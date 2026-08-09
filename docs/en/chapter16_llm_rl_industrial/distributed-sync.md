# 14.4 Distributed Sync, Async, and MoE Training

> [Appendix B.1 RL Training Systems](../appendix_industrial_training/rl-infrastructure) already covered the fundamentals — sampling, asynchrony, distributed parallelism. This chapter raises the vantage point to **framework-level architecture** and **frontier industrial practice**: how veRL uses HybridFlow to orchestrate multiple models under one roof, how AReaL/LlamaRL use full asynchrony to break down the wall between generation and training, how DeepSeek V3's DualPipe does pipeline parallelism on MoE, and how a 10,000-GPU cluster gets profiled and tuned.

## 36.1 A Deep Dive into the veRL Architecture

veRL (Volcano Engine Reinforcement Learning) is the RL training framework ByteDance open-sourced in 2024, described in [HybridFlow, arXiv:2409.19256](https://arxiv.org/abs/2409.19256). It has become the de facto mainstream framework for LLM RL training, adopted by the training scripts of teams like Qwen, DeepSeek, Llama, and Mistral.

### The Core Design of HybridFlow

HybridFlow abstracts RLHF/GRPO/PPO training as **single-controller multi-model orchestration**:

```
┌─────────────────────────────────────────────────────────┐
│              Single Controller (Driver)                  │
│  - Algorithm logic (the PPO/GRPO main loop)              │
│  - Resource scheduling (which GPUs run which model)      │
└──────────┬──────────────────────────────────────────────┘
           │
   ┌───────┼───────┬─────────────┬─────────────┐
   │       │       │             │             │
   ▼       ▼       ▼             ▼             ▼
┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Actor │ │Critic│ │Reference │ │Reward    │ │Rollout   │
│(FSDP)│ │(FSDP)│ │(Frozen)  │ │Model     │ │Engine    │
│      │ │      │ │          │ │          │ │(vLLM)    │
└──────┘ └──────┘ └──────────┘ └──────────┘ └──────────┘
   ▲       ▲       ▲             ▲             ▲
   │       │       │             │             │
   └───────┴───────┴─────────────┴─────────────┘
              ResourcePool (a GPU set)
```

### Three Core Abstractions

#### 1. ResourcePool

GPUs are grouped, and each group can host one or more models:

```python
# veRL config example (simplified)
resource_pools = {
    "actor_pool": num_gpus=8,    # Actor gets 8 GPUs
    "critic_pool": num_gpus=4,   # Critic gets 4 GPUs
    "rollout_pool": num_gpus=8,  # Rollout gets 8 GPUs
    "ref_pool": num_gpus=2,      # Reference model gets 2 GPUs
}
```

Different models can either **share GPUs** (colocate) or **have dedicated GPUs** (disaggregated):

```python
# Colocate: Actor and Rollout share the same GPU group
mapping = {
    "actor": "actor_rollout_pool",
    "rollout": "actor_rollout_pool",  # shared!
    "critic": "critic_pool",
    "ref": "ref_pool",
}
```

#### 2. Worker

Each Worker is an independent model instance that encapsulates the concrete training/inference logic:

```python
class ActorWorker:
    def __init__(self, model_config):
        self.model = FSDPActor(model_config)

    def update(self, batch):
        # PPO/GRPO loss computation + backward pass
        loss = compute_ppo_loss(batch, self.model)
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        # Sync weights out to the Rollout Engine
        return self.model.state_dict()

class RolloutWorker:
    def __init__(self, model_config):
        self.engine = vLLMEngine(model_config)

    def generate(self, prompts):
        return self.engine.generate(prompts)

    def sync_weights(self, new_weights):
        self.engine.load_weights(new_weights)
```

#### 3. Driver (Single Controller)

The Driver is the main loop of the RL algorithm — it orchestrates all the Workers:

```python
class PPODriver:
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # 1. Have the Actor expose its current weights to Rollout
            weights = self.actor_worker.get_weights()
            self.rollout_worker.sync_weights(weights)

            # 2. Sample with the current policy
            prompts = sample_prompts(self.dataset)
            responses = self.rollout_worker.generate(prompts)

            # 3. Score rewards with the Reward Model
            rewards = self.reward_worker.score(prompts, responses)

            # 4. Compute values with the Critic
            values = self.critic_worker.value(prompts, responses)

            # 5. Compute advantages + PPO loss, update the Actor
            advantages = compute_gae(rewards, values)
            self.actor_worker.update(prompts, responses, advantages)

            # 6. Update the Critic
            self.critic_worker.update(prompts, responses, rewards)
```

### What "Hybrid" Means in HybridFlow

Hybrid refers to a **unified hybrid-parallel strategy** — within a single framework you can combine:

- **3D Parallelism**: TP (tensor parallel) × PP (pipeline parallel) × DP (data parallel)
- **Colocate vs. Disaggregated**: models can share GPUs or hold them exclusively
- **Multiple training backends**: FSDP, Megatron, DeepSpeed ZeRO
- **Multiple inference backends**: vLLM, SGLang, HuggingFace generate

veRL is the first framework to make all of these dimensions configurable. DeepSpeed-Chat and OpenRLHF are more constrained along some of these axes.

### Core Differences from Other Frameworks

| Dimension               | veRL (HybridFlow)     | OpenRLHF          | NeMo-Aligner     | TRL            |
| ----------------------- | --------------------- | ----------------- | ---------------- | -------------- |
| **Orchestration**       | Single-controller     | Single-controller | Multi-controller | Single-process |
| **Resource allocation** | Arbitrary combination | Strict separation | NVIDIA stack     | Single GPU     |
| **Training backend**    | FSDP + Megatron       | FSDP/DeepSpeed    | Megatron         | Accelerate     |
| **Inference backend**   | vLLM/SGLang           | vLLM              | TRT-LLM          | HF generate    |
| **Typical scale**       | 8-1024 GPUs           | 8-256 GPUs        | 8-512 GPUs       | 1-8 GPUs       |

[Chapter 7's GRPO practice](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) is built on veRL.

## 36.2 OpenRLHF / NeMo-Aligner / TRL Compared

### OpenRLHF

[OpenRLHF, arXiv:2405.11143](https://arxiv.org/abs/2405.11143) is maintained by the OpenLLMAI team and is one of the earliest open-source RLHF frameworks.

**Core design**:

- Distributed scheduling built on **Ray**
- Strict **Actor/Critic/Ref/RM separation** — each model runs in its own Ray Actor process
- Emphasis on **simplicity** and **ease of use**

```python
# OpenRLHF PPO training (pseudocode)
from openrlhf import PPOTrainer, ModelGroup

actor = ModelGroup(num_gpus=8, backend="deepspeed")
critic = ModelGroup(num_gpus=8, backend="deepspeed")
ref = ModelGroup(num_gpus=4)
reward = ModelGroup(num_gpus=4)
vllm = VLLMRollout(num_gpus=8)

trainer = PPOTrainer(actor, critic, ref, reward, vllm)
trainer.train(dataset, num_epochs=100)
```

**Where it fits**: research settings and medium-scale training (8-256 GPUs). SimpleRL and Llama-3.1 post-training have both used OpenRLHF.

### NeMo-Aligner

[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) is NVIDIA's official stack, deeply integrated with Megatron-LM and TRT-LLM.

**Core design**:

- **Megatron** training backend (the strongest large-model parallelism story available)
- **TRT-LLM** inference backend (NVIDIA's own inference optimization)
- Favors full-stack NVIDIA optimization

**Where it fits**: NVIDIA clusters, very large models (70B+), workloads chasing peak performance. The Nemotron series and Llama-3 training on NVIDIA clusters both use NeMo.

### TRL (Transformer Reinforcement Learning)

[TRL](https://github.com/huggingface/trl) is HuggingFace's lightweight framework.

**Core design**:

- Built on **Accelerate** (HuggingFace's distributed abstraction)
- Single-process model, sharded automatically by Accelerate
- **Ease of use above all** — 10 lines of code to run PPO

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
config = PPOConfig(batch_size=8)
trainer = PPOTrainer(config, model)
trainer.train(dataset)
```

**Where it fits**: learning, prototyping, small-scale experiments (1-8 GPUs). Not suited for production-grade training.

### The Four Frameworks Compared

| Framework        | Ease of use | Performance | Scale ceiling | Industrial adoption                    |
| ---------------- | ----------- | ----------- | ------------- | -------------------------------------- |
| **veRL**         | Medium      | High        | 1024+ GPUs    | Qwen, DeepSeek, internal at ByteDance  |
| **OpenRLHF**     | High        | Medium      | 256 GPUs      | SimpleRL, various open-source projects |
| **NeMo-Aligner** | Low         | Very high   | 512+ GPUs     | NVIDIA customers, Nemotron             |
| **TRL**          | Very high   | Low         | 8 GPUs        | Research, teaching                     |

**Recommendation**:

- Learning, prototyping: TRL
- Research, medium scale: OpenRLHF or veRL
- Large-scale production: veRL or NeMo-Aligner (depending on the hardware stack)

## 36.3 The Rollout Engine and vLLM Integration

99% of RL training time is spent on rollout ([Appendix B.1](../appendix_industrial_training/async-training)). The rollout engine is the core of the performance bottleneck, and vLLM is the de facto standard.

### The Core Optimizations in vLLM

#### 1. PagedAttention

A conventional KV cache is allocated contiguously, which leads to severe memory fragmentation. vLLM borrows the paging mechanism from operating systems and splits the KV cache into fixed-size blocks:

```python
# Conventional: KV cache allocated contiguously
seq_len = 2048
kv_cache = torch.empty(batch_size, seq_len, num_heads, head_dim)
# Memory utilization: 50-70%

# vLLM PagedAttention: block-based
block_size = 16
blocks = allocate_blocks(num_blocks)
# Memory utilization: 95%+
```

Memory utilization goes from 50-70% to 95%+, and batch size grows 2-4×.

#### 2. Continuous Batching

Conventional batching waits until an entire batch has finished generating before swapping it out. vLLM does **dynamic batching** instead — the moment a sequence finishes, a new one takes its place:

```
Time:  ──────────────────────────────────────►
Seq A: [tok][tok][tok][tok][EOS]
Seq B: [tok][tok][tok][tok][tok][tok][EOS]
Seq C:           [tok][tok][tok][tok][EOS]  ← joins right after A finishes
Seq D:                    [tok][tok][tok][EOS]  ← joins right after C finishes
```

Throughput improves 5-10× over static batching.

#### 3. Speculative Decoding

A small model drafts a handful of tokens first, and the large model verifies them in parallel:

```python
def speculative_decode(prompt, draft_model, target_model, num_draft=4):
    while not done:
        # 1. The small model generates num_draft tokens
        draft_tokens = draft_model.generate(prompt, max_tokens=num_draft)

        # 2. The large model verifies them in parallel
        target_logits = target_model.forward(prompt + draft_tokens)

        # 3. Accept tokens that match, reject and regenerate otherwise
        for i, token in enumerate(draft_tokens):
            if target_logits[i].argmax() == token:
                prompt.append(token)
            else:
                prompt.append(target_logits[i].argmax())
                break
```

Throughput improves 2-3× (for typical LLM inference).

### vLLM's Role in RL Training

Inside veRL, vLLM serves as the RolloutWorker:

```python
class VLLMRolloutWorker:
    def __init__(self, model_path, tensor_parallel_size=8):
        from vllm import LLM
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,  # key: reuse KV across multiple samples of the same prompt in GRPO
            gpu_memory_utilization=0.9,
        )

    def generate(self, prompts, sampling_params):
        # Batch generation
        return self.engine.generate(prompts, sampling_params)

    def sync_weights(self, new_weights):
        # vLLM 0.5+ supports online weight updates
        self.engine.load_weights(new_weights)
```

**Prefix Caching** matters especially for GRPO — when the same prompt produces $G=8$ responses, the KV cache for the shared prefix (the prompt portion) can be reused, saving 70-80% of the memory and time.

### SGLang: vLLM's Challenger

[SGLang](https://github.com/sgl-project/sglang), developed by the LMSYS team, outperforms vLLM in agentic settings:

- **RadixAttention**: manages the KV cache with a radix tree, reusing it across requests
- **Programmatic Frontend**: supports complex control flow (multi-turn calls, branches, loops)
- **Constrained Decoding**: built-in JSON and regex-constrained generation

In industrial practice:

- **vLLM**: general-purpose rollout, single-turn generation
- **SGLang**: agentic rollout, multi-turn, structured output
- **TRT-LLM**: extreme optimization on NVIDIA hardware

## 36.4 GPU Memory Optimization: ZeRO, FSDP, Gradient Checkpointing

Memory is the core bottleneck in LLM training. A 70B model trained in bf16 with full parameters needs ~1.5 TB of memory — an 80GB H100 can't come close to holding that on its own.

### Breaking Down Training Memory

Training memory has four components:

$$\text{Memory} = \underbrace{|\theta| \cdot 2}_{\text{weights (bf16)}} + \underbrace{|\theta| \cdot 2}_{\text{gradients}} + \underbrace{|\theta| \cdot 8 + \text{optimizer state}}_{\text{Adam state}} + \underbrace{\text{activation}}_{\text{activations}}$$

For a 70B model:

- Weights: 140 GB
- Gradients: 140 GB
- Adam state (m, v, master weights): 560 GB
- Activations: ~100 GB (depends on batch size and sequence length)
- **Total**: ~940 GB

A single 80GB H100 falls far short.

### ZeRO (Zero Redundancy Optimizer)

[DeepSpeed ZeRO, arXiv:1910.02054](https://arxiv.org/abs/1910.02054) shards the training state across multiple GPUs:

| Stage      | What's sharded                | Savings                   | Communication cost |
| ---------- | ----------------------------- | ------------------------- | ------------------ |
| **ZeRO-1** | Optimizer state               | 4×                        | Low                |
| **ZeRO-2** | Optimizer + Gradient          | 8×                        | Medium             |
| **ZeRO-3** | Optimizer + Gradient + Weight | $N$× (N = number of GPUs) | High               |

ZeRO-3 shards the weights too, so each GPU only stores $1/N$ of them — but the forward and backward passes need an all-gather to reconstruct them.

```python
# DeepSpeed ZeRO-3 config
config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
    },
    "bf16": {"enabled": True}
}
```

### FSDP (Fully Sharded Data Parallel)

PyTorch's native equivalent of ZeRO-3, and easier to use than DeepSpeed:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = LlamaForCausalLM(config)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # equivalent to ZeRO-3
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    cpu_offload=CPUOffload(offload_params=False),  # optional CPU offload
)
```

veRL defaults to FSDP — it's more stable than DeepSpeed and fits more naturally into the PyTorch ecosystem.

### Gradient Checkpointing

Instead of sharding the model, this trades compute for memory — intermediate activations aren't saved during the forward pass, and are recomputed during the backward pass:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        # Wrap the transformer block with checkpoint
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.transformer_block(x)
```

Activation memory drops from $O(L)$ to $O(\sqrt{L})$ ($L$ is the number of layers), at the cost of running the forward pass twice — training slows down 20-30%.

### The Memory Budget When Combining All Three

For a 70B model (8 H100 80GB GPUs):

| Configuration                          | Per-GPU memory | Training speed |
| -------------------------------------- | -------------- | -------------- |
| Full parameters + Adam (baseline)      | 940 GB (OOM)   | -              |
| ZeRO-3                                 | 118 GB (OOM)   | -              |
| ZeRO-3 + Gradient Checkpointing        | 30 GB          | 1×             |
| ZeRO-3 + Gradient Checkpointing + LoRA | 8 GB           | 1.2×           |

LoRA ([Chapter 6](../chapter08_rlhf/industrial-post-training)) trains only a small set of parameters, cutting memory requirements sharply. Industrial-grade 70B RL training typically uses LoRA + FSDP.

## 36.5 Asynchronous RL Training

The bottleneck of synchronous training is covered in detail in [Appendix B.1](../appendix_industrial_training/async-training) — the GPU sits 99% idle waiting for rollout. Asynchronous training decouples generation from training so the two run at the same time. Below are three flagship 2025 frameworks that put this into practice.

### LlamaRL

[LlamaRL, Meta arXiv:2505.24034](https://arxiv.org/abs/2505.24034) is the distributed RL framework Meta released in May 2025.

**Core innovation**: **fully decentralized** — there is no master node; every worker is autonomous.

```python
# LlamaRL architecture (simplified)
class LlamaRLWorker:
    def run(self):
        while True:
            # Each worker decides its own job
            if self.role == "rollout":
                prompts = self.fetch_from_queue()
                responses = self.generate(prompts)
                self.push_to_train_queue(responses)

            elif self.role == "train":
                batch = self.fetch_from_rollout_queue()
                self.update(batch)
                self.broadcast_weights()  # asynchronous broadcast
```

**Advantages**:

- No single point of failure
- Easy to scale horizontally (just add workers)
- Suited to extreme scale (10k+ GPUs)

**Measured result**: running Llama-3-70B GRPO on 4096 GPUs, it's **10.4×** faster than synchronous training.

### AReaL (Asynchronous RL)

[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning, arXiv:2505.24298](https://arxiv.org/abs/2505.24298) is a large-scale asynchronous LLM RL system open-sourced by Ant Group and Tsinghua in 2025.

**Core innovation**: **fully asynchronous rollout + staleness-aware PPO**. Rollout workers continuously generate samples, and training workers consume a batch the moment it's available. The system controls sample staleness and adds a correction term in the PPO update for samples generated by a stale policy, mitigating the drift that comes from "the generating policy is already K steps behind the current training policy."

```python
# AReaL's key algorithm (simplified)
def staleness_aware_update(batch, current_weights):
    # batch records the policy version and log-prob at rollout time
    gen_log_probs = batch["gen_log_probs"]
    current_log_probs = compute_log_probs(batch, current_weights)
    importance_weights = torch.exp(current_log_probs - gen_log_probs)

    # Clip importance weights so stale samples don't cause outsized gradients
    clipped_weights = torch.clamp(importance_weights, 0.8, 1.2)
    loss = -(clipped_weights * advantages).mean()

    return loss
```

**Advantages**:

- Allows training on stale data — strict on-policy sampling isn't required
- The buffer can accumulate a large volume of data
- Training and generation are fully decoupled

**Measured result**: running 671B MoE GRPO on 1024 GPUs, it's **2.77×** faster than synchronous training.

### AgentRL

[AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework, arXiv:2510.04206](https://arxiv.org/abs/2510.04206) is a multi-turn, multi-task agentic RL framework released in October 2025, with code at [THUDM/AgentRL](https://github.com/THUDM/AgentRL).

**Core innovation**: a **fully-asynchronous generation-training pipeline + a unified environment interface**. On the training side, three pools of workers — rollout, actor, and reference — sample and update asynchronously. On the environment side, a function-call API, containerized environments, a centralized controller, and task workers manage heterogeneous tasks. On the algorithm side, cross-policy sampling and task advantage normalization strengthen multi-turn exploration and stabilize multi-task training, respectively.

```python
# AgentRL's asynchronous training structure (simplified)
rollout_workers.stream_trajectories(task_manager)
actor_workers.update_policy(buffer.sample())
reference_workers.compute_kl(buffer.sample())
controller.route_function_calls(task_workers)
```

**Advantages**:

- Supports multi-turn, multi-task agentic RL
- Decouples trajectory collection from policy updates asynchronously
- Manages environment deployment through a controller / task worker / transport layer
- Used to build AutoGLM

**Where it fits**: training SWE-Agent, Computer Use, and Deep Research Agent workloads.

### The Three Async Frameworks Compared

| Framework   | Main contributor       | Core mechanism                                        | Speedup                   | Where it fits              |
| ----------- | ---------------------- | ----------------------------------------------------- | ------------------------- | -------------------------- |
| **LlamaRL** | Meta                   | Fully decentralized                                   | 10.4×                     | Extreme-scale dense models |
| **AReaL**   | Ant Group and Tsinghua | Fully asynchronous rollout + staleness-aware PPO      | 2.77×                     | Large-scale LLM RL         |
| **AgentRL** | THUDM / Zhipu          | Multi-turn multi-task + unified environment interface | not reported in the paper | Agent training             |

## 36.6 MoE + RL Training

DeepSeek V3, Qwen3, and GLM-4.5 are all MoE architectures. MoE brings new challenges to RL training.

### What Makes MoE Different

An MoE model's parameters are distributed unevenly — most parameters live in the experts, and each sample only activates a small subset of them:

```
MoE model structure (DeepSeek V3):
┌──────────────────────────────────────────┐
│ Dense part (attention, etc.): 20B params │
├──────────────────────────────────────────┤
│ MoE part:                                │
│  - 256 experts × 5B params = 1.28T       │
│  - each sample activates 8 experts       │
│  - actual active params: 40B             │
└──────────────────────────────────────────┘
Total params: 1.3T, active params: 60B
```

### Three Challenges of MoE RL Training

#### 1. Uneven Expert Load

Some experts get activated frequently while others sit idle. This causes:

- Uneven compute load (some GPUs overloaded)
- Skewed training-data distribution (some experts undertrained)

**Fix**: an **Expert Balancing Loss**:

```python
def expert_balancing_loss(router_logits, num_experts):
    # Compute the activation frequency of each expert
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_freq = router_probs.mean(dim=0)  # [num_experts]

    # Encourage a uniform distribution
    target_freq = 1.0 / num_experts
    balance_loss = ((expert_freq - target_freq) ** 2).mean()

    return balance_loss
```

#### 2. Communication Overhead

MoE experts are spread across multiple GPUs (Expert Parallelism), so every sample requires an all-to-all communication step:

```
GPU 0: expert 0,1,2     ──┐
GPU 1: expert 3,4,5     ──┼── all-to-all ── all-to-all back after processing
GPU 2: expert 6,7,8     ──┤
GPU 3: expert 9,10,11   ──┘
```

**Fix**: **DeepEP** (DeepSeek Expert Parallelism), which optimizes the all-to-all communication pattern.

#### 3. High Token-Level IS Variance

[The GRPO family](../chapter09_grpo_rlvr/grpo-family) already noted this — under MoE, different tokens route to different experts, so token-level importance-sampling ratios swing wildly and gradient variance balloons.

**Fix**: **GSPO (Group Sequence Policy Optimization)** — move the IS ratio from the token level to the sequence level:

```python
# PPO/GRPO: token-level IS
token_ratio = exp(log_prob_new - log_prob_old)  # each token independent

# GSPO: sequence-level IS
sequence_log_prob_new = sum(log_prob_new_per_token)
sequence_log_prob_old = sum(log_prob_old_per_token)
sequence_ratio = exp(sequence_log_prob_new - sequence_log_prob_old)
# the whole sequence shares one ratio
```

The entire Qwen3 lineup (including 235B-A22B) is trained on GSPO.

### DeepSeek V3's MoE RL

The RL training practice behind DeepSeek V3 (671B MoE, 37B active):

- **DualPipe**: pipeline-parallel optimization (details in 36.7)
- **FP8 training**: uses FP8 to cut memory and compute ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
- **MTP (Multi-Token Prediction)**: predicts multiple tokens at once, raising the density of the training signal

### Step Flash (StepFun)

Step Flash is a MoE RL optimization StepFun released in 2025:

- **Dynamic Expert Allocation**: adjusts the number of experts dynamically based on the token distribution within a batch
- **Sparse Gradient Sync**: only syncs gradients for experts that were actually activated
- **Cache-aware Routing**: takes KV-cache locality into account when routing

### GLM-4.5 (Zhipu)

GLM-4.5 is trained with the **slime** framework ([THUDM/slime](https://github.com/THUDM/slime)):

- Megatron training backend
- SGLang inference backend
- Native MoE optimization (DeepEP communication, fp8 rollout)

## 36.7 DualPipe and Best-Fit Packing

### DualPipe

The [DeepSeek V3 paper, arXiv:2412.19437](https://arxiv.org/abs/2412.19437) introduces **DualPipe** — bidirectional pipeline parallelism.

Conventional pipeline parallelism (PP) suffers from a bubble problem:

```
GPU 0: [F0][F1][F2][F3]              [B3][B2][B1][B0]
GPU 1:       [F0][F1][F2][F3]   [B3][B2][B1][B0]
GPU 2:             [F0][F1][F2][F3][B3][B2][B1][B0]
                   ↑                ↑
                 forward           backward
                    the bubble is large
```

DualPipe runs forward and backward **at the same time** — the forward pass of stage N and the backward pass of stage N-1 overlap on the same GPU:

```
GPU 0: [F0|B0][F1|B1][F2|B2][F3|B3]  ← forward and backward overlap
GPU 1:       [F0|B0][F1|B1][F2|B2][F3|B3]
GPU 2:             [F0|B0][F1|B1][F2|B2][F3|B3]
                                    almost no bubble
```

The bubble ratio drops from the conventional $\frac{P-1}{M}$ ($P$ is the number of PP stages, $M$ is the number of micro-batches) to $\frac{P-1}{2M}$.

```python
# DualPipe pseudocode
class DualPipeScheduler:
    def schedule(self, num_stages, num_micro_batches):
        schedule = []
        for step in range(num_micro_batches + num_stages - 1):
            for stage in range(num_stages):
                # The same stage does both forward and backward in the same step
                fwd_mb = step - stage
                bwd_mb = step - (num_stages - 1 - stage)
                if fwd_mb >= 0 and fwd_mb < num_micro_batches:
                    schedule.append(("forward", stage, fwd_mb))
                if bwd_mb >= 0 and bwd_mb < num_micro_batches:
                    schedule.append(("backward", stage, bwd_mb))
        return schedule
```

### Best-Fit Packing

Conventional micro-batch assignment is uniform — every GPU gets the same number of micro-batches. But under MoE, different experts carry different loads, so uniform assignment leaves the cluster unbalanced.

**Best-Fit Packing**: use a bin-packing algorithm to assign differently sized micro-batches to GPUs:

```python
def best_fit_pack(items, bin_capacity):
    """items are micro-batches of varying size; bin_capacity is the capacity of one GPU"""
    bins = [[]]
    for item in sorted(items, reverse=True):  # largest first
        # Find the fullest bin that still has room
        best_bin = None
        best_remaining = float('inf')
        for bin in bins:
            remaining = bin_capacity - sum(bin)
            if item <= remaining < best_remaining:
                best_bin = bin
                best_remaining = remaining
        if best_bin is None:
            bins.append([item])
        else:
            best_bin.append(item)
    return bins
```

Best-Fit Packing raises DeepSeek V3's GPU utilization from 70% to 95%.

## 36.8 Performance Profiling and Bottleneck Analysis

Performance tuning for RL training has to be grounded in profiling — not guesswork.

### Profiling Tools

#### 1. PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    trainer.train_step()

# Print the top 10 most time-consuming operations
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### 2. NVIDIA Nsight Systems

```bash
# Run training under nsys
nsys profile -o rl_train_profile python train.py

# View the timeline in the Nsight Systems GUI
nsys-ui rl_train_profile.qdrep
```

This visualizes the execution time of every CUDA kernel, CPU-GPU synchronization points, and communication overhead.

#### 3. veRL's Built-In Profiler

veRL ships RL-specific profiling:

```python
from verl.utils.profiler import RLProfiler

with RLProfiler() as p:
    trainer.train()
    p.print_summary()
# Output:
#   rollout time: 3500s (85%)
#   actor update time: 120s (3%)
#   critic update time: 80s (2%)
#   weight sync time: 30s (0.7%)
#   communication: 400s (10%)
```

### Typical Bottlenecks and Fixes

| Bottleneck                   | Symptom                                                        | Fix                                            |
| ---------------------------- | -------------------------------------------------------------- | ---------------------------------------------- |
| **Slow rollout**             | Rollout eats 80%+ of the time                                  | Add rollout GPUs, use vLLM prefix caching      |
| **Slow weight sync**         | Sync eats 5%+ of the time                                      | Use LoRA, pack transfers over NCCL             |
| **Communication overhead**   | All-reduce eats 10%+ of the time                               | Increase batch size, use gradient accumulation |
| **Activation memory blowup** | OOM                                                            | Gradient checkpointing                         |
| **Uneven expert load**       | Some GPUs at 90%+, others at 30%                               | Expert balancing loss, dynamic routing         |
| **Straggler problem**        | The longest sequence in the batch dictates the wall-clock time | Length bucketing, Seer divided rollout         |

### MFU (Model FLOPs Utilization)

The gold-standard metric for training efficiency:

$$\text{MFU} = \frac{\text{actual FLOPs}}{\text{peak FLOPs} \times \text{time}}$$

The H100's bf16 peak is ~1000 TFLOPS. Typical MFU for LLM RL training:

| Configuration                                   | MFU                                                 |
| ----------------------------------------------- | --------------------------------------------------- |
| Dense + FSDP + checkpointing                    | 35-45%                                              |
| MoE + EP + DualPipe                             | 50-60%                                              |
| Asynchronous RL (generation/training separated) | 70-80% (the rollout portion is accelerated by vLLM) |

An MFU below 30% signals significant room for optimization — usually a communication or rollout bottleneck.

## 36.9 Practice at 10,000-GPU Scale

Put everything above together and you get RL training practice on a 2025-era 10,000-GPU cluster.

### A Typical Configuration

Take GRPO training of Qwen3-235B-A22B (235B total parameters, 22B active MoE) as an example:

```yaml
# Cluster configuration
total_gpus: 12288 # 12k H100s
intra_node_bandwidth: 900 GB/s # NVLink
inter_node_bandwidth: 50 GB/s # InfiniBand

# Model parallelism
tensor_parallel: 8 # TP=8 (within a node)
pipeline_parallel: 4 # PP=4 (across nodes)
expert_parallel: 16 # EP=16
data_parallel: 24 # DP=24

# Training configuration
algorithm: GSPO # a GRPO variant optimized for MoE
batch_size_per_gpu: 1
gradient_accumulation: 32
seq_len: 32768
group_size: 8 # GRPO generates 8 samples per prompt

# Async configuration
async_mode: disaggregated
rollout_buffer_size: 100000
weight_sync: lora # sync only the LoRA adapter
weight_sync_method: nccl_packed
```

### Measured Performance

```text
Training 1 epoch (10B tokens):
  Total time: 24 hours
  GPU hours: 294912

Time breakdown:
  Rollout: 18 hours (75%)
  Actor update: 3 hours (12.5%)
  Critic update: 2 hours (8%)
  Weight sync: 0.5 hours (2%)
  Other: 0.5 hours (2.5%)

MFU: 52% (MoE + DualPipe + FP8)
```

### Key Lessons from 10,000-GPU Training

#### 1. Failure Is the Norm

Across 12,288 GPUs, 5-10 fail on an average day. This forces certain practices:

- **Checkpoint frequency**: save every 30 minutes so you can roll back on failure
- **Redundant design**: keep 8 spare GPUs for every 1024 in use
- **Automatic restart**: detect failure and resume automatically from the most recent checkpoint

#### 2. Communication Is the Performance Killer

Cross-node communication is slow, which shapes how a 10,000-GPU cluster's network gets designed:

- **Topology-aware placement**: prefer neighboring GPUs when forming a tensor-parallel group
- **Overlap communication with compute**: kick off the gradient all-reduce while the backward pass is still running
- **Gradient bucketing**: merge small gradients to cut the number of communication rounds

#### 3. MoE Routing Stability

During MoE training the expert routing can suddenly collapse — every token starts routing to a handful of experts. Monitor for it:

```python
# Monitor expert load in real time
def monitor_expert_balance(model):
    while training:
        for layer in model.moe_layers:
            router_probs = layer.router.get_recent_probs()
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10))
            if entropy < threshold:  # routing entropy too low
                alert(f"Layer {layer.id}: expert routing collapse!")
        time.sleep(60)
```

#### 4. The Data Pipeline Is a Hidden Bottleneck

A 10,000-GPU cluster consumes millions of tokens per second, and data loading itself can become the bottleneck:

- **Prefetching**: prepare the next 10 batches of data ahead of time
- **Data compression**: store data in a more compact format
- **Distributed storage**: spread data across multiple SSDs to avoid a single-point I/O bottleneck

## Chapter Summary

Distributed RL training systems are the core engineering discipline of the LLM era:

1. **veRL (HybridFlow)** is the mainstream framework — single-controller multi-model orchestration with flexible resource allocation
2. **OpenRLHF/NeMo-Aligner/TRL** each occupy their own niche — research, the NVIDIA stack, lightweight teaching
3. **vLLM/SGLang** are the core of the rollout engine — PagedAttention, Continuous Batching, Prefix Caching
4. **ZeRO/FSDP/Checkpointing** solve the memory problem — LoRA + FSDP + Checkpointing is the standard recipe for 70B training
5. **Asynchronous training (LlamaRL/AReaL/AgentRL)** is the direction for 2025 — 10× speedups, tolerance for off-policy data
6. **MoE + RL** needs GSPO, Expert Balancing, DualPipe, and Best-Fit Packing working together
7. **10,000-GPU clusters** push engineering to its limits — routine failures, communication bottlenecks, monitoring and alerting, data pipeline pressure

[Chapter 15, Industrial LLM RL Practice](../chapter09_alignment/industrial-post-training) revisits how these techniques land in practice from a product perspective — this chapter has been the engineering perspective.

## Further Reading

- [Sheng et al. 2024 "HybridFlow: A Flexible and Efficient RLHF Framework"](https://arxiv.org/abs/2409.19256)
- [Hu et al. 2024 "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework"](https://arxiv.org/abs/2405.11143)
- [Kwon et al. 2023 "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)](https://arxiv.org/abs/2309.06180)
- [Zheng et al. 2023 "SGLang"](https://arxiv.org/abs/2312.07104)
- [Rajbhandari et al. 2020 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054)
- [LlamaRL (Meta GenAI) 2025 "LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework"](https://arxiv.org/abs/2505.24034)
- [Fu et al. 2025 "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"](https://arxiv.org/abs/2505.24298)
- [Zhang et al. 2025 "AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework"](https://arxiv.org/abs/2510.04206)
- [DeepSeek-AI 2024 "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI 2025 "DeepSeek-R1: Incentivizing Reasoning Capability via RL"](https://arxiv.org/abs/2501.12948)
- [Qwen Team 2025 "Qwen3 Technical Report"](https://arxiv.org/abs/2505.09388)
- [Zheng et al. 2025 "GSPO: Group Sequence Policy Optimization"](https://arxiv.org/abs/2507.18071)
- [Qin et al. 2025 "Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning"](https://arxiv.org/abs/2511.14617)
