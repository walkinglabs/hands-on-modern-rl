# 14.3 Optimizers and Training Stability

[Section 9.8, Industrial Practice](./industrial-post-training) covered the mainstream industrial approaches from 2024-2025 — MiniMax, Qwen, Kimi, Seed, DeepSeek. This section adds a few of the newest industrial practices from 2025-2026:

- **GLM-4.5 / GLM-4.6** (Zhipu): a rising star among Chinese open-source reasoning models
- **Llama 4** (Meta): the evolution of an open-source flagship
- **Seed-Thinking** (ByteDance Seed): an industrial recipe for reasoning models
- **MuonClip + QK-clip** (Kimi K2): new tools for training stability

These works represent the **latest state of the art** in industrial LLM RL training — pushing the algorithms from [earlier chapters](./intro) to their limits.

## 9.9.1 Zhipu GLM-4.5 / GLM-4.6

[GLM-4.5](https://github.com/zai-org/GLM-4.5) (Zhipu AI, released July 2025) and GLM-4.6 (October 2025) mark an important advance for Chinese open-source reasoning models.

### What Sets the GLM Series Apart

The GLM (General Language Model) series differs from Qwen and DeepSeek in a few ways:

- **Mixture of Experts (MoE)**: GLM-4.5 uses an MoE architecture — 355B total parameters, 32B active
- **Dual modes**: thinking / non-thinking, similar to Qwen3
- **Fully open**: weights, training methodology, and part of the data are all released
- **Coding ability**: code generation and agentic capability were specifically reinforced

### The GLM-4.5 Training Pipeline

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Base pretraining (MoE architecture)              │
│   - 15T tokens of high-quality data                       │
│   - MoE: 355B total / 32B active                          │
│   - RoPE scaling for long context                         │
├──────────────────────────────────────────────────────────┤
│ Phase 2: General SFT                                      │
│   - Multilingual dialogue data                            │
│   - Tool-call format training                             │
├──────────────────────────────────────────────────────────┤
│ Phase 3: Reasoning RL                                      │
│   - Math, code, and reasoning tasks                        │
│   - GRPO + rule-based rewards                              │
│   - Self-validation integration                            │
├──────────────────────────────────────────────────────────┤
│ Phase 4: General RLHF                                      │
│   - Dialogue quality, safety                               │
│   - Dual objectives: helpfulness / harmlessness            │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Thinking / non-thinking unification                │
│   - Mixed-data SFT                                          │
│   - Teaches the model to switch modes                       │
└──────────────────────────────────────────────────────────┘
```

This pipeline closely mirrors [the DeepSeek-R1 training pipeline](../chapter18_grpo/deepseek-dapo) — both follow a multi-stage paradigm of SFT + reasoning RL + general RLHF.

### GLM-4.6 Improvements

The October 2025 GLM-4.6 upgrade brought:

- **Longer thinking**: supports chains of thought over 100K tokens
- **Stronger agentic behavior**: more tools integrated internally (search, code execution, file operations)
- **Better multimodality**: works in tandem with the GLM-4.5V vision model
- **Finer thinking budgets**: users can specify a budget (similar to Qwen3)

### GLM-4.6 Benchmark Results

| Benchmark     | GLM-4.5 | GLM-4.6 |
| ------------- | ------- | ------- |
| AIME 2025     | 75.3    | 83.6    |
| MATH-500      | 92.1    | 95.4    |
| LiveCodeBench | 56.2    | 62.7    |
| GPQA Diamond  | 68.5    | 72.4    |

GLM-4.6 reaches open-source SOTA on several benchmarks, close to Claude Opus 4.5 / GPT-5.

### The Industrial Significance of GLM

The GLM series matters industrially for a few reasons:

1. **Diversity in open-source reasoning models** — GLM is a third influential Chinese open-source model, alongside Qwen and DeepSeek
2. **Validation of MoE for reasoning RL** — it confirms the effectiveness of [GSPO](../chapter18_grpo/grpo-family) on MoE architectures
3. **Integration of code and reasoning** — GLM-4.6 puts particular emphasis on agentic capability, competing directly with products like Claude Code

## 9.9.2 Meta Llama 4

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (Meta, released April 2025) is Meta's open-source flagship.

### The Llama 4 Family

Llama 4 ships in three variants:

- **Llama 4 Scout**: 109B total / 17B active (MoE), 10M context
- **Llama 4 Maverick**: 400B total / 17B active, 1M context
- **Llama 4 Behemoth** (unreleased): 2T total / 288B active, trained internally at Meta

### Key Innovations in Llama 4

**Innovation 1: Native multimodality**

Llama 4 was not trained on text first and then bolted onto vision — it was **multimodal from the start**, treating text and image tokens the same way. This gives Llama 4 an edge in visual understanding over "vision-bolted-on-afterward" designs.

**Innovation 2: Early fusion**

Instead of late fusion — processing text and images separately and merging them afterward — Llama 4 uses **early fusion**, combining multimodal information early in the model.

**Innovation 3: MoE architecture**

The entire Llama 4 lineup uses MoE — Meta's first large-scale deployment of the architecture. MoE lets the model scale up total parameters under a fixed activation budget, raising capability without raising inference cost.

**Innovation 4: Ultra-long context**

Llama 4 Scout supports a **10M-token context** — the longest of any open-source model at the time it shipped. This is achieved through iRoPE (interleaved RoPE) and attention sparsity.

### Llama 4 Training Method

Meta has not disclosed the full training details for Llama 4, but the papers and blog posts allow us to reconstruct the outline:

```text
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Multimodal pretraining                          │
│   - Joint training on text + images + video               │
│   - On the order of 22T tokens (estimated)                 │
│   - Early-fusion architecture                              │
├─────────────────────────────────────────────────────────┤
│ Phase 2: Mid-training (medium-scale SFT)                  │
│   - General instruction following                          │
│   - Tool-call format                                       │
├─────────────────────────────────────────────────────────┤
│ Phase 3: Post-training RL                                  │
│   - RLHF + RLVR combined                                   │
│   - Multiple objectives: helpfulness / safety / reasoning  │
└─────────────────────────────────────────────────────────┘
```

### Controversies Around Llama 4

The Llama 4 release sparked several controversies.

**Controversy 1: The gap between benchmark scores and real-world performance**

Llama 4 Maverick scored well on many benchmarks, but users found it fell short of Claude 3.5 / GPT-5 in actual use. Meta later acknowledged a gap between benchmark evaluation and real-world experience.

**Controversy 2: Maverick's "special version"**

The Maverick version that ran on LM Arena was a **specially tuned build** — using an adjusted chat template and prompt engineering. The open-source release of Maverick differed from the arena version.

This episode is a similar integrity problem to the [Qwen3 data contamination](../chapter30_alignment_failures/modern-incidents) case — it exposes **how fragile benchmark evaluation can be**.

### The Industrial Significance of Llama 4

Despite the controversy, Llama 4 remains an important step forward for open-source LLMs:

1. **Open-source MoE reaching maturity** — it proves MoE is viable in the open-source ecosystem
2. **A new multimodal paradigm** — early fusion is a reference point for later work
3. **Ultra-long context** — a 10M context window opens up new application scenarios

## 9.9.3 Seed-Thinking and ByteDance's Reasoning Recipe

[Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) (ByteDance Seed, April 2025) is ByteDance's systematic summary of industrial training for reasoning models.

### The Core Contribution of Seed-Thinking

Seed-Thinking is not a new algorithm — it is the **systematization of an industrial recipe**, combining several components to reach SOTA.

**Component 1: Data curation**

```text
Math data:
  - High-quality math problems (historical AIME, Putnam questions)
  - Auto-generated problems (new problems generated by a strong LLM)
  - Difficulty grading (by the base model's pass rate)

Code data:
  - Codeforces problems (with test cases)
  - SWE-bench / SWE-smith (with PR data)
  - Function generation (HumanEval, extended)
```

**Component 2: GRPO + DAPO improvements**

Seed-Thinking adopts the four engineering modifications from [DAPO](../chapter18_grpo/deepseek-dapo), plus some new improvements:

- **Dynamic KL**: strong KL penalty early in training, weakened later
- **Adaptive clip**: the clip range adjusts as training progresses
- **Group-size scheduling**: large groups early on, small groups later

**Component 3: Self-verification**

The model verifies its own answer after generating it:

```python
def self_verification_reward(response, ground_truth):
    answer = extract_answer(response)

    # Have the model re-read the problem and verify the answer
    verification_prompt = f"Check whether this answer is correct: {answer}"
    verification = model.generate(verification_prompt)

    if "correct" in verification and answer == ground_truth:
        return 1.0  # correct answer + verification passed
    elif "incorrect" in verification and answer != ground_truth:
        return 0.5  # wrong answer, but the model caught its own error
    else:
        return 0.0  # wrong answer, and the model missed it
```

This self-verification reward teaches the model to **reflect** — not just to get the answer right, but to recognize when it is wrong.

**Component 4: Curriculum learning**

Training data is ordered by difficulty, easy examples first and harder ones later. Curriculum learning makes training more stable and avoids an overly sparse reward signal early on.

### Seed-Thinking's Results

Seed-Thinking 1.5 across several benchmarks:

| Benchmark         | Score |
| ----------------- | ----- |
| AIME 2024         | 86.4% |
| MATH-500          | 96.2% |
| GPQA Diamond      | 75.1% |
| Codeforces Rating | 1822  |

This is the core recipe behind ByteDance Seed's internal reasoning models, later used in products like the Doubao Pro reasoning edition.

## 9.9.4 Kimi K2's MuonClip + QK-clip

One of the industrial contributions of [Kimi K2](https://arxiv.org/abs/2507.20534) (Moonshot, July 2025) is **MuonClip + QK-clip** — new tools for training stability.

### The Muon Optimizer

[Muon](https://kellerjordan.github.io/posts/muon/) (February 2025) is a newly proposed optimizer — Muon stands for Momentum + Orthogonalization. It combines:

- **Momentum** (as in Adam)
- **Orthogonalization** (orthogonalizing the gradient)

Orthogonalization makes the update direction more stable, avoiding the oscillation Adam can exhibit along certain directions.

### The Core Idea of MuonClip

MuonClip adds a **clip** on top of Muon:

```python
def muon_clip_update(grad, momentum, clip_threshold=1.0):
    # Main Muon update
    momentum = beta * momentum + (1 - beta) * grad
    orthogonalized = orthogonalize(momentum)

    # Clip to prevent explosion
    norm = torch.norm(orthogonalized)
    if norm > clip_threshold:
        orthogonalized = orthogonalized * (clip_threshold / norm)

    return -lr * orthogonalized
```

The clip matters a great deal at large scale — it stops a single outlier gradient from wrecking the entire training run.

### QK-clip and Attention Stabilization

QK-clip is Kimi K2's other innovation — **clipping the Q·K product inside attention**:

```python
def attention_with_qk_clip(Q, K, V, clip_value=30.0):
    # Standard attention
    scores = Q @ K.T / sqrt(d)

    # QK-clip: keep attention scores from getting too large
    scores = torch.clamp(scores, min=-clip_value, max=clip_value)

    # Softmax + weighted sum
    attn = softmax(scores)
    output = attn @ V

    return output
```

**Why is QK-clip needed?**

During long-context training, attention scores can blow up because of **attention sink** (certain tokens absorbing a disproportionate amount of attention) — this distorts the softmax distribution and can trigger gradient explosion.

QK-clip avoids this by bounding the range of the scores.

### Training Results for Kimi K2

With MuonClip + QK-clip, Kimi K2's large-scale training showed:

- **Training stability**: loss spikes dropped from about once every 1T tokens to about once every 10T tokens
- **Training speed**: roughly 15% faster than Adam
- **Final performance**: Kimi K2 reaches open-source SOTA on multiple benchmarks

### The Industrial Significance of MuonClip

MuonClip is a key tool for training **extremely large LLMs**:

- **Trillion-parameter-scale training**: traditional Adam becomes unstable at trillion-parameter scale, and MuonClip fixes this
- **Ultra-long context**: QK-clip makes training at 1M+ context feasible
- **Open-source ecosystem**: Muon has already spread through the open-source community (OpenLM and PyTorch both support it)

## 9.9.5 Summary of Chinese Industrial Practice

By mid-2026, the landscape of Chinese industrial LLM RL practice looks like this:

| Company            | Flagship Model       | RL Algorithm        | Industrial Contribution    |
| ------------------ | -------------------- | ------------------- | -------------------------- |
| **DeepSeek**       | R1, V3.2             | GRPO + improvements | transparency, open source  |
| **Alibaba Qwen**   | Qwen3 series         | GSPO                | stable MoE training        |
| **ByteDance Seed** | Doubao Pro, Seedance | DAPO + VAPO         | multi-stream parallelism   |
| **Moonshot Kimi**  | K2, K2.5             | GRPO + MuonClip     | training-stability tooling |
| **Zhipu GLM**      | GLM-4.6              | GSPO-style          | open-source MoE            |
| **MiniMax**        | M1, M2               | CISPO               | low-precision training     |
| **StepFun**        | Step3                | internal method     | reasoning + multimodality  |

A few things stand out:

1. **Every company has its own "signature RL algorithm"** — GRPO, GSPO, DAPO, CISPO, VAPO, MuonClip
2. **Algorithmic innovation is coming mainly out of China** — a striking contrast with the closed-source camp in the US (OpenAI, Anthropic)
3. **These industrial contributions complement each other** rather than replace one another — each tackles RL training problems from a different angle

## 9.9.6 Future Industrial Directions

### Trillion-Parameter Models + Ultra-Long Context

- Base models at trillion-parameter scale (DeepSeek V3 sits at 671B; the next generation will push past 1T)
- 10M+ token context (Llama 4 Scout already supports this)
- Training stability remains the core challenge — MuonClip points the way forward

### Natively Multimodal RL

- The pattern is shifting away from "text RL plus vision SFT" toward "joint multimodal RL"
- Llama 4's early fusion is an early step in that direction
- Expect more natively multimodal RL algorithms going forward

### The Industrialization of Agentic RL

- Agent training is becoming mainstream, not just for SWE but for customer service, research, and operations as well
- Agent trajectory data is the key resource
- Investment in agent RL infrastructure is substantial

### Falling Training Costs

- In 2024, training a SOTA model required $100M+
- By 2026 that could fall to $10M, driven by better algorithms and cheaper compute
- This opens the door for smaller teams to compete in SOTA research

## Training-Inference Mismatch: The Hidden Killer of LLM-RL Training Stability

GLM-4.6, Llama 4, MuonClip, and the other methods discussed above all address "explicit" training instability — loss spikes, gradient explosions, KV cache collapse. But there is another problem in LLM-RL that has been **overlooked for a long time**: **Training-Inference Mismatch**.

Strictly speaking, training-inference mismatch is not unique to large models. Any RL system where the sampling policy drifts from the policy being optimized will produce a similar distributional bias — the AlphaGo and Atari-DQN era already had experience with Policy Lag causing training instability. But this problem gets **dramatically amplified in the engineering of large-model RL**, because LLM-RL systems use entirely different engines and precisions for sampling and training, which creates a fundamental split.

### The Root Cause: $\pi_{\text{rollout}}$ and $\pi_{\text{old}}$ Are Not the Same Policy

> **"When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"**
> _(Liu et al., 2025)_

In almost every LLM-RL implementation, $\pi_{\text{rollout}}$ (the inference policy responsible for sampling data) and $\pi_{\text{old}}$ (the "old policy" recorded by the training framework) **are not actually the same policy**:

- **Inference side** (generating rollout data): vLLM / SGLang, FP8/BF16 precision, KV-cache optimizations
- **Training side** (computing log-probs and gradients): FSDP/Megatron, BF16/FP32 precision, activation recomputation

The same model parameters, run under different precision and a different computation graph, naturally produce different log-probabilities. You might assume the behavior policy $\mu$ equals the target policy $\pi_\theta$, but that "approximately equals" in $\mu \approx \pi_\theta$ can already have drifted by tens of percentage points.

### Precision Is the Prime Suspect

> **"Defeating the Training-Inference Mismatch via FP16"**
> _(Qi et al., 2025)_

This paper traces the root cause to floating-point precision. BF16 has too few mantissa bits, which introduces systematic rounding error into token-level log-probability computation. Simply switching precision back to FP16 makes the bias nearly disappear — a few lines of code fix one of the most frustrating sources of training collapse in LLM-RL.

> **"Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning"**
> _(arXiv 2512.23087, 2025)_

This paper further reveals the **asymmetry** of training-inference mismatch: the bias is proportional to $(1-p)$ — high-frequency tokens carry negligible error, but low-frequency, long-tail tokens produce a systematic bias that accumulates steadily in the gradient estimate and eventually causes collapse.

> **"Stabilizing Reinforcement Learning with LLMs: Formulation and Practices"**
> _(Zheng et al., Qwen Team, arXiv 2512.01374, 2025)_

The Alibaba Qwen team proposed a unified theoretical framework: the token-level REINFORCE objective is fundamentally a **first-order approximation** of the sequence-level reward, and that approximation only holds under two conditions — **(1) training and inference match**, and **(2) the policy is not stale**. Once training-inference mismatch holds, the first-order approximation breaks down.

### Relationship to PPO Clipping

You might ask what this has to do with PPO. The answer: **PPO's clipping mechanism is a defense against training-inference mismatch, but it can only cover half of it**.

The core PPO objective is:

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the importance-sampling ratio. But PPO's clipping carries a default assumption: **the denominator $\pi_{\text{old}}$ really is "the policy that was actually executed at sampling time."**

In classic RL, the sampling process and the training process are the same Python process, and $\pi_{\text{old}}$ is exactly the network weights saved at the instant of sampling. In LLM-RL, however:

- $\pi_{\text{rollout}}$: the policy that was **actually in effect** when the vLLM engine sampled under FP8
- $\pi_{\text{old}}$: the policy the training framework **recomputes afterward** in BF16/FP32 — "what you think was used at sampling time"

These are simply **not the same policy**. The denominator of the importance-sampling ratio $r_t$ is **itself biased** — PPO's clipping is trying to correct for drift caused by optimization, but it has no mechanism for correcting the mismatch between the inference engine and the training engine.

Here's an analogy: PPO's clipping guarantees that **you don't stray too far from the old policy**, but it never checks whether the "old policy" map itself was accurate to begin with. Training-inference mismatch means **the map was already wrong from the start**, and clipping cannot detect that.

### Mainstream Industrial Fixes

Fixes for training-inference mismatch in the field fall roughly along a few lines:

- **Precision fixes**: use FP16/BF16 instead of FP8 for rollout, reducing the numerical gap between $\pi_{\text{rollout}}$ and $\pi_{\text{old}}$ (Qi et al., 2025). Some work goes the other direction and lowers precision on the training side instead — FP8-RL, implemented in the veRL framework, runs full W8A8 low-precision training end to end, paired with importance-sampling correction, and gets 44% higher rollout throughput while matching the BF16 baseline (Qiu et al., arXiv 2601.18150).
- **Importance-sampling (IS) correction**: since $\pi_{\text{rollout}} \neq \pi_{\text{old}}$, correct the distributional shift explicitly with importance weights. Truncated IS (TIS) is the most direct approach — it clips extreme IS ratios to avoid gradient explosion (Yao et al., NeurIPS 2025). More recent work is MinPRO (Lei et al., arXiv 2601.22718), which replaces the cumulative product with the minimum token-level ratio within a prefix, giving more stability when off-policy drift is large.
- **Pruning long-tail tokens**: training-inference mismatch concentrates in low-probability regions, so directly removing extreme long-tail tokens eliminates the largest source of bias at its origin ("Taming the Tail", arXiv 2512.23087).
- **MoE routing replay**: expert routing during inference naturally differs from routing during training. R3 (Rollout Routing Replay) replays the inference-time routing distribution during training, addressing an amplification effect unique to training-inference mismatch in MoE-RL (Zheng et al., arXiv 2512.01374).
- **An optimization-based view**: treat training-inference mismatch as a dynamic optimization problem, triggering learning-rate scheduling from signals such as a sudden jump in response length (Zhang et al., arXiv 2602.01826).
- **Engineering-side rollback correction**: before training, recompute the rollout policy's log-probability using the current training engine, forcibly aligning $\pi_{\text{rollout}}$ and $\pi_{\text{old}}$ — expensive, but the most reliable option.

### Making Peace with Reality

These papers all point toward the same conclusion: in the engineering practice of LLM-RL, there is no such thing as "pure" on-policy learning. The best we can do is **keep the gap between $\mu$ and $\pi_\theta$ within an acceptable range** — PPO's clipping is one form of control, FP16 is another, and R3's routing replay is yet another. The on-policy/off-policy classification described in [Chapter 4's algorithm taxonomy](../chapter03_mdp/algorithm-taxonomy) is a clean binary distinction in theory, but engineering reality is a **continuous spectrum** — what looks on-policy on paper always carries a trace of off-policy in practice.

## Summary

The core trends in modern industrial LLM RL practice:

- **GLM-4.6 / Llama 4**: the evolution of open-source flagships, combining MoE and multimodality
- **Seed-Thinking**: the systematization of an industrial recipe for reasoning models
- **MuonClip + QK-clip**: new tools for training stability
- **Chinese companies leading algorithmic innovation**: each with its own signature method

Together, these works have pushed RL's role in LLM training **from optional to essential** — without RL, there is no path to a SOTA model.

Coming up next:

- [Chapter 8, Reasoning Models](../chapter19_reasoning/intro) — a detailed discussion of reasoning models
- [Chapter 9, PRM](../chapter20_prm_search/intro) — industrial practice for process rewards
- [Chapter 10, RL-based SWE](../chapter23_rl_based_swe/intro) — training code agents
