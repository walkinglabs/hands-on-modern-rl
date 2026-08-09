# C.2 GPU-Hour Estimation Tables

> The question engineers get asked most often is: "How many GPU hours will it take to train this model, and how much will it cost?" This isn't a question for a compute vendor's sales team — it's a core constraint on whether a research plan is even feasible. This appendix organizes the pretraining and post-training costs disclosed in public tech reports (DeepSeek, Qwen, Kimi, Llama, Claude) into lookup tables, then lays out three budget tiers for training your own models.
>
> How to read this appendix: jump straight to the section you need and look up the numbers. If you're planning a budget, start at [G.4](#g-4-budgeting-your-own-training-run). Every number here comes from a public tech report or a scaling-law-based estimate — **none of it includes undisclosed internal data**.

## G.1 Pretraining Cost by Model Scale

Three quantities determine pretraining cost:

$$\text{GPU hours} \approx \frac{6 \cdot N \cdot D}{\text{hardware utilization (MFU)} \cdot \text{FLOPS per GPU}}$$

Here $N$ is the parameter count, $D$ is the number of training tokens, and MFU (Model FLOPs Utilization) typically falls between 30% and 55%. The table below collects training token counts and the corresponding GPU hours from public tech reports.

| Model           | Params    | Training tokens | GPU type    | GPU hours   | Source                      |
| --------------- | --------- | --------------- | ----------- | ----------- | --------------------------- |
| Llama 2 7B      | 7B        | 2.0T            | A100-80G    | 184,320     | Meta 2023                   |
| Llama 2 13B     | 13B       | 2.0T            | A100-80G    | 432,000     | Meta 2023                   |
| Llama 2 70B     | 70B       | 1.7T            | A100-80G    | 1,700,000   | Meta 2023                   |
| Llama 3 8B      | 8B        | 15T             | H100-80G    | 130,000     | Meta 2024                   |
| Llama 3 70B     | 70B       | 15T             | H100-80G    | 6,400,000   | Meta 2024                   |
| Llama 3.1 405B  | 405B      | 15T             | H100-80G    | 30,000,000  | Meta 2024 (16K-GPU cluster) |
| DeepSeek-V2     | 236B-A21B | 8.1T            | H800-80G    | 2,800,000   | DeepSeek 2024               |
| DeepSeek-V3     | 671B-A37B | 14.8T           | H800-80G    | 2,664,000   | DeepSeek 2024               |
| Qwen2.5 7B      | 7B        | 18T             | undisclosed | ~1,000,000  | Qwen 2024 (estimated)       |
| Qwen2.5 72B     | 72B       | 18T             | undisclosed | ~5,000,000  | Qwen 2024 (estimated)       |
| Qwen3 235B-A22B | 235B-A22B | 36T             | undisclosed | ~14,000,000 | Qwen 2025 (estimated)       |
| Kimi K2         | 1T-A32B   | 15.5T           | H800-80G    | ~9,000,000  | Kimi 2025 (estimated)       |

::: tip Two things to notice in this table

1. **MoE cuts activated compute sharply.** DeepSeek-V3 has 671B total parameters but only activates 37B, so its effective compute is roughly that of a 60B–70B dense model — yet it still costs 2.66M GPU hours.
2. **Token count is the deciding variable.** Llama 3 70B and Llama 2 70B have the same parameter count, but because the token count went from 1.7T to 15T, GPU hours quadrupled. **Every mainstream model released after 2024 trains on more than 10T tokens** — the Chinchilla ratio ($D \approx 20N$) has been broadly surpassed.
   :::

### Cost Estimates (at public cloud list prices)

| Model tier | GPU hours | A100 @ $2.5/h | H100 @ $3.5/h | H800 @ $3.0/h | B200 @ $6.0/h |
| ---------- | --------- | ------------- | ------------- | ------------- | ------------- |
| 7B dense   | ~200K     | $0.5M         | $0.7M         | $0.6M         | $1.2M         |
| 70B dense  | ~5M       | $12.5M        | $17.5M        | $15M          | $30M          |
| 405B dense | ~30M      | $75M          | $105M         | $90M          | $180M         |
| 671B MoE   | ~2.7M     | $6.8M         | $9.5M         | $8M           | $16M          |
| 1T MoE     | ~9M       | $22.5M        | $31.5M        | $27M          | $54M          |

::: warning Real cost is much higher than the table above
The table above counts only **bare GPU rental**. Real training adds on top of that: (1) storage, networking, and power, roughly +30%; (2) failed experiments and hyperparameter search, roughly ×3–5; (3) data collection and labeling, roughly 10%–20%. A model whose public report says "$15M" often costs the company $50M–$100M in practice.
:::

## G.2 Cost by Stage: SFT / RLHF / RLVR

Pretraining is only part of the cost. The GPU-hour share of **post-training** (SFT, RLHF, RLVR) has risen from about 5% in 2022 to over 30% in 2026 — because rollout in RLHF/RLVR is far slower than a single forward pass.

The table below estimates the **share of total training cost by stage**, based on public data from DeepSeek-V3/R1, Qwen3, Llama 3.1, and Claude 3.5:

| Training stage                   | Share of total cost | GPU hours (70B tier) | Main cost driver                       |
| -------------------------------- | ------------------- | -------------------- | -------------------------------------- |
| Pretraining                      | 60%–75%             | 4M–5M                | dense forward + backward               |
| Continued pretraining (CPT)      | 5%–10%              | 300K–500K            | long context + domain data             |
| SFT (supervised fine-tuning)     | 3%–5%               | 200K–350K            | short-sequence forward + backward      |
| Reward model (RM) training       | 1%–2%               | 50K–100K             | similar to SFT                         |
| RLHF / PPO                       | 10%–20%             | 600K–1.2M            | rollout (generation) is the bottleneck |
| RLVR (GRPO / DAPO)               | 5%–15%              | 300K–800K            | rollout + verifier compute             |
| DPO / preference tuning          | 1%–3%               | 50K–200K             | cheaper than RLHF, no rollout          |
| Offline evaluation + experiments | 5%–10%              | 300K–600K            | many benchmarks run in parallel        |

### RLHF Training Token Volume and GPU Hours

| Model tier | SFT sample count | RLHF rollout token count | GPU hours per round |
| ---------- | ---------------- | ------------------------ | ------------------- |
| 7B         | 100K–500K pairs  | 5B–20B generated tokens  | 30K–80K             |
| 13B        | 200K–800K pairs  | 10B–30B tokens           | 60K–150K            |
| 70B        | 1M–3M pairs      | 30B–100B tokens          | 500K–1.2M           |
| 405B       | 3M–10M pairs     | 100B–300B tokens         | 3M–8M               |

::: details Why is RLHF so much more expensive than SFT?
One SFT step — one forward plus backward pass — processes a fixed prompt-target pair, at a cost roughly equal to one token of pretraining. One round of RLHF includes:

1. Actor rollout (generating a 1–4K token response)
2. Critic forward + backward
3. Reward model forward
4. Reference model forward (to compute KL)
5. PPO/GRPO update

Total compute per token is roughly **30–100 times** that of SFT. This is why RLHF's share of total cost rose from 5% in 2022 to 30% in 2026.
:::

### The Training Cost of RLVR (DeepSeek-R1 Style)

DeepSeek-R1 reports that its RL stage (R1-Zero + R1) cost about 128K H800 GPU hours total, not counting pretraining of the base model. That number is surprisingly small, for these reasons:

| Key factor                                                   | Why it saves cost          |
| ------------------------------------------------------------ | -------------------------- |
| The base model is already V3 (no backbone retraining needed) | saves 90%+ of the compute  |
| Rule-based reward (math verification, code execution)        | no need to train an RM     |
| GRPO has no critic                                           | cuts ~40% of the compute   |
| Curriculum learning + difficulty-based sampling              | improves token utilization |

::: tip Why the R1 route is cheap
R1's engineering contribution is proving that **starting from an already-strong base model, pure RL — with no SFT warmup — is enough to trigger long-CoT reasoning**. That means as long as you have a V3-tier base model, a few tens of thousands of GPU hours can get you an R1-tier reasoning model. This is the root reason the open-source community produced so many R1 reproductions through 2025.
:::

## G.3 Public Training Data Reference

The table below collects **publicly disclosed training data from tech reports** current through 2026, as an anchor point for budget planning. Every number here comes from a vendor's public report or the scaling-law estimate it cites.

### DeepSeek Series

| Item                            | Data                                        | Source                         |
| ------------------------------- | ------------------------------------------- | ------------------------------ |
| DeepSeek-V2 pretraining         | 8.1T tokens / 2.8M H800 hours               | DeepSeek-V2 tech report        |
| DeepSeek-V3 pretraining         | 14.8T tokens / 2.664M H800 hours            | DeepSeek-V3 tech report        |
| DeepSeek-V3 total training cost | ~$5.576M (GPU rental only, H800 at $2/h)    | DeepSeek-V3 tech report        |
| DeepSeek-R1 RL stage            | ~128K H800 hours (RL on top of the V3 base) | DeepSeek-R1 tech report        |
| DeepSeek-R1-Zero RL stage       | ~80K H800 hours (no SFT warmup)             | DeepSeek-R1 tech report        |
| DeepSeek-Prover-V2              | not disclosed; estimated ~50K–80K GPU hours | DeepSeek-Prover-V2 tech report |

::: details DeepSeek-V3 cost breakdown
The $5.576M DeepSeek-V3 reports breaks down as:

- Pretraining: 2.664M GPU hours × $2/h = $5.33M
- Post-training (SFT + RL): about 12K GPU hours
- Validation and ablations: about 8K GPU hours

Re-estimated at the market rate of $3/h for H800, the real cost is around $8M.
:::

### Qwen Series

| Item                          | Data                                                         | Source                               |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------ |
| Qwen2.5 7B pretraining        | 18T tokens                                                   | Qwen2.5 tech report                  |
| Qwen2.5 72B pretraining       | 18T tokens                                                   | Qwen2.5 tech report                  |
| Qwen3 full-series pretraining | 36T tokens (largest model: 235B-A22B)                        | Qwen3 tech report (arXiv:2505.09388) |
| Qwen3 post-training           | 4 stages: SFT → cold start → RL → synthetic data             | Qwen3 tech report                    |
| Qwen3 RL stage cost           | not disclosed; estimated ~500K–800K GPU hours (largest tier) | estimated                            |

::: warning Qwen3's four-stage post-training
The Qwen3 tech report describes an elaborate four-stage post-training pipeline (including cold start, RL, and synthetic data augmentation), and total post-training cost may exceed 10% of pretraining cost. This is the trend in reasoning-model training through 2025 — **post-training is no longer a "small tail" tacked onto pretraining**.
:::

### Kimi Series

| Item                        | Data                                      | Source                                   |
| --------------------------- | ----------------------------------------- | ---------------------------------------- |
| Kimi K2 pretraining         | 15.5T tokens (1T MoE)                     | Kimi K2 tech report (arXiv:2507.20534)   |
| Kimi K2 total training cost | ~$25M (MoE training + post-training)      | Kimi K2 tech report                      |
| Kimi K2 RL stage            | not disclosed; estimated ~1M–2M GPU hours | estimated                                |
| Kimi K2.5                   | not disclosed (next generation)           | Kimi K2.5 tech report (arXiv:2602.02276) |

### Llama Series

| Item                       | Data                                           | Source                |
| -------------------------- | ---------------------------------------------- | --------------------- |
| Llama 2 7B pretraining     | 2.0T tokens / 184K A100 hours                  | Llama 2 tech report   |
| Llama 2 70B pretraining    | 1.7T tokens / 1.7M A100 hours                  | Llama 2 tech report   |
| Llama 3 70B pretraining    | 15T tokens / ~6.4M H100 hours                  | Llama 3 tech report   |
| Llama 3.1 405B pretraining | 15T tokens / ~30M H100 hours (16K-GPU cluster) | Llama 3.1 tech report |

### Other Public Models

| Item         | Data                                      | Source                 |
| ------------ | ----------------------------------------- | ---------------------- |
| Mistral 7B   | ~8T tokens / ~700K A100 hours             | Mistral 7B tech report |
| Mixtral 8×7B | not disclosed; estimated ~2M A100 hours   | Mixtral tech report    |
| Step-2       | 1T parameters / token count not disclosed | StepFun                |
| GLM-4.6      | training details not disclosed            | Zhipu AI, 2025         |

## G.4 Budgeting Your Own Training Run

Now convert the public numbers above into **three budget tiers for training your own model**. This section assumes you're a researcher or a small team whose goal is to **reproduce and improve on an open-source baseline**, not to train a trillion-parameter model from scratch.

### Single-GPU / Small-Scale Experiments (0.5B–1.5B models)

Good for learning the full RLHF/RLVR/DPO pipeline; you can finish one complete training run within a week.

| Resource      | Configuration                                | Cost              |
| ------------- | -------------------------------------------- | ----------------- |
| GPU           | 1× A100 80GB or 1× H100 80GB                 | $2.5–$3.5/h       |
| Model scale   | 0.5B–1.5B (e.g. Qwen2.5-0.5B, Llama-3.2-1B)  | -                 |
| Data          | 1K–10K SFT samples + 1M–5M RL rollout tokens | -                 |
| Training time | 1–5 days                                     | ~50–100 GPU hours |
| Total cost    | $100–$500                                    | -                 |
| Framework     | TRL, verl, OpenRLHF, LLaMA-Factory           | -                 |

::: tip Recommended starter tasks

- Reproduce R1-Zero's GRPO training curve on GSM8K ([Chapter 7](../chapter18_grpo/intro))
- Fine-tune with DPO on the Anthropic HH-RLHF dataset ([Chapter 2](../chapter17_dpo/intro))
- Run SAC/TD3 on CartPole / MuJoCo ([Chapter 10](../chapter11_continuous_control/intro))
  :::

### Multi-GPU Experiments (7B–13B models)

Good for reproducing baselines from mainstream papers (R1, DPO, GRPO); needs 1–2 weeks for one complete training run.

| Resource      | Configuration                                             | Cost                   |
| ------------- | --------------------------------------------------------- | ---------------------- |
| GPU           | 4×–8× A100 80GB or 4×–8× H100 80GB                        | $20–$50/h (whole node) |
| Model scale   | 7B–13B (e.g. Qwen2.5-7B, Llama-3-8B, DeepSeek-V2-Lite)    | -                      |
| Data          | 100K–1M SFT samples + 10B–50B RL rollout tokens           | -                      |
| Training time | 1–3 weeks (including multiple experiments)                | ~5K–20K GPU hours      |
| Total cost    | $10K–$80K                                                 | -                      |
| Framework     | OpenRLHF, verl, TRL + DeepSpeed / Megatron                | -                      |
| Key challenge | memory (7B + long context), rollout speed, KL computation | -                      |

::: warning The real cost of the mid-tier
This tier is the easiest one to **blow the budget on**. Why:

1. **Multiple experiments.** Your first RLHF run will almost certainly fail (reward hacking, training divergence) — it typically takes 3–5 iterations to stabilize.
2. **Rollout is slow.** Rollout accounts for 60%–80% of total RLHF time. Accelerating it with vLLM/SGLang is not optional.
3. **Evaluation cost.** Every checkpoint needs to run on AIME/MATH/HumanEval, and benchmark evaluation alone can eat up 20% of your GPU hours.

Budget rule of thumb: plan for **5–10 times** the cost of a single training run as your project budget.
:::

### 70B+ Cluster Experiments

Suited to industrial-scale training or large academic research. This tier needs a dedicated cluster and a team.

| Resource      | Configuration                                                        | Cost                      |
| ------------- | -------------------------------------------------------------------- | ------------------------- |
| GPU           | 64×–256× H100/H800 80GB (8–32 nodes of 8 GPUs each)                  | $1,000–$5,000/h (cluster) |
| Model scale   | 70B+ dense or 30B+ MoE                                               | -                         |
| Data          | 1M+ SFT samples + 100B+ RL rollout tokens                            | -                         |
| Training time | 2–8 weeks (including ablations and restarts)                         | ~500K–5M GPU hours        |
| Total cost    | $2M–$20M+                                                            | -                         |
| Framework     | Megatron-LM, DeepEP, veRL, Ray + in-house infra                      | -                         |
| Key challenge | communication, fault tolerance, checkpoint management, eval pipeline | -                         |

::: details Compute breakdown of a 70B RLHF run
Take one complete RLHF run (100K PPO steps) on a 70B model as an example:

- Actor forward + backward: 30% of GPU time
- Critic forward + backward: 20%
- Reference model forward: 10%
- Reward model forward: 5%
- **Rollout (generation)**: **35%**

This means that if your rollout engine isn't optimized — say, you haven't used vLLM/SGLang — the cost of a single RLHF experiment can double. That's why frameworks like OpenRLHF and verl treat rollout-engine integration as a first-class concern.
:::

### Three-Tier Comparison Table

| Dimension       | Starter tier                  | Mid tier                           | Large tier                          |
| --------------- | ----------------------------- | ---------------------------------- | ----------------------------------- |
| Model scale     | 0.5B–1.5B                     | 7B–13B                             | 70B+                                |
| Number of GPUs  | 1                             | 4–8                                | 64–256                              |
| Total GPU hours | 50–100                        | 5K–20K                             | 500K–5M                             |
| Total cost      | $100–$500                     | $10K–$80K                          | $2M–$20M                            |
| Training cycle  | 1–5 days                      | 1–3 weeks                          | 2–8 weeks                           |
| Suited for      | learning, small reproductions | reproducing paper baselines        | industrial-scale training           |
| Risk level      | low (cheap to fail)           | medium (needs multiple iterations) | high (every experiment burns money) |

## G.5 Cost Optimization Checklist

Whichever tier you're in, these techniques cut cost significantly:

### Pretraining Stage

1. **Use MoE instead of dense.** DeepSeek-V3's 671B-A37B gets results close to a 70B dense model, at roughly the compute cost of a 37B model.
2. **Mixed-precision training.** BF16 + FP8 (already used by DeepSeek-V3) cuts memory and compute by 30%–50%.
3. **Sequence packing.** Concatenating multiple short samples into one long sequence drops padding waste from 30% to 5%.
4. **Data curriculum.** Train on easy examples before hard ones, reducing wasted token training.

### Post-Training Stage

1. **Rollout acceleration.** vLLM / SGLang speed up rollout by 3–10x.
2. **Off-policy reuse.** Reuse old rollouts via importance sampling (this is what GRPO's group sampling relies on).
3. **DPO instead of RLHF.** When you don't need a complex reward signal, DPO is 10–50x cheaper than PPO.
4. **Verifiers instead of an RM.** For math/code tasks, use a rule-based verifier (Lean, unit tests) — no need to train an RM.
5. **Curriculum learning.** Sample in order of increasing difficulty to improve token utilization.

### Evaluation and Experimentation

1. **Small-model ablations.** Run hyperparameter search on a 1B model, then transfer the settings to the large model.
2. **Early stopping.** Use KL divergence from reward shaping or reward plateauing as a stop signal.
3. **Shared checkpoints.** Start multiple experiments from the same SFT checkpoint, saving the SFT cost each time.

::: tip Choosing a cloud GPU

- **A100 80GB**: best price-performance, good for the mid and starter tiers
- **H100 80GB**: 2–3x faster training, 40% higher unit price — good for the mid tier when budget allows
- **H800 80GB**: purchasable in China, slightly weaker than H100 (halved NVLink bandwidth) — good for large-tier teams in China
- **B200**: new as of 2025, 2.5x the BF16 compute of H100, priced around $6/h — good for very large-scale training
- **L40S / A10**: inference-oriented, not suited to training
  :::

## Chapter Summary

The cost estimates in this appendix serve one core judgment call: **is your experiment worth running**. Once you've internalized the tables above, the next time you read a paper claiming "we propose method X," you should be able to immediately convert that into "training X will cost roughly how many GPU hours, how many weeks, and how many failed experiments." This engineering instinct matters more than any algorithmic detail — it determines whether you can produce meaningful research within your compute budget.

Suggested next steps:

- **Run a starter-tier experiment**: follow the GRPO/DPO code in [Appendix D: Code Cheatsheet](../appendix_code_cheatsheet/intro) and run it on a single GPU.
- **Plan a mid-tier experiment**: see the distributed training and monitoring sections in [Appendix B: Engineering Practice](../appendix_industrial_training/intro).
- **Read the cost disclosures in frontier papers**: look for training details in the tech reports covered in [Appendix F](../appendix_paper_reading/intro).
