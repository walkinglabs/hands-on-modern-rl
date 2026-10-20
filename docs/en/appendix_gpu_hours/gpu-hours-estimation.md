# C.2 GPU Hour Estimation Table

> In engineering practice, the most frequently asked question is: "How many GPU hours and how much money will it cost to train this model?" — This is not a question from a compute company's sales team, but rather a core constraint in assessing the feasibility of a research plan. This appendix compiles the pre-training and fine-tuning costs from publicly available tech reports (DeepSeek, Qwen, Kimi, Llama, Claude) into a searchable table, and provides three tiers of self-training budget planning.

> **Reading recommendation:** Jump directly to the table you need. For budget planning, start with [C.2.4](#budget-planning). All numbers come from public technical reports or scaling-law estimates and **do not include unpublished internal data**.

## C.2.1 Pre-training Costs for Different Model Sizes

Pre-training costs are determined by three factors:

$$\text{GPU hours} \approx \frac{6 \cdot N \cdot D}{\text{Hardware Compute Utilization (MFU)} \cdot \text{GPU Single-card FLOPS}}$$

Where $N$ is the number of model parameters, $D$ is the number of training tokens, and MFU (Model FLOPs Utilization) typically ranges from 30% to 55%. The following table summarizes the training token counts and corresponding GPU hours from publicly available tech reports.

| Model           | Parameters | Training Tokens | GPU Type   | GPU Hours   | Data Source                 |
| --------------- | ---------- | --------------- | ---------- | ----------- | --------------------------- |
| Llama 2 7B      | 7B         | 2.0T            | A100-80G   | 184,320     | Meta 2023                   |
| Llama 2 13B     | 13B        | 2.0T            | A100-80G   | 432,000     | Meta 2023                   |
| Llama 2 70B     | 70B        | 1.7T            | A100-80G   | 1,700,000   | Meta 2023                   |
| Llama 3 8B      | 8B         | 15T             | H100-80G   | 130,000     | Meta 2024                   |
| Llama 3 70B     | 70B        | 15T             | H100-80G   | 6,400,000   | Meta 2024                   |
| Llama 3.1 405B  | 405B       | 15T             | H100-80G   | 30,000,000  | Meta 2024 (16K GPU Cluster) |
| DeepSeek-V2     | 236B-A21B  | 8.1T            | H800-80G   | 2,800,000   | DeepSeek 2024               |
| DeepSeek-V3     | 671B-A37B  | 14.8T           | H800-80G   | 2,664,000   | DeepSeek 2024               |
| Qwen2.5 7B      | 7B         | 18T             | Not Public | ~1,000,000  | Qwen 2024 (Estimate)        |
| Qwen2.5 72B     | 72B        | 18T             | Not Public | ~5,000,000  | Qwen 2024 (Estimate)        |
| Qwen3 235B-A22B | 235B-A22B  | 36T             | Not Public | ~14,000,000 | Qwen 2025 (Estimate)        |
| Kimi K2         | 1T-A32B    | 15.5T           | H800-80G   | ~9,000,000  | Kimi 2025 (Estimate)        |

::: tip Two Key Takeaways from This Table

1. **MoE Dramatically Reduces Activation Computation**: DeepSeek-V3 has a total parameter count of 671B, but its activation count is only 37B, which is equivalent to a dense model with 60B–70B parameters. However, it still requires 2.66M GPU hours.
2. **Token Count is the Determining Factor**: Llama 3 70B and Llama 2 70B have the same number of parameters, but the token count increased from 1.7T to 15T, leading to a fourfold increase in GPU hours. **All mainstream models trained in 2024 and onwards have training token counts exceeding 10T**, surpassing the Chinchilla ratio ($D \approx 20N$).
   :::

### Cost Estimation (Based on Public Cloud Pricing)

| Model Size | GPU Hours | A100 @ $2.5/h | H100 @ $3.5/h | H800 @ $3.0/h | B200 @ $6.0/h |
| ---------- | --------- | ------------- | ------------- | ------------- | ------------- |
| 7B dense   | ~200K     | $0.5M         | $0.7M         | $0.6M         | $1.2M         |
| 70B dense  | ~5M       | $12.5M        | $17.5M        | $15M          | $30M          |
| 405B dense | ~30M      | $75M          | $105M         | $90M          | $180M         |
| 671B MoE   | ~2.7M     | $6.8M         | $9.5M         | $8M           | $16M          |
| 1T MoE     | ~9M       | $22.5M        | $31.5M        | $27M          | $54M          |

::: warning The Actual Cost is Much Higher than the Table Above
The table only accounts for the **bare GPU rental cost**. Real training also includes: (1) Storage/network/power costs, approximately +30%, (2) Multiple failed experiments and hyperparameter search, approximately ×3–5, (3) Data collection and annotation, approximately 10%–20%. A publicly reported "$15M" model, for example, often costs a company $50M–$100M in reality.
:::

## C.2.2 Cost of Each Stage: SFT / RLHF / RLVR

Pretraining accounts for only part of the cost. The **post-training** (SFT, RLHF, RLVR) GPU hours have increased from 5% in 2022 to over 30% in 2026 — this is because the rollout in RLHF/RLVR is significantly slower than a single forward pass.

The following table is based on the publicly available data of **stage cost proportions** for DeepSeek-V3 / R1, Qwen3, Llama 3.1, and Claude 3.5:

| Training Stage                   | Proportion of Total Training Cost | GPU Hours (70B Scale) | Main Expenses                          |
| -------------------------------- | --------------------------------- | --------------------- | -------------------------------------- |
| Pretraining                      | 60%–75%                           | 4M–5M                 | dense forward + backward               |
| Continuous Pretraining (CPT)     | 5%–10%                            | 300K–500K             | long context + domain data             |
| SFT (Supervised Fine-tuning)     | 3%–5%                             | 200K–350K             | short sequence forward + backward      |
| Reward Model Training (RM)       | 1%–2%                             | 50K–100K              | Similar to SFT                         |
| RLHF / PPO                       | 10%–20%                           | 600K–1.2M             | Rollout (generation) is the bottleneck |
| RLVR (GRPO / DAPO)               | 5%–15%                            | 300K–800K             | Rollout + verifier computation         |
| DPO / Preference                 | 1%–3%                             | 50K–200K              | Cheaper than RLHF, no rollout          |
| Offline Evaluation + Experiments | 5%–10%                            | 300K–600K             | Multiple benchmarks in parallel        |

### RLHF Training Token Count and GPU Hours

| Model Scale | SFT Sample Count | RLHF Rollout Token Count | Per Round GPU Hours |
| ----------- | ---------------- | ------------------------ | ------------------- |
| 7B          | 100K–500K pairs  | 5B–20B generated tokens  | 30K–80K             |
| 13B         | 200K–800K pairs  | 10B–30B tokens           | 60K–150K            |
| 70B         | 1M–3M pairs      | 30B–100B tokens          | 500K–1.2M           |
| 405B        | 3M–10M pairs     | 100B–300B tokens         | 3M–8M               |

::: details Why is RLHF Cost Much Higher than SFT?
SFT processes one fixed prompt-target pair with a single forward + backward pass, with a cost approximately equivalent to 1 token of pre-training. RLHF involves:

1. Actor rollout (generating 1–4K tokens of response)
2. Critic forward + backward
3. Reward model forward
4. Reference model forward (calculating KL)
5. PPO/GRPO update

The total computational cost is approximately **30–100 times higher per token** compared to SFT. This is why the cost ratio of RLHF has risen from 5% in 2022 to 30% in 2026.
:::

### Training Cost of RLVR (DeepSeek-R1 Style)

DeepSeek-R1 reports that its RL phase (R1-Zero + R1) has a total cost of approximately 128K H800 GPU hours (excluding the pre-training of the base model). This number is surprisingly low, due to the following reasons:

| Key Factor                                             | Explanation                |
| ------------------------------------------------------ | -------------------------- |
| Base model is already V3 (no need to retrain backbone) | Saves 90%+ computation     |
| Rule-based reward (math verification, code execution)  | No need to train RM        |
| GRPO without critic                                    | Reduces ~40% computation   |
| Curriculum learning + difficulty sampling              | Improves token utilization |

::: tip Why is the R1 approach cost-effective?
The engineering value of R1 lies in demonstrating: **Long CoT reasoning can be triggered purely through RL (without SFT warmup) on an already strong base model.** This means that with just a V3-level base model, a few tens of thousands of GPU hours are sufficient to obtain an R1-level reasoning model. This is the fundamental reason why the open-source community has seen a surge of R1 reproductions by 2025.

## C.2.3 Reference of Public Training Data

The following table summarizes the **public tech report training data** up to 2026, serving as a reference point for budget planning. All figures are derived from the manufacturers' public reports or their referenced scaling law estimates.

### DeepSeek Series

| Item                            | Data                                         | Source                         |
| ------------------------------- | -------------------------------------------- | ------------------------------ |
| DeepSeek-V2 Pretraining         | 8.1T tokens / 2.8M H800 hours                | DeepSeek-V2 tech report        |
| DeepSeek-V3 Pretraining         | 14.8T tokens / 2.664M H800 hours             | DeepSeek-V3 tech report        |
| DeepSeek-V3 Total Training Cost | ~$5.576M (only GPU rental cost, H800 × $2/h) | DeepSeek-V3 tech report        |
| DeepSeek-R1 RL Phase            | ~128K H800 hours (RL on V3 base)             | DeepSeek-R1 tech report        |
| DeepSeek-R1-Zero RL Phase       | ~80K H800 hours (no SFT warmup)              | DeepSeek-R1 tech report        |
| DeepSeek-Prover-V2              | Not disclosed, estimated ~50K–80K GPU hours  | DeepSeek-Prover-V2 tech report |

::: details Decomposition of DeepSeek-V3 Cost
The reported $5.576M for DeepSeek-V3 includes:

- Pretraining: 2.664M GPU hours × $2/h = $5.33M
- Post-training (SFT + RL): Approximately 12K GPU hours
- Validation and ablation: Approximately 8K GPU hours

Reestimated at the H800 market rate of $3/h, the actual cost is approximately $8M.
:::

### Qwen Series

| Project                       | Data                                                           | Source                               |
| ----------------------------- | -------------------------------------------------------------- | ------------------------------------ |
| Qwen2.5 7B Pretraining        | 18T tokens                                                     | Qwen2.5 Tech Report                  |
| Qwen2.5 72B Pretraining       | 18T tokens                                                     | Qwen2.5 Tech Report                  |
| Qwen3 Full Series Pretraining | 36T tokens (Maximum 235B-A22B)                                 | Qwen3 Tech Report (arXiv:2505.09388) |
| Qwen3 Post-training           | 4 Stages: SFT → Cold Start → RL → Synthetic Data               | Qwen3 Tech Report                    |
| Qwen3 RL Stage Cost           | Not disclosed, estimated at 500K–800K GPU hours (Maximum Tier) | Estimate                             |

::: warning Qwen3's 4-Stage Post-training
The Qwen3 tech report describes a complex 4-stage post-training process (including cold start, RL, and synthetic data augmentation). The total post-training cost may exceed 10% of the pretraining cost. This reflects the 2025 trend in reasoning model training — **post-training is no longer just a small tail of pretraining**.
:::

### Kimi Series

| Project               | Data                                     | Source                                 |
| --------------------- | ---------------------------------------- | -------------------------------------- |
| Kimi K2 Pretraining   | 15.5T tokens (1T MoE)                    | Kimi K2 tech report (arXiv:2507.20534) |
| Kimi K2 Training Cost | ~$25M (MoE training + fine-tuning)       | Kimi K2 tech report                    |
| Kimi K2 RL Phase      | Not disclosed, estimated 1M–2M GPU hours | Estimation                             |
| Kimi K2.5             | Not disclosed (next generation)          | Kimi K5 tech report (arXiv:2602.02276) |

### Llama Series

| Project                    | Data                                           | Source                |
| -------------------------- | ---------------------------------------------- | --------------------- |
| Llama 2 7B Pretraining     | 2.0T tokens / 184K A100 hours                  | Llama 2 tech report   |
| Llama 2 70B Pretraining    | 1.7T tokens / 1.7M A100 hours                  | Llama 2 tech report   |
| Llama 3 70B Pretraining    | 15T tokens / ~6.4M H100 hours                  | Llama 3 tech report   |
| Llama 3.1 405B Pretraining | 15T tokens / ~30M H100 hours (16K GPU cluster) | Llama 3.1 tech report |

### Other Public Models

| Project      | Data                                      | Source                 |
| ------------ | ----------------------------------------- | ---------------------- |
| Mistral 7B   | ~8T tokens / ~700K A100 hours             | Mistral 7B tech report |
| Mixtral 8×7B | Not disclosed, estimated ~2M A100 hours   | Mixtral tech report    |
| Step-2       | 1T parameters / Not disclosed token count | StepFun                |
| GLM-4.6      | Not disclosed training details            | Zhipu 2025             |

## C.2.4 Budget Planning for Self-Trained Models {#budget-planning}

Convert the above public figures into **three tiers of budgets for self-trained models**. This section assumes the reader is a researcher or a small team aiming to **reproduce and improve an open-source baseline**, rather than training a model with trillions of parameters from scratch.

### Single-GPU / Small-Scale Experiment (0.5B–1.5B Model)

Suitable for learning the full process of RLHF/RLVR/DPO, and can complete a full training within a week.

| Resource      | Configuration                                | Cost              |
| ------------- | -------------------------------------------- | ----------------- |
| GPU           | 1× A100 80GB or 1× H100 80GB                 | $2.5–$3.5/h       |
| Model Size    | 0.5B–1.5B (e.g., Qwen2.5-0.5B, Llama-3.2-1B) | -                 |
| Data          | 1K–10K SFT samples + 1M–5M RL rollout tokens | -                 |
| Training Time | 1–5 days                                     | ~50–100 GPU hours |
| Total Cost    | $100–$500                                    | -                 |
| Framework     | TRL, verl, OpenRLHF, LLaMA-Factory           | -                 |

::: tip Recommended Tasks for Beginners

- Reproduce the training curves of R1-Zero on GSM8K using GRPO ([Chapter 15](../chapter18_grpo/grpo-practice-and-mechanism))
- Fine-tune on Anthropic HH-RLHF data using DPO ([Chapter 14](../chapter17_dpo/dpo-objective-derivation))
- Run SAC/TD3 on CartPole / MuJoCo ([Chapter 9](../chapter11_continuous_control/ddpg))
  :::

### Multi-GPU Experiments (7B–13B Models)

Suitable for reproducing mainstream paper baselines (e.g., R1, DPO, GRPO), and requires 1–2 weeks to complete a full training run.

| Resource       | Configuration                                                  | Cost                    |
| -------------- | -------------------------------------------------------------- | ----------------------- |
| GPU            | 4×–8× A100 80GB or 4×–8× H100 80GB                             | $20–$50/h (per machine) |
| Model Size     | 7B–13B (e.g., Qwen2.5-7B, Llama-3-8B, DeepSeek-V2-Lite)        | -                       |
| Data           | 100K–1M SFT samples + 10B–50B RL rollout tokens                | -                       |
| Training Time  | 1–3 weeks (including multiple experiments)                     | ~5K–20K GPU hours       |
| Total Cost     | $10K–$80K                                                      | -                       |
| Framework      | OpenRLHF, verl, TRL + DeepSpeed / Megatron                     | -                       |
| Key Challenges | VRAM (7B + long context), rollout acceleration, KL computation | -                       |

::: warning Real Cost of the Mid-Range Tier
The mid-range tier is the easiest to **go over budget**. Reasons:

- High computational demands for large-scale models.
- Increased complexity in training and hyperparameter tuning.
- Potential for unexpected costs due to resource contention or inefficiencies in distributed training.
- The need for additional resources to handle data preprocessing, model checkpointing, and logging.

1. **Multiple Experiments**: The first RLHF is almost certainly going to fail (reward hacking, training divergence), and it typically takes at least 3–5 iterations to stabilize.

2. **Slow Rollout**: In RLHF, rollout accounts for 60%–80% of the total time. Accelerating with vLLM/SGLang is essential.

3. **Evaluation Cost**: Each checkpoint requires running AIME/MATH/HumanEval benchmarks, and benchmark evaluation may consume 20% of GPU hours.

Budget Recommendation: Allocate **5–10 times** the cost of a single training run as the project budget.
:::

### 70B+ Cluster Experiments

Suitable for industrial-scale training or large-scale academic research. This section requires a dedicated cluster and a team.

| Resource       | Configuration                                                              | Cost                      |
| -------------- | -------------------------------------------------------------------------- | ------------------------- |
| GPU            | 64×–256× H100/H800 80GB (8–32 nodes with 8 GPUs each)                      | $1,000–$5,000/h (cluster) |
| Model Size     | 70B+ dense or 30B+ MoE                                                     | -                         |
| Data           | 1M+ SFT samples + 100B+ RL rollout tokens                                  | -                         |
| Training Time  | 2–8 weeks (including ablation and restarts)                                | ~500K–5M GPU hours        |
| Total Cost     | $2M–$20M+                                                                  | -                         |
| Framework      | Megatron-LM, DeepEP, veRL, Ray + in-house infrastructure                   | -                         |
| Key Challenges | Communication, fault tolerance, checkpoint management, evaluation pipeline | -                         |

::: details Power Consumption Breakdown for 70B RLHF
As an example, consider one full RLHF run (100,000 steps of PPO) for a 70B model:

- Actor forward + backward: 30% GPU time
- Critic forward + backward: 20%
- Reference model forward: 10%
- Reward model forward: 5%
- **Rollout (generation)**: **35%**

This means that if your rollout engine is not optimized (e.g., not using vLLM/SGLang), the cost of a single RLHF experiment may double. This is also why frameworks like OpenRLHF and verl consider integrating the rollout engine as a first-class citizen.

:::

### Three-Tier Comparison Table

| Dimension       | Entry-Level Tier            | Mid-Tier                              | Large-Scale Tier                 |
| --------------- | --------------------------- | ------------------------------------- | -------------------------------- |
| Model Size      | 0.5B–1.5B                   | 7B–13B                                | 70B+                             |
| Number of GPUs  | 1                           | 4–8                                   | 64–256                           |
| Total GPU Hours | 50–100                      | 5K–20K                                | 500K–5M                          |
| Total Cost      | $100–$500                   | $10K–$80K                             | $2M–$20M                         |
| Training Cycle  | 1–5 days                    | 1–3 weeks                             | 2–8 weeks                        |
| Suitable Tasks  | Learning, small experiments | Paper baseline reproduction           | Industrial-scale training        |
| Risk Level      | Low (low cost of failure)   | Medium (requires multiple iterations) | High (each experiment is costly) |

## C.2.5 Cost Optimization Checklist

Regardless of the scale, the following techniques can significantly reduce costs:

### Pre-training Phase

1. **Use MoE instead of dense**: DeepSeek-V3 achieves performance close to 70B dense models using 671B-A37B, while maintaining a computational cost comparable to 37B.
2. B**F16 + FP8 mixed precision training**: Used in DeepSeek-V3, this reduces memory and computation by 30%–50%.
3. **Sequence packing**: Combine multiple short samples into a single long sequence, reducing padding waste from 30% to 5%.
4. **Curriculum learning**: Start with easy data and gradually introduce harder data, reducing the training of invalid tokens.

### Post-training Phase

1. **Rollout acceleration**: vLLM / SGLang can speed up rollouts by 3–10 times.
2. **Off-policy reuse**: Reuse old rollouts using importance sampling (refer to GRPO's population sampling).
3. **DPO instead of RLHF**: When complex reward signals are not needed, DPO is 10–50 times cheaper than PPO.
4. **Verifier instead of RM**: For mathematical and coding tasks, use rule-based verifiers (e.g., Lean, unit tests), eliminating the need to train RM.
5. **Curriculum learning**: Sample data in increasing order of difficulty to improve token utilization.

### Evaluation and Experimentation

1. **Small Model Ablation**: Conduct hyperparameter search on a 1B model, then transfer the findings to larger models. 2.**
   Early Stopping**: Use reward shaping's KL divergence or reward stagnation as stopping signals.
2. **Shared Checkpoint**: Multiple experiments start from the same SFT checkpoint, saving SFT costs.

::: tip GPU Selection for Cloud

- **A100 80GB**: Offers the best cost-performance ratio, suitable for mid-tier and entry-level models.
- **H100 80GB**: Training speed is 2–3 times faster than A100, but the price is 40% higher, suitable for mid-tier models with sufficient budget.
- **H800 80GB**: Available in China, performance is slightly lower than H100 (NVLink bandwidth is halved), suitable for large-scale models by Chinese teams.
- **B200**: A new model released in 2025, with BF16 performance 2.5 times that of H100, at a price of ~$6/h, suitable for ultra-large-scale training.
- **L40S / A10**: Primarily for inference, not suitable for training.
  :::

## Summary of This Chapter

This appendix provides cost estimates to serve a core judgment: **whether your experiment is worth doing**. After memorizing the above table, when you see any paper that says "we propose X method," you should be able to immediately calculate in your mind: "How many GPU hours does training X take? How many weeks? How many failed experiments?" This engineering intuition is more important than any algorithm detail—it determines whether you can conduct meaningful research within your computational budget.

Next Steps Recommendation:

- **Do a Baseline Experiment**: Refer to the GRPO/DPO code in [Appendix B Code Quick Reference](../appendix_code_cheatsheet/sft-kl) and run it on a single GPU.
- **Plan a Mid-Scale Experiment**: Refer to the diagnostic sequence and troubleshooting checklist in [Appendix A.1 Training Debugging Guide](../appendix_industrial_training/training-debugging).
- **Read the Cost Disclosure in Frontier Papers**: Look for training details in the tech reports in [Appendix C](../appendix_paper_reading/learning-resources).
