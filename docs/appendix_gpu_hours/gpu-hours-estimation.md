# C.2 GPU 小时估算表

> 工程实践中最常被问的问题是："训这个模型大概要多少 GPU 小时、烧多少钱？"——这不是算力公司销售的问题，而是研究方案可行性的核心约束。本附录把公开 tech report（DeepSeek、Qwen、Kimi、Llama、Claude）的预训练与后训练成本整理成可查表，再给出三档自训预算规划。
>
> 阅读建议：直接跳到所需小节查表。若需做预算规划，从 [C.2.4](#budget-planning) 起读。所有数字均来自公开 tech report 或基于 scaling law 的估算，**不包含未公开的内部数据**。

## C.2.1 不同模型规模的预训练成本

预训练成本由三个量决定：

$$\text{GPU 小时} \approx \frac{6 \cdot N \cdot D}{\text{硬件算力利用率 (MFU)} \cdot \text{GPU 单卡 FLOPS}}$$

其中 $N$ 为模型参数量，$D$ 为训练 token 数，MFU（Model FLOPs Utilization）通常在 30%–55% 之间。下面表格汇总公开 tech report 的训练 token 数与对应 GPU 小时。

| 模型            | 参数量    | 训练 token | GPU 类型 | GPU 小时    | 数据来源                 |
| --------------- | --------- | ---------- | -------- | ----------- | ------------------------ |
| Llama 2 7B      | 7B        | 2.0T       | A100-80G | 184,320     | Meta 2023                |
| Llama 2 13B     | 13B       | 2.0T       | A100-80G | 432,000     | Meta 2023                |
| Llama 2 70B     | 70B       | 1.7T       | A100-80G | 1,700,000   | Meta 2023                |
| Llama 3 8B      | 8B        | 15T        | H100-80G | 130,000     | Meta 2024                |
| Llama 3 70B     | 70B       | 15T        | H100-80G | 6,400,000   | Meta 2024                |
| Llama 3.1 405B  | 405B      | 15T        | H100-80G | 30,000,000  | Meta 2024 (16K GPU 集群) |
| DeepSeek-V2     | 236B-A21B | 8.1T       | H800-80G | 2,800,000   | DeepSeek 2024            |
| DeepSeek-V3     | 671B-A37B | 14.8T      | H800-80G | 2,664,000   | DeepSeek 2024            |
| Qwen2.5 7B      | 7B        | 18T        | 无公开   | ~1,000,000  | Qwen 2024 (估算)         |
| Qwen2.5 72B     | 72B       | 18T        | 无公开   | ~5,000,000  | Qwen 2024 (估算)         |
| Qwen3 235B-A22B | 235B-A22B | 36T        | 无公开   | ~14,000,000 | Qwen 2025 (估算)         |
| Kimi K2         | 1T-A32B   | 15.5T      | H800-80G | ~9,000,000  | Kimi 2025 (估算)         |

::: tip 解读这张表的两个要点

1. **MoE 大幅降低激活计算量**：DeepSeek-V3 总参数 671B 但激活只 37B，等效计算量约等于一个 60B–70B 的 dense 模型，但仍需 2.66M GPU 小时。
2. **token 数是决定性变量**：Llama 3 70B 与 Llama 2 70B 参数相同，但因 token 数从 1.7T 涨到 15T，GPU 小时翻 4 倍。**2024 年后所有主流模型的训练 token 都在 10T 以上**，Chinchilla ratio ($D \approx 20N$) 已被普遍超越。
   :::

### 成本估算（按公开云单价）

| 模型档位   | GPU 小时 | A100 @ $2.5/h | H100 @ $3.5/h | H800 @ $3.0/h | B200 @ $6.0/h |
| ---------- | -------- | ------------- | ------------- | ------------- | ------------- |
| 7B dense   | ~200K    | $0.5M         | $0.7M         | $0.6M         | $1.2M         |
| 70B dense  | ~5M      | $12.5M        | $17.5M        | $15M          | $30M          |
| 405B dense | ~30M     | $75M          | $105M         | $90M          | $180M         |
| 671B MoE   | ~2.7M    | $6.8M         | $9.5M         | $8M           | $16M          |
| 1T MoE     | ~9M      | $22.5M        | $31.5M        | $27M          | $54M          |

::: warning 真实成本远高于上表
上表只算**裸 GPU 租金**。真实训练还需叠加：(1) 存储/网络/电力约 +30%，(2) 多次失败实验与超参搜索约 ×3–5，(3) 数据采集与标注约 10%–20%。一个公开报告 "$15M" 的模型，公司实际投入往往在 $50M–$100M 量级。
:::

## C.2.2 SFT / RLHF / RLVR 各阶段成本

预训练只是成本的一部分。**后训练**（SFT、RLHF、RLVR）的 GPU 小时占比已从 2022 年的 5% 上升到 2026 年的 30% 以上——这是因为 RLHF/RLVR 的 rollout 远比一次 forward 慢。

下面表格基于 DeepSeek-V3 / R1、Qwen3、Llama 3.1、Claude 3.5 公开数据估算的**阶段成本占比**：

| 训练阶段            | 占总训练成本 | GPU 小时（70B 级） | 主要开销                  |
| ------------------- | ------------ | ------------------ | ------------------------- |
| 预训练              | 60%–75%      | 4M–5M              | dense forward + backward  |
| 持续预训练（CPT）   | 5%–10%       | 300K–500K          | 长上下文 + 领域数据       |
| SFT（监督微调）     | 3%–5%        | 200K–350K          | 短序列 forward + backward |
| 奖励模型训练（RM）  | 1%–2%        | 50K–100K           | 与 SFT 类似               |
| RLHF / PPO          | 10%–20%      | 600K–1.2M          | rollout（生成）是瓶颈     |
| RLVR（GRPO / DAPO） | 5%–15%       | 300K–800K          | rollout + verifier 计算   |
| DPO / preference    | 1%–3%        | 50K–200K           | 比 RLHF 便宜，无 rollout  |
| 离线评测 + 实验     | 5%–10%       | 300K–600K          | 多 benchmark 并发         |

### RLHF 训练 token 量与 GPU 小时

| 模型档位 | SFT 样本数      | RLHF rollout token 数   | 单轮 GPU 小时 |
| -------- | --------------- | ----------------------- | ------------- |
| 7B       | 100K–500K pairs | 5B–20B generated tokens | 30K–80K       |
| 13B      | 200K–800K pairs | 10B–30B tokens          | 60K–150K      |
| 70B      | 1M–3M pairs     | 30B–100B tokens         | 500K–1.2M     |
| 405B     | 3M–10M pairs    | 100B–300B tokens        | 3M–8M         |

::: details RLHF 成本为何远高于 SFT？
SFT 一次 forward + backward 处理一个固定 prompt-target 对，成本约等于预训练的 1 token。RLHF 一轮包含：

1. Actor rollout（生成 1–4K tokens 响应）
2. Critic forward + backward
3. Reward model forward
4. Reference model forward（计算 KL）
5. PPO/GRPO 更新

总计算量约为 SFT 的 **30–100 倍/token**。这是为什么 RLHF 成本占比从 2022 年的 5% 涨到 2026 年的 30%。
:::

### RLVR（DeepSeek-R1 风格）的训练成本

DeepSeek-R1 报告其 RL 阶段（R1-Zero + R1）总开销约 128K H800 GPU 小时（不含基础模型预训练）。这个数字令人惊讶地小，原因是：

| 关键因素                             | 说明              |
| ------------------------------------ | ----------------- |
| 基础模型已是 V3（无需重训 backbone） | 节省 90%+ 计算    |
| 规则奖励（数学验证、代码执行）       | 无需训练 RM       |
| GRPO 无 critic                       | 减少 ~40% 计算    |
| 课程学习 + 难度采样                  | 提高 token 利用率 |

::: tip R1 路线为什么便宜
R1 的工程价值在于证明：**在已有强基础模型上，纯 RL（无 SFT warmup）即可触发长 CoT 推理**。这意味着只要有一个 V3 级 base model，几十 K GPU 小时即可得到一个 R1 级推理模型。这是开源社区在 2025 年涌现大量 R1 复现的根本原因。
:::

## C.2.3 公开训练数据参考

下面表格汇总截至 2026 年的**公开 tech report 训练数据**，作为预算规划的锚点。所有数字均来自厂商公开报告或其引用的 scaling law 估算。

### DeepSeek 系列

| 项目                     | 数据                                   | 来源                           |
| ------------------------ | -------------------------------------- | ------------------------------ |
| DeepSeek-V2 预训练       | 8.1T tokens / 2.8M H800 小时           | DeepSeek-V2 tech report        |
| DeepSeek-V3 预训练       | 14.8T tokens / 2.664M H800 小时        | DeepSeek-V3 tech report        |
| DeepSeek-V3 训练总成本   | ~$5.576M（仅算 GPU 租金，H800 × $2/h） | DeepSeek-V3 tech report        |
| DeepSeek-R1 RL 阶段      | ~128K H800 小时（V3 base 上的 RL）     | DeepSeek-R1 tech report        |
| DeepSeek-R1-Zero RL 阶段 | ~80K H800 小时（无 SFT warmup）        | DeepSeek-R1 tech report        |
| DeepSeek-Prover-V2       | 未公开具体数字，估计 ~50K–80K GPU 小时 | DeepSeek-Prover-V2 tech report |

::: details DeepSeek-V3 成本分解
DeepSeek-V3 报告的 $5.576M 包括：

- 预训练：2.664M GPU 小时 × $2/h = $5.33M
- 后训练（SFT + RL）：约 12K GPU 小时
- 验证与消融：约 8K GPU 小时

按 H800 市场 $3/h 重新估，真实成本约 $8M。
:::

### Qwen 系列

| 项目               | 数据                                        | 来源                                 |
| ------------------ | ------------------------------------------- | ------------------------------------ |
| Qwen2.5 7B 预训练  | 18T tokens                                  | Qwen2.5 tech report                  |
| Qwen2.5 72B 预训练 | 18T tokens                                  | Qwen2.5 tech report                  |
| Qwen3 全系列预训练 | 36T tokens（最大 235B-A22B）                | Qwen3 tech report (arXiv:2505.09388) |
| Qwen3 后训练       | 4 阶段：SFT → 冷启动 → RL → 合成数据        | Qwen3 tech report                    |
| Qwen3 RL 阶段成本  | 未公开，估算约 500K–800K GPU 小时（最大档） | 估算                                 |

::: warning Qwen3 的 4 阶段后训练
Qwen3 tech report 描述了复杂的 4 阶段后训练（含冷启动、RL、合成数据增强），总后训练成本可能超过预训练的 10%。这是 2025 年推理模型训练的趋势——**后训练不再是预训练的"小尾巴"**。
:::

### Kimi 系列

| 项目               | 数据                          | 来源                                     |
| ------------------ | ----------------------------- | ---------------------------------------- |
| Kimi K2 预训练     | 15.5T tokens（1T MoE）        | Kimi K2 tech report (arXiv:2507.20534)   |
| Kimi K2 总训练成本 | ~$25M（MoE 训练 + 后训练）    | Kimi K2 tech report                      |
| Kimi K2 RL 阶段    | 未公开，估算约 1M–2M GPU 小时 | 估算                                     |
| Kimi K2.5          | 未公开（下一代）              | Kimi K2.5 tech report (arXiv:2602.02276) |

### Llama 系列

| 项目                  | 数据                                        | 来源                  |
| --------------------- | ------------------------------------------- | --------------------- |
| Llama 2 7B 预训练     | 2.0T tokens / 184K A100 小时                | Llama 2 tech report   |
| Llama 2 70B 预训练    | 1.7T tokens / 1.7M A100 小时                | Llama 2 tech report   |
| Llama 3 70B 预训练    | 15T tokens / ~6.4M H100 小时                | Llama 3 tech report   |
| Llama 3.1 405B 预训练 | 15T tokens / ~30M H100 小时（16K GPU 集群） | Llama 3.1 tech report |

### 其他公开模型

| 项目         | 数据                               | 来源                   |
| ------------ | ---------------------------------- | ---------------------- |
| Mistral 7B   | ~8T tokens / ~700K A100 小时       | Mistral 7B tech report |
| Mixtral 8×7B | 未公开具体数字，估算 ~2M A100 小时 | Mixtral tech report    |
| Step-2       | 1T 参数 / 未公开 token 数          | StepFun                |
| GLM-4.6      | 未公开训练细节                     | 智谱 2025              |

## C.2.4 自训模型预算规划 {#budget-planning}

把上述公开数字换算成**自训模型的三档预算**。本节假设读者是研究者或小团队，目标是**复现并改进某个开源 baseline**，而非从零训练万亿参数模型。

### 单卡 / 小规模实验（0.5B–1.5B 模型）

适合学习 RLHF/RLVR/DPO 全流程，可在一周内完成一次完整训练。

| 资源     | 配置                                       | 成本             |
| -------- | ------------------------------------------ | ---------------- |
| GPU      | 1× A100 80GB 或 1× H100 80GB               | $2.5–$3.5/h      |
| 模型规模 | 0.5B–1.5B（如 Qwen2.5-0.5B、Llama-3.2-1B） | -                |
| 数据     | 1K–10K SFT 样本 + 1M–5M RL rollout tokens  | -                |
| 训练时间 | 1–5 天                                     | ~50–100 GPU 小时 |
| 总成本   | $100–$500                                  | -                |
| 框架     | TRL、verl、OpenRLHF、LLaMA-Factory         | -                |

::: tip 入门档推荐任务

- 用 GRPO 复现 R1-Zero 在 GSM8K 上的训练曲线（[第 15 章](../chapter18_grpo/grpo-practice-and-mechanism)）
- 用 DPO 在 Anthropic HH-RLHF 数据上微调（[第 14 章](../chapter17_dpo/dpo-objective-derivation)）
- 在 CartPole / MuJoCo 上跑 SAC/TD3（[第 9 章](../chapter11_continuous_control/ddpg)）
  :::

### 多卡实验（7B–13B 模型）

适合复现主流论文 baseline（如 R1、DPO、GRPO），需要 1–2 周完成一次完整训练。

| 资源     | 配置                                                  | 成本              |
| -------- | ----------------------------------------------------- | ----------------- |
| GPU      | 4×–8× A100 80GB 或 4×–8× H100 80GB                    | $20–$50/h（整机） |
| 模型规模 | 7B–13B（如 Qwen2.5-7B、Llama-3-8B、DeepSeek-V2-Lite） | -                 |
| 数据     | 100K–1M SFT 样本 + 10B–50B RL rollout tokens          | -                 |
| 训练时间 | 1–3 周（含多次实验）                                  | ~5K–20K GPU 小时  |
| 总成本   | $10K–$80K                                             | -                 |
| 框架     | OpenRLHF、verl、TRL + DeepSpeed / Megatron            | -                 |
| 关键挑战 | 显存（7B + 长 context）、rollout 加速、KL 计算        | -                 |

::: warning 中型档的真实成本
中型档最容易**超预算**。原因：

1. **多次实验**：第一次 RLHF 几乎一定失败（reward hacking、训练发散），至少 3–5 次迭代才稳定。
2. **rollout 慢**：RLHF 中 rollout 占总时间的 60%–80%。用 vLLM/SGLang 加速是必须的。
3. **评测成本**：每个 checkpoint 都要跑 AIME/MATH/HumanEval，benchmark 评测可能花掉 GPU 小时的 20%。

预算建议：把单次训练成本的 **5–10 倍** 作为项目预算。
:::

### 70B+ 集群实验

适合工业级训练或大规模学术研究。本档需要专用集群与团队。

| 资源     | 配置                                        | 成本                    |
| -------- | ------------------------------------------- | ----------------------- |
| GPU      | 64×–256× H100/H800 80GB（8–32 台 8 卡节点） | $1,000–$5,000/h（集群） |
| 模型规模 | 70B+ dense 或 30B+ MoE                      | -                       |
| 数据     | 1M+ SFT 样本 + 100B+ RL rollout tokens      | -                       |
| 训练时间 | 2–8 周（含消融与重启）                      | ~500K–5M GPU 小时       |
| 总成本   | $2M–$20M+                                   | -                       |
| 框架     | Megatron-LM、DeepEP、veRL、Ray + 自研 infra | -                       |
| 关键挑战 | 通信、容错、checkpoint 管理、评测流水线     | -                       |

::: details 70B RLHF 的算力分解
以 70B 模型做一次完整 RLHF（10 万步 PPO）为例：

- Actor forward + backward：30% GPU 时间
- Critic forward + backward：20%
- Reference model forward：10%
- Reward model forward：5%
- **Rollout（生成）**：**35%**

这意味着如果你的 rollout engine 不优化（如未用 vLLM/SGLang），单 RLHF 实验的成本可能翻倍。这也是为什么 OpenRLHF、verl 等框架把 rollout engine 集成视为一等公民。
:::

### 三档对比总表

| 维度        | 入门档           | 中型档             | 大型档             |
| ----------- | ---------------- | ------------------ | ------------------ |
| 模型规模    | 0.5B–1.5B        | 7B–13B             | 70B+               |
| GPU 数      | 1                | 4–8                | 64–256             |
| 总 GPU 小时 | 50–100           | 5K–20K             | 500K–5M            |
| 总成本      | $100–$500        | $10K–$80K          | $2M–$20M           |
| 训练周期    | 1–5 天           | 1–3 周             | 2–8 周             |
| 适合任务    | 学习、复现小实验 | 论文 baseline 复现 | 工业级训练         |
| 风险等级    | 低（失败成本低） | 中（需多次迭代）   | 高（每次实验烧钱） |

## C.2.5 成本优化清单

无论哪一档，以下技巧可显著降低成本：

### 预训练阶段

1. **用 MoE 替代 dense**：DeepSeek-V3 用 671B-A37B 取得接近 70B dense 的效果，但计算量与 37B 相当。
2. **混合精度训练**：BF16 + FP8（DeepSeek-V3 已用），减少 30%–50% 显存与计算。
3. **Sequence packing**：把多个短样本拼成一个长序列，padding 浪费从 30% 降到 5%。
4. **数据课程**：先 easy 后 hard，减少无效 token 训练。

### 后训练阶段

1. **rollout 加速**：vLLM / SGLang 把 rollout 速度提升 3–10 倍。
2. **off-policy 重用**：用 importance sampling 重用旧 rollout（参考 GRPO 的群体采样）。
3. **DPO 替代 RLHF**：当不需要复杂奖励信号时，DPO 比 PPO 便宜 10–50 倍。
4. **Verifier 替代 RM**：数学/代码任务用规则 verifier（如 Lean、unit test），无需训练 RM。
5. **课程学习**：按难度递增采样，提高 token 利用率。

### 评测与实验

1. **小模型消融**：在 1B 模型上做超参搜索，再迁移到大模型。
2. **早期停止**：用 reward shaping 的 KL 发散或 reward 停滞作为停止信号。
3. **共享 checkpoint**：多次实验从同一 SFT checkpoint 出发，节省 SFT 成本。

::: tip 云 GPU 选型

- **A100 80GB**：性价比最高，适合中型档与入门档
- **H100 80GB**：训练速度快 2–3 倍，单价高 40%，适合预算充足的中型档
- **H800 80GB**：中国区可采购，性能略低于 H100（NVLink 带宽减半），适合大型档中国团队
- **B200**：2025 年新出，BF16 算力是 H100 的 2.5 倍，单价 ~$6/h，适合超大规模训练
- **L40S / A10**：推理为主，不适合训练
  :::

## 本章总结

本附录的成本估算服务于一个核心判断：**你的实验是否值得做**。把上述表格记住后，看到任何"我们提出 X 方法"的论文，你应能立刻在脑中换算成"$X 训这个要多少 GPU 小时、要多少周、要多少失败实验"。这种工程直觉比任何算法细节都重要——它决定了你能不能在算力预算内做出有意义的研究。

下一步建议：

- **做入门实验**：参考 [附录 B 代码速查](../appendix_code_cheatsheet/sft-kl) 中的 GRPO/DPO 代码，在单卡上跑一遍。
- **规划中型实验**：参考 [附录 A.1 训练跑偏](../appendix_industrial_training/training-debugging) 中的诊断顺序与排查清单。
- **阅读前沿论文的成本披露**：在 [附录 C](../appendix_paper_reading/learning-resources) 的 tech report 中寻找训练细节。
