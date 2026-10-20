---
search: false
---

# 分布式并行策略（已并入 A.2）

> 这一页保留为旧链接入口。核心内容已经合并到[附录 A.2 训练系统底座](./rl-infrastructure)的“分布式并行与显存优化”部分。下面保留原文，方便从旧链接进入的读者对照。

> PPO 训练需要同时加载 4 个模型（Actor、Critic、Reference、Reward Model）。7B 模型 FP16 每个 ~14GB，4 个就是 ~56GB，一张 A100（80GB）装不下。更大的模型更不用说了。
>
> 本节只回答一个问题：**模型太大一张卡装不下，怎么切到多张卡上？** 先讲四种并行策略各自怎么切，再讲混合精度怎么选，最后说 RL 训练相比普通微调有什么额外挑战。

## 四种并行策略

### 数据并行（DP）

最简单的方式。每个 GPU 持有模型的完整副本，但处理不同的数据 batch，梯度通过 AllReduce 同步。

限制：单卡必须能装下整个模型。7B 模型 FP16 需要 ~14GB 参数 + 优化器状态 + 梯度，实际需要 ~56GB，超出了大多数消费级显卡。

### 张量并行（TP）

把单个矩阵运算拆到多个 GPU 上。例如一个 4096×4096 的矩阵乘法，4 个 GPU 各算 1024×4096，结果拼接。

- 需要 NVLink 等高带宽互联（通常在节点内使用）
- Megatron-LM 是工业标准实现
- 适合单节点内多卡（如一台 8×H100）

### 流水线并行（PP）

按层切分模型。GPU 0 负责第 1-10 层，GPU 1 负责第 11-20 层，以此类推。

缺点是有"气泡"——前面的 GPU 算完后后面的才能开始。微批次调度（Micro-batching）可以缓解，但不能完全消除。

### 专家并行（EP）

专为 MoE（混合专家）模型设计。不同的专家网络分布在不同 GPU 上，Router 决定每个 token 发给哪个专家。Mixtral、DeepSeek-V3 等模型必须用这种策略。

**典型组合**：70B 密集模型常用 DP+TP+PP 三维混合并行。MoE 模型额外使用 EP。

| 策略 | 切什么   | 通信量                | 适用范围            |
| ---- | -------- | --------------------- | ------------------- |
| DP   | 数据     | 梯度 AllReduce        | 单卡能装下模型时    |
| TP   | 层内矩阵 | 每个 forward/backward | 节点内（需 NVLink） |
| PP   | 层间     | 激活值传递            | 跨节点              |
| EP   | 专家网络 | Router 结果           | MoE 模型            |

## 两种显存优化方案

上面四种是"怎么算"的并行。还有一类是"怎么省显存"的方案：

**FSDP（Fully Sharded Data Parallel）**：PyTorch 原生。把模型参数、梯度、优化器状态切分到所有 GPU，计算时临时聚合。通用性强，不需要改模型代码。

**DeepSpeed ZeRO**：类似 FSDP，分三档。ZeRO-1 只切优化器状态，ZeRO-2 加切梯度，ZeRO-3 全部切分。ZeRO-3 可以训练任意大小的模型，但通信开销也最大。

实践中 FSDP 和 ZeRO 经常和 TP/PP 组合使用。

## 混合精度怎么选

| 精度   | 显存    | 速度 | 稳定性     | 用在哪                             |
| ------ | ------- | ---- | ---------- | ---------------------------------- |
| BF16   | 减半    | 快   | 好         | **训练首选**（A100/H100 原生支持） |
| FP16   | 减半    | 快   | 有溢出风险 | 需 loss scaling                    |
| FP32   | 基准    | 慢   | 最好       | 关键精度场景                       |
| FP8    | 1/4     | 最快 | 较差       | 实验性训练                         |
| INT8/4 | 1/4~1/8 | 极快 | 有损       | **推理部署**                       |

实践建议：训练用 BF16，推理用量化（INT4/INT8）。

## RL 训练的额外挑战

RL 训练（PPO/GRPO）比普通微调复杂，因为要同时管理多个模型和两个阶段：

**Rollout 阶段是推理密集型**：GRPO 的 k=16 意味着每个 prompt 要生成 16 个回答，需要大量 GPU 做批量生成。vLLM 的 PagedAttention 可以把批量推理吞吐提升 2-4x。

**Training 阶段是计算密集型**：反向传播需要大量 GPU。

**两个阶段对 GPU 的需求是波动的**：Rollout 时训练闲置，反之亦然。解法是异步架构——Rollout 和训练用不同的 GPU 组，详见 [A.2 RL 训练系统](./rl-infrastructure) 中的异步训练架构部分。

**显存优化技巧**：

| 技巧                   | 原理                                | 节省           |
| ---------------------- | ----------------------------------- | -------------- |
| Reference 模型共享     | Ref 不参与训练，可和 Actor 共享权重 | ~25%           |
| LoRA Rollout           | 推理时只加载 LoRA adapter           | ~50%           |
| Gradient Checkpointing | 重计算代替存储中间激活              | ~40%（换时间） |

## MoE 和 PRM 带来的新问题

**MoE 路由不一致**：MoE 模型有多个专家网络，训练框架（Megatron）和推理框架（vLLM）对 Router 的浮点计算有微小差异，可能导致同一个 token 在训练时走 Expert A、推理时走 Expert B——梯度方向直接错了。DeepSeek-V3.2 的解法叫 Keep Routing：推理时记录路由决策，训练时强制执行相同路径。

**PRM 的额外计算**：Process Reward Model 对推理链的每一步打分，计算量可能接近生成本身。这意味着在推理集群和训练集群之间可能要再加一组 GPU 跑 PRM scoring。目前只有少数框架（如 PRIME-RL）实现了这种流水线。

## 参考文献

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026.

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.

[^3]: OpenRLHF, [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), EMNLP 2025 Demo.

[^4]: DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
