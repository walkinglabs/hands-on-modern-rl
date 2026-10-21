# 19.2 多轮强化学习

<ChapterNotebookLink chapter="agentic" />

[19.1 总览](./overview) 用订机票的例子说明了 Agentic RL 与单轮 RL 的根本差异。本节把这些差异写成精确的数学对象——采用 [AppWorld](https://arxiv.org/abs/2407.18901) 论文中的 POMDP 形式化，它把"模型生成的 token"与"环境返回的 token"显式区分开，是后续讨论 action mask、step-level advantage、credit assignment 的基础。

## 单轮 RL 的简化视角

前面章节的 GRPO 本质是一个**退化的 MDP**。模型接收 prompt，自回归地生成 token 序列，最后由奖励模型或 verifier 给出一个标量 reward。

- **状态** $s$：当前 token context（prompt + 已生成 token）
- **动作** $a$：下一个 token
- **转移**：确定性 append——把选中的 token 加到 context
- **奖励** $r$：整条 rollout 结束后给一次

每一步的动作是从 LLM 的 next-token 分布中采样的——每个 token 都是一个独立的动作。优化目标是让单轮输出的期望奖励最大：

$$
\mathbb{E}_{a \sim \pi_\theta}[r(a)] \quad \longrightarrow \quad \max_\theta
$$

这个视角的关键假设是：**所有 token 都是模型生成的，因此都参与梯度更新**。这个假设在多轮交互中不再成立。

## 多轮交互的 POMDP

当模型不再只是闭门生成，而是在每一步可以调用工具、改变环境状态时，状态空间必须扩展。记一条轨迹为 $x$，完整状态写作：

$$
z_t = [s_0,\ c,\ x_{1:t}]
$$

三个部分分别对应：

- $s_0$：**隐藏的初始环境状态**——AppWorld 里的数据库快照、Python REPL 的初始状态、文件系统内容。模型看不到，只能通过工具调用间接观测。
- $c$：**任务上下文**——用户请求、system prompt、可用的工具规格说明。
- $x_{1:t}$：**到当前为止的完整 token 历史**——既包括模型生成的 thought / action token，也包括环境返回的 observation token。

之所以叫**部分可观测**（POMDP 的 PO），是因为模型只能看到 $(c, x_{1:t})$ 这部分文本历史；隐藏环境状态 $s_0$ 和它随时间的演化都对模型不可见。模型可以调用 API 查日历，但它不能直接"读取整个世界状态"。

### 动作：文本 token 与工具调用

在 token 层面，模型仍然是在做 next-token prediction：

$$
x_{t+1} \sim p_\theta(\cdot \mid c, x_{1:t})
$$

但语义上，token 流分成两类：

- **普通文本 token**（thought、代码片段的一部分）：只更新 context，转移写作 $[s_0, c, x_{1:t}] \to [s_0, c, x_{1:t+1}]$
- **结构化工具调用 token**（如 `<tool_call>...</tool_call>`）：触发环境执行，环境运行代码或 API，并把返回结果也追加进 context：$[s_0, c, x_{1:t}] \to [s_0, c, x_{1:t+1+k}]$

这里多出来的 $k$ 个 token 是**环境 observation**，不是 policy action。比如 API 返回的 JSON 会影响模型下一步决策，但不是模型自己采样出来的。

### 轨迹概率的链式分解

把一条完整轨迹的概率拆开，关键是要**只在模型生成的 token 位置上连乘**。记 $a(x)$ 为轨迹 $x$ 中由 LLM 生成的 token 位置集合（即"action token"），环境返回的 observation 位置不在这个集合里。轨迹分布写作：

$$
\rho_\theta(x \mid s_0, c) = \mathbb{I}(s_0, x) \prod_{t \in a(x)} p_\theta(x_t \mid c, x_{1:t-1})
$$

这里的 $\mathbb{I}(s_0, x)$ 只是把环境动力学记进公式：给定初始数据库、REPL 状态和前面的 API 调用，环境返回什么 observation 是由环境决定的（确定性或随机），**不是模型自由生成的**。这条公式是后续 action mask 推导的起点——只有 $a(x)$ 中的 token 对 $\theta$ 有梯度。

### 优化目标

最大化给定初始状态和任务上下文下的期望回报：

$$
\max_\theta\ \mathbb{E}_{(s_0, c) \sim \mathcal{D}} \left[ \mathbb{E}_{x \sim \rho_\theta(\cdot \mid s_0, c)} \big[ R(s_0, c, x) \big] \right]
$$

外层期望对训练集 $\mathcal{D}$ 中的任务采样，内层期望对策略 $\rho_\theta$ 在固定任务下生成的轨迹采样。Reward $R(s_0, c, x)$ 评估的是整条轨迹是否完成了任务——这是 ORM（outcome reward）的典型形态。

## Reward 的四种类型

实际工程中，reward 不只是"最终答案对不对"。订票 agent 不是说"订好了"就成功，而是数据库里真的多了一张符合约束的订单、且没有误改其他字段。[XiaoRed5 的入门资料](https://github.com/XiaoRed5/Agentic-RL-Most-Detailed-Intro) 把 reward 分成四类：

| 类型        | 含义                                         | 例子                                       |
| ----------- | -------------------------------------------- | ------------------------------------------ |
| **Outcome** | 最终答案是否正确，或最终环境状态是否满足任务 | QA 任务答案匹配、AppWorld 单元测试通过     |
| **Format**  | action 是否能被环境解析和执行                | JSON 参数齐全、工具名拼写正确              |
| **Cost**    | 轨迹长度、工具调用次数、API 花费             | 限制 rollout 不超过 20 步、惩罚重复搜索    |
| **Process** | 中间步骤是否真的推进任务                     | 搜索是否找到有效证据、代码是否通过中间测试 |

入门时通常先实现 Outcome + Format 两类，确保训练能跑通；Process reward 是后续信用分配章节的主题，它把稀疏的 outcome 信号 dense 化。

## Action Mask：模型生成与环境返回必须区分

把上面的轨迹概率公式 $\rho_\theta$ 翻译成 loss，就得到 **action mask** 的数学根据。策略梯度只应该更新模型实际生成的 token：

$$
\nabla J(\theta) = \mathbb{E}_{x \sim \rho_\theta} \left[ R(x) \sum_{t \in a(x)} \nabla_\theta \log p_\theta(x_t \mid c, x_{1:t-1}) \right]
$$

如果让环境返回的 observation token 也参与梯度，相当于让模型"学习预测环境返回的网页内容"——这会污染策略梯度，导致训练不稳定。

实现上，action mask 是一个与轨迹等长的 0/1 向量：

```python
# 一次 rollout 的 token 序列与对应的 action mask
# 1 = 模型生成的 token（参与梯度）
# 0 = prompt / 工具返回 / padding（不参与梯度）

# <prompt>       <think>...搜索</think>  <search>query</search>  <information>...网页内容...</information>  <answer>...</answer>
#  0 0 0 0 0      1 1 1 1 1 1 1 1 1      1 1 1 1 1 1 1 1 1      0 0 0 0 0 0 0 0 0 0 0 0 0                  1 1 1 1 1 1 1
```

[Search-R1](https://github.com/PeterGriffinJin/Search-R1) 这个最小可跑案例把这件事做得非常清楚：用四类标签把 token 流分段——`<think>` 是模型推理（参与训练）、`<search>` 是模型 action（参与训练）、`<information>` 是检索器返回的 observation（masked 不训练）、`<answer>` 是最终回答（参与训练）。配置里的 `state_masking=true` 就是这件事。

[Agent-R1](https://arxiv.org/abs/2511.14460) 进一步发现：**完全排除非 agent token 不是最优**——可以对环境 token 施加 SFT loss（学习预测环境行为），相当于边学策略边学世界模型。这条路线被 Echo、PaW 等后续工作进一步发展。

## Step-Level 轨迹结构

理论上的轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, a_T)$ 已经够用，但工业实现中如何**存储**轨迹直接影响训练稳定性和工程效率。

### Flat Token Sequence 的问题

最简单的存储方式是把整条轨迹拍平成一个 token 序列。问题有两个：

1. **Step 边界隐式**：哪几个 token 属于"第 3 轮的模型输出"、哪几个属于"第 3 轮的工具返回"——全靠特殊 token 切分，错误处理容易出 bug。
2. **Retokenization drift**：rollout 时模型在 token 空间生成，存储时常常解析成 message list，训练时再把 message 重新 tokenize。tokenization 不是可逆操作——同一个文本可能对应不同的 token 序列，导致训练数据和 rollout 时不一致。

### Step-Level 记录（Agent-R1 风格）

把轨迹存储为**结构化的 step-level 记录**，每一步显式保存：

```python
@dataclass
class Step:
    state_before: str          # 该步开始时的 context
    action_tokens: List[int]   # 模型生成的原始 token id（不重新 tokenize）
    observation: str           # 工具返回的观测（如果有）
    reward: float              # 该步的 reward（process reward 时非零）
    is_terminal: bool          # 是否是最后一步
```

好处有三个：精确的 step 边界、无 retokenization drift、灵活的 context 管理策略（append-only、sliding-window、LLM summarization、selective retention）。Agent-R1 的实验显示，**sliding-window 在 GSM8K 上比 append-only 表现更好**——"less is more"，模型不需要看到所有历史也能做出好决策。

## 与单轮 RL 的对照表

|                  | 单轮 RL（GRPO）                        | 多轮 Agentic RL                                              |
| ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| **状态**         | prompt + 已生成 token                  | $[s_0, c, x_{1:t}]$——隐藏环境状态 + 任务上下文 + token 历史  |
| **动作**         | 纯文本 token                           | 文本 token + 结构化工具调用（最终都是 token，但语义不同）    |
| **转移**         | 确定性 append                          | 文本 token 确定性 append；工具调用触发环境动力学（可能随机） |
| **observation**  | 不区分（全是模型生成）                 | 必须显式区分 observation token 与 action token               |
| **奖励**         | 单步标量 $r(a)$                        | Outcome / Format / Cost / Process 四类                       |
| **优化目标**     | $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$ | $\mathbb{E}_{(s_0,c)}[\mathbb{E}_x[R(s_0,c,x)]]$             |
| **rollout 周期** | 几百毫秒                               | 秒到分钟级（受环境延迟主导）                                 |
| **训练表示**     | token 序列                             | Step-level 结构化记录（Agent-R1）                            |

## 本节总结

本节建立了 Agentic RL 的形式化骨架。采用 AppWorld 论文的 POMDP 形式化，把状态拆成 $[s_0, c, x_{1:t}]$ 三部分，是后续讨论的数学基础。轨迹概率的链式分解 $\rho_\theta(x | s_0, c) = \mathbb{I}(s_0, x) \prod_{t \in a(x)} p_\theta(x_t | \cdot)$ 直接推出 action mask——只有模型生成的 action token 参与 policy gradient。Step-level 轨迹结构（Agent-R1）解决了工业实现中的 retokenization drift 和 context 管理问题。

接下来的核心问题是：**轨迹概率公式里 $R(x)$ 是 trajectory-level 标量，它怎么回拆到每一步的 advantage？** 形式化告诉我们"应该只在 action token 上算梯度"，但没告诉我们"每个 action token 应该乘以多大的 advantage"。这就是信用分配——[19.3 轨迹信用分配](./credit-assignment)。
