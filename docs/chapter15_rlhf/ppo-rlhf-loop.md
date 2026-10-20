# 13.9 补充阅读：PPO-RLHF 训练循环

## 本节导读

**核心内容**

- 理解 PPO-RLHF 为什么需要 Actor、Reference、Reward Model、Critic 四个角色。
- 掌握 KL 惩罚、token-level reward、advantage、PPO clip 在 LLM 训练中的对应关系。
- 学会阅读 PPO-RLHF 训练曲线：reward、KL、长度、entropy、value loss 同时看。

**核心公式**

$$
r_t =
\begin{cases}
-\beta(\log \pi_\theta(y_t\mid s_t)-\log \pi_{ref}(y_t\mid s_t)), & t<T \\
r_{RM}(x,y)-\beta(\log \pi_\theta(y_t\mid s_t)-\log \pi_{ref}(y_t\mid s_t)), & t=T
\end{cases}
\quad \text{（RLHF token 奖励：每步 KL，末尾加 RM 分）}
$$

$$
\rho_t(\theta)=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\theta_{old}}(y_t\mid s_t)}
\quad \text{（新旧策略概率比）}
$$

$$
\mathcal{L}_{clip}(\theta)
=-\mathbb{E}_t\left[
\min(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)
\right]
\quad \text{（PPO 裁剪目标）}
$$

> **先记住一句话**
>
> PPO-RLHF 不是让模型“无约束追求高 reward”，而是在 Reference 拉住、PPO 裁剪、Critic 降噪的情况下，小步提高高质量回答的概率。

有了 SFT 模型和 Reward Model，经典 RLHF 的最后一步就是用 PPO 继续优化策略。InstructGPT 这类流程里，PPO 阶段不是“一个模型自己训练自己”，而是四个角色一起工作：

| 角色                 | 来源                | 作用                              |
| -------------------- | ------------------- | --------------------------------- |
| Actor                | SFT 模型继续训练    | 生成回答，并被 PPO 更新           |
| Reference            | 冻结的 SFT 模型     | 提供 KL 约束，防止 Actor 偏离太远 |
| Reward Model         | 偏好数据训练得到    | 给 Actor 生成的回答打分           |
| Critic / Value Model | 通常从 Actor 初始化 | 估计价值函数，降低 PPO 方差       |

```mermaid
flowchart LR
    Prompt["Prompt"] --> Actor["Actor\n可训练策略"]
    Actor --> Response["Response"]
    Response --> RM["Reward Model\n打分"]
    Actor --> Logp["Actor log-prob"]
    Ref["Reference\n冻结 SFT"] --> RefLogp["Reference log-prob"]
    Critic["Critic\n价值估计"] --> Adv["Advantage"]

    RM --> Reward["RM reward"]
    Logp --> KL["KL penalty"]
    RefLogp --> KL
    Reward --> Total["Total reward"]
    KL --> Total
    Total --> Adv
    Adv --> PPO["PPO update"]
    PPO --> Actor
    PPO --> Critic

    style Actor fill:#e3f2fd,stroke:#1565c0
    style Ref fill:#f5f5f5,stroke:#616161
    style RM fill:#fff3e0,stroke:#e65100
    style Critic fill:#e8f5e9,stroke:#2e7d32
```

## 把 LLM 回答拆成一条轨迹

第 3 章里，一条 RL 轨迹长这样：

$$
s_0,a_0,r_0,s_1,a_1,r_1,\ldots
$$

在 LLM 里，prompt 固定后，生成回答也可以写成轨迹：

```text
s_0 = prompt
a_0 = 第 1 个 token
s_1 = prompt + 第 1 个 token
a_1 = 第 2 个 token
...
s_T = prompt + 完整回答
```

动作就是 token，策略就是语言模型：

$$
\pi_\theta(a_t\mid s_t)=P_\theta(y_t\mid x,y_{<t})
$$

和 CartPole 不同，LLM 通常不是每生成一个 token 就有一个人类奖励。Reward Model 往往只在完整回答 $y$ 结束后给一个分数 $r_{RM}(x,y)$。为了让 PPO 能在 token 级别更新，工程上会把奖励拆成两部分：

1. 每个 token 都有 KL 惩罚，防止偏离 reference。
2. 最后一个 token 或 EOS 位置加上 RM 的整段奖励。

这就是本节开头的 token 奖励公式。它看起来有点绕，但直觉很简单：

> 你可以尝试写得更好，但每一步都要为”偏离原来的 SFT 模型”付一点成本；只有整段回答完成后，裁判才给总分。

### Token 级 vs 序列级策略梯度损失

上面的奖励拆分引出一个问题：**策略梯度损失应该在哪个粒度上计算？**

先说两种粒度各自是什么意思。

**序列级：整条回答共用一个梯度信号。** 模型生成完整回答后，RM 给一个总分 $R$。这个 $R$ 被均匀分配到回答中的每一个 token 上——不管是关键的解题数字还是无关的语气词，梯度更新幅度完全相同。

$$
\nabla_\theta J \approx \frac{1}{T}\sum_{t=1}^{T} R \cdot \nabla_\theta \log \pi_\theta(y_t\mid s_t)
$$

**Token 级：每个 token 有独立的梯度信号。** 虽然奖励仍然集中在末尾，但通过 Critic 估计每个位置的 value，计算出每个 token 独立的 advantage $A_t$，再用 PPO 裁剪目标更新：

$$
\mathcal{L}_{clip}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\min(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)
$$

一个具体的例子。用户问"3 + 3 × 6 等于几？"，模型生成了 9 个 token：

```text
这  道  题  的  答  案  是  2  1
t1  t2  t3  t4  t5  t6  t7  t8  t9
```

RM 给了正分 $R = +1.5$。

- **序列级做法**：每个 token 的梯度都乘上 $R = +1.5$。t8 的”2”和 t1 的”这”获得完全相同的更新力度。
- **Token 级做法**：模型会单独评估”每个 token 对最终得分有多大贡献”。”21”直接决定了答案对不对，贡献最大，梯度更新也最强；”这道题的答案是”只是常见套话，换成别的说法也不影响得分，梯度更新就很弱。模型把学习精力集中在真正重要的 token 上。

这个区别在 CartPole 里不明显（每步动作直接影响倒立摆），但在 LLM 里就很关键：一条回答通常几十到几百个 token，其中真正决定质量的往往只有几个。如果所有 token 被平均更新，梯度信号就被大量无关 token 稀释了。

| 维度     | 序列级                    | Token 级                          |
| -------- | ------------------------- | --------------------------------- |
| 梯度信号 | 所有 token 共享同一个 $R$ | 每个 token 有独立的 $A_t$         |
| 信用分配 | 无法区分关键与无关 token  | 通过 GAE 反向传播区分贡献         |
| 学习效率 | 低：大量 token 被平均更新 | 高：关键 token 获得更强的梯度信号 |
| 典型方法 | REINFORCE                 | PPO、GRPO                         |

**粒度越细，模型越能区分哪些决策真正重要，学习效率越高。** 后面 GRPO（第 15 章）和 Agentic RL（第 19 章）会进一步利用这一特性，在更长、更复杂的轨迹上做细粒度信用分配。

#### 目前论文的共识

学术界和工业界在这一问题上的结论已经比较明确：

- **Token 级优于序列级。** TDPO 的实验表明，token 级 DPO 在长文本生成任务上显著优于标准 DPO；ReMax 从信用分配角度证明，序列级 REINFORCE 的梯度信号被无关 token 稀释，是样本效率低的重要原因。
- **信用分配是核心难点。** 奖励通常只给整条回答打一个总分，如何把这个分数合理地分配给每个 token，是 token 级方法的关键。PPO 用 Critic + GAE 来估计每个 token 的 advantage，GRPO 用组内相对排名替代 Critic，两者的目标都是更精确的信用分配。
- **序列越长，token 级的优势越大。** DeepSeekMath 在数学推理场景中发现，推理链越长，序列级方法的梯度信号被稀释得越严重，token 级方法的收益越明显。这也是 GRPO 在长推理任务中效果突出的原因之一。
- **实践建议。** 如果训练资源有限、回答较短（如单轮问答），序列级和 token 级差异不大；如果回答较长（如推理链、多轮对话），优先使用 token 级方法。目前主流开源框架（TRL、OpenRLHF、veRL）默认都采用 token 级策略梯度损失。

::: details 代码上的区别

序列级和 token 级的核心差异只有一行：**advantage 怎么算**。

```python
# ---------- 序列级 ----------
# 整条回答共用一个 reward，所有 token 的 advantage 相同
reward = rm_score - beta * kl_sum          # 标量
advantages = torch.full_like(logprobs, reward)  # 每个 token 都填同一个值

loss_seq = -(advantages * logprobs).mean()

# ---------- Token 级 ----------
# 先用 Critic 算每个 token 的 value，再用 GAE 得到逐 token 的 advantage
values = critic(prompt, response)          # [seq_len]
rewards = build_token_rewards(rm_score, kl_per_token)  # 末尾 token 加 rm_score
advantages = compute_gae(rewards, values)  # 每个 token 一个独立的 advantage

loss_token = -(advantages * logprobs).mean()
```

序列级不需要 Critic，直接把总 reward 广播到每个位置；token 级多了一步 GAE 计算，但每个 token 得到不同的 advantage 值。
:::

::: details 反向传播过程的区别

两种方法的反向传播路径相同——都是对 `loss.backward()` 求梯度更新 Actor 参数。区别在于**梯度信号的强度分布**。

假设回答有 $T$ 个 token，每个 token 的策略梯度近似为：

$$
\Delta\theta_t \propto A_t \cdot \nabla_\theta \log \pi_\theta(y_t \mid s_t)
$$

- **序列级**：$A_t = R$ 对所有 $t$ 相同，梯度更新均匀分布在所有 token 对应的参数上。
- **Token 级**：$A_t$ 随位置变化。靠近奖励源头的 token（如最终的答案数字）$A_t$ 大，梯度更新强；远离奖励的前缀 token $A_t$ 小，梯度更新弱。

从参数视角看，语言模型的底层参数被所有 token 共享。序列级方法让底层参数收到一个被所有 token 平均后的梯度；token 级方法让底层参数收到的梯度更偏向关键 token 的方向。这就是"更精细的梯度信号"在反向传播中的具体含义。
:::

::: details 延伸阅读：Token 级策略梯度的相关论文

- **InstructGPT** (Ouyang et al., 2022) — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)。PPO 应用于 RLHF 的经典工作。奖励是序列级的，但策略梯度损失在 token 级计算，是工业界 token 级策略梯度的标准做法。
- **DeepSeekMath** (Shao et al., 2024) — [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)。提出 GRPO，在数学推理场景中分析了 token 级信用分配对长推理链的重要性。
- **TDPO** (Zeng et al., 2024) — [arxiv.org/abs/2404.11999](https://arxiv.org/abs/2404.11999)。Token-level Direct Preference Optimization，直接在 token 级做 DPO。论文 Section 3 对 token 级与序列级损失做了明确的数学对比。
- **ReMax** (Li et al., 2024) — [arxiv.org/abs/2310.10505](https://arxiv.org/abs/2310.10505)。讨论 token 级与序列级信用分配的差异，提出基于 REINFORCE 的改进方法。
- **Sutton & Barto, _Reinforcement Learning: An Introduction_** 第 11 章 — [incompleteideas.net/book](http://incompleteideas.net/book/the-book.html)。策略梯度的逐时间步（per time-step）推导，是 token 级策略梯度的理论基础。
  :::

## PPO-RLHF 单步更新

PPO-RLHF 的核心循环可以拆成六步：

1. 从 prompt 数据集中采样一批问题。
2. Actor 生成回答。
3. Reward Model 给回答打分。
4. Reference 计算同一段回答的 log-prob，用来得到 KL 惩罚。
5. Critic 估计 value，和 total reward 一起算 advantage。
6. PPO 用裁剪目标更新 Actor 和 Critic。

```python
# ==========================================
# PPO-RLHF 训练循环 与 概念版
# ==========================================
for batch in prompt_dataloader:
    prompts = batch["prompt"]

    # 1. Actor 生成回答
    responses, actor_logprobs = actor.generate_with_logprobs(prompts)

    # 2. Reward Model 打分
    rm_scores = reward_model.score(prompts, responses)

    # 3. Reference 计算 KL
    ref_logprobs = reference_model.logprobs(prompts, responses)
    kl_penalty = actor_logprobs - ref_logprobs

    # 4. 总奖励 = RM 分数 - KL 惩罚
    rewards = rm_scores - beta * kl_penalty

    # 5. Critic 估计优势
    values = critic.value(prompts, responses)
    advantages, returns = compute_gae(rewards, values)

    # 6. PPO 更新 Actor 和 Critic
    ppo_update(
        actor=actor,
        critic=critic,
        prompts=prompts,
        responses=responses,
        old_logprobs=actor_logprobs,
        advantages=advantages,
        returns=returns,
    )
```

这段代码省略了很多工程细节，但它抓住了经典 RLHF 的本质：Reward Model 给方向，Reference 拉住边界，Critic 降低方差，PPO 控制更新幅度。

### 手算一个 token 的 KL 惩罚

假设某个位置上，Actor 和 Reference 对实际生成 token 的 log-prob 分别是：

$$
\log \pi_\theta(y_t\mid s_t)=-1.2,\qquad
\log \pi_{ref}(y_t\mid s_t)=-1.6
$$

Actor 比 Reference 更喜欢这个 token，因为 $-1.2$ 对应的概率更大。KL 近似项是：

$$
\log \pi_\theta-\log \pi_{ref}=0.4
$$

如果 $\beta=0.05$，这一步 KL 惩罚就是：

$$
-\beta \cdot 0.4 = -0.02
$$

如果整段回答最后 RM 给了 $1.3$ 分，总奖励就可以理解为：

```text
前面每个 token：只扣 KL
最后 EOS token：RM 分数 - 最后一步 KL
```

这也是为什么 RLHF 的 reward 曲线必须和 KL 一起看。Actor 得分上涨，可能是回答真的变好，也可能是它离 reference 越来越远。

## PPO 更新的目标

对每个生成 token，PPO 都比较“旧策略生成它时的概率”和“当前新策略给它的概率”。概率比是：

$$
\rho_t(\theta)=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\theta_{old}}(y_t\mid s_t)}
$$

如果 advantage $A_t>0$，说明这个 token 所在轨迹比 Critic 预期好，PPO 希望提高它的概率；如果 $A_t<0$，说明比预期差，PPO 希望降低它的概率。

但不能无限提高或降低，所以要裁剪：

| 情况    | PPO 想做什么      | clip 的作用                        |
| ------- | ----------------- | ---------------------------------- |
| $A_t>0$ | 提高该 token 概率 | 提高到 $1+\epsilon$ 附近就别再猛推 |
| $A_t<0$ | 降低该 token 概率 | 降到 $1-\epsilon$ 附近就别再猛压   |

这和第 8 章 PPO 的直觉完全一致。区别只是：动作从“LunarLander 的推力方向”变成了“词表里的某个 token”。

### 一个极简 PPO 数值例子

假设某个 token 的旧概率是 $0.10$，新概率是 $0.13$：

$$
\rho=\frac{0.13}{0.10}=1.3
$$

clip range 设为 $\epsilon=0.2$，则上界是 $1.2$。如果这个 token 的 advantage 是 $A=2$：

$$
\rho A=1.3\times2=2.6
$$

裁剪后：

$$
\mathrm{clip}(\rho,0.8,1.2)A=1.2\times2=2.4
$$

PPO 取较小值 $2.4$，等于告诉优化器：这个 token 确实好，但这一步已经提高得够多了，别再推太猛。

如果没有这个裁剪，LLM 的 PPO 很容易因为少数高 reward 样本，把某些模板 token 的概率一下推得过高，出现输出坍缩。

## PPO-RLHF 的训练不稳定性

PPO-RLHF 比普通监督微调更容易出问题，原因不只是“超参数多”。它有三个结构性风险：

| 风险            | 发生了什么                                            | 训练里会看到什么                           |
| --------------- | ----------------------------------------------------- | ------------------------------------------ |
| 非平稳数据      | Actor 每更新一步，下一批回答分布就变了                | reward / KL / length 曲线互相拉扯          |
| RM 分布外错误   | 策略会主动搜索 Reward Model 没见过、但给高分的区域    | reward 上升，但人工观感下降                |
| Reference drift | Actor 离 SFT reference 太远，忘掉原本的语言和指令能力 | 输出变长、重复、模板化，甚至乱码或胡言乱语 |

所以 PPO-RLHF 的训练目标不是“让 reward 越快涨越好”，而是让 reward 在 KL、长度、多样性和回归评测都健康的前提下缓慢变好。

## Reference Model 的作用

如果只最大化 RM 分数，Actor 会很快偏离 SFT 模型，进入 RM 没见过的区域。这个区域里 RM 的分数不再可靠，模型可能写出很长、很空、很模板化甚至有害的回答，却拿到高分。

Reference 的作用是提供一个“不要离原来的 assistant 太远”的约束：

$$
R_{total}(x, y) = r_{RM}(x, y) - \beta D_{KL}(\pi_\theta(y|x) \| \pi_{ref}(y|x))
$$

这里的 $\pi_{ref}$ 通常就是冻结的 SFT 模型。$\beta$ 越大，Actor 越难偏离 SFT；$\beta$ 越小，Actor 越容易探索，也越容易 reward hacking。

Reference 不是“保守派摆设”，而是 RM 泛化边界的保险绳。RM 是在某个回答分布上训练的，通常来自 SFT 模型或相近模型的采样。Actor 如果离这个分布太远，RM 就会进入分布外预测区域。分布外高分往往最危险，因为它会被 PPO 当成真奖励继续放大。

可以把 $\beta$ 想成一个旋钮：

| $\beta$ | 训练现象                 | 风险                           |
| ------- | ------------------------ | ------------------------------ |
| 太大    | KL 很低，reward 不动     | 学不动，RLHF 几乎退化成 SFT    |
| 合适    | reward 缓慢上升，KL 稳定 | 健康更新                       |
| 太小    | reward 快速上升，KL 失控 | reward hacking、乱码、模式坍缩 |

## Critic 的作用

PPO 不是只看“这个回答得了几分”，还要判断“这个回答比当前平均水平好多少”。Critic 用来估计 value，进而计算 advantage：

$$
A_t = R_t - V_\phi(s_t)
$$

如果没有 Critic，奖励信号的方差会很大，训练更不稳定。后面的 GRPO 会尝试用组内相对分数替代 Critic，但在经典 RLHF 里，Critic 是 PPO 阶段的重要组件。

更完整一点，PPO-RLHF 通常会用 GAE 估计 advantage。它先计算 TD error：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

再把多个时间步的 TD error 加权累积：

$$
A_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

这和第 6、7 章的 Actor-Critic / PPO 是同一套东西。LLM 场景下状态是上下文，动作是 token，reward 多数集中在结尾，但 advantage 估计仍然要沿着 token 序列传播。

Critic 的质量也要监控。如果 value 预测太差，advantage 会很吵；如果 value loss 爆炸，Actor 更新通常也会跟着不稳。

## TRL 实验对应关系

在 TRL 的小参数实验里，你不一定需要手写四个模型类，但要知道每个配置项背后的角色：

| TRL 概念                        | RLHF 角色    |
| ------------------------------- | ------------ |
| policy model                    | Actor        |
| ref model                       | Reference    |
| reward model 或 reward function | Reward Model |
| value head                      | Critic       |
| `kl_coef` / `target_kl`         | KL 约束      |
| `ppo_epochs` / `cliprange`      | PPO 更新强度 |

小模型实验的目标不是追求最强效果，而是让你真正看懂这四个角色如何一起工作。大参数框架只是把同一个结构拆成分布式服务。

### Rollout batch 和 PPO batch 的区别

PPO-RLHF 里经常同时出现几个 batch 概念，容易混：

| 名称          | 含义                                    | 影响                 |
| ------------- | --------------------------------------- | -------------------- |
| prompt batch  | 一次拿多少 prompt 去生成                | rollout 吞吐         |
| rollout batch | Actor 生成出的 prompt-response 轨迹集合 | reward / KL 统计     |
| mini-batch    | PPO 更新时切成的小批                    | 梯度稳定性           |
| PPO epochs    | 同一批 rollout 复用几轮更新             | 样本效率和过拟合风险 |

On-policy PPO 的关键是：rollout 来自当前或刚刚过去的策略，不能无限复用旧数据。`ppo_epochs` 太大时，虽然看起来更“充分训练”，但实际上可能让策略在旧 rollout 上过拟合，破坏 on-policy 假设。

## 训练稳定性工具箱

PPO-RLHF 的难点不只是“能不能更新”，而是“更新后别崩”。稳定性和 reward hacking 都应该收进 PPO 主线一起看：

| 工具           | 作用                               | 重点观察                   |
| -------------- | ---------------------------------- | -------------------------- |
| KL 惩罚        | 防止 Actor 偏离 SFT reference 太远 | `kl_mean` 是否超过目标区间 |
| 自适应 `beta`  | KL 过高就拉紧，KL 过低就放松       | reward 是否被 KL 完全压住  |
| 学习率 warmup  | 避免训练初期梯度过猛               | loss / grad norm 是否异常  |
| 梯度裁剪       | 防止极端样本导致爆炸               | `grad_norm` 是否尖刺       |
| 奖励归一化     | 控制 RM 分数尺度                   | reward 分布是否漂移        |
| 长度和重复监控 | 捕捉 reward hacking                | 回答长度、n-gram 重复率    |

健康的 PPO-RLHF 通常不是 reward 一路狂飙，而是 reward 缓慢上升、KL 保持在目标范围、回答长度没有异常增长、输出多样性没有明显下降。只要出现“reward 上升但长度和重复率同步暴涨”，就要先暂停训练，回到 RM 数据和奖励设计检查。

这些工具的顺序也很重要。KL 惩罚是第一道边界，warmup 和梯度裁剪负责让更新不要一开始就炸，奖励归一化负责控制 RM 分数尺度，长度和重复监控负责捕捉 reward hacking。一个常见的自适应 KL 规则是：

```python
def update_kl_coef(beta, observed_kl, target_kl, horizon=1000):
    """KL 高于目标就拉紧，低于目标就放松。"""
    error = (observed_kl - target_kl) / max(target_kl, 1e-8)
    multiplier = 1.0 + error / horizon
    return max(0.0, beta * multiplier)
```

这里的 `beta` 不是越大越安全。太大时 Actor 被 reference 拽住，reward 学不动；太小时 Actor 会快速钻进 RM 的盲区。实际训练里要同时看 `reward_mean`、`kl_mean`、`response_length`、`entropy` 和固定回归集，而不是只看 reward 曲线。

常见失败模式可以快速定位到对应的修复动作：

| 失败现象           | 可能原因                 | 检查方法                  | 修复方案                 |
| ------------------ | ------------------------ | ------------------------- | ------------------------ |
| Loss 变成 NaN      | 梯度爆炸 / 学习率太大    | 检查梯度范数              | 降低学习率、加强梯度裁剪 |
| 奖励不动           | 学习率太小或 KL 惩罚太大 | 检查 KL 散度变化          | 降低 `beta` 或增大学习率 |
| 模型输出乱码       | 参考漂移太严重           | 检查 KL 是否异常大        | 增大 `beta`、降低学习率  |
| 模式坍缩           | 策略熵过低               | 检查 entropy 和重复率     | 加熵正则、降低学习率     |
| 奖励上升但质量下降 | reward hacking           | 人工抽检和 judge 对比     | 多维奖励、对抗性数据增强 |
| 回答越来越长       | 长度 hack                | 检查 length-reward 相关性 | 加长度惩罚，重新校准 RM  |

## 训练日志解读

PPO-RLHF 最容易误读的曲线是 reward。健康训练通常不是 reward 一路冲天，而是多个指标保持张力：

| 指标              | 健康信号             | 危险信号               |
| ----------------- | -------------------- | ---------------------- |
| `reward_mean`     | 缓慢上升             | 快速上升但人工质量下降 |
| `kl_mean`         | 围绕目标区间波动     | 持续升高或接近 0       |
| `response_length` | 稳定或按任务自然变化 | 和 reward 同步暴涨     |
| `entropy`         | 缓慢下降但不塌       | 快速降到很低           |
| `value_loss`      | 可控波动             | 爆炸或长期不降         |
| `clip_fraction`   | 有一定比例被裁剪     | 接近 0 或长期过高      |
| `judge_win_rate`  | 小样本胜率逐步改善   | 与 RM reward 背离      |

两个典型读法：

**情况一：reward 上升，KL 稳定，长度稳定，win rate 上升。**  
这是最健康的信号，说明 Actor 在 reference 附近找到了更好回答。

**情况二：reward 上升，KL 上升，长度暴涨，人工抽检变差。**  
这不是“继续训练就好了”，而是 reward hacking。应该暂停 PPO，回到 RM 数据、长度惩罚和对抗样本检查。

## 最小调参顺序

如果 PPO-RLHF 跑不稳，不要同时乱调所有参数。建议按这个顺序排查：

1. **先固定生成参数**：temperature、top_p、max_new_tokens 不要在实验间飘。
2. **检查 RM 分数尺度**：均值和方差是否离谱，是否需要标准化。
3. **调 KL 系数 `beta`**：让 `kl_mean` 回到目标区间。
4. **调学习率和 batch**：loss NaN 或 KL 尖刺时先降学习率、加梯度裁剪。
5. **看长度和重复率**：如果 reward 和长度强相关，先修奖励，不要只调 PPO。
6. **跑固定评估集**：每个 checkpoint 都用同一套 prompt 比较。

这套顺序的核心是先确认“奖励可信”和“边界稳定”，再追求更高的 reward。

## 从小参数到大参数

PPO-RLHF 的算法结构在小模型和大模型上是一致的。区别在于，大模型训练需要把这条简单流水线扩展成分布式系统。

本章的小参数实验用 TRL 跑通，是为了让你在一张消费级显卡上看清楚 RLHF 的完整结构。但工业训练关心的是：当模型从 360M、0.5B 放大到 7B、32B、70B 甚至更大时，这条流水线还能不能跑起来。

答案是：**算法结构基本不变，系统工程完全变重**。

```mermaid
flowchart LR
    Small["TRL 小参数实验\n单机 / 单卡 / LoRA"] --> Mid["OpenRLHF\n7B-32B / Ray / vLLM / ZeRO"]
    Mid --> Large["NeMo RL / NeMo Aligner\n多机多卡 / Megatron / 生产监控"]

    style Small fill:#e3f2fd,stroke:#1565c0
    style Mid fill:#fff3e0,stroke:#e65100
    style Large fill:#e8f5e9,stroke:#2e7d32
```

### 小参数版本（TRL）

小参数实验最重要的价值是可理解。你能直接看到 SFT 阶段如何把 base model 改造成 assistant，Reward Model 如何用 chosen/rejected 学会排序，PPO 阶段如何同时使用 Actor、Reference、Reward Model 和 Critic。

这时最合适的技术栈是 `transformers`、`datasets`、`peft`、`trl`、`accelerate`。模型可以选 `HuggingFaceTB/SmolLM2-360M`、`Qwen/Qwen2.5-0.5B`、`EleutherAI/pythia-410m` 这类参数量较小的 base model。

小参数阶段建议先把下面这些问题全部跑通：

| 问题                            | 通过标准                           |
| ------------------------------- | ---------------------------------- |
| SFT 是否真的改变 base 行为？    | 固定 prompt 对比明显更像 assistant |
| RM 是否能区分 chosen/rejected？ | held-out accuracy 和 margin 合理   |
| PPO 是否稳定？                  | reward 缓慢升，KL 和长度不失控     |
| 评估是否可复现？                | 同一 checkpoint 重跑结果接近       |
| badcase 是否能回放？            | 失败样本能进入下一轮数据           |

如果这些问题在 0.5B 上还没搞清楚，直接上 7B 只会让调试成本乘上十倍。

### 中等参数版本（OpenRLHF）

当模型到 7B 以上，瓶颈开始从"代码能不能写出来"变成"rollout 和训练吞吐能不能跟上"。PPO-RLHF 需要模型不断生成回答，再让 RM 打分，再回到训练，这个 generate-train loop 会让普通训练框架很吃力。

OpenRLHF 这类框架把几个系统问题打包解决：

| 问题         | 小参数 TRL      | 大参数 OpenRLHF 思路              |
| ------------ | --------------- | --------------------------------- |
| Rollout 速度 | 直接 `generate` | 用 vLLM / Ray 做高吞吐生成        |
| 显存压力     | LoRA 或单卡     | ZeRO、张量并行、流水并行          |
| 多模型调度   | 同进程较简单    | Actor、RM、Critic、Ref 分角色部署 |
| 数据流       | Python loop     | 分布式队列和 rollout buffer       |
| 监控         | 本地日志        | 实验平台、checkpoint、异常恢复    |

### 大参数版本（NeMo）

70B 级别以后，训练框架不仅要跑得动，还要可恢复、可观测、可复现。NVIDIA NeMo RL / NeMo Aligner 这类框架更接近生产训练视角：多机多卡、Megatron/FSDP、分布式 checkpoint、混合精度、模型并行、数据并行和完整监控都必须一起考虑。

大参数 RLHF 最难的地方通常不是 PPO 公式，而是四模型常驻成本（Actor、Reference、Reward Model、Critic 都要占显存或推理资源）、生成和训练切换、奖励模型吞吐、KL 和长度监控、checkpoint 与恢复、评估闭环。

经典 PPO-RLHF 至少涉及四个模型角色：

| 角色         | 是否需要梯度 | 资源特征                               |
| ------------ | ------------ | -------------------------------------- |
| Actor        | 需要         | 最重，训练和生成都要用                 |
| Critic       | 需要         | 可和 Actor 共享部分 backbone，也可独立 |
| Reference    | 不需要       | 冻结推理，但要算 log-prob              |
| Reward Model | 不需要       | 冻结推理，吞吐可能成为瓶颈             |

这意味着"训练一个 7B 模型"不等于显存里只放一个 7B。即使 Reference 和 RM 冻结，它们也要占推理资源。工业系统会做很多工程折中：Actor 和 Critic 共享底座只加 value head，Reference 用同一底座的冻结副本必要时 offload，RM 用较小模型或服务化部署，Rollout 阶段和 PPO update 阶段复用 GPU。

### 框架选型

| 规模    | 推荐路线                                           |
| ------- | -------------------------------------------------- |
| 135M-1B | TRL，优先理解流程                                  |
| 1B-7B   | TRL + Accelerate / DeepSpeed，可以继续用 LoRA      |
| 7B-32B  | OpenRLHF，重点解决 rollout 与分布式训练            |
| 70B+    | NeMo RL / NeMo Aligner，重点解决多机多卡与生产监控 |

不要过早上重框架。如果还没在小模型上跑通 SFT、RM、PPO 和评估，直接上 7B/70B 只会把算法问题和系统问题混在一起。

### 小模型实验和大模型工程的映射

| 本章小实验         | 大参数训练对应物                                  |
| ------------------ | ------------------------------------------------- |
| `SFTTrainer`       | 分布式 SFT，通常配合 LoRA、FSDP、ZeRO 或 Megatron |
| `RewardTrainer`    | 分布式 RM 训练，单独验证 RM accuracy / margin     |
| `PPOTrainer`       | Actor-RM-Critic-Ref 分布式 PPO 系统               |
| 本地 JSON 偏好数据 | 标注平台、数据版本、质量审计、去重和去污染        |
| 简单 judge prompt  | 多 judge、多维 rubric、人类仲裁                   |
| 本地评估脚本       | 自动 benchmark、A/B test、红队、安全回归          |

这张表说明一件事：小模型实验不是玩具，它是大模型训练的缩影。只要你理解了每个 artifact 的角色，换成大参数框架时就不会迷路。

## 本节小结

经典 RLHF 的 PPO 阶段可以压缩成一句话：**让 Actor 追求 RM 给出的偏好奖励，同时用 Reference 和 PPO 约束它不要走偏，再用 Critic 降低更新噪声。**

小参数实验用 TRL 跑通，大参数训练用 OpenRLHF 或 NeMo RL 扩展——算法结构不变，系统工程变重。

PPO-RLHF 训练循环搭好之后，不能只看 reward 是否上涨。13.6 节会用 benchmark、偏好评估和人工抽检确认模型真的变好，也会专门检查 reward hacking 和能力回退——[评估与奖励黑客](./evaluation)。

## 练习

1. 假设 Actor log-prob 为 -2.0，Reference log-prob 为 -2.4，$\beta=0.1$，手算这个 token 的 KL 惩罚。
2. 为什么 `ppo_epochs` 不能无限增大？用 on-policy 的角度解释。
3. 设计一个训练日志表，至少包含 reward、KL、长度、entropy、judge win rate 五个字段。
4. 画出你自己的 RLHF 系统图，标出 Actor、Reference、RM、Critic 分别在哪些 GPU 或进程上。
5. 写一份从 0.5B TRL 实验迁移到 7B OpenRLHF 的检查清单。
