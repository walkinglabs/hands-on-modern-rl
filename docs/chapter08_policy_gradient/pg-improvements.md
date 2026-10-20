# 6.2 策略梯度改进方法

6.4 节的 CartPole 实验会展示 REINFORCE 高方差的直接后果：训练曲线抖动剧烈，策略被运气牵着走。本节回答一个关键问题：**能不能在不改变梯度方向的前提下，降低 $G_t$ 的方差？**

答案是肯定的。策略梯度定理有一个重要性质：梯度估计中可以减去一个不依赖于动作的基线。

## 基线不改变期望

回顾策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

现在把 $G_t$ 换成 $G_t - b(s_t)$，其中 $b(s_t)$ 是一个只依赖状态、不依赖动作的函数：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left(G_t - b(s_t)\right) \right]$$

为什么这不改变期望？因为基线项的贡献为零：

$$\mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t) \right] = \sum_t b(s_t) \cdot \underbrace{\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \right]}_{= \, 0} = 0$$

最后一步用到了一个关键恒等式：对数概率梯度的期望为零。直觉上，$\nabla_\theta \log \pi_\theta(a|s)$ 衡量的是"怎么调参数能增加某个动作的概率"，把所有动作的这个量按概率加权平均，增减刚好抵消。

:::details 证明：$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$

概率分布的归一化条件：$\sum_a \pi_\theta(a|s) = 1$。两边对 $\theta$ 求梯度：

$$\sum_a \nabla_\theta \pi_\theta(a|s) = 0$$

利用 $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$，把 $\nabla_\theta \pi$ 替换成 $\pi \cdot \nabla_\theta \log \pi$：

$$\sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) = 0$$

左边恰好就是 $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)]$。

:::

所以基线不改变梯度的期望方向。但它的实际效果是改变梯度的**方差**。

## 基线降低方差的直觉

减掉基线后，更新信号从"这趟跑了多少分"变成"这趟比预期好了多少"。

考虑 CartPole 中的一个例子。假设当前策略已经比较好了，从状态 $s$ 出发平均能坚持 100 步（$V(s) \approx 100$）：

| 情况                    | $G_t$ | $G_t - V(s)$ | 无基线时更新方向 | 有基线时更新方向 |
| ----------------------- | ----- | ------------ | ---------------- | ---------------- |
| 运气好，坚持了 150 步   | 150   | +50          | 强烈强化         | 适度强化         |
| 运气一般，坚持了 100 步 | 100   | 0            | 中等强化         | 不更新           |
| 运气差，坚持了 50 步    | 50    | -50          | 轻微强化         | 降低概率         |

没有基线时，三种情况都给出正的 $G_t$，策略都会被强化——即使"运气差"的那次其实表现不如平均水平。有基线后，"运气一般"的情况不产生更新，"运气差"的情况被正确地惩罚。

基线做的事情是给每个状态建立一个"及格线"：超过及格线的强化，低于及格线的抑制。这个及格线不是固定的——不同的状态有不同的 $V(s)$，因为不同状态的"正常表现"本来就不一样。

## 用 $V(s)$ 构造状态基线

基线可以是任何不依赖动作的函数。最简单的选择是常数（比如所有 episode 的平均回报）。常数基线在无状态赌博机中就有用，但它无法区分不同状态。

更好的选择是状态相关的基线 $b(s)$。理论分析表明，当 $b(s) = V^\pi(s)$ 时，方差缩减效果接近最优 [^greensmith2004]。直觉上，$V^\pi(s)$ 恰好回答了"从这个状态出发，按当前策略做，平均能拿多少分"——用它做基线，更新信号变成了"实际表现比平均好了多少"。

把 $G_t - V(s_t)$ 这个量叫做**优势**（Advantage）：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

在 REINFORCE 中，$G_t$ 是 $Q^\pi(s_t,a_t)$ 的蒙特卡洛估计，所以优势的估计形式为：

$$\hat{A}_t = G_t - V(s_t)$$

- $\hat{A}_t > 0$：这个动作比当前状态下的平均水平好，应该增加概率
- $\hat{A}_t < 0$：这个动作比平均水平差，应该降低概率
- $\hat{A}_t \approx 0$：表现和预期差不多，不产生强更新

## 优势函数的意义

优势函数 $A^\pi(s,a)$ 是策略梯度方法中最重要的概念之一。它回答的不是"这个动作有多好"，而是"这个动作比平均水平好了多少"。这个"相对好坏"的信号，比 $G_t$ 的"绝对回报"信号稳定得多。

后续章节会反复用到优势函数：

- **第 7 章 Actor-Critic**：用 Critic 网络直接估计 $V(s)$，实现每步更新（不必等 episode 结束）
- **第 8 章 PPO**：用 GAE（Generalized Advantage Estimation）在偏差和方差之间取折中
- **第 13 章 RLHF**：奖励模型提供标量奖励，PPO-RLHF 再结合 Critic 将奖励转化为优势估计

## 加入价值网络

实际操作中，$V(s)$ 由一个额外的神经网络（价值网络）来估计：

```python
# 价值网络学习 V(s)
values = value_net(states_t)
value_loss = nn.MSELoss()(values, returns_t)  # 用 G_t 作为训练目标

# 用优势更新策略
with torch.no_grad():
    values_pred = value_net(states_t)
advantages = returns_t - values_pred  # Â_t = G_t - V(s_t)
policy_loss = -(log_probs * advantages).mean()
```

价值网络的训练目标是让 $V(s_t)$ 尽量接近 $G_t$——也就是说，它在学习"从这个状态出发，平均能拿多少分"。策略网络不再直接使用 $G_t$，而是使用优势 $\hat{A}_t = G_t - V(s_t)$。

这就是 **REINFORCE with Value Baseline**。它仍然是 REINFORCE（必须等 episode 结束，用蒙特卡洛回报更新），但更新信号从 $G_t$ 变成了 $\hat{A}_t$。

6.6 节在 CartPole 上对比 vanilla REINFORCE 和 REINFORCE + Value Baseline：[动手：CartPole 对比实验](./cartpole-baseline)。

---

[^greensmith2004]: Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. _Journal of Machine Learning Research_, 5, 1471-1530.
