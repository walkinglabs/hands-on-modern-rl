# D.3.5 强化学习优化公式汇总

> **前置知识**：本页汇总 D.3 模块所有公式，建议在读完 [D.3.1](./calculus-basics) 到 [D.3.4](./calculus-derivations) 后再来回顾。

前面几页我们分别推导了策略梯度的单样本形式、对数导数技巧、PPO 裁剪和 GRPO 组归一化。这一页把这些结果整理成完整的公式，并补充 DQN 的损失函数和 GAE 的推导。可以把这一页当作速查手册——遇到不熟悉的符号时回来翻一翻。

---

## 策略梯度定理

前面我们已经看过单样本形式：

$$
\nabla_\theta J(\theta) \approx G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

完整的策略梯度定理写作：

$$
\nabla_\theta J(\theta)
=\sum_s d^\pi(s)\sum_a q_\pi(s,a)\nabla_\theta\pi_\theta(a\mid s).
$$

这里每个符号的含义是：

- $d^\pi(s)$ 是策略 $\pi$ 下状态 $s$ 被访问的频率（可以理解成"策略运行时，有多少时间会处在状态 $s$"）。
- $q_\pi(s,a)$ 是动作价值函数，表示在状态 $s$ 执行动作 $a$ 后，未来能拿到多少平均回报。
- $\nabla_\theta\pi_\theta(a\mid s)$ 是参数 $\theta$ 变化时，选择动作 $a$ 的概率如何跟着变化。

利用上一节推导过的对数导数技巧，这个公式可以改写成更常见的 log 形式。因为：

$$
\nabla_\theta\pi_\theta(a\mid s)
=\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s),
$$

所以：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^\pi,a\sim\pi}
\left[q_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

如果用采样回报 $G_t$ 估计 $q_\pi(s_t,a_t)$，就得到 REINFORCE：

$$
\nabla_\theta J(\theta)
\approx
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

如果用优势函数替代动作价值，就得到更稳定的形式：

$$
\nabla_\theta J(\theta)
=\mathbb{E}
\left[A_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

把推导过程串起来看，复杂公式并不是凭空出现的——它只是把”好动作概率上升、坏动作概率下降”这个直觉写成了对所有状态和动作的加权平均。

---

## 价值函数近似的损失

策略梯度处理的是”怎么更新策略”，但训练中还需要一个模块来估计”某个状态或动作值多少分”——这就是 Critic 或 DQN 的工作。**为什么需要这个模块？** 因为策略梯度里的优势估计 $\hat{A}_t$ 依赖于价值 $V(s)$ 的准确估计；如果价值估计不准，策略更新的方向就会偏。它们的学习目标很直接：让预测值尽量接近真实值。

给定样本 $(s_t,a_t,r_{t+1},s_{t+1})$，DQN 的 TD 目标是：

$$
y_t=r_{t+1}+\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a').
$$

其中 $\theta^-$ 表示目标网络参数。损失函数是：

$$
L(\theta)=\frac{1}{2}\left(Q_\theta(s_t,a_t)-y_t\right)^2.
$$

求梯度：

$$
\nabla_\theta L(\theta)
=\left(Q_\theta(s_t,a_t)-y_t\right)
\nabla_\theta Q_\theta(s_t,a_t).
$$

这个式子里，前一项是 TD 误差：

$$
\delta_t=y_t-Q_\theta(s_t,a_t).
$$

后一项 $\nabla_\theta Q_\theta(s_t,a_t)$ 告诉参数如何改变预测值。DQN 的训练就是反复减小这种预测误差。

---

## 用 TD 误差累积估计优势

策略梯度需要优势函数 $\hat{A}_t$ 来衡量"一个动作比平均水平好多少"，但它没法直接观测到。有两种极端的估计方式：蒙特卡洛方法用整条轨迹的回报——偏差低但方差高（随机性累积了很多步）；时序差分（TD）只用一步的"奖励 + 下一状态估计"——方差低但偏差高（只看了一步，信息不够）。**GAE（Generalized Advantage Estimation）的引入就是为了在这两个极端之间灵活调节**——把未来多步的 TD 误差按递减权重累加，用参数 $\lambda$ 控制"偏向 MC 还是偏向 TD"。先看一步 TD 误差：

$$
\delta_t=R_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

如果 $\delta_t>0$，说明实际结果比 Critic 预期更好；如果 $\delta_t<0$，说明实际结果比预期差。TD 误差只看一步，GAE 的做法是把未来多步的 TD 误差按递减权重累加起来：

$$
\hat{A}^{GAE}_t
=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

其中 $\lambda\in[0,1]$ 控制偏差和方差的权衡：

- $\lambda$ 小：更依赖短期 TD 误差，方差较低，但偏差可能较大。
- $\lambda$ 大：更接近完整回报，偏差较低，但方差可能较大。

PPO 中常用 GAE，是因为它在稳定性和准确性之间提供了一个方便调节的旋钮。

---

## PPO 裁剪目标

前面我们已经看过概率比和裁剪的直觉：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

PPO 的裁剪目标是：

$$
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

这个公式看起来复杂，但逐个拆开看并不难。

如果 $\hat{A}_t>0$，说明动作比平均好。我们希望提高它的概率，但最多提高到 $1+\epsilon$ 倍。

如果 $\hat{A}_t<0$，说明动作比平均差。我们希望降低它的概率，但最多降低到 $1-\epsilon$ 倍。

因此 `min` 和 `clip` 的组合实现了一个简单而有效的原则：**让策略朝正确方向更新，但不让它一次走太远**。

---

## GRPO 的组归一化优势

GRPO 用于一组回答的相对比较。假设同一个问题生成 $n$ 个回答，奖励分别为：

$$
r_1,r_2,\ldots,r_n.
$$

先计算均值：

$$
\mu=\frac{1}{n}\sum_{i=1}^n r_i.
$$

再计算标准差：

$$
\sigma=\sqrt{\frac{1}{n}\sum_{i=1}^n(r_i-\mu)^2}.
$$

每个回答的标准化优势是：

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma+\epsilon}.
$$

例如奖励是 $[2,4,10]$，均值是 $5.33$。第三个回答明显高于平均，优势为正；第一个回答低于平均，优势为负。这种组内相对比较的好处是：模型不需要额外训练一个 Critic 网络，仅靠同组回答之间的好坏对比就能更新策略。

---

## 小结

本页汇总了 D.3 模块所有核心公式：

| 公式         | 核心表达式                                                             | 用途                       |
| ------------ | ---------------------------------------------------------------------- | -------------------------- |
| 策略梯度定理 | $\nabla_\theta J=\mathbb{E}[\nabla\log\pi\cdot Q^\pi]$                 | 策略优化的理论基础         |
| DQN 损失     | $L=\frac{1}{2}(Q_\theta-y_t)^2$                                        | 价值函数的训练目标         |
| GAE          | $\hat{A}^{GAE}_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$                  | 偏差-方差权衡的优势估计    |
| PPO 裁剪     | $\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)$ | 限制策略更新幅度           |
| GRPO 组优势  | $\hat{A}_i=(r_i-\mu)/(\sigma+\epsilon)$                                | 无需 Critic 的组内相对比较 |

遇到不熟悉的符号时可以回到这一页查阅。下一篇用练习题来检验对这些公式的理解。

> **下一篇**：[D.3.6 公式速查与练习](./calculus-formulas-exercises) —— 汇总本模块所有公式，用练习检验理解。
