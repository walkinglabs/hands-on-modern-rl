# D.3.4 策略梯度、Taylor 与 GRPO 推导

> **前置知识**：[D.3.2 策略梯度与优势函数](./calculus-policy-gradient)——你需要知道策略梯度的基本形式。

---

## 对数导数技巧如何推出策略梯度

前面我们一直在用策略梯度的结论，现在来看看它是怎么推导出来的。直接对策略概率 $\pi_\theta(a\mid s)$ 求梯度往往不好算——$\pi$ 是由 softmax 等复杂函数生成的，梯度表达式很繁。**对数导数技巧**就是把难算的 $\nabla_\theta \pi$ 转换成好算的 $\pi \cdot \nabla_\theta \log \pi$，从而让梯度可以用采样的方式估计，而不需要知道环境的转移概率。策略梯度最常见的形式就是由这个技巧推导出来的：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi[
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
].
$$

其中 $\mathbb{E}_\pi[\cdot]$ 表示”按照策略 $\pi$ 采样时的期望值”，也就是对所有可能的状态-动作对取加权平均，权重就是策略选择每个动作的概率。推导这个公式的关键是一个简单的恒等式：

$$
\nabla_\theta \log \pi_\theta(a\mid s)
=
\frac{\nabla_\theta \pi_\theta(a\mid s)}
{\pi_\theta(a\mid s)}.
$$

两边乘以 $\pi_\theta(a\mid s)$，就得到一个等价但更好用的形式：

$$
\nabla_\theta \pi_\theta(a\mid s)
=
\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

这个变换的好处在于：直接对 $\pi_\theta$ 求梯度往往不好算，但 $\log\pi_\theta$ 的梯度通常很简单。接下来把这个技巧代入目标函数。在离散动作空间中，目标函数可以写成：

$$
J(\theta)=\sum_a \pi_\theta(a\mid s)Q^\pi(s,a).
$$

对参数求梯度时，$Q^\pi(s,a)$ 不依赖 $\theta$，只有 $\pi_\theta$ 里有 $\theta$：

$$
\nabla_\theta J(\theta)
=\sum_a \nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a).
$$

把对数导数技巧代入，把 $\nabla_\theta\pi_\theta$ 替换掉：

$$
\nabla_\theta J(\theta)
=\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
Q^\pi(s,a).
$$

仔细看这个求和式：每一项都含 $\pi_\theta(a\mid s)$ 作为权重，这恰好是"按策略采样时的加权平均"——也就是期望：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
[
\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)
].
$$

上面只考虑了单个状态。如果对所有状态做加权平均（权重 $d^\pi(s)$ 是策略 $\pi$ 下访问状态 $s$ 的频率），就得到完整的策略梯度定理：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)Q^\pi(s_t,a_t)
].
$$

在实际算法中，$Q^\pi(s_t,a_t)$ 不容易精确知道，所以常用采样的累计回报 $G_t$ 或优势估计 $\hat{A}_t$ 来替代：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t
].
$$

这就是 REINFORCE、Actor-Critic 和 PPO 这些算法背后共同的梯度结构。

---

## Taylor 展开、Hessian 与 PPO 的二阶直觉

梯度下降只看一阶导数——"当前位置的斜率"，然后沿斜率方向走一步。但如果步子走得太大，一阶近似就会失准：你以为还在上坡，其实已经过了山顶开始下坡了。**Taylor 展开**就是用来分析"步子走多大时一阶近似还靠谱"的工具。一阶展开只看斜率，二阶展开额外考虑曲率（弯不弯、往哪个方向弯）。PPO 和 TRPO 背后的"信任域"思想，正是担心参数更新过大时一阶近似不再可靠——Taylor 展开帮助我们从数学上理解这件事。

$$
f(x+h)\approx f(x)+f'(x)h.
$$

看一个数字例子。令：

$$
f(x)=x^2,\qquad x=3,\qquad h=0.1.
$$

真实值是：

$$
f(3.1)=9.61.
$$

一阶近似是：

$$
f(3)+f'(3)h=9+6\times0.1=9.6.
$$

已经很接近了，差了 $0.01$。二阶 Taylor 展开再加上一个曲率修正项：

$$
f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2.
$$

对 $f(x)=x^2$，$f''(x)=2$，所以：

$$
9+6\times0.1+\frac{1}{2}\times2\times0.1^2=9.61.
$$

多变量时，二阶项中的 $f''$ 变成了 Hessian 矩阵 $H$（它记录了函数在每个方向上的弯曲程度）：

$$
f(\theta+\Delta\theta)
\approx
f(\theta)
\nabla f(\theta)^\top\Delta\theta
\frac{1}{2}\Delta\theta^\top H\Delta\theta.
$$

PPO 和 TRPO 背后的"信任域"思想，正是担心参数更新过大时，一阶近似不再可靠，二阶曲率项开始变得重要——此时如果还按一阶信息走大步，可能会把策略搞坏。

对 PPO 的概率比：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)},
$$

在 $\theta_{old}$ 附近展开：

$$
r_t(\theta)
\approx
1
+\nabla_\theta r_t^\top(\theta-\theta_{old})
+\frac{1}{2}(\theta-\theta_{old})^\top
\nabla_\theta^2 r_t
(\theta-\theta_{old}).
$$

这里三项分别是：

| 项     | 含义                           |
| ------ | ------------------------------ |
| $1$    | 新旧策略相同时，概率比为 $1$   |
| 一阶项 | 小步更新带来的线性变化         |
| 二阶项 | 步子变大后，曲率带来的额外变化 |

PPO 的裁剪虽然没有显式去算 Hessian，但它通过限制 $r_t(\theta)$ 的范围，间接避免了高阶项失控带来的风险。

---

## GRPO 的组归一化

前面讲了策略梯度、PPO 裁剪和 Taylor 展开，这些方法都需要一个优势估计 $\hat{A}_t$。传统方法（如 PPO）用训练好的 Critic 网络来估计优势，但 Critic 网络本身也需要训练，增加了工程复杂度。**GRPO 的核心想法是：不用 Critic，改用"同组回答之间的相对比较"来构造优势。** 想象老师批改一道开放题：四个学生的得分放在一起，高于平均的给正向信号，低于平均的给负向信号，不用额外请一个"标准分评判员"。具体来说，假设同一个 prompt 采样 4 个回答，奖励分别是：

$$
r=[2,4,6,8].
$$

均值是：

$$
\mu=\frac{2+4+6+8}{4}=5.
$$

标准差是：

$$
\sigma=
\sqrt{
\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}
}
=\sqrt{5}.
$$

第 4 个回答的标准化优势是：

$$
\hat{A}_4=\frac{8-5}{\sqrt{5}}\approx1.34.
$$

一般形式是：

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma}.
$$

整个计算分两步：

1. 减去均值：判断这个回答比组内平均好还是差。
2. 除以标准差：把不同题目的奖励尺度拉平——有的题分数天然偏高，有的偏低，除以标准差后就可以跨题比较。

GRPO 之所以能省掉传统 PPO 里的 Critic 网络，正是因为它用组内相对比较来构造 baseline。它不关心”这个回答绝对多少分”，只关心”这个回答在同组里排得怎么样”。

---

## 小结

本篇介绍了三个推导工具：

| 工具          | 核心公式                                         | 作用                            |
| ------------- | ------------------------------------------------ | ------------------------------- |
| 对数导数技巧  | $\nabla\pi = \pi\nabla\log\pi$                   | 把概率梯度改成可采样的 log 形式 |
| Taylor 展开   | $f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2$ | 理解 PPO 信任域和裁剪的二阶直觉 |
| GRPO 组归一化 | $\hat{A}_i=(r_i-\mu)/\sigma$                     | 用组内相对比较替代 Critic       |

这三个工具分别对应策略梯度的推导骨架、更新幅度的理论依据、以及不用 Critic 的替代方案。下一篇把它们整理成完整的公式速查表。

> **下一篇**：[D.3.5 强化学习优化公式汇总](./calculus-advanced-formulas)——集中整理 PG、DQN、GAE、PPO 与 GRPO 的公式。
