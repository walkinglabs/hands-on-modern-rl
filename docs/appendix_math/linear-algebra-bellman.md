# D.1.2 贝尔曼方程的矩阵形式

> **前置知识**：[D.1.1 向量与矩阵](./linear-algebra-basics)——需要了解向量、矩阵和矩阵乘法。同时建议先阅读第 3 章的[贝尔曼方程](../chapter03_mdp/value-bellman)，熟悉单个状态的贝尔曼方程形式。

---

## 从单个方程到矩阵方程

第 3 章给出了单个状态的贝尔曼方程：

$$
V^\pi(s) = \sum_{a} \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right].
$$

这行公式处理一个状态没有问题。但当环境包含 $n$ 个状态时，需要 $n$ 个这样的方程，每个状态一个。将这 $n$ 个方程合并，可以得到单一的矩阵表达：

$$
\boxed{\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}}
$$

其中 $\mathbf{v}$ 是所有状态的价值向量，$\mathbf{r}$ 是所有状态的即时奖励，$P$ 是状态转移矩阵。移项后还可以得到闭式解：

$$
\boxed{\mathbf{v} = (I - \gamma P)^{-1}\mathbf{r}}
$$

下面从单个方程出发，分三步推导出这两个公式：先将价值写成向量，再将转移关系写成矩阵，最后拼成方程组并求解。

---

## 把所有状态的价值排成向量

沿用附录导读中的两状态例子：

- 在 $s_1$，奖励是 $2$，下一步一定到 $s_2$。
- 在 $s_2$，奖励是 $1$，下一步一定到 $s_1$。
- 折扣因子 $\gamma = 0.5$。

两个状态的价值分别是 $v_1$ 和 $v_2$。将所有状态的价值排成一列向量：

$$
\mathbf{v} =
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix},
\qquad
\mathbf{r} =
\begin{bmatrix}
2 \\
1
\end{bmatrix}.
$$

$\mathbf{v}$ 是待求的量，$\mathbf{r}$ 是已知的即时奖励。两状态时向量有两个分量；推广到 $n$ 个状态，向量就有 $n$ 个分量，写法不变。

---

## 把转移关系写成矩阵

从 $s_1$ 出发一定到 $s_2$，从 $s_2$ 出发一定到 $s_1$。转移矩阵为：

$$
P =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}.
$$

矩阵的行对应"从哪个状态出发"，列对应"下一步到哪个状态"。矩阵乘向量的含义是：$P\mathbf{v}$ 的第 $i$ 行给出"从状态 $s_i$ 出发，下一步期望到达的价值"。验算如下：

$$
P\mathbf{v} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
v_2 \\
v_1
\end{bmatrix}.
$$

这正是在说：从 $s_1$ 出发下一步到 $s_2$，因此未来价值为 $v_2$；从 $s_2$ 出发下一步到 $s_1$，因此未来价值为 $v_1$。矩阵乘向量自动完成了概率加权求和。

---

## 拼成方程组

现在有了三样东西：价值向量 $\mathbf{v}$、奖励向量 $\mathbf{r}$、转移矩阵 $P$。将单个状态的贝尔曼方程逐个写出：

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

用矩阵语言重写右边：即时奖励是 $\mathbf{r}$，折扣后的未来价值是 $\gamma P\mathbf{v}$。二者相加：

$$
\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}.
$$

验证右边第一行：

$$
2 + 0.5 \times (0 \cdot v_1 + 1 \cdot v_2) = 2 + 0.5v_2.
$$

右边第二行：

$$
1 + 0.5 \times (1 \cdot v_1 + 0 \cdot v_2) = 1 + 0.5v_1.
$$

与逐个写出的方程完全一致。矩阵形式没有引入新内容，只是将大量结构相同的方程压缩为一个。

### 压缩有效的原理

关键在于 $P\mathbf{v}$ 这一步。$P$ 的每一行恰好是一组转移概率（行和为 $1$），矩阵乘法恰好是"概率 $\times$ 价值"的加权平均。贝尔曼方程表述"价值 = 即时奖励 + 折扣后的未来价值"，而矩阵方程表述的是完全相同的关系——只是对**所有状态同时**进行。

| 符号    | 含义                     | 维度  |
| ------- | ------------------------ | ----- |
| **v**   | 所有状态的价值（待求）   | n × 1 |
| **r**   | 所有状态的即时奖励       | n × 1 |
| γP**v** | 折扣后的概率加权未来价值 | n × 1 |

三个量的维度均为 $n \times 1$，等号两边形状一致。这也就是第 4 章中 DP（动态规划）方法背后的操作——反复应用 $v_{k+1} = r + \gamma P v_k$，直至收敛。

---

## 闭式解与不动点

$\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}$ 是一个线性方程组，因此可以直接求解。将含有 $\mathbf{v}$ 的项移至左边：

$$
\mathbf{v} - \gamma P\mathbf{v} = \mathbf{r}.
$$

提取公因子（$I$ 是单位矩阵——对角线为 $1$，其余为 $0$）：

$$
(I - \gamma P)\mathbf{v} = \mathbf{r}.
$$

若 $I - \gamma P$ 可逆，则存在闭式解：

$$
\mathbf{v} = (I - \gamma P)^{-1}\mathbf{r}.
$$

代入两状态的具体数值：

$$
I - \gamma P =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
- 0.5
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}.
$$

求解方程组：

$$
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\quad\Longrightarrow\quad
v_1 = 3.33,\quad v_2 = 2.67.
$$

### 交点就是不动点

这个方程组对应于 $(v_1, v_2)$ 平面上的两条直线：

- 第一个方程 $v_1 - 0.5v_2 = 2$
- 第二个方程 $-0.5v_1 + v_2 = 1$

两条直线的交点即为 $(3.33, 2.67)$。这个点同时也是贝尔曼更新的**不动点**：将 $(3.33, 2.67)$ 代入 $v_{new} = r + 0.5Pv$，得到的仍然是 $(3.33, 2.67)$。价值不再变化，说明这就是真实价值。

---

## 从两个状态到 $n$ 个状态

上述闭式解 $\mathbf{v} = (I-\gamma P)^{-1}\mathbf{r}$ 是在两个状态下推导的。将状态数从 2 推广到任意 $n$，方程的形式保持不变：

$$
\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi.
$$

其中：

- $\mathbf{v}_\pi \in \mathbb{R}^n$：策略 $\pi$ 下所有状态的价值
- $\mathbf{r}_\pi \in\mathbb{R}^n$：每个状态的期望即时奖励
- $P_\pi \in\mathbb{R}^{n\times n}$：策略诱导的转移矩阵（$P_\pi[i,j] = \sum_a \pi(a\mid s_i) p(s_j\mid s_i, a)$）

三个状态时，$P$ 为 $3\times3$，$\mathbf{v}$ 和 $\mathbf{r}$ 为 $3\times1$，方程 $\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}$ 依然成立。矩阵表示的核心优势正在于此：**无论状态数多大，方程的形式始终不变**。

### $I - \gamma P$ 的可逆性条件

求解的关键在于 $I - \gamma P$ 必须可逆。直观上，这要求贝尔曼更新不会发散。当 $0 < \gamma < 1$ 且 $P$ 是合法转移矩阵（每行概率和为 $1$）时，$I - \gamma P$ 几乎总是可逆的。

更严格地，$\gamma P$ 的谱半径（最大特征值的绝对值）满足 $\rho(\gamma P) \leq \gamma < 1$，因此 $I - \gamma P$ 的特征值均远离 $0$，矩阵必定可逆。D.1.4 将给出详细解释。

---

## 不直接求逆的原因

从推导来看，闭式解很简洁：写出矩阵形式，求逆，得到答案。但这一路径在实际中不可行，原因有三：

1. **规模过大。** 若状态数 $n=10^6$，矩阵 $I-\gamma P$ 为 $10^6 \times 10^6$，求逆的计算复杂度为 $O(n^3)$，几乎无法完成。
2. **矩阵未必显式存在。** 在许多实际问题中，$P$ 的具体数值不可知，只能通过采样观察状态转移。第 4 章介绍的 MC 和 TD 方法正是在不知 $P$ 的条件下运行的。
3. **状态可能连续。** 若状态是图像或文本，根本不存在有限大小的矩阵——无法构造 $P$。

实际算法采用迭代方法逼近这一解：

- **值迭代**：反复执行 $v_{k+1} = r + \gamma P v_k$，直至收敛。
- **策略评估**：在策略迭代中反复应用贝尔曼更新。
- **TD 学习**：利用采样数据进行增量更新。

这些方法本质上都是通过更可扩展的方式逼近 $(I-\gamma P)^{-1}\mathbf{r}$ 这一解，而无需实际求逆。第 4 章中 DP、MC、TD 三代方法的演进，对应的正是"已知模型直接迭代 → 不知模型用采样 → 采样仅需一步"这条路径。

::: warning 常见误区
看到 $\mathbf{v} = (I-\gamma P)^{-1}\mathbf{r}$ 时，不应认为实际算法真的在计算矩阵逆。这个公式是理论的闭式解，用以说明解的存在性与唯一性。实际算法是迭代的。
:::

---

## Q 函数的矩阵形式

前面只处理了状态价值 $V$。动作价值 $Q(s,a)$ 也可以写成矩阵形式，且与 $V$ 的矩阵形式之间存在清晰的代数联系。

### 符号定义

把所有 $(s,a)$ 对的 Q 值排成一个长向量：

$$
\mathbf{q} =
\begin{bmatrix}
Q(s_1, a_1) \\
Q(s_1, a_2) \\
\vdots \\
Q(s_2, a_1) \\
\vdots
\end{bmatrix}
\in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}.
$$

类似地，$\mathbf{r} \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ 存放每个 $(s,a)$ 对的即时奖励。

转移矩阵扩展为 $P \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}| \times |\mathcal{S}|}$，每一行对应一个 $(s,a)$ 对，每一列对应一个下一状态 $s'$：

$$
P[(s,a),\, s'] = P(s' \mid s, a).
$$

策略矩阵 $\Pi_\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}||\mathcal{A}|}$ 把 Q 向量"压缩"回 V 向量：

$$
\Pi_\pi[\,s,\, (s,a)\,] = \pi(a \mid s).
$$

### V-Q 关系

$$
\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi
$$

验证第 $i$ 行：$\sum_a \pi(a|s_i) Q(s_i, a) = V(s_i)$。这正是 $V(s) = \sum_a \pi(a|s) Q(s,a)$ 的矩阵写法。

### Q 的贝尔曼期望方程

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$

写成矩阵形式：

$$
\mathbf{q}_\pi = \mathbf{r} + \gamma P \mathbf{v}_\pi.
$$

将 $\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi$ 代入，得到纯 Q 的递推：

$$
\mathbf{q}_\pi = \mathbf{r} + \gamma P \Pi_\pi \mathbf{q}_\pi.
$$

闭式解为 $\mathbf{q}_\pi = (I - \gamma P \Pi_\pi)^{-1} \mathbf{r}$。

### Q 的贝尔曼最优方程

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')
$$

矩阵形式：

$$
\mathbf{q}_* = \mathbf{r} + \gamma P \cdot \mathrm{rowmax}(\mathbf{q}_*)
$$

其中 $\mathrm{rowmax}(\mathbf{q}) \in \mathbb{R}^{|\mathcal{S}|}$ 对每个状态取出该状态所有动作中的最大 Q 值。由于 max 不是线性运算，最优方程没有闭式解，只能通过迭代（如 Q-Learning）逼近。

### 从 Q 的矩阵形式推出 V 的矩阵形式

将 $\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi$ 代入 $\mathbf{q}_\pi = \mathbf{r} + \gamma P \mathbf{v}_\pi$，两边左乘 $\Pi_\pi$：

$$
\Pi_\pi \mathbf{q}_\pi = \Pi_\pi \mathbf{r} + \gamma \Pi_\pi P \mathbf{v}_\pi
\quad\Longrightarrow\quad
\mathbf{v}_\pi = \underbrace{\Pi_\pi \mathbf{r}}_{\mathbf{r}_\pi} + \gamma \underbrace{\Pi_\pi P}_{P_\pi} \mathbf{v}_\pi.
$$

这就是前面推导过的 $\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi$。V 的矩阵形式中 $\mathbf{r}_\pi$ 和 $P_\pi$ 已经把策略平均融进去了，而 Q 的矩阵形式保留了动作维度——策略平均由 $\Pi_\pi$ 单独完成。这正是"$Q$ 比 $V$ 携带更细粒度信息"的矩阵语言表达。

---

## DP 迭代的矩阵形式

第 4 章介绍了 DP 策略评估的逐状态更新：

$$
V(s) \leftarrow \sum_a \pi(a|s)\left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right].
$$

矩阵形式就是把贝尔曼期望方程拆成迭代：

$$
\mathbf{v}_{k+1} = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_k.
$$

每轮对所有状态同时做一次贝尔曼更新。前面用两状态例子演示过：从 $\mathbf{v}_0 = \mathbf{0}$ 开始，反复迭代会收敛到闭式解 $\mathbf{v} = (I - \gamma P_\pi)^{-1}\mathbf{r}_\pi$。

策略改进 $\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a)V^\pi(s')]$ 在矩阵视角下等价于：对每个状态 $s$，比较 $\mathbf{r} + \gamma P\mathbf{v}_\pi$ 中属于该状态的那几行（每行对应一个动作），选出值最大的动作。

---

## 对照总表

| 概念           | 逐状态形式（第 3 章）                                                                  | 矩阵形式                                                                  |
| -------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 贝尔曼期望方程 | $V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]$ | $\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi$           |
| 贝尔曼最优方程 | $V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]$                 | $\mathbf{v}_* = \mathbf{r}_* + \gamma P_* \mathbf{v}_*$（逐行取 max）     |
| 闭式解         | —                                                                                      | $\mathbf{v} = (I - \gamma P)^{-1}\mathbf{r}$                              |
| V-Q 关系       | $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$                                                | $\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi$                                 |
| Q 贝尔曼期望   | $Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')$    | $\mathbf{q}_\pi = \mathbf{r} + \gamma P \Pi_\pi \mathbf{q}_\pi$           |
| Q 贝尔曼最优   | $Q^*(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')$                      | $\mathbf{q}_* = \mathbf{r} + \gamma P \cdot\mathrm{rowmax}(\mathbf{q}_*)$ |
| DP 策略评估    | $V(s) \leftarrow \sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')]$         | $\mathbf{v}_{k+1} = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_k$           |

MC 和 TD 方法基于采样更新单个状态，没有对应的矩阵形式。

---

## 矩阵形式的局限性

本篇将贝尔曼方程压缩为矩阵形式 $\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}$，并给出了闭式解 $\mathbf{v} = (I-\gamma P)^{-1}\mathbf{r}$。无论状态数多少，方程始终是一行。

但矩阵形式隐含了一个前提：**每个状态都有独立的 $v(s)$ 可以存储**。

| 环境                         | 状态空间大小 | 能否存储 |
| ---------------------------- | ------------ | -------- |
| 两状态小环境                 | 2            | 是       |
| GridWorld 10×10              | 100          | 是       |
| 围棋棋盘                     | ~10¹⁷⁰       | 否       |
| 连续状态（如机器人关节角度） | 无穷         | 否       |

状态数量增大后，不仅矩阵求逆无法执行，连价值表本身都超出了存储能力。闭式解虽然形式简洁，但面对围棋的 $10^{170}$ 个状态，所需的存储量是不现实的。

解决思路是：不存储每个状态的值，而是用函数来**近似**表示价值。为状态提取特征，用特征与权重的点积计算价值——下一篇将展开这一思路。

> **下一篇**：[D.1.3 点积、范数与函数近似](./linear-algebra-function-approx) —— 当状态数量超出存储能力时，如何用特征向量和点积近似表示价值。
