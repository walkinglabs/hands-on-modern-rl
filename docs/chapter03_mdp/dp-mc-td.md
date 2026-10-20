# 4.1 动态规划、蒙特卡洛与时序差分

## 本节导读

**核心内容**

- 价值表：在有限状态问题里，先给每个状态存一个当前估计 $V(s)$。
- DP、MC、TD：三种更新价值表的方法，区别在于“目标值”从哪里来。
- TD Error：用“一步真实奖励 + 下一状态表格值”衡量当前格子预测偏差。

上一节讲了 $V^\pi(s)$ 的含义：从状态 $s$ 出发，按策略 $\pi$ 继续走，未来平均能拿多少回报。现在把这个概念落到实现上，它就是一张**价值表**。表里每一行对应一个状态，每个格子里存一个当前估计：

| 状态 | $V(s)$ 的当前估计 |
| ---- | ----------------- |
| $S$  | 0                 |
| $M$  | 0                 |
| $G$  | 0                 |

学习价值函数，本质上就是不断修改这张表。真正的问题不是“价值函数定义是什么”，而是：**每次要把表里的某个数改成什么？**

在上一节中，我们了解了如何用贝尔曼方程刻画状态价值。假设当前状态为 $s$，策略为 $\pi$，真实的状态价值为 $V^\pi(s)$。如果价值表已经准确，那么 $V^\pi(s)$ 应当满足下面的等式：

$$
V^\pi(s)
=
\mathbb{E}_\pi\left[
R_{t+1}+\gamma V^\pi(S_{t+1})
\mid S_t=s
\right].
$$

这个式子首先是一个自洽关系：左边是当前状态的真实价值，右边是“一步奖励加上下一个状态真实价值”的期望。换句话说，如果价值表已经正确，那么用下一步重新计算出来的结果，应该与表中原来的数值一致。

然而在学习开始时，我们手里的并不是 $V^\pi$，而是一张尚未填准的估计表 $V$。于是，贝尔曼方程在算法中被读成一种更新规则：先用当前能获得的信息计算等式右边，把这个量记作本次更新的目标值（target），再让旧的 $V(s)$ 向这个目标移动。这里的 target 并不是一个额外假设，而是贝尔曼等式右边在当前信息条件下的可计算版本。

本节就用“改表”的角度讲 DP、MC、TD。这里先说清一个词：**环境模型**指的是环境规则的数学描述，也就是在状态 $s$ 做动作 $a$ 后，会以什么概率到达各个下一状态 $s'$，以及平均会得到多少奖励。写成符号就是转移概率 $P(s'\mid s,a)$ 和奖励函数 $R(s,a)$。DP 知道这套规则，可以直接用模型算出一个贝尔曼目标；MC 不知道模型，只能等一整局结束，用完整回报改表；TD 也不知道模型，但不用等终点，走一步就用“即时奖励 + 下一状态当前表格值”改表。三种方法估计的是同一张价值表，区别只是更新目标来自哪里。

> 已经知道价值函数应该满足贝尔曼关系以后，怎样在有限状态问题中真正计算这些价值？如果环境模型未知，又能不能只靠采样经验来更新价值？

::: info 核心概念
DP、MC、TD 都是在更新价值表。DP 用已知规则算目标，MC 用整局真实回报算目标，TD 用一步真实奖励和下一格当前估计算目标。
:::

**核心公式**

$$
V_{k+1}(s)
\leftarrow
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')
\right]
\quad \text{（DP 策略评估：已知模型时迭代价值表）}
$$

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right]
\quad \text{（MC 更新：用完整回报修正当前估计）}
$$

$$
V(s)\leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right]
\quad \text{（TD(0) 更新：走一步后立刻自举更新）}
$$

$$
\delta=r+\gamma V(s')-V(s)
\quad \text{（TD Error：一步贝尔曼目标与旧预测的差）}
$$

> **把公式读成“改表”：**
>
> - $V(s)$：价值表中状态 $s$ 这一格的当前数字。
> - $V_k(s)$：第 $k$ 轮扫表时，状态 $s$ 这一格的旧数字。
> - $P(s'\mid s,a)$、$R(s,a)$：DP 用来算目标的环境规则。
> - $G_t$：MC 等整局结束后看到的完整回报。
> - $r+\gamma V(s')$：TD 走一步后立刻构造的目标。
> - $\alpha$：每次把旧数字往目标值移动多少。
> - $\delta$：TD 目标和旧表格值之间的差。

<span id="策略评估"></span>

## 价值表怎么被更新

上一节里，价值函数回答的是”从这个状态出发，未来大概值多少”。现在把问题再收窄一点：策略 $\pi$ 已经给定，我们先把它当作要评估的对象，看它照当前规则行动时，每个状态应该记多少分。比如在走廊里总是偏向右走，或者在 GridWorld 里按某个固定规则选方向，我们关心的是：从状态 $s$ 开始继续照这套策略走，平均能拿多少回报？这个数就是 $V^\pi(s)$。

上一节已经给出了这个目标的数学定义：

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\qquad
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

$V^\pi(s)$ 是”从 $s$ 出发、按 $\pi$ 行动”之后所有可能未来回报的平均值。但这个定义本身还不是一个可执行的更新步骤。要精确计算它，必须知道从 $s$ 出发的所有可能轨迹、每条轨迹的概率，以及对应的完整回报。在大多数实际问题中，这些信息无法逐一列出。

这正是 Sutton 与 Barto 在第 6 章所称的 prediction problem：给定策略 $\pi$，估计它的价值函数 $v_\pi$[^4]。算法在学习过程中不会直接拿到真实的 $v_\pi$，因为它是一个条件期望——是从 $s$ 出发、继续按 $\pi$ 行动时所有可能未来回报的平均，而不是某一条具体轨迹上的回报。实际中，算法只能通过采样得到若干条轨迹，无法穷尽所有可能。因此它只能维护一个当前估计 $V$，再用每次新得到的信息去逐步修正。

这就需要一个关键的中介：用一个**当前可获得的估计量**来替代未知的真实 $v_\pi$，去修正 $V(S_t)$。Sutton 与 Barto 直接把这个估计量称为 target。

三种经典方法的核心差异，就在于如何构造这个 target。

**DP** 假设已知完整的环境模型。它利用贝尔曼期望方程，把策略下所有动作分支和下一状态分支全部展开，直接计算期望。因为模型已经给出了所有分支的概率和奖励，不需要实际进入环境采样，遍历一遍状态表即可完成一轮更新。

**MC** 不假设知道模型。它等到一次完整的 episode 结束后，把实际发生的总回报 $G_t$ 直接当作 target。$G_t$ 是这条轨迹上从该状态出发的真实回报样本，不是模型算出的平均值。它的更新只能在 episode 结束后进行。

**TD** 同样不知道模型，但它不等 episode 结束。每走一步，它就把当前观察到的即时奖励 $R_{t+1}$ 和下一状态的当前估计值 $V(S_{t+1})$ 组合成 target，即 $R_{t+1} + \gamma V(S_{t+1})$。

TD 能这样做的原因是回报具有递归结构[^1]：

$$
G_t = R_{t+1} + \gamma G_{t+1}.
$$

$G_{t+1}$ 是从下一状态开始的完整回报。在 $t+1$ 时刻，$G_{t+1}$ 尚未发生，TD 就用表里对下一状态的当前估计 $V(S_{t+1})$ 代替它：

$$
\text{target}_{\mathrm{TD}} = R_{t+1}+\gamma V(S_{t+1}).
$$

这一步叫自举（bootstrapping）。它让 TD 能一步一更新，也会带来后面要讨论的偏差。

三种方法虽然 target 不同，但更新都可以归入同一个框架：把当前估计向 target 移动。

$$
V(s)\leftarrow V(s)+\alpha\left[\text{target}-V(s)\right].
$$

$V(s)$ 是表里的旧数字，$\text{target}$ 是这次算出来的新估计，$\alpha$ 控制步幅。$\text{target}-V(s)$ 表示”这次认为旧表错了多少”；乘上 $\alpha$，就是这次实际改多少。$\alpha=1$ 时直接覆盖旧值，$\alpha$ 较小时只往目标方向挪一小步。

因此，本节的重点不是分别学三种算法，而是比较三种构造 target 的方式。**三种方法都在改同一张价值表，区别只在 target 的来源。**

先看一个小走廊。它只有三个状态，但已经能体现“按策略平均”这件事：

![可左右移动的三格走廊](./images/three-state-corridor.svg)

| 当前状态 | 动作 | 策略概率 | 下一状态 | 奖励 |
| -------- | ---- | -------- | -------- | ---- |
| $S$      | 左走 | 0.2      | $S$      | $-2$ |
| $S$      | 右走 | 0.8      | $M$      | $-1$ |
| $M$      | 左走 | 0.2      | $S$      | $-2$ |
| $M$      | 右走 | 0.8      | $G$      | $-1$ |
| $G$      | 结束 | 1.0      | 无       | $0$  |

这里 $S$ 是起点，$M$ 是中间格，$G$ 是终点，到达终点后结束。向右前进扣 1 分；左走代表走错方向，代价更高，扣 2 分。$S$ 左走会撞墙，仍然留在 $S$；$M$ 左走会回到 $S$。

::: details 为什么这里不是每步 +1？
先看直觉：这个走廊任务想表达的不是“走路本身值得奖励”，而是**“尽快到达终点”**。每多走一步，就多花一次时间、能量或机会成本，所以这里把普通前进写成 $r=-1$，把走错方向写成代价更高的 $r=-2$。

如果把每一步都设成 $+1$，**回报的含义就会变掉**。走一步到终点得到 $+1$，走两步到终点得到 $+2$，绕很多步再到终点反而可能得到更大的累计回报。这样一来，智能体最大化回报时，未必会选择最短路线；在没有严格步数上限的任务里，它甚至可能学会来回走，因为**“多走一步”本身也在加分**。

所以，在这类“到达目标后结束”的例子里，常把每一步奖励写成负数。最大化回报并不是追求更多惩罚，而是在一组负数里选择**没那么负**的结果：$-2$ 比 $-10$ 好。换句话说，这里的价值可以读成**“从当前位置到结束还要付出多少未来代价”**。离终点越近，未来要扣的分越少，价值也就越接近 $0$。

当然，奖励并不一定必须是负数。也可以设计成“到达终点给 $+1$，其他步骤给 $0$”。那是另一种任务表达。当前例子选择每步负奖励，是为了让 DP、MC、TD 的价值更新更清楚地展示：**算法怎样把未来的步数代价一步步传播回前面的状态**。
:::

为了做策略评估，**我们固定一套策略**：**在 $S$ 和 $M$ 都有 $0.8$ 的概率右走、$0.2$ 的概率左走**。它大多数时候朝终点走，但偶尔会退回去。这样一来，$V(M)$ 不再只是“还差一步”，因为从 $M$ 也可能左走回 $S$；$V(S)$ 也不再只是“两步到终点”，因为起点可能先撞墙。

接下来要看的不是直接解出答案，而是从一张初始表出发，怎样把 $V(S)$ 和 $V(M)$ 改到当前策略对应的位置。下面三节分别只追问一件事：在同一张表上，DP、MC、TD 各自怎样构造 $\text{target}$。

## 动态规划

上一节推导出贝尔曼期望方程：

$$
V^\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s' \mid s) V^\pi(s').
$$

其中 $r_\pi(s)$ 是策略在状态 $s$ 的期望即时奖励，$P_\pi(s' \mid s)$ 是策略下的状态到状态转移概率。动态规划（dynamic programming, DP）从这条等式出发，利用一个前提：**环境模型已知**。算法知道每个动作会把智能体带到哪些下一状态、各自的转移概率 $P(s' \mid s,a)$ 是多少、这一步的平均奖励 $R(s,a)$ 是多少。

模型已知意味着可以直接使用 $R(s,a)$ 和 $P(s' \mid s,a)$ 的具体数值。以走廊里的 $S$ 为例：

$$
r_\pi(S) = \underbrace{\pi(\text{右} \mid S)}_{0.8} \cdot \underbrace{R(S, \text{右})}_{-1} + \underbrace{\pi(\text{左} \mid S)}_{0.2} \cdot \underbrace{R(S, \text{左})}_{-2} = -1.2.
$$

$$
P_\pi(M \mid S) = \underbrace{\pi(\text{右} \mid S)}_{0.8} \cdot \underbrace{P(M \mid S, \text{右})}_{1} + \underbrace{\pi(\text{左} \mid S)}_{0.2} \cdot \underbrace{P(M \mid S, \text{左})}_{0} = 0.8,
$$

另一分支 $P_\pi(S \mid S) = 0.2$ 由归一性可得。

这里 $r_\pi(S)$ 是期望即时奖励，$P_\pi(s' \mid S)$ 是策略下的总转移概率。一般形式即

$$
r_\pi(s) = \sum_a \pi(a \mid s) R(s,a), \quad P_\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s,a).
$$

代入贝尔曼方程，得到完全展开形式：

$$
\text{target}_{\mathrm{DP}}(s)=
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')
\right].
$$

这个式子有两层：外层按策略 $\pi$ 对动作加权，内层按环境转移概率对下一状态加权。各符号含义如下：

| 符号                             | 含义                                         |
| -------------------------------- | -------------------------------------------- |
| $\text{target}_{\mathrm{DP}}(s)$ | 这一次准备写进状态 $s$ 的新目标值。          |
| $a$                              | 在状态 $s$ 可以选择的动作，比如左走、右走。  |
| $\pi(a\mid s)$                   | 当前固定策略在状态 $s$ 选择动作 $a$ 的概率。 |
| $R(s,a)$                         | 执行动作 $a$ 后，这一步拿到的平均奖励。      |
| $s'$                             | 动作之后可能到达的下一状态。                 |
| $P(s'\mid s,a)$                  | 在状态 $s$ 做动作 $a$ 后，到达 $s'$ 的概率。 |
| $V_k(s')$                        | 第 $k$ 轮旧表里，下一状态 $s'$ 的价值估计。  |
| $\gamma$                         | 折扣因子，决定下一状态价值要打多少折扣。     |

先看中括号内部：

$$
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s').
$$

它是在问：**如果这一步先做动作 $a$，眼前奖励是多少，后面接上的旧价值平均是多少？** 内层求和 $\sum_{s'}$ 是对环境可能给出的下一状态求平均。

再看外层：

$$
\sum_a\pi(a\mid s)[\cdots].
$$

它是在问：**当前策略会以不同概率选择不同动作，所以这些动作后果也要按策略概率平均。** 这就是 DP 策略评估的目标值。

注意这里没有 $\max_a$。DP 策略评估不是在替智能体选最好的动作，而是在评价已经给定的策略 $\pi$。策略怎么选动作，就照它的概率去平均。

现在把这个 target 公式用到走廊的具体例子上。状态转移本身没有随机性：**选了右走，就去右边**；**选了左走，就按规则撞墙或退回 $S$**。随机性来自策略本身：同一个状态下，它有 $0.8$ 的概率右走，也有 $0.2$ 的概率左走。DP 要评估的是这套策略，所以不能只看“更好的那条路”，而要把两种动作后果按概率混在一起。

取 $\gamma=1$。实际算的时候，先从完整的求和式开始，再把这个走廊里的动作、概率、奖励和下一状态代进去。因为转移是确定的，内层 $\sum_{s'}$ 里只有真正到达的下一状态概率为 1，其他下一状态概率都是 0。比如 **$S$ 右走** 时，只有 $P(M\mid S,\text{右})=1$，所以这一支会留下 $V_{\text{old}}(M)$。

对 $S$ 来说，动作集合只有“右走”和“左走”，所以外层的 $\sum_a$ 会展开成两项：

$$
\begin{aligned}
\text{target}_{\mathrm{DP}}(S)
&=\sum_a\pi(a\mid S)
\left[
R(S,a)+\sum_{s'}P(s'\mid S,a)V_{\text{old}}(s')
\right]\\
&=\pi(\text{右}\mid S)
\left[R(S,\text{右})+P(M\mid S,\text{右})V_{\text{old}}(M)\right]\\
&\quad+\pi(\text{左}\mid S)
\left[R(S,\text{左})+P(S\mid S,\text{左})V_{\text{old}}(S)\right]\\
&=0.8[-1+1\cdot V_{\text{old}}(M)]+0.2[-2+1\cdot V_{\text{old}}(S)].
\end{aligned}
$$

对 $M$ 也是一样，只是右走会到终点 $G$，左走会退回 $S$：

$$
\begin{aligned}
\text{target}_{\mathrm{DP}}(M)
&=\sum_a\pi(a\mid M)
\left[
R(M,a)+\sum_{s'}P(s'\mid M,a)V_{\text{old}}(s')
\right]\\
&=\pi(\text{右}\mid M)
\left[R(M,\text{右})+P(G\mid M,\text{右})V_{\text{old}}(G)\right]\\
&\quad+\pi(\text{左}\mid M)
\left[R(M,\text{左})+P(S\mid M,\text{左})V_{\text{old}}(S)\right]\\
&=0.8[-1+1\cdot V_{\text{old}}(G)]+0.2[-2+1\cdot V_{\text{old}}(S)].
\end{aligned}
$$

也可以把这两个展开式读成一张分支表：

| 要更新的状态 | 右走分支                    | 左走分支                    | 加权后写入  |
| ------------ | --------------------------- | --------------------------- | ----------- |
| $S$          | $0.8[-1+V_{\text{old}}(M)]$ | $0.2[-2+V_{\text{old}}(S)]$ | 新的 $V(S)$ |
| $M$          | $0.8[-1+V_{\text{old}}(G)]$ | $0.2[-2+V_{\text{old}}(S)]$ | 新的 $V(M)$ |

这张表的读法很直接。下面四张图把对应的线加粗了：

| 分支                                       | 对应图                                                                                                            |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **从 $S$ 右走**：付出 $-1$，到达 $M$       | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="高亮 S 右走分支" style="max-width: 320px;" /> |
| **从 $S$ 左走**：付出 $-2$，撞墙并留在 $S$ | <img src="./images/three-state-corridor-highlight-s-left.svg" alt="高亮 S 左走分支" style="max-width: 320px;" />  |
| **从 $M$ 右走**：付出 $-1$，到达终点 $G$   | <img src="./images/three-state-corridor-highlight-m-right.svg" alt="高亮 M 右走分支" style="max-width: 320px;" /> |
| **从 $M$ 左走**：付出 $-2$，退回 $S$       | <img src="./images/three-state-corridor-highlight-m-left.svg" alt="高亮 M 左走分支" style="max-width: 320px;" />  |

每一轮都读旧表，把这些分支的结果按策略概率加权，算出新表。

先从全 0 的旧表开始：

| 初始旧表 | $V(S)$ | $V(M)$ | $V(G)$ |
| -------- | ------ | ------ | ------ |
| 第 0 轮  | 0      | 0      | 0      |

用上面的展开式代入旧表数值。第一轮时 $V_0$ 全为 0，所以目标只剩下眼前动作代价的平均：

$$
\begin{aligned}
V_1(S) &= 0.8[-1+V_0(M)] + 0.2[-2+V_0(S)] \\
       &= 0.8(-1+0) + 0.2(-2+0) \\
       &= -1.2.
\end{aligned}
$$

$$
\begin{aligned}
V_1(M) &= 0.8[-1+V_0(G)] + 0.2[-2+V_0(S)] \\
       &= 0.8(-1+0) + 0.2(-2+0) \\
       &= -1.2.
\end{aligned}
$$

第二轮把第一轮结果当作旧表：

| 第一轮后的旧表 | $V(S)$ | $V(M)$ | $V(G)$ |
| -------------- | ------ | ------ | ------ |
| 第 1 轮        | -1.2   | -1.2   | 0      |

每个分支不只包含眼前奖励，还会接上下一状态的旧价值：

$$
\begin{aligned}
V_2(S) &= 0.8[-1+V_1(M)] + 0.2[-2+V_1(S)] \\
       &= 0.8[-1+(-1.2)] + 0.2[-2+(-1.2)] \\
       &= -2.4.
\end{aligned}
$$

$$
\begin{aligned}
V_2(M) &= 0.8[-1+V_1(G)] + 0.2[-2+V_1(S)] \\
       &= 0.8(-1+0) + 0.2[-2+(-1.2)] \\
       &= -1.44.
\end{aligned}
$$

再用第二轮的结果做一次：

| 第二轮后的旧表 | $V(S)$ | $V(M)$ | $V(G)$ |
| -------------- | ------ | ------ | ------ |
| 第 2 轮        | -2.4   | -1.44  | 0      |

$$
\begin{aligned}
V_3(S) &= 0.8[-1+V_2(M)] + 0.2[-2+V_2(S)] \\
       &= 0.8[-1+(-1.44)] + 0.2[-2+(-2.4)] \\
       &= -2.832.
\end{aligned}
$$

$$
\begin{aligned}
V_3(M) &= 0.8[-1+V_2(G)] + 0.2[-2+V_2(S)] \\
       &= 0.8(-1+0) + 0.2[-2+(-2.4)] \\
       &= -1.68.
\end{aligned}
$$

把每轮结束后的价值表放在一起看，数字的变化方向就很清楚了：

| 轮次 | $V(S)$ | $V(M)$ | $V(G)$ |
| ---- | ------ | ------ | ------ |
| 0    | 0      | 0      | 0      |
| 1    | -1.2   | -1.2   | 0      |
| 2    | -2.4   | -1.44  | 0      |
| 3    | -2.832 | -1.68  | 0      |
| 收敛 | -3.375 | -1.875 | 0      |

这个例子比单向走廊更能体现 DP 的含义：它不是只把终点价值向前传，而是在每个状态上计算“按当前策略行动时的平均后果”。右走通常更好，但策略偶尔会左走；这部分绕路和撞墙的代价也必须进入价值表。随着一轮轮更新，$S$ 和 $M$ 的数字逐渐稳定下来，最后得到的不是最优价值，而是这套固定策略本身的价值。

评估完成后，DP 还可以继续做策略改进。此时问题换成：如果在状态 $s$ 先试一个动作 $a$，之后仍然按原策略 $\pi$ 行动，这个动作值多少？在模型已知时，这个动作分数可以直接由 $V^\pi$ 算出来：

$$
Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s'),
$$

再选动作分数最高的动作：

$$
\pi'(s)=\arg\max_a Q^\pi(s,a).
$$

这就是策略迭代的两步：先评估当前策略，再根据价值表改进策略。需要注意的是，无论评估还是改进，DP 都依赖同一个强前提：环境模型 $P$ 和 $R$ 必须已知。现实中的机器人控制、游戏任务和大模型生成通常没有这样的完整说明书。因此，DP 更像是理论基准：它说明如果知道一切，价值表应该怎样被计算。

## 蒙特卡洛

蒙特卡洛（Monte Carlo, MC）是第二种改表方法：**不知道环境模型，就用整局真实回报当目标**。

Monte Carlo 方法是一类用随机抽样和大量模拟来近似求解复杂问题的数值方法，核心是**用频率逼近概率、用样本均值逼近期望**。它起源于 1940 年代曼哈顿计划，数学家冯·诺伊曼与乌拉姆为模拟中子扩散等难题而提出；代号取自摩纳哥赌城，象征随机性。在强化学习中，就是让智能体按策略实际走多局，用每条轨迹上真实拿到的完整回报来估计状态价值。

DP 的展开式里需要知道 $P(s'\mid s,a)$ 和 $R(s,a)$，才能按概率对所有动作和下一状态求平均。MC 去掉这个前提：**环境模型未知**。

模型未知意味着不能枚举分支算期望。但智能体仍可以进入环境，按策略行动，观察到真实的轨迹。轨迹结束后，从时刻 $t$ 开始的整段折扣回报

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

就是真实发生的样本。MC 把它作为 target：

$$
\text{target}_{\mathrm{MC}} = G_t.
$$

多次访问同一个状态后，这些完整回报的平均值会逼近 $V^\pi(s)$。实际写程序时，不一定保存所有历史回报再求平均，也可以让旧值向本次回报移动一步：

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right].
$$

其中 $G_t-V(s)$ 是“这次真实回报”和“表里旧估计”的差。如果这次回报比旧估计高，表格值会上调；如果更低，表格值会下调。

在这个可左右走的走廊里，从 $S$ 出发不再只有一条轨迹。某一局可能很顺利：

$$
S\xrightarrow{-1}M\xrightarrow{-1}G.
$$

也可能先左走撞墙，或者从 $M$ 左走退回 $S$。例如某次采样得到如下轨迹：

$$
S\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-1}G.
$$

各步对应的状态转移如下：

| 步骤 | 实际发生的一步           | 对应图                                                                                                                        |
| ---- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1    | **$S\xrightarrow{-2}S$** | <img src="./images/three-state-corridor-highlight-s-left.svg" alt="采样轨迹第 1 步：S 左走撞墙" style="max-width: 260px;" />  |
| 2    | **$S\xrightarrow{-1}M$** | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="采样轨迹第 2 步：S 右走到 M" style="max-width: 260px;" /> |
| 3    | **$M\xrightarrow{-2}S$** | <img src="./images/three-state-corridor-highlight-m-left.svg" alt="采样轨迹第 3 步：M 左走回 S" style="max-width: 260px;" />  |
| 4    | **$S\xrightarrow{-1}M$** | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="采样轨迹第 4 步：S 右走到 M" style="max-width: 260px;" /> |
| 5    | **$M\xrightarrow{-1}G$** | <img src="./images/three-state-corridor-highlight-m-right.svg" alt="采样轨迹第 5 步：M 右走到 G" style="max-width: 260px;" /> |

MC 等整局结束后才更新。从每条轨迹倒着数，把从每次访问位置开始到终点的回报累加起来。取 $\gamma=1$，这条轨迹的各次访问目标如下：

| 访问位置 | 状态 | 后面实际拿到的奖励 | MC 目标 $G_t$ |
| -------- | ---- | ------------------ | ------------- |
| 第 1 步  | $S$  | $-2,-1,-2,-1,-1$   | $-7$          |
| 第 2 步  | $S$  | $-1,-2,-1,-1$      | $-5$          |
| 第 3 步  | $M$  | $-2,-1,-1$         | $-4$          |
| 第 4 步  | $S$  | $-1,-1$            | $-2$          |
| 第 5 步  | $M$  | $-1$               | $-1$          |

接下来才把这些完整回报代入更新式。这里每一行都在用同一个公式：

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right].
$$

初始价值表全为 0，学习率 $\alpha=0.5$，使用每次访问都更新的 MC。更新式为

$$
V(s) \leftarrow V(s) + 0.5\left[G_t - V(s)\right].
$$

$G_t$ 是从该访问位置到终点的完整回报。第 1 次访问 $S$ 时，后续奖励为 $-2,-1,-2,-1,-1$，故 $G_t=-7$；第 2 次访问 $S$ 时，该位置之后仅包含 $-1,-2,-1,-1$，故 $G_t=-5$。

| 被更新的状态 | MC 目标 | 旧值  | 新值                           |
| ------------ | ------- | ----- | ------------------------------ |
| 第 1 次 $S$  | $-7$    | 0     | $0+0.5(-7-0)=-3.5$             |
| 第 2 次 $S$  | $-5$    | -3.5  | $-3.5+0.5[-5-(-3.5)]=-4.25$    |
| 第 1 次 $M$  | $-4$    | 0     | $0+0.5(-4-0)=-2$               |
| 第 3 次 $S$  | $-2$    | -4.25 | $-4.25+0.5[-2-(-4.25)]=-3.125$ |
| 第 2 次 $M$  | $-1$    | -2    | $-2+0.5[-1-(-2)]=-1.5$         |

MC 的目标 $G_t$ 覆盖从当前访问到终点的全部奖励。episode 结束之前，从中间状态出发的完整回报尚未确定，因此无法计算 $G_t$，也不能更新该状态的价值估计。只有 episode 结束后，整条轨迹的奖励全部已知，才能从终点倒推，依次算出每次访问的 $G_t$，再回头更新。

MC 用真实发生的完整回报作为目标，在策略固定且采样充分时，样本均值收敛到真实期望。因此 MC 目标无偏。

但 $G_t$ 包含从当前时刻到终点的所有随机性。轨迹越长，不同采样间的回报波动越大，需要大量轨迹才能抵消这种波动。此外，许多任务没有自然终点，等待完整回报会让学习信号来得太晚。

## 时序差分

时序差分（temporal difference, TD）是第三种改表方法：**不知道规则，也不等整局结束，走一步就改表**[^3]。

Temporal difference 的字面意思是"时间上的差异"——用当前预测和走一步后的新估计之间的差来更新价值表。它既不需要 DP 那样知道完整模型，也不需要 MC 那样等整局结束。

从 MC 到 TD，又拿掉了一个前提：**不再等待完整回报 $G_t$**。MC 需要一整条 episode 结束才能更新；TD 每走一步就用即时奖励和下一状态的当前估计来更新。

TD 用贝尔曼方程作为依据。上一节推导出：

$$
V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right].
$$

等式右边是”一步奖励 + 下一状态价值”的期望。这个期望可以用样本估计：走一步后，$R_{t+1}$ 和 $S_{t+1}$ 都已经真实发生，只要把真实的 $V^\pi(S_{t+1})$ 换成表里已有的估计 $V(S_{t+1})$，就得到 TD 的目标：

$$
\text{target}_{\mathrm{TD}} = R_{t+1} + \gamma V(S_{t+1}).
$$

对应的改表规则是

$$
V(s)\leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right].
$$

括号里的差值就是 TD 误差（TD error）：

$$
\delta=r+\gamma V(s')-V(s).
$$

它度量了旧预测和一步目标之间的偏差。$\delta>0$ 表示目标值高于当前估计，价值估计向增大的方向修正；$\delta<0$ 表示目标值低于当前估计，价值估计向减小的方向修正；$\delta=0$ 表示当前估计与一步目标一致。

回到同一条采样轨迹：

$$
S\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-1}G.
$$

初始表仍然全为 0，学习率仍取 $\alpha=0.5$。这一次不等整局结束，而是每走一步就改一次。每一行都按同一个模板算：

$$
\text{新值}=\text{旧值}+0.5(\text{TD 目标}-\text{旧值}).
$$

注意 TD 读的是**当下这张表**。第 1 步更新完 $S$ 以后，第 3 步如果从 $M$ 左走回 $S$，就会立刻读到已经变成 $-1$ 的 $V(S)$。这就是 TD 和 MC 的一个明显差别：MC 在整局结束后统一回头算；TD 在轨迹进行中不断把刚学到的数拿来继续用。

| 步骤 | 对应线                                                                                                                   | 实际发生的一步           | 被更新的旧值 | TD 目标 $r+V(s')$     | 写入新值                         |
| ---- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ | ------------ | --------------------- | -------------------------------- |
| 1    | <img src="./images/three-state-corridor-highlight-s-left.svg" alt="TD 第 1 步：S 左走撞墙" style="max-width: 220px;" />  | **$S\xrightarrow{-2}S$** | $V(S)=0$     | $-2+V(S)=-2+0=-2$     | $V(S)=0+0.5(-2-0)=-1$            |
| 2    | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="TD 第 2 步：S 右走到 M" style="max-width: 220px;" /> | **$S\xrightarrow{-1}M$** | $V(S)=-1$    | $-1+V(M)=-1+0=-1$     | $V(S)=-1+0.5[-1-(-1)]=-1$        |
| 3    | <img src="./images/three-state-corridor-highlight-m-left.svg" alt="TD 第 3 步：M 左走回 S" style="max-width: 220px;" />  | **$M\xrightarrow{-2}S$** | $V(M)=0$     | $-2+V(S)=-2-1=-3$     | $V(M)=0+0.5(-3-0)=-1.5$          |
| 4    | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="TD 第 4 步：S 右走到 M" style="max-width: 220px;" /> | **$S\xrightarrow{-1}M$** | $V(S)=-1$    | $-1+V(M)=-1-1.5=-2.5$ | $V(S)=-1+0.5[-2.5-(-1)]=-1.75$   |
| 5    | <img src="./images/three-state-corridor-highlight-m-right.svg" alt="TD 第 5 步：M 右走到 G" style="max-width: 220px;" /> | **$M\xrightarrow{-1}G$** | $V(M)=-1.5$  | $-1+V(G)=-1+0=-1$     | $V(M)=-1.5+0.5[-1-(-1.5)]=-1.25$ |

把更新后的表单独列出来，可以更容易看清 TD 的节奏：

| 时刻 | 刚发生的一步         | $V(S)$ | $V(M)$ | 读法                                                   |
| ---- | -------------------- | ------ | ------ | ------------------------------------------------------ |
| 初始 | 尚未开始             | 0      | 0      | 初始估计。                                             |
| 1    | $S\xrightarrow{-2}S$ | -1     | 0      | $S$ 左走撞墙，即时奖励 $-2$ 使 $V(S)$ 下降。           |
| 2    | $S\xrightarrow{-1}M$ | -1     | 0      | 右走到 $M$，但 $V(M)=0$，$V(S)$ 不变。                 |
| 3    | $M\xrightarrow{-2}S$ | -1     | -1.5   | $M$ 左走回 $S$，$V(S)$ 已降至 $-1$，故 $V(M)$ 被拉低。 |
| 4    | $S\xrightarrow{-1}M$ | -1.75  | -1.5   | $S$ 右走接 $V(M)=-1.5$，$V(S)$ 继续下降。              |
| 5    | $M\xrightarrow{-1}G$ | -1.75  | -1.25  | $M$ 右走到 $G$，一步代价 $-1$，$V(M)$ 回升。           |

MC 与 TD 的差异在上述表格中清晰可见。MC 在 episode 结束后计算从每次访问到终点的完整回报；TD 每走一步就用即时奖励与下一状态当前估计构造目标。

以第 1 步 $S\xrightarrow{-2}S$ 为例。MC 在 episode 结束后给出的目标是 $-7$（完整轨迹回报）；TD 当场给出的目标是 $-2+V(S)=-2$（仅一步信息加当前估计）。TD 目标的信息量更少，但更新更及时。随着后续状态的估计被逐步修正，前面的状态估计也随之更新。

将已有估计代入目标来更新另一个估计的方式称为自举（bootstrapping）。自举使 TD 更新及时、方差较低，但引入偏差：若 $V(s')$ 本身不准确，$r+\gamma V(s')$ 的目标也会不准确。TD 的有效性依赖于连续修正：后继状态估计趋准后，前面状态的估计被逐步带准。

TD 误差在后续章节会反复出现。Critic 的训练、优势函数的估计、GAE 的构造，都可以看成对 TD 误差的不同使用方式。这里先记住它最基本的含义：TD 误差是价值估计与一步贝尔曼目标之间的差。

## 三种方法的关系

DP、MC 和 TD 不是三套独立的算法，而是同一个问题在不同信息条件下的三种求解方式。

它们共享同一个起点——贝尔曼期望方程（3.1 节）：

$$
V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right].
$$

要更新 $V(s)$，本质上都在估计这个期望。算法在学习过程中拿不到精确值，只能维护一个当前估计 $V$，再用每次新获得的信息构造一个 target 来逐步修正它。

三种方法的分化，来自"能拿到什么信息"这个约束的差异。下面从一般到特殊，依次推导三种 target。

**DP：模型已知，直接展开期望。**

当转移概率 $P(s'\mid s,a)$ 和期望奖励 $R(s,a)$ 已知时，贝尔曼期望方程可以完全展开。外层按策略对动作求平均，内层按转移概率对下一状态求平均：

$$
\text{target}_{\mathrm{DP}}(s)
=
\sum_a
\underbrace{\pi(a\mid s)}_{\text{策略：动作概率}}
\left[
\underbrace{R(s,a)}_{\text{奖励模型}}
+\gamma
\sum_{s'}
\underbrace{P(s'\mid s,a)}_{\text{转移模型}}
\underbrace{V_k(s')}_{\text{旧价值表}}
\right].
$$

DP 不需要实际采样，遍历状态表做同步更新即可迭代求解。

**MC：模型未知，用完整轨迹的回报替代。**

$P$ 和 $R$ 未知时，无法展开期望。但 $V^\pi(s)$ 的原始定义仍在：

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s].
$$

一条 episode 结束后，从时刻 $t$ 开始的完整回报

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

就是上述期望的一个样本。MC 用 $G_t$ 作为 target：

$$
\text{target}_{\mathrm{MC}} = G_t.
$$

样本均值在采样充分时收敛到真实期望，因此 MC 目标无偏，但更新必须等到 episode 结束。

**TD：不等完整未来，用一步自举。**

MC 已经不需要模型了，但它还要等待完整 $G_t$。利用回报的递归结构

$$
G_t = r_{t+1} + \gamma G_{t+1},
$$

将尚未观测的 $G_{t+1}$ 替换为当前价值表中的估计 $V_k(S_{t+1})$，得到：

$$
\text{target}_{\mathrm{TD}} = r_{t+1} + \gamma V_k(S_{t+1}).
$$

TD 保留真实发生的一步，用已有估计代替未知未来，每走一步就能更新。

三种 target 的信息条件逐步弱化，形成一条退化链：

| 方法              | 从哪个 target 出发                                               | 缺少什么           | 替代方式                           | 得到                          |
| ----------------- | ---------------------------------------------------------------- | ------------------ | ---------------------------------- | ----------------------------- |
| 贝尔曼完整 target | $\mathbb{E}_\pi[r+\gamma V(s')\mid s]$                           | 无                 | 完整展开动作、奖励、下一状态的期望 | 理论上的统一目标              |
| DP                | $\sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')]$ | 不缺模型           | 直接用 $R,P$ 计算期望              | $\text{target}_{\mathrm{DP}}$ |
| MC                | $\mathbb{E}_\pi[G_t\mid S_t=s]$                                  | 不知道 $R,P$       | 用一条真实轨迹的完整回报当样本     | $G_t$                         |
| TD                | $G_t=r_{t+1}+\gamma G_{t+1}$                                     | 不等完整 $G_{t+1}$ | 用旧表 $V_k(S_{t+1})$ 估计后续未来 | $r_{t+1}+\gamma V_k(S_{t+1})$ |

这条链可以写成更紧凑的形式：

$$
\underbrace{\mathbb{E}_\pi[r + \gamma V(s') \mid s]}_{\text{贝尔曼期望}}
\xrightarrow{\text{模型已知}}
\underbrace{\text{target}_{\mathrm{DP}}}_{\text{展开计算}}
\xrightarrow{\text{模型未知}}
\underbrace{G_t}_{\text{MC：完整采样}}
\xrightarrow{\text{不等完整未来}}
\underbrace{r + \gamma V_k(s')}_{\text{TD：一步自举}}
$$

箭头表示信息条件的弱化，而非算法的执行顺序。模型已知时用 DP 精确计算；模型未知时用 MC 采样估计完整回报；若不愿等待完整回报，则用 TD 将已发生的一步与现有估计结合。

在这个固定策略下，三种方法会收敛到同一个状态价值：

$$
V(S)=-3.375,\qquad V(M)=-1.875,\qquad V(G)=0.
$$

三种方法收敛到同一价值，但更新路径不同。DP 在模型已知时直接迭代求解，不依赖环境交互；MC 依赖完整轨迹的采样回报；TD 在每一步利用即时奖励和下一状态估计进行更新。

下面用程序验证。走廊环境与策略同上：$S$ 和 $M$ 均以 0.8 概率右走、0.2 概率左走。DP 迭代 1000 轮；MC 和 TD 各采样 100 万条 episode，取 5 个随机种子重复。三种方法估计同一 $V^\pi$。

```python
import random

STATES = ["S", "M", "G"]
GAMMA = 1.0


def step(state, action):
    # 环境本身是确定的：给定状态和动作，下一状态、奖励都固定。
    # 随机性只来自后面的策略 sample_action()。
    if state == "S":
        return ("M", -1) if action == "right" else ("S", -2)
    if state == "M":
        return ("G", -1) if action == "right" else ("S", -2)
    return "G", 0


def sample_action():
    # 固定策略 pi：80% 向右，20% 向左。
    return "right" if random.random() < 0.8 else "left"


def dp_policy_evaluation(n_iter=1_000):
    V = {s: 0.0 for s in STATES}
    for _ in range(n_iter):
        # DP 知道模型，因此可以直接枚举两个动作分支。
        # 这里用 old 做同步更新：本轮读旧表，写出新表。
        old = V.copy()
        V["S"] = 0.8 * (-1 + GAMMA * old["M"]) + 0.2 * (-2 + GAMMA * old["S"])
        V["M"] = 0.8 * (-1 + GAMMA * old["G"]) + 0.2 * (-2 + GAMMA * old["S"])
        V["G"] = 0.0
    return V


def generate_episode():
    # MC 和 TD 都不知道模型，只能让智能体真的走一局。
    episode = []
    state = "S"
    while state != "G":
        action = sample_action()
        next_state, reward = step(state, action)
        episode.append((state, reward, next_state))
        state = next_state
    return episode


def mc_every_visit(n_episodes=1_000_000, seed=0):
    random.seed(seed)
    V = {s: 0.0 for s in STATES}
    N = {s: 0 for s in STATES}
    for _ in range(n_episodes):
        episode = generate_episode()
        G = 0.0
        # MC 等整局结束后，从后往前累加完整回报 G_t。
        for state, reward, _ in reversed(episode):
            G = reward + GAMMA * G
            N[state] += 1
            # 每次访问都更新；1/N 是样本平均的增量写法。
            V[state] += (G - V[state]) / N[state]
    return V


def td_zero(n_episodes=1_000_000, seed=0):
    random.seed(seed)
    V = {s: 0.0 for s in STATES}
    N = {s: 0 for s in STATES}
    for _ in range(n_episodes):
        state = "S"
        while state != "G":
            action = sample_action()
            next_state, reward = step(state, action)
            N[state] += 1
            alpha = 1.0 / N[state]
            # TD 不等整局结束：一步奖励 + 下一状态当前估计。
            target = reward + GAMMA * V[next_state]
            V[state] += alpha * (target - V[state])
            state = next_state
    return V


def show(name, values):
    print(f"{name}: S={values['S']:.6f}, M={values['M']:.6f}, G={values['G']:.6f}")


def summarize(name, runs):
    mean_s = sum(v["S"] for v in runs) / len(runs)
    mean_m = sum(v["M"] for v in runs) / len(runs)
    min_s, max_s = min(v["S"] for v in runs), max(v["S"] for v in runs)
    min_m, max_m = min(v["M"] for v in runs), max(v["M"] for v in runs)
    print(
        f"{name}: mean S={mean_s:.6f} [{min_s:.6f}, {max_s:.6f}], "
        f"mean M={mean_m:.6f} [{min_m:.6f}, {max_m:.6f}]"
    )


print("single run")
show("DP", dp_policy_evaluation())
show("MC", mc_every_visit(seed=0))
show("TD", td_zero(seed=0))

print("\n5-run summary")
seeds = range(5)
summarize("MC", [mc_every_visit(seed=s) for s in seeds])
summarize("TD", [td_zero(seed=s) for s in seeds])
```

单次运行结果如下。DP 是确定性计算；MC 和 TD 的采样结果已接近 DP：

```text
single run
DP: S=-3.375000, M=-1.875000, G=0.000000
MC: S=-3.373813, M=-1.874359, G=0.000000
TD: S=-3.374871, M=-1.874966, G=0.000000
```

取 5 个随机种子重复运行，观察均值和波动范围：

```text
5-run summary
MC: mean S=-3.374261 [-3.376874, -3.372061], mean M=-1.874122 [-1.874833, -1.872401]
TD: mean S=-3.375231 [-3.380956, -3.366307], mean M=-1.874380 [-1.876858, -1.870551]
```

DP 的输出是模型迭代后的确定结果；MC 和 TD 的输出来自采样，因此存在随机波动。采样次数足够多时，三种方法收敛到同一组策略价值。

这一区分构成了后续算法设计的基础。DQN 和 Actor-Critic 采用 TD 类目标，因为它们面对的是无模型的大规模环境，无法等待完整回报。REINFORCE 则使用完整轨迹回报更新策略，与 MC 更为接近。DP、MC、TD 的关系本质上是”模型、采样、偏差与方差”之间的权衡。

## 小结

本节讨论了三种价值估计方法。

1. 状态价值 $V^\pi(s)$ 是从状态 $s$ 出发并继续按照策略 $\pi$ 行动时的期望折扣回报。
2. DP 假设环境模型已知，直接使用贝尔曼期望方程对所有动作和下一状态求平均。它适合作为理论基准，但完整模型在现实任务中通常不可得。
3. MC 不需要模型，而是使用完整轨迹的回报 $G_t$ 来更新价值。它的目标无偏，但必须等待 episode 结束，并且方差较大。
4. TD 也不需要模型，并且走一步就能更新。它使用 $r+\gamma V(s')$ 作为目标，学习更及时、方差更低，但因为目标中含有估计值，所以会引入偏差。
5. DP、MC、TD 的共同核心是贝尔曼思想：当前价值可以由即时奖励和未来价值来刻画。它们的差别在于未来价值是由模型精确计算、由完整经历给出，还是由下一状态估计近似。

下一节将从状态价值 $V(s)$ 转向动作价值 $Q(s,a)$。$V(s)$ 只能说明一个状态整体有多好，而 $Q(s,a)$ 可以直接比较“在状态 $s$ 下先做动作 $a$”的好坏，从而让智能体真正做出动作选择。

下一节：[4.2 策略采样与数据来源](./algorithm-taxonomy)

## 参考文献

[^1]: Bellman, R. (1957). _Dynamic Programming_. Princeton University Press.

[^2]: Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. _Journal of the American Statistical Association_, 44(247), 335-341.

[^3]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[^4]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press, Chapter 6, Section 6.1 and Chapter 7, Section 7.1. Section 6.1 explicitly compares the Monte Carlo target, TD target, and DP target; Section 7.1 explicitly calls the return used in a backup the target of the backup.
