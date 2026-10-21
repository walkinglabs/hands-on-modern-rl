# 3.3 动手：价值迭代与 Q-Learning

<ChapterNotebookLink chapter="mdp" />

> **本节目标**：在同一个 4×4 GridWorld 中运行价值迭代和 Q-Learning，观察目标奖励怎样传到起点，并比较两种算法获取信息的方式。

> **学习路径**：[3.1 状态价值与贝尔曼期望方程](./value-bellman) → [3.2 动作价值与贝尔曼最优方程](./value-q) → **3.3 价值迭代与 Q-Learning**

> **本节代码与资源**：[实验脚本](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter03_mdp/gridworld_q_learning.py) · [GridWorld 环境图](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter03_mdp/images/gridworld-environment.svg) · [价值迭代图](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter03_mdp/images/gridworld-value-iteration.svg) · [Q-Learning 曲线](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter03_mdp/images/gridworld-q-learning.svg)

## 3.3.1 运行 GridWorld 实验

[3.1](./value-bellman) 和 [3.2](./value-q) 已经介绍了状态价值、动作价值和贝尔曼方程。现在先不继续增加公式，而是把贝尔曼最优方程放进一个完整的小实验里，观察一张价值表怎样从全零开始，逐轮算到稳定。

任务只有 16 个状态。智能体从左上角的 $S=(0,0)$ 出发，要到达右下角的 $G=(3,3)$；$X=(1,1)$ 是陷阱。每一步可以向上、下、左、右移动，撞墙时位置不变。

<img src="./images/gridworld-environment.svg" alt="4×4 GridWorld：起点 S 位于左上角，陷阱 X 位于坐标一一，目标 G 位于右下角" />

<div class="figure-caption">图 3-3：价值迭代与 Q-Learning 使用同一个 4×4 GridWorld。</div>

这个实验只依赖 Python 标准库，在普通电脑上几秒内即可完成。在仓库根目录运行：

```bash
python3 code/chapter03_mdp/gridworld_q_learning.py \
  --output-dir output/value-experiment
```

脚本先让价值迭代读取完整的网格规则，反复更新所有状态；随后让 Q-Learning 从起点出发，只根据实际走过的经验更新 Q 表。运行结束后，重点看三个结果：价值迭代用了多少轮收敛，目标奖励经过几轮到达起点，以及关闭探索后 Q-Learning 是否找到六步最短路径。

## 3.3.2 GridWorld 的奖励

进入目标得到 $+1$，进入陷阱得到 $-1$，这两种情况都会结束当前回合。其余每一步得到 $-0.01$。这项很小的负奖励会催促智能体尽快到达目标：同样能够成功时，走 6 步会比走 8 步得到更高的回报。折扣因子取 $\gamma=0.95$。

先算最短距离。起点与目标的行坐标相差 3，列坐标也相差 3，因此至少需要

$$
|3-0|+|3-0|=6
$$

步。

陷阱会挡住部分走法，却没有堵住所有六步路径。沿上边界向右走三步，再向下走三步，就能避开陷阱到达目标。因此，我们可以先记住一个检查标准：如果算法学到的路线超过 6 步，它就还没有找到最短路径。

### 终止奖励在进入时计算

正式更新价值表之前，先看最靠近目标的格子 $(3,2)$。智能体在这里向右走一步，便会进入目标并得到 $+1$：

$$
(3,2)\xrightarrow{\;\rightarrow,\,+1\;}G\quad\text{终止}.
$$

回合已经结束，目标后面不会再产生新的动作和奖励。因此，从 $(3,2)$ 向右走的回报就是 1：

$$
Q((3,2),\rightarrow)=1.
$$

把它写成贝尔曼更新的形式：

$$
Q((3,2),\rightarrow)=1+\gamma V(G)=1,
$$

其中 $V(G)=0$。这里的 0 表示进入目标以后没有后续回报；目标奖励已经在进入 $G$ 的那一步计入。

陷阱采用相同的处理方式。进入 $X$ 的那一步得到 $-1$，随后回合结束，所以 $V(X)=0$。

代码按照这条时间顺序返回奖励和终止标记：

```python
def transition(state, action):
    if state in TERMINALS:
        return state, 0.0, True

    next_state = move_or_stay(state, action)
    if next_state == GOAL:
        return next_state, 1.0, True
    if next_state == TRAP:
        return next_state, -1.0, True
    return next_state, -0.01, False
```

价值迭代和 Q-Learning 都调用这个转移函数，因此它们面对的是同一个任务。

## 3.3.3 价值迭代：读取环境规则

价值迭代适用于环境规则已知的情况。这里的“规则已知”是指：给定任意格子和任意动作，我们都能查出下一步到达哪个格子、得到多少奖励、回合是否结束。

算法先把所有格子的价值设为 0：

$$
V_0(s)=0,\qquad \forall s.
$$

### 一轮更新能传多远

接下来计算一张新的价值表。对每个非终止格子，分别计算上、下、左、右四个动作的价值，再保留其中最大的一个：

$$
V_{k+1}(s)=\max_a\left[r+\gamma V_k(s')\right].
$$

式中的 $V_k$ 是更新前的旧表，$V_{k+1}$ 是这一轮得到的新表。这种“整张新表只读取上一张旧表”的做法称为**同步更新**。

先算第一轮。对 $(3,2)$ 来说，向右一步进入目标，得到的价值为

$$
V_1(3,2)=\max_a\left[r+0.95V_0(s')\right]=1.
$$

目标上方的 $(2,3)$ 也可以一步进入目标，所以它的价值也是 1。

现在看离目标两步的 $(3,1)$。虽然它的右边就是刚才算过的 $(3,2)$，但第一轮只能读取旧表，而旧表中的 $V_0(3,2)$ 仍然是 0：

$$
V_1(3,1)=-0.01+0.95V_0(3,2)=-0.01.
$$

到了第二轮，$(3,1)$ 才能读到新表中的 $V_1(3,2)=1$：

$$
V_2(3,1)=-0.01+0.95V_1(3,2)=0.94.
$$

这两个格子展示了价值传播的过程：第一轮先更新距离目标一步的格子，第二轮再影响距离目标两步的格子。一次同步更新只能把目标奖励向外传一层。

程序实现与上面的手算相同。`values` 保存 $V_k$，`updated` 保存正在计算的 $V_{k+1}$：

```python
values = {state: 0.0 for state in all_states()}

for sweep in range(1000):
    updated = values.copy()
    for state in all_states():
        if state not in TERMINALS:
            updated[state] = max(
                reward if done else reward + GAMMA * values[next_state]
                for action in range(4)
                for next_state, reward, done in [transition(state, action)]
            )
    values = updated
```

下图列出第 0、1、3、6 轮的价值表。读图时先看蓝色较深的格子：它们从目标附近开始，随着更新轮数增加逐渐向左上角扩展。$V_1$ 中只有目标相邻的两个格子得到正值；到 $V_6$ 时，起点才得到沿六步路径传回的目标奖励。

<img src="./images/gridworld-value-iteration.svg" alt="GridWorld 价值迭代的第 0、1、3、6 轮价值表，以及收敛价值与最优策略" />

<div class="figure-caption">图 3-4：同步价值迭代。每轮使用上一轮的完整价值表。</div>

### 从价值表读出策略

第 6 轮后，价值表已经不再变化。程序继续计算一轮，发现每个格子的数值都没有改变，于是在第 7 轮停止。我们把这种状态称为**收敛**。

最终结果为：

| 行 / 列 |         0 |             1 |     2 |             3 |
| ------- | --------: | ------------: | ----: | ------------: |
| 0       | **0.729** |         0.777 | 0.829 |         0.883 |
| 1       |     0.777 | 0.000（陷阱） | 0.883 |         0.940 |
| 2       |     0.829 |         0.883 | 0.940 |         1.000 |
| 3       |     0.883 |         0.940 | 1.000 | 0.000（目标） |

现在检查起点的数值。沿最短路径移动时，前 5 步各得到 $-0.01$，最后一步进入目标并得到 $+1$。把这 6 笔奖励按先后顺序折扣：

$$
\begin{aligned}
V^*(0,0)
&= -0.01-0.95\times0.01-\cdots-0.95^4\times0.01+0.95^5\times1 \\
&\approx 0.728537.
\end{aligned}
$$

计算结果约为 0.729，与价值表左上角的数值相符。这说明程序得到的起点价值可以由具体的最短路径直接验证。

图中的多个箭头表示这个格子有不止一个最优动作。例如在起点先向右或先向下，都能避开陷阱，并在 6 步内到达目标。

## 3.3.4 Q-Learning：从交互中学习

价值迭代每次更新一个格子时，可以直接查询四个动作的结果。Q-Learning 没有这项条件。它不知道一个动作会把自己带到哪里，只能从起点开始，选择一个动作，观察奖励 $r$ 和下一个状态 $s'$。

走完一步，我们就得到一条经验 $(s,a,r,s')$：从状态 $s$ 执行动作 $a$，获得奖励 $r$，然后进入状态 $s'$。Q-Learning 使用这条经验更新一个 Q 值：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right].
$$

方括号中的第一部分

$$
r+\gamma\max_{a'}Q(s',a')
$$

称为 **TD 目标**。它由“这一步已经得到的奖励”和“下一状态目前估计的最佳价值”组成。学习率 $\alpha$ 决定本次更新向 TD 目标靠近多少。

如果 $s'$ 是目标或陷阱，回合已经结束，下一状态价值取 0：

```python
next_state, reward, done = transition(state, action)
next_best = 0.0 if done else max(Q[next_state])
td_target = reward + gamma * next_best
Q[state][action] += alpha * (td_target - Q[state][action])
```

现在可以看出两种算法的区别。价值迭代的一轮扫描会访问所有非终止状态，并比较每个状态的四个动作；Q-Learning 的一次更新只使用刚刚走过的一步。它需要运行许多回合，才能让不同的状态—动作对都得到更新。

### 探索率怎样影响训练回报

训练开始时，Q 表全部为 0，智能体还不知道应该往哪里走。如果它每次都选择当前 Q 值最大的动作，许多没有尝试过的路线将一直得不到更新。这里使用 $\varepsilon$-贪心策略：以 $\varepsilon$ 的概率随机选择动作，以 $1-\varepsilon$ 的概率选择当前 Q 值最大的动作。

实验取学习率 $\alpha=0.15$，训练 500 回合。为了避免某一次随机运行碰巧特别好或特别差，每项设置使用 30 个不同的随机种子独立运行。

绘制曲线时，先计算同一回合在 30 次运行中的平均奖励，再取 20 回合滑动平均。这样可以保留整体变化趋势，同时减小单次随机运行造成的波动。

比较三种 $\varepsilon$-贪心设置：

- $\varepsilon$ 从 $1.00$ 线性降到 $0.05$；
- 固定 $\varepsilon=0.05$；
- 固定 $\varepsilon=0.30$。

<img src="./images/gridworld-q-learning.svg" alt="三种探索率设置下的 Q-Learning 多随机种子训练曲线" />

<div class="figure-caption">图 3-5：不同探索率下的训练回报。每条曲线汇总 30 个随机种子。</div>

| 探索率设置                        | 末 100 回合平均奖励 | 关闭探索后的成功率 | 平均路径长度 |
| --------------------------------- | ------------------: | -----------------: | -----------: |
| $\varepsilon:1.00\rightarrow0.05$ |               0.803 |               100% |       6.0 步 |
| 固定 $\varepsilon=0.05$           |               0.900 |               100% |       6.0 步 |
| 固定 $\varepsilon=0.30$           |               0.563 |               100% |       6.0 步 |

固定 $\varepsilon=0.30$ 时，每一步仍有 30% 的概率随机选择动作。即使 Q 表已经学好，智能体在训练过程中仍会绕路或进入陷阱，因此曲线停在较低的位置。这不一定表示 Q 表没有学会最短路径，还可能只是探索动作降低了当局得分。

为了单独检查学到的策略，我们在训练结束后令 $\varepsilon=0$。此时智能体不再随机探索，始终选择 Q 值最大的动作。三种设置都到达目标，路径长度都是 6，单回合未折扣奖励为

$$
5\times(-0.01)+1=0.95.
$$

训练曲线记录智能体一边探索、一边行动时得到的奖励。关闭探索后的测试则始终选择 Q 值最大的动作，衡量 Q 表最终学到的策略。

固定的 $\varepsilon=0.30$ 会降低训练回报。在这个小环境和当前训练预算下，三种设置最终仍然学到了相同长度的最短路径。

## 本节小结

- 价值迭代使用完整的环境模型，对所有状态进行同步更新。在本节的 GridWorld 中，目标奖励经过 6 轮更新传递到起点。
- 终止状态之后没有后续回报。进入目标或陷阱时获得终止奖励，并令终止状态的价值为 0。
- Q-Learning 不需要环境模型。每次交互产生一个转移样本，并更新一个状态—动作对的 Q 值。
- 训练时的探索动作会影响回合奖励。将 $\varepsilon$ 设为 0 后进行评估，可以得到 Q 表所表示的贪心策略。

下一节 [动态规划、蒙特卡洛与时序差分](./dp-mc-td) 将从这一区别出发，比较完整扫描、整局采样和单步自举三种更新方式。
