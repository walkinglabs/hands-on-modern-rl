# 6.5 动手：价值基线控制 CartPole

<ChapterNotebookLink chapter="pg" />

> **本节目标**：用 `CartPole-v1` 对比原始 REINFORCE 和带价值基线（Value Baseline, VB）的 REINFORCE，观察 $V(s)$ 如何让策略梯度训练更快、更稳。

> **学习路径**：[6.3 摇骰子赌博机](./dice-game) → [6.4 REINFORCE 控制 CartPole](./cartpole) → **6.5 价值基线控制 CartPole**

> **本节代码与资源**：[reinforce_with_baseline.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_with_baseline.py) · [render_cartpole_baseline.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/render_cartpole_baseline.py) · [reinforce_cartpole.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_cartpole.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/requirements.txt)

前两节已经用 REINFORCE 训练过赌博机和 CartPole。完整回报可以指导策略更新，同一策略在不同回合中却可能得到差异很大的回报，使梯度受到偶然轨迹影响。

本节保持 CartPole 环境、策略网络和训练预算不变，只增加一个估计 $V(s)$ 的价值网络。这样可以把性能差异归因于基线，并通过奖励曲线、梯度方差和回放共同判断它是否有效。

## 6.5.1 Value Baseline 从哪里来

先把名字说清楚。
**baseline** 不是一个单独的强化学习算法，
而是策略梯度估计中的方差缩减技巧。
Williams 在 1992 年提出 REINFORCE 时，
更新式中已经允许从奖励信号里减去一个不依赖当前动作的基线项。[^williams1992]
这样做的关键性质是：
只要基线不随当前动作改变，
它不会改变策略梯度的期望方向，
但可能显著降低采样估计的方差。

后来，Sutton、McAllester、Singh 和 Mansour 在 policy gradient theorem 中把策略梯度写成更清楚的形式：
策略更新可以看作
$\nabla_\theta \log \pi_\theta(a \mid s)$
乘上某种“动作好坏”的估计。[^sutton1999]
这个估计可以是完整回报 $G_t$，
也可以是动作价值 $Q^\pi(s,a)$，
还可以减去一个状态相关的 baseline。
当 baseline 取为状态价值函数 $V^\pi(s)$ 时，
就得到优势形式

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s).
$$

本节说的 **Value Baseline**，
就是用一个价值网络近似 $V(s)$，
再用

$$
A_t = G_t - V(s_t)
$$

来替代原始的 $G_t$ 更新策略。
因此，更准确地说，
本节实验不是“随便加一个 baseline”，
而是“用 learned value function 作为 baseline”。
它已经很接近下一章的 Critic，
但仍然保留 REINFORCE 的特点：
必须等一个完整 episode 结束后，
用 Monte Carlo 回报 $G_t$ 来更新。

还有其他 baseline 形式。
最简单的是常数基线，
例如减去所有回合的平均回报；
在无状态赌博机中，这已经有用。
更进一步，可以使用状态相关基线 $b(s)$，
其中最常见的选择就是 $V(s)$。
Greensmith、Bartlett 和 Baxter 从方差缩减角度系统分析了 baseline 和 actor-critic，
并指出常用的平均回报基线并不总是最优；
理论上还可以推导更接近最小方差的 optimal baseline。[^greensmith2004]
在现代算法中，
PPO、A2C、A3C 等通常不直接使用原始 $G_t$，
而是使用 advantage；
GAE 则是在 bias 和 variance 之间进一步折中。

所以，本节的 Value Baseline 是一条中间路线：
它比常数基线更懂状态，
但还没有进入完整 Actor-Critic 的在线 TD 更新。

## 6.5.2 为什么 CartPole 更适合看 Value Baseline

CartPole 的状态有 4 个数字：
小车位置、小车速度、杆子角度和杆子角速度。
动作只有两个：
向左推或向右推。
每坚持一个时间步，环境给 `+1` 奖励；
如果杆子倒得太厉害，或者小车离开边界，episode 结束。
`CartPole-v1` 的最高回合长度是 `500`，
所以回合回报也可以直接理解为“杆子立住了多少步”。

这个任务刚好暴露 REINFORCE 的高方差问题。
在训练早期，
策略可能只是偶然多坚持了几十步。
原始 REINFORCE 会把这一整段轨迹中的动作都当作“好动作”来强化，
即使其中有些动作只是碰巧没有立刻造成失败。
下一回合若开局扰动不同，
同一个策略又可能很快倒下。
于是训练曲线会出现明显抖动。

Value Baseline 要解决的不是“让公式方向改变”，
而是让同一个回报放回具体状态中解释。
原始 REINFORCE 直接用从时间步 $t$ 开始的折扣累计回报 $G_t$
作为这一动作的学习信号；
加入价值基线后，
我们先用价值网络估计当前状态的平均回报 $V(s_t)$，
再计算优势

$$
A_t = G_t - V(s_t).
$$

也就是说，
策略更新看的是“实际回报”和“这个状态本来预期能拿到的回报”之间的差值。
减掉 $V(s_t)$ 之后，
策略网络不再只问“这一步之后拿了多少分”，
而是问“这一步之后比原本预期好了多少”。
如果实际结果比预期好，
就强化这个动作；
如果实际结果比预期差，
就降低这个动作的概率。

这就是价值基线在 CartPole 中最直观的作用：
不是让小车多一个动作，
而是让它少被偶然的好坏回合误导。

## 6.5.3 运行对比实验

先安装依赖：

```bash
pip install -r code/chapter05_policy_gradient/requirements.txt
```

然后运行对比实验：

```bash
python code/chapter05_policy_gradient/reinforce_with_baseline.py
```

这个脚本会训练两个策略：

| 实验                       | 更新信号       | 额外网络      | 直观含义                             |
| -------------------------- | -------------- | ------------- | ------------------------------------ |
| Vanilla REINFORCE          | `G_t`          | 无            | 只看这一回合之后实际拿了多少分       |
| REINFORCE + Value Baseline | `G_t - V(s_t)` | Value Network | 看实际结果比当前状态的平均预期好多少 |

两个版本都使用同一个 CartPole 环境和同一种策略网络。
区别只在更新权重：
原始版本用完整回报 $G_t$；
Value Baseline 版本先训练一个价值网络估计 $V(s_t)$，
再用优势 $G_t - V(s_t)$ 更新策略。

脚本结束后会生成两张图：

| 输出文件                                            | 说明                       |
| --------------------------------------------------- | -------------------------- |
| `output/reinforce_baseline_reward_comparison.png`   | 两种方法的回合奖励曲线     |
| `output/reinforce_baseline_variance_comparison.png` | 两种方法的梯度估计方差曲线 |

本节讲义中的图像就是由这个脚本导出的。

如果希望同时导出回放 GIF，
可以运行：

```bash
python code/chapter05_policy_gradient/render_cartpole_baseline.py \
  --episodes 500 \
  --seed 0
```

这个脚本会重新训练两个策略，
然后用确定性动作渲染各自的 CartPole 回放，
并把 GIF 写入 `docs/chapter08_policy_gradient/images/`。

## 6.5.4 看奖励曲线

先看最直接的结果：小车能立住多久。

![CartPole 上原始 REINFORCE 与 REINFORCE + Value Baseline 的奖励曲线对比。Value Baseline 版本更早接近 500 步上限，原始版本学习更慢且波动更明显。](./images/reinforce-baseline-cartpole-reward.png)

图中浅色线是单个 episode 的原始回报，
深色线是滑动平均。
单个 episode 的回报会剧烈跳动，
这是策略梯度任务中很正常的现象：
同一个策略在不同初始状态下可能撑很久，
也可能很快失败。
因此，更应该看滑动平均趋势。

这次运行中，
原始 REINFORCE 的最后 50 回合平均回报约为 `95.1`。
它确实在学习，
但学习过程比较慢，
中途还有明显回落。
加入价值基线后，
最后 50 回合平均回报约为 `493.0`，
已经非常接近 CartPole 的 `500` 步上限。

这个差异说明：
价值基线不是一个装饰性的数学项。
在同一个任务中，
它能让策略更快进入“基本能立住杆子”的区域，
并减少训练后期突然退步的概率。

## 6.5.5 看回放

曲线说明平均趋势，
回放则说明策略到底在做什么。
下面两段 GIF 使用同一个渲染脚本生成，
展示训练 500 回合后两种策略的确定性表现。

**Vanilla REINFORCE：能坚持一段时间，但仍然容易越调越偏。**
这次渲染回报为 `166`。
它已经不再是随机策略，
但杆子偏离后修正不够稳定，
小车会逐渐把局面推到不可恢复的位置。

![Vanilla REINFORCE 训练后的 CartPole 回放：策略已经学到一部分平衡动作，但仍会逐渐失稳。](./images/cartpole-vanilla-reinforce.gif)

**REINFORCE + Value Baseline：更稳定地把杆子拉回中心附近。**
这次渲染回报为 `355`。
它不是每次都达到 500 步上限，
但动作修正明显更连贯，
杆子偏离时更容易被拉回来。

![REINFORCE + Value Baseline 训练后的 CartPole 回放：策略能更稳定地修正杆子角度。](./images/cartpole-reinforce-baseline.gif)

这两段回放的意义不是证明某一个 seed 永远如此，
而是帮助理解奖励曲线中的差别。
原始 REINFORCE 的更新信号更吵，
策略可能学到一些有用动作，
但对不同状态的判断不够稳定。
Value Baseline 版本通过 $G_t - V(s_t)$ 过滤掉一部分“这个状态本来就容易/本来就危险”的影响，
更容易把学习集中在动作本身带来的增益上。

## 6.5.6 看方差曲线

奖励曲线回答“策略表现是否变好”。
方差曲线回答另一个问题：
下面用梯度方差解释价值基线带来的变化。

![CartPole 上原始 REINFORCE 与 REINFORCE + Value Baseline 的梯度估计方差对比。Value Baseline 把回报变成优势后，梯度信号更集中。](./images/reinforce-baseline-cartpole-variance.png)

这张图画的是滑动窗口中的梯度估计方差。
数值越大，
说明不同 episode 给出的更新方向差异越大；
数值越小，
说明策略每次更新更一致。

在这次运行中，
原始 REINFORCE 的梯度估计方差约为 `100.41`，
Value Baseline 版本约为 `38.27`。
也就是说，
Value Baseline 把方差降到了原来的约 `38.1%`。
这和奖励曲线中的现象对应起来：
更新信号更稳，
策略就更容易持续朝着“让杆子站住”的方向移动。

## 6.5.7 代码里到底改了什么

原始 REINFORCE 的核心更新是：

```python
returns_t = torch.FloatTensor(returns)
log_probs = torch.log(action_probs + 1e-8)
loss = -(log_probs * returns_t).mean()
```

这里的 `returns_t` 就是 $G_t$。
如果某一回合刚好撑了很久，
这段轨迹里的所有动作都会被较大权重强化。
这并不总是错，
但它会把很多“碰巧没有出事”的动作也一起强化。

加入价值基线后，
脚本多了一个价值网络：

```python
values = value_net(states_t)
value_loss = nn.MSELoss()(values, returns_t)
```

价值网络学习的是：
从状态 $s_t$ 出发，
通常能拿多少分。
然后策略网络不再直接使用 $G_t$，
而是使用优势：

```python
with torch.no_grad():
    values_pred = value_net(states_t)

advantages = returns_t - values_pred
policy_loss = -(log_probs * advantages).mean()
```

这几行代码就是价值基线的核心。
如果 `advantages` 为正，
说明这一步之后比预期更好，
对应动作应该更常出现；
如果 `advantages` 为负，
说明这一步之后比预期更差，
对应动作应该减少。

注意，价值基线不依赖当前动作本身，
因此不会改变策略梯度的期望方向。
它改变的是估计的噪声大小。
这也是为什么它叫“降方差”，
而不是“改目标”。

## 6.5.8 回到画面中理解

想象小车已经把杆子扶到接近竖直的位置。
如果它本来就能从这个状态继续坚持很久，
那么再多坚持几步并不一定说明刚才那个动作特别神奇；
这只是一个好状态本来就应该有的结果。
此时 $V(s_t)$ 会比较高，
减掉它以后，
优势不会被夸大。

反过来，
如果杆子已经明显倾斜，
小车却通过一个正确动作把局面救回来，
实际回报可能明显超过价值网络的预期。
这时 $G_t - V(s_t)$ 为正，
策略会更明确地强化这个补救动作。

这就是价值基线比“只看总分”更细的地方：
它让策略知道，
同样是拿到 100 分，
在危险状态下拿到 100 分，
和在容易状态下拿到 100 分，
含义并不一样。

## 6.5.9 常见误读

**误读一：价值基线会让奖励变大。**
价值基线不改环境奖励。
CartPole 每一步仍然只给 `+1`。
它改变的是训练时如何解释这些奖励。

**误读二：基线越大越好。**
如果基线估计很差，
优势也会很吵。
这里使用价值网络学习 $V(s)$，
是因为状态不同，合理的平均回报也不同。
一个固定常数基线只能处理很简单的无状态问题。

**误读三：有价值基线就是 Actor-Critic。**
本节仍然是 REINFORCE with Value Baseline。
它要等一个完整 episode 结束，
用 Monte Carlo 回报 $G_t$ 更新。
下一章的 Actor-Critic 会进一步用 TD 目标替代完整回报，
做到每一步都可以更新。

## 本节小结

- CartPole 比赌博机更适合展示价值基线的作用，因为它有状态、有失败形态，也能通过回合长度直观看出策略好坏。
- 原始 REINFORCE 使用 $G_t$ 更新策略，容易被单个 episode 的运气误导。
- 价值基线学习 $V(s_t)$，把更新信号从 $G_t$ 改成 $G_t - V(s_t)$。
- 价值基线不改变策略梯度的期望方向，但能显著降低方差，使训练更稳定。
- 本节中的价值基线已经出现了 Critic 的影子；下一章会把它发展成真正的 Actor-Critic。

## 练习

1. 把 `num_episodes` 改成 `200`，观察两种方法谁更早学到可用策略。
2. 把学习率从 `1e-3` 改成 `5e-4` 或 `2e-3`，比较价值基线是否仍然更稳。
3. 在脚本中打印 `advantages.mean()` 和 `advantages.std()`，观察优势信号是否围绕 0 波动。
4. 把 Value Network 的隐藏层从 `128` 改成 `32`，观察基线估计变弱后训练曲线是否更抖。

## 参考文献

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Gymnasium. CartPole-v1 documentation. <https://gymnasium.farama.org/environments/classic_control/cart_pole/>

[^williams1992]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8, 229-256. DOI: <https://doi.org/10.1007/BF00992696>.

[^sutton1999]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^greensmith2004]: Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. _Journal of Machine Learning Research_, 5, 1471-1530. <https://jmlr.org/papers/v5/greensmith04a.html>.
