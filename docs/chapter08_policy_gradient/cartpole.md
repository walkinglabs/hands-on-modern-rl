# 6.4 动手：REINFORCE 控制 CartPole

> **本节目标**：用 REINFORCE 训练 `CartPole-v1`，观察策略梯度在高方差环境中的训练过程，理解"好结果强化动作概率"这件事在真实控制任务中的表现。

> **学习路径**：[6.1 策略梯度与 REINFORCE](./reinforce) → [6.3 摇骰子赌博机](./dice-game) → **6.4 REINFORCE 控制 CartPole**

> **本节代码与资源**：[reinforce_cartpole.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_cartpole.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/requirements.txt)

上一节推导了策略梯度定理和 REINFORCE 算法。赌博机例子展示了最简情况——无状态、单步、只有两个动作。现在换一个更有代表性的任务：`CartPole-v1`。小车可以向左或向右推，目标是让杆子尽量久地保持竖直。每坚持一个时间步，环境给 `+1` 奖励；杆子倒得太厉害或者小车离开边界，episode 结束。

与赌博机不同，CartPole 有状态（4 维向量：小车位置、速度、杆子角度、角速度），有多步决策，有明确的失败方式。它仍然只是离散动作（左/右），但已经足够暴露 REINFORCE 的高方差问题。

## 6.4.1 运行训练

先安装依赖：

```bash
pip install -r code/chapter05_policy_gradient/requirements.txt
```

然后运行训练：

```bash
python code/chapter05_policy_gradient/reinforce_cartpole.py
```

这个脚本会训练一个 REINFORCE 策略，500 个 episode。核心代码只有三步：

```python
# 用当前策略跑完一个完整 episode
states, actions, rewards, episode_reward = collect_episode(policy, env)

# 从后向前计算每一步的折扣累计回报 G_t
returns = compute_returns(rewards, gamma=0.99)

# 策略梯度更新
loss = -(log_probs * returns_tensor).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这就是 REINFORCE 的全部。跑完一个 episode，计算每步的 $G_t$，用 `loss = -log_prob * G_t` 更新策略。

运行结束后，脚本会在 `output/` 下生成训练曲线。

## 6.4.2 看训练曲线

![REINFORCE 在 CartPole-v1 上的训练曲线：回合奖励随训练进度变化](./images/reinforce-cartpole-reward.png)

训练曲线通常呈现以下特征：

**初期（episode 0–50）**：策略接近随机，杆子很快倒下，回合奖励在 10–30 之间波动。此时策略网络几乎是均匀分布——"向左和向右都试试"。

**中期（episode 50–200）**：如果运气好，某个 episode 偶然坚持了较长时间，$G_t$ 普遍较大，策略会把那次轨迹中的动作都强化。回合奖励开始出现向上的趋势，但波动剧烈——好的 episode 之后可能跟着一连串差的。

**后期（episode 200–500）**：策略逐渐收敛到一个比较稳定的平衡策略，回合奖励可能达到 100–200。但曲线仍然有明显的回落——这是高方差的直接后果。

## 6.4.3 高方差的表现

CartPole 比赌博机更能暴露方差问题，因为每个 episode 有几十到几百步。策略在某一步做了一个好的动作，但后续步骤的表现完全取决于采样运气。$G_t$ 把整条轨迹的随机性都压进了一个数里——这个数既反映了当前动作的好坏，也反映了后续所有步骤的运气。

体现在训练曲线上：

- **突然的奖励尖峰**：某次恰好采样到了一条好轨迹，回合奖励冲到 200+，但下一回合可能又掉回 30。策略被这次好运气大幅推动，然后又被下一次坏运气拉回来。
- **学习不稳定**：同样的超参数，不同随机种子可能导致完全不同的训练结果。有时候 500 个 episode 就能学到不错的策略，有时候 1000 个也不够。
- **对学习率敏感**：学习率太大，策略在好动作和坏动作之间摇摆；学习率太小，策略几乎不动。合适的窗口很窄。

这些现象的本质是同一个问题：REINFORCE 用 $G_t$ 来判断"这个动作好不好"，但 $G_t$ 太不稳定了。一个真正好的动作，可能因为后续运气差而被惩罚；一个偶然出现在好轨迹中的平庸动作，可能因为后续运气好而被过度强化。

## 6.4.4 代码里的关键细节

**折扣累计回报的计算**——从后向前递推：

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G  # G_t = r_t + γ * G_{t+1}
        returns.insert(0, G)
    return returns
```

从最后一个时间步开始：$G_T = r_T$。往前一步：$G_{T-1} = r_{T-1} + \gamma G_T$。依此类推。这个递推保证了每一步的 $G_t$ 都包含从该步到结束的所有折扣奖励。

**按概率采样，不是取 argmax**——这是策略梯度与 DQN 的一个关键区别：

```python
probs = policy(state_tensor)
dist = torch.distributions.Categorical(probs)
action = dist.sample()  # 按概率随机选
```

DQN 选 `argmax Q`，策略是确定性的。REINFORCE 从概率分布中采样，探索是内建的——如果网络认为某个动作有 60% 的概率值得尝试，它就会以 60% 的概率去试。

**on-policy 的含义**——REINFORCE 必须用当前策略产生数据，用完即丢：

```python
# 每个 episode 都要重新收集数据
states, actions, rewards, episode_reward = collect_episode(policy, env)
```

DQN 的经验回放池可以反复使用旧数据。REINFORCE 的梯度估计中 $\mathbb{E}_{\pi_\theta}$ 要求必须用当前策略 $\pi_\theta$ 产生的数据。策略一更新，旧数据就失效了。这也是策略梯度数据效率低于 DQN 的原因。

## 6.4.5 回到方差问题

CartPole 实验说明 REINFORCE 能学，但学得不够稳。根源在于 $G_t$ 的方差太大。策略梯度定理有一个奇妙的性质：可以在梯度估计中减去一个不依赖于动作的基线 $b(s_t)$，把更新信号从 $G_t$ 改成 $G_t - b(s_t)$，既不改变梯度的期望方向，又能大幅降低方差。

## 本节小结

- REINFORCE 使用完整轨迹的折扣回报更新动作概率，不需要 Q 网络或经验回放。
- CartPole 的奖励曲线能够上升，但不同随机种子之间仍会有明显波动。
- 高方差来自用整段后续回报评价单个动作，下一节将用状态基线降低这种噪声。

这个基线背后的数学原理见 [6.2 策略梯度改进方法](./pg-improvements)；下一节进入[价值基线控制 CartPole](./baseline-experiment)实验。
