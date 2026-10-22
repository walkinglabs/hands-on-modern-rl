# %% [markdown]
# # 回报减去一个基线 / REINFORCE with a baseline
#
# 一次动作之后得了 100 分，还不足以判断它是否值得鼓励。
# 如果同样状态通常能得 120 分，这次表现就低于预期；如果通常只能得 60 分，这次就高于预期。
# 我们让价值网络学习这个状态预期，再比较原始回报与“回报减预期”带来的更新。
#
# CPU 默认每种方法 2 个种子、每个种子 192 回合；改成 960 回合和 5 个种子可延长实验。
# 对应源码：`code/chapter05_policy_gradient/reinforce_with_baseline.py`。

# %%
import importlib.util
import subprocess
import sys

packages = {"torch": "torch>=2.5,<3", "numpy": "numpy>=1.26,<3",
            "matplotlib": "matplotlib>=3.8,<4", "PIL": "pillow>=10,<13",
            "gymnasium": "gymnasium[classic-control]>=1.0,<2", "pygame": "pygame>=2.5,<3"}
missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "reinforce"
OUTPUT.mkdir(parents=True, exist_ok=True)
EPISODES, SEEDS, BATCH_EPISODES = 192, [0, 1], 8
print({"episodes": EPISODES, "seeds": SEEDS, "batch_episodes": BATCH_EPISODES})

# %% [markdown]
# ## 1. 基线不改变期望梯度 / Check the score identity
#
# 基线必须与“本次选中的动作”无关。对固定状态，策略对所有动作的分数梯度期望为 0，
# 因此减去一个状态基线不会改变期望梯度。基线的取值会影响梯度方差。
# 下面直接枚举一个两动作例子，比较梯度的均值与方差。

# %%
p = 0.3
probabilities = np.array([1 - p, p])
scores = np.array([-p, 1 - p])
action_returns = np.array([4.0, 6.0])
variance = {}
for baseline in [0.0, 5.0, 20.0]:
    gradients = scores * (action_returns - baseline)
    mean = np.sum(probabilities * gradients)
    var = np.sum(probabilities * (gradients - mean) ** 2)
    variance[baseline] = float(var)
    print({"baseline": baseline, "expected_gradient": float(mean), "variance": float(var)})
    assert abs(mean - 0.42) < 1e-10
assert variance[5.0] < variance[0.0] < variance[20.0]

# %% [markdown]
# ## 2. 用完整回合学习 / Monte Carlo updates
#
# 策略输出左右动作的概率，价值网络输出一个状态值。两组都把回报除以 100，保持相同尺度。
# 每 8 个完整回合更新一次，按“每回合梯度之和，再对回合取平均”计算损失。
# 优势通过 `detach()` 与价值网络断开，价值网络另用回归损失训练。

# %%
def make_network(outputs):
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, outputs))


def collect(policy, env):
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    while True:
        with torch.no_grad():
            action = torch.distributions.Categorical(logits=policy(torch.tensor(state))).sample()
        states.append(state)
        actions.append(int(action))
        state, reward, terminated, truncated, _ = env.step(int(action))
        rewards.append(reward)
        if terminated or truncated:
            break
    returns, running = [], 0.0
    for reward in reversed(rewards):
        running = reward + 0.99 * running
        returns.insert(0, running / 100)
    tensors = torch.tensor(np.array(states)), torch.tensor(actions), torch.tensor(returns)
    return *tensors, sum(rewards)


@torch.no_grad()
def evaluate(policy):
    env = gym.make("CartPole-v1")
    result = []
    for seed in range(1000, 1010):
        state, _ = env.reset(seed=seed)
        total = 0
        while True:
            action = int(policy(torch.tensor(state)).argmax())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        result.append(total)
    env.close()
    return float(np.mean(result))


def train(use_baseline, seed):
    torch.manual_seed(seed)
    policy, critic = make_network(2), make_network(1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-3)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    returns, logs, gradient_norms = [], [(0, evaluate(policy))], []
    for first in range(0, EPISODES, BATCH_EPISODES):
        batch = [collect(policy, env) for _ in range(BATCH_EPISODES)]
        losses = []
        for states, actions, targets, total in batch:
            baseline = critic(states).squeeze(1) if use_baseline else torch.zeros_like(targets)
            advantage = (targets - baseline).detach()
            logp = torch.distributions.Categorical(logits=policy(states)).log_prob(actions)
            losses.append(-(logp * advantage).sum())
            returns.append(total)
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        norm = nn.utils.clip_grad_norm_(policy.parameters(), 10)
        gradient_norms.append(float(norm))
        optimizer.step()
        if use_baseline:
            states = torch.cat([row[0] for row in batch])
            targets = torch.cat([row[2] for row in batch])
            for _ in range(5):
                critic_optimizer.zero_grad()
                loss = nn.functional.mse_loss(critic(states).squeeze(1), targets)
                loss.backward()
                critic_optimizer.step()
        if (first + BATCH_EPISODES) % 48 == 0:
            logs.append((first + BATCH_EPISODES, evaluate(policy)))
    env.close()
    return policy, np.array(returns), np.array(logs), gradient_norms


runs = {name: [train(enabled, seed) for seed in SEEDS]
        for name, enabled in [("REINFORCE", False), ("With baseline", True)]}
for name, samples in runs.items():
    print(name, "held-out return:", [float(row[2][-1, 1]) for row in samples])

# %% [markdown]
# ## 3. 对照曲线与回放 / Results
#
# 训练中策略不断变化，梯度范数的波动同时受到状态分布、回报和学习进度影响，
# 不能单凭它估计同一策略的梯度方差。前面的固定状态枚举才隔离了基线的方差作用。
# 下面比较两个种子的独立评估曲线，并回放训练后的策略。

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
metrics = {}
for name, samples in runs.items():
    logs = np.stack([sample[2] for sample in samples])
    axes[0].plot(logs[0, :, 0], logs[:, :, 1].mean(0), label=name)
    axes[0].fill_between(logs[0, :, 0], logs[:, :, 1].min(0),
                         logs[:, :, 1].max(0), alpha=0.15)
    norms = np.array([sample[3] for sample in samples])
    axes[1].plot(norms.mean(0), label=name)
    metrics[name] = {"heldout_returns": logs[:, -1, 1].tolist()}
axes[0].set(xlabel="Training episodes", ylabel="Held-out return (seed range)")
axes[1].set(xlabel="Update", ylabel="Gradient norm before clipping")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=120)
plt.show()

from IPython.display import Image, display
from PIL import Image as PILImage

policy = runs["With baseline"][0][0]
env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = env.reset(seed=2000)
frames = []
for step in range(500):
    if step % 2 == 0:
        frames.append(PILImage.fromarray(env.render()).resize((360, 240)))
    with torch.no_grad():
        action = int(policy(torch.tensor(state)).argmax())
    state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
frames[0].save(OUTPUT / "policy.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
display(Image(filename=str(OUTPUT / "policy.gif")))
torch.save(policy.state_dict(), OUTPUT / "policy-seed0.pt")
assert all(np.isfinite(row[3]).all() for samples in runs.values() for row in samples)
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 完成检查 / Checklist
# - [ ] 能用枚举解释状态基线不改变期望梯度。
# - [ ] 能说明一个过大的基线为什么反而增加方差。
# - [ ] 能区分理论的固定策略方差与训练中不断变化的梯度范数。
#
# ## 练习 / Exercise
# 回报为 100、状态预期为 120 时，填入尚未缩放的优势。提示：回报减基线。

# %%
RUN_EXERCISE = False
answer = None
if RUN_EXERCISE:
    assert answer == -20
else:
    print("Optional exercise: fill answer and enable RUN_EXERCISE.")

# %% [markdown]
# ## 参考资料 / References
# - [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html)：
#   REINFORCE 的回合更新及与动作无关的状态基线。
# - [本书基线实验](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_policy_gradient/baseline-experiment)：
#   继续比较价值网络的拟合误差如何影响更新。
