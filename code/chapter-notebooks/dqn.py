# %% [markdown]
# # 把选择动作和评价动作分开 / DQN and Double DQN
#
# 两个动作真实价值都为 0，估计时却各自带有噪声。每次取较大的估计，平均值就可能大于 0。
# 我们先重现这个偏差，再在 CartPole 中比较 DQN 与 Double DQN。
#
# 默认 CPU，每种算法 2 个种子、每个种子 15,000 步。可将 `STEPS` 提高到 80,000、
# `SEEDS` 增加到 5 做更长的对照。短训练用于观察过程，不承诺每个种子都解出任务。
# 对应源码：`code/chapter04_dqn/double_dqn_cartpole.py`。

# %%
import importlib.util
import subprocess
import sys

packages = {"numpy": "numpy>=1.26,<3", "matplotlib": "matplotlib>=3.8,<4",
            "torch": "torch>=2.5,<3", "gymnasium": "gymnasium[classic-control]>=1.0,<2",
            "pygame": "pygame>=2.5,<3", "PIL": "pillow>=10,<13"}
missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import collections
import copy
import json
import os
import random
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "dqn"
OUTPUT.mkdir(parents=True, exist_ok=True)
STEPS, SEEDS, GAMMA = 15000, [0, 1], 0.99
rng = np.random.default_rng(0)
selection, evaluation = rng.normal(size=(2, 10000, 2))
same = selection.max(1).mean()
separate = evaluation[np.arange(10000), selection.argmax(1)].mean()
print({"noisy_max": float(same), "separate_evaluation": float(separate)})
assert same > 0.4 and abs(separate) < 0.05

# %% [markdown]
# ## 1. 看清目标值的一个差别 / The bootstrap target
#
# 网络输入 `[batch, 4]`，输出 `[batch, 2]` 的动作价值。
# DQN 用目标网络同时选择和评价；Double DQN 用在线网络选动作、目标网络给分。
# `terminated` 会清除后继价值，时间上限 `truncated` 只结束采样回合。

# %%
def network():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, 2))


@torch.no_grad()
def bootstrap(online_values, target_values, double):
    if double:
        actions = online_values.argmax(-1, keepdim=True)
        return target_values.gather(1, actions).squeeze(1)
    return target_values.max(1).values


online_test = torch.tensor([[3.0, 1.0]])
target_test = torch.tensor([[1.0, 4.0]])
assert bootstrap(online_test, target_test, False).item() == 4
assert bootstrap(online_test, target_test, True).item() == 1
print("Toy target test passed: DQN=4, Double DQN=1.")

# %% [markdown]
# ## 2. 同样的预算与独立评估 / Training and evaluation
#
# 两种算法使用相同网络大小、探索安排和评估种子。回放池保存真实转移，目标网络每 250 步同步。
# 每 1,500 步在独立环境上关闭探索评估；这些回合不会进入训练池。

# %%
@torch.no_grad()
def evaluate(model, seeds=range(1000, 1010)):
    env = gym.make("CartPole-v1")
    returns, estimates = [], []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        estimates.append(float(model(torch.tensor(state)).max()))
        total = 0
        while True:
            action = int(model(torch.tensor(state)).argmax())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.mean(estimates))


def train(double, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    online = network()
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=5e-4)
    buffer = collections.deque(maxlen=20000)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)
    log = [(0, *evaluate(online))]
    for step in range(1, STEPS + 1):
        epsilon = max(0.05, 1 - step / (0.6 * STEPS))
        with torch.no_grad():
            action = int(online(torch.tensor(state)).argmax())
        if random.random() < epsilon:
            action = env.action_space.sample()
        nxt, reward, terminated, truncated, _ = env.step(action)
        buffer.append((state, action, reward, nxt, float(terminated)))
        state = nxt
        if terminated or truncated:
            state, _ = env.reset()
        if len(buffer) >= 256 and step % 4 == 0:
            s, a, r, ns, terminal = zip(*random.sample(buffer, 64))
            s, ns = torch.tensor(np.array(s)), torch.tensor(np.array(ns))
            q = online(s).gather(1, torch.tensor(a).view(-1, 1)).squeeze(1)
            with torch.no_grad():
                next_value = bootstrap(online(ns), target(ns), double)
                y = torch.tensor(r) + GAMMA * (1 - torch.tensor(terminal)) * next_value
            loss = nn.functional.smooth_l1_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10)
            optimizer.step()
        if step % 250 == 0:
            target.load_state_dict(online.state_dict())
        if step % 1500 == 0:
            log.append((step, *evaluate(online)))
    env.close()
    return online, np.array(log)


runs = {}
for name, double in [("DQN", False), ("Double DQN", True)]:
    runs[name] = [train(double, seed) for seed in SEEDS]
    print(name, "held-out returns:", [float(log[-1, 1]) for _, log in runs[name]])

# %% [markdown]
# ## 3. 奖励、价值与真实回放 / Inspect the learned policy
#
# Q 值是折扣回报的预测，不能直接与未折扣的 500 分上限相减来证明高估。
# 下面只报告它的变化。算法的优劣要在更长训练、更多种子和其他环境中继续检验。
# 回放固定使用 Double DQN 的种子 0，避免从多个运行中挑选最好的一条。

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
metrics = {}
for name, samples in runs.items():
    logs = np.stack([log for _, log in samples])
    for column, ax in [(1, axes[0]), (2, axes[1])]:
        mean, std = logs[:, :, column].mean(0), logs[:, :, column].std(0)
        ax.plot(logs[0, :, 0], mean, label=name)
        ax.fill_between(logs[0, :, 0], mean - std, mean + std, alpha=0.15)
    metrics[name] = {"evaluation_returns": logs[:, -1, 1].tolist(),
                     "initial_state_q": logs[:, -1, 2].tolist()}
for ax, label in zip(axes, ["Held-out return", "Initial-state max Q estimate"]):
    ax.set(xlabel="Environment steps", ylabel=label)
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=120)
plt.show()

from IPython.display import Image, display
from PIL import Image as PILImage

model = runs["Double DQN"][0][0]
env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = env.reset(seed=2000)
frames = []
for step in range(500):
    if step % 2 == 0:
        frames.append(PILImage.fromarray(env.render()).resize((360, 240)))
    with torch.no_grad():
        action = int(model(torch.tensor(state)).argmax())
    state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
frames[0].save(OUTPUT / "policy.gif", save_all=True, append_images=frames[1:],
               duration=40, loop=0)
display(Image(filename=str(OUTPUT / "policy.gif")))
torch.save(model.state_dict(), OUTPUT / "double-dqn-seed0.pt")
assert np.isfinite([v for item in metrics.values() for v in item["evaluation_returns"]]).all()
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 完成检查 / Checklist
# - [ ] 能在两行目标计算中指出动作选择与评价的区别。
# - [ ] 能区分任务终止和时间上限，解释后继价值的掩码。
# - [ ] 保存了两个种子的评估曲线和固定种子回放，包括未学好的结果。
#
# ## 练习 / Exercise
# 在线网络给出 `[3, 1]`、目标网络给出 `[1, 4]` 时，填入 Double DQN 的后继价值。
# 提示：先看在线网络选择的动作下标。

# %%
RUN_EXERCISE = False
answer = None
if RUN_EXERCISE:
    assert answer == 1.0
else:
    print("Optional exercise: fill answer and enable RUN_EXERCISE.")

# %% [markdown]
# ## 参考资料 / References
# - [Double Q-learning with deep networks](https://arxiv.org/abs/1509.06461)：
#   动作选择和动作评价的分离方式，以及实验结论的适用范围。
# - [Gymnasium Env](https://gymnasium.farama.org/api/env/)：
#   `terminated` 与 `truncated` 的接口含义。
