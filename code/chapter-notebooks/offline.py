# %% [markdown]
# # 固定数据之外的动作 / Offline contextual bandit
#
# 数据里某个状态只试过接近 0 的动作，价值网络却可能认为动作 1 更好。
# 训练误差很小，也无法证明这个从未试过的动作真的有效。
# 我们固定一份数据，比较直接最大化 Q、加入保守正则和加入行为约束的策略。
#
# CPU 默认 3 个种子，每种价值网络训练 600 步，策略训练 400 步，数据完全合成。
# 任务采用单步 contextual bandit，比较 CQL 风格正则与行为约束。
# 对应书中 `minimal_offline_rl_contextual_bandit.py`。

# %%
import importlib.util
import subprocess
import sys

packages = {"torch": "torch>=2.5,<3", "numpy": "numpy>=1.26,<3",
            "matplotlib": "matplotlib>=3.8,<4"}
missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "offline"
OUTPUT.mkdir(parents=True, exist_ok=True)
SEEDS, Q_STEPS, POLICY_STEPS = [0, 1, 2], 600, 400


def target_action(state):
    return 0.35 * torch.sin(2.5 * state)


def behavior(state):
    return 0.2 * target_action(state)


def reward_fn(state, action):
    return -(action - target_action(state)).square()


torch.manual_seed(2)
states = torch.empty(6000, 1).uniform_(-1, 1)
actions = (behavior(states) + 0.04 * torch.randn_like(states)).clamp(-1, 1)
rewards = reward_fn(states, actions)
# 数据固定后，训练器只看到下面的数组，不调用真实奖励函数。
train = (states[:4800], actions[:4800], rewards[:4800])
test = (states[4800:], actions[4800:], rewards[4800:])
print({"train_samples": len(train[0]), "test_samples": len(test[0]),
       "observed_action_range": [float(actions.min()), float(actions.max())]})

# %% [markdown]
# ## 1. 看见数据覆盖，再训练价值 / Coverage before optimization
#
# 状态覆盖了 −1 到 1，动作却只分布在一条窄带。
# 保守正则比较随机动作的 Q 值和数据动作的 Q 值，抑制在数据之外凭空给出的高分。
# 两种网络在同一个种子下从相同参数开始，使用同样的训练样本索引。

# %%
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1))


def train_q(seed, alpha):
    torch.manual_seed(seed)
    q = QNetwork()
    optimizer = torch.optim.Adam(q.parameters(), lr=5e-4)
    generator = torch.Generator().manual_seed(2000 + seed)
    logs = []
    for step in range(Q_STEPS):
        index = torch.randint(len(train[0]), (128,), generator=generator)
        state, action, reward = [item[index] for item in train]
        q_data = q(state, action)
        fit_loss = (q_data - reward).square().mean()
        # 均匀动作上的 log-mean-exp；常数不影响对 Q 的梯度。
        candidates = torch.rand(128, 10, 1, generator=generator) * 2 - 1
        repeated = state[:, None].expand(-1, 10, -1).reshape(-1, 1)
        q_random = q(repeated, candidates.reshape(-1, 1)).reshape(128, 10)
        penalty = (torch.logsumexp(q_random, -1) - np.log(10)).mean() - q_data.mean()
        loss = fit_loss + alpha * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 30 == 0:
            with torch.no_grad():
                mse = (q(test[0], test[1]) - test[2]).square().mean().item()
            logs.append((step, mse))
    return q.requires_grad_(False), logs


def train_actor(q, seed, bc_weight):
    torch.manual_seed(seed + 100)
    actor = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1), nn.Tanh())
    optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4)
    generator = torch.Generator().manual_seed(seed + 3000)
    for _ in range(POLICY_STEPS):
        index = torch.randint(len(train[0]), (128,), generator=generator)
        state, data_action = train[0][index], train[1][index]
        chosen = actor(state)
        loss = -q(state, chosen).mean() + bc_weight * (chosen - data_action).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return actor


runs = []
for seed in SEEDS:
    naive_q, naive_log = train_q(seed, 0.0)
    conservative_q, conservative_log = train_q(seed, 0.2)
    runs.append({"naive_q": naive_q, "conservative_q": conservative_q,
                 "naive_log": naive_log, "conservative_log": conservative_log,
                 "Naive Q": train_actor(naive_q, seed, 0),
                 "Conservative": train_actor(conservative_q, seed, 0),
                 "Q + BC": train_actor(naive_q, seed, 3)})
assert len(runs) == 3
print("Trained three seeds without collecting additional environment rewards.")

# %% [markdown]
# ## 2. 训练误差和真实回报可能分离 / Independent policy evaluation
#
# 测试数据上的均方误差仍只覆盖行为策略附近。现在冻结所有参数，才用已知模拟器评估新动作。
# 评估网格不会反馈给优化器。真实最优策略只作为 oracle 上界，训练时不提供给策略。
# 保守方法也可能过于保守，因此同时记录数据支持距离和真实回报。

# %%
grid = torch.linspace(-1, 1, 1000)[:, None]
metrics = {}
with torch.no_grad():
    for name in ["Naive Q", "Conservative", "Q + BC"]:
        policy_actions = [run[name](grid) for run in runs]
        returns = [float(reward_fn(grid, action).mean()) for action in policy_actions]
        gap = [float((action - behavior(grid)).abs().mean()) for action in policy_actions]
        metrics[name] = {"true_returns": returns, "support_gaps": gap}
    metrics["Behavior"] = {"true_return": float(reward_fn(grid, behavior(grid)).mean())}
    metrics["Oracle"] = {"true_return": 0.0}
print(json.dumps(metrics, indent=2))
assert np.isfinite([value for name in ["Naive Q", "Conservative", "Q + BC"]
                    for value in metrics[name]["true_returns"]]).all()

# %% [markdown]
# ## 3. 看见分布外估计 / Inspect the extrapolation
#
# 左图画出数据覆盖；中图固定一个状态，比较真实奖励和两种 Q 估计；右图画策略。
# 图固定使用种子 0，多种子结果见上表。真实奖励曲线仅供最终诊断，训练器从未读取它。
# 可以调整数据噪声和初始化，观察三种策略的变化。

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].scatter(states[::10], actions[::10], s=3, alpha=0.25, label="Fixed dataset")
axes[0].plot(grid, target_action(grid), color="black", label="Oracle")
axes[0].set(xlabel="State", ylabel="Action", ylim=(-1, 1))
candidate_actions = torch.linspace(-1, 1, 300)[:, None]
probe_state = torch.full_like(candidate_actions, 0.5)
with torch.no_grad():
    for key, label in [("naive_q", "Naive Q"), ("conservative_q", "Conservative Q")]:
        axes[1].plot(candidate_actions, runs[0][key](probe_state, candidate_actions), label=label)
    axes[1].plot(candidate_actions, reward_fn(probe_state, candidate_actions), "k--", label="True")
    for name in ["Naive Q", "Conservative", "Q + BC"]:
        axes[2].plot(grid, runs[0][name](grid), label=name)
axes[1].set(xlabel="Action at state=0.5", ylabel="Estimated Q / actual reward")
axes[2].plot(grid, behavior(grid), "k--", label="Behavior mean")
axes[2].set(xlabel="State", ylabel="Policy action", ylim=(-1, 1))
for ax in axes:
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUTPUT / "coverage-and-policy.png", dpi=120)
plt.show()
fig, ax = plt.subplots(figsize=(6, 3))
for key in ["naive_log", "conservative_log"]:
    logs = np.array([run[key] for run in runs])
    ax.semilogy(logs[0, :, 0], logs[:, :, 1].mean(0), label=key)
ax.set(xlabel="Update", ylabel="Held-out behavior-data MSE")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "fit-errors.png", dpi=120)
plt.show()
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
torch.save({name: runs[0][name].state_dict() for name in ["Naive Q", "Conservative", "Q + BC"]},
            OUTPUT / "policies-seed0.pt")

# %% [markdown]
# ## 完成检查 / Checklist
# - [ ] 固定数据后，训练阶段没有查询新动作的真实奖励。
# - [ ] 同时报告行为数据拟合误差、支持距离和独立策略回报。
# - [ ] 能说明保守正则与行为约束分别怎样影响策略。
#
# ## 练习 / Exercise
# 训练数据的动作接近 0，填入“Q(状态, 动作=1) 是否被这些数据直接验证”。
# 提示：网络能够输出一个数，不等于这个数有观测支持。

# %%
RUN_EXERCISE = False
answer = None
if RUN_EXERCISE:
    assert answer is False
else:
    print("Optional exercise: fill answer and enable RUN_EXERCISE.")

# %% [markdown]
# ## 参考资料 / References
# - [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)：
#   为什么抑制数据之外的过高估计可以帮助离线策略学习。
# - [TD3+BC](https://arxiv.org/abs/2106.06860)：
#   行为约束如何限制策略偏离数据。
