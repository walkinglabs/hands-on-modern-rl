# %% [markdown]
# # 从价值表到一条路径 / GridWorld
#
# 我们从左上角出发，进入右下角得到 1 分，进入 (1, 1) 的陷阱得到 −1 分并结束。
# 普通一步扣 0.01 分。向墙外移动会留在原地，因此一直绕路也有代价。
#
# 先用已知转移规则计算价值，再让 Q-learning 只通过实际动作学习。
# 默认 CPU、30 个种子、每个种子 500 回合，数据完全在本地生成。
# 与 `code/chapter03_mdp/gridworld_q_learning.py` 使用相同的地图、奖励和终止约定。

# %%
import importlib.util
import subprocess
import sys

packages = {"numpy": "numpy>=1.26,<3", "matplotlib": "matplotlib>=3.8,<4"}
missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "gridworld"
OUTPUT.mkdir(parents=True, exist_ok=True)
START, TRAP, GOAL = (0, 0), (1, 1), (3, 3)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAMMA, ALPHA, EPISODES, SEEDS = 0.95, 0.15, 500, 30
print({"start": START, "trap": TRAP, "goal": GOAL, "episodes": EPISODES})

# %% [markdown]
# ## 1. 奖励属于一次转移 / Transitions and terminal states
#
# 陷阱和终点的奖励在“进入”时结算，结束后不再给下一状态价值。
# 例如从 (3, 2) 向右走，目标值就是 1；再加上终点的价值会重复计算奖励。

# %%
def transition(state, action):
    if state in (TRAP, GOAL):
        return state, 0.0, True
    dr, dc = MOVES[action]
    nxt = (state[0] + dr, state[1] + dc)
    if not (0 <= nxt[0] < 4 and 0 <= nxt[1] < 4):
        nxt = state
    reward = 1.0 if nxt == GOAL else -1.0 if nxt == TRAP else -0.01
    return nxt, reward, nxt in (TRAP, GOAL)


def action_values(values, state):
    result = []
    for action in range(4):
        nxt, reward, done = transition(state, action)
        result.append(reward + GAMMA * values[nxt] * (not done))
    return np.array(result)


values, history = np.zeros((4, 4)), []
for sweep in range(1000):
    history.append(values.copy())
    updated = values.copy()
    for state in np.ndindex(4, 4):
        if state not in (TRAP, GOAL):
            updated[state] = action_values(values, state).max()
    delta = np.abs(updated - values).max()
    values = updated
    if delta < 1e-12:
        break
assert transition((3, 2), 3) == (GOAL, 1.0, True)
assert abs(values[START] - 0.728537125) < 1e-9
print("Optimal state values:\n", np.round(values, 3))

# %% [markdown]
# ## 2. 不提前提供地图答案 / Learning by interaction
#
# Q 表形状为 `[4, 4, 4]`，最后一维分别对应上、下、左、右。
# 每一步只更新刚选择的格子和动作。探索率从 1 降到 0.05，让前期尝试逐渐变成后期利用。

# %%
def train(seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((4, 4, 4))
    returns = []
    for episode in range(EPISODES):
        state, total = START, 0.0
        epsilon = 1 - 0.95 * episode / (EPISODES - 1)
        for _ in range(100):
            candidates = np.flatnonzero(np.isclose(q[state], q[state].max()))
            action = int(rng.integers(4) if rng.random() < epsilon else rng.choice(candidates))
            nxt, reward, done = transition(state, action)
            bootstrap = 0.0 if done else q[nxt].max()
            q[state][action] += ALPHA * (reward + GAMMA * bootstrap - q[state][action])
            total += reward
            state = nxt
            if done:
                break
        returns.append(total)
    return q, returns


def evaluate(q):
    state, path, total = START, [START], 0.0
    for _ in range(100):
        state, reward, done = transition(state, int(q[state].argmax()))
        path.append(state)
        total += reward
        if done:
            break
    return {"success": state == GOAL, "steps": len(path) - 1, "return": total, "path": path}


runs = [train(seed) for seed in range(SEEDS)]
evaluations = [evaluate(q) for q, _ in runs]
metrics = {"success_rate": float(np.mean([r["success"] for r in evaluations])),
           "mean_steps": float(np.mean([r["steps"] for r in evaluations]))}
print(metrics)
print("Seed 0 greedy path:", evaluations[0]["path"])
assert metrics["success_rate"] == 1.0
assert metrics["mean_steps"] == 6.0

# %% [markdown]
# ## 3. 同时查看路径和曲线 / Policy and learning curves
#
# 曲线来自仍带探索的训练回合，路径来自关闭探索后的单独评估。
# 两者的回报通常不同。30 个种子全部成功，只支持这个确定的小地图，不能推断到陌生地图。

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].imshow(values, cmap="Blues")
q = runs[0][0]
axes[1].imshow(q.max(-1), cmap="Blues")
arrows = ["↑", "↓", "←", "→"]
for state in np.ndindex(4, 4):
    row, col = state
    axes[0].text(col, row, f"{values[state]:.2f}", ha="center", va="center")
    label = "X" if state == TRAP else "G" if state == GOAL else arrows[q[state].argmax()]
    axes[1].text(col, row, label, ha="center", va="center", fontsize=15)
path = np.array(evaluations[0]["path"])
axes[1].plot(path[:, 1], path[:, 0], color="orange", alpha=0.6)
rewards = np.stack([reward for _, reward in runs])
window = np.ones(20) / 20
smooth = np.stack([np.convolve(row, window, mode="valid") for row in rewards])
x = np.arange(19, EPISODES)
axes[2].plot(x, smooth.mean(0))
axes[2].fill_between(x, smooth.mean(0) - smooth.std(0),
                     smooth.mean(0) + smooth.std(0), alpha=0.2)
for ax, title in zip(axes, ["Value iteration", "Q-learning policy / seed 0", "Training return"]):
    ax.set_title(title)
axes[2].set(xlabel="Episode", ylabel="Return (20-episode mean)")
fig.tight_layout()
fig.savefig(OUTPUT / "gridworld.png", dpi=120)
plt.show()
np.save(OUTPUT / "q-table.npy", q)
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 完成检查 / Checklist
# - [ ] 能说明终止状态的后继价值为什么取 0。
# - [ ] 能从 Q 表读出一条实际抵达终点的路径。
# - [ ] 能区分带探索的训练回报与确定性评估回报。
#
# ## 练习 / Exercise
# 填入从 (3, 2) 向右进入终点时的 TD 目标。提示：这一步已经终止。

# %%
RUN_EXERCISE = False
answer = None
if RUN_EXERCISE:
    assert answer == 1.0
else:
    print("Optional exercise: fill answer and enable RUN_EXERCISE.")

# %% [markdown]
# ## 参考资料 / References
# - [本书价值实验](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/value-experiment)：
#   对应相同的地图和奖励约定。
# - [Sutton & Barto, Chapters 4 and 6](http://incompleteideas.net/book/the-book-2nd.html)：
#   解释已知模型的价值迭代与通过采样学习的差别。
