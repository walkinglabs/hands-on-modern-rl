# %% [markdown]
# # 把一组答案变成可检查的优势 / GRPO mechanism
#
# 同一道题得到四个回答，奖励依次为 `[0, 0, 1, 1]`。
# 组内平均奖励为 0.5，标准差为 0.5，因此优势接近 `[−1, −1, 1, 1]`。
# 全错的一组没有相对好坏，优势应为 0，不能除以 0。
#
# 这本 Notebook 是 CPU 机制实验，包含验证器、逐 token 裁剪和一次真实梯度更新。
# 策略是一个小型 token 概率表；语言模型训练见 DPO 和工具调用 Notebook。
# 对应 `code/chapter09_grpo_rlvr`，所有示例在本地生成。

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
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)
torch.manual_seed(7)
OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "grpo"
OUTPUT.mkdir(parents=True, exist_ok=True)
rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
advantages = (rewards - rewards.mean(-1, keepdim=True)) / (
    rewards.std(-1, correction=0, keepdim=True) + 1e-4)
assert torch.allclose(advantages[1], torch.zeros(4))
assert torch.isfinite(advantages).all()
print("Rewards:\n", rewards, "\nAdvantages:\n", advantages)

# %% [markdown]
# ## 1. 验证器要拒绝含糊答案 / A strict verifier
#
# 我们只接受形如 `答案: 42` 的单一整数答案，拒绝多个互相矛盾的答案和混入的说明。
#
# 下面用正确、错误、冲突和非整数答案检查验证器的返回值。

# %%
def verify(response, expected):
    match = re.fullmatch(r"\s*答案\s*[:：]\s*(-?\d+)\s*", response)
    return float(match is not None and int(match.group(1)) == expected)


cases = [("答案: 42", 42, 1), ("答案: 35", 42, 0),
         ("答案: 42 答案: 35", 42, 0), ("答案: -3", -3, 1),
         ("答案: 42.5", 42, 0), ("42", 42, 0)]
for response, expected, score in cases:
    actual = verify(response, expected)
    assert actual == score
    print(repr(response), "→", actual)

# %% [markdown]
# ## 2. 先裁剪每个 token，再平均 / Preserve the token dimension
#
# 两个 token 的概率比值是 1.5 和 0.5，逐 token 裁剪区间为 `[0.8, 1.2]`。
# 若先把整段概率相乘，得到 0.75，再裁剪就改变了目标。
# 下面保留 `[回答, token]` 的形状，padding 不计入平均，优势在同一回答内广播。

# %%
ratios = torch.tensor([[1.5, 0.5]])
positive = torch.ones(1, 1)
per_token = torch.minimum(ratios * positive, ratios.clamp(0.8, 1.2) * positive).mean()
sequence = torch.minimum(ratios.prod(-1), ratios.prod(-1).clamp(0.8, 1.2)).mean()
assert abs(per_token.item() - 0.85) < 1e-6
assert abs(sequence.item() - 0.75) < 1e-6
print({"token_surrogate": per_token.item(), "sequence_surrogate": sequence.item()})


def grpo_loss(logp, old_logp, ref_logp, advantage, mask, beta=0.04):
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.minimum(ratio * advantage[:, None],
                              ratio.clamp(0.8, 1.2) * advantage[:, None])
    log_ref_ratio = ref_logp - logp
    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
    per_response = ((-surrogate + beta * kl) * mask).sum(-1) / mask.sum(-1)
    return per_response.mean()


logits = torch.nn.Parameter(torch.zeros(4, 3, 2))
actions = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1]])
mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
old_logp = logits.detach().log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
ref_logp = old_logp.clone()
optimizer = torch.optim.SGD([logits], lr=0.5)
losses = []
for _ in range(4):
    logp = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
    loss = grpo_loss(logp, old_logp, ref_logp, advantages[0], mask)
    optimizer.zero_grad()
    loss.backward()
    assert torch.equal(logits.grad[mask == 0], torch.zeros_like(logits.grad[mask == 0]))
    optimizer.step()
    losses.append(float(loss.detach()))
change = (logits.detach().log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
          - old_logp)
mean_change = (change * mask).sum(-1) / mask.sum(-1)
assert (mean_change[:2] < 0).all() and (mean_change[2:] > 0).all()
print("Mean selected-token log-probability change:", mean_change.tolist())

# %% [markdown]
# ## 3. 梯度与概率变化 / Gradients and probability changes
#
# 概率表的更新支持三件事：较好的回答增加概率、较差的回答降低概率、padding 没有梯度。
#
# 我们使用总体标准差 `correction=0`；部分框架使用样本标准差，数值尺度会有差别。

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].bar(np.arange(4), advantages[0].numpy())
axes[0].set(xlabel="Response", ylabel="Group-relative advantage")
axes[1].bar(np.arange(4), mean_change.numpy())
axes[1].set(xlabel="Response", ylabel="Mean token log-probability change")
fig.tight_layout()
fig.savefig(OUTPUT / "mechanism.png", dpi=120)
plt.show()
metrics = {"verifier_tests": len(cases), "token_surrogate": per_token.item(),
           "sequence_surrogate": sequence.item(), "logprob_change": mean_change.tolist()}
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 完成检查 / Checklist
# - [ ] 全错或全对的组有有限数值，零方差时优势为 0。
# - [ ] 概率比值和裁剪保留 token 维，回答长度归一化在裁剪之后。
# - [ ] 验证器拒绝矛盾答案，padding 的梯度为 0。
#
# ## 练习 / Exercise
# 填入比值 `[1.5, 0.5]`、正优势为 1 时的逐 token 裁剪目标平均值。
# 提示：先分别取 `min(ratio, clip(ratio))`，再平均。

# %%
RUN_EXERCISE = False
answer = None
if RUN_EXERCISE:
    assert abs(answer - 0.85) < 1e-6
else:
    print("Optional exercise: fill answer and enable RUN_EXERCISE.")

# %% [markdown]
# ## 参考资料 / References
# - [DeepSeekMath, Section 4.1](https://arxiv.org/html/2402.03300)：
#   GRPO 的逐 token 概率比值、长度平均和参考策略 KL 项。
# - [本书 RLVR](https://walkinglabs.github.io/hands-on-modern-rl/chapter18_grpo/rlvr)：
#   把验证规则连接到可训练策略，并讨论奖励设计的限制。
