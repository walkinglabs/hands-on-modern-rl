# %% [markdown]
# # 用成对偏好更新语言模型 / Tiny language model DPO
#
# 同一句请求 `help topic0`，可以接上 `please read topic0 carefully`，也可以接上
# `go away topic0 now`。我们先让小模型学会生成这两种回答，再用成对偏好提高第一种回答的相对概率。
#
# 本实验用 CPU 从零训练一个微型 GRU 自回归语言模型，再执行真实 DPO 参数更新。
# 数据由代码生成，默认 400 步预训练、120 步 DPO。
# 书中 Qwen 0.5B 训练见 `code/chapter17_dpo` 五步脚本。

# %%
import importlib.util
import subprocess
import sys

packages = {"torch": "torch>=2.5,<3", "numpy": "numpy>=1.26,<3",
            "matplotlib": "matplotlib>=3.8,<4"}
missing = [spec for name, spec in packages.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import copy
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(7)
random.seed(7)
OUTPUT = Path(os.getenv("HANDS_ON_LAB_OUTPUT", "lab-output")) / "dpo"
OUTPUT.mkdir(parents=True, exist_ok=True)
SFT_STEPS, DPO_STEPS, BETA = 400, 120, 0.2
words = ["<pad>", "<sep>", "<eos>", "help", "explain", "please", "read", "carefully",
         "go", "away", "now"] + [f"topic{i}" for i in range(12)]
vocab = {word: index for index, word in enumerate(words)}
print({"vocabulary": len(words), "sft_steps": SFT_STEPS, "dpo_steps": DPO_STEPS})

# %% [markdown]
# ## 1. 把提示词和回答分开 / Completion-only likelihood
#
# 输入是提示词、分隔符和回答。每个位置的 logits 预测下一个词。
# DPO 使用“整段回答的对数概率之和”，排除提示词、分隔符本身和 padding，保留回答的 EOS。
# 这里两个回答同长；扩展到不同长度时仍要对有效回答词求和。

# %%
def example(topic, verb="help"):
    return {"prompt": f"{verb} topic{topic}",
            "chosen": f"please read topic{topic} carefully <eos>",
            "rejected": f"go away topic{topic} now <eos>"}


def batch(rows, field):
    inputs, masks = [], []
    for row in rows:
        prompt = row["prompt"].split() + ["<sep>"]
        response = row[field].split()
        inputs.append([vocab[w] for w in prompt + response])
        # 移位后，第 len(prompt)-1 个预测位置对应回答的第一个词。
        masks.append([0.0] * (len(prompt) - 1) + [1.0] * len(response))
    return torch.tensor(inputs), torch.tensor(masks)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(words), 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.head = nn.Linear(64, len(words))

    def forward(self, tokens):
        hidden, _ = self.gru(self.embedding(tokens))
        return self.head(hidden)


def sequence_logp(model, tokens, mask):
    logits = model(tokens[:, :-1])
    per_token = logits.log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
    return (per_token * mask).sum(-1)


test_tokens, test_mask = batch([example(0)], "chosen")
assert test_tokens.shape == (1, 8) and test_mask.sum() == 5
print("Example pair:", example(0))
print("Tokens:", test_tokens.tolist(), "answer mask:", test_mask.tolist())

# %% [markdown]
# ## 2. 先学语言，再固定参考模型 / SFT and the reference policy
#
# 预训练覆盖 12 个主题，两种措辞等量出现。DPO 只使用主题 0–7 的偏好，
# 主题 8–11 留作偏好评估。留出主题在语言预训练中出现过，因此这里检验的是偏好迁移。
# 预训练后复制出参考模型，DPO 期间它的参数保持冻结。

# %%
all_rows = [example(i, verb) for i in range(12) for verb in ["help", "explain"]]
train_rows = [row for i in range(8) for row in [example(i), example(i, "explain")]]
heldout = [row for i in range(8, 12) for row in [example(i), example(i, "explain")]]
model = TinyLM()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
sft_losses = []
for step in range(SFT_STEPS):
    rows = random.choices(all_rows, k=16)
    chosen, chosen_mask = batch(rows, "chosen")
    rejected, rejected_mask = batch(rows, "rejected")
    loss = -(sequence_logp(model, chosen, chosen_mask).mean()
             + sequence_logp(model, rejected, rejected_mask).mean()) / 10
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sft_losses.append(float(loss.detach()))
reference = copy.deepcopy(model).eval().requires_grad_(False)
reference_snapshot = {name: value.clone() for name, value in reference.state_dict().items()}


@torch.no_grad()
def evaluate(policy, rows):
    c, cm = batch(rows, "chosen")
    r, rm = batch(rows, "rejected")
    gap = sequence_logp(policy, c, cm) - sequence_logp(policy, r, rm)
    ref_gap = sequence_logp(reference, c, cm) - sequence_logp(reference, r, rm)
    return {"chosen_pair_rate": float((gap > 0).float().mean()),
            "logprob_gap": float(gap.mean()), "reward_margin": float(BETA * (gap - ref_gap).mean())}


@torch.no_grad()
def generate(policy, prompt):
    tokens = [vocab[w] for w in prompt.split() + ["<sep>"]]
    output = []
    for _ in range(8):
        token = int(policy(torch.tensor([tokens]))[0, -1].argmax())
        tokens.append(token)
        if words[token] == "<eos>":
            break
        output.append(words[token])
    return " ".join(output)


before = evaluate(model, heldout)
before_text = [generate(model, row["prompt"]) for row in heldout]
print("SFT held-out preference metrics:", before)
print("Sample before:", before_text[0])

# %% [markdown]
# ## 3. 比较相对于参考模型的改变 / DPO update
#
# 假设当前模型让好回答的对数概率增加 2、坏回答增加 0.5，相对间隔就是 1.5。
# 乘上 β 后，`−logsigmoid` 鼓励这个间隔增大。
# 参考模型只参与数值计算，当前模型的 chosen 和 rejected 两条路径都参与梯度。

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
losses, margins = [], []
for step in range(DPO_STEPS):
    rows = random.choices(train_rows, k=16)
    chosen, cm = batch(rows, "chosen")
    rejected, rm = batch(rows, "rejected")
    current_gap = sequence_logp(model, chosen, cm) - sequence_logp(model, rejected, rm)
    with torch.no_grad():
        ref_gap = sequence_logp(reference, chosen, cm) - sequence_logp(reference, rejected, rm)
    margin = BETA * (current_gap - ref_gap)
    loss = -nn.functional.logsigmoid(margin).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    losses.append(float(loss.detach()))
    margins.append(evaluate(model, heldout)["reward_margin"])
after = evaluate(model, heldout)
after_text = [generate(model, row["prompt"]) for row in heldout]
topic_before = np.mean([row["prompt"].split()[-1] in text.split()
                        for row, text in zip(heldout, before_text)])
topic_after = np.mean([row["prompt"].split()[-1] in text.split()
                       for row, text in zip(heldout, after_text)])
assert all(torch.equal(reference.state_dict()[key], value)
           for key, value in reference_snapshot.items())
assert after["reward_margin"] > 0
assert any(not torch.equal(value, reference.state_dict()[key])
           for key, value in model.state_dict().items())
print("Before:", before, "\nAfter:", after)
print("Requested topic retention before / after:", topic_before, topic_after)
for row, old, new in zip(heldout[:4], before_text, after_text):
    print(row["prompt"], "\n before:", old, "\n after:", new)

# %% [markdown]
# ## 4. 偏好指标与生成内容 / Evaluation
#
# 我们同时查看 chosen 与 rejected 的偏好间隔，以及留出主题上的生成内容。
# 下表逐条展示更新前后的回答，图中记录偏好率与主题保持率。

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, data, label in zip(axes, [sft_losses, losses, margins],
                          ["SFT token loss", "DPO loss", "Held-out reward margin"]):
    ax.plot(data)
    ax.set(xlabel="Update", ylabel=label)
fig.tight_layout()
fig.savefig(OUTPUT / "training.png", dpi=120)
plt.show()
metrics = {"before": before, "after": after,
           "requested_topic_retention": {"before": float(topic_before),
                                         "after": float(topic_after)},
           "examples": [{"prompt": row["prompt"], "before": old, "after": new}
                        for row, old, new in zip(heldout, before_text, after_text)]}
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUTPUT / "preferences.json").write_text(json.dumps(train_rows, indent=2))
torch.save({"state_dict": model.state_dict(), "vocabulary": words}, OUTPUT / "tiny-dpo.pt")

# %% [markdown]
# ## 完成检查 / Checklist
# - [ ] 训练了一个自回归模型，DPO 实际改变了模型参数。
# - [ ] 参考模型保持冻结，概率只累计有效回答词。
# - [ ] 将留出集的偏好间隔与生成内容分开评价。
#
# ## 练习 / Exercise
# 当前 chosen/rejected 的相对对数概率改变分别为 2 和 0.5，β=0.2，填入奖励间隔。
# 提示：先相减，再乘 β。

# %%
RUN_EXERCISE = False
answer = None
if RUN_EXERCISE:
    assert abs(answer - 0.3) < 1e-6
else:
    print("Optional exercise: fill answer and enable RUN_EXERCISE.")

# %% [markdown]
# ## 参考资料 / References
# - [DPO 原论文](https://arxiv.org/abs/2305.18290)：
#   DPO 使用成对回答相对于固定参考策略的对数概率差。
# - [本书 Qwen DPO 实验](https://walkinglabs.github.io/hands-on-modern-rl/chapter17_dpo/dpo-hands-on)：
#   使用同样的目标训练 Qwen 语言模型。
