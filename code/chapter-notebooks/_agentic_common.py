import random

import numpy as np
import torch
from torch import nn


class AgentPolicy(nn.Module):
    def __init__(self, dimensions, actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, 32), nn.Tanh(), nn.Linear(32, actions))

    def forward(self, observation, legal):
        logits = self.net(torch.tensor(observation, dtype=torch.float32))
        return logits.masked_fill(~torch.tensor(legal, dtype=torch.bool), -1e9)


def rollout(policy, task, factory, greedy=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    env = factory(task)
    logprobs, entropies, trace, rewards = [], [], [], []
    for turn in range(env.max_turns):
        observation, legal = env.observe()
        distribution = torch.distributions.Categorical(logits=policy(observation, legal))
        action = distribution.logits.argmax() if greedy else distribution.sample()
        logprobs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        reward, done, response = env.step(int(action))
        rewards.append(reward)
        trace.append({"turn": turn, "action": env.actions[int(action)],
                      "observation": response, "reward": reward})
        if done:
            break
    success = bool(env.success)
    if hasattr(env, "close"):
        env.close()
    return {"return": float(sum(rewards)), "success": success, "trace": trace,
            "logprob": torch.stack(logprobs).sum(),
            "entropy": torch.stack(entropies).sum()}


@torch.no_grad()
def evaluate_agent(policy, tasks, factory, greedy=True):
    samples = [rollout(policy, task, factory, greedy=greedy, seed=10000 + i)
               for i, task in enumerate(tasks)]
    return {"success_rate": float(np.mean([item["success"] for item in samples])),
            "return": float(np.mean([item["return"] for item in samples])),
            "mean_turns": float(np.mean([len(item["trace"]) for item in samples]))}


def train_agent(factory, tasks, heldout, seed, updates=160):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    probe = factory(tasks[0])
    policy = AgentPolicy(len(probe.observe()[0]), len(probe.actions))
    if hasattr(probe, "close"):
        probe.close()
    before = evaluate_agent(policy, heldout, factory)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    history = []
    for update in range(updates):
        losses = []
        for task in rng.choices(tasks, k=4):
            group = [rollout(policy, task, factory) for _ in range(4)]
            total = sum(item["return"] for item in group)
            for item in group:
                baseline = (total - item["return"]) / (len(group) - 1)
                advantage = item["return"] - baseline
                losses.append(-item["logprob"] * advantage - 0.002 * item["entropy"])
        optimizer.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 2)
        optimizer.step()
        if update % 20 == 0 or update == updates - 1:
            history.append({"update": update, "loss": float(loss.detach()),
                            **evaluate_agent(policy, heldout, factory)})
    after = evaluate_agent(policy, heldout, factory)
    return policy, before, after, history


def run_comparison(factory, training, heldout, output, updates=160):
    results, policies, curves = [], [], []
    for seed in [3, 7]:
        policy, before, after, history = train_agent(
            factory, training, heldout, seed, updates=updates)
        results.append({"seed": seed, "before": before, "after": after})
        policies.append(policy)
        curves.append(history)
    for row in results:
        print(row)
    import json
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for row, curve in zip(results, curves):
        ax.plot([point["update"] for point in curve],
                [point["success_rate"] for point in curve], label=f"seed {row['seed']}")
    ax.set(xlabel="Policy update", ylabel="Held-out greedy success", ylim=(-0.05, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=120)
    plt.show()
    trace = rollout(policies[0], heldout[0], factory, greedy=True)["trace"]
    print(json.dumps(trace, indent=2))
    torch.save(policies[0].state_dict(), output / "policy-seed3.pt")
    metrics = {"runs": results, "fixed_heldout_trace": trace,
               "train_tasks": len(training), "heldout_tasks": len(heldout),
               "updates": updates, "algorithm": "REINFORCE with leave-one-out baseline"}
    assert all(np.isfinite(row["after"]["return"]) for row in results)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return policies, metrics


print("Agent contract: observe → action → tool observation → next decision → verifier")
