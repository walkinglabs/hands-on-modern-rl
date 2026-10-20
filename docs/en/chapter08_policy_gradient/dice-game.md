---
title: '6.3 Hands-on: A Two-Armed Bandit (Dice-Game Slot Machine)'
---

# 6.3 Hands-on: Two-Armed Dice-Game Bandit

> **Section goal**: Implement the smallest policy-gradient experiment with a two-armed bandit and observe how rewards change action probabilities.

> **Learning path**: [2.1 Exploration and Exploitation](../chapter03_mdp/bandit) → [6.1 Policy Gradient and REINFORCE](./reinforce) → **6.3 Dice-Game Bandit**

> **Code and resources**: This page contains the complete environment, policy network, and training loop. Run the code blocks in order.

Save the complete code in Section 6.3.2 as `policy_gradient_bandit.py`, then run:

```bash
python policy_gradient_bandit.py
```

A two-armed slot machine has one red arm and one blue arm. The red arm pays $1 with probability 0.3, while the blue arm pays $1 with probability 0.7. The agent chooses one arm per round and starts without knowing either probability.

::: tip How This Relates To Chapter 3
In Chapter 2's [multi-armed bandit experiment](../chapter03_mdp/bandit), we discussed regret: "how many points did we lose because we did not pick the optimal arm?" There the emphasis was **analysis**: given a strategy, how fast does regret grow? Here the emphasis is **learning**: let the AI discover the best arm by interacting with the environment.

The bridge between the two is the same mathematical object: the [expected reward](../chapter03_mdp/bandit) $\mathbb{E}[R_a]$.
:::

This is the experimental playground for this section: an extremely minimal bandit. There are only two actions and no "state" (a stateless environment). The rules can be explained in one sentence. Yet in this tiny setting, we can watch a policy network evolve from "knowing nothing" to "choosing the optimal action." More importantly, we can see with our own eyes the core (and essentially the only) learning signal of policy gradients:

**good outcomes reinforce the probability of the action that produced them.**

This differs from the coin-guessing game in Chapter 2. There we wrote a [deterministic policy](../chapter03_mdp/mdp), “always guess heads,” by hand. Here the model learns a [parameterized policy](../chapter03_mdp/policy-value) $\pi_\theta(a|s)$.

## 6.3.1 Two-Armed Bandit Environment

```
┌──────────────────────────────────┐
│                                  │
│   ┌───┐          ┌───┐           │
│   │ A │          │ B │           │
│   │🔴│          │🔵│           │
│   └─┬─┘          └─┬─┘           │
│     │ win rate 30% │ win rate 70% │
│     │               │            │
│     └───────────────┘            │
│                                  │
│   Rule: pick one arm; win => +1  │
│   Goal: let AI discover B is best │
└──────────────────────────────────┘
```

## 6.3.2 Implementing A Policy Network In PyTorch

Our policy network is extremely simple: it is essentially a single Softmax layer. The input is a constant (because there is no state), and the output is a probability over the two actions:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# ==========================================
# 1. Policy network: a single Softmax layer
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)  # 1 input (constant), 2 outputs (prob of A and B)

    def forward(self, x):
        logits = self.linear(x)
        return torch.softmax(logits, dim=-1)  # Softmax ensures a valid distribution

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# ==========================================
# 2. Environment: two-armed bandit
# ==========================================
win_probs = [0.3, 0.7]  # A: 30%, B: 70%

def pull_arm(action):
    return 1.0 if random.random() < win_probs[action] else 0.0

# ==========================================
# 3. REINFORCE training (watch the effect first; theory next)
# ==========================================
prob_history = []
num_episodes = 300

for ep in range(num_episodes):
    state = torch.tensor([1.0])

    # Policy network outputs action probabilities; sample according to the distribution
    probs = policy(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()            # sample an action
    log_prob = dist.log_prob(action)  # log π(a|s)

    # Execute the action and observe the reward
    reward = pull_arm(action.item())

    # REINFORCE core: good outcome => increase probability
    loss = -log_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track the probability of choosing B
    with torch.no_grad():
        prob_history.append(policy(state)[1].item())  # P(B)

print(f"Initial P(B): {prob_history[0]:.3f}")
print(f"Final   P(B): {prob_history[-1]:.3f}")
```

The heart of this code is the single line `loss = -log_prob * reward`. Intuitively:

If the sampled action leads to a good outcome (`reward = 1`), then `-log_prob * 1` produces a gradient that pushes up the probability of that action. If the outcome is bad (`reward = 0`), the gradient is zero, and the probability stays unchanged. The minus sign is there because PyTorch performs gradient descent (minimizing a loss), while we conceptually want gradient ascent (maximizing expected return).

This formula is the single-step case of the policy-gradient estimator introduced with the [policy objective](../chapter03_mdp/policy-value):
$\nabla_\theta J(\theta) \propto \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$.
In the next section we will derive it carefully.

## 6.3.3 What You Observe During Training

After you run the code, the evolution of the policy typically looks like this:

```
Evolution of P(B)

 1.0 ┤
     │                    ╱━━━━━━━━━━━━━━━━━━  ← converges: stabilizes around 0.85-0.95
 0.9 ┤                ╱━╱
     │            ╱╱╱╱╱
 0.8 ┤        ╱╱╱╱
     │     ╱╱╱╱              ← climbs: discovers B yields reward more often
 0.7 ┤  ╱╱╱╱
     │╱╱╱╱
 0.6 ┤╲╱
     │ ╲ ╱╲  ╱
 0.5 ┤─╲╱╲╱╲╱╲────────────  ← early: fluctuates near 0.5 ("try both")
     └────────────────────────────────────────
     0    50   100  150  200  250  300
                  Episode
```

Three phases are easy to see. Early on, the policy is close to uniform ("try both," exploration-dominated). Then the probability of choosing B climbs ("B seems to give reward more often"). Finally it stabilizes in a high-probability region ("just pull B").

But you probably also noticed the curve does not rise smoothly. It is jagged, with obvious oscillations. This is the central pain point of policy gradients:

**high variance.**

## 6.3.4 Gradient Noise And Training Oscillations

Policy-gradient updates are driven by samples. At each update, the network only sees the single outcome it happened to sample in that round:

- Sometimes you pull B but still lose (30% probability). The network receives a `reward = 0` signal and does not reinforce B, even though B is the better choice on average.
- Sometimes you pull A and win (30% probability). The network receives a `reward = 1` signal and reinforces A, even though A is the worse choice on average.

These "bad-luck" samples make the gradient estimator noisy, and the policy can wobble back and forth even while drifting in the correct direction overall. It is like walking blindfolded: you move roughly the right way, but each step is crooked.

If you change the learning rate from `0.01` to `0.1`, you may see the policy swing dramatically between A and B: one lucky win on A pushes the policy heavily toward A; the next lucky win on B pushes it back. The policy fails to settle. This is like steering a car with an overly sensitive wheel: every correction overshoots, and you keep swaying around the target.

## 6.3.5 The Exploration-Exploitation Tradeoff

The oscillations also reveal the central tension in reinforcement learning: exploration vs exploitation.

- **Early training:** the policy is near-uniform (exploration) ("try both").
- **Late training:** the policy becomes close to deterministic (exploitation) ("just pick B").
- **The transition must be smooth:** converge too fast and you may lock into a suboptimal choice; converge too slowly and you waste samples.

This is close to how humans learn. When you first learn to cook, you try many recipes (exploration). Once you discover a dish you really like, you make it repeatedly (exploitation). But if you lock in too early, you might miss something even better.

> In Chapter 8's [PPO](../chapter10_ppo/ppo-clip-objective), an entropy bonus is a mechanism that forces the policy to keep exploring. It adds a term to the loss that prevents the policy from becoming "too certain" too early.

<details>
<summary>Thinking: If B only wins with probability 55% (instead of 70%), can the policy still learn?</summary>

Yes, but it learns more slowly and oscillates more. The gap between A and B is smaller (55% vs 30%), so misleading samples are more common. This again highlights the core difficulty of policy gradients: when the difference between "good" and "bad" is small, high variance can make learning extremely hard. This is why we will later introduce baselines to reduce variance.

</details>

This experiment shows that policy gradients can increase the probability of the better action. Section 6.1, [the REINFORCE algorithm](./reinforce), explains why the update `-log_prob * reward` has this effect.

## Section Summary

- A policy network can represent a probability distribution even when the environment has no state.
- REINFORCE increases the probability of sampled actions that receive positive reward.
- Individual rewards are noisy, so action probability can fluctuate even while its average trend improves.
