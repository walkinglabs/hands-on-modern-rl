---
title: '1.3 Hands-On: PPO Training and Visualization'
---

# 1.3 Hands-On: PPO Training and Visualization

> [1.1](./principles) covered CartPole's physical model and the core idea behind the PPO algorithm. This section is hands-on: run a complete PPO training loop, understand what each hyperparameter does, and use visualization tools to analyze the training process.

## Training Pipeline Overview

The full pipeline for training PPO on CartPole:

```
┌──────────────────────────────────────────────┐
│ 1. Initialize: policy network π_θ(a|s) with  │
│    random weights                            │
├──────────────────────────────────────────────┤
│ 2. Rollout: run N trajectories with the      │
│    current π_θ, collecting (s_t, a_t, r_t,   │
│    s_{t+1}, done)                            │
├──────────────────────────────────────────────┤
│ 3. Compute advantage Â_t (using GAE)         │
├──────────────────────────────────────────────┤
│ 4. PPO update: maximize the clipped          │
│    objective L = E[min(r_t Â_t, clip(r_t,    │
│    1±ε) Â_t)]                                │
├──────────────────────────────────────────────┤
│ 5. Repeat steps 2-4 until convergence        │
│    (reward reaches 500)                      │
└──────────────────────────────────────────────┘
```

## Full Training Code

Below is a minimal PPO + CartPole implementation. The complete, runnable code lives in `code/chapter01_cartpole/train_ppo.py`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

# === Policy network: state → action probabilities ===
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, s):
        logits = self.net(s)
        return torch.distributions.Categorical(logits=logits)

# === Value network: state → V(s) ===
class ValueNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)

# === GAE: Generalized Advantage Estimation ===
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    next_value = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages

# === Main training loop ===
def train_ppo(env_name='CartPole-v1', n_iters=200, n_steps=2048,
              gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4,
              n_epochs=10, batch_size=64):
    env = gym.make(env_name)
    policy = PolicyNetwork()
    value_fn = ValueNetwork()
    optimizer = optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=lr)

    reward_history = deque(maxlen=20)

    for iter in range(n_iters):
        # === 1. Rollout ===
        states, actions, rewards, dones, log_probs_old, values = [], [], [], [], [], []
        s, _ = env.reset()
        ep_reward = 0

        for step in range(n_steps):
            s_tensor = torch.FloatTensor(s)
            dist = policy(s_tensor)
            v = value_fn(s_tensor)
            a = dist.sample()

            s_next, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated

            states.append(s); actions.append(a.item()); rewards.append(r)
            dones.append(done); log_probs_old.append(dist.log_prob(a).item()); values.append(v.item())
            ep_reward += r

            if done:
                reward_history.append(ep_reward)
                ep_reward = 0
                s, _ = env.reset()
            else:
                s = s_next

        # === 2. Compute advantage ===
        advantages = compute_gae(rewards, values, gamma, lam)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + torch.FloatTensor(values)

        # === 3. PPO update (multiple epochs) ===
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        log_probs_old_t = torch.FloatTensor(log_probs_old)

        for epoch in range(n_epochs):
            idx = torch.randperm(len(states_t))
            for start in range(0, len(states_t), batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = log_probs_old_t[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                dist = policy(mb_states)
                new_lp = dist.log_prob(mb_actions)
                ratio = (new_lp - mb_old_lp).exp()

                # PPO Clip
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_pred = value_fn(mb_states)
                value_loss = ((v_pred - mb_ret) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if iter % 10 == 0:
            avg_r = np.mean(reward_history) if reward_history else 0
            print(f"Iter {iter}: avg_reward = {avg_r:.1f} / 500")

    return policy

if __name__ == '__main__':
    policy = train_ppo()
```

## Training Curves and Visualization

Run the code above, and the typical training curve looks like this:

```
reward
 500 │                              ╭───── converged
 400 │                          ╭───╯
 300 │                      ╭───╯
 200 │                  ╭───╯
 100 │              ╭───╯
     0 │───────────╯
       └─────────────────────────────────
        0    50   100   150   200 iterations
```

Watch for four distinct phases:

| Phase       | Iteration | Average reward | What's happening                                                                            |
| ----------- | --------- | -------------- | ------------------------------------------------------------------------------------------- |
| Exploration | 0-20      | 10-30          | The agent balances the pole essentially at random; almost no learning signal comes through. |
| Learning    | 20-100    | 30-200         | Reward climbs quickly as the policy starts to stabilize.                                    |
| Convergence | 100-150   | 200-450        | Progress slows but improvement continues steadily.                                          |
| Solved      | 150+      | 475+           | Gymnasium defines "solved" as an average of ≥ 475 over the most recent 100 episodes.        |

## The Effect of Hyperparameters

PPO on CartPole is not very sensitive to hyperparameters, but it still pays to understand what each one does:

| Hyperparameter | Default | Effect                                                                                                      |
| -------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `lr`           | 3e-4    | Too high (>1e-3) and training collapses; too low (<1e-5) and training crawls.                               |
| `clip_eps`     | 0.2     | Larger means a more aggressive update (closer to vanilla policy gradient); smaller means more conservative. |
| `gamma`        | 0.99    | The discount factor; at 0.99 on CartPole it's nearly equivalent to no discounting at all.                   |
| `lam` (GAE)    | 0.95    | Larger moves the estimate closer to Monte Carlo; smaller moves it closer to TD.                             |
| `n_epochs`     | 10      | How many times each rollout's data gets reused; too large and the policy overfits to it.                    |
| `n_steps`      | 2048    | Rollout length; for a short task like CartPole this can be cut down to 512.                                 |

### Debugging Failure Modes

If training fails to converge, check these in order:

1. **Has policy entropy collapsed to zero?** If action probabilities collapse to 1.0/0.0 too early, add an entropy bonus: `loss += -0.01 * dist.entropy().mean()`
2. **Is the advantage normalized correctly?** Skipping normalization leaves the gradient scale unstable.
3. **Has the value loss exploded?** Check the returns for outliers.
4. **Is the learning rate too large?** Retry with `lr=1e-4`.

## Visualizing with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/cartpole_ppo')

for iter in range(n_iters):
    # ... training code ...

    writer.add_scalar('train/reward_mean', avg_r, iter)
    writer.add_scalar('train/policy_loss', policy_loss.item(), iter)
    writer.add_scalar('train/value_loss', value_loss.item(), iter)
    writer.add_scalar('train/entropy', dist.entropy().mean().item(), iter)
    writer.add_scalar('train/clip_frac', clip_fraction, iter)
```

Launch TensorBoard:

```bash
tensorboard --logdir=runs
```

Watch five key metrics:

- **reward_mean**: the core training metric; it should climb monotonically toward 500.
- **policy_loss**: the loss after the PPO clip; oscillation here is normal.
- **value_loss**: should decline steadily.
- **entropy**: the policy's entropy, which should drift down slowly from about 0.69 (the initial ln 2) to about 0.1.
- **clip_frac**: the fraction of samples that got clipped; it should stay stable around 0.1-0.3 — above 0.5 means the policy is changing too fast.

## Comparing Experiments with Plotting Tools

```python
import matplotlib.pyplot as plt

def plot_experiments(results):
    """results: dict of name -> list of rewards"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rewards in results.items():
        # moving average
        smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
        ax.plot(smoothed, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward (20-episode mean)')
    ax.set_title('PPO on CartPole: Hyperparameter Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=475, color='r', linestyle='--', label='Solved threshold')
    plt.savefig('cartpole_ppo_sweep.png', dpi=120, bbox_inches='tight')

# compare different hyperparameters
results = {
    'lr=3e-4 (default)': run_experiment(lr=3e-4),
    'lr=1e-4 (slow)': run_experiment(lr=1e-4),
    'lr=1e-3 (fast)': run_experiment(lr=1e-3),
    'clip=0.1 (conservative)': run_experiment(clip_eps=0.1),
    'clip=0.3 (aggressive)': run_experiment(clip_eps=0.3),
}
plot_experiments(results)
```

Expected results:

- `lr=3e-4` + `clip=0.2` (default): converges in about 150 iterations.
- `lr=1e-4`: converges in about 300 iterations, but more stably.
- `lr=1e-3`: about a 50% chance of failing to converge at all.
- `clip=0.1`: converges slowly but stably.
- `clip=0.3`: converges quickly but with more oscillation.

## Section Summary

Training PPO on CartPole is RL's classic "hello world." This section walked through a complete, runnable PPO implementation, covering rollout collection, GAE advantage estimation, the PPO clipped update, hyperparameter tuning, and TensorBoard visualization end to end.

Key takeaways:

1. **PPO is a three-step loop**: rollout, GAE, and clipped update.
2. **Low sensitivity to hyperparameters** is a big part of why PPO caught on — the defaults work almost every time on CartPole.
3. **Visualizing training** is the central tool for debugging RL code — auxiliary metrics like entropy and clip_frac often surface problems before the reward curve does.

The next chapter, [The Basic Formulation of Reinforcement Learning](../chapter03_mdp/intro), pulls the perspective back to RL's simplest form — the stateless, immediate-reward multi-armed bandit — to study exploration and exploitation on their own, before state transitions and long-horizon return enter the picture.
