# 2.3 Policy, Value, and Return

> [2.2](./mdp) defined the MDP tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. But the MDP itself is just the "environment" — how does the agent actually make decisions inside it? This section introduces three core concepts: the **policy** (how the agent picks actions), the **return** (how we measure the quality of a trajectory), and the **value function** (how we evaluate the long-term payoff of a state or action). These three concepts are the foundation for every RL algorithm that follows.

## Policies and Decision Rules

A **policy** is the agent's mapping from states to actions. It comes in two flavors:

- **Deterministic policy**: $\pi: \mathcal{S} \to \mathcal{A}$, which takes a state and outputs an action directly, $a = \pi(s)$
- **Stochastic policy**: $\pi: \mathcal{S} \to \Delta(\mathcal{A})$, which takes a state and outputs a distribution over actions, $a \sim \pi(\cdot \mid s)$

The stochastic policy is the more general of the two — a deterministic policy is just the special case where the distribution collapses onto a single point. RL almost always works with stochastic policies.

```python
# A simple stochastic policy for CartPole
import torch
import torch.nn as nn

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 2)  # logits for 2 actions
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
```

### The Optimal Policy

The goal of RL is to find the **optimal policy** $\pi^*$, the one that maximizes long-run cumulative reward:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

Every algorithm covered later in this book — DQN, PPO, SAC — is, at bottom, an approximate solver for this optimization problem.

## Return: Measuring a Trajectory

Over the course of an episode, the agent lives through a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$. The **return** is the cumulative reward from time $t$ onward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### What the Discount Factor γ Does

$\gamma \in [0, 1]$ is the **discount factor**: it makes rewards further in the future count for less. It serves three purposes:

1. **Guarantees convergence mathematically**: the infinite sum $\sum \gamma^k R$ converges whenever $|\gamma| < 1$
2. **Reflects uncertainty**: rewards far in the future are inherently harder to predict, so they should be weighted less
3. **Stabilizes training**: it keeps delayed rewards from blowing up the variance of the return

| γ value | Meaning                                  | Typical use                          |
| ------- | ---------------------------------------- | ------------------------------------ |
| 0       | Only the immediate step matters (greedy) | Rarely used                          |
| 0.9     | Short horizon (~10 steps)                | Board games, recommender systems     |
| 0.99    | Medium horizon (~100 steps)              | Atari, CartPole                      |
| 0.999   | Long horizon (~1000 steps)               | Long-horizon tasks, robot navigation |
| 1.0     | No discounting                           | Finite-horizon tasks                 |

### Return in CartPole

In CartPole, every step gives a reward of 1 as long as the pole is still up. The episode ends when the pole falls or the cart goes out of bounds. The return is:

$$G_0 = 1 + \gamma + \gamma^2 + \cdots + \gamma^{T-1} = \frac{1 - \gamma^T}{1 - \gamma}$$

where $T$ is the episode length. With $\gamma = 0.99$ and $T = 500$, $G_0 \approx 99$.

## Value Functions: Long-Term Payoff

A **value function** measures how much return you can expect to collect from a given state or action while following policy $\pi$. There are two kinds.

### State Value V(s)

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

In words: starting from state $s$ and following policy $\pi$ from there, this is the expected cumulative return.

### Action Value Q(s, a)

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

In words: starting from state $s$, **taking action $a$ first, then following $\pi$** afterward, this is the expected cumulative return.

### How V and Q Relate

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

$V^\pi(s)$ is just the expectation of $Q^\pi(s, a)$ under the action distribution that $\pi$ prescribes.

### A Numerical Example: GridWorld

Consider a 4×4 GridWorld where the goal sits in the bottom-right corner (reward = +1, episode ends there) and every other step gives reward 0:

```
┌───┬───┬───┬───┐
│0.0│0.5│0.8│0.9│   ← V(s) values
├───┼───┼───┼───┤
│0.5│0.7│0.9│1.0│ ★ (goal)
├───┼───┼───┼───┤
│0.7│0.9│0.95│  │
├───┼───┼───┼───┤
│0.8│0.95│ │  │   ← unlabeled cells have lower V
└───┴───┴───┴───┘
```

The closer a cell is to the goal, the higher its V — the reward is fewer steps away, so it gets discounted less.

## The Advantage Function: Judging Actions

The **advantage function** measures how much better action $a$ is than the average action at state $s$:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A > 0$: action $a$ beats the average
- $A < 0$: action $a$ falls short of the average
- $A = 0$: action $a$ is exactly average

The advantage function is central to policy gradient methods ([Chapter 6](../chapter08_policy_gradient/policy-gradient)) and Actor-Critic methods ([Chapter 7](../chapter09_actor_critic/actor-critic)).

## A Preview of the Bellman Equation

Value functions satisfy the **Bellman equation** — a recursive relationship that writes $V(s)$ as a function of $V(s')$:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

This equation sits at the core of every RL algorithm. The next section, [2.4 Discounting, Trajectories, and POMDPs](./panorama), works out more of the details of trajectories, and [Chapter 3, Value Functions and the Bellman Equation](./value-bellman) will dig into the Bellman equation in full.

## Section Summary

Policy, return, and value function are the three core concepts of the MDP:

1. **Policy $\pi$**: the agent's decision rule; the stochastic form $a \sim \pi(\cdot \mid s)$ is the most general version
2. **Return $G_t$**: the discounted cumulative reward from time $t$ onward, $\sum \gamma^k r$
3. **Value functions**: $V^\pi(s)$ for state value, $Q^\pi(s, a)$ for action value; the advantage $A = Q - V$ measures relative quality

The next section, [3.3 Discounting, Trajectories, and POMDPs](./panorama), formalizes trajectories and introduces the POMDP (partially observable MDP) extension.
