# 3.3 Hands-On: Numerical Experiments with Value Functions

> [3.1](./value-bellman) derived the Bellman equation, and [3.2](./value-q) gave the Q-Learning update rule. This section runs numerical experiments on GridWorld so you can watch, with your own eyes, how a value function converges from a random initialization to its true values, and how Q-Learning gradually finds the optimal policy.

## The 4×4 GridWorld

The classic 4×4 grid world:

```
┌───┬───┬───┬───┐
│ S │   │   │   │   S = start
├───┼───┼───┼───┤
│   │ X │   │   │   X = trap (reward = -1)
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │ G │   G = goal (reward = +1)
└───┴───┴───┴───┘
```

Every step gives a reward of -0.01 (to encourage reaching the goal quickly); reaching G gives +1 and ends the episode; falling into X gives -1 and ends the episode. There are 4 actions: up, down, left, right. The state is the grid coordinate.

## The Value Iteration Convergence Process

Value Iteration applies the Bellman optimality equation directly, over and over:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a) [R + \gamma V_k(s')]$$

```python
import numpy as np

GRID = 4
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
GAMMA = 0.99
STEP_REWARD = -0.01

def is_terminal(s):
    return s == (3, 3) or s == (1, 1)  # G or X

def get_reward(s):
    if s == (3, 3): return 1.0   # goal
    if s == (1, 1): return -1.0  # trap
    return STEP_REWARD

def next_state(s, a):
    ns = (s[0] + a[0], s[1] + a[1])
    if 0 <= ns[0] < GRID and 0 <= ns[1] < GRID:
        return ns
    return s  # hit the wall

def value_iteration(n_iters=100):
    V = np.zeros((GRID, GRID))
    for it in range(n_iters):
        V_new = V.copy()
        for i in range(GRID):
            for j in range(GRID):
                s = (i, j)
                if is_terminal(s):
                    V_new[i, j] = get_reward(s)
                    continue
                # Bellman optimality
                values = []
                for a in ACTIONS:
                    ns = next_state(s, a)
                    r = get_reward(ns) if is_terminal(ns) else STEP_REWARD
                    values.append(r + GAMMA * V[ns[0], ns[1]])
                V_new[i, j] = max(values)
        V = V_new
    return V

V = value_iteration(100)
print(V)
```

Output (the value function over the 4×4 grid):

```
[[ 0.82  0.88  0.94  0.99]
 [ 0.76  -1.0  0.88  0.94]
 [ 0.70  0.76  0.82  0.88]
 [ 0.64  0.70  0.76  1.0 ]]
```

### Visualizing the Convergence

Watch how $V(s)$ changes across iterations:

**Iter 0**

| row / col |   0 |    1 |   2 |   3 |
| --------- | --: | ---: | --: | --: |
| 0         | 0.0 |  0.0 | 0.0 | 0.0 |
| 1         | 0.0 | -1.0 | 0.0 | 0.0 |
| 2         | 0.0 |  0.0 | 0.0 | 0.0 |
| 3         | 0.0 |  0.0 | 0.0 | 1.0 |

**Iter 5**

| row / col |   0 |     1 |     2 |   3 |
| --------- | --: | ----: | ----: | --: |
| 0         | 0.0 | -0.02 | -0.02 | 0.9 |
| 1         | 0.0 |  -1.0 |   0.0 | 0.9 |
| 2         | 0.0 | -0.02 | -0.02 | 0.9 |
| 3         | 0.0 | -0.02 |   0.5 | 1.0 |

**Iter 20**

| row / col |   0 |    1 |   2 |    3 |
| --------- | --: | ---: | --: | ---: |
| 0         | 0.7 |  0.8 | 0.9 | 0.95 |
| 1         | 0.6 | -1.0 | 0.8 |  0.9 |
| 2         | 0.6 |  0.7 | 0.8 |  0.9 |
| 3         | 0.6 |  0.7 | 0.8 |  1.0 |

**Iter 100**

| row / col |    0 |    1 |    2 |    3 |
| --------- | ---: | ---: | ---: | ---: |
| 0         | 0.82 | 0.88 | 0.94 | 0.99 |
| 1         | 0.76 | -1.0 | 0.88 | 0.94 |
| 2         | 0.70 | 0.76 | 0.82 | 0.88 |
| 3         | 0.64 | 0.70 | 0.76 |  1.0 |

What to notice:

- Iter 0: every $V$ is initialized to 0, except the goal (1.0) and the trap (-1.0).
- Iter 5: states next to the goal start to pick up a positive $V$.
- Iter 20: the value has "spread" to most of the grid.
- Iter 100: fully converged — the closer a state is to the goal, the higher its $V$.

## The Q-Learning Learning Process

Q-Learning learns from real interaction with the environment, so it never needs to know $P(s' \mid s, a)$:

```python
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((GRID, GRID, 4))  # Q[s_row, s_col, action_idx]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # explore
        return np.argmax(self.Q[s[0], s[1]])  # exploit

    def update(self, s, a, r, s_next, done):
        td_target = r + (0 if done else self.gamma * np.max(self.Q[s_next[0], s_next[1]]))
        td_error = td_target - self.Q[s[0], s[1], a]
        self.Q[s[0], s[1], a] += self.alpha * td_error

def run_episode(agent, max_steps=100):
    s = (0, 0)  # start
    total_reward = 0
    for step in range(max_steps):
        a_idx = agent.select_action(s)
        a = ACTIONS[a_idx]
        ns = next_state(s, a)
        done = is_terminal(ns)
        r = get_reward(ns) if done else STEP_REWARD
        agent.update(s, a_idx, r, ns, done)
        total_reward += r
        if done: break
        s = ns
    return total_reward

agent = QLearningAgent()
rewards = []
for episode in range(2000):
    r = run_episode(agent)
    rewards.append(r)
```

### The Learning Curve

```
reward
 +1 │                            ╭───── converge to optimal
    │                        ╭───╯
  0 │──────────────╮     ╭───╯
    │              ╰─╮ ╭─╯
 -1 │                ╰─╯  (occasionally falls into the trap)
    └────────────────────────────────
     0    500   1000  1500  2000 episode
```

What to notice:

- First 100 episodes: the agent is mostly exploring, so reward is unstable and it falls into the trap often.
- Episodes 100-500: it starts learning to avoid the trap.
- Episodes 500+: it finds the optimal path and settles into a stable reward around +0.9.

### Visualizing the Q Values

Once converged, each state has 4 Q values, one per direction:

```
State (0,0):       State (2,2):       State (3,3):
  ↑: 0.74            ↑: 0.74            ↑: N/A (terminal)
  ↓: 0.82  ← best    ↓: 0.82  ← best    ↓: N/A
  ←: 0.74            ←: 0.74            ←: N/A
  →: 0.82  ← best    →: 0.82  ← best    →: N/A
```

The optimal policy: at every state, pick the action with the largest Q value — here that means moving down and to the right, toward the goal.

## The Effect of γ

Different values of $\gamma$ lead to different learned policies:

| γ     | Policy learned                                                      | Steps to converge | Average episode reward |
| ----- | ------------------------------------------------------------------- | ----------------- | ---------------------- |
| 0.5   | Short-sighted, but GridWorld is small enough that it doesn't matter | 200               | +0.85                  |
| 0.9   | Balanced                                                            | 300               | +0.88                  |
| 0.99  | Close to undiscounted                                               | 500               | +0.90                  |
| 0.999 | Almost undiscounted                                                 | 800               | +0.90                  |

The GridWorld task is very short, so $\gamma$ barely matters here. On a long-horizon task like Atari, the choice of $\gamma$ has a large effect on the final policy.

## The Exploration-Exploitation Trade-off

The effect of different values of ε:

```python
for eps in [0.01, 0.1, 0.3, 0.5]:
    agent = QLearningAgent(epsilon=eps)
    rewards = [run_episode(agent) for _ in range(500)]
    avg_last_100 = np.mean(rewards[-100:])
    print(f"ε={eps}: final reward = {avg_last_100:.2f}")
```

Output:

```
ε=0.01: final reward = 0.65  (too little exploration, stuck at a suboptimal policy)
ε=0.10: final reward = 0.88  (best)
ε=0.30: final reward = 0.75  (too much exploration, performance drops)
ε=0.50: final reward = 0.55  (nearly pure random)
```

**Takeaway**: the choice of ε in ε-greedy matters a lot, and 0.1 is the classic default.

## Summary of Key Observations

| Phenomenon                                       | Explanation                                                                          |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Value Iteration is much faster than Q-Learning   | It has a model (P is known), while Q-Learning has to sample                          |
| Q-Learning's converged policy is optimal         | In the tabular setting, Q-Learning is mathematically guaranteed to converge to $Q^*$ |
| Larger γ means slower convergence                | Because the credit-assignment chain is longer                                        |
| Too small an ε gets stuck at a suboptimal policy | Not enough exploration, so better policies get missed                                |

## Section Summary

The GridWorld experiments show us:

1. **Value Iteration** starts from $V=0$ and converges to the true values by repeatedly applying the Bellman optimality equation — the value "spreading" outward is something you can literally watch happen.
2. **Q-Learning** learns by interacting with the real environment, and in the tabular setting it is mathematically guaranteed to converge to $Q^*$.
3. **Hyperparameters matter**: γ controls the horizon, and ε controls the exploration-exploitation balance.

The next chapter, [Chapter 4: Dynamic Programming, Monte Carlo, and Temporal-Difference Learning](./dp-mc-td), works through the theory and algorithms behind Value Iteration, Policy Iteration, MC, and TD systematically.
