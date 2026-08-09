# 9.3 Model-Based RL: Dyna, PETS, MBPO

> [11.2](./td3-sac) got model-free algorithms for continuous control to a stable, usable place -- SAC and TD3 can train for a million steps on MuJoCo and come out with a good policy. But a million steps is a burden a real robot cannot bear: motor wear, battery life, and safety constraints make every sample from a real environment extremely expensive. **Model-Based RL** starts from a different premise: **learn a model of the environment** $\hat{P}(s' \mid s, a), \hat{R}(s, a)$, train the policy against that model, and bring sample efficiency down from millions of steps to tens of thousands.

## The Fundamental Difference Between Model-Based and Model-Free

Every algorithm we've covered so far (DDPG/TD3/SAC) is **model-free** -- the agent never tries to understand the environment, it only learns a policy from the reward the environment hands back. **Model-Based RL** takes the opposite approach: first learn a model of the environment $\hat{P}(s' \mid s, a), \hat{R}(s, a)$, then use that model for planning or for generating data.

### Why Bother With a Model?

The motivation is **sample efficiency**. MuJoCo's physics simulator is cheap to query, but real robot experiments are **expensive per sample** -- a robot arm can be damaged, batteries drain, parts wear out. Model-free methods need millions of steps to train a good policy, which a real robot simply cannot absorb. Model-based methods need only tens of thousands of steps, because once the model is learned, you can sample from it "inside the model" as many times as you like.

### Three Paradigms at a Glance

| Paradigm | Core Idea                                     | Representative Algorithm | Best Suited For                                         |
| -------- | --------------------------------------------- | ------------------------ | ------------------------------------------------------- |
| **Dyna** | Model as data augmentation                    | Dyna-Q                   | discrete actions, fast training                         |
| **PETS** | Probabilistic ensembles + trajectory sampling | PETS                     | high-precision control, where model uncertainty matters |
| **MBPO** | Short-horizon rollouts                        | MBPO                     | general-purpose continuous control                      |

Let's take these apart one at a time.

## The Model as Data Augmentation

This is Sutton's classic 1990 idea. Dyna splits every real interaction into four steps; the third step trains the model, and the fourth uses the model to generate "fake" data that speeds up model-free training:

```python
for step in range(total_steps):
    # 1. Real interaction
    a = policy.select(s)
    s_prime, r = env.step(a)
    replay_buffer.add(s, a, r, s_prime)

    # 2. Update the model-free algorithm (e.g. Q-Learning) with real data
    q_learning_update(replay_buffer.sample())

    # 3. Train the environment model with real data
    model.train(s, a, r, s_prime)

    # 4. Generate "fake" data from the model, and do N more Q-Learning updates
    for _ in range(N):  # N = 10-100
        s_sim, a_sim = replay_buffer.sample_state_action()
        s_sim_next, r_sim = model.predict(s_sim, a_sim)
        q_learning_update(s_sim, a_sim, r_sim, s_sim_next)
```

Dyna treats the model as an "extra data generator": after every real interaction, it performs $N$ simulated updates, which roughly **multiplies sample efficiency by $N$**.

### Dyna's Key Limitation

Dyna assumes the model is deterministic -- it predicts $s'$ directly from $(s, a)$. That's fine in discrete environments like GridWorld, but in continuous physical environments like MuJoCo, model error accumulates:

$$\|s_T^{\text{predicted}} - s_T^{\text{true}}\| \sim \mathcal{O}(\epsilon^T)$$

Here $\epsilon$ is the single-step prediction error. With $\epsilon = 0.1$ and $T = 10$, the predicted error reaches $10^{10}$ -- completely unusable. This is exactly why the work that followed (PETS, MBPO) is all about answering one question: how do you quantify model error?

## Probabilistic Ensembles with Trajectory Sampling

The key observation behind Probabilistic Ensembles with Trajectory Sampling (Chua et al., 2018) is that a learned model carries **two distinct kinds of uncertainty**:

- **Epistemic uncertainty**: the model itself is uncertain because training data is limited -- captured with an **ensemble** $M_1, \ldots, M_K$
- **Aleatoric uncertainty**: the environment itself is stochastic (think of a die roll) -- captured with a **probabilistic output** $p(s' \mid s, a)$

### Model Architecture

PETS's model is an ensemble of $K$ probabilistic neural networks:

```python
class PEtsModel:
    def __init__(self, n_models=5):
        self.models = [ProbabilisticNN() for _ in range(n_models)]

    def predict(self, s, a):
        # Each model outputs (mean, var)
        means, vars = [], []
        for m in self.models:
            mu, sigma = m(s, a)
            means.append(mu); vars.append(sigma)
        return means, vars  # spread across the ensemble = epistemic uncertainty
```

During planning, PETS doesn't rely on a single model -- it samples randomly from the ensemble, so the policy stays robust to the possibility that the model is wrong.

### The Trajectory Sampling Strategy

PETS plans with **CEM** (the Cross-Entropy Method): at each step, it samples candidate action sequences $\{a_1, \ldots, a_H\}$ and selects among them:

```python
def cem_planning(model, s, horizon=10, n_samples=500, n_iters=5):
    # Initialize the action distribution
    action_mean = zeros(horizon, action_dim)
    action_var = ones(horizon, action_dim)

    for it in range(n_iters):
        # 1. Sample N action sequences
        action_seqs = sample_normal(action_mean, action_var, n_samples)

        # 2. Roll each sequence out with the model, using a randomly chosen ensemble member per sequence
        rewards = []
        for seq in action_seqs:
            model_id = random_int(0, K)
            s_pred = s
            total_r = 0
            for a in seq:
                s_pred, r = model[model_id].predict(s_pred, a)
                total_r += r
            rewards.append(total_r)

        # 3. Keep the top 20% of sequences and update the distribution
        elite = top_k_indices(rewards, k=0.2 * n_samples)
        action_mean = action_seqs[elite].mean(0)
        action_var = action_seqs[elite].var(0)

    return action_mean[0]  # execute only the first step (this is the MPC idea)
```

### PETS's Experimental Results

PETS was the first model-based method to match model-free performance on MuJoCo, while cutting the number of sampled steps by **10-50x**. The cost is that **planning is computationally expensive** -- every step requires 500 model rollouts.

## Model-Based Policy Optimization

The core innovation of Model-Based Policy Optimization (Janner et al., 2019) is to **generate rollouts of bounded length from the model** (for example, 5 steps), then switch back to the real environment. This sidesteps the explosive accumulation of model error over long rollouts.

### Short-Horizon Rollouts

The key parameter in MBPO is the rollout length $k$. The paper proves that when the single-step model error is $\epsilon$, the accumulated error over a $k$-step rollout is bounded by $k\epsilon$ -- controllable.

```python
# Short-horizon rollout keeps model error under control
for rollout_step in range(K_short):  # K_short = 5
    a = policy(s_sim)
    s_sim, r = model.predict(s_sim, a)
    replay_buffer.add(s_sim, a, r, s_sim)
    # Key point: every 5 steps we must "reset" to a real state
    if rollout_step % K_short == 0:
        s_sim = real_env.state
```

### The MBPO Training Loop

```
┌─────────────────────────────────────────────┐
│ 1. Train model M on real data               │
│    M.predict(s, a) → s', r                  │
├─────────────────────────────────────────────┤
│ 2. Generate short rollouts (5 steps) with M │
│    Start from some s in the real data       │
│    Each step: a = policy(s), s' = M(s, a)   │
│    Result: (s, a, r, s') × 5 added to replay│
├─────────────────────────────────────────────┤
│ 3. Run SAC updates on the replay buffer     │
│    (a mix of real and simulated data)       │
└─────────────────────────────────────────────┘
```

MBPO matches model-free SAC's performance on MuJoCo while cutting the number of sampled steps by **10-100x**.

### Comparing the Three Model-Based Algorithms

| Algorithm | Model Type             | Planning Method  | Sample Efficiency | Compute Cost |
| --------- | ---------------------- | ---------------- | ----------------- | ------------ |
| Dyna      | deterministic          | 1-step fake data | ~10x              | low          |
| PETS      | probabilistic ensemble | CEM MPC          | ~50x              | high         |
| MBPO      | deterministic          | short rollout    | ~100x             | medium       |

In practice:

- **Fast experimentation**: Dyna (simple and stable)
- **High-precision control**: PETS (robotic manipulation, precision machining)
- **General-purpose continuous control**: MBPO (the full range of MuJoCo environments)

## Section Summary

Model-Based RL improves sample efficiency by **learning a model of the environment**:

1. **Dyna** uses the model as data augmentation, performing N simulated updates after every real interaction
2. **PETS** represents model uncertainty with a probabilistic ensemble and stays robust through CEM planning
3. **MBPO** avoids error accumulation with short-horizon rollouts, matching SAC's performance with 100x fewer samples

The next section, [11.4 AlphaZero, MuZero, and Dreamer V3](./search-world-models), turns to the other flagship line of model-based RL -- explicit search combined with neural value estimation, tracing the arc from AlphaGo to Dreamer V3.
