# 9.4 AlphaZero, MuZero, and Dreamer V3

> [11.3](./model-based) covered the "data augmentation" line of model-based RL — Dyna/PETS/MBPO use a model to generate data that speeds up model-free training. This section covers the other flagship line of model-based RL: **explicit search + neural network value estimation**. From AlphaGo (2016) to AlphaZero (2017) to MuZero (2019) to Dreamer V3 (2023), this line represents the theoretical ceiling of model-based RL, and it directly inspired Process Reward Model search in the LLM era.

## AlphaZero and the Extreme of Search + Learning

AlphaGo (2016) → AlphaGo Zero (2017) → AlphaZero (2017) → MuZero (2019) traces out another philosophy of model-based RL: **explicit search + neural network value estimation**.

### AlphaZero's Core Loop

```python
def alphazero_search(state, neural_net, n_simulations=800):
    root = MCTSNode(state)
    for _ in range(n_simulations):
        # 1. Selection: pick the best child by PUCT
        node = root
        while not node.is_leaf():
            node = node.select_child()

        # 2. Expansion: evaluate the leaf with the neural net
        policy, value = neural_net(node.state)
        node.expand(policy)

        # 3. Backup: propagate the value back to the root
        node.backup(value)

    # return the root's visit-count distribution as the action probabilities
    return root.compute_action_distribution()
```

AlphaZero combines MCTS (Monte Carlo Tree Search) with a neural network:

- **Policy network** $p_\theta(a \mid s)$: narrows the search width (only search promising actions)
- **Value network** $v_\theta(s)$: shortens the search depth (estimate the leaf's value directly instead of searching to the end of the game)

### The PUCT Formula

AlphaZero uses PUCT (Predictor + UCB) to select child nodes:

$$\text{PUCT}(a) = Q(s, a) + c_{\text{prior}} \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

- $Q(s, a)$: the current value estimate for action $a$
- $p_\theta(a \mid s)$: the policy network's prior
- $\sqrt{N(s)} / (1 + N(s, a))$: the exploration bonus (the UCB idea)

The first term exploits current knowledge, the second term uses the prior to narrow the search, and the third term makes sure every action gets tried.

### Self-Play Training

The two networks are trained through **self-play**:

1. Play one game against yourself using the current network + MCTS
2. The search result becomes a "better policy label" — the action distribution MCTS produces is an improved policy
3. The game outcome becomes a "better value label" — a win is +1, a loss is -1
4. Use these labels to train the network with supervision

```python
def self_play_training(network, n_games=10000):
    for game in range(n_games):
        # 1. Self-play
        trajectory = []
        state = initial_state()
        while not state.is_terminal():
            policy = alphazero_search(state, network)
            action = sample_from(policy)
            trajectory.append((state, policy, action))
            state = state.next(action)

        # 2. Label with the outcome
        winner = state.winner()
        for s, p, a in trajectory:
            value = +1 if winner == s.current_player else -1
            train_network(s, p, value)
```

**No human game records needed** — starting from scratch, AlphaZero beat Stockfish in 4 hours and surpassed every human Go program in 72 hours.

## MuZero and Implicit Model Learning

AlphaZero needs to know the rules of the game (state transitions, legal actions). MuZero's (Schrittwieser et al. 2019) key innovation: **learn an implicit model** that maps a state $s$ to a hidden representation $h(s)$, and do planning and value estimation entirely in that hidden space.

$$s \xrightarrow{h} x_0 \xrightarrow{g} x_1 \xrightarrow{g} x_2 \to \ldots$$

### MuZero's Three Networks

- **Representation network** $h(s) \to x$: encodes the real state into hidden space
- **Dynamics network** $g(x, a) \to x', r$: predicts the next hidden state and the reward within hidden space
- **Prediction network** $f(x) \to p, v$: predicts the policy and the value from the hidden state

```python
class MuZero:
    def plan(self, state, n_simulations):
        # 1. Encode the real state into hidden space
        root_hidden = self.representation(state)
        root_policy, root_value = self.prediction(root_hidden)

        # 2. MCTS search inside hidden space
        for _ in range(n_simulations):
            self._mcts_iteration(root_hidden)

        # 3. Return the root's action distribution
        return root.action_distribution()

    def _mcts_iteration(self, root):
        # Select, expand, and back up entirely within hidden space
        path = self._select_path(root)
        next_hidden, reward = self.dynamics(path[-1].hidden, path[-1].action)
        policy, value = self.prediction(next_hidden)
        path[-1].expand(policy, reward)
        for node in path:
            node.update(value, reward)
```

### What MuZero Means

MuZero can learn without knowing the rules of the game — it **learns the rules itself**. This lets it generalize to:

- **Atari** (no simulator needed — it learns directly from pixels)
- **Board games** (Go, chess, shogi)
- **Poker** (partially observable)
- **Any MDP**

MuZero is the "unified architecture" of model-based RL — the same algorithm and the same network structure span visual input, vector input, discrete actions, and continuous actions.

## Dreamer V3 and a New Generation of World Models

The Dreamer series (Hafner et al. 2020-2023) is the modern flagship of model-based RL. The core idea: **learn a recurrent latent-variable world model**, and train an actor-critic by "dreaming" inside that model.

### The Recurrent State-Space Model

Dreamer uses a **Recurrent State-Space Model** (RSSM) to jointly model:

- A **deterministic trajectory** (the RNN hidden state $h_t$)
- A **stochastic posterior** (an encoder infers $z_t$ from the observation)
- A **stochastic prior** (predicts $\hat{z}_t$ from $h_t$)

During training, $\hat{z}_t$ is pushed to match $z_t$, which lets the model "imagine" trajectories that stay consistent with the real environment.

```python
class RSSM:
    def forward(self, obs_seq, action_seq):
        h = zeros(batch, hidden_dim)
        posterior_zs = []
        prior_zs = []

        for t in range(T):
            # Prior: predict z_t from h_t
            prior_mean, prior_std = self.prior(h)
            prior_zs.append((prior_mean, prior_std))

            # Posterior: infer z_t from h_t and obs_t
            posterior_mean, posterior_std = self.posterior(h, encoder(obs_seq[t]))
            z = reparameterize(posterior_mean, posterior_std)
            posterior_zs.append((posterior_mean, posterior_std))

            # Update the RNN hidden state
            h = self.rnn(h, z, action_seq[t])

        return prior_zs, posterior_zs
```

### Actor-Critic in Imagination

The actor is trained not on real data but on rollouts inside the world model:

```python
# "Dream" inside the world model
h = world_model.encode(real_observation_sequence)
for t in range(H):  # H = 15, the imagination horizon
    a = actor(h)
    h, r = world_model.predict(h, a)
    imagined_trajectory.append((h, a, r))

# Train the actor-critic on the imagined trajectory
for (h, a, r) in imagined_trajectory:
    critic_loss = ...
    actor_loss = ...
```

### What Makes Dreamer V3 Unified

Dreamer V3's (Hafner et al. 2023) key contribution: **a single hyperparameter setting** that works across 150+ tasks, including:

- Atari (discrete actions + visual input)
- MuJoCo (continuous actions + vector input)
- Crafter (open-world survival)
- DMLab (first-person 3D navigation)
- BSuite (cognitive tasks)

With no tuning at all, Dreamer V3 **beats model-free SOTA** on most benchmarks. This is the first time model-based RL has beaten SAC, PPO, and similar algorithms on generality.

### Three Key Engineering Innovations

1. **Discretized latents**: changing $z$ from a Gaussian distribution to a categorical distribution makes training more stable
2. **symlog loss**: $\text{symlog}(x) = \text{sign}(x) \log(|x| + 1)$ compresses the value function's range so it adapts to different tasks' reward scales
3. **No KL annealing**: directly maximize the ELBO and let the posterior match the prior

These three tricks are what let Dreamer V3 work "out of the box" across 150+ tasks.

## Model-Based vs. Model-Free: Which One to Use

| Dimension                      | Model-Free               | Model-Based                                |
| ------------------------------ | ------------------------ | ------------------------------------------ |
| **Sample efficiency**          | Low (millions of steps)  | High (tens of thousands of steps)          |
| **Asymptotic performance**     | High                     | Limited by model error                     |
| **Compute cost**               | Low (uses data directly) | High (train the model + search/plan)       |
| **Interpretability**           | Black box                | The model can be analyzed                  |
| **Transferability**            | Weak                     | The model can transfer to downstream tasks |
| **Hyperparameter sensitivity** | Medium                   | High (model quality determines everything) |

**When to choose model-free:**

- The simulator is cheap (Atari, MuJoCo, StarCraft)
- You care about final performance and aren't limited by sample count
- You want zero model-inference overhead at deployment

**When to choose model-based:**

- Sampling the real environment is expensive (robotics, autonomous driving, chemical reactions)
- You need fast adaptation (meta-RL, online learning)
- You need interpretability (safety-critical settings)

## The Connection to LLM RL

In LLM training:

- **Model-free**: RLHF/GRPO train directly on RM rewards (model-free)
- **Model-based**: Process Reward Models and verifier models are a kind of "environment model" — PRM-guided search ([Chapter 18, PRM and Search](../chapter20_prm_search/inference-time-search)) is analogous to AlphaZero
- **World model**: the Code World Model ([Chapter 21, SWE-Agent](../chapter23_rl_based_swe/world-model-and-deep-swe)) predicts the results of code execution — it's the MuZero of the LLM era

Once you understand the tradeoff between model-based and model-free, you can see why Tongyi DeepResearch uses PRM-guided search, and why SWE-Agent uses a Code World Model to improve sample efficiency.

## Chapter Summary

Continuous control and model-based RL are the two advanced directions of classical deep RL:

1. **DDPG → TD3 → SAC** is the evolution of deterministic policy gradients: from adding exploration noise, to stabilizing with twin Q-networks and delayed updates, to automatic exploration through maximum entropy
2. **Dyna → PETS → MBPO** is the evolution of model-based data augmentation: the model as a data generator
3. **AlphaZero → MuZero → Dreamer V3** is the flagship line of explicit search + learned models, representing the ceiling of model-based RL

The next chapter, [Chapter 10, Offline Reinforcement Learning](../chapter12_offline_rl/intro), turns to a different angle — **what do you do when the agent can't interact with the environment and can only use historical data?** This is the core problem behind LLM post-training, recommender systems, and other real-world settings.

## Further Reading

- [Silver et al. 2018 "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. 2020 "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero)](https://arxiv.org/abs/1911.08265)
- [Hafner et al. 2023 "Mastering Diverse Domains through World Models" (Dreamer V3)](https://arxiv.org/abs/2301.04104)
- [Janner et al. 2019 "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)](https://arxiv.org/abs/1906.08253)
