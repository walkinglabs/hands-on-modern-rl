# 11.2 Inverse RL and GAIL

> [13.1](./bc-dagger) covered behavior cloning — directly imitating expert actions. BC has a fundamental limit: it never learns _why_ the expert does what it does. This section takes a different route — **inverse RL**: infer a reward function from expert trajectories, then train a standard RL algorithm on that reward. This sidesteps BC's distribution shift, and it also yields a reward signal that transfers better.

## Maximum-Entropy Inverse RL

Inverse RL assumes the expert is good because it is **optimizing some hidden reward function**. Instead of imitating actions directly, we **first recover this reward function**, then hand it to ordinary RL to solve.

### The basic setup of inverse RL

Given expert trajectories $\mathcal{D}_{\text{expert}} = \{\tau_1, \ldots, \tau_M\}$, each $\tau = (s_0, a_0, \ldots, s_T)$, the goal is to learn a reward function $r_\psi(s, a)$ such that:

$$\text{the expert policy is optimal under } r_\psi$$

This condition is **severely underdetermined** — every constant reward $r_\psi \equiv c$ satisfies it. We need extra **regularization**, or a **maximum-entropy principle**, to pin down $r_\psi$ uniquely.

### The MaxEnt IRL objective

Ziebart et al. 2008 proposed maximum-entropy inverse RL. Assume the expert policy follows a **maximum-entropy** distribution (it matches feature expectations while staying as random as possible otherwise):

$$\pi(a \mid s) \propto \exp\left(Q^{\text{soft}}_{r_\psi}(s, a)\right)$$

Then the likelihood of an expert trajectory is:

$$p(\tau \mid r_\psi) = \frac{1}{Z(r_\psi)} \exp\left(\sum_t r_\psi(s_t, a_t)\right)$$

We maximize the log-likelihood of the expert data:

$$\max_\psi \; \mathcal{L}(\psi) = \sum_{\tau \in \mathcal{D}_{\text{expert}}} \left[\sum_t r_\psi(s_t, a_t)\right] - |\mathcal{D}_{\text{expert}}| \log Z(r_\psi)$$

The first term is the cumulative reward of the expert trajectories; the second term, $\log Z$, is the log partition function (the log of the sum of exponentiated rewards over all possible trajectories). The gradient is:

$$\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \text{expert}}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right] - \mathbb{E}_{\tau \sim p(\cdot \mid r_\psi)}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right]$$

In plain terms: **push up the reward on $(s, a)$ pairs the expert visits, and push down the reward on $(s, a)$ pairs the current policy (the policy rolled out under $r_\psi$) visits**. When the two feature expectations match, the gradient is zero.

### Where MaxEnt IRL gets hard

$\log Z(r_\psi)$ has **no closed form** over continuous state-action spaces. There are three main approximations:

1. **Model-based**: use a learned environment model to do forward rollouts and estimate $Z$
2. **Sampling-based soft Q iteration**: approximate with soft Bellman backups (Guided Cost Learning, Finn et al. 2016)
3. **Adversarial (GAIL)**: express $r_\psi$ implicitly via a discriminator (next section)

```python
def maxent_irl_step(reward_net, expert_states_actions, env_sampler, soft_q_planner):
    # 1. Do soft Q planning under the current reward to get a sampling distribution
    current_rewards = reward_net(states_actions_tensor)
    sampled_trajectories = soft_q_planner.rollout(reward_net)

    # 2. Compute the difference in feature expectations
    expert_feat = feature_expectation(expert_states_actions, reward_net)
    sampled_feat = feature_expectation(sampled_trajectories, reward_net)

    # 3. Gradient ascent update on the reward
    grad = expert_feat - sampled_feat
    reward_net.update(grad)
```

MaxEnt IRL is expensive: every outer update requires solving a full soft Q problem in the inner loop. That makes it hard to scale to high-dimensional problems like visual input. **GAIL** avoids the explicit $Z$ computation with adversarial training.

## Generative Adversarial Imitation Learning

Generative Adversarial Imitation Learning (Ho & Ermon 2016) borrows the idea behind GANs: it recasts inverse RL as a game between a **discriminator $D_\phi$ and a generator $\pi_\theta$**.

### The GAIL objective

The discriminator distinguishes "expert data" from "policy data":

$$\max_\phi \; \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}}\left[\log D_\phi(s, a)\right] + \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right]$$

The policy learns by "fooling the discriminator":

$$\min_\theta \; \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log D_\phi(s, a)\right] - \lambda \mathcal{H}(\pi_\theta)$$

The second term is entropy regularization, which keeps the policy from collapsing prematurely. Here $-\log D_\phi(s, a)$ acts as an **implicit reward** — equivalent to the adversarial derivation of $r_\psi(s, a) = \log D_\phi(s, a) - \log(1 - D_\phi(s, a))$ from MaxEnt IRL.

```python
class GAIL:
    def __init__(self, expert_data, policy, discriminator):
        self.expert_buffer = expert_data   # expert (s, a) pairs
        self.policy = policy               # any RL algorithm (PPO/TRPO/SAC)
        self.disc = discriminator          # binary classifier network

    def update(self, n_policy_steps=5, n_disc_steps=1):
        # === 1. Train the discriminator ===
        for _ in range(n_disc_steps):
            # Sample policy data
            policy_states, policy_actions = self.policy.sample_rollout()
            # Binary cross-entropy
            expert_logits = self.disc(self.expert_buffer.sample())
            policy_logits = self.disc(policy_states, policy_actions)
            d_loss = (
                F.binary_cross_entropy_with_logits(expert_logits, ones) +
                F.binary_cross_entropy_with_logits(policy_logits, zeros)
            )
            self.disc_optim.zero_grad(); d_loss.backward(); self.disc_optim.step()

        # === 2. Train the policy: use -log D as the reward ===
        for _ in range(n_policy_steps):
            states, actions, next_states, _ = self.policy.rollout()
            with torch.no_grad():
                rewards = -F.logsigmoid(self.disc(states, actions))  # r = -log(1 - D)
            # Feed into any RL algorithm (assume PPO here)
            self.policy.ppo_update(states, actions, rewards, next_states)
```

### The equivalence between GAIL and MaxEnt IRL

Formally, GAIL is the dual of MaxEnt IRL when the reward function is **unconstrained** (an arbitrary neural network). The optimal discriminator $D_\phi^*$ has the closed form:

$$D_\phi^*(s, a) = \frac{p_{\text{expert}}(s, a)}{p_{\text{expert}}(s, a) + p_{\pi_\theta}(s, a)}$$

Substituting this back in, the optimal reward is exactly $r^*(s, a) = \log D^* - \log(1 - D^*) = \log \frac{p_{\text{expert}}}{p_{\pi_\theta}}$ — the **log-likelihood ratio**. This matches the reward that MaxEnt IRL derives, but GAIL never has to compute the partition function $Z$ explicitly.

### Comparing the three imitation-learning approaches

| Dimension                    | BC   | MaxEnt IRL                     | GAIL                          |
| ---------------------------- | ---- | ------------------------------ | ----------------------------- |
| Addresses distribution shift | ❌   | ✅                             | ✅                            |
| Needs an environment model   | ❌   | ✅ (or a soft-Q approximation) | ❌                            |
| Explicit reward function     | —    | ✅ (interpretable)             | ❌ (implicit)                 |
| Compute cost                 | Low  | High (inner-loop RL)           | Medium (adversarial training) |
| Scales to high dimensions    | Easy | Hard                           | Medium                        |
| LLM counterpart              | SFT  | —                              | Implicit in DPO (see 14.6)    |

::: details GAIL's training instability
This is the classic GAN failure mode: when the discriminator is too strong, the generator's gradient vanishes; when it's too weak, there's no signal to learn from. Common tricks in practice:

- Gradient penalty on the discriminator (Wasserstein GAIL)
- Update the discriminator more slowly than the policy (1 discriminator step per 5 policy steps)
- Tune the entropy coefficient $\lambda$ to 0.1–1.0 to keep the policy from collapsing
  :::

GAIL reaches near-expert performance on MuJoCo, but it needs millions of steps of environment interaction — **sample efficiency is still the bottleneck**. This has driven research into **offline imitation learning** (e.g., DemoDICE, DWBC), which combines expert data with suboptimal data and needs no online interaction.

## Section summary

Inverse RL (IRL) infers a reward function from expert behavior, and maximum-entropy IRL resolves IRL's ill-posedness. GAIL uses the GAN framework to sidestep explicit reward inference, giving imitation learning a large boost in scalability. GAIL went on to inspire later work on adversarial RL and on reward-model training in RLHF.

The next section, [13.3 Meta-RL: MAML, RL², PEARL, In-Context RL](./meta-rl), turns to a different question — **how does an agent adapt quickly to new tasks when the environment keeps changing?**
