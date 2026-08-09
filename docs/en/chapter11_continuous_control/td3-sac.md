# 9.2 TD3 and SAC

> [11.1](./intro) laid out deterministic policy gradients (DPG) and DDPG — carrying DQN's off-policy idea over to continuous actions. But DDPG has three widely criticized flaws: Q-value overestimation, hyperparameter sensitivity, and unstable training. This section gives two complementary fixes: **TD3** stabilizes DDPG with engineering tricks, and **SAC** rebuilds the objective from the ground up using maximum-entropy RL.

## Stability Patches for DDPG

Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al. 2018) targets DDPG's three flaws with three fixes.

### 1. Twin Q-Networks

Borrowing the idea from Double DQN: train **two independent critics** $Q_{\phi_1}, Q_{\phi_2}$, and take the smaller value as the target:

$$y = r + \gamma \cdot \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \mu_{\theta'}(s'))$$

This suppresses Q-value overestimation structurally — the probability that both networks overestimate at once is much lower than for a single network.

```python
class TD3Critic:
    def __init__(self, state_dim, action_dim):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)  # independently initialized

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)

    def target_min(self, s, a):
        return torch.min(self.Q1(s, a), self.Q2(s, a))
```

### 2. Delayed Policy Updates

The critic is much harder to learn than the actor — the critic has to fit the two-argument function $Q(s,a)$, while the actor only needs to learn the single-argument function $\mu(s)$. TD3 updates the actor only once every $d$ steps ($d=2$), giving the critic a few extra rounds to converge before it starts feeding signal to the actor:

```python
for step in range(total_steps):
    # update the critic every step
    update_critic()

    # update the actor + target networks only every d=2 steps
    if step_count % policy_delay == 0:
        update_actor()
        soft_update_targets()
```

The intuition: while the critic hasn't learned well yet, the gradient the actor receives is mostly noise. Delaying the update keeps the actor from being pulled off course by bad gradients.

### 3. Target Policy Smoothing

DDPG's target action $a' = \mu_{\theta'}(s')$ is deterministic, but the function approximator's value near $s'$ can swing sharply. TD3 adds a bit of smoothing noise to the target action:

$$a' = \text{clip}(\mu_{\theta'}(s') + \epsilon, a_{\text{low}}, a_{\text{high}}), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

This is equivalent to "local averaging in action space" — it smooths the Q-function along the action dimension, **reducing the critic's sensitivity to small perturbations**. $\sigma = 0.2, c = 0.5$ is a common configuration.

### The Combined Effect of the Three Tricks

TD3 **substantially stabilizes DDPG** on MuJoCo, surpassing the performance of early SAC versions from the same period. Even today, TD3 remains a strong baseline for continuous control.

```python
class TD3:
    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic update (twin Q) ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # target policy smoothing
            noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-self.action_max, self.action_max)
            # take the min over the twin Q's
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor update (delayed) ===
        if self.step_count % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad(); actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
```

## Maximum-Entropy RL

Soft Actor-Critic (Haarnoja et al. 2018) approaches the problem from a completely different angle: **instead of requiring the policy to maximize expected return, maximize return plus entropy**.

### The Maximum-Entropy RL Objective

$$J(\pi) = \mathbb{E}_{(s_t, a_t) \sim \pi}\left[\sum_t \gamma^t \big(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\big)\right]$$

Here $\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ is the policy entropy, and $\alpha$ is a temperature coefficient controlling how much weight the entropy term gets.

**Why add entropy?**

- **Encourages exploration**: a high-entropy policy doesn't collapse prematurely onto a single action
- **Robustness**: a multimodal policy — one that spreads probability across several good actions — is more robust to perturbations in the environment
- **Training stability**: entropy regularization smooths out the Q-function and helps avoid overestimation

### The Soft Bellman Equation

The modified Bellman backup:

$$Q^\pi(s, a) = \mathbb{E}_{s'}\left[r + \gamma \cdot V^\pi(s')\right], \quad V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] + \alpha \mathcal{H}(\pi(\cdot \mid s))$$

The key change: $V$ is no longer $\max_a Q$, it's a **soft max** — a log-sum-exp form that takes an expectation over continuous actions:

$$V^\pi(s) = \alpha \log \int \exp\left(\frac{Q^\pi(s, a)}{\alpha}\right) da$$

### Reparameterizing the Stochastic Policy

SAC's policy $\pi_\theta(a \mid s)$ is a Gaussian distribution. It uses the reparameterization trick to compute the actor gradient:

$$a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This makes the actor loss differentiable:

$$\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{D}, \epsilon}\left[\alpha \log \pi_\theta(a \mid s) - Q_\phi(s, a)\right]$$

### Automatic Temperature Tuning

The hardest hyperparameter to set is $\alpha$. SAC's engineering innovation is **automatic temperature adjustment**:

$$\alpha^* = \arg\max_\alpha \mathbb{E}\left[-\alpha \log \pi(a \mid s) - \alpha \mathcal{H}_0\right]$$

where $\mathcal{H}_0$ is the target entropy (usually set to $-|\mathcal{A}|$). This lets $\alpha$ adjust automatically during training: if entropy is too high, $\alpha$ goes down; if entropy is too low, $\alpha$ goes up.

```python
# alpha optimization for automatic temperature tuning
def update_alpha(self, states, actions):
    # alpha is a learnable parameter; the goal is to push policy entropy toward target_entropy
    log_pi = -self.actor.log_prob(states, actions)  # negative log-likelihood of the current policy
    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.exp()
```

### Why SAC Works So Well

SAC has topped the leaderboard on MuJoCo for a long stretch, for these reasons:

1. **High sample efficiency from being off-policy** (inherited from DDPG)
2. **Automatic exploration from maximum entropy** (no noise schedule to tune)
3. **Stable training** (twin Q + soft targets)
4. **Superhuman performance** (15000+ on HalfCheetah)

### Comparing the Three Algorithms

| Dimension                  | DDPG          | TD3           | SAC                      |
| -------------------------- | ------------- | ------------- | ------------------------ |
| Policy type                | Deterministic | Deterministic | Stochastic (Gaussian)    |
| Q-networks                 | 1             | 2 (Twin)      | 2 (Twin)                 |
| Exploration                | Added noise   | Added noise   | Entropy bonus (built-in) |
| Stability                  | Poor          | Moderate      | Strong                   |
| Hyperparameter sensitivity | High          | Moderate      | Low                      |
| Recommended default        | ❌            | ⚠️            | ✅                       |

**Practical advice**: for continuous control, default to SAC; if you need a deterministic policy (no randomness at deployment time), pick TD3.

## Training Curves on HalfCheetah

A comparison of training for 1M steps on the MuJoCo HalfCheetah-v3 environment:

```
Return
12000 │                    ╭─────── SAC (stable convergence)
10000 │                  ╭─╯
 8000 │                ╭─╯  ╭─────── TD3 (stable but somewhat slower)
 6000 │              ╭─╯   ╱
 4000 │            ╭─╯    ╱
 2000 │          ╭─╯     ╱  ╭───── DDPG (diverges, with occasional recoveries)
     0 │─────────╯──────╱──╯
       └───────────────────────────────
        0    200K  400K  600K  800K  1M steps
```

Three observations:

- **SAC** converges fastest and most reliably — the exploration built into maximum entropy makes early learning fast
- **TD3** is somewhat slower than SAC but ends up close in final performance — the stability patches make DDPG usable
- **DDPG** spends most of its time diverging — occasionally training succeeds, but only for certain random seeds

## Section Summary

DDPG → TD3 → SAC traces a three-step evolution of continuous control:

1. **DDPG** extends the DQN idea to continuous actions, but is unstable
2. **TD3** stabilizes DDPG with three tricks: twin Q, delayed updates, and target smoothing
3. **SAC** rebuilds the problem at the level of the objective function using maximum-entropy RL, giving it built-in exploration and automatic temperature tuning

In practice, SAC is the default choice, TD3 is the fallback for scenarios that need a deterministic policy, and DDPG is no longer recommended.

The next section, [11.3 Model-Based RL](./model-based), turns to a different direction — when sampling from the real environment is expensive, learn a model of the environment to generate "fake" data, boosting sample efficiency by 10-100x.
