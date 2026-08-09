---
title: 'Chapter 9: Continuous Control & Model-Based RL'
---

# Chapter 9: Continuous Control and Model-Based Deep RL

> [Chapter 8, PPO](../chapter10_ppo/intro) solved policy learning in continuous action spaces — a Gaussian policy outputs continuous actions, and clipping keeps updates stable. But PPO is on-policy: every policy update forces a fresh round of sampling, so **sample efficiency is very poor**. This chapter tackles two problems: (1) how to handle continuous actions off-policy (DDPG/TD3/SAC), and (2) how to push sample efficiency further by learning a model of the environment (Model-Based RL + AlphaZero/Dreamer).

## 11.1 Deterministic Policy Gradients and DDPG

In problems like CartPole and Atari, actions are discrete — left/right, up/down/left/right — and Q-Learning or a softmax policy handles them directly. But in robot control, autonomous driving, or robotic-arm manipulation, actions are **continuous**: a joint angle $\theta \in \mathbb{R}^n$, a throttle position in $[0, 1]$, a steering angle in $[-\pi, \pi]$.

Continuous actions raise two challenges:

1. **The Q function can't be maximized by argmax**: in the discrete case, $a^* = \arg\max_a Q(s, a)$ is computed by direct enumeration; in the continuous case there's nothing to enumerate over.
2. **The policy's output changes shape**: instead of softmax probabilities, it now outputs distribution parameters (a mean, a variance).

### The continuous version of the policy gradient theorem

[Chapter 6's policy gradient](../chapter08_policy_gradient/reinforce) gives us:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a\mid s) \cdot Q^\pi(s, a)\right]$$

This requires the policy $\pi_\theta(a \mid s)$ to be **stochastic** — a probability distribution. But Silver et al. 2014 proved that if the policy is **deterministic**, $a = \mu_\theta(s)$, an analogous gradient theorem still holds:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\mu}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}\right]$$

This is the **Deterministic Policy Gradient (DPG)** theorem. It is more sample-efficient than the stochastic version:

- **No integral over $a$ is needed**: the stochastic PG has to take an expectation over every possible action; the deterministic PG only takes an expectation over states.
- **It's off-policy friendly**: a deterministic policy can be trained on data collected by any behavior policy.

But there's a fatal problem: **a deterministic policy doesn't explore**. If $\mu_\theta(s)$ always returns the same $a$, the agent never tries anything else. DDPG's fix: **add noise to the action at training time**.

### Deep Deterministic Policy Gradient

Deep Deterministic Policy Gradient (Lillicrap et al. 2015) combines DPG with the deep-network tricks from DQN:

- **Actor**: $\mu_\theta(s)$ outputs a continuous action (the network regresses it directly).
- **Critic**: $Q_\phi(s, a)$ evaluates the value of that action.
- **Target networks**: stabilize training (inherited from DQN).
- **Experience replay**: reuse data off-policy (inherited from DQN).

### The main training loop

```python
class DDPG:
    def __init__(self, state_dim, action_dim, action_max):
        # main networks
        self.actor = Actor(state_dim, action_dim, action_max)
        self.critic = Critic(state_dim, action_dim)
        # target networks (soft update)
        self.actor_target = copy(self.actor)
        self.critic_target = copy(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        self.gamma = 0.99
        self.tau = 0.005  # soft-update coefficient

    def select_action(self, state, explore=True):
        with torch.no_grad():
            action = self.actor(state)
        if explore:
            # Ornstein-Uhlenbeck or Gaussian exploration noise
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -self.action_max, self.action_max)

    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic update ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor update: maximize Q(s, mu(s)) ===
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        self.actor_optim.step()

        # === Soft-update the target networks ===
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
```

On MuJoCo physics environments (HalfCheetah, Hopper, Walker2d), DDPG was the first case of deep RL beating methods based on linear features, like TRPO/CES. But DDPG has several widely criticized flaws:

- **Q-value overestimation**: the target Q uses a max, which is easily pushed up by noise.
- **Sensitivity to hyperparameters**: a small change in learning rate, noise scale, or network architecture can make it diverge.
- **Unstable training**: once the critic learns something wrong, the actor follows it downhill — a positive-feedback death spiral.

## Section Summary

The Deterministic Policy Gradient (DPG) theorem extends the policy gradient from stochastic policies to deterministic ones, making off-policy training possible in continuous action spaces too. DDPG combines DPG with DQN's deep-network tricks, and on MuJoCo it was the first time deep RL beat classical methods.

But DDPG carries three major flaws: Q-value overestimation, hyperparameter sensitivity, and unstable training. The next section, [11.2 TD3 and SAC](./td3-sac), gives two complementary fixes — TD3 stabilizes DDPG with engineering tricks, while SAC rebuilds the objective from the ground up using maximum-entropy RL.
