"""
Chapter 6: Implementing PPO (Proximal Policy Optimization) from scratch
——Understanding every step of PPO in pure PyTorch on CartPole-v1

Core PPO formula:
    ratio = exp(new_logprob - old_logprob)
    clipped_ratio = clip(ratio, 1-eps, 1+eps)
    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

How to run:
    python ppo_from_scratch.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# Create output directory
os.makedirs("output", exist_ok=True)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Actor-Critic network
# ==========================================
class ActorCritic(nn.Module):
    """
    Actor-Critic network: shared trunk + separate action head and value head

    Structure:
        Shared layers: state_dim → 64 → 64 (ReLU)
        Actor:   64 → action_dim (outputs action logits)
        Critic:  64 → 1 (outputs state value V(s))

    Benefits of a shared trunk:
        - Feature reuse, fewer parameters
        - Actor and Critic share the underlying representation
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared trunk network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Actor head: outputs action logits
        self.actor_head = nn.Linear(64, action_dim)

        # Critic head: outputs state value
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        """Forward pass, returns action probabilities and value"""
        shared_features = self.shared_net(x)

        # Actor: outputs the action distribution
        action_logits = self.actor_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic: outputs the state value
        value = self.critic_head(shared_features)

        return action_probs, value

    def get_action(self, state):
        """Sample an action from the current state, return action, log-prob, value"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state_tensor)

        # Sample using a Categorical distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def evaluate(self, states, actions):
        """
        Evaluate the given (state, action) pairs
        Returns: log-probs, state values, distribution entropy
        """
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy


# ==========================================
# Part 2: GAE (Generalized Advantage Estimation)
# ==========================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)

    Core idea of GAE:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)    # TD error
        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}      # GAE advantage

    Args:
        rewards: reward at each step
        values:  value estimate V(s) at each step
        dones:   whether each step ends the episode
        gamma:   discount factor (controls the weight of distant returns)
        lam:     GAE lambda (controls the bias-variance tradeoff)
            λ=0: low variance, high bias (only looks at single-step TD error)
            λ=1: high variance, low bias (Monte Carlo return)

    Returns:
        advantages: advantage estimates
        returns:    target returns (used to train the Critic)
    """
    advantages = []
    gae = 0

    # Convert list to something easier to work with
    values = list(values)
    # The final step needs a terminal-state V(s)=0 appended
    next_value = 0

    # Iterate backwards to compute GAE
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # Episode ended, next step's value is 0
            next_value = 0
            gae = 0

        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE accumulation: A_t = δ_t + (γλ) * A_{t+1}
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

        # Update next step's V(s)
        next_value = values[t]

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)

    # Normalize advantages (improves training stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ==========================================
# Part 3: PPO clipped loss
# ==========================================
def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip_eps=0.2):
    """
    PPO clipped objective function

    Core formula:
        ratio = exp(new_logprob - old_logprob) = π_new(a|s) / π_old(a|s)
        L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    When ratio > 1+ε or ratio < 1-ε, the gradient is cut off
    → prevents the policy update step from being too large

    Args:
        old_logprobs: log-probs under the old policy
        new_logprobs: log-probs under the new policy
        advantages:   advantage estimates
        clip_eps:     clip range ε (default 0.2)

    Returns:
        policy_loss: policy loss
        clip_frac:   fraction that was clipped (used for monitoring training)
    """
    # Compute the importance-sampling ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Unclipped objective
    surr1 = ratio * advantages

    # Clipped objective
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    # Take the smaller of the two (conservative update)
    policy_loss = -torch.min(surr1, surr2).mean()

    # Compute the fraction that was clipped (monitoring metric)
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    return policy_loss, clip_frac


# ==========================================
# Part 4: Collect trajectory data
# ==========================================
def collect_trajectories(model, env, n_steps=2048):
    """
    Use the current policy to collect n_steps of trajectory data in the environment

    Collected data:
        - states:  states
        - actions: actions
        - logprobs: log-probs under the old policy (used in the subsequent PPO update)
        - rewards: rewards
        - dones:   episode-done flags
        - values:  value estimates

    Returns:
        batch dict + list of cumulative episode rewards
    """
    states = []
    actions = []
    old_logprobs = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()
    episode_rewards = []
    current_ep_reward = 0

    for step in range(n_steps):
        state_tensor = torch.FloatTensor(obs)

        # Sample an action with the current policy
        with torch.no_grad():
            action_probs, value = model(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Store the data
        states.append(obs.copy())
        actions.append(action.item())
        old_logprobs.append(log_prob.item())
        values.append(value.item())

        # Execute the action
        next_obs, reward, done, truncated, _ = env.step(action.item())
        rewards.append(reward)
        dones.append(done or truncated)

        current_ep_reward += reward

        if done or truncated:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0
            next_obs, _ = env.reset()

        obs = next_obs

    # Convert to tensors
    batch = {
        "states": torch.FloatTensor(np.array(states)),
        "actions": torch.LongTensor(actions),
        "old_logprobs": torch.FloatTensor(old_logprobs),
        "rewards": rewards,
        "dones": dones,
        "values": values,
    }

    return batch, episode_rewards


# ==========================================
# Part 5: PPO update
# ==========================================
def ppo_update(model, optimizer, batch, n_epochs=10, batch_size=64,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """
    Run multiple rounds of PPO updates using the collected data

    Each round:
        1. Re-evaluate the old data with the new policy → get new log_probs
        2. Compute the PPO clipped loss (policy loss)
        3. Compute the value function loss (Critic)
        4. Compute the entropy bonus (encourages exploration)
        5. Total loss = policy loss + value loss - entropy bonus

    Returns:
        dict of training metrics (for monitoring)
    """
    # First compute GAE advantages and target returns
    advantages, returns = compute_gae(
        batch["rewards"], batch["values"], batch["dones"],
        gamma=0.99, lam=0.95
    )

    # Keep the data on CPU (keep things simple)
    states = batch["states"]
    actions = batch["actions"]
    old_logprobs = batch["old_logprobs"]

    dataset_size = states.shape[0]
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_clip_frac = 0
    update_count = 0

    for epoch in range(n_epochs):
        # Shuffle the data randomly
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_logprobs = old_logprobs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # Evaluate the old data with the new policy
            new_logprobs, new_values, entropy = model.evaluate(mb_states, mb_actions)

            # ---- Policy loss (PPO-Clip) ----
            policy_loss, clip_frac = ppo_clip_loss(
                mb_old_logprobs, new_logprobs, mb_advantages, clip_eps
            )

            # ---- Value function loss ----
            value_loss = F.mse_loss(new_values, mb_returns)

            # ---- Entropy bonus ----
            entropy_bonus = entropy.mean()

            # ---- Total loss ----
            # total loss = policy loss + vf_coef * value loss - ent_coef * entropy
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            # Gradient update
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (prevents exploding gradients)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Accumulate statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            total_clip_frac += clip_frac
            update_count += 1

    # Return averaged metrics
    metrics = {
        "policy_loss": total_policy_loss / update_count,
        "value_loss": total_value_loss / update_count,
        "entropy": total_entropy / update_count,
        "clip_fraction": total_clip_frac / update_count,
    }

    return metrics


# ==========================================
# Part 6: Main training loop
# ==========================================
def train():
    """PPO main training function"""
    print("=" * 50)
    print("Chapter 6: Implementing PPO from scratch — CartPole-v1")
    print("=" * 50)

    # Create the environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    # Create the model and optimizer
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    print(f"\nNetwork structure:")
    print(model)
    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")

    # Training hyperparameters
    n_steps = 2048        # Steps collected per rollout
    n_epochs = 10         # Update rounds per batch of data
    batch_size = 64       # Mini-batch size
    clip_eps = 0.2        # PPO clip range
    total_episodes = 1000 # Total number of training episodes

    # Record training metrics
    all_rewards = []
    all_policy_losses = []
    all_value_losses = []
    all_entropies = []
    all_clip_fracs = []

    print(f"\nStarting training (target: {total_episodes} episodes)...")
    print("-" * 50)

    episode_count = 0
    iteration = 0

    while episode_count < total_episodes:
        iteration += 1

        # Step 1: collect trajectories
        batch, ep_rewards = collect_trajectories(model, env, n_steps=n_steps)
        episode_count += len(ep_rewards)
        all_rewards.extend(ep_rewards)

        # Step 2: PPO update
        metrics = ppo_update(
            model, optimizer, batch,
            n_epochs=n_epochs,
            batch_size=batch_size,
            clip_eps=clip_eps,
        )

        all_policy_losses.append(metrics["policy_loss"])
        all_value_losses.append(metrics["value_loss"])
        all_entropies.append(metrics["entropy"])
        all_clip_fracs.append(metrics["clip_fraction"])

        # Periodically print training info
        if iteration % 5 == 0 or len(ep_rewards) > 0:
            recent_rewards = all_rewards[-20:] if len(all_rewards) >= 20 else all_rewards
            avg_reward = np.mean(recent_rewards)
            print(
                f"  Iter {iteration:3d} | "
                f"Episodes: {episode_count:4d} | "
                f"Avg reward: {avg_reward:6.1f} | "
                f"Policy loss: {metrics['policy_loss']:.4f} | "
                f"Value loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.3f} | "
                f"Clip fraction: {metrics['clip_fraction']:.3f}"
            )

    print("-" * 50)
    print(f"Training complete! Trained {episode_count} episodes across {iteration} iterations")

    # Final evaluation
    test_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0
        while not (done or truncated):
            state_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)

    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f"\n20-episode test result: mean reward = {mean_reward:.1f} ± {std_reward:.1f}")

    env.close()

    # ==========================================
    # Part 7: Plot training curves
    # ==========================================
    print("\nPlotting training curves...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO from Scratch — CartPole-v1 Training Curves", fontsize=16, fontweight="bold")

    # Subplot 1: episode rewards
    ax1 = axes[0, 0]
    window = min(20, len(all_rewards))
    if window > 0:
        smoothed = np.convolve(all_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(len(all_rewards)), all_rewards, alpha=0.3, color="#90CAF9", label="Raw reward")
        ax1.plot(range(window - 1, len(all_rewards)), smoothed, color="#2196F3",
                 linewidth=2, label=f"Moving average (window={window})")
        ax1.axhline(y=475, color="green", linestyle="--", alpha=0.5, label="Target (475)")
    ax1.set_title("Episode reward", fontsize=13)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: policy loss & value loss
    ax2 = axes[0, 1]
    if all_policy_losses:
        ax2.plot(all_policy_losses, color="#F44336", alpha=0.8, linewidth=1.2, label="Policy loss")
        ax2.plot(all_value_losses, color="#2196F3", alpha=0.8, linewidth=1.2, label="Value loss")
    ax2.set_title("Loss curves", fontsize=13)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: policy entropy
    ax3 = axes[1, 0]
    if all_entropies:
        ax3.plot(all_entropies, color="#FF9800", alpha=0.8, linewidth=1.5)
        ax3.set_title("Policy entropy (exploration level)", fontsize=13)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Entropy")
        ax3.annotate("Decreasing entropy = more deterministic policy", xy=(len(all_entropies) * 0.6, max(all_entropies) * 0.8),
                     fontsize=10, color="gray", style="italic")
    ax3.grid(True, alpha=0.3)

    # Subplot 4: clip fraction
    ax4 = axes[1, 1]
    if all_clip_fracs:
        ax4.plot(all_clip_fracs, color="#9C27B0", alpha=0.8, linewidth=1.5)
        ax4.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range = 0.2")
        ax4.set_title("Clip fraction", fontsize=13)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Fraction clipped")
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/ppo_from_scratch_curves.png", dpi=150, bbox_inches="tight")
    print("Training curves saved to: output/ppo_from_scratch_curves.png")
    plt.show()


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    train()
