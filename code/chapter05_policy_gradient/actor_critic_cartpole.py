"""
Chapter 5: Actor-Critic algorithm -- CartPole-v1
A single-step-update policy gradient method, more efficient than REINFORCE

Problems with REINFORCE:
    - Must wait for a full episode to finish before updating (Monte Carlo method)
    - If episodes are long, data efficiency is low
    - The high-variance G_t computed after the episode ends is used to update every prior step

Actor-Critic's improvement:
    - Uses a TD(0) estimate in place of the full-episode return
    - advantage = r + gamma * V(s') - V(s)
    - Can update immediately at every step, no need to wait for the episode to end
    - Because it bootstraps from V(s'), variance is lower

Network architecture:
    An Actor-Critic network with a shared backbone
    - Shared layer: extracts state features (reuses parameters, reduces compute)
    - Actor head: outputs action probabilities (the policy)
    - Critic head: outputs the state value (the value function)

How to run:
    python actor_critic_cartpole.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the output directory
os.makedirs("output", exist_ok=True)

# Configure Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Actor-Critic network architecture
# ==========================================
class ActorCritic(nn.Module):
    """
    Actor-Critic network: shared backbone, two output heads

    Architecture diagram:
        Input state (dim=4)
            |
        +-------+
        | Linear| 4 -> 128
        |  ReLU |
        +-------+
            |
        +-------+
        | Actor  | 128 -> 2 -> Softmax  (policy: which action to pick)
        +-------+
            |
        +-------+
        | Critic | 128 -> 1            (value: how good the current state is)
        +-------+

    Benefits of a shared backbone:
        - State features only need to be computed once
        - The Actor and Critic can share the underlying representation
        - Fewer parameters, faster training
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared backbone layer: extracts a feature representation of the state
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: outputs the action probability distribution
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head: outputs the state value (scalar)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Forward pass, producing both action probabilities and the state value

        Args:
            x: state tensor [batch_size, state_dim]
        Returns:
            probs: action probabilities [batch_size, action_dim]
            value: state value [batch_size]
        """
        # Shared layer extracts features
        features = self.shared_backbone(x)

        # Actor outputs action probabilities
        action_logits = self.actor_head(features)
        probs = torch.softmax(action_logits, dim=-1)

        # Critic outputs the state value
        value = self.critic_head(features).squeeze(-1)

        return probs, value


# ==========================================
# Part 2: Compute the TD error and advantage
# ==========================================
def compute_advantage(reward, value, next_value, gamma=0.99, done=False):
    """
    Compute the TD(0) advantage function

    TD advantage = r + gamma * V(s') - V(s)

    Intuition:
        - V(s) is the Critic's "predicted score" for the current state
        - r + gamma * V(s') is "the reward actually received + the updated prediction for the future"
        - The difference between the two is the "prediction error": better than expected (positive) or worse (negative)

    Difference from REINFORCE:
        REINFORCE: advantage = G_t (the full episode's cumulative return)
        Actor-Critic: advantage = r + gamma * V(s') - V(s) (single-step TD error)

    Args:
        reward: immediate reward r_t
        value: current state value V(s_t)
        next_value: next state value V(s_{t+1})
        gamma: discount factor
        done: whether the episode has ended
    Returns:
        advantage: the TD advantage value
    """
    if done:
        # When the episode ends, there is no next state, so target = r_t
        target = reward
    else:
        # TD target: r_t + gamma * V(s_{t+1})
        target = reward + gamma * next_value

    advantage = target - value
    return advantage, target


# ==========================================
# Part 3: Main training loop
# ==========================================
def train():
    """
    The full Actor-Critic training pipeline

    Core difference (compared to REINFORCE):
        REINFORCE: collect a full episode -> compute all G_t -> one backward pass
        Actor-Critic: compute the TD error at every step -> update the network immediately

    This means Actor-Critic can learn online, without waiting for episodes
    to end, giving it higher data efficiency.
    """
    # ---------- Hyperparameters ----------
    num_episodes = 500
    gamma = 0.99
    learning_rate = 1e-3
    hidden_dim = 128

    # ---------- Initialization ----------
    env = gym.make("CartPole-v1")
    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Record training data
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []

    print("=" * 60)
    print("  Actor-Critic -- CartPole-v1 training")
    print("=" * 60)
    print(f"  Hyperparameters:")
    print(f"    Episodes: {num_episodes}")
    print(f"    Discount factor gamma: {gamma}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Hidden layer dimension: {hidden_dim}")
    print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        steps = 0

        done = False
        truncated = False

        while not (done or truncated):
            # ========== Step 1: observe the current state ==========
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Forward pass: get both the action probabilities and the state value
            probs, value = model(state_tensor)
            probs = probs.squeeze(0)     # [action_dim]
            value = value.squeeze()       # scalar

            # ========== Step 2: choose an action (sample by probability) ==========
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # Save log pi(a|s), used later to compute the policy gradient
            log_prob = dist.log_prob(action)

            # ========== Step 3: execute the action, observe the transition ==========
            next_state, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward
            steps += 1

            # ========== Step 4: compute the value of the next state ==========
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, next_value = model(next_state_tensor)
                next_value = next_value.squeeze()

            # ========== Step 5: compute the TD advantage and loss ==========
            # TD advantage: A(s,a) = r + gamma * V(s') - V(s)
            is_done = done or truncated
            advantage, target = compute_advantage(
                reward, value, next_value, gamma, done=is_done
            )

            # Actor loss: -log pi(a|s) * A(s,a)
            # Same form as REINFORCE, but the advantage is a single-step TD estimate
            actor_loss = -log_prob * advantage

            # Critic loss: pushes V(s) toward the TD target r + gamma * V(s')
            critic_loss = nn.MSELoss()(value, target.detach())

            # Combine the losses (could be weighted; here they're equally weighted)
            total_loss = actor_loss + critic_loss

            # ========== Step 6: update the network immediately ==========
            # Note: REINFORCE only updates after the episode ends; here every step updates!
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_losses_save = critic_loss.item()
            total_critic_loss += total_critic_losses_save

            # Move to the next state
            state = next_state

        # Record this episode's data
        episode_rewards.append(episode_reward)
        episode_actor_losses.append(total_actor_loss / max(steps, 1))
        episode_critic_losses.append(total_critic_loss / max(steps, 1))

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(
                f"  Episode {episode + 1:4d}/{num_episodes} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Last-50 avg: {recent_avg:6.1f} | "
                f"Steps: {steps:3d}"
            )

    env.close()

    # ---------- Training results summary ----------
    print("=" * 60)
    print("  Training complete!")
    print(f"  Average reward over the last 50 episodes: {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  Best episode reward: {np.max(episode_rewards):.1f}")
    print("=" * 60)

    # ---------- Convergence speed comparison with REINFORCE ----------
    compare_with_reinforce(episode_rewards)

    # ---------- Plot the training curve ----------
    plot_training_curve(episode_rewards)


# ==========================================
# Part 4: Convergence speed comparison with REINFORCE
# ==========================================
def compare_with_reinforce(actor_critic_rewards):
    """
    Compare Actor-Critic's convergence speed against REINFORCE

    Convergence speed metric: how many episodes it takes to first reach the target reward (e.g. 195)
    CartPole-v1's "solved" criterion: average reward >= 195 over 100 consecutive episodes
    """
    target_reward = 195

    # Compute Actor-Critic's convergence speed
    ac_solve_episode = None
    for i in range(len(actor_critic_rewards) - 99):
        window_avg = np.mean(actor_critic_rewards[i:i + 100])
        if window_avg >= target_reward:
            ac_solve_episode = i + 100
            break

    print("\n" + "-" * 60)
    print("  Convergence speed comparison")
    print("-" * 60)
    print(f"  CartPole-v1 solved criterion: average reward >= {target_reward} over 100 consecutive episodes")

    if ac_solve_episode:
        print(f"  Actor-Critic solved the environment at episode {ac_solve_episode}")
    else:
        print(f"  Actor-Critic did not reach the solved criterion within {len(actor_critic_rewards)} episodes")

    # A note about REINFORCE
    print(f"\n  [Reference] Typical figures:")
    print(f"    REINFORCE usually needs 300-500+ episodes to solve CartPole")
    print(f"    Actor-Critic usually solves it within 200-350 episodes")
    print(f"    Reason: Actor-Critic updates every step, giving higher data efficiency")
    print(f"            and TD(0) has lower variance than the Monte Carlo return")
    print("-" * 60)


# ==========================================
# Part 5: Plot the training curve
# ==========================================
def plot_training_curve(episode_rewards):
    """
    Plot the Actor-Critic training reward curve

    Includes both the raw rewards and the moving-average line
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw reward curve
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='Episode reward (raw)')

    # Moving-average curve
    window = 50
    moving_avg = [np.mean(episode_rewards[max(0, i - window + 1):i + 1])
                  for i in range(len(episode_rewards))]
    ax.plot(moving_avg, color='crimson', linewidth=2.0,
            label=f'Moving average (window={window})')

    # Mark the solved-criterion line
    ax.axhline(y=195, color='green', linestyle='--', alpha=0.7,
               label='Solved criterion (reward=195)')

    ax.set_xlabel('Training episode', fontsize=12)
    ax.set_ylabel('Episode reward', fontsize=12)
    ax.set_title('Actor-Critic -- CartPole-v1 training curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/actor_critic_cartpole_rewards.png', dpi=150, bbox_inches='tight')
    print("  Training curve saved to output/actor_critic_cartpole_rewards.png")
    plt.show()


# ==========================================
# Program entry point
# ==========================================
if __name__ == "__main__":
    train()
