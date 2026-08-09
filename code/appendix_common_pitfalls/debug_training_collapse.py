"""
Appendix A - Common Pitfalls and Solutions: Diagnosing and Fixing Training Collapse

This script demonstrates another common pitfall in reinforcement learning: training collapse.

What is training collapse?
    The loss suddenly explodes during training, the policy completely degenerates,
    and the reward drops off a cliff.
    Unlike "reward hacking", training collapse is caused by training instability,
    not by a reward design problem.

Typical symptoms:
    - The loss suddenly becomes NaN or infinite
    - The reward curve suddenly drops to near zero and never recovers
    - The gradient norm keeps growing
    - The agent picks only one action (policy degeneration)

This script includes three failure scenarios and their corresponding fixes:
    Scenario 1: Learning rate too high -> loss explosion
    Scenario 2: Missing gradient clipping -> gradient explosion
    Scenario 3: Epsilon not decaying -> never converges

It then demonstrates correct training after the fixes, along with a 4-step debugging methodology.

How to run:
    python debug_training_collapse.py
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the output directory
os.makedirs("output", exist_ok=True)

# Set a CJK-capable font so chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Core components (Q-network, replay buffer)
# ==========================================
class QNetwork(nn.Module):
    """
    Q-network: maps a state to the Q-value of each action
    Architecture: state_dim -> 128 -> 128 -> action_dim
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        """Forward pass: takes a state, outputs Q-values for each action"""
        return self.net(x)


class ReplayBuffer:
    """
    Experience replay buffer: stores and samples training data
    """

    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# Part 2: Configurable DQN agent
# ==========================================
class ConfigurableDQNAgent:
    """
    Configurable DQN agent -- supports deliberately triggering various training failures

    Parameters:
        lr: learning rate (set deliberately high to trigger loss explosion)
        clip_grad: whether to enable gradient clipping
        clip_max_norm: maximum norm for gradient clipping
        epsilon_start: initial exploration rate
        epsilon_end: final exploration rate
        epsilon_decay: exploration rate decay factor (set to 1.0 for no decay)
    """

    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99,
                 clip_grad=True, clip_max_norm=10.0,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma

        # Q-network and target network
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Experience replay
        self.buffer = ReplayBuffer(capacity=10000)

        # Gradient clipping configuration
        self.clip_grad = clip_grad
        self.clip_max_norm = clip_max_norm

        # Exploration rate configuration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # For diagnostics: track the gradient norm
        self.last_grad_norm = 0.0

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, batch_size):
        """
        Sample from the replay buffer and update the Q-network
        Returns: the loss value (a special marker if NaN occurs)
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Compute the gradient norm (for diagnostics)
        total_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.last_grad_norm = total_norm

        # Whether gradient clipping is enabled
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), max_norm=self.clip_max_norm
            )

        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy the Q-network's weights to the target network"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# Part 3: Unified training function
# ==========================================
def train_scenario(agent, env, label, num_episodes=300, batch_size=64,
                   target_update_freq=10, verbose=True):
    """
    Train the DQN agent under a given configuration

    Returns rich diagnostic data for later analysis and visualization

    Args:
        agent: the configurable DQN agent
        env: the training environment
        label: scenario name
        num_episodes: number of training episodes
        batch_size: batch size
        target_update_freq: target network update frequency
        verbose: whether to print training logs
    Returns:
        results: a dict containing all diagnostic data
    """
    # Diagnostic data collection
    reward_history = []
    loss_history = []
    grad_norm_history = []
    action_counts = {0: [], 1: []}  # Track the count of left/right actions per episode
    nan_detected = False

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Training scenario: {label}")
        print(f"{'─' * 60}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_grad_norm = 0.0
        steps = 0
        actions_taken = {0: 0, 1: 0}

        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size)

            # Detect NaN (a clear sign of training collapse)
            if np.isnan(loss) or np.isinf(loss):
                nan_detected = True
                if verbose and episode < num_episodes - 1:
                    print(f"  [!] Episode {episode + 1}: NaN/Inf detected! loss={loss}")
                loss = 0.0  # Replace with 0 so plotting can continue

            state = next_state
            episode_reward += reward
            episode_loss += loss
            episode_grad_norm += agent.last_grad_norm
            actions_taken[action] = actions_taken.get(action, 0) + 1
            steps += 1

            if done or truncated:
                break

        # Decay the exploration rate
        agent.decay_epsilon()

        # Record diagnostic data
        reward_history.append(episode_reward)
        loss_history.append(episode_loss / max(steps, 1))
        grad_norm_history.append(episode_grad_norm / max(steps, 1))
        action_counts[0].append(actions_taken.get(0, 0))
        action_counts[1].append(actions_taken.get(1, 0))

        # Periodically update the target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_loss = np.mean(loss_history[-50:])
            avg_grad = np.mean(grad_norm_history[-50:])
            print(
                f"  Episode {episode + 1:4d}/{num_episodes} | "
                f"Reward: {avg_reward:6.1f} | "
                f"Loss: {avg_loss:8.4f} | "
                f"Grad norm: {avg_grad:8.2f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    return {
        'label': label,
        'rewards': reward_history,
        'losses': loss_history,
        'grad_norms': grad_norm_history,
        'action_counts': action_counts,
        'nan_detected': nan_detected,
    }


# ==========================================
# Part 4: 4-step debugging methodology
# ==========================================
def four_step_diagnosis(results):
    """
    4-step debugging methodology: systematically diagnose training issues

    Step 1: Check the loss curve (is it exploding/vanishing?)
    Step 2: Check the reward curve (is the agent improving?)
    Step 3: Check the action distribution (is the agent exploring?)
    Step 4: Check the gradient norm (is the gradient stable?)
    """
    label = results['label']
    losses = results['losses']
    rewards = results['rewards']
    action_counts = results['action_counts']
    grad_norms = results['grad_norms']

    print(f"\n{'=' * 60}")
    print(f"  4-step diagnostic report -- {label}")
    print(f"{'=' * 60}")

    # -- Step 1: check the loss curve --
    print(f"\n  Step 1: Check the loss curve")
    print(f"  {'─' * 40}")

    # Filter out NaN/Inf for statistics
    valid_losses = [l for l in losses if not np.isnan(l) and not np.isinf(l)]

    if len(valid_losses) == 0:
        print(f"    All loss values are NaN/Inf -> training has completely collapsed!")
        print(f"    Most likely cause: learning rate too high")
        print(f"    Fix: lower the learning rate to 1e-3 or smaller")
    else:
        avg_loss = np.mean(valid_losses)
        max_loss = np.max(valid_losses)
        min_loss = np.min(valid_losses)

        print(f"    Loss range: [{min_loss:.4f}, {max_loss:.4f}]")
        print(f"    Average loss: {avg_loss:.4f}")

        if max_loss > 1000:
            print(f"    ⚠️  Loss explosion! Max loss {max_loss:.1f} is far outside the normal range")
            print(f"    Possible causes: learning rate too high, or missing gradient clipping")
        elif max_loss > 100:
            print(f"    ⚠️  Loss is on the high side, some instability risk")
        else:
            print(f"    ✓  Loss is within a reasonable range")

    # -- Step 2: check the reward curve --
    print(f"\n  Step 2: Check the reward curve")
    print(f"  {'─' * 40}")

    n = len(rewards)
    quarter = max(n // 4, 1)
    early_reward = np.mean(rewards[:quarter])
    late_reward = np.mean(rewards[-quarter:])
    reward_change = late_reward - early_reward

    print(f"    Average reward, early phase: {early_reward:.1f}")
    print(f"    Average reward, late phase: {late_reward:.1f}")
    print(f"    Trend: {'↑' if reward_change > 0 else '↓'} {abs(reward_change):.1f}")

    if reward_change > 10:
        print(f"    ✓  Reward is rising, the agent is learning")
    elif reward_change > 0:
        print(f"    ~  Reward is rising slightly, learning is slow")
    else:
        print(f"    ⚠️  Reward is falling or stagnant, the agent is not learning")
        print(f"    Possible causes: insufficient exploration, an unsuitable learning rate, or the policy has degenerated")

    # -- Step 3: check the action distribution --
    print(f"\n  Step 3: Check the action distribution")
    print(f"  {'─' * 40}")

    # Compute the action distribution over the last 50 episodes
    last_n = min(50, n)
    left_actions = np.sum(action_counts[0][-last_n:])
    right_actions = np.sum(action_counts[1][-last_n:])
    total_actions = left_actions + right_actions

    if total_actions > 0:
        left_ratio = left_actions / total_actions
        right_ratio = right_actions / total_actions
        print(f"    Action distribution over the last {last_n} episodes:")
        print(f"      Push left: {left_ratio:.1%} ({left_actions} times)")
        print(f"      Push right: {right_ratio:.1%} ({right_actions} times)")

        if left_ratio > 0.95 or right_ratio > 0.95:
            print(f"    ⚠️  Action distribution is extremely skewed! The policy has degenerated (almost always picks one action)")
            print(f"    Possible cause: epsilon decays too fast or does not decay at all")
        elif left_ratio > 0.8 or right_ratio > 0.8:
            print(f"    ~  Action distribution is somewhat skewed, but still within an acceptable range")
        else:
            print(f"    ✓  Action distribution is fairly even, exploration looks normal")
    else:
        print(f"    Not enough data")

    # -- Step 4: check the gradient norm --
    print(f"\n  Step 4: Check the gradient norm")
    print(f"  {'─' * 40}")

    valid_grads = [g for g in grad_norms if not np.isnan(g) and not np.isinf(g)]

    if len(valid_grads) == 0:
        print(f"    All gradient norms are NaN/Inf -> the gradient has fully exploded!")
    else:
        avg_grad = np.mean(valid_grads)
        max_grad = np.max(valid_grads)

        print(f"    Average gradient norm: {avg_grad:.2f}")
        print(f"    Maximum gradient norm: {max_grad:.2f}")

        if max_grad > 1000:
            print(f"    ⚠️  Gradient explosion! Max gradient norm {max_grad:.1f}")
            print(f"    Fix: add gradient clipping via torch.nn.utils.clip_grad_norm_")
        elif max_grad > 100:
            print(f"    ~  Gradient is occasionally large, consider adding clipping as a precaution")
        else:
            print(f"    ✓  Gradient norm is within a reasonable range")

    print(f"{'=' * 60}")


# ==========================================
# Part 5: Main experiment flow
# ==========================================
def run_all_scenarios():
    """
    Run the training experiments for all scenarios

    Scenario 1: Learning rate too high (lr=1.0)
    Scenario 2: Missing gradient clipping
    Scenario 3: Epsilon not decaying (fixed epsilon=1.0)
    Scenario 4: Correct training after the fixes
    """
    NUM_EPISODES = 300

    print("=" * 60)
    print("  Appendix A: Training collapse diagnosis and fix experiments")
    print("=" * 60)
    print(f"  Each scenario trains for {NUM_EPISODES} episodes")
    print(f"  The first 3 scenarios deliberately trigger a failure; the 4th shows the correct configuration")
    print("=" * 60)

    # -- Scenario 1: learning rate too high --
    print("\n" + "=" * 60)
    print("  Scenario 1: Learning rate too high (lr=1.0)")
    print("  The normal learning rate is 1e-3; here it is deliberately set to 1.0 (1000x higher)")
    print("  Expected: loss explosion, possibly NaN very soon")
    print("=" * 60)

    env1 = gym.make("CartPole-v1")
    state_dim = env1.observation_space.shape[0]
    action_dim = env1.action_space.n

    agent_lr_high = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1.0,               # Learning rate too high! Normally 1e-3
        clip_grad=True,        # Keep clipping enabled to isolate the pure learning-rate issue
        clip_max_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    results_lr_high = train_scenario(agent_lr_high, env1, "lr=1.0 (too high)")

    # -- Scenario 2: missing gradient clipping --
    print("\n" + "=" * 60)
    print("  Scenario 2: Missing gradient clipping")
    print("  Gradient clipping disabled, normal learning rate")
    print("  Expected: occasional gradient explosions, unstable training")
    print("=" * 60)

    env2 = gym.make("CartPole-v1")
    agent_no_clip = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,              # Normal learning rate
        clip_grad=False,       # No gradient clipping!
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    results_no_clip = train_scenario(agent_no_clip, env2, "No gradient clipping")

    # -- Scenario 3: epsilon not decaying --
    print("\n" + "=" * 60)
    print("  Scenario 3: Epsilon not decaying (fixed epsilon=1.0)")
    print("  The agent always explores fully at random, never exploiting what it has learned")
    print("  Expected: the reward curve stays low and noisy forever, never converges")
    print("=" * 60)

    env3 = gym.make("CartPole-v1")
    agent_no_decay = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,
        clip_grad=True,
        clip_max_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=1.0,       # Epsilon does not decay! Always fully random
        epsilon_decay=1.0,     # Decay factor of 1.0 = no decay
    )
    results_no_decay = train_scenario(agent_no_decay, env3, "epsilon=1.0 (no decay)")

    # -- Scenario 4: correct training after the fixes --
    print("\n" + "=" * 60)
    print("  Scenario 4: Correct training after the fixes")
    print("  Appropriate learning rate + gradient clipping + epsilon decay")
    print("  Expected: stable learning, reward steadily increasing")
    print("=" * 60)

    env4 = gym.make("CartPole-v1")
    agent_correct = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,              # Appropriate learning rate
        clip_grad=True,        # Gradient clipping enabled
        clip_max_norm=10.0,    # Clipping threshold
        epsilon_start=1.0,     # Initial epsilon
        epsilon_end=0.01,      # Final epsilon
        epsilon_decay=0.995,   # Decay factor
    )
    results_correct = train_scenario(agent_correct, env4, "Correct configuration")

    return [results_lr_high, results_no_clip, results_no_decay, results_correct]


# ==========================================
# Part 6: Diagnostic visualization
# ==========================================
def plot_diagnosis(all_results):
    """
    Plot the diagnostic comparison across the 4 scenarios

    A 4-row x 4-column grid:
    - Row 1: loss curve (did it explode?)
    - Row 2: reward curve (is it improving?)
    - Row 3: action distribution (is it exploring?)
    - Row 4: gradient norm (is it stable?)

    Each column is a scenario, giving a direct visual comparison of the failure and the fix.
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("Training collapse diagnosis: the 4-step debugging methodology",
                 fontsize=18, fontweight='bold', y=1.01)

    colors = ['#e74c3c', '#e67e22', '#9b59b6', '#27ae60']
    titles = [r['label'] for r in all_results]

    for col, (results, color, title) in enumerate(zip(all_results, colors, titles)):
        losses = results['losses']
        rewards = results['rewards']
        action_counts = results['action_counts']
        grad_norms = results['grad_norms']

        # Handle NaN/Inf: replace with None so the plot breaks there
        safe_losses = [l if not (np.isnan(l) or np.isinf(l)) else None for l in losses]
        safe_grads = [g if not (np.isnan(g) or np.isinf(g)) else None for g in grad_norms]

        # -- Row 1: loss curve --
        ax1 = axes[0, col]
        # Plot only the valid values
        valid_x = [i for i, l in enumerate(safe_losses) if l is not None]
        valid_y = [l for l in safe_losses if l is not None]
        if valid_y:
            ax1.plot(valid_x, valid_y, color=color, alpha=0.6, linewidth=0.8)
            # Take the log of the loss to more clearly see the explosion trend
            log_losses = [np.log10(max(l, 1e-8)) for l in valid_y]
            ax1.plot(valid_x, log_losses, color='navy', alpha=0.8, linewidth=1.5,
                     label='log10(loss)')
            ax1.legend(fontsize=8)
        ax1.set_title(f"{title}\nLoss curve", fontsize=11)
        ax1.set_ylabel('Loss / log10(loss)')
        ax1.grid(True, alpha=0.3)

        # -- Row 2: reward curve --
        ax2 = axes[1, col]
        ax2.plot(rewards, color=color, alpha=0.3, linewidth=0.8)
        window = 20
        if len(rewards) >= window:
            moving_avg = [
                np.mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            ax2.plot(moving_avg, color='navy', linewidth=2, label='Moving average')
            ax2.legend(fontsize=8)
        ax2.set_title(f"Reward curve", fontsize=11)
        ax2.set_ylabel('Cumulative reward')
        ax2.grid(True, alpha=0.3)

        # -- Row 3: action distribution --
        ax3 = axes[2, col]
        left_counts = np.array(action_counts[0], dtype=float)
        right_counts = np.array(action_counts[1], dtype=float)
        total_counts = left_counts + right_counts
        total_counts = np.where(total_counts == 0, 1, total_counts)  # Avoid division by zero

        left_ratio = left_counts / total_counts
        right_ratio = right_counts / total_counts

        ax3.fill_between(range(len(left_ratio)), 0, left_ratio,
                         color='#3498db', alpha=0.6, label='Push left')
        ax3.fill_between(range(len(right_ratio)), left_ratio, 1,
                         color='#e74c3c', alpha=0.6, label='Push right')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title(f"Action distribution", fontsize=11)
        ax3.set_ylabel('Action ratio')
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, alpha=0.3)

        # -- Row 4: gradient norm --
        ax4 = axes[3, col]
        valid_gx = [i for i, g in enumerate(safe_grads) if g is not None]
        valid_gy = [g for g in safe_grads if g is not None]
        if valid_gy:
            ax4.plot(valid_gx, valid_gy, color=color, alpha=0.6, linewidth=0.8)
            # Take the log of the gradient norm
            log_grads = [np.log10(max(g, 1e-8)) for g in valid_gy]
            ax4.plot(valid_gx, log_grads, color='navy', alpha=0.8, linewidth=1.5,
                     label='log10(grad_norm)')
            ax4.axhline(y=np.log10(10), color='red', linestyle='--', alpha=0.5,
                        label='Clipping threshold (10)')
            ax4.legend(fontsize=8)
        ax4.set_title(f"Gradient norm", fontsize=11)
        ax4.set_xlabel('Training episode')
        ax4.set_ylabel('Gradient norm / log10(gradient norm)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/training_collapse_diagnosis.png", dpi=150, bbox_inches='tight')
    print("\nDiagnostic plot saved to output/training_collapse_diagnosis.png")
    plt.show()


# ==========================================
# Part 7: Summary of fixes
# ==========================================
def print_fix_summary():
    """
    Print a summary of common causes of training collapse and their fixes
    """
    print("\n" + "=" * 60)
    print("  Training collapse fix cheat sheet")
    print("=" * 60)

    fixes = [
        {
            "故障": "Loss explosion (loss suddenly becomes NaN/Inf)",
            "症状": "The loss curve suddenly spikes to an astronomical value",
            "原因": "Learning rate too high",
            "修复": "Lower the learning rate (recommended 1e-4 ~ 1e-3), use a learning rate scheduler",
            "代码": "optimizer = Adam(params, lr=1e-3)  # not 1.0",
        },
        {
            "故障": "Gradient explosion (gradient norm keeps growing)",
            "症状": "Gradient norm > 1000 and not converging",
            "原因": "Missing gradient clipping",
            "修复": "Add gradient clipping via clip_grad_norm_",
            "代码": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)",
        },
        {
            "故障": "Never converges (reward stays low and noisy)",
            "症状": "No upward trend in reward even after hundreds of episodes",
            "原因": "Exploration rate epsilon does not decay",
            "修复": "Use an exponential or linear epsilon decay schedule",
            "代码": "epsilon = max(0.01, epsilon * 0.995)  # decay every episode",
        },
        {
            "故障": "Policy degeneration (only picks one action)",
            "症状": "Extremely skewed action distribution (e.g. 99:1)",
            "原因": "Epsilon decays too fast or its initial value is too low",
            "修复": "Slow down epsilon decay, raise the final epsilon floor",
            "代码": "epsilon = max(0.05, epsilon * 0.999)  # slower decay",
        },
    ]

    for i, fix in enumerate(fixes, 1):
        print(f"\n  Failure {i}: {fix['故障']}")
        print(f"    Symptom: {fix['症状']}")
        print(f"    Cause: {fix['原因']}")
        print(f"    Fix: {fix['修复']}")
        print(f"    Code: {fix['代码']}")

    # Print the debugging methodology
    print(f"\n{'=' * 60}")
    print("  4-step debugging methodology")
    print(f"{'=' * 60}")

    steps = [
        ("Step 1: Check the loss curve",
         "Observe whether the loss fluctuates within a reasonable range.",
         "If loss > 1000 or NaN appears -> a learning-rate problem or gradient explosion.",
         "Fix: lower the learning rate, add gradient clipping."),
        ("Step 2: Check the reward curve",
         "Observe whether the reward has an upward trend.",
         "If the reward never changes -> the policy is not learning.",
         "Fix: check the exploration rate, learning rate, and reward function design."),
        ("Step 3: Check the action distribution",
         "Tally how often the agent picks each action.",
         "If one action accounts for > 95% -> the policy has degenerated.",
         "Fix: adjust the epsilon schedule to ensure enough exploration."),
        ("Step 4: Check the gradient norm",
         "Monitor the trend of the gradient norm over time.",
         "If the gradient norm stays > 100 -> gradient explosion.",
         "Fix: add or tighten gradient clipping."),
    ]

    for title, line1, line2, line3 in steps:
        print(f"\n  {title}")
        print(f"    {line1}")
        print(f"    {line2}")
        print(f"    {line3}")

    # Full diagnostic checklist
    print(f"\n{'=' * 60}")
    print("  Diagnostic checklist")
    print(f"{'=' * 60}")

    checklist = [
        "[ ] Is the loss within a reasonable range (not NaN/Inf/astronomical)?",
        "[ ] Does the loss trend downward (rather than continuing to rise)?",
        "[ ] Is the reward curve steadily rising?",
        "[ ] Is the action distribution reasonable (not just one action)?",
        "[ ] Is the gradient norm stable (not continuously growing)?",
        "[ ] Is epsilon decaying on schedule (not fixed)?",
        "[ ] Is the learning rate within a reasonable range (1e-4 ~ 1e-3)?",
        "[ ] Is gradient clipping enabled (max_norm around 10)?",
    ]

    for item in checklist:
        print(f"  {item}")

    print("=" * 60)


# ==========================================
# Program entry point
# ==========================================
if __name__ == "__main__":
    # Run all scenarios
    all_results = run_all_scenarios()

    # Run the 4-step diagnosis on each scenario
    for results in all_results:
        four_step_diagnosis(results)

    # Plot the diagnostic comparison
    plot_diagnosis(all_results)

    # Print the summary of fixes
    print_fix_summary()
