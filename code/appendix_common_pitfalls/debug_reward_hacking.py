"""
Appendix A - Common Pitfalls and Solutions: Diagnosing and Fixing Reward Hacking

This script demonstrates one of the most common and dangerous pitfalls in reinforcement
learning: reward hacking.

What is reward hacking?
    The agent learns to "game the system" -- exploiting a flaw in the reward function's
    design to earn a high score, without actually accomplishing the task we really wanted.

Typical symptoms:
    - The reward curve keeps rising (training looks like it's succeeding)
    - But actual task performance is declining instead (or the agent stops doing the
      real task altogether)

This script includes three scenarios:
    Scenario 1: Passive survival -- the agent in CartPole learns to "just stand there"
        to collect reward
    Scenario 2: Bad reward shaping -- an ill-conceived intermediate reward causes
        unintended behavior
    Scenario 3: Correct reward design -- how to avoid the problems above

How to run:
    python debug_reward_hacking.py
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
# Part 1: Utility components (Q-network, replay buffer, agent)
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
        """Implemented with a deque; once full, old data is dropped automatically"""
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one transition (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of data to break temporal correlation"""
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


class DQNAgent:
    """
    DQN agent: combines the Q-network, target network, replay buffer, and
    epsilon-greedy policy

    To keep the code reusable, we use a single unified agent structure here;
    the different scenarios change the reward function by modifying the
    environment wrapper instead.
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        """Sample from the replay buffer and update the Q-network"""
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
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy the Q-network's weights to the target network (hard update)"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# Part 2: Custom environment wrappers (simulating reward hacking)
# ==========================================
class HackedRewardWrapper(gym.Wrapper):
    """
    Passive-survival reward wrapper -- simulates a "reward hacking" scenario

    Original CartPole reward: +1 per step until the pole falls.
    This is actually correct by design: the agent needs to keep the pole
    upright in order to score high.

    But if we additionally reward "time survived" without requiring anything
    about the pole's angle, the agent might learn a strange strategy: making
    the pole swing rapidly back and forth, or finding some quirk of the
    physics engine to prolong survival time.

    Here we simulate a more extreme case: giving an extra reward for states
    where "the pole is nearly horizontal" (i.e. rewarding it even when it's
    about to fall), which encourages the agent to linger near the edge state
    instead of actually staying balanced.
    """

    def __init__(self, env, survival_bonus=2.0):
        super().__init__(env)
        self.survival_bonus = survival_bonus

    def step(self, action):
        """
        Modify the reward function:
        - Original reward: +1 per step
        - Hacked reward: an extra bonus just for "surviving", regardless of
          the pole's angle
        - As long as the episode hasn't ended, a large bonus is given ->
          the agent learns to "stall"
        """
        next_state, reward, done, truncated, info = self.env.step(action)

        # Hacked reward: no matter what it does, as long as it's still alive
        # it gets an extra reward. This lets the agent discover that "it
        # doesn't matter what you do, as long as you're alive"
        hacked_reward = reward + self.survival_bonus

        return next_state, hacked_reward, done, truncated, info


class BadShapingWrapper(gym.Wrapper):
    """
    Bad reward-shaping wrapper -- simulates a "reward shaping gone wrong" scenario

    Scenario: we want the agent to push the cart to the center of the screen.
    We designed an intermediate reward for "the closer to center, the better".
    The problem is: the agent learns to oscillate back and forth near the
    center, instead of stably keeping its balance!

    This situation is quite common in practice: what you think is "helping"
    can actually be "misleading".
    """

    def __init__(self, env, position_reward_weight=5.0):
        super().__init__(env)
        self.position_reward_weight = position_reward_weight

    def step(self, action):
        """
        Modify the reward function:
        - Original reward: +1 per step (keep the pole upright)
        - Shaped reward: an extra bonus for "the cart being close to center"

        The problem is that position_reward_weight is too large, causing the
        agent to focus mainly on "pushing the cart to center" while ignoring
        "keeping the pole upright". Result: the cart does end up centered,
        but the pole falls over.
        """
        next_state, reward, done, truncated, info = self.env.step(action)

        # next_state[0] is the cart's position, roughly in the range [-2.4, 2.4]
        # Reward being "close to center"
        position = next_state[0]
        position_reward = self.position_reward_weight * (1.0 - abs(position) / 2.4)

        shaped_reward = reward + position_reward

        return next_state, shaped_reward, done, truncated, info


# ==========================================
# Part 3: Unified training function
# ==========================================
def train_agent(env, label, num_episodes=300, batch_size=64):
    """
    Train a DQN agent in the given environment, returning the training curves
    and task metrics

    Args:
        env: the training environment (possibly wrapped)
        label: scenario name (for printing and plotting)
        num_episodes: number of training episodes
        batch_size: batch size
    Returns:
        reward_history: list of per-episode rewards (the modified reward)
        task_score_history: list of per-episode true task scores (sum of the
            original reward)
        loss_history: history of loss values
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    epsilon = EPSILON_START
    reward_history = []
    task_score_history = []  # True task score (original reward)
    loss_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_task_score = 0.0  # Tracks only the original CartPole reward
        episode_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state, epsilon)

            # Record the modified reward (used for training)
            next_state, reward, done, truncated, _ = env.step(action)

            # Also compute the true task score (excluding the hacked bonus)
            # The original CartPole reward is 1.0 per step
            original_reward = 1.0

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size)

            state = next_state
            episode_reward += reward
            episode_task_score += original_reward
            episode_loss += loss
            steps += 1

            if done or truncated:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(episode_reward)
        task_score_history.append(episode_task_score)
        loss_history.append(episode_loss / max(steps, 1))

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_task = np.mean(task_score_history[-50:])
            print(
                f"  [{label}] Episode {episode + 1:4d}/{num_episodes} | "
                f"Reward (modified): {avg_reward:7.1f} | "
                f"Task score (original): {avg_task:5.1f}"
            )

    env.close()
    return reward_history, task_score_history, loss_history


# ==========================================
# Part 4: Diagnostic function -- detect reward hacking
# ==========================================
def diagnose_reward_hacking(reward_history, task_score_history, label):
    """
    Reward hacking diagnostic tool

    Core idea:
        - The signature of reward hacking is "reward rising, but task
          performance declining"
        - We judge this by looking at the correlation between the reward
          curve and the task score curve
        - If both are positively correlated and rising -> normal
        - If reward rises but task score falls -> hacking!

    Args:
        reward_history: the modified reward curve
        task_score_history: the true task score curve
        label: scenario name
    Returns:
        is_hacked: whether hacking was detected (True/False)
    """
    print(f"\n{'=' * 60}")
    print(f"  Reward hacking diagnostic report -- {label}")
    print(f"{'=' * 60}")

    # Compare the trend between the first quarter and the last quarter
    n = len(reward_history)
    quarter = max(n // 4, 1)

    # Average reward in the early and late segments
    early_reward = np.mean(reward_history[:quarter])
    late_reward = np.mean(reward_history[-quarter:])
    reward_trend = late_reward - early_reward

    # Average task score in the early and late segments
    early_task = np.mean(task_score_history[:quarter])
    late_task = np.mean(task_score_history[-quarter:])
    task_trend = late_task - early_task

    print(f"  Average reward, early phase (modified): {early_reward:.1f}")
    print(f"  Average reward, late phase (modified): {late_reward:.1f}")
    print(f"  Reward trend: {'↑' if reward_trend > 0 else '↓'} {abs(reward_trend):.1f}")
    print(f"  ─────────────────────────────")
    print(f"  Average task score, early phase (original): {early_task:.1f}")
    print(f"  Average task score, late phase (original): {late_task:.1f}")
    print(f"  Task score trend: {'↑' if task_trend > 0 else '↓'} {abs(task_trend):.1f}")

    # Diagnosis: reward rising but task performance falling = hacking
    is_hacked = (reward_trend > 0) and (task_trend < 0)

    # Compute the correlation coefficient between reward and task score
    if n > 10:
        correlation = np.corrcoef(reward_history, task_score_history)[0, 1]
        print(f"  Correlation between reward and task score: {correlation:.3f}")
    else:
        correlation = 1.0

    print(f"  ─────────────────────────────")
    if is_hacked:
        print(f"  ⚠️  Reward hacking detected!")
        print(f"  The reward is rising, but true task performance is actually declining.")
        print(f"  The agent may have learned to exploit a flaw in the reward function.")
        print(f"  Recommendation: revisit the reward function design.")
    elif correlation < 0.3:
        print(f"  ⚠️  Correlation between reward and task score is very low ({correlation:.3f})")
        print(f"  The reward function may not correctly reflect the task objective.")
        print(f"  Recommendation: check whether the reward function is aligned with the true goal.")
    else:
        print(f"  ✓  Reward and task performance are consistent, no hacking detected.")

    print(f"{'=' * 60}")

    return is_hacked


# ==========================================
# Part 5: Main experiment flow
# ==========================================
def run_all_experiments():
    """
    Run the comparison experiment across all three scenarios

    Scenario 1: Passive survival hacking (HackedRewardWrapper)
    Scenario 2: Bad reward shaping (BadShapingWrapper)
    Scenario 3: Correct design (original CartPole reward)
    """

    NUM_EPISODES = 300  # Train each scenario for 300 episodes

    print("=" * 60)
    print("  Appendix A: Reward Hacking experiments")
    print("=" * 60)
    print(f"  Each scenario trains for {NUM_EPISODES} episodes")
    print(f"  Hacking is diagnosed by comparing the 'modified reward' against the 'true task score'")
    print("=" * 60)

    # -- Scenario 1: passive survival hacking --
    print("\n" + "─" * 60)
    print("  Scenario 1: Passive-survival reward hacking")
    print("  Problem: an extra reward is given for merely staying alive, no matter what")
    print("  Expected: the agent learns to stall rather than genuinely stay balanced")
    print("─" * 60)

    hacked_env = HackedRewardWrapper(
        gym.make("CartPole-v1"),
        survival_bonus=2.0,  # Extra survival bonus
    )
    hacked_rewards, hacked_task_scores, hacked_losses = train_agent(
        hacked_env, "Hacked reward", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(hacked_rewards, hacked_task_scores, "Scenario 1 - Passive survival hacking")

    # -- Scenario 2: bad reward shaping --
    print("\n" + "─" * 60)
    print("  Scenario 2: Bad reward shaping")
    print("  Problem: over-rewarding 'being near center' causes 'staying balanced' to be neglected")
    print("  Expected: the cart really does end up near center, but the pole may fall")
    print("─" * 60)

    bad_shaping_env = BadShapingWrapper(
        gym.make("CartPole-v1"),
        position_reward_weight=5.0,  # Position reward weight too large
    )
    shaping_rewards, shaping_task_scores, shaping_losses = train_agent(
        bad_shaping_env, "Bad shaping", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(shaping_rewards, shaping_task_scores, "Scenario 2 - Bad shaping")

    # -- Scenario 3: correct reward design --
    print("\n" + "─" * 60)
    print("  Scenario 3: Correct reward design (baseline)")
    print("  Uses the original CartPole reward: +1 per step until the pole falls")
    print("  This is a model example of a simple, effective reward design")
    print("─" * 60)

    correct_env = gym.make("CartPole-v1")
    correct_rewards, correct_task_scores, correct_losses = train_agent(
        correct_env, "Correct reward", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(correct_rewards, correct_task_scores, "Scenario 3 - Correct design")

    return (hacked_rewards, hacked_task_scores,
            shaping_rewards, shaping_task_scores,
            correct_rewards, correct_task_scores)


# ==========================================
# Part 6: Visual comparison
# ==========================================
def plot_comparison(hacked_rewards, hacked_task_scores,
                    shaping_rewards, shaping_task_scores,
                    correct_rewards, correct_task_scores):
    """
    Plot the comparison across the three scenarios

    Top row: the modified reward curves (they all look fine)
    Bottom row: the true task score curves (the hacking scenarios expose the problem)

    This "top vs. bottom" comparison is the standard method for diagnosing
    reward hacking.
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Reward hacking diagnosis: modified reward vs. true task score",
                 fontsize=16, fontweight='bold')

    scenarios = [
        ("Scenario 1: Passive survival hacking", hacked_rewards, hacked_task_scores, "#e74c3c"),
        ("Scenario 2: Bad reward shaping", shaping_rewards, shaping_task_scores, "#e67e22"),
        ("Scenario 3: Correct reward design", correct_rewards, correct_task_scores, "#27ae60"),
    ]

    window = 20  # Moving average window

    for col, (title, rewards, task_scores, color) in enumerate(scenarios):
        # -- Top row: the modified reward curve --
        ax_top = axes[0, col]
        ax_top.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= window:
            moving_avg = [
                np.mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            ax_top.plot(moving_avg, color=color, linewidth=2, label='Moving average')
        ax_top.set_title(f"{title}\nModified reward", fontsize=12)
        ax_top.set_xlabel('Training episode')
        ax_top.set_ylabel('Reward (modified)')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=9)

        # -- Bottom row: the true task score curve --
        ax_bot = axes[1, col]
        ax_bot.plot(task_scores, alpha=0.2, color=color)
        if len(task_scores) >= window:
            moving_avg_task = [
                np.mean(task_scores[max(0, i - window): i + 1])
                for i in range(len(task_scores))
            ]
            ax_bot.plot(moving_avg_task, color=color, linewidth=2, label='Moving average')
        ax_bot.set_title(f"{title}\nTrue task score", fontsize=12)
        ax_bot.set_xlabel('Training episode')
        ax_bot.set_ylabel('Task score (original)')
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("output/reward_hacking_diagnosis.png", dpi=150, bbox_inches='tight')
    print("\nDiagnostic plot saved to output/reward_hacking_diagnosis.png")
    plt.show()


# ==========================================
# Part 7: Summary of reward design principles
# ==========================================
def print_reward_design_principles():
    """
    Print the principles of correct reward design

    These principles are distilled from practical experience in reinforcement
    learning, drawing in particular on the reward-engineering guidance from
    OpenAI and DeepMind.
    """
    print("\n" + "=" * 60)
    print("  Reward design best practices (avoiding reward hacking)")
    print("=" * 60)

    principles = [
        ("1. Keep the reward simple",
         "The reward function should be as simple and direct as possible. CartPole's reward "
         "is just +1 per step, with no extra shaping term, yet it works best."),

        ("2. Make sure the reward is aligned with the goal",
         "The reward must accurately reflect the behavior you actually want. If a 'bad' "
         "behavior can earn a high reward, the agent will surely find and exploit it."),

        ("3. Avoid over-shaping",
         "Reward shaping is a double-edged sword. "
         "Poorly designed shaping not only fails to help learning, it can introduce new local optima. "
         "If shaping is necessary, use potential-based shaping, "
         "which does not change the optimal policy."),

        ("4. Monitor multiple metrics",
         "Never look at just one metric. Monitor reward, task score, and behavior "
         "distribution together. "
         "If reward rises while task performance falls, that indicates hacking."),

        ("5. Run a baseline first, then add complexity",
         "Start with the simplest possible reward function to establish a baseline and confirm "
         "that learning is happening at all. "
         "Then gradually add shaping terms, verifying at each step that it genuinely helps learning."),

        ("6. Run adversarial tests",
         "After training, test the agent from different initial states. "
         "If the agent performs unusually poorly in certain states, "
         "it may indicate it has learned some kind of 'cheating' strategy."),
    ]

    for title, description in principles:
        print(f"\n  {title}")
        print(f"    {description}")

    print("\n" + "=" * 60)
    print("  Diagnostic checklist: how to quickly spot reward hacking")
    print("=" * 60)

    checklist = [
        "[ ] Is the reward curve continuously rising?",
        "[ ] Is task performance (true score) also rising in sync?",
        "[ ] Is the correlation coefficient between reward and task performance greater than 0.5?",
        "[ ] Does the agent's behavior match expectations (rather than finding a strange strategy)?",
        "[ ] Does the agent still perform well under different initial conditions?",
        "[ ] Does the reward function contain any exploitable 'loophole'?",
    ]

    for item in checklist:
        print(f"  {item}")

    print("=" * 60)


# ==========================================
# Program entry point
# ==========================================
if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments()

    # Unpack the results
    (hacked_rewards, hacked_task_scores,
     shaping_rewards, shaping_task_scores,
     correct_rewards, correct_task_scores) = results

    # Plot the comparison
    plot_comparison(
        hacked_rewards, hacked_task_scores,
        shaping_rewards, shaping_task_scores,
        correct_rewards, correct_task_scores,
    )

    # Print the reward design principles
    print_reward_design_principles()
