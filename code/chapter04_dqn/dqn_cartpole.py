"""
Chapter 4: Deep Q-Network (DQN) -- implemented from scratch
Trains a complete DQN agent on the CartPole-v1 environment

Core idea:
    Approximate Q(s, a) with a neural network, stabilizing training via
    experience replay and a target network. This is the milestone work
    published by DeepMind in 2015 that kicked off the deep reinforcement
    learning era.

Key components:
    1. Q network: approximates the action-value function with a
       multilayer perceptron (MLP)
    2. Experience replay buffer: breaks data correlation, improves
       sample efficiency
    3. Target network: updated with a delay, provides a stable training
       target
    4. Epsilon-greedy exploration: balances exploration and exploitation

How to run:
    python dqn_cartpole.py
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

# Create output directory
os.makedirs("output", exist_ok=True)


# ==========================================
# Part 1: Q network
# ==========================================
class QNetwork(nn.Module):
    """
    Q network: maps states to Q values for each action
    Structure: state_dim -> 128 -> 128 -> action_dim

    The input is a state vector (4-dimensional in CartPole), and the
    output is the estimated Q value for each action (2-dimensional in
    CartPole, corresponding to left/right).
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # First layer: state -> 128
            nn.ReLU(),
            nn.Linear(128, 128),          # Second layer: 128 -> 128
            nn.ReLU(),
            nn.Linear(128, action_dim),   # Output layer: 128 -> number of actions
        )

    def forward(self, x):
        """Forward pass: takes a state, outputs the Q value for each action"""
        return self.net(x)


# ==========================================
# Part 2: Experience replay buffer
# ==========================================
class ReplayBuffer:
    """
    Experience replay buffer: stores and samples training data

    Why do we need experience replay?
    - Deep learning assumes data is independent and identically
      distributed (i.i.d.)
    - But in reinforcement learning, consecutive transitions
      (s, a, r, s') are highly correlated
    - Experience replay breaks this correlation by randomly shuffling
      the sampling order
    - It also improves data utilization (a single experience can be
      used multiple times)
    """

    def __init__(self, capacity=10000):
        """Implemented with a deque, which automatically discards old data once full"""
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one transition (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of data to break temporal correlation"""
        batch = random.sample(self.buffer, batch_size)
        # Unpack and convert to numpy arrays for easy conversion to tensors
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
# Part 3: DQN agent
# ==========================================
class DQNAgent:
    """
    DQN agent: combines the Q network, target network, experience
    replay, and epsilon-greedy policy

    DQN's three main innovations:
    1. Experience Replay: breaks data correlation
    2. Target Network: provides a stable training target
    3. Epsilon-Greedy: balances exploration and exploitation
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor: decay rate for future rewards

        # Q network: the main network, updated in real time
        self.q_net = QNetwork(state_dim, action_dim)
        # Target network: periodically copies weights from the Q network,
        # used to compute stable target values
        self.target_net = QNetwork(state_dim, action_dim)
        # Sync the target network with the Q network at initialization
        self.target_net.load_state_dict(self.q_net.state_dict())
        # The target network doesn't need gradients
        self.target_net.eval()

        # Optimizer: only optimizes the Q network's parameters
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Experience replay buffer
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """
        Epsilon-greedy action selection

        With probability epsilon, choose a random action (exploration);
        with probability 1-epsilon, choose the action with the highest
        Q value (exploitation).
        """
        if random.random() < epsilon:
            # Explore: pick a random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: pick the action with the highest Q value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        """
        Sample from experience replay and update the Q network

        Core DQN update formula:
            target = r + gamma * max_a' Q_target(s', a')
            loss = (target - Q(s, a))^2

        Note: the target value is computed with target_net and the Q
        value with q_net, which avoids the instability of "chasing your
        own tail".
        """
        if len(self.buffer) < batch_size:
            return 0.0  # Don't update if there isn't enough data yet

        # Randomly sample from experience replay
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q(s, a): the current network's estimate
        # gather selects the Q value corresponding to each action
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target value: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            # Use the target network to compute the max Q value for the next state
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            # When done = 1, there is no future reward
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        # Mean squared error loss
        loss = nn.MSELoss()(q_values, targets)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy the Q network's weights to the target network (hard update)"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        """Save the model"""
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")


# ==========================================
# Part 4: Training loop
# ==========================================
def train():
    """The full DQN training pipeline"""

    # Hyperparameter settings
    NUM_EPISODES = 500       # Number of training episodes
    BATCH_SIZE = 64          # Batch size per update
    EPSILON_START = 1.0      # Initial exploration rate
    EPSILON_END = 0.01       # Final exploration rate
    EPSILON_DECAY = 0.995    # Exploration rate decay coefficient
    TARGET_UPDATE_FREQ = 10  # Target network update frequency (every N episodes)

    # Create the environment and the agent
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    print("=" * 60)
    print("  Deep Q-Network (DQN) -- CartPole-v1 training")
    print("=" * 60)
    print(f"  State space dimension: {state_dim}")
    print(f"  Action space dimension: {action_dim}")
    print(f"  Number of training episodes: {NUM_EPISODES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Initial exploration rate: {EPSILON_START}")
    print(f"  Target network update frequency: every {TARGET_UPDATE_FREQ} episodes")
    print("=" * 60)

    # Record training data
    reward_history = []
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # Select an action (epsilon-greedy policy)
            action = agent.select_action(state, epsilon)
            # Execute the action
            next_state, reward, done, truncated, _ = env.step(action)
            # Store into experience replay
            agent.buffer.push(state, action, reward, next_state, float(done))
            # Update the Q network
            agent.update(BATCH_SIZE)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Decay the exploration rate
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(episode_reward)

        # Periodically update the target network
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            recent = reward_history[-50:]
            avg_reward = np.mean(recent)
            print(
                f"  Episode {episode + 1:4d}/{NUM_EPISODES} | "
                f"Avg reward (last 50): {avg_reward:6.1f} | "
                f"epsilon: {epsilon:.3f}"
            )

    env.close()

    # ==========================================
    # Part 5: Training result visualization
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_history, alpha=0.3, color='steelblue', label='Per-episode reward')

    # Plot the moving average curve for a clearer view of the trend
    window = 20
    if len(reward_history) >= window:
        moving_avg = [
            np.mean(reward_history[max(0, i - window): i + 1])
            for i in range(len(reward_history))
        ]
        ax.plot(moving_avg, color='red', linewidth=2,
                label=f'{window}-episode moving average')

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Cumulative reward')
    ax.set_title('DQN training curve -- CartPole-v1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/dqn_cartpole_training.png", dpi=150)
    print("\nTraining curve saved to output/dqn_cartpole_training.png")
    plt.show()

    # ==========================================
    # Part 6: Test the trained agent
    # ==========================================
    print("\n" + "=" * 60)
    print("  Test phase: running 10 episodes")
    print("=" * 60)

    test_env = gym.make("CartPole-v1")
    test_rewards = []

    for ep in range(10):
        state, _ = test_env.reset()
        total_reward = 0

        while True:
            # No exploration during testing, always pick the optimal action
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward

            if done or truncated:
                break

        test_rewards.append(total_reward)
        print(f"  Test episode {ep + 1:2d}: score = {total_reward:.0f}")

    print("-" * 60)
    print(f"  Average test score: {np.mean(test_rewards):.1f} +/- {np.std(test_rewards):.1f}")
    print("=" * 60)

    test_env.close()

    # Save the model
    agent.save("output/dqn_cartpole.pth")

    return reward_history


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    train()
