"""
Chapter 4: Double DQN -- solving DQN's overestimation problem
Compares standard DQN and Double DQN on CartPole-v1

Background:
    When computing the target value, standard DQN uses the same network
    to both select the action and evaluate its Q value:
        target = r + gamma * max_a Q_target(s', a)
    This leads to systematic overestimation of Q values.

Solution -- Double DQN (Hasselt et al., 2016):
    Decouple "action selection" from "value evaluation":
    1. Use the Q network to select the optimal action: a* = argmax_a Q(s', a)
    2. Use the target network to evaluate that action's Q value: Q_target(s', a*)
    i.e.: target = r + gamma * Q_target(s', argmax_a Q(s', a))

    This effectively mitigates overestimation and improves training
    stability and final performance.

How to run:
    python double_dqn_cartpole.py
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
# Part 1: Q network (same as standard DQN)
# ==========================================
class QNetwork(nn.Module):
    """
    Q network: maps states to Q values for each action
    Structure: state_dim -> 128 -> 128 -> action_dim
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
        return self.net(x)


# ==========================================
# Part 2: Experience replay buffer (same as standard DQN)
# ==========================================
class ReplayBuffer:
    """Experience replay buffer: stores and samples training data"""

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
# Part 3: Standard DQN agent
# ==========================================
class DQNAgent:
    """
    Standard DQN agent

    Target value computation:
        target = r + gamma * max_a' Q_target(s', a')

    Note: the max operation is used for both "selecting the action" and
    "evaluating the Q value" -- this is the root cause of overestimation.
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
        """
        Standard DQN update

        Target value: r + gamma * Q_target(s').max()
        Uses the target network directly to take the max Q value.
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Standard DQN target value: take the max directly from the target network
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
        """Hard update of the target network"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# Part 4: Double DQN agent
# ==========================================
class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent (inherits from standard DQN)

    The only difference is how the target value is computed:
        Standard DQN: target = r + gamma * Q_target(s').max()
        Double DQN: target = r + gamma * Q_target(s')[argmax_a Q(s')]

    Intuition:
    - The Q network "nominates" the best action (action selection)
    - The target network "votes" on that action's value (value evaluation)
    - The two networks are independent, greatly reducing the chance of
      overestimation
    """

    def update(self, batch_size):
        """
        Double DQN update (the key difference is here!)

        Step by step:
        1. Use q_net to pick the optimal action for the next state:
           a* = argmax_a q_net(s')
        2. Use target_net to evaluate that action's Q value: Q_target(s', a*)
        3. Compute the target value: target = r + gamma * Q_target(s', a*)
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values (same as standard DQN)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # * Double DQN core: decouple action selection from value evaluation *
        with torch.no_grad():
            # Step 1: use the Q network to select the optimal action
            # q_net outputs Q values for all actions; argmax gives the best action index
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)

            # Step 2: use the target network to evaluate those actions' Q values
            # target_net picks out the Q value corresponding to the best action index
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)

            # Step 3: compute the target value
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()


# ==========================================
# Part 5: Generic training function
# ==========================================
def train_agent(agent, num_episodes=300, batch_size=64,
                epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay=0.995, target_update_freq=10):
    """
    Generic training function, works for both DQN and Double DQN

    Args:
        agent: a DQN or Double DQN agent
        the remaining args are training hyperparameters

    Returns:
        reward_history: list of cumulative rewards per episode
    """
    env = gym.make("CartPole-v1")
    reward_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update(batch_size)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        reward_history.append(episode_reward)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target()

    env.close()
    return reward_history


# ==========================================
# Part 6: Comparison experiment
# ==========================================
def main():
    """Run the comparison experiment between DQN and Double DQN"""

    # Training parameters
    NUM_EPISODES = 300
    BATCH_SIZE = 64
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    # Create an environment to get dimension info
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    env.close()

    print("=" * 60)
    print("  DQN vs Double DQN comparison -- CartPole-v1")
    print("=" * 60)
    print(f"  State space dimension: {state_dim}")
    print(f"  Action space dimension: {action_dim}")
    print(f"  Number of training episodes: {NUM_EPISODES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Discount factor: {GAMMA}")
    print("=" * 60)

    # ------------------------------------------
    # Train standard DQN
    # ------------------------------------------
    print("\n[1/2] Training standard DQN...")
    print("-" * 60)

    dqn_agent = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA)
    dqn_rewards = train_agent(
        dqn_agent,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    dqn_avg = np.mean(dqn_rewards[-50:])
    print(f"  DQN training complete! Average reward over last 50 episodes: {dqn_avg:.1f}")

    # ------------------------------------------
    # Train Double DQN
    # ------------------------------------------
    print("\n[2/2] Training Double DQN...")
    print("-" * 60)

    double_dqn_agent = DoubleDQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA)
    double_dqn_rewards = train_agent(
        double_dqn_agent,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    ddqn_avg = np.mean(double_dqn_rewards[-50:])
    print(f"  Double DQN training complete! Average reward over last 50 episodes: {ddqn_avg:.1f}")

    # ==========================================
    # Part 7: Comparison result visualization
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: raw reward curves ---
    ax1.plot(dqn_rewards, alpha=0.3, color='steelblue')
    ax1.plot(double_dqn_rewards, alpha=0.3, color='coral')

    # Plot moving averages
    window = 20
    dqn_ma = [np.mean(dqn_rewards[max(0, i - window): i + 1])
              for i in range(len(dqn_rewards))]
    ddqn_ma = [np.mean(double_dqn_rewards[max(0, i - window): i + 1])
               for i in range(len(double_dqn_rewards))]

    ax1.plot(dqn_ma, color='steelblue', linewidth=2, label='DQN')
    ax1.plot(ddqn_ma, color='coral', linewidth=2, label='Double DQN')
    ax1.set_xlabel('Training episode')
    ax1.set_ylabel('Cumulative reward')
    ax1.set_title('DQN vs Double DQN training curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: moving average comparison (clearer) ---
    ax2.plot(dqn_ma, color='steelblue', linewidth=2, label='DQN')
    ax2.plot(ddqn_ma, color='coral', linewidth=2, label='Double DQN')
    ax2.fill_between(
        range(len(dqn_ma)), dqn_ma, ddqn_ma,
        where=[d > dd for d, dd in zip(dqn_ma, ddqn_ma)],
        alpha=0.15, color='steelblue', label='DQN ahead'
    )
    ax2.fill_between(
        range(len(dqn_ma)), dqn_ma, ddqn_ma,
        where=[dd >= d for d, dd in zip(dqn_ma, ddqn_ma)],
        alpha=0.15, color='coral', label='Double DQN ahead'
    )
    ax2.set_xlabel('Training episode')
    ax2.set_ylabel('Cumulative reward (moving average)')
    ax2.set_title(f'{window}-episode moving average comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/dqn_vs_double_dqn.png", dpi=150)
    print("\nComparison plot saved to output/dqn_vs_double_dqn.png")
    plt.show()

    # ==========================================
    # Part 8: Final results summary
    # ==========================================
    print("\n" + "=" * 60)
    print("  Final results summary")
    print("=" * 60)

    print(f"\n  Standard DQN:")
    print(f"    Average reward over last 50 episodes: {dqn_avg:.1f}")
    print(f"    Highest episode reward: {max(dqn_rewards):.0f}")

    print(f"\n  Double DQN:")
    print(f"    Average reward over last 50 episodes: {ddqn_avg:.1f}")
    print(f"    Highest episode reward: {max(double_dqn_rewards):.0f}")

    print(f"\n  Difference (Double DQN - DQN): {ddqn_avg - dqn_avg:+.1f}")

    print("\n" + "-" * 60)
    print("  Key difference recap:")
    print("  Standard DQN: target = r + gamma * Q_target(s').max()")
    print("  Double DQN: target = r + gamma * Q_target(s')[Q(s').argmax()]")
    print("  ^ decouples action selection from value evaluation, reducing overestimation")
    print("=" * 60)


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    main()
