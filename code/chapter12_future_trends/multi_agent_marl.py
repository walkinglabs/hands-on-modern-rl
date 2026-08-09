"""
Chapter 13: Multi-Agent Reinforcement Learning (Multi-Agent RL) Experiment
——From independent learning to cooperative learning

This experiment builds a simplified multi-agent resource-collection scenario:
- 3 agents move around a grid world
- Resources are scattered across the grid, and agents need to collect them
- Agents "share" reward when they are close to each other (simulating cooperation)
- Two learning paradigms are compared:
    1. Independent learning (Independent Q-Learning): each agent trains independently
    2. Shared policy (Shared Policy): all agents share a single Q-table

How to run:
    python multi_agent_marl.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Create output directory
os.makedirs("output", exist_ok=True)

# Set Chinese font (kept for compatibility with any Chinese labels)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Multi-agent grid world environment
# ==========================================
class MultiAgentGridWorld:
    """
    Multi-agent grid world

    Map layout (8x8):
        - 0: empty space
        - 1: resource point (disappears when collected, regenerated every episode)
        - 2: wall (impassable)

    Agents:
        - 3 agents, initial positions scattered around the edges of the map
        - Each step can choose: up, down, left, right, stay
        - Automatically collects resources upon arrival

    Cooperation mechanism:
        - Two agents simultaneously collecting adjacent resources: extra +3 cooperation reward
        - Encourages agents to spread out rather than cluster together

    Reward design:
        - Collecting a resource: +5
        - Cooperative collection (adjacent): extra +3
        - Each move: -0.5 (encourages efficiency)
        - Hitting a wall: -1
    """

    # Actions: up, down, left, right, stay
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Stay']
    N_ACTIONS = 5

    def __init__(self, grid_size=8, n_agents=3, n_resources=6):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_resources = n_resources

        # Fixed wall positions
        self.walls = set()
        # Place some walls in the center of the map, forming a structure that requires going around
        wall_positions = [(3, 3), (3, 4), (4, 3)]
        for wp in wall_positions:
            if 0 <= wp[0] < grid_size and 0 <= wp[1] < grid_size:
                self.walls.add(wp)

        # Agents' initial positions (scattered around the edges)
        self.agent_starts = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0)]

        # Resource positions (regenerated every episode)
        self.resource_positions = set()

        # Agents' current positions
        self.agent_positions = []

    def reset(self):
        """
        Reset the environment

        Re-place the agents and resources, and return the initial observation.
        """
        # Reset agent positions
        self.agent_positions = [list(pos) for pos in self.agent_starts[:self.n_agents]]

        # Randomly generate resource positions
        self.resource_positions = set()
        while len(self.resource_positions) < self.n_resources:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            pos = (r, c)
            # Resources cannot be placed on walls or agent starting positions
            if pos not in self.walls and pos not in self.agent_starts:
                self.resource_positions.add(pos)

        return self._get_observations()

    def _get_observations(self):
        """
        Get each agent's observation

        The observation includes:
            - its own position
            - the direction and distance of the nearest resource
            - the relative positions of the other agents

        To simplify Q-learning, the observation is encoded as a single discrete state ID.
        """
        obs = []
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            obs.append((r, c))
        return obs

    def step(self, actions):
        """
        Execute all agents' actions

        Args:
            actions: list of actions of length n_agents

        Returns:
            observations: the new observations
            rewards: each agent's reward
            done: whether the episode has ended
            info: extra information
        """
        total_resources_before = len(self.resource_positions)
        rewards = [0.0] * self.n_agents
        collected_positions = []

        # Execute each agent's action in turn
        for i in range(self.n_agents):
            action = actions[i]
            dr, dc = self.ACTIONS[action]

            new_r = self.agent_positions[i][0] + dr
            new_c = self.agent_positions[i][1] + dc

            # Check validity
            if (new_r < 0 or new_r >= self.grid_size
                    or new_c < 0 or new_c >= self.grid_size
                    or (new_r, new_c) in self.walls):
                # Hit a wall: position unchanged, penalty applied
                rewards[i] -= 1.0
                continue

            # Update position
            self.agent_positions[i] = [new_r, new_c]

            # Movement penalty
            if action < 4:  # not "stay"
                rewards[i] -= 0.5

            # Check whether a resource was collected
            pos = (new_r, new_c)
            if pos in self.resource_positions:
                rewards[i] += 5.0
                collected_positions.append((i, pos))
                self.resource_positions.discard(pos)

        # Cooperation reward: check whether multiple agents collected at adjacent positions
        # at the same time, as well as distance-based rewards between agents
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = abs(self.agent_positions[i][0] - self.agent_positions[j][0]) + \
                       abs(self.agent_positions[i][1] - self.agent_positions[j][1])
                # Adjacent but not overlapping: grant a cooperation reward (encourages spreading out while staying in contact)
                if dist == 1:
                    # Check whether both agents are near a resource
                    for _, res_pos in collected_positions:
                        dist_i = abs(self.agent_positions[i][0] - res_pos[0]) + \
                                 abs(self.agent_positions[i][1] - res_pos[1])
                        dist_j = abs(self.agent_positions[j][0] - res_pos[0]) + \
                                 abs(self.agent_positions[j][1] - res_pos[1])
                        if dist_i <= 1 and dist_j <= 1:
                            rewards[i] += 1.5  # cooperation reward
                            rewards[j] += 1.5

        # Check whether all resources have been collected
        done = len(self.resource_positions) == 0

        # If all resources are collected, give an extra team reward
        if done:
            for i in range(self.n_agents):
                rewards[i] += 3.0  # completion reward

        info = {
            'resources_remaining': len(self.resource_positions),
            'resources_collected': total_resources_before - len(self.resource_positions),
        }

        return self._get_observations(), rewards, done, info


# ==========================================
# Part 2: Q-Learning agent
# ==========================================
class QLearningAgent:
    """
    Q-Learning agent

    Uses an epsilon-greedy policy and standard Q-learning updates.
    Each agent maintains its own Q-table.
    """

    def __init__(self, agent_id, n_actions=5, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: stored as a dict, keyed by (state, action)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs, env):
        """
        Convert an observation into a Q-table key

        State encoding:
            (own row, own column, direction of nearest resource, resources remaining)

        Here it is simplified to just its own position + resources remaining.
        """
        r, c = obs
        resources_left = len(env.resource_positions)
        return (r, c, resources_left)

    def select_action(self, state_key):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key, done):
        """
        Q-Learning update

        Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class SharedPolicyAgent:
    """
    Shared-policy agent

    All agents share a single Q-table.
    This simulates a parameter-sharing multi-agent method (similar to the shared
    network in IPPO).

    Pros:
        - Higher sample efficiency (experience from 3 agents is pooled for training)
        - Automatically learns cooperative behavior

    Cons:
        - Cannot distinguish policies for different roles
    """

    def __init__(self, n_agents=3, n_actions=5, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Shared Q-table
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs, env, agent_id):
        """
        State encoding for the shared policy

        Includes agent_id to distinguish policies across agents,
        but the Q-table itself is shared.
        """
        r, c = obs
        resources_left = len(env.resource_positions)
        return (agent_id, r, c, resources_left)

    def select_action(self, state_key):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key, done):
        """Q-Learning update (all agents update the same Q-table)"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ==========================================
# Part 3: Training functions
# ==========================================
def train_independent(env, n_episodes=800, max_steps=100, verbose=True):
    """
    Independent learning training (Independent Q-Learning, IQL)

    Each agent independently maintains its own Q-table,
    unaware of the other agents (treating them as part of the environment).

    This is the simplest multi-agent method, but it may fail to learn cooperative behavior.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training mode: independent learning (Independent Q-Learning)")
        print(f"  Number of agents: {env.n_agents}")
        print(f"  Training episodes: {n_episodes}")
        print(f"{'='*60}")

    agents = [QLearningAgent(agent_id=i) for i in range(env.n_agents)]

    # Record training data
    episode_rewards = []       # total team reward
    agent_rewards = [[] for _ in range(env.n_agents)]  # each agent's reward
    cooperation_counts = []    # count of cooperation events
    completion_rates = []      # completion rate

    for episode in range(n_episodes):
        obs = env.reset()
        total_team_reward = 0
        individual_rewards = [0.0] * env.n_agents
        cooperation_events = 0

        # Record the initial state keys
        state_keys = [agents[i].get_state_key(obs[i], env) for i in range(env.n_agents)]

        for step in range(max_steps):
            # Each agent independently selects an action
            actions = [agents[i].select_action(state_keys[i]) for i in range(env.n_agents)]

            # Execute the actions
            next_obs, rewards, done, info = env.step(actions)

            # Get the new state keys
            next_state_keys = [agents[i].get_state_key(next_obs[i], env)
                               for i in range(env.n_agents)]

            # Each agent independently updates its Q-table
            for i in range(env.n_agents):
                agents[i].update(state_keys[i], actions[i], rewards[i],
                                 next_state_keys[i], done)
                individual_rewards[i] += rewards[i]

            # Statistics
            total_team_reward += sum(rewards)
            # Simple cooperation detection: two or more agents getting a positive reward at the same time
            positive_count = sum(1 for r in rewards if r > 2.0)
            if positive_count >= 2:
                cooperation_events += 1

            state_keys = next_state_keys

            if done:
                break

        # Decay exploration rate
        for agent in agents:
            agent.decay_epsilon()

        # Record this episode's data
        episode_rewards.append(total_team_reward)
        for i in range(env.n_agents):
            agent_rewards[i].append(individual_rewards[i])
        cooperation_counts.append(cooperation_events)
        completion_rates.append(1.0 if done else 0.0)

        # Print progress every 200 episodes
        if verbose and (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_coop = np.mean(cooperation_counts[-200:])
            avg_complete = np.mean(completion_rates[-200:])
            print(f"  Episode {episode+1:4d} | "
                  f"Avg team reward: {avg_reward:7.2f} | "
                  f"Cooperation events: {avg_coop:4.1f} | "
                  f"Completion rate: {avg_complete:.0%} | "
                  f"epsilon: {agents[0].epsilon:.3f}")

    return {
        'episode_rewards': episode_rewards,
        'agent_rewards': agent_rewards,
        'cooperation_counts': cooperation_counts,
        'completion_rates': completion_rates,
        'agents': agents,
    }


def train_shared(env, n_episodes=800, max_steps=100, verbose=True):
    """
    Shared-policy training

    All agents share a single Q-table.
    Equivalent to parameter (weight) sharing, which improves sample efficiency.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training mode: shared policy (Shared Policy)")
        print(f"  Number of agents: {env.n_agents}")
        print(f"  Training episodes: {n_episodes}")
        print(f"{'='*60}")

    shared_agent = SharedPolicyAgent(n_agents=env.n_agents)

    # Record training data
    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n_agents)]
    cooperation_counts = []
    completion_rates = []

    for episode in range(n_episodes):
        obs = env.reset()
        total_team_reward = 0
        individual_rewards = [0.0] * env.n_agents
        cooperation_events = 0

        state_keys = [shared_agent.get_state_key(obs[i], env, i)
                      for i in range(env.n_agents)]

        for step in range(max_steps):
            # All agents select actions using the same Q-table
            actions = [shared_agent.select_action(state_keys[i])
                       for i in range(env.n_agents)]

            next_obs, rewards, done, info = env.step(actions)
            next_state_keys = [shared_agent.get_state_key(next_obs[i], env, i)
                               for i in range(env.n_agents)]

            # Shared Q-table update (all 3 agents' experience updates the same table)
            for i in range(env.n_agents):
                shared_agent.update(state_keys[i], actions[i], rewards[i],
                                    next_state_keys[i], done)
                individual_rewards[i] += rewards[i]

            total_team_reward += sum(rewards)
            positive_count = sum(1 for r in rewards if r > 2.0)
            if positive_count >= 2:
                cooperation_events += 1

            state_keys = next_state_keys

            if done:
                break

        shared_agent.decay_epsilon()

        episode_rewards.append(total_team_reward)
        for i in range(env.n_agents):
            agent_rewards[i].append(individual_rewards[i])
        cooperation_counts.append(cooperation_events)
        completion_rates.append(1.0 if done else 0.0)

        if verbose and (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_coop = np.mean(cooperation_counts[-200:])
            avg_complete = np.mean(completion_rates[-200:])
            print(f"  Episode {episode+1:4d} | "
                  f"Avg team reward: {avg_reward:7.2f} | "
                  f"Cooperation events: {avg_coop:4.1f} | "
                  f"Completion rate: {avg_complete:.0%} | "
                  f"epsilon: {shared_agent.epsilon:.3f}")

    return {
        'episode_rewards': episode_rewards,
        'agent_rewards': agent_rewards,
        'cooperation_counts': cooperation_counts,
        'completion_rates': completion_rates,
        'agent': shared_agent,
    }


# ==========================================
# Part 4: Visualization
# ==========================================
def visualize_results(ind_results, shared_results, n_agents=3):
    """
    Visualize the comparison experiment results

    Contains 4 subplots:
        1. Total team reward comparison curve
        2. Completion rate comparison
        3. Cooperation events comparison
        4. Per-agent individual reward comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Agent RL — Independent Learning vs Shared Policy',
                 fontsize=16, fontweight='bold')

    window = 50  # moving-average window
    episodes = range(len(ind_results['episode_rewards']))

    # ---- Subplot 1: total team reward comparison ----
    ax1 = axes[0, 0]

    # Independent learning
    ind_rewards = ind_results['episode_rewards']
    if len(ind_rewards) >= window:
        ind_avg = np.convolve(ind_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(ind_rewards)), ind_avg,
                 color='#F44336', linewidth=2.5, label='Independent learning (IQL)')

    # Shared policy
    shared_rewards = shared_results['episode_rewards']
    if len(shared_rewards) >= window:
        shared_avg = np.convolve(shared_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(shared_rewards)), shared_avg,
                 color='#2196F3', linewidth=2.5, label='Shared policy (Shared)')

    ax1.set_xlabel('Training episode', fontsize=12)
    ax1.set_ylabel('Total team reward', fontsize=12)
    ax1.set_title('Total team reward comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: completion rate comparison ----
    ax2 = axes[0, 1]

    ind_completion = ind_results['completion_rates']
    shared_completion = shared_results['completion_rates']

    # Compute moving-average completion rate
    if len(ind_completion) >= window:
        ind_comp_avg = np.convolve(ind_completion, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(ind_completion)), ind_comp_avg * 100,
                 color='#F44336', linewidth=2.5, label='Independent learning (IQL)')

    if len(shared_completion) >= window:
        shared_comp_avg = np.convolve(shared_completion, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(shared_completion)), shared_comp_avg * 100,
                 color='#2196F3', linewidth=2.5, label='Shared policy (Shared)')

    ax2.set_xlabel('Training episode', fontsize=12)
    ax2.set_ylabel('Completion rate (%)', fontsize=12)
    ax2.set_title('Task completion rate comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: cooperation events comparison ----
    ax3 = axes[1, 0]

    ind_coop = ind_results['cooperation_counts']
    shared_coop = shared_results['cooperation_counts']

    if len(ind_coop) >= window:
        ind_coop_avg = np.convolve(ind_coop, np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(ind_coop)), ind_coop_avg,
                 color='#F44336', linewidth=2.5, label='Independent learning (IQL)')

    if len(shared_coop) >= window:
        shared_coop_avg = np.convolve(shared_coop, np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(shared_coop)), shared_coop_avg,
                 color='#2196F3', linewidth=2.5, label='Shared policy (Shared)')

    ax3.set_xlabel('Training episode', fontsize=12)
    ax3.set_ylabel('Number of cooperation events', fontsize=12)
    ax3.set_title('Cooperation events comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: individual agent rewards (shared policy) ----
    ax4 = axes[1, 1]

    agent_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    for i in range(n_agents):
        rewards_i = shared_results['agent_rewards'][i]
        if len(rewards_i) >= window:
            avg_i = np.convolve(rewards_i, np.ones(window) / window, mode='valid')
            ax4.plot(range(window - 1, len(rewards_i)), avg_i,
                     color=agent_colors[i], linewidth=2, label=f'Agent {i+1}')

    ax4.set_xlabel('Training episode', fontsize=12)
    ax4.set_ylabel('Individual cumulative reward', fontsize=12)
    ax4.set_title('Per-agent reward (shared policy)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_trajectories(env, shared_agent, max_steps=30):
    """
    Visualize the trained agents' rollout trajectories

    Shows how the 3 agents cooperate to collect resources in the grid world.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Multi-Agent Rollout Trajectory Visualization', fontsize=16, fontweight='bold')

    # Run one episode
    obs = env.reset()
    agent_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    trajectories = [[tuple(obs[i])] for i in range(env.n_agents)]
    initial_resources = set(env.resource_positions)
    collected_steps = []  # record the steps at which resources were collected

    for step in range(max_steps):
        state_keys = [shared_agent.get_state_key(obs[i], env, i)
                      for i in range(env.n_agents)]
        actions = [shared_agent.select_action(state_keys[i])
                   for i in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)

        for i in range(env.n_agents):
            trajectories[i].append(tuple(next_obs[i]))

        if info['resources_collected'] > 0:
            collected_steps.append(step)

        obs = next_obs
        if done:
            break

    # ---- Left plot: grid world trajectories ----
    ax1 = axes[0]

    # Draw the grid
    grid_display = np.zeros((env.grid_size, env.grid_size))
    for wr, wc in env.walls:
        grid_display[wr][wc] = -1

    ax1.imshow(grid_display, cmap='Greys', alpha=0.3,
               extent=[-0.5, env.grid_size - 0.5, env.grid_size - 0.5, -0.5])

    # Draw the initial resource positions
    for rr, rc in initial_resources:
        ax1.plot(rc, rr, '*', color='gold', markersize=15,
                 markeredgecolor='orange', markeredgewidth=1.5)

    # Draw each agent's trajectory
    for i in range(env.n_agents):
        traj = trajectories[i]
        rows = [t[0] for t in traj]
        cols = [t[1] for t in traj]

        # Trajectory line
        ax1.plot(cols, rows, '-', color=agent_colors[i], linewidth=2,
                 alpha=0.7, label=f'Agent {i+1}')
        # Start point
        ax1.plot(cols[0], rows[0], 'o', color=agent_colors[i], markersize=12)
        # End point
        ax1.plot(cols[-1], rows[-1], 's', color=agent_colors[i], markersize=12)

        # Annotate step number
        for step_idx, (r, c) in enumerate(traj):
            if step_idx % 5 == 0 and step_idx > 0:  # label every 5 steps
                ax1.text(c + 0.2, r - 0.2, str(step_idx), fontsize=7,
                         color=agent_colors[i], alpha=0.7)

    # Draw walls
    for wr, wc in env.walls:
        ax1.add_patch(plt.Rectangle((wc - 0.5, wr - 0.5), 1, 1,
                                     facecolor='gray', alpha=0.5))
        ax1.text(wc, wr, 'Wall', ha='center', va='center', fontsize=10,
                 color='white', fontweight='bold')

    ax1.set_xlim(-0.5, env.grid_size - 0.5)
    ax1.set_ylim(env.grid_size - 0.5, -0.5)
    ax1.set_xticks(range(env.grid_size))
    ax1.set_yticks(range(env.grid_size))
    ax1.set_title('Agent rollout trajectories', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---- Right plot: step number vs. resources collected ----
    ax2 = axes[1]

    total_steps = len(trajectories[0])
    resources_per_step = []
    remaining = len(initial_resources)

    # Re-simulate to track how the resource count changes
    obs2 = env.reset()
    env.resource_positions = set(initial_resources)
    remaining_track = [len(env.resource_positions)]

    for step in range(total_steps - 1):
        state_keys = [shared_agent.get_state_key(obs2[i], env, i)
                      for i in range(env.n_agents)]
        actions = [shared_agent.select_action(state_keys[i])
                   for i in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)
        remaining_track.append(info['resources_remaining'])
        obs2 = next_obs
        if done:
            break

    ax2.fill_between(range(len(remaining_track)), remaining_track,
                     alpha=0.3, color='#4CAF50')
    ax2.plot(range(len(remaining_track)), remaining_track,
             '-o', color='#4CAF50', linewidth=2, markersize=5)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Resources remaining', fontsize=12)
    ax2.set_title('Resource collection progress', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Annotate completion time
    if remaining_track[-1] == 0:
        complete_step = len(remaining_track) - 1
        ax2.axvline(x=complete_step, color='red', linestyle='--', linewidth=1.5)
        ax2.annotate(f'All resources collected\n(step {complete_step})',
                     xy=(complete_step, 0), xytext=(complete_step + 2, 2),
                     fontsize=10, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    return fig


def print_cooperation_statistics(ind_results, shared_results, n_agents):
    """
    Print cooperation statistics
    """
    print("\n" + "=" * 60)
    print("  Cooperation statistics comparison")
    print("=" * 60)

    # Use the last 200 episodes of statistics
    last_n = 200

    for name, results in [("Independent learning", ind_results), ("Shared policy", shared_results)]:
        rewards = results['episode_rewards'][-last_n:]
        completion = results['completion_rates'][-last_n:]
        coop = results['cooperation_counts'][-last_n:]

        print(f"\n  [{name}] statistics over the last {last_n} episodes:")
        print(f"    Avg team reward:      {np.mean(rewards):.2f}")
        print(f"    Task completion rate: {np.mean(completion):.0%}")
        print(f"    Avg cooperation events: {np.mean(coop):.2f}")
        print(f"    Team reward std dev:  {np.std(rewards):.2f}")

        # Each agent's contribution
        print(f"    Avg reward per agent:", end="")
        for i in range(n_agents):
            avg_r = np.mean(results['agent_rewards'][i][-last_n:])
            print(f"  A{i+1}={avg_r:.1f}", end="")
        print()

    print("\n" + "-" * 60)


# ==========================================
# Part 5: Main program
# ==========================================
def main():
    """
    Main function: create environment -> train both strategies -> compare -> visualize

    Experiment flow:
        1. Create the multi-agent grid world
        2. Train the independent-learning (IQL) agents
        3. Train the shared-policy agent
        4. Compare team reward, completion rate, and degree of cooperation for both methods
        5. Visualize the results and agent trajectories
    """

    print("=" * 60)
    print("  Chapter 13: Multi-Agent Reinforcement Learning (MARL) Experiment")
    print("=" * 60)
    print("  Scenario: multi-agent resource collection")
    print("  Agents: 3 (Independent Q-Learning vs Shared Policy)")
    print("  Task: cooperatively collect 6 resources in an 8x8 grid world")
    print("-" * 60)

    # Set random seed
    np.random.seed(42)

    # ---- Step 1: create the environment ----
    print("\n[Step 1] Create the multi-agent grid world environment")
    env = MultiAgentGridWorld(grid_size=8, n_agents=3, n_resources=6)
    obs = env.reset()
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Agents' initial positions: {env.agent_starts}")
    print(f"  Number of resources: {env.n_resources}")
    print(f"  Wall positions: {env.walls}")

    # ---- Step 2: train the independent-learning agents ----
    print("\n[Step 2] Training independent-learning (IQL) agents...")
    ind_results = train_independent(env, n_episodes=800, max_steps=80)

    # ---- Step 3: train the shared-policy agent ----
    print("\n[Step 3] Training the shared-policy agent...")
    shared_results = train_shared(env, n_episodes=800, max_steps=80)

    # ---- Step 4: comparative analysis ----
    print("\n[Step 4] Comparative analysis")
    print_cooperation_statistics(ind_results, shared_results, env.n_agents)

    # ---- Step 5: visualization ----
    print("[Step 5] Generating visualization figures...")

    # Figure 1: training curve comparison
    fig1 = visualize_results(ind_results, shared_results, env.n_agents)
    fig1.savefig('output/marl_training_comparison.png', dpi=150, bbox_inches='tight')
    print("  Training curve comparison figure saved to: output/marl_training_comparison.png")

    # Figure 2: agent trajectory visualization
    fig2 = visualize_trajectories(env, shared_results['agent'], max_steps=40)
    fig2.savefig('output/marl_trajectories.png', dpi=150, bbox_inches='tight')
    print("  Agent trajectory figure saved to: output/marl_trajectories.png")

    # ---- Key conclusions ----
    print("\n" + "=" * 60)
    print("  Key conclusions")
    print("=" * 60)
    print("  1. Independent learning (IQL): each agent optimizes independently — simple but lacks cooperation")
    print("  2. Shared policy: parameter sharing improves sample efficiency and naturally gives rise to cooperative behavior")
    print("  3. Designing cooperation rewards is one of the core challenges of multi-agent RL")
    print("  4. Real multi-agent scenarios need to consider:")
    print("     - Communication mechanisms (how agents pass information to each other)")
    print("     - Credit assignment (how to allocate the team reward to individuals)")
    print("     - Scalability (computational complexity as the number of agents grows)")
    print("  5. Frontier directions in multi-agent RL:")
    print("     - Self-play-based multi-agent training")
    print("     - LLM-driven multi-agent cooperation")
    print("     - Multi-agent adaptability in open-ended environments")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
