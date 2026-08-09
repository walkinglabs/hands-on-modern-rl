"""
Chapter 3: 4x4 GridWorld Q-Learning experiment
Learn the optimal path in a grid world to build intuition for Q-values and the
Bellman equation

How to run:
    python gridworld_q_learning.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)


# ==========================================
# Part 1: GridWorld environment
# ==========================================
class GridWorld:
    """
    4x4 grid world environment

    Grid layout (4 rows, 4 columns):
        0,0  0,1  0,2  0,3
        1,0  1,1  1,2  1,3
        2,0  2,1  2,2  2,3
        3,0  3,1  3,2  3,3

    - Start: (0, 0)
    - Goal: (3, 3), reaching it gives a +10 reward
    - Obstacles: (1, 1) and (2, 2), hitting one gives a -5 penalty
    - Each step: -1 reward (encourages reaching the goal quickly)
    - Hitting a wall: -5 reward (position unchanged)

    Action space:
        0 = up (↑), 1 = down (↓), 2 = left (←), 3 = right (→)
    """

    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacles = [(1, 1), (2, 2)]
        self.n_actions = 4  # up, down, left, right
        self.action_names = ['up(↑)', 'down(↓)', 'left(←)', 'right(→)']
        self.reset()

    def reset(self):
        """Reset the environment to the start, return the initial state"""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        """
        Execute an action, return (next state, reward, done)

        Action mapping:
            0 = up → row -1
            1 = down → row +1
            2 = left → col -1
            3 = right → col +1
        """
        row, col = self.agent_pos

        # Compute the new position based on the action
        if action == 0:    # up
            new_pos = (row - 1, col)
        elif action == 1:  # down
            new_pos = (row + 1, col)
        elif action == 2:  # left
            new_pos = (row, col - 1)
        elif action == 3:  # right
            new_pos = (row, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check whether it hits a wall (out of bounds)
        new_row, new_col = new_pos
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            # Hit a wall: position unchanged, apply penalty
            return self.agent_pos, -5, False

        # Check whether it hits an obstacle
        if new_pos in self.obstacles:
            # Hit an obstacle: position unchanged, apply penalty
            return self.agent_pos, -5, False

        # Legal move: update position
        self.agent_pos = new_pos

        # Check whether the goal has been reached
        if self.agent_pos == self.goal:
            return self.agent_pos, 10, True  # Reached the goal, +10 reward

        # Normal move: -1 reward (encourages reaching the goal quickly)
        return self.agent_pos, -1, False


# ==========================================
# Part 2: Q-Learning algorithm
# ==========================================
def epsilon_greedy(Q, state, epsilon, n_actions):
    """
    epsilon-greedy action selection policy

    Explore randomly with probability epsilon, otherwise pick the action with
    the highest current Q value with probability 1-epsilon.
    This is the standard way Q-Learning balances "exploration" and "exploitation".
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # Explore: pick a random action
    else:
        return np.argmax(Q[state])  # Exploit: pick the action with the highest Q value


def train_q_learning(env, n_episodes=500, alpha=0.1, gamma=0.95,
                     epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Q-Learning training

    The core update formula of Q-Learning (the iterative form of the Bellman
    optimality equation):
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    Where:
        - s: current state
        - a: current action
        - r: reward received
        - s': next state
        - α: learning rate (controls the update step size)
        - γ: discount factor (how much future rewards matter)
        - max_a' Q(s', a'): the maximum Q value of the next state

    Args:
        n_episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
        epsilon_start: initial exploration rate
        epsilon_end: minimum exploration rate
        epsilon_decay: exploration rate decay factor
    """
    # Initialize the Q table: all Q values set to 0
    # Q[state][action] = estimated optimal action value
    Q = np.zeros((env.rows, env.cols, env.n_actions))

    # Record data during training
    episode_rewards = []  # Cumulative reward per episode
    episode_steps = []    # Number of steps per episode
    epsilon = epsilon_start

    print("=" * 60)
    print("  Q-Learning training")
    print("=" * 60)
    print(f"  Learning rate α = {alpha}")
    print(f"  Discount factor γ = {gamma}")
    print(f"  Initial exploration rate ε = {epsilon_start}")
    print(f"  Number of training episodes = {n_episodes}")
    print("-" * 60)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # 1. Select an action using the epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon, env.n_actions)

            # 2. Execute the action, observe the reward and next state
            next_state, reward, done = env.step(action)

            # 3. Q-Learning update (core formula)
            #    Note: this uses max_a' Q(s', a'), regardless of the policy
            #    actually followed -- this is the "off-policy" property of Q-Learning
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            # 4. Transition to the next state
            state = next_state
            total_reward += reward
            steps += 1

            # Safety valve: prevent an infinite loop
            if steps > 200:
                break

        # Decay the exploration rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"  Episode {episode + 1:4d} | "
                  f"avg reward (last 100): {avg_reward:7.2f} | "
                  f"avg steps: {avg_steps:5.1f} | "
                  f"ε: {epsilon:.4f}")

    print("-" * 60)
    return Q, episode_rewards, episode_steps


# ==========================================
# Part 3: Results visualization
# ==========================================
def print_q_table(Q, env):
    """
    Print a formatted Q table

    The Q table shows the Q value for each action in each state (the estimated
    optimal action value). A higher Q value means a larger expected cumulative
    reward for taking that action in that state.
    """
    print("\n" + "=" * 60)
    print("  Final Q table")
    print("=" * 60)
    print(f"{'State':<10s}", end="")
    for name in env.action_names:
        print(f"{name:<12s}", end="")
    print(f"{'Best action':<12s}")
    print("-" * 60)

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.obstacles:
                print(f"({r},{c}) obstacle  ", end="")
                print("    ---      ---      ---      ---     obstacle")
                continue
            if state == env.goal:
                print(f"({r},{c}) goal  ", end="")
                print("    ---      ---      ---      ---     goal")
                continue

            print(f"({r},{c})       ", end="")
            for a in range(env.n_actions):
                print(f"{Q[r][c][a]:>8.2f}   ", end="")
            best_action = np.argmax(Q[r][c])
            print(f"  {env.action_names[best_action]}")

    print("-" * 60)


def extract_optimal_path(Q, env):
    """
    Extract the optimal path from the Q table

    Select the action with the highest Q value at each state, and compute the
    next state according to the grid rules.
    Does not depend on the environment's step() function, avoiding unintended
    mutation of the environment state.
    """
    # Displacement corresponding to each action: 0=up, 1=down, 2=left, 3=right
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    state = env.start
    path = [state]
    visited = set()

    while state != env.goal:
        if state in visited:
            break  # Prevent an infinite loop
        visited.add(state)
        action = np.argmax(Q[state])
        dr, dc = deltas[action]
        new_state = (state[0] + dr, state[1] + dc)

        # Check whether the new position is legal (in bounds, not an obstacle)
        if (0 <= new_state[0] < env.rows and 0 <= new_state[1] < env.cols
                and new_state not in env.obstacles):
            state = new_state
        # If out of bounds or hitting an obstacle, state stays unchanged
        # (would otherwise risk an infinite loop, guarded by `visited`)
        path.append(state)
        if state == env.goal:
            break

    return path


def visualize_results(Q, episode_rewards, env):
    """
    Visualize the Q-Learning results
    - Figure 1: Q-value heatmap for each action
    - Figure 2: the optimal path shown on the grid
    - Figure 3: the cumulative reward curve over episodes
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))

    # ------------------------------------------
    # Figure 1: Q-value heatmaps for the four actions
    # ------------------------------------------
    action_names_short = ['up(↑)', 'down(↓)', 'left(←)', 'right(→)']

    for i in range(4):
        ax = fig.add_subplot(2, 3, i + 1)
        q_values = Q[:, :, i]  # Extract the Q values of one action across all states

        im = ax.imshow(q_values, cmap='RdYlGn', aspect='equal')
        # Annotate the Q value on each cell
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.obstacles:
                    ax.text(c, r, 'X', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='black')
                elif (r, c) == env.goal:
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='blue')
                else:
                    ax.text(c, r, f'{q_values[r, c]:.1f}', ha='center', va='center',
                            fontsize=9)
        ax.set_title(f'Q(s, {action_names_short[i]})', fontsize=12)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_xticklabels(range(env.cols))
        ax.set_yticklabels(range(env.rows))
        plt.colorbar(im, ax=ax, shrink=0.8)

    # ------------------------------------------
    # Figure 2: optimal path visualization
    # ------------------------------------------
    ax_path = fig.add_subplot(2, 3, 5)
    # Draw the grid background
    grid = np.zeros((env.rows, env.cols))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 2

    ax_path.imshow(grid, cmap='Set3', aspect='equal', vmin=-2, vmax=3)

    # Extract and draw the optimal path
    path = extract_optimal_path(Q, env)
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax_path.plot(path_cols, path_rows, 'b-o', linewidth=2.5, markersize=10)

    # Annotate the start, goal, and obstacles
    ax_path.text(0, 0, 'S', ha='center', va='center', fontsize=16,
                 fontweight='bold', color='green')
    ax_path.text(3, 3, 'G', ha='center', va='center', fontsize=16,
                 fontweight='bold', color='red')
    for obs in env.obstacles:
        ax_path.text(obs[1], obs[0], 'X', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='black')

    # Annotate step numbers along the path
    for idx, (r, c) in enumerate(path):
        ax_path.text(c, r, str(idx), ha='center', va='center',
                     fontsize=8, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.5))

    ax_path.set_title('Optimal path', fontsize=12)
    ax_path.set_xticks(range(env.cols))
    ax_path.set_yticks(range(env.rows))
    ax_path.set_xticklabels(range(env.cols))
    ax_path.set_yticklabels(range(env.rows))
    ax_path.grid(True, alpha=0.3)

    # ------------------------------------------
    # Figure 3: training reward curve
    # ------------------------------------------
    ax_reward = fig.add_subplot(2, 3, 6)
    ax_reward.plot(episode_rewards, alpha=0.3, color='lightblue', label='Per-episode reward')
    # Compute the moving average
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards,
                                 np.ones(window) / window, mode='valid')
        ax_reward.plot(range(window - 1, len(episode_rewards)),
                       moving_avg, color='blue', linewidth=2,
                       label=f'{window}-episode moving average')
    ax_reward.set_xlabel('Episode', fontsize=11)
    ax_reward.set_ylabel('Cumulative reward', fontsize=11)
    ax_reward.set_title('Training reward curve', fontsize=12)
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/gridworld_q_learning_results.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to output/gridworld_q_learning_results.png")
    plt.show()


# ==========================================
# Part 4: Main program
# ==========================================
def main():
    """Main function: create environment → train → print Q table → visualize"""

    # Create the GridWorld environment
    env = GridWorld()
    print("GridWorld environment created")
    print(f"  Start: {env.start}")
    print(f"  Goal: {env.goal}")
    print(f"  Obstacles: {env.obstacles}")
    print(f"  Action space: {env.action_names}")

    # Train Q-Learning
    Q, episode_rewards, episode_steps = train_q_learning(env, n_episodes=500)

    # Print the final Q table
    print_q_table(Q, env)

    # Extract and print the optimal path
    path = extract_optimal_path(Q, env)
    print(f"\nOptimal path: {' → '.join([str(p) for p in path])}")
    print(f"Path length: {len(path) - 1} steps")

    # Compute the total reward of the optimal path
    total_r = 0
    for i in range(len(path) - 1):
        if i < len(path) - 2:
            total_r += -1  # Normal step: -1
        else:
            total_r += 10  # Final step reaching the goal: +10
    print(f"Total reward of the optimal path: {total_r}")

    # Visualize the results
    visualize_results(Q, episode_rewards, env)


if __name__ == "__main__":
    main()
