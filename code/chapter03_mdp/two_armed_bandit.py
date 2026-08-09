"""
Chapter 3: Two-armed bandit experiment -- the classic exploration vs. exploitation comparison
Compare four strategies: random, greedy, epsilon-greedy, UCB
Evaluate each strategy's performance via cumulative average reward and cumulative regret

How to run:
    python two_armed_bandit.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)


# ==========================================
# Part 1: Two-armed bandit environment
# ==========================================
class TwoArmedBandit:
    """
    Two-armed bandit environment
    - Arm A: win probability 0.6 (the better choice)
    - Arm B: win probability 0.4

    In reinforcement learning, this is the simplest possible decision problem:
    there is only one state (always the same), and two actions (A and B).
    The goal is to maximize cumulative reward. Although it is a degenerate
    form of an MDP, it perfectly illustrates the core "exploration vs.
    exploitation" tension.
    """

    def __init__(self, prob_a=0.6, prob_b=0.4):
        self.prob_a = prob_a  # Win probability of arm A
        self.prob_b = prob_b  # Win probability of arm B
        # Probability of the best arm, used to compute regret
        self.best_prob = max(prob_a, prob_b)

    def pull(self, arm):
        """
        Pull the specified arm, return the reward (0 or 1)

        Args:
            arm: 0 for arm A, 1 for arm B
        Returns:
            reward: 1 for a win, 0 for no win
        """
        if arm == 0:
            return 1 if np.random.random() < self.prob_a else 0
        else:
            return 1 if np.random.random() < self.prob_b else 0


# ==========================================
# Part 2: Implementation of the four strategies
# ==========================================

def strategy_random(bandit, n_steps):
    """
    Strategy 1: random strategy
    Choose arm A or B completely at random at every step, ignoring history.

    This is the most basic baseline strategy. Since it does no learning at
    all, the average reward should be close to the mean of the two arms'
    probabilities: (0.6 + 0.4) / 2 = 0.5
    """
    rewards = []
    for _ in range(n_steps):
        arm = np.random.choice([0, 1])  # Choose randomly with equal probability
        reward = bandit.pull(arm)
        rewards.append(reward)
    return np.array(rewards)


def strategy_greedy(bandit, n_steps):
    """
    Strategy 2: greedy strategy
    Always choose the arm with the highest current estimate.

    Problem: since the initial estimates are equal, the first step picks an
    arm at random; if it happens to win, that arm gets pulled forever and the
    other arm is never explored again. This is the classic example of
    "premature convergence".
    """
    rewards = []
    # Q[a] denotes the current estimated expected reward of arm a
    Q = np.zeros(2)       # Initial estimates
    counts = np.zeros(2)  # Number of times each arm has been pulled

    for _ in range(n_steps):
        # Always choose the arm with the highest current estimate (exploit only)
        arm = np.argmax(Q)
        reward = bandit.pull(arm)
        rewards.append(reward)

        # Update that arm's estimate: incremental running average
        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


def strategy_epsilon_greedy(bandit, n_steps, epsilon=0.1):
    """
    Strategy 3: epsilon-greedy strategy (epsilon = 0.1)
    Explore randomly with probability epsilon, otherwise choose the current
    best arm with probability 1-epsilon.

    epsilon = 0.1 means about 10% of the time is spent on random exploration.
    This is the simplest and most common way to resolve the
    "exploration vs. exploitation" tension.
    epsilon too large → too much time wasted on known-bad choices;
    epsilon too small → the optimal arm may never be found.
    """
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)

    for _ in range(n_steps):
        # Explore randomly with probability epsilon, otherwise choose greedily
        if np.random.random() < epsilon:
            arm = np.random.choice([0, 1])  # Explore
        else:
            arm = np.argmax(Q)  # Exploit

        reward = bandit.pull(arm)
        rewards.append(reward)

        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


def strategy_ucb(bandit, n_steps, c=2.0):
    """
    Strategy 4: Upper Confidence Bound strategy (UCB)
    Choose the arm with the highest "estimate + uncertainty bound".

    UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
    - Q(a): the current estimated expected reward of arm a
    - c: parameter controlling the amount of exploration (typically c=2)
    - t: current total step count
    - N(a): number of times arm a has been chosen

    Core idea: if an arm has rarely been chosen (N(a) is small), the estimate
    for it is uncertain, so its uncertainty bound is large -- UCB therefore
    tends to try the arms that are "still uncertain". As the number of trials
    grows, uncertainty shrinks and the strategy naturally shifts toward
    greedy behavior.
    """
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)

    for t in range(1, n_steps + 1):
        # Pull each arm once in the first two steps, so every arm has been tried
        if t <= 2:
            arm = t - 1
        else:
            # Compute the UCB values
            ucb_values = np.zeros(2)
            for a in range(2):
                # Uncertainty bound: fewer pulls means a larger bound
                uncertainty = c * np.sqrt(np.log(t) / counts[a])
                ucb_values[a] = Q[a] + uncertainty
            arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        rewards.append(reward)

        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


# ==========================================
# Part 3: Running the experiment and comparing results
# ==========================================
def run_experiment():
    """
    Main experiment: run each strategy n_runs times, n_steps steps each, then average
    """
    n_steps = 1000  # Number of steps per run
    n_runs = 200    # Number of repeated runs (for smoother curves)

    # Used to accumulate results for each strategy
    all_rewards = {
        'Random': np.zeros(n_steps),
        'Greedy': np.zeros(n_steps),
        'ε-greedy (ε=0.1)': np.zeros(n_steps),
        'UCB (c=2)': np.zeros(n_steps),
    }

    print("=" * 60)
    print("  Two-armed bandit experiment: comparing exploration/exploitation strategies")
    print("=" * 60)
    print(f"  Arm A win probability: 0.6 (optimal)")
    print(f"  Arm B win probability: 0.4")
    print(f"  Steps per run: {n_steps}")
    print(f"  Number of runs: {n_runs}")
    print("-" * 60)
    print("Running experiment...")

    for run in range(n_runs):
        bandit = TwoArmedBandit(prob_a=0.6, prob_b=0.4)

        # Run all four strategies
        all_rewards['Random'] += strategy_random(bandit, n_steps)
        all_rewards['Greedy'] += strategy_greedy(bandit, n_steps)
        all_rewards['ε-greedy (ε=0.1)'] += strategy_epsilon_greedy(bandit, n_steps)
        all_rewards['UCB (c=2)'] += strategy_ucb(bandit, n_steps)

        if (run + 1) % 50 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs...")

    # Compute the averages
    for key in all_rewards:
        all_rewards[key] /= n_runs

    print("Experiment complete!")
    print()

    # ==========================================
    # Part 4: Plot the cumulative average reward curves
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Figure 1: cumulative average reward
    ax1 = axes[0]
    colors = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50']
    for (name, rewards), color in zip(all_rewards.items(), colors):
        # Compute the cumulative average reward
        cumulative_avg = np.cumsum(rewards) / np.arange(1, n_steps + 1)
        ax1.plot(cumulative_avg, label=name, color=color, alpha=0.85)

    ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Optimal (prob=0.6)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline (0.5)')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Cumulative average reward', fontsize=12)
    ax1.set_title('Cumulative average reward comparison', fontsize=14)
    ax1.legend(fontsize=9, loc='right')
    ax1.set_ylim(0.35, 0.7)
    ax1.grid(True, alpha=0.3)

    # Figure 2: cumulative regret
    ax2 = axes[1]
    best_prob = 0.6
    for (name, rewards), color in zip(all_rewards.items(), colors):
        # Regret = optimal reward per step - actual reward obtained
        regret = best_prob - rewards
        cumulative_regret = np.cumsum(regret)
        ax2.plot(cumulative_regret, label=name, color=color, alpha=0.85)

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Cumulative regret', fontsize=12)
    ax2.set_title('Cumulative regret comparison (lower is better)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/two_armed_bandit_results.png', dpi=150, bbox_inches='tight')
    print("Figure saved to output/two_armed_bandit_results.png")
    plt.show()

    # ==========================================
    # Part 5: Print the results summary table
    # ==========================================
    print()
    print("=" * 60)
    print("  Experiment results summary")
    print("=" * 60)
    print(f"{'Strategy':<20s} {'Cum. avg reward':<15s} {'Final avg reward':<15s} {'Total regret':<10s}")
    print("-" * 60)

    for (name, rewards), color in zip(all_rewards.items(), colors):
        cum_avg = np.cumsum(rewards)
        final_avg = cum_avg[-1] / n_steps
        total_reward = np.sum(rewards)
        total_regret = best_prob * n_steps - total_reward
        print(f"{name:<20s} {final_avg:<15.4f} {rewards[-1]:<15.4f} {total_regret:<10.1f}")

    print("-" * 60)
    print()
    print("Analysis:")
    print("  - Random strategy: does no learning, reward stays near 0.5 (the mean of the two arm probabilities)")
    print("  - Greedy strategy: may lock in prematurely on a suboptimal arm, results are unstable")
    print("  - epsilon-greedy strategy: balances exploration and exploitation, performs well")
    print("  - UCB strategy: guides exploration via uncertainty, usually performs best")


if __name__ == "__main__":
    run_experiment()
