"""
Chapter 9: GRPO Core Mechanism Demo — Group Relative Policy Optimization
==========================================================

This script uses synthetic data to walk step by step through the core idea of
GRPO (Group Relative Policy Optimization):
  1. Generate a group of responses and obtain their raw rewards
  2. Compute the group's mean and standard deviation
  3. Obtain advantages via group-relative normalization
  4. Compare against the PPO Critic-baseline approach
  5. Visualize the reward distribution before/after normalization and compare advantages

Key innovations of GRPO:
  - No need to train an extra Value Network (Critic)
  - Uses the group-level statistics of multiple sampled responses to the same question as the baseline
  - Greatly simplifies the training pipeline while preserving the variance-reduction benefit of the advantage function

Core formula:
  advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

How to run:
  python grpo_mechanism.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create the output directory
os.makedirs("output", exist_ok=True)

# Set a CJK-capable font so chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: GRPO group-relative advantage computation
# ==========================================
def compute_grpo_advantages(rewards):
    """
    The core operation of GRPO: normalize a group of rewards within the group to obtain advantages

    Formula:
        mean_r = mean(rewards)
        std_r  = std(rewards) + eps   (eps prevents division by zero)
        advantage_i = (reward_i - mean_r) / std_r

    Intuition:
        - advantage > 0 means this response is "better than the group average" and should be encouraged
        - advantage < 0 means this response is "worse than the group average" and should be discouraged
        - After normalization, advantages fluctuate around 0 with unit variance

    Args:
        rewards: numpy array, the raw reward values for a group of responses
    Returns:
        advantages: numpy array, the normalized advantage values
    """
    eps = 1e-8  # a tiny constant to prevent division by zero when std=0
    mean_r = rewards.mean()
    std_r = rewards.std() + eps
    advantages = (rewards - mean_r) / std_r
    return advantages


def compute_ppo_advantages(rewards, value_predictions):
    """
    PPO-style advantage computation: uses the Critic network's value prediction as a baseline

    Formula:
        advantage_i = reward_i - V(s_i)

    This requires training an additional Value Network to estimate V(s),
    which adds training complexity but is theoretically more precise.

    Args:
        rewards: numpy array, the raw reward values for a group of responses
        value_predictions: numpy array, the Critic network's value estimate for each state
    Returns:
        advantages: numpy array, the Critic-baseline advantage values
    """
    advantages = rewards - value_predictions
    return advantages


# ==========================================
# Part 2: Generate synthetic reward data
# ==========================================
def generate_synthetic_rewards(group_size=8, seed=42):
    """
    Generate a group of synthetic reward data, simulating the GRPO sampling process

    Scenario setup:
        Suppose we give the model a question and have it generate group_size different
        responses, then score each response with a reward model (or rule-based function).

    Rewards are set in the range [0, 1], simulating typical reward-model output:
        - 0.0 ~ 0.3: low-quality responses
        - 0.3 ~ 0.7: medium-quality responses
        - 0.7 ~ 1.0: high-quality responses

    Args:
        group_size: number of sampled responses per question (GRPO paper default is 8~16)
        seed: random seed, to ensure reproducibility
    Returns:
        rewards: numpy array of shape (group_size,)
    """
    np.random.seed(seed)
    # Simulate the rewards for a group of responses: mostly medium, a few good or bad
    rewards = np.array([0.35, 0.52, 0.68, 0.41, 0.89, 0.73, 0.28, 0.61])
    return rewards


def simulate_critic_predictions(rewards, noise_scale=0.08, seed=123):
    """
    Simulate the Critic network's value predictions in PPO

    In reality, the Critic is an independently trained neural network; here we simulate
    it by adding noise:
        V(s) ≈ reward + noise
    Critic predictions are usually close to, but not exactly, the true reward.

    Args:
        rewards: the array of true rewards
        noise_scale: the noise standard deviation, simulating the Critic's prediction error
        seed: random seed
    Returns:
        value_predictions: numpy array, the simulated Critic value predictions
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_scale, size=rewards.shape)
    value_predictions = rewards + noise
    # Ensure the value predictions stay within a reasonable range
    value_predictions = np.clip(value_predictions, 0.0, 1.0)
    return value_predictions


# ==========================================
# Part 3: Step-by-step walkthrough of the GRPO computation
# ==========================================
def demonstrate_grpo_step_by_step():
    """
    Fully demonstrate GRPO's group-relative normalization process, printing detailed
    values at each step

    Steps:
        Step 1: show the raw rewards
        Step 2: compute the group statistics (mean and standard deviation)
        Step 3: normalize to obtain the GRPO advantages
        Step 4: sort by advantage
        Step 5: compare against PPO's Critic-baseline advantages
    """
    group_size = 8

    print("=" * 70)
    print("  GRPO Group-Relative Normalization Mechanism — Step-by-Step Demo")
    print("=" * 70)

    # ---------- Step 1: raw rewards ----------
    rewards = generate_synthetic_rewards(group_size=group_size)
    print(f"\n[Step 1] Raw rewards (group_size = {group_size})")
    print("-" * 70)
    print("  Scenario: give the model a question, generate 8 different responses, "
          "score them with a reward function")
    print()
    for i, r in enumerate(rewards):
        bar = "|" * int(r * 30)  # a simple text bar chart
        print(f"  Response {i+1}: reward = {r:.2f}  {bar}")

    # ---------- Step 2: group statistics ----------
    mean_r = rewards.mean()
    std_r = rewards.std()
    print(f"\n[Step 2] Group statistics")
    print("-" * 70)
    print(f"  Group mean(r) = {mean_r:.4f}")
    print(f"  Group std(r) = {std_r:.4f}")
    print(f"  std(r) + 1e-8 = {std_r + 1e-8:.4f}  (epsilon added to prevent division by zero)")

    # ---------- Step 3: GRPO normalized advantages ----------
    grpo_advantages = compute_grpo_advantages(rewards)
    print(f"\n[Step 3] GRPO normalized advantage = (reward - mean) / (std + eps)")
    print("-" * 70)
    for i in range(group_size):
        adv = grpo_advantages[i]
        sign = "+" if adv >= 0 else ""
        print(f"  Response {i+1}: ({rewards[i]:.2f} - {mean_r:.4f}) / {std_r + 1e-8:.4f}"
              f" = {sign}{adv:.4f}")

    print()
    print(f"  Mean of advantages: {grpo_advantages.mean():.6f}  (theoretically should be near 0)")
    print(f"  Std of advantages: {grpo_advantages.std():.6f}  (theoretically should be near 1)")

    # ---------- Step 4: sort by advantage ----------
    sorted_indices = np.argsort(grpo_advantages)[::-1]  # descending order
    print(f"\n[Step 4] Sorted by GRPO advantage (high to low)")
    print("-" * 70)
    print(f"  {'Rank':>4s}  {'Resp':>4s}  {'RawRwd':>8s}  {'GRPOAdv':>10s}  {'Interpretation'}")
    print(f"  {'----':>4s}  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  {'----'}")
    for rank, idx in enumerate(sorted_indices):
        adv = grpo_advantages[idx]
        if adv > 0.5:
            interpretation = "significantly above group average, strongly encouraged"
        elif adv > 0:
            interpretation = "slightly above group average, moderately encouraged"
        elif adv > -0.5:
            interpretation = "slightly below group average, moderately discouraged"
        else:
            interpretation = "significantly below group average, strongly discouraged"
        print(f"  {rank+1:>4d}  {idx+1:>4d}  {rewards[idx]:>8.4f}  {adv:>+10.4f}  {interpretation}")

    # ---------- Step 5: comparison with PPO ----------
    value_preds = simulate_critic_predictions(rewards)
    ppo_advantages = compute_ppo_advantages(rewards, value_preds)
    print(f"\n[Step 5] GRPO vs PPO advantage comparison")
    print("-" * 70)
    print(f"  {'Resp':>4s}  {'RawRwd':>8s}  {'CriticPred':>10s}  "
          f"{'PPOAdv':>10s}  {'GRPOAdv':>10s}  {'Diff':>8s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  "
          f"{'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    for i in range(group_size):
        diff = grpo_advantages[i] - ppo_advantages[i]
        print(f"  {i+1:>4d}  {rewards[i]:>8.4f}  {value_preds[i]:>10.4f}  "
              f"{ppo_advantages[i]:>+10.4f}  {grpo_advantages[i]:>+10.4f}  {diff:>+8.4f}")

    print()
    print(f"  PPO advantages  -> mean: {ppo_advantages.mean():.4f}, std: {ppo_advantages.std():.4f}")
    print(f"  GRPO advantages -> mean: {grpo_advantages.mean():.6f}, std: {grpo_advantages.std():.4f}")
    print()
    print("  Key difference:")
    print("    PPO  advantage = reward - V(s), requires training an extra Critic network")
    print("    GRPO advantage = (reward - mean) / std, relies only on group statistics, no Critic needed")

    return rewards, grpo_advantages, ppo_advantages


# ==========================================
# Part 4: Visualization — reward before/after normalization
# ==========================================
def plot_reward_normalization(rewards, grpo_advantages):
    """
    Plot a bar chart comparing rewards before and after normalization

    Left plot: raw rewards (between 0 and 1)
    Right plot: GRPO-normalized advantages (fluctuating around 0)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    group_size = len(rewards)
    x = np.arange(group_size)

    # Left plot: raw rewards
    colors_raw = []
    for r in rewards:
        if r >= 0.7:
            colors_raw.append('#2ecc71')   # green: high quality
        elif r >= 0.4:
            colors_raw.append('#f39c12')   # orange: medium quality
        else:
            colors_raw.append('#e74c3c')   # red: low quality

    axes[0].bar(x, rewards, color=colors_raw, edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=rewards.mean(), color='black', linestyle='--',
                     linewidth=1.5, label=f'mean = {rewards.mean():.3f}')
    axes[0].set_xlabel('Response index', fontsize=12)
    axes[0].set_ylabel('Raw reward', fontsize=12)
    axes[0].set_title('Before normalization: raw reward', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right plot: GRPO-normalized advantage
    colors_adv = []
    for a in grpo_advantages:
        if a > 0:
            colors_adv.append('#27ae60')   # green: positive advantage (encouraged)
        else:
            colors_adv.append('#c0392b')   # red: negative advantage (discouraged)

    axes[1].bar(x, grpo_advantages, color=colors_adv, edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    axes[1].set_xlabel('Response index', fontsize=12)
    axes[1].set_ylabel('GRPO advantage', fontsize=12)
    axes[1].set_title('After normalization: GRPO advantage', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add annotations
    axes[1].annotate('Positive advantage -> encouraged', xy=(4.5, grpo_advantages[4]),
                      xytext=(5.5, grpo_advantages[4] + 0.3),
                      fontsize=10, color='#27ae60',
                      arrowprops=dict(arrowstyle='->', color='#27ae60'))
    axes[1].annotate('Negative advantage -> discouraged', xy=(6, grpo_advantages[6]),
                      xytext=(0.5, grpo_advantages[6] - 0.4),
                      fontsize=10, color='#c0392b',
                      arrowprops=dict(arrowstyle='->', color='#c0392b'))

    plt.suptitle('Effect of GRPO group-relative normalization', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('output/grpo_reward_normalization.png', dpi=150, bbox_inches='tight')
    print("  Reward normalization comparison chart saved to output/grpo_reward_normalization.png")
    plt.show()


# ==========================================
# Part 5: Visualization — GRPO vs PPO advantage distribution comparison
# ==========================================
def plot_advantage_comparison(grpo_advantages, ppo_advantages):
    """
    Plot a comparison of the GRPO and PPO advantage distributions

    Top plot: side-by-side bar chart of advantages from both methods
    Bottom plot: histogram of the advantage-value distributions
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    group_size = len(grpo_advantages)
    x = np.arange(group_size)
    bar_width = 0.35

    # Top plot: side-by-side bar chart
    bars1 = axes[0].bar(x - bar_width/2, grpo_advantages, bar_width,
                        label='GRPO advantage (group-relative normalization)', color='#3498db',
                        edgecolor='white', linewidth=1.0)
    bars2 = axes[0].bar(x + bar_width/2, ppo_advantages, bar_width,
                        label='PPO advantage (Critic baseline)', color='#e67e22',
                        edgecolor='white', linewidth=1.0)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('Response index', fontsize=12)
    axes[0].set_ylabel('Advantage', fontsize=12)
    axes[0].set_title('GRPO vs PPO advantage comparison (per response)', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Bottom plot: distribution histogram
    bins = np.linspace(-2, 2, 20)
    axes[1].hist(grpo_advantages, bins=bins, alpha=0.6, color='#3498db',
                 label='GRPO advantage distribution', edgecolor='white')
    axes[1].hist(ppo_advantages, bins=bins, alpha=0.6, color='#e67e22',
                 label='PPO advantage distribution', edgecolor='white')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('Advantage', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('GRPO vs PPO advantage distribution comparison', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (
        f"GRPO: mean={grpo_advantages.mean():.4f}, std={grpo_advantages.std():.4f}\n"
        f"PPO:  mean={ppo_advantages.mean():.4f}, std={ppo_advantages.std():.4f}"
    )
    axes[1].text(0.02, 0.95, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/grpo_vs_ppo_advantages.png', dpi=150, bbox_inches='tight')
    print("  Advantage distribution comparison chart saved to output/grpo_vs_ppo_advantages.png")
    plt.show()


# ==========================================
# Part 6: Multiple experiments — demonstrating GRPO's stability
# ==========================================
def run_multiple_groups(num_groups=5, group_size=8):
    """
    Run multiple experiments to show GRPO's behavior under different reward distributions

    Compares the statistical properties of GRPO vs. PPO advantages:
        - Whether the mean stays reliably near 0
        - Whether the standard deviation stays reliably near 1
    """
    print("\n" + "=" * 70)
    print("  Multiple Experiments: Stability of GRPO Normalization")
    print("=" * 70)
    print()
    print(f"  Setup: {num_groups} groups, {group_size} responses per group")
    print()

    grpo_stats = {"mean": [], "std": []}
    ppo_stats = {"mean": [], "std": []}

    for g in range(num_groups):
        # Randomly generate a different reward distribution for each group
        np.random.seed(g * 10 + 7)
        # Simulate problems of varying difficulty: different reward distributions
        base = np.random.uniform(0.2, 0.8)
        spread = np.random.uniform(0.1, 0.3)
        rewards = np.clip(np.random.normal(base, spread, size=group_size), 0.0, 1.0)

        # Compute GRPO advantages
        grpo_adv = compute_grpo_advantages(rewards)
        grpo_stats["mean"].append(grpo_adv.mean())
        grpo_stats["std"].append(grpo_adv.std())

        # Compute PPO advantages
        value_preds = simulate_critic_predictions(rewards, noise_scale=0.1, seed=g * 5 + 3)
        ppo_adv = compute_ppo_advantages(rewards, value_preds)
        ppo_stats["mean"].append(ppo_adv.mean())
        ppo_stats["std"].append(ppo_adv.std())

        print(f"  Group {g+1}:")
        print(f"    Raw reward: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        print(f"    GRPO advantage: mean={grpo_adv.mean():.6f}, std={grpo_adv.std():.4f}")
        print(f"    PPO  advantage: mean={ppo_adv.mean():.4f}, std={ppo_adv.std():.4f}")

    print()
    print("  [Summary statistics]")
    print(f"  GRPO advantage mean -> mean: {np.mean(grpo_stats['mean']):.6f}, "
          f"std: {np.std(grpo_stats['mean']):.6f}")
    print(f"  GRPO advantage std -> mean: {np.mean(grpo_stats['std']):.4f}, "
          f"std: {np.std(grpo_stats['std']):.4f}")
    print()
    print(f"  PPO  advantage mean -> mean: {np.mean(ppo_stats['mean']):.4f}, "
          f"std: {np.std(ppo_stats['mean']):.4f}")
    print(f"  PPO  advantage std -> mean: {np.mean(ppo_stats['std']):.4f}, "
          f"std: {np.std(ppo_stats['std']):.4f}")
    print()
    print("  Conclusion: GRPO's advantage mean is strictly 0 (mathematically guaranteed), "
          "and its standard deviation is strictly 1.")
    print("  Whereas PPO's advantage statistics depend on the quality of the Critic network, and vary.")


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    # Step-by-step demo of the core GRPO computation
    rewards, grpo_advantages, ppo_advantages = demonstrate_grpo_step_by_step()

    # Visualization: reward before/after normalization
    print("\n" + "=" * 70)
    print("  Generating visualizations...")
    print("=" * 70)
    plot_reward_normalization(rewards, grpo_advantages)

    # Visualization: GRPO vs PPO advantage distribution comparison
    plot_advantage_comparison(grpo_advantages, ppo_advantages)

    # Multiple experiments
    run_multiple_groups()

    # Final summary
    print("\n" + "=" * 70)
    print("  GRPO Core Mechanism Summary")
    print("=" * 70)
    print("""
  The core idea of GRPO (Group Relative Policy Optimization):

  1. For the same question, generate a group of multiple responses
  2. Score each response using a reward function (rule-based or a model)
  3. Normalize within the group:
     advantage = (reward - mean) / (std + eps)
  4. Update the policy using the normalized advantage values

  Advantages over PPO:
    - No need to train a Critic (Value Network), saving substantial compute
    - Group-relative normalization naturally guarantees a mean advantage of 0 and variance of 1
    - A simpler training pipeline with fewer hyperparameters

  Advantages over DPO:
    - No need for paired preference data (chosen/rejected)
    - Only requires a verifiable reward signal (rule-based or model-based)
    - Especially well suited to tasks with a clear correct answer, like math reasoning (RLVR)

  DeepSeek-R1 achieved its leap in reasoning ability precisely by using GRPO + RLVR.
    """)
