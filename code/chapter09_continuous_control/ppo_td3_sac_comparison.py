"""
Chapter 9: Continuous Control Algorithm Showdown —— PPO vs TD3 vs SAC
——A fair comparison of three mainstream continuous control algorithms under the same environment

Usage:
    python ppo_td3_sac_comparison.py

Core differences between the three algorithms:

    PPO (Proximal Policy Optimization) —— the simple, robust "veteran"
        Type: on-policy
        Policy: stochastic (Gaussian distribution)
        Core mechanism: clipped objective function, limits how much the policy can change per update
        Pros: simple to implement, robust to hyperparameters, stable training
        Cons: low sample efficiency (data can only be used once)
        Use cases: fast prototyping, scenarios that demand high stability

    TD3 (Twin Delayed DDPG) —— the "perfectionist" of deterministic policies
        Type: off-policy
        Policy: deterministic (outputs actions directly)
        Core mechanism: twin Q-networks + delayed policy updates + target policy smoothing
        Pros: high sample efficiency, strong performance on deterministic tasks
        Cons: insufficient exploration from a deterministic policy, sensitive to hyperparameters
        Use cases: high-dimensional action spaces, fine-grained control with sparse rewards

    SAC (Soft Actor-Critic) —— the "all-rounder" of maximum entropy reinforcement learning
        Type: off-policy
        Policy: stochastic (Gaussian distribution + entropy regularization)
        Core mechanism: entropy regularization + automatic temperature tuning + twin Q-networks
        Pros: high sample efficiency, thorough exploration, strong robustness
        Cons: slightly higher computational cost, more complex in theory
        Use cases: general-purpose continuous control, scenarios that need strong exploration

Keys to a fair comparison:
    - Same environment (HalfCheetah-v4)
    - Same training budget (50000 timesteps)
    - Same random seed
    - Same network architecture size
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# Create output directory
os.makedirs("output", exist_ok=True)

# Configure CJK font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Shared training callback —— logs episode rewards
# ==========================================
class RewardCallback(BaseCallback):
    """
    Shared training callback: logs episode rewards during training for each algorithm

    This callback works for any SB3 algorithm because episode rewards
    are consistently obtained via info["episode"]["r"].
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []  # Timestep at which each episode ended

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_timesteps.append(self.num_timesteps)
        return True


# ==========================================
# Part 2: Environment configuration
# ==========================================
print("=" * 60)
print("Chapter 9: Continuous Control Algorithm Showdown — PPO vs TD3 vs SAC")
print("=" * 60)

# Try to use HalfCheetah-v4 (requires MuJoCo)
# Fall back to Pendulum-v1 if MuJoCo is unavailable
ENV_NAME = "HalfCheetah-v4"
try:
    test_env = gym.make(ENV_NAME)
    test_env.reset()
    test_env.close()
    print(f"\nUsing environment: {ENV_NAME} (MuJoCo continuous control)")
except Exception as e:
    ENV_NAME = "Pendulum-v1"
    print(f"\nMuJoCo unavailable ({e}), falling back to: {ENV_NAME}")

# Print environment info
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_type = "continuous" if isinstance(env.action_space, gym.spaces.Box) else "discrete"
print(f"  State dimension:  {state_dim}")
print(f"  Action dimension: {action_dim}")
print(f"  Action type:      {action_type}")
env.close()

# Shared training parameters
TOTAL_TIMESTEPS = 50_000    # Training budget (for demonstration; real use needs 1M+)
SEED = 42                   # Shared random seed
NET_ARCH = [256, 256]       # Shared network architecture


# ==========================================
# Part 3: Train the three algorithms
# ==========================================

# ---- Algorithm 1: PPO ----
print("\n" + "-" * 60)
print("[1/3] Training PPO (Proximal Policy Optimization)")
print("-" * 60)
print("  Characteristics: on-policy, clipped objective, simple but sample-inefficient")

# Key PPO hyperparameters:
#   - n_steps=2048: number of steps collected per rollout, PPO's "batch size"
#   - batch_size=64: mini-batch update size
#   - n_epochs=10: the same batch of data is reused 10 times
#   - clip_range=0.2: clipping range for the policy ratio
#   - ent_coef=0.01: entropy coefficient (set manually, unlike SAC's automatic tuning)
ppo_model = PPO(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

ppo_callback = RewardCallback()
ppo_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=ppo_callback,
    progress_bar=True,
)
print(f"  PPO training complete, {len(ppo_callback.episode_rewards)} episodes total")


# ---- Algorithm 2: TD3 ----
print("\n" + "-" * 60)
print("[2/3] Training TD3 (Twin Delayed DDPG)")
print("-" * 60)
print("  Characteristics: off-policy, deterministic policy, twin Q-networks, delayed updates")

# TD3's three core improvements (building on DDPG):
#   1. Twin Q-networks (Clipped Double-Q): take the smaller of two Q-values
#      → mitigates Q-value overestimation
#   2. Delayed Policy Updates:
#      → the actor is updated only after the critic has been updated several times
#      → policy_delay=2 means the actor is updated once for every 2 critic updates
#   3. Target Policy Smoothing:
#      → adds noise to the target action to prevent sharp spikes in the Q-value at certain actions
td3_model = TD3(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,           # Delayed policy update: update the actor once every 2 critic updates
    action_noise=None,        # Action noise (TD3 uses its own internal exploration noise)
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

td3_callback = RewardCallback()
td3_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=td3_callback,
    progress_bar=True,
)
print(f"  TD3 training complete, {len(td3_callback.episode_rewards)} episodes total")


# ---- Algorithm 3: SAC ----
print("\n" + "-" * 60)
print("[3/3] Training SAC (Soft Actor-Critic)")
print("-" * 60)
print("  Characteristics: off-policy, stochastic policy, entropy regularization, automatic temperature tuning")

# SAC's core innovation —— the maximum entropy framework:
#   Standard RL: max Σ r(s,a)
#   Maximum entropy RL: max Σ [r(s,a) + α * H(π(·|s))]
#
# where H is the policy entropy and α is the temperature parameter
# This lets SAC pursue high returns while keeping the policy stochastic
#
# How automatic temperature tuning works:
#   alpha is optimized so that the policy entropy approaches the target entropy
#   Target entropy = -dim(A) (negative of the action dimension)
#   When the policy is too deterministic (entropy too low) → alpha increases → encourages exploration
#   When the policy is too random (entropy too high) → alpha decreases → encourages exploitation
sac_model = SAC(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",          # Automatic temperature tuning (SAC's core innovation!)
    train_freq=1,
    gradient_steps=1,
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

sac_callback = RewardCallback()
sac_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=sac_callback,
    progress_bar=True,
)
print(f"  SAC training complete, {len(sac_callback.episode_rewards)} episodes total")


# ==========================================
# Part 4: Evaluate all models
# ==========================================
print("\n" + "=" * 60)
print("Evaluation phase: 10 test episodes per algorithm")
print("=" * 60)

eval_env = gym.make(ENV_NAME)
n_eval = 10

results = {}
for name, model in [("PPO", ppo_model), ("TD3", td3_model), ("SAC", sac_model)]:
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval, deterministic=True
    )
    # Test episode by episode
    test_rewards = []
    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        done, truncated = False, False
        total_r = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = eval_env.step(action)
            total_r += r
        test_rewards.append(total_r)

    results[name] = {
        "mean": mean_reward,
        "std": std_reward,
        "rewards": test_rewards,
    }
    print(f"  {name:4s}: mean reward = {mean_reward:8.2f} ± {std_reward:6.2f}")

eval_env.close()


# ==========================================
# Part 5: Plot the comparison charts
# ==========================================
print("\nPlotting comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"Continuous Control Algorithm Comparison — {ENV_NAME} ({TOTAL_TIMESTEPS:,} timesteps)",
    fontsize=16, fontweight="bold",
)

# Color scheme
colors = {"PPO": "#2196F3", "TD3": "#F44336", "SAC": "#4CAF50"}

# Subplot 1: training curve comparison (raw values)
ax1 = axes[0, 0]
for name, cb in [("PPO", ppo_callback), ("TD3", td3_callback), ("SAC", sac_callback)]:
    if cb.episode_rewards:
        ax1.plot(cb.episode_rewards, alpha=0.3, color=colors[name], linewidth=0.8)
ax1.set_title("Training Episode Reward (raw)", fontsize=13)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Cumulative Reward")
# Manually add the legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors[n], linewidth=2, label=n)
    for n in ["PPO", "TD3", "SAC"]
]
ax1.legend(handles=legend_elements)
ax1.grid(True, alpha=0.3)

# Subplot 2: training curve comparison (moving average)
ax2 = axes[0, 1]
for name, cb in [("PPO", ppo_callback), ("TD3", td3_callback), ("SAC", sac_callback)]:
    if cb.episode_rewards:
        rewards = cb.episode_rewards
        window = min(20, len(rewards))
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(rewards)), smoothed,
                     color=colors[name], linewidth=2, label=f"{name}")
ax2.set_title("Training Episode Reward (moving average)", fontsize=13)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Cumulative Reward")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: final evaluation comparison (bar chart)
ax3 = axes[1, 0]
algo_names = list(results.keys())
means = [results[n]["mean"] for n in algo_names]
stds = [results[n]["std"] for n in algo_names]
bar_colors = [colors[n] for n in algo_names]
bars = ax3.bar(algo_names, means, yerr=stds, color=bar_colors,
               alpha=0.8, capsize=5, edgecolor="white", linewidth=1.5)
ax3.set_title("Final Evaluation Comparison (10-episode average)", fontsize=13)
ax3.set_ylabel("Mean Reward")
# Annotate the bars with values
for bar, mean, std in zip(bars, means, stds):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 10,
             f"{mean:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")

# Subplot 4: test episode reward distribution (box plot)
ax4 = axes[1, 1]
box_data = [results[n]["rewards"] for n in algo_names]
bp = ax4.boxplot(box_data, labels=algo_names, patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_title("Test Episode Reward Distribution", fontsize=13)
ax4.set_ylabel("Episode Reward")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("output/ppo_td3_sac_comparison.png", dpi=150, bbox_inches="tight")
print("Comparison chart saved to: output/ppo_td3_sac_comparison.png")
plt.show()


# ==========================================
# Part 6: Print the comparison summary table
# ==========================================
print("\n" + "=" * 60)
print("Algorithm Comparison Summary Table")
print("=" * 60)

# Header
print(f"{'Metric':<20s} {'PPO':>10s} {'TD3':>10s} {'SAC':>10s}")
print("-" * 60)

# Final reward
print(f"{'Final mean reward':<18s}", end="")
for name in algo_names:
    print(f" {results[name]['mean']:>10.1f}", end="")
print()

# Reward std dev
print(f"{'Reward std dev':<18s}", end="")
for name in algo_names:
    print(f" {results[name]['std']:>10.1f}", end="")
print()

# Number of training episodes
print(f"{'Training episodes':<18s}", end="")
for cb in [ppo_callback, td3_callback, sac_callback]:
    print(f" {len(cb.episode_rewards):>10d}", end="")
print()

# Algorithm type
print(f"{'Algorithm type':<18s} {'on-policy':>10s} {'off-policy':>10s} {'off-policy':>10s}")

# Policy type
print(f"{'Policy type':<18s} {'stochastic':>10s} {'deterministic':>10s} {'stoch+entropy':>10s}")

# Sample efficiency
print(f"{'Sample efficiency':<18s} {'low':>10s} {'high':>10s} {'highest':>10s}")

# Exploration mechanism
print(f"{'Exploration':<18s} {'intrinsic':>10s} {'action noise':>10s} {'entropy reg.':>10s}")

# Hyperparameter sensitivity
print(f"{'Hyperparam sens.':<18s} {'low':>10s} {'medium':>10s} {'low':>10s}")

print("-" * 60)

# Determine the winner
winner = max(results.keys(), key=lambda k: results[k]["mean"])
print(f"\nIn this experiment, {winner} achieved the highest mean reward!")
print()
print("Notes:")
print("  - 50k timesteps is for demonstration only; real comparisons typically need 1M+ timesteps")
print("  - Rankings may differ across environments")
print("  - PPO's on-policy nature gives it unique advantages in distributed training")
print("  - SAC performs best on most MuJoCo environments (especially with longer training)")
print("  - TD3 is highly competitive in scenarios requiring precise control (e.g. robotic manipulation)")
print("=" * 60)
