"""
Chapter 7: Training BipedalWalker-v3 with Stable-Baselines3's PPO
——Demonstrating PPO's capability on a continuous action space

How to run:
    python ppo_bipedal_walker.py
    python ppo_bipedal_walker.py --total-timesteps 100000    # quick check
    python ppo_bipedal_walker.py --total-timesteps 2000000   # full training

Why BipedalWalker-v3 is instructive:
    1. Continuous action space (4-dim joint torques) — DQN cannot handle this directly, PPO natively supports it
    2. 24-dim observation space (10 lidar rays + joint angles + velocities)
    3. Harder than LunarLander, requires longer training time
    4. The environment's "solved" criterion: average score over 100 episodes >= 300
"""

import argparse
import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Create output directory
os.makedirs("output", exist_ok=True)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train BipedalWalker-v3 with PPO")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps (default 1000000)")
    return parser.parse_args()


# ==========================================
# Part 1: Custom training callback — record key metrics
# ==========================================
class TrainingMonitorCallback(BaseCallback):
    """
    Custom callback: records PPO's key training metrics after each rollout ends
    Includes: episode reward, policy entropy, clip fraction, approximate KL divergence
    Also supports saving checkpoint models at specified step counts
    """

    def __init__(self, check_freq=2048, checkpoint_steps=None, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_steps = checkpoint_steps or []
        self.episode_rewards = []
        self.entropy_list = []
        self.clip_fraction_list = []
        self.approx_kl_list = []
        self.timesteps_list = []
        self._saved_checkpoints = set()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(
                    ep_info["r"] if isinstance(ep_info, dict) else ep_info
                )

        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                name_to_value = logger.name_to_value

                entropy = name_to_value.get("train/entropy_loss", 0)
                clip_frac = name_to_value.get("train/clip_fraction", 0)
                approx_kl = name_to_value.get("train/approx_kl", 0)

                self.entropy_list.append(entropy)
                self.clip_fraction_list.append(clip_frac)
                self.approx_kl_list.append(approx_kl)
                self.timesteps_list.append(self.num_timesteps)

        # Save checkpoints
        for ckpt_step in self.checkpoint_steps:
            if self.num_timesteps >= ckpt_step and ckpt_step not in self._saved_checkpoints:
                path = f"output/ppo_bipedal_walker_{ckpt_step // 1000}k"
                self.model.save(path)
                print(f"\n  [Checkpoint] Saved {ckpt_step // 1000}k-step model → {path}.zip")
                self._saved_checkpoints.add(ckpt_step)

        return True


# ==========================================
# Part 2: Create the vectorized environment
# ==========================================
args = parse_args()

print("=" * 50)
print("Chapter 7: Training BipedalWalker-v3 with PPO")
print("=" * 50)

print("\nCreating vectorized environment (8 parallel envs)...")

def make_env():
    """Environment factory function"""
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

num_envs = 8
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
print(f"Created {num_envs} parallel environments")


# ==========================================
# Part 3: Configure PPO hyperparameters
# ==========================================
print("\nConfiguring PPO hyperparameters...")

model = PPO(
    policy="MlpPolicy",       # Multi-layer perceptron policy
    env=vec_env,              # Vectorized environment
    learning_rate=3e-4,       # Learning rate
    n_steps=2048,             # Steps collected per rollout (per environment)
    batch_size=256,           # Mini-batch size (larger than LunarLander, for better stability)
    n_epochs=10,              # Update rounds per batch of data
    clip_range=0.2,           # PPO clip range
    ent_coef=0.005,           # Entropy coefficient (continuous spaces have more intrinsic exploration, so slightly lower is fine)
    vf_coef=0.5,              # Value function loss coefficient
    gamma=0.99,               # Discount factor
    gae_lambda=0.95,          # GAE lambda
    verbose=1,
    seed=42,
    device="auto",
)

clip_val = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
print(f"  Learning rate:  {model.learning_rate}")
print(f"  Rollout steps:  {model.n_steps}")
print(f"  Batch size:     {model.batch_size}")
print(f"  Update epochs:  {model.n_epochs}")
print(f"  Clip range:     [{1 - clip_val:.1f}, {1 + clip_val:.1f}]")
print(f"  Entropy coef:   {model.ent_coef}")
print(f"  Action space:   continuous, {vec_env.num_envs} dims (joint torques)")


# ==========================================
# Part 4: Train the model
# ==========================================
total_timesteps = args.total_timesteps
print(f"\nStarting training ({total_timesteps:,} timesteps)...")
print("-" * 50)

# Automatically save checkpoints during training (for three-stage comparison playback)
checkpoint_steps = []
if total_timesteps >= 500_000:
    checkpoint_steps = [100_000, 500_000]
callback = TrainingMonitorCallback(check_freq=2048, checkpoint_steps=checkpoint_steps)

model.learn(
    total_timesteps=total_timesteps,
    callback=callback,
    progress_bar=True,
)

print("-" * 50)
print("Training complete!")


# ==========================================
# Part 5: Plot training curves (4 separate figures)
# ==========================================
print("\nPlotting training curves...")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Figure 1: episode reward curve (raw values + moving average)
if callback.episode_rewards:
    rewards = callback.episode_rewards
    window = min(50, max(1, len(rewards)))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, color="#90CAF9", alpha=0.4, linewidth=0.8, label="Raw")
    x_smooth = np.arange(window // 2, window // 2 + len(smoothed))
    ax.plot(x_smooth, smoothed, color="#1565C0", alpha=0.9, linewidth=1.8, label="50-episode moving average")
    ax.axhline(y=300, color="green", linestyle="--", alpha=0.5, label="solved (300)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("PPO BipedalWalker-v3 Episode Reward", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_reward.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Reward curve → output/ppo_bipedal_walker_reward.png")

# Figure 2: policy entropy
if callback.entropy_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.entropy_list,
            color="#FF9800", alpha=0.8, linewidth=1.5)
    ax.set_title("PPO BipedalWalker-v3 Policy Entropy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Policy entropy curve → output/ppo_bipedal_walker_entropy.png")

# Figure 3: clip fraction
if callback.clip_fraction_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.clip_fraction_list,
            color="#F44336", alpha=0.8, linewidth=1.5, label="clip fraction")
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range=0.2")
    ax.set_title("PPO BipedalWalker-v3 Clip Fraction", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fraction clipped")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_clip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Clip fraction curve → output/ppo_bipedal_walker_clip.png")

# Figure 4: approximate KL divergence
if callback.approx_kl_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.approx_kl_list,
            color="#4CAF50", alpha=0.8, linewidth=1.5)
    ax.set_title("PPO BipedalWalker-v3 Approximate KL Divergence", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("KL divergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_kl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  KL divergence curve → output/ppo_bipedal_walker_kl.png")


# ==========================================
# Part 6: Evaluate the trained model
# ==========================================
print("\nEvaluating the final model (20 test episodes)...")
print("-" * 50)

eval_env = gym.make("BipedalWalker-v3")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"20-episode test result:")
print(f"  Mean reward: {mean_reward:.2f}")
print(f"  Std dev:     {std_reward:.2f}")

test_rewards = []
for ep in range(20):
    obs, _ = eval_env.reset()
    done, truncated = False, False
    total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)

print(f"\nPer-episode rewards:")
for i, r in enumerate(test_rewards):
    status = "solved" if r >= 300 else ("moderate" if r >= 100 else "not solved")
    print(f"  Episode {i + 1:2d}: {r:8.2f}  [{status}]")

print(f"\nSolve rate (>= 300 points): {sum(1 for r in test_rewards if r >= 300)}/20")
eval_env.close()


# ==========================================
# Part 7: Save the model
# ==========================================
model.save("output/ppo_bipedal_walker")
print(f"\nModel saved to: output/ppo_bipedal_walker.zip")
print("=" * 50)
