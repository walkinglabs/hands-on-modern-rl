"""
Chapter 6: Training Pendulum-v1 with A2C (Advantage Actor-Critic)
-- Demonstrates the Gaussian policy Actor-Critic uses in a continuous action space

Usage:
    python actor_critic_pendulum.py
    python actor_critic_pendulum.py --total-timesteps 20000     # quick sanity check
    python actor_critic_pendulum.py --total-timesteps 300000    # full training

What Pendulum-v1 teaches:
    1. The action is a 1-dimensional continuous torque in the range [-2, 2]
    2. The Actor outputs a continuous action distribution rather than discrete action probabilities
    3. The Critic estimates V(s), using the advantage to reduce policy gradient variance
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


os.makedirs("output", exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pendulum-v1 with A2C")
    parser.add_argument("--total-timesteps", type=int, default=300_000,
                        help="Total number of training steps (default 300000)")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel environments (default 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of episodes for the final evaluation")
    return parser.parse_args()


def make_env(seed, rank):
    def _init():
        env = gym.make("Pendulum-v1")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


class TrainingMonitorCallback(BaseCallback):
    """Records episode rewards, policy entropy, value loss, and policy loss."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.timesteps = []
        self.entropy_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.update_numbers = set()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        logger_values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = logger_values.get("train/entropy_loss")
        policy_loss = logger_values.get("train/policy_loss")
        value_loss = logger_values.get("train/value_loss")
        n_updates = logger_values.get("train/n_updates")

        if entropy_loss is not None and n_updates not in self.update_numbers:
            self.update_numbers.add(n_updates)
            self.timesteps.append(self.num_timesteps)
            self.entropy_losses.append(float(entropy_loss))
            self.policy_losses.append(float(policy_loss or 0.0))
            self.value_losses.append(float(value_loss or 0.0))

        return True


def moving_average(values, window):
    if not values:
        return np.array([])
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def moving_average_xy(values, window):
    averaged = moving_average(values, window)
    start = 1 if len(values) < window else window
    x_values = np.arange(start, start + len(averaged))
    return x_values, averaged


def save_plots(callback, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    rewards = callback.episode_rewards
    if rewards:
        episodes = np.arange(1, len(rewards) + 1)
        smooth_x, smooth_rewards = moving_average_xy(rewards, 20)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards, color="#90CAF9", alpha=0.45, linewidth=1.0, label="Raw return")
        ax.plot(smooth_x, smooth_rewards, color="#1565C0", linewidth=2.0,
                label="20-episode moving average")
        ax.axhline(y=-800, color="green", linestyle="--", alpha=0.6,
                   label="A2C baseline reference line (-800)")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.35)
        ax.set_title("A2C Pendulum-v1 Episode Reward", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_pendulum_reward.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    if callback.entropy_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(callback.timesteps, callback.entropy_losses,
                color="#EF6C00", linewidth=1.5)
        ax.set_title("A2C Pendulum-v1 Policy Entropy Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("entropy_loss (negative entropy)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_pendulum_entropy.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    if callback.policy_losses and callback.value_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(callback.timesteps, callback.policy_losses,
                color="#00897B", linewidth=1.5, label="Policy loss")
        ax.plot(callback.timesteps, callback.value_losses,
                color="#C62828", linewidth=1.5, label="Value loss")
        ax.set_title("A2C Pendulum-v1 Actor/Critic Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_pendulum_loss.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("Training curves saved to output/actor_critic_pendulum_*.png")


def main():
    args = parse_args()

    print("=" * 50)
    print("Chapter 6: A2C Training on Pendulum-v1")
    print("=" * 50)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs:   {args.num_envs}")
    print("Action space:    continuous 1-dimensional torque [-2, 2]")

    vec_env = DummyVecEnv([make_env(args.seed, i) for i in range(args.num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
    )

    callback = TrainingMonitorCallback()
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)

    output_dir = Path("output")
    model.save(output_dir / "actor_critic_pendulum")
    vec_env.save(output_dir / "actor_critic_pendulum_vecnormalize.pkl")
    print("\nModel saved to output/actor_critic_pendulum.zip")
    print("Normalization statistics saved to output/actor_critic_pendulum_vecnormalize.pkl")
    save_plots(callback, output_dir)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    eval_env = VecNormalize.load(
        output_dir / "actor_critic_pendulum_vecnormalize.pkl", eval_env
    )
    eval_env.training = False
    eval_env.norm_reward = False

    episode_rewards = []
    for episode in range(args.eval_episodes):
        eval_env.seed(args.seed + 10_000 + episode)
        obs = eval_env.reset()
        done = [False]
        total_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += float(reward[0])
        episode_rewards.append(total_reward)

    eval_env.close()
    print("\nFinal deterministic policy evaluation:")
    print(f"  Mean reward:   {np.mean(episode_rewards):.1f}")
    print(f"  Std dev:       {np.std(episode_rewards):.1f}")
    print(f"  Best episode:  {np.max(episode_rewards):.1f}")
    print(f"  Worst episode: {np.min(episode_rewards):.1f}")


if __name__ == "__main__":
    main()
