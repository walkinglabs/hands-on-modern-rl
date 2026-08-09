"""
Chapter 1: Training CartPole with PPO from Stable-Baselines3

Training progress is logged via SwanLab (reward curve, losses, etc.), and
a GUI window can optionally pop up after training to show off what the
agent learned.

Usage:
    # Default: train + SwanLab curves (no GUI, fast)
    python 1-ppo_cartpole.py

    # Show the GUI demo (pops up the cart animation window after training)
    python 1-ppo_cartpole.py --gui

About the --gui flag:
    Training is always headless (no rendering), so its speed is unaffected
    by the GUI setting.
    --gui only controls whether the CartPole animation window pops up
    during the demo stage after training finishes.
    With the GUI on, the demo stage has to wait for a screen refresh
    (~16ms) each frame, so it runs noticeably slower;
    with the GUI off, the demo stage is pure computation and finishes in
    a few seconds.
"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from swanlab.integration.sb3 import SwanLabCallback
import swanlab


class LogApproxKL(BaseCallback):
    """Backfills train/approx_kl into SwanLab.

    SB3's PPO.train() records this metric internally via
    logger.record("train/approx_kl", ...), but the value is a
    numpy.float32. SwanLab's SB3 callback does a type check in write()
    using isinstance(value, (int, float)), which numpy.float32 fails
    (numpy.float64 and Python float pass), so approx_kl gets silently
    skipped.

    This callback runs after every train() call, pulls the approx_kl
    value out of the logger's cache, converts it to a Python float, and
    logs it directly via swanlab.log to backfill it.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # train() has already finished running by the time _on_rollout_end
        # fires, so the logger's cache contains all of this round's metrics.
        logger = self.model.logger
        if hasattr(logger, "name_to_value") and "train/approx_kl" in logger.name_to_value:
            value = float(logger.name_to_value["train/approx_kl"])
            swanlab.log({"train/approx_kl": value}, step=self.num_timesteps)


class RestoreStdoutLog(BaseCallback):
    """Adds back the scrolling log table SB3 normally prints to the terminal.

    SwanLabCallback._init_callback() internally calls
    self.model.set_logger(...), which wholesale-replaces SB3's default
    logger with a "SwanLab-only" logger, incidentally removing the
    HumanOutputFormat that's responsible for printing the
    ep_rew_mean / fps / approx_kl table to stdout (i.e. the verbose=1
    scrolling log).

    This callback runs during the _init_callback phase (by which point
    SwanLabCallback has already swapped the logger out), and adds a
    stdout output sink back onto the current logger, so the terminal
    starts scrolling again while SwanLab logging is unaffected. It needs
    to come after SwanLabCallback in the callback list.
    """

    def _init_callback(self) -> None:
        # SwanLabCallback has already replaced the logger with one that
        # only contains SwanLabOutputFormat; here we add back a stdout
        # output sink to restore the verbose=1 scrolling table.
        self.model.logger.output_formats.append(HumanOutputFormat(sys.stdout))

    def _on_step(self) -> bool:
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO CartPole training")
    parser.add_argument(
        "--gui", action="store_true",
        help="Pop up a GUI window to demo the agent after training finishes (off by default, only prints scores)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    # ==========================================
    # Stage 1: Training
    # ==========================================
    env = gym.make("CartPole-v1")

    # Print environment info (observation space, action space, termination thresholds)
    print("=" * 50)
    print("CartPole-v1 environment info")
    print("=" * 50)
    print(f"  Observation space:  {env.observation_space}")
    print(f"  Action space:  {env.action_space}")
    print(f"  Observation upper bound:  {env.observation_space.high}")
    print(f"  Observation lower bound:  {env.observation_space.low}")
    print(f"  Termination condition:  position > ±{env.unwrapped.x_threshold}, "
          f"angle > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
          f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)")
    print("=" * 50)

    model = PPO("MlpPolicy", env, verbose=1)

    print("Starting training (with SwanLab logging)...")
    swanlab_cb = SwanLabCallback(
        project="cartpole-ppo",
        experiment_name="PPO-CartPole-v1",
        mode="local",
    )
    model.learn(
        total_timesteps=80000,
        callback=[swanlab_cb, RestoreStdoutLog(), LogApproxKL()],
    )

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Training complete! Mean reward: {mean_reward} +/- {std_reward}")

    model.save("output/ppo_cartpole")
    env.close()

    # ==========================================
    # Stage 2: Demoing what the agent learned
    # ==========================================
    print("\nDemoing what the agent learned...")
    render_mode = "human" if args.gui else None
    vis_env = gym.make("CartPole-v1", render_mode=render_mode)
    model = PPO.load("output/ppo_cartpole")

    for episode in range(5):
        obs, info = vis_env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vis_env.step(action)
            score += reward
        print(f"  Episode {episode + 1} score: {score}")

    vis_env.close()

    if args.gui:
        print("\nGUI demo finished.")
    else:
        print("\nTip: add --gui to pop up the cart animation window and watch the demo.")

    print("SwanLab experiment dashboard: swanlab watch swanlog")


if __name__ == "__main__":
    main()
