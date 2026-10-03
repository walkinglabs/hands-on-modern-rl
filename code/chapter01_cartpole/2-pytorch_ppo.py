"""
Chapter 1: Cracking open the black box - implementing PPO for CartPole in
pure PyTorch, to reveal the core logic behind SB3's model.learn().

Training metrics (reward curves, losses, etc.) are logged via SwanLab, and a GUI
window can optionally pop up afterwards to show off what the agent learned.

How to run:
    # Default: train + SwanLab curves (no GUI, fast)
    python 2-pytorch_ppo.py

    # Show the GUI demo (pops up the cart animation window after training)
    python 2-pytorch_ppo.py --gui

About the --gui flag:
    Training itself is always headless (no rendering), so GUI has no effect on its speed.
    --gui only controls whether the post-training demo pops up a CartPole animation window.
    With the GUI on, the demo waits for a screen refresh (~16ms) each frame and is noticeably slower;
    with it off, the demo is pure computation and finishes in a few seconds.
"""

import argparse
import csv
import os
import random
import sys
from collections import deque
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import swanlab

from device_utils import describe_device, print_device_report, resolve_torch_device


# ==========================================
# Part 1: Actor-Critic network (separate heads + orthogonal initialization)
# ==========================================
class ActorCritic(nn.Module):
    """
    Separate Actor-Critic networks (matching SB3's MlpPolicy):
    - Actor and Critic use their own hidden layers, avoiding gradient interference
    - Orthogonal init: gain=0.01 on the actor output layer keeps the initial policy near-uniform
    """

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization, matching SB3's default"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # Use a small gain on the actor's output layer -> initial policy close to uniform
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # Critic output layer gain=1
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ==========================================
# Part 2: Collecting trajectories (Rollout)
# ==========================================
def collect_rollout(
    model,
    env,
    obs,
    device,
    episode_reward=0.0,
    episode_length=0,
    num_steps=2048,
):
    """
    Collect a trajectory, correctly handling terminated vs truncated:
    - terminated (the pole fell over): V(s')=0
    - truncated (hit the step limit): V(s') needs to be bootstrapped
    - end of rollout without termination: also bootstrap with V(s')
    """
    transitions = []
    completed_rewards = []
    completed_lengths = []

    for _ in range(num_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                _, next_value_tensor = model(next_obs_tensor)
                next_value = next_value_tensor.item()

        # Store this step's V(s') so termination, truncation, and end-of-rollout all share one GAE formula.
        transitions.append({
            "obs": obs,
            "action": action.item(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "next_value": next_value,
        })

        episode_reward += float(reward)
        episode_length += 1

        obs = next_obs
        if terminated or truncated:
            completed_rewards.append(episode_reward)
            completed_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            obs, _ = env.reset()

    return (
        transitions,
        obs,
        completed_rewards,
        completed_lengths,
        episode_reward,
        episode_length,
    )


# ==========================================
# Part 3: Computing GAE advantages
# ==========================================
def compute_gae(transitions, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation, handling correctly:
    - terminated (a genuine episode end): do not propagate GAE, V(s')=0
    - truncated (time limit): bootstrap with V(s'), but do not propagate GAE across the reset
    - end of rollout: bootstrap with the stored V(s')
    """
    raw_advantages = []
    gae = 0

    for step in reversed(range(len(transitions))):
        t = transitions[step]
        episode_end = t["terminated"] or t["truncated"]
        delta = t["reward"] + gamma * t["next_value"] - t["value"]
        gae = delta + gamma * lam * (1.0 - float(episode_end)) * gae
        raw_advantages.insert(0, gae)

    raw_advantages = torch.tensor(raw_advantages, dtype=torch.float32)
    values = torch.tensor([t["value"] for t in transitions], dtype=torch.float32)
    # The Critic learns the unnormalized return target; normalization is only for the policy loss.
    returns = raw_advantages + values
    advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )

    return advantages, returns


# ==========================================
# Part 4: PPO update
# ==========================================
def ppo_update(model, optimizer, transitions, advantages, returns, device,
               clip_eps=0.2, epochs=10, batch_size=64):
    """PPO clipped objective update"""
    obs = np.array([t["obs"] for t in transitions])
    actions = np.array([t["action"] for t in transitions])
    old_log_probs = np.array([t["log_prob"] for t in transitions])

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    total_clip_frac = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))

        for start in range(0, len(transitions), batch_size):
            idx = indices[start:start + batch_size]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = ((values - batch_returns) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.0 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                log_ratio = new_log_probs - batch_old_log_probs
                # Non-negative KL approximation, matching SB3's approx_kl computation.
                total_kl += ((log_ratio.exp() - 1) - log_ratio).mean().item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }


# ==========================================
# Part 5: Training loop
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pure PyTorch PPO CartPole training")
    parser.add_argument(
        "--gui", action="store_true",
        help="Pop up a GUI window to demo the agent after training finishes (off by default, only prints scores)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training random seed")
    parser.add_argument("--iterations", type=int, default=40, help="Number of PPO iterations")
    parser.add_argument("--steps-per-rollout", type=int, default=2048, help="Steps sampled per iteration")
    parser.add_argument(
        "--log-csv", default="output/training_metrics.csv",
        help="Where to save the raw training-metrics CSV",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training device: auto prefers CUDA, then Apple MPS, then CPU",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["local", "cloud", "disabled"],
        default="local",
        help="SwanLab logging mode; set to disabled to reproduce runs without a dashboard",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    device = resolve_torch_device(args.device)
    print_device_report(device)

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(args.log_csv)),
        "pytorch_ppo_cartpole.pth",
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = gym.make("CartPole-v1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    env.action_space.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)

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

    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    total_iterations = args.iterations
    steps_per_rollout = args.steps_per_rollout

    # Initialize SwanLab
    swanlab.init(
        project="cartpole-pytorch",
        experiment_name="PPO-PyTorch-CartPole-v1",
        mode=args.swanlab_mode,
        config={
            "algorithm": "PPO",
            "lr": 3e-4,
            "total_iterations": total_iterations,
            "steps_per_rollout": steps_per_rollout,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "epochs": 10,
            "batch_size": 64,
            "seed": args.seed,
            "device": str(device),
        },
    )

    print(f"Starting training on {describe_device(device)} (pure PyTorch PPO + SwanLab)...")
    print("-" * 60)

    total_timesteps = 0

    csv_dir = os.path.dirname(args.log_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    metric_rows = []
    ongoing_episode_reward = 0.0
    ongoing_episode_length = 0
    # Match SB3's stats_window_size=100: rolling across rollouts, not reset each iteration.
    ep_rew_buffer = deque(maxlen=100)
    ep_len_buffer = deque(maxlen=100)

    for iteration in range(total_iterations):
        # Collect data
        (
            transitions,
            obs,
            ep_rewards,
            ep_lengths,
            ongoing_episode_reward,
            ongoing_episode_length,
        ) = collect_rollout(
            model,
            env,
            obs,
            device,
            ongoing_episode_reward,
            ongoing_episode_length,
            steps_per_rollout,
        )

        total_timesteps += len(transitions)

        # Compute advantages and the Critic's unnormalized return targets
        advantages, returns = compute_gae(transitions)

        # Set the learning rate before this iteration's update.
        frac = 1.0 - iteration / total_iterations
        lr = 3e-4 * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # PPO update
        metrics = ppo_update(
            model, optimizer, transitions, advantages, returns, device
        )

        # Explained variance compares the value predictions made during rollout collection against the return targets.
        return_values = returns.numpy()
        rollout_values = np.array([t["value"] for t in transitions])
        var_returns = np.var(return_values)
        if var_returns < 1e-6:
            # All returns are identical (e.g. every episode scored 500), EV is meaningless, set to 0
            explained_variance = 0.0
        else:
            explained_variance = 1 - np.var(return_values - rollout_values) / var_returns

        mean_reward_rollout = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_ep_len_rollout = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        ep_rew_buffer.extend(ep_rewards)
        ep_len_buffer.extend(ep_lengths)
        mean_reward = float(np.mean(ep_rew_buffer)) if ep_rew_buffer else 0.0
        mean_ep_len = float(np.mean(ep_len_buffer)) if ep_len_buffer else 0.0

        # Log to SwanLab. ep_rew_mean / ep_len_mean use a rolling 100-episode window
        # (SB3's default); *_rollout keep the per-rollout means for teaching.
        swanlab.log({
            "rollout/ep_rew_mean": mean_reward,
            "rollout/ep_len_mean": mean_ep_len,
            "rollout/ep_rew_mean_rollout": mean_reward_rollout,
            "rollout/ep_len_mean_rollout": mean_ep_len_rollout,
            "train/policy_gradient_loss": metrics["policy_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/entropy_loss": -metrics["entropy"],
            "train/approx_kl": metrics["approx_kl"],
            "train/clip_fraction": metrics["clip_fraction"],
            "train/clip_range": 0.2,
            "train/explained_variance": explained_variance,
            "train/learning_rate": lr,
            "train/n_updates": (iteration + 1) * 10 * (steps_per_rollout // 64),
            "time/total_timesteps": total_timesteps,
            "time/iterations": iteration + 1,
        }, step=total_timesteps)

        metric_rows.append({
            "seed": args.seed,
            "iteration": iteration + 1,
            "total_timesteps": total_timesteps,
            "completed_episodes": len(ep_rewards),
            "mean_episode_reward": mean_reward,
            "mean_episode_length": mean_ep_len,
            "mean_episode_reward_rollout": mean_reward_rollout,
            "mean_episode_length_rollout": mean_ep_len_rollout,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "clip_fraction": metrics["clip_fraction"],
            "explained_variance": explained_variance,
            "learning_rate": lr,
        })

        print(
            f"  Iteration {iteration + 1:2d}/{total_iterations} | "
            f"Episodes: {len(ep_rewards):3d} | "
            f"Mean reward: {mean_reward:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"clip%: {metrics['clip_fraction']:.1%}"
        )

    print("-" * 60)

    fieldnames = list(metric_rows[0].keys())
    temporary_csv = f"{args.log_csv}.tmp"
    with open(temporary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    os.replace(temporary_csv, args.log_csv)
    print(f"Raw training metrics saved to {args.log_csv}")

    # Final evaluation
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset(seed=args.seed + 10_000 + len(eval_rewards))
        done, truncated, score = False, False, 0
        while not (done or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = model.get_action(obs_tensor, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            score += reward
        eval_rewards.append(score)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\nTraining complete! 20-episode evaluation: {mean_reward:.1f} +/- {std_reward:.1f}")

    swanlab.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
    })

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # GUI demo
    if args.gui:
        try:
            vis_env = gym.make("CartPole-v1", render_mode="human")
            print("\nDemoing what the agent learned (5 episodes)...")
            for ep in range(5):
                obs, _ = vis_env.reset(seed=args.seed + 20_000 + ep)
                done, truncated, score = False, False, 0
                while not (done or truncated):
                    obs_tensor = torch.FloatTensor(obs)
                    with torch.no_grad():
                        action, _, _ = model.get_action(obs_tensor, deterministic=True)
                    obs, reward, done, truncated, _ = vis_env.step(action.item())
                    score += reward
                print(f"  Demo episode {ep + 1} score: {score}")
            vis_env.close()
            print("\nGUI demo finished.")
        except Exception:
            print("(Skipping GUI demo, no display available)")
    else:
        print("\nTip: add --gui to pop up the cart animation window and watch the demo.")

    env.close()
    swanlab.finish()

    print("SwanLab experiment dashboard: swanlab watch swanlog")


if __name__ == "__main__":
    train()
