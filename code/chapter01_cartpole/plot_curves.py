"""
Extract training data from SwanLab backup files and generate comparison
curves for all metrics.
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


EXPECTED_METRICS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/value_loss",
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/learning_rate",
]


def parse_swanlab_backup(path):
    with open(path, "rb") as f:
        raw = f.read()

    records = []
    i = 0
    while i < len(raw):
        json_start = raw.find(b'{"model_type"', i)
        if json_start < 0:
            break
        depth = 0
        j = json_start
        while j < len(raw):
            if raw[j:j + 1] == b'{':
                depth += 1
            elif raw[j:j + 1] == b'}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        chunk = raw[json_start:j + 1]
        try:
            obj = json.loads(chunk.decode("utf-8", errors="replace"))
            records.append(obj)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        i = j + 1

    metrics = {}
    for rec in records:
        if rec.get("model_type") != "Scalar":
            continue
        d = rec.get("data", {})
        key = d.get("key", "")
        step = d.get("step", 0)
        value = d.get("metric", {}).get("data")
        if key and value is not None:
            metrics.setdefault(key, []).append((step, float(value)))

    for key in metrics:
        seen = set()
        unique = []
        for s, v in metrics[key]:
            if s not in seen:
                seen.add(s)
                unique.append((s, v))
        metrics[key] = sorted(unique, key=lambda x: x[0])

    return metrics


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def classify_run(run_dir, metrics):
    metadata_path = os.path.join(run_dir, "files", "swanlab-metadata.json")
    command = os.path.basename(str(read_json(metadata_path).get("command", "")))

    if command == "1-ppo_cartpole.py":
        return "SB3 PPO"
    if command == "2-pytorch_ppo.py":
        return "PyTorch PPO"

    config_path = os.path.join(run_dir, "files", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = f.read().lower()
    except OSError:
        config = ""

    if "stable_baselines3" in config:
        return "SB3 PPO"
    if "total_iterations" in config or "steps_per_rollout" in config:
        return "PyTorch PPO"

    # Fallback for older local logs: SB3 usually logs real timesteps,
    # while the custom PyTorch script logs iteration indices.
    reward_steps = metrics.get("rollout/ep_rew_mean", [])
    if reward_steps and reward_steps[0][0] >= 1000:
        return "SB3 PPO"
    return "PyTorch PPO"


def metric_coverage(metrics):
    return sum(1 for key in EXPECTED_METRICS if metrics.get(key))


def latest_recorded_step(metrics):
    last_steps = [values[-1][0] for values in metrics.values() if values]
    return max(last_steps) if last_steps else -1


def select_best_runs(run_records):
    selected = {}
    for record in run_records:
        metrics = record["metrics"]
        score = (
            metric_coverage(metrics),
            latest_recorded_step(metrics),
            record["run_dir"],
        )
        previous = selected.get(record["name"])
        if previous is None or score > previous["score"]:
            selected[record["name"]] = {**record, "score": score}

    order = {"SB3 PPO": 0, "PyTorch PPO": 1}
    return sorted(selected.values(), key=lambda r: order.get(r["name"], 99))


def smooth(data, window=3):
    """Moving-average smoothing"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def save_chart(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    swanlog_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swanlog")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    run_records = []
    for d in sorted(os.listdir(swanlog_dir)):
        if not d.startswith("run-"):
            continue
        run_dir = os.path.join(swanlog_dir, d)
        backup = os.path.join(run_dir, "backup.swanlab")
        if not os.path.exists(backup):
            continue
        metrics = parse_swanlab_backup(backup)
        name = classify_run(run_dir, metrics)
        run_records.append({"name": name, "metrics": metrics, "run_dir": d})

    selected_runs = select_best_runs(run_records)
    for run in selected_runs:
        print(f"Using {run['name']}: {run['run_dir']}")
    runs = [(run["name"], run["metrics"]) for run in selected_runs]

    # ---- Global style ----
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    C = {
        "SB3 PPO": {"color": "#3B82F6", "fill": "#DBEAFE"},
        "PyTorch PPO": {"color": "#F97316", "fill": "#FED7AA"},
    }

    def get_iters_vals(name, metrics, key):
        """
        Normalize the x-axis to Total Timesteps.
        SB3's step is already in timesteps (0, 2048, 4096, ...), used as-is.
        PyTorch PPO's step is an iteration index (0, 1, 2, ...), so it
        needs to be multiplied by 2048.
        """
        data = metrics.get(key, [])
        if not data:
            return [], []
        if "PyTorch" in name:
            timesteps = np.array([r[0] * 2048 for r in data])
        else:
            timesteps = np.array([r[0] for r in data])
        vals = np.array([r[1] for r in data])
        return timesteps, vals

    # ================================================
    # 1. Reward curve (large chart) - SB3 vs PyTorch comparison
    # ================================================
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "rollout/ep_rew_mean")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=6, color=c["color"],
                label=name, linewidth=2.2, zorder=3)
        ax.fill_between(iters, 0, vals, alpha=0.08, color=c["color"])

    ax.axhline(y=195, color="#94A3B8", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.3, 205, "Solve Line (195)", fontsize=10, color="#64748B", style="italic")
    ax.axhline(y=500, color="#94A3B8", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(0.3, 485, "Max (500)", fontsize=10, color="#94A3B8", style="italic")

    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontweight="bold")
    ax.set_title("CartPole-v1 Training: SB3 PPO vs PyTorch PPO", fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9, edgecolor="#E2E8F0")
    ax.set_ylim(-10, 560)
    all_iters = []
    for n, m in runs:
        it, _ = get_iters_vals(n, m, "rollout/ep_rew_mean")
        all_iters.extend(it)
    if all_iters:
        ax.set_xlim(-0.5, max(all_iters) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "rollout/ep_rew_mean")
        if len(iters) > 0:
            ax.annotate(
                f"{vals[-1]:.0f}",
                xy=(iters[-1], vals[-1]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=11, fontweight="bold", color=C[name]["color"],
                arrowprops=dict(arrowstyle="->", color=C[name]["color"], lw=1.5),
            )

    plt.tight_layout()
    save_chart(fig, output_dir, "training_curves.png")

    # ================================================
    # 2. Episode length (Episode Length Mean)
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "rollout/ep_len_mean")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Mean Episode Length", fontweight="bold")
    ax.set_title("Episode Length Mean", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "ep_len_mean.png")

    # ================================================
    # 3. Value Loss
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/value_loss")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Value Loss", fontweight="bold")
    ax.set_title("Value Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "value_loss.png")

    # ================================================
    # 4. Entropy Loss
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/entropy_loss")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Entropy Loss", fontweight="bold")
    ax.set_title("Entropy Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "entropy_loss.png")

    # ================================================
    # 5. Policy Gradient Loss
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/policy_gradient_loss")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Policy Gradient Loss", fontweight="bold")
    ax.set_title("Policy Gradient Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "policy_gradient_loss.png")

    # ================================================
    # 6. Approx KL Divergence
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/approx_kl")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.axhline(y=0.03, color="#EF4444", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.3, 0.032, "Danger Zone (0.03)", fontsize=9, color="#EF4444", style="italic")
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Approx KL", fontweight="bold")
    ax.set_title("Approx KL Divergence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "approx_kl.png")

    # ================================================
    # 7. Clip Fraction
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/clip_fraction")
        if len(iters) == 0:
            continue
        c = C[name]
        vals_pct = vals * 100
        ax.plot(iters, vals_pct, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.axhline(y=30, color="#EF4444", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.3, 32, "Danger Zone (30%)", fontsize=9, color="#EF4444", style="italic")
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Clip Fraction (%)", fontweight="bold")
    ax.set_title("Clip Fraction", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "clip_fraction.png")

    # ================================================
    # 8. Explained Variance
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/explained_variance")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.axhline(y=1.0, color="#22C55E", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.3, 0.92, "Perfect Prediction (1.0)", fontsize=9, color="#22C55E", style="italic")
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Explained Variance", fontweight="bold")
    ax.set_title("Explained Variance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "explained_variance.png")

    # ================================================
    # 9. Learning Rate
    # ================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, metrics in runs:
        iters, vals = get_iters_vals(name, metrics, "train/learning_rate")
        if len(iters) == 0:
            continue
        c = C[name]
        ax.plot(iters, vals, "-o", markersize=5, color=c["color"],
                label=name, linewidth=2)
    ax.set_xlabel("Total Timesteps", fontweight="bold")
    ax.set_ylabel("Learning Rate", fontweight="bold")
    ax.set_title("Learning Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_chart(fig, output_dir, "learning_rate.png")

    # ================================================
    # 10. Training metrics overview (4-in-1)
    # ================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_configs = [
        ("train/value_loss", "Value Loss", axes[0, 0]),
        ("train/entropy_loss", "Entropy Loss", axes[0, 1]),
        ("train/approx_kl", "Approx KL", axes[1, 0]),
        ("train/clip_fraction", "Clip Fraction", axes[1, 1]),
    ]

    for key, title, ax in plot_configs:
        for name, metrics in runs:
            iters, vals = get_iters_vals(name, metrics, key)
            if len(iters) == 0:
                continue
            if key == "train/clip_fraction":
                vals = vals * 100
            c = C[name]
            ax.plot(iters, vals, "-o", markersize=4, color=c["color"],
                    label=name, linewidth=1.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Total Timesteps")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key == "train/clip_fraction":
            ax.set_ylabel("Clip Fraction (%)")

    fig.suptitle("Training Metrics Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_chart(fig, output_dir, "training_metrics.png")

    # ================================================
    # 11. Full metrics overview (6-in-1)
    # ================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    overview_configs = [
        ("rollout/ep_rew_mean", "Episode Reward Mean", False),
        ("rollout/ep_len_mean", "Episode Length Mean", False),
        ("train/value_loss", "Value Loss", False),
        ("train/entropy_loss", "Entropy Loss", False),
        ("train/approx_kl", "Approx KL", False),
        ("train/clip_fraction", "Clip Fraction (%)", True),
    ]

    for idx, (key, title, is_pct) in enumerate(overview_configs):
        ax = axes[idx // 3, idx % 3]
        for name, metrics in runs:
            iters, vals = get_iters_vals(name, metrics, key)
            if len(iters) == 0:
                continue
            if is_pct:
                vals = vals * 100
            c = C[name]
            ax.plot(iters, vals, "-o", markersize=4, color=c["color"],
                    label=name, linewidth=1.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Total Timesteps")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("CartPole Training: All Metrics Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_chart(fig, output_dir, "all_metrics_overview.png")

    print("\nDone! All charts saved to:", output_dir)


if __name__ == "__main__":
    main()
