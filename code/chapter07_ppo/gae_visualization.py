"""
Chapter 6: GAE (Generalized Advantage Estimation) Visualization
——Building intuition for how λ and γ control the bias-variance tradeoff

GAE formula:
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)           # TD error
    A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}   # GAE advantage

Meaning of λ:
    λ → 0: high bias, low variance (only looks at single-step TD error)
    λ → 1: low bias, high variance (tends toward Monte Carlo return)

Meaning of γ:
    γ → 0: short-sighted (only cares about immediate reward)
    γ → 1: far-sighted (values long-term cumulative reward)

How to run:
    python gae_visualization.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: GAE computation function
# ==========================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)

    Args:
        rewards: list of rewards
        values:  list of value estimates V(s)
        dones:   list of episode-done flags
        gamma:   discount factor
        lam:     GAE lambda

    Returns:
        advantages: list of advantage estimates
        returns:    list of target returns
    """
    advantages = []
    gae = 0

    # Append a V(s_T+1)=0 at the end
    values = list(values) + [0.0]

    # Iterate backwards to compute GAE
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # Episode ended, reset
            gae = 0
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE accumulation: A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    # Target return = advantage + value
    returns = [a + v for a, v in zip(advantages, values[:-1])]

    return advantages, returns


def compute_mc_returns(rewards, gamma=0.99):
    """
    Compute Monte Carlo returns (accumulate discounted rewards from the end)
    Used as a reference for comparison
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_td_residuals(rewards, values, gamma=0.99):
    """
    Compute single-step TD error
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    values = list(values) + [0.0]
    residuals = []
    for t in range(len(rewards)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        residuals.append(delta)
    return residuals


# ==========================================
# Part 2: Create a synthetic reward sequence
# ==========================================
print("=" * 60)
print("Chapter 6: GAE (Generalized Advantage Estimation) Visualization")
print("=" * 60)

# Scenario: a 5-step sparse-reward sequence
# No reward for the first 4 steps, +1 reward on the last step
# This mimics the "delayed reward" problem seen in real RL
rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
n_steps = len(rewards)

# Assumed value function estimates (imperfect but roughly correct)
# V(s) gradually increases as it approaches the goal state
values = [0.1, 0.2, 0.4, 0.6, 0.9]

# Assume no early termination
dones = [False] * n_steps

print(f"\nSynthetic scenario setup:")
print(f"  Reward sequence: {rewards}")
print(f"  Value estimates: {values}")
print(f"  Feature: sparse reward — only the last step gets a reward")

# Compute Monte Carlo returns (reference baseline)
mc_returns = compute_mc_returns(rewards, gamma=0.99)
print(f"  MC returns:      {[f'{r:.4f}' for r in mc_returns]}")

# Compute single-step TD error
td_residuals = compute_td_residuals(rewards, values, gamma=0.99)
print(f"  TD errors:       {[f'{r:.4f}' for r in td_residuals]}")


# ==========================================
# Part 3: Comparing GAE across different λ values
# ==========================================
print("\n" + "=" * 60)
print("Comparison of GAE advantage estimates across different λ values")
print("=" * 60)

lambda_values = [0.0, 0.5, 0.9, 0.95, 1.0]
gamma_fixed = 0.99

# Store advantage values for different λ
advantages_by_lambda = {}
returns_by_lambda = {}

for lam in lambda_values:
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma_fixed, lam=lam)
    advantages_by_lambda[lam] = adv
    returns_by_lambda[lam] = ret

# Print comparison table
print(f"\n{'λ value':<8}", end="")
for t in range(n_steps):
    print(f"{'Step ' + str(t):>12}", end="")
print()
print("-" * (8 + 12 * n_steps))

for lam in lambda_values:
    label = f"{lam:<8.2f}"
    print(label, end="")
    for t in range(n_steps):
        print(f"{advantages_by_lambda[lam][t]:>12.4f}", end="")
    print()

print(f"\nExplanation:")
print(f"  λ=0.0: only looks at single-step TD error → high bias, low variance")
print(f"  λ=1.0: equivalent to Monte Carlo → low bias, high variance")
print(f"  λ=0.95: common PPO setting → a middle-ground compromise")


# ==========================================
# Part 4: Comparing GAE across different γ values
# ==========================================
print("\n" + "=" * 60)
print("Comparison of GAE advantage estimates across different γ values (λ=0.95 fixed)")
print("=" * 60)

gamma_values = [0.5, 0.9, 0.95, 0.99, 1.0]
lambda_fixed = 0.95

advantages_by_gamma = {}
returns_by_gamma = {}

for gamma in gamma_values:
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=lambda_fixed)
    advantages_by_gamma[gamma] = adv
    returns_by_gamma[gamma] = ret

# Print comparison table
print(f"\n{'γ value':<8}", end="")
for t in range(n_steps):
    print(f"{'Step ' + str(t):>12}", end="")
print()
print("-" * (8 + 12 * n_steps))

for gamma in gamma_values:
    label = f"{gamma:<8.2f}"
    print(label, end="")
    for t in range(n_steps):
        print(f"{advantages_by_gamma[gamma][t]:>12.4f}", end="")
    print()

print(f"\nExplanation:")
print(f"  γ=0.5:  short-sighted — only cares about near-term rewards")
print(f"  γ=0.99: common PPO setting — values long-term return")
print(f"  γ=1.0:  fully far-sighted — no discounting of future rewards")


# ==========================================
# Part 5: Plot the visualization
# ==========================================
print("\nGenerating visualization plots...")

# Create figure: 2 rows x 2 columns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("GAE Generalized Advantage Estimation — Bias-Variance Tradeoff", fontsize=18, fontweight="bold")

# Color scheme
colors_lambda = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
colors_gamma = ["#E91E63", "#FF5722", "#009688", "#3F51B5", "#000000"]

steps = np.arange(n_steps)
step_labels = [f"Step {i}\n(r={rewards[i]})" for i in range(n_steps)]

# ---- Subplot 1: advantage curves for different λ ----
ax1 = axes[0, 0]
for i, lam in enumerate(lambda_values):
    adv = advantages_by_lambda[lam]
    ax1.plot(steps, adv, marker="o", linewidth=2.5, markersize=8,
             color=colors_lambda[i], label=f"λ = {lam}")

ax1.set_xticks(steps)
ax1.set_xticklabels(step_labels)
ax1.set_title("Advantage estimates for different λ values", fontsize=14, fontweight="bold")
ax1.set_ylabel("Advantage A(s)", fontsize=12)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

# Add annotations explaining the meaning of λ
ax1.annotate("λ→0: high bias, low variance\n(single-step TD)", xy=(0.5, 0.02),
             xycoords="axes fraction", fontsize=10, color="#F44336",
             style="italic", ha="left")
ax1.annotate("λ→1: low bias, high variance\n(Monte Carlo)", xy=(0.5, 0.15),
             xycoords="axes fraction", fontsize=10, color="#9C27B0",
             style="italic", ha="left")

# ---- Subplot 2: advantage curves for different γ ----
ax2 = axes[0, 1]
for i, gamma in enumerate(gamma_values):
    adv = advantages_by_gamma[gamma]
    ax2.plot(steps, adv, marker="s", linewidth=2.5, markersize=8,
             color=colors_gamma[i], label=f"γ = {gamma}")

ax2.set_xticks(steps)
ax2.set_xticklabels(step_labels)
ax2.set_title("Advantage estimates for different γ values (λ=0.95)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Advantage A(s)", fontsize=12)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

# Add annotations explaining the meaning of γ
ax2.annotate("γ→0: short-sighted\n(only immediate reward)", xy=(0.02, 0.02),
             xycoords="axes fraction", fontsize=10, color="#E91E63",
             style="italic", ha="left")
ax2.annotate("γ→1: far-sighted\n(values long-term return)", xy=(0.02, 0.15),
             xycoords="axes fraction", fontsize=10, color="#000000",
             style="italic", ha="left")

# ---- Subplot 3: target returns for different λ ----
ax3 = axes[1, 0]
for i, lam in enumerate(lambda_values):
    ret = returns_by_lambda[lam]
    ax3.plot(steps, ret, marker="o", linewidth=2.5, markersize=8,
             color=colors_lambda[i], label=f"λ = {lam}")

# Also plot MC returns as a reference
ax3.plot(steps, mc_returns, marker="*", linewidth=2, markersize=12,
         color="black", linestyle="--", label="MC return (reference)")

ax3.set_xticks(steps)
ax3.set_xticklabels(step_labels)
ax3.set_title("Target returns for different λ values", fontsize=14, fontweight="bold")
ax3.set_xlabel("Time step", fontsize=12)
ax3.set_ylabel("Target return G(s)", fontsize=12)
ax3.legend(fontsize=10, loc="upper left")
ax3.grid(True, alpha=0.3)

# ---- Subplot 4: bias-variance tradeoff illustration ----
ax4 = axes[1, 1]

# Create theoretical bias and variance curves
lams = np.linspace(0, 1, 100)
# Bias decreases as λ increases (illustrative)
bias = np.exp(-3 * lams) * 1.0
# Variance increases as λ increases (illustrative)
variance = (np.exp(2 * lams) - 1) / (np.exp(2) - 1) * 1.0
# Total error = bias^2 + variance
total_error = bias ** 2 + variance

ax4.fill_between(lams, 0, bias ** 2, alpha=0.3, color="#2196F3", label="Bias²")
ax4.fill_between(lams, bias ** 2, bias ** 2 + variance, alpha=0.3, color="#F44336", label="Variance")
ax4.plot(lams, total_error, color="black", linewidth=2.5, label="Total error")

# Mark the location of the optimal λ
optimal_idx = np.argmin(total_error)
optimal_lam = lams[optimal_idx]
ax4.axvline(x=optimal_lam, color="green", linestyle="--", linewidth=2, alpha=0.8)
ax4.annotate(f"Optimal λ ≈ {optimal_lam:.2f}", xy=(optimal_lam, total_error[optimal_idx]),
             xytext=(optimal_lam + 0.15, total_error[optimal_idx] + 0.3),
             fontsize=12, color="green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="green", lw=2))

# Mark the commonly used range
ax4.axvspan(0.9, 0.97, alpha=0.15, color="gold", label="Common PPO range (0.9~0.97)")

ax4.set_xlabel("λ value", fontsize=13)
ax4.set_ylabel("Error", fontsize=13)
ax4.set_title("Bias-variance tradeoff (illustrative)", fontsize=14, fontweight="bold")
ax4.legend(fontsize=11, loc="center right")
ax4.set_xlim(0, 1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/gae_visualization.png", dpi=150, bbox_inches="tight")
print("Plot saved to: output/gae_visualization.png")
plt.show()


# ==========================================
# Part 6: Print the full comparison table
# ==========================================
print("\n" + "=" * 60)
print("Full comparison table: advantage values for different (γ, λ) combinations")
print("=" * 60)

# Selected combinations
combos = [
    (0.99, 0.0,  "high bias / low variance extreme"),
    (0.99, 0.5,  "moderate balance"),
    (0.99, 0.95, "recommended PPO config"),
    (0.99, 1.0,  "low bias / high variance extreme"),
    (0.5,  0.95, "short-sighted + GAE"),
    (1.0,  0.95, "undiscounted + GAE"),
]

print(f"\n{'Config':<20} {'γ':>5} {'λ':>5}", end="")
for t in range(n_steps):
    print(f"  {'A(s'+str(t)+')':>8}", end="")
print()
print("-" * (20 + 5 + 5 + 10 * n_steps))

for gamma, lam, desc in combos:
    adv, _ = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    print(f"{desc:<20} {gamma:>5.2f} {lam:>5.2f}", end="")
    for t in range(n_steps):
        print(f"  {adv[t]:>8.4f}", end="")
    print()

print("\n" + "=" * 60)
print("Key takeaways:")
print("  1. λ controls the bias-variance tradeoff of the advantage estimate")
print("  2. γ controls how much weight is given to future rewards")
print("  3. Common PPO config: γ=0.99, λ=0.95")
print("  4. λ=0 → one-step TD, λ=1 → Monte Carlo return")
print("=" * 60)
