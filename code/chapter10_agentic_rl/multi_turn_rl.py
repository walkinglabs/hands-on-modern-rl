"""
Chapter 12: Multi-Turn Dialogue RL -- ORM vs. PRM Credit Assignment Comparison
==========================================================

This script simulates a multi-turn tool-calling Agent (3~5 turns per episode),
comparing two reward-assignment strategies:

  1. ORM (Outcome Reward Model):
     Only the final result receives a reward (1.0 or 0.0); intermediate steps get no signal

  2. PRM (Process Reward Model):
     Every step receives a partial reward (0.0~1.0), providing a timely learning signal

Core concepts:
  - Credit assignment problem: how do you attribute the final reward to each individual step?
  - Discounted return: G_t = r_t + gamma * G_{t+1}, where gamma controls how far credit propagates
  - Larger gamma -> credit propagates further back in time; smaller gamma -> only recent steps matter

How to run:
    python multi_turn_rl.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create the output directory
os.makedirs("output", exist_ok=True)

# Configure a CJK-capable font so chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Define the simulated tools
# ==========================================
# The Agent can call three tools to complete a user task:
#   - calculator: math calculator
#   - search:      knowledge search
#   - code_executor: code executor
# Each tool returns a simulated result and a correctness score

def tool_calculator(query):
    """
    Simulated calculator tool
    Correctly handles math expressions, occasionally simulates a computation error (for realism)
    """
    # Simple simulation: return a result for common math problems
    if "123 + 456" in query:
        return {"result": "579", "correct": True}
    elif "25 * 4" in query:
        return {"result": "100", "correct": True}
    elif "sqrt(144)" in query or "144" in query:
        return {"result": "12", "correct": True}
    else:
        # Unknown computation, simulate a 70% success rate
        correct = np.random.random() < 0.7
        return {"result": "simulated result", "correct": correct}


def tool_search(query):
    """
    Simulated search tool
    Returns a knowledge retrieval result
    """
    # Simulate search quality
    if "Python" in query or "python" in query:
        return {"result": "Python is a widely used high-level programming language...", "correct": True}
    elif "RL" in query or "强化学习" in query:
        return {"result": "Reinforcement learning is a branch of machine learning...", "correct": True}
    else:
        correct = np.random.random() < 0.7
        return {"result": "found related content...", "correct": correct}


def tool_code_executor(code):
    """
    Simulated code executor
    Executes code and returns the run result
    """
    # Simulate code execution success rate
    if "print" in code or "def " in code:
        return {"result": "code executed successfully", "correct": True}
    elif "import" in code:
        return {"result": "module imported successfully", "correct": True}
    else:
        correct = np.random.random() < 0.6
        return {"result": "simulated execution result", "correct": correct}


# Tool registry: tool name -> (call function, tool description)
TOOLS = {
    "calculator": (tool_calculator, "Math calculator, used for numerical operations"),
    "search": (tool_search, "Knowledge search engine, used for factual lookups"),
    "code_executor": (tool_code_executor, "Code executor, used to run code snippets"),
}


# ==========================================
# Part 2: Simulate multi-turn dialogue scenarios
# ==========================================
# Predefine several multi-turn tasks, each containing 3~5 rounds of tool calls

SCENARIOS = [
    {
        "task": "Compute 123 + 456, then search for Python-related knowledge, then write code to print the result",
        "turns": [
            {"tool": "calculator",  "query": "123 + 456",   "description": "Step 1: call the calculator to perform the addition"},
            {"tool": "search",      "query": "Python 编程",  "description": "Step 2: search for Python-related knowledge"},
            {"tool": "code_executor","query": "print(579)",  "description": "Step 3: execute code to print the result"},
        ],
    },
    {
        "task": "Search for reinforcement learning material, compute 25*4, then run a simple program",
        "turns": [
            {"tool": "search",       "query": "强化学习入门",       "description": "Step 1: search for reinforcement learning material"},
            {"tool": "calculator",   "query": "25 * 4",            "description": "Step 2: compute the multiplication"},
            {"tool": "code_executor","query": "def hello(): pass",  "description": "Step 3: run a simple program"},
        ],
    },
    {
        "task": "Compute a square root, search for algorithm material, execute code, then search for deep learning",
        "turns": [
            {"tool": "calculator",    "query": "sqrt(144)",         "description": "Step 1: compute the square root"},
            {"tool": "search",        "query": "排序算法比较",       "description": "Step 2: search for algorithm material"},
            {"tool": "code_executor", "query": "import numpy",       "description": "Step 3: import a module"},
            {"tool": "search",        "query": "深度学习框架",       "description": "Step 4: search for deep learning material"},
        ],
    },
    {
        "task": "Search for a math formula, run a computation script, verify the result",
        "turns": [
            {"tool": "search",        "query": "欧拉公式推导",       "description": "Step 1: search for the math formula"},
            {"tool": "code_executor", "query": "import math",        "description": "Step 2: run the computation script"},
            {"tool": "calculator",    "query": "圆周率计算",         "description": "Step 3: numerical computation"},
        ],
    },
    {
        "task": "Search for RL policy gradients, run training code, compute the reward, search for PPO",
        "turns": [
            {"tool": "search",        "query": "RL 策略梯度",        "description": "Step 1: search for policy gradients"},
            {"tool": "code_executor", "query": "def train(): pass",   "description": "Step 2: run the training code"},
            {"tool": "calculator",    "query": "计算累积奖励",        "description": "Step 3: compute the reward"},
            {"tool": "search",        "query": "PPO 算法详解",       "description": "Step 4: search for PPO"},
            {"tool": "code_executor", "query": "print('done')",       "description": "Step 5: run the wrap-up code"},
        ],
    },
]


# ==========================================
# Part 3: ORM and PRM reward computation
# ==========================================

def compute_orm_rewards(turns):
    """
    ORM (Outcome Reward Model): only the final result receives a reward

    If the task ultimately succeeds, all steps share a reward of 1.0;
    if the task ultimately fails, all steps receive 0.0.

    It's like an exam that only grades the final answer -- no credit for the process.
    """
    n = len(turns)
    rewards = [0.0] * n  # Initialize: every step's reward is 0

    # Simulate the final outcome: the task succeeds only if every step is correct
    all_correct = all(turn.get("correct", False) for turn in turns)
    final_reward = 1.0 if all_correct else 0.0

    # Only the last step receives a reward
    rewards[-1] = final_reward

    return rewards


def compute_prm_rewards(turns):
    """
    PRM (Process Reward Model): every step receives a partial reward

    Each step is evaluated independently for correctness, receiving a reward between 0.0 and 1.0.
    It's like an exam that grades each question separately -- credit is given for the process too.

    Reward strategy:
    - Correct tool call: base reward + bonus for a sensible tool choice
    - Incorrect tool call: a lower reward
    """
    rewards = []
    for turn in turns:
        correct = turn.get("correct", False)

        if correct:
            # Correct step: higher reward
            # Also factors in whether the tool choice was reasonable
            base_reward = 0.7 + np.random.uniform(0, 0.3)  # 0.7 ~ 1.0
        else:
            # Incorrect step: still gets a small reward (to encourage exploration)
            base_reward = np.random.uniform(0.0, 0.3)  # 0.0 ~ 0.3

        rewards.append(round(base_reward, 3))

    return rewards


def compute_discounted_returns(rewards, gamma=0.99):
    """
    Compute the discounted return: G_t = r_t + gamma * G_{t+1}

    Recurses backward to compute the discounted cumulative return at each step.
    gamma (the discount factor) controls how far credit propagates:
      - gamma close to 1.0: credit propagates further back (far-sighted)
      - gamma close to 0.0: credit only affects the current step (short-sighted)

    Args:
        rewards: list of immediate rewards for each step
        gamma:   discount factor
    Returns:
        returns: list of discounted cumulative returns for each step
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ==========================================
# Part 4: Run the simulation experiment
# ==========================================
print("=" * 70)
print("  Chapter 12: Multi-Turn Dialogue RL -- ORM vs PRM Credit Assignment Comparison")
print("=" * 70)

np.random.seed(42)  # Fix the random seed for reproducibility

# Store the results for every scenario
all_orm_rewards = []   # Per-step ORM rewards
all_prm_rewards = []   # Per-step PRM rewards
all_orm_returns = {}   # ORM discounted returns (per gamma)
all_prm_returns = {}   # PRM discounted returns (per gamma)
gamma_values = [0.5, 0.9, 0.99]  # Test different discount factors

for gamma in gamma_values:
    all_orm_returns[gamma] = []
    all_prm_returns[gamma] = []

# Iterate over each scenario and run the simulation
for idx, scenario in enumerate(SCENARIOS):
    print(f"\n{'─' * 70}")
    print(f"  Scenario {idx + 1}: {scenario['task']}")
    print(f"  {len(scenario['turns'])} tool calls total")
    print(f"{'─' * 70}")

    # Simulate the tool-call result for each step
    turns = scenario["turns"]
    for t, turn in enumerate(turns):
        tool_name = turn["tool"]
        query = turn["query"]

        # Call the corresponding tool
        tool_func, _ = TOOLS[tool_name]
        result = tool_func(query)
        turn["correct"] = result["correct"]

        status = "correct" if result["correct"] else "incorrect"
        print(f"  Turn {t+1}: {turn['description']}")
        print(f"         calling {tool_name}({query}) -> {status}")

    # ---- ORM reward computation ----
    orm_rewards = compute_orm_rewards(turns)
    all_orm_rewards.append(orm_rewards)

    print(f"\n  [ORM rewards] only the final result carries a signal:")
    for t, r in enumerate(orm_rewards):
        bar = "█" * int(r * 20)
        print(f"    turn {t+1} reward: {r:.1f}  {bar}")

    # ---- PRM reward computation ----
    prm_rewards = compute_prm_rewards(turns)
    all_prm_rewards.append(prm_rewards)

    print(f"\n  [PRM rewards] every step carries a learning signal:")
    for t, r in enumerate(prm_rewards):
        bar = "█" * int(r * 20)
        print(f"    turn {t+1} reward: {r:.3f}  {bar}")

    # ---- Discounted return comparison (across multiple gamma values) ----
    for gamma in gamma_values:
        orm_returns = compute_discounted_returns(orm_rewards, gamma=gamma)
        prm_returns = compute_discounted_returns(prm_rewards, gamma=gamma)
        all_orm_returns[gamma].append(orm_returns)
        all_prm_returns[gamma].append(prm_returns)

    # Walk through the discounted-return computation in detail for gamma=0.99
    gamma_demo = 0.99
    orm_ret_demo = compute_discounted_returns(orm_rewards, gamma=gamma_demo)
    prm_ret_demo = compute_discounted_returns(prm_rewards, gamma=gamma_demo)

    print(f"\n  Discounted return computation (gamma = {gamma_demo}):")
    print(f"    {'Turn':<6} {'Immediate reward':<12} {'Discounted return G_t':<16} {'Computation'}")
    print(f"    {'─' * 60}")

    # Step through the ORM discounted returns
    print(f"    [ORM mode]")
    G = 0.0
    for t in reversed(range(len(orm_rewards))):
        old_G = G
        G = orm_rewards[t] + gamma_demo * old_G
        formula = f"G_{t} = {orm_rewards[t]:.1f} + {gamma_demo} * {old_G:.4f} = {G:.4f}"
        print(f"    turn {t+1}  r={orm_rewards[t]:<8.1f}  G={G:<12.4f}  {formula}")

    # Step through the PRM discounted returns
    print(f"    [PRM mode]")
    G = 0.0
    for t in reversed(range(len(prm_rewards))):
        old_G = G
        G = prm_rewards[t] + gamma_demo * old_G
        formula = f"G_{t} = {prm_rewards[t]:.3f} + {gamma_demo} * {old_G:.4f} = {G:.4f}"
        print(f"    turn {t+1}  r={prm_rewards[t]:<8.3f}  G={G:<12.4f}  {formula}")


# ==========================================
# Part 5: Comprehensive ORM vs. PRM analysis
# ==========================================
print("\n" + "=" * 70)
print("  Comprehensive ORM vs PRM Analysis")
print("=" * 70)

print("\n  [Reward signal density comparison]")
for idx in range(len(SCENARIOS)):
    n_turns = len(SCENARIOS[idx]["turns"])
    orm_nonzero = sum(1 for r in all_orm_rewards[idx] if r > 0)
    prm_nonzero = sum(1 for r in all_prm_rewards[idx] if r > 0)
    print(f"    Scenario {idx+1} ({n_turns} turns):"
          f" ORM steps with signal = {orm_nonzero}/{n_turns},"
          f" PRM steps with signal = {prm_nonzero}/{n_turns}")

print(f"\n  Key takeaways:")
print(f"    - ORM signal is sparse: only the last step has a reward, intermediate steps lack any signal")
print(f"    - PRM signal is dense: every step has feedback, so learning is more efficient")
print(f"    - For multi-turn Agents, PRM can significantly speed up policy learning")

print("\n  [Effect of the discount factor gamma on credit propagation]")
for gamma in gamma_values:
    print(f"\n    gamma = {gamma}:")
    for idx in range(len(SCENARIOS)):
        orm_ret = all_orm_returns[gamma][idx]
        prm_ret = all_prm_returns[gamma][idx]
        n = len(orm_ret)
        print(f"      Scenario {idx+1} ({n} turns):")
        print(f"        ORM discounted returns: {[f'{v:.4f}' for v in orm_ret]}")
        print(f"        PRM discounted returns: {[f'{v:.4f}' for v in prm_ret]}")

print(f"\n  Summary of gamma's effect:")
print(f"    - gamma=0.5: credit decays quickly, only the last few steps feel the final reward")
print(f"    - gamma=0.9: credit propagates moderately, balancing near-term and long-term signals")
print(f"    - gamma=0.99: credit propagates far, even early steps receive a meaningful return signal")


# ==========================================
# Part 6: Visualization charts
# ==========================================
print("\nGenerating visualization charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Multi-Turn Dialogue RL -- ORM vs PRM Credit Assignment Comparison", fontsize=18, fontweight="bold")

# ---- Subplot 1: turn-level reward heatmap ----
ax1 = axes[0, 0]

# Build the heatmap data matrix
max_turns = max(len(r) for r in all_orm_rewards)
n_scenarios = len(SCENARIOS)

heatmap_orm = np.zeros((n_scenarios, max_turns))
heatmap_prm = np.zeros((n_scenarios, max_turns))

for i in range(n_scenarios):
    for j in range(len(all_orm_rewards[i])):
        heatmap_orm[i, j] = all_orm_rewards[i][j]
        heatmap_prm[i, j] = all_prm_rewards[i][j]

# Plot the PRM heatmap (more instructive)
im = ax1.imshow(heatmap_prm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(max_turns))
ax1.set_xticklabels([f"Turn {i+1}" for i in range(max_turns)])
ax1.set_yticks(range(n_scenarios))
ax1.set_yticklabels([f"Scenario {i+1}" for i in range(n_scenarios)])
ax1.set_title("PRM Per-Step Reward Heatmap", fontsize=14, fontweight="bold")
ax1.set_xlabel("Dialogue turn", fontsize=12)
ax1.set_ylabel("Scenario", fontsize=12)

# Annotate the heatmap cells with values
for i in range(n_scenarios):
    for j in range(len(all_prm_rewards[i])):
        ax1.text(j, i, f"{heatmap_prm[i, j]:.2f}",
                 ha="center", va="center", fontsize=10, fontweight="bold")

fig.colorbar(im, ax=ax1, label="Reward value")

# ---- Subplot 2: ORM vs PRM discounted return comparison (gamma=0.99) ----
ax2 = axes[0, 1]

gamma_plot = 0.99
colors_orm = plt.cm.Blues(np.linspace(0.4, 0.9, n_scenarios))
colors_prm = plt.cm.Reds(np.linspace(0.4, 0.9, n_scenarios))

for i in range(n_scenarios):
    n = len(all_orm_returns[gamma_plot][i])
    x = np.arange(n)

    # ORM uses a dashed line, PRM uses a solid line
    ax2.plot(x, all_orm_returns[gamma_plot][i],
             marker="o", linestyle="--", linewidth=2, markersize=6,
             color=colors_orm[i],
             label=f"Scenario {i+1} ORM" if i == 0 else None)
    ax2.plot(x, all_prm_returns[gamma_plot][i],
             marker="s", linestyle="-", linewidth=2, markersize=6,
             color=colors_prm[i],
             label=f"Scenario {i+1} PRM" if i == 0 else None)

# Just draw two representative lines (to avoid a cluttered legend)
ax2.plot([], [], marker="o", linestyle="--", color="steelblue", linewidth=2, label="ORM discounted return")
ax2.plot([], [], marker="s", linestyle="-", color="crimson", linewidth=2, label="PRM discounted return")

ax2.set_title(f"ORM vs PRM Discounted Return (gamma={gamma_plot})", fontsize=14, fontweight="bold")
ax2.set_xlabel("Dialogue turn", fontsize=12)
ax2.set_ylabel("Discounted return G_t", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.annotate("ORM: only the last step has a signal\n-> intermediate step gradient is ~0",
             xy=(0.02, 0.95), xycoords="axes fraction",
             fontsize=10, color="steelblue", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
ax2.annotate("PRM: every step has a signal\n-> dense gradient signal",
             xy=(0.02, 0.75), xycoords="axes fraction",
             fontsize=10, color="crimson", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3))

# ---- Subplot 3: effect of different gamma values on credit propagation (scenario 5, PRM) ----
ax3 = axes[1, 0]

# Choose the longest scenario (scenario 5 has 5 turns)
demo_idx = 4  # scenario 5 (index 4)
colors_gamma = ["#E91E63", "#FF9800", "#4CAF50"]

for gi, gamma in enumerate(gamma_values):
    ret = all_prm_returns[gamma][demo_idx]
    x = np.arange(len(ret))
    ax3.plot(x, ret, marker="o", linewidth=2.5, markersize=8,
             color=colors_gamma[gi], label=f"gamma = {gamma}")

ax3.set_title(f"Effect of Discount Factor gamma on Credit Propagation (Scenario 5, PRM)", fontsize=14, fontweight="bold")
ax3.set_xlabel("Dialogue turn", fontsize=12)
ax3.set_ylabel("Discounted return G_t", fontsize=12)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(len(all_prm_returns[0.99][demo_idx])))
ax3.set_xticklabels([f"Turn {i+1}" for i in range(len(all_prm_returns[0.99][demo_idx]))])

# ---- Subplot 4: ORM vs PRM reward signal density bar chart ----
ax4 = axes[1, 1]

x_pos = np.arange(n_scenarios)
bar_width = 0.35

# Compute the fraction of nonzero-reward steps for each scenario
orm_density = []
prm_density = []
for i in range(n_scenarios):
    n = len(all_orm_rewards[i])
    orm_density.append(sum(1 for r in all_orm_rewards[i] if r > 0) / n * 100)
    prm_density.append(sum(1 for r in all_prm_rewards[i] if r > 0) / n * 100)

bars1 = ax4.bar(x_pos - bar_width/2, orm_density, bar_width,
                label='ORM (outcome reward)', color='steelblue', alpha=0.8)
bars2 = ax4.bar(x_pos + bar_width/2, prm_density, bar_width,
                label='PRM (process reward)', color='crimson', alpha=0.8)

ax4.set_title("Reward Signal Density Comparison (Share of Nonzero-Reward Steps)", fontsize=14, fontweight="bold")
ax4.set_xlabel("Scenario", fontsize=12)
ax4.set_ylabel("Share of steps with a signal (%)", fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"Scenario {i+1}" for i in range(n_scenarios)])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 110)

# Annotate the bars with their percentages
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("output/multi_turn_orm_vs_prm.png", dpi=150, bbox_inches="tight")
print("Chart saved to: output/multi_turn_orm_vs_prm.png")
plt.show()


# ==========================================
# Part 7: Summary
# ==========================================
print("\n" + "=" * 70)
print("  Key Takeaways")
print("=" * 70)
print("""
  1. Credit assignment is the core challenge of multi-turn Agent RL
     - An Agent needs multiple rounds of interaction to complete a task
     - How do you attribute the final success/failure to each individual step?

  2. ORM's pros and cons:
     + Simple to implement, only requires labeling the final outcome
     - Sparse signal, intermediate steps "fly blind"
     - Credit propagation relies on the discount factor, which may decay too fast

  3. PRM's pros and cons:
     + Dense signal, every step has a learning signal
     + Can distinguish "good intermediate steps" from "bad intermediate steps"
     - High labeling cost, requires scoring every step
     - The reward model may introduce noise

  4. The role of the discount factor gamma:
     - The larger gamma is, the further credit propagates (long-term rewards are also considered)
     - The smaller gamma is, the faster credit decays (only recent steps matter)
     - For multi-turn dialogue, gamma >= 0.9 is recommended

  5. Practical recommendations:
     - Simple tasks: ORM is sufficient (e.g. single-turn QA)
     - Complex multi-step reasoning: PRM works better (e.g. math proofs, code generation)
     - Hybrid approach: PRM process reward + ORM outcome verification
""")
print("=" * 70)
