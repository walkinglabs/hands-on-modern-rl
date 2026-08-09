"""
Chapter 12: Reinforcement Learning Training for a Tool-Calling Agent
==========================================================

This script simulates a tool-calling Agent that uses the REINFORCE algorithm to learn
to pick the right tool for a given user query.

Scenario setup:
  - The user asks a variety of questions
  - The Agent must choose the appropriate one of 3 tools:
      1. search(query)      -- knowledge search
      2. calculate(expr)    -- math computation
      3. run_code(code)     -- code execution
  - Choosing the correct tool -> positive reward (+1)
  - Choosing the wrong tool -> negative reward (-0.1)

Training method:
  - Policy: a simple probability distribution (softmax parameterization)
  - Algorithm: REINFORCE (Monte Carlo policy gradient)
  - Train for 50 episodes, observing how the policy evolves

How to run:
    python tool_use_agent.py
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
# Part 1: Define the tools and the query dataset
# ==========================================

# The three available tools
TOOL_NAMES = ["search", "calculate", "run_code"]
TOOL_DESCRIPTIONS = {
    "search": "Search engine, suited to knowledge-based questions",
    "calculate": "Math calculator, suited to numerical operations",
    "run_code": "Code executor, suited to programming-related tasks",
}

# Training dataset: each entry contains a question text and the correct tool
# Design rationale: question types are evenly distributed (roughly 1/3 each) for diversity
TRAINING_QUERIES = [
    # ---- search type ----
    {"query": "中国的首都是哪里？",                     "correct_tool": "search"},
    {"query": "Python 是什么时候发明的？",               "correct_tool": "search"},
    {"query": "光速是多少？",                           "correct_tool": "search"},
    {"query": "法国的人口有多少？",                      "correct_tool": "search"},
    {"query": "什么是量子计算？",                        "correct_tool": "search"},
    {"query": "第二次世界大战什么时候结束？",             "correct_tool": "search"},
    {"query": "地球到月球的距离是多少？",                "correct_tool": "search"},
    {"query": "谁发明了电话？",                          "correct_tool": "search"},
    # ---- calculate type ----
    {"query": "请计算 123 + 456",                       "correct_tool": "calculate"},
    {"query": "25 乘以 37 等于多少？",                   "correct_tool": "calculate"},
    {"query": "1024 除以 8 是多少？",                   "correct_tool": "calculate"},
    {"query": "求 17 的平方根",                          "correct_tool": "calculate"},
    {"query": "计算圆的面积，半径为 5",                  "correct_tool": "calculate"},
    {"query": "3 的 10 次方是多少？",                    "correct_tool": "calculate"},
    {"query": "99 乘法表中 7×8 是多少？",               "correct_tool": "calculate"},
    {"query": "把 1024 转换成二进制",                    "correct_tool": "calculate"},
    # ---- run_code type ----
    {"query": "帮我写一个冒泡排序",                      "correct_tool": "run_code"},
    {"query": "写一个 Python 函数判断回文",              "correct_tool": "run_code"},
    {"query": "执行这段排序代码并输出结果",              "correct_tool": "run_code"},
    {"query": "帮我调试这段代码的语法错误",              "correct_tool": "run_code"},
    {"query": "写一个爬虫抓取网页标题",                  "correct_tool": "run_code"},
    {"query": "实现一个简单的 REST API",                 "correct_tool": "run_code"},
    {"query": "运行这段数据分析脚本",                    "correct_tool": "run_code"},
    {"query": "帮我写一个单元测试",                      "correct_tool": "run_code"},
]


def simulate_tool_result(tool_name, query, correct_tool):
    """
    Simulate the result of executing a tool

    Args:
        tool_name:     the tool actually called
        query:         the user's query text
        correct_tool:  the name of the correct tool
    Returns:
        result_dict:   a dict containing the result text and a correctness flag
    """
    if tool_name == correct_tool:
        # Picked the right tool
        return {"success": True, "message": f"used {tool_name} to successfully handle the query"}
    else:
        # Picked the wrong tool
        return {"success": False, "message": f"tool {tool_name} is not suited to this query"}


# ==========================================
# Part 2: Policy parameterization
# ==========================================

class ToolPolicy:
    """
    A simple policy model: maintains tool-selection probabilities for each query type

    Policy parameterization:
      - Uses logits (unnormalized scores) to represent the preference for each tool
      - Converts logits to a probability distribution via softmax
      - Training amounts to adjusting the logit values

    Intuition:
      - If a tool frequently receives a positive reward, its logit grows
      - If a tool frequently receives a negative reward, its logit shrinks
      - softmax guarantees the probabilities always sum to 1

    Args:
        n_tools: number of available tools
        learning_rate: learning rate
    """

    def __init__(self, n_tools=3, learning_rate=0.05):
        self.n_tools = n_tools
        self.learning_rate = learning_rate

        # Initialize logits: all zero -> uniform selection probability (1/3, 1/3, 1/3)
        self.query_type_logits = {
            "search":    np.zeros(n_tools),
            "calculate": np.zeros(n_tools),
            "run_code":  np.zeros(n_tools),
        }

    def get_query_type(self, query):
        """
        Determine the query type from its content (a simplified classifier)

        In a real system this would be an NLP classifier.
        Here we simulate it with keyword matching:
          - contains a math keyword -> calculate
          - contains a programming keyword -> run_code
          - otherwise -> search
        """
        calc_keywords = ["计算", "乘", "除", "平方", "面积", "次方", "等于多少",
                         "加", "减", "根", "乘法", "进制"]
        code_keywords = ["写", "执行", "代码", "函数", "调试", "爬虫", "API",
                         "排序", "运行", "测试", "编程", "脚本"]

        for kw in calc_keywords:
            if kw in query:
                return "calculate"
        for kw in code_keywords:
            if kw in query:
                return "run_code"
        return "search"

    def get_probabilities(self, query_type):
        """
        Get the tool-selection probabilities for a given query type

        Converts logits to probabilities via softmax:
          pi(a|s) = exp(logit_a) / sum(exp(logit_i))
        """
        logits = self.query_type_logits[query_type]
        # Numerically stable softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def sample_action(self, query_type):
        """
        Sample an action (tool choice) according to the current policy

        Not argmax! Sampling is key to exploration.
        Even if one tool has the highest probability, other tools can still be chosen.
        """
        probs = self.get_probabilities(query_type)
        action = np.random.choice(self.n_tools, p=probs)
        return action, probs[action]

    def update(self, query_type, action, reward):
        """
        REINFORCE policy gradient update

        Core formula:
          theta <- theta + alpha * grad(log pi(a|s)) * G

        For a softmax policy, the gradient is:
          grad(log pi(a_k|s)) = e_k - pi(s)
          where e_k is a one-hot vector and pi(s) is the probability vector

        Intuition:
          - If reward > 0: increase the probability of the chosen tool
          - If reward < 0: decrease the probability of the chosen tool
        """
        probs = self.get_probabilities(query_type)

        # Compute the gradient of the log probability
        # grad(log pi(a_k|s)) = e_k - pi(s)
        grad = -probs.copy()       # -pi(s)
        grad[action] += 1.0        # +e_k

        # Policy gradient update: theta <- theta + alpha * grad * reward
        self.query_type_logits[query_type] += self.learning_rate * grad * reward


# ==========================================
# Part 3: Training loop
# ==========================================

def train(policy, n_episodes=50, queries_per_episode=8):
    """
    Main REINFORCE training loop

    Each episode:
      1. Randomly sample a batch of queries
      2. The Agent picks a tool for each query
      3. Receives a reward (+1.0 for correct, -0.1 for incorrect)
      4. Immediately updates the policy parameters (online learning)

    Args:
        policy:             the policy object
        n_episodes:         number of training episodes
        queries_per_episode: number of queries sampled per episode
    Returns:
        history: the training history record
    """
    # Record the training process
    history = {
        "episode_rewards": [],      # average reward per episode
        "episode_accuracy": [],     # accuracy per episode
        "tool_probs_history": {     # tool-selection probabilities per episode
            "search": [],
            "calculate": [],
            "run_code": [],
        },
    }

    for episode in range(n_episodes):
        episode_reward = 0.0
        correct_count = 0

        # Randomly sample queries
        indices = np.random.choice(len(TRAINING_QUERIES),
                                   size=min(queries_per_episode, len(TRAINING_QUERIES)),
                                   replace=False)

        for idx in indices:
            query_data = TRAINING_QUERIES[idx]
            query = query_data["query"]
            correct_tool = query_data["correct_tool"]

            # Step 1: determine the query type
            query_type = policy.get_query_type(query)

            # Step 2: sample a tool according to the policy
            action, prob = policy.sample_action(query_type)
            chosen_tool = TOOL_NAMES[action]

            # Step 3: execute the tool and get the reward
            result = simulate_tool_result(chosen_tool, query, correct_tool)

            if result["success"]:
                reward = 1.0    # Correct choice: positive reward
                correct_count += 1
            else:
                reward = -0.1   # Wrong choice: negative reward (small penalty)

            # Step 4: update the policy
            policy.update(query_type, action, reward)
            episode_reward += reward

        # Record this episode's statistics
        avg_reward = episode_reward / queries_per_episode
        accuracy = correct_count / queries_per_episode
        history["episode_rewards"].append(avg_reward)
        history["episode_accuracy"].append(accuracy)

        # Record the current tool-selection probabilities for each query type
        for qt in ["search", "calculate", "run_code"]:
            probs = policy.get_probabilities(qt)
            history["tool_probs_history"][qt].append(probs.copy())

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"  episode {episode+1:3d}/{n_episodes} | "
                  f"avg reward: {avg_reward:+.3f} | "
                  f"accuracy: {accuracy:.1%}")

    return history


# ==========================================
# Part 4: Run training
# ==========================================
print("=" * 70)
print("  Chapter 12: Reinforcement Learning Training for a Tool-Calling Agent")
print("=" * 70)

np.random.seed(42)  # Fix the random seed

# Initialize the policy
policy = ToolPolicy(n_tools=3, learning_rate=0.05)

# ---- Pre-training test ----
print("\n[Before training] tool-selection probabilities (random init, uniform distribution):")
print(f"  {'Query type':<12} {'search':<12} {'calculate':<12} {'run_code':<12}")
print(f"  {'─' * 48}")
for qt in ["search", "calculate", "run_code"]:
    probs = policy.get_probabilities(qt)
    print(f"  {qt:<12} {probs[0]:<12.4f} {probs[1]:<12.4f} {probs[2]:<12.4f}")

# Quickly test the pre-training accuracy
correct_before = 0
total_before = len(TRAINING_QUERIES)
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    action, _ = policy.sample_action(qt)
    if TOOL_NAMES[action] == qd["correct_tool"]:
        correct_before += 1
accuracy_before = correct_before / total_before
print(f"\n  Pre-training accuracy: {correct_before}/{total_before} = {accuracy_before:.1%}")

# ---- Start training ----
print("\n" + "─" * 70)
print("  Starting REINFORCE training (50 episodes)")
print("─" * 70)

history = train(policy, n_episodes=50, queries_per_episode=8)

# ---- Post-training test ----
print("\n" + "─" * 70)
print("  Training complete!")
print("─" * 70)

print("\n[After training] tool-selection probabilities (should converge to the correct assignment):")
print(f"  {'Query type':<12} {'search':<12} {'calculate':<12} {'run_code':<12}  {'Optimal tool'}")
print(f"  {'─' * 64}")
for qt, optimal in [("search", "search"), ("calculate", "calculate"), ("run_code", "run_code")]:
    probs = policy.get_probabilities(qt)
    print(f"  {qt:<12} {probs[0]:<12.4f} {probs[1]:<12.4f} {probs[2]:<12.4f}  {optimal}")

# Test post-training accuracy (deterministic policy: pick the highest-probability tool)
correct_after = 0
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    probs = policy.get_probabilities(qt)
    best_action = np.argmax(probs)
    if TOOL_NAMES[best_action] == qd["correct_tool"]:
        correct_after += 1
accuracy_after = correct_after / total_before
print(f"\n  Post-training accuracy (deterministic policy): {correct_after}/{total_before} = {accuracy_after:.1%}")
print(f"  Accuracy improvement: {accuracy_after - accuracy_before:+.1%}")

# ---- Show predictions per query ----
print("\n[Per-query prediction breakdown]:")
print(f"  {'Question':<30} {'Correct tool':<10} {'Predicted tool':<10} {'Result'}")
print(f"  {'─' * 65}")
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    probs = policy.get_probabilities(qt)
    best_action = np.argmax(probs)
    predicted_tool = TOOL_NAMES[best_action]
    correct = predicted_tool == qd["correct_tool"]
    mark = "correct" if correct else "incorrect"
    query_short = qd["query"][:28] + ".." if len(qd["query"]) > 28 else qd["query"]
    print(f"  {query_short:<30} {qd['correct_tool']:<10} {predicted_tool:<10} {mark}")


# ==========================================
# Part 5: Visualization
# ==========================================
print("\nGenerating visualization charts...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Tool-Calling Agent -- REINFORCE Training Process", fontsize=18, fontweight="bold")

# ---- Subplot 1: evolution of tool-selection probability ----
ax1 = axes[0]

episodes = np.arange(1, len(history["episode_accuracy"]) + 1)

# Extract the tool-selection probability trajectory for each query type
for qt_idx, (qt, color, marker) in enumerate([
    ("search",    "#2196F3", "o"),
    ("calculate", "#FF9800", "s"),
    ("run_code",  "#4CAF50", "^"),
]):
    probs_history = np.array(history["tool_probs_history"][qt])
    # Plot the probability that the correct tool is chosen for this query type
    ax1.plot(episodes, probs_history[:, qt_idx],
             marker=marker, linewidth=2.5, markersize=6,
             color=color, label=f"{qt} queries -> chooses {TOOL_NAMES[qt_idx]}")

ax1.axhline(y=1/3, color="gray", linestyle="--", alpha=0.5, label="Random baseline (1/3)")
ax1.set_title("Evolution of Tool-Selection Probability", fontsize=14, fontweight="bold")
ax1.set_xlabel("Training episode", fontsize=12)
ax1.set_ylabel("Probability of choosing the correct tool", fontsize=12)
ax1.legend(fontsize=10, loc="center right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Add annotations
ax1.annotate("Before training: uniform distribution (~33%)",
             xy=(1, 1/3), xytext=(8, 0.15),
             fontsize=10, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax1.annotate("After training: converges to the correct assignment",
             xy=(50, 0.9), xytext=(30, 0.95),
             fontsize=10, color="green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

# ---- Subplot 2: accuracy curve ----
ax2 = axes[1]

ax2.plot(episodes, history["episode_accuracy"],
         linewidth=1.5, alpha=0.4, color="steelblue", label="Episode accuracy (raw)")

# Moving average
window = 5
if len(history["episode_accuracy"]) >= window:
    moving_avg = []
    for i in range(len(history["episode_accuracy"])):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(history["episode_accuracy"][start:i+1]))
    ax2.plot(episodes, moving_avg, color="crimson", linewidth=2.5,
             label=f"Moving average (window={window})")

ax2.axhline(y=1/3, color="gray", linestyle="--", alpha=0.5, label="Random baseline (33.3%)")
ax2.axhline(y=accuracy_after, color="green", linestyle=":", alpha=0.7,
            label=f"Final accuracy ({accuracy_after:.1%})")

ax2.set_title("Tool-Selection Accuracy Over Training", fontsize=14, fontweight="bold")
ax2.set_xlabel("Training episode", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.legend(fontsize=10, loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("output/tool_use_agent_training.png", dpi=150, bbox_inches="tight")
print("Chart saved to: output/tool_use_agent_training.png")
plt.show()


# ==========================================
# Part 6: Summary
# ==========================================
print("\n" + "=" * 70)
print("  Key Takeaways")
print("=" * 70)
print(f"""
  1. Tool calling is a core capability of Agentic RL
     - The Agent must learn to choose the right tool based on user intent
     - This is the key leap from "passively answering" to "actively acting"

  2. REINFORCE can effectively learn a tool-selection policy
     - Before training: uniform distribution, accuracy about {accuracy_before:.1%}
     - After training: converges to the correct assignment, accuracy rises to {accuracy_after:.1%}
     - A reasonable policy can be learned in just 50 episodes

  3. Directions for extending this to a real system:
     - A more sophisticated policy network (e.g. a Transformer)
     - Multi-step tool chains (chaining multiple tools to complete complex tasks)
     - A process reward model (PRM) to guide intermediate steps
     - Learning tool parameters (not just which tool, but how to construct its arguments)

  4. From a single tool to a multi-tool chain:
     - This experiment: only one tool is chosen per query
     - Next step: the Agent needs to plan a multi-step sequence of tool calls
     - This is the multi-turn credit assignment problem discussed in multi_turn_rl.py
""")
print("=" * 70)
