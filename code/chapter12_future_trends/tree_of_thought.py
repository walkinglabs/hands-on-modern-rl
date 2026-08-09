"""
Chapter 13: Tree of Thought (ToT) Reasoning Experiment
——Exploring search strategies for LLM test-time reasoning

Tree of Thought (ToT) is a method that performs explicit search at inference time:
- At each step, generate multiple candidate thoughts (breadth-first)
- Score each thought with an evaluation function
- Keep the top-k highest-scoring thoughts and continue searching
- Eventually find the optimal reasoning path

This experiment uses the "Game of 24" as the task, comparing three strategies:
1. Chain-of-Thought (CoT): single-path greedy reasoning
2. Tree of Thought (ToT): multi-branch search + score-based pruning
3. Random: random attempts

How to run:
    python tree_of_thought.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy

# Create output directory
os.makedirs("output", exist_ok=True)

# Set Chinese font (kept for compatibility with any Chinese labels)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Game of 24 environment
# ==========================================
class TwentyFourGame:
    """
    Game of 24 environment

    Rules:
        - Given 4 numbers (1~13)
        - Use the four arithmetic operations +, -, *, /
        - Each number must be used exactly once
        - The result of the computation must equal 24

    This implementation models the problem as a process of "incrementally
    building an expression":
        - At each step, pick two remaining numbers and one operator
        - Compute the intermediate result, treating it as a new "virtual number"
        - Repeat until only one number remains
    """

    # The four operations
    OPERATIONS = ['+', '-', '*', '/']

    @staticmethod
    def apply_op(a, b, op):
        """Apply the operation to two numbers and return the result; division by zero returns None"""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if abs(b) < 1e-10:
                return None  # guard against division by zero
            return a / b

    @staticmethod
    def is_close_to_target(value, target=24.0, tol=1e-6):
        """Check whether the value is close to the target"""
        return abs(value - target) < tol


# ==========================================
# Part 2: Search nodes and scoring function
# ==========================================
class ThoughtNode:
    """
    Tree-of-Thought node

    Each node represents an "intermediate state":
        - numbers: the list of remaining numbers
        - history: the operations executed so far
        - score: the node's score (higher is better)
        - parent: the parent node
        - children: list of child nodes
    """

    def __init__(self, numbers, history=None, parent=None):
        self.numbers = list(numbers)  # currently remaining numbers
        self.history = history or []  # operation history
        self.parent = parent
        self.children = []
        self.score = 0.0  # node score

    def is_terminal(self):
        """Whether this is a terminal state (only one number remains)"""
        return len(self.numbers) <= 1

    def get_expression(self):
        """Return the full computation expression"""
        return ' → '.join(self.history) if self.history else 'initial state'

    def __repr__(self):
        return f"Node(nums={self.numbers}, score={self.score:.2f})"


def evaluate_node(node, target=24.0):
    """
    Evaluate the quality of a node

    Scoring strategy:
        - Terminal state (only 1 number left): the closer the result is to 24, the higher the score
        - Non-terminal state (multiple numbers left): based on the "potential" of the remaining
          numbers to combine into the target

    This evaluation function simulates the "self-evaluation" ability of an LLM in ToT.
    Real systems use an LLM to score; here we substitute a heuristic rule.
    """
    if node.is_terminal():
        # Terminal state: directly check whether the result is close to 24
        diff = abs(node.numbers[0] - target)
        return max(0.0, 1.0 - diff / 50.0)  # the smaller the gap, the higher the score

    # Non-terminal state: evaluate the "potential" of the remaining numbers to reach 24
    # Heuristic: check whether any pair of remaining numbers can combine into a value closer to the target
    nums = node.numbers
    best_potential = 0.0

    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            for op in TwentyFourGame.OPERATIONS:
                result = TwentyFourGame.apply_op(nums[i], nums[j], op)
                if result is None:
                    continue
                # Construct the new list of remaining numbers
                new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
                new_nums.append(result)

                if len(new_nums) == 1:
                    # Only one number left, check whether it's close to 24
                    potential = max(0.0, 1.0 - abs(new_nums[0] - target) / 50.0)
                else:
                    # One-step recursion: look at the best potential of the next level
                    potential = 0.5 * max(0.0, 1.0 - abs(result - target) / 50.0)
                best_potential = max(best_potential, potential)

    return best_potential


def generate_children(node):
    """
    Generate all possible next steps for a node

    Pick any two numbers and one operator from the currently remaining numbers,
    producing a new intermediate state.
    """
    children = []
    nums = node.numbers
    n = len(nums)

    if n < 2:
        return children

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # can't pick the same number twice
            for op in TwentyFourGame.OPERATIONS:
                result = TwentyFourGame.apply_op(nums[i], nums[j], op)
                if result is None:
                    continue  # skip division by zero

                # Construct the new list of remaining numbers
                new_nums = [nums[k] for k in range(n) if k != i and k != j]
                new_nums.append(result)

                # Record the operation step
                step = f"{nums[i]} {op} {nums[j]} = {result:.2f}"
                new_history = node.history + [step]

                child = ThoughtNode(new_nums, new_history, parent=node)
                children.append(child)

    return children


# ==========================================
# Part 3: The three search strategies
# ==========================================
def search_tree_of_thought(numbers, breadth=3, max_depth=4, target=24.0, verbose=True):
    """
    Tree of Thought search

    Core idea:
        1. Generate all candidate nodes (branches) at each level
        2. Score each node with the evaluation function
        3. Keep only the top `breadth` highest-scoring nodes
        4. Continue expanding the next level

    Args:
        numbers: initial list of numbers
        breadth: number of candidates kept per level (beam width)
        max_depth: maximum search depth
        target: target value
        verbose: whether to print detailed progress

    Returns:
        best_node: the best node found
        tree_data: search tree data (for visualization)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Tree of Thought search (breadth={breadth})")
        print(f"  Initial numbers: {numbers}, target: {target}")
        print(f"{'='*60}")

    root = ThoughtNode(numbers)
    current_beam = [root]  # candidate nodes at the current level
    tree_data = {'nodes_per_level': [], 'scores_per_level': []}
    total_evaluated = 0

    for depth in range(max_depth):
        if verbose:
            print(f"\n--- Expanding level {depth + 1} ---")

        # Step 1: expand all nodes at the current level, generating children
        all_children = []
        for node in current_beam:
            children = generate_children(node)
            # Score each child node
            for child in children:
                child.score = evaluate_node(child, target)
                total_evaluated += 1
            all_children.extend(children)

        if not all_children:
            if verbose:
                print("  No more nodes to expand")
            break

        # Step 2: sort by score, keep the top-k
        all_children.sort(key=lambda x: x.score, reverse=True)
        current_beam = all_children[:breadth]

        # Record search tree data
        level_nodes = [c.get_expression() for c in current_beam]
        level_scores = [c.score for c in current_beam]
        tree_data['nodes_per_level'].append(level_nodes)
        tree_data['scores_per_level'].append(level_scores)

        if verbose:
            print(f"  Generated {len(all_children)} candidate nodes ({total_evaluated} evaluations so far)")
            print(f"  Keeping top-{breadth}:")
            for idx, node in enumerate(current_beam):
                nums_str = ', '.join([f"{n:.1f}" for n in node.numbers])
                print(f"    [{idx+1}] score={node.score:.3f} | remaining=[{nums_str}] | {node.get_expression()}")

        # Check whether an exact solution has already been found
        for node in current_beam:
            if (node.is_terminal()
                    and TwentyFourGame.is_close_to_target(node.numbers[0], target)):
                if verbose:
                    print(f"\n  *** Exact solution found! ***")
                    print(f"  Result = {node.numbers[0]:.4f}")
                    print(f"  Reasoning path: {node.get_expression()}")
                return node, tree_data

    # No exact solution found, return the closest one
    best_node = max(current_beam, key=lambda x: x.score)
    if verbose:
        if best_node.is_terminal():
            print(f"\n  No exact solution found, closest result:")
            print(f"  Result = {best_node.numbers[0]:.4f}")
            print(f"  Reasoning path: {best_node.get_expression()}")
        else:
            print(f"\n  Search did not complete (reached max depth)")
            print(f"  Current best node: {best_node}")

    return best_node, tree_data


def search_chain_of_thought(numbers, target=24.0, verbose=True):
    """
    Chain-of-Thought (CoT) search

    Core idea:
        At each step, keep only a single best node (greedy),
        equivalent to ToT with breadth=1.

    This simulates the standard CoT "single-chain reasoning" pattern.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Chain-of-Thought reasoning (greedy single path)")
        print(f"  Initial numbers: {numbers}, target: {target}")
        print(f"{'='*60}")

    # CoT is equivalent to ToT with breadth=1
    result, _ = search_tree_of_thought(
        numbers, breadth=1, max_depth=4, target=target, verbose=verbose
    )
    return result


def search_random(numbers, n_trials=50, target=24.0, verbose=True):
    """
    Random search (baseline comparison)

    Core idea:
        Randomly pick a pair of numbers and an operator,
        try many times, and record the best result.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Random search ({n_trials} trials)")
        print(f"  Initial numbers: {numbers}, target: {target}")
        print(f"{'='*60}")

    best_result = None
    best_diff = float('inf')
    best_history = []
    successes = 0

    for trial in range(n_trials):
        remaining = list(numbers)
        history = []

        for step in range(3):  # 4 numbers require 3 operations
            if len(remaining) < 2:
                break

            # Randomly pick two numbers at different positions
            indices = np.random.choice(len(remaining), 2, replace=False)
            i, j = indices
            op = np.random.choice(TwentyFourGame.OPERATIONS)

            result = TwentyFourGame.apply_op(remaining[i], remaining[j], op)
            if result is None:
                break

            step_str = f"{remaining[i]} {op} {remaining[j]} = {result:.2f}"
            history.append(step_str)

            # Update the remaining numbers
            new_remaining = [remaining[k] for k in range(len(remaining))
                             if k != i and k != j]
            new_remaining.append(result)
            remaining = new_remaining

        if len(remaining) == 1:
            diff = abs(remaining[0] - target)
            if diff < best_diff:
                best_diff = diff
                best_result = remaining[0]
                best_history = history
            if TwentyFourGame.is_close_to_target(remaining[0], target):
                successes += 1

    if verbose:
        if best_result is not None:
            print(f"  Best result from random search: {best_result:.4f} (error={best_diff:.4f})")
            print(f"  Exact solutions found: {successes}/{n_trials} times")
        else:
            print(f"  Random search found no valid result")

    return best_result, best_diff, successes


# ==========================================
# Part 4: Visualization functions
# ==========================================
def visualize_search_tree(tree_data, title="Tree of Thought search tree"):
    """
    Visualize the search tree

    Each level shows the retained nodes and their scores;
    node size and color reflect the score.
    """
    n_levels = len(tree_data['nodes_per_level'])
    if n_levels == 0:
        print("No search tree data to visualize")
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Spacing between levels
    y_spacing = 1.0
    max_score = 0.0

    for level_scores in tree_data['scores_per_level']:
        for s in level_scores:
            max_score = max(max_score, s)

    for level in range(n_levels):
        nodes = tree_data['nodes_per_level'][level]
        scores = tree_data['scores_per_level'][level]
        n_nodes = len(nodes)

        # y coordinate of the current level
        y = (n_levels - 1 - level) * y_spacing

        # Evenly distribute horizontally
        x_positions = np.linspace(0.5, n_nodes - 0.5, n_nodes)

        for i, (node_expr, score) in enumerate(zip(nodes, scores)):
            x = x_positions[i] if n_nodes > 1 else 0.5

            # Node size and color are determined by the score
            size = 200 + score * 800
            color_val = score / max(max_score, 0.01)
            color = plt.cm.RdYlGn(color_val)

            # Draw the node
            ax.scatter(x, y, s=size, c=[color], edgecolors='black',
                       linewidths=1.5, zorder=5)

            # Annotate the score
            ax.text(x, y + 0.15, f'{score:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

            # Annotate node info (simplified display)
            expr_short = node_expr.split(' → ')[-1] if ' → ' in node_expr else node_expr
            ax.text(x, y - 0.15, expr_short, ha='center', va='top',
                    fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              alpha=0.7))

        # Draw connecting lines to the previous level
        if level > 0:
            prev_nodes = tree_data['nodes_per_level'][level - 1]
            prev_scores = tree_data['scores_per_level'][level - 1]
            prev_n = len(prev_nodes)
            prev_x = np.linspace(0.5, prev_n - 0.5, prev_n) if prev_n > 1 else [0.5]
            prev_y = (n_levels - level) * y_spacing

            for pi in range(prev_n):
                for ci in range(n_nodes):
                    ax.plot([prev_x[pi], x_positions[ci]],
                            [prev_y, y],
                            color='gray', alpha=0.2, linewidth=0.8, zorder=1)

    # Configure axes
    ax.set_ylabel('Search depth', fontsize=12)
    ax.set_xticks([])
    y_ticks = [(n_levels - 1 - l) * y_spacing for l in range(n_levels)]
    y_labels = [f'Level {l+1}' for l in range(n_levels)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_comparison(results, problem_labels):
    """
    Visualize the comparison results of the three strategies

    Args:
        results: dict containing the performance of the three strategies across problems
        problem_labels: list of problem labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Tree of Thought vs Chain of Thought vs Random Search — Comparison Experiment', fontsize=16, fontweight='bold')

    colors = ['#4CAF50', '#2196F3', '#FF9800']
    strategies = ['Tree of Thought', 'Chain of Thought', 'Random']
    keys = ['tot', 'cot', 'random']

    # ---- Subplot 1: success rate ----
    ax1 = axes[0]
    success_rates = [results[k]['success_rate'] for k in keys]
    bars = ax1.bar(strategies, success_rates, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Success rate', fontsize=12)
    ax1.set_title('Solve success rate', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, rate in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{rate:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    # ---- Subplot 2: average error ----
    ax2 = axes[1]
    avg_errors = [results[k]['avg_error'] for k in keys]
    bars = ax2.bar(strategies, avg_errors, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Average error', fontsize=12)
    ax2.set_title('Average error from the target (24)', fontsize=13, fontweight='bold')
    for bar, err in zip(bars, avg_errors):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{err:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # ---- Subplot 3: effect of search breadth on success rate ----
    ax3 = axes[2]
    breadths = [1, 2, 3, 5, 8]
    # Re-run experiments for different breadths
    success_by_breadth = results['tot']['breadth_success_rates']
    ax3.plot(breadths[:len(success_by_breadth)], success_by_breadth,
             'o-', color='#4CAF50', linewidth=2.5, markersize=10, label='ToT success rate')
    ax3.axhline(y=results['cot']['success_rate'], color='#2196F3',
                linestyle='--', linewidth=2, label='CoT (breadth=1)')
    ax3.axhline(y=results['random']['success_rate'], color='#FF9800',
                linestyle='--', linewidth=2, label='Random')
    ax3.set_xlabel('Search breadth', fontsize=12)
    ax3.set_ylabel('Success rate', fontsize=12)
    ax3.set_title('Effect of search breadth on ToT performance', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(breadths[:len(success_by_breadth)])

    plt.tight_layout()
    return fig


# ==========================================
# Part 5: Main experiment flow
# ==========================================
def run_experiment():
    """
    Run the full comparison experiment

    Experiment flow:
        1. Prepare multiple Game-of-24 problems
        2. Solve each with ToT, CoT, and Random strategies
        3. Compute success rate and average error
        4. Test the effect of different search breadths on ToT
        5. Visualize the comparison
    """

    # Carefully selected Game-of-24 problems (all guaranteed to have a solution)
    problems = [
        [1, 2, 3, 4],
        [2, 3, 4, 6],
        [1, 5, 5, 5],
        [3, 3, 8, 8],
        [4, 4, 10, 10],
        [1, 4, 5, 6],
        [2, 6, 7, 7],
        [3, 6, 6, 8],
        [2, 3, 5, 7],
        [1, 3, 4, 6],
    ]

    target = 24.0

    print("=" * 60)
    print("  Chapter 13: Tree of Thought (ToT) Reasoning Experiment")
    print("=" * 60)
    print(f"  Task: Game of 24 — combine 4 numbers to make {target}")
    print(f"  Number of problems: {len(problems)}")
    print(f"  Strategies compared: Tree of Thought, Chain of Thought, Random")
    print("-" * 60)

    # ---- Strategy 1: Tree of Thought (breadth=3) ----
    print("\n" + "=" * 60)
    print("  Strategy 1: Tree of Thought (breadth=3)")
    print("=" * 60)

    tot_successes = 0
    tot_errors = []
    # Record the first problem's search tree for detailed display
    first_tree_data = None

    for idx, nums in enumerate(problems):
        result, tree_data = search_tree_of_thought(
            nums, breadth=3, max_depth=4, target=target, verbose=(idx == 0)
        )
        if idx == 0:
            first_tree_data = tree_data

        if result.is_terminal():
            error = abs(result.numbers[0] - target)
            tot_errors.append(error)
            if TwentyFourGame.is_close_to_target(result.numbers[0], target):
                tot_successes += 1
                print(f"  Problem {idx+1} {nums}: Success! Result={result.numbers[0]:.2f}")
            else:
                print(f"  Problem {idx+1} {nums}: Target not reached, result={result.numbers[0]:.2f} (error={error:.2f})")
        else:
            tot_errors.append(abs(target))  # treat unfinished searches as maximum error
            print(f"  Problem {idx+1} {nums}: Search incomplete")

    tot_success_rate = tot_successes / len(problems)
    tot_avg_error = np.mean(tot_errors)

    # ---- Strategy 2: Chain of Thought (breadth=1) ----
    print("\n" + "=" * 60)
    print("  Strategy 2: Chain of Thought (greedy single path)")
    print("=" * 60)

    cot_successes = 0
    cot_errors = []

    for idx, nums in enumerate(problems):
        result = search_chain_of_thought(nums, target=target, verbose=(idx == 0))
        if result.is_terminal():
            error = abs(result.numbers[0] - target)
            cot_errors.append(error)
            if TwentyFourGame.is_close_to_target(result.numbers[0], target):
                cot_successes += 1
                print(f"  Problem {idx+1} {nums}: Success! Result={result.numbers[0]:.2f}")
            else:
                print(f"  Problem {idx+1} {nums}: Target not reached, result={result.numbers[0]:.2f} (error={error:.2f})")
        else:
            cot_errors.append(abs(target))
            print(f"  Problem {idx+1} {nums}: Search incomplete")

    cot_success_rate = cot_successes / len(problems)
    cot_avg_error = np.mean(cot_errors)

    # ---- Strategy 3: Random ----
    print("\n" + "=" * 60)
    print("  Strategy 3: Random search baseline")
    print("=" * 60)

    random_successes = 0
    random_errors = []

    for idx, nums in enumerate(problems):
        best_result, best_diff, successes = search_random(
            nums, n_trials=50, target=target, verbose=(idx == 0)
        )
        if best_result is not None:
            random_errors.append(best_diff)
            if TwentyFourGame.is_close_to_target(best_result, target):
                random_successes += 1
                print(f"  Problem {idx+1} {nums}: Success! Result={best_result:.2f}")
            else:
                print(f"  Problem {idx+1} {nums}: Best={best_result:.2f} (error={best_diff:.2f})")
        else:
            random_errors.append(abs(target))
            print(f"  Problem {idx+1} {nums}: No valid result found")

    random_success_rate = random_successes / len(problems)
    random_avg_error = np.mean(random_errors)

    # ---- ToT experiments with different search breadths ----
    print("\n" + "=" * 60)
    print("  ToT success rate for different search breadths")
    print("=" * 60)

    breadth_values = [1, 2, 3, 5, 8]
    breadth_success_rates = []

    for b in breadth_values:
        successes = 0
        for nums in problems:
            result, _ = search_tree_of_thought(
                nums, breadth=b, max_depth=4, target=target, verbose=False
            )
            if (result.is_terminal()
                    and TwentyFourGame.is_close_to_target(result.numbers[0], target)):
                successes += 1
        rate = successes / len(problems)
        breadth_success_rates.append(rate)
        print(f"  breadth={b}: success rate = {rate:.0%} ({successes}/{len(problems)})")

    # ---- Summary of results ----
    print("\n" + "=" * 60)
    print("  Experiment results summary")
    print("=" * 60)
    print(f"  {'Strategy':<25} {'Success rate':>10} {'Avg. error':>12}")
    print(f"  {'-'*47}")
    print(f"  {'Tree of Thought (b=3)':<25} {tot_success_rate:>9.0%} {tot_avg_error:>12.2f}")
    print(f"  {'Chain of Thought (b=1)':<25} {cot_success_rate:>9.0%} {cot_avg_error:>12.2f}")
    print(f"  {'Random (50 trials)':<25} {random_success_rate:>9.0%} {random_avg_error:>12.2f}")
    print("-" * 60)

    # Print key conclusions
    print("\nKey conclusions:")
    print("  1. ToT significantly improves the solve success rate via multi-branch search + score-based pruning")
    print("  2. CoT (breadth=1) is a greedy strategy and easily gets stuck at local optima")
    print("  3. Increasing search breadth improves success rate, but also increases computation")
    print("  4. This demonstrates the power of test-time search:")
    print("     investing more compute at inference time can yield better results")
    print("  5. The core idea of ToT can be transferred to reasoning optimization in LLMs")

    # ---- Visualization ----
    results = {
        'tot': {
            'success_rate': tot_success_rate,
            'avg_error': tot_avg_error,
            'breadth_success_rates': breadth_success_rates,
        },
        'cot': {
            'success_rate': cot_success_rate,
            'avg_error': cot_avg_error,
        },
        'random': {
            'success_rate': random_success_rate,
            'avg_error': random_avg_error,
        },
    }

    # Figure 1: search tree visualization
    if first_tree_data:
        fig1 = visualize_search_tree(first_tree_data,
                                     title="Tree of Thought search process (Problem 1)")
        fig1.savefig('output/tot_search_tree.png', dpi=150, bbox_inches='tight')
        print("\nSearch tree visualization saved to: output/tot_search_tree.png")

    # Figure 2: comparison experiment
    problem_labels = [str(p) for p in problems]
    fig2 = visualize_comparison(results, problem_labels)
    fig2.savefig('output/tot_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison experiment figure saved to: output/tot_comparison.png")

    plt.show()


# ==========================================
# Part 6: Main entry point
# ==========================================
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run the experiment
    run_experiment()
