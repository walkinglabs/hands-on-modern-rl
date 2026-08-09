"""
Chapter 3: Numerical verification of the Bellman equations
Verify the Bellman expectation equation and the Bellman optimality equation with code

This experiment demonstrates:
1. Manually computing the Bellman expectation equation V^π(s)
2. Numerically solving for V^π(s) and V*(s) using Value Iteration
3. Comparing the results of both methods to verify consistency
4. Showing the step-by-step convergence process of value iteration

How to run:
    python bellman_equation_verify.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)


# ==========================================
# Part 1: Define a simple 3-state MDP
# ==========================================
# We hand-build a tiny MDP so it's easy to compute manually and verify in code.
#
# State set: S = {s0, s1, s2}
# Action set: A = {a0, a1} (two actions available in every state)
#
# The transition probabilities P(s'|s,a) and rewards R(s,a) are defined as follows:
#
# From s0:
#   Take a0 → transition to s1 (probability 1.0), reward = 1
#   Take a1 → transition to s2 (probability 1.0), reward = 2
#
# From s1:
#   Take a0 → transition to s0 (probability 0.5), reward = -1
#                 transition to s2 (probability 0.5), reward = -1
#   Take a1 → transition to s1 (probability 0.8), reward = 0
#                 transition to s2 (probability 0.2), reward = 0
#
# From s2:
#   Take a0 → transition to s1 (probability 1.0), reward = 3
#   Take a1 → transition to s2 (probability 1.0), reward = 1 (self-loop)

# Number of states and actions
N_STATES = 3
N_ACTIONS = 2

# Transition probabilities: P[s][a] = {next state: probability}
P = {
    0: {  # s0
        0: {1: 1.0},    # a0 → s1 with probability 1.0
        1: {2: 1.0},    # a1 → s2 with probability 1.0
    },
    1: {  # s1
        0: {0: 0.5, 2: 0.5},  # a0 → s0 with probability 0.5, s2 with probability 0.5
        1: {1: 0.8, 2: 0.2},  # a1 → s1 with probability 0.8, s2 with probability 0.2
    },
    2: {  # s2
        0: {1: 1.0},    # a0 → s1 with probability 1.0
        1: {2: 1.0},    # a1 → s2 with probability 1.0 (self-loop)
    },
}

# Reward function: R[s][a] = immediate reward
R = {
    0: {0: 1, 1: 2},     # s0: a0 reward 1, a1 reward 2
    1: {0: -1, 1: 0},    # s1: a0 reward -1, a1 reward 0
    2: {0: 3, 1: 1},     # s2: a0 reward 3, a1 reward 1
}

GAMMA = 0.9  # Discount factor


# ==========================================
# Part 2: Bellman expectation equation -- manual computation
# ==========================================
def manual_bellman_expectation():
    """
    Manually compute the Bellman expectation equation V^π(s)

    Given a fixed policy π, the Bellman expectation equation is:
        V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')]

    We use a simple uniform random policy:
        π(a|s) = 0.5 (each action chosen with equal probability)

    Manual derivation (let V^π(s0) = v0, V^π(s1) = v1, V^π(s2) = v2):

    V^π(s0) = 0.5 * [R(s0,a0) + γ * V^π(s1)] + 0.5 * [R(s0,a1) + γ * V^π(s2)]
            = 0.5 * [1 + 0.9 * v1] + 0.5 * [2 + 0.9 * v2]
            = 0.5 + 0.45*v1 + 1 + 0.45*v2
            = 1.5 + 0.45*v1 + 0.45*v2  ........................ (Equation 1)

    V^π(s1) = 0.5 * [R(s1,a0) + γ * (0.5*V^π(s0) + 0.5*V^π(s2))]
            + 0.5 * [R(s1,a1) + γ * (0.8*V^π(s1) + 0.2*V^π(s2))]
            = 0.5 * [-1 + 0.9*(0.5*v0 + 0.5*v2)]
            + 0.5 * [0 + 0.9*(0.8*v1 + 0.2*v2)]
            = -0.5 + 0.225*v0 + 0.225*v2 + 0.36*v1 + 0.09*v2
            = -0.5 + 0.225*v0 + 0.36*v1 + 0.315*v2  ............ (Equation 2)

    V^π(s2) = 0.5 * [R(s2,a0) + γ * V^π(s1)] + 0.5 * [R(s2,a1) + γ * V^π(s2)]
            = 0.5 * [3 + 0.9 * v1] + 0.5 * [1 + 0.9 * v2]
            = 1.5 + 0.45*v1 + 0.5 + 0.45*v2
            = 2.0 + 0.45*v1 + 0.45*v2  ........................ (Equation 3)
    """
    print("=" * 60)
    print("  Bellman expectation equation -- manual derivation")
    print("=" * 60)
    print()
    print("Given policy: uniform random π(a|s) = 0.5")
    print(f"Discount factor: γ = {GAMMA}")
    print()
    print("System of equations (v0 = V^π(s0), v1 = V^π(s1), v2 = V^π(s2)):")
    print("  v0 = 1.5   + 0.45*v1 + 0.45*v2  ...... (Equation 1)")
    print("  v1 = -0.5  + 0.225*v0 + 0.36*v1 + 0.315*v2  (Equation 2)")
    print("  v2 = 2.0   + 0.45*v1 + 0.45*v2  ...... (Equation 3)")
    print()

    # Solve the linear system A * v = b
    # Equation 1: v0 - 0.45*v1 - 0.45*v2 = 1.5
    # Equation 2: -0.225*v0 + (1-0.36)*v1 - 0.315*v2 = -0.5
    # Equation 3: -0.45*v1 + (1-0.45)*v2 = 2.0

    A = np.array([
        [1.0,   -0.45,   -0.45],
        [-0.225, 0.64,   -0.315],
        [0.0,   -0.45,    0.55],
    ])
    b = np.array([1.5, -0.5, 2.0])

    manual_V = np.linalg.solve(A, b)

    print("Solving the linear system by hand gives:")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {manual_V[i]:.6f}")
    print()
    return manual_V


# ==========================================
# Part 3: Policy evaluation -- iteratively solve the Bellman expectation equation
# ==========================================
def policy_evaluation(policy, max_iter=1000, tol=1e-8):
    """
    Policy evaluation: iteratively solve the Bellman expectation equation

    Bellman expectation equation (iterative form):
        V(s) ← Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    Iterate repeatedly until V(s) converges; the converged V(s) is V^π(s).

    Args:
        policy: policy π(a|s), shape (N_STATES, N_ACTIONS)
        max_iter: maximum number of iterations
        tol: convergence threshold
    Returns:
        V: state value function
        history: record of V values at each iteration (for visualizing convergence)
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # Bellman expectation equation: weighted sum over all actions
            for a in range(N_ACTIONS):
                # π(a|s) * [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
                action_value = R[s][a]
                for next_s, prob in P[s][a].items():
                    action_value += GAMMA * prob * V[next_s]
                V_new[s] += policy[s][a] * action_value

        # Check for convergence
        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    return V, history


# ==========================================
# Part 4: Value iteration -- solve the Bellman optimality equation
# ==========================================
def value_iteration(max_iter=1000, tol=1e-8):
    """
    Value iteration: solve the Bellman optimality equation to find V*(s)

    Bellman optimality equation (iterative form):
        V(s) ← max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    Difference from the Bellman expectation equation:
    - Expectation equation: given a policy π, solve for V^π(s)
    - Optimality equation: optimize over all policies to solve for V*(s)

    V*(s) satisfies:
        V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ * V*(s')]

    Args:
        max_iter: maximum number of iterations
        tol: convergence threshold
    Returns:
        V_star: optimal state value function
        optimal_policy: optimal policy
        history: convergence history
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # Compute Q(s, a) for each action
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s, prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)

            # Bellman optimality equation: take the max instead of the expectation
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    # Extract the optimal policy from V*
    optimal_policy = extract_optimal_policy(V)

    return V, optimal_policy, history


def extract_optimal_policy(V):
    """
    Extract the optimal policy π* from the optimal value function V*

    π*(s) = argmax_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s')]
    """
    policy = np.zeros((N_STATES, N_ACTIONS))

    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s, prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)

        best_action = np.argmax(q_values)
        policy[s][best_action] = 1.0  # Deterministic policy

    return policy


# ==========================================
# Part 5: Comparing results and visualization
# ==========================================
def verify_results():
    """Verify the consistency between manual computation and iterative computation"""

    # Uniform random policy
    uniform_policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    print("=" * 60)
    print("  Numerical verification of the Bellman equations")
    print("=" * 60)
    print()

    # ------------------------------------------
    # Comparison 1: manual computation vs iterative solution (Bellman expectation equation)
    # ------------------------------------------
    print("-" * 60)
    print("  Comparison 1: two ways of solving the Bellman expectation equation V^π(s)")
    print("-" * 60)

    # Manual computation
    manual_V = manual_bellman_expectation()

    # Iterative computation
    iter_V, iter_history = policy_evaluation(uniform_policy)
    print("Policy evaluation (iterative solution) result:")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")
    print()

    # Comparison
    print(">>> Comparison result:")
    print(f"  {'State':<8s} {'Manual':<15s} {'Iterative':<15s} {'Error':<15s}")
    for i in range(N_STATES):
        error = abs(manual_V[i] - iter_V[i])
        print(f"  s{i:<6d} {manual_V[i]:<15.8f} {iter_V[i]:<15.8f} {error:<15.2e}")

    all_match = np.allclose(manual_V, iter_V, atol=1e-6)
    print(f"\n  Conclusion: {'Fully consistent ✓' if all_match else 'Discrepancy found ✗'}")
    print(f"  (Manually solving the linear system = iteratively converging -- different roads, same destination!)")
    print()

    # ------------------------------------------
    # Comparison 2: Bellman expectation equation vs Bellman optimality equation
    # ------------------------------------------
    print("-" * 60)
    print("  Comparison 2: V^π(s) vs V*(s)")
    print("-" * 60)
    print()

    V_star, optimal_policy, vi_history = value_iteration()

    print("Bellman expectation equation V^π(s) (under the uniform random policy):")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")

    print()
    print("Bellman optimality equation V*(s) (under the optimal policy):")
    for i in range(N_STATES):
        print(f"  V*(s{i}) = {V_star[i]:.6f}")

    print()
    print("Optimal policy π*:")
    action_names = ['a0', 'a1']
    for s in range(N_STATES):
        best = np.argmax(optimal_policy[s])
        print(f"  π*(s{s}) = {action_names[best]}")

    print()
    print(f"  {'State':<8s} {'V^π(s) random policy':<20s} {'V*(s) optimal policy':<20s} {'Gain':<10s}")
    for i in range(N_STATES):
        improvement = V_star[i] - iter_V[i]
        print(f"  s{i:<6d} {iter_V[i]:<20.6f} {V_star[i]:<20.6f} {improvement:>+10.6f}")

    print()
    print("  Analysis: V*(s) >= V^π(s) holds for all states (the optimal policy is never worse)")

    # ------------------------------------------
    # Visualization: value iteration convergence process
    # ------------------------------------------
    visualize_convergence(iter_history, vi_history)


def visualize_convergence(expectation_history, optimal_history):
    """
    Visualize the convergence process of value iteration

    Left plot: convergence of policy evaluation (Bellman expectation equation)
    Right plot: convergence of value iteration (Bellman optimality equation)
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------------------------
    # Left plot: Bellman expectation equation convergence
    # ------------------------------------------
    ax1 = axes[0]
    history_arr = np.array(expectation_history)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    state_labels = ['V^π(s0)', 'V^π(s1)', 'V^π(s2)']

    for s in range(N_STATES):
        ax1.plot(history_arr[:, s], color=colors[s], label=state_labels[s], linewidth=2)

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('State value V(s)', fontsize=11)
    ax1.set_title('Policy evaluation convergence\n(Bellman expectation equation)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Only show the first 30 iterations (it has already converged after that)
    n_show = min(30, len(expectation_history))
    ax1.set_xlim(0, n_show)

    # ------------------------------------------
    # Right plot: Bellman optimality equation convergence
    # ------------------------------------------
    ax2 = axes[1]
    history_arr2 = np.array(optimal_history)

    state_labels_star = ['V*(s0)', 'V*(s1)', 'V*(s2)']
    for s in range(N_STATES):
        ax2.plot(history_arr2[:, s], color=colors[s], label=state_labels_star[s], linewidth=2)

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('State value V(s)', fontsize=11)
    ax2.set_title('Value iteration convergence\n(Bellman optimality equation)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    n_show2 = min(30, len(optimal_history))
    ax2.set_xlim(0, n_show2)

    plt.tight_layout()
    plt.savefig('output/bellman_equation_verify_results.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to output/bellman_equation_verify_results.png")
    plt.show()


# ==========================================
# Part 6: Step-by-step display of the value iteration process
# ==========================================
def print_value_iteration_steps(n_steps=10):
    """
    Print the first n_steps steps of value iteration, showing the step-by-step
    convergence process

    This lets the reader see, intuitively:
    - V(s) starts at 0 (nothing is known yet)
    - At each step, V(s) is updated via the Bellman equation, moving closer to
      the true value
    - After enough steps, V(s) converges to the optimal value
    """
    print()
    print("=" * 60)
    print("  Step-by-step convergence of value iteration")
    print("=" * 60)
    print()
    print("  Iter  |  V*(s0)    V*(s1)    V*(s2)   |  Max change")
    print("  " + "-" * 55)

    V = np.zeros(N_STATES)

    for iteration in range(n_steps):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s, prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))

        print(f"  {iteration + 1:4d}  |"
              f"  {V_new[0]:>8.4f}  {V_new[1]:>8.4f}  {V_new[2]:>8.4f} |"
              f"  {delta:>8.6f}")

        V = V_new

        if delta < 1e-8:
            print(f"\n  Converged at step {iteration + 1}!")
            break

    print("  " + "-" * 55)
    print(f"\n  Final optimal state value function:")
    for i in range(N_STATES):
        print(f"    V*(s{i}) = {V[i]:.6f}")

    # Print the optimal policy
    print(f"\n  Optimal policy:")
    action_names = ['a0', 'a1']
    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s, prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)
        best = np.argmax(q_values)
        print(f"    π*(s{s}) = {action_names[best]}  "
              f"(Q(s{s},a0)={q_values[0]:.4f}, Q(s{s},a1)={q_values[1]:.4f})")


# ==========================================
# Main program
# ==========================================
def main():
    """Main function: run all verification experiments"""

    # 1. Verify the consistency between manual computation and iterative computation
    verify_results()

    # 2. Show the step-by-step convergence of value iteration
    print_value_iteration_steps(n_steps=20)


if __name__ == "__main__":
    main()
