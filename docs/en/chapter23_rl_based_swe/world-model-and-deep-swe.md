# 21.2 Code World Model and DeepSWE

The previous section showed the core bottleneck of Meta SWE-RL: **long-horizon training is unstable**. Once a trajectory runs past 16 steps, RL struggles to learn credit assignment.

There's a deeper problem underneath: **every rollout has to run the real test suite, and that's slow and expensive**. A single trajectory involves multiple `pytest` calls, each taking anywhere from seconds to minutes. If one RL training run needs a million rollouts, the total wall-clock time can stretch to weeks.

Two breakthrough directions emerged in the second half of 2025:

- **Code World Model (CWM)**: train a model that "simulates" code execution, so training doesn't depend on real tests
- **DeepSWE**: train a deep agent by combining a world model with long-sequence RL

This section works through both in detail.

## 12.3.1 Code World Model (CWM)

The core idea behind [Code World Model](https://arxiv.org/abs/2510.02387) (CWM, September 2025): **model code execution as an MDP, and train a world model to predict how the code's state changes**.

### CWM's MDP formulation

Cast the SWE task as an MDP:

| MDP element                         | SWE counterpart                                                           |
| ----------------------------------- | ------------------------------------------------------------------------- |
| State $s_t$                         | The repo's code + the current edit history + test results                 |
| Action $a_t$                        | The model's next move (read a file, edit code, run tests)                 |
| Transition $T(s_{t+1} \| s_t, a_t)$ | Code execution—how the state changes once files are edited                |
| Reward $r_t$                        | Per-step feedback (intermediate state) plus the final reward (tests pass) |

### Training the world model

CWM trains a separate **world model** $\hat{T}$:

$$\hat{T}(s_{t+1} | s_t, a_t) \approx T(s_{t+1} | s_t, a_t)$$

This world model is itself an LLM: it takes $(s_t, a_t)$ as input and outputs $s_{t+1}$.

Training data:

- Collect trajectories from real SWE tasks
- Use $(s_t, a_t, s_{t+1})$ triples as training examples
- Train the world model to learn "given the current code state and an action, predict the next state"

### CWM's training pipeline

```text
┌────────────────────────────────────────────────────────────┐
│ Phase 1: World model pretraining                            │
│   - Collect trajectories from real SWE tasks                │
│   - Train the world model to predict code-state changes     │
├────────────────────────────────────────────────────────────┤
│ Phase 2: RL with the world model                            │
│   - The policy interacts with the world model                │
│   - The world model quickly simulates "code execution"      │
│   - No real tests needed—100x faster                        │
├────────────────────────────────────────────────────────────┤
│ Phase 3: Fine-tune on real tests                             │
│   - Take the policy trained inside the world model and run   │
│     a final round of RL in the real environment              │
│   - Correct for drift between the world model and reality    │
└────────────────────────────────────────────────────────────┘
```

### CWM's advantages

**Advantage one: speed**

A world-model call is a single LLM forward pass—milliseconds. A real test takes seconds to minutes. **CWM speeds up training by 100-1000x.**

**Advantage two: it can simulate failure**

The world model can simulate "what happens if I make this change"—the policy can explore failure modes extensively inside the world model and learn to avoid them.

**Advantage three: high data efficiency**

The world model learns the underlying regularities of code execution, and those regularities generalize to new tasks.

### CWM's limitations

**Limitation one: world model accuracy**

The world model is an LLM, and LLMs make mistakes. If it predicts the wrong "code execution result," the policy learns the wrong strategy.

The industrial mitigation: **periodically recalibrate the world model against real tests**—every N rollout steps, correct it using real-test ground truth.

**Limitation two: complex dependencies**

Code execution involves complex dependencies—library versions, environment variables, external services. The world model struggles to fully simulate these.

**Limitation three: training cost**

Training the world model itself requires large volumes of trajectory data and compute—a more complex undertaking than training the policy directly.

### CWM's relationship to model-based RL

CWM is model-based RL applied to the SWE domain. Classic model-based RL methods—MuZero, Dreamer—already proved their value in games and control tasks. CWM carries that same idea into the LLM-plus-SWE setting.

See also: [Chapter 5 model-based RL](../chapter10_ppo/rl-long-horizon-planning) and [Chapter 12 future trends / model-based RL](../chapter28_vla/embodied-intelligence/model-based-rl/).

## 12.3.2 DeepSWE and RL for Long-Horizon Agents

[DeepSWE-Preview](https://www.together.ai/blog/deepswe) (Agentica × Together AI, July 2025) is another SWE-RL breakthrough. Its core contribution: **training a long-horizon agent with verifiable reward on trajectories of 32+ steps**.

### DeepSWE's core approach

DeepSWE's key insight: **the root cause of long-horizon RL instability is the difficulty of credit assignment**. A 32-step trajectory only produces a reward at the final test—how do you propagate that single signal back across all 32 steps?

DeepSWE addresses this with three techniques:

**Technique one: step-level reward shaping**

Rather than relying on the final reward alone, give every step a shaping reward:

```python
def deep_swe_reward(trajectory, final_test_result):
    # Base reward: the final test outcome
    base_reward = 1.0 if final_test_result else 0.0

    # Shaping reward: how much each step "contributed"
    step_rewards = []
    for step in trajectory:
        # Use an LLM judge to score whether this step was "meaningful"
        step_quality = llm_judge(step)
        step_rewards.append(step_quality)

    # Total reward = base + sum(step rewards)
    return base_reward + sum(step_rewards) * 0.1
```

This shaping gives the model feedback at every step, sidestepping the credit-assignment problem.

**Technique two: a value model**

DeepSWE reintroduces a value model, in the same spirit as VAPO—see [Chapter 7 VAPO](../chapter18_grpo/grpo-family).

The value model $V_\phi(s_t)$ estimates the expected future reward from the current state. That lets RL use GAE for credit assignment:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.

**Technique three: hierarchical RL**

Split the long trajectory into layers:

- **High-level policy**: decides "which file to fix next" (coarse-grained)
- **Low-level policy**: decides "exactly how to edit this file" (fine-grained)

The high-level policy trains on sparse reward (the final test); the low-level policy trains on dense reward (per-step shaping).

### DeepSWE's training pipeline

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Data collection                                   │
│   - Roll out the SFT model on SWE-bench                    │
│   - Collect trajectories of 32-64 steps                    │
├──────────────────────────────────────────────────────────┤
│ Phase 2: World model training (similar to CWM)              │
│   - Speeds up the RL phases that follow                    │
├──────────────────────────────────────────────────────────┤
│ Phase 3: Value model training                               │
│   - Train V_phi on the collected trajectories               │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Hierarchical RL                                    │
│   - High-level policy: PPO + sparse reward                  │
│   - Low-level policy: GRPO + dense reward                   │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Test-time search                                   │
│   - Use MCTS or beam search at inference time               │
│   - Evaluate intermediate states with the value model        │
└──────────────────────────────────────────────────────────┘
```

### DeepSWE's results

DeepSWE's results on SWE-bench Verified:

| Model                   | SWE-bench Verified |
| ----------------------- | ------------------ |
| Meta SWE-RL             | 41.0%              |
| **DeepSWE (ByteDance)** | **50.0%**          |
| SWE-Lancer (OpenAI)     | 45.0%              |
| Claude Opus 4.5 + tools | 60%+               |

DeepSWE reaches 50% among open-source models—evidence that long-horizon RL training is viable.

### DeepSWE's relationship to VAPO

DeepSWE's design closely resembles [ByteDance's VAPO](../chapter18_grpo/grpo-family)—both replace GRPO's critic-free approach with a value model. That reflects a shared conclusion inside ByteDance Seed: **long-horizon tasks need a critic**.

This also confirms the conclusion from [Chapter 7's GRPO improvement family](../chapter18_grpo/grpo-family): **going critic-free was an engineering compromise, not an algorithmic necessity**. On long-horizon tasks—long CoT reasoning, long SWE trajectories—the value model has re-earned its keep.

## 12.3.3 Test-Time Search Integration

Both CWM and DeepSWE integrate **test-time search**—using MCTS or beam search at inference time to boost performance.

### MCTS on CWM

CWM's world model makes MCTS efficient:

```python
def cwm_mcts(issue, model, world_model, depth=10):
    # Run MCTS on top of the world model
    root_state = initialize_state(issue)

    for _ in range(N_iter):
        # Selection: pick the best child node using UCB
        node = select(root_state)

        # Expansion: generate an action with the policy, simulate the next state with the world model
        action = model.policy(node.state)
        next_state = world_model.predict(node.state, action)

        # Simulation: a quick rollout to termination
        rollout_reward = quick_rollout(next_state, world_model)

        # Backprop: update node statistics
        backpropagate(node, rollout_reward)

    # Return the best action from the root
    return best_action(root_state)
```

The entire MCTS run happens inside the world model—**no real tests needed**, so it's extremely fast.

### Beam Search on DeepSWE

At inference time, DeepSWE uses beam search:

```python
def deep_swe_beam_search(issue, model, value_model, K=4):
    beams = [{"state": init_state(issue), "score": 0}]

    for step in range(MAX_STEPS):
        candidates = []
        for beam in beams:
            # Generate K candidate actions
            actions = model.generate_actions(beam["state"], n=K)

            for action in actions:
                next_state = apply_action(beam["state"], action)
                # Score with the value model
                value = value_model.estimate(next_state)
                candidates.append({
                    "state": next_state,
                    "score": beam["score"] + value
                })

        # Keep the top-K
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

    return beams[0]["state"]
```

Beam search lets DeepSWE trade extra inference-time compute for accuracy—consistent with [Chapter 17's test-time compute scaling](../chapter19_reasoning/test-time-scaling).

## 12.3.4 Comparing Industrial Approaches

As of mid-2026, here's where the mainstream industrial SWE-RL approaches stand:

| Approach                    | Representative   | Characteristics        | SWE-bench Verified |
| --------------------------- | ---------------- | ---------------------- | ------------------ |
| Simple GRPO                 | Meta SWE-RL      | Open-source, simple    | 41.0%              |
| + World model               | Code World Model | Fast training          | ~45%               |
| + Value + search            | DeepSWE          | Long-horizon           | 50.0%              |
| + Multi-agent collaboration | Claude Opus 4.7  | Closed-source, complex | 65%+               |

The pattern is clear: **algorithmic complexity correlates with performance**. Moving from plain GRPO to multi-agent collaboration, each refinement buys a few more percentage points.

## Summary

Code World Model and DeepSWE are two important breakthroughs in SWE-RL:

- **CWM**: uses a world model to speed up training, avoiding the high cost of real tests
- **DeepSWE**: uses a value model, hierarchical RL, and test-time search to handle long horizons

Both reflect a shared truth about SWE-RL: **long-horizon tasks demand more sophisticated algorithms**. Plain GRPO is a good fit for short tasks (under 8 steps), but the 16-64 step trajectories typical of SWE tasks call for stronger tools.

The next section looks at Self-play SWE-RL—**letting the model generate its own training data**, cutting the dependence on human-curated data even further.
