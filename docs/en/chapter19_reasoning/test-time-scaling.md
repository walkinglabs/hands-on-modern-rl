# 17.3 Test-time Compute Scaling

The previous section showed o1/o3/o4 far outperforming traditional LLMs on hard tasks. Yet o1 isn't larger than GPT-4o in parameter count, and its training compute isn't significantly higher either. **So why is it so strong?**

OpenAI's official answer at the o1 launch was that o1 spends more compute at **inference time** — that is, test-time compute. In the second half of 2024, [Snell et al.](https://arxiv.org/abs/2408.03314) turned that answer into a systematic research program, giving rise to the field now called **Test-time Compute Scaling**.

## 10.2.1 Training Compute vs. Inference Compute

Compute allocation in a traditional LLM is heavily skewed toward one end:

```text
Pretraining compute:            ~10^23 FLOPs (GPT-4 scale)
Post-training compute:          ~10^21 FLOPs
Inference compute (per call):   ~10^15 FLOPs
```

In other words, **pretraining uses eight orders of magnitude more compute than inference**. That split makes sense under the "model answers in one forward pass" paradigm: inference gets little compute because the model isn't supposed to think.

Reasoning models break that assumption. Working an AIME problem, o1 might generate a chain of thought running 10K–100K tokens — two orders of magnitude longer than the 200–500 token answer a traditional LLM would give. **o1 pushes inference compute from ~10^15 up to ~10^17**.

Snell et al.'s key question follows directly: **given a fixed total budget (training + inference), where should you spend it?**

## 10.2.2 Snell 2024's Core Findings

[Snell et al. 2024](https://arxiv.org/abs/2408.03314) ("Scaling LLM Test-Time Compute Optimally") is the foundational paper for test-time compute scaling. Its experimental design is elegant:

**Setup:** fix a base model (Llama-3-8B-Instruct), and on math problems of varying difficulty, compare two ways of improving performance:

- **Option A:** spend more inference compute — have the model generate N candidate solutions and use a verifier to pick the best one (best-of-N).
- **Option B:** spend more training compute — upgrade the base model to a larger one (more parameters).

**Core findings:**

1. **On easy problems**, the return on extra inference compute **exceeds** the return on extra training compute. An 8B model given enough inference compute can beat a 70B model given none.
2. **On hard problems**, the return on extra inference compute **diminishes** — the base model's capability ceiling caps what inference can achieve.
3. **The best inference strategy depends on problem difficulty**: best-of-N works for easy problems, sequential revision works for hard ones.

This finding has major engineering implications:

- **Inference compute is tunable** — you can decide how much to spend based on task difficulty, at runtime.
- **Training compute is fixed** — once training finishes, the parameters are locked in.

This is why a reasoning model's core advantage is flexibility in compute allocation: parameter count is locked in at training time, while inference compute can be dialed up or down per task.

## 10.2.3 Two Paradigms of Test-time Compute

Snell et al. group how test-time compute gets used into two categories:

### Parallel Sampling

Have the model independently generate N candidate solutions, then use a verifier to pick the best one. This is the idea behind best-of-N.

```python
# Parallel sampling sketch
candidates = [model.generate(prompt) for _ in range(N)]
scores = [verifier.score(prompt, c) for c in candidates]
best = candidates[argmax(scores)]
```

**Advantages:**

- Naturally parallel, so it's fast.
- Works well on easy problems — the larger N is, the higher the odds of hitting a correct solution.

**Drawbacks:**

- Weak on hard problems — if the base model's single-attempt success probability is below 1/N, N samples are still likely to all be wrong.
- Requires a verifier (the core topic of [Chapter 9's PRM section](../chapter18_grpo/grpo-family)).

### Sequential Revision

Have the model generate an initial solution, then produce revised versions based on that solution, iterating repeatedly.

```python
# Sequential revision sketch
solution = model.generate(prompt)
for _ in range(K):
    feedback = model.critique(prompt, solution)
    solution = model.revise(prompt, solution, feedback)
```

**Advantages:**

- Suited to hard problems — each revision is a chance to correct an error.
- No external verifier needed.

**Drawbacks:**

- Sequential, so it's slow.
- Revision can make things worse — the feedback itself can be wrong.

### Tree Search

A more elaborate approach is tree search: unroll the reasoning process into a tree, where each node is an intermediate reasoning step, and use a search algorithm (MCTS, beam search) to find the best path. This is the core content of [Chapter 9, PRM and Inference-Time Search](../chapter18_grpo/grpo-family) — we won't expand on it here.

## 10.2.4 Gemini 3 Pro Deep Think and the Flagship of Parallel Reasoning

In October 2025, Google released [Gemini 3 Pro Deep Think](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/), pushing test-time compute scaling to a new extreme. Deep Think's core idea is to **stack a "parallel reasoning layer" on top of an MoE model**.

Traditional reasoning models (o1, R1) think **serially** — token 1, then token 2, then token 3, each depending on the one before it. That serial structure caps reasoning speed at the speed of autoregressive generation.

Deep Think introduces **parallel reasoning**:

- Generate multiple independent reasoning paths at the same time.
- Aggregate information across paths (similar to an ensemble).
- Use a "coordinator" to decide when to stop and how to merge the paths.

This structure lets Deep Think **generate N times more reasoning tokens than a serial model in the same wall-clock time**, where N is the number of parallel paths.

### Deep Think's Benchmark Results

A few key numbers from the launch:

- **IMO 2025**: Gold medal, demonstrating math-reasoning ability at the level of top IMO competitors.
- **HLE (Humanity's Last Exam)**: 48.4%, far ahead of contemporaries GPT-5 (~30%) and Claude Opus 4.5 (~35%).
- **ARC-AGI-2**: 84.6%, a further jump past o3's 75%.
- **Codeforces rating**: above 3000 (top 0.01% of humans).

### The February 2026 3.1 Deep Think Upgrade

In February 2026, Google released Gemini 3.1 Pro Deep Think. The main improvements:

- **Dynamic parallel-path count**: the degree of parallelism adjusts automatically to problem difficulty — 4 paths for easy problems, 32 for hard ones.
- **Cross-path attention**: different reasoning paths can "see" each other's intermediate results, forming loose coordination.
- **Longer context**: expanded from 1M to 10M tokens, supporting extremely long chains of thought.

  3.1 Deep Think reaches 91.2% on ARC-AGI-2 and 52.7% on HLE — pushing the ceiling of test-time scaling higher once again.

## 10.2.5 The Economics of Inference Compute

Test-time compute scaling isn't free. Every doubling of inference compute means:

- **Latency doubles** — users wait longer.
- **API cost doubles** — for token-billed models, thinking tokens cost money too.
- **Energy use doubles** — energy costs rise at deployment scale.

That raises an engineering question: **when should reasoning be turned on, and when shouldn't it be?**

| Task type                                             | Recommended strategy                                     |
| ----------------------------------------------------- | -------------------------------------------------------- |
| Simple Q&A ("what's the weather today")               | No reasoning — answer directly                           |
| Medium difficulty ("write a sorting algorithm")       | Light reasoning, tens to hundreds of tokens              |
| Math competitions / code generation                   | Full reasoning, thousands to tens of thousands of tokens |
| Research-grade reasoning (OpenAI o1-pro / Deep Think) | Maximal reasoning, hundreds of thousands of tokens       |

This is the engineering motivation behind Hybrid Thinking (next section): **let the model decide for itself when deep reasoning is warranted**.

## 10.2.6 A Reflection: Does the Scaling Law Saturate?

Snell et al.'s experiments found that the return on test-time compute diminishes on hard problems. Later work — the [DeepSeek R1 paper](https://arxiv.org/abs/2501.12948) and the [Qwen3 technical report](https://arxiv.org/abs/2505.09388) — confirmed the pattern at larger scale:

- **Easy problems**: test-time compute has almost no ceiling — the model can keep checking and revising indefinitely.
- **Medium problems**: returns start diminishing past a certain point.
- **Hard problems**: test-time compute saturates quickly — the base model's limited capability is a hard constraint.

The deeper implication is that **test-time compute scaling can't substitute for training compute scaling without limit**. The two are complementary:

- Training compute sets the **capability ceiling**.
- Test-time compute determines **how close to that ceiling you get**.

No amount of test-time compute rescues a base model that isn't strong enough. That's why reasoning models like R1-Zero and o1 are all built on trillion-parameter-class pretrained bases: **scaling reasoning requires a strong base to scale from**.

## Summary

Test-time Compute Scaling is more than "throw more compute at inference." Its core insight is that **the optimal point for allocating compute shifts from the training stage to the inference stage**. Snell et al.'s work proved that shift pays off on math and reasoning tasks; Gemini Deep Think proved it can push past SOTA at industrial scale too.

But this shift raises new engineering problems: how do you control reasoning depth? How do you keep the model from thinking too long? How does the model learn on its own that "this problem doesn't need deep thought"? Those are the questions the next section, Hybrid Thinking and thinking budgets, sets out to solve.
