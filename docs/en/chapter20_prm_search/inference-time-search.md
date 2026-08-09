# 18.5 Inference-time Search

The previous three sections covered how to train a PRM — discriminative, generative, formal. This section takes a different angle: **how do you use a PRM at inference time?**

The most direct application is [Best-of-N + Re-ranking](./discriminative-prm) (covered in Section 11.2) — generate N candidates, use the PRM to pick the best one. But Best-of-N is **memoryless parallel sampling**: it never touches the intermediate states of the reasoning process.

**Inference-time search** is a more structured approach — unroll the reasoning process into a **Thought Tree**, score every node with the PRM, and use a search algorithm (DFS, BFS, beam search, MCTS) to find the best path.

This section covers the main inference-time search methods.

## 11.5.1 Why Do We Need Search?

Consider a math problem:

```text
Solve x² + 5x + 6 = 0
```

A model can generate several different reasoning paths:

```text
Path A: quadratic formula
  → x = (-5 ± √(25-24)) / 2 = (-5 ± 1) / 2
  → x = -2 or x = -3

Path B: factoring
  → x² + 5x + 6 = (x+2)(x+3) = 0
  → x = -2 or x = -3

Path C: completing the square
  → x² + 5x = -6
  → x² + 5x + 25/4 = 25/4 - 6 = 1/4
  → (x + 5/2)² = 1/4
  → x + 5/2 = ±1/2
  → x = -2 or x = -3
```

All three paths reach the correct answer. But if the model makes a mistake somewhere on one path — say it botches the square root in Path A — a single sample along that path gives the wrong answer.

Best-of-N addresses this: generate several independent paths and let the PRM pick the best one. But Best-of-N has real limits:

- **It ignores similarity between paths.** If two paths share an identical first half, Best-of-N regenerates that shared prefix from scratch, twice.
- **It cannot backtrack mid-path.** If a path goes wrong halfway through, Best-of-N's only recourse is to start over.
- **It searches inefficiently.** N independent samples amount to brute-force enumeration.

**Inference-time search** fixes these problems with a structured search tree:

- **Shared prefixes**: identical reasoning prefixes are computed only once
- **Intermediate evaluation**: the PRM scores intermediate states and decides whether to continue or backtrack
- **Resource allocation**: search compute goes toward the most promising directions

## 11.5.2 Beam Search over Thoughts

**Beam search** is the simplest search method — it maintains K best "partial reasoning" candidates (the beam), expands every beam at each step, scores the expansions with the PRM, and keeps the K best.

### Algorithm

```python
def beam_search_thoughts(prompt, model, prm, K=4, max_steps=10):
    # initial beam: a single empty state
    beams = [{"thought": "", "score": 1.0}]

    for step in range(max_steps):
        # expand each beam: have the model generate the next reasoning step
        candidates = []
        for beam in beams:
            for _ in range(N_expansions):
                next_thought = model.generate_next(prompt, beam["thought"])
                score = prm.score(prompt, beam["thought"] + next_thought)
                candidates.append({
                    "thought": beam["thought"] + next_thought,
                    "score": score
                })

        # keep the top-K as the new beams
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

        # stop once a complete answer is found
        if any(is_complete(b["thought"]) for b in beams):
            break

    return beams[0]["thought"]  # return the best beam
```

### Characteristics of Beam Search

**Advantages**:

- Simple to implement
- Fast — K beams expand in parallel
- Well suited to wide search spaces

**Drawbacks**:

- K is fixed — wasted compute on easy problems, not enough on hard ones
- No backtracking — once a beam is eliminated, it's gone for good
- Prone to getting stuck in local optima

### When to Use It

Beam search fits well when:

- The reasoning space is wide (many valid solution methods)
- Single steps are easy to score
- The task is easy to moderately difficult

## 11.5.3 Tree of Thoughts (ToT)

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al., 2023) extends beam search — it adds **branching, backtracking, and a mix of DFS/BFS**.

### The Core Structure of ToT

```text
                Root
              /      \
            A1        A2
           /  \      /  \
         B1   B2   B3   B4
        / \    |    |   / \
       C1  C2  C3   C4 C5  C6

       Search algorithm: BFS (breadth-first) or DFS (depth-first)
       Evaluation: PRM scores every step
       Backtracking: low-scoring nodes get pruned
```

### The ToT Algorithm

```python
def tree_of_thoughts(prompt, model, prm, max_depth=10, breadth=4):
    # DFS starting from the root
    def dfs(thought, depth):
        if depth >= max_depth:
            return [{"thought": thought, "score": prm.score(prompt, thought)}]

        # generate N candidate next steps
        candidates = []
        for _ in range(breadth):
            next_thought = model.generate_next(prompt, thought)
            full_thought = thought + next_thought
            score = prm.score(prompt, full_thought)
            candidates.append({"thought": full_thought, "score": score})

        # sort by score, prune the low scorers
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:breadth // 2]  # prune half

        # recurse on the surviving candidates
        results = []
        for c in candidates:
            results.extend(dfs(c["thought"], depth + 1))

        return results

    return dfs("", 0)
```

### Characteristics of ToT

**Advantages**:

- Supports backtracking and pruning
- Handles deep reasoning tasks
- More efficient than Best-of-N

**Drawbacks**:

- Slow — the tree expands exponentially
- Requires many PRM evaluations
- A poor fit for very long CoT tasks

### Experimental Results for ToT

On the [Game of 24](https://arxiv.org/abs/2305.10601) task:

| Method                                 | Success Rate |
| -------------------------------------- | ------------ |
| Greedy decoding                        | 7.3%         |
| CoT prompting                          | 4.0%         |
| Self-consistency (multi-sample + vote) | 9.0%         |
| **Tree of Thoughts**                   | **74.0%**    |

This is a massive gain — the same base GPT-4, with nothing changed but the inference-time search strategy, goes from a 7% success rate to 74%.

## 11.5.4 MCTS over Thoughts

**Monte Carlo Tree Search (MCTS)** is the algorithm behind AlphaGo. Applied to LLM reasoning, the core idea is:

- Use the PRM as the value function (to score nodes)
- Use the model as the policy (to propose the next step)
- Use the UCB formula to balance exploration and exploitation

### The Four Steps of MCTS

Each iteration runs:

1. **Selection**: starting from the root, use the UCB formula to pick child nodes until reaching a leaf
2. **Expansion**: generate N child nodes at the leaf
3. **Simulation**: run a rollout from the child nodes (quickly generate a full reasoning trace)
4. **Backpropagation**: propagate the rollout's reward back up to all ancestor nodes

### The UCB Formula

UCB (Upper Confidence Bound) balances exploration and exploitation:

$$\text{UCB}(n) = Q(n) + c \cdot \sqrt{\frac{\ln N(p)}{N(n)}}$$

where:

- $Q(n)$: the average reward of node $n$ (from the PRM)
- $N(n)$: the number of times node $n$ has been visited
- $N(p)$: the number of times the parent node has been visited
- $c$: the exploration constant

The intuition: the first term is "known value" (exploitation), the second term is "unexplored potential" (exploration).

### Characteristics of MCTS

**Advantages**:

- Adaptive — it spends more effort exploring promising directions
- Comes with theoretical guarantees (converges to the optimum)
- Handles deep search well

**Drawbacks**:

- Complex to implement
- Expensive — many rollouts per step
- Sensitive to PRM quality

### Representative Work

- **rStar** ([arXiv:2408.06195](https://arxiv.org/abs/2408.06195)): MCTS + self-play, applied to math reasoning
- **AlphaProof** ([DeepMind 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)): MCTS + Lean4 verifier
- **RAP** ([Reasoning via Planning](https://arxiv.org/abs/2305.14992)): MCTS with the LLM acting as a world model

## 11.5.5 AlphaCodium and Search for Code Generation

[AlphaCodium](https://arxiv.org/abs/2401.08500) (January 2024) is a search method purpose-built for code generation. Its core idea:

- "Correctness" for a coding task can be verified automatically with **unit tests** (much like Lean4 for math)
- Use iterative search: generate → test → fix → test again

### The AlphaCodium Pipeline

```text
1. Problem understanding: have the LLM extract key information and generate test cases
2. Initial solution: generate a candidate solution
3. Iterative repair:
   a. Run the test cases
   b. If a test fails, analyze the error
   c. Have the LLM fix the error
   d. Repeat until all tests pass
4. Output the final solution
```

### Characteristics of AlphaCodium

- No PRM needed — the unit tests are the verifier
- Iterative, not tree search — simple and efficient
- Improves over single-shot generation by 30%+ on Codeforces

## 11.5.6 The Compute Cost of Inference-time Search

The compute cost of different search methods, measured in "number of model forward passes":

| Method                  | Forward Passes (typical) |
| ----------------------- | ------------------------ |
| Greedy decoding         | 1                        |
| Best-of-N               | N (usually 4–64)         |
| Beam search (K, D)      | K × D                    |
| Tree of Thoughts (B, D) | O(B^D) (exponential)     |
| MCTS (N_iter, N_expand) | N_iter × N_expand        |

**Tree of Thoughts and MCTS both scale exponentially in compute.** That's why they haven't caught on in industrial deployment the way Best-of-N has.

But on tasks where "correct is correct" — scientific computing, formal proof, competitive programming — the compute cost of search is worth paying, because these tasks demand extremely high correctness.

## 11.5.7 Search at Training Time vs. Inference Time

A deeper question follows: **should search happen at training time, or at inference time?**

**Search at training time** (as in AlphaProof's self-play):

- The results of search become training data
- The trained model "internalizes" the search capability
- No search is needed at inference time

**Search at inference time** (as in ToT, MCTS):

- No search during training — just ordinary RL
- Search is used at inference time to boost performance
- High compute cost, but flexible

**Industrial practice** is usually a hybrid:

- Light search during training (to speed up convergence)
- Search at inference time only when the task's difficulty warrants it

This lines up with the thinking behind [Chapter 8, Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) — where to spend compute is an engineering trade-off.

## Summary

Inference-time search is the other side of the PRM coin — not just providing dense reward during training, but also guiding search at inference time.

The main methods:

- **Beam search**: simple and parallel, fits medium-difficulty tasks
- **Tree of Thoughts**: supports backtracking and pruning, fits complex tasks
- **MCTS**: adaptive exploration, with theoretical guarantees
- **AlphaCodium**: code-specific, uses unit tests as the verifier

Search compute scales exponentially, which is why **Best-of-N remains the industrial mainstream**, and search is reserved for high-value tasks in research, formal verification, and competitive settings.

The next section covers PaCoRe — a new reasoning paradigm that turns "deep search" into "wide parallelism," balancing compute against quality.
