# 29.4 Evolutionary LLM Search and Scientific Discovery

The [previous sections](../) covered RL's traditional frontiers — embodied intelligence, self-play, multi-agent systems. This section turns to a different direction altogether: **LLM-driven scientific discovery**.

Three things define this direction:

- **The LLM takes on more than the actor role** — it also works as idea generator, code writer, and experiment designer
- **RL takes on more than policy optimization** — it drives search, evolution, self-improvement
- **The target is not gameplay or dialogue** — it's discovering new algorithms, new mathematics, new science

A few representative works from 2024–2026:

- **AlphaEvolve** (DeepMind, May 2024): LLM + evolutionary algorithms discover new mathematics
- **Genie 3** (DeepMind, August 2025): a generative world model
- **Titans** (Google, December 2024): a long-term memory architecture
- **Multi-Agent Deep Research** (ByteDance Seed, November 2025): trains multi-agent search systems with M-GRPO

Together these works represent the "next-generation paradigm" for combining RL with LLMs — the object of training shifts from "a policy" to "a research system."

## 29.4.1 AlphaEvolve and Mathematical Discovery via LLM + Evolution

[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind, announced May 2024, full version 2025) is the flagship case of LLM-driven scientific discovery.

### The Core Idea Behind AlphaEvolve

It models mathematical discovery as **evolutionary search plus LLM code generation**:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Population initialization: existing algorithms/proofs    │
│    (e.g., Strassen's algorithm for matrix multiplication)   │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM mutation: have Gemini generate branches               │
│    "Improve this algorithm" / "Try a different approach"    │
│    Output: new code                                         │
├─────────────────────────────────────────────────────────────┤
│ 3. Automatic evaluation: run the code, measure its effect   │
│    (e.g., number of multiplications used)                   │
├─────────────────────────────────────────────────────────────┤
│ 4. Selection: keep what performs well, discard what doesn't │
├─────────────────────────────────────────────────────────────┤
│ 5. Iterate: return to step 2                                │
└─────────────────────────────────────────────────────────────┘
```

This loop is almost identical to a [classic genetic algorithm (GA)](../chapter03_mdp/dp-mc-td) — the only difference is that the mutation operator changes from "random tweaks" to "intelligent LLM generation."

### AlphaEvolve's Key Innovations

**Innovation 1: the LLM as an intelligent mutation operator**

Mutation in a traditional GA is a random edit — its success rate is low. Mutation by an LLM means "understand the current code and propose a meaningful improvement" — its success rate is much higher.

**Innovation 2: code as the gene**

Instead of using a bit string as the gene, AlphaEvolve uses **executable code**. This makes fitness **automatically measurable** — running the code tells you how well it performs.

**Innovation 3: Gemini as the LLM backend**

AlphaEvolve runs Gemini Pro/Ultra as its LLM backend — a strong LLM substantially raises the quality of each mutation.

### What AlphaEvolve Discovered

AlphaEvolve produced **genuinely new discoveries** across several domains.

**Discovery 1: a new matrix multiplication algorithm**

In 1969, Strassen showed that $4 \times 4$ matrix multiplication could be done with 49 multiplications (previously thought to require 64). AlphaEvolve found a new algorithm that uses only **48** — improving on more than 50 years of human research.

**Discovery 2: new bounds in combinatorics**

On problems like [tensor decomposition](https://en.wikipedia.org/wiki/Tensor_decomposition) and [sorting networks](https://en.wikipedia.org/wiki/Sorting_network), AlphaEvolve found several new bounds that beat the previously best-known results.

**Discovery 3: optimizing Google's own infrastructure**

DeepMind used AlphaEvolve internally to optimize:

- Data-center scheduling algorithms (saving 0.7% of global compute resources)
- TPU matrix-multiplication hardware design
- Machine learning kernel optimization

### What AlphaEvolve Demonstrates

AlphaEvolve makes three points:

1. **LLMs can do genuine scientific research** — not just "answer questions," but "discover new knowledge"
2. **Evolution + LLM is a powerful combination** — the LLM supplies intelligence, evolution supplies exploration
3. **Automatic evaluation is the key enabler** — this paradigm only works in domains where results can be evaluated automatically

## 29.4.2 Genie 3 and Generative World Models

[Genie 3](https://deepmind.google/models/genie/) (DeepMind, August 2025) is the flagship example of a generative world model.

### What Is a World Model?

A world model is a model that **predicts environment dynamics**:

```text
Input: current state s_t + action a_t
Output: next state s_{t+1}
```

In RL, a world model can **substitute for the real environment** — the policy trains inside the world model, avoiding costly real-environment interaction.

### The Evolution of the Genie Series

**Genie 1** (February 2024): learning a world model from video

- Input: internet video
- Output: a controllable "game" environment it can generate
- Key point: there are no explicit action labels — the model learns on its own "what an action is"

**Genie 2** (December 2024): a 3D world model

- Input: a single image
- Output: an interactive 3D environment it can generate
- Key point: the environment stays consistent for several minutes

**Genie 3** (August 2025): large-scale, controllable, long-horizon

- Input: a natural-language description
- Output: a fully controllable, long-horizon 3D environment
- Key point: usable for training embodied agents

### How Genie 3 Is Trained

```text
┌───────────────────────────────────────────────────┐
│ Phase 1: Video pretraining                        │
│   - Large amounts of unlabeled video               │
│   - Learn "how the world works"                   │
├───────────────────────────────────────────────────┤
│ Phase 2: Action labeling                           │
│   - Have an LLM label the actions in the video     │
│   - Learn "which action causes which change"       │
├───────────────────────────────────────────────────┤
│ Phase 3: World model training                      │
│   - (s_t, a_t, s_{t+1}) triples                    │
│   - Train a model that predicts s_{t+1}            │
├───────────────────────────────────────────────────┤
│ Phase 4: RL training                                │
│   - The policy trains inside the world model        │
│   - Avoids expensive real-environment interaction   │
└───────────────────────────────────────────────────┘
```

### Applications of Genie 3

**Application 1: training embodied agents**

A robot learns to walk, grasp, and manipulate objects inside the world model — avoiding trial and error on a real robot, which is expensive and dangerous.

**Application 2: game generation**

Genie 3 can generate playable games automatically — a player describes the game they want, and Genie 3 generates the complete environment.

**Application 3: simulation-based training**

High-risk domains such as autonomous driving, industrial control, and medical surgery — train inside the world model, then deploy to the real environment.

### Genie 3's Limitations

- **Accuracy**: the world model isn't 100% accurate — long-horizon predictions drift
- **Generalization**: environments outside the training distribution are hard to simulate
- **Compute cost**: inference with a high-quality world model is expensive

## 29.4.3 Titans and Long-Term Memory Architectures

[Titans](https://arxiv.org/abs/2501.00663) (Google, published December 2024, revised 2025) opens up a new direction in LLM architecture: **long-term memory**.

### The Motivation Behind Titans

The Transformer has a fundamental limitation: the **context window**. Even extended to 1M tokens, it still cannot handle "infinitely long" input. Titans tries to solve this problem.

### Titans's Design

Titans introduces **neural long-term memory**:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Short-term memory: attention (standard Transformer)                        │
│   - Processes the most recent tokens                                       │
│   - Limited capacity (context window)                                      │
├────────────────────────────────────────────────────────────────────────────┤
│ Long-term memory: neural memory module (new)                               │
│   - Continuously learns and stores                                         │
│   - Unlimited capacity                                                     │
├────────────────────────────────────────────────────────────────────────────┤
│ Persistent memory: task-relevant knowledge (system prompt, knowledge base) │
│   - Fixed, does not change                                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

The three memory tiers together let Titans handle **arbitrarily long input** — the long-term memory keeps accumulating historical information as the sequence grows.

### How Titans Relates to RL

At its core, Titans is about **learning how to remember** — and that is itself an RL problem:

- State: the current input plus the current memory
- Action: how to update the memory (write / forget / update)
- Reward: future task performance (if the memory stores useful information, task performance improves)

Titans uses **surprise** as an internal reward signal — memory strengthens when the input is "surprising" and weakens when the input is "repetitive." This is a form of **self-supervised RL**: the model generates its own reward.

### Titans's Experimental Results

On long-horizon tasks, Titans substantially outperforms the Transformer:

| Task                            | Transformer | Titans                     |
| ------------------------------- | ----------- | -------------------------- |
| Language modeling (10M context) | OOM         | 67% perplexity improvement |
| Long-document QA                | 55%         | 78%                        |
| Time-series forecasting         | 65%         | 82%                        |

Titans's results argue that **long-term memory is the next axis of scaling** — not just "wider and deeper," but also "able to remember more."

## 29.4.4 M-GRPO and Multi-Agent Search Training

[Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO](https://arxiv.org/abs/2511.13288) (ByteDance Seed, November 2025) trains a multi-agent search system with M-GRPO — a multi-agent extension of Group Relative Policy Optimization.

### The System Design Behind M-GRPO

The multi-agent system consists of a main agent and several sub-agents:

```text
┌────────────────────────────────────────────────────────────┐
│ Main agent (planner): overall planning                     │
│   - Receives the task                                      │
│   - Decomposes it into subtasks                             │
│   - Dispatches sub-agents                                  │
├────────────────────────────────────────────────────────────┤
│ Sub-agents (tool executors): tool execution                │
│   - Call search, code, and other tools over multiple turns │
│   - Call frequency and count vary by sub-agent             │
├────────────────────────────────────────────────────────────┤
│ Hierarchical credit assignment                              │
│   - Main agent and sub-agents each compute group-relative  │
│     advantages                                              │
│   - Exchange minimal statistics through a shared store     │
└────────────────────────────────────────────────────────────┘
```

### How M-GRPO Trains

M-GRPO addresses three difficulties specific to multi-agent RL training:

- **Hierarchical credit assignment**: the main agent and the sub-agents each compute their own group-relative advantage, avoiding "credit confusion" between levels
- **Trajectory alignment**: because the number of sub-agent calls varies, a trajectory-alignment scheme produces fixed-size batches
- **Decoupled training**: agents run on separate servers and exchange statistics through a shared store, so no gradients need to cross servers

On benchmarks such as GAIA, XBench-DeepSearch, and WebWalkerQA, M-GRPO consistently outperforms both single-agent GRPO and multi-agent GRPO with frozen sub-agents.

### How M-GRPO Relates to AlphaEvolve

Both combine LLMs with RL or search, but they approach the problem from different angles:

- **AlphaEvolve**: evolutionary search (gradient-free, population-based), aimed at algorithm discovery
- **M-GRPO**: multi-agent RL (built on GRPO), aimed at tool-augmented deep research

They are two complementary paradigms within LLM-driven discovery.

## 29.4.5 Recursive Self-Improvement

**Recursive Self-Improvement (RSI)** is the ultimate form of LLM-driven discovery — **the model improves itself**.

### The Core Loop of RSI

```text
┌─────────────────────────────────────────────────────────────────┐
│ 1. Current model M_t evaluates its own capabilities             │
│    - Which tasks does it do well on? Which tasks is it weak on? │
├─────────────────────────────────────────────────────────────────┤
│ 2. Generate an improvement plan                                 │
│    - Design new training data                                   │
│    - Adjust training hyperparameters                            │
│    - Improve the algorithm                                      │
├─────────────────────────────────────────────────────────────────┤
│ 3. Execute the improvement                                      │
│    - Train a new model M_{t+1} using the plan                   │
├─────────────────────────────────────────────────────────────────┤
│ 4. Evaluate the new model                                       │
│    - Is M_{t+1} better than M_t?                                │
│    - If better, keep M_{t+1}; if worse, roll back                │
├─────────────────────────────────────────────────────────────────┤
│ 5. Return to step 1                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Where RSI Stands Today

As of mid-2026, RSI is still a **research concept**, with no industrial-grade implementation. Three reasons why.

**Challenge 1: self-evaluation is unreliable**

A model has a hard time accurately judging its own capabilities — it tends to overestimate, a Dunning-Kruger effect for machines.

**Challenge 2: the search space of improvement plans explodes**

The combinations of possible training data, hyperparameters, and algorithms are astronomical in number.

**Challenge 3: safety risk**

If a model can improve itself without limit, it could escape human control — this is the central concern in [AI safety](../chapter30_alignment_failures/intro).

### Partial Implementations of RSI

Full RSI doesn't exist yet, but several **partial implementations** point in that direction:

- **AutoGPT** (2023): an early attempt, with limited results
- **SRPO** ([arXiv:2406.01660](https://arxiv.org/abs/2406.01660)): trains a preference model through a self-improvement process (Cohere, 2024)
- **Voyager** ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)): a Minecraft agent that autonomously learns new skills
- **DeepMind's self-play systems**: models that improve by playing against themselves

These are early forms of RSI — they prove partial feasibility, but remain far from true "recursive self-improvement."

## 29.4.6 The Shared Pattern Across LLM-Driven Discovery

AlphaEvolve, Genie 3, Titans, MIRAS, and RSI share a common pattern.

### The LLM as an Intelligent Guide for Search

Traditional search methods (MCTS, beam search) need a hand-designed heuristic. An LLM can **generate the heuristic automatically** — making the search smarter.

### Automatic Evaluation Is the Key Enabler

AlphaEvolve can discover new algorithms precisely because **an algorithm's effect can be measured automatically** — running the code tells you the answer. This is the precondition for LLM-driven discovery: **only domains that support automatic evaluation are a fit**.

### Combination Beats Any Single Method

- AlphaEvolve = LLM + evolution
- Genie 3 = LLM + world model
- Titans = LLM + long-term memory
- Multi-Agent Deep Research = LLM + multi-agent + RL

**Combining multiple methods** outperforms any single method alone — this is what RL looks like in the LLM era.

### From "Training a Policy" to "Training a System"

Traditional RL trains a single policy. LLM-driven discovery trains a **complete research system** — multiple agents plus memory plus search plus tools, working together.

## 29.4.7 Future Directions

### Scientific Discovery

Extending the AlphaEvolve approach to:

- **Biology**: protein design, drug discovery
- **Chemistry**: new molecular synthesis routes
- **Physics**: new experiment design, new theory validation

### Education

Using LLM-driven discovery to personalize education — finding the learning path best suited to each individual student.

### AGI

Recursive self-improvement is one candidate path to AGI — if a model can keep improving itself, it might at some point surpass human capability.

This comes with **serious safety risk**, which is also the central topic of [alignment research](../chapter30_alignment_failures/scaling-and-defenses).

## Summary

LLM-driven discovery is RL's new frontier for 2025–2026:

- **AlphaEvolve**: LLM + evolution, discovering new mathematics
- **Genie 3**: a generative world model, used for embodied agents
- **Titans**: a long-term memory architecture, extending context
- **M-GRPO**: multi-agent RL training
- **RSI**: recursive self-improvement (partially realized)

Together, these works point to a new paradigm — training a research system, not just a policy. This is the next-generation direction where RL and LLMs fuse most deeply, and it sits at the very frontier of AGI research.
