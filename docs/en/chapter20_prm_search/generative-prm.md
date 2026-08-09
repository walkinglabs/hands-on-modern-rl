# 18.3 The Generative PRM Approach

The previous section showed the core bottleneck of discriminative PRMs: **annotation cost that explodes with scale**. PRM800K took 12 person-years of labeling, and every new domain requires labeling from scratch. That makes industrial deployment of PRMs nearly impractical.

In April 2025, [ThinkPRM](https://arxiv.org/abs/2504.16828) proposed a fundamentally different idea: **have the LLM "critique" reasoning steps in natural language**, instead of training a classifier. This approach is called **generative PRM**.

ThinkPRM reported a striking result: **with 100x fewer labels than PRM800K, a generative PRM achieves better results on MATH**. How is this possible?

## 11.3.1 The Core Idea Behind Generative PRM

### Discriminative vs. Generative

Discriminative PRM ([OpenAI PRM800K](./discriminative-prm)):

```text
Input: prompt + step i of the reasoning
Output: good / bad / neutral (three-way classification)
```

Model architecture: encoder + classification head (the last layer is a softmax over three class probabilities).

Generative PRM (ThinkPRM):

```text
Input: prompt + step i of the reasoning + "please critique this step"
Output: a natural-language critique + a final verdict
```

Model architecture: a standard LLM (decoder-only), producing text.

### Why Is Generative More Efficient?

The generative PRM's advantage comes from two things.

**Advantage one: it reuses the LLM's pretrained knowledge**

A discriminative PRM trains a classifier from scratch — the classification head starts from random initialization and needs a large amount of labeled data to learn "what a good reasoning step looks like."

A generative PRM uses the LLM directly. During pretraining, the LLM has already seen a massive amount of math, code, and logical-reasoning text, so it already "knows" what good reasoning looks like. A generative PRM only needs a small amount of fine-tuning (or even zero-shot prompting) to work.

**Advantage two: the flexibility of natural-language reasoning**

A discriminative PRM can only output one of three labels — good / bad / neutral. This is a **hard constraint**. If a step is "logically correct but poorly phrased," or "uses an uncommon trick," a discriminative PRM struggles to express that.

A generative PRM can say: "This step uses factoring, and the logic is correct. But the right-hand side of the factorization `x² - 1 = (x-1)(x+1)` was written as `(x-1)(x-2)` — that's a typo. So this step contains an error." This kind of fine-grained critique is much more useful for the model to learn from.

## 11.3.2 ThinkPRM's Concrete Design

[ThinkPRM](https://arxiv.org/abs/2504.16828) (April 2025) is the flagship work in the generative PRM line. Its design has several key elements.

### Generating Step-by-Step Critiques

Instead of simply asking "is this step correct," ThinkPRM has the model generate a step-by-step critique:

```text
Prompt: Evaluate step 3 of the following reasoning:
Problem: Solve x² - 5x + 6 = 0
Reasoning:
  Step 1: This is a quadratic equation, so we can use the quadratic formula
  Step 2: Quadratic formula: x = (-b ± √(b²-4ac)) / 2a
  Step 3: Substituting a=1, b=-5, c=6: x = (5 ± √(25-24)) / 2 = (5 ± 1) / 2
  Step 4: x = 3 or x = 2
Please evaluate Step 3.

ThinkPRM output:
Let me check Step 3:
- a = 1, b = -5, c = 6 ✓ (matches the original equation)
- Quadratic formula: x = (-b ± √(b²-4ac)) / 2a ✓
- Substituting: x = (5 ± √((-5)² - 4·1·6)) / (2·1)
       = (5 ± √(25 - 24)) / 2
       = (5 ± √1) / 2
       = (5 ± 1) / 2 ✓
Step 3 is correct.
Verdict: CORRECT
```

### Step-by-Step Generation vs. All-at-Once Generation

ThinkPRM supports two inference modes.

**Mode one: step-by-step generation**

ThinkPRM is called separately for each step. N steps require N LLM calls. Slow but precise.

**Mode two: all-at-once generation**

The entire reasoning chain is handed to ThinkPRM in a single call, and it outputs a critique for every step. One LLM call. Fast, but it can miss details.

The ThinkPRM paper shows that step-by-step generation gives better results, but all-at-once generation is 5-10x faster. In practice, the choice is often made based on task difficulty.

### A Remarkably Small Training Set

ThinkPRM's training only needs **fewer than 8K** step-level critiques (compared to PRM800K's 800K). How is this possible?

The key insight: **the LLM already knows how to critique reasoning — it just needs a small number of examples to "activate" that ability**.

ThinkPRM uses three kinds of data:

1. **Seed examples** (~1K): high-quality step critiques that show the model what a critique should look like
2. **Automatically generated data** (~5K): additional critiques generated with GPT-4 / Claude, which come with some noise
3. **Rejection sampling** (~2K): high-quality samples filtered out of the automatically generated data

That's fewer than 8K labels in total — 100x fewer than PRM800K.

## 11.3.3 Generative PRM Experimental Results

ThinkPRM's performance across several benchmarks:

### The MATH Dataset (PRM800K Test Set)

| Verifier                    | Labels used           | Best-of-64 accuracy |
| --------------------------- | --------------------- | ------------------- |
| ORM                         | 0 (final answer only) | 53.9%               |
| OpenAI PRM (discriminative) | 800K                  | 78.2%               |
| **ThinkPRM (generative)**   | **<8K**               | **79.4%**           |

This is a striking result — **with 1% of the labels, ThinkPRM beats the discriminative PRM**.

### Qwen32B-MATH (a Larger Base Model)

On a larger base model, ThinkPRM's advantage grows:

| Verifier           | Best-of-64 |
| ------------------ | ---------- |
| ORM                | 65.3%      |
| Discriminative PRM | 81.2%      |
| **ThinkPRM**       | **84.7%**  |

### Cross-Domain Generalization

ThinkPRM is trained on math tasks and then applied to code generation:

| Verifier                             | Code generation accuracy |
| ------------------------------------ | ------------------------ |
| Discriminative PRM (trained on math) | 32.1% (a clear drop)     |
| **ThinkPRM (trained on math)**       | **54.6% (stays high)**   |

This confirms ThinkPRM's **cross-domain generalization**: a natural-language critique generalizes better than a classification label.

## 11.3.4 Verifier Compute Scaling

ThinkPRM also reports an important finding: **the verifier itself has test-time compute scaling**.

The method: call ThinkPRM multiple times on the same reasoning step (each time with random sampling), and aggregate the results with majority vote.

| Verifier calls | Accuracy |
| -------------- | -------- |
| 1              | 79.4%    |
| 4              | 81.2%    |
| 16             | 82.5%    |
| 64             | 83.1%    |

This shows that **a generative PRM is also test-time compute scalable** — spending more compute at inference time keeps improving accuracy.

The engineering payoff of this finding:

- Training-time compute can be reduced (fewer labels needed)
- Inference-time compute can be increased (more sampling)
- Total compute allocation becomes more flexible

## 11.3.5 Generative vs. Discriminative: A System Comparison

| Dimension                   | Discriminative PRM                        | Generative PRM                      |
| --------------------------- | ----------------------------------------- | ----------------------------------- |
| Model                       | Classifier (encoder + head)               | LLM (decoder-only)                  |
| Output                      | Discrete label (good/bad/neutral)         | Natural-language critique + verdict |
| Labeling requirement        | High (PRM800K scale)                      | Low (<8K seed examples)             |
| Cross-domain generalization | Weak                                      | Strong                              |
| Interpretability            | Weak (label only)                         | Strong (natural-language rationale) |
| Inference speed             | Fast (single forward pass)                | Slow (generates text)               |
| Test-time scaling           | Not applicable                            | Available (multiple samples)        |
| Training cost               | High (learns classification from scratch) | Low (fine-tuning)                   |
| Inference cost              | Low (small model)                         | High (large model)                  |

### When to Use Which

**Discriminative PRM is a good fit when:**

- You're doing large-scale RL training (needs fast PRM calls)
- The task is single-domain (math, code) and doesn't need to generalize
- Compute is plentiful and can absorb the labeling cost

**Generative PRM is a good fit when:**

- The application spans multiple domains (needs generalization)
- Compute for labeling is limited
- Interpretability matters (research, education)
- Test-time compute scaling is available (inference compute is plentiful)

## 11.3.6 Generative PRM in Industry

By mid-2026, generative PRMs have become the mainstream approach in industry:

- **OpenAI o1/o3** (speculated): the internal verifier is likely generative
- **DeepSeek V3.2 Speciale**: its self-verification RLVR approach aligns with generative PRM
- **Anthropic Claude**: internal verifier (speculated)
- **Alibaba Qwen3-Thinking**: uses an LLM-as-Judge approach
- **Moonshot Kimi K2**: self-verification + best-of-N

Industry trends:

1. **Mixing PRM and ORM**: use PRM to evaluate the process and ORM to evaluate the outcome, then combine them with a weighted sum
2. **Mixing generative and discriminative**: generative PRM provides high-quality critiques, which are distilled into a discriminative PRM for large-scale training
3. **Self-PRM**: have the model evaluate its own reasoning (consistent with the [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) approach)

## 11.3.7 A Reflection: PRM or RLVR?

A PRM's core value is providing dense process signal. But [DeepSeek-R1](https://arxiv.org/abs/2501.12948) reported a counterintuitive phenomenon: **pure outcome reward (RLVR) can also teach a model to self-verify**.

R1-Zero's training uses only a 0/1 outcome reward, yet late in training the model spontaneously develops self-verification behavior — it checks its own steps during reasoning and backtracks when it finds an error. This self-verification behavior isn't trained by a PRM; it emerges naturally under outcome reward.

This finding raises a deep question: **is a PRM necessary, or can outcome reward implicitly learn process signal at sufficient scale?**

The current consensus:

- **Small-scale training** (<100B parameters × training tokens): PRM clearly outperforms ORM
- **Large-scale training** (R1-Zero scale): ORM can also give rise to process awareness, but PRM still accelerates convergence
- **Cross-domain applications**: ORM's emergent behavior isn't guaranteed, so PRM is more reliable

So PRM isn't strictly "necessary," but **for applications with limited resources or that need fast convergence, PRM remains irreplaceable**.

## Summary

By having the LLM critique reasoning steps in natural language, generative PRM matches discriminative PRM's performance with 1% of the labels, and it has a clear edge in cross-domain generalization and interpretability.

ThinkPRM is the flagship work in this line, and it opened a new direction for PRM research: **replacing human annotation with the LLM's own intrinsic reasoning ability**. This is consistent with the broader trend in the LLM era that generation outperforms discrimination.

Generative PRM still has one fundamental limitation: **it depends on the LLM's own judgment** — if the LLM itself is weak at a certain kind of reasoning, the PRM will make mistakes too. The next section covers formal PRM, which introduces an **external verifier** (Lean4) to solve this problem at the root.
