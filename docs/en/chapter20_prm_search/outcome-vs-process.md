# 18.1 Outcome vs Process Reward

This section starts from the most basic question: **why isn't outcome reward enough for long-CoT tasks, and why do we need process reward?**

## 11.1.1 The Sparse Reward Problem

Consider a concrete example: asking a model to prove that "√2 is irrational."

The model's generated CoT looks like this (simplified):

```text
Step 1: Assume √2 = p/q, where p and q are coprime
Step 2: Then 2 = p²/q², i.e., p² = 2q²
Step 3: So p² is even
Step 4: So p is even (this step uses the contrapositive of "the square of an even number is even")
Step 5: Let p = 2k
Step 6: Substitute: 4k² = 2q², i.e., 2k² = q²
Step 7: So q² is even, and q is also even
Step 8: This contradicts p, q being coprime
Step 9: So √2 is irrational  ✓
```

Suppose this proof goes wrong at Step 6 — say it writes "4k² = 2q², i.e., 4k = q²" (dropping the square). The final conclusion, "√2 is irrational," is still correct, but the reasoning that got there has a mistake.

**Outcome reward** scores this response as follows:

- If the final answer (√2 is irrational) is used as the pass/fail criterion → correct → reward = 1
- But the reasoning actually went wrong, and the model should have learned "Step 6 is wrong"

**The problems with outcome reward**:

1. **Sparse signal**: a 10,000-token reasoning chain gets exactly 1 reward signal
2. **Wrong attribution**: the model has no idea which step was wrong, so it can't correct precisely
3. **Mislabeled reward**: reasoning went wrong but the answer happened to be right anyway → positive feedback that reinforces flawed reasoning
4. **Inefficient learning**: the model can only infer which steps matter by backing out from the overall reward — extremely inefficient

This is the **sparse reward problem** — the reward signal is spread too thin across time to give an effective learning signal.

## 11.1.2 The Credit Assignment Problem

The sparse reward problem has a more formal name in RL: the **credit assignment problem**.

The precise statement: given a sequential decision task where the final reward is $r_T$, how do you distribute that reward back across every step in the sequence, $a_1, a_2, \ldots, a_T$? Which steps should be reinforced, and which should be suppressed?

Classic RL has a few ways of addressing this.

### Discounted Return

Discount future rewards back to the present:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{T-t} r_T$$

This is the classic approach discussed in [Chapter 3, MDPs](../chapter03_mdp/value-bellman). It carries an implicit assumption: **the farther a reward is from the current step, the less it should influence the current decision**. This assumption holds in physical control tasks (when pushing a cart, the reward 10 steps from now really does matter less to the current push), but it doesn't hold in LLM reasoning — the first step of a math proof matters just as much as the tenth.

### GAE (Generalized Advantage Estimation)

[Chapter 5's GAE in PPO](../chapter10_ppo/gae-reward-model) trades off bias against variance by introducing a $\lambda$ parameter. GAE is PPO's standard approach, but its fundamental limitation is that **it needs a value function** — and GRPO deliberately omits the value function.

### Token-Level Loss

[DAPO](../chapter18_grpo/deepseek-dapo)'s token-level loss is one way to approximate PRM — instead of scoring the "entire reasoning chain" as a unit, it scores "each token." But token-level loss still relies on backpropagating outcome reward — it has no independent verifier that evaluates "is this token good?"

### PRM (Process Reward Model)

This is the protagonist of this chapter: **training an independent verifier that scores every step of the reasoning**. A PRM's output is dense — every step gets a score, so the model can know precisely which step was good and which was bad.

## 11.1.3 A Formal Comparison of Outcome Reward and Process Reward

Let's formalize the difference between the two.

### Outcome Reward Model (ORM)

An ORM takes a prompt $q$ and a complete response $o$, and outputs a scalar score:

$$\text{ORM}(q, o) \in \mathbb{R}$$

This score represents "how good the response is overall." In math tasks, it's usually 0 or 1 (wrong or right).

ORM training data takes the form:

```text
(prompt, response, final_correctness)
```

Example: ("Prove √2 is irrational", "`<full proof>`", 1)

### Process Reward Model (PRM)

A PRM takes a prompt $q$, a response $o$, and a step position $i$ within the response, and outputs a score for that step:

$$\text{PRM}(q, o, i) \in \mathbb{R}$$

This score represents "how good step $i$ of the reasoning is." In math tasks, it can be:

- Binary: 1 (correct) / 0 (incorrect) / -1 (irrelevant)
- Continuous: a probability in [0, 1]

PRM training data takes the form:

```text
(prompt, response, step_index, step_correctness)
```

Example: ("Prove √2 is irrational", "`<full proof>`", 4, 1) # Step 4 is correct

### Using Them in RL Training

When ORM is used for RL training:

$$r_{\text{ORM}} = \text{ORM}(q, o)$$

The entire sequence shares one reward.

When PRM is used for RL training (a common approach):

$$r_t = \text{PRM}(q, o, \text{step}(t))$$

Each token $t$ receives the PRM score of the reasoning step it belongs to. Tokens within the same reasoning step share that step's score.

This turns a sparse reward into a dense reward — every token now has a clear training signal.

## 11.1.4 Why PRM Is Irreplaceable for Long-CoT Tasks

PRM's value is most obvious in long-CoT tasks. Consider three scenarios.

### Short-Response Tasks (Function Calling, Simple Q&A)

- CoT length: 100–500 tokens
- ORM signal density: one reward per 100–500 tokens
- PRM value: **limited** — the sequence is short, and the ORM signal is already dense enough

### Medium Reasoning Tasks (GSM8K, MATH)

- CoT length: 500–2,000 tokens
- ORM signal density: one reward per 500–2,000 tokens
- PRM value: **significant** — can pinpoint exactly which step went wrong

### Long-CoT Reasoning Tasks (AIME, IMO, research-level math)

- CoT length: 5,000–50,000 tokens
- ORM signal density: one reward per 5,000+ tokens
- PRM value: **irreplaceable** — the ORM signal is barely enough to give any effective learning signal at all

[DeepSeek-R1](https://arxiv.org/abs/2501.12948)'s training run reported a telling pattern: early in training, the model's CoT length grew rapidly from a few hundred tokens to several thousand, but AIME accuracy improved slowly. Only later in training, once the model learned to "check itself at critical steps" (self-verification), did AIME accuracy break through significantly. This shows that long-CoT tasks **need process-level signal to learn efficiently**.

## 11.1.5 Two Industrial Approaches to PRM

There are two main industrial approaches to implementing PRM, corresponding to two different training methods.

### Discriminative PRM

Treat the PRM as a **classifier** — it takes a prompt + step as input and outputs the probability that "this step is correct."

Representative work: OpenAI's [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al. 2023).

Training data: human-annotated step correctness (the PRM800K dataset).

Model: a BERT-style encoder, or a decoder-only LLM with a classification head.

### Generative PRM

Treat the PRM as a **generator** — have an LLM "critique" each step in natural language.

Representative work: [ThinkPRM](https://arxiv.org/abs/2504.16828) (April 2025).

Training data: a small number of seed examples plus LLM-generated critiques.

Model: any LLM (LLaMA, Qwen, DeepSeek), driven with prompting plus light fine-tuning.

### Formal PRM

Treat the PRM as a **formal verifier** — use a theorem prover like Lean4 or Coq to verify automatically.

Representative work: DeepMind's [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) (July 2024) and DeepSeek's [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) (April 2025).

Training data: formalized mathematical theorems (in Lean4 format).

Model: an LLM paired with a Lean4 verifier.

These three approaches are the subject of the next three sections: [11.2 Discriminative PRM](./discriminative-prm), [11.3 Generative PRM](./generative-prm), and [11.4 Formal PRM](./formal-prm).

## Summary

Outcome reward is sufficient for simple tasks, but on long-CoT tasks its signal is too sparse to give an effective learning signal. Process reward scores each step of the reasoning and turns a sparse reward into a dense one, making it the key technique for long-CoT tasks.

PRM has three industrial approaches:

- **Discriminative**: a classifier — accurate, but annotation is expensive
- **Generative**: an LLM critic — cheap to annotate, but precision depends on prompt engineering
- **Formal**: Lean4 verification — zero false positives, but only applicable to formalizable tasks

The next three sections work through each of these approaches in detail. The final two sections cover PRM's role in inference-time search (MCTS, ToT) and in coordinating parallel reasoning (PaCoRe).
