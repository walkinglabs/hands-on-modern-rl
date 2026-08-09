# 18.4 Formal PRM Verifiers

The two PRM routes we've covered so far — discriminative and generative — share one fundamental problem: **both are LLM-based, so both inherit the LLM's errors.**

- Discriminative PRM mislabels: the annotator judges wrong, so the PRM learns the wrong label.
- Generative PRM misjudges: the LLM itself misunderstands the reasoning, so it outputs the wrong verdict.

If a verifier existed that **always said "correct" for correct reasoning and always said "wrong" for wrong reasoning, with zero misjudgment**, every one of these PRM problems would disappear.

That verifier exists — it's the **formal theorem prover**, systems like Lean4, Coq, and Isabelle. This section covers the formal PRM route, which is PRM research's "ultimate verifier" direction.

## 11.4.1 Why Formalization Is the Ultimate Verifier

### Formal language vs. natural language

A mathematical proof can be expressed in two languages.

**Natural language** (informal):

```text
Proof that √2 is irrational:
Suppose √2 = p/q, where p and q are coprime.
Then p² = 2q², so p is even.
Let p = 2k. Substituting gives 2k² = q², so q is also even.
This contradicts p and q being coprime. Hence √2 is irrational.
```

This is readable to a human, but it **can be ambiguous, can skip steps, and can contain errors.**

**Formal language** (Lean4):

```lean
theorem sqrt_two_irrational : Irrational (√2) := by
  intro h
  rcases h with ⟨p, q, h1, h2⟩
  -- assume √2 = p/q
  have h3 : p^2 = 2 * q^2 := by
    have : (√2)^2 = (p/q)^2 := by rw [h1]
    simp at this
    rw [div_pow] at this
    field_simp at this
    linarith
  -- ...
  sorry  -- (full proof omitted)
```

There's no ambiguity here — **every step of reasoning must be strictly proved using Lean4's rules.** If a proof compiles under Lean4's compiler, **it is mathematically correct** — there's no room for debate.

### What makes a Lean4 verifier special

A few properties of Lean4 make it an ideal PRM:

- **Zero misjudgment**: compiling successfully means the proof is correct. This is guaranteed by mathematical theorems themselves.
- **Automated**: the compilation process runs automatically, with no human judgment required.
- **Extensible**: you can define new mathematical structures and new theorems as needed.
- **Community support**: [Mathlib](https://github.com/leanprover-community/mathlib4) has already formalized undergraduate-level mathematics.

### The cost of formalization

Formalization also comes with costs.

- **Limited domain**: Lean4 is mainly used for mathematics. Other domains — natural-language reasoning, code logic — don't have mature formal systems.
- **Scarce data**: there's relatively little Lean4 code out there, so LLMs get insufficient pretraining data on Lean4.
- **High barrier to entry**: writing Lean4 requires specialized training, and most mathematicians aren't fluent in it.

## 11.4.2 AlphaProof and DeepMind's IMO Silver Medal

In July 2024, DeepMind announced that [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) had reached **silver-medal level** at the 2024 International Mathematical Olympiad (IMO), solving 4 of the 6 problems. This was a milestone for the formal PRM route.

### AlphaProof's architecture

AlphaProof's core idea is **AlphaZero-style self-play combined with a Lean4 verifier**:

```text
┌────────────────────────────────────────────────────┐
│ 1. Problem formalization: translate the math        │
│    problem into Lean4                                │
│                                                       │
│ 2. AlphaZero-style search:                            │
│    - Policy network: proposes the next Lean4 tactic  │
│    - Value network: evaluates the current proof state│
│    - MCTS: searches the proof tree                    │
│                                                       │
│ 3. Lean4 verifier: automatically checks each tactic  │
│                                                       │
│ 4. Self-play training: uses search results to train  │
│    the policy and value networks                     │
└────────────────────────────────────────────────────┘
```

This architecture is nearly identical to [AlphaGo Zero](https://www.nature.com/articles/nature24270) — the only difference is that the action space shifts from "moves on a Go board" to "a sequence of Lean4 tactics."

### AlphaProof's key design choices

**Design one: problem formalization**

IMO problems are stated in natural language, so AlphaProof first needs to translate them into Lean4. DeepMind used a **formalizer** — a specially trained LLM that translates natural-language math problems into Lean4.

This step is challenging in its own right: natural language is ambiguous, mathematical notation varies, and there's more than one way to state a theorem. DeepMind reported roughly a 50% success rate for this step — only about half the problems get translated correctly.

**Design two: large-scale Lean4 training**

AlphaProof was trained on roughly **one million** Lean4 problems. These included:

- Theorems already present in Mathlib
- Automatically generated problems (Lean4 statements generated by an LLM)
- Lean4 versions of past IMO and Putnam problems

**Design three: Lean4 MCTS**

AlphaProof uses MCTS to search for proofs. Each node is a proof state, each action is a Lean4 tactic, and the reward is "whether the proof is complete." The Lean4 compiler serves as the MCTS environment — it tells the search "whether this tactic is valid."

### AlphaProof's results

Of the 6 problems at IMO 2024, AlphaProof solved 4:

- **Algebra 1**: ✓ (full marks)
- **Algebra 2**: ✓
- **Combinatorics**: ✓
- **Number Theory**: ✓
- **Geometry 1**: ✗ (Lean4 translation failed)
- **Geometry 2**: ✗

The total score was about 25 out of 42, equivalent to silver-medal level (the gold-medal threshold was about 29).

The geometry failures weren't due to weak reasoning — they came from **Lean4 translation failures**. Formalizing geometry problems is considerably harder than formalizing algebra problems.

## 11.4.3 AlphaGeometry 2 and Geometry-Specific Formalization

To address AlphaProof's weakness in geometry, DeepMind released [AlphaGeometry 2](https://www.nature.com/articles/s41586-024-07819-5), a formal system built specifically to solve geometry problems.

AlphaGeometry 2's key innovations:

- **Synthetic data**: automatically generating 500 million geometry problems along with their proofs
- **Auxiliary constructions**: teaching the model the key technique of "adding auxiliary lines"
- **Symbolic + neural**: a hybrid of symbolic reasoning (DD, a Deductive Database) and a neural network (an LM)

  As of the July 2024 report, AlphaGeometry 2 reached gold-medal level on IMO geometry problems.

## 11.4.4 DeepSeek-Prover-V2 and Open-Source Formal PRM

[DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) (April 2025) is DeepSeek's open-source formal PRM work. Its goals are:

- Train an open-source model, using Lean4 and RL, that can solve math competition problems
- Advance the industrial usability of formal PRM

### Prover-V2's method

DeepSeek's method resembles AlphaProof's but adds several improvements.

**Improvement one: recursive proof search**

Prover-V2 uses **recursive theorem proving** — it decomposes a hard theorem into a set of subgoals, and decomposes each subgoal further, until the subgoals can be proved independently.

```text
Main goal: prove A
  ├── Subgoal 1: prove B (if B holds, then A holds)
  │     ├── Sub-subgoal 1.1: prove C
  │     └── Sub-subgoal 1.2: prove D
  └── Subgoal 2: prove E
```

This decomposition makes the proof search more structured, instead of hunting blindly through one enormous proof space.

**Improvement two: binary reward**

Prover-V2 uses the simplest possible reward — proof succeeds = 1, proof fails = 0. The Lean4 verifier supplies this reward with zero noise.

**Improvement three: large-scale data synthesis**

DeepSeek automatically generated a large volume of Lean4 theorems and proofs for training. The generation process includes:

- Using an LLM to generate Lean4 statements from natural-language math problems
- Using Monte Carlo Tree Search to find proofs
- Using the proofs it finds as training data

### Prover-V2's results

Prover-V2's results on MiniF2F (a formal-mathematics benchmark):

| Model                              | MiniF2F (validation set) |
| ---------------------------------- | ------------------------ |
| AlphaProof (public version)        | ~70%                     |
| **DeepSeek-Prover-V2**             | **88.9%**                |
| GPT-5 (natural-language reasoning) | ~50%                     |

This is state-of-the-art among open-source models on MiniF2F. 88.9% means Prover-V2 comes close to perfect on undergraduate-level formal mathematics.

## 11.4.5 The Cost of Formal PRM

Formal PRM achieves "zero misjudgment" in mathematics, but it comes with several costs.

### Limited domain

Lean4 is mainly used for mathematics. In other domains:

- **Code logic**: formal methods exist (Dafny, F\*), but industrial usability is low
- **Natural-language reasoning**: no mature formal system exists
- **Multimodal reasoning**: no formal method exists at all

So today, formal PRM is only applicable to **mathematics and tasks closely related to formal mathematics**.

### Scarce formal data

There's relatively little Lean4 code available:

- Mathlib contains roughly 1 million lines of Lean4 code
- Compared to natural-language pretraining data (trillions of tokens), that's about 6 orders of magnitude less

LLM capability on Lean4 lags far behind natural language. This is the fundamental bottleneck for formal PRM.

### Translation cost

Formal PRM requires translating a task into Lean4. That translation is itself an LLM's job, and it introduces errors. AlphaProof reported a 50% translation success rate — half the problems get translated incorrectly.

### Training cost

Training with Lean4 MCTS is extremely expensive computationally — every tactic call triggers a Lean4 compilation, and each compilation takes anywhere from a few hundred milliseconds to a few seconds. AlphaProof trained for several months, consuming compute comparable to GPT-4's training run.

## 11.4.6 The Future of Formal PRM

Despite these costs, the formal PRM research direction remains important. A few directions for the future:

### Autoformalization

Teaching LLMs to automatically translate natural language into Lean4. This is the direction taken by [AlphaProof's formalizer](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) and by work such as [Autoformalization with Large Language Models](https://arxiv.org/abs/2205.12615).

### Hybrid Lean4 + LLM

Rather than requiring the LLM to generate Lean4 in full, let the LLM produce natural-language reasoning while Lean4 verifies it. This keeps the LLM's flexibility while adding formal rigor.

### Extending to other domains

Extending formal methods to code (using Dafny, F\*), physics (using Lean4 physics libraries), and biology (using Lean4 chemistry libraries). This is the direction for growth beyond Mathlib.

### Neuro-symbolic integration

Combining the LLM's "intuition" with formal systems' "rigor" — the LLM proposes a proof idea, and the formal system verifies it. This is the shared direction behind both DeepMind's AlphaProof and AlphaGeometry 2.

## Summary

Formal PRM is PRM research's ultimate direction — achieving zero-misjudgment verification through formal systems like Lean4. AlphaProof reached silver-medal level at the IMO, and DeepSeek-Prover-V2 reached 88.9% on MiniF2F, demonstrating that this approach is technically viable.

But formal PRM is constrained by domain (mainly mathematics), by scarce data (little Lean4 training data exists), and by translation cost (natural-language-to-Lean4 translation isn't perfect). Its industrial application today is concentrated in mathematical reasoning, and it's hard to generalize to general-purpose LLM tasks.

That covers the three PRM routes:

- **Discriminative PRM** (11.2): high annotation cost, but precise classification
- **Generative PRM** (11.3): less annotation needed, generalizes well, but depends on the LLM's own judgment
- **Formal PRM** (11.4): zero misjudgment, but limited in domain

The next two sections cover how PRMs are used in inference-time search (MCTS, ToT, PaCoRe) — a PRM isn't only a training-time reward, it's also a search guide at inference time.
