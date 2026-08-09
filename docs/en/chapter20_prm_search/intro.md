# Chapter 18 · Process Reward Models and Inference-Time Search

In the last chapter we saw reasoning models break through on hard tasks—o1, DeepSeek-R1, and Claude Opus 4.6 can all handle serious math, code, and scientific reasoning. But all of these models lean on one key assumption: **whether the final answer is right or wrong can serve as the RL reward signal**.

That assumption holds for simple tasks—a math problem is either solved or it isn't. As tasks get more complex, though, the assumption starts to break down:

- **Long CoT tasks**: a 10,000-token reasoning chain might have the first 8,000 tokens right and the last 2,000 wrong. The model only sees "wrong answer"—it has no idea where things went off the rails.
- **Code generation**: a program fails to compile. Which line caused it? Which piece of logic is broken?
- **Multi-step agent tasks**: a 10-step trajectory fails. At which step did it fail?

This is the **sparse reward problem**—the reward signal only shows up at the end of the sequence, and the intermediate steps get no feedback at all. The **Process Reward Model (PRM)** exists precisely to solve this problem: it scores every step of the reasoning process, turning a sparse reward into a dense one.

## Questions This Chapter Answers

- What is the essential difference between **outcome reward** and **process reward**?
- How does the **discriminative PRM** (OpenAI's classic approach) work, and why is labeling cost its bottleneck?
- Why can a **generative PRM** (ThinkPRM) outperform the discriminative approach while using far fewer labels?
- How does a **formal PRM** (AlphaProof, Lean4) achieve "zero false positive" verification?
- How does **inference-time search** (MCTS, Tree of Thoughts, Beam Search) use a PRM to guide it?
- How does **parallel coordinated reasoning** (PaCoRe) replace traditional depth-first reasoning?

## Chapter Map

```text
11.1 Outcome Reward vs. Process Reward
     ├── The sparse reward problem
     ├── The nature of credit assignment
     └── Why PRM is irreplaceable for long-CoT tasks
11.2 Discriminative PRM (the classic approach)
     ├── OpenAI's "Let's Verify Step by Step"
     ├── The PRM800K dataset
     ├── PRM as a re-ranking model
     └── Limitations: high labeling cost, weak generalization
11.3 Generative PRM (the new approach)
     ├── ThinkPRM: a generative verifier
     ├── The trick behind using 100x fewer labels
     ├── Verifier Compute Scaling
     └── Generative vs. discriminative compared
11.4 Formal PRM (the ultimate verifier)
     ├── Lean4 / Coq: zero false-positive verification
     ├── AlphaProof: IMO silver medal
     ├── AlphaGeometry 2: specialized for geometry
     └── DeepSeek-Prover-V2: 88.9% on MiniF2F
11.5 Inference-Time Search
     ├── Beam Search over Thoughts
     ├── MCTS over Thoughts
     ├── Tree of Thoughts
     └── AlphaCodium / rStar
11.6 Parallel Coordinated Reasoning (PaCoRe)
     ├── 16-way parallel rollouts
     ├── Outcome-based RL training
     ├── AIME 2025: 94.4
     └── The depth-vs-breadth trade-off
11.7 GenRM and Verifier Models
     ├── Generative Reward Model
     ├── LLM-as-Judge
     └── Self-Rewarding Language Models
```

## Relationship to Other Chapters

This chapter assumes you've already read:

- [Chapter 13, RLHF Reward Models](../chapter15_rlhf/reward-function-design)—the foundations of the Outcome Reward Model
- [Chapter 16, The GRPO Improvement Family](../chapter18_grpo/grpo-family)—how the credit assignment problem shows up in GRPO
- [Chapter 17, Reasoning Models](../chapter19_reasoning/intro)—why reasoning models need a PRM in the first place

Later chapters that build on this one:

- [Chapter 20, Agentic RL](../chapter22_agentic/intro)—process reward for multi-step trajectories
- [Chapter 28, Reward Hacking](../chapter30_alignment_failures/intro)—the reward-hacking problem in PRMs

## An Intuitive Opening

Before we get into the formal material, let's establish two key intuitions.

**Intuition one: PRM turns "exam grading" into "homework grading."** A traditional outcome reward works like exam grading—it only looks at whether the final answer is right, 100 points if correct and 0 if not. A PRM works like a teacher grading homework—every step of the reasoning gets scored, correct steps earn positive points, wrong steps earn negative points, and half-right steps earn partial credit. Grading this way takes more time, but the feedback is far more granular, and the student (the model) learns more from it.

**Intuition two: a PRM is a verifier, not a policy.** A common point of confusion is treating a PRM as "just another reward model." Strictly speaking, a PRM is a **verifier**—its job isn't to "generate good reasoning" but to "judge whether reasoning is good." The verifier and the policy are trained toward different objectives: the policy has to learn _what to do_, while the verifier has to learn _what to judge_. This distinction matters a lot for understanding GenRM and LLM-as-Judge later in the chapter.

With these two intuitions in hand, let's move on to 18.1.
