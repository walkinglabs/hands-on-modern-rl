# Chapter 21 · RL for Coding Agents

[Chapter 20 Agentic RL](../chapter22_agentic/intro) introduced RL training for agents that call tools and act across multiple turns. This chapter zooms in on the subfield with the most industrial weight right now: **RL-based SWE (Software Engineering)**—using RL to train models that fix bugs, implement features, and write tests on their own.

Why does this deserve its own chapter? Three reasons:

1. **SWE is RLVR's natural battlefield**—unit tests are a perfect "zero-noise verifier," the same idea that underlies [Chapter 9's formalized-PRM approach](../chapter20_prm_search/formal-prm).
2. **2025 produced a wave of industrial-grade breakthroughs in this area**—Meta's SWE-RL, ByteDance's DeepSWE, Tsinghua's SSR, Alibaba's CWM—each pushing SWE-bench accuracy to a new high.
3. **SWE-RL is Agentic RL's "algorithm lab"**—many of its findings (long-horizon credit assignment, self-play, world models) generalize to other domains.

## Questions this chapter answers

- **What is SWE-bench**, and why is it the core benchmark for SWE-RL?
- **How does Meta SWE-RL** reach SOTA with open-source data and plain GRPO?
- **How does the Code World Model (CWM)** model code execution as an MDP?
- **How does DeepSWE** train a long-horizon agent with verifiable reward?
- **How does Self-play SWE-RL (SSR)** get a model to generate its own training data?
- **Where is SWE-RL headed**—what do multi-language, multi-repo, multi-agent extensions look like?

## Chapter map

```text
12.1 SWE-bench and the RL-based SWE paradigm
     ├── SWE-bench task definition
     ├── Why SWE is the ideal battlefield for RLVR
     └── Manufacturing data: SWE-smith and SWE-gym
12.2 Meta SWE-RL: the open-source SOTA
     ├── Data scale and composition
     ├── Algorithm choice: GRPO + rule-based reward
     ├── Engineering details: context management, test sampling
     └── SWE-bench Verified 41.0%
12.3 Code World Model (CWM)
     ├── Modeling code execution as an MDP
     ├── World model training
     ├── RL built on top of CWM
     └── Relationship to model-based RL
12.4 DeepSWE: RL for long-horizon agents
     ├── The challenge of trajectories with 16+ steps
     ├── Step-level reward shaping
     ├── Test-time search integration
     └── ByteDance Seed's industrial practice
12.5 Self-play SWE-RL (SSR)
     ├── The model generates its own bugs and fixes
     ├── Curriculum learning
     ├── Tsinghua's SSR work
     └── The formation of a data flywheel
12.6 RL-based SWE in industry
     ├── Cursor, Cognition Devin, ByteDance Trae
     ├── Business models and cost structure
     └── Multi-language, multi-repo extensions
```

## Relationship to other chapters

This chapter assumes you've already read:

- [Chapter 7 The GRPO improvement family](../chapter18_grpo/grpo-family)—the base RL algorithm
- [Chapter 20 Agentic RL](../chapter22_agentic/intro)—the multi-turn interaction foundations for agents
- [Chapter 9 PRM](../chapter20_prm_search/intro)—the idea of a formalized verifier

Later sections point forward to:

- [Section 12.4 Agent training systems](../chapter22_agentic/build-agentic-training-system)—the engineering behind SWE-RL
- [Chapter 12 Reward hacking](../chapter15_rlhf/evaluation)—the hacking patterns specific to SWE-RL

## An intuitive opening

**The intuition: SWE-RL takes the PRM formalization idea and applies it to code.** Lean4 is math's formal verifier; unit tests are code's formal verifier. Both rest on the same core idea: **replace human or LLM subjective judgment with zero-error-rate external verification.**

But SWE has something Lean4 doesn't—**an enormous body of real-world practice to draw on.** GitHub holds hundreds of millions of lines of code, millions of PRs, tens of millions of commits—a scale of training data the math world simply can't match.

With that intuition in hand, let's move on to 12.1.
