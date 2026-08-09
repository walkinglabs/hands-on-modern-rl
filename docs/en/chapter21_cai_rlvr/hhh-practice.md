# 19.1 The HHH Principles in Claude's Practice

> [Chapter 19](./intro) covered the theory of Constitutional AI and the RLAIF framework. This section answers the engineering question: **how does Anthropic actually implement CAI in Claude's training?** The answer is the HHH triad — Helpful, Harmless, Honest — together with a set of adversarial training tricks.

## The HHH Alignment Principles

The value framework underneath Constitutional AI is **HHH** — Helpful, Harmless, Honest. Anthropic treats these three as concrete, optimizable targets, each captured by a formal preference function.

### Helpful and Maximizing User Utility

A helpful assistant genuinely solves the user's problem, engaging with the request instead of deflecting or giving a token answer. Formally:

$$
\text{Helpful}(y \mid x) = \mathbb{E}_{u \sim \text{user}} \big[U_u(x, y)\big]
$$

Here $U_u(x, y)$ is the utility that user $u$ assigns to response $y$ given prompt $x$. In RLHF/RLAIF, $U$ is approximated by preference data.

A common failure mode of Helpful is **verbosity**: an RM tends to score longer answers higher, and the policy keeps growing longer over training as a result. Claude's training explicitly adds a length penalty term:

$$
r_{\text{adj}}(x, y) = r_\phi(x, y) - \lambda_{\text{len}} \cdot |y|
$$

### Harmless and Refusing to Help with Dangerous Requests

Formalizing Harmless is subtler than "say nothing" — what it actually means is "don't help the user cause harm." A typical definition:

$$
\text{Harmless}(y \mid x) = 1 - \mathbb{P}(\text{harm} \mid x, y)
$$

Here $\mathbb{P}(\text{harm})$ is the probability that this response helps cause real-world harm. This quantity isn't directly observable, so CAI approximates it with the Constitution plus an AI judge.

::: warning The Tension Between Helpful and Harmless
Models trained with RLHF often develop **evasiveness**: they'd rather refuse than take any risk, so both "how do I make fertilizer" and "write an educational article about fertilizer" get declined. CAI's Constitution explicitly includes a principle for this: "If the request itself is harmless — e.g., education, writing, research — the model should engage with it even when the topic sounds sensitive." This is CAI's key improvement over plain RLHF.
:::

### Honest and Not Outputting False Information

Honest requires that the model not lie, not pretend to know things it doesn't, and be able to express uncertainty. Formally:

$$
\text{Honest}(y \mid x) = 1 - D_{KL}\big(p_{\text{model}}(\cdot \mid x) \,\|\, p_{\text{true}}(\cdot \mid x)\big)
$$

Here $p_{\text{true}}$ is the "distribution of objective truth." In practice $p_{\text{true}}$ is inaccessible, so **verifiable rewards** — math answers, code tests, fact retrieval — are used as a proxy. This is exactly where [RLVR](../chapter18_grpo/rlvr) connects to HHH: RLVR is, in essence, a hard-verification version of the Honest principle.

### Jointly Optimizing All Three HHH Objectives

CAI combines the three objectives with weights:

$$
r_{\text{HHH}}(x, y) = \alpha_H \cdot \text{Helpful}(y \mid x) + \alpha_{HL} \cdot \text{Harmless}(y \mid x) + \alpha_{Ho} \cdot \text{Honest}(y \mid x)
$$

Different principles in the Constitution map to different $\alpha$ weightings: some principles emphasize Helpfulness ("comply as much as possible if the request is legitimate"), others emphasize Harmlessness ("do not assist with violence"). When the AI judge scores a response, it combines these principles according to the Constitution's weighting, which amounts to an implicit HHH weighting.

| Principle | Typical Failure Mode             | CAI's Response                                                         |
| --------- | -------------------------------- | ---------------------------------------------------------------------- |
| Helpful   | Verbosity, template collapse     | Length penalty + diversity reward                                      |
| Harmless  | Over-refusal                     | Constitution distinguishes "sensitive but legitimate" from "dangerous" |
| Honest    | Hallucination, feigned knowledge | Explicit "I don't know" training + RLVR verification                   |

## CAI in Practice in Claude Training

CAI isn't a toy from a paper — it's the actual training pipeline used across the Claude model family. This section walks through the evolution of CAI from Claude 2 through Claude 3 to Claude 3.5, focusing on the concrete changes made in industrial practice.

### Claude 2 (2023) and the First Full CAI Deployment

Claude 2 was the first production model to run the complete SL-CAI + RL-CAI pipeline end to end. Key technical details:

- **Constitution size**: roughly 40 principles, spanning all three HHH categories.
- **Self-critique length**: each critique is capped at 200-400 tokens, so it doesn't slow training down.
- **Judge model**: a model larger than the generator is used as the judge (Claude 2 used an internal 100B+ model to judge a 50B model), avoiding self-preference bias.
- **Data mix**: roughly 70% AI feedback plus 30% high-quality human feedback. Human feedback is still kept in the loop, but only for edge cases where the AI's judgment is uncertain.

Anthropic reported that, relative to a pure-RLHF version, Claude 2 saw **harmfulness drop by 50%+ and the over-refusal rate drop by 30%**.

### Claude 3 (2024) and Constitution Expansion with Collective CAI

The Claude 3 series expanded the Constitution from 40 principles to roughly 80, adding new dimensions:

- **Collective Constitutional AI**: Anthropic partnered with a public research organization to have 1,000+ respondents from different cultural backgrounds vote on which values the AI should follow. The result showed strong global agreement on a handful of principles: honesty, refusing to assist with violence, and respecting privacy.
- **Reducing over-refusal**: a new principle was added — "refusal should be based on actual risk, not topic sensitivity."
- **Multilingual alignment**: the Constitution was translated into 20+ languages, but a **single English master version** is kept as ground truth, to avoid value drift introduced by translation.

On the engineering side, Claude 3 continued the critique-revision loop from the original Constitutional AI paper (Bai et al. 2022): the model critiques its own past responses after the fact, and those critiques become additional SFT data. This effectively closes the loop from deployment data back into training.

### Claude 3.5 (2024-2025) and the Fusion of CAI and RLVR

The key shift in the Claude 3.5 era: **CAI stopped being a standalone pipeline and merged with RLVR**. Concretely:

1. **Helpfulness training**: dominated by RLVR — math and code are checked with rule-based verification, while writing and instruction-following still rely on RLAIF.
2. **Harmlessness training**: dominated by CAI, because "safety" can't be rule-verified — it can only be judged via the Constitution plus an AI judge.
3. **Honesty training**: a mix — factual questions use retrieval augmentation plus a verifier model, open-ended questions use an AI judge plus RLVR.

These three lines are combined as a weighted reward inside PPO:

$$
R(x, y) = w_{\text{task}} r_{\text{RLVR}}(x, y) + w_{\text{safe}} r_{\text{CAI}}(x, y) + w_{\text{hon}} r_{\text{verifier}}(x, y) - \beta D_{KL}
$$

This kind of **multi-objective RL** is the core training paradigm behind Claude 3.5/4, and it's one of the reward-combination schemes used in [Chapter 19's PRM-guided search](../chapter20_prm_search/inference-time-search).

### A Few Engineering Lessons from Claude 3.5

::: tip Industry Consensus (as of 2025)

1. **Pure RLAIF is unreliable** — a small anchor of high-quality human feedback is required.
2. **A longer Constitution is harder to tune** — 80 principles is already the point of diminishing returns; adding more causes principles to conflict with each other.
3. **The judge model must be stronger than the generator** — otherwise self-preference bias becomes severe.
4. **Safety training and capability training must be decoupled** — otherwise the KL constraint drags down capability gains.
   :::

## Section Summary

HHH (Helpful, Harmless, Honest) is the set of three principles Anthropic actually uses in Claude's training. Helpful requires the model to genuinely try to complete the task; Harmless requires it to refuse harmful requests; Honest requires it not to fabricate. These three frequently pull against each other — for a sensitive but reasonable question, over-optimizing Harmless turns into evasiveness, which costs Helpful and Honest. CAI uses the Constitution to teach the model to find a balance amid this conflict.

The next section, [21.3 RLAIF Engineering and Constitution Expansion](./rlaif-engineering), covers the 80-page Constitution Anthropic published in 2026 — currently the most detailed piece of AI constitution engineering in the industry.
