# 17.4 Hybrid Thinking and Thinking Budgets

The previous section showed the promise of test-time compute scaling — the more a model thinks, the better its answers get. Industrial deployment immediately runs into the opposite problem: **if a model thinks deeply about every question, every API call burns 10K+ tokens on thinking, and the service becomes too slow to use**.

The deeper issue is that **most user requests don't need deep reasoning at all**. "What's the weather today," "translate this sentence for me," "write a simple hello world" — for tasks like these, making the model think only slows down the response and drives up cost.

In the second half of 2025 a new engineering paradigm emerged to address this: **Hybrid Thinking** — the same model supports both a "thinking" mode and a "non-thinking" mode, and the user or the model itself decides which one to use.

## 10.3.1 DeepSeek V3.1 and Hybrid Mode Fusion

[DeepSeek V3.1](https://api-docs.deepseek.com/news/news250821) (August 2025) is an early industrial implementation of Hybrid Thinking. V3.1's design rests on three ideas:

- **One model, not two**: it doesn't maintain separate "thinking model" and "non-thinking model" checkpoints
- **Mode switching**: a special marker in the prompt (such as `<think>` / `</think>`) controls whether the model enters thinking mode
- **Optional thinking content**: the thinking portion can be kept in the output (visible to the user) or stripped out (the user sees only the final answer)

V3.1's training pipeline has two stages:

1. **Reasoning RL**: RL on math and code tasks, teaching the model to reason deeply
2. **General RL**: RL on dialogue and tool-calling tasks, teaching the model to answer directly when reasoning isn't needed

The key challenge in the second stage is **preserving the reasoning ability the model already has while teaching it when to reason and when not to**. DeepSeek's approach is to mix the training data — some prompts explicitly trigger reasoning (math problems), others don't (small talk) — so the model learns the judgment call of switching modes on its own.

## 10.3.2 Qwen3: Thinking Mode Fusion and Thinking Budget

The [Qwen3 technical report](https://arxiv.org/abs/2505.09388) (May 2025) proposes a more systematic Hybrid Thinking scheme. Every model in the Qwen3 family, from 0.6B to 235B, supports both modes.

### Thinking Mode Fusion

Qwen3 doesn't simply "train a reasoning model plus a non-reasoning model." Instead it **fuses the two modes into a single model**. Concretely:

- The training data mixes thinking and non-thinking samples
- Thinking samples: a full long CoT followed by the answer
- Non-thinking samples: a direct, short answer
- A `<think>` token triggers the mode switch

This training scheme lets the model **automatically learn to pick a mode based on the prompt** — it turns thinking on for a math problem and leaves it off for small talk.

### Thinking Budget: Controllable Reasoning Depth

Qwen3 introduces an engineering parameter called **thinking budget**. In an API call, the user can specify `thinking_budget=N`, capping how many tokens the model may spend thinking.

```python
# Qwen3 thinking budget example
response = client.chat.completions.create(
    model="qwen3-235b-a22b",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational"}],
    extra_body={"thinking_budget": 2000}  # think for at most 2000 tokens
)
```

The engineering value of this parameter:

- **Controllable latency**: a high-traffic service can set a small budget to guarantee response time
- **Controllable cost**: under token-based billing, the budget caps spend
- **Quality trade-off**: set a large budget for hard problems, a small one for easy ones

But the thinking budget also raises an **algorithmic challenge**: how does the model "stop gracefully when the budget runs out"? Qwen3's answer is to add a **length penalty** during RL training — answers that exceed the budget are penalized. This is the same idea as [DAPO's Overlong Reward Shaping](../chapter18_grpo/deepseek-dapo).

## 10.3.3 NoThinking + Best-of-N: A Counterintuitive Finding

In April 2025, Ma et al. published an intriguing study, [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858) (commonly called NoThinking). The paper makes a counterintuitive claim: on many tasks, **"don't think + Best-of-N" outperforms "think."**

### The Core NoThinking Experiment

Setup:

- Base model: Qwen3-32B
- Tasks: AIME, GPQA, MMLU-Pro
- Two inference strategies compared:
  - **Thinking**: standard thinking mode, a single generation (with CoT)
  - **NoThinking + BoN**: thinking disabled, but N candidate solutions are generated and majority vote picks the best one

Results:

| Task         | Thinking (single) | NoThinking + BoN (N=32) |
| ------------ | ----------------- | ----------------------- |
| AIME 2024    | 60.2              | **65.1**                |
| GPQA Diamond | 55.3              | **58.7**                |
| MMLU-Pro     | 72.1              | 71.8 (roughly tied)     |

In other words, **redirecting the compute that would have gone into N rounds of thinking toward N rounds of nothinking-plus-voting instead produces better results**.

### Why Does NoThinking Work?

Ma et al. offer a few explanations:

1. **A thinking CoT isn't free** — a wrong reasoning step compounds and drags the final answer down with it. NoThinking skips reasoning and answers directly, sidestepping that failure mode.
2. **Best-of-N supplies diversity** — N independent samples are more diverse than a single long chain, raising the odds that one of them hits the correct answer.
3. **CoT is over-thinking on some tasks** — an easy task doesn't need a long chain of reasoning; thinking just injects noise.

But NoThinking has limits:

- **It falls short of Thinking on hard problems** — on the hardest AIME problems, NoThinking + BoN still can't beat Thinking
- **It depends on a verifier** — BoN needs some mechanism for judging "which answer is better," and majority voting only works for tasks with a well-defined answer

The point of this study isn't to dismiss Thinking. It's to show that **test-time compute can be spent in more than one way, and no single way is optimal across the board** — the right strategy depends on the task.

## 10.3.4 Long CoT Compression and Kimi k1.5's long2short RL

Reasoning models tend to develop a common problem late in training: **the CoT keeps getting longer**. R1-Zero, o1, and Qwen3 have all reported the same pattern — the more training steps, the longer the model's answers, until length can balloon past 50K tokens. There are a few reasons:

- **The reward signal rewards correctness**, and a longer CoT means more chances to double-check, which raises the odds of getting it right
- **Reflection and verification behavior gets reinforced** — the model learns to "check this again"
- **There's no explicit length constraint**, so the model has no incentive to keep the CoT short

But a 50K-token CoT is unacceptable in deployment — latency and cost are both too high. How do you compress a long, post-training CoT down to a deployment-friendly length while keeping the reasoning quality?

[Kimi k1.5](https://arxiv.org/abs/2501.12599) (January 2025) proposes **long2short RL** — a two-stage training scheme.

### Stage One: Train a Long-CoT Reasoning Model

First, standard RL (GRPO plus a math reward) trains a long-CoT reasoning model. This stage places no limit on length, letting the model fully develop its reasoning ability. After training, the model's typical CoT length runs 5K–20K tokens.

### Stage Two: long2short Distillation Plus RL

The second stage transfers the ability captured in long CoT onto short CoT:

1. **Data generation**: the stage-one model generates long-CoT answers (high quality)
2. **Distillation compression**: a small model (or another LLM) compresses each long CoT into a short one, preserving the key reasoning steps
3. **SFT**: the original model is fine-tuned on the compressed short-CoT data
4. **Length-penalty RL**: a further round of RL is run, this time with a length penalty added to the reward

The length penalty takes this form:

$$r_{\text{total}} = r_{\text{correct}} - \lambda \cdot \max(0, |o| - L_{\text{target}})$$

Here $|o|$ is the response length, $L_{\text{target}}$ is the target length, and $\lambda$ is the penalty strength. This is the same shape as [DAPO's Overlong Reward Shaping](../chapter18_grpo/deepseek-dapo), but Kimi treats it as a training signal for **active compression** rather than just an engineering guardrail against runaway length.

### The Effect of long2short

The Kimi k1.5 paper reports:

- **Long-CoT model** (baseline): AIME score 60.1, average CoT length 12K tokens
- **long2short model**: AIME score 58.7 (a 1.4-point loss), average CoT length 3.2K tokens (a 73% reduction)

In other words, **using 25% of the tokens retains 97.7% of the capability**. That trade-off is a strong deal for industrial deployment.

### How long2short Relates to Thinking Budget

long2short and thinking budget are complementary:

- **long2short** is compression at **training time** — it changes the model itself so it's inclined to generate short CoT
- **thinking budget** is control at **inference time** — it doesn't change the model, but forces a cutoff at generation time

Industrial deployments typically combine both:

```text
Training time: long2short RL teaches the model to produce short CoT
Inference time: thinking budget sets a safety ceiling
```

## 10.3.5 The Algorithmic Challenges of Hybrid Thinking

Hybrid Thinking isn't as simple as "adding a switch." It raises several new algorithmic questions.

### Deciding Which Mode to Use

How does the model decide whether a given question calls for thinking or not?

- **Explicit user control**: simplest — the user adds `<think>` in the prompt or leaves it out
- **Automatic model judgment**: the model learns during training to decide based on the content of the prompt

Qwen3 uses a hybrid policy — by default the model judges automatically, but the user can explicitly override it. The training data for automatic judgment needs to cover both cases: prompts where thinking helps (math, code) and prompts where it doesn't (small talk, translation).

### Consistency Between the Two Modes

The thinking mode and the non-thinking mode should give **consistent answers** — thinking can't say "A" while non-thinking says "B." But the two modes are optimized separately during training, which makes inconsistency easy to introduce.

DeepSeek's fix is to add a **consistency constraint** during RL — the two modes' answers to the same prompt should agree, and the reward is lowered when they don't.

### Stopping Gracefully When the Budget Runs Out

When the thinking budget is exhausted, the model has to interrupt its current train of thought and give an answer directly. But during RL training the model has never seen this situation, so a hard cutoff tends to produce a poor answer.

Qwen3's fix is to **simulate the budget-exhausted scenario during training** — the rollout stage randomly truncates the CoT, so the model learns to produce a reasonable answer even when cut off mid-thought.

## Summary

Hybrid Thinking and thinking budgets are the path reasoning models must take to go from "lab prototype" to "industrial product." The core problem they solve is: **how can a model capable of thinking still work efficiently when thinking isn't needed?**

DeepSeek V3.1, Qwen3, and Kimi K2 have all made important progress in this direction:

- **V3.1**: mode fusion plus optional thinking content
- **Qwen3**: Thinking Mode Fusion plus the `thinking_budget` parameter
- **Kimi k1.5**: long2short RL for active CoT compression
- **The NoThinking study**: shows that "don't think, then vote" can beat thinking on certain tasks

But Hybrid Thinking raises a new question of its own: **what the model is thinking — should the user get to see it?** That's the subject of the next section, Hidden vs. Visible CoT.
