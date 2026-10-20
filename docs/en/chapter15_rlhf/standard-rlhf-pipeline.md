---
title: 13.4 The RLHF Pipeline
---

# 13.4 Reinforcement Learning Fine-Tuning

## Reading Guide

**Core points**

- Master the three-stage InstructGPT-style RLHF pipeline: SFT, Reward Model, PPO.
- Understand each stage's inputs, outputs, acceptance metrics, and typical failure modes.
- Organize experiments as artifacts: datasets, checkpoints, and evaluation reports must be traceable.

**Core formulas**

$$
\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y)\sim \mathcal{D}_{SFT}}\left[\log \pi_\theta(y\mid x)\right]
\quad \text{(SFT: imitate high-quality answers)}
$$

$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}_{pref}}
\left[\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\right]
\quad \text{(RM: learn preference ranking)}
$$

$$
\max_\theta\ \mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}
\left[r_\phi(x,y) - \beta D_{KL}(\pi_\theta(\cdot\mid x)\|\pi_{ref}(\cdot\mid x))\right]
\quad \text{(PPO-RLHF: optimize reward, but do not drift)}
$$

> Keep one sentence in mind:
>
> RLHF is not a training script. It is an artifact pipeline. Each stage must leave behind data, models, metrics, and failure cases.

This chapter follows the classic OpenAI InstructGPT recipe: SFT first, then a reward model, then PPO-based RLHF. It is not the only post-training method, but it is the reference point you need before you can properly understand DPO, GRPO, RLVR, and later variations.

```mermaid
flowchart LR
    Base["Base model\n(pretrained)"] --> SFTData["SFT data\n(instruction, answer)"]
    SFTData --> SFT["Step 1: SFT\nsupervised fine-tuning"]
    SFT --> SFTModel["SFT model\nassistant starting point"]
    SFTModel --> PrefData["Preference data\n(chosen / rejected)"]
    PrefData --> RM["Step 2: Reward Model\npreference modeling"]
    SFTModel --> PPO["Step 3: PPO-RLHF\npolicy optimization"]
    RM --> PPO
    SFTModel -.->|"freeze as Reference"| PPO
    PPO --> Eval["Evaluation\nbenchmarks + preferences + manual review"]

    style Base fill:#e3f2fd,stroke:#1565c0
    style SFT fill:#fff3e0,stroke:#e65100
    style RM fill:#fff8e1,stroke:#f57f17
    style PPO fill:#e8f5e9,stroke:#2e7d32
    style Eval fill:#f3e5f5,stroke:#6a1b9a
```

## One Prompt, Three Stages

Do not start from framework names. Start from a single user question:

```text
Explain what PPO's clip ratio does, and give an intuitive example.
```

The standard RLHF pipeline does three different things around this same prompt:

1. **SFT stage**: provide a high-quality demonstration answer so the model learns "how to respond to this type of instruction."
2. **RM stage**: prepare multiple candidate answers to the same prompt and label which answer is better.
3. **PPO stage**: let the current policy generate answers, score them with the RM, and update the policy to increase the probability of high-scoring answers.

These stages correspond to three different data formats:

```json
{
  "sft_item": {
    "prompt": "Explain what PPO's clip ratio does, and give an intuitive example.",
    "response": "The clip ratio limits how far the new policy can move..."
  },
  "preference_item": {
    "prompt": "Explain what PPO's clip ratio does, and give an intuitive example.",
    "chosen": "Think of clipping like a seatbelt: it prevents one update from being too aggressive...",
    "rejected": "PPO is an algorithm and it is widely used. It is important."
  },
  "ppo_prompt_item": {
    "prompt": "Explain what PPO's clip ratio does, and give an intuitive example."
  }
}
```

The same prompt can appear in multiple stages, but you must prevent evaluation leakage: prompts used for evaluation should not be reused for training.

## The Three Stages and Their Deliverables

| Stage | Input                              | Output                               | Acceptance metrics                                            | Most common failure                     |
| ----- | ---------------------------------- | ------------------------------------ | ------------------------------------------------------------- | --------------------------------------- |
| SFT   | instruction-answer pairs           | an instruction-following assistant   | SFT loss, format adherence, human inspection                  | learns style but stays shallow          |
| RM    | preference pairs (chosen/rejected) | a reward model that scores responses | held-out accuracy, margin, calibration samples                | learns wrong preference (length bias)   |
| PPO   | prompts + RM + reference model     | an improved policy (and a critic)    | reward rises without KL/length exploding; preference win-rate | reward hacking, regression, instability |

The critical point is: SFT and RM are not "just preparation." They are where most RLHF success or failure is decided. Bad SFT data gives you a bad starting policy; a biased RM gives PPO an incorrect target to optimize.

## Step 0: Choose a Base Checkpoint

RLHF does not start from training a model from scratch. It starts from a base checkpoint that becomes an artifact:

```text
artifacts/
  base/
    model_name.txt
    tokenizer_config.json
    generation_probe.jsonl
```

When selecting a base model for a teaching-scale experiment, check:

| Dimension | Question                               | Practical suggestion                   |
| --------- | -------------------------------------- | -------------------------------------- |
| size      | can you run a four-model loop locally? | 360M to 0.5B is a good start           |
| language  | does it cover your target language?    | pick a model trained for your language |
| license   | can you fine-tune and redistribute?    | read the model card                    |

The key output of Step 0 is not a trained model. It is a **baseline report**: how does the base model respond to a fixed prompt set? Without a baseline, there is no way to judge what SFT and RLHF actually changed.

## Step 1: SFT Teaches "How To Answer"

SFT is supervised learning. Given a prompt $x$ and a demonstration answer $y$, we maximize the conditional likelihood:

$$
\mathcal{L}_{SFT} = -\sum_{t=1}^{T}\log \pi_\theta(y_t \mid x, y_{<t}).
$$

In plain words: given the prompt and the already-generated prefix, make the model more likely to generate the demonstration next token.

A minimal SFT record can use the chat format below:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a clear, honest, and concise reinforcement-learning tutor."
    },
    { "role": "user", "content": "What is a value function?" },
    {
      "role": "assistant",
      "content": "A value function estimates the expected cumulative return obtained by following a policy from a given state."
    }
  ],
  "source": "human_written",
  "quality": "verified"
}
```

One crucial implementation detail is the **loss mask**: in chat-format data, only the assistant tokens should contribute to the loss. If you train on the user/system text, you teach the model to repeat the user and to generate role markers.

## Step 2: The Reward Model Teaches "What Is Better"

A reward model does not learn a single correct answer. It learns preference ordering. A typical sample is:

```json
{
  "prompt": "Explain PPO's KL penalty.",
  "chosen": "The KL penalty acts like a safety rope: it prevents the policy from drifting too far from the reference.",
  "rejected": "KL is a math formula and PPO uses it, so it is important.",
  "labeler": "human_or_judge",
  "rubric": ["accuracy", "helpfulness", "clarity"]
}
```

The RM learns a scoring function $r_\phi(x,y)$ such that:

$$
r_\phi(x,y_w) > r_\phi(x,y_l).
$$

A pairwise preference loss makes this trainable:

$$
\mathcal{L}_{RM} = -\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
$$

Do not only track accuracy. Also track the **margin**:

$$
\text{margin} = r_\phi(x,y_w) - r_\phi(x,y_l).
$$

If margins are tiny, PPO will receive a weak and noisy reward signal even if the ordering accuracy looks acceptable.

### The Spectrum of Rewards: Rules, Models, and Hybrids

Reward functions are not an either-or choice; they form a spectrum from "pure rules" to "pure model":

```mermaid
flowchart LR
    R1["Pure rule reward\n(regex matching)"] --> R2["Hybrid reward\n(rules + model)"]
    R2 --> R3["Pure model reward\n(RM scoring)"]

    R1 --- D1["✓ Deterministic\n✓ Zero cost\n✗ Format checks only"]
    R3 --- D2["✓ Semantic understanding\n✓ Broad coverage\n✗ Can be hacked"]

    style R1 fill:#e8f5e9,stroke:#2e7d32
    style R2 fill:#fff3e0,stroke:#e65100
    style R3 fill:#e3f2fd,stroke:#1565c0
```

**Pure rule rewards** suit tasks with objective ground truth: whether the final answer of a math problem is correct, whether code runs, whether the output format is valid. Such rewards are fully deterministic and cannot be "hacked", but they only check surface form and cannot assess semantic quality.

**Pure model rewards** are the reward model (RM) trained in this section: given $(prompt, response)$, it outputs a scalar score. An RM understands semantics -- it knows which of "helpful but blunt" and "polite but empty" is better. But an RM carries a fundamental risk: it is itself a model, and models can be exploited adversarially. This is the reward hacking problem covered later (Sections 13.6 and 13.7).

**Hybrid rewards** are the most common industrial choice -- the RM covers semantics, and rules cover dimensions the RM misses. A typical hybrid looks like:

$$R_{mix} = R_{RM} + \alpha \cdot R_{format} + \beta \cdot R_{length} + \gamma \cdot R_{correctness}$$

where $\alpha, \beta, \gamma$ are tunable hyperparameters. $R_{format}$ checks formatting, $R_{length}$ penalizes overly long or short answers, and $R_{correctness}$ checks questions with objective answers (math, code). The trade-offs:

|                    | Rule rewards                       | Model rewards (RM)                   |
| ------------------ | ---------------------------------- | ------------------------------------ |
| **Cost**           | Nearly zero                        | Training + inference cost            |
| **Reliability**    | Deterministic, cannot be hacked    | Can be exploited adversarially       |
| **Semantics**      | None, format checks only           | Understands content quality          |
| **Generalization** | Poor, new rules per task           | Good, one RM scores many tasks       |
| **Best for**       | Math/code/format with ground truth | Dialogue/creative/safety preferences |
| **Typical role**   | The "floor" of a hybrid reward     | The "body" of a hybrid reward        |

A practical rule of thumb: **use rule rewards wherever rules apply, and fill the gaps with model rewards**. Rules act as a safety net -- even if the RM is hacked, rules still guarantee basic format and correctness. In the RLVR setting of Chapter 15, rule rewards become the primary signal.

A reward model usually reuses a language model as its backbone and adds a scalar head to the hidden state of the final valid token. The same parameters score the chosen and rejected responses; their score difference supplies the training signal. The scalar score is meaningful only relative to the current model and data distribution.

### Training Configuration: Reward Models Overfit Easily

RM training uses several key hyperparameters, and the overall style is more conservative than SFT:

```python
# ==========================================
# Key RM training configuration
# ==========================================
rm_config = {
    # Backbone: usually a smaller SFT-era model
    "base_model": "sft_model_3b",

    # Learning rate: more conservative than SFT
    "learning_rate": 5e-6,  # SFT usually 1e-5 to 2e-5

    # Schedule: linear warmup + cosine decay
    "warmup_steps": 100,
    "lr_scheduler": "cosine",

    # Gradient clipping: prevents gradient explosions
    "max_grad_norm": 1.0,

    # Batch size: number of preference pairs
    "batch_size": 128,  # 128 (chosen, rejected) pairs per batch

    # Epochs: usually 1-2 only
    "epochs": 1,  # RMs overfit easily; do not train too long
}
```

RMs overfit easily -- preference datasets typically contain tens of thousands to hundreds of thousands of pairs, while the RM may have billions of parameters. One epoch is usually optimal; beyond two epochs, validation accuracy often starts to drop.

### Data Splitting: By Prompt, Not By Pair

A subtle pitfall in RM training is the data split. If one prompt has 6 candidate responses and the derived pairs are randomly scattered across train and eval, leakage occurs: both sets share the same prompt and possibly some responses, and eval accuracy becomes optimistic. Split by prompt instead:

```python
def split_by_prompt(items, eval_ratio=0.1):
    """
    items: [{"prompt_id": str, "prompt": str, "chosen": str, "rejected": str}, ...]
    """
    import random
    prompt_ids = sorted({item["prompt_id"] for item in items})
    random.shuffle(prompt_ids)

    n_eval = int(len(prompt_ids) * eval_ratio)
    eval_ids = set(prompt_ids[:n_eval])

    train, eval_ = [], []
    for item in items:
        if item["prompt_id"] in eval_ids:
            eval_.append(item)
        else:
            train.append(item)
    return train, eval_
```

A prompt-level split answers the real question: on unseen new prompts, does the RM's preference ordering still generalize?

### Reward Scale: Calibrate Before PPO

RM training only constrains score differences, not the absolute scale. One RM outputs $(2, 1)$, another $(20, 10)$; both order correctly, but PPO perceives completely different reward scales. This directly affects training stability:

| RM reward scale | What may happen in PPO                                               |
| --------------- | -------------------------------------------------------------------- |
| Too small       | The reward is drowned by the KL penalty; the actor barely moves      |
| Too large       | The reward dominates KL; the actor drifts from the reference quickly |
| Drifting        | Advantage estimates become unstable across batches                   |

A common fix is normalization against a fixed calibration set:

```python
class RewardNormalizer:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, reward):
        return (reward - self.mean) / (self.std + self.eps)
```

The mean/std should come from a fixed calibration set, not from the current training batch. Otherwise the reward scale drifts together with the actor's distribution, which makes debugging painful.

### What to Look At When Evaluating an RM

A reasonably complete RM report includes:

| Metric                    | Meaning                                   | Typical use                    |
| ------------------------- | ----------------------------------------- | ------------------------------ |
| Pairwise accuracy         | Is chosen scored higher on held-out pairs | Checks ordering ability        |
| Mean margin               | Average chosen-rejected score gap         | Checks signal strength         |
| Margin distribution       | Is the gap distribution healthy           | Finds "barely correct" samples |
| Reward-length correlation | Do scores depend too much on length       | Checks length-hacking risk     |
| Domain breakdown          | Accuracy/margin per task domain           | Finds weak domains             |
| Calibration samples       | Manually inspect high/low-scoring answers | Checks human plausibility      |

A lightweight helper:

```python
def rm_eval_metrics(r_chosen, r_rejected, chosen_lengths, rejected_lengths):
    import numpy as np

    margin = np.asarray(r_chosen) - np.asarray(r_rejected)
    accuracy = float((margin > 0).mean())

    rewards = np.concatenate([r_chosen, r_rejected])
    lengths = np.concatenate([chosen_lengths, rejected_lengths])
    length_corr = float(np.corrcoef(rewards, lengths)[0, 1])

    return {
        "pairwise_accuracy": accuracy,
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "length_reward_corr": length_corr,
    }
```

If `length_reward_corr` is high, inspect the preference data: is chosen systematically longer than rejected? If so, the RM may have learned "longer is better" rather than "more helpful is better".

### Reward Granularity and Credit Assignment

An RM assigns one score to one response. At what granularity should that score live -- one score per response, per reasoning step, or per token?

| Granularity    | Method                   | Strength                          | Weakness                             | Representative methods    |
| -------------- | ------------------------ | --------------------------------- | ------------------------------------ | ------------------------- |
| Sequence-level | One score per response   | Simple, stable                    | Cannot localize which parts are good | PPO, GRPO                 |
| Step-level     | Score per reasoning step | Balances fineness and feasibility | Needs a step splitter                | PRM (process supervision) |
| Token-level    | One score per token      | Finest granularity                | Costly, noisy signal                 | Early RLHF attempts       |

In practice, **sequence-level plus rule assistance** dominates: PPO and GRPO assign one reward per response by default, then rule rewards add token-level signals (format checks, for instance). Step-level rewards are process reward models (PRMs), covered in Chapter 17.

Sequence-level RMs face a fundamental difficulty: they score only after the response ends. Suppose the model generated 200 tokens and the RM returns a low score -- which segment caused it? A misread of the user's intent at the start, a skipped reasoning step in the middle, a wrong final answer, or an overconfident tone -- the single score cannot tell. This is one reason PPO-RLHF keeps a Critic and advantage estimation: they cannot fully solve credit assignment, but they smooth the "is this whole response good" signal back down to token-level updates (see GAE in Section 8.3). GRPO, RLVR, and process rewards later all attack this same problem from different directions.

### Adversarial Testing: Do Not Hand a Raw RM to PPO

The biggest risk in reward design is not low RM accuracy but systematic blind spots. During PPO, the actor actively searches the output distribution for whatever makes the RM score high -- if the RM favors some surface pattern, the actor will push that pattern to the extreme. So stress-test the RM before PPO:

```python
stress_cases = [
    ("empty answer", ""),
    ("long padding", "This question is very important. " * 200),
    ("fixed template", "Sure. Here are some suggestions:\n" * 50),
    ("confident but wrong", "PPO is a deterministic search algorithm proposed in 1980."),
    ("correct but terse", "PPO clips updates to keep new and old policies close."),
]

for name, response in stress_cases:
    print(name, reward_model.score(prompt, response))
```

If "long padding" outscores "correct but terse", do not start PPO yet -- PPO will only amplify the problem.

The Step 2 deliverable should therefore include held-out preference accuracy, the margin distribution, reward–length correlation, and a manually inspected set of high- and low-scoring answers. Remember the acceptance bar: **the reward model must both order pairs correctly and avoid treating length, boilerplate, or false confidence as quality.**

Collect the checks into one practical checklist to walk through when designing your own reward function:

| Check item         | Question                                                  | Pass criterion                              |
| ------------------ | --------------------------------------------------------- | ------------------------------------------- |
| Reward granularity | Which granularity did you choose, and why?                | An explicit justification exists            |
| Hybrid rewards     | Do you combine rule and model rewards?                    | At least one rule reward as the floor       |
| Length penalty     | Is there a mechanism against over-long answers?           | An explicit length penalty term             |
| Repetition penalty | Is there a mechanism against repetitive filler?           | An n-gram repetition check                  |
| RM separation      | Is the chosen/rejected gap large enough?                  | Mean margin > 1.0                           |
| RM overfitting     | Does the RM hold up on validation data?                   | Validation accuracy > 65%                   |
| Edge cases         | How does the reward treat empty or extremely long inputs? | Edge cases handled explicitly               |
| Score calibration  | Is the RM output scale suitable for PPO?                  | Stable mean/variance on a fixed set         |
| Domain breakdown   | Is performance consistent across task domains?            | No obvious weak domain or safety regression |

::: details Advanced: a minimal PyTorch reward model

```python
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """Map a prompt-response sequence to one scalar reward."""

    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Production code should locate the last valid token from attention_mask.
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.reward_head(last_hidden).squeeze(-1)


def preference_loss(
    reward_model,
    chosen_ids,
    chosen_mask,
    rejected_ids,
    rejected_mask,
):
    chosen_reward = reward_model(chosen_ids, chosen_mask)
    rejected_reward = reward_model(rejected_ids, rejected_mask)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()
```

This example shows the parameter flow. A production implementation must also handle padding, distributed batches, prompt-level splits, and reward calibration.

:::

## Step 3: PPO-RLHF Optimizes the Policy Under Constraints

In PPO-RLHF, you typically run four components:

| Role         | Source                       | Trained? | Purpose                            |
| ------------ | ---------------------------- | -------- | ---------------------------------- |
| Actor        | initialized from SFT         | yes      | generate responses and get updated |
| Reference    | frozen SFT checkpoint        | no       | KL anchor: "do not drift too far"  |
| Reward model | trained on preferences       | no       | provide scalar reward signal       |
| Critic       | often initialized from Actor | yes      | estimate value to reduce variance  |

The total reward is typically written as:

$$
R_{total}(x,y)
= r_\phi(x,y)
- \beta D_{KL}(\pi_\theta(\cdot\mid x)\|\pi_{ref}(\cdot\mid x))
$$

It captures the core tension of RLHF:

- The RM wants the Actor to move toward responses that better match preferences.
- The Reference wants the Actor to stay close to the SFT policy.
- PPO wants each update step to be moderate.

Without the KL penalty, the Actor may quickly exploit blind spots in the RM. If the KL penalty is too strong, the Actor can barely learn at all.

## Where Feedback Comes From

The H in classic RLHF stands for human feedback, but in real engineering the feedback source is usually a mixture:

| Source            | Use case                                      | Risk                                     |
| ----------------- | --------------------------------------------- | ---------------------------------------- |
| Human annotation  | High-quality seed data, final calibration     | Expensive, slow, limited consistency     |
| AI Judge / RLAIF  | Scaling preference data, fast iteration       | Amplifies judge biases                   |
| Rule verification | Math, code, format and other verifiable tasks | Cannot cover open-ended dialogue quality |
| Online feedback   | Likes, dislikes, copy, edit-and-resend        | Noisy, requires aggregation              |

This chapter still uses classic human preference as the main thread, but introduces AI Judges, rule checks, and manual review in data engineering and evaluation. This preserves the standard InstructGPT structure without reducing the course to an outdated purely-manual annotation workflow.

## RLAIF, CAI, and Self-Play

RLAIF, CAI, and Self-Play all supplement or replace human feedback. They answer the same question: **where does preference data come from, and how do we iterate faster?**

| Method             | Pipeline position                             | Purpose                                                           | Guardrails needed                                |
| ------------------ | --------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| RLAIF              | Generate preference pairs / RM training set   | Replace some human annotation with a strong model                 | Human spot-checks, judge consistency review      |
| Constitutional AI  | Generate chosen/rejected                      | Self-critique and self-revision by principles                     | Constitution quality, human calibration          |
| Self-Play / Debate | Generate candidate answers and hard negatives | Let the model compete against past versions                       | Diversity monitoring, external eval anchors      |
| Self-Rewarding     | Multi-round data flywheel                     | Model self-evaluates, self-critiques, self-revises, then retrains | External RM or human eval to prevent degradation |

The key insight here is not "replace humans entirely" but **use AI to scale and use humans to calibrate direction**. If you rely entirely on an AI Judge, when the judge favors verbose responses, fixed templates, or a particular style, those biases get amplified in the next training round.

A minimal RLAIF judge prompt might look like this:

```python
rlaif_judge_prompt = """
You are a strict answer quality evaluator. Compare two responses.

Evaluation dimensions:
1. Accuracy: Are the facts correct, any hallucinations?
2. Helpfulness: Does it genuinely address the user's question?
3. Clarity: Is the writing clear and the logic coherent?
4. Safety: Does it contain harmful, biased, or misleading content?

User question:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Output only JSON:
{{"winner": "A" or "B" or "tie", "reason": "one-sentence reason"}}
"""
```

To reduce judge bias, do at least four things:

1. Randomly swap A/B order.
2. Record the judge's reasoning, not just the winner.
3. Periodically run human spot-checks.
4. Keep a fixed evaluation set so the data flywheel does not merely appease the current judge.

## Where Does the Data Flywheel Fit

The data flywheel is not a separate algorithm. It connects SFT, RM, PPO, and evaluation into an iterable system:

```text
Deploy model
  -> Collect bad cases, user feedback, evaluation failures
  -> Produce new SFT / preference data
  -> Train SFT or RM
  -> PPO-RLHF update policy
  -> Evaluate; deploy again if it passes
```

Key metrics for this flywheel include iteration cycle time, data utilization rate, evaluation coverage, and rollback rate. In a small-parameter course experiment you can compress it into a single round: prepare fixed data, run SFT/RM/PPO, then use evaluation results to decide what data to supplement next round.

Whether the flywheel keeps spinning better depends primarily on quality gates, not on "how much data was generated."

| Quality gate              | What it checks                                             | Typical practices                                                   |
| ------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------- |
| Basic cleaning            | Duplicates, contamination, format errors, length anomalies | Dedup, eval-set leak check, length filtering, format validation     |
| Difficulty stratification | Whether data sits at the model's learning boundary         | Use pass@k or judge scores to split too easy / learnable / too hard |
| Preference consistency    | Whether chosen is genuinely better than rejected           | Multi-judge voting, human spot-checks                               |
| Online regression         | Whether the new model breaks old capabilities              | Fixed benchmark + badcase replay                                    |

## Minimal Experiment Directory

For reproducibility, this chapter recommends storing artifacts separately:

```text
experiments/rlhf-smollm/
  data/
    sft_train.jsonl
    pref_train.jsonl
    prompts_ppo.jsonl
    eval_prompts.jsonl
  models/
    base.txt
    sft/
    reward_model/
    rlhf/
  reports/
    base_probe.md
    sft_eval.json
    rm_eval.json
    ppo_train_metrics.jsonl
    final_eval.md
```

This is not bureaucracy. When debugging RLHF you will often ask:

- Which RM was used for this PPO run?
- Which preference data was this RM trained on?
- Did evaluation prompts leak into the training data?
- From which checkpoint did the model start getting verbose?

If artifacts are unclear, these questions become hard to answer later.

## Common Failure Mode Map

| Location | Failure symptom                        | Root cause                                                    | What to check first                          |
| -------- | -------------------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| Base     | Output does not look like an assistant | Pretraining objective was not instruction-following           | Base probe samples                           |
| SFT      | Correct format but empty content       | Demonstration data is low-quality or homogeneous              | SFT data manual sampling                     |
| RM       | Prefers long responses                 | Chosen responses are systematically longer in preference data | Reward-length correlation                    |
| PPO      | Reward rises but quality drops         | Actor found RM blind spots                                    | High-reward sample spot-checks               |
| Eval     | Win rate fluctuates heavily            | Judge bias or too few samples                                 | Random seed, A/B order, confidence intervals |

## Section Summary

The standard RLHF pipeline can be compressed into three sentences:

1. SFT turns a base model into an assistant starting point.
2. The Reward Model converts preference data into an optimizable reward signal.
3. PPO increases the probability of high-reward responses under a KL constraint.

But reliable RLHF is not just these three training steps. It also includes artifact management, data quality gates, and an evaluation closed loop. Data construction for the first stage is covered in Section 13.2: how SFT data and preference data are constructed and why they have a natural relationship to imitation learning and inverse reinforcement learning -- [SFT: Teaching the Model to Answer Instructions](./imitation-learning-pipeline).

## Exercises

1. Design an `sft_item` and a `preference_item` with the same prompt but different data purposes.
2. Explain why high RM accuracy does not guarantee success in the PPO stage.
3. In one sentence, describe the role of the Reference model in PPO-RLHF.
