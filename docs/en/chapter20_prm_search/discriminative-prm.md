# 18.2 The Discriminative PRM Approach

Discriminative PRM is the first PRM approach that was systematically studied. Its idea is the most direct one: model "judging whether a reasoning step is right or wrong" as a **classification task** — input a prompt plus a step, output the probability that "this step is correct."

This section walks through OpenAI's [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al. 2023) as the main thread, covering the method, dataset, applications, and limitations of discriminative PRM.

## 11.2.1 OpenAI's Process Supervision Research

In May 2023, OpenAI published [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). The paper's motivation:

- GPT-4 had made a breakthrough in mathematical reasoning, but still often made mistakes in multi-step reasoning
- ORM (Outcome Reward Model) can only score the final answer — it can't locate which step went wrong
- An intuitive fix: **have humans label reasoning steps one by one, and train a PRM on that**

OpenAI compared three supervision schemes:

| Supervision scheme      | Labeling target                    | Labeling cost |
| ----------------------- | ---------------------------------- | ------------- |
| **ORM** (Outcome RM)    | correctness of the final answer    | low           |
| **PRM** (Process RM)    | correctness of each reasoning step | high          |
| **PRM800K** (full-step) | 800K step-level annotations        | very high     |

Experimental results (on the MATH dataset):

- ORM: 53.9%
- PRM (small amount of step labeling): 56.6%
- **PRM800K** (full step labeling): **78.2%**

This is a **substantial improvement** — with the same base model, refining the supervision signal alone lifted accuracy from 53.9% to 78.2%. That result is the proof of PRM's value.

## 11.2.2 The PRM800K Dataset

[PRM800K](https://github.com/openai/prm800k) is OpenAI's public step-level annotation dataset. Its scale:

- **800K** step-level annotations
- drawn from the solution traces of **75K** math problems
- each step labeled as one of three states:
  - **good** (correct): the reasoning is sound and can continue
  - **bad** (incorrect): the reasoning is flawed and should stop or backtrack
  - **neutral** (neutral): possibly a transitional step, no verdict given

The labeling pipeline:

1. Have a base model generate a solution trace
2. Split the solution trace into "steps" (by newline, sentence period, or explicit reasoning markers)
3. Have human annotators judge each step as good / bad / neutral
4. Collect 800K annotations

### The Cost of PRM800K

OpenAI never disclosed PRM800K's annotation cost, but it can be estimated from scale:

- 800K steps × an average of 30 seconds/step = 24,000 hours (roughly 12 person-years)
- at a US ML-annotator hourly rate of $30-50, the cost comes out to roughly $720K-$1.2M

Industrial labs (OpenAI, Anthropic, Google) can absorb this cost, but it is nearly infeasible for academic labs and small companies. This is the core bottleneck of discriminative PRM — **the annotation cost is too high**.

## 11.2.3 Training a Discriminative PRM

Given PRM800K, training a discriminative PRM is a standard classification task.

### Model Architecture

The architecture OpenAI used (in the paper):

- Base: GPT-4's pretrained backbone (or a smaller fine-tuned version)
- Input: `<prompt> <response_up_to_step_i> <step_i>`
- Output: a 3-way classification (good / bad / neutral)

Follow-up work (such as [Math-Shepherd](https://arxiv.org/abs/2312.08935)) uses a smaller base model (LLaMA-7B, Qwen-7B) with a classification head on top.

### Training Objective

Cross-entropy loss:

$$\mathcal{L}_{\text{PRM}} = -\sum_{i} \sum_{c \in \{good, bad, neutral\}} y_{i,c} \log p_\theta(c | q, o_{\leq i}, s_i)$$

Here $y_{i,c}$ is the one-hot encoding of the true label for step $i$, and $s_i$ is the content of step $i$.

### Training Data Augmentation

PRM800K's 800K annotations alone aren't enough to train a strong PRM. Common augmentation strategies:

- **Auto-labeling**: use GPT-4 to automatically label unannotated steps (noisy)
- **Synthetic data**: generate steps that "look wrong" as negative samples, starting from solution traces known to be correct
- **Data mixing**: merge PRM800K with other datasets such as Math-Shepherd

## 11.2.4 PRM as a Re-ranking Model

The main application of PRM isn't training RL directly — it's **re-ranking**.

The re-ranking workflow:

1. **Generate**: have the base model produce N candidate solutions for a math problem (N is typically 4-64)
2. **Score**: use the PRM to score every step of every candidate solution, producing a total score for each full solution trace
3. **Select**: pick the candidate with the highest total score as the final answer

OpenAI's experiments show that **re-ranking with PRM performs far better than single-shot generation plus ORM**:

| Method                          | MATH accuracy |
| ------------------------------- | ------------- |
| single-shot generation (greedy) | ~40%          |
| single-shot generation + ORM    | ~50%          |
| **N=64 + PRM re-rank**          | **78.2%**     |

What this shows is that PRM's value isn't confined to "supplying a dense reward during training" — it's at least as useful for "selecting the best candidate at inference time."

### Token-Level vs. Step-Level Re-ranking

PRM re-ranking can score in two ways:

**Token-level**: one score per token, and the total score for the whole response is some aggregate (mean, sum, min) of all the token scores.

**Step-level**: one score per reasoning step, and the total score for the whole response is some aggregate of all the step scores.

OpenAI used step-level scoring — it matches the human intuition of judging "is this step right or wrong" much more closely.

Choice of aggregation:

- **Mean**: the average of all step scores. Robust, but it can dilute the impact of one critical wrong step
- **Min**: the lowest of all step scores. Conservative — it leans toward "if any one step is wrong, veto the whole trace"
- **Product**: the product of all step scores. Stricter than min

OpenAI's experiments found that **min** works best on math tasks — if one step is wrong, the whole reasoning trace becomes untrustworthy. On tasks like creative writing, mean tends to work better.

## 11.2.5 Using Discriminative PRM in RL Training

Re-ranking is an inference-time application. PRM can also be used in RL training — by treating the PRM's step-level scores as a dense reward for RL.

Concretely:

```python
# Use a PRM as the reward function for RL training
def prm_reward(prompt, response):
    # Split the response into reasoning steps
    steps = split_into_steps(response)

    # Score each step
    step_scores = [prm(prompt, steps[:i+1]) for i in range(len(steps))]

    # Aggregate into a single reward for the whole response
    # Choosing min: any one wrong step vetoes the trace
    return min(step_scores)
```

This reward function can substitute for the ORM in RLHF, feeding into PPO / GRPO training.

[Math-Shepherd](https://arxiv.org/abs/2312.08935) is the flagship example of this approach. It trained LLaMA on GSM8K and MATH using a PRM reward, and beat an ORM-reward-trained baseline by 5-10 percentage points.

## 11.2.6 The Limitations of Discriminative PRM

Discriminative PRM works, but it has several fundamental limitations.

### Annotation Cost Explosion

PRM800K's annotation cost runs into the millions of dollars. For a new domain (code generation, biomedical reasoning, say), labeling a PRM800K-scale dataset from scratch is close to infeasible.

This is the biggest obstacle to scaling discriminative PRM beyond its original domain — **every new domain needs its own annotation effort from zero**.

### Weak Generalization

Discriminative PRM performs well on its training domain but generalizes poorly across domains. A PRM trained on math data, for instance, sees a sharp drop in performance when applied to code generation — what it has learned about "what counts as a good reasoning step" is domain-specific.

[Generative PRM](./generative-prm) (the next section) generalizes better precisely because it evaluates in natural language.

### Noisy Annotations

Even with PRM800K, the annotations aren't 100% accurate:

- Different annotators can judge the same step differently
- Correctness for complex reasoning steps is inherently somewhat subjective
- The boundary of the neutral category is fuzzy

[Lightman et al.](https://arxiv.org/abs/2305.20050) acknowledge the annotation-noise problem in the paper and mitigate it with an ensemble.

### Fixed Step Segmentation

Discriminative PRM needs the response split into "steps" — but how to split is itself a problem:

- splitting on newlines: too mechanical, a single "step" may span multiple lines
- splitting on sentence periods: too fine-grained, a complete reasoning move may contain multiple sentences
- splitting with an LLM: introduces the cost of a new LLM call

Different segmentation choices noticeably change the PRM's judgments. This is where a lot of the engineering complexity of discriminative PRM comes from.

## 11.2.7 Industrial Practice with Discriminative PRM

Despite these limitations, discriminative PRM remains the mainstream approach in industry:

- **OpenAI o1/o3**: reportedly use a PRM internally to guide reasoning (undisclosed)
- **DeepSeek**: follows the Math-Shepherd approach, using PRM for RL training
- **Anthropic**: Claude's internal PRM (speculative)
- **Alibaba Qwen-Math**: uses PRM re-ranking to improve MATH benchmark scores

A few industry trends:

1. **PRM + ORM hybrids**: use PRM for process reward and ORM for outcome reward, combined with a weighted sum
2. **Domain-specialized PRMs**: train separate PRMs for different tasks (a math PRM, a code PRM, a biomedical PRM)
3. **PRM auto-labeling**: use a strong LLM (GPT-4, Claude) to automatically label steps, cutting human annotation cost

## Summary

Discriminative PRM is the classic PRM approach. OpenAI's [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) and the PRM800K dataset laid its foundation. Its core idea — **train a classifier to judge whether each reasoning step is correct** — is simple and direct, and it proved its value on math tasks.

But discriminative PRM has three fundamental bottlenecks: high annotation cost, weak cross-domain generalization, and sensitivity to step segmentation. These three bottlenecks are what motivate the next section's **generative PRM** approach — achieving better results with far fewer labels.
