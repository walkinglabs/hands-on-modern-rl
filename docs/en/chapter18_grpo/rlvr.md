---
title: '15.3 RLVR: Verifiable Rewards'
---

# 15.3 Hands-on: Building RLVR Rewards

<ChapterNotebookLink chapter="grpo" locale="en" />

> **Section goal**: Implement a verifiable reward for mathematical reasoning, connect it to a minimal GRPO loop, and test whether format, answer, and total rewards behave as intended.

> **Learning path**: [15.1 How GRPO Training Works](./grpo-practice-and-mechanism) → [15.2 DeepSeek-R1-Zero and DAPO](./deepseek-dapo) → **15.3 Building RLVR Rewards**

> **Code and resources**: [rule-based reward](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter09_grpo_rlvr/rule_based_reward.py) · [GRPO math reasoning](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter09_grpo_rlvr/grpo_math_reasoning.py) · [dependencies](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter09_grpo_rlvr/requirements.txt)

GRPO and DAPO both depend on a reliable reward signal. Mathematics provides a clear starting point: the model may write different reasoning paths, while the final answer can still be checked by a deterministic rule. Reinforcement Learning with Verifiable Rewards (RLVR) uses that check as the training signal.

Run the verifier before starting model training:

```bash
cd code/chapter09_grpo_rlvr
pip install -r requirements.txt
python rule_based_reward.py
python grpo_math_reasoning.py
```

The first script checks reward behavior without a model. The second script requires model dependencies and compute; its result must be evaluated on held-out problems.

## 15.3.1 The Core Idea of RLVR

Traditional RLHF depends on a Reward Model to provide the training signal. The RM needs human-labeled preference pairs, has noise, is vulnerable to reward hacking, and is expensive to label. RLVR starts from a simple question: **if the task itself has objective criteria for correctness, why train a noisy RM?**

```mermaid
flowchart TD
    subgraph rlhf ["RLHF flow"]
        R1["Prompt"] --> R2["Model generates answer"]
        R2 --> R3["Reward Model scoring\n(requires RM training)"]
        R3 --> R4["Update model"]
    end

    subgraph rlvr ["RLVR flow"]
        V1["Prompt"] --> V2["Model generates answer"]
        V2 --> V3["Verifier checks\n(automatic and reproducible)"]
        V3 --> V4["Update model"]
    end

    R3 -.->|"Costs: expensive labels, noise, easy to hack"| COST1["Label cost\nRM noise\nReward hacking"]
    V3 -.->|"Benefits: repeatable checks"| COST2["No preference labels\nExplicit criteria\nAuditable output"]

    style R3 fill:#fce4ec,stroke:#c62828
    style V3 fill:#e8f5e9,stroke:#2e7d32
```

### Compared with Traditional RLHF

| Aspect                 | RLHF                                                    | RLVR                                           |
| ---------------------- | ------------------------------------------------------- | ---------------------------------------------- |
| **Data cost**          | Extremely high; requires human-labeled preference pairs | Very low; automatic verification               |
| **Reward quality**     | Noisy because human judgment is subjective              | Precise because correctness is objective       |
| **Scalability**        | Limited by labeling speed                               | Almost unlimited                               |
| **Scope**              | Subjective preferences, such as politeness and safety   | Objective tasks, such as math, code, and logic |
| **Training stability** | Affected by RM quality                                  | Very stable because the reward signal is clear |
| **Risk of hacking**    | High; the model may exploit RM loopholes                | Low; rules are hard constraints                |

## 15.3.2 Verifiers: The Key Design in RLVR

Different domains use different verification methods:

| Domain                   | Verification method | Example                            |
| ------------------------ | ------------------- | ---------------------------------- |
| Mathematics              | Answer matching     | `\boxed{42}` == ground truth       |
| Code                     | Unit tests          | Execute code + test case pass rate |
| Logical reasoning        | Formal verification | Lean/Coq theorem provers           |
| Multilingual translation | Automatic scoring   | BLEU/COMET score                   |

Verifier design is the key to RLVR. A good verifier needs three properties: **determinism** (the same input always receives the same result), **correctness** (the verifier's judgment truly reflects answer quality), and **efficiency** (verification must be fast enough not to become the training bottleneck).

Among these, "correctness" is the subtlest requirement. Consider math answer matching. If the ground truth is $\frac{22}{7}$ and the model answers $3.1428...$, is it correct? If the ground truth is $(x+1)(x-2)$ and the model answers $x^2 - x - 2$, is it correct? These boundary cases require careful verifier design. In practice, math verifiers usually perform numerical comparison with tolerance and symbolic simplification so that equivalent representations can be handled.

## 15.3.3 1-Shot RLVR: RL with Minimal Data

More surprisingly, ICLR 2025 research showed that RLVR can work with **only 1 training sample**. Researchers found that even when the training set contains only one math problem, RL training can still improve the model on many unseen problems.

This suggests that RLVR effectiveness does not mainly depend on data volume. The model has already learned reasoning ability during pretraining; RLVR's role is to "unlock" those latent abilities rather than teach them from scratch. It is like a student who already understands the math concepts but has never been forced to solve problems. RL training acts like an exam: it pressures the student to organize existing knowledge into usable problem-solving strategies.

This finding has important practical implications. It means the cost-performance ratio of RLVR training can be very high. You do not need massive data to obtain a significant gain. The key is not simply the amount of data, but the training setup: reward function design, group size, number of training steps, and so on. It also explains why DeepSeek-R1-Zero could succeed: even without any SFT data, if the RL training setup is correct, the model can "evolve" reasoning behavior by itself.

<details>
<summary>Reflection question: if RLVR can work with only 1 sample, why does practical training still need large datasets?</summary>

One sample can "start" the training process, but making the model perform well across diverse scenarios still requires training data with different types and difficulty levels. The reasons include:

- **Generalization**: with only 1 sample, the model may improve only in the "neighborhood" of that problem. Covering broad problem types requires diverse data.
- **Avoiding overfitting**: with too little data, the model may memorize a specific solution path rather than learn a general reasoning strategy.
- **Statistical stability**: success on 1 sample can be accidental. Averaging over a large dataset helps ensure the training direction is correct.

The real meaning of 1-Shot RLVR is theoretical: it shows that the value of RL is not "injecting new knowledge", but "activating existing ability". This changes how we understand the role of RL in LLM training.

</details>

## 15.3.4 Hands-on: A Minimal RLVR Training Implementation

The discussion above clarifies the concept of RLVR. Now we turn it into a concrete implementation with runnable code.

Specifically, we will train a 0.6B Qwen3 model on the MATH dataset. Given a math problem, the model generates a reasoning process and final answer, the verifier checks whether the answer is correct, and GRPO updates the model. The full implementation stays under 200 lines and can run on a single GPU.

This implementation follows the RLVR GRPO script from Chapter 6 of Sebastian Raschka's [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) project. It reconstructs the core structure of RLVR training with minimal code. Our goal is not merely to produce a runnable script, but to **understand how the structure of an RLVR training system naturally follows from "verifiable rewards + GRPO"**.

### What the RLVR Training Loop Looks Like

The RLVR training loop is the same as traditional GRPO, except that rewards come from a verifier rather than an RM:

```mermaid
flowchart TD
    subgraph Rollout ["Rollout: sample multiple answers"]
        P["Math problem prompt"]
        M["Model pi_theta"]
        P --> M
        M -->|"sample 1"| R1["Answer 1"]
        M -->|"sample 2"| R2["Answer 2"]
        M -->|"..."| R3["Answer G"]
    end
    subgraph Reward ["Reward: verifier scoring"]
        R1 --> V["Verifier\nextract + grade"]
        R2 --> V
        R3 --> V
        V -->|"correct=1, wrong=0"| Scores["reward_1, reward_2, ..., reward_G"]
    end
    subgraph Train ["Train: GRPO update"]
        Scores --> Adv["advantage = (r - mean) / std"]
        Adv --> Loss["pg_loss = -(advantage * log_prob).mean()"]
        Loss --> Update["Gradient descent theta -> theta'"]
    end
    Update -->|"next round"| M
```

Concretely:

- **Rollout phase**: for each math problem, the model samples $G$ answers under the current policy $π_θ$, for example $G=4$. Each answer contains a reasoning process and a final answer in `\boxed{}` format.
- **Reward phase**: the verifier extracts the answer inside `\boxed{}` and compares it with the ground truth. A correct answer gets reward 1; a wrong answer, or an answer that cannot be extracted, gets reward 0.
- **Train phase**: use GRPO within-group normalization to compute the advantage, then perform a policy gradient update.

### Verifier: Extract and Grade the Answer

The "verifiable" part of RLVR lives in the verifier. A math verifier does two things: extract the final answer from the model output and compare it with the ground truth.

```python
import re

def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer inside \\boxed{...} from model output.

    The model is trained to mark the final answer with \\boxed{} at the end
    of its reasoning process. If extraction fails, return None (reward = 0).
    """
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return None

def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Judge whether the predicted answer is correct.

    Simplified version: direct string comparison + numerical comparison.
    Production verifiers handle equivalent forms such as fraction reduction
    and polynomial expansion.
    """
    predicted = predicted.strip().replace(" ", "")
    ground_truth = ground_truth.strip().replace(" ", "")
    if predicted == ground_truth:
        return True
    # Try numerical comparison, such as "22/7" vs "3.1428...".
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        return False

def reward_rlvr(response: str, ground_truth: str) -> float:
    """RLVR reward function: extract answer + judge correctness.

    This is the core of RLVR: no RM, no human labels, only one rule that
    provides an exact 0/1 reward.
    """
    predicted = extract_boxed_answer(response)
    if predicted is None:
        return 0.0  # If no answer can be extracted, reward is 0.
    return float(grade_answer(predicted, ground_truth))
```

Design points:

- `extract_boxed_answer()` only accepts the `\boxed{}` format. If the model does not follow the format, reward is directly 0. This itself is a training signal that forces the model to learn the required output format.
- `grade_answer()` first performs string matching and then numerical comparison. A production verifier, such as the one used in [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch), uses more complex `sympy` equivalence checking, but the core logic is the same.
- The whole verification process is deterministic, reproducible, and zero-cost in labels. This is the essential advantage of RLVR over RM-based training.

### One-Update Limit of Original GRPO

With the verifier in place, the next step is to connect "sample multiple answers -> compute rewards -> GRPO update" into a training loop.

```python
import torch
import torch.nn.functional as F


def compute_grpo_loss(model, tokenizer, prompt, ground_truth,
                      device, num_rollouts=4, max_new_tokens=512,
                      temperature=1.0):
    """One online update of the original GRPO policy term.

    Args:
        model: policy model
        tokenizer: tokenizer
        prompt: math problem prompt
        ground_truth: ground truth answer
        num_rollouts: number of sampled answers per problem (GRPO group size)
        max_new_tokens: maximum generation length
        temperature: sampling temperature

    Returns:
        dict: training statistics including loss, rewards, and advantages
    """
    roll_rewards, rollout_data = [], []

    # ==================== Phase 1: Rollout ====================
    # Sample num_rollouts independent answers for the same problem.
    with torch.no_grad():
        for _ in range(num_rollouts):
            input_ids = torch.tensor(
                tokenizer.encode(prompt), device=device
            ).unsqueeze(0)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            # Extract generated part only, excluding the prompt.
            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            # Compute reward with the verifier: correct=1, wrong=0.
            reward = reward_rlvr(response, ground_truth)
            roll_rewards.append(reward)
            rollout_data.append((output_ids[0], input_ids.shape[1]))

    # ==================== Phase 2: GRPO Advantage ====================
    # Core idea: normalize multiple answers to the same problem inside the group.
    # advantage = (reward - mean) / std
    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # If all advantages are 0, meaning all answers are correct or all are wrong,
    # skip the update.
    if torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-8):
        return {"loss": 0.0, "loss_tensor": None, "rewards": roll_rewards}

    # ==================== Phase 3: Compute log prob ====================
    roll_logps = []
    for token_ids, prompt_len in rollout_data:
        logits = model(token_ids.unsqueeze(0)).logits.squeeze(0).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        # Keep only the log probabilities of the response tokens.
        targets = token_ids[1:]
        selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
        # The paper averages tokens inside each response before averaging responses.
        roll_logps.append(selected[prompt_len - 1:].mean())

    logps = torch.stack(roll_logps)

    # ==================== Phase 4: Policy gradient loss ====================
    # pg_loss = -(advantage * log_prob).mean()
    # Answers with positive advantage become more likely; negative ones become less likely.
    pg_loss = -(advantages.detach() * logps).mean()

    return {
        "loss": pg_loss.item(),
        "loss_tensor": pg_loss,
        "rewards": roll_rewards,
        "advantages": advantages.tolist(),
    }


def train_rlvr(model, tokenizer, train_data, device,
               steps=100, num_rollouts=4, lr=1e-5, **kwargs):
    """Main RLVR training loop.

    Args:
        train_data: list of examples, each containing "problem" and "answer"
        steps: number of training steps
        num_rollouts: GRPO group size
        lr: learning rate
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(steps):
        example = train_data[step % len(train_data)]
        prompt = (
            f"Solve the following problem. Put your final answer within "
            f"\\boxed{{}}.\n\nProblem: {example['problem']}"
        )

        stats = compute_grpo_loss(
            model, tokenizer, prompt, example["answer"],
            device, num_rollouts=num_rollouts, **kwargs,
        )

        if stats["loss_tensor"] is not None:
            optimizer.zero_grad()
            stats["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        reward_avg = sum(stats["rewards"]) / len(stats["rewards"])
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} | loss={stats['loss']:.4f} | "
                  f"reward_avg={reward_avg:.3f}")

    return model
```

Design points:

- `compute_grpo_loss()` shows the one-update limit of the original GRPO policy term. At the forward-pass value, the current and old policies coincide, so the ratio is 1 and clipping is inactive; the ratio still has the gradient of the token log probability. The code therefore writes the same gradient directly as group advantage times mean token log probability.
- **The reward comes from the verifier, not an RM.** `reward_rlvr()` only extracts and compares answers. It has no trainable parameters, but an incomplete parser or equivalence check can still be exploited.
- **All-zero advantage is skipped.** If every rollout for a problem is correct or every rollout is wrong, all advantages are 0 and the gradient is also 0. Skipping the update saves compute, especially early in training when the model is weak and most problems receive all-zero rewards.
- The Reference KL term is omitted here to keep the verifier example focused. This is not the complete [DeepSeekMath Equations (3) and (4)](https://arxiv.org/html/2402.03300#S3.SS1); the complete original objective also includes token-wise ratio, clipping, and KL with $\beta=0.04$, as shown in the preceding code map.

### Running It

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Start with a small model. Measure memory under the selected sequence length and batch size.
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MATH training data (example format)
train_data = [
    {"problem": "What is the value of $x$ if $2x + 3 = 11$?",
     "answer": "4"},
    {"problem": "Compute $\\sum_{k=1}^{10} k$.", "answer": "55"},
    # ... more problems
]

model = train_rlvr(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    device=model.device,
    steps=100,
    num_rollouts=4,
    lr=1e-5,
    max_new_tokens=512,
)
```

### Gap from Production Implementations

The implementation above runs the minimal RLVR + GRPO loop. Compared with the production script in [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) and frameworks such as veRL and OpenRLHF, the main gaps are:

| Aspect               | Minimal implementation in this section      | Production RLVR training                                             |
| -------------------- | ------------------------------------------- | -------------------------------------------------------------------- |
| Verifier             | String matching + numerical comparison      | `sympy` equivalence checking, LaTeX parsing, multiple format support |
| Sampling engine      | Generate one by one with `model.generate()` | Continuous batching, KV cache, vLLM/SGLang                           |
| GRPO variant         | No KL penalty; simplest version             | Clip, KL penalty, length reward, Dr. GRPO, and other improvements    |
| Distributed training | Single GPU                                  | FSDP / Megatron, multi-GPU, gradient accumulation                    |
| Evaluation           | Mean reward during training                 | Standard benchmarks such as MATH-500, periodic checkpoint + eval     |
| Memory optimization  | None                                        | Gradient checkpointing, sequence truncation, zero-advantage skipping |

Each gap is an independent optimization direction. Chapter 7 of [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) discusses several GRPO improvements, including Olmo3 fixes, DeepSeek-V3.2 fixes, and GDPO, and provides systematic comparisons on MATH-500.

## 15.3.5 Limitations and Debates Around RLVR

RLVR is not a universal solution. It has several important limitations:

1. **It only applies to domains with objective answers.** Mathematics, code, and logical reasoning have clear correctness criteria. Subjective preferences such as "more polite", "more creative", and "safer" cannot receive precise reward signals from RLVR. These domains still require RMs or preference data.

2. **The verifier may be hacked.** Even when rewards are generated by rules, the model may still find shortcuts that satisfy the rule without genuine understanding. In math, for example, the model may learn a special trick that passes a particular verifier for certain problem types but fails under a rephrased question.

3. **Does RLVR really improve reasoning ability?** This is the sharp question raised by a NeurIPS 2025 oral paper. The authors questioned whether RLVR improves true reasoning ability or merely improves search efficiency, meaning the model becomes better at finding correct answers during inference. This remains an open frontier debate.

---

GRPO removes the Critic from the policy side, and RLVR removes the RM from the reward side. Together, they compress the complexity of RL training to an extreme degree. But the RL story does not end here. The next exciting directions are RL Scaling and test-time scaling. In [Chapter 26](../chapter32_selfplay/rl-scaling-outlook), we will look at these frontier directions: RL Scaling and the future outlook.

## Section Summary

- RLVR replaces a learned reward model with an explicit verifier when correctness can be checked.
- A verifier is only as reliable as its parser, equivalence rules, and test coverage.
- Training reward must be paired with held-out evaluation and adversarial verifier tests.
