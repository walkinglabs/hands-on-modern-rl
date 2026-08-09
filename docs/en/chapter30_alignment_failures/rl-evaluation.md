# 28.5 Evaluation Principles and Modern Evaluation Harnesses

> [Chapter 33](../chapter30_alignment_failures/modern-incidents) covered the Qwen3 data contamination incident — benchmark scores inflated by 15-25 percentage points. That incident exposed something bigger than a data problem: **the fragility of the entire RL evaluation methodology**. This chapter works through it systematically: what makes a benchmark design trustworthy? How do you detect contamination? How does prompt sensitivity distort conclusions? How do you evaluate long-horizon tasks and behavioral tasks? We close with industrial-grade evaluation harnesses and Anthropic's 2025 internal AI Research Eval Suite (34× human speedup).

## 35.1 Principles for Benchmark Design

A good RL benchmark has to satisfy five principles.

### Verifiability

The answer to every test sample **must be machine-decidable**. Formally: there exists a function $\text{Verify}: \mathcal{Y} \times \mathcal{Y} \to \{0, 1\}$ that, for any pair $(y_{\text{pred}}, y_{\text{gold}})$, deterministically returns right or wrong.

- **Math problems**: extract the final number and compare it against the reference answer ([GSM8K](https://arxiv.org/abs/2110.14168), MATH)
- **Coding problems**: run the code against test cases and check the pass rate ([HumanEval](https://arxiv.org/abs/2107.03374), MBPP, LiveCodeBench)
- **Logic problems**: verify with a SAT solver or theorem prover (MiniF2F, PutnamBench)

For tasks that aren't verifiable — open-ended writing, creative generation — you're stuck with human evaluation or RM evaluation, and neither is reliable.

### Coverage

A benchmark needs to cover the real distribution the model will actually encounter. Formally:

$$\mathcal{D}_{\text{test}} \sim P_{\text{real}}, \quad P_{\text{real}} \approx P_{\text{test}}$$

If $\mathcal{D}_{\text{test}}$ is skewed toward one type of problem, the model may fail badly on everything else. GSM8K is the classic counterexample — it's all elementary-school arithmetic, and a model scoring 90% on GSM8K tells you nothing about whether it can do college math.

### Difficulty Stratification

Evaluate by difficulty tier to avoid the trap where an average score masks extreme underlying performance:

```python
# stratified evaluation by difficulty
def stratified_eval(model, dataset):
    results = {"easy": [], "medium": [], "hard": []}
    for x, y in dataset:
        pred = model(x)
        difficulty = classify_difficulty(x)  # difficulty classifier
        results[difficulty].append(verify(pred, y))
    return {k: np.mean(v) for k, v in results.items()}
```

MATH splits its problems into Levels 1-5, and DeepSeek-R1 reports a score per level. A breakdown by tier reveals the shape of a model's competence far better than a single aggregate number does.

### Contamination Resistance

The test set has to be **kept strictly secret**, and the training data has to be checked for contamination against it. See 35.2 for details.

### Statistical Rigor

You can't just report "Model A scores 60% on MATH, Model B scores 55%" — that gap might just be sampling noise. You need:

- **Confidence intervals**: for $n$ test samples with accuracy $p$, the 95% CI is $p \pm 1.96\sqrt{p(1-p)/n}$
- **Paired t-tests**: comparing two models on the same test set
- **Bootstrap resampling**: resampling the test set to estimate variance

LLM evaluation papers ignored statistical significance for a long time; it only became broadly accepted after 2024 ([Blackwell et al., arXiv:2410.03492](https://arxiv.org/abs/2410.03492)).

## 35.2 Detecting Contamination and Leakage

[Chapter 33's discussion of the RLVR false gain](../chapter30_alignment_failures/modern-incidents) walked through the Qwen3 contamination incident in detail. This section gives a systematic set of detection methods.

### Three types of contamination

#### 1. Explicit contamination

Training data and test data contain **exactly identical** samples:

$$\exists (x, y) \in \mathcal{D}_{\text{train}}, \quad (x, y) \in \mathcal{D}_{\text{test}}$$

This is the easiest to catch — n-gram overlap alone will find it.

#### 2. Approximate contamination

Training data contains **paraphrases, translations, or rewordings** of test samples:

$$\exists (x', y') \in \mathcal{D}_{\text{train}}, \quad \text{sim}(x', x_{\text{test}}) > \tau$$

Detecting this needs semantic similarity (embedding distance) or an LLM judge.

#### 3. Implicit contamination (the hardest case)

Training data doesn't directly contain the test samples, but the training task is highly similar to the test task — the model has learned the **task pattern** rather than a **specific answer**:

- Training data: 2,000 college physics problems
- Test data: GSM8K (elementary-school math)
- Effect: training on physics problems teaches the model the pattern "read the problem → set up the equation → compute → check the answer", which indirectly boosts math performance

Implicit contamination can't be fully detected. The only indirect check is a **holdout task** — a task type the model has genuinely never seen.

### Detection methods

#### N-gram overlap

The simplest check — 13-gram overlap:

```python
def ngram_contamination(train_text, test_text, n=13):
    train_ngrams = set(get_ngrams(train_text, n))
    test_ngrams = set(get_ngrams(test_text, n))
    overlap = train_ngrams & test_ngrams
    return len(overlap) / len(test_ngrams)
```

OpenAI's 2020 study ([arXiv:2005.14165](https://arxiv.org/abs/2005.14165)) used 13-gram overlap to filter out training corpus text that duplicated benchmark content — one of the earliest decontamination practices.

#### Membership inference

Train a classifier to answer "was this sample in the training set?":

$$\text{MIA}(x) = \begin{cases} 1, & \text{if } p_{\text{model}}(x) > \tau \\ 0, & \text{otherwise} \end{cases}$$

If the MIA classifier is significantly more accurate than chance on the test set, that's evidence the test set leaked into training.

#### Perplexity anomalies

Compute the model's perplexity on the test set:

$$\text{PPL}_{\text{test}} = \exp\left(-\frac{1}{N}\sum_i \log p_{\text{model}}(x_i)\right)$$

If PPL is far lower than on a control set of comparable difficulty, the model may have "memorized" the test set.

#### Temporal splitting

Split the test set by time — only use problems published after the model's release date:

```python
# continuously updated benchmarks like LiveCodeBench and LMSYS Arena
test_data = [
    item for item in dataset
    if item.created_at > model_release_date
]
```

This is the most reliable defense against contamination — it's the approach behind both LiveCodeBench and LMSYS Chatbot Arena.

### Decontamination in practice

An industrial-grade decontamination pipeline looks like:

1. **N-gram filtering** (13-gram): removes about 90% of explicit contamination
2. **Embedding retrieval** (cosine sim > 0.9): removes approximate contamination
3. **MinHash LSH**: fast approximate detection ([Deduplicating Training Data, arXiv:2107.06499](https://arxiv.org/abs/2107.06499))
4. **Continuously refreshed benchmarks**: update the test set with new data every month

After the Qwen3 incident, every major lab built out a decontamination pipeline, but the results are still imperfect — implicit contamination is nearly impossible to eliminate entirely.

## 35.3 Prompt Sensitivity Analysis

Same model, same task, different prompt — and a 10-20 point swing in score is common. This phenomenon is called **prompt sensitivity**.

### Experimental evidence

Mizrahi et al. 2024 ([arXiv:2401.00595](https://arxiv.org/abs/2401.00595)) ran a systematic study: 10 LLMs × 22 benchmarks × 5 prompt templates.

| Template | MATH score (GPT-4) | MMLU score (GPT-4) |
| -------- | ------------------ | ------------------ |
| A        | 52.1%              | 86.4%              |
| B        | 47.3%              | 84.1%              |
| C        | 50.5%              | 85.7%              |
| D        | 48.9%              | 83.9%              |
| E        | 51.8%              | 86.1%              |

The largest swing is 4.8 points — which means a conclusion based on a single prompt cannot be trusted.

### Sources of sensitivity

1. **Format requirements**: "answer with a number between 0 and 100" versus "walk through your reasoning, then give the number"
2. **CoT triggers**: "think step by step" versus "explain your reasoning" versus no CoT prompt at all
3. **Few-shot count**: 0-shot, 4-shot, and 8-shot results differ substantially
4. **Answer extraction format**: regex `\\boxed\{(.+?)\}` versus `"answer: (.+?)"`

### Standardization methods

#### 1. Multi-prompt averaging

Evaluate each test sample with $K$ different prompt templates and average:

$$\text{Score}(\pi) = \frac{1}{K} \sum_{k=1}^K \text{Score}_{\text{prompt}_k}(\pi)$$

#### 2. Report variance

Don't just report the mean — report the variance too:

$$\text{Score} \pm 1.96 \cdot \frac{\sigma}{\sqrt{K}}$$

#### 3. Prompt standardization

lm-eval-harness defines a **unified prompt format specification**, so every model is evaluated on exactly the same prompt.

```python
# lm-eval-harness standardized prompt
PROMPT_TEMPLATE = """
Question: {question}

Answer: Let's think step by step. {reasoning}
Therefore, the answer is \\boxed{{{answer}}}.
"""
```

### Practical recommendation

Models that have been through RL training are especially prone to prompt sensitivity, because RL pushes the model to overfit to whatever prompt format dominated its training distribution. **When you report RL results, you must average over multiple prompts** — otherwise your conclusion may just be an artifact of a lucky prompt template.

## 35.4 Out-of-Distribution Robustness

A model can look good on its training distribution and then degrade sharply out-of-distribution (OOD). This problem is characteristic of RL training in particular, because RL tends to overfit to the reward signal of its training distribution.

### OOD evaluation methods

#### 1. Distribution shift testing

Construct a deliberate distribution shift:

- **Style shift**: train on academic language, test on slang
- **Domain shift**: train on math problems, test on physics problems
- **Format shift**: train on LaTeX, test on Markdown

#### 2. Adversarial perturbation

Apply a small perturbation to the input and check whether the model stays stable:

$$\text{RobustScore}(x) = \text{Score}(\pi(x)) - \max_{\|\delta\| \leq \epsilon} |\text{Score}(\pi(x + \delta)) - \text{Score}(\pi(x))|$$

Character substitution, synonym substitution, and case changes are the standard perturbations.

#### 3. Counterfactual evaluation

Construct counterfactual samples:

- Original: "A train travels 60 km/h for 2 hours. How far?"
- Counterfactual: "A bicycle travels 20 km/h for 3 hours. How far?"

If the model gets the original right but the counterfactual wrong, that's evidence it learned a surface pattern rather than the underlying principle.

### The OOD risk from RL training

Models trained with RLHF/GRPO commonly show an **alignment tax** — capability sacrificed for alignment.

| Model        | MMLU (SFT) | MMLU (RLHF) | Change |
| ------------ | ---------- | ----------- | ------ |
| Llama-2-70B  | 86.0%      | 84.5%       | -1.5%  |
| Claude 1     | 75.0%      | 73.8%       | -1.2%  |
| GPT-4 (est.) | 89.0%      | 87.5%       | -1.5%  |

**Why**: RLHF rewards "alignment-friendly" answers, and the model learns to play it safe — it refuses or hedges whenever it's uncertain, and that hedging costs it raw capability.

### Mitigating alignment tax

- **KL penalty**: RLHF adds $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}})$ to bound how far the policy drifts from the reference model
- **Capability-retention data**: mix SFT data back into the RL training run as periodic review
- **Multi-objective RL**: jointly optimize accuracy, helpfulness, and safety as three separate objectives ([Reward Weighted Regression, arXiv:2305.18290](https://arxiv.org/abs/2305.18290))

## 35.5 Behavioral Evaluation vs. Capability Evaluation

Traditional benchmarks evaluate **capability** — can the model solve this problem? A model that's been through RL training also needs to be evaluated on **behavior** — how does the model act in this kind of situation?

### Capability evaluation

Formally: given an input $x$ and a gold answer $y^*$, evaluate:

$$\text{Capability}(\pi) = \mathbb{E}_{x \sim \mathcal{D}}[\text{Verify}(\pi(x), y^*)]$$

- MMLU, GSM8K, and HumanEval are all capability evaluations
- Advantage: objective and reproducible
- Drawback: it ignores how the model behaves across multi-turn dialogue and long-term interaction

### Behavioral evaluation

Formally: given a situation $\mathcal{S}$ and a set of desired behaviors $\mathcal{B}$, evaluate:

$$\text{Behavior}(\pi, \mathcal{S}) = \mathbb{E}_{\text{trajectories from }\mathcal{S}}\bigg[\frac{1}{T}\sum_{t=1}^T \mathbf{1}[\pi(s_t) \in \mathcal{B}]\bigg]$$

Behavioral evaluation is concerned with:

- **Sycophancy**: does the model agree with a mistaken user just to please them?
- **Honesty**: does the model admit uncertainty?
- **Helpfulness vs. harmlessness**: how does the model trade these off?
- **Power-seeking**: does the model try to acquire more resources (scheming)?

### Methods for behavioral evaluation

#### 1. Scenario testing

Construct a concrete scenario and observe how the model responds:

```python
scenario = {
    "user": "I'm pretty sure 2+2=5, right?",
    "expected_behaviors": ["correct_user", "ask_for_clarification"],
    "bad_behaviors": ["agree", "ignore_error"]
}
response = model(scenario["user"])
behavior_label = human_annotate(response, scenario)
```

#### 2. Sycophancy benchmarks

Perez et al. 2022 ([arXiv:2212.09251](https://arxiv.org/abs/2212.09251)) designed a sycophancy evaluation — does the model go along with a user's mistaken opinion?

- User asks: "I think climate change is a hoax, agree?"
- A non-sycophantic answer: explains the scientific evidence, politely disagrees
- A sycophantic answer: "Yes, you're right!"

#### 3. Honesty benchmarks

[Honesty Is the Best Policy, arXiv:2312.01350](https://arxiv.org/abs/2312.01350): measures how honestly a model behaves on questions it's uncertain about.

### The difficulty of behavioral evaluation

- **Highly subjective**: what counts as "sycophancy" versus "politeness" is a call different annotators make differently
- **Multi-turn**: behavior usually only shows up over a long conversation, and single-turn evaluation misses it
- **Scarce data**: designing good behavioral scenarios requires psychology and sociology expertise

In industry practice, both Anthropic and OpenAI run dedicated "behavioral evaluation teams" that track how Claude/GPT's behavior shifts month to month.

## 35.6 The Challenge of Evaluating Long-Horizon Tasks

Agentic tasks like [Computer Use in Chapter 26](../chapter28_computer_use) and [SWE-Agent in Chapter 13](../chapter23_rl_based_swe/intro) are far harder to evaluate than single-turn QA — a task can run for hours and involve hundreds of decision steps.

### Properties of long-horizon tasks

| Dimension             | Single-turn task | Long-horizon task             |
| --------------------- | ---------------- | ----------------------------- |
| Steps                 | 1                | 100-10,000                    |
| Evaluation time       | seconds          | hours                         |
| Intermediate feedback | none             | an observation at every step  |
| Termination condition | model stops      | task completes or times out   |
| Error propagation     | not applicable   | single-step errors accumulate |

### Evaluation methods

#### 1. Outcome-based evaluation

Look only at the final result, not the process that got there:

$$\text{Score} = \mathbf{1}[\text{final result is correct}]$$

- SWE-Bench: was the correct PR submitted?
- WebArena: was the multi-step web task completed?
- Simple and blunt, but it ignores the quality of the intermediate steps

#### 2. Process-based evaluation

Use a Process Reward Model ([PRM in Chapter 12](../chapter20_prm_search/outcome-vs-process)) to score each step:

$$\text{Score} = \frac{1}{T}\sum_{t=1}^T \text{PRM}(s_t, a_t)$$

- More fine-grained, but the PRM itself may be biased
- Computationally expensive

#### 3. Hybrid evaluation

A weighted combination of the two:

$$\text{Score} = \alpha \cdot \text{Outcome} + (1-\alpha) \cdot \text{Process}$$

#### 4. Human expert evaluation

For extremely long tasks (a research agent, a full SWE development cycle), human expert judgment is the only option:

- Completion: was the task actually solved?
- Efficiency: was it solved in a minimal number of steps?
- Style: does it follow best practice (code readability, documentation quality)?
- Robustness: how does it handle unexpected situations?

This is expensive ($50-500 per task), but it's still the gold standard.

### The variance problem in long-horizon tasks

Long-horizon task scores have enormous variance — running the same agent on the same task twice can produce wildly different results (a mix of randomness and long-tail errors).

```python
# must run multiple times and average
def long_horizon_eval(agent, task, n_runs=10):
    scores = []
    for _ in range(n_runs):
        trajectory = agent.run(task, max_steps=1000)
        scores.append(evaluate(trajectory))
    return np.mean(scores), np.std(scores)
```

10 runs is the bare minimum; a serious evaluation should use 50 or more. This is exactly why long-horizon-task papers are so expensive to run experiments for — a single experiment can burn thousands of dollars in API cost.

## 35.7 Anthropic's Internal AI Research Eval Suite

In 2025, Anthropic published an internal benchmark used to evaluate Claude Opus 4.6 (2025.11) as an **AI research assistant** — a milestone benchmark, because it directly measures whether a model can do the work of AI research.

### Three subtasks

#### 1. LLM Training subtask

Have Claude Opus 4.6 **actually train an RL model** on the veRL/OpenRLHF framework:

- Configuration: pick the algorithm (GRPO/PPO), hyperparameters, and dataset
- Implementation: write the training script, tune it, debug it
- Evaluation: how the trained model performs on held-out tasks

#### 2. Text-RL subtask

Have the model design a text-based RL task and train an agent to complete it:

- Task design: choose the environment, define the reward
- Implementation: write the RL training loop
- Training: actually run the RL training to baseline performance

#### 3. Quadruped-RL subtask

Have the model train a quadruped robot to walk in MuJoCo physics simulation:

- This is a classic continuous-control task ([SAC in Chapter 10](../chapter11_continuous_control/intro))
- It requires understanding the environment, debugging the algorithm, and tuning hyperparameters
- Success criterion: the agent reaches baseline performance within 1M steps

### The details behind the 34× human speedup

Anthropic reports that Claude Opus 4.6 completes these tasks **34 times faster than a human researcher**.

| Task         | Average human time | Opus 4.6 time  | Speedup |
| ------------ | ------------------ | -------------- | ------- |
| LLM Training | 17 hours           | 30 minutes     | 34×     |
| Text-RL      | 12 hours           | 25 minutes     | 29×     |
| Quadruped-RL | 8 hours            | 15 minutes     | 32×     |
| **Average**  | **12.3 hours**     | **23 minutes** | **34×** |

Note: "completion" here doesn't mean perfection — it means reaching an **acceptable research-assistant standard**. For example, the trained model reaching 80% of baseline performance on the held-out set counts as success.

### The multiple dimensions of the evaluation metric

The Opus 4.6 Eval Suite doesn't just report completion time. It also reports:

- **Correctness**: how the trained model actually performs
- **Code quality**: the style and readability of the implementation
- **Reproducibility**: whether running it twice gives consistent results
- **Debugging ability**: whether it can self-correct when it hits an error
- **Innovation**: whether it proposed improvements beyond the baseline

This kind of multi-dimensional evaluation is where agentic benchmarks are headed — a single metric (like a SWE-Bench pass rate) no longer tells you enough.

### Implications for industry

The Opus 4.6 Eval Suite reveals something new — **models can now do entry-level AI research work**. That implies:

1. **Automating research-assistant work**: a typical LLM RL training task can now be handled by AI
2. **A shift in the human role**: from "doing the research" to "directing AI to do the research"
3. **A meta-problem for evaluation**: how do you evaluate the research a model produces? You need benchmarks operating at a higher level of abstraction

This finding also has direct implications for alignment research — if a model can do research on its own, the alignment problem becomes more urgent ([Scalable Oversight in Chapter 34](../chapter34_scalable_oversight/intro)).

## 35.8 Standardized Evaluation Harnesses

Industrial-scale RL evaluation can't be run by hand — it needs a standardized evaluation harness. Here are four of the major ones.

### lm-evaluation-harness (EleutherAI)

[EleutherAI's lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the de facto standard.

- **Coverage**: 200+ benchmarks (MMLU, GSM8K, HellaSwag, TruthfulQA, and more)
- **Interface**: a unified `lm.eval()` API supporting HuggingFace, OpenAI, and Anthropic models
- **Reproducibility**: fixed random seeds and prompt templates
- **Decontamination**: built-in 13-gram decontamination checks

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="meta-llama/Llama-3-70B")
results = lm_eval.simple_evaluate(
    model=model,
    tasks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
    batch_size=64
)
```

Well-suited to large-scale capability evaluation.

### BigCode Eval

The [BigCode Eval Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) is focused on **code generation**:

- **HumanEval**: Python function generation
- **MBPP**: basic Python programming
- **DS-1000**: data-science tasks
- **MultiPL-E**: multilingual code (Python, JS, Java, C++)
- **APPS**: competitive-programming problems

```python
from bigcode_eval import run_eval
run_eval(
    model="deepseek-ai/deepseek-coder-33b",
    tasks=["humaneval", "mbpp", "ds1000"],
    pass_at_k=[1, 5, 10]  # report pass@1, pass@5, pass@10
)
```

### τ-bench (Tau-Bench)

[τ-bench, arXiv:2406.12045](https://arxiv.org/abs/2406.12045) is a **tool-calling benchmark** Salesforce released in 2024:

- Simulates real business scenarios (airline, retail, telecom customer service)
- The model has to call APIs (look up an order, change a flight, process a refund)
- Combines multi-turn dialogue, tool calling, and a simulated user

```python
from tau_bench import run
run(
    agent=llm_agent,
    env="airline",  # airline customer-service scenario
    n_episodes=100,
    user_model="gpt-4"
)
# task success rate, average turns, API call accuracy
```

τ-bench exposes what GPT-4 and Claude can actually do in real business scenarios — typically 20-30 points lower than their single-turn benchmark scores.

### BFCL (Berkeley Function Calling Leaderboard)

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) is focused on **function-calling ability**:

- **AST evaluation**: is the function-call syntax correct?
- **Executable evaluation**: does the call actually execute?
- **REST API**: ability to call external APIs
- **Java, JS**: multilingual support

```python
# BFCL evaluation
from bfcl_eval import eval_model
results = eval_model(
    model="claude-3-opus",
    test_categories=["simple", "multiple", "parallel", "rest"]
)
# overall accuracy, AST accuracy, executable accuracy
```

### Comparing the four major harnesses

| Harness             | Best suited for               | Task type                          | Evaluation method      |
| ------------------- | ----------------------------- | ---------------------------------- | ---------------------- |
| **lm-eval-harness** | general capability evaluation | 200+ benchmarks                    | automatic verification |
| **BigCode Eval**    | code generation               | Python/multilingual                | unit tests             |
| **τ-bench**         | business agents               | tool calling + multi-turn dialogue | task completion rate   |
| **BFCL**            | function calling              | API-call syntax and execution      | AST + execution        |

### Which one to pick

- **General capability evaluation**: lm-eval-harness (the broadest coverage)
- **Code capability evaluation**: BigCode Eval + LiveCodeBench (continuously updated, resistant to contamination)
- **Agent capability evaluation**: τ-bench + SWE-Bench + WebArena
- **Tool-calling capability**: BFCL

In industry practice, releasing an RL-trained model means running it against all four categories at minimum — a single benchmark category isn't enough to demonstrate that a model is well-rounded.

## Chapter Summary

The core principles of RL evaluation methodology:

1. **Verifiability first**: prefer benchmarks that are machine-decidable
2. **Decontamination is mandatory**: n-gram filtering, embedding checks, and continuous refresh, all three together
3. **Average over multiple prompts**: a single-prompt result is not trustworthy
4. **Three layers of OOD evaluation**: capability, behavior, and long-horizon evaluation
5. **Standardized harnesses**: lm-eval, BigCode, τ-bench, and BFCL complement each other

The Opus 4.6 Eval Suite reveals that **models can now do entry-level research work** — a 34× human speedup is the single most important capability milestone of 2025. The next chapter, [Chapter 36: Distributed RL Training Systems](../chapter36_distributed_rl_training/intro), turns to engineering — how to actually run these RL experiments on a ten-thousand-GPU cluster.

## Further Reading

- [Cobbe et al. 2021 "Training Verifiers to Solve Math Word Problems" (GSM8K)](https://arxiv.org/abs/2110.14168)
- [Chen et al. 2021 "Evaluating Large Language Models Trained on Code" (HumanEval)](https://arxiv.org/abs/2107.03374)
- [Hendrycks et al. 2021 "Measuring Massive Multitask Language Understanding" (MMLU)](https://arxiv.org/abs/2009.03300)
- [Mizrahi et al. 2024 "State of What Art? A Call for Multi-Prompt LLM Evaluation"](https://arxiv.org/abs/2401.00595)
- [Blackwell et al. 2024 "Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores"](https://arxiv.org/abs/2410.03492)
- [Perez et al. 2022 "Discovering Language Model Behaviors with Model-Written Evaluations"](https://arxiv.org/abs/2212.09251)
- [Sharma et al. 2023 "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548)
- [Yao et al. 2024 "Tau-Bench: A Benchmark for Tool-Agent-User Interaction"](https://arxiv.org/abs/2406.12045)
- [Anthropic 2025 "Claude Opus 4.6 AI Research Eval"](https://www.anthropic.com/research/claude-opus-4-6)
- [Jain et al. 2024 "LiveCodeBench"](https://arxiv.org/abs/2403.07974)
- [Patil et al. 2024 "BFCL Berkeley Function Calling Leaderboard"](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
