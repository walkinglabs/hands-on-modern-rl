# 21.2 Meta SWE-RL and the Open-Source SOTA

[Meta SWE-RL](https://arxiv.org/abs/2502.18449) (Feb 2025) is the representative work of open-source SWE-RL. Its core contributions:

- Trained on open-source data (SWE-bench + SWE-gym)
- Uses the simplest possible setup: GRPO + a test-based reward
- Reaches 41.0% on SWE-bench Verified (open-source SOTA)

This section looks in detail at Meta SWE-RL's data, algorithm, and engineering details.

## 12.2.1 Data Scale and Composition

Meta SWE-RL's training data comes from:

| Data Source             | Scale         | Purpose               |
| ----------------------- | ------------- | --------------------- |
| SWE-bench (open-source) | 2,294 items   | High-quality baseline |
| SWE-gym (open-source)   | 6,800 items   | Training expansion    |
| Internal PR data (Meta) | 80,000+ items | Large-scale RL        |

**About 90,000 SWE tasks in total** — 40 times larger than SWE-bench alone.

### Data Preprocessing

Meta reports several key data-cleaning steps.

**Step one: repository filtering**

- Exclude: repositories with test coverage below 50% (cannot be reliably verified)
- Exclude: inactively maintained repositories (last commit more than 6 months ago)
- Exclude: repositories on Python < 3.8 (incompatible with the latest dependencies)

**Step two: PR filtering**

- Exclude: PRs that touch more than 10 files (too complex for early-stage RL training)
- Exclude: PRs that are pure dependency bumps (not a real "bug fix")
- Exclude: PRs that remove functionality (inconsistent with the semantics of "fixing")

**Step three: test filtering**

- Keep: PRs that include new tests (a clear verification criterion)
- Exclude: PRs whose tests can't run independently
- Exclude: PRs whose tests depend on external services (e.g. a database or an API key)

These filters substantially raised the final data quality. Meta reports that training on the unfiltered data performed poorly, while the filtered data brought a large improvement — **data quality matters more than data quantity**.

## 12.2.2 Algorithm: GRPO + a Simple Reward

Meta SWE-RL's algorithm choice is extremely simple: **GRPO + a binary test-based reward**.

### Why GRPO?

The Meta team compared PPO, GRPO, and DPO in the [SWE-RL paper](https://arxiv.org/abs/2502.18449):

| Algorithm      | SWE-bench Verified |
| -------------- | ------------------ |
| DPO (baseline) | 25.3%              |
| PPO            | 33.2%              |
| **GRPO**       | **41.0%**          |

GRPO's advantages:

- **No critic needed**: saves memory, which suits large-scale training
- **Within-group normalization**: naturally handles tasks of varying difficulty (easy tasks show high within-group variance, hard tasks show low within-group variance)
- **Simple and stable**: easier to implement than PPO

This matches [DeepSeek-R1's findings](../chapter18_grpo/deepseek-dapo) exactly — **GRPO is the default choice for SWE-RL**.

### The Reward Function

Meta SWE-RL's reward is extremely simple:

```python
def swe_rl_reward(test_results):
    """Binary reward for passing tests"""
    if all(test_results):
        return 1.0  # all tests pass
    else:
        return 0.0  # any failure
```

Meta reports several reward-shaping attempts, all of which **performed worse than the simple binary reward**:

- **Continuous reward (passed/total)**: the model learns to "half-fix" the bug to collect partial reward, instead of learning a complete fix
- **Length penalty**: the model learns to "use fewer steps without fixing the bug" — it simply aborts the task to get a zero-step trajectory
- **Complex shaping**: severe reward hacking

This confirms the R1-Zero finding: **a simple reward plus large-scale RL is the best combination**.

## 12.2.3 Engineering Details: Context Management

Context management is an engineering challenge for SWE-RL. A complete SWE task can involve:

- An entire repository (10K–1M lines of code)
- The issue description (a few hundred tokens)
- Tool calls (read file, edit, run test)
- The model's reasoning (CoT)

Put together, this can exceed 100K tokens — beyond the context window of most models. How do you handle it?

### Tool Calls Instead of Reading the Whole Repository

Instead of having the model read the entire repository up front, let it **call tools to actively fetch only the parts it needs**:

```text
<issue> Django filter(field__in=[]) should short-circuit </issue>

<thought>
I need to find where this filter is implemented.
</thought>

<tool_call> grep -r "def filter" django/db/models/sql/query.py </tool_call>

<tool_result> def filter(self, *args, **kwargs): ... </tool_result>

<thought>
Found it. Let me look at the as_sql method.
</thought>

<tool_call> read django/db/models/sql/query.py lines 100-150 </tool_call>

<tool_result> def as_sql(self, compiler, connection):
    ...
</tool_result>

<thought>
I see there's no short-circuit logic here. Let me add one.
</thought>

<tool_call> edit django/db/models/sql/query.py
+ if not self.bloom_metadata and not value:
+     return '', []
</tool_call>
```

This agentic approach lets the model explore the repository on its own, **reading only what's necessary** — context usage drops substantially.

### Long-Context Models

Use a model that supports 128K / 1M context (e.g. Claude 3.5 Sonnet, Gemini 1.5 Pro, Qwen3-1M) and simply read the entire repository.

Meta SWE-RL uses **Llama-3-70B with RoPE scaling**, extended to 128K context. But long context brings costs of its own:

- Higher training cost (attention is O(n²))
- Slower inference
- Position bias (the model is less sensitive to the middle of a long context)

### RAG (Retrieval-Augmented Generation)

Build an embedding index over the repository ahead of time, retrieve the files relevant to the issue description, and put only those files into the context.

```python
def build_context(issue, repo):
    # 1. Retrieve relevant files via embeddings
    relevant_files = retrieve(issue, repo, top_k=5)

    # 2. Concatenate into a context
    context = ""
    for file in relevant_files:
        context += f"### {file.path}\n{file.content}\n\n"

    return context
```

RAG is the most common approach in industry — simple, efficient, and compatible with existing models.

Meta SWE-RL uses **a hybrid of the tool-call approach and RAG**: RAG builds the base context, and tool calls let the model explore further from there.

## 12.2.4 Training Stability Techniques

SWE-RL is harder to stabilize during training than math RL, because:

- Trajectories are long (16–100+ steps)
- Reward is extremely sparse (reward only arrives when the final test passes)
- Most trajectories fail (reward = 0)

Meta reports several stability techniques.

### Success Rate Filtering

During RL training, **keep only prompts that have succeeded at least once**. If all N rollouts for a prompt fail (reward is 0 for all of them), the within-group variance is also 0, and it provides no training signal.

```python
def filter_prompts(prompts, model, num_rollouts=8):
    useful_prompts = []
    for prompt in prompts:
        rollouts = [model.generate(prompt) for _ in range(num_rollouts)]
        rewards = [compute_reward(r) for r in rollouts]
        if max(rewards) > 0:  # at least one success
            useful_prompts.append(prompt)
    return useful_prompts
```

This is the same idea as [DAPO's Dynamic Sampling](../chapter18_grpo/deepseek-dapo) — filter out the "graduated" problems.

### Curriculum Learning

Order prompts by difficulty, and train on the easy ones first (small PRs, single-file changes, clear issues) before moving to the complex ones (multi-file changes, ambiguous issues).

```python
def curriculum_order(prompts):
    # sort by number of files changed
    prompts.sort(key=lambda p: p.num_files_changed)
    return prompts
```

### KL Constraint

Late in SWE-RL training, the model tends to "forget how to write code" — over-optimizing for test passage damages code style. Meta addresses this with a KL constraint:

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

$\pi_{\text{ref}}$ is the pre-RL model (the SFT checkpoint), and $\beta$ controls the strength of the constraint.

This contrasts with DeepSeek V3.2's "zero KL for math tasks" — **SWE needs to preserve code style, so it needs the KL term**, while math is pure logic and doesn't.

## 12.2.5 SWE-bench Verified: 41.0%

Meta SWE-RL's final results on SWE-bench Verified:

| Model                           | SWE-bench Verified                     |
| ------------------------------- | -------------------------------------- |
| GPT-4 (zero-shot)               | 1.96%                                  |
| Claude 3 Opus                   | 3.21%                                  |
| SWE-agent (GPT-4)               | 12.5%                                  |
| SWE-Gym (open-source)           | 20.0%                                  |
| **Meta SWE-RL (open-source)**   | **41.0%**                              |
| Cognition Devin (closed-source) | 13.95% (note: different eval protocol) |
| Claude 3.5 Sonnet + tools       | 49.0% (closed-source)                  |

Meta SWE-RL is SOTA among open-source models — proof that **open-source data + GRPO + a simple reward can get you close to closed-source performance**.

## 12.2.6 The Limits of Meta SWE-RL

Meta SWE-RL still has several limitations.

### Python Only

All of Meta SWE-RL's training data is Python. There's no corresponding data for other languages (JavaScript, Java, C++, Go).

### Dependence on Test Suites

A repository with no tests can't be trained on. This is a real problem in industrial practice — a lot of companies' code doesn't have complete unit test coverage.

### Instability on Long-Horizon Training

Trajectories longer than 16 steps are unstable to train on — credit assignment in RL is hard over long horizons. Meta reports that training performance drops noticeably once trajectories exceed 32 steps.

### Data Diversity

90K data points sounds like a lot, but they all come from GitHub PRs — the distribution skews toward the open-source ecosystem. Industrial code (e.g. an enterprise's internal Java systems) has characteristics this data doesn't cover.

## Summary

Meta SWE-RL is the representative work of open-source SWE-RL. Its core contributions:

- **Data**: 90K open-source SWE tasks spanning 100+ repositories
- **Algorithm**: GRPO + a simple binary reward, sharing its lineage with R1-Zero
- **Engineering**: context management and training stability techniques
- **Results**: 41.0% on SWE-bench Verified (open-source SOTA)

Meta SWE-RL demonstrates that RLVR is viable in the SWE domain. But its limitations — Python-only, unstable long-horizon training, dependence on tests — point toward the topic of the next section: **how to use a world model to let the model "simulate" code execution, instead of running the real test suite every single time**.
