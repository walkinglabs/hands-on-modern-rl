# 21.1 SWE-bench and the RL-based SWE Paradigm

This section lays the groundwork for SWE-RL: what SWE-bench is, why software engineering is the ideal battlefield for RLVR, and what actually separates SWE-RL from traditional code generation.

## 12.1.1 The SWE-bench Task Definition

[SWE-bench](https://arxiv.org/abs/2310.06770) (Jimenez et al. 2023) is the core benchmark for SWE-RL. Its task definition is:

```text
Input:
  - A GitHub repository (the full codebase)
  - An issue description (natural language, describing a bug or feature request)
  - Test cases (used to verify whether the fix is correct)

Output:
  - A code patch (the modified code)

Verification:
  - Apply the patch to the repository
  - Run the test cases
  - All pass → task succeeds
  - Any test fails → task fails
```

### A concrete example

```text
Repository: django/django (the Django web framework)

Issue:
  "In Django 4.2, using `Model.objects.filter(field__in=[])`
   returns an empty queryset, but the SQL query still executes.
   It should short-circuit to an empty result and avoid the
   unnecessary database call."

Test case:
  def test_empty_in_lookup_short_circuits(self):
      # Expected: filter(field__in=[]) does not trigger SQL
      with self.assertNumQueries(0):
          list(Model.objects.filter(field__in=[]))

Model output:
  - Modify django/db/models/sql/query.py
  - In the as_sql method, add: if not self.bloom_metadata and not value: return '', []

Verification:
  - Apply the patch
  - Run the tests: passed
  - Task succeeds
```

### How hard is SWE-bench

SWE-bench is far harder than traditional code generation:

| Dimension       | Ordinary code generation              | SWE-bench                           |
| --------------- | ------------------------------------- | ----------------------------------- |
| Context         | A single function / short description | An entire repository (10K–1M lines) |
| Output          | A complete code snippet               | An exact patch (diff)               |
| Verification    | Manual or by testing                  | An automated test suite             |
| Multi-file      | Rare                                  | Cross-file edits are common         |
| Reasoning depth | 1–10 steps                            | 10–100+ steps                       |

SOTA performance on SWE-bench Verified (the high-quality 500-problem subset):

- Early 2024: about 12% (OpenAI SWE-agent)
- Mid 2024: about 25% (Cognition Devin)
- Early 2025: about 40% (open-source SWE-RL systems)
- Late 2025: about 53% (NVIDIA and others)
- Early 2026: about 65% (Claude Opus 4.7 with tool calling)

## 12.1.2 Why SWE Is the Ideal Battlefield for RLVR

Recall [Chapter 7's RLVR](../chapter18_grpo/rlvr)—the core idea is **replacing the reward model with rule-based verification**. RLVR needs three conditions:

1. **The task has a definite answer**: it's either right or wrong
2. **Verification can be automated**: no human judgment required
3. **There's enough training data**: enough to support large-scale RL

SWE satisfies all three conditions perfectly.

### A definite answer

Code either passes the tests or it doesn't—there's no "half right" or "subjective call." Outside of mathematics, this is the purest domain of clean right-or-wrong.

### Automated verification

Test frameworks like `pytest` and `unittest` run tests automatically and output PASS/FAIL. The entire verification process requires no human intervention.

### Massive data

- GitHub hosts over 400 million repositories
- Every PR is a naturally occurring SWE task (issue + patch + tests)
- Internal commit history at industrial companies is an even larger pool of training data

These three conditions make SWE-RL one of RLVR's **biggest industrial success stories**. Meta, ByteDance, Cognition, Alibaba, and Tsinghua have all poured substantial resources into this direction.

## 12.1.3 SWE-RL vs. Traditional Code Generation

Traditional code generation (HumanEval, MBPP) frames the task as:

```text
Input: a function signature + docstring
Output: a complete function implementation
```

This is a **short-context, single-file, no-test-feedback** setting. RL has limited impact here, because the generation space is small and SFT alone already reaches SOTA.

SWE-RL frames the task as:

```text
Input: a full repository + issue + test cases
Output: an exact patch
Allowed: multi-step interaction (read file, edit, run test, edit again)
```

This is a **long-context, multi-file, test-feedback** setting. RL has a substantial impact here, for three reasons:

- **The exploration space is enormous**: the number of possible patches is astronomical, and RL can explore it efficiently
- **Feedback is delayed**: test results arrive as delayed reward, which matches naturally with RL's advantage estimation
- **Decisions are multi-step**: read → think → edit → test → fix → submit is a textbook agent trajectory

## 12.1.4 Manufacturing SWE-bench Data

SWE-RL training needs large numbers of (issue, patch, tests) triples. There are three sources.

### Real PRs (the SWE-bench method)

Scrape PRs from GitHub and extract:

- The issue text (the issue linked to the PR)
- The code diff (the PR's changes)
- Test cases (tests the PR added or modified)

Scale: about 2,300 examples (the original SWE-bench)

Limitations:

- **Too little data**: 2,300 examples isn't enough to train a large model
- **Depends on PR quality**: low-quality PRs get swept in too
- **Tests may be missing**: many PRs don't come with a complete test suite

### Synthetic data (the SWE-smith method)

[SWE-smith](../chapter22_agentic/agent-data-swe-smith) ([arXiv:2504.21798](https://arxiv.org/abs/2504.21798)) **deliberately injects bugs into good code, then runs the tests to see which bugs get caught.**

Scale: 50,000+ examples (spanning 128 Python repositories)

Advantages:

- **Data volume**: 20x the size of SWE-bench
- **Controllable**: bug type and difficulty can be tuned
- **Complete tests**: every bug comes with a matching test

### Model self-generation (the self-play SSR method)

Let the model itself:

1. Find a spot in the repository that "looks like a bug"
2. Write a "fix"
3. Run the tests to see if it passes
4. Keep the (issue, patch, test) triples that pass as training data

This is the core idea behind [Section 12.5's SSR](./self-play-ssr)—**the model generates its own training data.**

## 12.1.5 The SWE-RL Reward Function

The SWE-RL reward is usually extremely simple:

```python
def swe_reward(test_results):
    """Test results as the reward"""
    passed = sum(test_results)
    total = len(test_results)
    return passed / total  # or binary: 1.0 if passed == total else 0.0
```

This reward function is essentially identical to R1-Zero's math reward—**a 0/1 binary reward**.

### The details of reward shaping

In industrial practice, though, a few shaping terms often get added.

**Term 1: fraction of tests passed**

```python
reward = passed / total
```

Instead of binary, this is continuous. It lets the model get partial reward even when it "fixed half of it."

**Term 2: a length penalty**

```python
reward -= 0.01 * len(trajectory)
```

Encourages the model to finish in fewer steps—avoiding the wasteful pattern of "make a random edit, run the tests, fail, edit again."

**Term 3: edit quality**

```python
patch_quality = score_patch(model_output)  # scored by an LLM judge
reward += 0.1 * patch_quality
```

Encourages the model to produce cleaner patches (no duplicated code, no broken existing logic).

**Term 4: context usage**

```python
context_efficiency = relevant_files_read / total_files_read
reward += 0.05 * context_efficiency
```

Encourages the model to read only relevant files instead of wastefully "reading everything."

But [Meta SWE-RL](https://arxiv.org/abs/2502.18449) reported an important finding: **the simplest reward (binary test-pass) works best.** Complex shaping tends to invite reward hacking—the model learns to optimize the shaping terms instead of actually fixing the bug.

This matches [the finding from R1-Zero](../chapter18_grpo/deepseek-dapo): **a simple reward at large RL scale beats a complex reward at small RL scale.**

## 12.1.6 The SWE-RL Training Pipeline

A complete SWE-RL training pipeline looks like this:

```text
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Choose a base model                                 │
│   - Usually a code-tuned LLM (Qwen-Coder, DeepSeek-Coder)   │
│   - Already pretrained on a large volume of code            │
├─────────────────────────────────────────────────────────────┤
│ Step 2: SFT cold start (optional)                           │
│   - SFT on SWE-bench / SWE-smith data                       │
│   - Teaches the model the basic trajectory format           │
├─────────────────────────────────────────────────────────────┤
│ Step 3: RL training                                         │
│   - GRPO / PPO                                              │
│   - Reward: binary test-pass                                │
│   - Long horizon: each trajectory can run 16-100+ steps     │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Rejection sampling + a second round of SFT          │
│   - Generate multiple candidates from the RL-trained model  │
│   - Pick the best ones for SFT                              │
├─────────────────────────────────────────────────────────────┤
│ Step 5: Evaluation                                          │
│   - SWE-bench Verified                                      │
│   - Internal evaluation sets                                │
└─────────────────────────────────────────────────────────────┘
```

This pipeline closely mirrors [DeepSeek-R1's training pipeline](../chapter18_grpo/deepseek-dapo)—both combine SFT, RL, and a second round of SFT. The only difference is what the reward measures:

- R1's reward is whether the math answer is right
- SWE-RL's reward is whether the tests pass

This similarity points to something general: **RLVR's training paradigm is domain-agnostic**—given the right verifier, the same algorithm carries over to a completely different domain.

## Summary

SWE-bench is SWE-RL's core benchmark, defining the (issue, patch, tests) task format. SWE is the ideal battlefield for RLVR—it has a definite answer, automated verification, and massive data.

SWE-RL differs fundamentally from traditional code generation—long context, multiple files, test feedback, multi-step decisions. That's exactly what makes it line up so closely with Agentic RL, and why it's one of RL's most valuable industrial applications.

Next, we turn to Meta SWE-RL—the flagship open-source SWE-RL system.
