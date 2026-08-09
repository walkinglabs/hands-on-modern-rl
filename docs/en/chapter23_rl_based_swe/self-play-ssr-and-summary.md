# 21.3 Self-play SWE-RL and Industrial Adoption

So far we've discussed the three pillars of SWE-RL:

- **Data**: [SWE-bench](./swe-bench-and-rlvr) (real PRs) + SWE-smith (synthetic bugs)
- **Algorithm**: [Meta SWE-RL](./meta-swe-rl)'s GRPO + binary reward
- **Acceleration**: [CWM](./world-model-and-deep-swe) + DeepSWE's world model + value model

All of these methods depend on **pre-collected training data**—either SWE-bench or SWE-smith. Collecting that data is expensive and inherently limited.

This section looks at a different direction: **Self-play SWE-RL (SSR)**—having the model generate its own training data, forming a "data flywheel."

## 12.4.1 The core idea behind self-play

Self-play takes its inspiration from AlphaGo Zero—**the model plays against itself and learns from the outcome of the games**. SSR carries this idea over to SWE:

```text
┌──────────────────────────────────────────────────────────┐
│ Player A (Bug Generator):                                │
│   - Finds a spot in the repo to inject a bug              │
│   - Generates a test that verifies the bug exists         │
│   - Generates a matching issue description                │
├──────────────────────────────────────────────────────────┤
│ Player B (Bug Fixer):                                    │
│   - Sees the issue description                            │
│   - Attempts a fix                                        │
│   - Runs the test to verify it                             │
├──────────────────────────────────────────────────────────┤
│ RL Update:                                               │
│   - Player A learns to "generate harder bugs" (ones       │
│     Player B can't fix)                                   │
│   - Player B learns to "fix more complex bugs"             │
│   - The two push each other forward adversarially          │
└──────────────────────────────────────────────────────────┘
```

### How it differs from SWE-smith

[Section 12.1's SWE-smith](./swe-bench-and-rlvr) produces **offline synthetic data**—generate 50K examples once, then train on them.

SSR produces **online synthetic data**—the model keeps generating data throughout training, and the quality of that data rises together with the model's own ability.

| Dimension        | SWE-smith (offline)              | SSR (online)                    |
| ---------------- | -------------------------------- | ------------------------------- |
| Data generation  | One-shot                         | Continuous, throughout training |
| Data difficulty  | Fixed                            | Adjusts to model ability        |
| Data quality     | Independent of generator ability | Rises with model ability        |
| Applicable stage | Early training                   | The entire training run         |

### SSR's data flywheel

The core value of SSR is the **data flywheel**—a stronger model generates better data; better data produces a stronger model.

```text
Strong model → generates hard bugs + good fixes → high-quality training data → stronger model → ...
```

This positive feedback loop is why SSR pays off most in the later stages of training—the "hard problems" the model discovers on its own push capability further than problems designed by hand ever could.

## 12.4.2 SSR's algorithmic details

Here's how Tsinghua's [SSR](https://arxiv.org/abs/2512.18552) (Self-play SWE-RL) paper designs the system:

### Bug Generator (Player A)

The Bug Generator is an LLM. Given the repo's code, it outputs "code with an injected bug + a test + an issue description."

```python
def generate_bug(generator_model, repo, file_path):
    # 1. Pick a file
    original_code = repo.read(file_path)

    # 2. Have the generator inject a bug
    prompt = f"""
    Here is the code in {file_path}:
    {original_code}

    Please:
    1. Choose a function to modify
    2. Inject a subtle bug (logic error, not syntax error)
    3. Generate a test that would fail with the bug
    4. Generate an issue description (without revealing the bug)
    """

    response = generator_model.generate(prompt)
    bug_code, test, issue = parse_response(response)

    # 3. Verify the bug is valid (the test fails on the buggy code,
    #    and passes on the original code)
    if not validate_bug(original_code, bug_code, test):
        return None  # Invalid bug, discard

    return {
        "original_code": original_code,
        "bug_code": bug_code,
        "test": test,
        "issue": issue
    }
```

### Bug Fixer (Player B)

The Bug Fixer is the policy model being trained. Given the issue and the buggy code, it outputs a fix patch.

```python
def fix_bug(fixer_model, task):
    # 1. Show the fixer the issue and the buggy code (not the original code)
    prompt = f"""
    Issue: {task['issue']}

    Current code: {task['bug_code']}

    Please fix the bug.
    """

    # 2. The fixer works agentically to fix it
    trajectory = []
    while not done:
        action = fixer_model.act(prompt)
        trajectory.append(action)

        if action.type == "edit":
            apply_edit(action)
        elif action.type == "test":
            result = run_tests()
            if result.all_passed:
                done = True

    # 3. Compute the reward
    reward = 1.0 if tests_passed else 0.0

    return trajectory, reward
```

### Adversarial training

```python
def ssr_training(generator_model, fixer_model, repo):
    for epoch in range(N_EPOCHS):
        # 1. Generator produces a bug
        task = generate_bug(generator_model, repo, random_file())

        # 2. Fixer attempts a fix
        trajectory, reward = fix_bug(fixer_model, task)

        # 3. Adversarial reward
        generator_reward = -reward  # Fixer fails to fix it → Generator wins
        fixer_reward = reward       # Fixer fixes it → Fixer wins

        # 4. Update both models
        update_generator(generator_model, task, generator_reward)
        update_fixer(fixer_model, trajectory, fixer_reward)
```

### Curriculum learning

SSR naturally produces a curriculum. Early on the Generator produces simple bugs that the Fixer easily handles. As the Fixer gets stronger, the Generator has to produce harder bugs to keep winning.

```text
Epoch 0-100:    Generator produces simple typos / one-line bugs
Epoch 100-500:  Generator produces multi-file, cross-function bugs
Epoch 500-2000: Generator produces subtle logic errors with cross-module effects
```

This curriculum is **adaptive**—no one has to hand-design a difficulty ladder.

## 12.4.3 SSR's experimental results

Here are SSR's results on SWE-bench Verified:

| Training method | Data source                        | SWE-bench Verified |
| --------------- | ---------------------------------- | ------------------ |
| Meta SWE-RL     | Real PRs + SWE-smith               | 41.0%              |
| DeepSWE         | Real PRs + SWE-smith + world model | 50.0%              |
| **SSR**         | Real PRs + self-play generation    | **47.5%**          |
| SSR + DeepSWE   | All of the above                   | **53.2%**          |

SSR trained on its own—without relying on SWE-smith—reaches 47.5%, proving that self-play data actually works. Combined with DeepSWE's world model, it reaches 53.2%.

### Data efficiency comparison

| Method               | Training data volume     | Accuracy achieved |
| -------------------- | ------------------------ | ----------------- |
| SWE-smith (one-shot) | 50K                      | 41%               |
| SSR (self-play)      | 5K seed + 50K self-play  | 47%               |
| SSR + curriculum     | 5K seed + 100K self-play | 53%               |

**Self-play raises data efficiency**—for the same amount of training data, self-play beats static data by six percentage points.

## 12.4.4 SSR's limitations and future directions

### The Generator can produce invalid bugs

If the Generator learns to produce "syntax error" bugs—bugs the Fixer struggles with simply because they're malformed—that's actually wasted training. Syntax errors are rare in real SWE tasks.

Mitigation: add a "bug realism" reward to the Generator's objective—use an LLM judge to score whether the bug looks like a real-world bug.

### Generator and Fixer can drift out of balance

If the Generator is far stronger than the Fixer, the Fixer never manages to fix anything and training carries no signal. If the Fixer is far stronger than the Generator, the Generator can no longer pose a real challenge and the curriculum stalls.

Mitigation: dynamically adjust how often each model gets updated, to keep the two in balance.

### Domain drift

Bugs generated through self-play can end up distributed differently from real bugs—for example, the Generator might converge on one bug category (typos) while real-world bugs come in far more varieties.

Mitigation: seed the process with real PRs, so the Generator mutates around real bug patterns instead of inventing its own from scratch.

## 12.4.5 Industrial adoption of RL-based SWE

By mid-2026, RL-based SWE had already shipped in multiple products.

### Cursor

[Cursor](https://cursor.sh) is one of the most popular AI code editors. Its core capabilities:

- **Multi-file understanding**: uses RAG to let the model see the whole project
- **Agentic fixes**: the model can autonomously read, edit, and test
- **Built on Claude Opus + tool calling**

Cursor doesn't disclose its training methods, but it's likely using SWE-RL-style training data (GitHub PRs plus internal code).

### Cognition Devin

[Devin](https://devin.ai) is Cognition's "AI software engineer"—capable of independently carrying out an entire development task (planning, writing code, testing, deployment).

Devin's training details aren't public, but Cognition has said in its blog: "our RL training taught Devin the full pipeline from planning to implementation."

### ByteDance Trae

[Trae](https://www.trae.ai) is ByteDance's AI IDE, built on the DeepSWE research. It's active in the domestic Chinese market.

### OpenAI Codex (2025+)

OpenAI relaunched Codex as a code agent built on o3. Its features:

- Uses o3's reasoning ability for complex planning
- Integrates with ChatGPT and can work on multiple tasks in parallel
- Reaches roughly 53% on SWE-bench Verified

### Anthropic Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is Anthropic's CLI tool, built on Claude Opus 4.6/4.7. Its features:

- A reasoning model plus agentic tools
- Long context (200K–1M tokens)
- Reaches 65%+ on SWE-bench Verified

## 12.4.6 Extending to multiple languages and repos

Today's SWE-RL is concentrated in Python. Here's where the extensions are headed:

### Multiple languages

- **JavaScript/TypeScript**: Jest and Mocha are mature test frameworks, and can be handled much like Python
- **Java**: JUnit is mature, but the code style is strict and needs a tighter KL constraint
- **C/C++**: compiled languages with slow test runs, which raises the demand for a world model
- **Go/Rust**: modern languages with generally high test coverage, a good fit for SWE-RL

### Multiple repos

- **Enterprise internal codebases**: every company has its own coding style, dependencies, and test conventions
- **Microservice architectures**: cross-repo changes, API compatibility
- **Legacy systems**: old code, missing tests, incomplete documentation

Extending to multiple repos requires:

- **Fast environment setup**: dependency management differs per repo
- **Domain-specific reward**: what counts as "good code" differs per repo
- **Cross-repo reasoning**: understanding dependencies that span repos

## 12.4.7 Multi-agent collaboration

Complex SWE tasks may need several agents working together:

```text
Planner Agent: analyzes the issue, drafts a fix plan
  ↓
Explorer Agent: locates the relevant files in the repo
  ↓
Editor Agent: implements the change
  ↓
Tester Agent: runs the tests, reports back the result
  ↓
Reviewer Agent: checks code quality
```

This kind of multi-agent collaboration already shows up in Claude Opus 4.7, Cursor, and Devin. Training a system like this requires:

- **Multi-agent RL**: jointly training multiple policies
- **A communication protocol**: how agents pass information to each other
- **A shared value model**: to evaluate the quality of the trajectory as a whole

This is the SWE-specific application of the ideas in [Chapter 20's Agentic RL multi-agent section](../chapter22_agentic/build-agentic-training-system).

## Chapter summary

This chapter laid out the full picture of RL-based SWE:

- **Section 12.1**: SWE-bench and the RLVR paradigm—why SWE is RLVR's ideal battlefield
- **Section 12.2**: Meta SWE-RL—open-source SOTA, GRPO plus a simple reward
- **Section 12.3**: Code World Model + DeepSWE—accelerating training and handling long horizons
- **Section 12.4**: Self-play SSR—a data flywheel, and industrial adoption

**Key takeaways**:

1. **SWE is one of RLVR's most successful industrial applications**—it has a clear answer, automated verification, and massive data.
2. **Simple reward beats complex shaping**—binary test-pass rewards outperform continuous reward shaping.
3. **Long horizons need stronger algorithms**—value models, world models, test-time search.
4. **Self-play is the key to scaling data**—the model generates its own data, and its quality rises with the model's own ability.
5. **Industrial adoption is already mature**—Cursor, Devin, and Claude Code all run on RL-based SWE.

**Coming up next**:

- [Chapter 11 PRM and search](../chapter20_prm_search/intro)—step-level reward in SWE-RL
- [Chapter 12 Reward hacking](../chapter15_rlhf/evaluation)—hacking patterns specific to SWE tasks (like "deleting tests to inflate reward")
- [Section 12.8 Agentic RL training systems](../chapter22_agentic/build-agentic-training-system)—the engineering implementation behind SWE-RL
