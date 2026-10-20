# 20.3 Self-Play SWE-RL

So far, we have discussed the three pillars of SWE-RL:

- **Data**: [SWE-bench](./swe-bench-and-rlvr) (real PRs) + SWE-smith (synthetic bugs)
- **Algorithm**: GRPO + binary reward in [Meta SWE-RL](./meta-swe-rl)
- **Long-horizon training**: [CWM](./world-model-and-deep-swe) learns from environment trajectories, while DeepSWE scales pure RL in real R2E-Gym environments

However, these methods all rely on **pre-collected training data** — SWE-bench or SWE-smith. Data collection itself is expensive and limited.

This section discusses a new direction: **Self-play SWE-RL (SSR)** — letting the model generate its own training data, forming a "data flywheel."

Suppose we have only an open-source repository whose dependencies can be installed and whose tests can run—no human issue and no gold patch. The model first edits the code to make a newly added test fail. It then sees only the test patch and current repository and tries to repair the defect. A successful inject-and-repair round becomes a new training experience.

<img src="../../chapter23_rl_based_swe/images/ssr-self-play-loop.svg" alt="A self-play loop alternates between injecting a verifiable defect and repairing it">

## 20.3.1 Core Idea of Self-play

The inspiration for self-play comes from AlphaGo Zero — **the model plays against itself, learning from the outcomes of the games.** SSR applies this idea to SWE:

```text
┌──────────────────────────────────────────────────────────┐
│ Player A (Bug Generator):                                │
│   - Injects a bug into a repository                      │
│   - Generates corresponding tests (to verify the bug)    │
│   - Generates corresponding issue descriptions          │
├──────────────────────────────────────────────────────────┤
│ Player B (Bug Fixer):                                    │
│   - Sees the issue description                           │
│   - Tries to fix it                                      │
│   - Runs tests to verify                                  │
├──────────────────────────────────────────────────────────┤
│ RL Update:                                               │
│   - Player A learns to "generate harder bugs" (Player B can't fix them) │
│   - Player B learns to "fix more complex bugs"          │
│   - Forming adversarial improvement                      │
└──────────────────────────────────────────────────────────┘
```

### Difference from SWE-smith

[Section 20.1 SWE-smith](./swe-bench-and-rlvr) is **offline synthetic data** — 50K data is generated all at once, then training is performed.

SSR is **online synthetic data** — the model continuously generates data during training, and the data quality improves as the model's capability increases.

| Dimension        | SWE-smith (Offline)                 | SSR (Online)                   |
| ---------------- | ----------------------------------- | ------------------------------ |
| Data Generation  | One-time                            | Continuous during training     |
| Data Difficulty  | Fixed                               | Adjusted with model capability |
| Data Quality     | Independent of generator capability | Improved with model capability |
| Applicable Stage | Early training                      | Entire training process        |

### Data Flywheel of SSR

The core value of SSR is the **data flywheel** — the stronger the model, the better the generated data; the better the data, the stronger the model.

```text
Strong model → Generate hard bugs + excellent fixes → High-quality training data → Model becomes stronger → ...
```

This positive feedback loop makes SSR particularly effective in the later stages of training — the "hard problems" explored by the model itself can break through the model's capability limits more effectively than those designed by humans.

## 20.3.2 Algorithm Details of SSR

The specific design of the **Self-play SWE-RL (SSR)** from Tsinghua University [SSR](https://arxiv.org/abs/2512.18552):

### Bug Generator (Player A)

The Bug Generator is a Large Language Model (LLM) that takes a code repository as input and outputs "bug-injected code + test case + issue description".

```python
def generate_bug(generator_model, repo, file_path):
    # 1. Select a file
    original_code = repo.read(file_path)

    # 2. Let the generator inject a bug
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

    # 3. Validate the bug (the test fails on the bug code, passes on the original code)
    if not validate_bug(original_code, bug_code, test):
        return None  # Invalid bug, discard

    return {
        "original_code": original_Code,
        "bug_code": bug_code,
        "test": test,
        "issue": issue
    }
```

### Bug Fixer (Player B)

Bug Fixer is the trained policy model, which takes an issue and bug code as input and outputs a patch to fix the bug.

```python
def fix_bug(fixer_model, task):
    # 1. Show the fixer the issue and bug code (without the original code)
    prompt = f"""
    Issue: {task['issue']}

    Current code: {task['bug_code']}

    Please fix the bug.
    """

    # 2. Fixer fixes the bug in an agentic manner
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

    # 3. Calculate the reward
    reward = 1.0 if tests_passed else 0.0

    return trajectory, reward
```

### Adversarial Training

```python
def ssr_training(generator_model, fixer_model, repo):
    for epoch in range(N_EPOCHS):
        # 1. Generator generates bugs
        task = generate_bug(generator_model, repo, random_file())

        # 2. Fixer attempts to fix
        trajectory, reward = fix_bug(fixer_model, task)

        # 3. Adversarial reward
        generator_reward = -reward  # Fixer fails to fix → Generator wins
        fixer_reward = reward       # Fixer succeeds in fixing → Fixer wins

        # 4. Update both models
        update_generator(generator_model, task, generator_reward)
        update_fixer(fixer_model, trajectory, fixer_reward)
```

### Curriculum Learning

SSR naturally generates a curriculum — the Generator initially creates simple bugs, which the Fixer can easily fix; as the Fixer becomes stronger, the Generator must produce more complex bugs to win.

```text
Epoch 0-100:  Generator generates simple typos / single-line bugs
Epoch 100-500: Generator generates multi-file, cross-function bugs
Epoch 500-2000: Generator generates subtle logic errors, cross-module impacts
```

This curriculum is **adaptive** — it does not require manual design of difficulty gradients.

## 20.3.3 Experimental Results of SSR

Experimental results of SSR on SWE-bench Verified:

| Training Method | Data Source                        | SWE-bench Verified |
| --------------- | ---------------------------------- | ------------------ |
| Meta SWE-RL     | Real PR + SWE-smith                | 41.0%              |
| DeepSWE         | Real PR + SWE-smith + world model  | 50.0%              |
| **SSR**         | Real PR + self-play generated data | **47.5%**          |
| SSR + DeepSWE   | All data                           | **53.2%**          |

SSR achieves 47.5% performance with training alone (without relying on SWE-smith) — this demonstrates the effectiveness of self-play generated data. Combining it with DeepSWE's world model further improves performance to 53.2%.

### Comparison of Data Efficiency

| Method               | Training Data Size        | Achieved Accuracy |
| -------------------- | ------------------------- | ----------------- |
| SWE-smith (one-time) | 50K                       | 41%               |
| SSR (self-play)      | 5K seeds + 50K self-play  | 47%               |
| SSR + curriculum     | 5K seeds + 100K self-play | 53%               |

**Self-play improves data efficiency** — with the same amount of training data, self-play achieves 6 percentage points higher accuracy than static data.

## 20.3.4 Limitations and Future of SSR

### Generator May Generate Invalid Bugs

If the Generator learns to "generate syntactically incorrect" bugs (which Fixer finds hard to fix), this is actually ineffective training — syntactic errors are rare in real SWE tasks.

Mitigation: Add a "bug authenticity" reward to the Generator's reward — use an LLM judge to determine whether a bug resembles a real bug.

### Imbalance Between Generator and Fixer

If the Generator is much stronger than the Fixer, the Fixer will never be able to fix the issues — training has no signal. If the Fixer is much stronger than the Generator, the Generator will be unable to generate effective challenges — the curriculum stagnates.

**Mitigation:** Dynamically adjust the training frequency of both — maintain balance.

### Domain Shift

The bugs generated by self-play may differ from the real-world bug distribution — for example, the Generator may focus on a particular type of bug (e.g., typos), while real-world bugs are diverse in type.

**Mitigation:** Use real PRs as seeds, allowing the Generator to mutate based on real bug patterns.

## 20.3.5 Industrial Deployment of RL-based SWE

As of mid-2026, RL-based SWE has been deployed in multiple products:

### Cursor

[Cursor](https://cursor.sh) is one of the most popular AI code editors. Its core capabilities include:

- **Multi-file Understanding:** Using RAG to let the model see the entire project
- **Agentic Fixing:** The model can autonomously read, edit, and test
- **Based on Claude Opus + Tool Invocation**

Cursor does not publicly disclose its training methods, but it is speculated that it uses training data similar to SWE-RL (GitHub PRs + internal code).

### Cognition Devin

[Devin](https://devin.ai) is the "AI Software Engineer" developed by Cognition — capable of independently completing full development tasks (planning, writing code, testing, and deployment).

Devin's training details are not publicly disclosed, but Cognition mentioned in a blog post: "Our RL training enables Devin to learn the full process from planning to implementation."

### Byte Trae

[Trae](https://www.trae.ai) is the AI IDE from Byte, based on the research findings of DeepSWE. It is actively operating in the domestic market.

### OpenAI Codex (2025+)

OpenAI has re-released Codex — a code agent based on o3. Its features include:

- Using o3's reasoning capabilities for complex planning
- Integration with ChatGPT to handle multiple tasks in parallel
- Achieving approximately 53% on SWE-bench Verified

### Anthropic Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is the CLI tool from Anthropic, based on Claude Opus 4.6/4.7. Its features include:

- Reasoning model + agentic tools
- Long context (200K-1M)
- Achieving 65%+ on SWE-bench Verified

## 20.3.6 Multi-Language and Multi-Repository Extensions

Currently, SWE-RL is primarily focused on Python. Future directions for expansion include:

### Multi-Language

- **JavaScript/TypeScript**: Jest and Mocha testing frameworks are mature and can be handled similarly to Python.
- **Java**: JUnit testing is mature, but the code style is strict, requiring stronger KL constraints.
- **C/C++**: Compiled languages, with slow test execution, and a greater need for a world model.
- **Go/Rust**: Modern languages with generally high test coverage, making them suitable for S0-RL.

### Multi-Repository

- **Enterprise Internal Code**: Each company has its own code style, dependencies, and testing standards.
- **Microservices Architecture**: Cross-repository modifications and API compatibility.
- **Legacy Systems**: Old code, lack of tests, and incomplete documentation.

Expanding to multiple repositories requires:

- **Fast Environment Setup**: Dependency management for each repository.
- **Domain-Specific Rewards**: Different "good code" standards for different repositories.
- **Cross-Repository Reasoning**: Understanding the dependencies between repositories.

## 20.3.7 Multi-Agent Collaboration

Complex SWE tasks may require multiple agents to collaborate:

```text
Planner Agent: Analyze the issue and develop a repair plan
  ↓
Explorer Agent: Locate relevant files in the repository
  ↓
Editor Agent: Implement the changes
  ↓
Tester Agent: Run tests and provide feedback
  ↓
Reviewer Agent: Check code quality
```

This type of multi-agent collaboration has already appeared in systems like Claude Opus 4.7, Cursor, and Devin. Training such systems requires:

- **Multi-Agent RL**: Joint training of multiple policies.
- **Communication Protocol**: How agents pass information to each other.
- **Shared Value Model**: Evaluating the overall quality of a trajectory.

This is a specific application of the [Chapter 19 on Agentic RL Multi-Agent Systems](../chapter22_agentic/build-agentic-training-system) in the domain of SWE.

## Chapter Summary

In this chapter, we have reviewed the full picture of RL-based Software Engineering (SWE):

- **Section 20.1**: SWE-bench and the RLVR paradigm — why SWE is an ideal battlefield for RLVR
- **Section 20.1**: Meta SWE-RL — open-source SOTA, GRPO + simple reward
- **Section 20.2**: Code World Model + DeepSWE — accelerating training + handling long horizons
- **Section 20.3**: Self-play SSR — data flywheel, industrial deployment

**Key Takeaways**:

1. **SWE is one of the most successful industrial applications of RLVR** — clear answers, automated verification, and massive data
2. **Simple reward > complex shaping** — binary test pass rates are more effective than continuous rewards
3. **Long horizons require stronger algorithms** — value models, world models, and test-time search
4. **Self-play is key to data expansion** — models generate their own data, with quality improving as capability increases
5. **Industrial deployment is already mature** — Cursor, Devin, and Claude Code all use RL-based SWE

**Next Chapters**:

- [Chapter 17: PRM and Search](../chapter20_prm_search/outcome-vs-process) — Step-level reward in SWE-RL
- [Chapter 25: Reward Hacking](../chapter30_alignment_failures/classical-failures) — Reward hacking in SWE tasks (e.g., "delete tests to increase reward")
- [Section 19.10: Agentic RL Training Systems](../chapter22_agentic/build-agentic-training-system) — Engineering implementation of SWE-RL
