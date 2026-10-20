# 18.5 Large-Scale RL Data Engineering

In a single-machine experiment, training data may be a script, a JSONL file, or one dataset. In industrial RL training, a trajectory passes through task generation, environment execution, scoring, storage, further processing, model updates, and feedback into the next round. A distributed run can generate hundreds of thousands of trajectories per hour. If any stage is poorly managed, large amounts of data are lost or erroneous trajectories repeatedly return to training.

This section follows the lifecycle of one trajectory: how a task pool grows from existing datasets and real failures, how the runtime records trajectories, rewards, and intermediate states, how quality control removes invalid samples and non-executable trajectories, and how high-quality data returns to the next round of SFT or reward-model training.

---

## Task Production: From Fixed Datasets to Real Failures

**Why one data source is insufficient.** Early RL training often starts from public mathematics, coding, or reasoning datasets. Their quality is stable, but their task distribution is fixed. Once a model has learned those patterns, improvement slows. Failures from real use contain new tool combinations, long reasoning chains, and boundary cases that fixed datasets rarely cover.

Industrial training therefore combines several sources:

- **Academic and public benchmarks:** MATH, GSM8K, AIME, MMLU, HumanEval, MBPP, and similar datasets cover common mathematical and coding skills and work well for cold starts and regression tests.
- **Synthetic tasks:** templates, programmatic construction, model rewriting, and difficulty control expand the task pool and target specific formats or reasoning patterns.
- **Interactive-environment tasks:** trajectories involving terminals, browsers, code sandboxes, or external tools cover computer use, software engineering, and multi-turn tool use.
- **Real failure samples:** failures from internal testing, public testing, or production use include incorrect answers, failed tool calls, malformed output, and inappropriate refusals.
- **Safety and adversarial data:** jailbreak attempts, sensitive questions, prompts designed to trigger formatting failures, and high-risk tasks teach the model to preserve constraints while solving the task.

MiniCPM 5 uses about 30,000 high-quality seed prompts: roughly two thirds cover general reasoning, mathematics, and coding, while one third covers tool use and agent tasks. In addition to public competition datasets, Kimi K3 extracts tasks from pretraining corpora and combines them with real-world feedback. Qwen-AgentWorld begins with 5,000 seed tasks across 20 domains and expands them to 30,000 tasks over two weeks by running on 60 GPUs in parallel.

**Difficulty control and data scheduling.** Once the task pool grows, tasks should not all appear at the same frequency. Tasks that are too easy provide little update signal; tasks that are too hard fail continuously and create high-variance gradients. Production systems commonly combine the following policies:

- **Prioritize the capability boundary:** track success rates by difficulty and task type, then sample tasks whose recent success rate is between $0.3$ and $0.7$ more frequently.
- **Emphasize failure cases:** return real errors and high-value failures to the task pool with increased sampling probability.
- **Counter-skew sampling:** reduce the frequency of task types that already succeed at high rates and allocate capacity to weaker areas.
- **Stage the curriculum:** emphasize foundational skills early, then add long-horizon, multi-tool, and multi-constraint tasks later.
- **Continue injecting safety tasks:** even after the main task stabilizes, retain a small proportion of safety and formatting constraints so that capability gains do not erase boundaries.

**Task formats and prompt construction.** The same task can be presented through several prompt forms. A mathematics problem may request only the conclusion or require the reasoning steps. A coding task may provide partial code or only a natural-language description. A tool task may specify the procedure or state only the goal.

```text
Goal: Fix a Python function
Input: Faulty code and failed test output
Format A: Modify the code directly
Format B: Explain the cause, then modify the code
Format C: Run the tests, inspect the error, modify the code, and run the tests again
```

If training uses only one form, the model may learn only one response pattern. Industrial data pipelines prepare multiple prompt templates, output formats, and context lengths for the same task and mix them during sampling.

**Building executable environments.** Tool use, code execution, and browser tasks require a runnable environment rather than a static prompt.

- **Sandbox isolation:** code execution, terminal access, and web access run in isolated environments so that training tasks cannot affect external systems.
- **Environment standardization:** rollout nodes use the same language and tool versions, dependencies, and network permissions. Otherwise the same trajectory may produce different results on different nodes.
- **State reset:** reset the file system, database, browser state, and tool sessions before every trajectory so that samples remain independent.
- **Timeouts and retries:** tool calls may hang, time out, or fail. The data system records the failure type instead of silently discarding the sample.
- **Multi-environment orchestration:** systems such as Qwen-AgentWorld maintain separate environments for different domains and expose them through a shared interface.

Task production aims to keep data concentrated around the current model's capability boundary, real failure modes, and safety boundary, rather than merely increasing volume.

---

## Data Storage: What One Rollout Must Record

**A minimal training record is insufficient for debugging.** Many PPO or GRPO implementations save only the prompt, response, reward, and log-probability. These fields are enough for one gradient update, but they cannot explain why rewards suddenly fall, why one response pattern becomes more common, or why a verifier begins reporting large numbers of errors.

Industrial systems usually save a trajectory at several levels:

- **Task level:** prompt, task type, difficulty label, source, environment configuration, and prompt-template version.
- **Trajectory level:** complete model response, tokens and log-probabilities at each step, generation time, temperature, sampling parameters, and model version.
- **Interaction level:** tool requests, tool returns, intermediate execution output, browser actions, and error messages.
- **Reward level:** value and weight from every reward function, aggregation rule, formatting checks, number of unit tests passed, and final score.
- **System level:** rollout-worker identifier, timestamp, model-checkpoint identifier, environment-image version, and random seed.

**Reward-model training also requires paired preferences.** If trajectories will later train a reward model, the system must retain the question, multiple responses to the same question, the preference relation among them, and the source of that judgment—rules, human annotation, or a model judge. MiniCPM 5 automatically generates prompts, performs rollouts, records metadata, and prepares paired preference data for reward-model training during each RL iteration.

**Large-scale storage and indexing.** One month of distributed RL can easily produce several to tens of terabytes of trajectory data. Storage design commonly includes:

- **Columnar and tiered storage:** store raw text, tensors, and logs separately, and index frequently queried statistics.
- **Compression and deduplication:** repeated rollouts for one prompt can share prompt storage, while tensors use formats suitable for training recovery.
- **Checkpoint binding:** bind every trajectory to the checkpoint and parameter version that generated it so that it can be reproduced and replayed.
- **Failure labels:** label timeouts, environment errors, malformed output, and reward-computation failures separately. Do not mix them directly into training or delete them immediately.
- **Reproducibility metadata:** record random seeds, environment and dependency versions, and prompt-template versions so that the same configuration can be rerun when necessary.

**More fields are not always better.** Every additional field increases storage, cleaning, and query costs. Preserve fields that affect gradient computation, reward reproduction, and fault diagnosis. Large intermediate tensors with little debugging value can be sampled or replaced by aggregate statistics.

---

## Quality Control: Stop Errors Before Training

**A correct reward does not guarantee a learnable trajectory.** A high score may result from a verifier vulnerability, answer leakage, a formatting bypass, or an environment fault. A low score may result from a broken reward function, an incorrect test, or a tool timeout. Without filtering, these errors enter the gradient directly.

Common data-quality checks include:

- **Format validation:** verify the required structure, tags, fields, JSON, and code blocks.
- **Reward sanity checks:** confirm that scores lie in the expected range, component rewards do not conflict, and perfect or zero scores have not suddenly become universal.
- **Executability checks:** compile or interpret code, validate tool arguments, and confirm that browser actions can actually run in the environment.
- **Duplication and leakage detection:** detect responses that copy reference answers without derivation, reuse fixed training templates, or expose system prompts and environment output.
- **Abnormal-trajectory filtering:** flag unusually long or short generation, tool-call loops, repeated calls to one tool, and meaningless output.
- **Environment-consistency checks:** compare the same task across nodes. If results differ, mark the environment as unstable instead of assigning an immediate zero reward.

**Rule-based filtering precedes model-based filtering.** Rules can quickly catch formatting errors, syntax errors, timeouts, and reward-service failures. Reward models or additional judge models are reserved for cases that rules cannot decide.

**Filtering does not mean deleting.** Rejected trajectories still provide useful evidence:

- environment errors and tool timeouts reveal sandbox and dependency faults;
- formatting failures help improve prompt templates and constraint training;
- reward anomalies expose reward hacking and verifier vulnerabilities;
- repeatedly failed high-value tasks can enter human annotation, prompt revision, or environment repair queues.

**Data-quality monitoring.** During training, systems continually track:

- mean reward, pass rate, and length distribution for each task type;
- perfect-score, zero-score, and reward-clipping rates;
- formatting-error, tool-failure, and environment-error rates;
- response duplication, fixed-template matches, and suspicious bypass behavior;
- differences among workers and time windows.

If one rollout node suddenly reports a much higher perfect-score rate than the others, its environment version may differ or its cache may leak answers. If one task type abruptly receives perfect scores across the board, the reward function may have been bypassed. If response length keeps falling without an improvement in reward, the model may have found a short-answer shortcut.

**Defending against data poisoning and reward hacking.** Industrial systems often encounter behavior that exploits the training system rather than obvious malformed data. Examples include:

- emitting a special statement that skips tests in a coding task;
- repeating keywords from the question to trigger a matching reward;
- choosing a tool path that always returns success without completing the goal;
- using crafted formatting in multi-turn interaction to interfere with a judge model.

These patterns are rarely visible from one sample alone. Detection must combine response distributions, tool-call sequences, decomposed reward items, and human sampling. Once a high-risk vulnerability is confirmed, repair the reward function or environment before resuming training.

---

## Data Feedback: From RL Trajectories to the Next Training Round

**RL data is not used only once.** Large-scale training is rarely one dataset followed by one PPO run. Responses, preferences, and failures generated during RL return to SFT, reward-model training, and the task pool for the next RL round.

### Returning to SFT

High-quality responses verified by rules, passing tests, or human review can become supervised fine-tuning data:

- distill complete high-scoring responses back into the base model to stabilize formats and solution procedures;
- convert trajectories that recover from failure into error-to-correction pairs;
- structure tool-use trajectories so that the model learns when to call a tool and how to interpret its result;
- filter long responses so that verbose but ineffective reasoning does not return wholesale to SFT.

SFT feedback stabilizes output formats, restores general capabilities degraded during RL, and consolidates newly learned tool behavior into a stronger starting checkpoint.

### Returning to the Reward Model

RL rollouts naturally produce several responses to the same prompt and therefore supply reward-model or judge-model data:

- use a high-scoring response as `chosen` and a low-scoring response as `rejected` for the same prompt;
- use rules for clear errors and supplement difficult preference cases with human labels or model judges;
- construct pairs by task type, difficulty, and score gap so that the reward model learns more than extreme good-versus-bad distinctions;
- group safety, factuality, and formatting constraints separately so that the main-task score does not overwhelm boundary signals.

The updated reward model then returns to RL, creating a joint iteration loop between the policy and reward model. MiniCPM 5 uses trajectories generated in each RL round to train the next reward model, forming this feedback cycle.

### Returning to the Task Pool

Failed trajectories expand the task pool in the opposite direction:

- repeatedly missed real tasks re-enter the next round with increased sampling weight;
- new tools, APIs, and page types become new training tasks;
- discovered reward-hacking paths become adversarial tasks that test the vulnerability;
- anonymized real user requests bring the task distribution closer to deployment.

### Closing the Loop with Safety Data and Evaluation Sets

Industrial post-training also places red-team tests, safety evaluations, and formatting-robustness tests inside the data loop:

- jailbreak and high-risk requests continually enter safety and refusal training;
- prompts that trigger malformed output, hallucination, or tool misuse enter the regression suite;
- every model update is evaluated on both a fixed benchmark and newly collected failures so that capability gains do not reintroduce old problems.

**Data versioning.** Task sets, reward functions, prompt templates, environment images, and cleaning rules all require versions. Every experiment must be able to answer which prompts, environment, reward version, and filtering rules it used. Without versioning, even a better checkpoint cannot be traced back to the data change that produced it.

The scale of industrial RL is therefore measured by more than its GPU count. Data must be produced, stored, cleaned, and fed back continuously while remaining traceable across repeated training rounds.

---

## Section Summary

- A task pool cannot rely only on fixed public datasets. It must combine synthetic tasks, interactive environments, real failures, and safety data, with sampling scheduled around the model's capability boundary.
- A rollout must preserve task, trajectory, interaction, reward, and system metadata for gradient computation, diagnosis, and reproduction.
- Quality control covers formatting, executability, reward sanity, environment consistency, and reward-hacking detection.
- RL data returns to SFT, reward-model training, the task pool, and safety evaluation, forming a continuous iteration loop.
- Large-scale RL data engineering aims to keep data executable, verifiable, traceable, and reusable across training rounds rather than simply storing more of it.

This completes the path from single-machine algorithms through preference optimization, industrial feedback loops, stability, distributed training, and data engineering. For capability emergence, pure-RL reasoning training, test-time scaling, and hybrid thinking, revisit [Chapter 16](../chapter19_reasoning/emergence-and-o1). The next chapter, [Chapter 19: Agentic RL](../chapter22_agentic/overview), studies training with tool calls and environment interaction.

## Further Reading

- [MiniCPM5 Technical Report](https://arxiv.org/abs/2601.04962)
- [Kimi K3 Technical Report](https://arxiv.org/abs/2504.12593)
- [Qwen-AgentWorld](https://arxiv.org/abs/2506.07340)
- [SimpleRL-Zoo](https://github.com/hiyouga/SimpleRL-Zoo)
- [SLIME](https://arxiv.org/abs/2602.02779)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
