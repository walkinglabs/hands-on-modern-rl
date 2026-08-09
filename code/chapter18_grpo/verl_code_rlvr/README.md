# 16.8 Hands-On: veRL Code Generation RL Experiment

This directory holds the companion code for _Hands-On Modern Reinforcement Learning_, [Section 16.8](../../../docs/chapter18_grpo/verl-code-sandbox.md): **running PPO training on a code generation task with veRL**.

Code problems share a key advantage with math problems: the answer isn't graded by a human, it **can be verified by running tests**. Passing the tests earns positive reward, failing earns low reward — this is "hard feedback." This section is about getting a model to write programs that actually run.

## The data: what Eurus-2-RL-Data actually looks like

This dataset is easy to get confused about (see [issue #53](https://github.com/walkinglabs/hands-on-modern-rl/issues/53)). It is **not** in the HumanEval-style format with `entry_point` / `tests` fields; its real structure is:

| Field          | Meaning                                                                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `prompt`       | A chat message list. `system` is PRIME's reasoning action template (`[ASSESS]`/`[ADVANCE]`…); `user` holds the actual problem                                            |
| `ability`      | `"math"` or `"code"` — this experiment uses only `code` (25,276 train rows / 1,024 val rows)                                                                             |
| `reward_model` | `{"ground_truth": <answer>, "style": "rule"}`. For code samples, `ground_truth` is a JSON string `{"inputs": [...], "outputs": [...]}`, i.e. **stdin/stdout test pairs** |
| `data_source`  | Origin: `codecontests` / `taco` / `apps` / `codeforces`                                                                                                                  |
| `extra_info`   | `{index, split}`                                                                                                                                                         |

In other words, these are **competitive-programming problems that read from stdin and write to stdout**, not "implement this function signature" problems. The verification info lives in `reward_model.ground_truth`, and veRL passes it to the reward function at training time.

## Getting it running in three steps

### 1. Prepare the data

Filter the dataset down to code samples, rebuild the prompt, sample 1000 rows, and produce the parquet files veRL needs:

```bash
conda activate test
pip install datasets pandas pyarrow
python prepare_data.py
# Output: ~/data/eurus2/train1000.parquet (1000 rows) and validation.parquet (1024 rows)
```

### 2. Validate the reward function (no GPU needed)

`code_reward.py` holds the core logic and can self-check standalone:

```bash
python code_reward.py
# correct code -> score=1.00 pass_rate=1.00 format=1
# wrong code   -> score=0.00 pass_rate=0.00 format=1
# no code      -> score=0.00 pass_rate=0.00 format=0
```

### 3. Start training (GPU required)

```bash
chmod +x run_qwen_coder_ppo_single_gpu.sh
./run_qwen_coder_ppo_single_gpu.sh
```

Actual output from running 8 PPO steps with 48 samples on 1 GPU (validation acc rises from 0 as training proceeds):

```
step:2   critic/score/mean:0.15        # code that passes tests starts appearing in the train set
step:8   val-core/apps/acc/mean@1:0.147
         val-core/codeforces/acc/mean@1:0.153
```

## Key design decisions explained

### Reward: why I/O tests instead of assert

HumanEval-style rewards write tests as `assert two_sum(...) == ...` and then `exec` them. But Eurus-2-RL-Data's code samples don't have that kind of test — `ground_truth` is a set of **stdin/stdout input-output pairs**. So the reward works like this:

1. Extract the ```python code block from the model's response
2. Write the code to a temp `.py` file
3. Spawn an independent process with `subprocess`, feed each input to stdin, and compare stdout against the expected output
4. Return the pass rate as the reward

Using `subprocess` instead of `exec` has two benefits: **full process isolation** (an infinite loop, file operations, or network requests written by the model can't affect the training process) and **realistic execution** (the program genuinely reads from stdin and writes to stdout).

### Prompt: why it must be in chat message format

veRL's RLHFDataset passes `prompt` to the model's `apply_chat_template`. **If `prompt` is a plain string, Qwen's template silently discards the content**, emitting only the system + assistant special tokens (24 tokens total in practice) — the model never sees the problem statement, and reward stays 0.

So `prepare_data.py` rebuilds the prompt in chat format:

```
[{"role": "system", "content": "You are a competitive programming assistant."},
 {"role": "user",   "content": "Read the problem below and write a Python solution...\n\nProblem:\n{problem}"}]
```

### The verl interface: the compute_score signature

veRL's RewardManager calls the reward function with a fixed signature (see `verl/workers/reward_manager/naive.py`):

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Returns {"score": pass_rate, "pass_rate": pass_rate, "format": whether code was extracted}"""
```

- `ground_truth` comes from the dataset's `reward_model["ground_truth"]` (you don't pass it yourself)
- When a dict is returned, veRL uses `"score"` as the main PPO reward; the other keys are logged

The training script must wire it into verl via `custom_reward_function`, or reward won't take effect:

```bash
REWARD=(
    reward_model.enable=False
    custom_reward_function.path="$(pwd)/code_reward.py"
    custom_reward_function.name=compute_score
)
```

## File overview

| File                               | Purpose                                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------------------- |
| `prepare_data.py`                  | Downloads Eurus-2-RL-Data → filters code samples → rebuilds chat-format prompt → samples → parquet |
| `code_reward.py`                   | I/O-based reward: extracts code → runs stdin/stdout tests in a subprocess → returns pass rate      |
| `run_qwen_coder_ppo_single_gpu.sh` | Single-GPU 0.5B PPO launch script (includes `custom_reward_function` wiring)                       |

## Environment notes

Running this experiment requires a machine with a GPU and veRL installed (including the vLLM rollout dependencies). Two easy pitfalls:

- **verl 0.9 requires `transfer_queue`**: `pip install git+https://github.com/Ascend/TransferQueue.git`, otherwise `import transfer_queue` fails immediately at startup.
- **On multi-GPU machines, specify a GPU explicitly**: use `CUDA_VISIBLE_DEVICES` (or `HIP_VISIBLE_DEVICES`) to pick a single GPU, so you don't compete for memory with other jobs.
