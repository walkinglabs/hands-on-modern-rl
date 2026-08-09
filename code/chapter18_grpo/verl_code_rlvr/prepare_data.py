# prepare_data.py
# Processes PRIME-RL/Eurus-2-RL-Data into a parquet format veRL can consume directly.
#
# Background (issue #53):
#   Eurus-2-RL-Data does not actually have the top-level entry_point / tests /
#   ground_truth fields that the docs describe. Its real structure is:
#     - prompt      : a chat message array ([{"role","content"}, ...]), where the
#                     system message is PRIME's reasoning action template
#                     ([ASSESS]/[ADVANCE]/...), which is useless for code generation
#     - ability     : "math" or "code"
#     - reward_model: {"ground_truth": <answer/tests>, "style": "rule"}
#                     for code samples, ground_truth is a JSON string:
#                     {"inputs": [...], "outputs": [...]} (stdin/stdout test pairs)
#     - data_source / extra_info : metadata
#   veRL's RewardManager reads reward_model["ground_truth"] from the dataset and
#   passes it to the reward function, so this format is already veRL-native and
#   needs no field conversion at training time.
#
# This script does exactly three things:
#   1. Filters to samples with ability == "code" (25K rows)
#   2. Rebuilds the prompt as plain text containing only "the problem statement +
#      code generation instruction" (drops the PRIME reasoning template)
#   3. Filters the train split by max_prompt_length + randomly samples 1000 rows,
#      then saves everything as veRL parquet
#
# Usage:
#   conda activate test
#   python prepare_data.py

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path.home() / "data" / "eurus2"
TRAIN_OUT = DATA_DIR / "train1000.parquet"
VAL_OUT = DATA_DIR / "validation.parquet"

# Filter out samples whose prompt exceeds 512 tokens (1 token ≈ 4 characters)
MAX_PROMPT_CHARS = 512 * 4
N_TRAIN_SAMPLES = 1000

# veRL's RLHFDataset / AgentLoop expect prompt to be a chat message list (list[dict]),
# which gets run through apply_chat_template during rollout. Note: it **cannot** be
# a plain string — when Qwen's apply_chat_template receives a string, it silently
# discards the content and only emits the system + assistant special tokens (24
# tokens total in practice), so the model never sees the problem and reward stays 0.
CODE_GEN_SYSTEM = "You are a competitive programming assistant."
CODE_GEN_USER_TEMPLATE = (
    "Read the problem below and write a Python solution that reads from stdin "
    "and writes to stdout.\n"
    "Return only one Python code block, with no explanations.\n\n"
    "Problem:\n{problem}"
)


def extract_user_content(prompt) -> str:
    """Extracts the user message (the actual problem statement) from the chat message array.

    The raw prompt is [{"role":"system",...},{"role":"user",...}], where the
    system message is PRIME's action template and meaningless for code
    generation, so we keep only the problem statement.
    """
    if isinstance(prompt, str):
        return prompt
    # numpy.ndarray / list of dict
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    if isinstance(prompt, (list, tuple)):
        return "\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in prompt)
    return str(prompt)


def build_prompt(problem: str) -> list:
    """Rebuilds the prompt in chat format: system instruction + user problem statement."""
    return [
        {"role": "system", "content": CODE_GEN_SYSTEM},
        {"role": "user", "content": CODE_GEN_USER_TEMPLATE.format(problem=problem)},
    ]


def prompt_len_chars(prompt) -> int:
    """Character length of the chat-format prompt (used for the overlong filter)."""
    if isinstance(prompt, str):
        return len(prompt)
    return sum(len(m.get("content", "")) for m in prompt)


def process_split(split: str, ds) -> pd.DataFrame:
    df = ds[split].to_pandas()
    code = df[df["ability"] == "code"].copy()
    print(f"[{split}] total rows={len(df)}, of which code={len(code)}")

    # Rebuild prompt: keep only the problem statement + code generation instruction,
    # as a chat message list
    code["prompt"] = code["prompt"].map(lambda p: build_prompt(extract_user_content(p)))

    if split == "train":
        before = len(code)
        code = code[code["prompt"].map(prompt_len_chars) < MAX_PROMPT_CHARS]
        print(f"[{split}] after filtering overlong prompts: {before} -> {len(code)}")
        n = min(N_TRAIN_SAMPLES, len(code))
        code = code.sample(n=n, random_state=42)
        print(f"[{split}] randomly sampled {n} rows")

    # Columns needed natively by veRL: prompt / reward_model / data_source / extra_info
    keep = ["prompt", "reward_model", "data_source", "ability", "extra_info"]
    code = code[keep].reset_index(drop=True)
    return code


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PRIME-RL/Eurus-2-RL-Data ...")
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")

    train = process_split("train", ds)
    train.to_parquet(TRAIN_OUT, index=False)
    print(f"Saved {len(train)} rows -> {TRAIN_OUT}")

    val = process_split("validation", ds)
    val.to_parquet(VAL_OUT, index=False)
    print(f"Saved {len(val)} rows -> {VAL_OUT}")

    # Sanity check: confirm reward_model.ground_truth is parseable I/O tests and prompt is chat format
    row = val.iloc[0]
    gt = json.loads(row["reward_model"]["ground_truth"])
    assert set(gt) >= {"inputs", "outputs"}, f"ground_truth missing inputs/outputs: {list(gt)}"
    assert len(gt["inputs"]) == len(gt["outputs"])
    assert isinstance(row["prompt"], list) and row["prompt"][0]["role"] == "system", \
        f"prompt should be a chat message list, actual type: {type(row['prompt'])}"
    print("\nSanity check passed. Sample row:")
    print("  data_source     :", row["data_source"])
    print("  prompt roles    :", [m["role"] for m in row["prompt"]])
    print("  user content head:", row["prompt"][-1]["content"][:60].replace("\n", "\\n"), "...")
    print("  number of test cases:", len(gt["inputs"]))


if __name__ == "__main__":
    main()
