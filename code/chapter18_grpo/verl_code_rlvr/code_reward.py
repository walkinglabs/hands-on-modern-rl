# code_reward.py
# Reward function for veRL code-generation RLVR: runs the generated code as a
# standalone program against stdin/stdout tests.
#
# Background (issue #53):
#   The original docs assumed Eurus-2-RL-Data had a tests field (Python assert
#   statements) that could be exec'd directly. In reality, the dataset's code
#   samples have no tests / entry_point; reward_model.ground_truth is a JSON
#   string {"inputs": [...], "outputs": [...]} (stdin/stdout test pairs).
#   So the approach here is: extract the code -> write it to a temp file ->
#   execute it as a real subprocess, feeding each input to stdin, comparing
#   stdout against the expected output, and returning the pass rate.
#
# veRL interface (verl/workers/reward_manager/naive.py):
#   score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
#   ground_truth comes from the dataset's reward_model["ground_truth"].
#   When a dict is returned, "score" is used as the main reward; the other
#   keys are logged as extra info.
#
# Usage:
#   python code_reward.py            # sanity check: runs a few constructed cases to verify reward logic
#
# Wired into training via the verl config:
#   custom_reward_function.path=.../code_reward.py
#   custom_reward_function.name=compute_score

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_TIMEOUT_S = 10.0


def extract_code(response: str) -> str:
    """Extracts the Python code block from the model's output.

    The model typically outputs something like:
        "```python\ndef solve():\n    ...```"
    We take only the part between ```python and ```.
    If the model didn't output a code block, fall back to treating the whole
    response as code (usually results in a syntax error and reward=0).
    """
    match = _CODE_BLOCK_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def _normalize(text: str) -> str:
    """Strips trailing whitespace from each line before comparing, to avoid
    false mismatches caused by \r or extra blank lines."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def run_io_tests(code: str, ground_truth_json: str, timeout_s: float = _TIMEOUT_S):
    """Runs code as a standalone program, testing it against the inputs/outputs in ground_truth.

    Returns (pass_rate, detailed results for the first few tests). Any
    exception (syntax error, crash, timeout, output mismatch) only affects
    the corresponding test case and doesn't abort the overall scoring.
    """
    try:
        tests = json.loads(ground_truth_json)
    except (TypeError, json.JSONDecodeError) as exc:
        return 0.0, f"failed to parse ground_truth: {exc!r}"

    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])
    if not inputs or len(inputs) != len(outputs):
        return 0.0, f"inputs/outputs count mismatch: {len(inputs)} vs {len(outputs)}"

    # Write the code to a temp .py file and execute it in a separate Python
    # interpreter process — safer than exec: full process isolation means an
    # infinite loop or file operations in the generated code can't affect the
    # training process.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        passed = 0
        details = []
        for inp, expected in zip(inputs, outputs):
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                if proc.returncode != 0:
                    details.append("FAIL(nonzero exit)")
                    continue
                got = _normalize(proc.stdout)
                want = _normalize(expected)
                if got == want:
                    passed += 1
                    details.append("PASS")
                else:
                    details.append("FAIL(output mismatch)")
            except subprocess.TimeoutExpired:
                details.append("FAIL(timeout)")
            except Exception as exc:  # noqa: BLE001
                details.append(f"FAIL({exc!r})")
        return passed / len(inputs), "; ".join(details[:5])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """veRL reward entry point.

    Args:
        data_source: dataset source (in this experiment, one of codecontests/taco/apps/codeforces)
        solution_str: the model's full generated response (markdown text)
        ground_truth: dataset's reward_model["ground_truth"]; for code samples this is a JSON string of I/O tests
        extra_info: dataset's extra_info column (this dataset only has index/split, unused)

    Returns:
        dict: {"score": pass_rate, "pass_rate": pass_rate, "format": whether code was extracted}
        veRL uses "score" as the main PPO reward.
    """
    # format only indicates whether the code was output as a ```python code block.
    # When not formatted correctly, extract_code falls back to running the whole
    # response as code (usually a syntax error, score=0), but the format metric
    # should honestly reflect "did the model learn to output a code block".
    match = _CODE_BLOCK_RE.search(solution_str)
    format_ok = 1.0 if match else 0.0
    code = extract_code(solution_str)
    if not code:
        return {"score": 0.0, "pass_rate": 0.0, "format": 0.0}

    pass_rate, detail = run_io_tests(code, ground_truth)
    return {"score": pass_rate, "pass_rate": pass_rate, "format": format_ok}


# ---------------------------------------------------------------------------
# Sanity check: verifies the reward logic directly, no training environment needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Construct ground_truth (inputs/outputs) using a simple A+B problem
    ab_gt = json.dumps({"inputs": ["1 2", "10 20", "-3 5"], "outputs": ["3", "30", "2"]})

    correct = "```python\nimport sys\n\n\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)\n```"
    wrong = "```python\nimport sys\n\n\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a - b)\n```"
    no_code = "I don't know how to solve this."

    for name, resp in [("correct code", correct), ("wrong code", wrong), ("no code", no_code)]:
        result = compute_score("synthetic", resp, ab_gt, None)
        print(f"{name:8s} -> score={result['score']:.2f} pass_rate={result['pass_rate']:.2f} "
              f"format={result['format']:.0f}")
