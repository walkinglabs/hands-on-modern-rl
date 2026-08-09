import re
from typing import Any

REWARD_NAME = "gsm8k"
REWARD_TYPE = "sequential"


def extract_answer(response: str) -> str | None:
    """Extract the final answer from the model output."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()

    answer_tag = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()

    lines = response.strip().split("\n")
    for line in reversed(lines):
        numbers = re.findall(r"-?\d+\.?\d*", line)
        if numbers:
            return numbers[-1]

    return None


def check_answer(predicted: str | None, ground_truth: str) -> float:
    """Compare the predicted answer against the ground truth."""
    if predicted is None:
        return 0.0

    try:
        predicted_value = float(predicted.replace(",", ""))
        ground_truth_value = float(ground_truth.replace(",", ""))
        return 1.0 if abs(predicted_value - ground_truth_value) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0


def compute_score(reward_input: dict[str, Any], **kwargs) -> dict[str, float]:
    """veRL custom reward entry point."""
    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]

    predicted = extract_answer(response)
    accuracy = check_answer(predicted, ground_truth)

    return {
        "overall": accuracy,
        "accuracy": accuracy,
        "format": 1.0 if predicted is not None else 0.0,
    }

