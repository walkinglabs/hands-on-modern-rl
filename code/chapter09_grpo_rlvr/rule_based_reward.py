"""
Chapter 9: Verifiable Reward Function — The Core Component of RLVR
==========================================================

This script implements the rule-based reward functions commonly used in
RLVR (Reinforcement Learning with Verifiable Rewards), for automatically
evaluating a model's generated reasoning process and final answer.

Reward function components:
  1. check_answer_correctness  —— checks whether the final answer is correct
  2. check_format               —— checks whether the response format is well-structured
  3. check_reasoning_quality    —— evaluates the quality of the reasoning process
  4. compute_total_reward       —— weighted combination of the above, computing the total reward

How to run:
  python rule_based_reward.py
"""

import re


# ==========================================
# Part 1: Answer correctness check
# ==========================================
def check_answer_correctness(response, ground_truth):
    """
    Extract the final answer from the model's response and compare it against the ground truth

    Extraction rules:
      1. Prefer extracting from \\boxed{...} (LaTeX format)
      2. If there's no boxed answer, try to match patterns like "答案是..." ("the answer is..."),
         "最终答案为..." ("the final answer is...")
      3. Supports integers, decimals, fractions, percentages, negative numbers, etc.

    Args:
        response: the model-generated response text
        ground_truth: the standard answer (string or number)
    Returns:
        dict: {
            "score": float (0.0 or 1.0),
            "extracted": str (the extracted answer),
            "method": str (the extraction method used),
            "correct": bool (whether it is correct),
        }
    """
    extracted = None
    method = "no answer extracted"

    # Method 1: extract from \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        extracted = boxed_match.group(1).strip()
        method = "\\boxed{} extraction"

    # Method 2: match Chinese patterns like "答案是/为/：" ("the answer is/:")
    if extracted is None:
        cn_patterns = [
            r'答案[是为：:]\s*([+-]?\d+\.?\d*)',       # "答案是 42" ("the answer is 42")
            r'最终答案[是为：:]\s*([+-]?\d+\.?\d*)',    # "最终答案为 42" ("the final answer is 42")
            r'结果[是为：:]\s*([+-]?\d+\.?\d*)',         # "结果是 42" ("the result is 42")
            r'所以[，,]\s*(?:答案[是为])?\s*([+-]?\d+\.?\d*)',  # "所以答案是 42" ("so the answer is 42")
        ]
        for pattern in cn_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = match.group(1).strip()
                method = "Chinese pattern match"
                break

    # Method 3: match English patterns like "The answer is ..."
    if extracted is None:
        en_patterns = [
            r'[Tt]he answer is\s*([+-]?\d+\.?\d*)',
            r'[Tt]herefore[,.]?\s*(?:the answer is\s*)?([+-]?\d+\.?\d*)',
            r'[Ss]o the answer is\s*([+-]?\d+\.?\d*)',
        ]
        for pattern in en_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = match.group(1).strip()
                method = "English pattern match"
                break

    # Method 4: extract the last standalone number (last resort)
    if extracted is None:
        all_numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
        if all_numbers:
            extracted = all_numbers[-1]
            method = "last number (fallback)"

    # Compare the answers
    correct = False
    if extracted is not None:
        try:
            # Convert both to float for comparison, with tolerance
            extracted_num = float(extracted)
            truth_num = float(ground_truth)
            correct = abs(extracted_num - truth_num) < 1e-6
        except (ValueError, TypeError):
            # If conversion to a number fails, do an exact string match
            correct = str(extracted).strip() == str(ground_truth).strip()

    score = 1.0 if correct else 0.0

    return {
        "score": score,
        "extracted": extracted if extracted else "(not extracted)",
        "method": method,
        "correct": correct,
    }


# ==========================================
# Part 2: Format compliance check
# ==========================================
def check_format(response):
    """
    Check whether the response has a well-structured reasoning format

    Checks performed:
      1. Whether it contains reasoning step markers (e.g. "步骤" ("step"), "第X步" ("step X"), "Step", etc.)
      2. Whether it contains a final-answer marker (e.g. "答案" ("answer"), "\\boxed{}", etc.)
      3. Whether the response length is reasonable (not too short, not too long)
      4. Whether it contains a mathematical expression

    Args:
        response: the model-generated response text
    Returns:
        dict: {
            "score": float (0.0 ~ 1.0),
            "details": dict (a detailed score for each check),
        }
    """
    details = {}

    # Check 1: reasoning step markers (0.25 points)
    step_patterns = [
        r'步骤\s*\d',         # "步骤1" ("step 1")
        r'第\s*\d+\s*步',     # "第1步" ("step 1")
        r'[Ss]tep\s*\d',      # "Step 1"
        r'\d+\)\s',           # "1) "
        r'首先|然后|接着|最后',  # Chinese connective words (first/then/next/finally)
    ]
    has_steps = any(re.search(p, response) for p in step_patterns)
    details["has_step_markers"] = 0.25 if has_steps else 0.0

    # Check 2: final-answer marker (0.25 points)
    answer_patterns = [
        r'\\boxed\{',         # LaTeX format
        r'答案[是为：:]',       # Chinese marker ("the answer is:")
        r'[Tt]he answer is',  # English marker
        r'最终结果',           # Chinese marker ("final result")
    ]
    has_answer = any(re.search(p, response) for p in answer_patterns)
    details["has_answer_marker"] = 0.25 if has_answer else 0.0

    # Check 3: reasonable response length (0.25 points)
    length = len(response)
    if 20 <= length <= 2000:
        details["reasonable_length"] = 0.25
    elif 10 <= length < 20 or 2000 < length <= 5000:
        details["reasonable_length"] = 0.10
    else:
        details["reasonable_length"] = 0.0

    # Check 4: contains a math expression (0.25 points)
    math_patterns = [
        r'\d+\s*[+\-*/×÷]\s*\d+',   # arithmetic operation: 3 + 5
        r'\d+\s*[=＝]\s*\d+',        # equation: x = 10
        r'[（(][^)]*[)）]',          # parenthesized expression
        r'\\frac|\\sqrt|\\times',     # LaTeX math commands
    ]
    has_math = any(re.search(p, response) for p in math_patterns)
    details["has_math_expression"] = 0.25 if has_math else 0.0

    total_score = sum(details.values())
    return {
        "score": total_score,
        "details": details,
    }


# ==========================================
# Part 3: Reasoning quality evaluation
# ==========================================
def check_reasoning_quality(response):
    """
    Evaluate the basic quality of the reasoning process using heuristic rules

    Dimensions evaluated:
      1. Number of reasoning steps — more steps (within a reasonable range) means more detailed reasoning
      2. Coherence of numeric computation — whether there are intermediate results
      3. Use of logical connectives — whether there is causal/sequential structure
      4. Presence of obvious errors — e.g. dividing by zero, square root of a negative number

    Args:
        response: the model-generated response text
    Returns:
        dict: {
            "score": float (0.0 ~ 1.0),
            "details": dict (a detailed score for each dimension),
        }
    """
    details = {}

    # Dimension 1: number of reasoning steps (0 ~ 0.3 points)
    # Use the number of sentences/lines in the response as a proxy for reasoning steps
    sentences = re.split(r'[。.！!？?\n]', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    num_steps = len(sentences)
    if num_steps >= 5:
        details["step_count"] = 0.30
    elif num_steps >= 3:
        details["step_count"] = 0.20
    elif num_steps >= 1:
        details["step_count"] = 0.10
    else:
        details["step_count"] = 0.0

    # Dimension 2: intermediate computation results (0 ~ 0.3 points)
    # Check for "=" or other computation markers
    calc_patterns = [
        r'\d+\s*[+－\-]\s*\d+\s*[=＝]\s*\d+',   # addition/subtraction
        r'\d+\s*[*×]\s*\d+\s*[=＝]\s*\d+',       # multiplication
        r'\d+\s*[/÷]\s*\d+\s*[=＝]\s*[\d.]+',   # division
        r'=+\s*\d+',                              # "=" followed by a number
    ]
    calc_count = sum(len(re.findall(p, response)) for p in calc_patterns)
    if calc_count >= 3:
        details["intermediate_calc"] = 0.30
    elif calc_count >= 1:
        details["intermediate_calc"] = 0.15
    else:
        details["intermediate_calc"] = 0.0

    # Dimension 3: logical connectives (0 ~ 0.2 points)
    logic_words = [
        '因此', '所以', '因为', '由于', '于是',
        '那么', '从而', '也就是说', '换句话说',
        '根据', '根据题意', '由题意可知',
        '首先', '然后', '接着', '最后',
    ]
    logic_count = sum(1 for word in logic_words if word in response)
    if logic_count >= 3:
        details["logical_connectives"] = 0.20
    elif logic_count >= 1:
        details["logical_connectives"] = 0.10
    else:
        details["logical_connectives"] = 0.0

    # Dimension 4: no obvious errors (0 ~ 0.2 points)
    # Check for common reasoning errors
    error_patterns = [
        r'除以\s*0',           # divide by zero (Chinese)
        r'÷\s*0',             # divide by zero (symbol form)
        r'/\s*0(?!\d)',        # divide by zero (slash form)
        r'负数.*开方',         # square root of a negative number
        r'负数.*开根号',       # square root of a negative number (alt phrasing)
    ]
    has_errors = any(re.search(p, response) for p in error_patterns)
    details["no_errors"] = 0.0 if has_errors else 0.20

    total_score = sum(details.values())
    return {
        "score": min(total_score, 1.0),
        "details": details,
    }


# ==========================================
# Part 4: Weighted total reward computation
# ==========================================
def compute_total_reward(response, ground_truth, weights=None):
    """
    Combine the individual reward components with weights to compute the final total reward

    total_reward = w1 * correctness + w2 * format + w3 * reasoning_quality

    Default weights:
        - Answer correctness (w1 = 0.6): the most important — is the answer right or wrong
        - Format compliance (w2 = 0.15): whether the format is well-structured
        - Reasoning quality (w3 = 0.25): whether the reasoning process is sound

    Args:
        response: the model-generated response text
        ground_truth: the standard answer
        weights: a weight dict {"correctness": w1, "format": w2, "reasoning": w3}
    Returns:
        dict: {
            "total_reward": float (0.0 ~ 1.0),
            "answer_check": dict,
            "format_check": dict,
            "reasoning_check": dict,
        }
    """
    if weights is None:
        weights = {
            "correctness": 0.6,
            "format": 0.15,
            "reasoning": 0.25,
        }

    # Compute each component score
    answer_result = check_answer_correctness(response, ground_truth)
    format_result = check_format(response)
    reasoning_result = check_reasoning_quality(response)

    # Weighted sum
    total_reward = (
        weights["correctness"] * answer_result["score"]
        + weights["format"] * format_result["score"]
        + weights["reasoning"] * reasoning_result["score"]
    )

    return {
        "total_reward": total_reward,
        "answer_check": answer_result,
        "format_check": format_result,
        "reasoning_check": reasoning_result,
        "weights": weights,
    }


# ==========================================
# Part 5: Print reward breakdown
# ==========================================
def print_reward_breakdown(result, response_label=""):
    """
    Print a clearly formatted breakdown of the reward

    Args:
        result: the return value of compute_total_reward
        response_label: a label for the response (used to distinguish test cases)
    """
    print(f"  [{response_label}] Total reward: {result['total_reward']:.4f}")
    print(f"  ├── Answer correctness ({result['weights']['correctness']:.0%} weight): "
          f"{result['answer_check']['score']:.2f}")
    print(f"  │   ├── Extracted answer: {result['answer_check']['extracted']}")
    print(f"  │   ├── Extraction method: {result['answer_check']['method']}")
    print(f"  │   └── Correct: {'yes' if result['answer_check']['correct'] else 'no'}")
    print(f"  ├── Format compliance ({result['weights']['format']:.0%} weight): "
          f"{result['format_check']['score']:.2f}")
    for item, score in result['format_check']['details'].items():
        icon = "+" if score > 0 else "-"
        print(f"  │   ├── [{icon}] {item}: {score:.2f}")
    print(f"  └── Reasoning quality ({result['weights']['reasoning']:.0%} weight): "
          f"{result['reasoning_check']['score']:.2f}")
    for item, score in result['reasoning_check']['details'].items():
        icon = "+" if score > 0 else "-"
        print(f"      ├── [{icon}] {item}: {score:.2f}")
    print()


# ==========================================
# Part 6: Test cases
# ==========================================
def run_tests():
    """
    Validate the reward function's behavior on several test cases

    Test scenarios covered:
      1. Perfect answer — correct answer, well-structured format, thorough reasoning
      2. Correct answer but poor format — no step markers or answer markers
      3. Incorrect answer but detailed reasoning — good format but a miscalculated final answer
      4. Fully non-compliant answer — too short, no reasoning, no format
      5. LaTeX-formatted answer — uses \\boxed{} to mark the answer
    """
    print("=" * 70)
    print("  Verifiable Reward Function Tests")
    print("=" * 70)

    # Define the test cases
    test_cases = [
        {
            "label": "Perfect answer",
            "ground_truth": "42",
            "response": (
                "我们来一步步解决这个问题。\n"
                "步骤1：根据题意，小明有 15 个苹果，小红给了他 27 个。\n"
                "步骤2：所以总数 = 15 + 27 = 42。\n"
                "因此，小明现在一共有 42 个苹果。\n"
                "答案是：42"
            ),
        },
        {
            "label": "Correct answer but poor format",
            "ground_truth": "42",
            "response": "42",
        },
        {
            "label": "Incorrect answer but detailed reasoning",
            "ground_truth": "42",
            "response": (
                "首先，我们需要计算苹果的总数。\n"
                "根据题意，小明有 15 个苹果，小红给了他 27 个。\n"
                "然后，我们计算 15 + 27 = 35。\n"
                "所以，总数 = 35 个苹果。\n"
                "由于这个加法比较简单，我们直接得出结果。\n"
                "答案是：35"
            ),
        },
        {
            "label": "Fully non-compliant",
            "ground_truth": "42",
            "response": "不知道",
        },
        {
            "label": "LaTeX-formatted answer",
            "ground_truth": "36",
            "response": (
                "首先，根据题意我们需要计算 4 × 9。\n"
                "第1步：4 × 9 = 36\n"
                "所以答案是 \\boxed{36}"
            ),
        },
    ]

    # Run the tests and print results
    results = []
    for tc in test_cases:
        print(f"\n{'─' * 70}")
        print(f"  Test case: {tc['label']}")
        print(f"  Ground truth: {tc['ground_truth']}")
        print(f"  Model response: {tc['response'][:80]}{'...' if len(tc['response']) > 80 else ''}")
        print(f"{'─' * 70}")

        result = compute_total_reward(tc["response"], tc["ground_truth"])
        print_reward_breakdown(result, tc["label"])
        results.append((tc["label"], result))

    # ==========================================
    # Part 7: Summary comparison
    # ==========================================
    print("=" * 70)
    print("  Reward Summary Comparison")
    print("=" * 70)
    print()
    print(f"  {'Test case':<20s}  {'Total':>8s}  {'Correct':>8s}  {'Format':>8s}  {'Reason':>8s}")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for label, result in results:
        total = result["total_reward"]
        correct = result["answer_check"]["score"]
        fmt = result["format_check"]["score"]
        reasoning = result["reasoning_check"]["score"]
        print(f"  {label:<20s}  {total:>8.4f}  {correct:>8.2f}  {fmt:>8.2f}  {reasoning:>8.2f}")

    print()
    print("=" * 70)
    print("  Reward Function Design Summary")
    print("=" * 70)
    print("""
  1. Answer correctness (60% weight)
     - The ultimate criterion: a correct answer is correct
     - Supports multiple answer-extraction methods for robustness
     - This is the most important signal in RLVR

  2. Format compliance (15% weight)
     - Encourages the model to produce structured reasoning
     - Includes step markers, answer markers, and math expressions
     - Doesn't force a specific format, but gives a modest bonus for one

  3. Reasoning quality (25% weight)
     - Encourages detailed intermediate computation steps
     - Checks for the use of logical connectives
     - Detects obvious mathematical errors

  Weight design rationale:
     - Answer correctness is the most important factor (60%), but not the only one
     - The quality of the reasoning process also matters (25%), to discourage the model from "getting lucky"
     - Format compliance earns a small bonus (15%), nudging the model toward good habits
    """)


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    run_tests()
