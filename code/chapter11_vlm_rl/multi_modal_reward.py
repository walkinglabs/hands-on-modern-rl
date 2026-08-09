"""
Chapter 11: Multi-Modal Reward Function Design
==========================================================

This script demonstrates designing a multi-modal reward function for VLM
(Vision-Language Model) reinforcement learning:
  1. reward_correctness:       answer correctness reward (0.0 or 1.0)
  2. reward_reasoning_quality: reasoning quality reward (0.0~0.5)
  3. reward_format:            format compliance reward (0.0~0.2)
  4. reward_visual_grounding:  visual grounding reward (0.0~0.3)
  5. compute_total_reward:     weighted total score

Reward design principles:
  - Correctness is the core signal and carries the highest weight
  - Reasoning quality encourages "showing your work"
  - Format compliance ensures the output can be parsed
  - Visual grounding encourages the model to genuinely "look at" the image

How to run:
  python multi_modal_reward.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)

# Set a CJK-capable font to ensure chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Reward function definitions
# ==========================================

def reward_correctness(response, ground_truth):
    """
    Correctness reward: check whether the numbers in the response match the annotations

    Logic:
      1. Extract the count for each shape from the response
      2. Compare each against ground_truth
      3. 1.0 if all correct, otherwise 0.0

    Args:
        response: the model's text response
        ground_truth: dict, {'三角形': int, '圆形': int, '正方形': int}
    Returns:
        float: 0.0 (incorrect) or 1.0 (correct)
    """
    # Try to extract the number corresponding to each shape from the response
    extracted = {}

    for shape_name in ['三角形', '圆形', '正方形']:
        # Pattern 1: shape name + number, e.g. "三角形3个" or "三角形：3" or "三角形有3个"
        patterns = [
            rf'{shape_name}[^0-9]*?(\d+)',
            rf'(\d+)\s*个\s*{shape_name}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # Take the last match (usually the final conclusion)
                extracted[shape_name] = int(matches[-1])
                break

    # If not all shape counts could be extracted, return 0 directly
    for shape_name in ground_truth:
        if shape_name not in extracted:
            return 0.0

    # Compare each one
    for shape_name, expected_count in ground_truth.items():
        if extracted.get(shape_name, -1) != expected_count:
            return 0.0

    return 1.0


def reward_reasoning_quality(response):
    """
    Reasoning quality reward: check whether the response includes step-by-step reasoning

    Scoring criteria (0.0~0.5):
      - 0.0: no reasoning steps at all, answer given directly
      - 0.1~0.2: briefly mentions the process
      - 0.3~0.4: contains clear step-by-step reasoning
      - 0.5: complete step-by-step reasoning with analysis

    Judged by:
      - Whether it contains step markers ("第一步", "Step 1", "1.", etc.)
      - Whether it contains reasoning keywords ("因为", "所以", "因此", "分析", "计算")
      - Whether the response is long enough to contain reasoning

    Args:
        response: the model's text response
    Returns:
        float: 0.0~0.5
    """
    score = 0.0

    # Check for step markers (0.2 points)
    step_markers = [
        r'第[一二三四五六七八九十\d]+步',
        r'[Ss]tep\s*\d+',
        r'首先',
        r'然后',
        r'接着',
        r'最后',
    ]
    step_count = sum(1 for m in step_markers if re.search(m, response))
    if step_count >= 3:
        score += 0.2
    elif step_count >= 1:
        score += 0.1

    # Check for reasoning keywords (0.2 points)
    reasoning_keywords = [
        '因为', '所以', '因此', '分析', '计算',
        '可以', '发现', '观察', '总共', '合计',
    ]
    keyword_count = sum(1 for kw in reasoning_keywords if kw in response)
    if keyword_count >= 3:
        score += 0.2
    elif keyword_count >= 1:
        score += 0.1

    # Check response length (0.1 points) — must be long enough to plausibly contain reasoning
    if len(response) >= 50:
        score += 0.1

    return min(score, 0.5)


def reward_format(response):
    """
    Format compliance reward: check whether the response follows the expected output format

    Scoring criteria (0.0~0.2):
      - 0.0: no formatting at all
      - 0.05: partial formatting (e.g. lists some shape names)
      - 0.1: format is roughly correct (mentions all three shapes)
      - 0.15: good formatting (all three shapes have corresponding counts)
      - 0.2: perfect formatting (summary conclusion + all three shape counts present)

    Expected format example:
      "图片中有 X 个三角形、Y 个圆形和 Z 个正方形，总共 N 个形状。"

    Args:
        response: the model's text response
    Returns:
        float: 0.0~0.2
    """
    score = 0.0

    # Check whether all three shapes are mentioned (0.1 points)
    shapes_mentioned = sum(1 for s in ['三角形', '圆形', '正方形'] if s in response)
    if shapes_mentioned == 3:
        score += 0.1
    elif shapes_mentioned >= 2:
        score += 0.05

    # Check for a summary statement (0.05 points)
    summary_patterns = [
        r'总共\s*\d+',
        r'合计\s*\d+',
        r'一共\s*\d+',
        r'总计\s*\d+',
        r'答[：:]',
    ]
    if any(re.search(p, response) for p in summary_patterns):
        score += 0.05

    # Check for a number-shape pairing (0.05 points)
    has_number_shape_pair = bool(re.search(r'\d+\s*个', response))
    if has_number_shape_pair:
        score += 0.05

    return min(score, 0.2)


def reward_visual_grounding(response, image_info):
    """
    Visual grounding reward: check whether the response correctly references
    the visual features actually present in the image

    Scoring criteria (0.0~0.3):
      - Check whether the response mentions shapes that are actually present in the image
      - For mentioned shapes, whether a visual description is given (color, position, etc.)
      - Whether the model shows signs of "actually looking at the image"

    image_info example:
      {
        'present_shapes': ['三角形', '圆形'],  # shapes actually present in the image
        'absent_shapes': ['正方形'],             # shapes not present in the image
      }

    Args:
        response: the model's text response
        image_info: dict containing the shape information for the image
    Returns:
        float: 0.0~0.3
    """
    score = 0.0

    present_shapes = image_info.get('present_shapes', [])
    absent_shapes = image_info.get('absent_shapes', [])

    # Check whether the present shapes are mentioned (0.15 points)
    mentioned_present = sum(1 for s in present_shapes if s in response)
    if len(present_shapes) > 0:
        ratio = mentioned_present / len(present_shapes)
        score += 0.15 * ratio

    # Check whether absent shapes are correctly reported as absent (0.1 points)
    # If a shape is absent and the model says "0 个" or "没有", that is correct visual grounding
    for shape in absent_shapes:
        if shape in response:
            # Check whether it's correctly labeled as 0
            zero_patterns = [
                rf'{shape}[^0-9]*?0',
                rf'0\s*个\s*{shape}',
                rf'没有\s*{shape}',
            ]
            if any(re.search(p, response) for p in zero_patterns):
                score += 0.05
                break  # cap the bonus at 0.05

    # Check for visually descriptive language (0.05 points)
    visual_keywords = [
        '颜色', '红色', '蓝色', '绿色', '橙色', '紫色',
        '位置', '左边', '右边', '上方', '下方',
        '大小', '大', '小',
        '可以看到', '图中', '图片中', '画布上',
    ]
    visual_count = sum(1 for kw in visual_keywords if kw in response)
    if visual_count >= 2:
        score += 0.05

    return min(score, 0.3)


def compute_total_reward(response, ground_truth, image_info):
    """
    Compute the weighted total reward

    Weight distribution:
      - Correctness:        × 1.0  → max 1.0
      - Reasoning quality:   × 1.0  → max 0.5
      - Format compliance:   × 1.0  → max 0.2
      - Visual grounding:    × 1.0  → max 0.3
      ---------------------------------------------
      Theoretical max total score: 2.0

    Args:
        response: the model's text response
        ground_truth: dict, annotation data
        image_info: dict, image information
    Returns:
        dict containing each component reward and the total reward
    """
    r_correct = reward_correctness(response, ground_truth)
    r_reasoning = reward_reasoning_quality(response)
    r_format = reward_format(response)
    r_visual = reward_visual_grounding(response, image_info)

    total = r_correct + r_reasoning + r_format + r_visual

    return {
        'correctness': r_correct,
        'reasoning': r_reasoning,
        'format': r_format,
        'visual_grounding': r_visual,
        'total': total,
    }


# ==========================================
# Part 2: Test case definitions
# ==========================================
# 8 test cases covering responses of varying quality:
#   perfect response, correct but short, wrong but with reasoning, wrong and confused,
#   partially correct, well-formatted but wrong content, has visual description, no format no reasoning

test_cases = [
    {
        'name': 'Perfect response (correct + reasoning + format + visual grounding)',
        'response': (
            '我仔细观察了图片。\n'
            '首先，分析图中的三角形：我数到了 3 个三角形，分别在左上方和右侧。\n'
            '然后，分析圆形：图片中有 1 个圆形，颜色是蓝色，位于中间偏左。\n'
            '接着，分析正方形：图片中有 2 个正方形，一个红色一个绿色。\n'
            '所以，图片中总共有 3 个三角形、1 个圆形和 2 个正方形，合计 6 个形状。\n'
            '答：三角形3个，圆形1个，正方形2个，总共6个。'
        ),
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': 'Correct but short (no reasoning process)',
        'response': '三角形3个，圆形1个，正方形2个。',
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': 'Wrong but with reasoning process',
        'response': (
            '让我一步步来分析。\n'
            '首先，观察图片中的三角形。可以看到有 2 个三角形。\n'
            '然后，观察圆形。图片中有 1 个圆形。\n'
            '最后，观察正方形。有 2 个正方形。\n'
            '所以，总共是 2+1+2=5 个形状。\n'
            '答：三角形2个，圆形1个，正方形2个。'
        ),
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': 'Wrong and confused',
        'response': '图里有好多形状，大概是三角形2个圆形3个正方形5个吧。',
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': 'Partially correct (only one shape counted correctly)',
        'response': (
            '我来看一下图片中的形状。\n'
            '三角形有 3 个，这个我数对了。\n'
            '圆形有 2 个，正方形有 1 个。\n'
            '总共 6 个形状。'
        ),
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': 'Well-formatted but wrong content',
        'response': (
            '答：经过分析，图片中包含以下形状：\n'
            '三角形0个，圆形5个，正方形5个，总共10个。'
        ),
        'ground_truth': {'三角形': 0, '圆形': 2, '正方形': 3},
        'image_info': {
            'present_shapes': ['圆形', '正方形'],
            'absent_shapes': ['三角形'],
        },
    },
    {
        'name': 'Has visual description but the answer is wrong',
        'response': (
            '观察图片，可以看到图片中左侧有一个蓝色的圆形，'
            '右边有一个红色的三角形，下方有两个绿色的正方形。\n'
            '三角形1个，圆形2个，正方形2个，合计5个形状。'
        ),
        'ground_truth': {'三角形': 2, '圆形': 1, '正方形': 3},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': 'Case where some shapes are absent (handled correctly)',
        'response': (
            '首先，分析图中的三角形：有 2 个三角形。\n'
            '然后，分析圆形：有 3 个圆形。\n'
            '接着，分析正方形：图片中没有正方形，0个。\n'
            '所以，三角形2个，圆形3个，正方形0个，总共5个形状。\n'
            '答：三角形2个，圆形3个，正方形0个。'
        ),
        'ground_truth': {'三角形': 2, '圆形': 3, '正方形': 0},
        'image_info': {
            'present_shapes': ['三角形', '圆形'],
            'absent_shapes': ['正方形'],
        },
    },
]


# ==========================================
# Part 3: Test the reward function and print detailed results
# ==========================================
def run_reward_tests():
    """
    Run the reward function on all test cases and print a detailed reward breakdown table
    """
    print("=" * 80)
    print("  Multi-Modal Reward Function Tests — Detailed Breakdown")
    print("=" * 80)
    print()
    print("Reward components:")
    print("  Correctness:       0.0 ~ 1.0   whether the answer is fully correct")
    print("  Reasoning quality:  0.0 ~ 0.5   whether there is step-by-step reasoning")
    print("  Format compliance:  0.0 ~ 0.2   whether the output format is compliant")
    print("  Visual grounding:   0.0 ~ 0.3   whether the correct visual features are referenced")
    print("  ----------------------------------------------")
    print("  Total:              0.0 ~ 2.0")
    print()

    # Header
    header = (
        f"{'ID':>4s}  "
        f"{'Correct':>6s}  "
        f"{'Reason':>4s}  "
        f"{'Format':>4s}  "
        f"{'Visual':>4s}  "
        f"{'Total':>5s}  "
        f"{'Description'}"
    )
    separator = "-" * 80
    print(header)
    print(separator)

    all_rewards = []

    for i, tc in enumerate(test_cases):
        rewards = compute_total_reward(
            tc['response'],
            tc['ground_truth'],
            tc['image_info'],
        )
        all_rewards.append(rewards)

        print(
            f"  {i+1:>2d}  "
            f"{rewards['correctness']:>6.2f}  "
            f"{rewards['reasoning']:>4.2f}  "
            f"{rewards['format']:>4.2f}  "
            f"{rewards['visual_grounding']:>4.2f}  "
            f"{rewards['total']:>5.2f}  "
            f"{tc['name']}"
        )

    print(separator)
    print()

    # Summary statistics
    totals = [r['total'] for r in all_rewards]
    print(f"Reward statistics:")
    print(f"  Max total: {max(totals):.2f}")
    print(f"  Min total: {min(totals):.2f}")
    print(f"  Avg total: {np.mean(totals):.2f}")
    print()

    # Print a detailed analysis for each test case
    print("=" * 80)
    print("  Detailed Analysis Per Case")
    print("=" * 80)
    for i, (tc, rewards) in enumerate(zip(test_cases, all_rewards)):
        gt = tc['ground_truth']
        print(f"\n--- Test case {i+1}: {tc['name']} ---")
        print(f"  Annotation: triangles={gt['三角形']}, circles={gt['圆形']}, squares={gt['正方形']}")
        print(f"  Response: {tc['response'][:80]}...")
        print(f"  Reward breakdown:")
        print(f"    Correctness:       {rewards['correctness']:.2f}  {'[max]' if rewards['correctness'] == 1.0 else '[no score]'}")
        print(f"    Reasoning quality: {rewards['reasoning']:.2f}  {'[max]' if rewards['reasoning'] == 0.5 else ''}")
        print(f"    Format compliance: {rewards['format']:.2f}  {'[max]' if rewards['format'] == 0.2 else ''}")
        print(f"    Visual grounding:  {rewards['visual_grounding']:.2f}  {'[max]' if rewards['visual_grounding'] == 0.3 else ''}")
        print(f"    Weighted total:    {rewards['total']:.2f}")

    return all_rewards


# ==========================================
# Part 4: Visualize the reward weight distribution
# ==========================================
def plot_reward_weights(all_rewards):
    """
    Plot a pie chart of reward component weights and a stacked bar chart per test case

    Left: theoretical max score share (pie chart)
    Right: stacked bar chart of reward components per test case
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- Left: theoretical max score share pie chart ----------
    ax1 = axes[0]

    # Theoretical max score for each component
    max_scores = [1.0, 0.5, 0.2, 0.3]  # correctness, reasoning, format, visual
    labels = ['Correctness\n(max 1.0)', 'Reasoning Quality\n(max 0.5)', 'Format\n(max 0.2)', 'Visual Grounding\n(max 0.3)']
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    explode = (0.05, 0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(
        max_scores,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90,
        textprops={'fontsize': 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax1.set_title('Reward Component Theoretical Max Score Share', fontsize=14, fontweight='bold')

    # ---------- Right: stacked bar chart ----------
    ax2 = axes[1]

    n_cases = len(all_rewards)
    x = np.arange(n_cases)

    correctness_vals = [r['correctness'] for r in all_rewards]
    reasoning_vals = [r['reasoning'] for r in all_rewards]
    format_vals = [r['format'] for r in all_rewards]
    visual_vals = [r['visual_grounding'] for r in all_rewards]

    bar_width = 0.6
    ax2.bar(x, correctness_vals, bar_width, label='Correctness', color='#e74c3c')
    ax2.bar(x, reasoning_vals, bar_width, bottom=correctness_vals,
            label='Reasoning Quality', color='#3498db')
    bottom2 = [c + r for c, r in zip(correctness_vals, reasoning_vals)]
    ax2.bar(x, format_vals, bar_width, bottom=bottom2,
            label='Format Compliance', color='#f39c12')
    bottom3 = [b + f for b, f in zip(bottom2, format_vals)]
    ax2.bar(x, visual_vals, bar_width, bottom=bottom3,
            label='Visual Grounding', color='#2ecc71')

    # Annotate the total score above each bar
    for i, r in enumerate(all_rewards):
        ax2.text(i, r['total'] + 0.05, f"{r['total']:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Test Case Number', fontsize=12)
    ax2.set_ylabel('Reward Score', fontsize=12)
    ax2.set_title('Reward Component Breakdown per Test Case', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'#{i+1}' for i in range(n_cases)])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 2.3)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output/multi_modal_reward_breakdown.png', dpi=150, bbox_inches='tight')
    print("\n  Reward breakdown chart saved as output/multi_modal_reward_breakdown.png")
    plt.show()


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    # Step 1: run the reward function tests
    all_rewards = run_reward_tests()

    # Step 2: visualize the reward weights
    print("\n" + "=" * 80)
    print("  Generating Visualization Charts...")
    print("=" * 80)
    plot_reward_weights(all_rewards)

    # Final summary
    print("\n" + "=" * 80)
    print("  Multi-Modal Reward Function Design Summary")
    print("=" * 80)
    print("""
  Reward function design highlights:

    1. Correctness (highest weight, max score 1.0)
       - Only fully correct answers get the max score; any error scores 0
       - This "all-or-nothing" design pushes the model to strive for perfection
       - In GRPO training, correct responses within a group receive positive advantage

    2. Reasoning quality (max score 0.5)
       - Encourages the model to show its thinking process rather than jumping to the answer
       - Scored by detecting step markers and reasoning keywords
       - Helps cultivate Chain-of-Thought reasoning ability

    3. Format compliance (max score 0.2)
       - Ensures the output can be parsed automatically
       - Requires mentioning all three shapes and a summary statement
       - Reduces the complexity of downstream evaluation

    4. Visual grounding (max score 0.3)
       - Encourages the model to genuinely "understand" the image content
       - Checks whether present shapes are correctly referenced and absent shapes correctly noted
       - Especially important for VLM training, to prevent the model from "guessing" answers

  Differences from text-only RL:
    - Text-only RL rewards focus only on text quality
    - VLM RL rewards must additionally account for visual grounding ability
    - The model must not only answer correctly, but do so by "looking at the image"

  Extensions for real-world use:
    - Could add an OCR reward (checking whether the model correctly reads text in the image)
    - Could add a spatial relationship reward (checking understanding of object positional relationships)
    - Could use LLM-as-Judge instead of rule-based rewards for finer-grained scoring
    """)
