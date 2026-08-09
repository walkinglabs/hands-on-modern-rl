"""
Chapter 11: VLM GRPO Training Simulation Demo
==========================================================

This script simulates the GRPO training process on a VLM (Vision-Language Model):
  1. Build training data for the geometry shape counting task
  2. Simulate the GRPO training loop (group sampling → reward computation → normalization → policy update)
  3. Show the key differences between VLM GRPO and text-only GRPO
  4. Track training metrics: accuracy, average reward, response quality
  5. Compare before and after training

Important note:
  This script is a **simplified demo**, using simulated data instead of a real VLM model.
  A full VLM GRPO training run would require:
    - GPU memory >= 40GB (e.g. an A100)
    - A VLM model from the transformers library (e.g. Qwen2-VL, LLaVA)
    - An image encoder + vision token processing
    - A distributed training framework (e.g. DeepSpeed)

  The purpose of this script is to help you understand the VLM GRPO training
  workflow and its key concepts.

How to run:
  python vlm_grpo_train.py
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)

# Set a CJK-capable font to ensure chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Differences between VLM GRPO and text-only GRPO
# ==========================================
# What makes VLM GRPO unique:
#   1. The input includes images → requires an image encoder (Vision Encoder)
#   2. Images are encoded as vision tokens → concatenated with text tokens before feeding into the LLM
#   3. The reward function needs to account for visual grounding ability
#   4. Training must process both image and text data simultaneously

def print_vlm_grpo_overview():
    """
    Print an architecture comparison between VLM GRPO and text-only GRPO
    """
    print("=" * 70)
    print("  VLM GRPO vs Text-only GRPO — Architecture Comparison")
    print("=" * 70)
    print()
    print("  Text-only GRPO pipeline:")
    print("  ┌──────────┐    ┌──────────┐    ┌──────────┐")
    print("  │ Text     │ → │ LLM      │ → │ Text     │")
    print("  │ prompt   │    │ sampling │    │ reward   │")
    print("  │ (prompt) │    │ (group)  │    │ (score)  │")
    print("  └──────────┘    └──────────┘    └──────────┘")
    print()
    print("  VLM GRPO pipeline:")
    print("  ┌──────┐ ┌──────────┐    ┌──────────────┐    ┌──────────────┐")
    print("  │Image │→│ Vision    │→│ VLM sampling  │→│ Multi-modal  │")
    print("  │(img) │ │ encoding  │    │(vision+text) │    │ reward       │")
    print("  │      │ │(Encoder) │    │              │    │(correct+vis) │")
    print("  └──────┘ └──────────┘    └──────────────┘    └──────────────┘")
    print("  ┌──────────┐     ↑")
    print("  │ Text     │ ────┘")
    print("  │ prompt   │")
    print("  │ (prompt) │")
    print("  └──────────┘")
    print()
    print("  Key differences:")
    print("    1. Input: text-only prompt → (image, text prompt) pair")
    print("    2. Model: LLM → VLM (LLM + Vision Encoder + projection layer)")
    print("    3. Tokens: text tokens only → text tokens + vision tokens")
    print("    4. Reward: text quality only → text quality + visual grounding ability")
    print("    5. Memory: smaller → significantly higher (image encoding + longer sequences)")
    print()


# ==========================================
# Part 2: Simulated data and response generation
# ==========================================
# Since full VLM inference would require substantial GPU resources,
# preset templates are used here to simulate model responses of varying quality

# Standard prompt for the geometry shape counting task
STANDARD_PROMPT = "请数一下图片中有多少个三角形、圆形和正方形"

# Templates simulating responses of varying quality levels
# Each template is a function that generates a response from the ground_truth
def generate_correct_response(gt):
    """Generate a fully correct response (including reasoning)"""
    shapes_desc = []
    for shape, count in gt.items():
        if count > 0:
            shapes_desc.append(f"{shape}{count}个")
        else:
            shapes_desc.append(f"{shape}0个（没有{shape}）")
    return (
        f"让我仔细观察图片中的形状。\n"
        f"首先，分析三角形：我数到了{gt['三角形']}个三角形。\n"
        f"然后，分析圆形：图片中有{gt['圆形']}个圆形。\n"
        f"接着，分析正方形：有{gt['正方形']}个正方形。\n"
        f"所以，{shapes_desc[0]}，{shapes_desc[1]}，{shapes_desc[2]}。\n"
        f"答：三角形{gt['三角形']}个，圆形{gt['圆形']}个，正方形{gt['正方形']}个。"
    )


def generate_short_correct_response(gt):
    """Generate a correct but short response"""
    return f"三角形{gt['三角形']}个，圆形{gt['圆形']}个，正方形{gt['正方形']}个。"


def generate_wrong_response(gt):
    """Generate a response with an error (randomly modify one number)"""
    wrong_gt = dict(gt)
    shape_to_modify = random.choice(list(gt.keys()))
    # Randomly increase or decrease by 1~2
    delta = random.choice([-2, -1, 1, 2])
    wrong_gt[shape_to_modify] = max(0, wrong_gt[shape_to_modify] + delta)
    return (
        f"我来看一下图片。\n"
        f"三角形{wrong_gt['三角形']}个，圆形{wrong_gt['圆形']}个，正方形{wrong_gt['正方形']}个。\n"
        f"总共{sum(wrong_gt.values())}个形状。"
    )


def generate_partially_correct_response(gt):
    """Generate a partially correct response (only some shape counts are right)"""
    shapes = list(gt.keys())
    # Keep one shape's count correct, randomly modify the others
    correct_shape = random.choice(shapes)
    wrong_gt = {}
    for s in shapes:
        if s == correct_shape:
            wrong_gt[s] = gt[s]
        else:
            wrong_gt[s] = max(0, gt[s] + random.choice([-1, 1]))
    return f"分析图片后，三角形{wrong_gt['三角形']}个，圆形{wrong_gt['圆形']}个，正方形{wrong_gt['正方形']}个。"


def generate_low_quality_response(gt):
    """Generate a low-quality response (no formatting, no reasoning)"""
    shapes = list(gt.keys())
    random.shuffle(shapes)
    nums = [max(0, gt[s] + random.choice([-2, -1, 0, 1, 2])) for s in shapes]
    return f"大概有{shapes[0]}{nums[0]}个{shapes[1]}{nums[1]}个{shapes[2]}{nums[2]}个吧"


# List of response generators, ordered from highest to lowest quality
RESPONSE_GENERATORS = [
    generate_correct_response,           # quality: high
    generate_short_correct_response,     # quality: medium-high (correct but no reasoning)
    generate_wrong_response,             # quality: medium-low (has reasoning but wrong)
    generate_partially_correct_response, # quality: low (partially correct)
    generate_low_quality_response,       # quality: very low
]


# ==========================================
# Part 3: Reward function (simplified version, reuses the logic from multi_modal_reward.py)
# ==========================================
def extract_numbers(response, shape_names):
    """Extract the number corresponding to each shape from the response"""
    import re
    extracted = {}
    for shape_name in shape_names:
        patterns = [
            rf'{shape_name}[^0-9]*?(\d+)',
            rf'(\d+)\s*个\s*{shape_name}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                extracted[shape_name] = int(matches[-1])
                break
    return extracted


def simple_reward(response, ground_truth):
    """
    Simplified reward function: computes a composite score for the response

    Components:
      - Correctness (0.0 or 1.0): the answer is fully correct
      - Reasoning bonus (0.0~0.3): includes reasoning steps
      - Format bonus (0.0~0.2): well-formatted output

    Total score range: 0.0 ~ 1.5
    """
    import re

    score = 0.0

    # Correctness
    extracted = extract_numbers(response, ['三角形', '圆形', '正方形'])
    all_correct = True
    for shape, expected in ground_truth.items():
        if extracted.get(shape, -1) != expected:
            all_correct = False
            break
    if all_correct and len(extracted) == len(ground_truth):
        score += 1.0

    # Reasoning bonus
    step_keywords = ['首先', '然后', '接着', '最后', '分析', '观察', '因为', '所以']
    step_count = sum(1 for kw in step_keywords if kw in response)
    score += min(step_count * 0.06, 0.3)

    # Format bonus
    if all(s in response for s in ['三角形', '圆形', '正方形']):
        score += 0.1
    if re.search(r'答[：:]', response):
        score += 0.1

    return score


# ==========================================
# Part 4: GRPO training data preparation
# ==========================================
def create_training_samples(num_samples=20, seed=42):
    """
    Create simulated training samples

    Each sample contains:
      - Image identifier (simulated)
      - Text prompt
      - ground_truth (the correct count for each shape)

    Args:
        num_samples: number of samples
        seed: random seed
    Returns:
        list[dict]: list of training samples
    """
    random.seed(seed)
    samples = []

    for i in range(num_samples):
        # Randomly generate the count of each shape (0~5)
        gt = {
            '三角形': random.randint(0, 5),
            '圆形': random.randint(0, 5),
            '正方形': random.randint(0, 5),
        }

        samples.append({
            'sample_id': i,
            'prompt': STANDARD_PROMPT,
            'ground_truth': gt,
        })

    return samples


# ==========================================
# Part 5: Simulated GRPO training loop
# ==========================================
def simulate_grpo_training(samples, group_size=4, num_epochs=5,
                           initial_quality=0.3, seed=42):
    """
    Simulate the VLM GRPO training process

    GRPO training loop (per epoch):
      1. For each training sample, generate group_size responses
      2. Score each response with the reward function
      3. Compute the within-group normalized advantage
      4. Update the policy using the advantage values (simulated)
      5. As training proceeds, the model's response quality gradually improves

    Args:
        samples: list of training samples
        group_size: number of samples per question (the GRPO paper recommends 4~16)
        num_epochs: number of training epochs
        initial_quality: initial response quality (0~1, higher means a stronger initial model)
        seed: random seed
    Returns:
        dict: training history
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("  VLM GRPO Training Simulation")
    print("=" * 70)
    print()
    print(f"Training configuration:")
    print(f"  Number of training samples: {len(samples)}")
    print(f"  Group size (group_size): {group_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Initial response quality: {initial_quality:.1f}")
    print()
    print("Note: this is a simulated training run. Full VLM GRPO requires:")
    print("  - A VLM model (e.g. Qwen2-VL, LLaVA)")
    print("  - An image encoder and vision token processing")
    print("  - GPU memory >= 40GB")
    print("  - DeepSpeed or FSDP for distributed training")
    print()

    # Track training history
    history = {
        'epoch': [],
        'accuracy': [],         # proportion fully correct
        'avg_reward': [],       # average reward
        'best_reward': [],      # average reward of the best response
        'avg_advantage_std': [],# average advantage std (measures discriminability)
    }

    # Simulate the training process: model quality gradually improves as epochs increase
    # quality_factor grows linearly from initial_quality toward ~1.0
    for epoch in range(num_epochs):
        quality_factor = initial_quality + (1.0 - initial_quality) * (epoch / max(num_epochs - 1, 1))

        epoch_correct = 0
        epoch_rewards = []
        epoch_best_rewards = []
        epoch_adv_stds = []

        for sample in samples:
            gt = sample['ground_truth']

            # Simulate generating group_size responses
            group_rewards = []
            for g in range(group_size):
                # Decide whether to generate a correct response based on the quality factor
                # The higher the quality factor, the higher the probability of a correct response
                if random.random() < quality_factor:
                    # Generate a correct or near-correct response
                    if random.random() < 0.7:
                        response = generate_correct_response(gt)
                    else:
                        response = generate_short_correct_response(gt)
                else:
                    # Generate a flawed response
                    choice = random.random()
                    if choice < 0.4:
                        response = generate_wrong_response(gt)
                    elif choice < 0.7:
                        response = generate_partially_correct_response(gt)
                    else:
                        response = generate_low_quality_response(gt)

                # Compute the reward
                reward = simple_reward(response, gt)
                group_rewards.append(reward)

            # GRPO core: within-group normalization
            rewards_arr = np.array(group_rewards)
            mean_r = rewards_arr.mean()
            std_r = rewards_arr.std() + 1e-8
            advantages = (rewards_arr - mean_r) / std_r

            # Record statistics
            epoch_rewards.extend(group_rewards)
            epoch_best_rewards.append(max(group_rewards))
            epoch_adv_stds.append(std_r)

            # Check whether the best response is correct
            best_idx = np.argmax(rewards_arr)
            if rewards_arr[best_idx] >= 1.0:
                epoch_correct += 1

        # Compute this epoch's metrics
        accuracy = epoch_correct / len(samples)
        avg_reward = np.mean(epoch_rewards)
        best_reward = np.mean(epoch_best_rewards)
        avg_adv_std = np.mean(epoch_adv_stds)

        history['epoch'].append(epoch + 1)
        history['accuracy'].append(accuracy)
        history['avg_reward'].append(avg_reward)
        history['best_reward'].append(best_reward)
        history['avg_advantage_std'].append(avg_adv_std)

        # Print this epoch's training log
        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f"Accuracy: {accuracy:.3f} | "
              f"Avg reward: {avg_reward:.3f} | "
              f"Best response reward: {best_reward:.3f} | "
              f"Advantage std: {avg_adv_std:.3f}")

    return history


# ==========================================
# Part 6: Before/after training comparison
# ==========================================
def print_before_after_comparison(samples, history, seed=42):
    """
    Print a comparison of results before and after training

    Shows:
      1. Differences in performance on the same samples before and after training
      2. Response comparisons for a few specific samples
    """
    random.seed(seed)

    print("\n" + "=" * 70)
    print("  Before/After Training Comparison")
    print("=" * 70)

    # Select 5 samples to display the comparison
    display_samples = samples[:5]

    print("\n--- Response examples before training (low-quality model) ---")
    for i, sample in enumerate(display_samples):
        gt = sample['ground_truth']
        # Before training: low-quality response
        response = generate_wrong_response(gt)
        reward = simple_reward(response, gt)
        extracted = extract_numbers(response, ['三角形', '圆形', '正方形'])
        is_correct = all(extracted.get(s, -1) == gt[s] for s in gt)
        print(f"\n  Sample {i+1} (GT: triangles={gt['三角形']}, circles={gt['圆形']}, "
              f"squares={gt['正方形']})")
        print(f"    Response: {response[:60]}...")
        print(f"    Reward: {reward:.2f} | {'correct' if is_correct else 'incorrect'}")

    print("\n\n--- Response examples after training (high-quality model) ---")
    for i, sample in enumerate(display_samples):
        gt = sample['ground_truth']
        # After training: high-quality response
        response = generate_correct_response(gt)
        reward = simple_reward(response, gt)
        extracted = extract_numbers(response, ['三角形', '圆形', '正方形'])
        is_correct = all(extracted.get(s, -1) == gt[s] for s in gt)
        print(f"\n  Sample {i+1} (GT: triangles={gt['三角形']}, circles={gt['圆形']}, "
              f"squares={gt['正方形']})")
        print(f"    Response: {response[:60]}...")
        print(f"    Reward: {reward:.2f} | {'correct' if is_correct else 'incorrect'}")

    # Summary
    print("\n" + "-" * 70)
    print("  Summary comparison:")
    print(f"  {'Metric':>12s}  {'Before':>10s}  {'After':>10s}  {'Change':>10s}")
    print(f"  {'----':>12s}  {'------':>10s}  {'------':>10s}  {'----':>10s}")

    before_acc = history['accuracy'][0]
    after_acc = history['accuracy'][-1]
    before_reward = history['avg_reward'][0]
    after_reward = history['avg_reward'][-1]
    before_best = history['best_reward'][0]
    after_best = history['best_reward'][-1]

    print(f"  {'Accuracy':>12s}  {before_acc:>10.3f}  {after_acc:>10.3f}  {after_acc - before_acc:>+10.3f}")
    print(f"  {'Avg reward':>12s}  {before_reward:>10.3f}  {after_reward:>10.3f}  {after_reward - before_reward:>+10.3f}")
    print(f"  {'Best reward':>12s}  {before_best:>10.3f}  {after_best:>10.3f}  {after_best - before_best:>+10.3f}")


# ==========================================
# Part 7: Plot training curves
# ==========================================
def plot_training_curves(history):
    """
    Plot the VLM GRPO training curves

    Contains 4 subplots:
      1. Accuracy over training epochs
      2. Average reward over training epochs
      3. Best response reward over time
      4. Advantage standard deviation over time (measures within-group discriminability)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VLM GRPO Training Curves (Simulated)", fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # Subplot 1: accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['accuracy'], 'o-', color='#2196F3', linewidth=2,
             markersize=8, label='Accuracy')
    ax1.fill_between(epochs, 0, history['accuracy'], alpha=0.1, color='#2196F3')
    ax1.set_title('Accuracy', fontsize=13)
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: average reward
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['avg_reward'], 's-', color='#FF9800', linewidth=2,
             markersize=8, label='Average Reward')
    ax2.fill_between(epochs, 0, history['avg_reward'], alpha=0.1, color='#FF9800')
    ax2.set_title('Average Reward', fontsize=13)
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: best response reward
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['best_reward'], 'D-', color='#4CAF50', linewidth=2,
             markersize=8, label='Best Response Reward')
    ax3.set_title('Average Reward of the Best Response per Group', fontsize=13)
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('Reward Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: advantage standard deviation
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['avg_advantage_std'], '^-', color='#9C27B0', linewidth=2,
             markersize=8, label='Average Advantage Std')
    ax4.set_title('Within-Group Advantage Std (Discriminability)', fontsize=13)
    ax4.set_xlabel('Training Epoch')
    ax4.set_ylabel('Standard Deviation')
    ax4.annotate('Decreasing std = responses within a group converge in quality\n(the model becomes more stable)',
                 xy=(epochs[-1] * 0.5, max(history['avg_advantage_std']) * 0.8),
                 fontsize=9, color='gray', style='italic')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/vlm_grpo_training_curves.png', dpi=150, bbox_inches='tight')
    print("\n  Training curves saved as output/vlm_grpo_training_curves.png")
    plt.show()


# ==========================================
# Part 8: GRPO core computation demo
# ==========================================
def demonstrate_grpo_normalization():
    """
    Demonstrate GRPO's within-group normalization process with a concrete example

    Shows: for a single image, the VLM generates 4 responses of varying quality,
    and how GRPO determines how "good" each response is through within-group comparison
    """
    print("\n" + "=" * 70)
    print("  GRPO Within-Group Normalization Demo (VLM Scenario)")
    print("=" * 70)

    # Simulate the ground truth for one sample
    gt = {'三角形': 3, '圆形': 1, '正方形': 2}

    print(f"\nInput image contains: triangles={gt['三角形']}, circles={gt['圆形']}, squares={gt['正方形']}")
    print(f"Prompt: {STANDARD_PROMPT}")
    print()

    # Simulate 4 responses and their rewards
    responses = [
        ("correct + full reasoning", generate_correct_response(gt)),
        ("correct but short", generate_short_correct_response(gt)),
        ("has reasoning but wrong", generate_wrong_response(gt)),
        ("low quality", generate_low_quality_response(gt)),
    ]

    rewards = []
    print("Generated responses and their rewards:")
    print("-" * 70)
    for i, (desc, resp) in enumerate(responses):
        r = simple_reward(resp, gt)
        rewards.append(r)
        print(f"  Response {i+1} ({desc})")
        print(f"    Content: {resp[:70]}...")
        print(f"    Reward: {r:.3f}")
        print()

    # GRPO normalization
    rewards_arr = np.array(rewards)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std() + 1e-8
    advantages = (rewards_arr - mean_r) / std_r

    print("-" * 70)
    print("GRPO normalization process:")
    print(f"  Group mean: {mean_r:.4f}")
    print(f"  Group std: {std_r:.4f}")
    print()
    print(f"  {'Resp':>4s}  {'Raw Reward':>8s}  {'GRPO Adv':>10s}  {'Meaning'}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  {'----'}")
    for i in range(len(responses)):
        adv = advantages[i]
        if adv > 0.5:
            meaning = "significantly above the group average, strongly encouraged"
        elif adv > 0:
            meaning = "slightly above the group average, moderately encouraged"
        elif adv > -0.5:
            meaning = "slightly below the group average, moderately suppressed"
        else:
            meaning = "significantly below the group average, strongly suppressed"
        print(f"  {i+1:>4d}  {rewards[i]:>8.4f}  {adv:>+10.4f}  {meaning}")

    print()
    print("  Key point: GRPO doesn't need absolute reward values, only the within-group relative ranking!")
    print("  This is why GRPO doesn't need to train a Critic network to estimate a baseline.")


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    # Step 1: print the VLM GRPO overview
    print_vlm_grpo_overview()

    # Step 2: GRPO core computation demo
    demonstrate_grpo_normalization()

    # Step 3: create training data
    print("\n" + "=" * 70)
    print("  Creating Simulated Training Data")
    print("=" * 70)
    samples = create_training_samples(num_samples=20, seed=42)
    print(f"  Created {len(samples)} training samples")

    # Print the ground truth for a few samples
    print("\n  Sample examples:")
    for i, s in enumerate(samples[:5]):
        gt = s['ground_truth']
        print(f"    Sample {i+1}: triangles={gt['三角形']}, circles={gt['圆形']}, squares={gt['正方形']}, "
              f"total={sum(gt.values())}")

    # Step 4: simulate GRPO training
    print()
    history = simulate_grpo_training(
        samples,
        group_size=4,
        num_epochs=5,
        initial_quality=0.3,
        seed=42,
    )

    # Step 5: before/after training comparison
    print_before_after_comparison(samples, history, seed=42)

    # Step 6: plot training curves
    print("\n" + "=" * 70)
    print("  Generating Visualization Charts...")
    print("=" * 70)
    plot_training_curves(history)

    # Final summary
    print("\n" + "=" * 70)
    print("  VLM GRPO Training Simulation Summary")
    print("=" * 70)
    print("""
  This script simulated the VLM GRPO training workflow and demonstrated the
  following key concepts:

  1. Differences between VLM GRPO and text-only GRPO:
     - Input goes from plain text to (image, text) pairs
     - A vision encoder is needed to convert images into vision tokens
     - Vision tokens are concatenated with text tokens before being fed into the LLM
     - The reward function needs to account for visual grounding ability

  2. GRPO core mechanism:
     - Generate multiple responses (a group) for the same image
     - Score with the reward function, normalize within the group to get advantages
     - Responses with positive advantage are encouraged, negative ones are suppressed
     - No need to train an additional Critic network

  3. Key implementation points for full VLM GRPO:
     - Use a VLM model from the transformers library
       e.g.: Qwen2VLForConditionalGeneration
     - Images are encoded via a Vision Transformer
     - Visual features are mapped into the LLM's embedding space through a projection layer
     - Training must handle longer sequences (vision tokens consume extra length)
     - DeepSpeed ZeRO-3 or FSDP is recommended for distributed training

  4. Application scenarios for VLM GRPO:
     - Improving reasoning ability in Visual Question Answering (VQA)
     - Optimizing quality in Image Captioning
     - Multi-modal mathematical reasoning (reasoning combining images and text)
     - Embodied intelligence (joint vision-language-action training for robots)

  5. Key hyperparameters:
     - group_size: number of samples per question (recommended 4~16)
     - clip_ratio: PPO-style clipping range (recommended 0.2)
     - learning_rate: learning rate (recommended 1e-6 ~ 5e-6)
     - KL_coefficient: KL divergence penalty coefficient (recommended 0.01~0.1)
    """)
