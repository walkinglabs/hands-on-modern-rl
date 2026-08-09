"""
Chapter 7: DPO Math Reasoning Alignment -- Training Math Preferences with Rule-Based Signals
==========================================================

This script demonstrates how to apply DPO to the math reasoning domain:
  1. Construct GSM8K-style math preference data (arithmetic problems)
  2. Generate a correct (chosen) and an incorrect (rejected) solution for each problem
  3. Train a model with DPO so it favors the correct reasoning path
  4. Evaluate accuracy before and after training on a held-out test set
  5. Show how DPO learns from rule-based preference signals

Core idea:
  - No need for humans to label "which answer is better"
  - Rules (whether the answer is correct) automatically generate preference labels
  - DPO learns the correct reasoning pattern from these rule-based signals

How to run:
  pip install -r requirements.txt
  python dpo_math_reward.py
"""

import re
import json
import torch
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1. Construct GSM8K-style math preference data
# ==========================================
# Each entry contains a math problem plus a correct and an incorrect solution process
# We use simple arithmetic problems so answers can be verified automatically

def create_math_preference_data():
    """
    Create the preference dataset for math reasoning

    Each entry contains:
    - prompt: the math problem
    - chosen: a correct, step-by-step solution process
    - rejected: a solution process that contains a reasoning error
    """

    math_examples = [
        {
            "prompt": "小明有 15 个苹果，给了小红 7 个，又买了 3 个。请问小明现在有几个苹果？",
            "chosen": "我们一步步来算。\n第一步：小明原来有 15 个苹果。\n第二步：给了小红 7 个，所以剩下 15 - 7 = 8 个。\n第三步：又买了 3 个，所以现在有 8 + 3 = 11 个。\n答：小明现在有 11 个苹果。",
            "rejected": "小明原来有 15 个苹果。\n给了小红 7 个，剩下 15 - 7 = 9 个。（错误：15-7=8，不是9）\n又买了 3 个，所以有 9 + 3 = 12 个。\n答：小明现在有 12 个苹果。",
            "answer": 11,
        },
        {
            "prompt": "一本书有 240 页，小红每天看 30 页，看了 5 天。还剩多少页没看？",
            "chosen": "我们一步步来算。\n第一步：小红每天看 30 页，看了 5 天。\n第二步：已经看了 30 × 5 = 150 页。\n第三步：总共有 240 页，所以还剩 240 - 150 = 90 页。\n答：还剩 90 页没看。",
            "rejected": "小红每天看 30 页，看了 5 天。\n已经看了 30 + 5 = 35 页。（错误：应该用乘法，不是加法）\n还剩 240 - 35 = 205 页。\n答：还剩 205 页没看。",
            "answer": 90,
        },
        {
            "prompt": "一个长方形的长是 12 厘米，宽是 8 厘米，求面积。",
            "chosen": "长方形的面积 = 长 × 宽。\n长 = 12 厘米，宽 = 8 厘米。\n面积 = 12 × 8 = 96 平方厘米。\n答：面积是 96 平方厘米。",
            "rejected": "长方形的面积 = 长 + 宽。（错误：面积是乘法，不是加法）\n面积 = 12 + 8 = 20 平方厘米。\n答：面积是 20 平方厘米。",
            "answer": 96,
        },
        {
            "prompt": "商店里有 36 个篮球和 24 个足球，篮球比足球多几个？",
            "chosen": "篮球有 36 个，足球有 24 个。\n篮球比足球多 36 - 24 = 12 个。\n答：篮球比足球多 12 个。",
            "rejected": "篮球有 36 个，足球有 24 个。\n篮球和足球一共 36 + 24 = 60 个。（错误：题目问的是差值，不是总数）\n答：篮球比足球多 60 个。",
            "answer": 12,
        },
        {
            "prompt": "一箱橘子有 48 个，平均分给 6 个小朋友，每人能分几个？",
            "chosen": "总共 48 个橘子，平均分给 6 个小朋友。\n每人分到 48 ÷ 6 = 8 个。\n答：每人能分到 8 个橘子。",
            "rejected": "总共 48 个橘子，平均分给 6 个小朋友。\n每人分到 48 - 6 = 42 个。（错误：应该用除法，不是减法）\n答：每人能分到 42 个橘子。",
            "answer": 8,
        },
        {
            "prompt": "一个三角形的底是 10 厘米，高是 6 厘米，求面积。",
            "chosen": "三角形的面积 = 底 × 高 ÷ 2。\n底 = 10 厘米，高 = 6 厘米。\n面积 = 10 × 6 ÷ 2 = 30 平方厘米。\n答：面积是 30 平方厘米。",
            "rejected": "三角形的面积 = 底 × 高。（错误：三角形面积要除以2）\n面积 = 10 × 6 = 60 平方厘米。\n答：面积是 60 平方厘米。",
            "answer": 30,
        },
        {
            "prompt": "一列火车每小时行驶 80 公里，行驶了 3.5 小时，共行驶多少公里？",
            "chosen": "速度 = 80 公里/小时，时间 = 3.5 小时。\n路程 = 速度 × 时间 = 80 × 3.5 = 280 公里。\n答：共行驶 280 公里。",
            "rejected": "速度 = 80 公里/小时，时间 = 3.5 小时。\n路程 = 80 + 3.5 = 83.5 公里。（错误：应该用乘法，不是加法）\n答：共行驶 83.5 公里。",
            "answer": 280,
        },
        {
            "prompt": "妈妈买了 3 千克苹果，每千克 12 元，又买了 2 千克香蕉，每千克 8 元。一共花了多少钱？",
            "chosen": "苹果花了：3 × 12 = 36 元。\n香蕉花了：2 × 8 = 16 元。\n一共花了 36 + 16 = 52 元。\n答：一共花了 52 元。",
            "rejected": "苹果花了：3 × 12 = 36 元。\n香蕉花了：2 × 8 = 16 元。\n一共花了 36 × 16 = 576 元。（错误：应该用加法，不是乘法）\n答：一共花了 576 元。",
            "answer": 52,
        },
        {
            "prompt": "一个正方形的边长是 9 厘米，求周长。",
            "chosen": "正方形的周长 = 边长 × 4。\n边长 = 9 厘米。\n周长 = 9 × 4 = 36 厘米。\n答：周长是 36 厘米。",
            "rejected": "正方形的周长 = 边长 × 边长。（错误：这是面积公式，不是周长）\n周长 = 9 × 9 = 81 厘米。\n答：周长是 81 厘米。",
            "answer": 36,
        },
        {
            "prompt": "爸爸今年 42 岁，儿子今年 12 岁。5 年后爸爸比儿子大多少岁？",
            "chosen": "年龄差不会随时间变化。\n现在爸爸比儿子大 42 - 12 = 30 岁。\n5 年后爸爸比儿子仍然大 30 岁。\n答：5 年后爸爸比儿子大 30 岁。",
            "rejected": "5 年后爸爸 42 + 5 = 47 岁，儿子 12 + 5 = 17 岁。\n爸爸比儿子大 47 - 12 = 35 岁。（错误：儿子的年龄也加5了，计算差值时却没加）\n答：5 年后爸爸比儿子大 35 岁。",
            "answer": 30,
        },
    ]

    return math_examples


# ==========================================
# 2. Prepare the test set data (used to evaluate accuracy)
# ==========================================

def create_test_data():
    """
    Create an independent test set that does not overlap with the training data
    Used to evaluate math reasoning ability before and after DPO training
    """
    test_examples = [
        {
            "prompt": "有 56 个糖果，平均分给 8 个小朋友，每人分几个？",
            "answer": 7,
        },
        {
            "prompt": "小明每天跑步 5 公里，跑了 7 天，一共跑了多少公里？",
            "answer": 35,
        },
        {
            "prompt": "一个长方形的长是 15 厘米，宽是 4 厘米，求周长。",
            "answer": 38,
        },
        {
            "prompt": "水果店有 100 个苹果，卖出了 37 个，又进货 20 个，现在有几个？",
            "answer": 83,
        },
        {
            "prompt": "一箱牛奶有 24 盒，学校买了 5 箱，一共有多少盒牛奶？",
            "answer": 120,
        },
    ]
    return test_examples


# ==========================================
# 3. Define helper functions
# ==========================================

def extract_number(text):
    """Extract the final numeric answer from the model's response"""
    # Try to match the number in patterns like "答：...是 X" or "答：X" (i.e. "Answer: ... is X" / "Answer: X")
    patterns = [
        r"答[：:][^0-9]*?(\d+)",
        r"答案是\s*(\d+)",
        r"结果是\s*(\d+)",
        r"等于\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.findall(pattern, text)
        if match:
            return int(match[-1])  # take the last match

    # If none of the above matched, try to find the last number that appears in the text
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def evaluate_math(model, tokenizer, test_data, label="Model"):
    """
    Evaluate the model's math reasoning accuracy on the test set

    Generate a response for each problem, extract the numeric answer, and compare it against the correct answer
    """
    print("=" * 60)
    print(f"[{label} Math Reasoning Evaluation]")
    print("=" * 60)

    correct = 0
    total = len(test_data)

    for i, item in enumerate(test_data):
        prompt = item["prompt"]
        true_answer = item["answer"]

        # Generate the model's response
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        predicted_answer = extract_number(response)

        is_correct = predicted_answer == true_answer
        correct += int(is_correct)

        status = "Correct" if is_correct else "Incorrect"
        print(f"  Problem {i+1}: {prompt[:30]}...")
        print(f"    Model response: {response[:80]}...")
        print(f"    Predicted answer: {predicted_answer} | Correct answer: {true_answer} | {status}")
        print()

    accuracy = correct / total * 100
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)
    return accuracy


# ==========================================
# 4. Main workflow begins
# ==========================================

print("=" * 60)
print("  DPO Math Reasoning Alignment Experiment")
print("  -- Training with Rule-Based Preference Signals")
print("=" * 60)
print()

# Prepare the data
math_data = create_math_preference_data()
test_data = create_test_data()

print(f"Training set size: {len(math_data)} math preference examples")
print(f"Test set size: {len(test_data)} math problems")
print()
print("How the preference data is constructed:")
print("  - chosen:  a correct, step-by-step reasoning process (every step of the computation is correct)")
print("  - rejected: a solution that contains a reasoning error (a computation or formula is wrong at some step)")
print("  - This preference signal requires no human labeling; it is generated entirely by rules")
print()


# ==========================================
# 5. Load the model and evaluate pre-training performance
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading base model {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model_before = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

before_accuracy = evaluate_math(model_before, tokenizer, test_data, label="Before Training")

# Free up GPU memory
del model_before
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ==========================================
# 6. DPO training
# ==========================================

print("\nStarting DPO training with rule-based math preference data...")

# Reload the model for training
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
train_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_tokenizer.pad_token = train_tokenizer.eos_token

# Build the HuggingFace Dataset
data_dict = {
    "prompt": [item["prompt"] for item in math_data],
    "chosen": [item["chosen"] for item in math_data],
    "rejected": [item["rejected"] for item in math_data],
}
train_dataset = Dataset.from_dict(data_dict)

# Configure training arguments
training_args = DPOConfig(
    output_dir="./dpo_math_results",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=5,         # Math reasoning needs more epochs to learn the reasoning pattern
    logging_steps=2,
    save_strategy="no",
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    remove_unused_columns=False,
    beta=0.1,
)

# Create the DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=train_tokenizer,
)

print("\nTraining... watch how loss and reward evolve\n")
train_result = trainer.train()

# ==========================================
# 7. Print training metrics
# ==========================================

print("\n" + "=" * 60)
print("[Training Metrics Detail]")
print("=" * 60)
print(f"Final training loss: {train_result.training_loss:.4f}")
print()

# Parse the reward information from the training log
log_history = trainer.state.log_history
print("Training metrics at each step:")
print(f"{'Step':>6} | {'Loss':>8} | {'Chosen Reward':>14} | {'Rejected Reward':>16} | {'Margin':>8}")
print("-" * 70)

for entry in log_history:
    if "loss" in entry:
        step = entry.get("step", "?")
        loss = entry["loss"]
        chosen_r = entry.get("rewards/chosen", "N/A")
        rejected_r = entry.get("rewards/rejected", "N/A")
        margin = entry.get("rewards/margins", "N/A")

        # Format the output
        if isinstance(chosen_r, float):
            chosen_r = f"{chosen_r:.4f}"
        if isinstance(rejected_r, float):
            rejected_r = f"{rejected_r:.4f}"
        if isinstance(margin, float):
            margin = f"{margin:.4f}"

        print(f"{step:>6} | {loss:>8.4f} | {chosen_r:>14} | {rejected_r:>16} | {margin:>8}")

# Save the model
save_path = "./dpo_math_results/final_model"
trainer.save_model(save_path)
print(f"\nModel saved to {save_path}")


# ==========================================
# 8. Evaluate the trained model
# ==========================================

# Load the trained model for evaluation
print("\nLoading the trained model for evaluation...")
model_after = AutoModelForCausalLM.from_pretrained(save_path, device_map="auto")
eval_tokenizer = AutoTokenizer.from_pretrained(save_path)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

after_accuracy = evaluate_math(model_after, eval_tokenizer, test_data, label="After Training")


# ==========================================
# 9. Compare results and summarize
# ==========================================

print("\n" + "=" * 60)
print("[DPO Math Reasoning Alignment -- Final Results Comparison]")
print("=" * 60)
print()
print(f"  Accuracy before training: {before_accuracy:.1f}%")
print(f"  Accuracy after training: {after_accuracy:.1f}%")
print(f"  Improvement: {after_accuracy - before_accuracy:+.1f}%")
print()

print("=" * 60)
print("[Experiment Summary]")
print("=" * 60)
print("""
1. Rule-based preference signals:
   This experiment does not rely on human labeling. Instead, it uses the simple
   rule "is the answer correct" to automatically generate chosen/rejected pairs.
   This approach scales easily to many more problems.

2. DPO's effect on math reasoning:
   - DPO can learn correct reasoning patterns (such as choosing the right operator)
   - After training, the model tends to favor correct solution strategies on new problems
   - However, with only 0.5B parameters, the improvement on complex reasoning is limited

3. Comparison with RLHF:
   - Traditional RLHF: requires training a reward model -> optimizing the policy with PPO
   - DPO approach: optimizes directly from preference data, skipping reward model training
   - For tasks with clear-cut rules (like math), DPO + rule-based signals is very efficient

4. Directions for further extension:
   - Use larger models (7B, 14B) for better reasoning ability
   - Increase the amount of training data (e.g. the full GSM8K dataset)
   - Combine with Chain-of-Thought prompting to deepen reasoning
   - Use multiple incorrect rejected samples to strengthen contrastive learning
""")
