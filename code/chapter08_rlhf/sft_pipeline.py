"""
Chapter 8: SFT (Supervised Fine-Tuning) Pipeline
==================================================

This script demonstrates the first stage of the three-stage RLHF pipeline —
supervised fine-tuning (SFT).
Contents:
  1. Generate Self-Instruct-style instruction-response training data
  2. Load the Qwen2.5-0.5B-Instruct base model
  3. Perform supervised fine-tuning with SFTTrainer
  4. Compare model output quality before and after fine-tuning
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Create the output directory
os.makedirs("output", exist_ok=True)

# ==========================================
# 1. Generate Self-Instruct-style training data
# ==========================================

def generate_sft_data():
    """
    Generate instruction-response training data for SFT.
    Covers four task categories: math, writing, coding, and knowledge Q&A.
    Each entry is an (instruction, response) pair.
    """
    sft_pairs = [
        # ---- Math ----
        {
            "instruction": "计算 (3 + 5) × 2 的结果。",
            "response": "我们来逐步计算：\n首先计算括号内的加法：3 + 5 = 8\n然后将结果乘以 2：8 × 2 = 16\n所以 (3 + 5) × 2 = 16。"
        },
        {
            "instruction": "一个长方形的长是 8 厘米，宽是 5 厘米，求面积。",
            "response": "长方形的面积公式为：面积 = 长 × 宽\n代入数值：面积 = 8 × 5 = 40 平方厘米\n所以这个长方形的面积是 40 平方厘米。"
        },
        {
            "instruction": "把 3/4 和 2/5 相加，结果用分数表示。",
            "response": "首先找到公分母：4 和 5 的最小公倍数是 20。\n将两个分数通分：\n  3/4 = 15/20\n  2/5 = 8/20\n然后相加：15/20 + 8/20 = 23/20\n结果为 23/20（即 1 又 3/20）。"
        },
        # ---- Writing ----
        {
            "instruction": "请用三句话描写春天的景色。",
            "response": "春风轻拂大地，樱花纷纷飘落在小径上，空气中弥漫着淡淡的花香。河边的柳树抽出嫩绿的新芽，在阳光下闪闪发光。孩子们在草地上奔跑嬉戏，欢笑声和鸟鸣声交织在一起，构成了一幅生机勃勃的画面。"
        },
        {
            "instruction": "写一段鼓励正在准备考试的同学的话。",
            "response": "亲爱的同学，你已经在备考的路上付出了那么多努力，这些付出一定不会白费。考试只是检验知识的一种方式，它不能定义你的全部价值。请相信自己日积月累的实力，保持平常心，认真对待每一道题。无论结果如何，你在这个过程中收获的知识和毅力，才是最珍贵的财富。加油！"
        },
        {
            "instruction": "请以'我的编程之旅'为题写一段短文，大约100字。",
            "response": "我的编程之旅始于大学第一堂 C 语言课。那时的我对着屏幕上跳动的字符感到既新奇又困惑。随着一行行代码从报错到成功运行，我逐渐体会到了创造的乐趣。从简单的计算器到复杂的 Web 应用，每一次调试成功都让我充满成就感。编程不仅教会了我逻辑思维，更让我学会了面对问题时的耐心与坚持。"
        },
        # ---- Coding ----
        {
            "instruction": "用 Python 写一个函数，判断一个数是否为偶数。",
            "response": "```python\ndef is_even(n):\n    \"\"\"判断一个数是否为偶数\"\"\"\n    return n % 2 == 0\n\n# 测试\nprint(is_even(4))   # True\nprint(is_even(7))   # False\n```\n这个函数利用取模运算（%）来判断：如果一个数除以 2 的余数为 0，它就是偶数。"
        },
        {
            "instruction": "解释 Python 中列表和元组的区别。",
            "response": "Python 中列表（list）和元组（tuple）的主要区别如下：\n\n1. **可变性**：列表是可变的，创建后可以增删改元素；元组是不可变的，创建后不能修改。\n2. **语法**：列表使用方括号 `[1, 2, 3]`，元组使用圆括号 `(1, 2, 3)`。\n3. **性能**：元组比列表更轻量，访问速度稍快。\n4. **用途**：列表适合需要频繁修改的数据集合；元组适合固定不变的数据，如坐标、配置等。\n\n简单记忆：列表像笔记本可以擦写，元组像石刻一旦刻好就不变了。"
        },
        {
            "instruction": "用 Python 实现一个简单的冒泡排序。",
            "response": "```python\ndef bubble_sort(arr):\n    \"\"\"冒泡排序：重复遍历列表，每次将最大的元素'冒泡'到最后\"\"\"\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        # 如果某一轮没有发生交换，说明已经有序\n        if not swapped:\n            break\n    return arr\n\n# 测试\nnumbers = [64, 34, 25, 12, 22, 11, 90]\nprint(\"排序前:\", numbers)\nprint(\"排序后:\", bubble_sort(numbers))\n```\n冒泡排序的时间复杂度为 O(n²)，适合小规模数据的排序教学。"
        },
        # ---- Knowledge Q&A ----
        {
            "instruction": "解释什么是机器学习，用通俗的语言。",
            "response": "想象你在教一个小朋友认水果。你给他看了很多苹果和香蕉的照片，告诉他哪些是苹果、哪些是香蕉。看了几百张之后，小朋友自己就能分辨新的水果了——这就是机器学习的核心思想。\n\n用术语来说：机器学习是让计算机通过数据自动学习规律，而不是靠人手动编写每一条规则。常见类型包括：\n- **监督学习**：给模型带标签的数据（比如标记好的水果图片），让它学会预测。\n- **无监督学习**：只给数据不给标签，让模型自己发现数据中的模式。\n- **强化学习**：通过试错和奖励反馈来学习最优策略。"
        },
        {
            "instruction": "太阳系有哪几大行星？按距离太阳由近到远排列。",
            "response": "太阳系八大行星按距离太阳由近到远排列如下：\n\n1. **水星**（Mercury）—— 最小、最靠近太阳的行星\n2. **金星**（Venus）—— 表面温度最高的行星\n3. **地球**（Earth）—— 我们的家园，唯一已知存在生命的行星\n4. **火星**（Mars）—— 被称为红色星球\n5. **木星**（Jupiter）—— 太阳系最大的行星\n6. **土星**（Saturn）—— 以美丽的光环著称\n7. **天王星**（Uranus）—— 侧躺着自转的冰巨星\n8. **海王星**（Neptune）—— 最远的行星，风速最快\n\n记忆口诀：「水金地火木土天海」。"
        },
        {
            "instruction": "什么是 HTTP 协议？GET 和 POST 请求有什么区别？",
            "response": "HTTP（超文本传输协议）是浏览器和服务器之间通信的规则，就像两个人打电话时的'语言规范'。\n\nGET 和 POST 是最常用的两种请求方法：\n\n| 特征       | GET                        | POST                      |\n|-----------|----------------------------|---------------------------|\n| 用途       | 获取数据                    | 提交数据                   |\n| 参数位置    | URL 中（可见）               | 请求体中（不可见）           |\n| 数据量     | 有限制（约 2KB）             | 无限制                     |\n| 安全性     | 较低（参数暴露在 URL 中）      | 较高                       |\n| 幂等性     | 幂等（多次请求结果相同）       | 非幂等                     |\n| 缓存      | 可被浏览器缓存                | 默认不缓存                  |\n\n简单来说：GET 像「查阅资料」，POST 像「提交表单」。"
        },
        {
            "instruction": "什么是递归？举一个生活中的例子。",
            "response": "递归是指函数在其定义中调用自身的编程技巧。\n\n**生活中的例子——俄罗斯套娃：**\n你拿到一个大套娃，打开后发现里面有个小一点的套娃，再打开又有一个更小的……直到最小的那个打不开了——这就对应递归的「终止条件」。\n\n**编程中的例子——计算阶乘：**\n```python\ndef factorial(n):\n    if n == 1:        # 终止条件（最小的套娃）\n        return 1\n    return n * factorial(n - 1)  # 打开一层，继续处理里面的小套娃\n\nprint(factorial(5))  # 120\n```\n\n递归的两个关键要素：\n1. **终止条件**：什么时候停下来（不然会无限循环）\n2. **递归步骤**：把大问题分解成小问题，用同样的方式处理"
        },
        {
            "instruction": "请解释什么是数据库索引，为什么它能加速查询？",
            "response": "数据库索引就像书的目录。\n\n**没有目录时**：你要找书中某个知识点，只能从头到尾一页页翻——这叫「全表扫描」，非常慢。\n\n**有了目录后**：你先在目录里找到关键词所在的页码，然后直接翻到那一页——这就是索引的原理。\n\n索引的工作方式：\n1. 数据库为某一列（如用户 ID）创建一个额外的数据结构（通常是 B+ 树）\n2. 这个结构按值排序存储，可以快速定位到对应的行\n3. 查询时先查索引，再通过指针找到实际数据\n\n**代价**：索引会占用额外的存储空间，并且在插入、更新数据时需要同步维护索引。所以索引要加在经常查询的列上，而不是越多越好。"
        },
    ]

    return sft_pairs


# ==========================================
# 2. Format data into conversational form
# ==========================================

def format_sft_dataset(sft_pairs, tokenizer):
    """
    Convert instruction-response pairs into a format the model can train on.
    Uses the chat template to wrap each entry as a multi-turn conversation.
    """
    formatted_texts = []
    for pair in sft_pairs:
        messages = [
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        # Use the tokenizer's chat_template to format the conversation into training text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)

    return Dataset.from_dict({"text": formatted_texts})


# ==========================================
# 3. Main flow: SFT training
# ==========================================

def main():
    print("=" * 60)
    print("Chapter 8: SFT (Supervised Fine-Tuning) Pipeline")
    print("=" * 60)

    # ---- 3.1 Generate and inspect training data ----
    print("\n[Step 1] Generating Self-Instruct-style training data...")
    sft_pairs = generate_sft_data()
    print(f"  Generated {len(sft_pairs)} instruction-response training examples")
    print(f"  Categories: math ({3}), writing ({3}), coding ({3}), knowledge Q&A ({6})")

    # Print a sample
    sample = sft_pairs[0]
    print(f"\n  Sample example:")
    print(f"    Instruction: {sample['instruction']}")
    print(f"    Response: {sample['response'][:50]}...")

    # ---- 3.2 Load model and tokenizer ----
    print("\n[Step 2] Loading Qwen2.5-0.5B-Instruct model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    )
    print(f"  Model loaded: {model_name}")

    # ---- 3.3 Test before fine-tuning ----
    print("\n[Step 3] Testing model output before fine-tuning...")
    test_instructions = [
        "用 Python 写一个判断奇偶数的函数。",
        "解释什么是递归。",
    ]

    print("  --- Output before fine-tuning ---")
    before_responses = []
    for inst in test_instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        before_responses.append(response)
        print(f"  Q: {inst}")
        print(f"  A: {response[:80]}...")
        print()

    # ---- 3.4 Prepare the SFT training dataset ----
    print("[Step 4] Formatting the training dataset...")
    train_dataset = format_sft_dataset(sft_pairs, tokenizer)
    print(f"  Dataset size: {len(train_dataset)} examples")
    print(f"  Data preview: {train_dataset[0]['text'][:100]}...")

    # ---- 3.5 Configure and run SFT training ----
    print("\n[Step 5] Starting SFT training...")
    print("  Hyperparameters:")
    print("    - per_device_train_batch_size = 2")
    print("    - learning_rate = 2e-5")
    print("    - num_train_epochs = 2")
    print("    - max_length = 512")

    sft_config = SFTConfig(
        output_dir="./output/sft_results",
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        num_train_epochs=2,
        max_length=512,
        logging_steps=1,          # Log every step (small dataset)
        save_strategy="epoch",    # Save once per epoch
        report_to="none",         # Don't upload to wandb, etc.
        fp16=False,               # Use float32 on CPU
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Start training
    train_result = trainer.train()

    # ---- 3.6 Print training loss curve data ----
    print("\n[Step 6] Training loss log:")
    print("  " + "-" * 40)
    print("  Step | Training Loss")
    print("  " + "-" * 40)

    # Extract loss data from the training log
    if hasattr(trainer, "state") and trainer.state.log_history:
        for log_entry in trainer.state.log_history:
            if "loss" in log_entry:
                step = log_entry.get("step", "?")
                loss = log_entry["loss"]
                print(f"  {str(step).rjust(4)} | {loss:.4f}")
    else:
        print(f"  Final training loss: {train_result.training_loss:.4f}")

    print("  " + "-" * 40)

    # ---- 3.7 Test after fine-tuning ----
    print("\n[Step 7] Testing model output after fine-tuning...")
    print("  --- Output after fine-tuning ---")

    # Load the fine-tuned model for inference
    model.eval()
    after_responses = []
    for inst in test_instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        after_responses.append(response)
        print(f"  Q: {inst}")
        print(f"  A: {response[:80]}...")
        print()

    # ---- 3.8 Before/after comparison summary ----
    print("=" * 60)
    print("Before/after fine-tuning comparison summary:")
    print("=" * 60)
    for i, inst in enumerate(test_instructions):
        print(f"\n  Instruction: {inst}")
        print(f"  Before: {before_responses[i][:60]}...")
        print(f"  After: {after_responses[i][:60]}...")

    # ---- 3.9 Save the SFT model ----
    print("\n[Step 8] Saving the SFT model...")
    save_path = "./output/sft_results/sft_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  SFT model saved to: {save_path}")
    print("  This model will be used in the next stage (reward model training / PPO alignment).")

    # Also save the training data for use in later stages
    data_save_path = "./output/sft_results/sft_training_data.json"
    with open(data_save_path, "w", encoding="utf-8") as f:
        json.dump(sft_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Training data saved to: {data_save_path}")

    print("\n" + "=" * 60)
    print("SFT stage complete!")
    print("Next, run reward_model_training.py to train the reward model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
