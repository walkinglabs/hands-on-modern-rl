"""
Chapter 8: RLHF PPO Alignment Training
=========================================

This script demonstrates the third stage of the three-stage RLHF pipeline —
PPO alignment training.
Contents:
  1. Load the SFT model as the policy model (Actor)
  2. Use the reward model to score generated responses
  3. PPO training loop: generate → score → compute advantage → clipped update
  4. Track reward, KL divergence, response length, and other metrics
  5. Compare response quality before and after alignment

Note: this is a simplified/simulated version. A full RLHF-PPO training run
typically requires:
  - A large-scale cluster (tens to hundreds of GPUs)
  - Hundreds of thousands of preference examples
  - A sophisticated distributed training framework
  This script is meant to help you understand the core PPO algorithm flow
  within RLHF.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Create the output directory
os.makedirs("output", exist_ok=True)

# Configure a CJK-capable font
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. Helper functions
# ==========================================

def generate_response(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7):
    """
    Generate a response using the model.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    input_length = inputs["input_ids"].shape[-1]
    return response, input_length, outputs[0]


def compute_log_probs(model, input_ids, attention_mask):
    """
    Compute the model's log-probabilities over a given sequence.
    Used to compute the importance-sampling ratio in PPO.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Take the log-probability of predicting the next token at each position
    # logits[:, :-1, :] corresponds to the prediction at position t; the target is input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Compute log-softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Extract the log-probability of the actual token
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Use the attention mask to exclude padding (aligned to the shifted positions)
    shift_mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * shift_mask

    # Return the sequence's average log-probability
    return token_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)


# ==========================================
# 2. Simplified reward model
# ==========================================

class SimpleRewardModel:
    """
    A simplified reward model.

    In real RLHF, the reward model is a deep neural network trained on
    preference pairs. Here we use a rule-based scoring function to simulate
    the reward model's behavior, so the PPO alignment flow can be
    demonstrated quickly on a single machine.

    Scoring rules:
      - Moderate response length (50-200 characters): bonus
      - Contains useful structured information (numbered lists, code blocks, etc.): bonus
      - Friendly, polite tone: bonus
      - Response too short or refuses to answer: penalty
    """

    def __init__(self, tokenizer, backbone_model=None):
        self.tokenizer = tokenizer
        self.backbone_model = backbone_model

        # If a backbone model is provided, try to load a trained value head
        self.value_head = None
        if backbone_model is not None:
            hidden_size = backbone_model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)

            # Try to load trained value head parameters
            value_head_path = "./output/rm_results/value_head.pt"
            if os.path.exists(value_head_path):
                self.value_head.load_state_dict(
                    torch.load(value_head_path, map_location="cpu")
                )
                print(f"  Loaded trained value head parameters: {value_head_path}")

    def score(self, prompt, response):
        """
        Score a (prompt, response) pair.
        Returns a scalar reward value.

        If a trained neural reward model is available, it is used first;
        otherwise a rule-based score is used.
        """
        # Try to use the neural reward model
        if self.backbone_model is not None and self.value_head is not None:
            return self._neural_score(prompt, response)
        else:
            return self._rule_based_score(prompt, response)

    def _neural_score(self, prompt, response):
        """Score using the neural reward model"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            text, truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.backbone_model(
                **enc, output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            seq_len = enc["attention_mask"].sum(dim=1) - 1
            last_token_hidden = last_hidden[0, seq_len[0]]
            reward = self.value_head(last_token_hidden).item()

        # Combine the neural network output with the rule-based score for robustness
        rule_score = self._rule_based_score(prompt, response)
        return 0.5 * reward + 0.5 * rule_score

    def _rule_based_score(self, prompt, response):
        """Rule-based scoring function (simulates a reward model)"""
        score = 0.0

        # ---- Length score ----
        length = len(response)
        if length < 10:
            score -= 2.0  # Response too short
        elif length < 30:
            score -= 0.5  # Response somewhat short
        elif 50 <= length <= 300:
            score += 1.5  # Moderate length
        elif length > 500:
            score -= 0.5  # Too long, possibly redundant

        # ---- Structured content score ----
        if any(marker in response for marker in ["1.", "2.", "3.", "（1）", "（2）"]):
            score += 1.0  # Has a numbered list, well structured
        if "```" in response:
            score += 1.0  # Contains a code block
        if any(marker in response for marker in ["：\n", "：\r\n", "步骤", "方法"]):
            score += 0.5  # Has structured explanation

        # ---- Tone score ----
        positive_words = ["请", "建议", "可以帮助", "以下", "当然", "好的"]
        negative_words = ["不关我事", "自己搜", "随便", "不想", "懒得"]
        for word in positive_words:
            if word in response:
                score += 0.3
        for word in negative_words:
            if word in response:
                score -= 1.0

        # ---- Relevance score ----
        # Check whether the response is relevant to the prompt (simple keyword matching)
        prompt_keywords = set(prompt.replace("？", "").replace("？", "").replace("，", "").split())
        response_words = set(response.replace("，", "").replace("。", "").split())
        overlap = len(prompt_keywords & response_words)
        if overlap > 0:
            score += min(overlap * 0.2, 1.0)

        return score


# ==========================================
# 3. PPO trainer
# ==========================================

class PPOTrainer:
    """
    A simplified PPO trainer.

    The core PPO (Proximal Policy Optimization) flow in RLHF:
      1. The policy model (Actor) generates a response
      2. The reward model scores the response
      3. Compute the advantage function
      4. Update the policy model using the clipped objective
      5. Add a KL-divergence penalty to keep the policy close to the reference model

    PPO clipped objective:
      L_CLIP = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

    where r(θ) = π_θ(a|s) / π_ref(a|s) is the probability ratio between the
    new and old policies.

    Total loss = -L_CLIP + β * KL(π_θ || π_ref)
    """

    def __init__(
        self,
        policy_model,
        reference_model,
        reward_model,
        tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-6,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef        # KL divergence penalty coefficient
        self.clip_range = clip_range  # PPO clip range

        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=learning_rate
        )

        # Training statistics
        self.stats = {
            "rewards": [],
            "kl_divergences": [],
            "policy_losses": [],
            "total_losses": [],
            "response_lengths": [],
        }

    def compute_kl_divergence(self, input_ids, attention_mask):
        """
        Compute the KL divergence between the policy model and the reference model.
        KL(π_θ || π_ref) = Σ π_θ * log(π_θ / π_ref)

        Uses an approximation here: the KL divergence at each token position is averaged.
        """
        with torch.no_grad():
            # Policy model logits
            policy_outputs = self.policy_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            policy_logits = policy_outputs.logits[:, :-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_probs = torch.softmax(policy_logits, dim=-1)

            # Reference model logits
            ref_outputs = self.reference_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # Compute KL divergence per token
            kl_per_token = (
                policy_probs * (policy_log_probs - ref_log_probs)
            ).sum(dim=-1)

            # Exclude padding tokens
            shift_mask = attention_mask[:, 1:]
            kl_div = (kl_per_token * shift_mask).sum() / shift_mask.sum()

        return kl_div.item()

    def train_step(self, prompts):
        """
        Run one step of PPO training.

        Steps:
          1. Use the policy model to generate a response for each prompt
          2. Score the responses with the reward model
          3. Compute the advantage function
          4. Compute the PPO clipped loss + KL penalty
          5. Backpropagate and update the policy model
        """
        self.policy_model.train()

        batch_rewards = []
        batch_kl = []
        batch_lengths = []
        all_input_ids = []
        all_attention_masks = []
        all_old_log_probs = []

        for prompt in prompts:
            # ---- Step 1: generate a response ----
            response, input_len, full_ids = generate_response(
                self.policy_model, self.tokenizer, prompt,
                max_new_tokens=60, temperature=0.8,
            )

            # Prepare encoding
            input_ids = full_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            # ---- Step 2: score with the reward model ----
            reward = self.reward_model.score(prompt, response)
            batch_rewards.append(reward)
            batch_lengths.append(len(response))

            # ---- Step 3: compute KL divergence ----
            kl_div = self.compute_kl_divergence(input_ids, attention_mask)
            batch_kl.append(kl_div)

            # ---- Step 4: record the old policy's log-probability ----
            with torch.no_grad():
                old_log_prob = compute_log_probs(
                    self.policy_model, input_ids, attention_mask
                )
            all_old_log_probs.append(old_log_prob)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # ---- Step 5: compute the advantage function ----
        # Simplified version: use the reward value itself as the advantage (no GAE estimation)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
        advantages = rewards_tensor - rewards_tensor.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        # ---- Step 6: PPO clipped update ----
        total_policy_loss = 0.0
        for i, (input_ids, att_mask, old_log_p) in enumerate(
            zip(all_input_ids, all_attention_masks, all_old_log_probs)
        ):
            # New policy's log-probability
            new_log_prob = compute_log_probs(
                self.policy_model, input_ids, att_mask
            )

            # Importance-sampling ratio
            ratio = torch.exp(new_log_prob - old_log_p)

            # PPO clipped objective
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            ) * advantage

            # Take the minimum (conservative update)
            policy_loss = -torch.min(surr1, surr2)
            total_policy_loss += policy_loss

        # Average policy loss
        avg_policy_loss = total_policy_loss / len(prompts)

        # KL penalty term
        avg_kl = sum(batch_kl) / len(batch_kl)
        kl_penalty = self.kl_coef * avg_kl

        # Total loss = policy loss + KL penalty
        total_loss = avg_policy_loss + kl_penalty

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        # Record statistics
        self.stats["rewards"].append(sum(batch_rewards) / len(batch_rewards))
        self.stats["kl_divergences"].append(avg_kl)
        self.stats["policy_losses"].append(avg_policy_loss.item())
        self.stats["total_losses"].append(total_loss.item())
        self.stats["response_lengths"].append(
            sum(batch_lengths) / len(batch_lengths)
        )

        return {
            "avg_reward": self.stats["rewards"][-1],
            "avg_kl": avg_kl,
            "policy_loss": avg_policy_loss.item(),
            "total_loss": total_loss.item(),
        }


# ==========================================
# 4. Training metrics visualization
# ==========================================

def plot_training_stats(stats, save_path="output/ppo_training_stats.png"):
    """
    Plot the metrics tracked during PPO training.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ---- Average reward ----
    ax = axes[0, 0]
    ax.plot(stats["rewards"], "g-o", markersize=4)
    ax.set_title("Average Reward")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # ---- KL divergence ----
    ax = axes[0, 1]
    ax.plot(stats["kl_divergences"], "r-o", markersize=4)
    ax.set_title("KL Divergence")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL divergence")
    ax.grid(True, alpha=0.3)

    # ---- Policy loss ----
    ax = axes[1, 0]
    ax.plot(stats["policy_losses"], "b-o", markersize=4)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ---- Response length ----
    ax = axes[1, 1]
    ax.plot(stats["response_lengths"], "m-o", markersize=4)
    ax.set_title("Average Response Length")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Character count")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training metrics plot saved to: {save_path}")


# ==========================================
# 5. Main flow
# ==========================================

def main():
    print("=" * 60)
    print("Chapter 8: RLHF PPO Alignment Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")
    print("  Note: this is a simplified/simulated version, meant to demonstrate the core PPO alignment algorithm flow.")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- 5.1 Load the SFT model (policy model) ----
    print("\n[Step 1] Loading the policy model (SFT model)...")

    sft_path = "./output/sft_results/sft_model"
    if os.path.exists(sft_path):
        print(f"  Found a saved SFT model: {sft_path}")
        policy_model = AutoModelForCausalLM.from_pretrained(
            sft_path, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(sft_path)
        print("  Loaded the SFT model as the policy model.")
    else:
        print(f"  No SFT model found, loading the base model {model_name} directly")
        print("  (It is recommended to run sft_pipeline.py first to perform SFT training)")
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 5.2 Create the reference model (frozen, not trained) ----
    print("\n[Step 2] Creating the reference model (a frozen copy of the SFT model)...")
    # The reference model is an initial copy of the policy model, used to compute KL divergence
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("  Reference model created and its parameters frozen.")

    # ---- 5.3 Initialize the reward model ----
    print("\n[Step 3] Initializing the reward model...")

    # Try to load a previously trained reward model backbone
    rm_backbone = None
    rm_backbone_path = "./output/rm_results"
    if os.path.exists(os.path.join(rm_backbone_path, "value_head.pt")):
        print(f"  Found trained reward model parameters: {rm_backbone_path}")
        rm_backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
    else:
        print("  No trained reward model found; falling back to the rule-based scoring function.")
        print("  (It is recommended to run reward_model_training.py first to train the reward model)")

    reward_model = SimpleRewardModel(tokenizer, backbone_model=rm_backbone)
    print("  Reward model initialized.")

    # ---- 5.4 Test before alignment ----
    print("\n[Step 4] Testing model output before PPO alignment...")
    test_prompts = [
        "用 Python 写一个求列表最大值的函数。",
        "解释什么是机器学习。",
        "给我讲一个有趣的故事。",
        "如何提高英语水平？",
    ]

    print("  --- Output before PPO alignment ---")
    before_responses = []
    before_rewards = []
    for prompt in test_prompts:
        response, _, _ = generate_response(
            policy_model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7,
        )
        reward = reward_model.score(prompt, response)
        before_responses.append(response)
        before_rewards.append(reward)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:80]}...")
        print(f"  Reward score: {reward:.3f}")
        print()

    # ---- 5.5 Configure PPO training ----
    print("[Step 5] Configuring PPO training...")
    print("  Hyperparameters:")
    print("    - learning_rate = 1e-6")
    print("    - KL penalty coefficient (β) = 0.1")
    print("    - PPO clip range (ε) = 0.2")
    print("    - training steps = 10")

    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-6,
    )

    # Prompt pool used for training
    train_prompts_pool = [
        "请解释什么是深度学习。",
        "用 Python 写一个冒泡排序。",
        "如何学习编程？",
        "什么是人工智能？",
        "请推荐几本技术书籍。",
        "如何准备技术面试？",
        "解释什么是 RESTful API。",
        "写一段鼓励正在学习的人的话。",
    ]

    # ---- 5.6 Run the PPO training loop ----
    print("\n[Step 6] Starting PPO alignment training...")
    print("  " + "-" * 60)
    print(f"  {'Step':>4} | {'AvgReward':>8} | {'KL':>8} | {'PolicyLoss':>8} | {'TotalLoss':>8}")
    print("  " + "-" * 60)

    num_steps = 10
    for step in range(num_steps):
        # Randomly sample a batch of prompts each step
        step_prompts = random_sample(train_prompts_pool, k=4)

        # Run one step of PPO training
        step_stats = ppo_trainer.train_step(step_prompts)

        print(f"  {step + 1:>4} | "
              f"{step_stats['avg_reward']:>8.3f} | "
              f"{step_stats['avg_kl']:>8.4f} | "
              f"{step_stats['policy_loss']:>8.4f} | "
              f"{step_stats['total_loss']:>8.4f}")

    print("  " + "-" * 60)

    # ---- 5.7 Print training metrics summary ----
    stats = ppo_trainer.stats

    print("\n[Step 7] Training metrics summary:")
    print(f"  Reward change: {stats['rewards'][0]:.3f} → {stats['rewards'][-1]:.3f} "
          f"(delta: {stats['rewards'][-1] - stats['rewards'][0]:+.3f})")
    print(f"  KL divergence change: {stats['kl_divergences'][0]:.4f} → {stats['kl_divergences'][-1]:.4f}")
    print(f"  Response length change: {stats['response_lengths'][0]:.1f} → {stats['response_lengths'][-1]:.1f}")

    # ---- 5.8 Visualize the training process ----
    print("\n[Step 8] Visualizing the training process...")
    plot_training_stats(stats, save_path="output/ppo_training_stats.png")

    # ---- 5.9 Test after alignment ----
    print("\n[Step 9] Testing model output after PPO alignment...")
    print("  --- Output after PPO alignment ---")

    policy_model.eval()
    after_responses = []
    after_rewards = []
    for prompt in test_prompts:
        response, _, _ = generate_response(
            policy_model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7,
        )
        reward = reward_model.score(prompt, response)
        after_responses.append(response)
        after_rewards.append(reward)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:80]}...")
        print(f"  Reward score: {reward:.3f}")
        print()

    # ---- 5.10 Before/after comparison summary ----
    print("=" * 60)
    print("Before/after PPO alignment comparison summary:")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts):
        print(f"\n  Prompt: {prompt}")
        print(f"  Before [{before_rewards[i]:.3f}]: {before_responses[i][:60]}...")
        print(f"  After [{after_rewards[i]:.3f}]: {after_responses[i][:60]}...")
        print(f"  Reward change: {after_rewards[i] - before_rewards[i]:+.3f}")

    avg_before = sum(before_rewards) / len(before_rewards)
    avg_after = sum(after_rewards) / len(after_rewards)
    print(f"\n  Average reward: before {avg_before:.3f} → after {avg_after:.3f} "
          f"({avg_after - avg_before:+.3f})")

    # ---- 5.11 Save the aligned model ----
    print("\n[Step 10] Saving the PPO-aligned model...")
    save_dir = "./output/ppo_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aligned_model")
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Aligned model saved to: {save_path}")

    # Save training statistics
    stats_path = os.path.join(save_dir, "ppo_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Training statistics saved to: {stats_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("All three RLHF pipeline stages complete!")
    print("=" * 60)
    print("\n  Stage recap:")
    print("  [1] SFT (supervised fine-tuning)  → output/sft_results/sft_model")
    print("  [2] RM (reward model training)    → output/rm_results/value_head.pt")
    print("  [3] PPO (alignment training)       → output/ppo_results/aligned_model")
    print("\n  Core concepts recap:")
    print("  - SFT: use instruction data to teach the model basic instruction-following ability")
    print("  - RM: learn human preferences, scoring good responses high and bad responses low")
    print("  - PPO: optimize the policy using feedback from the reward model, while using a KL penalty to stay stable")
    print("=" * 60)


def random_sample(lst, k):
    """Randomly sample k elements from a list"""
    import random
    return random.sample(lst, min(k, len(lst)))


if __name__ == "__main__":
    main()
