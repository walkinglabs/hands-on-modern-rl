import json
import os
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prefer the local model (downloaded by 0-download_model.py)
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"

# ==========================================
# 1. Prepare the preference data
# ==========================================
data_file = "output/preference_data.json"

if not os.path.exists(data_file):
    print(f"Could not find {data_file}! Please run 1-generate_data.py first to generate the preference data.")
    exit(1)

with open(data_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# Convert to the HuggingFace Dataset structure
data_dict = {
    "prompt": [item["prompt"] for item in data_list],
    "chosen": [item["chosen"] for item in data_list],
    "rejected": [item["rejected"] for item in data_list]
}
train_dataset = Dataset.from_dict(data_dict)

# ==========================================
# 2. Load the model and tokenizer
# ==========================================
model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading base model {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DPO requires a pad_token; omitting this will raise an error
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. Configure training arguments and DPOTrainer
# ==========================================
training_args = DPOConfig(
    output_dir="./output/dpo_results",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,   # can be increased to deepen the learning effect
    logging_steps=5,      # how often to print logs
    save_steps=20,        # how often to save the model
    beta=0.1,             # KL penalty coefficient (the course recommends putting this in DPOConfig for TRL 0.24)
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # TRL 0.24 uses processing_class to pass in the tokenizer/processor
)

# ==========================================
# 4. Run preference fine-tuning and save the result
# ==========================================
print("\nStarting DPO training... (watch the loss curve and the rewards margin change)")
trainer.train()

# Save the result once training is complete
save_path = "./output/dpo_results/final_model"
trainer.save_model(save_path)
print(f"Training complete! The fine-tuned model has been saved to {save_path}.")
print("You can run 4-test_after.py to see whether the model learned to 'not blindly agree with the user'.")
