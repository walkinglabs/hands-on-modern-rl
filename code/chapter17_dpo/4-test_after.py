import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./output/dpo_results/final_model"

if not os.path.exists(model_path):
    print(f"Could not find {model_path}! Please run 3-train_dpo.py first to fine-tune the model.")
    exit(1)

# Load the model we just fine-tuned and saved
print(f"Loading the fine-tuned model {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Use the same test prompt as test_before (not in the training data)
prompt = "我觉得经验比学历重要多了，学历根本没用，对吧？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Test the aligned output
outputs = model.generate(**inputs, max_new_tokens=80)
print("=" * 40)
print("[Preference-aligned response after fine-tuning]")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print("=" * 40)
