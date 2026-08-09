# 8.7 veRL + GSM8K Adaptation Code

This directory corresponds to the tutorial [8.7 Hands-on: Running PPO Training on GSM8K with veRL](../../../docs/chapter08_rlhf/verl-ppo-gsm8k.md).

This repository does not copy the veRL source code. It only provides the GSM8K reward functions and launch scripts used in the course; the actual training entry point still comes from the external veRL repository.

## External Dependencies

- veRL official repository: <https://github.com/volcengine/verl>
- veRL PPO training entry point: `python3 -m verl.trainer.main_ppo`
- veRL GSM8K data preprocessing: `examples/data_preprocess/gsm8k.py`

## Usage

First install veRL following the tutorial, and prepare the GSM8K data:

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

Then use the scripts in this directory within the veRL environment:

```bash
cd /path/to/hands-on-modern-rl/code/chapter08_rlhf/verl_gsm8k
chmod +x run_qwen2_5_0_5b_ppo_single_gpu.sh
./run_qwen2_5_0_5b_ppo_single_gpu.sh
```

To switch to the advanced reward:

```bash
./run_qwen2_5_0_5b_ppo_single_gpu.sh \
  custom_reward_function.path="$(pwd)/gsm8k_reward_advanced.py" \
  custom_reward_function.name=compute_score
```

## File Overview

| File                                 | Purpose                             |
| ------------------------------------ | ----------------------------------- |
| `gsm8k_reward.py`                    | Basic 0/1 accuracy reward           |
| `gsm8k_reward_advanced.py`           | Combined accuracy + format reward   |
| `run_qwen2_5_0_5b_ppo_single_gpu.sh` | Single-GPU 0.5B PPO launch script   |
| `run_qwen2_5_0_5b_ppo_8gpu.sh`       | Single-node 8-GPU PPO launch script |
