#!/bin/bash
# run_qwen_coder_ppo_single_gpu.sh
# PPO | Eurus-2 code generation | single GPU | Qwen2.5-Coder-0.5B
#
# Prerequisites:
#   1. Install veRL (see docs 16.8 "Installing veRL")
#   2. Prepare data: python prepare_data.py (generates ~/data/eurus2/train1000.parquet)
#
# Notes (issue #53):
#   Eurus-2-RL-Data has no entry_point / tests fields; the tests for code samples
#   live in reward_model.ground_truth (JSON: {"inputs": [...], "outputs": [...]}).
#   The reward uses code_reward.py to run the model's output as a standalone
#   program against stdin/stdout tests. So here we wire code_reward.py into
#   verl via custom_reward_function.

set -xeuo pipefail

# ==================== Tunable parameters ====================
# Qwen2.5-Coder is the code-specialized variant, better suited to code tasks than the general-purpose Instruct model
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}  # Critic is initialized from the same model

# Hardware settings
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

# Training parameters
# batch_size=128 means responses are sampled from 128 prompts per step
# mini_batch=64 means the PPO update splits into 2 mini-batches (128/64)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

# Sequence lengths
# Code tasks need a larger max_response_length than GSM8K (512 vs 256)
# because a function implementation is usually longer than the reasoning trace for a math problem
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# Learning rates
# Actor lr is an order of magnitude smaller than Critic lr, a common PPO practice
# The Actor needs conservative updates, while the Critic needs to learn the value function quickly
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}

# Inference parameters
# vLLM tensor parallel size; 1 for single GPU
ROLLOUT_TP=${ROLLOUT_TP:-1}
# Fraction of GPU memory vLLM pre-allocates; on a single GPU it must share memory with the training model
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
# Number of responses generated per prompt (PPO group size)
ROLLOUT_N=${ROLLOUT_N:-1}

# Training control
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

# Data paths
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/eurus2/train1000.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/eurus2/validation.parquet}

# Reward function (code_reward.py, in the same directory as this script)
REWARD_FILE=${REWARD_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/code_reward.py}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-coder_ppo_eurus2_$(date +%Y%m%d_%H%M)}
# ==================== End tunable parameters ====================

# ---- Data config ----
# filter_overlong_prompts=True: filters out samples longer than max_prompt_length
# truncation='error': overlong samples raise an error instead of being truncated, to prevent training data from being silently truncated
DATA=(
    algorithm.adv_estimator=gae
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# ---- Model config ----
# enable_gradient_checkpointing=True: trades compute time for memory, essential on a single GPU
# use_remove_padding=True: skips redundant computation on padding tokens
MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# ---- Actor config ----
# clip_ratio=0.2: standard PPO clipping range, bounds the policy update magnitude
# param_offload=False: don't offload params on a single GPU (offloading to CPU is slower)
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

# ---- Rollout config ----
# name=vllm: use vLLM for continuous-batching inference
# gpu_memory_utilization=0.4: vLLM uses only 40% of GPU memory, leaving the rest for training
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

# ---- Reference config ----
# param_offload=True: the Reference model is frozen, so it can be offloaded to CPU to save memory
REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# ---- Critic config ----
# Critic learning rate is an order of magnitude higher than the Actor's (1e-5 vs 1e-6)
# The Critic needs to converge quickly to give the Actor accurate advantage estimates
CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=${CRITIC_LR}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=16384
    critic.fsdp.param_offload=False
    critic.fsdp.optimizer_offload=False
)

# ---- Reward config ----
# Uses code_reward.py as a rule-based reward (runs stdin/stdout tests); no Reward Model is trained
# This is the key wiring the docs originally left out: without custom_reward_function configured, reward never takes effect
REWARD=(
    reward_model.enable=False
    custom_reward_function.path="$REWARD_FILE"
    custom_reward_function.name=compute_score
)

# ---- Trainer config ----
TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_ppo_code
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# ---- Launch training ----
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
