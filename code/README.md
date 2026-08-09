# Code Index

This directory contains the companion code for each chapter of the course. It is recommended to first enter the `code/` directory, then install dependencies and run scripts on a per-chapter basis.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Full set of dependencies, suitable when working through the whole course
pip install -r requirements.txt

# Or install dependencies for just one chapter
pip install -r chapter01_cartpole/requirements.txt
```

## Chapter Code Overview

| Chapter                 | Directory                       | Main Code                            | Description                                                                                                               |
| ----------------------- | ------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Ch01 CartPole           | `chapter01_cartpole/`           | `1-ppo_cartpole.py`                  | SB3 PPO trains CartPole, logging SwanLab metrics                                                                          |
|                         |                                 | `2-pytorch_ppo.py`                   | Pure PyTorch PPO: Actor-Critic, GAE, clipping, metric logging                                                             |
|                         |                                 | `plot_curves.py`                     | Reads training logs and plots metric curves                                                                               |
| Ch02 DPO                | `chapter02_dpo/`                | `0-download_model.py`                | Downloads Qwen2.5-0.5B-Instruct                                                                                           |
|                         |                                 | `1-generate_data.py`                 | Generates preference data correcting excessive compliance                                                                 |
|                         |                                 | `2-test_before.py`                   | Test before fine-tuning                                                                                                   |
|                         |                                 | `3-train_dpo.py`                     | TRL DPO training                                                                                                          |
|                         |                                 | `4-test_after.py`                    | Test after fine-tuning                                                                                                    |
| Ch03 MDP                | `chapter03_mdp/`                | `two_armed_bandit.py`                | Two-armed bandit policy comparison                                                                                        |
|                         |                                 | `bellman_equation_verify.py`         | Numerical verification of the Bellman equation                                                                            |
|                         |                                 | `gridworld_q_learning.py`            | GridWorld Q-Learning and path visualization                                                                               |
| Ch04 DQN                | `chapter04_dqn/`                | `dqn_cartpole.py`                    | DQN implemented from scratch to train CartPole                                                                            |
|                         |                                 | `double_dqn_cartpole.py`             | DQN vs. Double DQN comparison                                                                                             |
|                         |                                 | `dqn_gym_sb3.py`                     | SB3 DQN trains CartPole, MountainCar, LunarLander, and other discrete environments, logging SwanLab and evaluation curves |
|                         |                                 | `dqn_atari_sb3.py`                   | SB3 DQN trains Atari for real, including wrapper, SwanLab, evaluation, and logging                                        |
|                         |                                 | `export_dqn_curves.py`               | Exports lecture figures from the Chapter 4 DQN eval CSV                                                                   |
|                         |                                 | `dqn_pokemon_red_pyboy.py`           | PyBoy + SB3 DQN trains an early Pokemon exploration task                                                                  |
| Ch05 Policy Gradient    | `chapter05_policy_gradient/`    | `reinforce_cartpole.py`              | REINFORCE trains CartPole                                                                                                 |
|                         |                                 | `reinforce_with_baseline.py`         | REINFORCE vs. baseline comparison                                                                                         |
|                         |                                 | `actor_critic_cartpole.py`           | Actor-Critic and TD error                                                                                                 |
| Ch07 PPO                | `chapter07_ppo/`                | `ppo_lunar_lander.py`                | SB3 PPO trains LunarLander-v3                                                                                             |
|                         |                                 | `ppo_from_scratch.py`                | Pure PyTorch PPO                                                                                                          |
|                         |                                 | `gae_visualization.py`               | GAE parameter visualization                                                                                               |
| Ch08 RLHF               | `chapter08_rlhf/`               | `sft_pipeline.py`                    | SFT pipeline                                                                                                              |
|                         |                                 | `reward_model_training.py`           | Reward model training                                                                                                     |
|                         |                                 | `rlhf_ppo_train.py`                  | Simplified PPO-RLHF training loop                                                                                         |
|                         | `chapter08_rlhf/verl_gsm8k/`    | `run_qwen2_5_0_5b_ppo_single_gpu.sh` | 8.7 veRL + GSM8K external framework adaptation script                                                                     |
| Ch09 Alignment          | `chapter09_alignment/`          | `dpo_hands_on.py`                    | DPO alignment and beta comparison                                                                                         |
|                         |                                 | `dpo_math_reward.py`                 | DPO experiments on math preference data                                                                                   |
| Ch09 GRPO/RLVR          | `chapter09_grpo_rlvr/`          | `grpo_mechanism.py`                  | GRPO mechanism demonstration                                                                                              |
|                         |                                 | `grpo_math_reasoning.py`             | Small GRPO experiment on math reasoning                                                                                   |
|                         |                                 | `rule_based_reward.py`               | Rule-based reward function                                                                                                |
| Ch09 Continuous Control | `chapter09_continuous_control/` | `sac_halfcheetah.py`                 | SAC trains HalfCheetah-v4                                                                                                 |
|                         |                                 | `ppo_td3_sac_comparison.py`          | PPO, TD3, SAC comparison                                                                                                  |
| Ch10 Agentic RL         | `chapter10_agentic_rl/`         | `tool_use_agent.py`                  | Tool selection policy training                                                                                            |
|                         |                                 | `multi_turn_rl.py`                   | Multi-turn interaction credit assignment                                                                                  |
|                         |                                 | `generate_synthetic_data.py`         | Synthetic trajectory data                                                                                                 |
|                         |                                 | `mini_deep_research_grpo.py`         | Mini Deep Research GRPO example                                                                                           |
| Ch11 VLM RL             | `chapter11_vlm_rl/`             | `geometry_counting_dataset.py`       | Geometry counting dataset                                                                                                 |
|                         |                                 | `multi_modal_reward.py`              | Multi-modal rule-based reward                                                                                             |
|                         |                                 | `vlm_grpo_train.py`                  | VLM GRPO training example                                                                                                 |
| Ch12 Future Trends      | `chapter12_future_trends/`      | `tree_of_thought.py`                 | Tree of Thought search demonstration                                                                                      |
|                         |                                 | `multi_agent_marl.py`                | Multi-agent GridWorld                                                                                                     |
| Appendix Pitfalls       | `appendix_common_pitfalls/`     | `debug_reward_hacking.py`            | Reward hacking reproduction                                                                                               |
|                         |                                 | `debug_training_collapse.py`         | Training collapse diagnosis                                                                                               |

## Notes

- The `requirements.txt` in each chapter directory contains the minimal dependencies for that chapter.
- LLM-related chapters use small models by default, but running in a GPU environment is still recommended.
- Some scripts will generate `output/`, model weights, or image files in the current working directory.
- The Chapter 4 DQN scripts use SwanLab local mode by default; after running, you can view the curves with `swanlab watch swanlog`.
