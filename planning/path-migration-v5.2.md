# v5.2 Path Migration Map

> Old directory → new directory mapping, used as the basis for `git mv` operations.
> Processing order: handle 1:1 renames first (no splitting needed), then handle directories that need to be split.

## 1. 1:1 Renames (direct `git mv`)

| Old path                                   | New path                                   | Corresponding chapter                              |
| ------------------------------------------ | ------------------------------------------ | -------------------------------------------------- |
| `chapter02_dpo/`                           | `chapter17_dpo/`                           | Chapter 17 DPO                                     |
| `chapter03_bandits/`                       | `chapter02_bandits/`                       | Chapter 2 Bandits                                  |
| `chapter04_dqn/`                           | `chapter07_dqn/`                           | Chapter 7 DQN                                      |
| `chapter05_policy_gradient/`               | `chapter08_policy_gradient/`               | Chapter 8 Policy Gradient                          |
| `chapter06_actor_critic/`                  | `chapter09_actor_critic/`                  | Chapter 9 Actor-Critic                             |
| `chapter07_ppo/`                           | `chapter10_ppo/`                           | Chapter 10 PPO                                     |
| `chapter08_rlhf/`                          | `chapter15_rlhf/`                          | Chapter 15 RLHF                                    |
| `chapter09_grpo_rlvr/`                     | `chapter18_grpo/`                          | Chapter 18 GRPO                                    |
| `chapter12_continuous_control/`            | `chapter11_continuous_control/`            | Chapter 11 Continuous Control                      |
| `chapter13_offline_rl/`                    | `chapter12_offline_rl/`                    | Chapter 12 Offline RL                              |
| `chapter13_reasoning_models/`              | `chapter19_reasoning/`                     | Chapter 19 Reasoning                               |
| `chapter14_imitation_meta_rl/`             | `chapter13_imitation_meta_rl/`             | Chapter 13 Imitation/IRL/Meta-RL                   |
| `chapter14_prm_search/`                    | `chapter20_prm_search/`                    | Chapter 20 PRM                                     |
| `chapter15_exploration_marl_hierarchical/` | `chapter14_exploration_marl_hierarchical/` | Chapter 14 Exploration/MARL/Hierarchical           |
| `chapter15_rl_based_swe/`                  | `chapter23_rl_based_swe/`                  | Chapter 23 Code Agents                             |
| `chapter16_alignment_failures/`            | `chapter30_alignment_failures/`            | Chapter 30 Reward Hacking                          |
| `chapter17_llm_rl_industrial/`             | `chapter16_llm_rl_industrial/`             | Chapter 16 LLM RL in Industry                      |
| `chapter22_cai_rlvr/`                      | `chapter21_cai_rlvr/`                      | Chapter 21 CAI/RLAIF                               |
| `chapter28_computer_use/`                  | `chapter25_computer_use/`                  | Chapter 25 Computer Use                            |
| `chapter30_audio_rl/`                      | `chapter27_audio_rl/`                      | Chapter 27 Audio                                   |
| `chapter11_vlm_rl/`                        | `chapter26_vlm/`                           | Chapter 26 VLM (excluding visual-generation-rl.md) |

## 2. Kept As-Is (no rename)

| Path                  | Reason                                                         |
| --------------------- | -------------------------------------------------------------- |
| `chapter00_overview/` | Preface already merged; directory kept as a historical archive |
| `chapter01_cartpole/` | Chapter 1 numbering is already consistent                      |
| `chapter03_mdp/`      | Shared by Chapters 3-6 (MDP/value functions/DP/Q-Learning)     |
| `preface/`            | Preface directory                                              |

## 3. Directories That Need To Be Split

### `chapter09_alignment/`

- `industrial-post-training.md` → `chapter16_llm_rl_industrial/industrial-post-training.md` (16.2)
- `modern-industrial-practice.md` → `chapter16_llm_rl_industrial/modern-industrial-practice.md` (16.4)
- `dpo-theory-and-family.md` → `chapter17_dpo/dpo-theory-and-family.md` (17.3)
- Other files sorted into 16 or 17 by topic

### `chapter10_agentic_rl/`

- `multi-turn-rl.md`, `tool-use-*.md`, `industrial-*.md`, `trajectory-synthesis.md` → kept in `chapter22_agentic/` (main body of Chapter 22)
- `deep-research-agent.md` → moved to the new `chapter24_deep_research/intro.md` (Chapter 24)
- New addition: `chapter22_agentic/multi-agent-swarm.md` (22.6)

### `chapter12_future_trends/`

- `embodied-intelligence/` → moved to `chapter28_vla/` (Chapter 28 VLA)
- `llm-driven-discovery.md` → moved to `chapter31_alphaevolve/` (Chapter 31)
- `self-play-outlook/`, `rl-scaling-outlook.md`, `llm-multi-agent-rl/` → moved to `chapter32_selfplay/` (Chapter 32)

### `chapter23_rl_environments/`

- Entire directory contents → merged into `chapter18_grpo/` (as 18.5/18.6/18.7)

### `chapter35_rl_evaluation/`

- Entire directory contents → merged into `chapter30_alignment_failures/` (as 30.6/30.7)

### `chapter36_distributed_rl_training/`

- Entire directory contents → merged into `chapter16_llm_rl_industrial/` (as 16.5/16.6/16.7)

### `chapter34_scalable_oversight/`

- Already deleted (cut in v5.2). If the directory still exists, archive it to `archive/`.

### `chapter11_vlm_rl/visual-generation-rl.md` and similar

- Visual generation content → moved to the new `chapter29_visual_generation/` (Chapter 29)
- `video-generation-modern.md` → `chapter29_visual_generation/`

## 4. New Directories (brand-new v5.2 chapters)

| New path                       | Content                                                                  |
| ------------------------------ | ------------------------------------------------------------------------ |
| `chapter22_agentic/`           | Chapter 22 (renamed from chapter10_agentic_rl)                           |
| `chapter24_deep_research/`     | Chapter 24 Deep Research (new; split out from chapter10 + new 24.2/24.3) |
| `chapter25_computer_use/`      | Chapter 25 (renamed from chapter28_computer_use) + new 25.2/25.3         |
| `chapter29_visual_generation/` | Chapter 29 Visual Generation (new; split out from chapter11_vlm_rl)      |
| `chapter31_alphaevolve/`       | Chapter 31 (new; split out from chapter12_future_trends)                 |
| `chapter32_selfplay/`          | Chapter 32 (new; split out from chapter12_future_trends)                 |

## 5. Order of Operations

1. **Phase 1**: Simple 1:1 `git mv` (20+ directories)
2. **Phase 2**: Split compound directories (chapter09_alignment, chapter10_agentic_rl, chapter12_future_trends, chapter23_rl_environments, chapter35_rl_evaluation, chapter36_distributed_rl_training)
3. **Phase 3**: Create the new chapter24/29/31/32 directories and move in the corresponding files
4. **Phase 4**: Fill in missing files (22.6, 24.2, 24.3, 25.2, 25.3, etc.)
5. **Phase 5**: Rewrite config.mjs zhSidebar
6. **Phase 6**: Globally update cross-chapter links
7. **Phase 7**: Build verification
