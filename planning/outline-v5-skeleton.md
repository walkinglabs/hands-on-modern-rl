# Hands-On Modern Reinforcement Learning — v5 Outline (Structural Skeleton)

> This file verifies that the three-tier "Part / single article / in-article sub-outline" structure is clearly distinguished. Once the format is confirmed, expand it to all 38 chapters.

## Heading Level Rules (Mandatory, Unified)

| Level           | Marker            | Meaning                                                                                                                    |
| --------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Part**        | `#`               | Major part (Part I, Part II, preface...)                                                                                   |
| **Chapter**     | `##`              | A teaching unit, corresponding to a directory `chapterNN_xxx/`, must be tagged `[single article]` or `[multiple articles]` |
| **Article**     | `###`             | Appears only under `[multiple articles]`, one `.md` file                                                                   |
| **Sub-outline** | `-` indented list | The H2/H3 points to be expanded within that article                                                                        |

**Mandatory rules:**

- `###` is **not allowed** under a single-article chapter (to avoid misleading readers into thinking it's an independent file)
- `###` **must** be used to tag each file under a multiple-article chapter
- Design rationale and notes always go into `>` blockquotes, not mixed into the body text
- Version tags like `[v5 new]` and `[v5.1 expansion]` go in brackets after the chapter title, without polluting the heading hierarchy

---

## Reading Conventions

| Marker                | Meaning                                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `[single article]`    | This chapter has only 1 `.md` file; the "internal sections" below are the H2/H3 headings within that one article |
| `[multiple articles]` | This chapter has multiple `.md` files; each item below is a separate article                                     |
| `📄`                  | Article (one `.md` file)                                                                                         |
| `→ path`              | Actual file location (may include legacy directory names)                                                        |
| `Sub-outline:`        | The H2/H3 points to be expanded within that article                                                              |

---

# Preface · Introduction `[multiple articles]`

> **Design rationale**: This book promises "hands-on first, theory later." But the old `preface/intro.md` opened with Sutton's philosophical discussion of the bitter lesson, and readers didn't touch code until Chapter 1 — the preface itself broke its own promise. v5 fix: the first section of the preface immediately gives an instantly-playable CartPole entry point, so readers see an agent go from wobbling to balanced within 30 seconds; **play first, then explain why**.
>
> **Directory**: `docs/preface/` (3 files covering sections 0.1-0.6)

### Opening Words (covers 0.1-0.4 + 0.6) → `docs/preface/intro.md`

**Internal sections:**

- **0.1 Hands-on first: play with CartPole in 30 seconds** `[v5 new lead-in]`
  - ① One-click try (zero install): dual deployment on ModelScope (primary source) + HuggingFace (mirror)
  - ② One-line local run: `pip install gymnasium[...] stable-baselines3 && python 1-ppo_cartpole.py`
  - ③ Visual preview (offline fallback): training GIF + final policy demo video
- **0.2 A glimpse of what's ahead** `[v5 new]`: DPO refusing malicious requests / R1 reasoning emergence / Computer Use / SWE-Agent fixing bugs
- **0.3 Why we need RL**: Sutton's "The Bitter Lesson" / trial-and-error as the most primitive form of learning / the fundamental difference between supervised learning and decision-making
- **0.4 What is RL**: the agent-environment-state-action-reward loop / trajectory · return · discount / state vs. observation
- **0.6 Book structure and reader roadmap**: 8 progressive Parts / three paths for ML engineers, people with RL background, and students / notation conventions

### A Brief History of Reinforcement Learning (0.5) → `docs/preface/brief-history/index.md`

**Internal sections:**

- 1950s-1980s: trial-and-error learning, the Bellman equation, the birth of TD learning
- 1992: TD-Gammon defeats a human champion for the first time
- 2013: DQN plays Atari — the dawn of deep RL
- 2016: AlphaGo defeats Lee Sedol
- 2017-2019: AlphaGo Zero, MuZero, self-play
- 2017: PPO is published (the algorithm played with in 0.1 is PPO)
- 2022: InstructGPT / RLHF enters large-model training
- 2023-2024: DPO, GRPO, Constitutional AI
- 2025: DeepSeek-R1, o1/o3, the RLVR paradigm is established
- The rise of Chinese labs: Qwen3 GSPO, Step-Audio, DeepSeek's transparency

### Environment Setup Guide (0.7) → `docs/preface/env-setup.md`

**Internal sections:**

- Python environment: conda vs. venv
- PyTorch version and CUDA configuration
- Gymnasium installation and verification
- Preview of the veRL / OpenRLHF / TRL toolchain
- Training hardware checklist: entry-level / core / large-project tiers
- Repository code structure: each chapter's independent subdirectory under `code/`

---

# Part I · Foundations and Classical Reinforcement Learning (7 Chapters)

## Chapter 1 Overview of Reinforcement Learning `[single article]`

📄 File: `chapter00_overview/intro.md`

**Internal sections:**

- 1.1 From the preface's intuition to a formal definition
- 1.2 The core agent-environment-reward-state loop
- 1.3 The modern application landscape: control, games, alignment, agents
- 1.4 The fundamental difference between RL and supervised/unsupervised learning
- 1.5 How this book's later chapters connect

---

## Chapter 2 CartPole: The First Reinforcement Learning Experiment `[multiple articles]`

> Existing directory: `chapter01_cartpole/` (the directory name follows the old numbering and is inconsistent with chapter number 2 — a legacy artifact, not renamed for now)

### 2.1 CartPole Basics and Principles → `chapter01_cartpole/intro.md` + `principles.md`

**Sub-outline:**

- The CartPole problem and the Gym/Gymnasium interface
- Engineering definitions of state, action, and reward
- Random-policy baseline and failure modes

### 2.2 Training Metrics Design → `chapter01_cartpole/metrics.md`

**Sub-outline:**

- Return curves, success rate, stability
- Experiment: the full pipeline from random to converged

---

## Chapter 3 Multi-Armed Bandits and Exploration-Exploitation Theory `[single article]`

📄 File: `chapter03_bandits/intro.md`

**Internal sections:**

- 3.1 The multi-armed bandit problem and basic strategies
- 3.2 ε-greedy and decay schedules
- 3.3 Upper Confidence Bound (UCB) algorithm
- 3.4 Thompson sampling and the Bayesian perspective
- 3.5 Regret bounds and PAC analysis
- 3.6 Contextual bandits

---

# Part IV · Large Language Model Alignment and Post-Training (8 Chapters)

## Chapter 17 LLM RL in Industry `[multiple articles]` `[v5.1 expansion: shifting from PPO to the modern GRPO pipeline]`

> **Design rationale**: By 2025-2026, Llama 4 / Qwen3 / DeepSeek V3.2 / GLM-4.6 have all fully shifted to GRPO/Dr.GRPO + RLVR, and the old version's focus on the classic PPO implementation is outdated.
>
> **Directory**: `chapter17_llm_rl_industrial/` (currently only intro.md); some content is scattered across `chapter09_alignment/` and `chapter09_grpo_rlvr/` and needs to be consolidated

### 17.1 Comparison of Training Frameworks → `chapter17_llm_rl_industrial/01-frameworks.md`

**Sub-outline:**

- Synchronous frameworks: veRL (ByteDance's mainstream choice) / OpenRLHF (open-source-friendly) / TRL (HF ecosystem) / NeMo-Aligner (NVIDIA)
- Asynchronous frameworks: AReaL (Tsinghua + Zhipu) / AgentRL (Zhipu + Tsinghua) / SLIME / ROLL / LlamaRL
- Framework comparison table and a selection decision tree

### 17.2 Modern Post-Training Pipeline Paradigms → `chapter17_llm_rl_industrial/02-pipelines.md`

**Sub-outline:**

- DeepSeek-R1 multi-stage pipeline: cold-start SFT → reasoning RL → rejection sampling → full-scenario RL
- Llama 4: lightweight SFT → online RL → lightweight DPO + pass@k difficulty filtering
- Qwen3: Thinking Mode Fusion + Thinking Budget + GSPO
- GLM-4.5 / 4.6: difficulty-curriculum RL + Hybrid Thinking + RLCS curriculum sampling
- GLM-5 (2026.02): asynchronous Agent RL + DSA sparse attention
- Seed-Thinking-v1.5: dual-track reward + Pre-PPO + hybrid reward

### 17.3 Dual-Track Reward Design → `chapter17_llm_rl_industrial/03-dual-reward.md`

**Sub-outline:**

- Verifiable reward (math, code)
- Pairwise preference reward (open-ended dialogue)
- Pre-PPO: prompt-selection strategies to avoid reward hacking
- Hybrid reward: combining RTV + GenRM

### 17.4 Optimizers and Training Stability → `chapter09_alignment/modern-industrial-practice.md` (reuse existing)

**Sub-outline:**

- AdamW's stability issues in RL training
- The MuonClip optimizer (Kimi K2)
- QK-clip: attention numerical stability
- Early signals of KL explosion and how to handle them

### 17.5 Training Cost Estimation → `chapter17_llm_rl_industrial/05-cost.md`

**Sub-outline:**

- Pretraining costs at different model scales
- GPU-hours for the SFT / RLHF / RLVR stages
- DeepSeek's public data reference: V3 pretraining at 2.664M H800-hours, R1-Zero at 128K GPU-hours
- Budget planning for training your own model

### 17.6 Hands-On in Industry: GSM8K and AIME → `chapter09_grpo_rlvr/verl-code-sandbox.md` (reuse existing)

**Sub-outline:**

- Experiment: training GSM8K with GRPO
- Experiment: training AIME 2024 with DAPO
- Full open-source reproductions: Open-R1 / Sky-T1 / Tülu 3

### 17.7 Common Interview Topics for Chinese Alignment Teams → `chapter17_llm_rl_industrial/07-interview.md`

**Sub-outline:**

- The full derivation chain PG → REINFORCE → TRPO → PPO → GRPO (a real Zhipu interview question)
- The DPO family + DPO regularization
- Engineering comparison: DeepSpeed vs. Megatron
- On-the-spot estimation of training resource consumption
