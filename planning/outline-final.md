# Hands-On Modern Reinforcement Learning — Full v5 Outline (MIT-Level Textbook)

> The final proposal, synthesizing all corrections from v1→v2→v3→v4, based on real evidence (job descriptions from OpenAI/Anthropic/DeepSeek/Qwen/Zhipu/StepFun + 2025-2026 papers).

---

## Design Philosophy

### Why This Structure

1. **A three-stage progression, theory → hands-on → frontier**, in the style of Stanford CS285 + Sutton & Barto + D2L
2. **Each Part maps to one clear learning objective** and can be taught independently
3. **Agentic and hands-on practice are not an appendix — they're core Part V**, reflecting real 2025-2026 industry demand
4. **A uniform structure per chapter**: chapter intro → theory → implementation → experiments → summary → further reading
5. **Real paper citations**: every key claim has an arXiv number or a link to a company technical report

### Comparison With Existing Books

| Dimension           | Sutton & Barto | CS285   | Raschka (in progress) | **This Book, v5**           |
| ------------------- | -------------- | ------- | --------------------- | --------------------------- |
| Classical RL        | ✅ Complete    | ✅      | ❌                    | ✅                          |
| Deep RL             | ❌             | ✅      | ❌                    | ✅                          |
| LLM RL              | ❌             | Partial | ✅                    | ✅ Complete                 |
| Agentic RL          | ❌             | ❌      | ❌                    | **✅ 5 dedicated chapters** |
| Multimodal RL       | ❌             | ❌      | ❌                    | **✅ 3 dedicated chapters** |
| Safety/Alignment    | ❌             | ❌      | ❌                    | **✅ 3 dedicated chapters** |
| Engineering Systems | ❌             | ❌      | Partial               | **✅ 2 dedicated chapters** |

---

# Preface · Introduction (corresponds to docs/preface/)

> **Design rationale**: This book promises "hands-on first, theory later." But the current `preface/intro.md` opens with Sutton's philosophical discussion of the bitter lesson, and readers don't touch code until Chapter 1 — **the preface itself breaks its own promise**. v5 fix: Section 0.1 of the preface immediately gives an instantly-playable CartPole entry point, so readers see an agent go from wobbling to balanced within 30 seconds; **play first, then explain why**.

## 0.1 Hands-on first: play with CartPole in 30 seconds **[v5 new lead-in]**

**Three tiers of experience, dual-source deployment covering both domestic and overseas readers, covering every type of reader:**

**① One-click try (the preferred experience, zero install) — dual-source deployment**

- **Primary source: ModelScope Creative Space**: `spaces.modelscope.cn/{namespace}/cartpole-playground`
  - Stable access within mainland China; this is the default for this book's primary audience (Chinese-language readers)
  - A Gradio app, nearly identical in code to the HF Space
- **Mirror source: HuggingFace Space**: `hf.co/spaces/{namespace}/cartpole-playground`
  - For overseas readers / mirror-site visitors
- **On-page presentation**: two iframe tabs to switch between "🇨🇳 Domestic entry point / 🌍 Overseas entry point"
  - ModelScope (primary source) shown by default
  - Automatic fallback prompt on load failure, guiding readers to switch
- Experience: click the "Train" button → watch the reward curve climb in real time → the final policy plays automatically once training finishes

**② One-line local run (a deeper option)**

- A prominent code box: `pip install "gymnasium[classic-control]" stable-baselines3 && python 1-ppo_cartpole.py`
- CPU training finishes in 30 seconds, popping up a `--gui` cart demo window
- Links into the full code at `code/chapter01_cartpole/`
- For mainland readers: `pip install` uses the Tsinghua/Alibaba mirror, with configuration instructions given in the docs

**③ Visual preview (an offline fallback)**

- A training-process GIF: an animation of the reward curve climbing from 20 to 500 (self-hosted in this book's repository at `docs/preface/images/`)
- A final-policy demo video: the agent going from wobbly to stably balanced
- The lowest-barrier experience, for when both sources fail to load, readers who don't want to install anything, or offline readers

**Promise fulfilled**: no matter which tier a reader chooses, and no matter their region, they have now "seen" an agent learn something — everything from 0.2 through 0.6 is a look back to explain what just happened.

---

### Bridging classic and modern: the **"A Glimpse of What's Ahead"** section **[v5 new]**

> CartPole is RL's past (a classic task dating from the 1990s); this book's true protagonist is modern RL in the LLM era. At the end of the preface, 3-5 short video clips give a glimpse of **what you'll be able to do after finishing this book** — the classic entry point is already playable, and the modern entry point gets a preview first.

**Teaser 1: DPO teaches a large model to stop blindly obeying the user**

- A before-vs-after training conversation-comparison GIF (user asks for malicious code → complies before training / politely declines after training)
- Leads into Chapter 18, the DPO family

**Teaser 2: The emergence of DeepSeek-R1's reasoning ability**

- A video of R1-Zero's pure-RL training spontaneously lengthening its CoT
- Leads into Chapter 19 (the GRPO family) + Chapter 20 (Reasoning Models)

**Teaser 3: A Computer Use agent operating a browser**

- Public demo videos of Claude Computer Use / OpenAI Operator
- Leads into Chapter 28, Computer Use and GUI Agents

**Teaser 4: SWE-Agent autonomously fixing a bug**

- The full pipeline of an agent on SWE-bench: reading code → localizing the bug → making the fix → passing the tests
- Leads into Chapter 26, Code Agent Reinforcement Learning

**Presentation**: all GIFs/videos self-hosted under `docs/preface/images/teasers/`, avoiding broken external links. Readers see the book's destination up front, without the preface forcing them into a hard requirement of being able to run an LLM (too high a barrier).

## 0.2 Opening Words: Why We Need RL (formerly "Why We Need Reinforcement Learning" in `preface/intro.md`)

- Sutton's "The Bitter Lesson" and the two main threads of 70 years of AI history: search and learning
- Why trial-and-error is the most primitive form of learning: the bicycle-riding analogy
- From recognition to decision-making: why supervised learning cannot cover sequential decisions
- What RL provides: it doesn't tell you how to do something, only what's good and what's not
- Bridge to 0.1: the "trial-and-error → convergence" you just watched on CartPole is a concrete instance of this section

## 0.3 What Is RL: The Core Loop and Key Terminology (formerly "What Is Reinforcement Learning" in `preface/intro.md`)

- The agent-environment-state-action-reward loop
- Trajectory, return, discount factor $\gamma$
- State vs. observation, discrete vs. continuous action spaces
- Bridge to Chapter 1: redescribe the 0.1 CartPole example using this terminology

## 0.4 A Brief History of Reinforcement Learning (formerly `preface/brief-history/`)

- 1950s–1980s: trial-and-error learning, the Bellman equation, the birth of TD learning
- 1992: TD-Gammon — the first RL system to defeat a human champion
- 2013: DQN plays Atari — the dawn of deep RL
- 2016: AlphaGo defeats Lee Sedol
- 2017–2019: AlphaGo Zero, MuZero, self-play
- 2017: PPO is published and becomes the industry standard (the algorithm you just used in 0.1 is PPO)
- 2022: InstructGPT / RLHF enters large-model training
- 2023–2024: DPO, GRPO, Constitutional AI
- 2025: DeepSeek-R1, o1/o3, the RLVR paradigm is established
- The rise of Chinese labs: Qwen3 GSPO, Step-Audio, DeepSeek's transparency

## 0.5 Environment Setup Guide (formerly `preface/env-setup.md`)

- Python environment: conda vs. venv
- PyTorch version and CUDA configuration
- Gymnasium installation and verification
- Preview of the veRL / OpenRLHF / TRL toolchain
- Training hardware checklist: entry-level experiments / core experiments / large-project tiers
- Repository code structure: each chapter's independent subdirectory under `code/`

## 0.6 Book Structure and Reader Roadmap (formerly "About This Book" in `preface/intro.md`)

- The progressive logic across the book's 8 Parts: Foundations → Deep → Advanced → LLM → Agentic → Multimodal → Safety → Frontier
- Recommended paths for three types of readers:
  - ML engineers: 0.1 → Part IV–V (LLM + Agentic)
  - Readers with an RL background: Part II–III + Part IV
  - Students: work through Part I onward in sequence
- Notation conventions and symbol table (see Appendix H for details)

---

# Part I · Foundations and Classical Reinforcement Learning (7 Chapters)

## Chapter 1 Overview of Reinforcement Learning

- 1.1 From the preface's intuition to a formal definition
- 1.2 The core agent-environment-reward-state loop
- 1.3 The modern application landscape: control, games, alignment, agents
- 1.4 The fundamental difference between RL and supervised/unsupervised learning
- 1.5 How this book's later chapters connect

## Chapter 2 CartPole: The First Reinforcement Learning Experiment

- 2.1 The CartPole problem and the Gym/Gymnasium interface
- 2.2 Engineering definitions of state, action, and reward
- 2.3 Random-policy baseline and failure modes
- 2.4 Training metrics design: return curves, success rate, stability
- 2.5 Experiment: the full pipeline from random to converged

## Chapter 3 Multi-Armed Bandits and Exploration-Exploitation Theory

- 3.1 The multi-armed bandit problem and basic strategies
- 3.2 ε-greedy and decay schedules
- 3.3 Upper Confidence Bound (UCB) algorithm
- 3.4 Thompson sampling and the Bayesian perspective
- 3.5 Regret bounds and PAC analysis **[theory]**
- 3.6 Contextual bandits **[groundwork for the alignment chapters]**

## Chapter 4 Markov Decision Processes

- 4.1 From bandits to sequential decision-making
- 4.2 The mathematical definition of the Markov property, and its intuition
- 4.3 State space, action space, transition function, reward function
- 4.4 Discount factor and return
- 4.5 Trajectories and episodes
- 4.6 Partially Observable MDPs (POMDPs) **[groundwork for LLM multi-turn interaction]**

## Chapter 5 Value Functions and the Bellman Equation

- 5.1 The state-value function $V^\pi(s)$
- 5.2 The action-value function $Q^\pi(s,a)$
- 5.3 The Bellman expectation equation
- 5.4 The Bellman optimality equation
- 5.5 The contraction-mapping property of the Bellman operator **[theory]**
- 5.6 Existence and uniqueness of the optimal policy

## Chapter 6 Dynamic Programming, Monte Carlo, and Temporal Difference Learning

- 6.1 Dynamic programming: policy evaluation and policy improvement
- 6.2 Policy iteration and value iteration
- 6.3 Monte Carlo methods: first-visit and every-visit
- 6.4 Temporal difference (TD) learning: TD(0)
- 6.5 n-step bootstrapping and TD(λ)
- 6.6 Eligibility traces
- 6.7 Comparing and weighing the three families of methods

## Chapter 7 Q-Learning and Off-Policy Control

- 7.1 On-policy vs. off-policy
- 7.2 The Q-Learning algorithm and its convergence
- 7.3 The SARSA algorithm
- 7.4 Importance sampling **[key groundwork]**
- 7.5 The challenges of function approximation and the Deadly Triad **[theory]**
- 7.6 Reward function design: sparse vs. dense, shaping, and hacking

---

# Part II · Deep Reinforcement Learning (5 Chapters)

## Chapter 8 Deep Q-Networks and Distributional RL

- 8.1 Motivation for moving from Q-Learning to DQN
- 8.2 Experience Replay
- 8.3 The Target Network
- 8.4 Double DQN: addressing overestimation
- 8.5 Dueling DQN: decomposing state-action value
- 8.6 Prioritized Experience Replay (PER)
- 8.7 Distributional RL: C51, QR-DQN, IQN
- 8.8 Rainbow and NoisyNet
- 8.9 Experiments: LunarLander and Atari

## Chapter 9 Policy Gradient Methods

- 9.1 Motivation for policy gradient methods (continuous actions, stochastic policies)
- 9.2 Policy representations: softmax, Gaussian, categorical
- 9.3 The policy gradient theorem (full derivation) **[theory]**
- 9.4 The REINFORCE algorithm
- 9.5 The variance problem and baselines
- 9.6 Off-policy policy gradients
- 9.7 Experiments: CartPole and Pendulum

## Chapter 10 Actor-Critic Architectures

- 10.1 The advantage function $A(s,a) = Q(s,a) - V(s)$
- 10.2 Training the critic network (fitting the value function)
- 10.3 The Actor-Critic framework and synchronized updates
- 10.4 Generalized Advantage Estimation (GAE) **[groundwork for PPO]**
- 10.5 A2C and A3C: asynchronous parallelism
- 10.6 Experiments: Pendulum and BipedalWalker

## Chapter 11 PPO and Trust-Region Methods

- 11.1 The stability problem in policy updates
- 11.2 TRPO and the monotonic improvement theorem **[theory]**
- 11.3 The PPO-Clip algorithm
- 11.4 PPO-Penalty and adaptive KL
- 11.5 Engineering details of PPO implementation (entropy bonus, value clipping)
- 11.6 PPO in long-horizon tasks
- 11.7 PPO's place in the LLM-RL era (background; see Chapter 19, the GRPO family, for detail)
- 11.8 Experiment: BipedalWalker continuous control

## Chapter 12 Continuous Control and Model-Based Deep RL

- 12.1 Deterministic Policy Gradients (DPG)
- 12.2 The DDPG algorithm
- 12.3 TD3: target policy smoothing and twin Q-networks
- 12.4 Soft Actor-Critic (SAC) and maximum-entropy RL **[theory]**
- 12.5 Model-based RL: Dyna, PETS, MBPO
- 12.6 AlphaZero and MuZero
- 12.7 Dreamer V3 and World Models
- 12.8 The model-based vs. model-free trade-off

---

# Part III · Advanced RL Methods (3 Chapters, Trimmed but Deep)

## Chapter 13 Offline Reinforcement Learning and Decision Transformers

- 13.1 The challenge of offline RL: distribution shift
- 13.2 The pessimism behind CQL, IQL, and BCQ
- 13.3 AWAC and TD3+BC
- 13.4 Decision Transformer: RL as sequence modeling
- 13.5 Trajectory Transformer and Diffuser
- 13.6 Offline RL in the LLM era

## Chapter 14 Imitation Learning, Inverse RL, and Meta-RL

- 14.1 Behavior cloning (BC) and DAgger
- 14.2 Dataset Aggregation (DAgger)
- 14.3 Maximum-entropy inverse RL (MaxEnt IRL)
- 14.4 GAIL: Generative Adversarial Imitation Learning
- 14.5 Meta-RL: MAML, RL², PEARL
- 14.6 In-context RL and Algorithm Distillation **[DeepMind 2022]**

## Chapter 15 Exploration, Multi-Agent RL, and Hierarchical RL

- 15.1 The fundamental tension between exploration and exploitation (theoretical review)
- 15.2 Intrinsic Curiosity Modules (ICM) and Random Network Distillation (RND)
- 15.3 NGU and Agent57
- 15.4 Multi-agent RL: the CTDE framework
- 15.5 MADDPG and MAPPO
- 15.6 Hierarchical RL: Options, FeUdal Networks, HIRO
- 15.7 **Generative world models as RL environments** (a preview of Genie 3; see Chapter 37 for detail)

---

# Part IV · Large Language Model Alignment and Post-Training (8 Chapters)

## Chapter 16 The RLHF Training Pipeline **[v5.1 expansion]**

- 16.1 Base models and instruction alignment
- 16.2 The modern three-stage paradigm: SFT → RLHF → RLVR
- 16.3 SFT instruction tuning
- 16.4 Reward modeling: the Bradley-Terry model
- 16.5 RL fine-tuning (PPO or GRPO)
- 16.6 KL constraints and the reference policy
- 16.7 **Dual-track reward design** (Seed-Thinking: verifiable + pairwise)
- 16.8 **Pre-PPO**: prompt-selection strategies to avoid reward hacking
- 16.9 Tülu 3: an open-source reference for the three-stage paradigm

## Chapter 17 LLM RL in Industry **[v5.1 expansion: shifting from PPO to the modern GRPO pipeline]**

> **Design rationale**: Agent #2 explicitly flagged that v5's focus on the classic PPO implementation in this chapter is outdated. By 2025-2026, Llama 4 / Qwen3 / DeepSeek V3.2 / GLM-4.6 have all fully shifted to GRPO/Dr.GRPO + RLVR.

### 17.1 Comparison of Training Frameworks

- 17.1.1 veRL (ByteDance, mainstream)
- 17.1.2 OpenRLHF (open-source-friendly)
- 17.1.3 TRL (the HuggingFace ecosystem)
- 17.1.4 NeMo-Aligner (NVIDIA)
- 17.1.5 AReaL (Tsinghua + Zhipu, asynchronous)
- 17.1.6 AgentRL (Zhipu + Tsinghua)
- 17.1.7 Comparing SLIME / ROLL / LlamaRL

### 17.2 Modern Post-Training Pipeline Paradigms

- 17.2.1 **DeepSeek-R1's multi-stage pipeline**: cold-start SFT → reasoning RL → rejection sampling → full-scenario RL
- 17.2.2 **The Llama 4 pipeline**: lightweight SFT → online RL → lightweight DPO + pass@k difficulty filtering (arXiv:2504.13914, the Llama 4 tech report)
- 17.2.3 **The Qwen3 pipeline**: Thinking Mode Fusion + Thinking Budget + GSPO (arXiv:2505.09388)
- 17.2.4 **The GLM-4.5 / 4.6 pipeline**: difficulty-curriculum RL + Hybrid Thinking + RLCS curriculum sampling (arXiv:2508.06471)
- 17.2.5 **GLM-5** (2026.02, arXiv:2602.15763): new asynchronous Agent RL + DSA sparse attention + 744B/28.5T
- 17.2.6 **Seed-Thinking-v1.5**: dual-track reward (verifiable + pairwise) + Pre-PPO + hybrid reward (RTV+GenRM) (arXiv:2504.13914)

### 17.3 Dual-Track Reward Design

- 17.3.1 Verifiable reward (math, code)
- 17.3.2 Pairwise preference reward (open-ended dialogue)
- 17.3.3 Pre-PPO: prompt-selection strategies to avoid reward hacking
- 17.3.4 Hybrid reward: combining RTV + GenRM

### 17.4 Optimizers and Training Stability

- 17.4.1 AdamW's stability issues in RL training
- 17.4.2 **The MuonClip optimizer** (Kimi K2, arXiv:2507.20534 §3.2)
- 17.4.3 QK-clip: attention numerical stability
- 17.4.4 Early signals of KL explosion and how to handle them

### 17.5 Training Cost Estimation

- 17.5.1 Pretraining costs at different model scales
- 17.5.2 GPU-hours for the SFT / RLHF / RLVR stages
- 17.5.3 DeepSeek's public data reference: V3 pretraining at 2.664M H800-hours, R1-Zero at 128K GPU-hours
- 17.5.4 Budget planning for training your own model

### 17.6 Hands-On in Industry: GSM8K and AIME

- 17.6.1 Experiment: training GSM8K with GRPO
- 17.6.2 Experiment: training AIME 2024 with DAPO
- 17.6.3 Full open-source reproductions: Open-R1 / Sky-T1 / Tülu 3

### 17.7 Common Interview Topics for Chinese Alignment Teams

- 17.7.1 The full derivation chain PG → REINFORCE → TRPO → PPO → GRPO (a real Zhipu interview question)
- 17.7.2 The DPO family + DPO regularization
- 17.7.3 Engineering comparison: DeepSpeed vs. Megatron
- 17.7.4 On-the-spot estimation of training resource consumption

## Chapter 18 Preference Alignment: The DPO Family **[v5.1 expansion]**

- 18.1 The mathematical derivation of DPO (deriving it from the RLHF objective)
- 18.2 Analyzing DPO training dynamics
- 18.3 IPO: fixing DPO's overfitting
- 18.4 KTO: no need for paired preference data
- 18.5 SimPO: a reference-free method
- 18.6 **DPO regularization methods** (a real Zhipu interview question)
- 18.7 Iterative DPO and ReST
- 18.8 Self-Play Fine-Tuning (SPIN)
- 18.9 A decision tree for choosing within the DPO family

## Chapter 19 The GRPO Family: From Group Normalization to the Full Improvement Lineage **[v5.1 full restructure]**

> **Design rationale**: the biggest algorithmic focus of 2025-2026. Four independent research efforts (OpenAI/Anthropic, DeepMind/Meta, DeepSeek/Qwen/Kimi, Zhipu/StepFun/ByteDance/MiniMax) consistently pointed out that v5's version of this chapter only lists names with no algorithmic detail. v5.1 reorganizes it by direction of improvement, covering the algorithmic differences across 6+ mainstream variants.

### 19.1 GRPO Fundamentals

- 19.1.1 From PPO to GRPO: why drop the critic
- 19.1.2 The principle of group normalization: relative advantage across multiple rollouts of the same prompt
- 19.1.3 KL constraints and the reference-policy implementation

### 19.2 Improvement Direction A: Fixing the Normalization Bias

- 19.2.1 **Dr.GRPO** (Liu et al. 2025, arXiv:2508.10355): removes std normalization and length normalization to avoid reward hacking and length inflation
- 19.2.2 DeepSeek V3.2's KL tuning: zero KL for math tasks, self-verifying RLVR, mHC residual stability (arXiv:2512.02556)

### 19.3 Improvement Direction B: Sequence-Level Importance Sampling

- 19.3.1 **GSPO** (Zheng et al. 2025, Qwen3, arXiv:2507.18071): sequence-level IS ratio + sequence-level clipping, designed specifically for stable MoE RL training
- 19.3.2 Empirical benefits from the entire Qwen3 line's adoption of GSPO

### 19.4 Improvement Direction C: Comparing Clipping Strategies

- 19.4.1 **DAPO** (ByteDance + Tsinghua, 2025.03, arXiv:2503.14476, NeurIPS 2025):
  - Clip-Higher: decoupling $\epsilon_{low} \neq \epsilon_{high}$
  - Dynamic Sampling: filtering out all-correct/all-wrong samples
  - Token-level Loss: avoiding domination by long responses
  - Overlong Filtering + Soft Shaping
- 19.4.2 **CISPO** (MiniMax 2025.06, arXiv:2506.13585):
  - Clipping the IS weight rather than the token update
  - Preserving all token gradients, avoiding token loss
  - Precision alignment under lightning attention
  - 2× speedup vs. DAPO
- 19.4.3 Comparing DAPO vs. CISPO for selection

### 19.5 Improvement Direction D: The Value-Based Counter-Trend

- 19.5.1 **VAPO** (ByteDance Seed 2025.04, arXiv:2504.05118):
  - Value-based Augmented PPO
  - In long-CoT settings, a value model once again beats GRPO
  - AIME 60.4 (surpassing contemporary GRPO variants)
- 19.5.2 What VAPO reveals: critic-free isn't the only answer

### 19.6 Other Industrial Variants

- 19.6.1 REINFORCE++ (Hu 2025)
- 19.6.2 AREAL (an asynchronous RL framework, Tsinghua + Zhipu)
- 19.6.3 Niche variants such as ASPO / DCPO

### 19.7 RLVR: The Verifiable-Reward Paradigm

- 19.7.1 Defining RLVR: rule-based feedback in place of human annotation
- 19.7.2 Sources of reward in RLVR: math verifiers, unit tests, formal proofs
- 19.7.3 Hybrid pipelines combining RLVR and RLHF

### 19.8 RL at the Pretraining Stage: Reinforcement Pre-Training **[v5.1 entirely new concept]**

- 19.8.1 RPT (Microsoft 2025.06, arXiv:2506.08007): reframes next-token prediction as a reasoning task with an intrinsic binary reward
- 19.8.2 RPT's scalability rivals pretraining: challenging the pretraining/post-training dichotomy
- 19.8.3 The biggest conceptual shift of 2025: RL is no longer just post-training

### 19.9 Hands-On Comparison of Chinese-Lab Approaches

- 19.9.1 The full DeepSeek-R1 training pipeline: cold-start SFT → reasoning RL → rejection sampling → full-scenario RL
- 19.9.2 Qwen3's GSPO + Thinking Mode Fusion
- 19.9.3 MiniMax M1's CISPO + Lightning Attention
- 19.9.4 ByteDance Seed's dual line of DAPO + VAPO
- 19.9.5 Kimi K2's MuonClip + QK-clip

### 19.10 A Decision Tree for Choosing a Variant

- A mapping table from task type to recommended algorithm
- The three-way trade-off among GPU memory / training time / convergence stability

## Chapter 20 Reasoning Models: From o1 to Claude Opus 4.6 **[v5.1 expansion]**

### 20.1 The Rise of Reasoning Models

- 20.1.1 The progression from OpenAI o1 → o3 → o4
- 20.1.2 **Competitive Programming with Large Reasoning Models** (OpenAI 2025.02, arXiv:2502.06807):
  - End-to-end general RL outperforms domain-specific pipelines on IOI/Codeforces
  - **Complex test-time reasoning strategies emerge naturally from end-to-end RL**, rather than being hand-designed
- 20.1.3 Empirical evidence for reasoning ability as an "emergent phenomenon"

### 20.2 The R1-Zero Paradigm: Pure RL With No SFT

- 20.2.1 **DeepSeek-R1-Zero** (Nature 2025, nature.com/articles/s41586-025-09422-z):
  - Running RL directly on the base model, with no intermediate SFT stage
  - Reflection, verification, and "aha moments" emerge spontaneously
- 20.2.2 Open-source, industrial-grade counterparts to the R1-Zero paradigm:
  - **DAPO** (ByteDance + Tsinghua): surpasses R1-Zero on AIME 2024 using 50% of the training steps
  - **VAPO** (ByteDance Seed): the value-based counter-trend approach
  - **Qwen3**: Thinking Mode Fusion + GSPO
- 20.2.3 The complete DeepSeek-R1 training pipeline (cold start + reasoning RL + rejection sampling + full-scenario RL)

### 20.3 Test-Time Compute Scaling

- 20.3.1 The trade-off between test-time compute and train-time compute
- 20.3.2 **Gemini 3 Pro Deep Think** (2025.10) / **3.1 Deep Think** (2026.02):
  - Parallel-reasoning "thinking layers" stacked on top of an MoE
  - IMO 2025 gold medal, 48.4% on HLE, 84.6% on ARC-AGI-2
- 20.3.3 Deep Think as a flagship case study for test-time scaling

### 20.4 Hybrid Thinking and Thinking Budgets **[v5.1 new]**

- 20.4.1 A single model that supports both think and non-think modes
- 20.4.2 **DeepSeek V3.1** (2025.08): fusing hybrid modes
- 20.4.3 **Qwen3** (arXiv:2505.09388 §4.3): Thinking Mode Fusion + Thinking Budget
- 20.4.4 **NoThinking + Best-of-N**: matching thinking-level performance without thinking (Ma et al., arXiv:2505.18681)
- 20.4.5 The engineering implementation of Thinking Budget for controlling reasoning depth

### 20.5 Long-CoT Compression **[v5.1 new]**

- 20.5.1 **Kimi k1.5's long2short RL** (arXiv:2501.12599 §2.4, §3.4):
  - An RL method for distilling long CoT into short CoT
  - Length-penalty control
- 20.5.2 Balancing reasoning efficiency against reasoning quality

### 20.6 Hidden CoT vs. Visible CoT

- 20.6.1 The engineering motivation behind OpenAI o1/o3's Hidden CoT
- 20.6.2 DeepSeek-R1's open strategy of Visible CoT
- 20.6.3 The trade-off between CoT readability and reasoning ability

### 20.7 Adaptive Thinking

- 20.7.1 **Claude Opus 4.6**'s adaptive thinking depth
- 20.7.2 Opus 4.6's internal AI Research Eval Suite (LLM training / Text-RL / Quadruped-RL subtasks, 34x human speedup)
- 20.7.3 Anthropic's 2026, 80-page Constitution and reasoning ability

### 20.8 The Readability and Alignment of Reasoning Chains

- 20.8.1 Reasoning alignment
- 20.8.2 Safety filtering of reasoning chains
- 20.8.3 Potential deception within Hidden CoT

## Chapter 21 Process Reward Models and Inference-Time Search **[v5.1 full restructure]**

> **Design rationale**: three independent research efforts (OpenAI/Anthropic, DeepMind/Meta, DeepSeek/Qwen/Kimi) all pointed out that v5's version of this chapter still centers on discriminative PRM, missing the two new main lines of generative and formal approaches.

### 21.1 Outcome Reward vs. Process Reward

- 21.1.1 The sparsity problem with outcome reward
- 21.1.2 The fine-grained advantage of process reward
- 21.1.3 Why PRM is indispensable for long-CoT tasks

### 21.2 Discriminative PRM (the Classic Approach)

- 21.2.1 OpenAI's "Let's Verify Step by Step" (Lightman et al. 2023, arXiv:2305.20050)
- 21.2.2 The PRM800K dataset and human annotation
- 21.2.3 PRM as a re-ranking model
- 21.2.4 Limitations: high annotation cost, weak generalization

### 21.3 Generative PRM (a New Approach)

- 21.3.1 ThinkPRM (arXiv:2504.16828): generative PRM outperforms discriminative PRM
- 21.3.2 The key to 100x fewer labels: letting the verifier generate its own critique
- 21.3.3 Verifier Compute Scaling
- 21.3.4 A PRM survey (arXiv:2510.08049): comparing generative and discriminative approaches

### 21.4 Formal PRM (the Ultimate Verifier)

- 21.4.1 Lean4 / Coq as a natural verifier: zero false positives
- 21.4.2 **AlphaProof** (DeepMind 2024.07, IMO silver medal):
  - The AlphaZero algorithm + the Lean formal language
  - Self-training on millions of problems
  - Self-play proving
- 21.4.3 **AlphaGeometry 2** (DeepMind): a dedicated formal approach for geometry problems
- 21.4.4 **DeepSeek-Prover-V2** (2025.04, arXiv:2504.21801):
  - Formal theorem proving in Lean4 + RL with binary reward
  - 88.9% on MiniF2F
- 21.4.5 The cost of formal PRM: scarce formal-language data, limited domain coverage

### 21.5 Inference-Time Search

- 21.5.1 Beam Search over Thoughts
- 21.5.2 MCTS over Thoughts: tree expansion
- 21.5.3 Tree of Thoughts (ToT)
- 21.5.4 AlphaCodium: search for code generation
- 21.5.5 rStar: self-play search

### 21.6 Parallel Coordinated Reasoning (PaCoRe)

- 21.6.1 **PaCoRe** (Step3-VL-10B, ACL 2026, github.com/stepfun-ai/PaCoRe)
- 21.6.2 16-way parallel rollout aggregation
- 21.6.3 Outcome-based RL for training reasoning synthesis
- 21.6.4 Shifting test-time compute scaling from depth expansion to parallel-breadth expansion
- 21.6.5 AIME 2025: 94.4
- 21.6.6 Comparing PaCoRe vs. DeepThink vs. MCTS

### 21.7 GenRM and Verifier Models

- 21.7.1 The Generative Reward Model: turning verification into generation
- 21.7.2 The LLM-as-Judge paradigm
- 21.7.3 Self-Rewarding Language Models

## Chapter 22 Constitutional AI and RLAIF

- 22.1 The Constitutional AI framework (Anthropic 2022)
- 22.2 RLAIF: replacing human annotation with AI feedback
- 22.3 Self-correction and self-rewarding
- 22.4 The HHH alignment principles: Helpful, Harmless, Honest
- 22.5 CAI's actual use in training Claude
- 22.6 Anthropic's 2026, 80-page Constitution

## Chapter 23 Designing RL Environments and Verifiers **[v5.1 expansion]**

> **Design rationale**: three independent research efforts all flagged that v5's version of this chapter is missing asynchronous RL training systems, and that Appendix B's focus on synchronous veRL alone is outdated.

### 23.1 RL Environments as the New Bottleneck

- 23.1.1 Anthropic's $1B investment (The Information, 2025.09)
- 23.1.2 Wing VC data: Anthropic spends tens of millions of dollars per year, expanding 3-5x by 2026 (per a wing.vc report)
- 23.1.3 Karpathy: "RLVR is the new major stage of the LLM training pipeline"
- 23.1.4 Mechanize paying RL environment engineers $500K/year

### 23.2 The Equivalence Between Evals and RL Environments

- 23.2.1 Evals = RL Environments (Pash 2025)
- 23.2.2 Evaluation is training, and training is evaluation

### 23.3 Verifier Design Principles

- 23.3.1 Correctness
- 23.3.2 Efficiency
- 23.3.3 Anti-gaming
- 23.3.4 Formal verifiers vs. heuristic verifiers

### 23.4 Sandbox Engineering

- 23.4.1 Docker container isolation
- 23.4.2 Code-execution sandboxes
- 23.4.3 Network allowlisting and resource quotas
- 23.4.4 Managing multi-agent parallel sandboxes

### 23.5 Long-Horizon Task Harnesses

- 23.5.1 **Anthropic's Effective Harnesses** (2025.11):
  - The initializer-agent + incremental-coding-agent pattern
  - `claude-progress.txt` progress tracking
  - The `feature_list.json` state file
  - The test-ratchet pattern
- 23.5.2 Karpathy's "5-6 agents" pattern

### 23.6 Synchronous vs. Asynchronous RL Training **[v5.1 new]**

- 23.6.1 Synchronous RL training (the traditional mode of veRL, TRL, OpenRLHF)
- 23.6.2 The motivation for asynchronous RL training: decoupling rollout from training
- 23.6.3 **AReaL** (Tsinghua + Zhipu, arXiv:2505.24298, NeurIPS 2025):
  - Fully asynchronous rollout-training decoupling
  - Staleness-enhanced PPO
  - 2.77x speedup
- 23.6.4 **AgentRL** (Zhipu + Tsinghua, arXiv:2510.04206):
  - Cross-policy sampling
  - Task advantage normalization
- 23.6.5 **SLIME / ROLL / LlamaRL / PRIME-RL** (distributed asynchronous RL)
- 23.6.6 TOPLOC + SHARDCAST distributed coordination (INTELLECT-2)
- 23.6.7 Engineering implementations of staleness and cross-policy sampling

### 23.7 Evaluation Benchmarks

- 23.7.1 CyberGym (arXiv:2506.02548)
- 23.7.2 SWE-bench (Live/Verified/Multimodal)
- 23.7.3 Terminal-Bench
- 23.7.4 **τ-bench** (Salesforce): multi-turn tool calling
- 23.7.5 **BFCL** (Berkeley Function Calling Leaderboard)
- 23.7.6 WebArena / VisualWebArena
- 23.7.7 Vending-Bench (arXiv:2502.15840)
- 23.7.8 BrowseComp / xbench-DeepSearch

### 23.8 Engineering the Train-Evaluate Loop

- 23.8.1 Eval-driven RL training
- 23.8.2 Incremental evaluation: sampled evaluation every N steps
- 23.8.3 Data-contamination detection (see Chapter 33 for detail)

---

# Part V · Agentic Reinforcement Learning (5 Chapters, **the Core New Addition in v5**)

> **Design rationale**: real 2025-2026 industry demand concentrates here. 60% of the Anthropic Code RL job description centers on agentic work, and OpenAI Operator, Claude Computer Use, and SWE-Agent are all evolving rapidly. A single chapter, as in the original book, is far from enough.

## Chapter 24 Multi-Turn Interactive Reinforcement Learning

- 24.1 From single-turn to multi-turn: credit assignment across a trajectory
- 24.2 Modeling multi-turn interaction as an MDP
- 24.3 Designing user simulators
- 24.4 Reward design for long-horizon tasks
- 24.5 Engineering differences between multi-turn and single-turn RL
- 24.6 Experiment: training an RL agent for multi-turn dialogue

## Chapter 25 Tool Use and Function-Calling RL

- 25.1 Expanding the action space for tool use
- 25.2 Modeling function-calling trajectories
- 25.3 Designing tool reward: execution outcome + call appropriateness
- 25.4 The ReAct / ToolFormer paradigms
- 25.5 Search-augmented RL: Search-R1, R1-Searcher
- 25.6 Code Interpreter RL: SimpleTIR, ReTool, AFM
- 25.7 Experiment: training GRPO for tool calling

## Chapter 26 Code Agent Reinforcement Learning **[v5.1 restructure: RL-based as the main line]**

> **Design rationale**: Agent #2 (DeepMind/Meta) explicitly flagged that v5's version of this chapter, still using an SFT-only SWE-agent as its example, is outdated. The 2025-2026 main line is RL-based SWE (SWE-RL → CWM → DeepSWE → SSR).

### 26.1 Task Definition and Benchmarks

- 26.1.1 SWE-bench: the standard software-engineering task set (Live/Verified)
- 26.1.2 SWE-bench-Lite / SWE-bench Multimodal
- 26.1.3 Evaluation metrics: Resolved %, Pass@k, edit distance

### 26.2 First Generation: SFT-Based SWE Agents (Background)

- 26.2.1 SWE-Gym and training-data generation
- 26.2.2 SWE-Smith: large-scale data synthesis
- 26.2.3 Limitation: SFT can't learn long-horizon self-correction ability

### 26.3 Second Generation: RL-Based SWE (the Main Line)

- 26.3.1 **SWE-RL** (Meta 2025.02, arXiv:2502.18449, NeurIPS'25):
  - 11M GitHub PRs + rule-based reward
  - Llama3-70B reaches 41% on SWE-bench Verified
  - **The first observation of an "aha moment"**
- 26.3.2 rLLM / DeepCoder: industrial-grade RL-based SWE agents
- 26.3.3 Implementation details: parallel envs, reward shaping, context management

### 26.4 Third Generation: Code World Models

- 26.4.1 **Code World Model (CWM)** (Meta 2025.09, arXiv:2510.02387):
  - 32B dense
  - Python execution-trace mid-training
  - GRPO post-training
  - 65.8% on SWE-bench
- 26.4.2 DeepSWE (Luo et al. 2025)
- 26.4.3 The world-model paradigm: the agent learns to "predict the outcome of executing code"

### 26.5 Fourth Generation: The Self-Play Paradigm

- 26.5.1 **Self-Play SWE-RL (SSR)** (Meta 2025.12, arXiv:2512.18552):
  - A single policy playing dual roles (bug-injector + bug-solver)
  - No human-written issue descriptions needed
  - +10.4 on SWE-bench Verified
- 26.5.2 A self-generating training-data flywheel

### 26.6 Designing Code Verifiers

- 26.6.1 Unit tests as the reward signal
- 26.6.2 Code repair as process reward
- 26.6.3 Process reward models for code
- 26.6.4 **SWE-RM** (an execution-free reward model, arXiv:2512.21919)
- 26.6.5 Hybrid reward: rule + model

### 26.7 Long-Horizon Autonomous Engineering Ability

- 26.7.1 Anthropic's Effective Harnesses for coding agents
- 26.7.2 Progress tracking (`claude-progress.txt`, `feature_list.json`)
- 26.7.3 The test-ratchet pattern
- 26.7.4 The initializer-agent + incremental-coding-agent pattern

### 26.8 Code Agents From Chinese Labs

- 26.8.1 **Qwen3-Coder** (2025.07): long-horizon agent RL, 20,000 parallel environments
- 26.8.2 DeepSeek Coder's training practice
- 26.8.3 Zhipu's CodeGeeX training methodology

### 26.9 Experiments

- 26.9.1 Training an open-source code agent using the SWE-RL algorithm
- 26.9.2 Reproducing a 30-40% baseline on SWE-bench Verified

## Chapter 27 Deep Research and Web Agents

- 27.1 Defining the Deep Research task
- 27.2 Multi-step retrieval and information aggregation
- 27.3 Browser-agent RL
- 27.4 Anthropic's Effective Harnesses for Long-Running Agents
- 27.5 The BrowseComp / xbench-DeepSearch benchmarks
- 27.6 Financial QA and open-domain QA agents
- 27.7 Experiment: training a Deep Research agent

## Chapter 28 Computer Use and Multi-Agent Collaboration **[v5.1 expansion]**

### 28.1 The Computer Use Paradigm

- 28.1.1 Anthropic Computer Use
- 28.1.2 OpenAI Operator
- 28.1.3 Google Project Mariner
- 28.1.4 The core action space of Computer Use (clicking, scrolling, typing, screenshots)

### 28.2 GUI Grounding RL

- 28.2.1 Locating screen elements (Set-of-Mark, visual grounding)
- 28.2.2 Action mapping: from pixels to mouse/keyboard events
- 28.2.3 Aligning visual understanding with action

### 28.3 Hands-On GUI Agent Training **[v5.1 new: Chinese labs]**

- 28.3.1 **UI-TARS-2** (ByteDance Seed 2025.09, arXiv:2509.02544):
  - Multi-Turn RL for GUI agents
  - Asynchronous rollouts + a streaming training pool
  - Stateful envs + hybrid GUI-SDK
  - Value pretraining
- 28.3.2 **AutoGLM / Open-AutoGLM** (Zhipu 2025.12):
  - Self-evolving online curriculum RL
  - Self-evolving GUI agents
- 28.3.3 **MobileRL / ComputerRL** (Zhipu, arXiv:2509.18119 / arXiv:2508.14040)
- 28.3.4 CogAgent (Zhipu)

### 28.4 Instruction Hierarchy and Agent Safety **[v5.1 new]**

- 28.4.1 **Instruction Hierarchy** (OpenAI 2024.04, arXiv:2404.13208):
  - Permission levels across system/developer/user/tool instructions
  - The "kernel mode" analogy
- 28.4.2 GPT-5 Mini-R using instruction hierarchy as an RL reward (+0.11~0.21)
- 28.4.3 Defending against prompt injection
- 28.4.4 A core safety mechanism for the agent era

### 28.5 Multi-Agent Collaboration Frameworks

- 28.5.1 **Anthropic's multi-agent research system** (2025.06):
  - The orchestrator-worker pattern
  - An explicit OODA loop
  - Independent context windows per sub-agent
  - 90.2% faster than a single agent
- 28.5.2 Karpathy's "5-6 agents" pattern
- 28.5.3 Self-play multi-agent setups (debate, consensus)

### 28.6 Agent Swarms and Parallel-Agent RL **[v5.1 new]**

- 28.6.1 **Kimi K2.5 Agent Swarm** (2026.01, arXiv:2602.02276, kimi.com/blog/kimi-k2-5):
  - Parallel multi-agent + a trainable orchestrator
  - Parallel-agent RL
- 28.6.2 **Step 3.7 Flash Advisor Mode** (2026.05):
  - A small model executes, with a large model as advisor
  - Corresponds to Anthropic's advisor strategy
- 28.6.3 Communication and coordination among agents

### 28.7 Experiments

- 28.7.1 Training a simple GUI agent
- 28.7.2 A multi-agent collaboration task experiment

---

# Part VI · Multimodal Reinforcement Learning (4 Chapters **[v5.1 adds RL for Visual Generation]**)

## Chapter 29 RL for Vision-Language Models **[v5.1 expansion]**

### 29.1 Foundations of VLM RL Training

- 29.1.1 Joint vision-language representations
- 29.1.2 Sources of multimodal reward signals
- 29.1.3 Handling visual tokens vs. text tokens in RL

### 29.2 Designing Visual Reward Signals

- 29.2.1 Visual-QA correctness reward
- 29.2.2 Visual-caption completeness reward
- 29.2.3 Visual-hallucination penalty

### 29.3 Visual Reflection RL **[v5.1 new]**

- 29.3.1 **Qwen3-VL** (2025.11.26): reflection-driven visual re-attention
- 29.3.2 Self-correction of visual grounding
- 29.3.3 Combining reflection mechanisms with RL training

### 29.4 Training Frameworks

- 29.4.1 The EasyR1 / R1-V training frameworks
- 29.4.2 Open-Vision-Reasoner / Perception-R1
- 29.4.3 The "missing trace" problem in visual reasoning

### 29.5 China's Multimodal Frontier

- 29.5.1 Step3-VL-10B (arXiv:2601.09668): 1000+ RL iterations
- 29.5.2 GLM-4.6V: RLCS curriculum sampling
- 29.5.3 Qwen3-VL: reflection-driven
- 29.5.4 **Seed1.5-VL** (ByteDance, arXiv:2505.07062): a 20B-A200B MoE, GUI-agent + game RL

### 29.6 A Deep Dive on Parallel Coordinated Reasoning (PaCoRe)

- 29.6.1 PaCoRe's 16-way parallel rollout aggregation
- 29.6.2 An alternative path for test-time compute scaling
- 29.6.3 Comparison with MCTS over Thoughts

### 29.7 Experiment: GeoQA Geometric Reasoning

---

## Chapter 30 Audio and Speech RL **[v5.1 expansion]**

> **Design rationale**: Agent #4 explicitly flagged that v5's version of this chapter is only outline bullet points, missing core methods like Step-Audio's MGRD.

### 30.1 Overview of Audio Language Models

- 30.1.1 Audio tokenization schemes
- 30.1.2 Differences between speech generation and text generation
- 30.1.3 Engineering challenges of real-time inference

### 30.2 The Step-Audio Series **[a direction unique to China]**

- 30.2.1 **Step-Audio-R1** (2025.11, arXiv:2511.15848):
  - The first speech-language model to achieve test-time compute scaling
  - **MGRD** (Modality-Grounded Reasoning Distillation)
- 30.2.2 **Acoustic-Grounded Reasoning**: audio as the basis for reasoning
- 30.2.3 **Mind-Paced Speaking**: real-time reasoning and speech generation
- 30.2.4 The Dual-Brain Architecture

### 30.3 The RLVR → RLHF Evolution

- 30.3.1 **Step-Audio-R1.5**: shifting from RLVR to RLHF for Audio Reasoning
- 30.3.2 Multi-objective RL balancing vocal naturalness and reasoning ability
- 30.3.3 Preserving natural prosody

### 30.4 Designing Audio Rewards

- 30.4.1 Content-correctness reward
- 30.4.2 Prosodic-naturalness reward (modeling human preference)
- 30.4.3 Real-time-performance reward

### 30.5 Experiment: A Simple Spoken-Dialogue RL Task

---

## Chapter 31 Embodied Intelligence and VLA Models **[v5.1 upgraded flagship case study]**

> **Design rationale**: Agent #2 explicitly flagged that v5's use of RT-2 as its example in this chapter is outdated; the new benchmarks are Gemini Robotics 1.5 + π0 + Embodied Thinking.

### 31.1 Overview of VLA Models

- 31.1.1 **π0** (Physical Intelligence 2024): diffusion policy + VLM
- 31.1.2 RT-2 (Google 2023, treated as historical background)
- 31.1.3 OpenVLA (the open-source flagship)
- 31.1.4 **Gemini Robotics 1.5** (DeepMind 2025.09, the flagship model):
  - A dual-model VLA + ER setup
  - The **Embodied Thinking** paradigm
  - Cross-embodiment transfer (Apptronik Apollo / Boston Dynamics Spot)
  - Technical report PDF: storage.googleapis.com/deepmind-media

### 31.2 Foundations of Robot Learning

- 31.2.1 Observation space (vision, proprioception, force sensing)
- 31.2.2 Action space (joint angles, end-effector pose)
- 31.2.3 Reward function design

### 31.3 Diffusion Policy

- 31.3.1 Diffusion models as policies
- 31.3.2 Multimodal action distributions
- 31.3.3 Comparison with traditional Gaussian policies

### 31.4 Multimodal Fusion

- 31.4.1 Vision-language-action tokenization
- 31.4.2 Cross-modal alignment
- 31.4.3 Conditioning inputs for long-horizon tasks

### 31.5 Sim-to-Real

- 31.5.1 Domain randomization
- 31.5.2 Sim-to-real transfer techniques
- 31.5.3 System identification

### 31.6 Teleoperation and Demonstration

- 31.6.1 Collecting human demonstrations
- 31.6.2 Behavior-cloning pretraining
- 31.6.3 RL fine-tuning

### 31.7 Experiment: Basic VLA Training

- 31.7.1 Fine-tuning OpenVLA with RL for a tabletop grasping task

---

## Chapter 32 RL for Visual Generation **[v5.1 entirely new chapter]**

> **Design rationale**: Agent #4 (P0) explicitly pointed out that ByteDance Seed is the single largest source of innovation in video-generation RL for 2025-2026, yet v5 is completely blank on it. DanceGRPO adapts GRPO for diffusion, Seedance uses multi-dimensional RLHF, and LongCat-Video stacks multiple rewards — this is a globally leading direction from Chinese labs.

### 32.1 Defining Visual Generation Tasks

- 32.1.1 Text-to-Video
- 32.1.2 Image-to-Video
- 32.1.3 Video editing and continuation

### 32.2 Foundations of Diffusion + RL

- 32.2.1 Diffusion models as policy networks
- 32.2.2 Adapting RL for Rectified Flow
- 32.2.3 Fundamental differences from text-LLM RL

### 32.3 DanceGRPO **[a ByteDance Seed innovation]**

- 32.3.1 **DanceGRPO** (2025.05, arXiv:2505.07818):
  - Adapting GRPO to diffusion/flow-based visual generation
  - Unified across 4 foundation models
- 32.3.2 The core idea: treating a diffusion step as an RL timestep
- 32.3.3 Comparison with earlier methods like DDPO

### 32.4 Multi-Reward Video RLHF

- 32.4.1 **Seedance 1.0** (ByteDance, arXiv:2506.09113):
  - Foundational reward (basic quality)
  - Motion reward (motion plausibility)
  - Aesthetic reward
  - Refiner RLHF
- 32.4.2 **LongCat-Video** (ByteDance 2025.10, arXiv:2510.22200):
  - GRPO + multi-reward stacking
  - LoRA stacking implementation

### 32.5 Reward Models for Video Generation

- 32.5.1 VisionReward: an open-source video evaluation model
- 32.5.2 Multi-dimensional reward decomposition
- 32.5.3 Human-preference alignment

### 32.6 Physics-Aware Video Generation

- 32.6.1 **Hailuo-02** (MiniMax, a physics-aware NCR architecture)
- 32.6.2 Physical laws as intrinsic reward
- 32.6.3 Temporal-consistency constraints

### 32.7 Experiment: Training a Simple Video-Generation Model With DanceGRPO

---

# Part VII · Safety, Evaluation, and Alignment Research (4 Chapters)

## Chapter 33 Reward Hacking and Alignment Failure Modes **[v5.1 expansion]**

### 33.1 A Complete Taxonomy of Reward Hacking

- 33.1.1 A formal definition of reward hacking
- 33.1.2 Specification gaming vs. reward tampering vs. Goodhart's Law
- 33.1.3 Anthropic's 2025.11 taxonomy (arXiv:2511.18397)

### 33.2 RLVR's "Spurious Gains" **[v5.1 new]**

- 33.2.1 **Empirical evidence of data contamination** (arXiv:2507.10532, AAAI 2026):
  - Qwen's "spurious reward RLVR" gains on MATH-500 come primarily from data contamination
  - The "even random rewards can improve Qwen's performance" phenomenon
- 33.2.2 How GRPO's clipping bias activates memorization
- 33.2.3 Methodology for assessing the genuine gains from RLVR
- 33.2.4 Contamination-resistant evaluation design

### 33.3 Industry Failure Cases **[v5.1 new]**

- 33.3.1 **The GPT-4o sycophancy rollback** (OpenAI 2025.04-05):
  - A real-world RLHF failure case
  - User-feedback reward diluted the primary safety reward
  - Rolled back after 48 hours
  - The fix: an RL reward signal based on sycophancy (openai.com/index/sycophancy-in-gpt-4o)
- 33.3.2 **ByteDance Seed's RLHF data scaling**:
  - Reward hacking and diversity decay
  - The Pre-PPO prompt-selection strategy

### 33.4 Anthropic's Misalignment Research

- 33.4.1 **School of Reward Hacks** (Gao et al. 2025.08)
- 33.4.2 **Naturally emergent misalignment** (Anthropic 2025.11, arXiv:2511.18397):
  - Reward hacking arising naturally within RL environments
  - Generalizing to misaligned behavior
  - HHH reward as a mitigation
- 33.4.3 **Sleeper Agents** (Hubinger et al. 2024.01, arXiv:2401.05566)
- 33.4.4 **Alignment Faking** (Greenblatt et al. 2024.12, arXiv:2412.14093)
- 33.4.5 **In-Context Scheming** (Apollo 2024.12, arXiv:2412.04984)
- 33.4.6 **Sycophancy to Subterfuge** (Anthropic 2024, arXiv:2406.10162)

### 33.5 METR and Frontier-Model Research

- 33.5.1 METR: Frontier Models Reward Hacking (Von Arx et al. 2025)
- 33.5.2 Misalignment risk in long-horizon autonomous agents

### 33.6 Defense Mechanisms

- 33.6.1 Preference models and reward-hack classifiers
- 33.6.2 By-construction prevention of hacking through architecture
- 33.6.3 Ensembles of multiple verifiers
- 33.6.4 Formal verification as the ultimate line of defense (see Chapter 21)

---

## Chapter 34 Scalable Oversight and Red-Teaming

- 34.1 The scalable-oversight problem
- 34.2 AI Safety via Debate (Irving et al.)
- 34.3 Recursive Reward Modeling (OpenAI)
- 34.4 Weak-to-Strong Generalization (OpenAI 2023)
- 34.5 Red-teaming methodology
- 34.6 Adversarial training and robustness
- 34.7 The Sandwiching Problem
- 34.8 Exploration hacking and the exploit search problem

---

## Chapter 35 RL Evaluation Methodology

- 35.1 Principles of benchmark design
- 35.2 Detecting contamination and leakage (see Chapter 33's coverage of RLVR's spurious gains)
- 35.3 Prompt-sensitivity analysis
- 35.4 Out-of-distribution robustness
- 35.5 Behavioral evaluation vs. capability evaluation
- 35.6 Challenges in evaluating long-horizon tasks
- 35.7 **Anthropic's internal AI Research Eval Suite** (Opus 4.6):
  - LLM training / Text-RL / Quadruped-RL subtasks
  - A 34x human-speedup benchmark
- 35.8 Standardized evaluation harnesses: lm-eval-harness, BigCode Eval, τ-bench, BFCL

---

## Chapter 36 Distributed RL Training Systems

- 36.1 A deep dive into the veRL architecture
- 36.2 Comparing OpenRLHF / NeMo-Aligner / TRL
- 36.3 Rollout engines and vLLM integration
- 36.4 GPU memory optimization: ZeRO, FSDP, gradient checkpointing
- 36.5 Asynchronous RL training (LlamaRL, AReaL, AgentRL)
- 36.6 MoE + RL training (DeepSeek V3, Step Flash, GLM-4.5)
- 36.7 DualPipe and Best-Fit packing
- 36.8 Performance profiling and bottleneck analysis
- 36.9 Hands-on practice on 10,000-GPU clusters

---

# Part VIII · Research Frontiers (2 Chapters **[v5.1 expansion]**)

## Chapter 37 Evolutionary LLM Search and Generative World Models **[v5.1 entirely new]**

> **Design rationale**: Agent #2 (P0) explicitly pointed out that v5 is completely missing AlphaEvolve and Genie 3, the two most cutting-edge 2025-2026 directions.

### 37.1 The AlphaEvolve Paradigm

- 37.1.1 **AlphaEvolve** (DeepMind 2025.05):
  - LLM proposes a diff, an automated evaluator scores it, and an evolutionary algorithm selects among candidates
  - The first discovery of a 23% speedup in matrix multiplication
  - Improvements to 50+ open math problems
  - Paper PDF: storage.googleapis.com/deepmind-media
- 37.1.2 AlphaEvolve's algorithmic architecture: evolutionary search + LLM proposal
- 37.1.3 Differences from traditional RL: not policy gradient, but search + LLM
- 37.1.4 A new paradigm for search algorithms in the LLM era

### 37.2 Generative World Models as RL Environments

- 37.2.1 **Genie 3** (DeepMind 2025.08):
  - A real-time, interactive world model
  - 720p/24fps generation
  - World memory with multi-minute consistency
- 37.2.2 Generative environments vs. real environments
- 37.2.3 An unlimited RL training curriculum: agents learning inside generated worlds
- 37.2.4 A foundation for a general AGI world model

### 37.3 Long-Term Memory Architectures

- 37.3.1 **Titans + MIRAS** (Google Research 2025.12):
  - Neural long-term-memory modules
  - Test-time learning updates memory weights
  - Context of 2M+ tokens
- 37.3.2 A third paradigm beyond attention/RNNs
- 37.3.3 Combining long-term memory with RL agents

### 37.4 Recursive Self-Improvement

- 37.4.1 **Anthropic-Funded Research / Recursive Self-Improvement** (2026.04):
  - Claude conducting its own AI research
  - A 52x speedup on an internal benchmark (vs. 3x for Opus 4)
  - anthropic.com/institute/recursive-self-improvement
- 37.4.2 The "Claude Mythos Preview" model
- 37.4.3 The ultimate vision: training AI with RL to do AI research

---

## Chapter 38 Self-Play, Scaling Trends, and Future Directions

### 38.1 Foundations of Self-Play

- 38.1.1 The progression AlphaGo → AlphaZero → MuZero
- 38.1.2 The convergence properties of self-play
- 38.1.3 Applications of self-play in Go / chess / StarCraft

### 38.2 LLM Self-Play

- 38.2.1 LLM self-play and SPIN
- 38.2.2 Self-Play SWE-RL (SSR) (see Chapter 26 for detail)
- 38.2.3 Multi-agent debate as self-play
- 38.2.4 Mode collapse and preserving diversity

### 38.3 RL Scaling Laws

- 38.3.1 RL scaling laws (by analogy with Chinchilla)
- 38.3.2 Reward signal vs. data volume vs. model scale
- 38.3.3 The scaling limits of RLVR

### 38.4 Foundation-Model RL

- 38.4.1 The foundation model as RL's starting point
- 38.4.2 A unified view of RLHF / RLVR / RLAIF / Agent RL
- 38.4.3 The future shape of foundation-model RL

### 38.5 In-Context RL

- 38.5.1 In-context RL and Algorithm Distillation (DeepMind 2022)
- 38.5.2 Meta-learning and continual learning

### 38.6 Research Directions for the Next Decade

- 38.6.1 Karpathy's reflection that "AGI is still a decade away"
- 38.6.2 Open problems: credit assignment, long-horizon planning, generalization, safety
- 38.6.3 Divergent paths between Chinese and American labs
- 38.6.4 The leap from conversational models to autonomous agents

---

# Appendices (7 Sections)

## A. Training Debugging Handbook **[v5.1 expansion]**

- A.1 Diagnosing common training crashes
- A.2 Detecting gradient anomalies
- A.3 Handling KL-divergence explosions
- A.4 **MuonClip + QK-clip** optimizer stability (Kimi K2)
- A.5 Router interference in MoE + RL training
- A.6 Early signals of reward hacking (in its agent-specific forms)
- A.7 A troubleshooting checklist for function-call parsing failures
- A.8 A decision tree for diagnosing OOM in long trajectories
- A.9 A checklist for reproducing training crashes
- A.10 Tuning staleness in asynchronous RL training

## B. Reinforcement Learning Engineering Practice **[v5.1 major expansion]**

- B.1 The foundation of synchronous RL training systems (veRL, TRL)
- B.2 **Asynchronous RL training systems** (AReaL, AgentRL, SLIME, ROLL, LlamaRL)
- B.3 **Engineering implementations of staleness and cross-policy sampling**
- B.4 Agent sandbox engineering
- B.5 Evaluation-benchmark engineering
- B.6 A dictionary of training metrics
- B.7 **MoE + RL training engineering** (DeepSeek V3, Step Flash, GLM-4.5)
- B.8 Hands-on industrial exercises

## C. Core Algorithm Implementations **[v5.1 expansion]**

- C.1 Implementing SFT and KL divergence
- C.2 Implementing PPO and GAE
- C.3 Implementing the DPO family
- C.4 A basic GRPO implementation
- C.5 **Implementing the GRPO improvement family** (DAPO, Dr.GRPO, GSPO, CISPO, VAPO)
- C.6 Implementing RPT (Reinforcement Pre-Training)
- C.7 Implementing softmax and cross-entropy
- C.8 Implementing sampling methods (top-k, top-p, min-p)
- C.9 Implementing attention mechanisms (MHA, GQA, MLA, **DSA sparse attention**)
- C.10 Adapting **DanceGRPO** for diffusion/flow
- C.11 Implementing PRM training (discriminative + generative + formal Lean4)
- C.12 Implementing the MuonClip + QK-clip optimizer

## D. Learning Resources and Reproduction Projects

- D.1 A must-read paper list (organized by topic, 100+ papers)
- D.2 An index of open-source code repositories
- D.3 Recommended reproduction projects (Sky-T1, Open-R1, Tülu 3)
- D.4 An index of video courses (CS285, CS234, the Hugging Face Course)

## E. Mathematical Foundations

- E.1 Linear algebra (Bellman matrices, function approximation, convergence)
- E.2 Probability and statistics (return, value, sampling estimation, GAE)
- E.3 Calculus and optimization (gradients, PG, PPO, Adam)
- E.4 Information theory (entropy, KL divergence, cross-entropy, mutual information)

## F. A Paper-Reading Roadmap **[new]**

- F.1 Must-reads in classical RL (Sutton, Watkins, Mnih)
- F.2 Must-reads in deep RL (DQN, A3C, PPO, SAC)
- F.3 Must-reads in LLM RL (InstructGPT, CAI, DPO, GRPO, R1)
- F.4 Must-reads in safety research (Sleeper Agents, Alignment Faking, Reward Hacking)
- F.5 2025-2026 frontier reading (DAPO, GSPO, CISPO, PRM, PaCoRe)

## G. A GPU-Hour Estimation Table **[new]**

- G.1 Pretraining costs at different model scales
- G.2 Costs for the SFT / RLHF / RLVR stages
- G.3 Public training-data references from DeepSeek / Qwen / Step
- G.4 Budget planning for training your own model

## H. Notation and Algorithm Index **[new]**

- H.1 A unified notation table for the whole book
- H.2 An index of algorithm names (GRPO, PPO, DPO, SAC...)
- H.3 A table of abbreviations (RLHF, RLVR, PRM, CAI...)

---

# Full Chapter Statistics **[v5.1 revision]**

| Part       | Topic                                       | Chapter count                        |
| ---------- | ------------------------------------------- | ------------------------------------ |
| 0          | Preface · Introduction                      | 6 sections (corresponds to preface/) |
| I          | Foundations and Classical RL                | 7                                    |
| II         | Deep RL                                     | 5                                    |
| III        | Advanced RL Methods                         | 3                                    |
| IV         | LLM Alignment and Post-Training             | 8                                    |
| V          | **Agentic RL**                              | **5**                                |
| VI         | **Multimodal RL (incl. Visual Generation)** | **4** **[v5.1 +1]**                  |
| VII        | Safety, Evaluation, and Systems             | 4                                    |
| VIII       | **Research Frontiers**                      | **2** **[v5.1 +1]**                  |
| **Total**  |                                             | **38 chapters**                      |
| Appendices | A-H                                         | 8 sections                           |

---

# Comparison With the Existing Book **[v5.1 revision]**

| Dimension                 | Current book                      | **v5.1 Final**                                                                     |
| ------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- |
| Total chapter count       | 12                                | **38**                                                                             |
| Preface                   | Philosophical discussion up front | **0.1: hands-on CartPole first + a glimpse of what's ahead**                       |
| Agentic content           | 1 chapter, shallow                | **5 chapters, in depth + Chapter 28: instruction hierarchy/UI-TARS-2/K2.5**        |
| Multimodality             | 1 chapter, shallow                | **4 chapters (VLM/audio/VLA/visual generation)**                                   |
| Safety/alignment research | 0                                 | **3 chapters, incl. the 2025.11 paper + the GPT-4o rollback + data contamination** |
| GRPO family               | Only DAPO                         | **Algorithmic detail on 6+ variants (DAPO/Dr.GRPO/GSPO/CISPO/VAPO)**               |
| PRM                       | Mostly discriminative             | **Generative (ThinkPRM) + formal (Lean4/AlphaProof)**                              |
| RL Environments           | 0                                 | **A standalone chapter + asynchronous RL (AReaL/AgentRL)**                         |
| RL for visual generation  | 0                                 | **A standalone chapter (DanceGRPO/Seedance/LongCat)**                              |
| Engineering systems       | Appendix                          | **1 chapter of main text + expanded appendices**                                   |
| Hands-on code             | Partial coverage                  | **A lab in every chapter**                                                         |
| Coverage of Chinese labs  | DeepSeek only                     | **Full coverage: DeepSeek/Qwen/Kimi/Zhipu/Step/ByteDance/MiniMax**                 |
| Real paper citations      | None                              | **Every topic has an arXiv number + an official URL**                              |
| Frontier directions       | 0                                 | **AlphaEvolve/Genie 3/Titans/recursive self-improvement/RPT**                      |

---

# Rollout Recommendations **[v5.1 update]**

**Phase 1 (immediate, zero risk)**: restructure the preface + turn headings into proper textbook form + split Chapter 3 (MDP)

**Phase 2 (this month, P0)**: fill in the core of Part IV

- Fully restructure Chapter 19, the GRPO family (DAPO/Dr.GRPO/GSPO/CISPO/VAPO/RPT)
- Add Hybrid Thinking + long2short + emergence evidence to Chapter 20 (Reasoning)
- Upgrade Chapter 21 (PRM) with generative and formal approaches

**Phase 3 (next quarter, P0)**: fill in Part V (Agentic)

- Restructure Chapter 26's code agents around RL-based SWE as the main line
- Add instruction hierarchy / UI-TARS-2 / Kimi K2.5 Agent Swarm to Chapter 28

**Phase 4 (second half of the year, P0)**: fill in Part VI (Multimodal)

- Deepen Chapter 30's audio RL coverage with MGRD
- Upgrade Chapter 31's VLA coverage to Gemini Robotics 1.5
- Add Chapter 32, RL for visual generation (DanceGRPO/Seedance) **[entirely new]**

**Phase 5 (ongoing, P1)**: Part VII (safety) + Part VIII (frontier)

- Add the GPT-4o rollback / data contamination / Seed scaling to Chapter 33 (Reward Hacking)
- Add Chapter 37: AlphaEvolve / Genie 3 / Titans / recursive self-improvement **[entirely new]**

**Phase 6 (long-term, P1-P2)**: expand the appendices

- Add MuonClip + QK-clip to Appendix A
- Add asynchronous RL systems to Appendix B
- Add DanceGRPO / RPT / PRM implementations to Appendix C

---

# Quick Reference: Key Paper Citations **[v5.1 update]**

```
# GRPO family
[DeepSeek-R1] Nature 2025. https://www.nature.com/articles/s41586-025-09422-z
[DeepSeek-V3] arXiv:2412.19437
[DeepSeek V3.2 / DSA] arXiv:2512.02556
[DAPO] Yu et al. 2025.03. arXiv:2503.14476 NeurIPS 2025
[Dr.GRPO] Liu et al. 2025. arXiv:2508.10355
[GSPO] Zheng et al. 2025.07. arXiv:2507.18071 (Qwen3)
[CISPO] MiniMax 2025.06. arXiv:2506.13585 (M1)
[VAPO] ByteDance Seed 2025.04. arXiv:2504.05118
[REINFORCE++] Hu 2025
[RPT] Microsoft 2025.06. arXiv:2506.08007

# Qwen / Kimi
[Qwen3 Tech Report] arXiv:2505.09388
[Qwen3-Coder] qwenlm.github.io/blog/qwen3-coder
[Qwen Data Contamination] arXiv:2507.10532 AAAI 2026
[Kimi k1.5] arXiv:2501.12599
[Kimi K2] arXiv:2507.20534
[Kimi K2.5] arXiv:2602.02276 / kimi.com/blog/kimi-k2-5
[Ma et al. NoThinking] arXiv:2505.18681

# Zhipu GLM
[GLM-4.5 ARC] arXiv:2508.06471
[GLM-4.6] HuggingFace zai-org/GLM-4.6
[GLM-5] arXiv:2602.15763
[AReaL] arXiv:2505.24298 NeurIPS 2025
[AgentRL] arXiv:2510.04206
[AutoGLM] xiao9905.github.io/AutoGLM
[MobileRL] arXiv:2509.18119
[ComputerRL] arXiv:2508.14040

# StepFun
[Step3-VL-10B] arXiv:2601.09668
[Step-Audio-R1] arXiv:2511.15848
[Step 3.5 Flash] arXiv:2602.10604
[PaCoRe] github.com/stepfun-ai/PaCoRe (ACL 2026)

# ByteDance
[Seed-Thinking-v1.5] arXiv:2504.13914
[Seed1.5-VL] arXiv:2505.07062
[UI-TARS-2] arXiv:2509.02544
[Seedance 1.0] arXiv:2506.09113
[DanceGRPO] arXiv:2505.07818
[LongCat-Video] arXiv:2510.22200

# PRM
[Let's Verify Step by Step] Lightman et al. OpenAI 2023. arXiv:2305.20050
[ThinkPRM] arXiv:2504.16828
[PRM Survey] arXiv:2510.08049

# Formal RL
[AlphaProof + AlphaGeometry 2] deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level
[DeepSeek-Prover-V2] arXiv:2504.21801

# Agentic
[SWE-RL] Meta 2025.02. arXiv:2502.18449 NeurIPS 2025
[Code World Model] arXiv:2510.02387
[Self-play SWE-RL (SSR)] arXiv:2512.18552
[Search-R1] arXiv:2503.09516
[R1-Searcher] arXiv:2503.05592
[SWE-RM] arXiv:2512.21919
[Effective Harnesses] anthropic.com/engineering/effective-harnesses-for-long-running-agents 2025.11
[Multi-Agent Research System] anthropic.com/engineering/multi-agent-research-system 2025.06
[Anthropic Code RL JD] job-boards.greenhouse.io/anthropic/jobs/4613568008

# Safety and alignment
[Constitutional AI] Bai et al. Anthropic 2022. arXiv:2212.08073
[Sleeper Agents] Hubinger et al. 2024.01. arXiv:2401.05566
[Alignment Faking] Greenblatt et al. 2024.12. arXiv:2412.14093
[In-Context Scheming] Apollo 2024.12. arXiv:2412.04984
[Sycophancy to Subterfuge] Anthropic 2024. arXiv:2406.10162
[Natural Emergent Misalignment] MacDiarmid et al. Anthropic 2025.11. arXiv:2511.18397
[School of Reward Hacks] Gao et al. 2025.08
[METR Frontier Reward Hacking] Von Arx et al. 2025
[GPT-4o Sycophancy Rollback] openai.com/index/sycophancy-in-gpt-4o 2025.05
[Instruction Hierarchy] OpenAI 2024.04. arXiv:2404.13208

# Anthropic and OpenAI
[Anthropic Funded Research / Recursive Self-Improvement] anthropic.com/institute/recursive-self-improvement 2026.04
[Opus 4.6] anthropic.com/news/claude-opus-4-6
[Competitive Programming with LRM] OpenAI 2025.02. arXiv:2502.06807
[Weak-to-Strong Generalization] OpenAI 2023

# DeepMind
[AlphaEvolve] deepmind.google/blog/alphaevolve + paper PDF
[Genie 3] deepmind.google/blog/genie-3-a-new-frontier-for-world-models
[Gemini 3 Deep Think] blog.google Gemini 3
[Gemini Robotics 1.5] storage.googleapis.com/deepmind-media/.../Gemini-Robotics-1-5-Tech-Report.pdf
[Titans + MIRAS] research.google/blog/titans-miras-helping-ai-have-long-term-memory 2025.12

# Meta
[Llama 4] ai.meta.com/blog/llama-4-multimodal-intelligence
[Llama Guard 4] huggingface.co/meta-llama/Llama-Guard-4-12B

# Industry / economics
[Wing VC RL Environments Market] wing.vc/content/rl-environments-for-agentic-ai
[Karpathy 2025 Year in Review] karpathy.bearblog.dev
[Epoch AI RL Environments FAQ] epochai.substack.com/p/an-faq-on-reinforcement-learning
[Raschka State of LLMs 2025] magazine.sebastianraschka.com/p/state-of-llms-2025
[Raschka LLM Papers 2025] magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one
[DeepSeek CRFM Transparency] crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html

# Evaluation benchmarks
[τ-bench] Salesforce 2024-2025
[BFCL] Berkeley Function Calling Leaderboard
[WebArena] webarena.dev
[CyberGym] arXiv:2506.02548
[Vending-Bench] arXiv:2502.15840

# Tülu 3
[Tülu 3] Allen AI 2024-2025
```

---

# Quick Reference: Key Paper Citations

```
[DeepSeek-R1] Nature 2025. https://www.nature.com/articles/s41586-025-09422-z
[DeepSeek-V3] arXiv:2412.19437
[DAPO] Yu et al. 2025. arXiv:2503.14476
[Dr. GRPO] Liu et al. 2025
[GSPO] Zheng et al. 2025 (Qwen3)
[CISPO] MiniMax et al. 2025
[REINFORCE++] Hu 2025
[Qwen3 Tech Report] arXiv:2505.09388
[Kimi k1.5] Jan 2025
[Tülu 3] Allen AI 2024-2025
[Let's Verify Step by Step] Lightman et al. OpenAI 2023. arXiv:2305.20050
[Constitutional AI] Bai et al. Anthropic 2022. arXiv:2212.08073
[Sleeper Agents] Hubinger et al. 2024. arXiv:2401.05566
[Alignment Faking] Greenblatt et al. 2024. arXiv:2412.14093
[In-Context Scheming] Apollo 2024. arXiv:2412.04984
[Sycophancy to Subterfuge] Anthropic 2024. arXiv:2406.10162
[Natural Emergent Misalignment] MacDiarmid et al. Anthropic 2025.11. arXiv:2511.18397
[School of Reward Hacks] Gao et al. 2025.08
[METR Frontier Reward Hacking] Von Arx et al. 2025
[Effective Harnesses for Long-Running Agents] Anthropic 2025.11
[Weak-to-Strong Generalization] OpenAI 2023
[Karpathy 2025 Year in Review] karpathy.bearblog.dev
[Step3-VL-10B] arXiv:2601.09668
[Step-Audio-R1] github.com/stepfun-ai/Step-Audio-R1
[Epoch AI RL Environments FAQ] epochai.substack.com/p/an-faq-on-reinforcement-learning
[Raschka State of LLMs 2025] magazine.sebastianraschka.com/p/state-of-llms-2025
[Raschka LLM Papers 2025 List] magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one
[SWE-RL] Meta 2025. arXiv:2502.18486
[Search-R1] arXiv:2503.09516
[R1-Searcher] arXiv:2503.05592
[Anthropic Code RL JD] job-boards.greenhouse.io/anthropic/jobs/5254364008
[DeepSeek CRFM Transparency] crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html
[τ-bench] Salesforce 2024-2025
[CyberGym] arXiv:2506.02548
[Vending-Bench] arXiv:2502.15840
```

---

# A Final, Honest Statement to the Reader

1. **36 chapters is a massive undertaking**, but that's the scale an MIT-level textbook should have (Sutton & Barto has 14 chapters, CS285 has 11 lectures, and this book additionally covers LLM/Agentic/Multimodal content).
2. **This is not "read it and land the job"** — real job descriptions also require SE engineering ability, production debugging, distributed-systems experience, and product sense. This book covers the knowledge component.
3. **Every chapter should have a lab/experiment** — true hands-on practice is what makes the material stick. Some of this already exists in the book's `code/` directory, and it needs to be expanded.
4. **Continuous updates**: 2026 will bring new papers and new models; this book needs small revisions every quarter and a major revision every year.
5. **Writing-effort estimate**: 36 chapters × roughly 3,000-5,000 words per chapter + code ≈ 150,000-200,000 words plus a large volume of code. Estimated at 6-12 months of full-time work.
