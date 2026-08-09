# Hands-On Modern Reinforcement Learning — Full v5 Outline (MIT-Level Textbook)

> The final proposal, synthesizing all corrections from v1→v2→v3→v4→v5, based on real evidence (job descriptions from OpenAI / Anthropic / DeepSeek / Qwen / Zhipu / StepFun + 2025-2026 papers).
>
> **Core fix in this revision (0622)**: unify the three-tier heading hierarchy, strictly distinguishing "Part / single article / in-article sub-outline," eliminating the "X.Y numbering ambiguity" of the old version (where the old `1.1` could mean either a section within one article or a separate standalone article).

---

## Heading Level Rules (Mandatory, Unified)

| Level           | Marker            | Meaning                                                                                                                        |
| --------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Part**        | `#`               | Major part (Part I, Part II, preface...)                                                                                       |
| **Chapter**     | `##`              | A teaching unit, corresponding to a directory `chapterNN_xxx/`, **must be tagged `[single article]` or `[multiple articles]`** |
| **Article**     | `###`             | **Appears only under `[multiple articles]`**, one `.md` file                                                                   |
| **Sub-outline** | `-` indented list | The H2/H3 points to be expanded within that article (not a separate file)                                                      |

**Mandatory rules:**

- `###` is **forbidden** under a single-article chapter (to avoid misleading readers into thinking it's an independent file)
- `###` **must** be used to tag each file under a multiple-article chapter
- Design rationale and notes always go into `>` blockquotes, not mixed into the body text
- Version tags like `[v5 new]` and `[v5.1 expansion]` go in brackets after the chapter title, without polluting the heading hierarchy
- File paths are marked with a `→` arrow; legacy directory names are explained directly, not hidden

---

## Reading Conventions

| Marker                | Meaning                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| `[single article]`    | The chapter has only 1 `.md` file; the "internal sections" below are the H2/H3 headings within that file |
| `[multiple articles]` | The chapter has multiple `.md` files; each `###` item below is a separate article                        |
| `📄`                  | Article (one `.md` file)                                                                                 |
| `→ path`              | Actual file location                                                                                     |
| `Sub-outline:`        | The H2/H3 points to be expanded within that article (not a separate file)                                |

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
| Multimodal RL       | ❌             | ❌      | ❌                    | **✅ 4 dedicated chapters** |
| Safety/Alignment    | ❌             | ❌      | ❌                    | **✅ 3 dedicated chapters** |
| Engineering Systems | ❌             | ❌      | Partial               | **✅ 2 dedicated chapters** |

---

# Preface · Introduction `[multiple articles]`

> **Design rationale**: This book promises "hands-on first, theory later." But the old `preface/intro.md` opened with Sutton's philosophical discussion of the bitter lesson, and readers didn't touch code until Chapter 1 — the preface itself broke its own promise. v5 fix: the first section of the preface immediately gives an instantly-playable CartPole entry point, so readers see an agent go from wobbling to balanced within 30 seconds; **play first, then explain why**.
>
> **Directory**: `docs/preface/` (3 files total, covering sections 0.1-0.6)

### Opening Words (covers 0.1 / 0.2 / 0.3 / 0.6) → `docs/preface/intro.md`

**Sub-outline:**

- **0.1 Hands-on first: play with CartPole in 30 seconds** `[v5 new lead-in]`
  - Three tiers of experience, dual-source deployment covering both domestic and overseas readers, covering every type of reader:
  - **① One-click try (the preferred experience, zero install) — dual-source deployment**
    - Primary source: ModelScope Creative Space: `spaces.modelscope.cn/{namespace}/cartpole-playground`
      - Stable access within mainland China; this is the default for this book's primary audience (Chinese-language readers)
      - A Gradio app, nearly identical in code to the HF Space
    - Mirror source: HuggingFace Space: `hf.co/spaces/{namespace}/cartpole-playground`
      - For overseas readers / mirror-site visitors
    - On-page presentation: two iframe tabs to switch between "🇨🇳 Domestic entry point / 🌍 Overseas entry point"
      - ModelScope (primary source) shown by default
      - Automatic fallback prompt on load failure, guiding readers to switch
    - Experience: click the "Train" button → watch the reward curve climb in real time → the final policy plays automatically once training finishes
  - **② One-line local run (a deeper option)**
    - A prominent code box: `pip install "gymnasium[classic-control]" stable-baselines3 && python 1-ppo_cartpole.py`
    - CPU training finishes in 30 seconds, popping up a `--gui` cart demo window
    - Links into the full code at `code/chapter01_cartpole/`
    - For mainland readers: `pip install` uses the Tsinghua/Alibaba mirror, with configuration instructions given in the docs
  - **③ Visual preview (an offline fallback)**
    - A training-process GIF: an animation of the reward curve climbing from 20 to 500 (self-hosted in this book's repository at `docs/preface/images/`)
    - A final-policy demo video: the agent going from wobbly to stably balanced
    - The lowest-barrier experience, for when both sources fail to load, readers who don't want to install anything, or offline readers
  - **Promise fulfilled**: no matter which tier a reader chooses, and no matter their region, they have now "seen" an agent learn something — everything from 0.2 through 0.6 is a look back to explain what just happened
  - **Bridging classic and modern: a glimpse of what's ahead** `[v5 new]`
    - CartPole is RL's past (a classic task dating from the 1990s); this book's true protagonist is modern RL in the LLM era
    - Teaser 1: DPO teaches a large model to stop blindly obeying the user (a before-vs-after training conversation-comparison GIF: user asks for malicious code → complies before training / politely declines after training) → leads into Chapter 17, the DPO family
    - Teaser 2: the emergence of DeepSeek-R1's reasoning ability (a video of R1-Zero's pure-RL training spontaneously lengthening its CoT) → leads into Chapter 18 (the GRPO family) + Chapter 19 (Reasoning Models)
    - Teaser 3: a Computer Use agent operating a browser (public demo videos of Claude Computer Use / OpenAI Operator) → leads into Chapter 24, Computer Use and GUI Agents
    - Teaser 4: SWE-Agent autonomously fixing a bug (the full pipeline of an agent on SWE-bench: reading code → localizing the bug → making the fix → passing the tests) → leads into Chapter 23, Code Agent Reinforcement Learning
    - Presentation: all GIFs/videos self-hosted under `docs/preface/images/teasers/`, avoiding broken external links. Readers see the book's destination up front, without the preface forcing them into a hard requirement of being able to run an LLM (too high a barrier)

- **0.2 Opening Words: Why We Need RL**
  - Sutton's "The Bitter Lesson" and the two main threads of 70 years of AI history: search and learning
  - Why trial-and-error is the most primitive form of learning: the bicycle-riding analogy
  - From recognition to decision-making: why supervised learning cannot cover sequential decisions
  - What RL provides: it doesn't tell you how to do something, only what's good and what's not
  - Bridge to 0.1: the "trial-and-error → convergence" you just watched on CartPole is a concrete instance of this section

- **0.3 What Is RL: The Core Loop and Key Terminology**
  - The agent-environment-state-action-reward loop
  - Trajectory, return, discount factor $\gamma$
  - State vs. observation, discrete vs. continuous action spaces
  - Bridge to Chapter 1: redescribe the 0.1 CartPole example using this terminology

- **0.6 Book Structure and Reader Roadmap**
  - The progressive logic across the book's 7 Parts: Foundations → Deep → Advanced → LLM → Agentic → Multimodal → Safety and Frontier
  - Recommended paths for three types of readers:
    - ML engineers: 0.1 → Part IV–V (LLM + Agentic)
    - Readers with an RL background: Part II–III + Part IV
    - Students: work through Part I onward in sequence
  - Notation conventions and symbol table (see Appendix H for details)

### 0.4 A Brief History of Reinforcement Learning → `docs/preface/brief-history/index.md`

**Sub-outline:**

- 1950s-1980s: trial-and-error learning, the Bellman equation, the birth of TD learning
- 1992: TD-Gammon — the first RL system to defeat a human champion
- 2013: DQN plays Atari — the dawn of deep RL
- 2016: AlphaGo defeats Lee Sedol
- 2017-2019: AlphaGo Zero, MuZero, self-play
- 2017: PPO is published and becomes the industry standard (the algorithm you just used in 0.1 is PPO)
- 2022: InstructGPT / RLHF enters large-model training
- 2023-2024: DPO, GRPO, Constitutional AI
- 2025: DeepSeek-R1, o1/o3, the RLVR paradigm is established
- The rise of Chinese labs: Qwen3 GSPO, Step-Audio, DeepSeek's transparency

### 0.5 Environment Setup Guide → `docs/preface/env-setup.md`

**Sub-outline:**

- Python environment: conda vs. venv
- PyTorch version and CUDA configuration
- Gymnasium installation and verification
- Preview of the veRL / OpenRLHF / TRL toolchain
- Training hardware checklist: entry-level experiments / core experiments / large-project tiers
- Repository code structure: each chapter's independent subdirectory under `code/`

---

# Part I · Foundations and Classical Reinforcement Learning (6 Chapters)

## Chapter 1 CartPole: The First Reinforcement Learning Experiment `[multiple articles]`

> **Directory**: `chapter01_cartpole/` (the directory follows the old numbering, 01, which is consistent with chapter number 1)

### 1.1 CartPole Basics and Principles → `chapter01_cartpole/intro.md` + `principles.md`

**Sub-outline:**

- The CartPole problem and the Gym/Gymnasium interface
- Engineering definitions of state, action, and reward
- Random-policy baseline and failure modes

### 1.2 Training Metrics Design → `chapter01_cartpole/metrics.md`

**Sub-outline:**

- Return curves, success rate, stability
- Experiment: the full pipeline from random to converged

### 1.3 Hands-On PPO Training and Visualization → `chapter01_cartpole/training.md`

**Sub-outline:**

- Getting started with PPO training via stable-baselines3
- Visualizing training curves and what the hyperparameters mean
- Diagnosing failure modes (oscillating convergence, stalled reward)
- Connecting back to the training pipeline behind the preface's 0.1 hands-on demo

---

## Chapter 2 Multi-Armed Bandits and Exploration-Exploitation Theory `[multiple articles]`

> **Directory**: `chapter03_bandits/`

### 2.1 Problem Definition and ε-Greedy → `chapter03_bandits/intro.md`

**Sub-outline:**

- The multi-armed bandit problem and its formal definition
- Regret as a measure of performance
- The ε-greedy algorithm and decay schedules
- Regret-bound analysis for ε-greedy

### 2.2 UCB and Thompson Sampling → `chapter03_bandits/ucb-thompson.md`

**Sub-outline:**

- Upper Confidence Bound (UCB): optimism in the face of uncertainty
- Thompson sampling: the Bayesian perspective and probability matching
- The Beta conjugate prior for Bernoulli rewards
- Comparing UCB and Thompson sampling, with an industry case study (Google AdWords)

### 2.3 Regret Bounds, PAC, and Contextual Bandits → `chapter03_bandits/theory-contextual.md`

**Sub-outline:**

- The Lai-Robbins lower bound and asymptotic optimality `[theory]`
- The PAC framework and sample complexity
- Contextual bandits: LinUCB / LinTS / NeuralUCB
- The connection to RLHF: why GRPO samples multiple rollouts, why PPO needs importance sampling

---

## Chapter 3 Markov Decision Processes `[multiple articles]`

> **Legacy note**: currently merged together with Chapters 4/5/6 under `chapter03_mdp/`. v5 proposes splitting this into a dedicated directory, `chapter04_mdp/`, but because it involves extensive cross-references, that migration work is deferred to Phase 2.

### 3.1 MDP Fundamentals and the Markov Property → `chapter03_mdp/mdp.md`

**Sub-outline:**

- From bandits to sequential decision-making
- The mathematical definition of the Markov property, and its intuition
- State space, action space, transition function, reward function

### 3.2 Policies, Value, and Return → `chapter03_mdp/policy-value.md`

**Sub-outline:**

- Defining a policy: deterministic vs. stochastic
- Return and value (a first introduction to V/Q, covered in full in Chapter 4)
- Policy evaluation and episodic vs. continuing tasks

### 3.3 Discounting, Trajectories, and POMDPs → `chapter03_mdp/panorama.md`

**Sub-outline:**

- The mathematical meaning of the discount factor and return
- Trajectories and episodes
- Partially Observable MDPs (POMDPs) `[groundwork for LLM multi-turn interaction]`
- Real-world POMDP examples: robot perception, dialogue history

---

## Chapter 4 Value Functions and the Bellman Equation `[multiple articles]`

> **Legacy note**: merged together with Chapters 3/5/6. The actual content currently spans two files, `chapter03_mdp/value-bellman.md` and `value-q.md`.

### 4.1 V/Q Functions and the Bellman Expectation Equation → `chapter03_mdp/value-bellman.md`

**Sub-outline:**

- The state-value function $V^\pi(s)$
- The action-value function $Q^\pi(s,a)$
- Deriving the Bellman expectation equation
- The relationship between V and Q

### 4.2 Bellman Optimality, Contraction Mapping, and the Optimal Policy → `chapter03_mdp/value-q.md`

**Sub-outline:**

- The Bellman optimality equation
- The contraction-mapping property of the Bellman operator `[theory]`
- Existence and uniqueness of the optimal policy
- Applying the Banach fixed-point theorem

### 4.3 Numerical Experiments With Value Functions → `chapter03_mdp/value-experiment.md`

**Sub-outline:**

- Policy evaluation on a Gridworld
- Empirically measuring the convergence rate of value iteration
- Visualizing V/Q and policy improvement
- Bridge to Chapter 5, dynamic programming

---

## Chapter 5 Dynamic Programming, Monte Carlo, and Temporal Difference Learning `[multiple articles]`

> **Legacy note**: merged together with Chapters 3/4/6. The actual content currently lives in `chapter03_mdp/dp-mc-td.md`.

### 5.1 Dynamic Programming → `chapter03_mdp/dp-mc-td.md`

**Sub-outline:**

- Policy evaluation and policy improvement
- Policy iteration and value iteration
- The limitation of DP: it requires a perfect model

### 5.2 Monte Carlo Methods → `chapter03_mdp/dp-mc-td.md`

**Sub-outline:**

- First-visit and every-visit MC
- The model-free nature of MC
- The variance problem and control

### 5.3 Temporal Difference Learning, n-Step Methods, and Eligibility Traces → `chapter03_mdp/dp-mc-td.md`

**Sub-outline:**

- TD(0) learning
- The bias-variance trade-off in n-step bootstrapping and TD(λ)
- Eligibility traces
- Comparing and weighing DP / MC / TD

---

## Chapter 6 Q-Learning and Off-Policy Control `[multiple articles]`

> **Legacy note**: merged together with Chapters 3/4/5. The actual content currently lives in `chapter03_mdp/algorithm-taxonomy.md`.

### 6.1 On-Policy vs. Off-Policy, Q-Learning, and SARSA → `chapter03_mdp/algorithm-taxonomy.md`

**Sub-outline:**

- The distinction between on-policy and off-policy
- The Q-Learning algorithm and its convergence
- The SARSA algorithm
- Q-Learning vs. SARSA: the classic Cliff Walking example

### 6.2 Importance Sampling and the Deadly Triad → `chapter03_mdp/algorithm-taxonomy.md`

**Sub-outline:**

- Importance sampling `[key groundwork, foundational to PPO/GRPO]`
- The challenges of function approximation and the Deadly Triad (function approximation + bootstrapping + off-policy) `[theory]`
- A first look at off-policy gradient methods

### 6.3 An Introduction to Reward Function Design → `chapter03_mdp/reward-design.md`

**Sub-outline:**

- Sparse vs. dense reward
- Reward shaping and potential-based shaping
- Early examples of reward hacking
- Bridge to Chapter 30's dedicated coverage of reward hacking

---

# Part II · Deep Reinforcement Learning (5 Chapters)

## Chapter 7 Deep Q-Networks and Distributional RL `[multiple articles]`

> **Directory**: `chapter04_dqn/` (6 files; the directory follows the old numbering, 04, inconsistent with chapter number 7 — a legacy artifact)

### 7.1 From Q-Learning to DQN → `chapter04_dqn/from-q-to-dqn.md`

**Sub-outline:**

- Motivation for moving from Q-Learning to DQN
- Experience Replay
- The Target Network

### 7.2 The DQN Improvement Family → `chapter04_dqn/dqn-family.md`

**Sub-outline:**

- Double DQN: addressing overestimation
- Dueling DQN: decomposing state-action value
- Prioritized Experience Replay (PER)
- Rainbow and NoisyNet

### 7.3 Distributional RL → `chapter04_dqn/dqn-components.md`

**Sub-outline:**

- C51, QR-DQN, IQN
- The mathematical foundations of distributional value functions

### 7.4 Experiments: LunarLander and Atari → `chapter04_dqn/lunar-lander.md` + `visual-game-projects.md`

**Sub-outline:**

- The LunarLander training pipeline
- The Atari game benchmark
- Hyperparameter tuning and visualization

---

## Chapter 8 Policy Gradient Methods `[multiple articles]`

> **Directory**: `chapter05_policy_gradient/` (9 files; the directory follows the old numbering, 05, inconsistent with chapter number 8 — a legacy artifact)

### 8.1 Introduction to Policy Gradients → `chapter05_policy_gradient/intro.md` + `policy-gradient.md`

**Sub-outline:**

- Motivation for policy gradient methods (continuous actions, stochastic policies)
- Policy representations: softmax, Gaussian, categorical
- The policy gradient theorem (full derivation) `[theory]`

### 8.2 REINFORCE and Baselines → `chapter05_policy_gradient/reinforce.md` + `baseline-experiment.md`

**Sub-outline:**

- The REINFORCE algorithm
- The variance problem and baselines

### 8.3 Policy Gradient Improvements and Experiments → `chapter05_policy_gradient/pg-improvements.md` + `cartpole.md` + `cartpole-baseline.md` + `dice-game.md` + `pg-necessity.md`

**Sub-outline:**

- Off-policy policy gradients
- Experiments: CartPole and Pendulum
- Proving the necessity of policy gradients (a dice-game example)

---

## Chapter 9 Actor-Critic Architectures `[multiple articles]`

> **Directory**: `chapter06_actor_critic/` (8 files; the directory follows the old numbering, 06, inconsistent with chapter number 9 — a legacy artifact)

### 9.1 The Advantage Function and Training the Critic → `chapter06_actor_critic/advantage-function.md` + `critic-training.md`

**Sub-outline:**

- The advantage function $A(s,a) = Q(s,a) - V(s)$
- Training the critic network (fitting the value function)

### 9.2 The Actor-Critic Framework and Synchronized Updates → `chapter06_actor_critic/actor-critic.md` + `ac-frontier.md`

**Sub-outline:**

- The Actor-Critic framework and synchronized updates
- Generalized Advantage Estimation (GAE) `[groundwork for PPO]`
- A2C and A3C: asynchronous parallelism

### 9.3 Experiments → `chapter06_actor_critic/pendulum.md` + `bipedalwalker.md` + `alphago.md`

**Sub-outline:**

- Experiments: Pendulum and BipedalWalker
- AlphaGo as an application of Actor-Critic

---

## Chapter 10 PPO and Trust-Region Methods `[multiple articles]`

> **Directory**: `chapter07_ppo/` (7 files; the directory follows the old numbering, 07, inconsistent with chapter number 10 — a legacy artifact)

### 10.1 TRPO and Trust Regions → `chapter07_ppo/trust-region-clipping.md` + `ppo-math.md`

**Sub-outline:**

- The stability problem in policy updates
- TRPO and the monotonic improvement theorem `[theory]`
- The mathematical derivation of PPO

### 10.2 Engineering PPO-Clip → `chapter07_ppo/intro.md`

**Sub-outline:**

- The PPO-Clip algorithm
- PPO-Penalty and adaptive KL
- Engineering details of PPO implementation (entropy bonus, value clipping)

### 10.3 GAE and the Reward Model → `chapter07_ppo/gae-reward-model.md`

**Sub-outline:**

- Applying GAE within PPO
- The interface between the reward model and PPO (laying groundwork for RLHF in Chapter 15)

### 10.4 Long-Horizon Tasks and Experiments → `chapter07_ppo/rl-long-horizon-planning.md` + `ppo-bipedal-walker.md` + `ppo-game-benchmark.md`

**Sub-outline:**

- PPO in long-horizon tasks
- PPO's place in the LLM-RL era (see Chapter 18, the GRPO family, for detail)
- Experiments: BipedalWalker continuous control and game benchmarks

---

## Chapter 11 Continuous Control and Model-Based Deep RL `[multiple articles]`

> **Directory**: `chapter12_continuous_control/`

### 11.1 Deterministic Policy Gradients and DDPG → `chapter12_continuous_control/intro.md`

**Sub-outline:**

- The Deterministic Policy Gradient (DPG) theorem
- The DDPG algorithm: Actor-Critic + experience replay + target networks
- DDPG's flaws: Q-value overestimation, hyperparameter sensitivity, unstable training

### 11.2 TD3 and SAC → `chapter12_continuous_control/td3-sac.md`

**Sub-outline:**

- TD3's three fixes: twin Q-networks / delayed policy updates / target policy smoothing
- Soft Actor-Critic: the maximum-entropy RL framework
- The soft Bellman equation and automatic temperature tuning
- A comparison table across the three algorithms and selection guidance

### 11.3 Model-Based RL: Dyna, PETS, MBPO → `chapter12_continuous_control/model-based.md`

**Sub-outline:**

- Why use a model: a fundamental improvement in sample efficiency
- Dyna: the model as a form of data augmentation
- PETS: probabilistic trajectory sampling and ensembles
- MBPO: model-based policy iteration with short-horizon rollouts

### 11.4 AlphaZero, MuZero, and Dreamer V3 → `chapter12_continuous_control/search-world-models.md`

**Sub-outline:**

- AlphaZero: MCTS + neural network evaluation + self-play
- MuZero: implicit model learning (representation + dynamics + prediction networks)
- Dreamer V3: an RSSM world model + training an actor-critic in imagination
- A trade-off table for model-based vs. model-free approaches

---

# Part III · Advanced RL Methods (3 Chapters, Trimmed but Deep)

## Chapter 12 Offline Reinforcement Learning and Decision Transformers `[multiple articles]`

> **Directory**: `chapter13_offline_rl/`

### 12.1 The Challenge of Offline RL and Classic Methods → `chapter13_offline_rl/intro.md`

**Sub-outline:**

- The challenge of offline RL: distribution shift and extrapolation error
- The principle of pessimism: CQL, IQL, BCQ
- AWAC and TD3+BC: simplified conservative constraints
- The relationship to behavior cloning (BC)

### 12.2 Decision Transformer, Trajectory Transformer, and Diffuser → `chapter13_offline_rl/sequence-modeling.md`

**Sub-outline:**

- Decision Transformer: RL as sequence modeling
- Trajectory Transformer: planning-style sampling
- Diffuser: making decisions with a diffusion model
- Offline RL in the LLM era (viewing DPO through the lens of offline RL)

### 12.3 Offline RL Experiments and the LLM Perspective → `chapter13_offline_rl/experiments.md`

**Sub-outline:**

- The D4RL benchmark and classic tasks (HalfCheetah, AntMaze)
- Comparing CQL / IQL / DT experimentally
- The unifying view of DPO/IPO from an offline-RL perspective
- Bridge to Chapter 17, the DPO family

---

## Chapter 13 Imitation Learning, Inverse RL, and Meta-RL `[multiple articles]`

> **Directory**: `chapter14_imitation_meta_rl/`

### 13.1 Behavior Cloning and DAgger → `chapter14_imitation_meta_rl/bc-dagger.md`

**Sub-outline:**

- Behavior cloning (BC) and the covariate-shift problem
- DAgger: Dataset Aggregation to fix distribution drift
- The connection to SFT: the supervised-learning paradigm

### 13.2 Inverse RL and GAIL → `chapter14_imitation_meta_rl/irl-gail.md`

**Sub-outline:**

- Maximum-entropy inverse RL (MaxEnt IRL)
- GAIL: Generative Adversarial Imitation Learning
- The connection to RLHF: reward learning

### 13.3 Meta-RL: MAML, RL², PEARL, In-Context RL → `chapter14_imitation_meta_rl/meta-rl.md`

**Sub-outline:**

- MAML: Model-Agnostic Meta-Learning
- RL²: implicitly learning fast adaptation via an RNN
- PEARL: probabilistic meta-RL
- In-context RL and Algorithm Distillation `[DeepMind 2022]`

---

## Chapter 14 Exploration, Multi-Agent RL, and Hierarchical RL `[multiple articles]`

> **Directory**: `chapter15_exploration_marl_hierarchical/`

### 14.1 Exploration: ICM, RND, NGU, Agent57 → `chapter15_exploration_marl_hierarchical/intro.md`

**Sub-outline:**

- The fundamental tension between exploration and exploitation (theoretical review)
- Intrinsic Curiosity Modules (ICM) and Random Network Distillation (RND)
- NGU and Agent57: episodic + life-long intrinsic reward
- Callback to Chapter 2's coverage of bandit exploration theory

### 14.2 Multi-Agent RL: CTDE, MADDPG, MAPPO → `chapter15_exploration_marl_hierarchical/marl.md`

**Sub-outline:**

- The challenge of multi-agent RL: non-stationary environments
- The CTDE framework: centralized training, decentralized execution
- MADDPG: one critic per agent
- MAPPO: industrial practice with multi-agent PPO

### 14.3 Hierarchical RL and a Preview of Generative World Models → `chapter15_exploration_marl_hierarchical/hierarchical.md`

**Sub-outline:**

- The motivation for hierarchical RL: decomposing long-horizon tasks hierarchically
- The Options framework
- FeUdal Networks and HIRO
- Generative world models as RL environments (a preview of Genie 3; see Chapter 31 for detail)

---

# Part IV · Large Language Model Alignment and Post-Training (7 Chapters)

## Chapter 15 The RLHF Training Pipeline `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `chapter08_rlhf/` (10 files; the directory follows the old numbering, 08, inconsistent with chapter number 15 — a legacy artifact)

### 15.1 Base Models and Instruction Alignment → `chapter08_rlhf/base-model-to-assistant.md`

**Sub-outline:**

- Base models and instruction alignment
- The modern three-stage paradigm: SFT → RLHF → RLVR

### 15.2 SFT Instruction Tuning → `chapter08_rlhf/imitation-learning-pipeline.md`

**Sub-outline:**

- SFT instruction tuning
- Data construction and formatting

### 15.3 Reward Modeling: The Bradley-Terry Model → `chapter08_rlhf/reward-function-design.md`

**Sub-outline:**

- Reward modeling: the Bradley-Terry model
- The reward-model training pipeline

### 15.4 RL Fine-Tuning (PPO or GRPO) → `chapter08_rlhf/standard-rlhf-pipeline.md` + `ppo-rlhf-loop.md`

**Sub-outline:**

- The standard RLHF pipeline
- The loop structure of PPO within RLHF
- KL constraints and the reference policy

### 15.5 Dual-Track Reward and Pre-PPO `[v5.1 expansion]` → `chapter08_rlhf/intro.md`

**Sub-outline:**

- Dual-track reward design (Seed-Thinking: verifiable + pairwise)
- Pre-PPO: prompt-selection strategies to avoid reward hacking

### 15.6 Extended Practice and Large-Scale Training → `chapter08_rlhf/extended-practice.md` + `scaling-to-large-models.md`

**Sub-outline:**

- Extended hands-on practice
- Large-scale training (including a reference to Tülu 3's three-stage paradigm)

### 15.7 Evaluation → `chapter08_rlhf/evaluation.md`

**Sub-outline:**

- Post-RLHF-training evaluation metrics
- Safety and alignment evaluation

### 15.8 Hands-On Experiment: Training GSM8K With veRL + PPO → `chapter08_rlhf/verl-ppo-gsm8k.md`

**Sub-outline:**

- Getting started with the veRL framework
- Training on the GSM8K dataset
- Analyzing training curves and results

---

## Chapter 16 LLM RL in Industry and Distributed Training Systems `[multiple articles]` `[v5.1 expansion: shifting from PPO to the modern GRPO pipeline]`

> **Design rationale**: By 2025-2026, Llama 4 / Qwen3 / DeepSeek V3.2 / GLM-4.6 have all fully shifted to GRPO/Dr.GRPO + RLVR, and the old version's focus on the classic PPO implementation is outdated.
>
> **Merge note**: the original Chapter 36, Distributed RL Training Systems (veRL/OpenRLHF/async/MoE/10,000-GPU clusters), has been merged into this chapter — training frameworks and distributed systems are fundamentally the same subject, and forcing them apart just created duplicate content.
>
> **Directory**: `chapter17_llm_rl_industrial/` + `chapter36_distributed_rl_training/` (the distributed subsection keeps its own directory)

### 16.1 Comparison of Training Frameworks → `chapter17_llm_rl_industrial/01-frameworks.md`

**Sub-outline:**

- Synchronous frameworks: veRL (ByteDance's mainstream choice) / OpenRLHF (open-source-friendly) / TRL (HF ecosystem) / NeMo-Aligner (NVIDIA)
- Asynchronous frameworks: AReaL (Tsinghua + Zhipu) / AgentRL (Zhipu + Tsinghua) / SLIME / ROLL / LlamaRL
- Framework comparison table and a selection decision tree

### 16.2 Modern Post-Training Pipeline Paradigms → `chapter09_alignment/industrial-post-training.md` (reuse existing)

**Sub-outline:**

- DeepSeek-R1's multi-stage pipeline: cold-start SFT → reasoning RL → rejection sampling → full-scenario RL
- Llama 4: lightweight SFT → online RL → lightweight DPO + pass@k difficulty filtering
- Qwen3: Thinking Mode Fusion + Thinking Budget + GSPO
- GLM-4.5 / 4.6: difficulty-curriculum RL + Hybrid Thinking + RLCS curriculum sampling
- GLM-5 (2026.02): asynchronous Agent RL + DSA sparse attention
- Seed-Thinking-v1.5: dual-track reward + Pre-PPO + hybrid reward

### 16.3 Dual-Track Reward Design → `chapter17_llm_rl_industrial/03-dual-reward.md`

**Sub-outline:**

- Verifiable reward (math, code)
- Pairwise preference reward (open-ended dialogue)
- Pre-PPO: prompt-selection strategies to avoid reward hacking
- Hybrid reward: combining RTV + GenRM

### 16.4 Optimizers and Training Stability → `chapter09_alignment/modern-industrial-practice.md` (reuse existing)

**Sub-outline:**

- AdamW's stability issues in RL training
- The MuonClip optimizer (Kimi K2)
- QK-clip: attention numerical stability
- Early signals of KL explosion and how to handle them

### 16.5 Distributed Synchronous Frameworks and Rollout Engines → `chapter36_distributed_rl_training/intro.md`

**Sub-outline:**

- A deep dive into the veRL architecture
- Comparing OpenRLHF / NeMo-Aligner / TRL
- Rollout engines and vLLM integration
- GPU memory optimization: ZeRO, FSDP, gradient checkpointing
- Performance profiling and bottleneck analysis

### 16.6 Asynchronous RL Training Systems → `chapter36_distributed_rl_training/async.md`

**Sub-outline:**

- The staleness problem in asynchronous RL
- LlamaRL (Meta): a purely asynchronous pipeline
- AReaL (Tsinghua + Zhipu): heterogeneous compute scheduling
- AgentRL (Zhipu + Tsinghua): asynchronous training for long-trajectory agents
- Comparing the SLIME / ROLL frameworks

### 16.7 MoE + RL and 10,000-GPU Clusters → `chapter36_distributed_rl_training/scale.md`

**Sub-outline:**

- MoE + RL training (DeepSeek V3, Step Flash, GLM-4.5)
- DualPipe and Best-Fit packing
- Expert-load-balancing stability during the RL stage
- Practice on 10,000-GPU clusters: communication, fault tolerance, checkpointing
- Measured cost/throughput for RL training at 10,000-GPU scale

### 16.8 Hands-On in Industry: GSM8K and AIME → `chapter09_grpo_rlvr/verl-code-sandbox.md` (reuse existing)

**Sub-outline:**

- Experiment: training GSM8K with GRPO
- Experiment: training AIME 2024 with DAPO
- Full open-source reproductions: Open-R1 / Sky-T1 / Tülu 3

### 16.9 Common Interview Topics for Chinese Alignment Teams → `chapter17_llm_rl_industrial/07-interview.md`

**Sub-outline:**

- The full derivation chain PG → REINFORCE → TRPO → PPO → GRPO (a real Zhipu interview question)
- The DPO family + DPO regularization
- Engineering comparison: DeepSpeed vs. Megatron
- On-the-spot estimation of training resource consumption

---

## Chapter 17 Preference Alignment: The DPO Family `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `chapter02_dpo/` (3 files, old numbering 02) + `chapter09_alignment/dpo-theory-and-family.md`

### 17.1 DPO Basics and Derivation → `chapter02_dpo/intro.md` + `principles.md`

**Sub-outline:**

- The mathematical derivation of DPO (deriving it from the RLHF objective)
- Analyzing DPO training dynamics

### 17.2 DPO Training Metrics → `chapter02_dpo/metrics.md`

**Sub-outline:**

- Training-monitoring metrics
- Key metrics such as reward margin and accuracy

### 17.3 DPO Theory, Mathematics, and Choosing Within the Family → `chapter09_alignment/dpo-theory-and-family.md`

**Sub-outline:**

- IPO: fixing DPO's overfitting
- KTO: no need for paired preference data
- SimPO: a reference-free method
- DPO regularization methods (a real Zhipu interview question)
- Iterative DPO and ReST
- Self-Play Fine-Tuning (SPIN)
- A decision tree for choosing within the DPO family

---

## Chapter 18 The GRPO Family, RLVR, and Verifier Engineering `[multiple articles]` `[v5.1 full restructure]`

> **Design rationale**: the biggest algorithmic focus of 2025-2026. Four independent research efforts consistently pointed out that v5's version of this chapter only lists names with no algorithmic detail. v5.1 reorganizes it by direction of improvement, covering the algorithmic differences across 6+ mainstream variants.
>
> **Merge note**: the original Chapter 23, Designing RL Environments and Verifiers, has been merged into this chapter — verifiers are a core component of RLVR, and RLVR is in turn the training paradigm behind the GRPO family, so the three form one complete closed loop; splitting them apart created excessive cross-referencing.
>
> **Directory**: `chapter09_grpo_rlvr/` (7 files) + `chapter23_rl_environments/`

### 18.1 GRPO Training and Core Mechanics → `chapter09_grpo_rlvr/grpo-practice-and-mechanism.md`

**Sub-outline:**

- From PPO to GRPO: why drop the critic
- The principle of group normalization: relative advantage across multiple rollouts of the same prompt
- KL constraints and the reference-policy implementation

### 18.2 The R1-Zero Paradigm (DAPO) → `chapter09_grpo_rlvr/deepseek-dapo.md`

**Sub-outline:**

- DAPO (ByteDance + Tsinghua, 2025.03, arXiv:2503.14476, NeurIPS 2025)
  - Clip-Higher: decoupling $\epsilon_{low} \neq \epsilon_{high}$
  - Dynamic Sampling: filtering out all-correct/all-wrong samples
  - Token-level Loss: avoiding domination by long responses
  - Overlong Filtering + Soft Shaping
- Dr.GRPO (Liu et al. 2025, arXiv:2508.10355): removing std and length normalization
- DeepSeek V3.2's KL tuning: zero KL, self-verifying RLVR, mHC residual stability (arXiv:2512.02556)

### 18.3 RLVR: Verifiable Rewards → `chapter09_grpo_rlvr/rlvr.md`

**Sub-outline:**

- Defining RLVR: rule-based feedback in place of human annotation
- Sources of reward in RLVR: math verifiers, unit tests, formal proofs
- Hybrid pipelines combining RLVR and RLHF

### 18.4 The GRPO Improvement Family (Dr.GRPO / GSPO / CISPO / VAPO / RPT) → `chapter09_grpo_rlvr/grpo-family.md`

**Sub-outline:**

- GSPO (Zheng et al. 2025, Qwen3, arXiv:2507.18071): sequence-level IS ratio + sequence-level clipping
- CISPO (MiniMax 2025.06, arXiv:2506.13585): clipping the IS weight rather than the token update, 2x speedup
- VAPO (ByteDance Seed 2025.04, arXiv:2504.05118): the value-based counter-trend, beating GRPO on long CoT
- REINFORCE++ (Hu 2025) / AREAL (asynchronous) / ASPO / DCPO and other niche variants
- Comparing DAPO vs. CISPO for selection
- RPT (Microsoft 2025.06, arXiv:2506.08007): Reinforcement Pre-Training challenges the pretraining/post-training dichotomy
- A decision tree for choosing a variant: mapping from task type to recommended algorithm

### 18.5 RL Environments as the New Bottleneck → `chapter23_rl_environments/intro.md`

**Sub-outline:**

- Anthropic's $1B investment in RL Environments (The Information, 2025.09)
- Wing VC data: Anthropic spends tens of millions of dollars per year, expanding 3-5x by 2026
- Karpathy: "RLVR is the new major stage of the LLM training pipeline"
- Mechanize paying RL environment engineers $500K/year
- Evals = RL Environments (Pash 2025): evaluation is training, and training is evaluation

### 18.6 Verifier and Sandbox Engineering → `chapter23_rl_environments/verifier-sandbox.md`

**Sub-outline:**

- Verifier design principles: correctness, efficiency, anti-gaming
- Formal verifiers vs. heuristic verifiers
- Sandbox engineering: Docker isolation, code-execution sandboxes, network allowlisting, resource quotas
- Managing multi-agent parallel sandboxes
- Long-horizon task harnesses: Anthropic's Effective Harnesses (2025.11), Karpathy's "5-6 agents" pattern

### 18.7 Asynchronous RL and Evaluation Benchmarks → `chapter23_rl_environments/async-eval.md`

**Sub-outline:**

- Synchronous RL training (the traditional mode of veRL, TRL, OpenRLHF)
- The motivation for asynchronous RL training: decoupling rollout from training
- AReaL (Tsinghua + Zhipu, arXiv:2505.24298): staleness-enhanced PPO, 2.77x speedup
- AgentRL (Zhipu + Tsinghua): cross-policy sampling, task advantage normalization
- SLIME / ROLL / LlamaRL / PRIME-RL / TOPLOC + SHARDCAST
- Evaluation benchmarks: CyberGym, SWE-bench, Terminal-Bench, τ-bench, BFCL, WebArena, Vending-Bench, BrowseComp
- Engineering the train-evaluate loop: eval-driven RL, incremental evaluation, data-contamination detection (see Chapter 30 for detail)

### 18.8 Hands-On: GRPO for Financial API Tool Calling → `chapter09_grpo_rlvr/financial-tool-calling-grpo.md`

**Sub-outline:**

- A financial-API dataset
- The GRPO + tool-use training pipeline
- Evaluating tool-calling accuracy

### 18.9 On-Policy Distillation (OPD) → `chapter09_grpo_rlvr/on-policy-distillation.md`

**Sub-outline:**

- The principle behind on-policy distillation
- Combining it with RL training
- Practical gains observed

### 18.10 Hands-On: Code-Generation RL With veRL → `chapter09_grpo_rlvr/verl-code-sandbox.md`

**Sub-outline:**

- Getting started with the veRL framework
- Training RL for code generation
- Sandbox and verifier engineering

---

## Chapter 19 Reasoning Models: From o1 to Claude Opus 4.6 `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `chapter13_reasoning_models/` (6 files; the directory follows the old numbering, 13, inconsistent with chapter number 19 — a legacy artifact)

### 19.1 The Rise of Reasoning Models: From o1 to Reasoning as a Product → `chapter13_reasoning_models/emergence-and-o1.md`

**Sub-outline:**

- The progression from OpenAI o1 → o3 → o4
- Competitive Programming with Large Reasoning Models (OpenAI 2025.02, arXiv:2502.06807)
- Empirical evidence for reasoning ability as an "emergent phenomenon"

### 19.2 The R1-Zero Paradigm: Pure RL With No SFT → `chapter13_reasoning_models/intro.md`

**Sub-outline:**

- DeepSeek-R1-Zero (Nature 2025): running RL directly on the base model
- Reflection, verification, and "aha moments" emerge spontaneously
- Open-source, industrial-grade counterparts to R1-Zero (DAPO / VAPO / Qwen3)
- The complete DeepSeek-R1 training pipeline

### 19.3 Test-Time Compute Scaling → `chapter13_reasoning_models/test-time-scaling.md`

**Sub-outline:**

- The trade-off between test-time compute and train-time compute
- Gemini 3 Pro Deep Think (2025.10) / 3.1 Deep Think (2026.02)
- Parallel-reasoning "thinking layers" stacked on top of an MoE
- IMO 2025 gold medal, 48.4% on HLE, 84.6% on ARC-AGI-2

### 19.4 Hybrid Thinking and Thinking Budgets → `chapter13_reasoning_models/hybrid-thinking.md`

**Sub-outline:**

- A single model that supports both think and non-think modes
- DeepSeek V3.1 (2025.08): fusing hybrid modes
- Qwen3 (arXiv:2505.09388 §4.3): Thinking Mode Fusion + Thinking Budget
- NoThinking + Best-of-N: matching thinking-level performance without thinking (Ma et al., arXiv:2505.18681)
- The engineering implementation of Thinking Budget for controlling reasoning depth
- Long-CoT compression: Kimi k1.5's long2short RL

### 19.5 Adaptive Thinking → `chapter13_reasoning_models/adaptive-thinking.md`

**Sub-outline:**

- Claude Opus 4.6's adaptive thinking depth
- Opus 4.6's internal AI Research Eval Suite (LLM training / Text-RL / Quadruped-RL subtasks, 34x human speedup)
- Anthropic's 2026, 80-page Constitution and reasoning ability

### 19.6 The Readability and Alignment of Reasoning Chains → `chapter13_reasoning_models/cot-visibility-alignment.md`

**Sub-outline:**

- Reasoning alignment
- The engineering trade-off between Hidden CoT and Visible CoT
- Safety filtering of reasoning chains
- Potential deception within Hidden CoT

---

## Chapter 20 Process Reward Models and Inference-Time Search `[multiple articles]` `[v5.1 full restructure]`

> **Design rationale**: three independent research efforts all pointed out that v5's version of this chapter still centers on discriminative PRM, missing the two new main lines of generative and formal approaches.
>
> **Directory**: `chapter14_prm_search/` (7 files; the directory follows the old numbering, 14, inconsistent with chapter number 21 — a legacy artifact)

### 20.1 Outcome Reward vs. Process Reward → `chapter14_prm_search/outcome-vs-process.md`

**Sub-outline:**

- The sparsity problem with outcome reward
- The fine-grained advantage of process reward
- Why PRM is indispensable for long-CoT tasks

### 20.2 Discriminative PRM (the Classic Approach) → `chapter14_prm_search/discriminative-prm.md`

**Sub-outline:**

- OpenAI's "Let's Verify Step by Step" (Lightman et al. 2023, arXiv:2305.20050)
- The PRM800K dataset and human annotation
- PRM as a re-ranking model
- Limitations: high annotation cost, weak generalization

### 20.3 Generative PRM (a New Approach) → `chapter14_prm_search/generative-prm.md`

**Sub-outline:**

- ThinkPRM (arXiv:2504.16828): generative PRM outperforms discriminative PRM
- The key to 100x fewer labels: letting the verifier generate its own critique
- Verifier Compute Scaling
- A PRM survey (arXiv:2510.08049): comparing generative and discriminative approaches

### 20.4 Formal PRM (the Ultimate Verifier) → `chapter14_prm_search/formal-prm.md`

**Sub-outline:**

- Lean4 / Coq as a natural verifier: zero false positives
- AlphaProof (DeepMind 2024.07, IMO silver medal): AlphaZero + Lean
- AlphaGeometry 2 (DeepMind): a dedicated formal approach for geometry problems
- DeepSeek-Prover-V2 (2025.04, arXiv:2504.21801): Lean4 + RL with binary reward, 88.9% on MiniF2F
- The cost of formal PRM: scarce formal-language data, limited domain coverage

### 20.5 Inference-Time Search → `chapter14_prm_search/inference-time-search.md`

**Sub-outline:**

- Beam Search over Thoughts
- MCTS over Thoughts: tree expansion
- Tree of Thoughts (ToT)
- AlphaCodium: search for code generation
- rStar: self-play search
- PaCoRe (Step3-VL-10B, ACL 2026): 16-way parallel rollout aggregation, AIME 2025: 94.4
- GenRM and verifier models: the Generative Reward Model, LLM-as-Judge, Self-Rewarding

### 20.6 Parallel Coordinated Reasoning and Summary → `chapter14_prm_search/parallel-reasoning-and-summary.md`

**Sub-outline:**

- Comparing PaCoRe vs. DeepThink vs. MCTS
- Shifting test-time compute scaling from depth expansion to parallel-breadth expansion
- A decision tree for choosing among the PRM family

---

## Chapter 21 Constitutional AI and RLAIF `[multiple articles]`

> **Directory**: `chapter22_cai_rlvr/`

### 21.1 The Constitutional AI and RLAIF Frameworks → `chapter22_cai_rlvr/intro.md`

**Sub-outline:**

- The Constitutional AI framework (Anthropic 2022, arXiv:2212.08073)
- RLAIF: replacing human annotation with AI feedback
- Self-correction and self-rewarding
- Comparing the cost and effectiveness of CAI vs. RLHF

### 21.2 The HHH Principles and Claude's Practice → `chapter22_cai_rlvr/hhh-practice.md`

**Sub-outline:**

- The HHH alignment principles: Helpful, Harmless, Honest
- CAI's actual use in training Claude
- The engineering significance of Anthropic's 2026, 80-page Constitution
- Callback to Chapter 30's coverage of reward hacking

### 21.3 Engineering RLAIF and the Evolving Constitution → `chapter22_cai_rlvr/rlaif-engineering.md`

**Sub-outline:**

- Prompt engineering for RLAIF and choosing a feedback model
- The evolutionary mechanism behind a large-scale Constitution
- Cost comparison: annotation budgets for RLHF / RLAIF / CAI
- The version history of Anthropic's public Constitution across 2025-2026

---

# Part V · Agentic Reinforcement Learning (4 Chapters, `[the Core New Addition in v5]`)

> **Design rationale**: real 2025-2026 industry demand concentrates here. 60% of the Anthropic Code RL job description centers on agentic work, and OpenAI Operator, Claude Computer Use, and SWE-Agent are all evolving rapidly. A single chapter, as in the original book, is far from enough.
>
> **Merge note**: v5.1 compressed Part V down to 3 chapters — the original Chapter 24, Multi-Turn Interactive RL, was merged into tool calling (multi-turn and tool use are tightly coupled), and the original Chapter 27, Deep Research, was merged into Computer Use (both are agentic applications). **v5.2 splits them back apart**: the original Chapter 24's forced three-way combination of Deep Research + Computer Use + multi-agent has been undone — Deep Research (information retrieval), Computer Use (GUI control), and multi-agent collaboration are three distinct tasks, and forcing them together meant none of them got covered thoroughly. The final v5.2 structure: Chapter 22 (tools + multi-turn + multi-agent) / Chapter 23 (code) / Chapter 24 (Deep Research) / Chapter 25 (Computer Use), 4 chapters total.
>
> **Directory**: the content currently making up Part V is scattered across `chapter10_agentic_rl/` (14 files) and `chapter15_rl_based_swe/` (5 files).

## Chapter 22 Tool Use, Multi-Turn Interaction, and Multi-Agent RL `[multiple articles]`

> **Directory**: currently scattered across `chapter10_agentic_rl/` (multi-turn-rl.md / tool-use-agents.md / tool-use-and-trajectory.md / trajectory-synthesis.md / industrial-practice.md / industrial-evaluation.md / multi-agent-swarm.md)
>
> **Merge note**: the original Chapter 24, Multi-Turn Interactive RL, has been merged into this chapter — multi-turn MDPs and tool calling represent the same modeling and engineering problem within agentic RL (trajectory credit assignment, long-horizon reward, user simulators), and splitting them apart created conceptual duplication. v5.2 further folds multi-agent collaboration / Agent Swarms into this chapter as well — in the LLM era, multi-agent setups are fundamentally an extension of "multi-turn + multi-role" interaction, sharing the same trajectory-credit-assignment framework used for multi-turn MDP modeling in 22.1-22.2.

### 22.1 Modeling Multi-Turn Interaction as an MDP, and Credit Assignment → `chapter10_agentic_rl/multi-turn-rl.md`

**Sub-outline:**

- From single-turn to multi-turn: the fundamental shift in trajectory credit assignment
- Multi-turn MDP modeling: the state includes the full conversation history
- Reward design for long-horizon tasks: sparse vs. dense, the role of process reward (PRM)
- Bridge to Chapter 20's coverage of PRM

### 22.2 User Simulators and Multi-Turn RL Experiments → `chapter10_agentic_rl/multi-turn-rl.md`

**Sub-outline:**

- Designing user simulators: rule-based / model-based / human-in-the-loop
- Engineering differences between multi-turn and single-turn RL: context management, delayed reward
- Experiment: a multi-turn dialogue RL training pipeline
- Evaluation: the τ-bench multi-turn tool-calling benchmark

### 22.3 Introduction to Tool-Calling RL → `chapter10_agentic_rl/tool-use-and-trajectory.md`

**Sub-outline:**

- Expanding the action space for tool use
- Modeling function-calling trajectories
- Designing tool reward: execution outcome + call appropriateness
- The ReAct / ToolFormer paradigms

### 22.4 Search-Augmented RL → `chapter10_agentic_rl/tool-use-agents.md`

**Sub-outline:**

- Search-R1 (arXiv:2503.09516)
- R1-Searcher (arXiv:2503.05592)
- The retrieval-augmented RL training pipeline

### 22.5 Code Interpreter RL and Hands-On Industrial Practice → `chapter10_agentic_rl/industrial-practice.md` + `industrial-evaluation.md`

**Sub-outline:**

- SimpleTIR / ReTool / AFM
- Experiment: training GRPO for tool calling
- Industry evaluation: BFCL, τ-bench

### 22.6 Multi-Agent Collaboration and Agent Swarms → `chapter10_agentic_rl/multi-agent-swarm.md` `[v5.2 new]`

**Sub-outline:**

- LLM-era multi-agent setups vs. classic MARL (differences from Chapter 14's CTDE/MADDPG/MAPPO)
- Anthropic's orchestrator-worker pattern (90.2% speedup)
- Karpathy's "5-6 agents" orchestration pattern
- Agent Swarm products: Kimi K2.5 (2026.01, arXiv:2602.02276), Step 3.7 Flash Advisor Mode
- Self-play multi-agent training (callback to Chapter 32's coverage of self-play)
- Experiment: training a dual-agent collaboration task

---

## Chapter 23 Code Agent Reinforcement Learning `[multiple articles]` `[v5.1 restructure: RL-based as the main line]`

> **Design rationale**: the 2025-2026 main line is RL-based SWE (SWE-RL → CWM → DeepSWE → SSR); the SFT-only paradigm is outdated.
>
> **Directory**: `chapter15_rl_based_swe/` (5 files; the directory follows the old numbering, 15, inconsistent with chapter number 23 — a legacy artifact)

### 23.1 Task Definition and Benchmarks → `chapter15_rl_based_swe/intro.md`

**Sub-outline:**

- SWE-bench: the standard software-engineering task set (Live/Verified)
- SWE-bench-Lite / SWE-bench Multimodal
- Evaluation metrics: Resolved %, Pass@k, edit distance

### 23.2 SWE-RL and Basic Experiments → `chapter15_rl_based_swe/swe-bench-and-rlvr.md`

**Sub-outline:**

- The limitations of first-generation SFT-based SWE agents (SWE-Gym, SWE-Smith)
- SWE-RL (Meta 2025.02, arXiv:2502.18449, NeurIPS'25): 11M GitHub PRs + rule-based reward
- Llama3-70B reaches 41% on SWE-bench Verified; the first observation of an "aha moment"
- rLLM / DeepCoder: industrial-grade RL-based SWE agents

### 23.3 Code World Models and DeepSWE → `chapter15_rl_based_swe/world-model-and-deep-swe.md`

**Sub-outline:**

- Code World Model (CWM, Meta 2025.09, arXiv:2510.02387): 32B dense, 65.8% on SWE-bench
- DeepSWE (Luo et al. 2025)
- The world-model paradigm: the agent learns to "predict the outcome of executing code"

### 23.4 Self-Play SWE-RL and Summary → `chapter15_rl_based_swe/self-play-ssr-and-summary.md` + `meta-swe-rl.md`

**Sub-outline:**

- Self-Play SWE-RL (SSR, Meta 2025.12, arXiv:2512.18552): a single policy playing dual roles
- A self-generating training-data flywheel
- Designing code verifiers (unit tests, code repair, SWE-RM arXiv:2512.21919)
- Long-horizon autonomous engineering ability (Effective Harnesses, progress tracking, test ratchet)
- Code agents from Chinese labs (Qwen3-Coder's 20,000 parallel environments, DeepSeek Coder, CodeGeeX)

---

## Chapter 24 Deep Research and Browser Agents `[multiple articles]` `[v5.2 focused split]`

> **Directory**: `chapter10_agentic_rl/deep-research-agent.md` (split into 3 sections)
>
> **Design rationale**: v5.1 merged Deep Research with Computer Use, but the two tasks are positioned differently — Deep Research is about **information retrieval and synthesis** (multi-step search + answer aggregation), while Computer Use is about **general-purpose GUI control** (screenshots + clicking + desktop automation). Sharing a "click/scroll" action space isn't reason enough to merge them. v5.2 splits them into two focused chapters so each set of methods can be covered thoroughly.

### 24.1 Defining the Deep Research Task and Multi-Step Retrieval → `chapter10_agentic_rl/deep-research-agent.md`

**Sub-outline:**

- Defining the Deep Research task: from single-turn QA to multi-step research
- Scenarios such as open-domain QA, financial research, and academic literature review
- Multi-step retrieval strategies: query rewriting, iterative search, information aggregation
- The fundamental difference from RAG: agency and planning

### 24.2 The Browser-RL Action Space and Harness Engineering → `chapter10_agentic_rl/browser-rl-harness.md`

**Sub-outline:**

- The browser-agent action space: search, click, scroll, extract, go back
- Harness engineering and progress tracking (claude-progress.txt, feature_list.json)
- Memory management for long-horizon tasks: context compression, caching key information
- Reward design: answer correctness, retrieval efficiency, step-count penalty
- Hands-on: financial QA and open-domain QA agents

### 24.3 Evaluation Benchmarks and Open-Source Projects → `chapter10_agentic_rl/deep-research-eval.md`

**Sub-outline:**

- BrowseComp (Meta): a browser-agent benchmark
- xbench-DeepSearch: a deep-research evaluation benchmark
- GAIA: a general AI-assistant benchmark
- Open-source reproduction projects: GPT-Researcher, Stanford STORM, OpenResearcher
- End-to-end experiment: from a browser environment to training a Deep Research agent

---

## Chapter 25 Computer Use and GUI Agents `[multiple articles]` `[v5.2 new chapter split]`

> **Directory**: `chapter28_computer_use/`
>
> **Design rationale**: split out from the original Chapter 24. Computer Use is a major 2025-2026 direction (Anthropic Computer Use, OpenAI Operator, Google Project Mariner, ByteDance's UI-TARS-2, Zhipu's AutoGLM), and deserves its own chapter to thoroughly cover the modeling, training, and safety of GUI control.

### 25.1 The Computer Use Paradigm and GUI Grounding RL → `chapter28_computer_use/intro.md`

**Sub-outline:**

- The Computer Use paradigm (Anthropic Computer Use / OpenAI Operator / Google Project Mariner)
- The core action space: clicking, scrolling, typing, screenshots
- GUI Grounding RL: Set-of-Mark, visual grounding, action mapping
- Aligning visual understanding with action
- Differences across desktop, mobile, and Web environments

### 25.2 Hands-On GUI Agent Training → `chapter28_computer_use/training.md`

**Sub-outline:**

- UI-TARS-2 (ByteDance Seed 2025.09, arXiv:2509.02544): Multi-Turn RL + asynchronous rollouts + stateful envs + value pretraining
- AutoGLM / Open-AutoGLM (Zhipu 2025.12): self-evolving online curriculum RL
- MobileRL (arXiv:2509.18119) / ComputerRL (arXiv:2508.14040)
- CogAgent (Zhipu)
- Synthesizing training data: trajectory bootstrapping, collecting human demonstrations

### 25.3 Instruction Hierarchy and Defending Against Prompt Injection → `chapter28_computer_use/safety-swarm.md`

**Sub-outline:**

- Instruction hierarchy (OpenAI 2024.04, arXiv:2404.13208): permission levels across system/developer/user/tool instructions
- GPT-5 Mini-R using instruction hierarchy as an RL reward (+0.11~0.21)
- Defending against prompt injection, and the "kernel mode" analogy
- Safety challenges specific to the GUI context: malicious webpages, spoofed UIs, cross-app attacks
- Callback to Chapter 30's coverage of reward hacking (an agent being hijacked is a special form of reward hacking)

---

# Part VI · Multimodal Reinforcement Learning (4 Chapters `[v5.1 adds RL for Visual Generation]`)

## Chapter 26 RL for Vision-Language Models `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `chapter11_vlm_rl/` (8 files; the directory follows the old numbering, 11, inconsistent with chapter number 25 — a legacy artifact)

### 26.1 Foundations of VLM RL Training → `chapter11_vlm_rl/intro.md` + `vlm-frameworks.md`

**Sub-outline:**

- Joint vision-language representations
- Sources of multimodal reward signals
- Handling visual tokens vs. text tokens in RL
- Training frameworks: EasyR1 / R1-V / Open-Vision-Reasoner / Perception-R1

### 26.2 Visual Reward and Challenges → `chapter11_vlm_rl/vlm-challenges.md`

**Sub-outline:**

- Visual-QA correctness reward
- Visual-caption completeness reward
- Visual-hallucination penalty
- The "missing trace" problem in visual reasoning

### 26.3 Visual Reflection RL `[v5.1 new]` → `chapter11_vlm_rl/qwen3-vl-reflection.md`

**Sub-outline:**

- Qwen3-VL (2025.11.26): reflection-driven visual re-attention
- Self-correction of visual grounding
- Combining reflection mechanisms with RL training

### 26.4 China's Multimodal Frontier → `chapter11_vlm_rl/vlm-grpo-hands-on.md`

**Sub-outline:**

- Step3-VL-10B (arXiv:2601.09668): 1000+ RL iterations
- GLM-4.6V: RLCS curriculum sampling
- Seed1.5-VL (ByteDance, arXiv:2505.07062): a 20B-A200B MoE, GUI-agent + game RL
- Comparing PaCoRe's 16-way parallel rollout aggregation with MCTS over Thoughts

### 26.5 Experiment: GeoQA Geometric Reasoning → `chapter11_vlm_rl/easyr1-geoqa.md`

**Sub-outline:**

- Getting started with the EasyR1 framework
- The GeoQA dataset
- The VLM GRPO training pipeline

---

## Chapter 27 Audio and Speech RL `[multiple articles]` `[v5.1 expansion]`

> **Design rationale**: v5's version of this chapter is only outline bullet points, missing core methods like Step-Audio's MGRD.
>
> **Directory**: `chapter30_audio_rl/`

### 27.1 Overview of Audio Language Models and the Step-Audio Series → `chapter30_audio_rl/intro.md`

**Sub-outline:**

- Audio tokenization schemes (codec, semantic/acoustic separation)
- Differences between speech generation and text generation
- Engineering challenges of real-time inference
- The Step-Audio series `[a direction unique to China]`: Step-Audio-R1 (arXiv:2511.15848)
- MGRD (Modality-Grounded Reasoning Distillation)
- Acoustic-Grounded Reasoning / Mind-Paced Speaking / the Dual-Brain Architecture

### 27.2 The RLVR → RLHF Evolution and Designing Audio Rewards → `chapter30_audio_rl/reward-design.md`

**Sub-outline:**

- Step-Audio-R1.5: shifting from RLVR to RLHF for Audio Reasoning
- Multi-objective RL balancing vocal naturalness and reasoning ability
- Preserving natural prosody
- Designing audio rewards: content correctness / prosodic naturalness (modeling human preference) / real-time performance
- Experiment: a simple spoken-dialogue RL task

### 27.3 Multimodal Audio Agents and Future Directions → `chapter30_audio_rl/future.md`

**Sub-outline:**

- Real-time speech agents: GPT-4o Voice, GLM-4.6 Voice, Step-Audio-2
- Emotion-aware and vocal-style-control RL
- Unifying with VLMs: joint audio-visual RL
- Comparing audio-RL approaches across Chinese labs

---

## Chapter 28 Embodied Intelligence and VLA Models `[multiple articles]` `[v5.1 upgraded flagship case study]`

> **Design rationale**: the old version's use of RT-2 is outdated; the new benchmarks are Gemini Robotics 1.5 + π0 + Embodied Thinking.
>
> **Directory**: `chapter12_future_trends/embodied-intelligence/` (the directory follows the old numbering, 12, inconsistent with chapter number 31 — a legacy artifact)

### 28.1 Overview of VLA Models → `chapter12_future_trends/embodied-intelligence/index.md`

**Sub-outline:**

- π0 (Physical Intelligence 2024): diffusion policy + VLM
- RT-2 (Google 2023, treated as historical background)
- OpenVLA (the open-source flagship)
- Gemini Robotics 1.5 (DeepMind 2025.09, the flagship model): a dual-model VLA + ER setup, the Embodied Thinking paradigm, cross-embodiment transfer (Apollo / Spot)

### 28.2 Foundations of Robot Learning

**Sub-outline:**

- Observation space (vision, proprioception, force sensing)
- Action space (joint angles, end-effector pose)
- Reward function design

### 28.3 Diffusion Policy and Multimodal Fusion

**Sub-outline:**

- Diffusion models as policies
- Multimodal action distributions
- Vision-language-action tokenization and cross-modal alignment

### 28.4 Sim-to-Real and Teleoperation

**Sub-outline:**

- Domain randomization
- Sim-to-real transfer techniques
- System identification
- Collecting human demonstrations, behavior-cloning pretraining, RL fine-tuning

### 28.5 Experiment: Fine-Tuning OpenVLA With RL for a Tabletop Grasping Task

---

## Chapter 29 RL for Visual Generation `[multiple articles]` `[v5.1 entirely new chapter]`

> **Design rationale**: ByteDance Seed is the single largest source of innovation in video-generation RL for 2025-2026. DanceGRPO adapts GRPO for diffusion, Seedance uses multi-dimensional RLHF, and LongCat-Video stacks multiple rewards — a globally leading direction from Chinese labs.
>
> **Directory**: `chapter11_vlm_rl/visual-generation-rl.md` + `video-generation-modern.md` (currently housed under the VLM directory; recommend splitting it out)

### 29.1 Defining Visual Generation Tasks → `chapter11_vlm_rl/visual-generation-rl.md`

**Sub-outline:**

- Text-to-Video
- Image-to-Video
- Video editing and continuation
- Foundations of Diffusion + RL: diffusion models as policy networks, adapting RL for Rectified Flow, fundamental differences from text-LLM RL

### 29.2 DanceGRPO `[a ByteDance Seed innovation]` → `chapter11_vlm_rl/visual-generation-rl.md`

**Sub-outline:**

- DanceGRPO (2025.05, arXiv:2505.07818): adapting GRPO to diffusion/flow
- The core idea: treating a diffusion step as an RL timestep
- Comparison with earlier methods like DDPO
- Unified across 4 foundation models

### 29.3 Multi-Reward Video RLHF → `chapter11_vlm_rl/video-generation-modern.md`

**Sub-outline:**

- Seedance 1.0 (ByteDance, arXiv:2506.09113): foundational reward / motion reward / aesthetic reward / refiner RLHF
- LongCat-Video (ByteDance 2025.10, arXiv:2510.22200): GRPO + multi-reward stacking, LoRA stacking
- Reward models for video generation: VisionReward, multi-dimensional reward decomposition, human-preference alignment

### 29.4 Physics-Aware Video Generation and Experiment

**Sub-outline:**

- Hailuo-02 (MiniMax, a physics-aware NCR architecture)
- Physical laws as intrinsic reward
- Temporal-consistency constraints
- Experiment: training a simple video-generation model with DanceGRPO

---

# Part VII · Safety, Evaluation, and Research Frontiers (3 Chapters) `[v5.2 merges the original Part VII + VIII]`

> **Merge note**: in v5.1, Part VII (safety/evaluation) had only 1 chapter, making it an awkward standalone Part. v5.2 merges the original Part VIII (research frontiers, 2 chapters) into Part VII, renamed "Safety, Evaluation, and Research Frontiers," for 3 chapters total. The original Chapter 35, RL Evaluation Methodology, has been merged into Chapter 30 (Reward Hacking); the original Chapter 34, Scalable Oversight and Red-Teaming (pure AI-safety philosophy/process, not RL technique), has been removed; the original Chapter 36, Distributed Training, has been merged into Chapter 16.

## Chapter 30 Reward Hacking and RL Evaluation `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `chapter16_alignment_failures/` (5 files) + `chapter35_rl_evaluation/`

### 30.1 Classic Failure Modes → `chapter16_alignment_failures/classical-failures.md` + `intro.md`

**Sub-outline:**

- A complete taxonomy of reward hacking: specification gaming / reward tampering / Goodhart's Law
- Anthropic's 2025.11 taxonomy (arXiv:2511.18397)

### 30.2 RLVR's "Spurious Gains" `[v5.1 new]` → `chapter16_alignment_failures/modern-incidents.md`

**Sub-outline:**

- Empirical evidence of data contamination (arXiv:2507.10532, AAAI 2026): Qwen's "spurious reward RLVR" gains on MATH-500 come primarily from data contamination
- How GRPO's clipping bias activates memorization
- Methodology for assessing the genuine gains from RLVR
- Contamination-resistant evaluation design

### 30.3 Industry Failure Cases `[v5.1 new]`

**Sub-outline:**

- The GPT-4o sycophancy rollback (OpenAI 2025.04-05): user-feedback reward diluted the primary safety reward, rolled back after 48 hours
- ByteDance Seed's RLHF data scaling: reward hacking and diversity decay, the Pre-PPO prompt-selection strategy

### 30.4 Anthropic's Misalignment Research → `chapter16_alignment_failures/sleeper-and-faking.md`

**Sub-outline:**

- School of Reward Hacks (Gao et al. 2025.08)
- Naturally emergent misalignment (Anthropic 2025.11, arXiv:2511.18397): HHH reward as a mitigation
- Sleeper Agents (Hubinger et al. 2024.01, arXiv:2401.05566)
- Alignment Faking (Greenblatt et al. 2024.12, arXiv:2412.14093)
- In-Context Scheming (Apollo 2024.12, arXiv:2412.04984)
- Sycophancy to Subterfuge (Anthropic 2024, arXiv:2406.10162)
- METR: Frontier Models Reward Hacking (Von Arx et al. 2025)

### 30.5 Defense Mechanisms and Summary → `chapter16_alignment_failures/scaling-and-defenses.md`

**Sub-outline:**

- Preference models and reward-hack classifiers
- By-construction prevention of hacking through architecture
- Ensembles of multiple verifiers
- Formal verification as the ultimate line of defense (see Chapter 20)

### 30.6 Evaluation Principles and Contamination Robustness → `chapter35_rl_evaluation/intro.md`

**Sub-outline:**

- Principles of benchmark design
- Detecting contamination and leakage (callback to section 29.2's coverage of RLVR's spurious gains)
- Prompt-sensitivity analysis
- Out-of-distribution robustness
- Behavioral evaluation vs. capability evaluation
- Challenges in evaluating long-horizon tasks

### 30.7 Modern Evaluation Harnesses and Internal Benchmarks → `chapter35_rl_evaluation/harness.md`

**Sub-outline:**

- Standardized evaluation harnesses: lm-eval-harness, BigCode Eval, τ-bench, BFCL
- Anthropic's internal AI Research Eval Suite (Opus 4.6: LLM training / Text-RL / Quadruped-RL subtasks, 34x human speedup)
- Claude 4.6's self-evaluation and adversarial baselines
- Dynamic benchmarks such as LiveCodeBench and SWE-bench Verified

---

---

## Chapter 31 Evolutionary LLM Search and Generative World Models `[multiple articles]` `[v5.1 entirely new]`

> **Design rationale**: v5 is completely missing AlphaEvolve and Genie 3, the two most cutting-edge 2025-2026 directions.
>
> **Directory**: `chapter12_future_trends/` (some content already lives in llm-driven-discovery.md)

### 31.1 The AlphaEvolve Paradigm → `chapter12_future_trends/llm-driven-discovery.md`

**Sub-outline:**

- AlphaEvolve (DeepMind 2025.05): LLM proposes a diff, an automated evaluator scores it, and an evolutionary algorithm selects among candidates
- The first discovery of a 23% speedup in matrix multiplication
- Improvements to 50+ open math problems
- AlphaEvolve's algorithmic architecture: evolutionary search + LLM proposal
- Differences from traditional RL: not policy gradient, but search + LLM
- A new paradigm for search algorithms in the LLM era

### 31.2 Generative World Models as RL Environments

**Sub-outline:**

- Genie 3 (DeepMind 2025.08): a real-time, interactive world model, 720p/24fps generation, world memory with multi-minute consistency
- Generative environments vs. real environments
- An unlimited RL training curriculum: agents learning inside generated worlds
- A foundation for a general AGI world model

### 31.3 Recursive Self-Improvement

**Sub-outline:**

- Anthropic-Funded Research / Recursive Self-Improvement (2026.04): Claude conducting its own AI research, a 52x speedup on an internal benchmark
- The "Claude Mythos Preview" model
- The ultimate vision: training AI with RL to do AI research

---

## Chapter 32 Self-Play, Scaling Trends, and Future Directions `[multiple articles]`

> **Directory**: `chapter12_future_trends/` (some content already lives in self-play-outlook/ and rl-scaling-outlook.md)

### 32.1 Foundations of Self-Play and LLM Self-Play → `chapter12_future_trends/self-play-outlook/index.md`

**Sub-outline:**

- The progression AlphaGo → AlphaZero → MuZero
- The convergence properties of self-play
- Applications of self-play in Go / chess / StarCraft
- LLM self-play and SPIN
- Self-Play SWE-RL (SSR) (see Chapter 23 for detail)
- Multi-agent debate as self-play
- Mode collapse and preserving diversity

### 32.2 RL Scaling Laws and Foundation-Model RL → `chapter12_future_trends/rl-scaling-outlook.md`

**Sub-outline:**

- RL scaling laws (by analogy with Chinchilla)
- Reward signal vs. data volume vs. model scale
- The scaling limits of RLVR
- The foundation model as RL's starting point
- A unified view of RLHF / RLVR / RLAIF / Agent RL
- The future shape of foundation-model RL

### 32.3 In-Context RL and the Next Decade → `chapter12_future_trends/llm-multi-agent-rl/index.md`

**Sub-outline:**

- In-context RL and Algorithm Distillation (DeepMind 2022)
- Meta-learning and continual learning
- Karpathy's reflection that "AGI is still a decade away"
- Open problems: credit assignment, long-horizon planning, generalization, safety
- Divergent paths between Chinese and American labs
- The leap from conversational models to autonomous agents

---

# Appendices (8 Sections)

## Appendix A · Training Debugging Handbook `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `appendix_common_pitfalls/`

### A.1 Numerical Stability and Crash Diagnostics → `appendix_common_pitfalls/intro.md`

**Sub-outline:**

- Diagnosing common training crashes
- Detecting gradient anomalies
- Handling KL-divergence explosions
- A checklist for reproducing training crashes

### A.2 Optimizer and System Stability → `appendix_common_pitfalls/optimizer-stability.md`

**Sub-outline:**

- MuonClip + QK-clip optimizer stability (Kimi K2)
- Router interference in MoE + RL training
- Tuning staleness in asynchronous RL training

### A.3 An Agent / Long-Trace Troubleshooting Checklist → `appendix_common_pitfalls/agentic-failure.md`

**Sub-outline:**

- Early signals of reward hacking (in its agent-specific forms)
- A troubleshooting checklist for function-call parsing failures
- A decision tree for diagnosing OOM in long trajectories

---

## Appendix B · Reinforcement Learning Engineering Practice `[multiple articles]` `[v5.1 major expansion]`

> **Directory**: `appendix_industrial_training/`

### B.1 Synchronous and Asynchronous Training Systems → `appendix_industrial_training/intro.md`

**Sub-outline:**

- The foundation of synchronous RL training systems (veRL, TRL)
- Asynchronous RL training systems (AReaL, AgentRL, SLIME, ROLL, LlamaRL)
- Engineering implementations of staleness and cross-policy sampling

### B.2 Agent Sandbox and Evaluation Engineering → `appendix_industrial_training/agentic-rl-infra.md`

**Sub-outline:**

- Agent sandbox engineering
- Evaluation-benchmark engineering
- Resource management for long-trajectory rollouts

### B.3 A Metrics Dictionary and Hands-On Exercises → `appendix_industrial_training/metrics-exercises.md`

**Sub-outline:**

- A dictionary of training metrics
- MoE + RL training engineering (DeepSeek V3, Step Flash, GLM-4.5)
- Hands-on industrial exercises

---

## Appendix C · Core Algorithm Implementations `[multiple articles]` `[v5.1 expansion]`

> **Directory**: `appendix_code_cheatsheet/`

### C.1 Implementing SFT / PPO / DPO → `appendix_code_cheatsheet/intro.md`

**Sub-outline:**

- Implementing SFT and KL divergence
- Implementing PPO and GAE
- Implementing the DPO family

### C.2 Implementing the GRPO Family and RPT → `appendix_code_cheatsheet/grpo-family.md`

**Sub-outline:**

- A basic GRPO implementation
- Implementing the GRPO improvement family (DAPO, Dr.GRPO, GSPO, CISPO, VAPO)
- Implementing RPT (Reinforcement Pre-Training)
- Adapting DanceGRPO for diffusion/flow

### C.3 Implementing Sampling, Attention, and Optimizers → `appendix_code_cheatsheet/numerical.md`

**Sub-outline:**

- Implementing softmax and cross-entropy
- Implementing sampling methods (top-k, top-p, min-p)
- Implementing attention mechanisms (MHA, GQA, MLA, DSA sparse attention)
- Implementing the MuonClip + QK-clip optimizer
- Implementing PRM training (discriminative + generative + formal Lean4)

---

## Appendix D · Learning Resources and Reproduction Projects `[multiple articles]`

> **Directory**: `appendix_resources/`

### D.1 An Index of Papers and Courses → `appendix_resources/intro.md`

**Sub-outline:**

- A must-read paper list (organized by topic, 100+ papers)
- An index of video courses (CS285, CS234, the Hugging Face Course)

### D.2 Open-Source Reproduction Projects → `appendix_resources/open-projects.md`

**Sub-outline:**

- An index of open-source code repositories (veRL / OpenRLHF / TRL / trl-X)
- Recommended reproduction projects (Sky-T1, Open-R1, Tülu 3)
- Tracking open-source reproductions from Chinese labs

---

## Appendix E · Mathematical Foundations `[multiple articles]`

> **Directory**: `appendix_math/`

### E.1 Linear Algebra and Probability/Statistics → `appendix_math/probability-linear.md`

**Sub-outline:**

- Linear algebra (Bellman matrices, function approximation, convergence)
- Probability and statistics (return, value, sampling estimation, GAE)

### E.2 Calculus and Information Theory → `appendix_math/calculus-information.md`

**Sub-outline:**

- Calculus and optimization (gradients, PG, PPO, Adam)
- Information theory (entropy, KL divergence, cross-entropy, mutual information)

---

## Appendix F · A Paper-Reading Roadmap `[multiple articles]` `[new]`

> **Directory**: `appendix_paper_reading/`

### F.1 Must-Reads in Classical and Deep RL → `appendix_paper_reading/classical-deep-rl.md`

**Sub-outline:**

- Must-reads in classical RL (Sutton, Watkins, Mnih)
- Must-reads in deep RL (DQN, A3C, PPO, SAC)

### F.2 Must-Reads in LLM RL and Safety Research → `appendix_paper_reading/llm-rl-safety.md`

**Sub-outline:**

- Must-reads in LLM RL (InstructGPT, CAI, DPO, GRPO, R1)
- Must-reads in safety research (Sleeper Agents, Alignment Faking, Reward Hacking)
- 2025-2026 frontier reading (DAPO, GSPO, CISPO, PRM, PaCoRe)

---

## Appendix G · A GPU-Hour Estimation Table `[multiple articles]` `[new]`

> **Directory**: `appendix_gpu_hours/`

### G.1 Pretraining and Post-Training Costs → `appendix_gpu_hours/intro.md`

**Sub-outline:**

- Pretraining costs at different model scales
- Costs for the SFT / RLHF / RLVR stages
- Public training-data references from DeepSeek / Qwen / Step

### G.2 Budget Planning for Self-Training → `appendix_gpu_hours/budget-planning.md`

**Sub-outline:**

- Budget planning for training your own model
- Buying compute vs. renting cloud capacity
- Per-GPU-hour costs across different RL paradigms

---

## Appendix H · Notation and Algorithm Index `[multiple articles]` `[new]`

> **Directory**: `appendix_terminology/` (already exists)

### H.1 Notation and Abbreviation Tables → `appendix_terminology/intro.md`

**Sub-outline:**

- A unified notation table for the whole book
- A table of abbreviations (RLHF, RLVR, PRM, CAI...)

### H.2 Algorithm Index → `appendix_terminology/algorithm-index.md`

**Sub-outline:**

- An index of algorithm names (GRPO, PPO, DPO, SAC...)
- A diagram of algorithm-family relationships (PPO → GRPO → DAPO / GSPO / CISPO / VAPO)
- A cross-reference table between algorithms and chapters

---

# Full Chapter Statistics `[v5.1 revision]`

> **v5.1 (0622) merge log**: the original 38 chapters were compressed to 31 through 7 merges —
> ① Chapter 1 (Overview) → the preface (duplicate content)
> ② Chapter 23 (Verifiers) → Chapter 18 (the GRPO family; verifiers are an RLVR component)
> ③ Chapter 24 (multi-turn interaction) → Chapter 22 (tool use; tightly coupled)
> ④ Chapter 27 (Deep Research) → Chapter 24 (Computer Use; shared GUI-RL modeling)
> ⑤ Chapters 34+35 (Scalable Oversight + RL evaluation) → Chapter 30 (reward hacking; same subject)
> ⑥ Chapter 36 (distributed training) → Chapter 16 (LLM RL in industry; massive overlap)
>
> **v5.2 reverse-split log**: the original Chapter 24's forced three-way combination (Deep Research + Computer Use + multi-agent) has been undone —
> ⑦ Deep Research → a standalone Chapter 24 (information retrieval and browser RL)
> ⑧ Computer Use → a standalone Chapter 25 (GUI control and instruction-hierarchy safety)
> ⑨ Multi-agent collaboration / Agent Swarms → folded into section 22.6 of Chapter 22 (LLM-era multi-agent setups are fundamentally multi-turn + multi-role interaction)
> Part V has grown from 3 chapters to 4, taking the whole book from 31 chapters to 32.

| Part       | Topic                                       | Chapter count   | Chapter type                            |
| ---------- | ------------------------------------------- | --------------- | --------------------------------------- |
| 0          | Preface · Introduction                      | 7 sections      | Multiple articles (3 files)             |
| I          | Foundations and Classical RL                | 6               | 6 multi-article                         |
| II         | Deep RL                                     | 5               | 5 multi-article                         |
| III        | Advanced RL Methods                         | 3               | 3 multi-article                         |
| IV         | LLM Alignment and Post-Training             | 7               | 7 multi-article                         |
| V          | **Agentic RL**                              | **4**           | 4 multi-article                         |
| VI         | **Multimodal RL (incl. Visual Generation)** | **4**           | 4 multi-article                         |
| VII        | Safety, Evaluation, and Research Frontiers  | **3**           | 3 multi-article                         |
| **Total**  |                                             | **32 chapters** | **0 single-article + 32 multi-article** |
| Appendices | A-H                                         | 8 sections      | 0 single-article + 8 multi-article      |

---

# Comparison With the Existing Book `[v5.1 revision]`

| Dimension                 | Current book                      | **v5.1 Final**                                                                                      |
| ------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------- |
| Total chapter count       | 12                                | **32**                                                                                              |
| Preface                   | Philosophical discussion up front | **0.1: hands-on CartPole first + a glimpse of what's ahead**                                        |
| Agentic content           | 1 chapter, shallow                | **4 chapters, in depth + Chapter 25: instruction hierarchy / UI-TARS-2 / K2.5**                     |
| Multimodality             | 1 chapter, shallow                | **4 chapters (VLM / audio / VLA / visual generation)**                                              |
| Safety/alignment research | 0                                 | **1 merged chapter (reward hacking + evaluation)**                                                  |
| GRPO family               | Only DAPO                         | **Algorithmic detail on 6+ variants (DAPO / Dr.GRPO / GSPO / CISPO / VAPO) + verifier engineering** |
| PRM                       | Mostly discriminative             | **Generative (ThinkPRM) + formal (Lean4 / AlphaProof)**                                             |
| RL Environments           | 0                                 | **Merged into Chapter 18, the GRPO family + asynchronous RL (AReaL / AgentRL)**                     |
| RL for visual generation  | 0                                 | **A standalone chapter (DanceGRPO / Seedance / LongCat)**                                           |
| Engineering systems       | Appendix                          | **1 chapter of main text + expanded appendices**                                                    |
| Hands-on code             | Partial coverage                  | **A lab in every chapter**                                                                          |
| Coverage of Chinese labs  | DeepSeek only                     | **Full coverage: DeepSeek / Qwen / Kimi / Zhipu / Step / ByteDance / MiniMax**                      |
| Real paper citations      | None                              | **Every topic has an arXiv number + an official URL**                                               |
| Frontier directions       | 0                                 | **AlphaEvolve / Genie 3 / recursive self-improvement / RPT**                                        |
| **Heading hierarchy**     | **Confused (X.Y ambiguity)**      | **Strict three-tier: Part / chapter / article / sub-outline**                                       |

---

# Rollout Recommendations `[v5.1 update]`

**Phase 1 (immediate, zero risk)**: restructure the preface + turn headings into proper textbook form + split Chapter 3 (MDP)

**Phase 2 (this month, P0)**: fill in the core of Part IV

- Fully restructure Chapter 18, the GRPO family + verifier engineering (DAPO / Dr.GRPO / GSPO / CISPO / VAPO / RPT)
- Add Hybrid Thinking + long2short + emergence evidence to Chapter 19 (Reasoning)
- Upgrade Chapter 20 (PRM) with generative and formal approaches

**Phase 3 (next quarter, P0)**: fill in Part V (Agentic)

- Restructure Chapter 23's code agents around RL-based SWE as the main line
- Split Chapter 24 (Deep Research) and Chapter 25 (Computer Use) into their own chapters
- Add section 22.6, Multi-Agent Collaboration and Agent Swarms, to Chapter 22

**Phase 4 (second half of the year, P0)**: fill in Part VI (Multimodal)

- Deepen Chapter 27's audio RL coverage with MGRD
- Upgrade Chapter 28's VLA coverage to Gemini Robotics 1.5
- Add Chapter 29, RL for visual generation (DanceGRPO / Seedance) `[entirely new]`

**Phase 5 (ongoing, P1)**: Part VII, safety + frontier

- Add the GPT-4o rollback / data contamination / Seed scaling to Chapter 30 (the merged Reward Hacking + Evaluation chapter)
- Add Chapter 31: AlphaEvolve / Genie 3 / recursive self-improvement `[entirely new]`

**Phase 6 (long-term, P1-P2)**: expand the appendices

- Add MuonClip + QK-clip to Appendix A
- Add asynchronous RL systems to Appendix B
- Add DanceGRPO / RPT / PRM implementations to Appendix C

**Phase 7 (ongoing)**: renaming legacy directories (optional)

- `chapter01_cartpole/` → `chapter02_cartpole/`
- `chapter02_dpo/` → merged into `chapter18_dpo/`
- `chapter03_mdp/` → split into `chapter04_mdp/`, `chapter05_value/`, `chapter06_dp_mc_td/`, `chapter07_q_learning/`
- ...(see the "legacy artifact" notes in each chapter for detail)

---

# Quick Reference: Key Paper Citations `[v5.1 update]`

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

# A Final, Honest Statement to the Reader

1. **38 chapters is a massive undertaking**, but that's the scale an MIT-level textbook should have (Sutton & Barto has 14 chapters, CS285 has 11 lectures, and this book additionally covers LLM / Agentic / Multimodal content).
2. **This is not "read it and land the job"** — real job descriptions also require SE engineering ability, production debugging, distributed-systems experience, and product sense. This book covers the knowledge component.
3. **Every chapter should have a lab / experiment** — true hands-on practice is what makes the material stick. Some of this already exists in the book's `code/` directory, and it needs to be expanded.
4. **Continuous updates**: 2026 will bring new papers and new models; this book needs small revisions every quarter and a major revision every year.
5. **Writing-effort estimate**: 38 chapters × roughly 3,000-5,000 words per chapter + code ≈ 150,000-200,000 words plus a large volume of code. Estimated at 6-12 months of full-time work.
6. **The core fix in this revision (0622)**: enforcing a strict three-tier heading hierarchy (Part / chapter / article / sub-outline), eliminating the old version's X.Y numbering ambiguity, so that "sections within a single-article chapter" and "independent articles within a multiple-article chapter" are visually completely distinguishable.
