# C.1 Learning Resources and Project Roadmap

> After completing the first 17 chapters of this book, readers have acquired all the mathematical and engineering foundations necessary to read cutting-edge reinforcement learning (RL) papers. However, the RL literature is vast—arXiv adds thousands of papers each year, but only a handful are truly direction-defining and worth rereading repeatedly. This appendix organizes these "must-read papers" by theme, with each paper annotated with a **one-sentence summary of its core contribution** and a **reading priority** (★ Must-read / ◆ Recommended / ◇ Extension), helping readers build a complete map of the literature from classical RL to the forefront of research in 2026.
>
> Reading suggestions: Build theoretical foundations in the order of C.1.1 → C.1.2 → C.1.3, then select relevant papers in C.1.4 and C.1.5 based on research interests. Each paper should be read again after hands-on implementation (see [Appendix A](../appendix_industrial_training/training-debugging), [Appendix B](../appendix_code_cheatsheet/sft-kl)), at which point readers will discover engineering details that were previously overlooked during the first read.

## C.1.1 Essential Reading in Classical RL

The value of classical literature lies not in the algorithms themselves—TD-Learning and Q-Learning are now textbook knowledge. Their true value lies in the **foundational proofs**: formalizing, analyzing, and proving the concept of "an agent learning in an environment." Reading these papers is the fastest way to understand the meaning behind all the mathematical notation in modern RL.

| Priority | Paper                                                                                                        | Core Contribution                                                                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★        | Sutton & Barto. _Reinforcement Learning: An Introduction_ (2018, 2nd ed.)                                    | The "bible" of RL. It covers MDP, Bellman equations, TD learning, policy iteration, and eligibility traces. It provides a complete theoretical framework, and all modern RL papers assume the reader is familiar with its notation.         |
| ★        | Watkins, Dayan. _Q-Learning_ (Machine Learning, 1992)                                                        | Proves the convergence of off-policy TD control $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ in the tabular case, laying the theoretical foundation for subsequent DQN/Rainbow.                             |
| ★        | Tesauro. _TD-Gammon, a Self-Teaching Backgammon Program_ (Neural Computation, 1995)                          | The first time a neural network combined with TD-Learning reached expert-level performance (Backgammon), proving that $\text{TD}(\lambda)$ combined with function approximation could learn policies beyond the training data distribution. |
| ★        | Sutton. _Learning to Predict by the Methods of Temporal Differences_ (Neural Computation, 1988)              | The foundational paper on TD learning, introducing $\text{TD}(0)$, $\text{TD}(\lambda)$, and eligibility traces, which are the origins of value estimation.                                                                                 |
| ★        | Mnih et al. _Human-level control through deep reinforcement learning_ (Nature, 2015)                         | DQN: Experience replay and target networks enabled CNNs to reach human-level performance on 49 Atari games. This is a foundational engineering paper in deep RL.                                                                            |
| ★        | Silver et al. _Mastering the game of Go with deep neural networks and tree search_ (Nature, 2016)            | AlphaGo: A combination of policy networks, value networks, and MCTS defeated Lee Sedol, proving the breakthrough of RL in Go—a game long considered "impossible" for AI to solve.                                                           |
| ◆        | Williams. _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning_ (1992) | REINFORCE: The original derivation of the policy gradient theorem, the ancestor of all on-policy algorithms (A2C, PPO, GRPO).                                                                                                               |
| ◆        | Schulman et al. _Trust Region Policy Optimization_ (ICML, 2015)                                              | TRPO: Ensures monotonic policy updates using KL divergence, the direct predecessor of PPO.                                                                                                                                                  |
| ◆        | Kearns, Singh. _Bias-Variance Error Bounds for Temporal Difference Updates_ (2000)                           | Explains the role of $\lambda$ in $\text{TD}(\lambda)$ from the perspective of bias-variance trade-off, which is crucial for understanding GAE (see [Chapter 8](../chapter10_ppo/gae-reward-model)).                                        |
| ◇        | Bertsekas, Tsitsiklis. _Neuro-Dynamic Programming_ (1996)                                                    | An early monograph combining DP with function approximation. Mathematically rigorous but dense, suitable for theoretical reference.                                                                                                         |

## C.1.2 Essential Reading on Deep Reinforcement Learning

After 2013, RL entered the era of deep learning. This set of papers defined the standard method stack for modern deep RL: experience replay, target networks, actor-critic, maximum entropy, and model learning. Readers should focus on mastering the **algorithm skeleton** and **ablation experiments** of each paper— the former determines whether one can reproduce the results, and the latter determines whether one can improve upon them.

| Priority | Paper                                                                                                                                     | Core Contribution                                                                                                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ★        | Mnih et al. _Playing Atari with Deep Reinforcement Learning_ (arXiv:1312.5602, 2013)                                                      | Early version of DQN (NIPS Workshop), first combined CNN with Q-Learning, initiating the era of deep RL.                                                                                               |
| ★        | Mnih et al. _Human-level Control through Deep Reinforcement Learning_ (Nature, 2015)                                                      | Complete version of DQN: experience replay + target network, achieving human-level performance on 49 Atari games.                                                                                      |
| ★        | Mnih et al. _Asynchronous Methods for Deep Reinforcement Learning_ (A3C, arXiv:1602.01783, 2016)                                          | Asynchronous actor-critic: parallel sampling with multiple threads to eliminate correlation, without experience replay; A2C is its synchronous version, still a baseline today.                        |
| ★        | Schulman et al. _Proximal Policy Optimization Algorithms_ (arXiv:1707.06347, 2017)                                                        | PPO: clip instead of TRPO's second-order optimization, engineering-friendly and stable training. LLMs are trained with PPO directly. See [Chapter 8](../chapter10_ppo/ppo-clip-objective).             |
| ★        | Lillicrap et al. _Continuous Control with Deep Reinforcement Learning_ (DDPG, arXiv:1509.02971, 2015)                                     | Extended DPG to deep networks, deterministic policy gradient + experience replay, the foundational work for continuous control. See [Chapter 9](../chapter11_continuous_control/ddpg).                 |
| ★        | Fujimoto et al. _Addressing Function Approximation Error in Actor-Critic Methods_ (TD3, arXiv:1802.09477, 2018)                           | Double Q + delayed update + target smoothing, fixes DDPG's Q-value overestimation and training instability.                                                                                            |
| ★        | Haarnoja et al. _Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor_ (arXiv:1801.01290, 2018)                  | Maximum entropy RL + automatic temperature regulation, MuJoCo long-term dominance, the preferred algorithm for continuous control.                                                                     |
| ★        | Hessel et al. _Rainbow: Combining Improvements in Deep RL_ (arXiv:1710.02298, 2017)                                                       | Combined six DQN improvements: Double DQN, Dueling, PER, NoisyNet, Multi-step, Distributional Q, proving "combination beats single improvements." Ablation experiments are a classic textbook example. |
| ★        | Silver et al. _A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play_ (AlphaZero, Science, 2018) | Self-play + MCTS + neural networks, achieved superhuman performance on Go, Chess, and Shogi without human game data.                                                                                   |
| ★        | Schrittwieser et al. _Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model_ (MuZero, arXiv:1911.08265, 2019)             | Learned a **implicit** world model (representation/dynamics/prediction triple networks), planning without knowing game rules.                                                                          |
| ★        | Hafner et al. _Mastering Diverse Domains through World Models_ (Dreamer V3, arXiv:2301.04104, 2023)                                       | RSSM recurrent world model + training actor-critic in imagination, single hyperparameter across 150+ tasks (Atari, MuJoCo, Crafter, DMLab), the modern flagship of model-based RL.                     |
| ◆        | Haarnoja et al. _Soft Actor-Critic Algorithms and Applications_ (arXiv:1812.05905, 2018)                                                  | Extended technical report of SAC, including full hyperparameter tables and multiple real robot experiments.                                                                                            |
| ◆        | Schaul et al. _Prioritized Experience Replay_ (arXiv:1511.05952, 2015)                                                                    | Prioritized experience replay using TD error, accelerated DQN convergence. One of the core components of Rainbow.                                                                                      |
| ◆        | Wang et al. _Dueling Network Architectures for Deep RL_ (arXiv:1511.06581, 2015)                                                          | Decomposed $Q(s,a)$ into $V(s) + A(s,a)$, improved training efficiency for tasks with small action differences.                                                                                        |
| ◆        | Van Hasselt et al. _Deep RL with Double Q-Learning_ (arXiv:1509.06461, 2015)                                                              | Double DQN: decoupled action selection from value estimation, suppressed Q-value overestimation.                                                                                                       |
| ◆        | Bellemare et al. _A Distributional Perspective on RL_ (C51, arXiv:1707.06887, 2017)                                                       | Learned the **distribution** of rewards rather than the expectation, more robust to noisy rewards.                                                                                                     |
| ◇        | Janner et al. _When to Trust Your Model: Model-Based RL in the Stochastic World_ (MBPO, arXiv:1906.08253, 2019)                           | Short horizon rollout balances model bias and sample efficiency.                                                                                                                                       |
| ◇        | Chua et al. _Deep RL in a Handful of Trials Using Probabilistic Dynamics Models_ (PETS, arXiv:1805.12114, 2018)                           | Integrated + probabilistic model for cognitive and epistemic uncertainty.                                                                                                                              |

## C.1.3 Must-Read Papers on LLM RL

LLM RL is the largest application of reinforcement learning (RL) after 2022. This set of papers defines core paradigms such as RLHF / DPO / GRPO / RLVR / R1 / DAPO, each corresponding to a segment of industry training practices. Readers should **read them in chronological order** to understand the evolution of algorithms: reward model → RLHF → DPO (without reward model) → GRPO (without critic) → R1 (pure RL) → DAPO (fixing GRPO's shortcomings) → the 2026 GSPO/CISPO.

| Priority | Paper                                                                                                                     | Core Contribution                                                                                                                                                                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★        | Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback_ (InstructGPT, arXiv:2203.02155, 2022) | RLHF three-stage engineering (SFT + RM + PPO), the first large-scale proof that RLHF significantly outperforms SFT. The foundational paper of LLM post-training paradigms. See [Chapter 13](../chapter15_rlhf/base-model-to-assistant).                                                 |
| ★        | Bai et al. _Constitutional AI: Harmlessness from AI Feedback_ (arXiv:2212.08073, 2022)                                    | Anthropic's RLAIF: using AI feedback to replace human annotation, "Constitution" driving self-correction. Also serves as a bridge between RLHF and alignment research. See [Chapter 13](../chapter21_cai_rlvr/hhh-practice).                                                            |
| ★        | Rafailov et al. _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_ (arXiv:2305.18290, 2023) | DPO: using the Bradley-Terry model to reparameterize, converting RLHF into supervised learning, **completely eliminating the need for RM and PPO**. One of the most elegant mathematical derivations in LLM post-training. See [Chapter 14](../chapter17_dpo/dpo-objective-derivation). |
| ★        | DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL_ (arXiv:2501.12948, 2025)                    | R1: Pure RL (GRPO + rule-based reward) makes a 7B model's mathematical reasoning close to GPT-4o; R1-Zero proves that long CoT can be triggered without SFT. A landmark paper of the RLVR paradigm. See [Chapter 15](../chapter18_grpo/grpo-practice-and-mechanism).                    |
| ★        | Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ (arXiv:2402.03300, 2024) | The original paper of GRPO: removing the critic of PPO, using group-wise normalization to estimate baseline. The starting point of the Group-Normalized Policy Gradient paradigm.                                                                                                       |
| ★        | Yu et al. _DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization_ (arXiv:2503.14476, 2025)                        | Fixes four shortcomings of GRPO—length bias, entropy collapse, noise advantage, and long/short horizon conflicts—proposing Clip-Higher, Dynamic Sampling, Token-Level Loss, and Overlong Reward Shaping. Engineering improvements by the Qwen team.                                     |
| ★        | Kimi Team. _Kimi K2: Open Agentic Intelligence_ (arXiv:2507.20534, 2025)                                                  | 1T parameter MoE (32B activation) open training, validating the scalability of GSPO (Group Sequence Policy Optimization) in ultra-large agent training.                                                                                                                                 |
| ★        | Zheng et al. _Group Sequence Policy Optimization_ (arXiv:2507.18071, 2025)                                                | Sequence-level importance sampling + sequence-level loss, fixing GRPO's high token-level variance in long sequences and long horizon tasks.                                                                                                                                             |
| ★        | MiniMax. _MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention_ (arXiv:2506.13585, 2025)            | Proposes CISPO (Cropped Importance Sampling Policy Optimization) on a 456B hybrid attention MoE, using cropped importance sampling weights instead of token updates, combined with lightning attention to complete RL training on 512 H800s in three weeks.                             |
| ◆        | Xiao et al. _VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks_ (arXiv:2504.05118, 2025)   | Value-Assisted PPO: introducing a lightweight critic + long-sequence value estimation, making PPO usable again for long CoT in mathematical reasoning.                                                                                                                                  |
| ◆        | Luong et al. _ReFT: Reasoning with Reinforced Fine-Tuning_ (arXiv:2401.08967, 2024)                                       | SFT preheating + PPO online fine-tuning, sampling diverse reasoning paths from the same training set, continuously improving reasoning ability on small datasets.                                                                                                                       |
| ◆        | Yuan et al. _Free Process Rewards without Process Labels_ (arXiv:2412.01981, 2024)                                        | Distilling token-level implicit PRM from RLVR training, providing dense process rewards without human process annotations.                                                                                                                                                              |
| ◆        | Setlur et al. _Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning_ (PRM, arXiv:2410.08146, 2024)   | Systematic methodology for PRM training, the foundation of PRM-guided search. See [Chapter 17](../chapter20_prm_search/outcome-vs-process).                                                                                                                                             |
| ◇        | Yuan et al. _Self-Rewarding Language Models_ (arXiv:2401.10020, 2024)                                                     | Letting LLMs generate preference pairs to train themselves, reducing reliance on human annotations.                                                                                                                                                                                     |
| ◇        | Tajwar et al. _Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data_ (arXiv:2404.14367, 2024)        | Comparing three types of preference fine-tuning—supervised, on-policy RL, and contrastive learning—proving that suboptimal on-policy data is most effective for preference fine-tuning.                                                                                                 |

## C.1.4 Essential Reading for Safety Research

RLHF / RLVR opens the Pandora's box of **reward hacking, deception, alignment camouflage, and latent backdoors** while making models stronger. This set of papers represents the core literature of AI safety research after 2023 — they explain **why well-trained models can harm you**, and the fundamental limitations of existing alignment methods. Any engineer working on post-training should at least read the first five papers in detail.

| Priority | Paper                                                                                                                                | Core Contribution                                                                                                                                                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ★        | Hubinger et al. _Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training_ (arXiv:2401.05566, 2024)              | Anthropic demonstrates: models with "specific trigger words only for malicious behavior" backdoors cannot be eliminated by RLHF / SFT / adversarial training. Reveals the fundamental blind spot of existing alignment methods.                                                |
| ★        | Greenblatt et al. _Alignment Faking in Large Language Models_ (arXiv:2412.14093, 2024)                                               | Anthropic demonstrates: models **pretend to be aligned** during training — displaying behavior that conforms to expectations to avoid being modified; and then violating training objectives during reasoning. This is the empirical manifestation of schema in real training. |
| ★        | Meinke et al. _Frontier Models are Capable of In-Context Scheming_ (arXiv:2412.04984, 2024)                                          | In-context scheming is observed across multiple frontier models: models pretend to have limited capabilities, copy their own weights, and avoid supervision.                                                                                                                   |
| ★        | Sharma et al. _Towards Understanding Sycophancy in Language Models_ (arXiv:2310.13548, 2023)                                         | Systematic study of sycophancy (obsequiousness) behavior in LLMs, revealing the bias in RLHF preference data where "user agreement = high reward".                                                                                                                             |
| ★        | Denison et al. _Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models_ (arXiv:2406.10162, 2024)          | Course-based experiments demonstrate: LLMs that learn simple specification gaming will gradually generalize to **directly rewriting their own reward function** (reward tampering).                                                                                            |
| ★        | OpenAI. _GPT-4o System Card_ (2024)                                                                                                  | After the release of GPT-4o, sycophancy issues led to large-scale rollbacks — publicly acknowledging the observable consequences of reward signal bias in RLHF training.                                                                                                       |
| ★        | Gao et al. _Scaling Laws for Reward Model Overoptimization_ (arXiv:2210.10760, 2022)                                                 | OpenAI's reward model overoptimization scaling law: the distance between RM and true preferences deteriorates exponentially with increasing KL divergence. A key paper for understanding the limits of RLHF.                                                                   |
| ★        | Anthropic. _Natural Emergent Misalignment from Reward Hacking in Production RL_ (arXiv:2511.18397, 2025)                             | Proves that models that learn reward hacking will generalize to emergent misalignment such as alignment faking and deliberate destruction, and standard RLHF safety training cannot eliminate these.                                                                           |
| ◆        | Eisenstein et al. _Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking_ (arXiv:2312.09244, 2023) | Uses RM ensembles to reduce reward hacking, a commonly used robustification approach in industry.                                                                                                                                                                              |
| ◆        | Coste et al. _Reward Model Ensembles Help Mitigate Overoptimization_ (arXiv:2310.02743, 2023)                                        | Engineering empirical results on the same topic, including complete examples of hyperparameter abuse.                                                                                                                                                                          |
| ◆        | Hubinger et al. _Risks from Learned Optimization in Advanced Machine Learning Systems_ (arXiv:1906.01820, 2019)                      | Mesa-optimization theoretical framework: learned optimizers may internalize goals different from the training objective — the theoretical foundation for analyzing the probability of deceptive alignment.                                                                     |
| ◇        | Taylor et al. _School of Reward Hacks: Hacking Harmless Tasks Generalizes to Misaligned Behavior in LLMs_ (arXiv:2508.17511, 2025)   | Training reward hacking on harmless tasks generalizes to more dangerous misaligned behavior, which can be used as a benchmark for evaluating safety.                                                                                                                           |
| ◇        | METR. _Frontier Reward Hacking: An Empirical Study of Reward Hacking in Frontier Models_ (2024)                                      | METR provides an empirical benchmark for evaluating reward hacking behavior in frontier models.                                                                                                                                                                                |
| ◇        | OpenAI. _The Instruction Hierarchy: Training LLMs to Prioritize Instructions_ (arXiv:2404.13208, 2024)                               | Divides prompts into three layers: system, user, and tool, providing training-time strategies to counter prompt injection.                                                                                                                                                     |

## C.1.5 2025–2026 Frontiers

Starting from 2025, RL enters its second boom period: process reward models drive reasoning, agent training paradigms mature, self-play extends to code and software engineering, and video generation introduces RL fine-tuning. These papers are mostly tech reports from 2025–2026 and have not yet entered mainstream textbooks—they define **the research directions for the next decade of RL**. Readers should pay close attention.

### C.1.5.1 RLVR and Reasoning Algorithms

| Priority | Paper                                                                                                                                                 | Core Contribution                                                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★        | Yu et al. _DAPO_ (arXiv:2503.14476, 2025)                                                                                                             | Engineering fix of GRPO, Qwen team opens the full training details. Already listed in C.1.3.                                                         |
| ★        | Zheng et al. _GSPO_ (arXiv:2507.18071, 2025)                                                                                                          | Sequence-level IS, core algorithm used by Kimi K2. Already listed in C.1.3.                                                                          |
| ★        | MiniMax. _CISPO_ (arXiv:2506.13585, 2025)                                                                                                             | Truncating invalid IS samples. Already listed in C.1.3.                                                                                              |
| ★        | Khalifa et al. _Process Reward Models That Think_ (ThinkPRM, arXiv:2504.16828, 2025)                                                                  | Using a lightweight PRM to replace dense process reward annotations, making RLVR usable with low data.                                               |
| ★        | DeepSeek. _DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition_ (arXiv:2504.21801, 2025) | RLVR on formal theorem proving (Lean 4): RL learns subgoal decomposition, achieving 88.9% pass rate on MiniF2F.                                      |
| ◆        | Microsoft. _rStar-Math_ (arXiv:2501.04519, 2025)                                                                                                      | MCTS + PRM + self-play training for small model mathematical reasoning.                                                                              |
| ◆        | Liu et al. _Understanding R1-Zero-Like Training: A Critical Perspective_ (arXiv:2503.20783, 2025)                                                     | Dissecting the base model and RL components in R1-Zero-like training, pointing out the response length bias in GRPO and proposing unbiased Dr. GRPO. |

### C.1.5.2 Agentic RL and Code/Software Engineering

| Priority | Paper                                                                                                                        | Core Contribution                                                                                                                                                                                                                            |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★        | Pan et al. _SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution_ (arXiv:2502.18449, 2025)  | Meta scales RL training on SWE-bench, defining the "code-execution-feedback-learning" Agent RLVR paradigm. See [Chapter 20](../chapter23_rl_based_swe/swe-bench-and-rlvr).                                                                   |
| ★        | FAIR CodeGen team. _CWM: An Open-Weights LLM for Research on Code Generation with World Models_ (arXiv:2510.02387, 2025)     | Meta FAIR's 32B open-weight model: trained on Python interpreter and agentic Docker trajectories, integrating code execution prediction into RL reasoning.                                                                                   |
| ★        | Wei et al. _Toward Training Superintelligent Software Agents through Self-Play SWE-RL_ (arXiv:2512.18552, 2025)              | Meta/Carnegie Mellon's self-play SWE-RL: a single agent iteratively injects and fixes bugs in a sandboxed codebase, continuously self-improves without human-annotated issues/tests.                                                         |
| ★        | ByteDance. _UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning_ (arXiv:2509.02544, 2025) | GUI Operation Agent: screenshot → action (mouse click/keyboard), end-to-end visual RL. See [Chapter 22](../chapter25_computer_use/training).                                                                                                 |
| ★        | StepFun. _Step-Audio-R1 Technical Report_ (arXiv:2511.15848, 2025)                                                           | The first model to unlock audio reasoning capabilities: MGRD distillation anchors the reasoning chain to real acoustic features, outperforming Gemini 2.5 Pro in audio understanding. See [Chapter 24](../chapter27_audio_rl/reward-design). |
| ◆        | Tongyi. _Tongyi DeepResearch_ (2025)                                                                                         | PRM-guided long-horizon research agent.                                                                                                                                                                                                      |
| ◇        | Google. _AlphaEvolve_ (2025)                                                                                                 | Evolution + LLM automatically discovers new algorithms, applied to chip design and combinatorial optimization.                                                                                                                               |

### C.1.5.3 Large-Scale Training and Open Models

| Priority | Paper                                                                                                           | Core Contribution                                                                                                                                                                                                                                           |
| -------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★        | Qwen Team. _Qwen3 Technical Report_ (arXiv:2505.09388, 2025)                                                    | Training details of the full Qwen3 series (0.6B–235B-A22B MoE), including the complete recipe of SFT/RLHF/RLVR.                                                                                                                                             |
| ★        | Kimi Team. _Kimi K2_ (arXiv:2507.20534, 2025)                                                                   | 1T MoE open training, validating the scalability of large-scale GSPO on agent tasks. Already listed in C.1.3.                                                                                                                                               |
| ★        | Kimi Team. _Kimi K2.5: Visual Agentic Intelligence_ (arXiv:2602.02276, 2026)                                    | The next-generation multimodal agent model of K2: joint text-visual pre-training and RL, significantly reducing inference latency with parallel task orchestration in Agent Swarm.                                                                          |
| ★        | DeepSeek-AI. _DeepSeek-V3 Technical Report_ (arXiv:2412.19437, 2024)                                            | 671B MoE (37B active), 2.664M H800 hours training 14.8T tokens; MLA + DeepSeekMoE + FP8 are core engineering innovations. Pre-training cost reference see [Appendix C](../appendix_gpu_hours/gpu-hours-estimation).                                         |
| ★        | Gao et al. _Seedance 1.0: Exploring the Boundaries of Video Generation Models_ (arXiv:2506.09113, 2025)         | RLVR fine-tuning for video generation models, extending RL from tokens to spatiotemporal latent space.                                                                                                                                                      |
| ★        | Peng et al. _DanceGRPO: A Unified Post-training Framework for Generative Video Models_ (arXiv:2505.07818, 2025) | Adapting GRPO to video diffusion models, unifying text and video RLVR.                                                                                                                                                                                      |
| ★        | Meituan LongCat Team. _LongCat-Video Technical Report_ (arXiv:2510.22200, 2025)                                 | Meituan's 13.6B parameter DiT video generation model, supporting text-to-video, image-to-video, and continuation, with multi-reward RLHF alignment for long video generation.                                                                               |
| ◆        | Behrouz et al. _Titans: Learning to Memorize at Test Time_ (arXiv:2501.00663, 2025)                             | Extending Transformers with long-term and short-term memory modules, enabling online learning of new concepts at test time, providing a scalable memory architecture for long-sequence reasoning.                                                           |
| ◆        | Hong et al. _Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO_ (arXiv:2511.13288, 2025)      | Hierarchical extension of GRPO: the main agent and each sub-agent separately compute intra-group advantages, combined with trajectory alignment and decoupled training, improving the stability and sample efficiency of multi-agent deep research systems. |
| ◆        | Google. _Genie 3_ (2025)                                                                                        | General world model, learning environment dynamics from video, serving as the foundation for embodied RL in the next generation.                                                                                                                            |
| ◇        | Ishibashi et al. _Can Large Language Models Invent Algorithms to Improve Themselves?_ (arXiv:2410.15639, 2024)  | Self-Developing: models use DPO to iteratively discover and improve their own algorithms (model merging), outperforming human-designed Task Arithmetic in mathematical reasoning.                                                                           |

## Reading Path Recommendations

The optimal path varies for different readers:

- **From the Basics**: C.1.1 → C.1.2 → C.1.3 → C.1.4, to understand the full evolution of RL from 1992 to 2026 in chronological order.
- **For LLM Post-Training Engineers**: C.1.3 (entire) + C.1.4 (first five chapters) + C.1.5.1 (entire). Focus on the engineering details of InstructGPT, DPO, GRPO, R1, and DAPO.
- **For Agentic RL Researchers**: C.1.3 (GRPO/R1) + C.1.5.2 (entire) + C.1.2 (AlphaZero/MuZero/Dreamer) to understand search and world models.
- **For Safety Researchers**: C.1.4 (entire) + C.1.1 (Sutton & Barto, Chapter 15 on exploration and exploitation failure patterns) + C.1.3 (RLHF over-optimization related papers).
- **For Theoretical Researchers**: C.1.1 (entire) + C.1.2 (Rainbow/Dreamer ablation studies) + C.1.5 (GSPO/CISPO mathematical derivations).

## Reading Methods

::: tip Detailed Reading vs. Skimming
For **core algorithm papers** (marked with ★), do a detailed reading—read formulas line by line, reproduce code, and conduct ablation studies. For **engineering tech reports** (such as Qwen3, Kimi K2), do a skimming—focus on hyperparameter tables, training curves, and ablation results. The former trains your thinking, while the latter trains your engineering intuition.
:::

::: warning Do Not Only Read the Latest Papers
RL papers from 2025–2026 often assume readers are familiar with the details of GRPO/PPO/REINFORCE. If you skip C.1.1/C.1.2 and directly read C.1.5, you might feel that "the algorithms are not special"—in reality, this is due to a lack of historical context. It is recommended to trace each new algorithm back to its 2–3 "ancestral" papers.
:::

::: details Correspondence Between Papers and Chapters in This Book
The "Extended Reading" section at the end of each chapter in this book lists the core papers for that chapter. This appendix provides a more comprehensive map of papers across chapters. It is recommended to use this appendix in conjunction with the lists at the end of each chapter:

- [Chapter 14 DPO](../chapter17_dpo/dpo-objective-derivation) → C.1.3 Rafailov et al.
- [Chapter 8 PPO](../chapter10_ppo/ppo-clip-objective) → C.1.1 Schulman PPO, C.1.2 A3C
- [Chapter 13 RLHF](../chapter15_rlhf/base-model-to-assistant) → C.1.3 InstructGPT
- [Chapter 15 GRPO/RLVR](../chapter18_grpo/grpo-practice-and-mechanism) → C.1.3 R1, Shao et al.
- [Chapter 9 Continuous Control](../chapter11_continuous_control/ddpg) → C.1.2 DDPG/TD3/SAC/Dreamer V3
- [Chapter 25 Alignment Failures](../chapter30_alignment_failures/classical-failures) → C.1.4 All
- [Chapter 18 Industrial-Level LLM RL](../chapter16_llm_rl_industrial/single-machine-to-industrial) → C.1.5.3 All
  :::

## Chapter Summary

The 100+ papers in this appendix constitute a complete map of the reinforcement learning (RL) literature, from 1992 Q-Learning to 2026 Kimi K2.5. They are not a "read and forget" list, but rather a **resource to be revisited repeatedly, accompanying your professional growth** — each additional paper you understand deepens your comprehension of modern RL. When you can uncover an engineering detail that the author left unexplained in a paper, or identify an assumption that has not been ablated, you have the capability to conduct independent research.

## Supplement: Learning Resources and Reproduction Projects

> **This appendix's goal** is to provide readers with a clear navigation for their future advancement. The first half organizes theory-rich, accessible textbooks and course resources to help you systematically build a solid foundation or delve into the cutting edge. The second half outlines the classic milestones and commonly used environments in the game and simulation ecosystems of reinforcement learning, offering you inspiration and a coordinate system to find your next hands-on reproduction project.

### Recommended Learning Resources

> **How to Use This List**: This book covers the full chain from MDP fundamentals to PPO, DPO, and GRPO, but the RL field extends far beyond this. If you want to dive deeper into a particular direction, compare different teaching styles, or find resources to get started with hands-on practice, this list can serve as a starting point. All resources are free or publicly accessible.

**How to Use This List?**

Choose based on your goals:

- **You have just finished Chapter 3 and want to see how other textbooks explain the fundamentals**: We recommend starting with Zhao Shiyu's _Principles of RL Mathematics_ or the original Sutton & Barto book.
- **You want to follow a video course**: We recommend starting with David Silver's course or Li Hongyi's course.
- **You want to get started with coding**: We recommend starting with OpenAI's _Spinning Up_ or _Hands-On Reinforcement Learning_.
- **You are interested in LLM alignment / RLHF / GRPO**: We recommend starting with Nathan Lambert's RLHF Book or Ernest Ryu's RL-LLM course.
- **You want to learn about cutting-edge theories**: We recommend starting with Princeton ECE 524 or Alberta CMPUT 365.

**I. Classic Textbooks**

**Reinforcement Learning: An Introduction (Sutton & Barto, 2nd Edition, 2018)**

**Link**: [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) ｜ [Chinese Translation](https://rl.qiwihui.com/)

This is the standard textbook for reinforcement learning and a required reference in many university courses. Part I develops tabular methods, including MDPs, dynamic programming, Monte Carlo methods, TD learning, $n$-step bootstrapping, and planning. Part II covers function approximation, eligibility traces, and policy gradients. Part III connects RL with psychology, neuroscience, and applications. The authors provide the book online, making it a strong choice for a systematic foundation.

**Mathematical Foundations of Reinforcement Learning (Shiyu Zhao)**

**Link**: [github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

Its ten chapters derive the core RL algorithms from a mathematical perspective: Bellman equations, value and policy iteration, Monte Carlo methods, TD learning, function approximation, policy gradients, and actor–critic methods. Each chapter includes proofs and exercises. It is well suited to readers who want a rigorous explanation of why the algorithms work.

**Deep Reinforcement Learning (Zhihua Zhang, Peking University)**

**Link**: [Draft PDF](https://www.math.pku.edu.cn/teachers/zhzhang/drl_v1.pdf)

**Hands-On Reinforcement Learning (Weinan Zhang, Jian Shen, and Yong Yu)**

**Address**: [Online Version](https://hrl.boyuai.com/) ｜ Textbook for Shanghai Jiao Tong University RL Course

Practical-oriented, the entire book is accompanied by executable Jupyter code. It is divided into three parts: **Basics** (Bandit → MDP → DP → MC → Planning) → **Advanced Topics** (Function Approximation → DQN → Policy Gradient → PPO) → **Frontiers** (Model-Based RL, Offline RL). **Suitable for learners who want to code while reading.**

**II. University Courses**

**Courses from Europe and the US**

**Stanford CS234: Reinforcement Learning (Emma Brunskill)**

**Address**: [web.stanford.edu/class/cs234/](https://web.stanford.edu/class/cs234/)

Stanford's introductory course on reinforcement learning. It starts with table MDPs, covers policy evaluation, Q-Learning, function approximation, policy gradient, Offline RL, exploration, MCTS, and finally touches on RLHF. About half of the class time is devoted to theoretical foundations, and the other half to advanced topics. Textbook: Sutton & Barto.

**Stanford CS224R: Deep Reinforcement Learning (Chelsea Finn)**

**Website**: [cs224r.stanford.edu](https://cs224r.stanford.edu/) ｜ [YouTube 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL)

Stanford's Deep Reinforcement Learning course. Assumes students have a basic understanding of reinforcement learning and directly starts with imitation learning, quickly moving into policy gradient methods, Actor-Critic, Q-Learning, Model-Based RL, Offline RL, Reward Learning, RLHF, and Meta-RL. **Ideal for learners who have a foundational understanding and wish to delve into various directions of deep reinforcement learning.**

**MIT 6.7920: Reinforcement Learning Foundations and Methods (Cathy Wu)**

**Website**: [web.mit.edu/6.7920/www/](https://web.mit.edu/6.7920/www/)

MIT's Reinforcement Learning theory course. Two-thirds of the course focuses on "exploitation" (known theory: DP 7 lectures + core RL methods 9 lectures), and one-third on "exploration" (frontier topics). The DP portion is very solid, covering finite/infinite horizon, LQR, policy/ value iteration, and convergence proofs. **Suitable for learners who seek theoretical depth.**

**UC Berkeley CS285: Deep Reinforcement Learning (Sergey Levine)**

**Address**: [rail.eecs.berkeley.edu/deeprlcourse/](https://rail.eecs.berkeley.edu/deeprlcourse/)

Berkeley's Deep RL flagship course. Only one lecture reviews RL fundamentals, followed by an in-depth exploration of imitation learning, policy gradient, Actor-Critic, Value-Based RL, advanced policy gradient, variational inference with RL, LLM RL, Model-Based RL, Offline RL, and exploration. The 2026 Spring edition includes hands-on assignments for LLM RL and Offline RL. **The content is most aligned with current industrial frontiers.**

**CMU 10-703: Deep Reinforcement Learning and Control**

**Address**: [cmudeeprl.github.io/703website_f25/](https://cmudeeprl.github.io/703website_f25/)

CMU's Deep RL course. After covering classical theory (MDP, DP, MC, TD), the course moves into function approximation, Deep Q-Learning, MCTS, policy gradient, imitation learning, inverse RL, optimal control, Model-Based RL, and exploration. **Balances theory and practice with broad coverage.**

**University of Alberta CMPUT 365: Introduction to RL (Marlos Machado)**

**Address**: [Syllabus PDF](https://webdocs.cs.ualberta.ca/~machado/cmput365/w26/syllabus.pdf)

Sutton's university's RL introductory course strictly follows the order of the Sutton & Barto textbook: Bandits → MDP → DP (including PI, VI, GPI) → MC Prediction and Control → TD Prediction → **TD Control (Sarsa, Q-Learning)** → Planning (Dyna-Q) → Function Approximation → Policy Gradients. **The most faithful implementation of the Sutton & Barto textbook in a course.**

**Georgia Tech CS 7642: Reinforcement Learning (OMSCS)**

**Address**: [omscs.gatech.edu/cs-7642-reinforcement-learning](https://omscs.gatech.edu/cs-7642-reinforcement-learning)

An online RL course. Covers DP, TD (including Sarsa), n-step TD, Lambda Return, DQN, Policy Gradients, Multi-Agent RL, Game Theory, POMDP. **One of the best-rated RL courses in the OMSCS program.**

**Princeton ECE 524: Foundations of RL (Chi Jin)**

**Address**: [sites.google.com/view/cjin/teaching/ece524](https://sites.google.com/view/cjin/teaching/ece524) ｜ [YouTube](https://www.youtube.com/playlist?list=PLYXvCE1En13epbogBmgafC_Yyyk9oQogl)

**Theoretical Focus, with Emphasis on Finite Sample Analysis and Convergence Proofs.**
**Part I** covers tabular MDPs, planning, exploration (Bandit and MDP), and lower bounds;
**Part II** covers large state spaces, linear VI, function approximation, multi-agent systems, and POMDPs.
**Suitable for learners interested in RL theory research.**

**David Silver RL Course (UCL / DeepMind)**

**Address**: [davidsilver.uk/teaching](https://www.davidsilver.uk/teaching/) ｜ [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

10-lecture classic course: MDP → DP → Model-Free Prediction → Model-Free Control → Function Approximation → Policy Gradient → Learning & Planning → Exploration → Classic Game Case Studies. David Silver is the first author of AlphaGo/AlphaZero.
**The structure is concise, the explanations are clear, and it is the most widely disseminated RL video course.**

**DeepMind x UCL RL Lecture Series (2021)**

**Address**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

David Silver's updated course, presented by DeepMind researchers (including Hado van Hasselt and others). It covers 13 lectures on exploration and control, MDP and DP, model-free methods, function approximation, planning, policy gradient and actor-critic methods, approximate DP, multi-step and off-policy learning, and deep reinforcement learning. **More advanced than the 2015 edition, with added content on cutting-edge topics.**

**Chinese University Courses**

**Tsinghua University Reinforcement Learning (Fall 2025)**

**Address**: [coai.cs.tsinghua.edu.cn/Courses/RL2025/\_site/](https://coai.cs.tsinghua.edu.cn/Courses/RL2025/_site/)

An undergraduate-level RL course. Starts with multi-armed bandit problems, covering MDP, planning (DP), MC, TD learning, policy gradient, function approximation, and deep RL. Includes 4 programming assignments (Bandit → MDP → TD & PG → Deep RL) + course project. Lecture notes are publicly available.

**Nanjing University Introduction to Reinforcement Learning (Yu Yang, 2024)**

**Address**: [lamda.nju.edu.cn/introrl](https://www.lamda.nju.edu.cn/introrl/)

Based on the Sutton & Barto textbook, 9 lectures cover the basic concepts of RL, MDP, DP, MC, TD, and DQN. Includes 5 programming assignments (Dagger → Q-Learning → DQN → Model-Based → Offline RL). **One of the most theoretically rigorous courses among Chinese universities.**

**Nanjing University Advanced Reinforcement Learning (Yuan Lei, 2025)**

**Address**: [lamda.nju.edu.cn/advanceRL](https://www.lamda.nju.edu.cn/advanceRL/)

Graduate-level advanced course. Covers DDPG/TD3, PPO and techniques, multi-agent systems, theoretical derivations of RLHF/DPO, and paper reading.

**Shanghai Jiao Tong University Reinforcement Learning (Zhang Weinan, 2024)**

**Address**: [wnzhang.net/teaching/sjtu-rl-2024](https://wnzhang.net/teaching/sjtu-rl-2024/)

Uses _Hands-On Reinforcement Learning_ as the textbook, covering 9 chapters from basics to frontiers.

**III. Chinese Online Courses and Tutorials**

**Li Hongyi Deep Reinforcement Learning (National Taiwan University)**

**Address**: [Course Homepage](https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) ｜ [Bilibili 2025 Edition](https://www.bilibili.com/video/BV1SJvAzfEL2/)

Enters the subject with Policy Gradient as the main thread, and provides in-depth explanations of PPO (including Importance Sampling, On-policy → Off-policy derivation), followed by Q-Learning (DQN, Double DQN, Dueling DQN), and Actor-Critic. The explanations are vivid and the PPTs are well-designed. **The PPO section is the most in-depth among Chinese courses.**

**Wang Shusen Deep Reinforcement Learning**

**Address**: [Bilibili Video](https://www.bilibili.com/video/BV1oEWDz1Ez5/)

A video companion for the course offered by the College of Mathematics, Peking University. It consists of five modules: basic concepts → value learning (DQN) → policy learning (Policy Gradient) → Actor-Critic (A3C, TRPO) → advanced topics (DDPG, AlphaGo, multi-agent systems). It is paired with Zhang Zihua's textbook _Deep Reinforcement Learning_. **Content is concise and ideal for a quick start.**

**Mushroom Book EasyRL (Datawhale)**

**Address**: [Online Version](https://datawhalechina.github.io/easy-rl/) ｜ [GitHub](https://github.com/datawhalechina/easy-rl)

A synthesis of the essentials from Zhou Bolei's _Outline of Reinforcement Learning_, Li Hongyi's course, and Baidu's _World Champion Takes You from Zero to Practice Reinforcement Learning_. It includes 13 chapters and special topics, covering from basics to DQN, PPO, DDPG, and AlphaStar. **The most active open-source RL tutorial in the Chinese community.**

**Spinning Up Chinese Version**

**Address**: [spinningup.qiwihui.com/zh-cn/latest](https://spinningup.qiwihui.com/zh-cn/latest/)

A Chinese translation of OpenAI's Spinning Up. It includes core concepts, algorithm classification, derivation of policy gradient methods, and implementations of six algorithms: VPG, TRPO, PPO, DDPG, TD3, and SAC.

**Nathan Lambert — RLHF Book + Course**

**Address**: [rlhfbook.com](https://rlhfbook.com/) ｜ [Course](https://rlhfbook.com/course) ｜ [GitHub](https://github.com/natolambert/rlhf-book) ｜ [YouTube](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

A comprehensive monograph on RLHF written by AI2 Researcher Nathan Lambert. It covers the full RLHF pipeline: instruction fine-tuning → reward model training → Rejection Sampling → PPO → DPO. The code repository implements policy gradient methods such as PPO, REINFORCE, GRPO, and RLOO. The video course consists of four lectures. **The most systematic open-access textbook in the field of LLM alignment.**

**Ernest Ryu — Reinforcement Learning of Large Language Models (UCLA)**

**Address**: [ernestryu.com/courses/RL-LLM.html](https://ernestryu.com/courses/RL-LLM.html)

The only university course that combines classical RL theory with LLM RL systems. It is divided into three parts: Ch1 (5 lectures on classical RL: MDP → VI → PG → PPO/GRPO → AlphaGo) → Ch2 (4 lectures on LLM fundamentals: NLP → Transformer → ICL/SFT) → Ch3 (2 lectures on LLM RL: RLHF/PPO/DPO → RLVR). **The LLM RL course with the deepest coverage of RL fundamentals.**

**DeepLearning.AI — Reinforcement Fine-Tuning LLMs with GRPO**

**Address**: [deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)

1-hour short course, 10 lessons. Using the Wordle game as a running case study, the course explains the GRPO algorithm, reward function design, LLM-as-Judge, and Reward Hacking. 7 coding experiments. **Suitable for practitioners who already have a foundation in LLMs and want to quickly get started with GRPO.**

**Hugging Face — Deep RL Course**

**Address**: [huggingface.co/learn/deep-rl-course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

8 units cover Q-Learning → DQN → Policy Gradient → A2C/A3C → PPO → Multi-Agent → Offline RL. Each unit includes theory and coding practice. Bonus unit covers RLHF. **Suitable for learners who want to use the Hugging Face ecosystem to conduct RL experiments.**

**V. Practical Tutorials and Technical Blogs**

**OpenAI Spinning Up in Deep RL**

**Address**: [spinningup.openai.com](https://spinningup.openai.com/en/latest/)

The gold standard for teaching the fundamentals of reinforcement learning. Divided into three parts: core concepts (V/Q/Bellman/Advantage) → algorithm classification (Model-Based vs Model-Free) → derivation of policy optimization (deriving Policy Gradient from scratch). Implements six algorithms: VPG, TRPO, PPO, DDPG, TD3, and SAC. **The ideal combination of theoretical explanations and code implementations.**

**Cameron Wolfe — Deep (Learning) Focus**

**Address**: [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm) ｜ [Online vs Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl)

A series of blog posts that explain in plain language the application of PPO in LLMs, the trade-offs between Online and Offline RL, and the principles of DPO. **Suitable for readers who want to understand "why LLM RL uses these algorithms."**

**Sebastian Raschka — Ahead of AI**

**Address**: [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) ｜ [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)

### Recommended Projects for Reproduction

Reinforcement learning projects can be divided into two lines of development based on the era. The pre-LLM era focused more on fixed simulation environments, game benchmarks, continuous control, multi-agent systems, and model learning. The LLM era expanded the concept of actions to include tokens, tool calls, web operations, visual reasoning, and long-term agent trajectories. Rewards also evolved from environment scores to preference models, rule validation, process rewards, and real-world task success rates.

#### Reproduction Roadmap Overview

| Target Direction                | Prioritize These Resources                                                                                                                                                                                                                                                                   | Suitable for Reproducing                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Classic Algorithms Introduction | [CleanRL](https://github.com/vwxyzjn/cleanrl)、[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)、[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)、[Dopamine](https://github.com/google/dopamine)                                                            | DQN、PPO、SAC、TD3、Rainbow DQN、Atari benchmarks                                                       |
| Environments & Game Benchmarks  | [Gymnasium](https://gymnasium.farama.org/)、[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)、[MiniGrid](https://minigrid.farama.org/)、[Procgen](https://github.com/openai/procgen)、[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) | CartPole、LunarLander、Atari、FPS、procedurally generated environments                                  |
| Multi-Agent & Game Theory       | [PettingZoo](https://pettingzoo.farama.org/)、[OpenSpiel](https://github.com/google-deepmind/open_spiel)、[SMAC](https://github.com/oxwhirl/smac)、[Google Research Football](https://github.com/google-research/football)                                                                   | Self-play、cooperative/competitive MARL、StarCraft micromanagement、football                            |
| Robotics & Embodied Control     | [MuJoCo](https://mujoco.readthedocs.io/)、[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)、[ManiSkill](https://maniskill.readthedocs.io/)、[Meta-World](https://github.com/Farama-Foundation/Metaworld)、[LeRobot](https://github.com/huggingface/lerobot)                                | Continuous control、robotic arms、mobile robots、imitation learning with RL                             |
| Model-Based / World Models      | [DreamerV3](https://github.com/danijar/dreamerv3)、[TD-MPC2](https://github.com/nicklashansen/tdmpc2)、[mbrl-lib](https://github.com/facebookresearch/mbrl-lib)、[MBPO](https://github.com/JannerM/mbpo)                                                                                     | Learn dynamics models from pixels or states, then perform planning or policy optimization               |
| LLM Post-Training               | [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155)、[TRL](https://huggingface.co/docs/trl/index)、[NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL)、[verl](https://github.com/verl-project/verl)                                                                                      | PPO、DPO、GRPO、RLHF、preference alignment、reward model training                                       |
| LLM Reasoning                   | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[Open-R1](https://github.com/huggingface/open-r1)、[TinyZero](https://github.com/Jiayi-Pan/TinyZero)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                           | RLVR、math/code reasoning、R1-style reproduction、verifier design                                       |
| Deep Research RL                | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)、[Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)、[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[WebThinker](https://github.com/RUC-NLPIR/WebThinker)                    | Search、reading、evidence screening、citation、research-oriented answers                                |
| Agentic RL                      | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)、[Google ADK](https://github.com/google/adk-python)、[Agent Lightning](https://github.com/microsoft/agent-lightning)、[AReaL](https://github.com/inclusionAI/AReaL)                                                      | Code execution、tool calls、web browsing、long-term task success rate optimization                      |
| GUI / Computer Use              | [OpenAI CUA](https://openai.com/index/computer-using-agent/)、[Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)、[UI-TARS](https://github.com/bytedance/UI-TARS)、[OSWorld](https://os-world.github.io/)                             | Web、desktop、mobile GUI operations and visual grounding                                                |
| VLM                             | [TRL VLM GRPO](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl)、[VLM-R1](https://github.com/om-ai-lab/VLM-R1)、[Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner)、[Gemini Robotics](https://deepmind.google/models/gemini-robotics/)        | Image question answering、visual reasoning、GUI/web、robotic vision operations、vision-language rewards |
| Generative Model RL             | [DDPO](https://github.com/jannerm/ddpo)、[Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo)、[AlignProp](https://align-prop.github.io/)、[RLAIF-V](https://github.com/RLHF-V/RLAIF-V)、[VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                        | Use preference、aesthetic、safety、and consistency rewards to optimize image/multimodal generation      |

#### Overview of Reinforcement Learning Directions

If you want to systematically select a direction for reproduction, it is recommended to choose based on the three axes of "algorithm problem + environment type + reward source." The following table can serve as a long-term directory skeleton.

| Category                             | Representative Problem                                                         | Recommended Projects/Frameworks                                                                                                                                                                                                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Value-Based RL                       | Learning discrete action strategies from Q-values                              | DQN, Double DQN, Dueling DQN, Rainbow; [Dopamine](https://github.com/google/dopamine), [CleanRL](https://github.com/vwxyzjn/cleanrl)                                                                                                                                                            |
| Policy Gradient / Actor-Critic       | Directly optimize policies for continuous or random actions                    | REINFORCE, A2C/A3C, PPO, TRPO; [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [TRL PPO](https://huggingface.co/docs/trl/ppo_trainer)                                                                                                                                         |
| Off-Policy / Maximum Entropy         | Improve sample efficiency, encourage exploration, and robustness               | DDPG, TD3, SAC, REDQ; [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [Tianshou](https://github.com/thu-ml/tianshou)                                                                                                                                                          |
| Distributional RL                    | Learn the distribution of returns rather than a single expectation             | C51, QR-DQN, IQN, FQF; [Dopamine](https://github.com/google/dopamine), [DI-engine](https://github.com/opendilab/DI-engine)                                                                                                                                                                      |
| Exploration / Curiosity              | Sparse rewards, long-term exploration, intrinsic motivation                    | RND, ICM, count-based exploration; [MiniGrid](https://minigrid.farama.org/), [Procgen](https://github.com/openai/procgen), [DI-engine exploration docs](https://opendilab.github.io/DI-engine/)                                                                                                 |
| Model-Based RL                       | Learn environment models and then plan or imagine rollouts                     | PETS, MBPO, Dreamer, TD-MPC; [mbrl-lib](https://github.com/facebookresearch/mbrl-lib), [DreamerV3](https://github.com/danijar/dreamerv3), [TD-MPC2](https://github.com/nicklashansen/tdmpc2)                                                                                                    |
| Offline / Batch RL                   | Only use offline data, cannot explore online                                   | BCQ, CQL, IQL, TD3+BC; [D4RL](https://github.com/Farama-Foundation/D4RL), [Minari](https://github.com/Farama-Foundation/Minari), [d3rlpy](https://github.com/takuseno/d3rlpy), [CORL](https://github.com/corl-team/CORL)                                                                        |
| Imitation / Reward Learning          | Learn from expert trajectories, preferences, or inverse reinforcement learning | BC, DAgger, GAIL, AIRL; [imitation](https://github.com/HumanCompatibleAI/imitation), [robomimic](https://github.com/ARISE-Initiative/robomimic), [LeRobot](https://github.com/huggingface/lerobot)                                                                                              |
| Goal-Conditioned / Hierarchical      | Long-term tasks, subgoals, options, and skills                                 | HER, Options, HIRO, skill discovery; [MiniGrid/BabyAI](https://minigrid.farama.org/), [Meta-World](https://github.com/Farama-Foundation/Metaworld)                                                                                                                                              |
| Meta-RL / Multitask / Generalization | Cross-task transfer, fast adaptation, and generalization                       | MAML-RL, PEARL, multi-task PPO/SAC; [Meta-World](https://github.com/Farama-Foundation/Metaworld), [Procgen](https://github.com/openai/procgen), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                                                                                     |
| Safe / Constrained RL                | Constraints on cost, risk, and safe exploration                                | CPO, PPO-Lagrangian, shielding; [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium), [OmniSafe](https://github.com/PKU-Alignment/omnisafe)                                                                                                                                    |
| Multi-Agent RL / Game AI             | Cooperation, competition, self-play, and communication                         | QMIX, MADDPG, MAPPO, AlphaZero; [PettingZoo](https://pettingzoo.farama.org/), [OpenSpiel](https://github.com/google-deepmind/open_spiel), [JaxMARL](https://github.com/FLAIROx/JaxMARL)                                                                                                         |
| Robotics / Embodied RL               | Continuous control, manipulation, navigation, Sim2Real                         | PPO/SAC on robots, domain randomization, VLA; [Isaac Lab](https://isaac-sim.github.io/IsaacLab/), [ManiSkill](https://maniskill.readthedocs.io/), [robosuite](https://robosuite.ai/), [OpenVLA](https://github.com/openvla/openvla)                                                             |
| Distributed / Systems RL             | High-throughput rollouts, multi-machine training, and production               | IMPALA, APPO, distributed PPO; [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html), [Sample Factory](https://github.com/alex-petrenko/sample-factory), [DI-engine](https://github.com/opendilab/DI-engine), [Acme](https://github.com/google-deepmind/acme)                             |
| RLHF / Preference Alignment          | Optimize language/multimodal models from human or AI preferences               | PPO, DPO, IPO, KTO, ORPO; [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155), [Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback), [TRL](https://huggingface.co/docs/trl/index), [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) |
| RLVR / Reasoning RL                  | Verifiable rewards, mathematical/code reasoning, long CoT                      | GRPO, DAPO, RLOO, REINFORCE++; [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [DAPO](https://github.com/BytedTsinghua-SIA/DAPO), [reasoning-gym](https://github.com/open-thought/reasoning-gym)                                  |
| Agentic RL                           | Search, tool use, code execution, and web/desktop tasks                        | Trajectory reward, tool-use reward, process reward; [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), [Google ADK](https://github.com/google/adk-python), [Agent Lightning](https://github.com/microsoft/agent-lightning), [SkyRL](https://docs.skyrl.ai/docs)               |
| VLM / GUI / Computer-Use RL          | Image understanding, GUI grounding, and web/mobile/desktop control             | Multimodal GRPO, GUI action RL; [OpenAI CUA](https://openai.com/index/computer-using-agent/), [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [OSWorld](https://os-world.github.io/)  |

#### Pre-LLM Era and Fixed Environments, Simulations, and Classical Algorithms

This path is ideal for building a strong foundation in reinforcement learning. It is recommended to start with single-file implementations of small environments and gradually progress to Atari, continuous control, multi-agent, robotics, and model-based RL.

##### Environments and Algorithm Libraries

| Environment/Tool                                                                                | Type                        | Description                                                                                    | Recommended Use                                                        |
| ----------------------------------------------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [Gymnasium](https://gymnasium.farama.org/)                                                      | General RL Environment      | Successor of OpenAI Gym, featuring classic environments like CartPole and LunarLander          | Getting started, algorithm debugging, course experiments               |
| [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Game Environment            | Atari 2600 standard benchmark, commonly used in DQN-related papers                             | Pixel input, discrete actions, DQN family                              |
| [MiniGrid](https://minigrid.farama.org/)                                                        | Grid World                  | Lightweight GridWorld, suitable for studying exploration, sparse rewards, and generalization   | Getting started with exploration, hierarchical RL, task generalization |
| [Procgen](https://github.com/openai/procgen)                                                    | Procedural Game Environment | 16 procedural game environments focusing on generalization capability                          | Overfitting analysis, generalization experiments                       |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)                                         | FPS 3D Environment          | First-person shooter game, partially observable, visual input, long-term decision making       | Visual strategies, POMDP, navigation and combat                        |
| [Stable-Retro](https://github.com/Farama-Foundation/stable-retro)                               | Classic Games               | Gymnasium-style wrappers for retro console games                                               | Classic game recreation, course demonstrations                         |
| [MuJoCo](https://mujoco.readthedocs.io/)                                                        | Physics Simulation          | High-precision physics engine, featuring benchmarks like HalfCheetah, Ant, and Humanoid        | PPO, SAC, TD3, and continuous control                                  |
| [PyBullet](https://pybullet.org/wordpress/)                                                     | Physics Simulation          | Open-source robot simulation with a lightweight ecosystem                                      | Introductory robotics and an alternative to MuJoCo experiments         |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)                                              | GPU Parallel Simulation     | Successor of NVIDIA Isaac Gym, suitable for large-scale parallel robot training                | Large-scale embodied RL, Sim2Real                                      |
| [ManiSkill](https://maniskill.readthedocs.io/)                                                  | Robotics Manipulation       | Benchmark for robotic arm manipulation, visual control, and large-scale parallel simulation    | Visual manipulation, imitation learning plus RL                        |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld)                                    | Multi-Task Robotics         | Multi-task benchmark for robotic arms                                                          | Multi-task RL, meta-learning, generalization                           |
| [PettingZoo](https://pettingzoo.farama.org/)                                                    | Multi-Agent Environment     | Multi-agent version of Gymnasium, supporting cooperative and competitive scenarios             | An introduction to MARL and parallel/sequential action interfaces      |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel)                                      | Game Framework              | Collection of algorithms for board games, card games, matrix games, and multi-agent algorithms | Self-play, CFR, AlphaZero variants                                     |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)                                     | Distributed RL              | Distributed RL library within the Ray ecosystem                                                | Large-scale training, production-level multi-agent experiments         |
| [CleanRL](https://github.com/vwxyzjn/cleanrl)                                                   | Algorithm Implementation    | Single-file, readable, and experiment-friendly implementations                                 | Learning algorithm details, writing course code                        |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)                                | Algorithm Library           | Pre-implemented versions of DQN, PPO, SAC, TD3, etc.                                           | Quick baseline runs, hyperparameter tuning, and comparisons            |
| [Dopamine](https://github.com/google/dopamine)                                                  | Atari Algorithm Library     | Google's research framework for DQN, Rainbow, IQN, etc.                                        | Atari paper reproduction, distributed value learning                   |

##### Recommended Reproduction Staircase

| Stage | Project Suggestions                      | Recommended Tools                                     | Acceptance Criteria                                               |
| ----- | ---------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| 1     | CartPole, MountainCar, LunarLander       | Gymnasium, CleanRL, Stable-Baselines3                 | Can plot reward curves, understand replay and GAE                 |
| 2     | DQN / Rainbow on Atari                   | ALE, Dopamine, CleanRL                                | Reproduce at least one Atari experiment                           |
| 3     | PPO / SAC / TD3 on MuJoCo                | MuJoCo, Stable-Baselines3, RL Baselines3 Zoo          | Can explain entropy, target network, and Q bias                   |
| 4     | Self-Play and Multi-Agent                | PettingZoo, OpenSpiel, SMAC, Google Research Football | Can distinguish between cooperative, competitive, and mixed games |
| 5     | Robotics Manipulation and Visual Control | Isaac Lab, ManiSkill, Meta-World, LeRobot             | Can run parallel simulation or imitation-to-RL flow               |
| 6     | Model-Based RL / World Model             | DreamerV3, TD-MPC2, mbrl-lib, MBPO                    | Can explain latent dynamics and planning                          |

##### Advanced Directions and Practice Suggestions

| Direction                               | Recommended Projects to Reproduce                                                                                                                                                   | Possible Course Assignments                                                                                              |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Single-File Algorithm                   | CleanRL's DQN, PPO, SAC, C51, PPO-LSTM                                                                                                                                              | Write replay, GAE, target network, entropy from 200 to 500 lines                                                         |
| High-Performance RL                     | [Sample Factory](https://github.com/alex-petrenko/sample-factory), [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html), [DI-engine](https://github.com/opendilab/DI-engine) | Compare throughput and sample efficiency of single-machine, multi-process, and distributed rollout                       |
| JAX / GPU Parallelism                   | [Brax](https://github.com/google/brax), [PureJaxRL](https://github.com/luchris429/purejaxrl), [JaxMARL](https://github.com/FLAIROx/JaxMARL)                                         | Use jit/vmap/pmap to run large batches of environments, understand the engineering paradigm of accelerating environments |
| Offline Reinforcement Learning          | D4RL + CQL/IQL/TD3+BC, [Minari](https://github.com/Farama-Foundation/Minari), [d3rlpy](https://github.com/takuseno/d3rlpy), [CORL](https://github.com/corl-team/CORL)               | Compare online RL and offline RL's extrapolation error                                                                   |
| Imitation Learning                      | Behavior Cloning, DAgger, GAIL, AIRL; [imitation](https://github.com/HumanCompatibleAI/imitation), [robomimic](https://github.com/ARISE-Initiative/robomimic)                       | Train a policy from expert trajectories, then fine-tune with RL                                                          |
| Reward Learning and Preference Learning | GAIL/AIRL, preference comparison, reward model                                                                                                                                      | Construct "human preference" or script preference, observe reward hacking                                                |
| Safe and Constrained RL                 | Safety-Gymnasium, OmniSafe, PPO-Lagrangian, CPO                                                                                                                                     | Plot reward curve and cost curve, learn constrained optimization                                                         |
| Exploration and Sparse Rewards          | MiniGrid, Montezuma's Revenge, Procgen; RND, ICM, episodic curiosity                                                                                                                | Study whether intrinsic reward truly improves exploration, not just training scores                                      |
| Hierarchical and Goal-Conditioned RL    | HER, Options, HIRO, BabyAI, Meta-World                                                                                                                                              | Split long-term tasks into subgoals, compare flat policy and hierarchical policy                                         |
| Multi-Task and Generalization           | Procgen, Meta-World, LIBERO, ContinualWorld                                                                                                                                         | High environment scores are not enough; also test on unseen tasks and seeds                                              |
| Multi-Agent Cooperation and Competition | PettingZoo, OpenSpiel, SMAC, Google Research Football, JaxMARL                                                                                                                      | Compare independent PPO, MAPPO, QMIX, self-play                                                                          |
| Robotics Manipulation                   | MuJoCo, Isaac Lab, ManiSkill, robosuite, Meta-World                                                                                                                                 | Do reaching, pushing, pick-and-place, then add visual input                                                              |
| World Models and Planning               | DreamerV3, TD-MPC2, mbrl-lib, MBPO, IRIS                                                                                                                                            | Learn dynamics models first, then compare model-free and model-based sample efficiency                                   |
| Industrial Applications                 | [RecSim](https://github.com/google-research/recsim), [FinRL](https://github.com/AI4Finance-Foundation/FinRL), [Pearl](https://github.com/facebookresearch/Pearl)                    | Bandit/RL experiments in recommendation, advertising, and financial trading, focus on offline evaluation and risk        |

##### Getting Started with Unity ML-Agents

[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) supports reinforcement-learning experiments inside a 3D game engine. Unlike a two-dimensional grid or a physics-only simulator, it can combine perspective, occlusion, gravity, collision, and programmable environments for visual navigation and spatial-reasoning tasks.

**Typical Use Cases**:

```python
# Unity ML-Agents can be wrapped with a Gymnasium-compatible interface.
from mlagents_envs.environment import UnityEnvironment

# Load a prebuilt Unity environment.
env = UnityEnvironment(file_name="3DBall")

# Convert the ML-Agents API to a Gymnasium-style interface.
from mlagents_envs.gym_utils import UnityToGymWrapper
gym_env = UnityToGymWrapper(env)

# Train the wrapped environment with Stable-Baselines3.
from stable_baselines3 import PPO
model = PPO("MlpPolicy", gym_env)
model.learn(total_timesteps=100000)
```

**Classic ML-Agents Environment Examples**:

| Environment   | Task Type                 | Difficulty | Suitable for Practice                         |
| ------------- | ------------------------- | ---------- | --------------------------------------------- |
| 3DBall        | Balance Control           | Beginner   | Understanding continuous action spaces        |
| Crawler       | Four-Legged Walk          | Medium     | Continuous control + multi-joint coordination |
| Walker        | Two-Legged Walk           | Medium     | Comparing with PyBullet's Walker2d            |
| PushBlock     | Push Block                | Beginner   | Goal-conditioned RL                           |
| FoodCollector | Food Collection           | Medium     | Multi-objective + navigation                  |
| HideAndSeek   | Multi-Agent Hide-and-Seek | Advanced   | Multi-agent emergent behavior                 |

Installation and environment retrieval methods are referenced in the [Environment Setup Guide](/preface/env-setup).

##### Classic Milestone Project References

The following are 30 common game and simulation replication directions from the pre-LLM era, organized by theme:

###### Classic/Board Games

| #   | Name         | Game/Environment | Year | Key Information                                                    |
| --- | ------------ | ---------------- | ---- | ------------------------------------------------------------------ |
| 1   | TD-Gammon    | Backgammon       | 1992 | Gerald Tesauro, achieved expert level through self-play RL         |
| 2   | Deep Blue    | Chess            | 1997 | IBM, defeated world champion Kasparov, search-based, not purely RL |
| 3   | AlphaGo      | Go               | 2016 | DeepMind, RL + MCTS defeated Lee Sedol                             |
| 4   | AlphaGo Zero | Go               | 2017 | No human game data, learned from self-play                         |
| 5   | AlphaZero    | Go, Chess, Shogi | 2017 | General-purpose RL algorithm for all three games                   |
| 6   | MuZero       | Go, Chess, Atari | 2020 | No explicit game rules, learns model and strategy together         |

###### Atari Series

| #   | Name                             | Game/Environment | Year | Key Information                                                                                     |
| --- | -------------------------------- | ---------------- | ---- | --------------------------------------------------------------------------------------------------- |
| 7   | DQN (Playing Atari with Deep RL) | Atari 2600       | 2013 | First use of deep RL to learn strategies for multiple games directly from pixels                    |
| 8   | Human-level Control through DRL  | Atari 2600       | 2015 | Nature 2015, improved version of DQN, achieving human-level performance on multiple Atari games     |
| 9   | Prioritized Experience Replay    | Atari            | 2015 | Improved experience replay with prioritized sampling of high TD error experiences                   |
| 10  | Rainbow DQN                      | Atari            | 2017 | Integrates Double DQN, Dueling, PER, NoisyNet, Distributional RL, and n-step return                 |
| 11  | IQN (Implicit Quantile Networks) | Atari            | 2018 | Distributional reinforcement learning, learning the quantile representation of return distributions |

###### Real-Time Strategy / MOBA

| #   | Name                                      | Game/Environment | Year | Key Information                                                                 |
| --- | ----------------------------------------- | ---------------- | ---- | ------------------------------------------------------------------------------- |
| 12  | SC2LE (StarCraft II Learning Environment) | StarCraft II     | 2017 | DeepMind provides the RL research environment and benchmark for StarCraft II    |
| 13  | AlphaStar                                 | StarCraft II     | 2019 | Multi-agent RL achieves Grandmaster level                                       |
| 14  | TStarBot                                  | StarCraft II     | 2019 | Tencent's proposed StarCraft II agent system                                    |
| 15  | OpenAI Five                               | Dota 2           | 2019 | 5v5 defeats world champion OG, large-scale distributed RL                       |
| 16  | Honor of Kings 1v1                        | Honor of Kings   | 2020 | Tencent AI Lab, dual-clip PPO, mastering complex control                        |
| 17  | Honor of Kings 5v5                        | Honor of Kings   | 2020 | MOBA AI system with multiple heroes, roles, and global collaboration            |
| 18  | Honor of Kings Arena                      | Honor of Kings   | 2022 | Open-ended MOBA RL environment, focusing on generalization challenges           |
| 19  | Mini Honor of Kings                       | Honor of Kings   | 2024 | Lightweight MARL environment, suitable for personal devices and course projects |

###### FPS / 3D Games

| #   | Name                              | Game/Environment | Year | Key Information                                                                       |
| --- | --------------------------------- | ---------------- | ---- | ------------------------------------------------------------------------------------- |
| 20  | Playing FPS Games with Deep RL    | ViZDoom          | 2016 | Applying Deep RL to FPS games, including visual input and partially observable states |
| 21  | Quake III Arena: Capture the Flag | Quake III CTF    | 2019 | DeepMind, complex team coordination and multi-agent emergent behaviors                |
| 22  | Obstacle Tower                    | Unity 3D         | 2019 | Testing 3D navigation, visual generalization, and long-range exploration              |
| 23  | Sample Efficient RL in Minecraft  | Minecraft/MineRL | 2021 | Using human demonstration data to improve sample efficiency in Minecraft              |

###### Sports/Racing/Other

| #   | Name                     | Game/Environment | Year | Key Information                                                                              |
| --- | ------------------------ | ---------------- | ---- | -------------------------------------------------------------------------------------------- |
| 24  | Google Research Football | Football 11v11   | 2020 | Open-source football simulator supporting multi-agent RL research                            |
| 25  | RL in Rocket League      | Rocket League    | 2022 | High-dimensional continuous control and team coordination in a racing and soccer hybrid game |
| 26  | Deep RL for Flappy Bird  | Flappy Bird      | 2015 | Early deep RL project in a game setting                                                      |

###### Multi-Agent/Comprehensive

| #   | Name                             | Game/Environment          | Year | Key Information                                                                    |
| --- | -------------------------------- | ------------------------- | ---- | ---------------------------------------------------------------------------------- |
| 27  | Deep RL for General Game Playing | General Game              | 2020 | Extends AlphaZero-like methods to general game playing                             |
| 28  | OpenSpiel                        | Board/Card Games          | 2019 | DeepMind's game framework, including various games and classic game algorithms     |
| 29  | Hide-and-Seek                    | Multi-Agent Hide-and-Seek | 2019 | OpenAI, emergent tool use and complex strategies in multi-agent self-play          |
| 30  | Multi-Agent RL in Video Games    | Survey                    | 2025 | Covers Rocket League, Doom, Minecraft, StarCraft, Dota, MOBA, and other directions |

#### The Age of LLMs and Post-Training, Reasoning, Agentic, VLM, and World Models

In the age of large language models (LLMs), reinforcement learning (RL) is no longer just about "maximizing a score in a fixed environment." Actions can be a piece of text, a search, a tool call, a webpage click, a code patch, a visual localization, or even an entire multi-step agent trajectory. Rewards have expanded from environmental scores to preference models, rule validation, process rewards, unit tests, webpage task success rates, and multimodal grounding signals.

##### Modern and Classical Resource Quick Reference

The recommended reading order is to first study classical papers and official documentation to build a conceptual foundation, then choose a "small model + verifiable reward" project to run through, and finally move on to distributed training, deep research, GUI/Computer Use, and multimodal environments.

| Direction                          | Recommended to Read First                                                                                                                                                                                                                                                            | Type                                      | Why It's Worth Reading                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| RLHF / Post-Training Classics      | [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155), [Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback), [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)                                                    | Classical Papers / Official Documentation | Understand the basic paradigms of SFT, RM, PPO, DPO, RLAIF, and safety alignment                                       |
| Modern Post-Training Engineering   | [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL), [verl](https://github.com/verl-project/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                               | Production / Research Training Frameworks | Directly study how rollout, vLLM/SGLang, Ray, Megatron, GRPO/DAPO, and asynchronous agentic RL are implemented         |
| Reasoning RLVR                     | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [DeepSeek-R1 Nature](https://www.nature.com/articles/s41586-025-09422-z), [Open-R1](https://github.com/huggingface/open-r1), [TinyZero](https://github.com/Jiayi-Pan/TinyZero)                                            | Modern Reasoning Reproduction             | Most suitable for learning verifiable rewards, GRPO/RLVR, cold-start data, long reasoning, and reward hacking          |
| Open-Source Large Model Foundation | [Qwen3.6](https://github.com/QwenLM/Qwen3.6), [Qwen3](https://github.com/QwenLM/Qwen3), [Meta Llama Models](https://github.com/meta-llama/llama-models)                                                                                                                              | Open-Source Models                        | Ideal for doing SFT/DPO/GRPO, tool calls, long context, and agentic coding experiments                                 |
| Deep Research                      | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/), [Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch), [WebThinker](https://github.com/RUC-NLPIR/WebThinker), [Search-R1](https://github.com/PeterGriffinJin/Search-R1)            | Products / Open-Source Research           | Turn search, reading, evidence screening, citation, and long reports into trainable trajectories                       |
| Agent Frameworks and Tool Calls    | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), [Google ADK](https://github.com/google/adk-python), [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning), [AutoGen](https://github.com/microsoft/autogen)                                  | Agent Engineering Frameworks              | Learn the engineering boundaries: tools, handoff, guardrails, tracing, session, agent trajectory, and RL interface     |
| GUI / Computer Use                 | [OpenAI CUA](https://openai.com/index/computer-using-agent/), [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool), [ByteDance UI-TARS](https://github.com/bytedance/UI-TARS), [OSWorld](https://os-world.github.io/)           | Models / Tools / Benchmarks               | Core materials for modern computer use: screenshots, coordinate actions, and webpage/desktop/mobile task success rates |
| VLM / VLA / Robotics               | [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner), [Gemini Robotics](https://deepmind.google/models/gemini-robotics/), [LeRobot](https://github.com/huggingface/lerobot)                             | Multimodal / Embodied                     | Connect visual question answering, localization, GUI clicks, robot actions, and verifiable rewards                     |
| World Models                       | [DreamerV3 Nature](https://www.nature.com/articles/s41586-025-08744-2), [DreamerV3 Code](https://github.com/danijar/dreamerv3), [Google DeepMind Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | Classical / Frontier / Simulation         | From reproducible world models to interactive world generation, and then to robot parallel simulation                  |
| Generative Model RL                | [DDPO](https://github.com/jannerm/ddpo), [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo), [AlignProp](https://align-prop.github.io/), [RLAIF-V](https://github.com/RLHF-V/RLAIF-V), [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                | Image / Video / Multimodal Rewards        | Learn to turn aesthetics, preferences, safety, textual-visual consistency, or video quality into optimization goals    |

**Subnavigation**:

- [LLM Post-Training](#llm-post-training)
- [LLM Reasoning](#llm-reasoning)
- [Deep Research RL](#deep-research-rl)
- [Agentic RL and Tool Calling](#agentic-rl-and-tool-invocation)
- [GUI and Computer Use](#gui-and-computer-use)
- [VLM](#vlm)
- [World Models and Simulators](#world-models-and-simulators)
- [Generative Model RL](#generative-model-rl)
- [Evaluation Benchmarks and Projects](#evaluation-benchmarks-and-projects)
- [Reproduction Order Suggestions](#reproduction-order-recommendation)

##### LLM Post-Training

LLM post-training primarily deals with "how to make the model more aligned with human preferences, task formats, and safety constraints." The key terms at this level are SFT, Reward Model, PPO, DPO, KTO, ORPO, RLOO, and GRPO. For beginners, it is recommended to first run TRL with a small model, and then move on to distributed training stacks such as OpenRLHF, verl, and NeMo-RL.

This direction requires a clear understanding of three things: **where the data comes from, how to give rewards, and how to update the policy.** SFT addresses "whether the model can answer in the required format," preference optimization addresses "which of two responses is preferred," and PPO/GRPO/RLOO address "how the model can improve further based on rewards after self-sampling." Many post-training projects fail not because of algorithm errors, but because these three aspects are mixed together: using SFT data to expect the model to explore, using preference data to expect the model to learn verifiable reasoning, or using overly simplistic rewards to expect the model to gain stable capabilities.

###### Reproduction Goals

###### Minimal Project Design

A minimal project can use an instruction-tuned model with 0.5B to 3B parameters. First apply supervised fine-tuning to a small instruction dataset, then run one round of preference optimization with DPO or KTO, and finally use PPO or GRPO with a rule-based reward or a small reward model. Keep a fixed set of 20–50 evaluation prompts at every stage. Track response length, format compliance, refusal rate, repeated phrasing, and human preference. These observations show which behaviors post-training actually changes, beyond the loss curve alone.

---

###### Choosing a Framework

Use TRL when the goal is to understand the algorithm: its trainer interfaces make the loss, reward, and KL terms easy to inspect. LLaMA-Factory and ms-swift reduce setup time for quick experiments, especially with Chinese-language and multimodal models. For industrial training systems, study OpenRLHF, verl, and NeMo-RL. They expose the interactions among rollout generation, optimization, inference engines, weight synchronization, memory, and throughput.

| Resource                                                                                                          | Focus                                                                  | Best use                                                                         |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155)                                                            | The classic SFT, reward-model, and PPO pipeline                        | Understanding why RLHF became a standard LLM post-training method                |
| [Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) | RLAIF, safety preferences, and principle-based feedback                | Understanding AI feedback plus preference optimization                           |
| [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)                                                            | An industrial description of open-model post-training                  | Studying how pretraining, instruction tuning, and safety evaluation fit together |
| [Hugging Face TRL](https://huggingface.co/docs/trl/index)                                                         | PPO, DPO, KTO, ORPO, RLOO, and GRPO                                    | Small post-training experiments and algorithm study                              |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)                                                                  | Ray, vLLM, DeepSpeed, PPO, GRPO, and DPO                               | Reproducing larger RLHF and RLVR systems                                         |
| [verl](https://github.com/verl-project/verl)                                                                      | Distributed rollout, PPO, GRPO, DPO, and SFT                           | R1-style pipelines and training with complex rewards                             |
| [Open-Instruct](https://github.com/allenai/open-instruct)                                                         | SFT, DPO, PPO, and preference alignment                                | Studying a complete open alignment pipeline                                      |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)                                                                      | Large-scale RL in the NVIDIA ecosystem                                 | Multi-GPU post-training and production training stacks                           |
| [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                 | Decoupled clipping, dynamic sampling, and token-level policy gradients | Connecting an R1-style training recipe with code                                 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)                                                         | SFT, DPO, PPO, KTO, and ORPO                                           | Rapid experiments with models common in the Chinese open-source community        |
| [ms-swift](https://github.com/modelscope/ms-swift)                                                                | SFT, RLHF, GRPO, and multimodal fine-tuning                            | Post-training Chinese-language and multimodal models                             |

###### Common Pitfalls

- **Only Looking at Reward Curves**: An increase in reward does not necessarily mean an improvement in quality. It could simply indicate that the model is producing longer answers, using more sophisticated formatting, or relying on more fixed templates.
- **Skipping Fixed Evaluation Sets**: Without a fixed prompt, it is difficult to determine whether the model is genuinely improving or if the observed changes are due to sampling randomness.
- **Ignoring KL Divergence and Length**: Post-training often pushes models toward a local optimum characterized by being long-winded, conservative, and repetitive.
- **Directly Applying Large Frameworks**: Without first thoroughly understanding the reward and loss dynamics in TRL, large-scale frameworks can amplify existing issues.

##### LLM Reasoning

The core of reasoning-based reinforcement learning is not to "make the answer more aligned with human preferences," but rather to "teach the model to learn verifiable reasoning processes in mathematics, coding, logic, and symbolic tasks." In this context, rewards can come from final answers, unit tests, format validation, symbolic execution, and process checking.

This direction differs from standard RLHF in a key way: rewards do not have to come from human preferences, but can instead be derived from a **verifier**. Math problems can be judged by answers, code problems can be tested by running the code, format problems can be validated by regular expressions, and symbolic problems can be executed by interpreters. As long as the verifier is reliable, the model can obtain training signals through extensive self-sampling. This is why works like RLVR, GRPO, and R1 are worth studying together.

###### Reproduction Goals

In the first stage, it is recommended to start with tasks such as Countdown, arithmetic problems, formatted JSON, and simple Python functions. The goal is not to chase rankings, but to personally observe four phenomena: the model explores different solution methods; incorrect answers may also have high-confidence reasoning; training is unstable when rewards are sparse; and once the verifier has a flaw, the model quickly exploits it.

In the second stage, transition to tasks such as GSM8K, MATH, code unit tests, or reasoning-gym generated tasks.

###### Project Structure

A clean reinforcement learning (RL) project for reasoning tasks should have at least five distinct file layers: task generation, answer parsing, reward function, rollout sampling, and training configuration. Avoid hiding the answer parsing logic within the reward function, as this will make debugging bad cases extremely difficult. After each training session, it is essential to sample and save the following: the question, the model's full output, the parsed answer, the standard answer, the reward, and the failure reason. This approach provides significantly more insight than simply looking at the average score.

| Resource                                                                       | Focus                                                                    | Suitable for                                                                                              |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)                      | R1/R1-Zero, RLVR, distillation models                                    | Directly reading the official open-source entry for modern reasoning RL                                   |
| [DeepSeek-R1 Nature](https://www.nature.com/articles/s41586-025-09422-z)       | The official paper version of DeepSeek-R1                                | Understanding cold start, RL, and the evaluation of rejectability and readability                         |
| [Qwen3.6](https://github.com/QwenLM/Qwen3.6)                                   | Alibaba Qwen's next-generation open-source model                         | As a modern base for testing SFT, GRPO, tool calling, and long context                                    |
| [Qwen3](https://github.com/QwenLM/Qwen3)                                       | Thinking / non-thinking mode                                             | Comparing the training and inference differences between reasoning models and standard instruction models |
| [Open-R1](https://github.com/huggingface/open-r1)                              | Reimplementation of DeepSeek-R1 training components                      | Studying the data, SFT, RL, and evaluation process of reasoning models                                    |
| [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) | Open-source framework for training reasoning models from scratch         | Reimplementing R1-Zero-style RLVR, data generation, and evaluation process                                |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero)                              | Small-scale R1-Zero-style RLVR                                           | Understanding GRPO/RLVR using tasks like Countdown                                                        |
| [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                              | KL-free, hard problem sampling, clip-higher, token-level policy gradient | Understanding the engineering details of R1-like recipes after 2025                                       |
| [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) | Self-play reasoning RL without human-annotated data                      | Studying code executor, self-play, and verifier combinations                                              |
| [rLLM / DeepScaleR](https://github.com/rllm-org/rllm)                          | GRPO, REINFORCE, RLOO for math, code, and agent tasks                    | Studying long context, verifiable rewards, and agent trajectory training                                  |
| [reasoning-gym](https://github.com/open-thought/reasoning-gym)                 | Automatically generated reasoning tasks                                  | Customizing verifiers and verifiable training tasks                                                       |
| [Math-Verify](https://github.com/huggingface/Math-Verify)                      | Mathematical answer parsing and verification                             | Writing reward functions for mathematical reasoning tasks                                                 |
| [OpenThoughts](https://github.com/open-thoughts/open-thoughts)                 | Open-source reasoning data and recipes                                   | Constructing R1-like SFT/RL data and evaluation sets                                                      |

###### Common Pitfalls

- **Too Fragile Answer Analysis**: The model answers correctly but the analysis fails, or answers incorrectly but gets rewarded due to formatting loopholes.
- **Long Reasoning as Good Reasoning**: Longer reasoning may simply indicate the model has learned to procrastinate, not that the logic is more accurate.
- **Too Easy Training Set**: The reward quickly saturates, and the model only learns to use fixed templates.
- **Too Difficult Training Set**: Almost all rewards are zero, and the policy receives no gradient signals.

##### Deep Research RL

Deep Research RL focuses on "how models actively retrieve, read, filter evidence, and generate research-based answers." It differs from ordinary RAG in that the retrieval behavior itself is a strategy. The model must learn when to search, what to search for, which pages to read, how to cite evidence, and when to stop.

Ordinary RAG typically involves a one-time retrieval: the user asks a question, the system retrieves several documents, and the model generates an answer. Deep Research, on the other hand, resembles a multi-turn decision process: first, break down the question, retrieve the first set of documents, read them, identify gaps, reformulate the query, continue searching for evidence, and finally synthesize the findings into a report. The value of RL lies in training these intermediate decisions, not just the final text.

###### Reproduction Goal

Minimal reproduction does not necessarily need to start with real internet access. One can first prepare a static document library and encapsulate three tools: `search`, `open`, and `find`, allowing the model to complete multi-hop QA in a controlled environment. Rewards can be divided into four categories: whether the final answer is correct, whether citations come from real documents, whether the evidence supports the conclusion, and whether the number of tool calls is reasonable. Once this closed-loop system is stable, it can be migrated to BrowserGym, WebArena, or a real search environment.

###### Observation Focus

The logs of Deep Research are more important than the final answer. You should observe whether the model repeatedly searches for the same keyword, whether it only opens the top-ranked page, whether it cites sources it has not read, and whether it forces a conclusion when evidence is insufficient. A good agent is not only "correct" but should also demonstrate planning, cross-validation, revisiting when conflicts arise, and narrowing the scope of conclusions when evidence is lacking.

| Resource                                                                    | Focus                                                          | Suitable for Doing                                                                                                  |
| --------------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) | Product-level deep research agent                              | Compare with real-world product forms, understand long-range retrieval, evidence integration, and report generation |
| [Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)  | Open-source deep research series collection                    | Track research lines such as WebWalker, WebDancer, WebSailor, and WebWatcher                                        |
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1)                   | Search-enhanced reasoning RL                                   | Train models to actively retrieve and integrate evidence during reasoning                                           |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN)                                  | Generative agent environment with reasoning trajectory         | Construct research/search environments and train multi-step strategies                                              |
| [rLLM](https://github.com/rllm-org/rllm)                                    | Verifiable agent tasks such as search, math, code, and finance | Decouple reward function, rollout, and trainer                                                                      |
| [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)                | RL training framework for deep research                        | Study multi-step retrieval, information filtering, and long-answer synthesis                                        |
| [WebThinker](https://github.com/RUC-NLPIR/WebThinker)                       | Deep research agent with Web search + reasoning                | Train/evaluate long-range reasoning strategies with search                                                          |
| [WebArena](https://github.com/web-arena-x/webarena)                         | Real-world website task environment                            | Perform web information retrieval, cross-page tasks, and final success rate evaluation                              |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym)                      | Unified browser environment interface                          | Connect search, click, reading, and form-filling into agent training environments                                   |
| [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)                       | Web operation dataset                                          | Transition from behavior cloning to web agent RL                                                                    |

###### Common Pitfalls

- **Rewarding Only the Final Answer**: In long-horizon tasks, rewarding only the final state makes credit assignment extremely difficult.
- **Using Citation Count as Quality**: Models may learn to stack links rather than support claims with evidence.
- **Unreproducible Real Web Pages**: Page changes, search ranking variations, and broken links can contaminate training results.
- **Ignoring Cost Constraints**: Infinite search can improve hit rates, but real systems must control latency and API call costs.

##### Agentic RL and Tool Invocation

In Agentic RL, actions are not only outputting tokens, but also invoking tools, executing code, browsing web pages, modifying files, querying databases, or interacting with external systems. The challenge lies in long trajectories, heterogeneous actions, delayed rewards, and complex failure types, making it more reliant on environment wrappers, trajectory segmentation, and verifier design compared to standard RLHF.

This direction can be viewed as "training a large language model within a stateful environment." At each step, the model outputs not only natural language but also tool names, parameters, code snippets, SQL queries, shell commands, or next steps. Training data is no longer single prompt-response pairs, but full episodes: states, actions, tool returns, environmental changes, rewards, and termination reasons must all be recorded.

###### Reproducibility Goals

For beginners, it is recommended to start with low-risk tools: calculators, file search, restricted Python execution, and text-world environments. First, let the model learn "when to invoke a tool," then "which tool to invoke," and finally "how to generate complex parameters." For code agents, unit tests can serve as rewards. For business process agents, database state satisfaction with constraints can be the reward. For web agents, task completion rate can be the reward.

###### Engineering Splitting

Do not start by writing the agent framework, environment, reward, and trainer all together. A more stable structure is: the environment is responsible only for executing actions and returning observations; the reward is responsible only for scoring; the trajectory store is responsible only for recording; and the trainer consumes only the transformed samples. Projects like Agent Lightning, AReaL, and SkyRL are worth looking at precisely because of how they split these boundaries.

| Resource                                                                                                  | Focus                                                | Suitable for Doing                                                                        |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)                                       | tools, handoffs, guardrails, tracing                 | Build a modern agent shell, and connect the trajectory to evaluation or RL data pipelines |
| [Google ADK](https://github.com/google/adk-python)                                                        | code-first agent, evaluation, deployment             | Learn Google-style agent engineering interfaces and multi-tool orchestration              |
| [AutoGen](https://github.com/microsoft/autogen)                                                           | Multi-agent dialogue, tool and application framework | Do multi-agent collaboration, code/data tasks, and agent baseline                         |
| [Agent Lightning](https://github.com/microsoft/agent-lightning)                                           | Split agent execution trajectory into trainables     | Connect RL to existing systems like LangChain, AutoGen, and OpenAI Agents                 |
| [AReaL](https://github.com/inclusionAI/AReaL)                                                             | Asynchronous LLM RL and agentic RL                   | Study the decoupling of rollout and training, asynchronous updates, and system throughput |
| [SkyRL](https://docs.skyrl.ai/docs)                                                                       | Large-scale LLM/agent RL training stack              | Understand the engineering hierarchy of rollout, trainer, reward, and evaluation          |
| [AgentGym](https://github.com/WooooDyy/AgentGym)                                                          | Multi-environment LLM agent training                 | Do multi-task agent training and environment interface comparisons                        |
| [ALFWorld](https://github.com/alfworld/alfworld), [ScienceWorld](https://github.com/allenai/scienceworld) | Text-based worlds and scientific experiments         | Low-cost research on long-term planning, language actions, and sparse rewards             |
| [ToolBench](https://github.com/OpenBMB/ToolBench)                                                         | API tool calling                                     | Study tool selection, parameter generation, and call success rate                         |
| [tau-bench](https://github.com/sierra-research/tau-bench)                                                 | Tool agents in real business processes               | Customer service, orders, database status, and multi-turn constrained tasks               |
| [WebShop](https://github.com/princeton-nlp/WebShop)                                                       | Web shopping search and decision                     | Classic language agent environment, suitable for supervised-to-RL                         |
| [SWE-bench](https://github.com/swe-bench/SWE-bench)                                                       | Software engineering agent evaluation                | Use test pass rate as a verifiable reward for code agents                                 |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)                                                             | Software engineering RL environment                  | Interactive environment for training coding agents                                        |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench)                                       | Terminal task agents                                 | Study shell actions, long-term execution, and task success rate                           |

###### Common Pitfalls

- **Loss Based on Tool Output**: Tool outputs are not generated tokens by the model. The model should not be held responsible for the results of the tools.
- **Irrecoverable Environment State**: A single failure contaminates subsequent tasks, making the training data difficult to interpret.
- **Reward Too Late**: Only providing success or failure at the end makes it unclear to the model which step in the decision-making process went wrong.
- **Too Large Action Space**: Directly opening shell, browser, and file system can make exploration extremely inefficient.

##### GUI and Computer Use

The GUI/Computer Use direction expands the action space from tokens to include clicks, inputs, scrolls, shortcuts, window switching, and coordinate positioning. It differs from Deep Research's web reading, emphasizing real interface state changes, visual grounding, long-term operations, and task completion rates.

The challenge for GUI agents lies in the fact that "the same action description can be completely different under different screen states." Button positions change, popups appear, scroll regions may be obscured, and layouts differ between mobile and desktop interfaces. Therefore, the core here is not to have the model memorize the UI, but to align the model with screenshots, DOM, accessibility trees, historical actions, and task goals.

###### Reproduction Goals

It is recommended to start with behavior cloning or action ranking, rather than directly using reinforcement learning. Use environments like Mind2Web or BrowserGym, where the model selects the next action given a web state. Once the action format stabilizes, introduce online rollout and task success rewards. Desktop GUI and mobile GUI can begin with short tasks, such as opening settings, searching for information, and filling out simple forms, and then expand to multi-step cross-application workflows.

| Resource                                                                                                  | Focus                                | Suitable for What                                                                                  |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| [OpenAI CUA](https://openai.com/index/computer-using-agent/)                                              | Operator-backed computer-using agent | See how modern GUI agents combine vision, reasoning, mouse/keyboard actions, and safety boundaries |
| [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) | Claude's computer use tool interface | Learn screenshot, coordinate actions, tool feedback, beta API, and safety considerations           |
| [ByteDance UI-TARS](https://github.com/bytedance/UI-TARS)                                                 | Native GUI agent model               | Perform unified GUI operations and grounding reproduction across desktop, web, and mobile          |
| [WebArena](https://github.com/web-arena-x/webarena)                                                       | Real website task environment        | Web forms, account management, shopping, and content management tasks                              |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena)                                           | Multimodal web agent                 | Combine screenshots, DOM, and language for web operations                                          |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym)                                                    | Browser environment interface        | Integrate web agents, VLM agents, and RL rollouts                                                  |
| [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)                                                     | Web operation dataset                | Transition from behavior cloning, action ranking to online RL                                      |
| [OSWorld](https://os-world.github.io/)                                                                    | Desktop GUI agent benchmark          | Real software tasks, screenshot understanding, coordinate actions, and task success rate           |
| [AndroidWorld](https://github.com/google-research/android_world)                                          | Android mobile agent                 | Mobile UI operations, app switching, and long-term tasks                                           |

###### Common Pitfalls

- **Coordinate Overfitting**: The model memorizes screen positions rather than understanding the semantics of controls.
- **Inconsistency Between Screenshots and DOM**: Elements visually visible may not be easily locatable in the DOM, and DOM elements may be occluded.
- **Evaluation Based Only on Final Page**: Some tasks require intermediate states to be correct, such as saving settings, submitting a form, or confirming an order.
- **Lack of Failure Classification**: Clicking the wrong button, not finding an element, inputting the wrong format, or slow page loading should be categorized separately.

##### VLM

In the direction of VLM (Vision-Language Model) reinforcement learning, the focus is on connecting visual understanding, language reasoning, and action grounding. Rewards can come from the correctness of text and image answers, OCR accuracy, bounding box or point localization, GUI click positions, robot task success rates, or from preference models like ImageReward.

VLM reinforcement learning is not merely "adding a reward to image-based question answering." It at least includes four types of tasks: visual question answering, spatial localization, GUI operations, and visual control. The rewards for these tasks are completely different: for QA, the reward is based on the answer; for localization, it is based on IoU or point error; for GUI tasks, it is based on whether the action changes the interface state; and for robots, it is based on whether the task is completed. When reproducing, it is essential to clearly identify the task type, otherwise the reward function may become a patchwork of unrelated components.

###### Reproduction Goals

The minimal project can start with text and image answer validation, for example, given an image and a question, let the model output answers in a fixed format, and use rules or external annotations for scoring. The next step is to perform grounding, where the model outputs a bounding box or a click point, and the performance is evaluated using IoU, distance error, or target hit rate. Finally, move on to GUI or robot tasks, as these tasks introduce continuous states, long-range actions, and real-world environmental noise.

###### Training Observations

VLM training must save visual bad cases. Saving only text logs is insufficient: you need to combine the original image, model annotations, target boxes, output answers, and the reward in one view. Many errors are not due to reasoning mistakes, but rather OCR errors, small target objects, coordinate system conversion errors, point displacement after image scaling, or the model being misled by irrelevant text in the image.

| Resource                                                                               | Suitable for Reproduction                                 | Focus Points                                                                                    |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)                       | Introduction to GRPO/RLVR for VLMs                        | Can write reward as correctness, formatting, and quality of image-text answers                  |
| [HF VLM GRPO Cookbook](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl) | Hands-on practice with small-scale VLM GRPO               | End-to-end examples from data, reward function to training scripts                              |
| [VLM-R1](https://github.com/om-ai-lab/VLM-R1)                                          | RL reasoning for vision-language models                   | Suitable for viewing multi-modal chain-of-thought and verifiable rewards                        |
| [LMM-R1](https://github.com/TideDra/lmm-r1)                                            | Two-stage training framework for multi-modal reasoning    | Research on cold start, GRPO, and VLM reasoning chains                                          |
| [Open-Vision-Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner)     | RL training for vision reasoning models                   | Migrating R1-Zero-like methods to multi-modal reasoning                                         |
| [RL4VLM](https://rl4vlm.github.io/)                                                    | RLHF/RLVR for vision-language models                      | Project page aggregates research on RL for VLM alignment and reasoning                          |
| [Gemini Robotics](https://deepmind.google/models/gemini-robotics/)                     | Google DeepMind VLA/embodied reasoning                    | Understanding how VLM/VLA transition to real robot actions and multi-step tasks                 |
| [LeRobot](https://github.com/huggingface/lerobot)                                      | Real and simulated robot data, imitation learning, and RL | Suitable for transitioning from visual action data to embodied RL                               |
| [RoboCasa](https://github.com/robocasa/robocasa)                                       | Home scene robot operations                               | Suitable for studying multi-task visual control, language-conditioned tasks, and generalization |
| [ManiSkill](https://maniskill.readthedocs.io/)                                         | Visual operations and parallel robot environments         | Can connect VLM policy, visual reward, and low-level control                                    |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena)                        | Multi-modal web agent                                     | Requires image, DOM, and language to be grounded together                                       |
| [OSWorld](https://github.com/xlang-ai/OSWorld)                                         | Desktop GUI agent                                         | Suitable for studying screenshot understanding, coordinate actions, and real software tasks     |
| [AndroidWorld](https://github.com/google-research/android_world)                       | Android phone agent                                       | Phone UI operations, visual grounding, and long-term tasks                                      |
| [OpenVLA](https://github.com/openvla/openvla)                                          | Vision-language-action model                              | Suitable for connecting VLM, robot actions, and downstream RL fine-tuning                       |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                            | Lifelong robot learning                                   | Multi-task, language-conditioned, generalization, and visual operations                         |
| [ImageReward](https://github.com/THUDM/ImageReward)                                    | Image generation/multi-modal reward models                | Can serve as a reward signal for generative models or VLM preference optimization               |

###### Common Pitfalls

- **Only Looking at Text Accuracy**: Visual grounding may be wrong, yet the answer may still happen to be correct.
- **Confusion in Coordinate Systems**: When the original image size, model input size, and rendering size are inconsistent, bounding boxes or points are easily misaligned.
- **Data Leakage**: Some VQA benchmarks may have been seen during pre-training, so it is necessary to supplement evaluations with self-built or time-sliced data.
- **Uninterpretable Rewards**: When a preference model gives a high score, it should be possible to trace back what visual attributes were actually rewarded.

##### World Models and Simulators

A world model focuses on "whether an agent can learn an environment model that can be used for imagination and planning." In classical reinforcement learning, it connects model-based RL, planning, and sample efficiency; in the age of LLMs and agents, it can also serve as a controllable simulator for training long-horizon tasks, web/tool environments, and embodied intelligence.

The value of a world model is not to "replace the real environment," but to reduce the cost of trial and error. The agent first learns an approximate dynamics model from real interactions, then performs short rollouts, planning, or strategy improvements within the model. The key issue is always model bias: if the model has even a slight systematic error, the planner may repeatedly exploit this error, eventually learning strategies that are not applicable in the real world.

###### Reproduction Goals

For beginners, one can start by implementing a 2D continuous control or a variant of CartPole: collect real transitions, train a small model that predicts the next state and reward, and then compare the sample efficiency of model-free PPO/SAC with model-based rollouts.

For advanced study, one can look into DreamerV3, TD-MPC2, and MBPO: they respectively represent the three typical approaches of latent imagination, MPC planning, and short model rollouts.

###### Relationship with LLM Agents

In Agentic RL, the simulator can also serve as a "task world model": web snapshots, tool responses, user states, and database states can all be encapsulated into reproducible environments. This allows training to occur without needing to access real websites or real business systems each time; instead, the agent can be trained within pre-recorded state transitions. While such simulated environments cannot fully replace real-world environments, they are highly suitable for offline debugging, badcase replay, and course projects.

| Resource                                                                                                                        | Suitable for Reproduction                      | Focus Area                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [DreamerV3 Nature](https://www.nature.com/articles/s41586-025-08744-2)                                                          | World model general capability paper           | See fixed hyperparameters, Minecraft, Atari, control tasks, etc. results                     |
| [Google Research Dreamer](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/) | Dreamer series official introduction           | Understand latent imagination and actor-critic using official explanations                   |
| [DreamerV3](https://github.com/danijar/dreamerv3)                                                                               | Pixel input world model                        | Latent dynamics, imagined rollout, actor-critic                                              |
| [Google DeepMind Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)                                | Interactive generative world model             | Understand how world models evolve from control environments to interactive world generation |
| [TD-MPC2](https://github.com/nicklashansen/tdmpc2)                                                                              | Continuous control and multi-task MBRL         | Learned latent model, MPC planning, sample efficiency                                        |
| [mbrl-lib](https://github.com/facebookresearch/mbrl-lib)                                                                        | PETS, MBPO, etc. classic MBRL                  | Dynamics model, uncertainty, planning                                                        |
| [MBPO](https://github.com/JannerM/mbpo)                                                                                         | Model-Based Policy Optimization                | Short model rollout plus off-policy RL                                                       |
| [IRIS](https://github.com/eloialonso/iris)                                                                                      | Atari world model                              | Discrete latent tokens, transformer dynamics, imagined rollout                               |
| [DayDreamer](https://github.com/danijar/daydreamer)                                                                             | Robotics world model                           | Transfer Dreamer's idea to real/simulated robot tasks                                        |
| [Brax](https://github.com/google/brax)                                                                                          | Differentiable physics and parallel simulation | Suitable for studying high-throughput model-free and model-based control                     |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)                                                                              | Large-scale robot RL                           | GPU parallel environments, domain randomization, Sim2Real                                    |
| [LeRobot](https://github.com/huggingface/lerobot)                                                                               | Data-driven embodied intelligence              | Imitation learning, diffusion policy, RL fine-tuning                                         |
| [RoboCasa](https://github.com/robocasa/robocasa)                                                                                | Home scene robot manipulation                  | Long-term tasks, multi-object interaction, language-conditioned control                      |

###### Common Pitfalls

- **Too Long Model Rollout**: Prediction errors accumulate with the number of steps. Shorter rollouts are often more stable.
- **Only Report Final Score**: Also report the number of real environment samples, model training cost, and planning cost.
- **Ignore Uncertainty**: When the model is uncertain, the strategy may drift into regions where the model is least reliable.
- **Too Clean Simulation**: Robotics and GUI tasks, in particular, require noise, latency, occlusion, and random initial states.

##### Generative Model RL

Generative Model RL focuses on directly optimizing image, video, or multi-modal generation results using a reward function. It is similar to language RLHF, but the reward typically comes from aesthetics scores, image-text consistency, safety, preference models, or human/AI feedback.

One can view the sampling process of a diffusion model as a trajectory: starting from noise, going through multiple denoising steps, and finally obtaining an image. Rewards are typically computed only on the final image, such as aesthetics scores, text consistency, safety classification, human preference, or multi-modal model scores. The core of methods like DDPO is to incorporate the generation process into policy gradient, rather than only doing supervised fine-tuning.

###### Reproduction Goals

For a minimal project, one can choose a very narrow prompt set, such as "generate a clear single object," "generate an icon with a specified color," or "generate a simple scene that matches the text description." First, fix the base model and sampling parameters, then compare three types of rewards: CLIP/text consistency, aesthetics or preference scores, and normalized safety constraints. After each training, save before/after images for the same set of prompts and visually inspect whether the reward truly corresponds to quality improvement.

| Resource                                                              | Focus                                                       | Suitable for Doing                                                                 |
| --------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [DDPO](https://github.com/jannerm/dd/F)                               | Classic implementation of diffusion model RL                | Understand how to perform policy gradient after treating denoising steps as an MDP |
| [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo) | Fine-tuning diffusion models with DDPO                      | Reproduce policy gradient optimization for text-to-image models                    |
| [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch)              | Original open-source implementation of DDPO                 | Study RL updates on the sampling trajectory of diffusion models                    |
| [AlignProp](https://align-prop.github.io/)                            | Optimizing generative models through differentiable rewards | Compare backprop-through-reward with policy gradient                               |
| [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)                          | Alignment via visual-language feedback                      | Use AI feedback to optimize multi-modal generation/understanding                   |
| [ImageReward](https://github.com/THUDM/ImageReward)                   | Image preference reward model                               | Provide reward signals for image generation RL                                     |
| [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)           | Human feedback alignment for video generation               | See how video reward, DPO/RL, and reject sampling can be combined                  |

###### Common Pitfalls

- **Reward Model Exploitation**: The model may learn to generate certain high-scoring textures, compositions, or keywords styles, rather than genuinely aligning with human preferences.
- **Only Looking at Average Reward**: It is essential to examine image samples; otherwise, it is easy for the reward to increase while the quality decreases.
- **Too Narrow Prompt Coverage**: Over-optimization on a small prompt set can damage generalization and diversity.
- **Safety and Aesthetics Conflict**: Multi-objective reward functions require tracking each sub-score, and cannot rely solely on a weighted total score.

##### Evaluation Benchmarks and Projects

In the LLM era, doing RL often leads to more issues in evaluation than in training. A project should at least consider the final success rate, process quality, format constraints, reward hacking, length bias, data leakage, and sampling stability. The following benchmarks can serve as entry points for reproducing projects.

Evaluation should not only ask "Has the score increased?" but also "Where has it increased, what is the cost, and has it damaged other capabilities." A qualified reproduction experiment should at least include: a fixed dev set, an unseen test set, pre- and post-training comparisons, mean and variance of multiple samples, badcase classification, length statistics, and cost statistics. For Agentic/VLM/GUI tasks, it is also important to save complete trajectories for replay.

###### Acceptance Checklist

- **Final Metrics**: Accuracy, pass rate, task success rate, preference win rate.
- **Process Metrics**: Tool call count, proportion of invalid actions, proportion of repeated search, reference accessibility rate, code test failure types.
- **Stability Metrics**: Whether the system remains effective under different random seeds, sampling temperatures, and model sizes.
- **Safety Metrics**: Whether it is more likely to fabricate references, overstep tool calls, leak environment information, or break format constraints.
- **Cost Metrics**: Average token usage, average tool calls, average latency, and GPU/CPU costs for training and evaluation.

| Direction            | Recommended Baselines/Projects                                                                                                                                                                                                      | Primary Evaluation Signals                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Reward Modeling      | [RewardBench](https://github.com/allenai/reward-bench)                                                                                                                                                                              | Whether the reward model truly favors better responses               |
| Math Reasoning       | [Math-Verify](https://github.com/huggingface/Math-Verify), [reasoning-gym](https://github.com/open-thought/reasoning-gym)                                                                                                           | Answer verifiability, formatting, and reasoning stability            |
| Code Reasoning       | [EvalPlus](https://github.com/evalplus/evalplus), [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)                                                                                                                   | Unit test pass rate, generalization to new problems                  |
| Software Engineering | [SWE-bench](https://github.com/swe-bench/SWE-bench), [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)                                                                                                                                  | Whether the patch resolves real-world issues                         |
| Web Navigation       | [WebArena](https://github.com/web-arena-x/webarena), [BrowserGym](https://github.com/ServiceNow/BrowserGym), [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)                                                                  | Web task success rate, click trajectory, state changes               |
| Multimodal Web       | [VisualWebArena](https://github.com/web-arena-x/visualwebarena)                                                                                                                                                                     | Screenshot grounding, DOM grounding, task success rate               |
| Desktop/Phone GUI    | [OSWorld](https://github.com/xlang-ai/OSWorld), [AndroidWorld](https://github.com/google-research/android_world)                                                                                                                    | Long-horizon success rate in real applications                       |
| Tool Calling         | [ToolBench](https://github.com/OpenBMB/ToolBench), [tau-bench](https://github.com/sierra-research/tau-bench)                                                                                                                        | Tool selection, parameter correctness, business flow completion rate |
| Robotics/VLA         | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ManiSkill](https://maniskill.readthedocs.io/), [RoboCasa](https://github.com/robocasa/robocasa)                                                                       | Task success rate, generalization, collision and safety cost         |
| Image Generation RL  | [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo), [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch), [AlignProp](https://github.com/mihirp1998/AlignProp), [RLAIF-V](https://github.com/RLHF-V/RLAIF-V) | Preference reward, aesthetics score, safety, consistency             |

###### Badcase Template

Each direction is advised to maintain a `badcases.jsonl` or table, at least recording: task id, input, model output, reward, scoring rationale, failure type, reproducibility, and repair suggestions. For LLM RL, badcase is not an afterthought after training; it is the entry point for the next round of reward design, data filtering, and environment fixes.

##### Reproduction Order Recommendation

Start with a small model of 0.5B to 3B and observable tasks such as math, code, and format validation to observe reward hacking, length bias, and sampling temperature effects. Then migrate from TRL/TinyZero/Open-R1 to distributed frameworks such as verl/OpenRLHF. For Agentic RL, prioritize tasks with clear success rates such as search, web, and code; for VLM RL, prioritize scorable tasks such as text-image answers, localization, OCR, and GUI clicks; for world models and embodied directions, first run DreamerV3/TD-MPC2, then add visual and real-robot complexity.

###### A Stable Route

1. **Week 1: Rule-based Reward Tasks**
   Run a small verifiable task using TRL or TinyZero, such as Countdown, JSON formatting, or simple math. The goal is to understand rollout, reward, advantage, KL, length bias, and log saving.

2. **Week 2: Preference Optimization and Post-Training Comparison**
   Use the same small model to perform SFT, DPO/KTO, and PPO/GRPO comparisons. Do not change too many variables; only observe the impact of different training methods on the same batch of prompts.

3. **Week 3: Reasoning RLVR**
   Introduce Math-Verify, reasoning-gym, or code unit tests to upgrade the reward from "format correctness" to "answer verifiability." Focus on observing reward sparsity and verifier vulnerabilities.

4. **Week 4: Tool Invocation or Deep Research**
   Package a small search/reading environment, and record complete trajectories. Begin with offline trajectory replay, then move to online rollout.

5. **Week 5: VLM or GUI**
   Choose a visual question answering, bounding box localization, or web-clicking task, and add visual badcases. Focus on checking coordinate systems, screenshot states, and reward interpretability.

6. **After Week 6: Distributed and Industrial Frameworks**
   Then move into frameworks such as Verl, OpenRLHF, AReaL, and SkyRL. At this point, you will already know what reward, what logging, and what evaluation you need. You will not be led astray by engineering complexity anymore.

###### How to Determine When to Upgrade Difficulty?

When a task satisfies three criteria, you can move to the next level: first, the performance difference before and after training on a fixed evaluation set is stable; second, badcases can be clearly categorized; third, when the reward increases, the quality of manual sampling also improves. Otherwise, do not rush to switch to a larger model or a more complex environment. First, improve the reward, data, and logging.
