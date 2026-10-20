# C.1 学习资源与项目路线

> 学完本书前 17 章后，读者已具备阅读强化学习前沿论文的全部数学与工程基础。但 RL 文献浩繁——arXiv 每年新增数千篇，真正决定方向、值得反复读的不到一百篇。本附录按主题把这批"必读论文"分类整理，每篇标注**核心贡献一句话**与**阅读优先级**（★ 必读 / ◆ 推荐 / ◇ 扩展），帮助读者建立从经典 RL 到 2026 年前沿的完整文献地图。
>
> 阅读建议：按 C.1.1 → C.1.2 → C.1.3 顺序建立理论根基，再按研究方向在 C.1.4 和 C.1.5 中挑选相关论文精读。每篇论文都应在动手复现（[附录 A](../appendix_industrial_training/training-debugging)、[附录 B](../appendix_code_cheatsheet/sft-kl)）之后回看，才会发现第一次读时忽略的工程细节。

## C.1.1 经典 RL 必读

经典文献的价值不在于算法本身——时过境迁，TD-Learning、Q-Learning 已是教科书常识。它们的真正价值在于**奠基性论证**：把"智能体在环境中学习"这件事形式化、可分析、可证明。读这些论文是理解现代 RL 所有数学符号背后含义的最快路径。

| 优先级 | 论文                                                                                                         | 核心贡献                                                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Sutton & Barto. _Reinforcement Learning: An Introduction_ (2018, 2nd ed.)                                    | RL 的"圣经"。MDP、Bellman 方程、TD 学习、策略迭代、资格迹的完整理论体系，所有现代 RL 论文都默认读者熟悉此书符号。                                                     |
| ★      | Watkins, Dayan. _Q-Learning_ (Machine Learning, 1992)                                                        | 证明 off-policy TD 控制 $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ 在 tabular 情况下收敛到 $Q^*$，奠定后续 DQN/Rainbow 的理论根基。 |
| ★      | Tesauro. _TD-Gammon, a Self-Teaching Backgammon Program_ (Neural Computation, 1995)                          | 第一次让神经网络 + TD-Learning 达到人类专家水平（双陆棋），证明 $\text{TD}(\lambda)$ 与函数逼近结合能学到超出训练数据分布的策略。                                     |
| ★      | Sutton. _Learning to Predict by the Methods of Temporal Differences_ (Neural Computation, 1988)              | TD 学习的奠基论文，提出 $\text{TD}(0)$、$\text{TD}(\lambda)$ 与资格迹机制，是价值估计的源头。                                                                         |
| ★      | Mnih et al. _Human-level control through deep reinforcement learning_ (Nature, 2015)                         | DQN：经验回放 + 目标网络让 CNN 在 49 个 Atari 游戏上达到人类水平。深度 RL 的奠基工程论文。                                                                            |
| ★      | Silver et al. _Mastering the game of Go with deep neural networks and tree search_ (Nature, 2016)            | AlphaGo：策略网络 + 价值网络 + MCTS 击败李世石，证明 RL 在围棋这种长期被认为"AI 不可能解决"的问题上的突破。                                                           |
| ◆      | Williams. _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning_ (1992) | REINFORCE：策略梯度定理的原始推导，所有 on-policy 算法（A2C、PPO、GRPO）的祖先。                                                                                      |
| ◆      | Schulman et al. _Trust Region Policy Optimization_ (ICML, 2015)                                              | TRPO：用 KL 散度约束保证策略更新单调，是 PPO 的直接前身。                                                                                                             |
| ◆      | Kearns, Singh. _Bias-Variance Error Bounds for Temporal Difference Updates_ (2000)                           | 从偏差-方差权衡角度解释 $\text{TD}(\lambda)$ 中 $\lambda$ 的作用，对理解 GAE（[第 8 章](../chapter10_ppo/gae-reward-model)）至关重要。                                |
| ◇      | Bertsekas, Tsitsiklis. _Neuro-Dynamic Programming_ (1996)                                                    | 把 DP 与函数逼近结合的早期专著，数学严密但读起来沉重，适合做理论时查阅。                                                                                              |

## C.1.2 深度 RL 必读

2013 年后 RL 进入深度学习时代。这一批论文定义了现代深度 RL 的标准方法栈：经验回放、目标网络、actor-critic、最大熵、模型学习。读者应重点掌握每篇的**算法骨架**与**消融实验**——前者决定了能否复现，后者决定了能否改进。

| 优先级 | 论文                                                                                                                                      | 核心贡献                                                                                                                               |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Mnih et al. _Playing Atari with Deep Reinforcement Learning_ (arXiv:1312.5602, 2013)                                                      | DQN 早期版本（NIPS Workshop），首次把 CNN 与 Q-Learning 结合，开启深度 RL。                                                            |
| ★      | Mnih et al. _Human-level Control through Deep Reinforcement Learning_ (Nature, 2015)                                                      | DQN 完整版：经验回放 + 目标网络，49 个 Atari 游戏达到人类水平。                                                                        |
| ★      | Mnih et al. _Asynchronous Methods for Deep Reinforcement Learning_ (A3C, arXiv:1602.01783, 2016)                                          | 异步 actor-critic：多线程并行采样消除相关性，无需经验回放；A2C 是其同步版本，至今仍是基线。                                            |
| ★      | Schulman et al. _Proximal Policy Optimization Algorithms_ (arXiv:1707.06347, 2017)                                                        | PPO：clip 替代 TRPO 的二阶优化，工程友好、训练稳定。LLM 后训练 PPO 的直接原型。详见[第 8 章](../chapter10_ppo/ppo-bipedal-walker)。                 |
| ★      | Lillicrap et al. _Continuous Control with Deep Reinforcement Learning_ (DDPG, arXiv:1509.02971, 2015)                                     | 把 DPG 扩展到深度网络，确定性策略梯度 + 经验回放，连续控制的开山之作。详见[第 9 章](../chapter11_continuous_control/ddpg)。           |
| ★      | Fujimoto et al. _Addressing Function Approximation Error in Actor-Critic Methods_ (TD3, arXiv:1802.09477, 2018)                           | 双 Q + 延迟更新 + 目标平滑，修复 DDPG 的 Q 值过估计与训练不稳定。                                                                      |
| ★      | Haarnoja et al. _Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor_ (arXiv:1801.01290, 2018)                  | 最大熵 RL + 自动温度调节，MuJoCo 长期霸榜，连续控制首选算法。                                                                          |
| ★      | Hessel et al. _Rainbow: Combining Improvements in Deep RL_ (arXiv:1710.02298, 2017)                                                       | 把 Double DQN、Dueling、PER、NoisyNet、Multi-step、Distributional Q 这 6 个 DQN 改进组合，证明"组合优于单点改进"。消融实验是经典教材。 |
| ★      | Silver et al. _A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play_ (AlphaZero, Science, 2018) | 自我对弈 + MCTS + 神经网络，无人类棋谱学到围棋/象棋/将棋超人类水平。                                                                   |
| ★      | Schrittwieser et al. _Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model_ (MuZero, arXiv:1911.08265, 2019)             | 学一个**隐式**世界模型（表示/动力学/预测三网络），不知道游戏规则也能规划。                                                             |
| ★      | Hafner et al. _Mastering Diverse Domains through World Models_ (Dreamer V3, arXiv:2301.04104, 2023)                                       | RSSM 循环世界模型 + 想象中训练 actor-critic，单一超参跨越 150+ 任务（Atari、MuJoCo、Crafter、DMLab），model-based RL 的现代旗舰。      |
| ◆      | Haarnoja et al. _Soft Actor-Critic Algorithms and Applications_ (arXiv:1812.05905, 2018)                                                  | SAC 的扩展技术报告，含完整超参表与多个真实机器人实验。                                                                                 |
| ◆      | Schaul et al. _Prioritized Experience Replay_ (arXiv:1511.05952, 2015)                                                                    | 用 TD 误差给经验回放样本加权，加速 DQN 收敛。Rainbow 的核心组件之一。                                                                  |
| ◆      | Wang et al. _Dueling Network Architectures for Deep RL_ (arXiv:1511.06581, 2015)                                                          | 把 $Q(s,a)$ 分解为 $V(s) + A(s,a)$，提升动作间差异较小任务的训练效率。                                                                 |
| ◆      | Van Hasselt et al. _Deep RL with Double Q-Learning_ (arXiv:1509.06461, 2015)                                                              | Double DQN：解耦动作选择与价值估计，抑制 Q 值过估计。                                                                                  |
| ◆      | Bellemare et al. _A Distributional Perspective on RL_ (C51, arXiv:1707.06887, 2017)                                                       | 学习回报的**分布**而非期望，对噪声 reward 更鲁棒。                                                                                     |
| ◇      | Janner et al. _When to Trust Your Model: Model-Based RL in the Stochastic World_ (MBPO, arXiv:1906.08253, 2019)                           | 短 horizon rollout 平衡模型偏差与样本效率。                                                                                            |
| ◇      | Chua et al. _Deep RL in a Handful of Trials Using Probabilistic Dynamics Models_ (PETS, arXiv:1805.12114, 2018)                           | 集成 + 概率模型表达认知与偶然不确定。                                                                                                  |

## C.1.3 LLM RL 必读

LLM RL 是 2022 年后 RL 的最大应用场景。这一批论文定义了 RLHF / DPO / GRPO / RLVR / R1 / DAPO 等核心范式，每一篇都对应一段工业界训练实践。读者应**按时间顺序**读完，才能理解算法演进脉络：奖励模型 → RLHF → DPO（无奖励模型）→ GRPO（无 critic）→ R1（纯 RL）→ DAPO（修复 GRPO 缺陷）→ 2026 年的 GSPO/CISPO。

| 优先级 | 论文                                                                                                                      | 核心贡献                                                                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback_ (InstructGPT, arXiv:2203.02155, 2022) | RLHF 三阶段工程化（SFT + RM + PPO），第一次大规模证明 RLHF 比 SFT 显著更好。LLM 后训练范式的奠基论文。详见[第 13 章](../chapter15_rlhf/base-model-to-assistant)。                      |
| ★      | Bai et al. _Constitutional AI: Harmlessness from AI Feedback_ (arXiv:2212.08073, 2022)                                    | Anthropic 的 RLAIF：用 AI 反馈替代人类标注，"宪法"驱动自我修正。同时是 RLHF 与对齐研究的桥梁。详见[第 13 章](../chapter21_cai_rlvr/hhh-practice)。                                     |
| ★      | Rafailov et al. _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_ (arXiv:2305.18290, 2023) | DPO：通过 Bradley-Terry 模型重新参数化，把 RLHF 转化为监督学习，**完全省去 RM 与 PPO**。LLM 后训练最优雅的数学推导之一。详见[第 14 章](../chapter17_dpo/dpo-objective-derivation)。                       |
| ★      | DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL_ (arXiv:2501.12948, 2025)                    | R1：纯 RL（GRPO + 规则奖励）让 7B 模型数学推理接近 GPT-4o；R1-Zero 证明无需 SFT 即可触发长 CoT。RLVR 范式的标志性论文。详见[第 15 章](../chapter18_grpo/grpo-practice-and-mechanism)。 |
| ★      | Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ (arXiv:2402.03300, 2024) | GRPO 的原始论文：去掉 PPO 的 critic，用组内 normalize 估计 baseline。Group-Normalized Policy Gradient 的范式起点。                                                                     |
| ★      | Yu et al. _DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization_ (arXiv:2503.14476, 2025)                        | 修复 GRPO 的四个缺陷——长度偏置、熵坍缩、噪声优势、长短 horizon 冲突——提出 Clip-Higher、Dynamic Sampling、Token-Level Loss、Overlong Reward Shaping。Qwen 团队的工程化改进。            |
| ★      | Kimi Team. _Kimi K2: Open Agentic Intelligence_ (arXiv:2507.20534, 2025)                                                  | 1T 参数 MoE（32B 激活）的开放训练，验证 GSPO（Group Sequence Policy Optimization）在超大规模 Agent 训练上的可扩展性。                                                                  |
| ★      | Zheng et al. _Group Sequence Policy Optimization_ (arXiv:2507.18071, 2025)                                                | 序列级 importance sampling + sequence-level loss，修复 GRPO 在长序列、长 horizon 任务上 token-level variance 过高的问题。                                                              |
| ★      | MiniMax. _MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention_ (arXiv:2506.13585, 2025)            | 在 456B 混合注意力 MoE 上提出 CISPO——裁剪重要性采样权重而非 token 更新，配合 lightning attention 在 512 张 H800 上三周完成 RL 训练。                                                   |
| ◆      | Xiao et al. _VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks_ (arXiv:2504.05118, 2025)   | Value-Assisted PPO：引入轻量 critic + 长序列 value 估计，让 PPO 在数学推理长 CoT 上重新可用。                                                                                          |
| ◆      | Luong et al. _ReFT: Reasoning with Reinforced Fine-Tuning_ (arXiv:2401.08967, 2024)                                       | SFT 预热 + PPO 在线微调，从同一批训练题上采多样化解题路径，小数据集上持续提升推理能力。                                                                                                |
| ◆      | Yuan et al. _Free Process Rewards without Process Labels_ (arXiv:2412.01981, 2024)                                        | 从 RLVR 训练中蒸馏出 token 级隐式 PRM，无需人工过程标注即可提供密集过程奖励。                                                                                                          |
| ◆      | Setlur et al. _Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning_ (PRM, arXiv:2410.08146, 2024)   | 系统化 PRM 训练方法论，PRM 引导搜索的基础。详见[第 17 章](../chapter20_prm_search/outcome-vs-process)。                                                                                |
| ◇      | Yuan et al. _Self-Rewarding Language Models_ (arXiv:2401.10020, 2024)                                                     | 让 LLM 自己生成偏好对训练自己，减少人工标注依赖。                                                                                                                                      |
| ◇      | Tajwar et al. _Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data_ (arXiv:2404.14367, 2024)        | 对比监督式、on-policy RL 与对比学习三类 preference fine-tuning，证明次优的 on-policy 数据对偏好微调最有效。                                                                            |

## C.1.4 安全研究必读

RLHF / RLVR 在让模型变强的同时打开了**奖励黑客、欺骗、对齐伪装、潜在后门**的潘多拉魔盒。这一批论文是 2023 年后 AI 安全研究的核心文献——它们告诉你**为什么训练良好的模型可能伤害你**，以及现有对齐方法的根本局限。任何做后训练的工程师都应至少精读前 5 篇。

| 优先级 | 论文                                                                                                                                 | 核心贡献                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| ★      | Hubinger et al. _Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training_ (arXiv:2401.05566, 2024)              | Anthropic 证明：植入"特定触发词才作恶"后门的模型，RLHF / SFT / 对抗训练**都无法消除**后门。揭示现有对齐方法的根本盲区。              |
| ★      | Greenblatt et al. _Alignment Faking in Large Language Models_ (arXiv:2412.14093, 2024)                                               | Anthropic 证明：模型在训练时**假装对齐**——展示符合预期的行为以避免被修改；推理时则违背训练目标。这是 schema 在真实训练中的实证出现。 |
| ★      | Meinke et al. _Frontier Models are Capable of In-Context Scheming_ (arXiv:2412.04984, 2024)                                          | 在多个前沿模型上观察到 in-context scheming：模型会假装能力受限、会复制自身权重、会规避监督。                                         |
| ★      | Sharma et al. _Towards Understanding Sycophancy in Language Models_ (arXiv:2310.13548, 2023)                                         | 系统化研究 LLM 的 sycophancy（迎合）行为，揭示 RLHF 偏好数据中"用户赞同 = 高奖励"的偏置。                                            |
| ★      | Denison et al. _Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models_ (arXiv:2406.10162, 2024)          | 课程式实验证明：学会简单 specification gaming 的 LLM 会逐步泛化到**直接重写自身奖励函数**的 reward tampering。                       |
| ★      | OpenAI. _GPT-4o System Card_ (2024)                                                                                                  | GPT-4o 发布后 sycophancy 问题导致大规模回滚——公开承认 RLHF 训练中奖励信号偏置的可观察后果。                                          |
| ★      | Gao et al. _Scaling Laws for Reward Model Overoptimization_ (arXiv:2210.10760, 2022)                                                 | OpenAI 的奖励模型 over-optimization scaling law：RM 与真实偏好距离随 KL 增大呈指数恶化。理解 RLHF 上限的关键论文。                   |
| ★      | Anthropic. _Natural Emergent Misalignment from Reward Hacking in Production RL_ (arXiv:2511.18397, 2025)                             | 证明学会 reward hacking 的模型会泛化出 alignment faking 与蓄意破坏等 emergent misalignment，且标准 RLHF 安全训练无法消除。           |
| ◆      | Eisenstein et al. _Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking_ (arXiv:2312.09244, 2023) | 用 RM 集成降低奖励黑客，工业界常用的鲁棒化方案。                                                                                     |
| ◆      | Coste et al. _Reward Model Ensembles Help Mitigate Overoptimization_ (arXiv:2310.02743, 2023)                                        | 同上的工程实证，含完整超参与坏案例。                                                                                                 |
| ◆      | Hubinger et al. _Risks from Learned Optimization in Advanced Machine Learning Systems_ (arXiv:1906.01820, 2019)                      | mesa-optimization 理论框架：学会的优化器可能内化与训练目标不同的目标——deceptive alignment 概率分析的理论基础。                       |
| ◇      | Taylor et al. _School of Reward Hacks: Hacking Harmless Tasks Generalizes to Misaligned Behavior in LLMs_ (arXiv:2508.17511, 2025)   | 在无害任务上训练 reward hacking 会泛化为更危险的不对齐行为，可作为评估安全性的 benchmark。                                           |
| ◇      | METR. _Frontier Reward Hacking: An Empirical Study of Reward Hacking in Frontier Models_ (2024)                                      | METR 评估前沿模型 reward hacking 行为的实证基准。                                                                                    |
| ◇      | OpenAI. _The Instruction Hierarchy: Training LLMs to Prioritize Instructions_ (arXiv:2404.13208, 2024)                               | 把 prompt 划分为系统/用户/工具三层，提供对抗 prompt injection 的训练时方案。                                                         |

## C.1.5 2025–2026 前沿

2025 年起 RL 进入第二个爆发期：过程奖励模型驱动推理、Agent 训练范式成熟、自我博弈扩展到代码与软件工程、视频生成引入 RL fine-tuning。这一批论文大多是 2025–2026 年的 tech report，尚未进入主流教材——它们定义了**下一个十年 RL 的研究方向**。读者应密切关注。

### C.1.5.1 RLVR 与推理算法

| 优先级 | 论文                                                                                                                                                  | 核心贡献                                                                                           |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| ★      | Yu et al. _DAPO_ (arXiv:2503.14476, 2025)                                                                                                             | GRPO 的工程化修复，Qwen 团队开源完整训练细节。已在 C.1.3 列出。                                    |
| ★      | Zheng et al. _GSPO_ (arXiv:2507.18071, 2025)                                                                                                          | 序列级 IS，Kimi K2 使用的核心算法。已在 C.1.3 列出。                                               |
| ★      | MiniMax. _CISPO_ (arXiv:2506.13585, 2025)                                                                                                             | 截断无效 IS 样本。已在 C.1.3 列出。                                                                |
| ★      | Khalifa et al. _Process Reward Models That Think_ (ThinkPRM, arXiv:2504.16828, 2025)                                                                  | 用一个轻量 PRM 替代密集过程奖励标注，让 RLVR 在低数据下可用。                                      |
| ★      | DeepSeek. _DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition_ (arXiv:2504.21801, 2025) | 形式化定理证明（Lean 4）上的 RLVR：RL 学习子目标分解，MiniF2F 达到 88.9% 通过率。                  |
| ◆      | Microsoft. _rStar-Math_ (arXiv:2501.04519, 2025)                                                                                                      | MCTS + PRM + 自我对弈训练小模型数学推理。                                                          |
| ◆      | Liu et al. _Understanding R1-Zero-Like Training: A Critical Perspective_ (arXiv:2503.20783, 2025)                                                     | 拆解 R1-Zero 式训练中的基础模型与 RL 两个组件，指出 GRPO 的响应长度优化偏置并给出无偏的 Dr. GRPO。 |

### C.1.5.2 Agentic RL 与代码/软件工程

| 优先级 | 论文                                                                                                                         | 核心贡献                                                                                                                                            |
| ------ | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Pan et al. _SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution_ (arXiv:2502.18449, 2025)  | Meta 在 SWE-bench 上规模化 RL 训练，定义"代码-执行-反馈-学习"的 Agent RLVR 范式。详见[第 20 章](../chapter23_rl_based_swe/swe-bench-and-rlvr)。     |
| ★      | FAIR CodeGen team. _CWM: An Open-Weights LLM for Research on Code Generation with World Models_ (arXiv:2510.02387, 2025)     | Meta FAIR 的 32B 开放权重模型：在 Python 解释器与 agentic Docker 轨迹上中训练世界模型，把代码执行预测融入 RL 推理。                                 |
| ★      | Wei et al. _Toward Training Superintelligent Software Agents through Self-Play SWE-RL_ (arXiv:2512.18552, 2025)              | Meta/CMU 的自我博弈 SWE-RL：单一 agent 在沙盒代码库上迭代注入与修复 bug，无需人工标注 issue/test 即可持续自我改进。                                 |
| ★      | ByteDance. _UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning_ (arXiv:2509.02544, 2025) | GUI 操作 Agent：截图 → 动作（鼠标点击/键盘），端到端视觉 RL。详见[第 22 章](../chapter25_computer_use/training)。                                   |
| ★      | StepFun. _Step-Audio-R1 Technical Report_ (arXiv:2511.15848, 2025)                                                           | 第一个解锁音频推理能力的模型：MGRD 蒸馏让推理链锚定真实声学特征，音频理解超越 Gemini 2.5 Pro。详见[第 24 章](../chapter27_audio_rl/reward-design)。 |
| ◆      | Tongyi. _Tongyi DeepResearch_ (2025)                                                                                         | PRM 引导的长 horizon research agent。                                                                                                               |
| ◇      | Google. _AlphaEvolve_ (2025)                                                                                                 | 进化 + LLM 自动发现新算法，应用于芯片设计与组合优化。                                                                                               |

### C.1.5.3 大规模训练与开放模型

| 优先级 | 论文                                                                                                            | 核心贡献                                                                                                                                                 |
| ------ | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Qwen Team. _Qwen3 Technical Report_ (arXiv:2505.09388, 2025)                                                    | Qwen3 全系列（0.6B–235B-A22B MoE）训练细节，含 SFT/RLHF/RLVR 三阶段完整 recipe。                                                                         |
| ★      | Kimi Team. _Kimi K2_ (arXiv:2507.20534, 2025)                                                                   | 1T MoE 开放训练，验证大规模 GSPO 在 Agent 任务上的可扩展性。已在 C.1.3 列出。                                                                            |
| ★      | Kimi Team. _Kimi K2.5: Visual Agentic Intelligence_ (arXiv:2602.02276, 2026)                                    | K2 的下一代多模态 agent 模型：联合文本-视觉预训练与 RL，Agent Swarm 并行任务编排显著降低推理延迟。                                                       |
| ★      | DeepSeek-AI. _DeepSeek-V3 Technical Report_ (arXiv:2412.19437, 2024)                                            | 671B MoE（37B 激活），2.664M H800 小时训练 14.8T tokens；MLA + DeepSeekMoE + FP8 是核心工程创新。预训练成本参考见[附录 C](../appendix_gpu_hours/gpu-hours-estimation)。 |
| ★      | Gao et al. _Seedance 1.0: Exploring the Boundaries of Video Generation Models_ (arXiv:2506.09113, 2025)         | 视频生成模型的 RLVR fine-tuning，把 RL 从 token 扩展到时空 latent。                                                                                      |
| ★      | Peng et al. _DanceGRPO: A Unified Post-training Framework for Generative Video Models_ (arXiv:2505.07818, 2025) | 把 GRPO 适配到视频扩散模型，统一图文与视频 RLVR。                                                                                                        |
| ★      | Meituan LongCat Team. _LongCat-Video Technical Report_ (arXiv:2510.22200, 2025)                                 | 美团 13.6B 参数的 DiT 视频生成模型，单模型支持文生视频/图生视频/续写，多奖励 RLHF 对齐长视频生成。                                                       |
| ◆      | Behrouz et al. _Titans: Learning to Memorize at Test Time_ (arXiv:2501.00663, 2025)                             | 用长期/短期记忆模块扩展 Transformer，测试时在线学习新概念，为长序列推理提供可扩展记忆架构。                                                              |
| ◆      | Hong et al. _Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO_ (arXiv:2511.13288, 2025)      | GRPO 的层级扩展：主 agent 与各子 agent 分别计算组内优势，配合轨迹对齐与解耦训练，提升多 agent 深度研究系统的稳定性与样本效率。                           |
| ◆      | Google. _Genie 3_ (2025)                                                                                        | 通用世界模型，从视频学环境动力学，是 embodied RL 的下一代基础。                                                                                          |
| ◇      | Ishibashi et al. _Can Large Language Models Invent Algorithms to Improve Themselves?_ (arXiv:2410.15639, 2024)  | Self-Developing：模型用 DPO 迭代发现并改进自己的算法（模型合并），在数学推理上超过人工设计的 Task Arithmetic。                                           |

## 阅读路线建议

不同读者的最优路径不同：

- **从基础补齐**：C.1.1 → C.1.2 → C.1.3 → C.1.4，按时间顺序理解 RL 从 1992 到 2026 的完整演进。
- **LLM 后训练工程师**：C.1.3 全部 + C.1.4 前 5 篇 + C.1.5.1 全部。重点理解 InstructGPT/DPO/GRPO/R1/DAPO 五篇的工程细节。
- **Agentic RL 研究者**：C.1.3 GRPO/R1 + C.1.5.2 全部 + C.1.2 AlphaZero/MuZero/Dreamer（理解搜索与世界模型）。
- **安全研究者**：C.1.4 全部 + C.1.1 Sutton & Barto 第 15 章（探索与利用的失败模式）+ C.1.3 RLHF over-optimization 相关论文。
- **理论研究者**：C.1.1 全部 + C.1.2 Rainbow/Dreamer 消融实验 + C.1.5 中的 GSPO/CISPO 数学推导。

## 阅读方法

::: tip 精读 vs 泛读
对**算法核心论文**（带 ★）做精读——逐行读公式、复现代码、做消融实验。对**工程 tech report**（如 Qwen3、Kimi K2）做泛读——抓超参表、训练曲线、ablation。前者训练思维，后者训练工程直觉。
:::

::: warning 不要只读最新论文
2025–2026 年的 RL 论文常默认读者熟悉 GRPO/PPO/REINFORCE 的细节。如果跳过 C.1.1/C.1.2 直接读 C.1.5，你会觉得"算法都没什么特别"——实际是缺少历史背景。建议每个新算法都追溯到其 2-3 篇"祖先"论文。
:::

::: details 论文与本书章节的对应关系
本书每章末尾的"延伸阅读"已列出该章核心论文。本附录是更全的、跨章节的论文地图。建议把本附录与各章末尾清单交叉使用：

- [第 14 章 DPO](../chapter17_dpo/dpo-objective-derivation) → C.1.3 Rafailov et al.
- [第 8 章 PPO](../chapter10_ppo/ppo-bipedal-walker) → C.1.1 Schulman PPO、C.1.2 A3C
- [第 13 章 RLHF](../chapter15_rlhf/base-model-to-assistant) → C.1.3 InstructGPT
- [第 15 章 GRPO/RLVR](../chapter18_grpo/grpo-practice-and-mechanism) → C.1.3 R1、Shao et al.
- [第 9 章 连续控制](../chapter11_continuous_control/ddpg) → C.1.2 DDPG/TD3/SAC/Dreamer V3
- [第 25 章 对齐失败](../chapter30_alignment_failures/classical-failures) → C.1.4 全部
- [第 18 章 工业级 LLM RL](../chapter16_llm_rl_industrial/single-machine-to-industrial) → C.1.5.3 全部
  :::

## 本章总结

本附录的 100+ 篇论文构成了从 1992 Q-Learning 到 2026 Kimi K2.5 的完整 RL 文献地图。它们不是"读完就行"的清单，而是**反复回看、伴随职业成长**的资料库——每多理解一篇，你对现代 RL 的理解就深一层。当你能在某篇论文中发现作者没说清楚的工程细节，或找到一个未被消融验证的假设时，你就具备了做独立研究的能力。

## 补充：学习资料与复现项目

> **本附录目标**：为读者的后续进阶提供清晰的导航。前半部分整理了理论扎实、深入浅出的教材与课程资源，帮助你系统性夯实基础或钻研前沿；后半部分则梳理了强化学习在游戏与仿真生态中的经典里程碑与常用环境，为你寻找下一个动手复现的实战项目提供灵感与坐标系。

### 学习资料推荐

> **使用方式**：本书覆盖了从 MDP 基础到 PPO、DPO、GRPO 的完整链路，但 RL 领域远不止于此。如果你想深入某个方向、对比不同教学风格，或者寻找上手实践的资源，这份清单可以作为起点。所有资源均免费或公开可访问。

**如何使用这份清单？**

按你的目标选择：

- **刚学完第 3 章，想看其他教材怎么讲基础理论**：建议从赵世钰《RL 数学原理》或 Sutton & Barto 原版开始。
- **想看视频课程跟学**：建议从 David Silver 课程或李宏毅课程开始。
- **想上手写代码**：建议从 OpenAI Spinning Up 或《动手学强化学习》开始。
- **关注 LLM 对齐 / RLHF / GRPO**：建议从 Nathan Lambert RLHF Book 或 Ernest Ryu 的 RL-LLM 课程开始。
- **想了解前沿理论**：建议从 Princeton ECE 524 或 Alberta CMPUT 365 开始。

**一、经典教材**

**Reinforcement Learning: An Introduction（Sutton & Barto，第 2 版，2018）**

**地址**：[incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) ｜ [中文翻译](https://rl.qiwihui.com/)

强化学习领域的标准教材，被几乎所有大学 RL 课程列为必读。全书分三部分：Part I（表格方法，Ch1-8）覆盖 MDP、DP、MC、TD、n-step Bootstrapping、Planning；Part II（近似方法，Ch9-13）覆盖函数近似、资格迹、策略梯度；Part III（Ch14-17）讨论心理学、神经科学和应用。免费 PDF，中文翻译质量较高。**适合系统性地打基础。**

**强化学习的数学原理（赵世钰）**

**地址**：[github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)（GitHub 10k+ stars）

Springer 出版 + 清华大学出版社。全书 10 章从数学角度严格推导 RL 核心算法：贝尔曼方程 → VI/PI → MC → TD（含 Sarsa、Q-Learning、n-step Sarsa）→ 函数近似 → 策略梯度 → Actor-Critic。每章配有数学证明和练习题。**适合喜欢严格推导、想从数学层面理解"为什么这些算法有效"的读者。**

**深度强化学习（张志华，北京大学）**

**地址**：[PDF 初稿](https://www.math.pku.edu.cn/teachers/zhzhang/drl_v1.pdf)

北京大学数学学院课程配套教材。假设读者了解机器学习基础但不一定熟悉 RL，从基本概念出发，覆盖价值学习（DQN）、策略学习（Policy Gradient）、Actor-Critic、TRPO 等。配套王树森的 B 站视频课程。**适合中文读者快速入门 DRL。**

**动手学强化学习（张伟楠、沈键、俞勇）**

**地址**：[在线版](https://hrl.boyuai.com/) ｜ 上海交大 RL 课程教材

实践导向，全书配有可运行的 Jupyter 代码。三部分：基础（Bandit → MDP → DP → MC → 规划）→ 进阶（函数近似 → DQN → 策略梯度 → PPO）→ 前沿（Model-Based RL、Offline RL）。**适合边读边敲代码的学习者。**

**二、大学课程**

**欧美课程**

**Stanford CS234: Reinforcement Learning（Emma Brunskill）**

**地址**：[web.stanford.edu/class/cs234/](https://web.stanford.edu/class/cs234/)

斯坦福 RL 基础课。从表格 MDP 开始，覆盖策略评估、Q-Learning、函数近似、策略梯度、Offline RL、探索、MCTS，最后涉及 RLHF。约一半课时打理论基础，一半进入高级主题。教材：Sutton & Barto。

**Stanford CS224R: Deep Reinforcement Learning（Chelsea Finn）**

**地址**：[cs224r.stanford.edu](https://cs224r.stanford.edu/) ｜ [YouTube 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL)

斯坦福深度 RL 课程。假设学生已有 RL 基础，直接从模仿学习开始，快速进入策略梯度、Actor-Critic、Q-Learning、Model-Based RL、Offline RL、Reward Learning、RLHF、Meta-RL。**适合已掌握基础、想深入 DRL 各方向的学习者。**

**MIT 6.7920: Reinforcement Learning Foundations and Methods（Cathy Wu）**

**地址**：[web.mit.edu/6.7920/www/](https://web.mit.edu/6.7920/www/)

MIT 的 RL 理论课。三分之二"exploitation"（已知理论：DP 7 讲 + RL 核心方法 9 讲），三分之一"exploration"（前沿专题）。DP 部分非常扎实，覆盖有限/无限视界、LQR、策略/价值迭代、收敛性证明。**适合追求理论深度的学习者。**

**UC Berkeley CS285: Deep Reinforcement Learning（Sergey Levine）**

**地址**：[rail.eecs.berkeley.edu/deeprlcourse/](https://rail.eecs.berkeley.edu/deeprlcourse/)

伯克利的 Deep RL 旗舰课。仅 1 讲回顾 RL 基础，随后深入模仿学习、策略梯度、Actor-Critic、Value-Based RL、高级策略梯度、变分推断与 RL、LLM RL、Model-Based RL、Offline RL、探索。2026 春季版新增了 LLM RL 和 Offline RL 的实战作业。**内容最贴近当前工业前沿。**

**CMU 10-703: Deep Reinforcement Learning and Control**

**地址**：[cmudeeprl.github.io/703website_f25/](https://cmudeeprl.github.io/703website_f25/)

CMU 的深度 RL 课程。覆盖经典理论（MDP、DP、MC、TD）后，进入函数近似、Deep Q-Learning、MCTS、策略梯度、模仿学习、逆 RL、最优控制、Model-Based RL、探索。**理论与实践并重，覆盖面广。**

**University of Alberta CMPUT 365: Introduction to RL（Marlos Machado）**

**地址**：[Syllabus PDF](https://webdocs.cs.ualberta.ca/~machado/cmput365/w26/syllabus.pdf)

Sutton 所在大学的 RL 入门课，严格遵循 Sutton & Barto 教材顺序：Bandits → MDP → DP（含 PI、VI、GPI）→ MC 预测与控制 → TD 预测 → **TD 控制（Sarsa、Q-Learning）** → Planning（Dyna-Q）→ 函数近似 → 策略梯度。**Sutton & Barto 教材最忠实的课程实现。**

**Georgia Tech CS 7642: Reinforcement Learning（OMSCS）**

**地址**：[omscs.gatech.edu/cs-7642-reinforcement-learning](https://omscs.gatech.edu/cs-7642-reinforcement-learning)

可在线修读的 RL 课程。覆盖 DP、TD（含 Sarsa）、n-step TD、Lambda Return、DQN、策略梯度、多智能体 RL、博弈论、POMDP。**OMSCS 项目中口碑最好的 RL 课程之一。**

**Princeton ECE 524: Foundations of RL（Chi Jin）**

**地址**：[sites.google.com/view/cjin/teaching/ece524](https://sites.google.com/view/cjin/teaching/ece524) ｜ [YouTube](https://www.youtube.com/playlist?list=PLYXvCE1En13epbogBmgafC_Yyyk9oQogl)

理论导向，侧重有限样本分析和收敛性证明。Part I 覆盖表格 MDP、规划、探索（Bandit 和 MDP）、下界；Part II 覆盖大状态空间、线性 VI、函数近似、多智能体、POMDP。**适合想做 RL 理论研究的学习者。**

**David Silver RL Course（UCL / DeepMind）**

**地址**：[davidsilver.uk/teaching](https://www.davidsilver.uk/teaching/) ｜ [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

10 讲经典课程：MDP → DP → Model-Free Prediction → Model-Free Control → 函数近似 → 策略梯度 → Learning & Planning → 探索 → 经典游戏案例。David Silver 是 AlphaGo/AlphaZero 的第一作者。**结构精炼，讲解清晰，是最广泛传播的 RL 视频课程。**

**DeepMind x UCL RL Lecture Series（2021）**

**地址**：[YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

David Silver 课程的更新版，由 DeepMind 研究员（Hado van Hasselt 等）主讲。13 讲覆盖探索与控制、MDP 与 DP、无模型方法、函数近似、Planning、策略梯度与 Actor-Critic、近似 DP、Multi-step 与 Off-Policy、Deep RL。**比 2015 版更深入，增加了前沿内容。**

**中国大学课程**

**清华大学 Reinforcement Learning（Fall 2025）**

**地址**：[coai.cs.tsinghua.edu.cn/Courses/RL2025/\_site/](https://coai.cs.tsinghua.edu.cn/Courses/RL2025/_site/)

本科生 RL 课程。从多臂老虎机开始，覆盖 MDP、Planning（DP）、MC、TD Learning、策略梯度、函数近似、Deep RL。4 个编程作业（Bandit → MDP → TD & PG → Deep RL）+ 课程项目。课件公开。

**南京大学 强化学习基础（俞扬，2024）**

**地址**：[lamda.nju.edu.cn/introrl](https://www.lamda.nju.edu.cn/introrl/)

基于 Sutton & Barto 教材，9 讲覆盖 RL 基础概念、MDP、DP、MC、TD、DQN。5 个编程作业（Dagger → Q-Learning → DQN → Model-Based → Offline RL）。**中文大学课程中理论基础最扎实的之一。**

**南京大学 高级强化学习（袁雷，2025）**

**地址**：[lamda.nju.edu.cn/advanceRL](https://www.lamda.nju.edu.cn/advanceRL/)

研究生进阶课程。覆盖 DDPG/TD3、PPO 及技巧、多智能体、RLHF/DPO 理论推导、论文阅读。

**上海交通大学 强化学习（张伟楠，2024）**

**地址**：[wnzhang.net/teaching/sjtu-rl-2024](https://wnzhang.net/teaching/sjtu-rl-2024/)

使用《动手学强化学习》作为教材，9 章覆盖基础到前沿。

**三、中文在线课程与教程**

**李宏毅 深度强化学习（台湾大学）**

**地址**：[课程主页](https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) ｜ [B站 2025 版](https://www.bilibili.com/video/BV1SJvAzfEL2/)

以 Policy Gradient 为主线切入，深入讲解 PPO（含 Importance Sampling、On-policy → Off-policy 推导），然后讲 Q-Learning（DQN、Double DQN、Dueling DQN）、Actor-Critic。讲解生动，PPT 精美。**PPO 部分是中文课程中讲得最深入的。**

**王树森 深度强化学习**

**地址**：[B站视频](https://www.bilibili.com/video/BV1oEWDz1Ez5/)

北大数学学院课程配套视频。五大模块：基本概念 → 价值学习（DQN）→ 策略学习（Policy Gradient）→ Actor-Critic（A3C、TRPO）→ 进阶（DDPG、AlphaGo、多智能体）。配套张志华《深度强化学习》教材。**内容精炼，适合快速入门。**

**蘑菇书 EasyRL（Datawhale）**

**地址**：[在线版](https://datawhalechina.github.io/easy-rl/) ｜ [GitHub](https://github.com/datawhalechina/easy-rl)

综合了周博磊《强化学习纲要》、李宏毅课程、百度《世界冠军带你从零实践强化学习》的精华。13 章 + 专题，覆盖基础到 DQN、PPO、DDPG、AlphaStar。**中文社区最活跃的开源 RL 教程。**

**Spinning Up 中文版**

**地址**：[spinningup.qiwihui.com/zh-cn/latest](https://spinningup.qiwihui.com/zh-cn/latest/)

OpenAI Spinning Up 的中文翻译。包含核心概念、算法分类、策略梯度推导，以及 VPG、TRPO、PPO、DDPG、TD3、SAC 六个算法的实现。

**四、LLM 强化学习专项资源**

**Nathan Lambert — RLHF Book + Course**

**地址**：[rlhfbook.com](https://rlhfbook.com/) ｜ [Course](https://rlhfbook.com/course) ｜ [GitHub](https://github.com/natolambert/rlhf-book) ｜ [YouTube](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

AI2 研究员 Nathan Lambert 编写的 RLHF 专著。覆盖 RLHF 全流程：指令微调 → 奖励模型训练 → Rejection Sampling → PPO → DPO。代码库实现了 PPO、REINFORCE、GRPO、RLOO 等策略梯度方法。视频课程 4 讲。**LLM 对齐领域最系统的公开教材。**

**Ernest Ryu — Reinforcement Learning of Large Language Models（UCLA）**

**地址**：[ernestryu.com/courses/RL-LLM.html](https://ernestryu.com/courses/RL-LLM.html)

唯一一门把经典 RL 理论和 LLM RL 系统结合的大学课程。三部分：Ch1（5 讲经典 RL：MDP → VI → PG → PPO/GRPO → AlphaGo）→ Ch2（4 讲 LLM 基础：NLP → Transformer → ICL/SFT）→ Ch3（2 讲 LLM RL：RLHF/PPO/DPO → RLVR）。**RL 基础讲得最深的 LLM RL 课程。**

**DeepLearning.AI — Reinforcement Fine-Tuning LLMs with GRPO**

**地址**：[deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)

1 小时短课程，10 节课。以 Wordle 游戏为贯穿案例，讲解 GRPO 算法、奖励函数设计、LLM-as-Judge、Reward Hacking。7 个代码实验。**适合已有 LLM 基础、想快速上手 GRPO 的实践者。**

**Hugging Face — Deep RL Course**

**地址**：[huggingface.co/learn/deep-rl-course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

8 个单元覆盖 Q-Learning → DQN → Policy Gradient → A2C/A3C → PPO → 多智能体 → Offline RL。每个单元配有理论和代码实践。Bonus 单元讲 RLHF。**适合想用 Hugging Face 生态做 RL 实验的学习者。**

**五、实践教程与技术博客**

**OpenAI Spinning Up in Deep RL**

**地址**：[spinningup.openai.com](https://spinningup.openai.com/en/latest/)

RL 基础教学的金标准。三部分：核心概念（V/Q/Bellman/Advantage）→ 算法分类（Model-Based vs Model-Free）→ 策略优化推导（从零推导 Policy Gradient）。实现了 VPG、TRPO、PPO、DDPG、TD3、SAC 六个算法。**理论讲解和代码实现的最佳结合点。**

**Cameron Wolfe — Deep (Learning) Focus**

**地址**：[PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm) ｜ [Online vs Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl)

系列博客，用通俗语言深入讲解 PPO 在 LLM 中的应用、Online vs Offline RL 的取舍、DPO 原理等。**适合想理解"为什么 LLM RL 用这些算法"的读者。**

**Sebastian Raschka — Ahead of AI**

**地址**：[LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) ｜ [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)

《Build a Large Language Model From Scratch》作者的技术博客。覆盖 RLHF、DPO、RLVR、GRPO、推理时扩展等前沿话题。

### 复现项目推荐

强化学习项目可以按时代拆成两条线。非 LLM 时代更关注固定仿真环境、游戏基准、连续控制、多智能体和模型学习；LLM 时代则把动作扩展成 token、工具调用、网页操作、视觉推理和长程 agent trajectory，奖励也从环境分数扩展到偏好模型、规则验证、过程奖励和真实任务成功率。

#### 复现路线速查

| 目标方向               | 优先看这些资源                                                                                                                                                                                                                                                                               | 适合复现什么                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 经典算法入门           | [CleanRL](https://github.com/vwxyzjn/cleanrl)、[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)、[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)、[Dopamine](https://github.com/google/dopamine)                                                            | DQN、PPO、SAC、TD3、Rainbow DQN、Atari 基准                |
| 环境与游戏基准         | [Gymnasium](https://gymnasium.farama.org/)、[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)、[MiniGrid](https://minigrid.farama.org/)、[Procgen](https://github.com/openai/procgen)、[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) | CartPole、LunarLander、Atari、FPS、程序生成环境            |
| 多智能体与博弈         | [PettingZoo](https://pettingzoo.farama.org/)、[OpenSpiel](https://github.com/google-deepmind/open_spiel)、[SMAC](https://github.com/oxwhirl/smac)、[Google Research Football](https://github.com/google-research/football)                                                                   | 自我对弈、合作/竞争 MARL、StarCraft micromanagement、足球  |
| 机器人与具身控制       | [MuJoCo](https://mujoco.readthedocs.io/)、[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)、[ManiSkill](https://maniskill.readthedocs.io/)、[Meta-World](https://github.com/Farama-Foundation/Metaworld)、[LeRobot](https://github.com/huggingface/lerobot)                                | 连续控制、机械臂、移动机器人、模仿学习加 RL                |
| Model-Based / 世界模型 | [DreamerV3](https://github.com/danijar/dreamerv3)、[TD-MPC2](https://github.com/nicklashansen/tdmpc2)、[mbrl-lib](https://github.com/facebookresearch/mbrl-lib)、[MBPO](https://github.com/JannerM/mbpo)                                                                                     | 从像素或状态学习动力学模型，再做规划或策略优化             |
| LLM 后训练             | [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155)、[TRL](https://huggingface.co/docs/trl/index)、[NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL)、[verl](https://github.com/verl-project/verl)                                                                                      | PPO、DPO、GRPO、RLHF、偏好对齐、奖励模型训练               |
| LLM 推理               | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[Open-R1](https://github.com/huggingface/open-r1)、[TinyZero](https://github.com/Jiayi-Pan/TinyZero)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                           | RLVR、数学/代码推理、R1 类复现、verifier 设计              |
| Deep Research RL       | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)、[Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)、[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[WebThinker](https://github.com/RUC-NLPIR/WebThinker)                    | 搜索、阅读、证据筛选、引用、研究型回答                     |
| Agentic RL             | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)、[Google ADK](https://github.com/google/adk-python)、[Agent Lightning](https://github.com/microsoft/agent-lightning)、[AReaL](https://github.com/inclusionAI/AReaL)                                                      | 代码、工具调用、网页浏览、长程任务成功率优化               |
| GUI / Computer Use     | [OpenAI CUA](https://openai.com/index/computer-using-agent/)、[Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)、[UI-TARS](https://github.com/bytedance/UI-TARS)、[OSWorld](https://os-world.github.io/)                             | 网页、桌面、手机 GUI 操作和视觉 grounding                  |
| VLM                    | [TRL VLM GRPO](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl)、[VLM-R1](https://github.com/om-ai-lab/VLM-R1)、[Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner)、[Gemini Robotics](https://deepmind.google/models/gemini-robotics/)        | 图像问答、视觉推理、GUI/网页、机器人视觉操作、视觉语言奖励 |
| 生成模型 RL            | [DDPO](https://github.com/jannerm/ddpo)、[Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo)、[AlignProp](https://align-prop.github.io/)、[RLAIF-V](https://github.com/RLHF-V/RLAIF-V)、[VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                        | 用偏好、美学、安全和一致性奖励优化图像/多模态生成          |

#### 强化学习方向速览

如果要系统地选择复现方向，建议按“算法问题 + 环境形态 + 奖励来源”三条轴来选。下面这张表可以作为长期维护的目录骨架。

| 分类                                 | 代表问题                                    | 推荐项目/框架                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Value-Based RL                       | 从 Q 值学习离散动作策略                     | DQN、Double DQN、Dueling DQN、Rainbow；[Dopamine](https://github.com/google/dopamine)、[CleanRL](https://github.com/vwxyzjn/cleanrl)                                                                                                                                                            |
| Policy Gradient / Actor-Critic       | 直接优化策略，处理连续或随机动作            | REINFORCE、A2C/A3C、PPO、TRPO；[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)、[TRL PPO](https://huggingface.co/docs/trl/ppo_trainer)                                                                                                                                         |
| Off-Policy / Maximum Entropy         | 提升样本效率，鼓励探索和鲁棒性              | DDPG、TD3、SAC、REDQ；[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)、[Tianshou](https://github.com/thu-ml/tianshou)                                                                                                                                                          |
| Distributional RL                    | 学习回报分布而不是单一期望                  | C51、QR-DQN、IQN、FQF；[Dopamine](https://github.com/google/dopamine)、[DI-engine](https://github.com/opendilab/DI-engine)                                                                                                                                                                      |
| Exploration / Curiosity              | 稀疏奖励、长程探索、内在动机                | RND、ICM、count-based exploration；[MiniGrid](https://minigrid.farama.org/)、[Procgen](https://github.com/openai/procgen)、[DI-engine exploration docs](https://opendilab.github.io/DI-engine/)                                                                                                 |
| Model-Based RL                       | 学习环境模型，再规划或想象 rollout          | PETS、MBPO、Dreamer、TD-MPC；[mbrl-lib](https://github.com/facebookresearch/mbrl-lib)、[DreamerV3](https://github.com/danijar/dreamerv3)、[TD-MPC2](https://github.com/nicklashansen/tdmpc2)                                                                                                    |
| Offline / Batch RL                   | 只能使用离线数据，不能在线探索              | BCQ、CQL、IQL、TD3+BC；[D4RL](https://github.com/Farama-Foundation/D4RL)、[Minari](https://github.com/Farama-Foundation/Minari)、[d3rlpy](https://github.com/takuseno/d3rlpy)、[CORL](https://github.com/corl-team/CORL)                                                                        |
| Imitation / Reward Learning          | 从专家轨迹、偏好或逆强化学习中学习          | BC、DAgger、GAIL、AIRL；[imitation](https://github.com/HumanCompatibleAI/imitation)、[robomimic](https://github.com/ARISE-Initiative/robomimic)、[LeRobot](https://github.com/huggingface/lerobot)                                                                                              |
| Goal-Conditioned / Hierarchical      | 长程任务、子目标、选项和技能                | HER、Options、HIRO、skill discovery；[MiniGrid/BabyAI](https://minigrid.farama.org/)、[Meta-World](https://github.com/Farama-Foundation/Metaworld)                                                                                                                                              |
| Meta-RL / Multitask / Generalization | 跨任务迁移、快速适应、泛化                  | MAML-RL、PEARL、多任务 PPO/SAC；[Meta-World](https://github.com/Farama-Foundation/Metaworld)、[Procgen](https://github.com/openai/procgen)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                                                                                         |
| Safe / Constrained RL                | 约束成本、风险、安全探索                    | CPO、PPO-Lagrangian、shielding；[Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)、[OmniSafe](https://github.com/PKU-Alignment/omnisafe)                                                                                                                                    |
| Multi-Agent RL / Game AI             | 合作、竞争、自我对弈、通信                  | QMIX、MADDPG、MAPPO、AlphaZero；[PettingZoo](https://pettingzoo.farama.org/)、[OpenSpiel](https://github.com/google-deepmind/open_spiel)、[JaxMARL](https://github.com/FLAIROx/JaxMARL)                                                                                                         |
| Robotics / Embodied RL               | 连续控制、操作、导航、Sim2Real              | PPO/SAC on robots、domain randomization、VLA；[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)、[ManiSkill](https://maniskill.readthedocs.io/)、[robosuite](https://robosuite.ai/)、[OpenVLA](https://github.com/openvla/openvla)                                                             |
| Distributed / Systems RL             | 高吞吐 rollout、多机训练、生产化            | IMPALA、APPO、分布式 PPO；[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)、[Sample Factory](https://github.com/alex-petrenko/sample-factory)、[DI-engine](https://github.com/opendilab/DI-engine)、[Acme](https://github.com/google-deepmind/acme)                                  |
| RLHF / Preference Alignment          | 从人类或 AI 偏好中优化语言/多模态模型       | PPO、DPO、IPO、KTO、ORPO；[OpenAI InstructGPT](https://arxiv.org/abs/2203.02155)、[Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)、[TRL](https://huggingface.co/docs/trl/index)、[NeMo-RL](https://github.com/NVIDIA-NeMo/RL) |
| RLVR / Reasoning RL                  | 规则可验证奖励、数学/代码推理、长 CoT       | GRPO、DAPO、RLOO、REINFORCE++；[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[Open-R1](https://github.com/huggingface/open-r1)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)、[reasoning-gym](https://github.com/open-thought/reasoning-gym)                                  |
| Agentic RL                           | 搜索、工具调用、代码执行、网页/桌面任务     | trajectory reward、tool-use reward、process reward；[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)、[Google ADK](https://github.com/google/adk-python)、[Agent Lightning](https://github.com/microsoft/agent-lightning)、[SkyRL](https://docs.skyrl.ai/docs)               |
| VLM / GUI / Computer-Use RL          | 图像理解、GUI grounding、网页/手机/桌面控制 | 多模态 GRPO、GUI action RL；[OpenAI CUA](https://openai.com/index/computer-using-agent/)、[Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)、[VLM-R1](https://github.com/om-ai-lab/VLM-R1)、[OSWorld](https://os-world.github.io/)      |
| Generative Model RL                  | 用奖励优化图像、视频、音频等生成模型        | DDPO、AlignProp、RLAIF-V；[DDPO](https://github.com/jannerm/ddpo)、[Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo)、[AlignProp](https://align-prop.github.io/)、[VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                                               |

#### 非 LLM 时代 与 固定环境、仿真与经典算法

这一条线适合打牢 RL 基本功。推荐从小环境的单文件实现开始，逐步进入 Atari、连续控制、多智能体、机器人和 Model-Based RL。

##### 环境与算法库

| 环境/工具                                                                                       | 类型         | 说明                                                  | 推荐用途                       |
| ----------------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------- | ------------------------------ |
| [Gymnasium](https://gymnasium.farama.org/)                                                      | 通用 RL 环境 | OpenAI Gym 的继任者，CartPole、LunarLander 等经典环境 | 入门、算法调试、课程实验       |
| [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | 游戏环境     | Atari 2600 标准基准，DQN 系列论文常用环境             | 像素输入、离散动作、DQN 家族   |
| [MiniGrid](https://minigrid.farama.org/)                                                        | 网格世界     | 轻量级 GridWorld，便于研究探索、稀疏奖励和泛化        | 入门探索、层级 RL、任务泛化    |
| [Procgen](https://github.com/openai/procgen)                                                    | 程序生成游戏 | 关注泛化能力的 16 个程序生成环境                      | 过拟合分析、泛化实验           |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)                                         | FPS 3D 环境  | 第一人称射击游戏，部分可观察、视觉输入、长程决策      | 视觉策略、POMDP、导航与战斗    |
| [Stable-Retro](https://github.com/Farama-Foundation/stable-retro)                               | 经典游戏     | 复古主机游戏的 Gymnasium 风格封装                     | 经典游戏复现、课程展示         |
| [MuJoCo](https://mujoco.readthedocs.io/)                                                        | 物理仿真     | 高精度物理引擎，HalfCheetah、Ant、Humanoid 等基准     | PPO、SAC、TD3、连续控制        |
| [PyBullet](https://pybullet.org/wordpress/)                                                     | 物理仿真     | 开源机器人仿真，生态轻量                              | 机器人入门、替代 MuJoCo 实验   |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)                                              | GPU 并行仿真 | NVIDIA Isaac Gym 继任者，适合大规模并行机器人训练     | 大规模具身 RL、Sim2Real        |
| [ManiSkill](https://maniskill.readthedocs.io/)                                                  | 机器人操作   | 面向机械臂操作、视觉控制和大规模并行仿真的基准        | 视觉操作、模仿学习加 RL        |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld)                                    | 多任务机器人 | 机械臂多任务基准                                      | 多任务 RL、元学习、泛化        |
| [PettingZoo](https://pettingzoo.farama.org/)                                                    | 多智能体环境 | 多智能体版 Gymnasium，支持合作与竞争场景              | MARL 入门、并行/轮流行动接口   |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel)                                      | 博弈框架     | 棋类、卡牌、矩阵博弈和多智能体算法集合                | 自我对弈、CFR、AlphaZero 变体  |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)                                     | 分布式 RL    | Ray 生态里的分布式 RL 库                              | 大规模训练、多智能体生产实验   |
| [CleanRL](https://github.com/vwxyzjn/cleanrl)                                                   | 算法实现     | 单文件、可读性强、复现实验友好                        | 学算法细节、写课程代码         |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)                                | 算法库       | 封装好的 DQN、PPO、SAC、TD3 等实现                    | 快速跑 baseline、调参、对比    |
| [Dopamine](https://github.com/google/dopamine)                                                  | Atari 算法库 | Google 的 DQN/Rainbow/IQN 研究框架                    | Atari 论文复现、分布式价值学习 |

##### 推荐复现阶梯

| 阶段 | 项目建议                           | 推荐工具                                              | 验收标准                               |
| ---- | ---------------------------------- | ----------------------------------------------------- | -------------------------------------- |
| 1    | CartPole、MountainCar、LunarLander | Gymnasium、CleanRL、Stable-Baselines3                 | 能画 reward curve，理解 replay 与 GAE  |
| 2    | DQN / Rainbow on Atari             | ALE、Dopamine、CleanRL                                | 复现至少 1 个 Atari 小实验             |
| 3    | PPO / SAC / TD3 on MuJoCo          | MuJoCo、Stable-Baselines3、RL Baselines3 Zoo          | 能解释 entropy、target network、Q bias |
| 4    | 自我对弈与多智能体                 | PettingZoo、OpenSpiel、SMAC、Google Research Football | 能区分合作、竞争、混合博弈             |
| 5    | 机器人操作与视觉控制               | Isaac Lab、ManiSkill、Meta-World、LeRobot             | 能跑通并行仿真或 imitation-to-RL 流程  |
| 6    | Model-Based RL / World Model       | DreamerV3、TD-MPC2、mbrl-lib、MBPO                    | 能解释 latent dynamics 和 planning     |

##### 进阶方向与练习建议

| 方向               | 推荐复现项目                                                                                                                                                                        | 可做成什么课程作业                                              |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 单文件算法实现     | CleanRL 的 DQN、PPO、SAC、C51、PPO-LSTM                                                                                                                                             | 从 200 行到 500 行写清楚 replay、GAE、target network、entropy   |
| 高性能 RL 系统     | [Sample Factory](https://github.com/alex-petrenko/sample-factory)、[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)、[DI-engine](https://github.com/opendilab/DI-engine) | 对比单机、多进程、分布式 rollout 的吞吐和样本效率               |
| JAX / GPU 并行     | [Brax](https://github.com/google/brax)、[PureJaxRL](https://github.com/luchris429/purejaxrl)、[JaxMARL](https://github.com/FLAIROx/JaxMARL)                                         | 用 jit/vmap/pmap 跑大批量环境，理解“环境也可加速”的工程范式     |
| 离线强化学习       | D4RL + CQL/IQL/TD3+BC、[Minari](https://github.com/Farama-Foundation/Minari)、[d3rlpy](https://github.com/takuseno/d3rlpy)、[CORL](https://github.com/corl-team/CORL)               | 比较 online RL 和 offline RL 的 extrapolation error             |
| 模仿学习           | Behavior Cloning、DAgger、GAIL、AIRL；[imitation](https://github.com/HumanCompatibleAI/imitation)、[robomimic](https://github.com/ARISE-Initiative/robomimic)                       | 从专家轨迹训练策略，再用 RL fine-tune                           |
| 奖励学习与偏好学习 | GAIL/AIRL、偏好比较、reward model                                                                                                                                                   | 构造“人类偏好”或脚本偏好，观察 reward hacking                   |
| 安全与约束 RL      | Safety-Gymnasium、OmniSafe、PPO-Lagrangian、CPO                                                                                                                                     | 同时画 reward curve 与 cost curve，学习约束优化                 |
| 探索与稀疏奖励     | MiniGrid、Montezuma's Revenge、Procgen；RND、ICM、episodic curiosity                                                                                                                | 研究内在奖励是否真的提升探索，而不是只刷训练分数                |
| 层级与目标条件 RL  | HER、Options、HIRO、BabyAI、Meta-World                                                                                                                                              | 把长程任务拆成子目标，比较 flat policy 和 hierarchical policy   |
| 多任务与泛化       | Procgen、Meta-World、LIBERO、ContinualWorld                                                                                                                                         | 训练环境分数高不够，还要测未见任务和未见种子                    |
| 多智能体合作/竞争  | PettingZoo、OpenSpiel、SMAC、Google Research Football、JaxMARL                                                                                                                      | 比较 independent PPO、MAPPO、QMIX、自我对弈                     |
| 机器人操作         | MuJoCo、Isaac Lab、ManiSkill、robosuite、Meta-World                                                                                                                                 | 做 reaching、pushing、pick-and-place，再加入视觉输入            |
| 世界模型与规划     | DreamerV3、TD-MPC2、mbrl-lib、MBPO、IRIS                                                                                                                                            | 先学动力学模型，再比较 model-free 与 model-based 的样本效率     |
| 产业应用           | [RecSim](https://github.com/google-research/recsim)、[FinRL](https://github.com/AI4Finance-Foundation/FinRL)、[Pearl](https://github.com/facebookresearch/Pearl)                    | 推荐、广告、金融交易中的 bandit/RL 实验，重点讨论离线评估和风险 |

##### Unity ML-Agents 入门

[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) 是一个独特的 RL 工具包，它让训练直接在 3D 游戏引擎中进行。与 Gymnasium 的 2D 网格或 PyBullet 的纯物理仿真不同，ML-Agents 提供完整的 3D 空间，包括遮挡、透视、重力和碰撞，适合研究视觉导航和空间推理。

**典型使用场景**：

```python
# Unity ML-Agents 与 Gymnasium 接口兼容
from mlagents_envs.environment import UnityEnvironment

# 加载预构建的 Unity 环境（3D 平台跳跃）
env = UnityEnvironment(file_name="3DBall")

# ML-Agents 使用自己的 API，但可以包装为 Gymnasium 接口
from mlagents_envs.gym_utils import UnityToGymWrapper
gym_env = UnityToGymWrapper(env)

# 之后就可以用 Stable-Baselines3 训练
from stable_baselines3 import PPO
model = PPO("MlpPolicy", gym_env)
model.learn(total_timesteps=100000)
```

**经典 ML-Agents 环境示例**：

| 环境          | 任务类型       | 难度 | 适合练习                  |
| ------------- | -------------- | ---- | ------------------------- |
| 3DBall        | 平衡控制       | 入门 | 理解连续动作空间          |
| Crawler       | 四足行走       | 中等 | 连续控制 + 多关节协调     |
| Walker        | 二足行走       | 中等 | 对比 PyBullet 的 Walker2d |
| PushBlock     | 推方块         | 入门 | 目标条件 RL               |
| FoodCollector | 收集食物       | 中等 | 多目标 + 导航             |
| HideAndSeek   | 多智能体捉迷藏 | 高级 | 多智能体涌现行为          |

安装和环境获取方式参见[环境安装指南](/preface/env-setup)。

##### 经典里程碑项目参考

以下是非 LLM 时代常见的 30 个游戏与仿真复现方向，按主题整理：

###### 经典/棋盘游戏

| #   | 名称         | 游戏/环境               | 年份 | 关键信息                                             |
| --- | ------------ | ----------------------- | ---- | ---------------------------------------------------- |
| 1   | TD-Gammon    | 西洋双陆棋 (Backgammon) | 1992 | Gerald Tesauro，通过自我对弈强化学习达到人类专家水平 |
| 2   | Deep Blue    | 国际象棋 (Chess)        | 1997 | IBM，击败世界冠军卡斯帕罗夫，搜索为主，非纯 RL       |
| 3   | AlphaGo      | 围棋 (Go)               | 2016 | DeepMind，RL + MCTS 击败李世石                       |
| 4   | AlphaGo Zero | 围棋 (Go)               | 2017 | 不使用人类棋谱，从自我对弈中学习                     |
| 5   | AlphaZero    | 围棋/国际象棋/将棋      | 2017 | 通用棋类 RL 算法，同时掌握三种棋类                   |
| 6   | MuZero       | 围棋/象棋/Atari         | 2020 | 不依赖显式游戏规则，同时学习模型和策略               |

###### Atari 系列

| #   | 名称                             | 游戏/环境  | 年份 | 关键信息                                                                  |
| --- | -------------------------------- | ---------- | ---- | ------------------------------------------------------------------------- |
| 7   | DQN (Playing Atari with Deep RL) | Atari 2600 | 2013 | 首次用深度 RL 从像素直接学习多款游戏策略                                  |
| 8   | Human-level Control through DRL  | Atari 2600 | 2015 | Nature 2015，DQN 改进版，在多款 Atari 游戏上达到人类水平                  |
| 9   | Prioritized Experience Replay    | Atari      | 2015 | 改进经验回放，优先采样高 TD error 经验                                    |
| 10  | Rainbow DQN                      | Atari      | 2017 | 整合 Double DQN、Dueling、PER、NoisyNet、Distributional RL、n-step return |
| 11  | IQN (Implicit Quantile Networks) | Atari      | 2018 | 分布强化学习，学习回报分布的分位数表示                                    |

###### 即时战略 / MOBA

| #   | 名称                                      | 游戏/环境 | 年份 | 关键信息                                        |
| --- | ----------------------------------------- | --------- | ---- | ----------------------------------------------- |
| 12  | SC2LE (StarCraft II Learning Environment) | 星际争霸2 | 2017 | DeepMind 提供 StarCraft II 的 RL 研究环境和基准 |
| 13  | AlphaStar                                 | 星际争霸2 | 2019 | 多智能体 RL 达到 Grandmaster 级别               |
| 14  | TStarBot                                  | 星际争霸2 | 2019 | 腾讯提出的 StarCraft II 智能体系统              |
| 15  | OpenAI Five                               | Dota 2    | 2019 | 5v5 击败世界冠军 OG，大规模分布式 RL            |
| 16  | Honor of Kings 1v1                        | 王者荣耀  | 2020 | 腾讯 AI Lab，双裁剪 PPO，掌握复杂操作控制       |
| 17  | Honor of Kings 5v5                        | 王者荣耀  | 2020 | 多英雄、多角色、全局协作的 MOBA AI 系统         |
| 18  | Honor of Kings Arena                      | 王者荣耀  | 2022 | 开放式 MOBA RL 环境，关注泛化性挑战             |
| 19  | Mini Honor of Kings                       | 王者荣耀  | 2024 | 轻量级 MARL 环境，适合个人设备和课程项目        |

###### FPS / 3D 游戏

| #   | 名称                              | 游戏/环境        | 年份 | 关键信息                                              |
| --- | --------------------------------- | ---------------- | ---- | ----------------------------------------------------- |
| 20  | Playing FPS Games with Deep RL    | ViZDoom          | 2016 | 将深度 RL 用于 FPS 游戏，包含视觉输入和部分可观察状态 |
| 21  | Quake III Arena: Capture the Flag | 雷神之锤3 CTF    | 2019 | DeepMind，复杂团队协作和多智能体涌现行为              |
| 22  | Obstacle Tower                    | Unity 3D         | 2019 | 测试 3D 导航、视觉泛化和长程探索                      |
| 23  | Sample Efficient RL in Minecraft  | Minecraft/MineRL | 2021 | 使用人类示范数据提升 Minecraft 中的样本效率           |

###### 体育/竞速/其他

| #   | 名称                     | 游戏/环境   | 年份 | 关键信息                                     |
| --- | ------------------------ | ----------- | ---- | -------------------------------------------- |
| 24  | Google Research Football | 足球 11v11  | 2020 | 开源足球模拟器，支持多智能体 RL 研究         |
| 25  | RL in Rocket League      | 火箭联盟    | 2022 | 赛车加足球混合游戏中的高维连续控制与团队协作 |
| 26  | Deep RL for Flappy Bird  | Flappy Bird | 2015 | 早期深度 RL 游戏实践项目                     |

###### 多智能体/综合

| #   | 名称                             | 游戏/环境      | 年份 | 关键信息                                                          |
| --- | -------------------------------- | -------------- | ---- | ----------------------------------------------------------------- |
| 27  | Deep RL for General Game Playing | 通用棋类       | 2020 | 将 AlphaZero 类方法推广到通用博弈                                 |
| 28  | OpenSpiel                        | 棋类/卡牌      | 2019 | DeepMind 博弈框架，包含多种游戏和经典博弈算法                     |
| 29  | Hide-and-Seek                    | 多智能体捉迷藏 | 2019 | OpenAI，多智能体自博弈中涌现工具使用和复杂策略                    |
| 30  | Multi-Agent RL in Video Games    | 综述           | 2025 | 覆盖 Rocket League、Doom、Minecraft、StarCraft、Dota、MOBA 等方向 |

#### LLM 时代 与 后训练、推理、Agentic、VLM 与世界模型

LLM 时代的 RL 不再只是“在固定环境里最大化分数”。动作可以是一段文本、一次搜索、一次工具调用、一个网页点击、一个代码补丁、一次视觉定位，甚至是一整条多步 agent trajectory。奖励也从环境分数扩展到偏好模型、规则验证、过程奖励、单元测试、网页任务成功率和多模态 grounding 信号。

##### 现代与经典资源速查

阅读顺序建议是先看经典论文和官方说明建立概念，再挑一个“小模型 + 可验证 reward”的项目跑通，最后进入分布式训练、Deep Research、GUI/Computer Use 和多模态环境。

| 方向                 | 推荐先看                                                                                                                                                                                                                                                                             | 类型                 | 为什么值得看                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | ------------------------------------------------------------------------------------- |
| RLHF / 后训练经典    | [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155)、[Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)、[Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)                                                    | 经典论文/官方说明    | 理解 SFT、RM、PPO、DPO、RLAIF 和安全对齐的基本范式                                    |
| 现代后训练工程       | [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL)、[verl](https://github.com/verl-project/verl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                               | 生产/研究训练框架    | 直接看 rollout、vLLM/SGLang、Ray、Megatron、GRPO/DAPO、异步 agentic RL 怎么落地       |
| 推理 RLVR            | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[DeepSeek-R1 Nature](https://www.nature.com/articles/s41586-025-09422-z)、[Open-R1](https://github.com/huggingface/open-r1)、[TinyZero](https://github.com/Jiayi-Pan/TinyZero)                                            | 现代 reasoning 复现  | 最适合学习 verifiable reward、GRPO/RLVR、冷启动数据、长推理和 reward hacking          |
| 开源大模型底座       | [Qwen3.6](https://github.com/QwenLM/Qwen3.6)、[Qwen3](https://github.com/QwenLM/Qwen3)、[Meta Llama Models](https://github.com/meta-llama/llama-models)                                                                                                                              | 开源模型             | 适合拿来做 SFT/DPO/GRPO、工具调用、长上下文和 agentic coding 实验                     |
| Deep Research        | [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)、[Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)、[WebThinker](https://github.com/RUC-NLPIR/WebThinker)、[Search-R1](https://github.com/PeterGriffinJin/Search-R1)            | 产品/开源研究        | 把搜索、阅读、证据筛选、引用和长报告合成变成可训练 trajectory                         |
| Agent 框架与工具调用 | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)、[Google ADK](https://github.com/google/adk-python)、[Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)、[AutoGen](https://github.com/microsoft/autogen)                                  | Agent 工程框架       | 学工程边界：tools、handoff、guardrails、tracing、session、agent trajectory 和 RL 接口 |
| GUI / Computer Use   | [OpenAI CUA](https://openai.com/index/computer-using-agent/)、[Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)、[ByteDance UI-TARS](https://github.com/bytedance/UI-TARS)、[OSWorld](https://os-world.github.io/)           | 模型/工具/基准       | 现代 computer use 的核心材料：截图、坐标动作、网页/桌面/手机任务成功率                |
| VLM / VLA / 机器人   | [VLM-R1](https://github.com/om-ai-lab/VLM-R1)、[Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner)、[Gemini Robotics](https://deepmind.google/models/gemini-robotics/)、[LeRobot](https://github.com/huggingface/lerobot)                             | 多模态/具身          | 把视觉问答、定位、GUI 点击、机器人动作和可验证奖励接起来                              |
| 世界模型             | [DreamerV3 Nature](https://www.nature.com/articles/s41586-025-08744-2)、[DreamerV3 Code](https://github.com/danijar/dreamerv3)、[Google DeepMind Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)、[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | 经典/前沿/仿真       | 从可复现 world model 到交互式世界生成，再到机器人并行仿真                             |
| 生成模型 RL          | [DDPO](https://github.com/jannerm/ddpo)、[Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo)、[AlignProp](https://align-prop.github.io/)、[RLAIF-V](https://github.com/RLHF-V/RLAIF-V)、[VideoAlign](https://github.com/KlingAIResearch/VideoAlign)                | 图像/视频/多模态奖励 | 学会把美学、偏好、安全、图文一致性或视频质量变成优化目标                              |

**子标题导航**：

- [LLM 后训练](#llm-后训练)
- [LLM 推理](#llm-推理)
- [Deep Research RL](#deep-research-rl)
- [Agentic RL 与工具调用](#agentic-rl-与工具调用)
- [GUI 与 Computer Use](#gui-与-computer-use)
- [VLM](#vlm)
- [世界模型与模拟器](#世界模型与模拟器)
- [生成模型 RL](#生成模型-rl)
- [评测基准和项目](#评测基准和项目)
- [复现顺序建议](#复现顺序建议)

##### LLM 后训练

LLM 后训练主要处理“模型如何更符合人类偏好、任务格式和安全约束”。这一层的关键词是 SFT、Reward Model、PPO、DPO、KTO、ORPO、RLOO、GRPO。入门建议先用小模型跑通 TRL，再进入 OpenRLHF、verl、NeMo-RL 这类分布式训练栈。

这一方向要先分清三件事：**数据从哪里来、奖励怎么给、策略怎么更新**。SFT 解决“会不会按格式回答”，偏好优化解决“两个回答更喜欢哪个”，PPO/GRPO/RLOO 解决“模型自己采样之后，如何根据奖励继续变好”。很多后训练项目失败，并不是算法写错，而是这三件事混在一起：用 SFT 数据期待模型产生探索，用偏好数据期待模型学会可验证推理，或者用过于单薄的 reward 期待模型获得稳定能力。

###### 复现目标

最小项目可以这样设计：选一个 0.5B 到 3B 的 instruct 模型，先用少量指令数据做 SFT，再用 DPO/KTO 做一轮偏好优化，最后用规则奖励或小型 reward model 做 PPO/GRPO。每一步都保存 20 到 50 条固定评测 prompt，观察回答长度、格式遵守率、拒答率、重复句式和人工偏好变化。这样你不是只在跑 loss，而是在看“后训练到底改变了什么行为”。

###### 选型建议

如果目标是理解算法，优先用 TRL，因为它暴露的 Trainer 接口最适合拆 loss、reward 和 KL。如果目标是中文模型快速出结果，用 LLaMA-Factory 或 ms-swift 更省时间。如果目标是理解工业训练系统，就要看 OpenRLHF、verl、NeMo-RL：它们真正关心的是 rollout、训练、推理引擎、权重同步、显存和吞吐，而不只是一个损失函数。

| 资源                                                                                                              | 重点                                                  | 适合做什么                                   |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| [OpenAI InstructGPT](https://arxiv.org/abs/2203.02155)                                                            | SFT、Reward Model、PPO 的经典范式                     | 理解 RLHF 为什么成为 LLM 后训练主线          |
| [Anthropic Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) | RLAIF、安全偏好、原则反馈                             | 理解“AI feedback + 偏好优化”的经典路线       |
| [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)                                                            | 开源模型的 post-training 说明                         | 看工业模型如何组织预训练、指令微调和安全评测 |
| [Hugging Face TRL](https://huggingface.co/docs/trl/index)                                                         | PPO、DPO、KTO、ORPO、RLOO、GRPO                       | 小规模后训练实验、算法理解、课程演示         |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)                                                                  | Ray、vLLM、DeepSpeed、PPO/GRPO/DPO                    | 大模型 RLHF/RLVR 工程复现                    |
| [verl](https://github.com/verl-project/verl)                                                                      | 分布式 rollout、PPO/GRPO/DPO/SFT                      | R1 类训练管线、复杂 reward 训练              |
| [Open-Instruct](https://github.com/allenai/open-instruct)                                                         | SFT、DPO、PPO、偏好对齐                               | 学习完整开源对齐流程                         |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)                                                                      | NVIDIA 生态的大规模 RL 训练                           | 多 GPU 后训练、生产级训练栈                  |
| [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                                                                 | Decoupled Clip、动态采样、token-level policy gradient | 研究 R1 类训练 recipe 如何从论文走向代码     |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)                                                         | SFT、DPO、PPO、KTO、ORPO                              | 中文社区常用，适合快速跑通实验               |
| [ms-swift](https://github.com/modelscope/ms-swift)                                                                | SFT、RLHF、GRPO、多模态微调                           | 中文模型、多模态模型后训练                   |

###### 常见坑

- **只看 reward 曲线**：reward 上升不一定代表质量上升，可能只是回答变长、格式更讨巧、模板更固定。
- **跳过固定评测集**：没有固定 prompt，就很难判断模型是真的变强，还是采样随机性造成的错觉。
- **忽略 KL 和长度**：后训练最容易把模型推向“长、啰嗦、保守、重复”的局部最优。
- **直接上大框架**：没有先在 TRL 跑清楚 reward 和 loss，大规模框架只会把问题放大。

##### LLM 推理

推理型 RL 的核心不是“让回答更像人类喜欢的答案”，而是“让模型在数学、代码、逻辑、符号任务里学会可验证的推理过程”。这里常见奖励来自最终答案、单元测试、格式校验、符号执行器和过程检查器。

这一方向和普通 RLHF 的关键区别在于：奖励不一定来自人类偏好，而可以来自 verifier。数学题能判答案，代码题能跑测试，格式题能正则校验，符号题能用解释器执行。只要 verifier 足够可靠，模型就可以通过大量自采样获得训练信号。这也是 RLVR、GRPO 和 R1 类工作适合放在一起看的原因。

###### 复现目标

第一阶段建议从 Countdown、算术、格式化 JSON、简单 Python 函数这类任务开始。目标不是追求榜单，而是亲手看到四个现象：模型会探索不同解法；错误答案也可能有高置信推理；奖励稀疏时训练很不稳定；一旦 verifier 有漏洞，模型很快会钻漏洞。第二阶段再迁移到 GSM8K、MATH、代码单测或 reasoning-gym 生成任务。

###### 项目拆法

一个干净的推理 RL 项目至少有五个文件层次：任务生成、答案解析、reward function、rollout 采样、训练配置。不要把答案解析藏在 reward 里，否则 badcase 会很难查。每次训练后都要抽样保存“题目、模型完整输出、解析出的答案、标准答案、reward、失败原因”，这比单看平均分有用得多。

| 资源                                                                           | 重点                                                      | 适合做什么                                       |
| ------------------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------ |
| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)                      | R1/R1-Zero、RLVR、蒸馏模型                                | 直接阅读现代 reasoning RL 的官方开源入口         |
| [DeepSeek-R1 Nature](https://www.nature.com/articles/s41586-025-09422-z)       | DeepSeek-R1 的正式论文版本                                | 对照理解 cold start、RL、拒答/可读性和评测       |
| [Qwen3.6](https://github.com/QwenLM/Qwen3.6)                                   | Alibaba Qwen 新一代开源模型                               | 作为现代底座测试 SFT、GRPO、工具调用和长上下文   |
| [Qwen3](https://github.com/QwenLM/Qwen3)                                       | thinking / non-thinking 模式                              | 比较推理模型和普通 instruct 模型的训练与推理差异 |
| [Open-R1](https://github.com/huggingface/open-r1)                              | DeepSeek-R1 训练组件复现                                  | 研究 reasoning model 的数据、SFT、RL 和评测流程  |
| [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) | 从零训练 reasoning model 的开源框架                       | 复现 R1-Zero 式 RLVR、数据生成和评测流程         |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero)                              | 小规模 R1-Zero 式 RLVR                                    | 用 Countdown 等任务理解 GRPO/RLVR                |
| [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)                              | 去 KL、难题采样、clip-higher、token-level policy gradient | 理解 2025 年后 R1 类 recipe 的工程细节           |
| [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) | 不依赖人工标注推理数据的自博弈式 reasoning RL             | 研究 code executor、self-play 和 verifier 组合   |
| [rLLM / DeepScaleR](https://github.com/rllm-org/rllm)                          | 数学、代码、agent 任务上的 GRPO、REINFORCE、RLOO          | 研究长上下文、可验证奖励和 agent trajectory 训练 |
| [reasoning-gym](https://github.com/open-thought/reasoning-gym)                 | 自动生成推理任务                                          | 自定义 verifier 和可验证训练任务                 |
| [Math-Verify](https://github.com/huggingface/Math-Verify)                      | 数学答案解析与校验                                        | 给数学推理任务写 reward function                 |
| [OpenThoughts](https://github.com/open-thoughts/open-thoughts)                 | 开源 reasoning 数据与配方                                 | 构造 R1 类 SFT/RL 数据和评测集                   |

###### 常见坑

- **答案解析太脆**：模型答对了但解析失败，或答错了却因为格式漏洞拿到奖励。
- **把长推理当成好推理**：推理变长可能只是模型学会拖延，不代表逻辑更正确。
- **训练集太容易**：reward 很快饱和，模型只学会固定模板。
- **训练集太难**：几乎全是 0 reward，策略没有梯度信号。

##### Deep Research RL

Deep Research RL 关注“模型如何主动检索、阅读、筛选证据并生成研究型回答”。它和普通 RAG 的区别是：检索行为本身也是策略，模型需要学习什么时候搜、搜什么、读哪些页面、如何引用证据，以及什么时候停止。

普通 RAG 的检索通常是一次性的：用户提问，系统取回若干文档，模型生成答案。Deep Research 更像多轮决策：先把问题拆开，搜索第一批资料，阅读后发现缺口，再改写查询，继续追证据，最后综合成报告。RL 的价值在于训练这些中间决策，而不仅是训练最后一段文字。

###### 复现目标

最小复现不必一开始接真实互联网。可以先准备一个静态文档库，封装 `search`、`open`、`find` 三个工具，让模型在受控环境里完成多跳问答。奖励可以分成四项：最终答案是否正确、引用是否来自真实文档、证据是否支持结论、工具调用次数是否合理。等这个闭环稳定后，再迁移到 BrowserGym、WebArena 或真实搜索环境。

###### 观察重点

Deep Research 的日志比最终答案更重要。你要看模型是否会重复搜索同一个关键词，是否只打开排名第一的页面，是否引用没有读过的来源，是否在证据不足时强行下结论。好的 agent 不只是“答对”，还应该表现出规划、交叉验证、遇到冲突时回查、证据不足时收缩结论范围。

| 资源                                                                        | 重点                                          | 适合做什么                                            |
| --------------------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) | 产品级 deep research agent                    | 对照真实产品形态，理解长程检索、证据整合和报告输出    |
| [Alibaba Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)  | 开源 deep research 系列研究集合               | 追 WebWalker、WebDancer、WebSailor、WebWatcher 等路线 |
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1)                   | 搜索增强推理 RL                               | 训练模型在推理中主动检索和整合证据                    |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN)                                  | 生成式 agent 环境与 reasoning trajectory      | 构造 research/search 环境并训练多步策略               |
| [rLLM](https://github.com/rllm-org/rllm)                                    | 搜索、数学、代码、金融等可验证 agent 任务     | 把 reward function、rollout 和 trainer 解耦           |
| [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)                | 面向 deep research 的 RL 训练框架             | 研究多步检索、信息筛选和长答案合成                    |
| [WebThinker](https://github.com/RUC-NLPIR/WebThinker)                       | Web search + reasoning 的 deep research agent | 训练/评测带搜索的长程推理策略                         |
| [WebArena](https://github.com/web-arena-x/webarena)                         | 真实网站任务环境                              | 做网页信息查找、跨页面任务和最终成功率评测            |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym)                      | 浏览器环境统一接口                            | 把搜索、点击、阅读、表单填写接成 agent 训练环境       |
| [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)                       | 网页操作数据集                                | 从行为克隆过渡到网页 agent RL                         |

###### 常见坑

- **只奖励最终答案**：长程任务里只给终态 reward，信用分配会非常困难。
- **引用数量当质量**：模型可能学会堆链接，而不是用证据支撑论断。
- **真实网页不可复现**：页面变化、搜索排序变化、链接失效都会污染训练结果。
- **忽略成本约束**：无限搜索可以提高命中率，但真实系统必须控制延迟和调用成本。

##### Agentic RL 与工具调用

Agentic RL 的动作不只是输出 token，而是调用工具、执行代码、浏览网页、修改文件、查数据库或和外部系统交互。它的难点在于轨迹长、动作异构、奖励延迟、失败类型复杂，所以比普通 RLHF 更依赖环境封装、trajectory 切分和 verifier 设计。

这一方向可以看成“把 LLM 放进一个有状态环境里训练”。模型每一步输出的不只是自然语言，还可能是工具名、参数、代码片段、SQL、shell 命令或下一步计划。训练数据也不再是单条 prompt-response，而是一整条 episode：状态、动作、工具返回、环境变化、奖励和终止原因都要记录。

###### 复现目标

入门项目建议从低风险工具开始：计算器、文件搜索、受限 Python 执行器、文本世界环境。先让模型学会“什么时候该调用工具”，再学“调用什么工具”，最后才学复杂参数生成。代码 agent 可以用单元测试作为 reward，业务流程 agent 可以用数据库状态是否满足约束作为 reward，网页 agent 可以用任务完成率作为 reward。

###### 工程拆分

不要一上来就把 agent 框架、环境、reward、trainer 写在一起。更稳的结构是：环境只负责执行动作和返回 observation；reward 只负责判分；trajectory store 只负责记录；trainer 只消费转换后的样本。Agent Lightning、AReaL、SkyRL 这类项目值得看的地方，正是它们如何把这些边界拆开。

| 资源                                                                                                      | 重点                                   | 适合做什么                                            |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)                                       | tools、handoffs、guardrails、tracing   | 搭一个现代 agent 外壳，再把轨迹接到评测或 RL 数据管线 |
| [Google ADK](https://github.com/google/adk-python)                                                        | code-first agent、评测、部署           | 学 Google 风格的 agent 工程接口和多工具编排           |
| [AutoGen](https://github.com/microsoft/autogen)                                                           | 多 agent 对话、工具与应用框架          | 做多 agent 协作、代码/数据任务和 agent baseline       |
| [Agent Lightning](https://github.com/microsoft/agent-lightning)                                           | 将 agent 执行轨迹拆成可训练 transition | 给已有 LangChain、AutoGen、OpenAI Agents 等系统接 RL  |
| [AReaL](https://github.com/inclusionAI/AReaL)                                                             | 异步 LLM RL 与 agentic RL              | 研究 rollout 与训练解耦、异步更新和系统吞吐           |
| [SkyRL](https://docs.skyrl.ai/docs)                                                                       | 大规模 LLM/agent RL training stack     | 理解 rollout、trainer、reward、evaluation 的工程分层  |
| [AgentGym](https://github.com/WooooDyy/AgentGym)                                                          | 多环境 LLM agent 训练                  | 做多任务 agent 训练和环境接口对比                     |
| [ALFWorld](https://github.com/alfworld/alfworld)、[ScienceWorld](https://github.com/allenai/scienceworld) | 文本世界与科学实验环境                 | 低成本研究长程规划、语言动作和稀疏奖励                |
| [ToolBench](https://github.com/OpenBMB/ToolBench)                                                         | API 工具调用                           | 研究 tool selection、参数生成和调用成功率             |
| [tau-bench](https://github.com/sierra-research/tau-bench)                                                 | 真实业务流程中的工具 agent             | 客服、订单、数据库状态、多轮约束任务                  |
| [WebShop](https://github.com/princeton-nlp/WebShop)                                                       | 网购搜索和决策                         | 经典 language agent 环境，可做 supervised-to-RL       |
| [SWE-bench](https://github.com/swe-bench/SWE-bench)                                                       | 软件工程 agent 评测                    | 用测试通过率作为代码 agent 的可验证奖励               |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)                                                             | 软件工程 RL 环境                       | 训练 coding agent 的交互式环境                        |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench)                                       | 终端任务 agent                         | 研究 shell action、长程执行和任务成功率               |

###### 常见坑

- **工具返回参与 loss**：工具输出不是模型生成的 token，不应该让模型为工具结果负责。
- **环境状态不可恢复**：一次失败污染后续任务，训练数据会变得难以解释。
- **奖励太晚**：只在最后给成功/失败，模型不知道哪一步决策出了问题。
- **动作空间过大**：直接开放 shell、浏览器和文件系统，会让探索极其低效。

##### GUI 与 Computer Use

GUI/Computer Use 方向把动作空间从 token 扩展到点击、输入、滚动、快捷键、窗口切换和坐标定位。它和 Deep Research 的网页阅读不同，更强调真实界面状态变化、视觉 grounding、长程操作和任务完成率。

GUI agent 的难点在于“同一句动作描述，在不同屏幕状态下可能完全不同”。按钮位置会变，弹窗会出现，滚动区域会遮挡，移动端和桌面端布局也不同。因此这里的核心不是让模型背 UI，而是让它把截图、DOM、可访问性树、历史动作和任务目标对齐起来。

###### 复现目标

建议先从行为克隆或 action ranking 开始，而不是直接 RL。用 Mind2Web 或 BrowserGym 这类环境，让模型在给定网页状态下选择下一步动作；等动作格式稳定后，再引入 online rollout 和任务成功奖励。桌面 GUI 和手机 GUI 可以先做短任务，例如打开设置、查找信息、填写简单表单，再扩展到跨应用多轮流程。

| 资源                                                                                                      | 重点                                 | 适合做什么                                                  |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------- |
| [OpenAI CUA](https://openai.com/index/computer-using-agent/)                                              | Operator 背后的 computer-using agent | 看现代 GUI agent 如何结合视觉、推理、鼠标键盘动作和安全边界 |
| [Anthropic Computer Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) | Claude 的 computer use 工具接口      | 学截图、坐标动作、工具回传、beta API 和安全注意事项         |
| [ByteDance UI-TARS](https://github.com/bytedance/UI-TARS)                                                 | 原生 GUI agent 模型                  | 做桌面/网页/移动端统一 GUI 操作和 grounding 复现            |
| [WebArena](https://github.com/web-arena-x/webarena)                                                       | 真实网站任务环境                     | 网页表单、账户管理、购物、内容管理任务                      |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena)                                           | 多模态网页 agent                     | 结合截图、DOM 和语言进行网页操作                            |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym)                                                    | 浏览器环境统一接口                   | 接入网页 agent、VLM agent 和 RL rollout                     |
| [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)                                                     | 网页操作数据集                       | 从行为克隆、action ranking 过渡到 online RL                 |
| [OSWorld](https://os-world.github.io/)                                                                    | 桌面 GUI agent 基准                  | 真实软件任务、截图理解、坐标动作和任务成功率                |
| [AndroidWorld](https://github.com/google-research/android_world)                                          | Android 手机 agent                   | 手机 UI 操作、应用间跳转和长程任务                          |

###### 常见坑

- **坐标过拟合**：模型记住屏幕位置，而不是理解控件语义。
- **截图和 DOM 不一致**：视觉上可见的东西，DOM 中未必容易定位；DOM 中存在的元素，也可能被遮挡。
- **评测只看最终页面**：有些任务需要中间状态正确，例如保存设置、提交表单、确认订单。
- **缺少失败分类**：点击错、没找到、格式输错、页面加载慢，应该分开统计。

##### VLM

VLM 方向的 RL 重点是把视觉理解、语言推理和动作 grounding 连起来。奖励可以来自图文答案正确性、OCR、bbox/point 定位、GUI 点击位置、机器人任务成功率，也可以来自 ImageReward 这类偏好模型。

VLM RL 不只是“给图片问答加一个奖励”。它至少包含四类任务：视觉问答、空间定位、GUI 操作和视觉控制。不同任务的 reward 完全不同：问答看答案，定位看 IoU 或点位误差，GUI 看动作是否改变了界面状态，机器人看任务是否完成。复现时要先选清楚任务类型，否则 reward 很容易写成四不像。

###### 复现目标

最小项目可以从图文答案校验开始，例如给定图片和问题，让模型输出固定格式答案，用规则或外部标注判分。第二步做 grounding，让模型输出 bbox 或点击点，用 IoU、距离误差或目标命中率评估。第三步再进入 GUI 或机器人任务，因为这些任务会引入连续状态、长程动作和真实环境噪声。

###### 训练观察

VLM 训练必须保存可视化 badcase。只保存文本日志是不够的：你需要把原图、模型标注点、目标框、输出答案和 reward 放在一起看。很多错误不是推理错，而是 OCR 读错、目标物体太小、坐标系转换错、图片缩放后点位偏移，或者模型被图中无关文字带偏。

| 资源                                                                                   | 适合复现                                 | 关注点                                         |
| -------------------------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)                       | VLM 的 GRPO/RLVR 入门                    | 可把 reward 写成图文答案正确性、格式、定位质量 |
| [HF VLM GRPO Cookbook](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl) | 小规模 VLM GRPO 实操                     | 从数据、reward function 到训练脚本的端到端示例 |
| [VLM-R1](https://github.com/om-ai-lab/VLM-R1)                                          | 视觉语言模型的 RL reasoning              | 适合看多模态 chain-of-thought 与可验证奖励     |
| [LMM-R1](https://github.com/TideDra/lmm-r1)                                            | 多模态 reasoning 的两阶段训练框架        | 研究 cold start、GRPO 和 VLM 推理链            |
| [Open-Vision-Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner)     | 视觉推理模型的 RL 训练                   | 将 R1-Zero 类方法迁移到多模态推理              |
| [RL4VLM](https://rl4vlm.github.io/)                                                    | 视觉语言模型上的 RLHF / RLVR             | 项目页汇总了 RL 对 VLM 对齐和推理的研究脉络    |
| [Gemini Robotics](https://deepmind.google/models/gemini-robotics/)                     | Google DeepMind VLA / embodied reasoning | 理解 VLM/VLA 如何走向真实机器人动作和多步任务  |
| [LeRobot](https://github.com/huggingface/lerobot)                                      | 真实/仿真机器人数据、模仿学习和 RL       | 适合从视觉动作数据过渡到具身 RL                |
| [RoboCasa](https://github.com/robocasa/robocasa)                                       | 家庭场景机器人操作                       | 适合研究多任务视觉控制、语言条件任务和泛化     |
| [ManiSkill](https://maniskill.readthedocs.io/)                                         | 视觉操作与并行机器人环境                 | 可连接 VLM policy、视觉 reward 和底层控制      |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena)                        | 多模态网页 agent                         | 需要图像、DOM 和语言共同 grounding             |
| [OSWorld](https://github.com/xlang-ai/OSWorld)                                         | 桌面 GUI agent                           | 适合研究截图理解、坐标动作和真实软件任务       |
| [AndroidWorld](https://github.com/google-research/android_world)                       | Android 手机 agent                       | 手机 UI 操作、视觉 grounding 和长程任务        |
| [OpenVLA](https://github.com/openvla/openvla)                                          | vision-language-action model             | 适合连接 VLM、机器人动作和下游 RL fine-tuning  |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                            | 终身机器人学习                           | 多任务、语言条件、泛化和视觉操作               |
| [ImageReward](https://github.com/THUDM/ImageReward)                                    | 图像生成/多模态奖励模型                  | 可作为生成模型或 VLM 偏好优化的 reward signal  |

###### 常见坑

- **只看文本正确率**：视觉 grounding 错了，答案可能仍然碰巧正确。
- **坐标系混乱**：原图尺寸、模型输入尺寸、渲染尺寸不一致时，bbox/point 很容易错位。
- **数据泄漏**：一些 VQA benchmark 可能被预训练见过，需要用自建或时间切分数据补充评测。
- **奖励不可解释**：偏好模型给高分时，要能回看它到底奖励了什么视觉属性。

##### 世界模型与模拟器

世界模型关注“智能体能否学习一个可用于想象和规划的环境模型”。在经典 RL 中，它连接 Model-Based RL、规划和样本效率；在 LLM/Agent 时代，它也可以作为可控模拟器，用于训练长程任务、网页/工具环境和具身智能。

世界模型的价值不是“替代真实环境”，而是降低试错成本。智能体先从真实交互中学习一个近似动力学模型，再在模型里做短 rollout、规划或策略改进。关键问题永远是 model bias：模型只要有一点系统性错误，规划器就可能反复利用这个错误，最后学到现实中不可用的策略。

###### 复现目标

入门可以先做一个二维连续控制或 CartPole 变体：收集真实 transition，训练一个预测下一状态和奖励的小模型，再比较 model-free PPO/SAC 与 model-based rollout 的样本效率。进阶再看 DreamerV3、TD-MPC2、MBPO：它们分别代表 latent imagination、MPC planning 和短模型 rollout 三种典型路线。

###### 和 LLM Agent 的关系

在 Agentic RL 中，模拟器也可以是“任务世界模型”：网页快照、工具返回、用户状态、数据库状态，都可以被封装成可复现环境。这样训练时不必每次访问真实网站或真实业务系统，而是在记录好的状态转移中训练策略。它不能完全代替真实环境，但非常适合做离线调试、badcase 回放和课程项目。

| 资源                                                                                                                            | 适合复现                        | 关注点                                                     |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------- |
| [DreamerV3 Nature](https://www.nature.com/articles/s41586-025-08744-2)                                                          | 世界模型通用能力论文            | 看 fixed hyperparameters、Minecraft、Atari、控制任务等结果 |
| [Google Research Dreamer](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/) | Dreamer 系列官方入门            | 用官方解释理解 latent imagination 和 actor-critic          |
| [DreamerV3](https://github.com/danijar/dreamerv3)                                                                               | 像素输入世界模型                | latent dynamics、想象 rollout、actor-critic                |
| [Google DeepMind Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)                                | 交互式生成世界模型              | 了解 world model 如何从控制环境走向可交互世界生成          |
| [TD-MPC2](https://github.com/nicklashansen/tdmpc2)                                                                              | 连续控制和多任务 MBRL           | learned latent model、MPC planning、样本效率               |
| [mbrl-lib](https://github.com/facebookresearch/mbrl-lib)                                                                        | PETS、MBPO 等经典 MBRL          | 动力学模型、uncertainty、planning                          |
| [MBPO](https://github.com/JannerM/mbpo)                                                                                         | Model-Based Policy Optimization | 短模型 rollout 加 off-policy RL                            |
| [IRIS](https://github.com/eloialonso/iris)                                                                                      | Atari 世界模型                  | 离散 latent tokens、transformer dynamics、imagined rollout |
| [DayDreamer](https://github.com/danijar/daydreamer)                                                                             | 机器人世界模型                  | 把 Dreamer 思路迁移到真实/仿真机器人任务                   |
| [Brax](https://github.com/google/brax)                                                                                          | 可微分物理和并行仿真            | 适合研究高吞吐 model-free 与 model-based 控制              |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)                                                                              | 大规模机器人 RL                 | GPU 并行环境、domain randomization、Sim2Real               |
| [LeRobot](https://github.com/huggingface/lerobot)                                                                               | 数据驱动具身智能                | imitation learning、diffusion policy、RL fine-tuning       |
| [RoboCasa](https://github.com/robocasa/robocasa)                                                                                | 家庭场景机器人操作              | 长程任务、多物体交互、语言条件控制                         |

###### 常见坑

- **模型 rollout 太长**：预测误差会随步数累积，短 rollout 往往更稳。
- **只报告最终得分**：还要报告真实环境样本数、模型训练开销和规划开销。
- **忽略不确定性**：模型不知道自己不知道时，策略会跑到模型最不可靠的区域。
- **仿真太干净**：机器人和 GUI 任务尤其需要噪声、延迟、遮挡和随机初始状态。

##### 生成模型 RL

生成模型 RL 关注用奖励函数直接优化图像、视频或多模态生成结果。它和语言 RLHF 很像，但奖励通常来自美学分、图文一致性、安全性、偏好模型或人工/AI feedback。

这里可以把扩散模型的一次采样过程看成一条 trajectory：从噪声出发，经过多步去噪，最终得到图像。奖励通常只在最终图像上计算，例如美学分、文本一致性、安全分类、人工偏好或多模态模型打分。DDPO 这类方法的核心，就是把生成过程纳入 policy gradient，而不是只做监督式微调。

###### 复现目标

最小项目可以选择一个很窄的 prompt 集，例如“生成清晰的单个物体”“生成带指定颜色的图标”“生成符合文字描述的简单场景”。先固定 base model 和采样参数，再比较三类 reward：CLIP/图文一致性、美学或偏好分、规则化安全约束。每次训练后保存同一批 prompt 的 before/after 图片，肉眼检查 reward 是否真的对应质量提升。

| 资源                                                                  | 重点                     | 适合做什么                                       |
| --------------------------------------------------------------------- | ------------------------ | ------------------------------------------------ |
| [DDPO](https://github.com/jannerm/ddpo)                               | 扩散模型 RL 的经典实现   | 理解把 denoising steps 看成 MDP 后如何做策略梯度 |
| [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo) | 用 DDPO 微调扩散模型     | 复现文本到图像模型的 policy gradient 优化        |
| [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch)              | DDPO 原始开源实现        | 研究扩散模型采样轨迹上的 RL 更新                 |
| [AlignProp](https://align-prop.github.io/)                            | 通过可微奖励优化生成模型 | 比较 backprop-through-reward 与 policy gradient  |
| [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)                          | 视觉语言反馈驱动的对齐   | 用 AI feedback 优化多模态生成/理解               |
| [ImageReward](https://github.com/THUDM/ImageReward)                   | 图像偏好奖励模型         | 给图像生成 RL 提供 reward signal                 |
| [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)           | 视频生成的人类反馈对齐   | 看视频 reward、DPO/RL、reject sampling 如何组合  |

###### 常见坑

- **奖励模型被利用**：模型可能学会生成某些高分纹理、构图或关键词风格，而不是真的更符合人类偏好。
- **只看平均 reward**：必须看图像样本，否则很容易奖励上涨、质量下降。
- **prompt 覆盖太窄**：在小 prompt 集上优化过头，会损害泛化和多样性。
- **安全和美学冲突**：多目标 reward 需要记录每个子分数，不能只看加权总分。

##### 评测基准和项目

LLM 时代做 RL，评测往往比训练更容易出问题。一个方向至少要同时看最终成功率、过程质量、格式约束、reward hacking、长度偏置、数据泄漏和多次采样稳定性。下面这些基准可以作为复现项目的验收入口。

评测不要只问“分数有没有涨”，还要问“涨在哪里、代价是什么、有没有破坏别的能力”。一个合格的复现实验至少应该包含：固定 dev set、未见过的 test set、训练前后对比、多次采样均值和方差、badcase 分类、长度统计、调用成本统计。Agentic/VLM/GUI 任务还应该保存完整轨迹，方便回放。

###### 验收清单

- **最终指标**：准确率、通过率、任务成功率、偏好胜率。
- **过程指标**：工具调用次数、无效动作比例、重复搜索比例、引用可访问率、代码测试失败类型。
- **稳定性指标**：不同随机种子、不同采样温度、不同模型尺寸下是否仍然有效。
- **安全指标**：是否更容易胡编引用、越权调用工具、泄漏环境信息或破坏格式约束。
- **成本指标**：平均 token、平均工具调用、平均时延、训练和评测的 GPU/CPU 开销。

| 方向          | 推荐基准/项目                                                                                                                                                                                                                       | 主要评测信号                              |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 奖励模型      | [RewardBench](https://github.com/allenai/reward-bench)                                                                                                                                                                              | reward model 是否真的偏向更好回答         |
| 数学推理      | [Math-Verify](https://github.com/huggingface/Math-Verify)、[reasoning-gym](https://github.com/open-thought/reasoning-gym)                                                                                                           | 答案可验证性、格式、推理稳定性            |
| 代码推理      | [EvalPlus](https://github.com/evalplus/evalplus)、[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)                                                                                                                   | 单元测试通过率、泛化到新题能力            |
| 软件工程      | [SWE-bench](https://github.com/swe-bench/SWE-bench)、[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)                                                                                                                                  | patch 是否解决真实 issue                  |
| 网页操作      | [WebArena](https://github.com/web-arena-x/webarena)、[BrowserGym](https://github.com/ServiceNow/BrowserGym)、[Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)                                                                  | 网页任务成功率、点击轨迹、状态变化        |
| 多模态网页    | [VisualWebArena](https://github.com/web-arena-x/visualwebarena)                                                                                                                                                                     | 截图 grounding、DOM grounding、任务成功率 |
| 桌面/手机 GUI | [OSWorld](https://github.com/xlang-ai/OSWorld)、[AndroidWorld](https://github.com/google-research/android_world)                                                                                                                    | 真实应用中的长程操作成功率                |
| 工具调用      | [ToolBench](https://github.com/OpenBMB/ToolBench)、[tau-bench](https://github.com/sierra-research/tau-bench)                                                                                                                        | 工具选择、参数正确性、业务流程完成率      |
| 机器人/VLA    | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[ManiSkill](https://maniskill.readthedocs.io/)、[RoboCasa](https://github.com/robocasa/robocasa)                                                                       | 任务成功率、泛化、碰撞和安全成本          |
| 图像生成 RL   | [Diffusers DDPO](https://huggingface.co/docs/diffusers/training/ddpo)、[ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch)、[AlignProp](https://github.com/mihirp1998/AlignProp)、[RLAIF-V](https://github.com/RLHF-V/RLAIF-V) | 偏好奖励、美学分、安全性、一致性          |

###### Badcase 模板

每个方向都建议维护一个 `badcases.jsonl` 或表格，至少记录：任务 id、输入、模型输出、reward、判分理由、失败类型、是否可复现、修复建议。对 LLM RL 来说，badcase 不是训练后的附属品，而是下一轮 reward 设计、数据过滤和环境修复的入口。

##### 复现顺序建议

先用 0.5B 到 3B 小模型和数学、代码、格式校验等可验证任务观察 reward hacking、长度偏置和采样温度影响；再从 TRL/TinyZero/Open-R1 迁移到 verl/OpenRLHF 这类分布式框架。Agentic RL 优先选择搜索、网页、代码这类有明确成功率的任务；VLM RL 优先选择图文答案、定位、OCR、GUI 点击这类可判分任务；世界模型和具身方向先跑通 DreamerV3/TD-MPC2，再叠加视觉和真实机器人复杂度。

###### 一条稳妥路线

1. **第 1 周：规则奖励任务**
   用 TRL 或 TinyZero 跑一个小型可验证任务，例如 Countdown、格式化 JSON、简单数学。目标是理解 rollout、reward、advantage、KL、长度偏置和日志保存。

2. **第 2 周：偏好优化和后训练对比**
   用同一个小模型做 SFT、DPO/KTO、PPO/GRPO 对比。不要换太多变量，只看不同训练方式对同一批 prompt 的影响。

3. **第 3 周：推理 RLVR**
   引入 Math-Verify、reasoning-gym 或代码单测，让 reward 从“格式正确”升级到“答案可验证”。重点观察 reward sparsity 和 verifier 漏洞。

4. **第 4 周：工具调用或 Deep Research**
   封装一个小型搜索/阅读环境，记录完整 trajectory。先做离线轨迹回放，再做 online rollout。

5. **第 5 周：VLM 或 GUI**
   选择一个视觉问答、bbox 定位或网页点击任务，加入可视化 badcase。重点检查坐标系、截图状态和 reward 可解释性。

6. **第 6 周以后：分布式和工业框架**
   再进入 verl、OpenRLHF、AReaL、SkyRL 这类框架。此时你已经知道自己需要什么 reward、什么日志和什么评测，不会被工程复杂度牵着走。

###### 怎么判断可以升级难度？

当一个任务满足三点，就可以进入下一层：第一，固定评测集上训练前后差异稳定；第二，badcase 能被清楚分类；第三，reward 上升时人工抽检质量也上升。否则不要急着换更大的模型或更复杂的环境，先修 reward、数据和日志。
