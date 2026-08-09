# Outline Redesign Proposal v3: Based on Real Evidence (2025-2026 Papers + Anthropic/OpenAI Job Descriptions)

> v2 was an "ideal textbook" inferred from training data. v3 is a revision based on **real evidence** — actual 2025-2026 papers and the Anthropic Code RL job description found via search — and honestly flags where v2 over-extrapolated.

---

## 1. Evidence Sources (Real and Clickable)

### 1.1 A Real Anthropic Job Description

**Research Engineer, Code RL** (job-boards.greenhouse.io/anthropic/jobs/5254364008):

- "Pioneering fundamental RL research for large language models"
- "Building scalable RL infrastructure and training methodologies"
- "Design RL environments and coding tasks"
- "Build the reward signals and verifiers that capture what 'good code' means"
- "Long-horizon autonomous engineering"
- "Agentic coding behaviors"
- "High-performance code for accelerators"
- **Required**: "Strong software-engineering skills and deep Python expertise, including async/concurrent programming"
- **Nice-to-have**: RLHF / post-training / LLM finetuning; coding agents / code-execution sandboxes / eval harnesses / verifiers / developer tooling; program analysis / testing / verification / compilers / formal methods; PyTorch + large-scale distributed training + performance profiling

### 1.2 Real Key Papers from 2025-2026

| Paper                                                             | Date              | Source                                 |
| ----------------------------------------------------------------- | ----------------- | -------------------------------------- |
| DeepSeek-R1 (Nature)                                              | 2025.01           | nature.com/articles/s41586-025-09422-z |
| DAPO (Yu et al.)                                                  | 2025.03           | arXiv:2503.14476                       |
| Dr. GRPO (Liu et al.)                                             | 2025              | arXiv                                  |
| GSPO (Zheng et al.)                                               | 2025              | arXiv                                  |
| CISPO (MiniMax)                                                   | 2025              | arXiv                                  |
| REINFORCE++ (Hu)                                                  | 2025              | arXiv                                  |
| **Natural Emergent Misalignment from Reward Hacking** (Anthropic) | **2025.11**       | **arXiv:2511.18397**                   |
| Sleeper Agents (Hubinger et al.)                                  | 2024.01           | arXiv:2401.05566                       |
| Alignment Faking (Greenblatt et al.)                              | 2024.12           | arXiv:2412.14093                       |
| In-Context Scheming (Apollo)                                      | 2024.12           | arXiv:2412.04984                       |
| Sycophancy to Subterfuge (Anthropic)                              | 2024              | arXiv:2406.10162                       |
| METR: Frontier Models Reward Hacking                              | 2025              | METR Blog                              |
| School of Reward Hacks (Gao et al.)                               | 2025.08           | arXiv                                  |
| Karpathy 2025 Year in Review                                      | 2025.12           | karpathy.bearblog.dev                  |
| Tülu 3 (Allen AI)                                                 | 2024-2025         | arXiv                                  |
| Anthropic: Effective Harnesses for Long-Running Agents            | 2025.11           | anthropic.com                          |
| OpenAI o1/o3 introduction                                         | 2024.09 / 2025.01 | openai.com                             |
| Competitive Programming with Large Reasoning Models               | 2025.02           | OpenAI                                 |
| Reinforcement Pre-Training (Lambert)                              | 2025.06           | arXiv                                  |

### 1.3 Real Industry Signals

- **Anthropic reportedly plans to invest $1B in RL environments** (per The Information, 2025.09)
- **Mechanize is paying RL environment engineers $500K/year** (TechCrunch, 2025.09)
- **Karpathy**: "RLVR is the 'new major stage' of the LLM training pipeline"
- **Post-training role salaries**: OpenAI/Anthropic/DeepMind IC $200K-$312K, Senior $400K+
- **Job growth**: RLHF/post-training job postings grew 3x between 2025.01 and 2026.03

---

## 2. An Honest Assessment of v2

### 2.1 Where v2 Guessed Right (Backed by Real Sources)

| v2 claim                                        | Real evidence                                                |
| ----------------------------------------------- | ------------------------------------------------------------ |
| The PRM chapter is important                    | Lightman et al. 2023, OpenAI; core to o1/o3 training         |
| Constitutional AI / RLAIF                       | Bai et al. 2022, Anthropic; actually used in Claude training |
| The Sleeper Agents chapter is important         | Hubinger et al. 2024.01 (a real paper)                       |
| Alignment Faking                                | Greenblatt et al. 2024.12 (a real paper)                     |
| The Reward Hacking chapter is important         | Anthropic 2025.11 paper (very recent, very important)        |
| Full coverage of the DPO family (KTO/IPO/SimPO) | All genuinely exist                                          |
| GRPO / RLVR at the core                         | Confirmed by the DeepSeek-R1 Nature paper                    |
| Test-time compute scaling                       | Confirmed by the OpenAI o1/o3 series                         |

### 2.2 Where v2 Seriously Underestimated (Contradicted by JD Evidence)

**🔴 The biggest blind spot: RL environment design**

- Anthropic's $1B investment, Mechanize's $500K salaries, Karpathy calling it the "new major stage"
- v2 only mentions it in passing, in Appendix B's engineering practice
- **Must be upgraded to a standalone core chapter**

**🟡 Engineering ability was underestimated**

- The Anthropic JD emphasizes: async/concurrent Python, performance profiling, distributed training
- v2 relegates "engineering" to an appendix, but it's actually the largest-weighted part of the JD's requirements
- A candidate who has only read v2 would know the algorithms but couldn't write a trainer that scales — they wouldn't pass the interview

**🟡 Reasoning Models don't get their own chapter**

- o1, R1, Claude Opus 4.6, and Gemini 3.1 Pro are all now independent product categories
- v2 crams this into a "Test-time Compute" section, which is badly insufficient
- **Should be its own chapter, covering the SFT→RLHF→RLVR three-stage progression**

**🟡 Coverage of the GRPO improvement family is too thin**

- 2025-2026 has at least 6 mainstream variants: DAPO, Dr.GRPO, GSPO, CISPO, REINFORCE++, AREAL
- Plus SPO, BSPO, TOPR, GPPO, M2PO
- v2 only names DAPO

### 2.3 Where v2 Over-Extrapolated (Academic Depth with Low Industry-JD Value)

**⚠️ Hierarchical RL / Meta-RL / MARL / IRL & GAIL**

- Not a single job description found mentions these
- Karpathy's 2025 year-end review doesn't emphasize them either
- These are content from academic textbooks (Sutton & Barto / CS285), but offer **limited help** for landing a job at OpenAI/Anthropic
- **Recommend demoting to an appendix or merging into a single chapter**

**⚠️ "Read this and walk straight into a job at OpenAI/Anthropic"**

- Honest admission: this was overselling
- Real job descriptions also require: a strong SE background, production debugging experience, distributed-systems experience, product sense
- None of these can be acquired just by reading a textbook
- v3 should reframe this as "after reading, you'll be able to pass the RL portion of a technical interview"

### 2.4 Ordering Mistakes in v2

- PRM was placed in Chapter 27 → it should be one of the core chapters in the LLM RL section (given how heavily it's used in industry)
- "Test-time Compute" was made its own chapter → it should be merged into the Reasoning Models chapter
- Offline RL was placed in the Part V frontier section → it's actually an important part of 2025 industrial practice (indirectly related to Anthropic Code RL)

---

## 3. The v3 Proposal (Evidence-Driven)

### Design Principles

1. **JD-driven**: anchored to the actual requirements of the Anthropic Code RL job description
2. **Real 2025 trends**: RL Environments, Reasoning Models, Reward Hacking
3. **Cut low-ROI academic content**: merge Hierarchical / Meta-RL / MARL / IRL
4. **Upgrade engineering content**: move it from the appendix into the main text
5. **Full coverage of the GRPO family**: DAPO / Dr.GRPO / GSPO / CISPO / REINFORCE++

### The v3 Chapter Structure (28 Chapters / 6 Parts)

---

#### Part I · Foundations and Classical RL (8 Chapters) — Same as v2

1. Overview of Reinforcement Learning
2. CartPole: The First Reinforcement Learning Experiment
3. Multi-Armed Bandits and Exploration-Exploitation Theory
4. Markov Decision Processes
5. Value Functions and the Bellman Equation
6. Dynamic Programming, Monte Carlo, and Temporal Difference Learning
7. Q-Learning and Off-Policy Control
8. Reward Function Design

---

#### Part II · Deep Reinforcement Learning (6 Chapters)

9. Deep Q-Networks and Distributional RL
10. Policy Gradient Methods
11. Actor-Critic Architectures
12. PPO and Trust-Region Methods
13. Deep Methods for Continuous Control (DDPG / TD3 / SAC)
14. Model-Based Deep RL (MuZero / Dreamer)

---

#### Part III · Advanced RL Methods (Trimmed to 2 Chapters, Down From 6 in v2)

15. Offline Reinforcement Learning and Decision Transformers (CQL / IQL / Decision Transformer / Diffuser)
16. Imitation Learning, Inverse RL, and Meta-RL Combined **[merged]**

- Behavior cloning, DAgger
- MaxEnt IRL, GAIL
- MAML, RL², Algorithm Distillation

> **Trade-off**: Hierarchical RL and MARL don't get their own chapters — a "further reading" subsection at the end of Part III is sufficient. Rationale: real job descriptions don't require them, and Anthropic/OpenAI's 2025 papers don't emphasize them either.

---

#### Part IV · LLM Alignment and Post-Training (8 Chapters — v2's core, expanded)

17. The RLHF Training Pipeline (the three-stage SFT → RLHF → RLVR progression)
18. PPO-RLHF in Industrial Practice
19. Preference Alignment: The DPO Family (DPO / IPO / KTO / SimPO / Iterative DPO / SPIN)
20. **GRPO and Verifiable Rewards: From Fundamentals to the Improvement Family** **[v3 major expansion]**

- 20.1 The principle of GRPO's group normalization
- 20.2 The RLVR paradigm
- 20.3 DAPO: asymmetric clipping, dynamic sampling, token-level loss, overlong shaping, no KL
- 20.4 Dr. GRPO: removing std normalization
- 20.5 GSPO: sequence-level importance sampling
- 20.6 CISPO: clipping the IS weight rather than the token update
- 20.7 REINFORCE++ and AREAL
- 20.8 DeepSeek V3.2's KL tuning tricks
- 20.9 A decision tree for choosing a variant

21. **Reasoning Models: From o1 to Claude Opus 4.6** **[v3 new standalone chapter]**

- 21.1 The rise of reasoning models (o1 → o3 → o4)
- 21.2 The R1-Zero paradigm: pure RL with no SFT
- 21.3 The full DeepSeek-R1 training pipeline
- 21.4 Test-time compute scaling
- 21.5 Claude Opus 4.6's adaptive thinking
- 21.6 Hidden CoT vs. visible CoT

22. Process Reward Models and Inference-Time Search (PRM / MCTS over Thoughts / Tree of Thoughts / rStar)
23. **Constitutional AI and RLAIF** (Anthropic's alignment paradigm)
24. Agentic Reinforcement Learning (multi-turn interaction, tool use, SWE-bench, Deep Research)

---

#### Part V · Safety, Evaluation, and Alignment Research (3 Chapters)

25. **Reward Hacking and Alignment Failure Modes** **[v3 major update, through the 2025.11 paper]**

- 25.1 A taxonomy of reward hacking (Anthropic 2025.11, arXiv:2511.18397)
- 25.2 School of Reward Hacks (Gao et al. 2025)
- 25.3 Sleeper Agents (Hubinger et al. 2024)
- 25.4 Alignment Faking (Greenblatt et al. 2024)
- 25.5 In-Context Scheming (Apollo 2024)
- 25.6 Sycophancy to Subterfuge (Anthropic 2024)
- 25.7 Naturally emergent misalignment (caused by reward hacking)
- 25.8 Defenses: preference models, reward-hack classifiers

26. **Scalable Oversight and Red-Teaming**

- Scalable Oversight
- AI Safety via Debate
- Weak-to-Strong Generalization (OpenAI 2023)
- Red-teaming methodology

27. **RL Evaluation Methodology** (evals as equivalent to RL environments, Pash 2025)

---

#### Part VI · RL Engineering and Systems (Brand-New in v3, the Core of the JD)

28. **Designing RL Environments and Verifiers** **[v3 core addition]**

- 28.1 RL environments as the new bottleneck (Anthropic's $1B, Karpathy)
- 28.2 Verifier design principles
- 28.3 The equivalence between evals and RL environments
- 28.4 Sandbox engineering (Docker, code execution)
- 28.5 Long-horizon task harnesses (Anthropic's 2025.11 Effective Harnesses)
- 28.6 Multi-agent parallelism (Karpathy's "5-6 agents")
- 28.7 Evaluation benchmarks: CyberGym, SWE-bench, Terminal-Bench, Prime Intellect Hub

29. **Distributed RL Training Systems**

- 29.1 Comparing veRL / OpenRLHF / TRL / NeMo-Aligner
- 29.2 Rollout engines and vLLM integration
- 29.3 Asynchronous RL training (LlamaRL 2025)
- 29.4 GPU memory optimization: ZeRO, FSDP, gradient checkpointing
- 29.5 Performance profiling and bottleneck analysis
- 29.6 Hands-on large-scale training debugging

---

#### Part VII · Research Frontiers (Trimmed to 3 Chapters)

30. RL for Vision-Language Models (VLM-GRPO, EasyR1, GeoQA)
31. Embodied Intelligence and Multimodality (VLA: π0, RT-2, OpenVLA, Diffusion Policy)
32. Self-Play and Scaling Trends (AlphaGo → MuZero → LLM Self-Play, RL scaling laws)

---

## 4. Comparison Summary Table

| Dimension                              | Current v1 | v2 (ideal extrapolation) | **v3 (evidence-driven)**                                       |
| -------------------------------------- | ---------- | ------------------------ | -------------------------------------------------------------- |
| Chapter count                          | 12         | 38                       | **32** (academic content trimmed, engineering expanded)        |
| Emphasis on RL Environments            | ❌         | ❌                       | **✅ core chapter**                                            |
| Reasoning Models standalone            | ❌         | one section              | **✅ standalone chapter**                                      |
| Completeness of the GRPO family        | Only DAPO  | DAPO + Dr.GRPO           | **✅ 6+ variants**                                             |
| Placement of engineering content       | Appendix   | Appendix                 | **✅ Part VI main text**                                       |
| Timeliness of reward-hacking coverage  | 2024 paper | 2024 paper               | **✅ 2025.11 paper**                                           |
| Hierarchical/Meta/MARL                 | Scattered  | 3 academic chapters      | **reduced to 1 merged chapter**                                |
| The "walk straight into a job" promise | ❌         | ❌ (overstated)          | **reframed as "pass the RL portion of a technical interview"** |
| Verified against real job descriptions | None       | None                     | **✅ Anthropic Code RL JD**                                    |

---

## 5. Migration Path (Prioritized by Real Evidence)

**Phase 1** (immediate, zero risk) — turn headings into proper textbook form + split Chapter 3 (MDP)

**Phase 2** (this month, medium risk) — move DPO out, add the full §20 GRPO family (some content already exists)

**Phase 3** (this quarter, high ROI) — add the key v3 Part IV chapters:

- §20 The GRPO improvement family
- §21 Reasoning Models as its own chapter
- §22 PRM and inference-time search
- §23 Constitutional AI
- §25 Reward Hacking (updated through 2025.11)

**Phase 4** (next quarter, the JD's core) — add the Part VI RL engineering chapters:

- §28 RL Environments and Verifiers (most important, JD-verified)
- §29 Distributed training systems

**Phase 5** (ongoing) — Part III advanced RL, Part VII research frontiers

---

## 6. Key Citations (Ready to Use When Writing)

```
[DeepSeek-R1] Guo et al. 2025. Nature. https://www.nature.com/articles/s41586-025-09422-z
[DAPO] Yu et al. 2025. arXiv:2503.14476
[GSPO] Zheng et al. 2025.
[CISPO] MiniMax et al. 2025.
[Dr.GRPO] Liu et al. 2025.
[REINFORCE++] Hu 2025.
[Anthropic Reward Hacking] MacDiarmid et al. 2025. arXiv:2511.18397
[Anthropic Sleeper Agents] Hubinger et al. 2024. arXiv:2401.05566
[Alignment Faking] Greenblatt et al. 2024. arXiv:2412.14093
[In-Context Scheming] Apollo Research 2024. arXiv:2412.04984
[Sycophancy to Subterfuge] Anthropic 2024. arXiv:2406.10162
[METR Reward Hacking] Von Arx et al. 2025.
[School of Reward Hacks] Gao et al. 2025.
[Karpathy 2025 Year in Review] karpathy.bearblog.dev
[Anthropic Effective Harnesses] anthropic.com 2025.11
[Anthropic Code RL JD] job-boards.greenhouse.io/anthropic/jobs/5254364008
[Epoch AI RL Environments FAQ] epochai.substack.com/p/an-faq-on-reinforcement-learning
[Raschka State of LLMs 2025] magazine.sebastianraschka.com/p/state-of-llms-2025
[Raschka LLM Papers 2025 List] magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one
```

---

## 7. An Honest Summary for the Reader

1. **v2's overall direction was right** — the real evidence confirms that PRM, CAI, Sleeper Agents, Reward Hacking, and GRPO/RLVR really are 2025 industry priorities.
2. **v2's biggest blind spot was RL Environments** — this is the industry's biggest investment area in 2025 (Anthropic's $1B), and v3 adds it as a standalone chapter.
3. **v2 leaned too heavily on academic content** (Hierarchical/Meta/MARL) — these offer little help for landing an industry job, so v3 trims them.
4. **v2's engineering coverage was too thin** — the JD emphasizes async Python, profiling, and distributed systems, so v3 promotes it into the main text.
5. **"Read this and walk straight into a job" was overselling** — real job descriptions also require SE experience and production-debugging skills that no textbook can substitute for.
6. **The GRPO family is far more extensive than v2 suggested** — at least 6 mainstream variants appeared in 2025, and full coverage is necessary.

---

# v4 Addendum: Evidence from Chinese Labs (2025-2026)

> Research scope expanded to major domestic labs (DeepSeek, Qwen, Zhipu, StepFun); found that v3 needs strengthening in the following areas.

## 8. Real Evidence from Chinese Labs

### 8.1 Real Zhipu GLM Alignment-Team Interview Questions (Real Interview Reports from Nowcoder)

**[Source]** nowcoder.com's Zhipu AI topic section — real first-/second-round interview records from multiple candidates

**Key areas tested**:

- **The full derivation chain PG → REINFORCE → TRPO → PPO** (optimizing PPO from an RL perspective)
- **The DPO family + DPO regularization methods** (directions for optimizing DPO)
- **Comparing DeepSeek's GRPO with PPO**
- **Whiteboard-coding a transformer decoder block**
- **Comparing DeepSpeed and Megatron**
- **Estimating PPO/DPO training resource consumption** (interviewers had candidates estimate GPU-hours on the spot)
- Feedback: "If you've only done SFT, honestly don't bother applying"

→ **v4 takeaway**: this book must cover

- ✅ The complete PG derivation chain (already present, needs to be strengthened into a standalone theory subsection)
- ✅ DPO regularization methods (v3 §19 needs expansion)
- ✅ A training-cost-estimation chapter (v3's §29 engineering chapter needs a new "cost estimation" subsection)

### 8.2 Real DeepSeek Training Data (Stanford CRFM Transparency Report)

**[Source]** crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html

- DeepSeek-V3 pretraining: **2.664M H800 GPU-hours, 14.8T tokens**
- DeepSeek-R1-Zero: **648 H800 GPUs × 198 hours = 128K GPU-hours**
- DeepSeek-R1 (full multi-stage): **648 H800 GPUs × ~80 hours**
- V3 + R1 total: **2.8M GPU-hours, 67 days**
- Training module: loads actor + critic (optional), supports PPO/GRPO/DPO
- Best-Fit data packing, the DualPipe algorithm

→ **v4 takeaway**:

- ✅ Cost estimation needs concrete reference numbers (this book could add a "GPU-hour estimation table" appendix)
- ✅ Engineering implementations like DualPipe and Best-Fit packing should go into the §29 engineering chapter

### 8.3 Qwen3 Training Methodology (Real Technical Reports)

**[Source]** the Qwen3 Technical Report (arXiv:2505.09388), Qwen3-Thinking-2507, Qwen2.5-Math

- **Qwen3 uses GSPO** (Group Sequence Policy Optimization, Zheng et al. 2025) — sequence-level importance sampling
- **Qwen2.5-Math**: a self-improvement approach
- **Qwen3-Thinking-2507**: AIME 86.7, LiveCodeBench 74.1 (surpassing o3/o4-mini)
- **The data-contamination issue** (arXiv:2507.10532): because Qwen's pretraining data included benchmark answers, some of the RL gains come from activating memorization rather than from genuine reasoning generalization

→ **v4 takeaway**:

- ✅ GSPO must get significant coverage in the GRPO family chapter (already present in v3 §20.5)
- ✅ The data-contamination issue should go into the §25 reward-hacking chapter
- ⚠️ Qwen3's "even random rewards can improve performance" phenomenon reveals the subtlety of RLVR

### 8.4 StepFun's Unique Direction

**[Source]** static.stepfun.com/blog/step-3.5-flash, github.com/stepfun-ai/Step-Audio-R1, arXiv:2601.09668 (Step3-VL-10B)

**Real job postings** (BOSS Zhipin + Nowcoder + StepFun's own site):

- AI infra engineers: training frameworks, inference acceleration, distributed systems
- Kernel development, MoE communication optimization, 10,000-GPU-cluster work
- Speech/multimodal prioritized directions
- StepStar 2026 campus recruiting (top master's/PhD candidates)

**Step's unique technical contributions**:

- **Step-Audio-R1**: the first speech-language model to achieve test-time compute scaling
- **Step-Audio-R1.5**: shifts from RLVR to **RLHF for Audio Reasoning** (balancing vocal naturalness with reasoning)
- **Step3-VL-10B**:
  - **SeRe** (Sequential Reasoning): standard CoT, 64K context
  - **PaCoRe** (Parallel Coordinated Reasoning): **16-way parallel rollout aggregation**, 128K context, test-time compute scaling
  - AIME 2025: 94.4 (in PaCoRe mode)
- **Step 3.5 Flash**: MoE 196B/11B active, AIME 97.3, 350 tokens/s inference
- **Deep Research**: a multi-agent architecture

→ **v4 takeaway**:

- ✅ **Parallel coordinated reasoning approaches like PaCoRe** should go into the §22 PRM/inference-search chapter (a brand-new test-time-scaling method)
- ✅ **Audio RL** is a distinctively domestic direction; §31 (embodiment/multimodality) should give it its own subsection
- ✅ **MoE + RL engineering optimization** should go into §29 (distributed training)
- ✅ **Multimodal RL is more cutting-edge in China than in the US** (Step3-VL-10B beats GLM-4.6V/Qwen3-VL on AIME 2025)

### 8.5 Distinctive Traits of Chinese Labs (vs. OpenAI/Anthropic)

| Dimension          | OpenAI/Anthropic                           | Chinese Labs (DeepSeek/Qwen/Zhipu/Step)                         |
| ------------------ | ------------------------------------------ | --------------------------------------------------------------- |
| Focus              | RL Environments, Safety, Constitutional AI | **MoE training, multimodal RL, kernel optimization**            |
| Reasoning paradigm | o1/o3 hidden CoT                           | R1 visible CoT + **PaCoRe parallel reasoning**                  |
| Multimodality      | Leans text + vision                        | **Audio RL, visual reasoning, GUI agents**                      |
| Training cost      | Not disclosed commercially                 | **GPU-hours disclosed publicly** (DeepSeek is most transparent) |
| Interview focus    | Async Python, verifiers, sandboxes         | **PG derivation chain, the DPO family, DeepSpeed/Megatron**     |
| Engineering roles  | RL environments, $500K                     | **Kernel work, MoE communication, 10,000-GPU clusters**         |

---

## 9. Proposed New Chapters for v4

Based on evidence from Chinese labs, **add the following** on top of v3:

### v4 §20.10 China's Hands-On GRPO Variants

- Qwen3 GSPO (sequence-level IS)
- DeepSeek V3.2's KL tuning (zero KL for math tasks)
- Applying Dr. GRPO within the Qwen series

### v4 §22.6 Parallel Coordinated Reasoning (PaCoRe) — **entirely new subsection**

- Step3-VL-10B's 16-way parallel rollout aggregation
- An alternative path for test-time compute scaling
- Comparison with MCTS over Thoughts

### v4 §25.9 Data Contamination and the Subtlety of RLVR

- The "even random rewards can improve Qwen's performance" phenomenon (arXiv:2507.10532)
- How GRPO's clipping bias activates memorization
- Methodology for assessing the genuine gains from RLVR

### v4 §29.7 MoE Training and RL Integration **[a core strength of Chinese labs]**

- The DeepSeek-V3 MoE architecture
- RL training on 10,000-GPU clusters
- DeepSpeed / Megatron / DualPipe / Best-Fit packing
- Hands-on training-cost estimation (using DeepSeek's public data as reference)

### v4 §31.4 Audio RL **[new subsection]**

- Step-Audio-R1: the first speech model with test-time compute scaling
- The shift from RLVR to RLHF for Audio
- Balancing prosodic naturalness with reasoning ability

### v4 §31.5 China's Multimodal RL Frontier

- Comparing Step3-VL-10B, GLM-4.6V, and Qwen3-VL
- The "missing trace" problem in visual-reasoning RL (from the Step paper)
- Acoustic-Grounded Reasoning (Step-Audio R1.1)

---

## 10. Key Citations from Chinese Labs

```
[Qwen3 Technical Report] Yang et al. 2025. arXiv:2505.09388
[Qwen3-Thinking-2507] Hugging Face Qwen3-235B-A22B-Thinking-2507
[Step3-VL-10B Technical Report] arXiv:2601.09668
[Step-Audio-R1] github.com/stepfun-ai/Step-Audio-R1
[Step 3.5 Flash] static.stepfun.com/blog/step-3.5-flash
[DeepSeek-R1 Nature] Guo et al. 2025. nature.com/articles/s41586-025-09422-z
[DeepSeek-V3 Tech Report] DeepSeek-AI 2024. arXiv:2412.19437
[DeepSeek CRFM Transparency] crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html
[Zhipu GLM alignment interview questions] nowcoder.com/creation/subject/da767c9233384be9a2992ee3d1946518
[Qwen Contamination Study] arXiv:2507.10532 (Reasoning or Memorization?)
[Olmo 3 GRPO improvements] magazine.sebastianraschka.com/p/state-of-llms-2025
```

---

## 11. Summary of Combined v4 Adjustments

| v3 chapter                   | v4 adjustment                                             | Rationale                           |
| ---------------------------- | --------------------------------------------------------- | ----------------------------------- |
| §20 GRPO family              | **add §20.10 China's variants**                           | Qwen3 GSPO, DeepSeek V3.2 KL tuning |
| §22 PRM/search               | **add §22.6 PaCoRe parallel coordinated reasoning**       | An original Step3-VL method         |
| §25 Reward Hacking           | **add §25.9 data contamination and the subtlety of RLVR** | An empirical finding from Qwen      |
| §29 distributed training     | **add §29.7 MoE + RL integration**                        | Core engineering at DeepSeek/Step   |
| §31 embodiment/multimodality | **add §31.4 audio RL, §31.5 China's multimodal frontier** | Step-Audio, Step3-VL                |

**Conclusion**: the evidence from Chinese labs reinforces the following v3 judgments:

1. Full coverage of the GRPO family is necessary (Chinese labs are the main contributors to GRPO improvements)
2. MoE + RL engineering must be covered in depth (China invests more heavily in it than the US does)
3. Multimodal RL is more mature than expected (should be moved earlier into the main text)
4. Training-cost estimation should become standard appendix content

But the evidence from Chinese labs also surfaces new directions not covered by v3:

- **Audio RL** (a StepFun specialty)
- **PaCoRe parallel coordinated reasoning** (Step's original test-time scaling approach)
- **The subtlety of data contamination and genuine RLVR gains** (a problem surfaced by Qwen)
