# v5 Outline Gap Analysis (2026.06, Four-Team Research Roundup)

> Based on real-URL research findings from four independent sub-agents: OpenAI/Anthropic, DeepMind/Meta, DeepSeek/Qwen/Kimi, and Zhipu/StepFun/ByteDance/MiniMax.

---

## 1. Very-High-Confidence Gaps Cross-Validated by All Four Agents

### Gap 1 · Completing the GRPO Improvement Family (P0, all four agree)

**v5 status**: Chapter 19's GRPO family currently only names the seven variants GRPO, DAPO, Dr.GRPO, GSPO, CISPO, REINFORCE++, and AREAL, with no comparison of algorithmic detail.

**Evidence from all four teams**:

| Agent                 | Finding                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| #1 OpenAI/Anthropic   | Name DAPO explicitly; add the four techniques Clip-Higher/Dynamic Sampling/Token-level Loss/Overlong Filtering                |
| #2 DeepMind/Meta      | Dr.GRPO removes std normalization and length normalization to avoid reward hacking                                            |
| #3 DeepSeek/Qwen/Kimi | GSPO's sequence-level IS + clipping is the cornerstone of the entire Qwen3 line; deepen coverage of DAPO's engineering tricks |
| #4 China-lab group    | Full lineage of CISPO/DAPO/VAPO/GSPO/ASPO; missing **VAPO** (ByteDance Seed's value-based counter-trend approach)             |

**Revision proposal**: Restructure Chapter 19 as:

- 19.1 The principle of GRPO's group normalization
- 19.2 Two directions for correcting GRPO
  - 19.2.1 Removing normalization bias: Dr.GRPO
  - 19.2.2 Sequence-level IS: GSPO (Qwen3)
  - 19.2.3 Comparing clipping strategies: Clip-Higher (DAPO) vs. Clip IS-weight (CISPO, MiniMax)
  - 19.2.4 The value-based counter-trend: VAPO (ByteDance Seed)
- 19.3 A collection of engineering tricks
  - 19.3.1 Dynamic Sampling (filtering out all-correct/all-wrong groups)
  - 19.3.2 Token-level Loss (avoiding domination by long responses)
  - 19.3.3 Overlong Filtering + Soft Shaping
  - 19.3.4 No KL / precision alignment (CISPO lightning attention)
- 19.4 The RLVR paradigm
- 19.5 Hands-on comparison of Chinese-lab variants
- 19.6 A decision tree for choosing a variant

---

### Gap 2 · Upgrading PRM: Generative + Formal (P0, three teams agree)

**v5 status**: Chapter 21's PRM coverage still centers on the discriminative OpenAI "Let's Verify Step by Step" approach.

**Evidence from three teams**:

| Agent | Finding                                                                                                                      |
| ----- | ---------------------------------------------------------------------------------------------------------------------------- |
| #1    | Generative PRM (ThinkPRM) outperforms discriminative PRM with 100x fewer labels; scaling verifier compute is a new dimension |
| #2    | AlphaProof + AlphaGeometry 2 self-train with the Lean formal language + AlphaZero MCTS, winning an IMO silver medal          |
| #3    | DeepSeek-Prover-V2 does formal theorem-proving RL in Lean4, reaching 88.9% on MiniF2F                                        |

**Revision proposal**: Restructure Chapter 21 as:

- 21.1 Outcome reward vs. process reward
- 21.2 Discriminative PRM: OpenAI's "Let's Verify Step by Step"
- 21.3 Generative PRM: ThinkPRM (arXiv:2504.16828)
- 21.4 Formal PRM: Lean4/Coq as a natural verifier
  - AlphaProof (DeepMind, IMO silver medal)
  - DeepSeek-Prover-V2 (88.9% on MiniF2F)
- 21.5 Inference-time search: beam search, MCTS over thoughts, tree of thoughts
- 21.6 PaCoRe: parallel coordinated reasoning (a Step3-VL original)
- 21.7 GenRM and verifier models

---

### Gap 3 · Reward Hacking / Evaluation Pitfalls (P0, three teams agree)

**v5 status**: Chapter 32 already covers Anthropic's 2025.11 emergent misalignment work, but is missing data contamination and the GPT-4o rollback case.

**Evidence from three teams**:

| Agent | Finding                                                                                                  |
| ----- | -------------------------------------------------------------------------------------------------------- |
| #1    | The GPT-4o sycophancy rollback (2025.04-05, a real RLHF-failure case plus root-cause analysis)           |
| #2    | Anthropic's 2025.11 emergent misalignment work + HHH mitigation                                          |
| #3    | Data contamination and RLVR evaluation pitfalls (arXiv:2507.10532, Qwen's "spurious reward" on MATH-500) |
| #4    | ByteDance Seed's RLHF data scaling: reward hacking + diversity decay                                     |

**Revision proposal**: Add new subsections to Chapter 32:

- 32.10 RLVR's "spurious gains": empirical evidence of data contamination (the Qwen case)
- 32.11 An industry failure case: root-cause analysis of the GPT-4o sycophancy rollback
- 32.12 Diversity decay in ByteDance Seed's RLHF scaling

---

### Gap 4 · Agentic RL Engineering Infrastructure (P0, three teams agree)

**v5 status**: Chapter 23, RL Environments, mentions Anthropic's Effective Harnesses but doesn't cover asynchronous RL training systems.

**Evidence from three teams**:

| Agent | Finding                                                                                                                     |
| ----- | --------------------------------------------------------------------------------------------------------------------------- |
| #1    | Anthropic's multi-agent research system (orchestrator-worker + OODA + independent context, 90.2% speedup)                   |
| #3    | Agentic rollout infrastructure (Qwen3-Coder's 20,000 envs, Kimi K2's agentic rollout infra)                                 |
| #4    | Four open-sourced asynchronous systems: AReaL (Tsinghua + Zhipu, 2.77x speedup) + AgentRL (Zhipu + Tsinghua) + SLIME + ROLL |

**Revision proposal**: Add to both Chapter 23 and Appendix B:

- 23.9 Synchronous vs. asynchronous RL training
- 23.10 Open-source comparison of asynchronous systems: AReaL / AgentRL / SLIME / ROLL / LlamaRL
- Appendix B.2 Decoupling asynchronous rollout from training: staleness, cross-policy sampling

---

## 2. Entirely New Gaps (Not Covered Anywhere in v5)

### Gap 5 · Reinforcement Pre-Training (RPT) (P0, Agent #1)

**Gap**: Reframes next-token prediction as a reasoning task with an intrinsic binary reward, matching the scalability of pretraining. Challenges the pretraining/post-training dichotomy — the biggest conceptual shift of 2025.

**Source**: arXiv:2506.08007 (Microsoft, 2025.06)

**Proposed placement**: Add "RL at the Pretraining Stage" at the end of Chapter 19, or in a Part VIII frontier chapter

---

### Gap 6 · The AlphaEvolve Paradigm (P0, Agent #2)

**Gap**: LLM proposes diffs, an automated evaluator scores them, and an evolutionary algorithm selects among them. DeepMind (2025.05) found a 23% speedup in matrix multiplication and improved 50+ open math problems. This belongs to neither classic RL nor SFT — it's a new search paradigm for the LLM era.

**Source**: deepmind.google/blog/alphaevolve + the paper PDF

**Proposed placement**: Add "Evolutionary LLM Search" as a new Part VIII frontier chapter

---

### Gap 7 · Instruction Hierarchy (P0, Agent #1)

**Gap**: The "kernel mode" analogy for agent safety. Proposed by OpenAI in 2024.04; using it as an RL reward for GPT-5 Mini-R yielded a +0.11~0.21 improvement. A core safety mechanism for the agent era.

**Source**: openai.com/index/instruction-hierarchy-challenge, arXiv:2404.13208

**Proposed placement**: Add a new subsection to Chapter 28 (Computer Use) or the Chapter 32 safety material

---

### Gap 8 · VAPO, the Value-Based Counter-Trend (P0, Agent #4)

**Gap**: ByteDance Seed's 2025.04 Value-based Augmented PPO, where a value model once again beats GRPO in long-CoT settings, reaching AIME 60.4. This is the **counter-trend** direction within the GRPO family (critic-free → critic-resurrected).

**Source**: arXiv:2504.05118

**Proposed placement**: Chapter 19's GRPO family (see Gap 1)

---

### Gap 9 · RL for Video Generation (P0, Agent #4)

**Gap**: A direction unique to ByteDance Seed. DanceGRPO adapts GRPO to diffusion/flow-based visual generation; Seedance uses multi-dimensional RLHF (Foundational+Motion+Aesthetic); LongCat-Video combines GRPO with multiple rewards. Completely absent from v5.

**Sources**:

- DanceGRPO arXiv:2505.07818
- Seedance arXiv:2506.09113
- LongCat-Video arXiv:2510.22200

**Proposed placement**: Add a new Chapter 32 "RL for Visual Generation" under Part VI Multimodal (shifting the original Part VI chapters back)

---

### Gap 10 · Audio RL (P0, Agent #4 + Agent #3 already covers it but not deeply enough)

**Gap**: Step-Audio-R1's MGRD (Modality-Grounded Reasoning Distillation), Acoustic-Grounded Reasoning, and the RLVR→RLHF evolution. v5's Chapter 30 is only outline bullet points.

**Source**: arXiv:2511.15848

**Proposed placement**: Expand Part VI Chapter 30 into a full standalone chapter

---

### Gap 11 · GLM Series Training Paradigms (P1, Agent #4)

**Gap**: **v5 has zero Zhipu citations in the entire book**. GLM-4.5 ARC (arXiv:2508.06471, MoE 355B/32A, difficulty-curriculum RL), GLM-4.6 asynchronous RL + RLCS, GLM-5 (arXiv:2602.15763).

**Sources**:

- GLM-4.5 arXiv:2508.06471
- GLM-4.6 HuggingFace zai-org/GLM-4.6
- GLM-5 arXiv:2602.15763
- AReaL arXiv:2505.24298

**Proposed placement**: Add GLM case studies to Chapter 17's industrial post-training practice

---

### Gap 12 · Kimi K2 / K2.5 / Agent Swarm (P1, Agent #3)

**Gap**: The MuonClip optimizer + RL stability, K2 Thinking's 200-300-step tool calling, K2.5's Agent Swarm — parallel multi-agent + trainable orchestrator + parallel-agent RL.

**Sources**:

- K2 arXiv:2507.20534
- K2.5 arXiv:2602.02276, kimi.com/blog/kimi-k2-5

**Proposed placement**: Chapter 28's multi-agent collaboration (see Gap 4 for detail)

---

### Gap 13 · Hybrid Thinking + Thinking Budget (P0, Agent #3)

**Gap**: DeepSeek V3.1 and Qwen3 support think/non-think modes in a single model; thinking budget controls reasoning depth; NoThinking + best-of-N can match thinking-mode performance (Ma et al., arXiv:2505.18681).

**Sources**:

- Qwen3 report §4.3
- DeepSeek V3.1, api-docs.deepseek.com/updates
- Ma et al., arXiv:2505.18681

**Proposed placement**: Add a "Thinking Mode Fusion" subsection to Chapter 20, Reasoning Models

---

### Gap 14 · Kimi's long2short RL (P0, Agent #3)

**Gap**: An RL method for distilling long CoT into short CoT, controlled via a length penalty.

**Source**: arXiv:2501.12599 §2.4, §3.4

**Proposed placement**: Add a "Long-CoT Compression" subsection to Chapter 20, Reasoning

---

### Gap 15 · Self-Play SWE-RL (SSR) (P0, Agent #2)

**Gap**: Meta's 2025.12 single-policy dual-role approach (bug-injector + bug-solver), requiring no human-written issues, +10.4 on SWE-bench Verified.

**Source**: arXiv:2512.18552

**Proposed placement**: Add a "Self-Play Paradigm" subsection to Chapter 26, Code Agents

---

### Gap 16 · Code World Model (CWM) + DeepSWE (P1, Agent #2)

**Gap**: Meta's CWM (32B dense, Python execution-trace mid-training + GRPO post-training, 65.8% on SWE-bench). Represents the mainline shift of code agents from SFT-only to RL-based.

**Source**: arXiv:2510.02387

**Proposed placement**: Chapter 26, Code Agents

---

### Gap 17 · Generative vs. Discriminative PRM (P1, Agent #1)

**Already merged into Gap 2**

---

### Gap 18 · The GPT-4o Sycophancy Rollback (P1, Agent #1)

**Already merged into Gap 3**

---

### Gap 19 · Anthropic's Multi-Agent Research System (P1, Agent #1)

**Already merged into Gap 4**

---

### Gap 20 · Genie 3, a Generative World Model as an RL Environment (P1, Agent #2)

**Gap**: DeepMind's 2025.08 real-time interactive world model, 720p/24fps, positioned as the foundation for a general AGI world model, offering an unlimited RL training curriculum.

**Source**: deepmind.google/blog/genie-3

**Proposed placement**: Add "Generative World Models" to Part III Chapter 15 (Exploration/Hierarchical RL), or to a Part VIII frontier chapter

---

### Gap 21 · The Llama 4 Pipeline (P1, Agent #2)

**Gap**: Lightweight SFT → online RL → lightweight DPO; Behemoth prunes 95% of SFT data; RL uses pass@k difficulty filtering.

**Source**: ai.meta.com/blog/llama-4

**Proposed placement**: Chapter 17, industrial post-training practice

---

### Gap 22 · A Competitive-Programming Paper as Evidence of Reasoning Emergence (P1, Agent #1)

**Gap**: o3's complex test-time strategies emerge naturally from end-to-end RL, not from hand-design.

**Source**: arXiv:2502.06807 (OpenAI, 2025.02)

**Proposed placement**: Chapter 20, Reasoning, §20.1 or §20.4

---

### Gap 23 · Titans + MIRAS Long-Term Memory (P2, Agent #2)

**Source**: research.google/blog/titans-miras-helping-ai-have-long-term-memory

**Proposed placement**: Part VIII frontier

---

### Gap 24 · DeepSeek V3.2 / Speciale (P2, Agent #2 + #3)

**Gap**: DSA sparse attention + self-verification/self-refinement RLVR + mHC residual stability, 97% on AIME25.

**Sources**: arXiv:2512.02556, magazine.sebastianraschka.com/p/technical-deepseek

**Proposed placement**: Chapter 19's GRPO family, as the latest case study

---

### Gap 25 · Market-Economics Data on RL Environments (P2, Agent #1)

**Source**: wing.vc/.../rl-environments-for-agentic-ai

**Proposed placement**: Chapter 23 §23.1

---

### Gap 26 · Recursive Self-Improvement / Anthropic-Funded Research (P2, Agent #1)

**Source**: anthropic.com/institute/recursive-self-improvement

**Proposed placement**: Part VIII frontier, or the preface's "glimpse of what's ahead"

---

### Gap 27 · The MuonClip Optimizer (P1, Agent #3)

**Gap**: Kimi K2's MuonClip optimizer + QK-clip for RL training stability.

**Source**: arXiv:2507.20534 §3.2

**Proposed placement**: Part II Chapter 11 (PPO training details) or Appendix A (debugging handbook)

---

### Gap 28 · UI-TARS-2 + AutoGLM GUI Agent RL (P1, Agent #4)

**Sources**:

- UI-TARS-2 arXiv:2509.02544
- AutoGLM xiao9905.github.io/AutoGLM

**Proposed placement**: Chapter 28, Computer Use

---

### Gap 29 · MoE + RL Training Engineering (P1, Agent #4)

**Source**: Step 3.5 Flash arXiv:2602.10604, GLM-4.5

**Proposed placement**: Appendix B, the industrial training chapter

---

### Gap 30 · Seed-Thinking's Dual-Track Reward / Pre-PPO (P1, Agent #4)

**Source**: arXiv:2504.13914, seed.bytedance.com RLHF scaling

**Proposed placement**: Chapter 16 (RLHF pipeline) or Chapter 17 (industrial practice)

---

## 3. Outdated-Content Warnings (Must Update)

### Outdated 1 · Part IV Centers on PPO (Agent #2)

**Problem**: Llama 4 / Qwen3 / DeepSeek V3.2 have all fully shifted to GRPO/Dr.GRPO + RLVR.

**Revision**: Shift Part IV's focus from "the classic PPO implementation" to "the GRPO family as the main line, with PPO as historical background."

### Outdated 2 · The Code Chapter in Part V Uses an SFT-Only SWE-Agent (Agent #2)

**Problem**: The main line should be RL-based SWE (SWE-RL / CWM / DeepSWE / SSR).

**Revision**: Restructure Chapter 26 to treat the SFT-based approach as background and the RL-based approach as the main line.

### Outdated 3 · Part VI VLA Uses RT-2 (Agent #2)

**Problem**: The new benchmarks are Gemini Robotics 1.5 / π0 / OpenVLA.

**Revision**: In Chapter 31 (VLA), demote RT-2 to background material and use Gemini Robotics 1.5 + Embodied Thinking as the new flagship case study.

### Outdated 4 · The R1-Zero Paradigm Covers Only DeepSeek (Agent #4)

**Problem**: Should add DAPO (ByteDance) and VAPO (ByteDance Seed) as industrial-grade open-source counterparts to the R1-Zero approach.

**Revision**: Expand Chapter 20's R1-Zero paradigm section into a comparison across multiple open-source implementations.

### Outdated 5 · Appendix B on Training Systems Covers Only Synchronous veRL (Agent #4)

**Problem**: Four major asynchronous systems — AReaL / AgentRL / SLIME / ROLL — are all now open source.

**Revision**: Expand Appendix B.1 with a synchronous-vs-asynchronous comparison.

---

## 4. Revision Priority Matrix

| Priority | Gap #          | Summary                                                                | Effort     |
| -------- | -------------- | ---------------------------------------------------------------------- | ---------- |
| **P0**   | 1, 8           | GRPO family restructure (incl. VAPO)                                   | Large      |
| **P0**   | 2              | PRM upgrade (generative + formal)                                      | Medium     |
| **P0**   | 3              | Reward hacking / evaluation pitfalls (add GPT-4o + data contamination) | Medium     |
| **P0**   | 4              | Agentic RL engineering infrastructure (async systems)                  | Medium     |
| **P0**   | 5              | Reinforcement Pre-Training (RPT)                                       | Small      |
| **P0**   | 6              | The AlphaEvolve paradigm                                               | Medium     |
| **P0**   | 7              | Instruction Hierarchy                                                  | Small      |
| **P0**   | 9              | RL for video generation (DanceGRPO/Seedance)                           | Large      |
| **P0**   | 10             | Deepen audio RL (Step-Audio MGRD)                                      | Medium     |
| **P0**   | 13, 14         | Hybrid Thinking + long2short                                           | Medium     |
| **P0**   | 15             | Self-Play SWE-RL (SSR)                                                 | Medium     |
| **P1**   | 11             | GLM series training paradigms                                          | Medium     |
| **P1**   | 12             | Kimi K2/K2.5 + Agent Swarm                                             | Medium     |
| **P1**   | 16             | Code World Model + DeepSWE                                             | Medium     |
| **P1**   | 20             | Genie 3 generative world model                                         | Medium     |
| **P1**   | 21             | The Llama 4 pipeline                                                   | Small      |
| **P1**   | 22             | Competitive-programming reasoning emergence                            | Small      |
| **P1**   | 27             | The MuonClip optimizer                                                 | Small      |
| **P1**   | 28             | UI-TARS-2 + AutoGLM                                                    | Medium     |
| **P1**   | 29             | MoE + RL training engineering                                          | Medium     |
| **P1**   | 30             | Seed-Thinking dual-track reward                                        | Small      |
| **P2**   | 23, 24, 25, 26 | Titans/V3.2/market data/recursive improvement                          | Small each |

---

## 5. Proposed Chapter Additions/Changes for the v5.1 Revision

**New chapters**:

- Chapter 19 restructure: complete the GRPO family (including VAPO, Dr.GRPO, GSPO, CISPO, and the DAPO lineage)
- Chapter 21 restructure: upgrade PRM (generative ThinkPRM + formal Lean4)
- New Chapter 20 subsection: Hybrid Thinking + long2short + evidence of reasoning emergence
- New addition to Chapter 23: synchronous vs. asynchronous RL systems + economics data
- Chapter 26 restructure: RL-based SWE as the main line (SWE-RL/CWM/DeepSWE/SSR)
- New addition to Chapter 28: Instruction Hierarchy + UI-TARS-2 + AutoGLM
- Chapter 31 restructure: upgrade VLA to Gemini Robotics 1.5 + π0
- New addition to Chapter 32: data contamination + the GPT-4o rollback + Seed RLHF scaling
- **New Chapter 32 in Part VI**: RL for visual generation (DanceGRPO/Seedance)
- New additions to Part VIII: AlphaEvolve + Genie 3 + Titans + recursive self-improvement

**Appendix expansions**:

- Appendix A debugging handbook: add MuonClip + QK-clip
- Appendix B engineering practice: add the AReaL/AgentRL/SLIME/ROLL asynchronous systems + MoE+RL engineering

**Chapter-count change**:

- Original v5 plan: 36 chapters
- After the v5.1 revision: roughly 37-38 chapters (adding a standalone "RL for Visual Generation" chapter)

---

## 6. Summary of Key Findings

1. **The GRPO family is the biggest algorithmic focus of 2025-2026**: all four independent research teams named it, with 5+ mainstream variants each contributing something new — Chapter 19 in v5 must be restructured.
2. **Chinese labs lead the world in RL engineering**: asynchronous training systems (AReaL/AgentRL), MoE+RL engineering, RL for video generation, and audio RL all debuted first in China.
3. **Several "representative case studies" in v5 are now outdated**: the PPO main line, the SFT-only SWE-agent, and the RT-2 VLA case must all be updated.
4. **Formal verification is the next stop for PRM**: AlphaProof + DeepSeek-Prover-V2 use Lean4 as a natural verifier.
5. **The boundary between pretraining and post-training is dissolving**: Reinforcement Pre-Training is the biggest conceptual shift of 2025.
6. **Reward hacking research has matured**: the GPT-4o sycophancy rollback, Anthropic's 2025.11 work, and Qwen's data contamination together form a complete chain of evidence.
7. **The entire v5 book has zero Zhipu citations**: it must add the GLM-4.5/4.6/5 series, which represents real engineering practice from a Chinese alignment team.
