# 17.1 Emergence of Reasoning Models

On September 12, 2024, OpenAI released o1 — the first production-grade large model explicitly labeled a "reasoning model." On hard tasks with **unambiguous correct answers** — IOI (International Olympiad in Informatics), Codeforces, GPQA Diamond, AIME — it far outperformed every non-reasoning model of its era. The release of o1 was not just a product event. It marked **a paradigm shift**: from "the model produces an answer in a single forward pass" to "the model thinks explicitly, then gives an answer."

## 10.1.1 The Evolution from o1 to o3 to o4

### Proving the Paradigm Works

o1's core innovation is a shift in **training paradigm** — the model architecture itself is unchanged; it is still a Transformer decoder. OpenAI's official blog post states plainly that o1 uses RL to "reinforce the model's chain of thought." During training, the model is encouraged to generate long CoT, then rewarded based on whether the final answer is correct. This differs from traditional RLHF, which uses human preference as the reward signal: o1's reward is **the task outcome itself** — a correct answer scores, an incorrect one does not.

When o1 launched, several concrete results shook the industry:

- **AIME 2024**: a jump from GPT-4o's roughly 12% to o1's 83% (pass@1, single sample)
- **Codeforces**: o1 reached a rating of 1673, roughly the top 10% of human competitors
- **GPQA Diamond**: 78% (PhD-level science QA), far above GPT-4o's 53%

But o1 also shipped with an engineering choice that frustrated users: **its CoT is hidden**. Users see only the final answer, never the reasoning that produced it. OpenAI's stated reasons were safety, compliance, and preventing users from distilling the model — but this decision cast doubt on o1's trustworthiness. If a reasoning model cannot show its own reasoning, how does a user know it isn't just making things up?

### The First Breakthrough on ARC-AGI

In December 2024, OpenAI released o3 during its "12 Days of OpenAI" event (note: the o2 designation was skipped). o3's key breakthroughs were:

- **ARC-AGI-2**: o3 was the first model to exceed 75% on ARC-AGI-2 (GPT-4o scored only about 5%)
- **SWE-bench Verified**: 71.7% (a software engineering benchmark)
- **Codeforces rating**: 2727 (roughly the top 0.1% of human competitors)

o3 also introduced a new engineering parameter: **reasoning effort**. Users could specify `reasoning_effort` (low/medium/high) in the API to control how many tokens the model spends thinking. This was the prototype for the "thinking budget" feature that came later.

### Extending Reasoning to Tool Calling

In April 2025, OpenAI released o4-mini along with a full upgrade to o3 (the community sometimes calls this o4). o4's defining feature is **combining reasoning with tool calling**:

- During reasoning, the model can proactively call tools (search, code execution, image analysis)
- The tool's results feed back and shape the next step of reasoning
- This forms a loop of "think → call a tool → think again → call another tool"

This paradigm later came to be called **agentic reasoning**: reasoning becomes a multi-step process that interacts with the external environment, rather than a single-pass CoT. This connects deeply with the material in [Chapter 10, Agentic RL](../chapter22_agentic/intro).

## 10.1.2 The Competitive Programming Paper and Key Evidence for Emergence

After o1 shipped, the community's biggest question was: **how did OpenAI train this reasoning ability?** Did it depend on massive amounts of human-annotated CoT data?

In February 2025, OpenAI published a key paper, [Competitive Programming with Large Reasoning Models](https://arxiv.org/abs/2502.06807) (arXiv:2502.06807). The paper doesn't disclose training details, but it offers three key findings.

### End-to-End RL Beats Domain-Specific Pipelines

Traditionally, the SOTA pipeline for competitive tasks like Codeforces looked like this:

```text
Problem → compiler + test case generation → program synthesis → select the best solution
```

Every step in this pipeline is purpose-built — domain-specific search algorithms, specialized program synthesizers. OpenAI found that **general-purpose reasoning models trained end-to-end with RL, like o1 and o3, beat these specialized pipelines on Codeforces**.

This result challenges the common wisdom that specific tasks call for specific methods. The implication: **general reasoning ability can surpass specialized optimization**, given sufficient RL training.

### Complex Test-Time Reasoning Emerges Naturally from RL

The paper reports an observation that stunned the community: while solving Codeforces problems, o1 and o3 exhibit **complex, multi-stage reasoning**, including:

- **Generating multiple candidate solutions**: the model produces several solutions on its own and compares their merits
- **Execution verification**: using a tool to run the code and check whether the result matches expectations
- **Self-correction**: rederiving the solution after spotting an error
- **Strategy switching**: moving from a greedy algorithm to dynamic programming, then to backtracking

These behaviors were **never hand-designed** — OpenAI never labeled its training data with instructions like "try greedy first, then try DP." They **emerged naturally** from RL training.

The significance: **as long as the reward signal is clear enough — a correct answer scores — RL lets the model discover complex problem-solving strategies on its own**. This lines up exactly with R1-Zero's "aha moment": emergent behavior is an intrinsic property of RL training, not a coincidence.

### The Trade-off Between Test-Time and Train-Time Compute

The paper also reports a key trade-off:

- **Increasing training compute**: the model's baseline ability improves, but the number of reasoning tokens spent per problem stays roughly the same
- **Increasing inference compute** (letting the model think longer): for a fixed amount of training compute, this can push performance further

This finding is the core argument behind **Test-time Compute Scaling** — the subject of the next section.

## 10.1.3 Reasoning Ability: Emergence vs. Activation

Both o1 and R1-Zero reported "emergent reasoning behavior." Strictly speaking, though, the word "emergence" carries two distinct meanings that need to be separated.

**Meaning one: the reasoning behavior never appeared explicitly in the training data**

R1-Zero never saw any human-annotated CoT, yet after training it autonomously generates long chains of thought, reflects on its own reasoning, and verifies its answers. This kind of "emergence" is defined relative to the training data — the model was never taught to do this.

**Meaning two: reasoning ability is "created from scratch" during the RL stage**

This meaning **does not hold**. Follow-up research (SimpleRL-Zoo, Open-R1) found that the base model already possesses latent reasoning ability — give it a clear task and enough sampling attempts, and it can produce a correct reasoning process on its own. What RL does is **activate and reinforce reasoning ability that already exists**, not create reasoning from nothing.

This distinction matters in practice:

- If you believe RL creates reasoning ability, you will pour large amounts of compute into large-scale RL training.
- If you believe RL activates reasoning ability, you will first check the base model's latent ceiling — a small amount of RL combined with a good base model may already get you close to SOTA.

The DeepSeek team later stated this explicitly in the R1 paper: **"reasoning ability is primarily conferred during pretraining; RL simply organizes it."** This view is now industry consensus.

## 10.1.4 The Industrial Landscape of Reasoning Models

By mid-2026, reasoning models had become industrial-grade products. The main players:

| Model                           | Vendor    | Key Features                                      |
| ------------------------------- | --------- | ------------------------------------------------- |
| **o1 / o3 / o4**                | OpenAI    | Hidden CoT, `reasoning_effort` parameter          |
| **DeepSeek-R1 / V3.2 Speciale** | DeepSeek  | Visible CoT, pure-RL approach, open source        |
| **Claude Opus 4.6 / 4.7**       | Anthropic | Adaptive thinking, Extended Thinking API          |
| **Gemini 3 Pro Deep Think**     | Google    | Parallel-reasoning "thinking layer," long context |
| **Qwen3 Thinking series**       | Alibaba   | Hybrid Thinking, Thinking Budget                  |
| **Kimi k1.5 / K2.5**            | Moonshot  | long2short RL, Thinking Budget                    |
| **GLM-Zero / GLM-4.6**          | Zhipu     | Reasoning + tool calling                          |
| **MiniMax M1**                  | MiniMax   | CISPO + Lightning Attention                       |

Note a key distinction:

- **The Hidden CoT approach**: OpenAI's o-series. The model thinks but doesn't show its thinking. Advantage: users can't see the reasoning, so it can't be distilled. Disadvantage: poor interpretability and low user trust.
- **The Visible CoT approach**: DeepSeek, Qwen, Kimi, Anthropic, Google. The model thinks and shows its thinking. Advantage: interpretable and fine-tunable. Disadvantage: users can copy the CoT data.

This split isn't just a product decision — it touches deeper questions of **alignment and safety**, which Section 10.5 develops further.

## 10.1.5 A Key Concept: Reasoning Tokens

A traditional LLM output is a single token sequence: `<answer tokens>`. A reasoning model's output adds a "reasoning token" segment: `<reasoning tokens> <separator> <answer tokens>`.

In engineering terms, this is usually implemented with special tokens:

```text
<|begin_of_thought|>
The user's question asks me to solve ... let me first understand what's being asked ...
Possible approaches include ...
Approach one: ... the problem with this approach is ...
Approach two: ... this approach looks feasible ...
Let me verify ...
Yes, approach two gives the correct result.
<|end_of_thought|>
<|begin_of_solution|>
The final answer is X.
<|end_of_solution|>
```

This structure lets training assign separate scores to the "reasoning part" and the "answer part" — the foundation for techniques like Hybrid Thinking, thinking budgets, and long2short.

## Summary

The rise of reasoning models is a concentrated expression of the LLM training paradigm shift of 2024-2025, not an isolated event. o1 proved that RL can shape reasoning behavior. R1-Zero proved that pure RL needs no SFT cold start. The Competitive Programming paper proved that complex reasoning strategies emerge naturally from end-to-end RL.

But these findings raise a deeper question: **if reasoning ability mainly comes from pretraining, why are reasoning models so much stronger than traditional LLMs?** The answer is in the next section — test-time compute scaling changes how compute is allocated.
