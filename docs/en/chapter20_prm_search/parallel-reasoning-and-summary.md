# 18.6 Parallel Coordinated Reasoning and Chapter Summary

So far we've covered PRM's three training routes (discriminative, generative, formal) and four inference-time search methods (Beam Search, ToT, MCTS, AlphaCodium). All of these methods share an implicit assumption: **reasoning is sequential**—the model walks forward one step at a time.

But a new direction emerged in the second half of 2025: **Parallel Coordinated Reasoning (PaCoRe)**. Instead of going sequential, it **unrolls multiple independent reasoning chains in parallel, then aggregates them**. This idea is philosophically aligned with the "parallel reasoning layer" in [Gemini Deep Think](../chapter19_reasoning/test-time-scaling).

This section covers the PaCoRe route, along with a related PRM concept—**GenRM (Generative Reward Model)**.

## 11.6.1 Depth vs. Breadth: Two Ways to Spend Inference Compute

In [Chapter 8, Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling), we discussed two ways to spend inference compute:

- **Sequential depth**: the model generates one very long CoT
- **Parallel breadth**: the model generates multiple independent CoTs, aggregated by some mechanism

| Approach         | Representative              | Compute allocation           | Advantage              | Disadvantage                   |
| ---------------- | --------------------------- | ---------------------------- | ---------------------- | ------------------------------ |
| Sequential depth | o1, R1, Qwen3 thinking      | All compute on one chain     | Good for hard problems | Slow, errors accumulate        |
| Parallel breadth | Best-of-N, Self-Consistency | Compute spread across chains | Fast, diverse          | Needs an aggregation mechanism |

**PaCoRe** is the extreme version of parallel breadth—it aggregates 16-32 independent reasoning chains, but the aggregation isn't simple majority voting. It's **coordinated by an LLM**.

## 11.6.2 The Design of PaCoRe

The core design of [PaCoRe](https://github.com/stepfun-ai/PaCoRe) (StepFun, ACL 2026 paper):

### Core pipeline

```text
┌─────────────────────────────────────────────────────┐
│ Step 1: Generate N reasoning chains in parallel      │
│   - Same prompt                                      │
│   - Each chain generated independently               │
│     (different temperatures, different seeds)        │
│   - No PRM guidance needed                           │
├─────────────────────────────────────────────────────┤
│ Step 2: Coordinator aggregates                       │
│   - An LLM reads all N reasoning chains               │
│   - Makes a joint judgment: which chains are correct? │
│   - Outputs the final answer                         │
├─────────────────────────────────────────────────────┤
│ Step 3: (at training time) reinforce with             │
│         outcome reward                                │
└─────────────────────────────────────────────────────┘
```

### Coordinator vs. voting

The key difference between PaCoRe and Best-of-N + Majority Vote is the **coordinator**:

- **Majority Vote**: pick the answer that appears most often (simple statistics)
- **PaCoRe Coordinator**: have an LLM read all the reasoning chains and "judge" which one is most credible (semantic aggregation)

The advantages of a coordinator:

- **Handles problems with multiple valid solutions**: if the N chains reach different "correct" answers (say a problem has several equivalent solutions), majority vote picks one at random, while PaCoRe can recognize that they're all correct
- **Assesses reasoning quality**: even when two chains reach the same answer, PaCoRe can judge which reasoning process is more rigorous

### Training PaCoRe

PaCoRe is trained with **outcome-based RL**—only the correctness of the final answer is rewarded, not the intermediate reasoning. This matches R1-Zero: simple, and it needs no PRM annotation.

```python
def pacore_reward(prompt, target_answer):
    # 1. Generate N reasoning chains in parallel
    reasonings = [model.generate(prompt, temperature=t) for t in temperatures]

    # 2. Coordinator aggregates
    final_answer = coordinator.aggregate(prompt, reasonings)

    # 3. Train with outcome reward
    reward = 1.0 if final_answer == target_answer else 0.0

    # Backpropagate the reward to all reasonings and to the coordinator
    return reward
```

This training scheme means **the entire PaCoRe system is optimized as a single unit by RL**, rather than optimizing each reasoning chain separately.

## 11.6.3 PaCoRe's Experimental Results

PaCoRe's results on [AIME 2025](https://github.com/stepfun-ai/PaCoRe):

| Method                            | AIME 2025 | Reasoning tokens   |
| --------------------------------- | --------- | ------------------ |
| Single thinking pass (baseline)   | 60-70%    | ~10K               |
| Best-of-32 + Majority Vote        | 80%       | 320K (32×10K)      |
| **PaCoRe (16-way parallel)**      | **94.4%** | 160K (16×10K)      |
| Gemini 3.1 Deep Think (reference) | 90%+      | millions of tokens |

PaCoRe reaches a higher accuracy **with less compute than Best-of-N**—which shows that a coordinator (LLM aggregation) is more effective than simple voting.

## 11.6.4 PaCoRe vs. Deep Think vs. MCTS

Comparing the three reasoning paradigms:

| Dimension                 | PaCoRe                             | Deep Think                               | MCTS                       |
| ------------------------- | ---------------------------------- | ---------------------------------------- | -------------------------- |
| Reasoning structure       | N independent chains + coordinator | N parallel chains + cross-path attention | Tree expansion             |
| Compute overhead          | N × single chain                   | N × single chain                         | Exponential                |
| Training requirement      | outcome RL                         | model architecture changes               | PRM + value                |
| Best-suited tasks         | Medium difficulty, high diversity  | Hard problems needing coordination       | Formal, precision-critical |
| Implementation difficulty | Medium                             | High (model changes)                     | High                       |

### When should you use PaCoRe?

PaCoRe's advantages:

- **Simple**: no PRM needed, no model architecture changes needed
- **Efficient**: higher accuracy than Best-of-N
- **Scalable**: N-way parallelism scales linearly

Good-fit scenarios:

- Problems with multiple valid solutions (a problem that admits several correct solution methods)
- Medium-difficulty reasoning (doesn't need deep tree search)
- Plenty of compute available, but model changes are impractical

## 11.6.5 GenRM: The Generative Reward Model

When discussing PaCoRe, we mentioned that "the coordinator is an LLM." That points to a broader concept—**GenRM (Generative Reward Model)**.

GenRM's core idea: **turn reward computation into a generation task**. A traditional RM is a regression model—it takes a prompt + response as input and outputs a scalar score. GenRM is an LLM—it takes a prompt + response as input and outputs a natural-language critique plus a final judgment.

### The form of GenRM

```text
Input: prompt + response + "Please evaluate this response"
Output: natural-language critique + [GOOD/BAD]
```

GenRM can use [verbal confidence](https://arxiv.org/abs/2305.14992)—having the LLM output a probability as the reward:

$$\text{GenRM}(q, o) = P(\text{"good"} \mid q, o, \text{prompt})$$

That is: given the prompt, the model outputs the probability it assigns to the token "good."

### GenRM vs. discriminative RM

| Dimension        | Discriminative RM         | GenRM                               |
| ---------------- | ------------------------- | ----------------------------------- |
| Architecture     | Encoder + regression head | Standard LLM                        |
| Output           | Scalar score              | Natural language + probability      |
| Training         | Regression loss           | Language-model loss                 |
| Interpretability | Weak (score only)         | Strong (natural-language rationale) |
| Inference speed  | Fast                      | Slow                                |

### The relationship between GenRM and PRM

GenRM is a broader concept—it can implement an ORM (evaluating the whole response) or a PRM (evaluating each reasoning step).

[Section 11.3's ThinkPRM](./generative-prm) is a representative case of GenRM implementing a PRM. Other GenRM work includes:

- **Generative Verifiers** ([Zhang et al.](https://arxiv.org/abs/2408.15240)): evaluates using chain-of-thought
- **LLM-as-Judge** ([Zheng et al.](https://arxiv.org/abs/2306.05685)): uses GPT-4 to evaluate other models' outputs

## 11.6.6 LLM-as-Judge and Self-Rewarding

LLM-as-Judge is an industrial application of GenRM—using a strong LLM (GPT-4, Claude) to evaluate other models' outputs.

### Applications of LLM-as-Judge

- **Benchmark evaluation**: using GPT-4 as the judge for MT-Bench, AlpacaEval
- **Training-data filtering**: using an LLM to filter high-quality training data
- **RLHF replacement**: using an LLM in place of human preference annotation (RLAIF)

### Self-Rewarding Language Models

[Self-Rewarding LM](https://arxiv.org/abs/2401.10020) (Meta, 2024) pushes LLM-as-Judge to its limit—**having the model evaluate its own outputs**:

```python
def self_reward_training(prompt, model):
    # 1. Generate multiple responses
    responses = [model.generate(prompt) for _ in range(N)]

    # 2. Have the model judge itself
    rewards = [model.judge(prompt, r) for r in responses]

    # 3. Use self-evaluation for RL (DPO or PPO)
    model = rl_update(model, prompt, responses, rewards)
```

This approach **eliminates the external RM entirely**—the model is both policy and reward. The upside is no RM training is needed. The downside is that **self-evaluation can reinforce existing biases**: whatever the model already thinks it's good at gets reinforced, and whatever it's weak at gets suppressed further.

## 11.6.7 The Future of PRM and Verifiers

As of mid-2026, a few trends stand out in PRM and verifier research:

### From discriminative to generative

Discriminative PRM → generative PRM → self-evaluation—the trend is toward replacing external verifiers with the LLM's own reasoning capability.

### From depth to breadth

Tree of Thoughts → MCTS → PaCoRe—the trend is moving from deep sequential search to broad parallel coordination.

### From static to dynamic

Fixed PRM → dynamic verifier → adaptive search—the trend is letting the verifier adjust dynamically during reasoning.

### From single to hybrid

ORM-only → PRM-only → PRM + ORM + formal methods + LLM-as-Judge hybrids—the trend is combining multiple verifiers so they complement each other.

## Chapter Summary

This chapter has laid out the full picture of PRM and inference-time search:

- **Section 11.1**: Outcome vs. process reward—the sparse-reward problem and credit assignment
- **Section 11.2**: Discriminative PRM—OpenAI's Let's Verify and PRM800K
- **Section 11.3**: Generative PRM—ThinkPRM reaches SOTA with 1% of the labels
- **Section 11.4**: Formal PRM—AlphaProof, Lean4, DeepSeek-Prover-V2
- **Section 11.5**: Inference-time search—Beam Search, ToT, MCTS, AlphaCodium
- **Section 11.6**: Parallel coordinated reasoning (PaCoRe) and GenRM, LLM-as-Judge

**Core takeaways**:

1. **PRM is the key technique for long-CoT tasks**—it turns sparse reward into dense reward
2. **The three routes each have their own strength**—discriminative is precise, generative is efficient, formal has zero misjudgment
3. **Inference-time search can push performance further**—but it comes with heavy compute overhead, so Best-of-N remains the industrial mainstream
4. **Parallel coordination (PaCoRe) is a new direction**—it converts deep search into broad parallelism, balancing compute against quality
5. **GenRM and LLM-as-Judge are where verifiers are heading**—using the LLM's intrinsic capability to replace external verifiers

**Next chapters**:

- [Chapter 10, Agentic RL](../chapter22_agentic/intro)—PRM in multi-step trajectories
- [Chapter 12, Reward Hacking](../chapter15_rlhf/evaluation)—PRM's reward-hacking problem
- [Chapter 13, Industrial Practice](../chapter17_dpo/industrial-post-training)—how PRM is used in industrial training
