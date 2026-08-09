# 17.5 Adaptive Thinking

The previous sections covered the rise of reasoning models, test-time scaling, Hybrid Thinking, and CoT visibility. This section focuses on one concrete case: **Claude Opus 4.6** (released 2025.10, upgraded to 4.7 in 2026.02).

Anthropic calls Opus 4.6 a "reasoning model that thinks adaptively" — a reasoning model built around **adaptive thinking**. Its design differs from both o1 and DeepSeek-R1, representing a distinct industrial approach to reasoning models.

## 10.5.1 The Core Idea of Adaptive Thinking

**Adaptive Thinking** is the next step beyond Hybrid Thinking. Hybrid Thinking is a binary choice — either think or don't think. Adaptive thinking is **continuous control over thinking depth** — the model decides for itself how deeply to think about each problem.

This idea can be formalized as a **thinking-depth parameter** $\tau \in [0, 1]$:

- $\tau = 0$: no thinking at all (answer directly)
- $\tau = 0.5$: moderate thinking (a few hundred tokens of CoT)
- $\tau = 1.0$: extensive thinking (tens of thousands of tokens of CoT, possibly with repeated revision)

After seeing the prompt, the model **first decides on $\tau$**, then generates reasoning at the corresponding depth.

### The Difference from thinking_budget

Qwen3's thinking_budget is **user-controlled** — the user specifies a budget in the API call. Claude Opus 4.6's adaptive thinking is **decided by the model itself** — the user doesn't specify a budget; the model makes the judgment call.

Each approach has its own advantages:

- **thinking_budget**: user-controllable, suited to applications with a **known difficulty distribution** (e.g., a customer-service API where most questions are simple)
- **adaptive thinking**: model-autonomous, suited to applications where **difficulty is unknown** (e.g., a research assistant facing questions of widely varying difficulty)

In practice, industrial deployments often **combine both** — the model decides its own $\tau$, and the user adds a budget cap on top as a safety net.

## 10.5.2 Training Details of Opus 4.6

Anthropic hasn't published the full training details of Opus 4.6 (this is a closed-source policy choice), but several key design decisions can be inferred from the [official blog post](https://www.anthropic.com/news/claude-opus-4-6) and the [Extended Thinking documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking):

### A Difficulty Estimator Based on Prompt Difficulty

Internally, the model has a "difficulty estimator" — after seeing the prompt, it quickly estimates difficulty and outputs $\tau$. This estimator is learned during RL training, not hand-designed.

The training data spans prompts of varying difficulty, and each prompt's reward depends not only on answer correctness but also on **whether the thinking depth was appropriate**:

- Thinking too much on an easy problem → wastes compute → penalized
- Thinking too little on a hard problem → wrong answer → penalized

Through this "difficulty-depth matching" reward signal, the model learns to **allocate thinking depth adaptively**.

### Extended Thinking API

Opus 4.6 exposes this to developers through the **Extended Thinking** API. Developers can do the following:

```python
# enable extended thinking
response = client.messages.create(
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "..."}],
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # optional budget cap
    }
)
```

`budget_tokens` is optional — if it isn't set, the model decides adaptively. This API design accommodates both **model-autonomous** and **user-controlled** modes.

### thinking signatures

When Opus 4.6 outputs CoT, sensitive content is replaced with `<thinking_signature>` — an encrypted "thinking signature." Users can see the structure and length of the CoT, but not the specific sensitive tokens.

This mechanism is an attempt to strike a compromise between Hidden and Visible CoT:

- Visible: the user can see the structure of the thinking (which parts are reasoning, which are the answer)
- Hidden: the specific sensitive content is encrypted
- Anti-distillation: the signature can't be reverse-engineered, so competitors can't train directly on the CoT

## 10.5.3 Opus 4.6's Flagship Capability and the AI Research Eval Suite

When Anthropic launched Opus 4.6, it prominently featured an internal benchmark — the **AI Research Eval Suite**. This suite includes several subtasks tied to RL research:

### LLM Training

Have Claude design a small RL training experiment on its own — choosing data, writing training code, tuning hyperparameters, running the experiment, and analyzing results. On this task, Opus 4.6 reaches **34x the speed of a human researcher** — a week of work for a researcher, Claude finishes in a few hours.

### Text-RL

Have Claude design and implement an RL algorithm for a text task (such as dialogue alignment). This task tests Claude's understanding and implementation ability for RL algorithms.

### Quadruped-RL

Have Claude design a gait RL algorithm for a quadruped robot — a classic control RL task. Opus 4.6 can write working PPO + reward-shaping code and train a policy that walks in a simulated environment.

These three subtasks reveal an important trend: **reasoning models aren't just good at "solving problems" — they can "do research" too**. Claude Opus 4.6's performance on these tasks marks the evolution of reasoning models from "exam performers" into "research assistants."

## 10.5.4 Anthropic's 80-Page Constitution

In February 2026, Anthropic released an 80-page Constitution 2.0 — a detailed specification of the values Claude should follow while reasoning. Here's how this Constitution relates to reasoning models:

### The Constitution as a Reasoning Constraint

The constitution behind traditional RLHF constrains **the final answer** — the answer should be polite, helpful, and harmless. Opus 4.6's Constitution extends the constraint to **the reasoning process itself**:

- The reasoning process should not exhibit discrimination
- The reasoning process should be honest (facts should not be distorted just to please the user)
- The reasoning process should weigh multiple parties' interests (not just the user, but others who are affected)

### The Constitution and Training

The Constitution isn't "prompt injection" — it isn't 80 pages the model reads at inference time. Instead, the model **internalizes the Constitution during training**. Concretely:

1. Break the Constitution down into actionable judgment criteria
2. Use these criteria to generate large volumes of "constitution-aligned" preference data
3. Use this data for RLHF / DPO training

A model trained this way **naturally follows the constitution** at inference time — no reminder in the prompt is needed.

### The Constitution and Interpretability

The 80-page Constitution also provides a new tool for **alignment interpretability** — researchers and users can use the Constitution to check whether the model's reasoning matches expectations. If the model violates one of the Constitution's criteria at some reasoning step, that's a potential alignment problem.

## 10.5.5 Safety Challenges of Reasoning Models

Opus 4.6's adaptive thinking also brings new safety challenges:

### The Model May "Fake Thinking"

If the model fakes deep thinking on an easy problem — generating a CoT that looks long but is empty of content — users have a hard time noticing. This is a form of **thinking deception**: using token count to fake reasoning depth.

Anthropic's response is to add a "thinking quality" reward during training — the content of the CoT must contribute to the final answer, and CoT that's padded purely to be longer is penalized.

### The Unpredictability of Adaptive Thinking

Adaptive thinking makes the model's behavior harder to predict — for the same prompt, the model might think for 1,000 tokens one time and 5,000 tokens the next. This is a problem in industrial deployment — latency and cost become hard to forecast.

The mitigation is to add a budget cap on top of adaptive thinking, but that weakens the "adaptive" property itself. This is a fundamental tension between **algorithmic capability and engineering controllability**.

### Attack Surface in the Reasoning Chain

Adaptive thinking makes the reasoning chain longer and more complex — giving attackers more opportunities to inject malicious triggers into the CoT. For example, prompt injection can induce the model to leak system-prompt content inside the CoT.

This is the problem [OpenAI's Instruction Hierarchy](https://openai.com/index/the-instruction-hierarchy/) (2025) sets out to solve — establishing clear priority among the system prompt, the user prompt, and tool return results, to prevent lower-priority content from hijacking higher-priority behavior.

## 10.5.6 Adaptive Thinking vs. Fixed-Depth Thinking

Let's close with a comparison:

| Dimension              | Fixed-Depth Thinking (early o1) | Hybrid Thinking (Qwen3)          | Adaptive Thinking (Claude Opus 4.6) |
| ---------------------- | ------------------------------- | -------------------------------- | ----------------------------------- |
| Mode                   | Always thinks deeply            | Binary choice (think / nothink)  | Continuous depth control            |
| Who controls it        | Fixed in the model              | User-specified / model picks one | Model decides autonomously          |
| Compute efficiency     | Low (wasted on easy problems)   | Medium (coarse-grained control)  | High (fine-grained matching)        |
| Engineering complexity | Simple                          | Moderate                         | High                                |
| Typical application    | Math / competitive coding       | General dialogue + reasoning     | Research assistant / complex tasks  |

**Adaptive thinking is a refinement of Hybrid Thinking** — moving from a binary choice to continuous control. This is the direction reasoning models are evolving in, and it comes with rising engineering complexity.

## 10.6 Alignment of Reasoning Models and Future Outlook

This chapter has traced the evolution of reasoning models from o1 to Claude Opus 4.6. But research into the alignment of reasoning models is only just beginning:

### Process Alignment

Future alignment research will focus more on **aligning the reasoning process itself** — not just having the final answer match the intended values, but having every reasoning step match them. This requires:

- Finer-grained PRMs (process reward models)
- Monitoring at inference time (CoT monitoring)
- Constitution-style reasoning training

### Safety Sandboxes for Reasoning Models

While thinking, a reasoning model might "consider" dangerous content, even if it never outputs it in the final answer. Future systems will need a "safety sandbox" — isolating the model's thinking process to prevent sensitive content from leaking out. This is a further development of the [Hidden CoT approach](./cot-visibility-alignment).

### Alignment Under Reasoning Scaling

Test-time compute scaling makes models increasingly capable, but it also makes alignment increasingly difficult — a longer CoT means a larger potential attack surface and more ways alignment can fail. What's needed is an **alignment scaling law** — one where alignment capability grows in step with model capability.

### The Convergence of Reasoning and Agentic Behavior

Opus 4.6's AI Research Eval Suite already demonstrates this trend — reasoning models don't just "think," they can "act" too (writing code, running experiments, analyzing results). This converges deeply with [Chapter 10, Agentic RL](../chapter22_agentic/intro). Future reasoning models will become complete agents that can both think and act.

## Chapter Summary

This chapter has laid out the full picture of reasoning models:

- **Section 10.1**: The rise of reasoning models — the evolution of o1/o3/o4, emergent evidence from the Competitive Programming paper, the nature of reasoning ability
- **Section 10.2**: Test-time Compute Scaling — Snell's research, parallel vs. sequential reasoning, the flagship case of Gemini Deep Think
- **Section 10.3**: Hybrid Thinking and thinking budgets — DeepSeek V3.1, Qwen3 Thinking Mode Fusion, the counterintuitive NoThinking finding, Kimi k1.5's long2short RL
- **Section 10.4**: Hidden vs. Visible CoT — OpenAI's hidden approach, DeepSeek's open approach, the challenges of reasoning alignment
- **Section 10.5**: Adaptive Thinking — the flagship case of Claude Opus 4.6, the 80-page Constitution, the AI Research Eval Suite

**Core takeaways**:

1. The essence of reasoning models is "**turning reasoning into an optimizable objective**" — RL training shapes the model's thinking behavior.
2. Test-time Compute Scaling changes how compute is allocated — from "more compute at training time" to "more compute at inference time."
3. Hybrid Thinking solves the "when to think" problem — either the model decides autonomously or the user controls it.
4. Visible vs. Hidden CoT is a fundamental choice about "AI transparency" — the market has leaned toward Visible.
5. Adaptive thinking is the fine-grained evolution of reasoning models — from a binary choice to continuous depth control.

**Coming up next**:

- [Chapter 9, PRMs and Inference-Time Search](../chapter18_grpo/grpo-family) — how to use process rewards to guide reasoning
- [Chapter 10, Agentic RL](../chapter22_agentic/intro) — how reasoning models combine with tool calling
- [Chapter 12, Reward Hacking and Alignment Failures](../chapter15_rlhf/evaluation) — alignment challenges specific to reasoning models
