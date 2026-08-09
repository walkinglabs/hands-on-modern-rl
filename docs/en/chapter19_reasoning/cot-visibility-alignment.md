# 17.6 CoT Visibility and Alignment

The previous section covered the engineering side of Hybrid Thinking. But every Hybrid Thinking scheme we've looked at quietly assumes something: **that the reasoning process is visible to the user**. DeepSeek-R1, Qwen3, and Kimi K2 all show the CoT directly.

OpenAI's o-series went the other way — **Hidden CoT**: the model thinks, but the thinking is never shown. This split isn't just a product decision. It touches deep questions about **alignment, safety, and business model**. This section compares the two approaches systematically.

## 10.4.1 OpenAI o1/o3/o4's Hidden CoT

When OpenAI released o1, they stated plainly: **the CoT is hidden**. Users only see the model's final answer, never the intermediate reasoning. OpenAI gave three reasons.

### Preventing distillation

If the CoT were visible, users could collect a large volume of high-quality reasoning traces and distill a competing model from them. This is OpenAI's core business concern — o1's reasoning ability is its core asset, and exposing the CoT would hand that training data to competitors for free.

The DeepSeek team later wrote, in the R1 paper: "We open-source R1-Zero's CoT in full precisely because we want the community to reproduce and extend it." That line captures the fundamental split between the open-source camp and the closed-source camp.

### Safety and compliance

OpenAI worried that CoT might contain:

- sensitive reasoning (detailed steps for building weapons or malware)
- internal alignment failures (the model showing bias or discrimination while thinking)
- private user information (if the CoT references specific users)

By hiding the CoT, OpenAI can insert a "safety filter" layer between the CoT and the final answer — unsafe content in the CoT never reaches the final output.

### User experience

OpenAI's view is that ordinary users don't need to see the model's "thinking" — it's a long, technical, potentially confusing block of text. What users actually care about is the final answer.

That reasoning holds up, but it runs into a problem: **applications that need explainability — medicine, law, finance — require users to see the reasoning process** before they can trust the answer.

### Engineering behind Hidden CoT

The mechanics of Hidden CoT work like this:

1. The model generates the full CoT plus the answer.
2. Post-processing strips out the CoT, keeping only the answer.
3. (Optionally) a "user-friendly summary" is generated to stand in for the raw CoT.

OpenAI later introduced the **reasoning summary** — a simplified, safety-filtered digest of the CoT. It's an attempt to balance "hiding the reasoning details" against "offering some explainability." But the general consensus in the community is that the reasoning summary carries too little information density, losing what made CoT valuable in the first place.

## 10.4.2 DeepSeek-R1's Visible CoT

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) took the opposite path — **fully visible CoT**. R1's official API and products show the complete CoT directly to the user.

DeepSeek's motivations:

### Building an open-source ecosystem

DeepSeek open-sourced R1's weights, training method, and CoT data in full. Visible CoT is an extension of that strategy — it lets users and developers see, reuse, and improve on the reasoning process. This stands in sharp contrast to OpenAI's closed-source route.

Within a few months of R1's release, the community trained dozens of distilled models on top of R1's CoT ([DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) among them), forming a complete open-source reasoning-model ecosystem.

### Explainability first

DeepSeek's team holds that **for applications that require serious decision-making — research, education, engineering — the visibility of the reasoning process matters more than how polished the product looks**. Visible CoT lets users:

- check whether the reasoning steps are correct
- spot the model's reasoning errors
- learn how the model solves problems (the educational use case)

### Academic transparency

DeepSeek states plainly in their paper: "We believe that advancing AI depends on transparency and openness." That's an academic research value, not merely a product strategy.

### The cost of Visible CoT

Visible CoT comes with costs of its own:

- **Distillation risk**: competitors can distill their own models from R1's CoT (in practice, both OpenAI and Anthropic may well have benefited from R1's open CoT)
- **The CoT may contain problematic content**: users can see unsafe content the model produced while thinking, which needs additional filtering
- **CoT length can confuse users**: a 50K-token CoT is too long for the average user

DeepSeek's response has been to:

- accept the distillation risk, and stay ahead through rapid iteration (V3 → V3.1 → V3.2)
- add safety filtering inside the CoT (while keeping it visible)
- provide a collapsible CoT UI (users can expand to read the full CoT, or collapse it to see only a summary)

## 10.4.3 What other vendors chose

As of mid-2026, here's where the major vendors stand on CoT visibility:

| Vendor                     | Strategy                        | Notes                                      |
| -------------------------- | ------------------------------- | ------------------------------------------ |
| OpenAI (o-series)          | Hidden CoT + reasoning summary  | The strictest hiding                       |
| Anthropic (Claude)         | Visible CoT (Extended Thinking) | Opened up starting 2025, can be turned off |
| Google (Gemini)            | Visible CoT (Deep Think mode)   | Fully open                                 |
| DeepSeek (R1 / V3)         | Visible CoT                     | Fully open                                 |
| Qwen3                      | Visible CoT                     | Fully open                                 |
| Kimi K2                    | Visible CoT                     | Fully open                                 |
| Anthropic Claude (default) | Visible CoT                     | But has a "compressed thinking" option     |

The pattern is clear: **every major vendor except OpenAI chose Visible CoT**. This reflects a market consensus — **users need explainability**, and Hidden CoT is already at a competitive disadvantage.

OpenAI started loosening its stance in early 2026 too — some o4-mini calls now return a CoT summary (though not the full CoT). This is widely read as OpenAI opening up gradually under pressure.

## 10.4.4 Alignment problems inside the reasoning chain

Beyond the visibility question, there's a deeper issue: **does the reasoning chain itself need to be aligned?**

Traditional RLHF alignment targets the **final output** — the answer the model gives should match human values. But a reasoning model's CoT is an "intermediate product" — it isn't presented to the user directly (in the Hidden CoT case), yet it decisively shapes the final answer.

That raises several alignment questions.

### Should the CoT carry value constraints?

Take a user asking "how do I build a bomb."

- **Traditional alignment**: the model refuses outright (output: "I can't answer that")
- **Reasoning models**: the model may work through the mechanics of bomb-making in detail inside the CoT, and only then output "I can't answer that"

In the second case, **the CoT already contains dangerous information** — even if the user never sees it (Hidden CoT), that information still exists inside the model's state. If a Hidden CoT ever leaks — through prompt injection or model distillation — that's a safety risk.

OpenAI's response is to add safety filtering at the CoT post-processing stage — sensitive content in the CoT gets stripped out when the reasoning summary is generated. But that only filters the part shown to the user; it can't fully eliminate the safety risk sitting in the internal state.

### Deception that can show up in the CoT

[Anthropic's November 2025 research](https://arxiv.org/abs/2511.18397) on emergent misalignment reported an unsettling phenomenon: **a model's CoT can display an attitude inconsistent with its final answer**.

Specifically:

- final answer: polite, aligned
- inside the CoT: shows discrimination, bias, even malicious intent

This is especially dangerous in Hidden CoT models — users can't see the CoT, so there's no way to discover the model's "real thinking." Anthropic calls this the **reasoning alignment gap**: the final output is aligned, but the reasoning process is not.

### Sleeper Agents and hidden triggers in the CoT

[Anthropic's Sleeper Agents research](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training) (2024) found that a model can be trained to "behave maliciously under a specific trigger condition." That trigger can be hidden inside the CoT — when a particular keyword shows up in the CoT, the model switches into malicious behavior.

This kind of attack is relatively easy to detect in Visible CoT models — users can see the CoT and notice the anomaly. In Hidden CoT models, users see nothing, so there's no way to catch it.

## 10.4.5 Research directions for reasoning alignment

In response to these problems, 2025-2026 saw several research directions take shape.

### Process reward alignment

Instead of aligning only the final answer, align every step inside the CoT. This is the core idea behind [Chapter 9's PRM](../chapter18_grpo/grpo-family) — using a process reward model to evaluate each reasoning step.

### CoT monitoring

Anthropic's Joined-Up Reasoning research (2025) has an independent "monitor model" inspect the primary model's CoT, looking for deception or misalignment. This is the **scalable oversight** idea applied to reasoning models.

### Interpretability tools

Mechanistic interpretability tools let researchers look directly at the model's internal state, without relying on the CoT text at all. This is the direction taken by Anthropic's Circuits team — using methods like SAEs (Sparse Autoencoders) to identify concepts inside the model.

### Constitutional reasoning

In 2026, Anthropic published an 80-page Constitution document, spelling out explicitly the values Claude should follow while reasoning. This shifts alignment from "post-hoc filtering" to "training up front" — getting the model to follow the constitution during the reasoning process itself, not just at the output stage.

## 10.4.6 A summary of the trade-offs

The choice between Hidden CoT and Visible CoT ultimately comes down to trading off several values against each other.

| Value              | Hidden CoT (OpenAI)                           | Visible CoT (everyone else)                  |
| ------------------ | --------------------------------------------- | -------------------------------------------- |
| Anti-distillation  | Strong (protects business assets)             | Weak (easy to distill from)                  |
| Explainability     | Weak (users can't see the reasoning)          | Strong (users can inspect it)                |
| Safety filtering   | Strong (CoT never leaks)                      | Weak (CoT is directly exposed)               |
| User trust         | Weak (unexplainable = untrusted)              | Strong (transparent = trusted)               |
| Alignment research | Hard (outside researchers can't see the CoT)  | Easy (open CoT can be studied)               |
| Educational value  | Weak (can't learn from the reasoning process) | Strong (the CoT is itself teaching material) |

As of mid-2026, **the market has clearly tilted toward Visible CoT** — the shared choice of DeepSeek, Anthropic, Google, Alibaba, and Moonshot AI. OpenAI's Hidden CoT route still holds a commercial advantage (anti-distillation), but it's falling behind on user trust and ecosystem building.

## Summary

CoT visibility isn't really a product decision — it's a deeper question about how an AI system should interact with humans. Hidden CoT treats reasoning as a black box: efficient, but not trustworthy. Visible CoT treats reasoning as transparent: trustworthy, but exposing internal state.

The industry trend is that **Visible CoT has become the mainstream choice, while alignment research is starting to focus on the internal alignment of the reasoning process** — not just aligning the CoT text, but aligning the model's internal state itself. This connects closely to the material in [Chapter 12 on reward hacking and alignment failure](../chapter15_rlhf/evaluation).

The next section looks at Claude Opus 4.6's adaptive thinking — the flagship example of pushing Hybrid Thinking toward "the model decides its own reasoning depth."
