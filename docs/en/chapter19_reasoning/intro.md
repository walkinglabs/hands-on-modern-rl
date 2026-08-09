# 17.2 R1-Zero Pure RL Training

In the last chapter we lined up five improvements from the GRPO family side by side—Dr.GRPO, GSPO, CISPO, VAPO, RPT. These algorithm-level improvements answer the question of how to train more stably and more efficiently. But the deepest shift in the RL field during 2025 happened at a higher level: **RL doesn't just change training—it also reshapes how models behave at inference time**.

This chapter's subject is the **reasoning model**—a class of large language models that write "thinking" explicitly into their output, using RL to make the thinking process itself an optimization target. From OpenAI o1 (September 2024) to Claude Opus 4.6 (2025), in just over a year, reasoning models went from lab prototypes to production-grade products, redefining what large language models can do.

## Questions This Chapter Answers

By the end of this chapter, you should be able to clearly answer:

- **What fundamentally distinguishes reasoning models from traditional LLMs**—is it just "producing a longer CoT"?
- **The evolution path of OpenAI o1/o3/o4**—why is hidden CoT an engineering inevitability?
- **The theoretical basis for Test-time Compute Scaling**—why is "spending more compute at inference time" more cost-effective than "spending more compute at training time" for certain tasks?
- **Hybrid Thinking and thinking budgets**—why can't a model be allowed to think forever? How is it controlled?
- **Long CoT compression (long2short)**—how does Kimi k1.5 compress 10K tokens of reasoning down to 2K?
- **Adaptive thinking**—how does Claude Opus 4.6 decide "how long to think about this problem"?
- **The alignment implications of Hidden vs. Visible CoT**—should the reasoning process be shown to users?

## Chapter Map

```text
19.1 The Rise of Reasoning Models
     ├── The evolution of OpenAI o1 → o3 → o4
     ├── Emergent evidence from the Competitive Programming paper
     └── Is reasoning ability "emergent" or "activated"
19.3 Test-time Compute Scaling
     ├── The trade-off between training compute and inference compute
     ├── Snell et al.'s scaling law
     └── Gemini 3 Pro Deep Think's parallel thinking
19.4 Hybrid Thinking and Thinking Budgets
     ├── DeepSeek V3.1's dual-mode fusion
     ├── Qwen3 Thinking Mode Fusion
     └── The counterintuitive finding of NoThinking + Best-of-N
19.6 Readability and Alignment of the Reasoning Chain
     ├── OpenAI's engineering motivation for hiding reasoning
     ├── DeepSeek-R1's strategy of open reasoning
     └── The trade-off between readability and alignment
19.5 Adaptive Thinking
     ├── Claude Opus 4.6's adaptive depth
     ├── Anthropic's Constitution and reasoning ability
     └── Deception and alignment within the reasoning chain
```

## Relationship to Other Chapters

This chapter assumes you've already read:

- [Chapter 16, The GRPO Improvement Family](../chapter18_grpo/grpo-family)—training details of R1-Zero and DAPO
- [Chapter 16, DeepSeek-R1 and DAPO](../chapter18_grpo/deepseek-dapo)—concrete implementation of the pure-RL training paradigm
- [Chapter 13, RLHF](../chapter15_rlhf/intro)—the basics of reward signals

Later chapters this one points forward to:

- [Chapter 18, PRM and Inference-time Search](../chapter20_prm_search/intro)—process rewards and tree search
- [Chapter 28, Reward Hacking and Alignment Failures](../chapter30_alignment_failures/intro)—reward hacking specific to reasoning models
- [Chapter 20, Agentic RL](../chapter22_agentic/intro)—reasoning models as the "brain" of an agent

## An Intuitive Opening

Before diving into the formal content, let's build two key intuitions:

**Intuition one: a reasoning model is not CoT prompt engineering scaled up.** Traditional CoT (Chain-of-Thought) prompting adds a line at inference time—"let's think step by step"—which only activates reasoning ability the model already acquired during pretraining. A reasoning model, by contrast, is trained with RL so that **the model itself learns when to think, how long to think, and when to stop thinking and give an answer**. The difference is like the difference between reminding a student that they know how to solve a problem and training a student in the methodology of solving problems.

**Intuition two: the essence of a reasoning model is turning reasoning itself into an optimizable objective.** In the RLHF era, the model's optimization target was "give an answer that satisfies people"—this target places no direct constraint on the reasoning process. In the reasoning-model era, the optimization target is "reach the correct answer through reasoning"—this target directly shapes the reasoning process itself. That's why R1-Zero could produce an emergent aha moment: the reward signal selected for "reasoning paths that reflect," so reflective behavior got reinforced and retained.

With these two intuitions in hand, let's move on to 19.1.
