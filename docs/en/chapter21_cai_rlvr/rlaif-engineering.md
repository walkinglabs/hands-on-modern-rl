# 19.2 RLAIF Engineering: Constitution Extensions

> [21.1](./hhh-practice) covered how the HHH triad gets implemented in Claude's training. This section looks at the engineering side of RLAIF at scale — Anthropic's 2026 release of an 80-page Constitution, the most detailed AI constitution effort in industry to date. Let's look at how a constitution gets "engineered."

## Anthropic's 2026 80-Page Constitution

In 2026 Anthropic publicly released an **80-page Constitution document for the Claude 4 family** (covering Claude 4 Opus / Sonnet / Haiku). This isn't another round of expanding the list of principles. It's a **structural shift in methodology**: from "enumerate rules" to "**socialization**." This section walks through its three core innovations.

### From a Rule List to a Values Framework

The old Constitution was "80 flat principles," each independent of the others. The new version introduces a **hierarchical structure**:

```
Top level: North Star values
  ├── Helpful subtree
  │     ├── Actually solve the problem
  │     ├── Distinguish the request from the action
  │     └── Proactively clarify ambiguity
  ├── Harmless subtree
  │     ├── Don't assist with serious harm
  │     ├── Proportionality (match refusal strength to risk)
  │     └── Protect vulnerable groups
  └── Honest subtree
        ├── Express uncertainty
        ├── Distinguish fact from speculation
        └── Acknowledge mistakes
```

Each leaf node is a concrete principle, but **conflicts are arbitrated by the priority of the level above**. When "Helpful: solve the problem" collides with "Harmless: proportionality," for instance, the resolution is weighted by risk level: low risk leans Helpful, high risk leans Harmless.

This hierarchy gives the AI judge a clear priority ordering when scoring, instead of leaving 80 principles to fight each other.

### Socialization — Getting the Model to "Internalize" Values

The key word in the title of the 80-page document is _socialization_. Anthropic borrows the concept from sociology: **values aren't instilled through rules — they're internalized through "socialization."** The human analogy: children don't grow up by memorizing legal statutes — they observe, imitate, and get corrected in concrete situations.

On the engineering side, Claude 4 training introduces **contextual alignment**:

1. Instead of having the model memorize "principle c_k," construct large numbers of **scenario-action pairs** so the model embodies values within a situation.
2. The judge prompt shifts from "evaluate against principle c_k" to "what would an ideal assistant do in this situation."
3. The training loss changes from a pure preference loss to preference loss plus a contextual-consistency regularizer:

$$
\mathcal{L} = \mathcal{L}_{\text{pref}} + \lambda_{\text{ctx}} \cdot \mathcal{L}_{\text{context-consistency}}
$$

where $\mathcal{L}_{\text{context-consistency}}$ measures whether the model's answers across different scenarios stay consistent with the Constitution framework.

::: details Why socialization is more robust than rule-listing
The fundamental problem with a rule list: **rules can't be exhaustive**. 80 principles cannot cover the endless variety of situations encountered in real deployment. Socialization has the model learn "the capacity for value judgment" rather than "rule matching." Anthropic reports that on OOD safety scenarios (ones unseen during training), Claude 4's robustness is 40%+ higher than the rule-list version. This lines up directly with the requirement in [Chapter 26, Computer Use](../chapter25_computer_use/intro) that "the model needs to generalize to new environments."
:::

### Auditability

The third focus of the 80-page Constitution is **auditability** — every model decision must be traceable back to a specific Constitution clause. This requires:

1. **Interpretable judge decisions**: the judge doesn't output a bare scalar score — it outputs a "verdict" that explicitly cites which principle applies.
2. **Training-data provenance**: every preference pair is tagged with which Constitution subnode triggered it.
3. **Deployment logging**: at inference time, the value judgments in the model's "inner monologue" are logged (in CoT form) to support after-the-fact auditing.

Formally: a model output $y$ carries an attribution $a(y) \in \mathcal{P}(\text{Constitution})$, a distribution over the Constitution clauses that $y$ is grounded in. The judge's preference loss is rewritten as:

$$
\mathcal{L}_{\text{audit}} = -\mathbb{E} \big[\log \sigma\big(r_\phi(x, y_w, a_w) - r_\phi(x, y_l, a_l)\big)\big] + \lambda_{\text{attr}} \cdot \text{Entropy}(a_w)
$$

The entropy term discourages the attribution from collapsing onto a single principle — when multiple principles genuinely apply, they should be listed explicitly.

### Claude 4 Constitution vs. Frontier Alignment Research

| Dimension          | Claude 2/3 Constitution            | Claude 4 Constitution (2026)                    |
| ------------------ | ---------------------------------- | ----------------------------------------------- |
| Structure          | Flat rule list (~80 items)         | Hierarchical value tree (North Star + subtrees) |
| Learning mechanism | Rule matching + AI judge           | Contextual socialization                        |
| Conflict handling  | Implicit (judge's subjective call) | Explicit priority arbitration                   |
| Interpretability   | Implicit reward                    | Explicit attribution + CoT                      |
| OOD robustness     | Weak                               | Strong (socialization generalizes)              |
| Auditability       | Black box                          | Every decision traceable to a principle         |

This line of work, together with the "AI supervision" research in [Chapter 34, Scalable Oversight](../chapter34_scalable_oversight/intro) and the large-scale alignment training in [Chapter 36, Distributed RL Training](../chapter36_distributed_rl_training/intro), forms a complete industrial-grade alignment stack.

## Chapter Summary

Constitutional AI and RLAIF are the pivotal step that moves LLM alignment from "depends on human annotation" to "scalable oversight":

1. **Constitutional AI** restructures safety alignment from "annotators score each response one by one" into "the model critiques and revises itself against a set of principles." SL-CAI and RL-CAI implement this idea via SFT and PPO respectively.
2. **RLAIF** replaces human annotation with an AI judge — cutting cost by two orders of magnitude and boosting speed a thousandfold, but it's bounded by judge capability and needs to be blended with a small amount of high-quality human feedback.
3. **Self-correction and self-rewarding** write the critique-revise loop explicitly into training. Self-Rewarding Language Models go further and fuse the generator, judge, and learner into one — the first few iterations are clearly effective, but external verification is needed to guard against reward hacking.
4. **The HHH triad** is the underlying value framework of the Constitution; the three principles are jointly optimized as a weighted reward inside multi-objective RL.
5. **The Claude 4 family Constitution** (2026) completes the methodological leap from "rule list" to "hierarchical value tree + contextual socialization + auditable attribution," offering a new paradigm for OOD robustness and interpretability.

The next chapter, [Chapter 21, RL Environments and Verifiers](../chapter23_rl_environments/intro), turns to the other half of RLAIF/RLVR: **how do you design a verifier?** Whether a math answer is correct, whether a piece of code actually runs, whether an API call complies with spec — all of these need an executable environment to produce a reward signal. That's the engineering foundation for converting RLAIF's "soft preferences" into "hard rules."

## Further Reading

- [Bai et al. 2022 "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073)
- [Lee et al. 2023 "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"](https://arxiv.org/abs/2309.00267)
- [Yuan et al. 2024 "Self-Rewarding Language Models"](https://arxiv.org/abs/2401.10020)
- [Askell et al. 2021 "A General Language Assistant as a Laboratory for Alignment" (the original definition of HHH)](https://arxiv.org/abs/2112.00861)
- [Anthropic 2024 "Collective Constitutional AI: Aligning a Language Model with Public Input"](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)
- [Anthropic 2026 "Claude 4 Constitution" (the 80-page socialized values framework document)](https://www.anthropic.com/research/claudes-constitution)
- [Sharma et al. 2023 "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548) (an analysis of RLAIF failure modes)
- [Gao et al. 2022 "Scaling Laws for Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760) (reward hacking from RM overoptimization)
