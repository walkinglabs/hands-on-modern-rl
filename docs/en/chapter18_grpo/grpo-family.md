# 16.4 The GRPO Improvement Family

The previous section walked through DeepSeek-R1-Zero and DAPO — the first proved that "pure RL can replace SFT cold-start," the second showed that engineering fixes let GRPO reach R1-Zero's level in half the training steps. But DAPO is only **one member** of the 2025 GRPO improvement family. Between the R1 paper's release (January 2025) and early 2026, the open-source community and industry labs produced at least five influential GRPO variants in under a year. They aren't competing replacements for one another — each patches a different flaw in GRPO from a different angle.

This section reorganizes them by **direction of improvement** — putting Dr.GRPO, GSPO, CISPO, VAPO, and RPT side by side so you can see clearly which variant to reach for and when.

## Improvement Direction One: Removing the Normalization Bias

### What Dr.GRPO Found

GRPO's original form in the R1 paper applies two normalization steps to the group's rewards:

$$\tilde{r}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

Dividing by the standard deviation looks natural here — it puts the advantage values on a common scale. But Liu et al., in a 2025 study ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)), found that this seemingly harmless normalization introduces two kinds of bias:

- **Length bias**: when a prompt's responses have high reward variance (some right, some wrong), dividing by std compresses the advantage. When variance is low (all right or all wrong), std approaches zero and the advantage gets blown up to an unreasonable magnitude. The model ends up learning that "producing varied outputs matters more than getting the answer right."
- **A breeding ground for reward hacking**: dividing by std is equivalent to rewarding the model for increasing the reward variance within a group, and the easiest way to increase variance is to make some of the responses **longer** — more tokens, more chances to land on a correct answer. This is one of the direct causes of the response-length explosion seen late in R1-Zero training.

Dr.GRPO's fix is strikingly simple — **subtract the mean, don't divide by std**:

$$\tilde{r}_i^{\text{Dr.GRPO}} = r_i - \text{mean}(r_1, \ldots, r_G)$$

Experiments show this single change significantly eases the late-training length inflation problem and reduces reward-hacking behavior. The Qwen series adopted a similar fix internally.

### DeepSeek V3.2's Further Engineering

DeepSeek pushed Dr.GRPO's idea further in V3.2 (December 2025, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)), making three engineering adjustments specifically for math reasoning tasks:

- **Zero KL for math tasks**: standard GRPO uses a KL divergence penalty to keep the policy from straying too far from the reference model, but for math tasks the reward itself already provides enough constraint — a wrong answer scores zero. The KL penalty just suppresses exploration of new solution paths. DeepSeek turns KL off entirely during the pure-math RL stage.
- **Self-verifying RLVR**: the model is trained to append a "verification step" after generating its answer — reread the problem, check the arithmetic, confirm the answer. The reward for this verification step is folded into the RL objective, forming an internal self-check mechanism.
- **mHC residual stability**: numerical-stability improvements to the Modified Hamiltonian Monte Carlo sampler used during long-CoT training, avoiding gradient explosions.

The V3.2 Speciale variant scores 97 on AIME 2025, surpassing GPT-5's contemporaneous level.

## Improvement Direction Two: Sequence-Level Importance Sampling

### GSPO (Qwen3's Choice)

GRPO, like PPO, uses a **token-level importance sampling ratio**:

$$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

Every token gets its own ratio, and the gradient for the whole sequence is the combined effect of all the per-token ratios. In LLM training this creates a concrete problem: **under an MoE architecture, different tokens route to different experts, so token-level ratios fluctuate wildly**, driving up gradient variance and destabilizing training.

GSPO (Group Sequence Policy Optimization, [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) raises the ratio from the token level to the **sequence level**:

$$\rho^{\text{seq}} = \frac{\pi_\theta(o|q)}{\pi_{\theta_{\text{old}}}(o|q)} = \prod_{t=1}^{|o|} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

A single ratio now covers the entire response. Clipping shifts to the sequence level accordingly:

$$\mathcal{L}^{\text{GSPO}} = \mathbb{E}\left[\min\left(\rho^{\text{seq}} \cdot \tilde{r}, \; \text{clip}(\rho^{\text{seq}}, 1-\epsilon, 1+\epsilon) \cdot \tilde{r}\right)\right]$$

This change looks simple but has an enormous effect on training stability for MoE models — the entire Qwen3 lineup (Qwen3-235B-A22B, Qwen3-Thinking-2507, Qwen3-Coder, and others) trains on GSPO. A sequence-level ratio has far lower variance than a token-level one, which is what makes large-scale RL training on ten-thousand-GPU clusters practical.

GSPO's cost: a sequence-level ratio **couples the updates of every token together**, so its fine-grained credit assignment at the single-token level is weaker than a token-level scheme. That's why GSPO shines on long-CoT tasks (reasoning, math) but underperforms DAPO on code-generation tasks that need token-level reward signals.

## Improvement Direction Three: Rewriting What Gets Clipped

### CISPO (MiniMax's Innovation)

Both GRPO and DAPO clip "the product of the policy ratio and the advantage" — clipping happens at the level of the gradient update. MiniMax, in its M1 model ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)), proposes CISPO, which moves the clipping target from "the token update" to the **importance sampling weight**:

$$\tilde{\rho}_t = \text{clip}\left(\frac{\pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_\theta(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)$$

Notice the ratio's numerator and denominator are flipped here — $\pi_{\text{old}} / \pi_\theta$ rather than $\pi_\theta / \pi_{\text{old}}$. This inverted ratio acts as a **sampling weight** multiplied onto the advantage, but it **preserves every token's gradient contribution**.

The intuition: standard clipping says "if a token's policy has drifted too far, wipe it out of the gradient entirely." CISPO says "if it has drifted too far, lower its weight in the advantage estimate, but keep the gradient direction intact." The latter avoids the policy getting stuck when a large fraction of tokens stop updating altogether late in training.

CISPO carries a further engineering advantage — paired with MiniMax's own lightning attention, it resolves a precision-alignment issue. Lightning attention's recursive computation lets floating-point error accumulate in the token-level ratios, and standard clipping under low-precision training ends up wrongly discarding large numbers of tokens. CISPO sidesteps this by rescaling weights instead of clipping them outright. MiniMax M1 trains on 512 H800 GPUs at roughly twice DAPO's overall speed.

## Improvement Direction Four: The Value-Based Counter-Trend

### VAPO (ByteDance Seed's Counter-Trend)

By this point you might have formed the impression that **the critic network has been rendered obsolete by GRPO**. But ByteDance Seed's VAPO (Value-based Augmented PPO, [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)), published in April 2025, shows the opposite — at least for long-CoT reasoning tasks, **the value model beats GRPO again**.

VAPO's core argument: GRPO replaces the critic with the group mean, which in essence estimates the advantage from "the relative ranking among multiple rollouts of the same prompt." That's sufficient for short-answer tasks — function calling, simple math problems. But for long-CoT tasks:

- A single rollout spans hundreds of tokens, and **the real advantage signal lives at the token level** — one reasoning step might be good, the next bad.
- The group mean treats the entire rollout as one unit, throwing away that token-level signal.
- The longer training runs, the more the model learns to "get lucky on a fraction of rollouts" rather than "get every step right."

VAPO reintroduces a value model $V_\phi(s)$ and estimates token-level advantages with GAE:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is the TD error. PPO-style clipping is then applied on top of this token-level advantage.

VAPO scores 60.4 on AIME 2024, beating every contemporaneous GRPO variant (DAPO scores 50; R1-Zero scores 71 but needs twice the training steps). **ByteDance Seed's internal reasoning-model training has shifted from pure GRPO to VAPO.**

VAPO's cost: it requires training a separate value model, doubling memory usage and raising engineering complexity. This is exactly why GRPO became mainstream in 2024 — **being critic-free was an engineering compromise, not an algorithmic necessity.**

## Improvement Direction Five: Bringing RL into Pretraining

### RPT (Reinforcement Pre-Training)

The four improvement directions above all assume RL happens in the **post-training stage** — the model is already pretrained, and RL is just fine-tuning on top. But Microsoft's Reinforcement Pre-Training ([arXiv:2506.08007](https://arxiv.org/abs/2506.08007)), proposed in June 2025, challenges that split entirely.

RPT's core idea: **reframe next-token prediction as a reasoning task**. The standard pretraining loss is:

$$\mathcal{L}_{\text{LM}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t})\right]$$

Every token is an equally weighted teacher-forcing target. RPT changes this: the model first generates a piece of reasoning about the next token ("given the context, the next word is probably X because..."), then uses that reasoning to predict the next token, and gets rewarded if it's correct:

$$\mathcal{L}_{\text{RPT}} = -\mathbb{E}\left[\log \pi_\theta(a_t | s_{<t}, \text{reasoning}_t)\right] + \beta \cdot \text{RL loss}$$

The significance of this change is genuinely revolutionary — **RL can now happen during pretraining itself**, and RPT's scaling behavior matches that of standard pretraining. This suggests the sharp boundary between "pretraining" and "post-training" may disappear, with RL running through the entire training pipeline.

RPT is still at an early stage and hasn't seen broad industrial adoption yet. But its conceptual impact is significant enough to earn it a dedicated place in this improvement lineup.

## A Decision Tree for Choosing a Variant

The table below lays out the five variants' core differences, typical use cases, and representative users side by side:

| Algorithm   | Core Innovation                | Pain Point Solved                      | Typical Use Case                   | Representative User  |
| ----------- | ------------------------------ | -------------------------------------- | ---------------------------------- | -------------------- |
| **GRPO**    | Group mean replaces the critic | Critic memory overhead                 | General RLHF / RLVR                | DeepSeek-R1          |
| **Dr.GRPO** | Removes std normalization      | Length inflation, reward hacking       | Math reasoning                     | Qwen (internal)      |
| **GSPO**    | Sequence-level IS              | MoE training instability               | RL for MoE models                  | Qwen3 lineup         |
| **CISPO**   | Clips the IS weight            | Token loss, precision alignment        | Lightning attention, low precision | MiniMax M1           |
| **VAPO**    | Reintroduces the value model   | Long-CoT credit assignment             | Reasoning-model training           | ByteDance Seed       |
| **DAPO**    | Four engineering fixes         | Training efficiency, length control    | Math / code RL                     | ByteDance + Tsinghua |
| **RPT**     | Brings RL into pretraining     | The pretraining–post-training boundary | Next-generation base models        | Microsoft Research   |

The selection logic in actual industry practice looks roughly like this:

```text
What kind of task?
├── Math / code reasoning (long CoT)
│   ├── MoE architecture → GSPO + Dr.GRPO ideas
│   ├── Dense architecture → VAPO or DAPO
│   └── Extreme stability requirements → CISPO
├── General conversational alignment
│   └── GRPO / PPO (the basics suffice)
├── Multi-turn tool calling
│   └── DAPO + token-level loss
└── Next-generation base model
    └── RPT (experimental)
```

This decision tree isn't absolute — ByteDance Seed's internal teams frequently mix approaches (for example, DAPO's engineering tricks combined with VAPO's value model). But it gives you a checklist for "what should come to mind first once you know the task."

## Summary

The rapid evolution of the GRPO improvement family reflects a simple fact: **RL at the large-model scale is no longer a matter of "PPO is good enough."** Every lab has picked a different direction of improvement based on its own training infrastructure (MoE vs. dense, lightning attention vs. standard attention, memory budget) and task characteristics (reasoning vs. dialogue, long CoT vs. short answers).

The real value of this section isn't memorizing every algorithm's exact formula — it's building the judgment to, **on seeing a new GRPO variant, immediately ask which of these four lines it's patching: normalization, sequence-level ratios, clipping, or the value model.** That judgment is the crucial step from reading papers to actually improving algorithms yourself.
