# 25.1 RLVR → RLHF Audio Reward Design

> [27.0](./intro) covered how the Step-Audio series evolved. This section focuses on a core engineering problem: **how do you design an audio reward?** Text RMs can train directly on preference pairs, but audio adds dimensions like prosody, emotion, and accent — a single reward signal can't cover all of that.

## The Evolution from RLVR to RLHF

Step-Audio-R1 used MGRD + RLVR to hit SOTA on objective benchmarks. But once deployed in real conversations, the team ran into a counterintuitive problem: **the higher the benchmark score, the worse the conversation sounded.**

### 30.3.1 The Verifiable Reward Trap

[Step-Audio-R1.5](https://arxiv.org/abs/2604.25719) gave this problem a name: the **Verifiable Reward Trap**.

::: warning The Verifiable Reward Trap
When an audio benchmark's ground truth is just a discrete label (an emotion category, an ASR transcript, a scene tag), RLVR can only reward "guessing the label correctly." It **structurally ignores** prosodic naturalness, emotional coherence, and conversational fluency.
:::

The mechanism behind the trap:

```text
RLVR objective = answer correctness → model learns to be "maximally token-efficient" → responses turn short, mechanical, flat
                ↓
         benchmark score ↑    real conversation quality ↓
```

RLVR optimizes "what to say." What users actually care about is "how to say it." When the two decouple, the model degrades into an **answer machine** — technically accurate, experientially hollow.

### 30.3.2 Step-Audio-R1.5: From RLVR to RLHF

R1.5's fix is to bring in RLHF to patch what RLVR misses: train a holistic preference reward model that distills correctness, fluency, and emotional resonance into a single, unified supervision signal.

#### Audio-Centric Mid-Training

Before RLHF, the model goes through a round of mid-training that strengthens the underlying audio understanding and reasoning:

$$\mathcal{L}_{\text{mid}} = \mathbb{E}_{(x,q,r,y) \sim \mathcal{D}_{\text{audio}}}\left[\log \pi_\theta(r, y \mid x, q)\right] + \mathbb{E}_{(q,r,y) \sim \mathcal{D}_{\text{text}}}\left[\log \pi_\theta(r, y \mid q)\right]$$

Here $(x, q, r, y)$ denotes audio input + context + reasoning + response. The text-only data preserves long CoT reasoning structure, facilitating transfer to the audio modality.

#### Cold-Start SFT

At this stage, cold-start SFT is no longer about expanding domain knowledge. Its job is to **align interaction behavior**:

1. **Multi-turn continuity**: maintaining context and constraints across turns
2. **Instruction following**: responding according to the content, format, and style the user specifies
3. **Response naturalness**: coherent, conversationally appropriate delivery
4. **Interaction awareness**: handling follow-ups, clarifications, interruptions, and user corrections

This step gives the subsequent RLHF stage a better initialization, so preference optimization doesn't get wasted correcting basic conversational behavior.

#### RLHF with Rubric-based Reward Model

Audio interaction is a multi-objective optimization problem — content correctness, natural prosody, emotional coherence, and controlled latency all matter at once. R1.5 replaces the scalar RM with a **rubric-based Generated Reward Model (GRM)**:

```python
def audio_rlhf_reward(response, context, rubric):
    """Multi-dimensional scoring instead of a single scalar"""
    scores = {}
    scores["correctness"] = grm.score(response, context, rubric="Is the content correct?")
    scores["fluency"] = grm.score(response, context, rubric="Is the delivery fluent and natural?")
    scores["prosody"] = grm.score(response, context, rubric="Does the prosody match the emotion?")
    scores["emotional_resonance"] = grm.score(response, context, rubric="Emotional resonance")
    scores["latency"] = grm.score(response, context, rubric="Response latency")
    # Weighted aggregation (weights learned by regressing on human preferences)
    return sum(w[k] * scores[k] for k in scores)
```

GRM's advantage: **human preference is multi-dimensional**, and a scalar RM cannot capture that. Score each dimension separately with an LLM-as-judge (rubric prompting), then learn a weighted aggregator on top. That upgrades the [RLHF](../chapter15_rlhf/intro) RM from producing a single "overall score" to filling out a full "scorecard."

#### Multi-Objective RL Training Objective

R1.5's RL loss combines RLVR and RLHF:

$$\mathcal{L}_{\text{RL}} = \underbrace{\mathbb{E}_{\mathcal{D}_{\text{verified}}}\left[R_{\text{verify}}(r, a)\right]}_{\text{objective correctness (RLVR)}} + \lambda \cdot \underbrace{\mathbb{E}_{\mathcal{D}_{\text{pref}}}\left[\log\sigma\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}_{\text{subjective preference (DPO form)}}$$

The first term preserves objective reasoning ability, so RLHF doesn't erase what RLVR already learned. The second term uses a DPO loss (see [Chapter 17: GRPO/DPO](../chapter18_grpo/grpo-family)) to align subjective experience. $\lambda$ balances the two — this is the core hyperparameter of audio RL.

### 30.3.3 Preserving Prosodic Naturalness

RLVR's biggest casualty is **prosodic flattening**: to maximize answer correctness, the model turns speech into a monotone "recitation." R1.5 uses three mechanisms to keep prosody intact:

1. **Preference data includes a prosody dimension**: when annotators compare two responses, they don't just judge the content — they also listen for "which one sounds more natural, gets the emotion more right, and has a more human rhythm"
2. **The rubric explicitly scores prosody**: the GRM assigns prosody its own score, kept separate from correctness
3. **Codec-token-level supervision**: the RVQ's $c_2 \ldots c_K$ (acoustic layers) participate in the preference signal, so prosodic information is preserved as early as the generation stage

R1.5 matches or beats Gemini-2.5-Flash on AudioMultiChallenge (a multi-turn conversation benchmark testing Inference Memory / Instruction Retention / Self Coherence / Voice Editing), **while** losing no ground on traditional reasoning benchmarks. RLHF unwinds the trap that RLVR set.

## Audio Reward Design

Audio RL rewards are far more complex than text rewards. Text mostly comes down to correctness; audio has three layers to get right — content, prosody, and real-time responsiveness. This section works through the design of each.

### 30.4.1 Content Correctness Reward

The most direct approach: compare the final answer against ground truth.

$$R_{\text{content}}(r, a) = \begin{cases}1, & \text{if } a = a^* \\ 0, & \text{else}\end{cases}$$

Variants include:

- **ASR word error rate**: the lower the WER, the higher the reward, $R = 1 - \text{WER}$
- **Semantic matching**: embedding cosine similarity, $R = \cos(\text{emb}(a), \text{emb}(a^*))$
- **LLM-as-judge**: have a large model judge whether the answer is equivalent to the reference, $R \in [0, 1]$

Content rewards work well for objective tasks (math, knowledge QA, ASR), but they fail for open-ended conversation, which has no standard answer.

### 30.4.2 Prosodic Naturalness Reward

Prosody covers pitch, rhythm, intensity, and pausing. Modeling human preferences over naturalness is the hard part of audio RL.

#### The Limits of a Scalar RM

The conventional approach: train an RM $R_\phi(\text{audio}) \to \mathbb{R}$ on human pairwise preference data:

$$\mathcal{L}_{\text{RM}} = -\log\sigma(R_\phi(y_w) - R_\phi(y_l))$$

The problem: a scalar RM compresses multi-dimensional preference into one number. It loses the distinction between "content correct but prosody strange" and "content wrong but prosody natural."

#### Multi-Dimensional Preference Modeling

R1.5's GRM uses **rubric prompting** to have the LLM score each dimension separately:

```text
Evaluate the response using the following rubric (0-10 points):
1. Content correctness: is the answer accurate?
2. Fluency: is it coherent, with no stumbling?
3. Prosodic naturalness: do pitch and rhythm match how humans actually speak?
4. Emotional match: does the tone match the context's emotion?
5. Immersion: does it feel like talking to a person?

Response: [audio]
```

Each dimension is scored independently, and then a set of weights $w_k$ is learned to aggregate them:

$$R_{\text{prosody}}(y) = \sum_k w_k \cdot \text{GRM}_k(y), \quad w = \arg\min_w \|R_{\text{human}}(y) - \sum_k w_k \cdot \text{GRM}_k(y)\|^2$$

The weights are learned from human preferences via Bradley-Terry regression.

#### Direct Prosodic-Feature Reward

Besides preference modeling, you can also score prosody directly from acoustic features:

```python
def prosody_reward(audio):
    # Extract prosodic features
    f0 = extract_pitch(audio)          # fundamental frequency contour
    energy = extract_energy(audio)     # energy envelope
    duration = extract_durations(audio)  # phoneme durations

    # Compare against the reference (human) prosody distribution
    f0_score = -wasserstein(f0_dist(audio), f0_dist_human)
    energy_score = -wasserstein(energy_dist(audio), energy_dist_human)

    # Penalize monotonicity (guards against RLVR-induced flattening)
    f0_var = np.std(f0)
    monotonicity_penalty = -max(0, 0.2 - f0_var)  # penalize if f0 variance is too low

    return 0.5 * f0_score + 0.3 * energy_score + 0.2 * monotonicity_penalty
```

This kind of reward, built directly on the human prosody distribution, can suppress RLVR's flattening tendency even when no preference annotations are available.

### 30.4.3 Real-Time Responsiveness Reward

Real-time conversation requires first-packet latency under 1 s, with a reasonable overall response time. That brings latency into the reward:

$$R_{\text{latency}}(y) = \begin{cases}1, & T_{\text{first-packet}} < 0.5\text{s} \\ 0.5, & 0.5\text{s} \leq T_{\text{first-packet}} < 1.0\text{s} \\ 0, & T_{\text{first-packet}} \geq 1.0\text{s}\end{cases}$$

Or a continuous form:

$$R_{\text{latency}}(y) = \exp(-\alpha \cdot T_{\text{first-packet}})$$

The real-time reward conflicts with deep reasoning: the longer the model thinks, the later the first packet arrives. This is exactly where the [Dual-Brain Architecture](#_30-2-3-dual-brain-architecture-双脑架构) earns its keep: the expression brain can start synthesizing while the thinking brain is still working, hiding latency inside the generation pipeline.

### Combined Reward

The final audio RL reward is typically a weighted combination of all three:

$$R_{\text{total}} = w_c \cdot R_{\text{content}} + w_p \cdot R_{\text{prosody}} + w_l \cdot R_{\text{latency}}$$

The weights $(w_c, w_p, w_l)$ reflect the application: customer service weights content heavily ($w_c$ large), a companion robot weights prosody heavily ($w_p$ large), and real-time translation weights latency heavily ($w_l$ large). R1.5's core contribution is showing that **optimizing $w_c$ alone falls straight into the verifiable reward trap** — $w_p$ has to be part of the objective, or the real conversational experience doesn't survive.

## Section Summary

Audio reward design is far more involved than text reward design. Beyond content correctness, it has to account for prosody, emotion, accent, and speaking style. There are two engineering routes to a multi-dimensional reward: (1) weight together multiple RMs, or (2) use an LLM-as-Judge to evaluate overall quality directly. Step-Audio-R1.5 takes the second route, fusing audio understanding and evaluation into a single model.

The next section, [27.3 Multimodal Audio Agents and Future Directions](./future), moves further out toward the frontier — audio stops being just input and output, and becomes a tool an agent calls (voice search, voice translation, real-time conversation).
