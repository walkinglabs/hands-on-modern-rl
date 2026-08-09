# 27.2 Multi-Reward Video RLHF and Physically-Aware Generation

[Section 11.4, Visual Generation RL](./visual-generation-rl) covered the foundations of diffusion RL — algorithms like DDPO and DPOK. That section took an **algorithmic** perspective: how to model diffusion training as an MDP, and how to optimize it with policy gradients.

This section shifts the lens to the **industrial** side: how were the 2025-2026 video generation models — Seedance, LongCat-Video, Hailuo, Wan, Kling — actually trained with RL? These systems represent the industrial state of the art in video generation RL.

## 11.6.1 From Images to Video: New Challenges for RL

RL for image generation is already mature ([DDPO](./visual-generation-rl), DPOK). Video generation introduces a new set of challenges.

### Long Sequences

- Image: a single frame (1024×1024 pixels)
- Video: 30-300 frames (each 1024×1024) — 30 to 300 times the data volume of an image

This blow-up in sequence length makes credit assignment for RL extremely hard: in a 100-frame video, which frame, which pixel, is responsible for a problem?

### Temporal Consistency

A video needs more than each frame looking good on its own — it needs **consistency across frames**: the same character, the same scene, continuous motion.

```text
Image reward: per-frame quality (sharpness, aesthetics, prompt match)
Video reward: per-frame quality + temporal consistency + motion smoothness + physical plausibility
```

Video reward is far more complex than image reward.

### Compute Cost

- Image generation (diffusion): 50 denoising steps × one frame = a few seconds
- Video generation: 50 denoising steps × 100 frames = several minutes

RL training needs large numbers of rollouts. At several minutes per rollout, this pushes the training cost of video RL to more than 100x that of image RL.

### Scarcity of Reward Models

Image reward models have open-source options like [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) and [PickScore](https://arxiv.org/abs/2305.01569). Video reward models barely exist — labeling video preference data costs more than 10 times as much as labeling image preference data.

These challenges kept video generation RL moving slowly through 2024. The industrial breakthroughs of 2025 came mainly from two directions:

- **DanceGRPO**: bringing the GRPO idea to diffusion (image and video)
- **Seedance / LongCat**: RLHF-style training combined with engineering optimization

## 11.6.2 DanceGRPO and GRPO for Diffusion

[DanceGRPO](https://arxiv.org/abs/2505.07818) (ByteDance Seed, May 2025) is a major breakthrough in diffusion RL. Its core contribution is applying the GRPO idea directly to diffusion training.

### The Core Idea of DanceGRPO

Recall [Chapter 7, GRPO](../chapter18_grpo/grpo-practice-and-mechanism):

- Generate G rollouts for the same prompt
- Compute the reward for each rollout
- Normalize within the group to get the advantage
- No critic needed

DanceGRPO carries this idea over to diffusion:

```text
┌──────────────────────────────────────────────────────────┐
│ 1. For the same prompt, have diffusion generate G videos  │
│    (G is typically 4-8)                                   │
├──────────────────────────────────────────────────────────┤
│ 2. Score each video with a video reward model             │
├──────────────────────────────────────────────────────────┤
│ 3. Normalize within the group (subtract the mean,         │
│    optionally divide by std) to get the advantage         │
├──────────────────────────────────────────────────────────┤
│ 4. Update the diffusion model's parameters with a policy  │
│    gradient                                                │
└──────────────────────────────────────────────────────────┘
```

This pipeline is nearly identical to GRPO for LLMs. The only difference is what a rollout is made of:

- An LLM rollout is a token sequence
- A diffusion rollout is a denoising trajectory

### DanceGRPO versus DDPO

| Dimension            | DDPO                    | DanceGRPO                                                |
| -------------------- | ----------------------- | -------------------------------------------------------- |
| Advantage estimation | Single rollout + reward | Within-group normalization                               |
| Needs a critic       | No                      | No                                                       |
| Training stability   | Moderate                | Significantly improved                                   |
| Training efficiency  | Medium                  | High (group normalization strengthens the reward signal) |
| Applicable models    | Early diffusion models  | Modern video diffusion                                   |

The core advantages of DanceGRPO:

1. **Within-group normalization sharpens the reward signal** — comparing multiple videos from the same prompt makes it possible to tell which video is actually better
2. **No critic needed** — this saves a value model, just like GRPO
3. **Stable training** — within-group normalization makes the advantage estimate more stable

### DanceGRPO Experiments

ByteDance Seed used DanceGRPO to train several video generation models:

- **Image generation** (FLUX, SD3): aesthetic scores improved 15-20%
- **Video generation** (Wan, Seedance): dynamic quality improved 10-15%

In industry, DanceGRPO has already replaced DDPO/DPOK as the default choice for diffusion RL — the same position GRPO holds in the LLM world.

## 11.6.3 Seedance: ByteDance's Flagship Video Generation Model

[Seedance](https://seed.bytedance.com/) (ByteDance, released March 2025, upgraded to 1.0 Pro in October 2025) is one of China's state-of-the-art video generation models. It has repeatedly ranked first on VBench, the standard video generation benchmark.

### Seedance's Training Pipeline

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Large-Scale Video Pretraining                    │
│   - Hundreds of millions of video-text pairs               │
│   - Learn the basic distribution of video                  │
├──────────────────────────────────────────────────────────┤
│ Phase 2: High-Quality Data SFT                             │
│   - Filter for high-quality video (4K, professionally shot)│
│   - Teach the model what "high quality" looks like         │
├──────────────────────────────────────────────────────────┤
│ Phase 3: DanceGRPO RL                                      │
│   - Run RL with a video reward model                       │
│   - Optimize prompt following, dynamic quality, and         │
│     temporal consistency                                    │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Expert Iteration                                  │
│   - RL → collect new data → SFT → RL → ...                 │
│   - A data flywheel                                         │
└──────────────────────────────────────────────────────────┘
```

### Seedance's Reward Design

Seedance's reward is built from several components.

**Component 1: Prompt Following**

Does the video content match the prompt description? Scored with a video-text alignment model.

**Component 2: Aesthetic Quality**

Visual appeal — composition, color, lighting. Scored with an aesthetic model.

**Component 3: Motion Quality**

How natural the motion looks — do character and object movements follow physics? Scored with a motion model.

**Component 4: Temporal Consistency**

Consistency over time — are consecutive frames coherent? Scored with frame-to-frame similarity.

**Component 5: Human Preference**

Human preference — a reward model trained on RLHF preference data.

The final reward:

$$r_{\text{total}} = w_1 \cdot r_{\text{prompt}} + w_2 \cdot r_{\text{aesthetic}} + w_3 \cdot r_{\text{motion}} + w_4 \cdot r_{\text{temporal}} + w_5 \cdot r_{\text{human}}$$

The weights $w_1, \ldots, w_5$ are tuned via grid search.

### Seedance's Engineering Optimizations

**Optimization 1: Latent Diffusion**

Training happens in latent space (compressed with a VAE) rather than pixel space, which cuts compute substantially.

**Optimization 2: 3D Attention**

Attention operates in 3D (time × space) rather than within single frames, capturing temporal dependencies.

**Optimization 3: Classifier-Free Guidance**

During training, the prompt is randomly dropped (10-20% of the time) so the model also learns unconditional generation. At inference, a guidance scale controls how strongly the model follows the condition.

**Optimization 4: Flow Matching**

Flow matching replaces traditional diffusion — it's more stable and more efficient. This alternative to diffusion started gaining popularity in 2024.

### Seedance 1.0 Pro's Results

VBench rankings, October 2025:

| Model            | VBench Total |
| ---------------- | ------------ |
| Seedance 1.0 Pro | 86.7%        |
| Wan 2.5          | 84.2%        |
| Kling 2.0        | 83.1%        |
| Hailuo 02        | 81.5%        |
| Sora 2 (OpenAI)  | 80.8%        |
| Veo 3 (Google)   | 79.5%        |

Seedance is China's state of the art in video generation, ahead of both Sora 2 and Veo 3.

## 11.6.4 LongCat-Video: Efficient Long-Video Generation

[LongCat-Video](https://arxiv.org/abs/2510.22200) (Meituan, October 2025) is another important piece of work, focused on **long-video generation**.

### The Challenges of Long Video

Standard video generation targets 5-10 seconds. LongCat-Video aims for **30 seconds and beyond**, which brings new challenges:

- **Context blow-up**: the latent representation of a 30-second video is enormous
- **Narrative coherence**: a long video needs to tell a complete story, not just a fragment
- **Compute cost**: generating 30 seconds of video takes more than 6 times as long as generating 5 seconds

### The Design of LongCat-Video

**Design 1: Chunked Generation**

The long video is split into multiple 5-second chunks. Each chunk is generated independently, and coherence across chunks is maintained through an **overlap region**:

```text
Chunk 1: [0-5s]
Chunk 2: [4-9s]  ← overlaps Chunk 1 in [4-5s]
Chunk 3: [8-13s] ← overlaps Chunk 2 in [8-9s]
...
```

The generation results in the overlap region are averaged, which guarantees a smooth transition.

**Design 2: Story-Level Reward**

On top of frame-level reward, there's also a **story-level reward** — an LLM judges whether the video tells a coherent story.

```python
def story_reward(video, prompt):
    # Use an LLM to evaluate the video's narrative quality
    frames = sample_frames(video, n=10)
    description = vlm.describe(frames)
    story_quality = llm.judge_story(description, prompt)
    return story_quality
```

**Design 3: Hierarchical Diffusion**

Two levels of diffusion:

- **High level**: generates the video's "skeleton" (keyframes)
- **Low level**: interpolates the intermediate frames on top of that skeleton

This hierarchical structure follows the same idea as [DeepSWE's hierarchical RL](../chapter23_rl_based_swe/world-model-and-deep-swe).

### LongCat-Video's Results

LongCat-Video reaches state of the art in long-video generation:

| Model             | 30-Second Consistency | Narrative Coherence |
| ----------------- | --------------------- | ------------------- |
| Sora 2            | 65%                   | 60%                 |
| Veo 3             | 68%                   | 65%                 |
| Wan 2.5 Long      | 70%                   | 68%                 |
| **LongCat-Video** | **78%**               | **75%**             |

## 11.6.5 Hailuo: MiniMax's Video Generation

[Hailuo](https://hailuoai.video/) (MiniMax, released September 2024, upgraded to 02 in July 2025) is another Chinese state-of-the-art video generation model.

### Hailuo's Characteristics

- **Strong motion capture**: excels at character movement, dance, and sports scenes
- **Physics simulation**: models gravity, collisions, and fluids with relative accuracy
- **Open-source ecosystem**: some models are open-sourced (MiniMax-VL-01)

### Hailuo's Training Method

Hailuo follows a training pipeline similar to Seedance's:

- Large-scale pretraining
- High-quality SFT
- DanceGRPO-style RL
- Expert iteration

MiniMax's internal research — [CISPO](../chapter18_grpo/grpo-family), for instance — also contributed to Hailuo's training: CISPO's stability under low-precision training is what makes large-scale video RL feasible.

## 11.6.6 Other Mainstream Video Generation Models

### Wan (Alibaba)

[Wan](https://github.com/Wan-Video/Wan2.1) (Alibaba, February 2025) is the open-source state of the art in video generation. Wan 2.1 is open-sourced on HuggingFace and widely used by the community.

### Kling (Kuaishou)

[Kling](https://klingai.com/) (Kuaishou) — strong motion, strong physics simulation. Competes with Seedance across multiple benchmarks.

### Sora 2 (OpenAI)

[Sora 2](https://openai.com/sora/) (October 2025) — OpenAI's flagship video generation model. Known for long videos and strong physics simulation.

### Veo 3 (Google)

[Veo 3](https://deepmind.google/models/veo/) (May 2025) — Google's video generation model. Known for synchronized audio generation (joint video + audio).

## 11.6.7 The Industrial Landscape of Video Generation RL

As of mid-2026, here is the industrial landscape of video generation RL:

| Vendor         | Flagship Model(s)                 | Algorithm        | Characteristics                               |
| -------------- | --------------------------------- | ---------------- | --------------------------------------------- |
| ByteDance Seed | Seedance, LongCat                 | DanceGRPO        | China's SOTA, multiple parallel lines of work |
| MiniMax        | Hailuo                            | CISPO + GRPO     | Strong motion, open-source                    |
| Alibaba        | Wan                               | DanceGRPO        | Open-source ecosystem                         |
| Kuaishou       | Kling                             | Internal methods | Strong physics                                |
| OpenAI         | Sora 2                            | Undisclosed      | Long video                                    |
| Google         | Veo 3                             | Undisclosed      | Joint audio-video                             |
| Anthropic      | (does not build video generation) | -                | Focused on text                               |

A few things stand out:

- **Chinese vendors dominate video generation RL research** — they publish the most open papers
- **DanceGRPO is the mainstream algorithm** — an extension built on GRPO
- **Data and engineering outweigh algorithmic novelty** — most of the gains come from data quality and engineering optimization

## 11.6.8 Future Directions for Video Generation RL

### Longer Video

- Current SOTA: 30-60 seconds
- Future target: 5-10 minutes (short-film scale)
- Challenges: context, coherence, cost

### Joint Audio-Video Generation

- Current: audio and video are generated separately and composited afterward
- Future: joint generation with natural synchronization
- Challenges: multimodal RL, cross-modal consistency

### Interactive Video Generation

- Current: the full video is generated in one shot
- Future: users can intervene, edit, and steer generation
- Challenges: real-time RL, user-derived reward

### Controllable Generation

- Current: control is limited to text prompts
- Future: fine-grained control over pose, motion, camera, lighting, and more
- Challenges: multi-condition reward, control RL

### Physical Plausibility

- Current: physics is essentially a "hallucination" — the model paints from memory rather than simulating anything
- Future: genuine physics simulation
- Challenges: integrating with a physics engine, physics-based reward

## Summary

Video generation RL made major breakthroughs in 2025:

- **DanceGRPO** brought the GRPO idea to diffusion and became the mainstream algorithm
- **Seedance / LongCat** reached state of the art in video generation on the industrial side
- **Hailuo / Wan / Kling** together pushed Chinese video generation research into the lead

The core challenges of video generation RL — long sequences, temporal consistency, compute cost — are being worked out step by step through industrial practice. Going forward, 5-10 minute videos, joint audio-video generation, and interactive generation are the main directions.

This section and [Section 11.4, Visual Generation RL](./visual-generation-rl) form a complete picture:

- 11.4: algorithmic foundations (DDPO, DPOK)
- 11.6: industrial practice (DanceGRPO, Seedance, LongCat)

Together they cover the full picture of visual generation RL.
