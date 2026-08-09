# 24.2 Visual Reflection RL

This section covers two 2025 advances in multimodal RL:

- **Qwen3-VL's reflection mechanism**: lets a vision-language model explicitly reflect on visual content before answering
- **Audio RL (MGRD)**: multimodal reasoning in Step-Audio-R1

Together they mark multimodal RL's shift from "simple alignment" to "complex reasoning."

## 11.7.1 Qwen3-VL's Reflection Mechanism

[Qwen3-VL](https://arxiv.org/abs/2511.21631) (Alibaba, May 2025, released alongside Qwen3) is the vision-language counterpart in the Qwen3 series.

### Reflection in Visual Understanding

The traditional VLM (vision-language model) pipeline works like this:

```text
image + question → VLM → answer
```

The model produces an answer in a single forward pass, with no "thinking" step in between. That is fine for simple visual tasks — image classification, object recognition — but it breaks down on harder tasks like chart reasoning, geometric proofs, and visual math. The model misreads the image or misses a key detail.

Qwen3-VL addresses this with a **visual reflection mechanism**: the model explicitly reflects before answering.

```text
image + question
  → VLM looks: "I see in the image..."
  → VLM reflects: "Let me look again more carefully..."
  → VLM reasons: "Based on what I see, the question is..."
  → VLM answers
```

### Training Qwen3-VL

Qwen3-VL's training pipeline mirrors the text-only Qwen3 pipeline, with visual data layered in:

```text
Phase 1: multimodal pretraining (text + images)
Phase 2: multimodal SFT (visual QA, image captioning, geometric reasoning)
Phase 3: visual reasoning RL
  - math/geometry problems (with figures)
  - chart understanding
  - visual reasoning problems (pattern finding, spatial imagination)
Phase 4: general RLHF (dialogue quality + safety)
```

**The key data in Phase 3**:

- **Geometry problems**: math problems with geometric figures, which require looking at the figure before solving
- **Chart problems**: understanding bar charts, line graphs, tables
- **Visual reasoning**: Raven's Progressive Matrices, visual analogy tasks

This data teaches the model **joint vision-language reasoning**.

### Engineering the Reflection Mechanism

Qwen3-VL's reflection is implemented through **CoT prompting**:

```python
def qwen3_vl_inference(image, question):
    prompt = f"""
    Image: {image}
    Question: {question}

    Please think step by step:
    1. First, describe what you see in the image.
    2. Then, identify the key elements relevant to the question.
    3. Reason about the answer based on what you see.
    4. Verify your answer by re-checking the image.
    5. Provide the final answer.
    """

    response = model.generate(prompt)
    return response
```

This prompting scheme makes the model **reflect on the image explicitly**. During RL training, the model gets higher reward for "reflecting before answering," and this reinforces the reflection behavior.

### Qwen3-VL's Results

| Benchmark                       | Qwen2.5-VL | Qwen3-VL |
| ------------------------------- | ---------- | -------- |
| MathVista (visual math)         | 65.3%      | 78.2%    |
| MMMU (multimodal understanding) | 50.2%      | 58.7%    |
| DocVQA (document QA)            | 92.1%      | 95.4%    |
| ChartQA (chart understanding)   | 80.5%      | 87.3%    |

Qwen3-VL beats Qwen2.5-VL by a wide margin across visual reasoning benchmarks — the reflection mechanism buys more than 10 percentage points.

### What the Reflection Mechanism Means

1. **Visual understanding needs thinking too**: just like text reasoning, visual tasks benefit from CoT.
2. **Reflection is learned by RL**: this isn't prompt engineering — RL training is what internalizes the reflection behavior in the model.
3. **Multimodal reasoning RL is maturing**: the field has moved from "learning to look at images" to "learning to reflect on images."

## 11.7.2 Audio RL and Step-Audio-R1's MGRD

[Step-Audio-R1](https://arxiv.org/abs/2511.15848) (StepFun, November 2025) is a breakthrough in RL for audio — **Multimodal Generative Reasoning with Direct Preference Optimization (MGRD)**.

### The Challenge of Audio RL

Audio is a more complex modality than images:

- **Long time series**: a clip of audio can run from tens of seconds to several minutes.
- **Multiple information layers**: speech content, speaker identity, emotion, speaking rate, accent.
- **Expensive labeling**: audio preference annotation requires listening to the whole clip, which is slower than looking at an image.

Traditional audio models (Whisper, SpeechT5) handle a single task — speech recognition or speech synthesis, not both. Step-Audio-R1's breakthrough is joint training across **audio understanding + reasoning + generation**.

### Multimodal Generative Reasoning + DPO

The core idea behind MGRD:

```text
┌──────────────────────────────────────────────────────────┐
│ 1. Multimodal input                                       │
│    - audio (user speech)                                  │
│    - text (optional context)                              │
│    - image (optional visual context)                      │
├──────────────────────────────────────────────────────────┤
│ 2. Joint reasoning                                         │
│    - understand the audio content                         │
│    - identify speaker, emotion, intent                    │
│    - generate the reply content                           │
├──────────────────────────────────────────────────────────┤
│ 3. Multimodal output                                       │
│    - text reply                                            │
│    - speech synthesis (matched emotion, speaking rate)     │
├──────────────────────────────────────────────────────────┤
│ 4. RL training                                              │
│    - optimize multimodal output with DPO                   │
│    - preference data: good (audio + text) vs. bad (audio + text) │
└──────────────────────────────────────────────────────────┘
```

### MGRD's Training Data

Step-Audio-R1's training data includes:

- **Audio dialogue**: over 1 million turns of multimodal conversation.
- **Emotion annotation**: audio paired with emotion labels (happy, sad, angry, etc.).
- **Multiple languages**: Mandarin, English, regional dialects.
- **Professional domains**: customer service, education, healthcare, and other scenarios.

### How MGRD Relates to DPO

MGRD extends [DPO](../chapter17_dpo/dpo-theory-and-family) to the multimodal setting:

- DPO: trains text generation from text preference data.
- MGRD: trains multimodal generation from multimodal preference data.

MGRD's loss function follows the same form as DPO's:

$$\mathcal{L}_{\text{MGRD}} = -\log\sigma\left(\beta \log\frac{\pi_\theta(y_w^{\text{multi}} | x)}{\pi_{\text{ref}}(y_w^{\text{multi}} | x)} - \beta \log\frac{\pi_\theta(y_l^{\text{multi}} | x)}{\pi_{\text{ref}}(y_l^{\text{multi}} | x)}\right)$$

Here $y_w^{\text{multi}}$ and $y_l^{\text{multi}}$ are the winning and losing responses of a multimodal (audio + text) preference pair.

### Step-Audio-R1's Capabilities

Step-Audio-R1's capabilities in production:

- **Multi-turn spoken dialogue**: natural, fluent, emotionally expressive voice interaction.
- **Dialect understanding**: supports multiple Chinese dialects (Cantonese, Sichuanese, etc.).
- **Emotional feedback**: recognizes the user's emotion and matches the emotional tone of the reply.
- **Professional scenarios**: customer service, education, healthcare, and other vertical domains.

### What Audio RL Means

1. **Audio is the next RL battleground**: text RL has matured, image RL broke through in 2025, and audio RL is the new frontier for 2026.
2. **Multimodal joint training**: audio RL is never audio alone — it's audio joined with text and vision.
3. **Chinese labs are leading**: StepFun, ByteDance, and Alibaba are all investing heavily in audio RL.

## 11.7.3 The Industrial Landscape of Multimodal RL

By mid-2026, the industrial landscape of multimodal RL looks like this:

### Visual Understanding RL

| Lab       | Flagship Model  | Distinguishing Feature |
| --------- | --------------- | ---------------------- |
| Alibaba   | Qwen3-VL        | reflection mechanism   |
| ByteDance | Doubao-Vision   | visual reasoning       |
| Google    | Gemini 3 Vision | native multimodality   |
| OpenAI    | GPT-5 Vision    | general-purpose        |
| Anthropic | Claude Opus 4.6 | vision + agentic       |

### Visual Generation RL

(See [Section 11.6, Modern Video Generation RL](./video-generation-modern).)

### Audio RL

| Lab       | Flagship Model        | Distinguishing Feature    |
| --------- | --------------------- | ------------------------- |
| StepFun   | Step-Audio-R1         | MGRD multimodal reasoning |
| ByteDance | Doubao-Voice          | emotional speech          |
| Alibaba   | Qwen2-Audio           | audio understanding       |
| OpenAI    | GPT-4o Advanced Voice | real-time speech          |
| Google    | Gemini Live           | real-time multimodality   |

### VLA (Vision-Language-Action) RL

| Lab                   | Flagship Model      | Distinguishing Feature   |
| --------------------- | ------------------- | ------------------------ |
| Google                | Gemini Robotics 1.5 | embodied thinking        |
| Physical Intelligence | π0                  | general-purpose robotics |
| ByteDance             | RoboBrain           | Chinese SOTA             |
| Skild AI              | Skild Brain         | heavy-industry robotics  |

## 11.7.4 Shared Challenges Across Multimodal RL

The specific tasks differ, but multimodal RL faces a few challenges in common.

### Data Scarcity

- Visual RL: high-quality visual reasoning problems are scarce.
- Audio RL: audio preference annotation is expensive.
- VLA RL: collecting robot trajectory data is hard.

### Reward Design

- Visual RL: how do you automatically evaluate "image understanding"?
- Audio RL: how do you evaluate "speech emotion"?
- VLA RL: how do you evaluate "robot actions"?

### Long Horizons

- Visual RL: video generation (30+ frames).
- Audio RL: long conversations (dozens of turns).
- VLA RL: long trajectories (100+ steps of robot action).

All of these challenges point in the same direction — **stronger algorithms, finer-grained rewards, and longer context**.

## 11.7.5 Where Multimodal RL Is Headed

### Native Multimodal RL

The goal isn't "text RL plus multimodal SFT" — it's **multimodal RL from the ground up**. Llama 4's early fusion is a first step in this direction.

### Real-Time Multimodal RL

Real-time interaction — voice, vision, and action together — is at the core of the next generation of agentic RL.

### Cross-Modal Alignment

Getting the model to understand that "what the audio says" equals "what the image shows" equals "what the text describes" — semantic alignment across modalities.

### The Maturing of Embodied AI

VLA + world model + RL = a genuinely general-purpose robot. This is the central topic of [Chapter 12, Embodied Intelligence](../chapter28_vla/embodied-intelligence/).

## Summary

Advances in multimodal RL across 2025-2026:

- **Qwen3-VL**: a visual reflection mechanism that brings reasoning RL to vision.
- **Step-Audio-R1's MGRD**: multimodal audio reasoning combined with DPO.
- **Gemini Robotics 1.5**: the next step for VLA (see [Embodied Intelligence](../chapter28_vla/embodied-intelligence/)).

Multimodal RL is the natural extension of RL into the LLM era — from text to images, video, audio, and action. Each modality brings its own challenges, but the core RL ideas — policy optimization, reward design, credit assignment — carry across all of them.
