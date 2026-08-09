# 23.1 GUI Agent Training in Practice

> [25.1](./intro) laid out the MDP formulation of Computer Use and the visual alignment problem behind GUI grounding. This section answers the next engineering question: **how do you actually train a VLM into a GUI agent?** That means data synthesis, curriculum design, reward engineering, and virtual environments — a full industrial pipeline. We'll use representative 2025–2026 work from Chinese labs as our throughline — UI-TARS-2, AutoGLM, MobileRL, ComputerRL, CogAgent — comparing the trade-offs across their technical approaches.

## A Wave of GUI Agent Work from Chinese Labs

In the second half of 2025, Chinese labs produced a concentrated wave of GUI agent RL training work. This wasn't an accident — three conditions matured at the same time:

1. **Mature VLM backbones**: open-source VLMs like Qwen2.5-VL, InternVL3, and GLM-4.5V provided high-quality starting points
2. **Virtual environment tooling**: benchmarks like Android Worldwide, AndroidWorld, OSWorld, and WebArena provided reproducible training and evaluation environments
3. **Falling compute cost**: 4090 / H100 prices stabilized, making RL training of 7B models affordable

Here's how the representative systems compare:

| Model            | Institution     | arXiv      | Scale    | Core Innovation                                                 |
| ---------------- | --------------- | ---------- | -------- | --------------------------------------------------------------- |
| **UI-TARS-2**    | ByteDance Seed  | 2509.02544 | 7B / 72B | End-to-end VLM + long-horizon task RL + reflection augmentation |
| **Open-AutoGLM** | Zhipu           | 2411.00820 | 9B       | Chinese/English GUI mix + mobile + fully open source            |
| **MobileRL**     | Tencent         | 2509.18119 | 7B       | Difficulty curriculum for mobile apps                           |
| **ComputerRL**   | Shanghai AI Lab | 2508.14040 | 7B       | Backward curriculum + intermediate exploration reward           |
| **CogAgent-9B**  | Zhipu           | 2408.16500 | 9B       | High-resolution visual encoding + dual-branch fusion            |

We break each of these down below.

## UI-TARS-2: The End-to-End RL Approach

UI-TARS-2 treats Computer Use as a **pure LLM RL problem** — a single VLM handles perception, reasoning, and action output all at once. There's no explicit planner/actor split in the architecture; everything happens inside one transformer.

### A Four-Stage Training Pipeline

```
Stage 1: Vision-language pretraining
  └─ GUI screenshot + text pairs → base visual capability

Stage 2: Supervised fine-tuning (SFT)
  └─ Human demonstrations + model self-generated trajectories → base action capability

Stage 3: Reflective RL (reflection augmentation)
  └─ Multiple candidate trajectories + verifier selection → rejection sampling + SFT

Stage 4: Online RL
  └─ Rollouts in real GUI environments → PPO optimizes task completion rate
```

The rejection sampling in Stage 3 is the key transition: the model generates $K=8$ trajectories for the same task, a programmatic verifier judges which ones succeeded, and the successful trajectories get fed back in as high-quality SFT data. This is more stable than jumping straight to online RL — at low success rates (<10%), online RL barely learns anything from the signal.

### Reflection Augmentation

The core innovation in Stage 4 is a reflection mechanism. When the agent fails, it's made to explicitly emit a `<reflection>` tag:

```
<thought>I need to click the "Submit" button</thought>
<action>click(450, 320)</action>
<observation>The button turned gray, but nothing navigated</observation>
<reflection>The click position was probably off. The clickable area for "Submit" is (440-470, 310-330), and I clicked outside that boundary. On retry, move toward the center.</reflection>
<action>click(455, 320)</action>
<observation>Page navigated to the success page</observation>
<action>done</action>
```

This kind of self-correction can't be learned from SFT alone — it needs RL's trial-and-error signal. During RL training, trajectories that "successfully correct after reflecting" get an extra +0.3 reward, which encourages the model to learn to reflect.

### The Multi-Task RL Reward

UI-TARS-2's total reward function:

$$r = r_{\text{task}} + \alpha \cdot r_{\text{format}} + \beta \cdot r_{\text{reflection}} - \gamma \cdot r_{\text{invalid}}$$

- $r_{\text{task}} \in \{0, 1\}$: whether the task was completed
- $r_{\text{format}} \in \{0, 1\}$: whether the output format is valid (XML tags closed, coordinates in range)
- $r_{\text{reflection}} \in [0, 0.3]$: quality of a successful corrective reflection
- $r_{\text{invalid}}$: performing an out-of-bounds action (e.g. trying to close the browser)

The weights used in practice are $\alpha = 0.1, \beta = 0.3, \gamma = 2.0$. $\gamma$ is deliberately large — the cost of one out-of-bounds action far outweighs the benefit of one completed task.

## Open-AutoGLM: A Complete Open-Source Pipeline

Zhipu's AutoGLM series (Open-AutoGLM open-sourced in December 2025) is optimized for **the Chinese internet environment** — English-trained models (Operator, Mariner) perform poorly on Chinese apps like Weibo, Taobao, and WeChat Mini Programs. Its training innovations include:

### Chinese GUI Data Synthesis

English models draw their data from Common Crawl plus RPA recordings, but Chinese GUI data is scarce. Open-AutoGLM's approach:

1. **WeChat Mini Program crawling**: use the Android automation framework Appium to drive 100+ real devices, autonomously exploring mini programs and recording screenshots plus actions at every step
2. **Chinese e-commerce task synthesis**: automatically generate task templates on Taobao/JD/Pinduoduo like "search for a product → compare prices → add to cart → checkout (without actually placing the order)"
3. **Chinese social tasks**: posting on Weibo, commenting on Douyin, saving to favorites on Xiaohongshu, and so on

The result is a collection of **2.3M Chinese GUI trajectories** — 2.9 times the size of the English trajectory set (800K).

### A Unified Cross-Platform Action Space

Open-AutoGLM's key design is **cross-platform unification** — the same model works on desktop browsers, Android apps, and iOS apps (via WebDriverAgent). The unified action space:

```python
UNIFIED_ACTIONS = {
    "tap":       {"x": float, "y": float},           # click / touch
    "long_press":{"x": float, "y": float, "ms": int},
    "swipe":     {"start": [x,y], "end": [x,y]},     # swipe / drag
    "type":      {"text": str},
    "key":       {"name": str},                       # back, home, enter
    "scroll":    {"dy": int},
    "wait":      {"ms": int},
    "done":      {"summary": str},
}
```

Desktop's "click" and mobile's "tap" are unified into the same `tap` action — the platform-specific semantic differences are handled by an environment adapter.

### Full Open Source

Open-AutoGLM open sources its **model weights, training data, environment simulator, and training scripts** in full — currently the most complete open-source GUI agent training framework available:

```bash
git clone https://github.com/zai-org/Open-AutoGLM
cd Open-AutoGLM

# 1. Download pretrained weights
huggingface-cli download zhipuai/Open-AutoGLM-9B

# 2. Start the Android emulator
bash scripts/start_emulator.sh

# 3. RL training (single node, 8×H100)
bash train.sh \
    --model Open-AutoGLM-9B \
    --algo grpo \
    --platform android \
    --tasks curated-1k.jsonl
```

Measured on 8×H100, a single GRPO step processing 256 prompts takes about 4 minutes. Training 5000 steps to convergence takes roughly 14 days.

## MobileRL: RL for the Mobile Setting

Tencent's MobileRL (arXiv:2509.18119) tackles mobile app automation specifically. Mobile is harder than desktop for three reasons:

- **Small screens, dense elements**: a single app's home screen might pack 30 clickable elements into a tight layout
- **Complex gestures**: long-press, swipe, pinch-to-zoom, 3D Touch — a far richer vocabulary than mouse clicks
- **Frequent app switching**: push notifications, incoming calls, and low-battery dialogs can interrupt a task at any moment

### A Progressive Difficulty Curriculum

MobileRL's core innovation is a **progressive difficulty curriculum**:

$$\text{Curriculum}(\pi_\theta) = \arg\max_{\text{task } \tau} \; \text{Difficulty}(\tau) \quad \text{s.t.} \quad 0.3 \leq P_\theta(\text{success} \mid \tau) \leq 0.7$$

The idea is to only sample tasks that fall inside the model's current 30%–70% success-rate range — its "zone of proximal development." This avoids tasks that are too hard (signal too sparse) and tasks that are too easy (no learning signal at all).

### Quantifying Task Difficulty

MobileRL defines task difficulty as a weighted sum across four dimensions:

$$\text{Difficulty}(\tau) = w_1 \cdot \text{Steps}(\tau) + w_2 \cdot \text{Apps}(\tau) + w_3 \cdot \text{GestureComplexity}(\tau) + w_4 \cdot \text{Distraction}(\tau)$$

- $\text{Steps}$: the minimum number of steps needed to complete the task (5–50)
- $\text{Apps}$: the number of apps that must be switched between (1–4)
- $\text{GestureComplexity}$: the number of distinct gesture types required (tap=1, swipe=2, long_press=3, multi-touch=5)
- $\text{Distraction}$: the number of simulated interrupting events (notifications, calls)

The weights used in practice are $w_1=0.4, w_2=0.2, w_3=0.2, w_4=0.2$.

### The Curriculum Scheduler

```python
class CurriculumSampler:
    def __init__(self, tasks, model):
        self.tasks = tasks
        self.model = model
        self.success_rate = {}  # task_id -> moving average success rate

    def sample(self, batch_size):
        # 1. Evaluate each task's success rate under the current model
        for tau in self.tasks:
            if tau.id not in self.success_rate:
                self.success_rate[tau.id] = self._estimate(tau)

        # 2. Filter down to tasks in the 30%-70% success rate band
        candidates = [t for t in self.tasks
                      if 0.3 <= self.success_rate[t.id] <= 0.7]

        # 3. Sample weighted by difficulty
        weights = [t.difficulty for t in candidates]
        return weighted_sample(candidates, weights, batch_size)

    def _estimate(self, task):
        # Run 10 rollouts to estimate success rate
        successes = sum(self._rollout(task) for _ in range(10))
        return successes / 10
```

Task success rates get re-estimated every epoch, so the curriculum keeps tracking the model's actual capability as it improves.

## ComputerRL: Backward Curriculum + Exploration Reward

Shanghai AI Lab's ComputerRL (arXiv:2508.14040) found that a pure task-completion reward gives too sparse a signal on long-horizon tasks (50+ steps). Its answer is a **backward curriculum plus intermediate exploration reward**.

### Backward Curriculum

A traditional curriculum runs easy to hard — learn 5-step tasks first, then 10-step, then 20-step. The backward curriculum reverses this: **start from the end of the task**.

Take a 50-step task $T = (s_0, a_1, s_1, \ldots, a_{50}, s_{50})$. The backward curriculum's training order is:

```
Round 1: start from s_49, only need to execute a_50 → done (a 1-step task)
Round 2: start from s_48, execute a_49, a_50 → done (a 2-step task)
Round 3: start from s_47, execute a_48, a_49, a_50 → done (a 3-step task)
...
Round 50: start from s_0, the full task (50 steps)
```

**Why does this work?** A backward curriculum guarantees that RL is always training on states "close to the reward." Under forward training, the agent sees no reward signal at all from $s_0$; under backward training, the agent gets a reward one step after starting from $s_{49}$. That makes credit assignment simple — the action just taken gets immediate feedback.

### Intermediate Exploration Reward

The backward curriculum solves the problem of "the terminal reward is too far away," but the intermediate steps still carry no signal. ComputerRL adds an **intermediate state reward**:

$$r_t = \underbrace{r_{\text{task}}(t=T)}_{\text{sparse terminal reward}} + \lambda \cdot \underbrace{r_{\text{progress}}(s_t, s_{t+1})}_{\text{dense progress reward}}$$

where $r_{\text{progress}}$ is produced by a separate "progress evaluator" LLM:

```python
def compute_progress_reward(s_t, s_{t+1}, task):
    prompt = f"""
    Task: {task}
    State before: {describe(s_t)}
    State after: {describe(s_{t+1})}
    Question: did the agent make progress toward the task?
    Answer with a score in [0, 1]:
    - 1.0: significant progress (e.g., filled a required field)
    - 0.5: minor progress (e.g., navigated closer)
    - 0.0: no progress (e.g., clicked irrelevant element)
    - -0.5: regression (e.g., closed important dialog)
    """
    return float(llm_judge(prompt))
```

This LLM-as-judge style of intermediate reward mirrors the idea behind the [Chapter 18 Process Reward Model](../chapter20_prm_search/inference-time-search) — using an LLM to evaluate the quality of intermediate steps.

### Comparison with a Forward Curriculum

The ComputerRL paper reports a comparison experiment:

| Method                                    | OSLevel-3 Success Rate | Avg. Steps | Training Cost |
| ----------------------------------------- | ---------------------- | ---------- | ------------- |
| Forward curriculum + terminal reward      | 12.3%                  | 47         | 1×            |
| Forward curriculum + progress reward      | 28.7%                  | 35         | 2.3×          |
| **Backward curriculum + progress reward** | **51.2%**              | **28**     | 2.8×          |

The backward curriculum pushes the success rate from 12% to 51%, but training cost also rises 2.8× — mostly from the overhead of calling the progress evaluator LLM.

## CogAgent: The Cost of High-Resolution Vision

Zhipu's CogAgent-9B (arXiv:2408.16500) takes a different route: **trade higher-resolution visual encoding for accuracy**.

### A High-Resolution Vision Branch

A standard VLM takes 448×448 input images; CogAgent uses 1120×1120 — 4× the pixels, which means 4× as many visual tokens, but that's enough to read small text on a UI (a size-9 font inside a table, a PowerPoint toolbar icon).

CogAgent's architectural trick is **dual-branch fusion**:

```
┌──────────────────────────────────────────┐
│ Input screenshot (1120×1120)              │
└────────────┬─────────────────────────────┘
             ↓
   ┌─────────┴─────────┐
   │                   │
   ↓                   ↓
High-res branch    Low-res branch
(EVA-CLIP)          (SigLIP)
1120×1120            448×448
→ 3136 tokens        → 256 tokens
   │                   │
   └─────────┬─────────┘
             ↓
        Cross-Attention
             ↓
         LLM Decoder
```

The low-resolution branch supplies global context ("this is a shopping page"), and the high-resolution branch supplies detail ("the cart button is in the top-right corner"). The two are fused through cross-attention, which avoids having the LLM process all 3136 tokens directly.

### The Accuracy vs. Latency Trade-off

The cost is computational: the high-resolution visual tokens make inference 3–5× slower.

| Configuration           | Visual Tokens | Inference Latency | OSWorld Accuracy |
| ----------------------- | ------------- | ----------------- | ---------------- |
| 448×448 single branch   | 256           | 0.8s              | 38.2%            |
| 1120×1120 single branch | 3136          | 4.2s              | 47.5%            |
| **Dual-branch fusion**  | 3392          | 1.6s              | **46.8%**        |

The dual-branch design comes close to the high-resolution accuracy while only doubling latency. This trade-off — accuracy vs. latency — is a central engineering decision in GUI agent design.

## Three Challenges in Production Deployment

Moving these systems from a paper to a production environment surfaces three challenges the papers don't fully discuss.

### Environment Distribution Shift

The training environments in these papers are controllable benchmarks like OSWorld and AndroidWorld. A production environment is a real user's computer — everyone's OS version, browser extensions, and font sizes differ.

**Mitigations**:

- **Data diversification**: UI-TARS-2 collected training environments across 50+ different Windows/macOS/Linux configurations
- **Domain randomization**: randomize the UI theme, font, and resolution during training
- **Continual learning**: collect failure cases post-deployment and periodically retrain

### Long-Tail Tasks

Paper benchmarks are all "mainstream tasks" (book a flight, check the calendar, write an email). In production, users ask things like "change this computer's BIOS to UEFI mode" — a task with almost no training data.

**Mitigations**:

- **Task tiering**: use the trained policy for common tasks; fall back to "tree search + LLM planning" for rare ones
- **Human-in-the-loop**: proactively ask the user when confidence is low

### Safety Boundaries

A GUI agent can perform destructive operations — deleting files, transferring money, sending emails. A production system needs clear safety boundaries.

**Mitigations**:

- **Action whitelisting**: block `rm -rf`, transfers over $100, and mass emails by default
- **Confirmation gates**: pop up a confirmation dialog before high-risk operations
- **Audit logging**: record every action so it can be traced back

See [25.3 Instruction Hierarchy and Prompt Injection Defense](./safety-swarm) for more.

## Section Summary

Chinese labs have converged on four distinct routes for GUI agent RL training:

- **UI-TARS-2**: end-to-end VLM + reflection augmentation, treating Computer Use as a pure LLM RL problem
- **Open-AutoGLM**: Chinese GUI data synthesis + cross-platform unification, the most complete engineering effort
- **MobileRL**: progressive difficulty curriculum, focused on mobile apps
- **ComputerRL**: backward curriculum + intermediate exploration reward, targeting long-horizon tasks
- **CogAgent**: high-resolution visual encoding, targeting small-text recognition

These four routes aren't mutually exclusive — UI-TARS-2 later added a reflection curriculum (an idea in the spirit of MobileRL), and Open-AutoGLM adopted a backward curriculum too (an idea in the spirit of ComputerRL). **Production-grade systems tend to combine multiple ideas at once.**

The next section, [25.3 Instruction Hierarchy and Prompt Injection Defense](./safety-swarm), turns to safety — once an agent is actually deployed on a user's computer, how do you defend against malicious web pages, spoofed UIs, and cross-application attacks trying to hijack it?
