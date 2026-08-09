# Chapter 23 · Computer Use and GUI Agents

> [Chapter 20, Agentic RL](../chapter22_agentic/intro) taught the LLM to call tools, read tool outputs, and self-correct across multi-turn interactions — that's the single-agent form. But once the task moves from "write a function" to "book me a flight to Shanghai for next Wednesday on my computer," the agent has to cross a gap: **seeing the screen, clicking the mouse, and typing on the keyboard the way a human does**. This chapter covers two things: (1) under the Computer Use paradigm, how the agent maps the GUI pixel stream to atomic actions and optimizes them with RL (25.1–25.2); (2) GUI Agent training practice ([25.2](./training)) and safety defenses ([25.3](./safety-swarm)).

## 25.1 The Computer Use Paradigm

The tools in [Chapter 20, Tool Use](../chapter22_agentic/tool-use-and-trajectory) were **structured APIs** — `def search(query): return results`, with inputs and outputs both strings. But in the real world, a huge amount of software exposes only one interface: the **GUI**. Browsers, Excel, internal corporate OA systems, Photoshop, games — none of them has a public API, only a screen and mouse/keyboard events.

The **Computer Use** paradigm treats the entire operating system as the agent's environment:

- **Observation**: a screenshot $o_t \in \mathbb{R}^{H \times W \times 3}$ (1–4 frames per second)
- **Action**: an atomic GUI event (mouse move, click, scroll, key press, wait)
- **Reward**: a binary signal for task completion ("did it succeed in booking the flight")

This MDP looks nothing like the classic RL benchmarks. CartPole has a 4-dimensional state, a 2-dimensional action space, and dense per-step reward. Computer Use has a state space of millions of pixel dimensions, a mixed-type action space, and reward so sparse it only arrives on the final step.

### Mainstream Products

| Product             | Organization   | Released | Characteristics                                             |
| ------------------- | -------------- | -------- | ----------------------------------------------------------- |
| **Computer Use**    | Anthropic      | 2024.10  | Claude 3.5 Sonnet natively supports screenshot-action pairs |
| **Operator**        | OpenAI         | 2025.01  | CU Agent + GPT-4o vision, browser-specific                  |
| **Project Mariner** | Google         | 2024.12  | Gemini-driven, deeply integrated with Chrome                |
| **UI-TARS-2**       | ByteDance Seed | 2025.09  | End-to-end VLM + RL training                                |
| **Open-AutoGLM**    | Zhipu          | 2025.12  | Open-source AutoGLM upgrade                                 |

### The Core Action Space

Anthropic's Computer Use defines its action primitives as follows (OpenAI's Operator and Google's Mariner are much the same):

```python
ACTIONS = {
    "click":      {"x": int, "y": int, "button": "left|right|middle"},
    "double":     {"x": int, "y": int},
    "drag":       {"start": [x,y], "end": [x,y]},
    "type":       {"text": str},
    "key":        {"keys": "ctrl+c|enter|tab"},   # key combination
    "scroll":     {"x": int, "y": int, "dy": int},
    "wait":       {"ms": int},
    "screenshot": {},
    "done":       {"summary": str},
}
```

Three design choices stand out:

1. **Actions mix discrete tokens with continuous coordinates** — `click` requires both picking a token and predicting $(x, y)$. This is something LLMs don't handle naturally: a standard transformer outputs discrete tokens, while $(x, y) \in [0, W] \times [0, H]$ is continuous-valued.
2. **Screenshot frequency is far below the human eye's** — humans perceive 30–60 frames per second, Computer Use only 1–4. That means the state transition $P(s_{t+1} \mid s_t, a_t)$ hides a lot of state change between two consecutive observations.
3. **The wait action (`wait`)** — GUI animations, network loading, and popup transitions all require waiting. This is an action classic RL doesn't have: deliberately spending a timestep.

### MDP Formalization

Define the Computer Use MDP as $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$:

$$\mathcal{S} = \{\text{screenshots}\}, \quad \mathcal{A} = \{\text{click, type, scroll, key, wait, done}\}$$

The task description (e.g., "convert this PDF to Markdown for me") is prepended as an initial prompt $q$ to every step's observation. The policy is a conditional distribution:

$$\pi_\theta(a_t \mid q, o_{1:t}, a_{1:t-1})$$

The reward $R$ is typically sparse and binary: $r_T = \mathbb{1}[\text{task completed}]$, with $r_{t<T} = 0$ at every intermediate step. That makes credit assignment brutally hard — a single browser-automation task might take 50 steps, and only the last one gets a reward, leaving no way to tell which earlier steps were right and which were wrong.

::: warning RL's Real Difficulty
Sparse reward + long horizons (50–500 steps) + high-dimensional observations (1344×756-pixel screenshots) + a mixed action space — Computer Use hits every RL pain point at once. That's why almost every Computer Use system before 2024 was **pure prompt engineering**, and RL training only reached real industrial deployment in 2025.
:::

## GUI Grounding RL

The first challenge in Computer Use isn't decision-making, it's **localization**: how does the model know where on the screen — which $(x, y)$ — the "Submit" button is?

### Set-of-Mark Prompting

Yang et al. 2023 proposed **Set-of-Mark (SoM)** prompting: first use OCR / object detection to draw a box around every interactive element on the screen and number them $1, 2, \ldots, K$; the agent then only needs to reference the number when it outputs an action:

```
[screenshot + box 1: input field "username", box 2: input field "password", box 3: button "login"]

Agent: type("alice") → click(box 1) → type("***") → click(box 2) → click(box 3)
```

This reduces continuous coordinate prediction to a **discrete choice** — at the cost of depending on an external detector, and when the detector misses an element, the agent is helpless.

### Visual Grounding

UI-TARS, CogAgent, and other end-to-end models take a different route: **have the VLM output coordinates directly**. The model architecture splits into two heads:

$$\text{VLM}(o_t, q) \to \underbrace{(\text{thought}, \text{action token})}_{\text{language head}} + \underbrace{(x, y) \in [0,1]^2}_{\text{grounding head}}$$

The grounding head is usually an MLP that outputs normalized coordinates $(x, y) \in [0, 1]^2$, which are then multiplied by the screen dimensions to map to pixels.

Grounding is trained with **supervised imitation**: humans label the "button center point $(x_i, y_i)$," and the loss is:

$$\mathcal{L}_{\text{ground}} = \frac{1}{N}\sum_i \|\hat{p}_\theta(o_i) - p_i\|_2^2$$

But pure supervision has a problem: **the model can end up pointing at empty space**. Supervision only teaches "where the button is," not "the button needs to actually get pressed." This is where RL comes in.

### Joint RL over Grounding and Decision-Making

Put grounding and action selection into the same PPO objective:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right] - \beta \cdot \mathcal{L}_{\text{ground}}(\theta)$$

The second term is the grounding supervised loss, kept as a regularizer. This **joint SFT + RL training** is the standard recipe for GUI Agents — imitation learning first to acquire basic operation skills, then RL to optimize task success rate.

UI-TARS-2 pushes this idea to the extreme: it outputs the chain of thought, the action, and the coordinate as a **single sequence**, and optimizes all three with RL simultaneously:

```python
def ui_tars_forward(self, screenshot, task):
    # Encode the image
    visual_tokens = self.vision_encoder(screenshot)  # [B, N_vis, d]

    # Assemble the prompt
    prompt = f"<task>{task}</task>\n<image>{visual_tokens}</image>\n"

    # Autoregressively generate thought + action + coord
    # Key detail: coord is wrapped in special tokens <coord_x> <coord_y>
    output = self.llm.generate(prompt, max_new_tokens=256)

    # Parse the output: "<thought>...</thought>\n<action>click</action>\n<coord>(0.45, 0.62)</coord>"
    thought, action, coord = parse_action(output)
    return thought, action, coord
```

### Generating RL Training Data

Real GUI tasks can't be labeled by hand at scale — a single 50-step browser task costs roughly 30 minutes to demonstrate manually. The solution is **programmatic task generation**:

1. **Crawling real websites**: UI-TARS collects 200+ real apps and auto-generates 1,000+ task templates per app
2. **Environment snapshots**: record human operation sequences, saving each step's screenshot + action as SFT data
3. **Task verifiers**: programmatic rules check whether a task completed ("did a success message appear on the page")
4. **RL rollout**: the agent executes the task in a virtual machine, and the verifier produces the final reward

```python
class GUIEnv:
    def reset(self, task_id):
        self.vm.restore_snapshot(task_id)  # Restore the VM to the task's initial state
        self.task = self.tasks[task_id]
        return self.screenshot()

    def step(self, action):
        self.vm.execute(action)            # Inject mouse/keyboard events
        obs = self.screenshot()
        done = self.task.verifier(obs, self.vm.state)
        reward = 1.0 if done else 0.0
        return obs, reward, done, {}
```

::: details Why Not Use the Real Mouse
Directly controlling the operating system's mouse would put the agent in conflict with human user input. Industrial practice runs the agent inside a **virtual machine + VNC remote desktop**, injecting mouse and keyboard events over the RDP/VNC protocol so the agent stays isolated from the human user. This is also why Computer Use systems typically manage only 1–2 actions per second — the latency of screenshotting plus VNC injection.
:::

## Section Summary

Computer Use treats the GUI pixel stream as the RL state space and mouse/keyboard events as the action space, which amplifies every classic RL difficulty — sparse reward, long horizons, high-dimensional observation — all at once. **Set-of-Mark** and **visual grounding** are the two mainstream routes to solving the "localization" problem: the former relies on an external detector to simplify the action space, the latter has the VLM output coordinates end to end.

The next section, [25.2 GUI Agent Training Practice](./training), goes into industrial practice — you'll see how systems like UI-TARS-2, AutoGLM, MobileRL, and ComputerRL turn this theory into a reproducible training pipeline.
