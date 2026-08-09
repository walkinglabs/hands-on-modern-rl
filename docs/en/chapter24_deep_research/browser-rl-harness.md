# 22.1 Browser RL Action Space and Harness Engineering

> [24.1](./intro) covered the task definition for Deep Research and the mainstream models. But the moment you actually try to train a Deep Research agent, you run into two engineering problems: (1) **how do you design the action space**—a browser supports hundreds of possible operations, so which ones should you expose to the agent? (2) **how do you build the harness**—the actions the agent generates have to land on a real browser, which means you need a complete environment for execution, monitoring, and reward computation. This section works through both problems and gives you a reproducible engineering template.

## The Browser as an RL Environment

Deep Research's "environment" is the browser (or a search-engine API). From an RL perspective, this is a **partially observable, long-horizon, sparse-reward** MDP:

$$\mathcal{M}_{\text{browser}} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$$

- $\mathcal{S}$: the browser's state space, including the current URL, the DOM tree, visible text, scroll position, cookies/session, and so on
- $\mathcal{A}$: the action space (see below)
- $P$: the environment's transition function (determined by the real browser, unknown to the agent)
- $R$: a sparse binary reward, usually $r_T = \mathbb{1}[\text{answer is correct}]$, with intermediate steps $r_{t<T} = 0$
- $\gamma$: the discount factor; Deep Research tasks run $T = 20\text{–}100$ steps, and $\gamma = 1$ (undiscounted)
- $T$: the maximum number of steps (the budget), usually 30–50

Compared with the GUI MDP from [Chapter 23, Computer Use](../chapter25_computer_use/intro), Deep Research differs in a few key ways:

| Dimension                           | Deep Research                                      | Computer Use                       |
| ----------------------------------- | -------------------------------------------------- | ---------------------------------- |
| Observation space                   | DOM text / screenshots                             | Mostly screenshots                 |
| Action granularity                  | Abstract (search / click_link / extract)           | Atomic (pixel click / keypress)    |
| Predictability of state transitions | Fairly high (search results are relatively stable) | Low (GUI animations, popups)       |
| Reward sparsity                     | Extremely sparse (final step only)                 | Extremely sparse (final step only) |
| Typical episode length              | 20–50                                              | 50–500                             |

## Action Space Design and Three Mainstream Approaches

### Search API Abstraction

The simplest approach: don't expose a real browser at all, just give the agent a **search API**:

```python
ACTIONS = {
    "search":   {"query": str},          # call the search engine, return top-K results
    "visit":    {"url": str},            # fetch the plain text of a given URL
    "answer":   {"text": str},           # submit the final answer
}
```

This is the approach used by Search-R1 and R1-Searcher. Advantages:

- The action space has only 3 atomic operations, so it's easy to learn
- Each step's observation is clean Markdown, so no vision model is needed
- The engineering is simple—a single `requests.get()` handles it

Disadvantages:

- Can't handle pages that need JavaScript (SPAs, dynamically loaded content)
- Can't click, scroll, or paginate (only ever gets the first screen)
- Doesn't resemble what actually doing research on the web feels like

Good fit for: open-domain QA, academic paper retrieval, and other "mostly text" tasks.

### Real Browsers via Playwright

Use Playwright / Puppeteer to expose the full capability of a real browser:

```python
ACTIONS = {
    "goto":         {"url": str},
    "click":        {"selector": str},        # CSS selector or text match
    "fill":         {"selector": str, "value": str},
    "scroll":       {"dx": int, "dy": int},
    "back":         {},
    "extract_text": {"selector": str},        # extract text from a given element
    "screenshot":   {},
    "answer":       {"text": str},
}
```

This is the approach used by DeepResearcher and Tongyi DeepResearch. Advantages:

- Full real-browser capability, so it can handle any web page
- Can take screenshots as visual observations (for VLM agents)
- Closely resembles how a human actually does research

Disadvantages:

- The action space is large (7–10 actions), which needs more training data
- Real browsers are slow (1–3 seconds per step), so training cost is high
- CSS selectors fail often (page changes break selectors)

Good fit for: financial research, product comparisons, and tasks that need interactive pagination.

### Set-of-Mark Hybrid

Borrowing the SoM idea from [Chapter 23, GUI Grounding](../chapter25_computer_use/intro): at every step, number all interactive elements on the page, and the agent only needs to output a number:

```
Agent observes:
[page screenshot + numbering]
  [1] search box
  [2] "Next page" button
  [3] link to the first search result
  [4] link to the second search result
  ...

Agent action: click(3)  # click the first search result
```

This is the approach used by most SOTA systems in the BrowseComp benchmark. Advantages:

- The action space collapses to "pick a number," which is about as simple as it gets
- Doesn't depend on fragile CSS selectors
- Works with both VLMs (looking at the screenshot) and LLMs (looking at the numbered list)

Disadvantages:

- Needs OCR / DOM parsing to generate the numbering (an extra component)
- A numbering error is costly (the agent clicks the wrong link)

## The Five Core Modules of Harness Engineering

Whichever action space you choose, a Deep Research training harness needs the following five modules:

### Environment Wrapper

```python
class BrowserEnv:
    def __init__(self, mode='api' | 'playwright' | 'som'):
        self.mode = mode
        self.browser = None  # Playwright instance
        self.history = []    # trajectory history

    def reset(self, query: str) -> Observation:
        """Start a new trajectory, return the initial observation"""
        self.history = [{'role': 'user', 'content': query}]
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute an action, return (next_obs, reward, done, info)"""
        # 1. parse the action
        # 2. call the browser / API
        # 3. fetch the new observation
        # 4. decide whether done (agent called answer, or budget exceeded)
        # 5. compute reward (only nonzero when done, otherwise 0)
        ...
```

**Key engineering points**:

- **Timeout handling**: real web pages can hang, so you need a timeout (usually 10 seconds)
- **Error recovery**: CSS selector failures, network drops, JS errors—all of these need to be caught and turned into a friendly error observation
- **State persistence**: cookies/session must survive across steps (otherwise login state gets lost)

### Action Parser and Validator

The LLM's output is text, and it needs to be parsed into a structured action:

````python
def parse_action(output: str, mode: str) -> Action:
    """Parse an action from the LLM's output; return NoOp on failure"""
    try:
        if mode == 'api':
            # expected format: <action>search</action><query>...</query>
            return ApiAction.from_xml(output)
        elif mode == 'playwright':
            # expected format: ```python\nAction(...)\n```
            return PlaywrightAction.from_code(output)
        elif mode == 'som':
            # expected format: click(3)
            return SomAction.from_text(output)
    except ParseError as e:
        # parsing failed: return an error observation so the agent can retry
        return ErrorAction(f"Parse failed: {e}")
````

**Key engineering points**:

- **Format tolerance**: LLM output often has formatting errors, so the parser needs to be robust
- **Retry mechanism**: when parsing fails, return an error observation and let the agent self-correct (this is an important source of emergent behavior)
- **Action allowlist**: forbid dangerous actions (like `format_disk` or `rm -rf`) even if the agent tries to issue them

### Reward Verifier

Deep Research's reward measures **task completion**, and it needs to be designed per task type:

```python
class RewardVerifier:
    def __call__(self, query: str, answer: str, task_type: str) -> float:
        if task_type == 'qa':
            # answer matching (EM / F1 / LLM-as-Judge)
            return self.qa_score(query, answer)
        elif task_type == 'citation':
            # citation accuracy (CaRR metric)
            return self.citation_score(query, answer)
        elif task_type == 'multi_doc':
            # multi-document synthesis (needs LLM judging)
            return self.multi_doc_score(query, answer)
        elif task_type == 'browse_comp':
            # BrowseComp benchmark: exact string match
            return self.browse_comp_score(query, answer)
```

**Key engineering points**:

- **Mitigating reward sparsity**: you can add a process reward (PRM) as an auxiliary signal, but the primary reward stays end-to-end
- **LLM-as-Judge bias**: using GPT-4 / Claude as the judge introduces known biases (favoring longer answers, favoring its own style), which need to be calibrated for
- **Anti-cheating**: detect cheating strategies like the agent "restating the question" or "stitching together search snippets" as its answer

### Progress Tracker

Long-horizon tasks (30+ steps) need visible progress, otherwise you can't debug training:

```python
# a progress file in the style of claude-progress.txt
[2026-06-25 10:23:15] Step 1: search("2024 US GDP")
[2026-06-25 10:23:18] → Got 10 results, top: bea.gov
[2026-06-25 10:23:22] Step 2: visit("https://bea.gov/...")
[2026-06-25 10:23:25] → Page loaded, 15KB text
[2026-06-25 10:23:29] Step 3: extract("main table")
[2026-06-25 10:23:32] → Extracted table: 4 rows × 3 cols
[2026-06-25 10:23:36] Step 4: answer("2024 US GDP was $28.5T")
[2026-06-25 10:23:38] → Reward: 1.0 (correct)
```

This file serves two purposes:

1. **Debugging training**: for a failed trajectory, you can see at a glance which step went wrong
2. **Data synthesis**: successful trajectories can be used as SFT data

### Parallel Rollout Engine

A single Deep Research trajectory runs 30–50 steps at 1–3 seconds per step, so 60–150 seconds total. At a training batch size of 1024, running trajectories serially would take 25 hours per step. Parallelism is mandatory:

```python
async def parallel_rollout(
    agent, prompts: list[str], num_parallel: int = 256
) -> list[Trajectory]:
    semaphore = asyncio.Semaphore(num_parallel)

    async def rollout_one(prompt):
        async with semaphore:
            env = BrowserEnv(mode='playwright')
            obs = await env.reset(prompt)
            trajectory = []
            for t in range(MAX_STEPS):
                action = await agent.act(obs)
                next_obs, r, done, info = await env.step(action)
                trajectory.append((obs, action, r))
                if done:
                    break
                obs = next_obs
            return trajectory

    return await asyncio.gather(*[rollout_one(p) for p in prompts])
```

**Key engineering points**:

- **Browser pool**: reuse browser instances (startup overhead is large)
- **Network proxies**: avoid getting the target site's IP banned (use residential proxies)
- **Failure isolation**: one trajectory crashing shouldn't affect the others

## The Full Training Pipeline

Put the five modules together:

```
┌─────────────────────────────────────────────────┐
│ 1. Prompt Batch (1024 questions)                │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 2. Parallel Rollout (256 concurrent browsers)   │
│    ├─ Environment Wrapper (Playwright)          │
│    ├─ Action Parser (XML / code / SoM)          │
│    ├─ Progress Tracker (claude-progress.txt)    │
│    └─ Reward Verifier (QA / Citation / Browse)  │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 3. Trajectory Buffer                            │
│    {(s_t, a_t, r_t)}_{t=1..T} per trajectory    │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 4. GRPO Update                                  │
│    ├─ Group normalization (G=8 per prompt)      │
│    ├─ Advantage estimation                      │
│    └─ PPO-Clip policy update                    │
└─────────────────────────────────────────────────┘
```

In practice, on an 8×H100 GPU + 64-core CPU server, a single GRPO step processing 1024 prompts takes about 8–12 minutes. Training a 7B Deep Research model to convergence typically needs 5,000–10,000 steps, i.e., 4–7 days.

::: tip Connecting to [Chapter 16, GRPO](../chapter18_grpo/grpo-practice-and-mechanism)
Deep Research's RL training pipeline is not fundamentally different from the GRPO covered in [Chapter 16](../chapter18_grpo/grpo-practice-and-mechanism)—both are group-normalized advantage plus PPO-Clip. The differences are only in the environment (browser vs. text sandbox) and the reward (task completion vs. answer correctness). If you've already worked through [18.8, Financial API Tool-Calling GRPO](../chapter18_grpo/financial-tool-calling-grpo), moving to Deep Research only requires swapping out the `Environment Wrapper` and `Reward Verifier` modules.
:::

## Section Summary

Deep Research harness engineering comes down to **five modules**: environment wrapper, action parser, reward verifier, progress tracker, and parallel rollout. Of these, the **environment wrapper** and **reward verifier** are the hardest to reproduce—the former requires real-world browser engineering experience, and the latter requires task-specific verifier design.

The next section, [24.3, Evaluation Benchmarks and Open-Source Projects](./deep-research-eval), covers how to measure how good a Deep Research agent is—you'll find that evaluation itself is harder than training.
