# 20.1 Agentic RL Overview

[The chapter introduction](./intro) lays out the formal skeleton of Agentic RL—trajectories, POMDPs, a handful of credit-assignment formulas. This section brings those formulas back down to the concrete engineering picture: how an agent actually runs in a real environment, why training one is so much harder than training a single-turn LLM, and what frameworks industry currently uses to make it happen.

By the end of this section you should have a complete mental model of an "Agentic RL training system," and know which later section digs into which piece.

## The Paradigm Shift from Single-Turn to Multi-Turn

The reinforcement learning discussed in every earlier chapter is, at bottom, **single-turn decision-making**: the model receives a prompt, produces a complete response, a reward model assigns a score, and the policy updates once on that basis. Whether the underlying algorithm is PPO or GRPO, the skeleton of "one prompt, one response, one score" never changes.

Real agents do not work this way.

Consider a flight-booking agent. The user says, "Book me the cheapest early-morning flight from Beijing to Shanghai tomorrow." The agent has to act in steps: search for flights, compare prices and times, confirm seat availability, call the order API, and wait for the ticket to be confirmed. A mistake at any step—too broad a search query, picking the first result without comparing prices, misjudging inventory, malformed order parameters—fails the whole task. The environment gives a single binary signal at the very end: ticket issued (reward = 1) or not (reward = 0).

This shift from "one question, one answer" to "multi-step interaction with an environment" is exactly the core problem Agentic RL sets out to solve.

## Two Contrastive Trajectories

Same flight-booking task, same model, two rollouts:

```
Trajectory A (success)                   Trajectory B (failure)
─────────────────────────                ─────────────────────────
T1 search("Beijing Shanghai               T1 search("Beijing Shanghai flight")
   early morning cheap flight")              obs: 200 mixed results
   obs: 12 relevant flights

T2 filter(dep<9:00, sort=price)          T2 pick_first()
   obs: CA1501 6:30 ¥760                    obs: MU5101 9:30 ¥1280

T3 check_seat(CA1501)                    T3 order(MU5101)
   obs: seats available                     obs: order placed

T4 order(CA1501, seat=window)
   obs: ticket issued

reward = 1                               reward = 0
```

The two trajectories end with sharply different rewards, but **where exactly did it go wrong**? Did Trajectory B fail because T1's query was too broad, because T2 picked the first result without comparing prices, or because T3 placed the order without confirming? Looking at the final reward alone cannot answer that.

- For a precise formalization of "trajectory vs. single-turn completion," see [22.2 Multi-Turn RL Formalization](./formulation).
- For how to decompose the final reward back into per-step signals, see [22.3 Trajectory Credit Assignment](./credit-assignment).

## Basic Components of an Agent

An agent is more than an LLM. The minimal definition is **LLM backbone + instructions + tools + environment**—four pieces that cycle through the agentic loop.

### LLM Backbone

The decision-making core of the agent. It takes in the current observation, reasons about the next step, and produces an action—text or a tool call. Any sufficiently capable LLM can serve as the backbone, but in practice teams tend to pick a reasoning-trained model: it can emit a thinking trace before committing to an action, which is friendlier to multi-step decision-making.

### Instructions

Instructions tell the agent what problem to solve and what strategy to use. Beyond the task itself ("book the cheapest early-morning flight"), they also carry hints about how to solve it—"search first, then filter," "weigh both price and time," "retry on failure." The quality of the instructions sets a floor on how well the agent can behave.

### Tools and Environment

Tools are the agent's interface for acting on the environment: search APIs, code interpreters, CLIs, MCP servers, order APIs. Tool calls are usually marked with special tokens and embedded directly in the model's token stream:

```
<tool_call>{"name":"search_flights","args":{"from":"PEK","to":"SHA"}}</tool_call>
<tool_response>[CA1501 6:30 ¥760, CA1831 7:00 ¥690, ...]</tool_response>
```

The environment is stateful: search results change, inventory shifts, placing an order mutates a database. What a tool call returns depends not just on its arguments but on the environment's current state. This ability to anchor outputs to the real world rather than to parametric memory is called **grounding**—a major advantage agents hold over plain LLMs, and one of the core behavioral patterns RL training can instill.

### Agentic Loop

The four pieces cycle: **observe → reason about the next step → execute an action → receive a new observation**, until some termination condition is met—the task is done, the step budget runs out, or the model emits an end signal.

A complete loop is called a **rollout**; the full interaction record it produces is called a **trajectory**, written $\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)$. A trajectory is not just a text sequence—it mixes model-generated tokens, tool calls, tool returns, and environment state changes, structurally closer to a dialogue tree than to linear text.

## Four System-Level Challenges (from RAGEN)

System papers like RAGEN remind us that Agentic RL is not just "bolting GRPO onto tool calls"—it requires co-designing the environment, sampling, rewards, training stability, and evaluation all at once. [XiaoRed5's introductory guide](https://github.com/XiaoRed5/Agentic-RL-Most-Detailed-Intro) distills what sets Agentic RL apart from single-turn RL into four core challenges, each of which determines whether training actually learns anything.

### Challenge One: Long-Horizon Decisions—Early Actions Shape the Later State Distribution

The "long horizon" in Agentic RL is not just a surface fact about "trajectories getting longer"—it means **early actions change the distribution of later states**.

In the flight-booking example, T1's search query determines which flights the model even sees; T2's choice of flight determines what inventory needs checking next; whether T3 confirms determines whether T4 can place the order at all.

```
poorly written query  →  skewed search results  →  wrong evidence read  →  all later reasoning derailed
well-written query    →  finds the key source    →  rest is just verify-and-summarize
```

A small early mistake can get amplified downstream; a good early decision can just as easily fail to pay off because a later step stumbles. **The training signal usually arrives very late, but the decision that actually determined the outcome may have happened very early.** This is the root of the credit-assignment problem—see [22.3](./credit-assignment) for the details.

### Challenge Two: Environment Stochasticity Drives Up Reward Variance

When an agent interacts with an environment, that environment is not a perfectly stable text function. Search-engine results can change, web pages get updated, tool calls can fail, simulated environments can have their own randomness. Even with a fixed environment, sampling from the model alone produces different trajectories for the same task.

This creates a training problem: **for the same prompt, different rollouts can end up with wildly different final rewards.** One trajectory happens to surface the key evidence; another wanders into an irrelevant page. One answers correctly early; another takes a few extra detours and fails.

```
sample 8 rollouts of the same question   →  2 succeed (reward = 1) / 6 fail (reward = 0)
sample 8 more the next round             →  5 succeed (reward = 1) / 3 fail (reward = 0)
```

This kind of swing doesn't necessarily mean the model suddenly got better or worse—it can just be variance from sampling and environment feedback. So Agentic RL can't rely on a single reward curve; it also has to watch **reward variance, gradient spikes, whether the trajectory distribution is collapsing, and whether the model has fallen into some repetitive behavior pattern**. Work like AEM and RAGEN-2 attacks the problem from this stability angle.

### Challenge Three: Rollout Design—Three Overlooked Dimensions

In Agentic RL, a rollout is not simply "have the model generate a few more answers." It determines what states the model can explore, which behaviors it can compare, and whether the reward signal carries enough information.

**Initial-state diversity** matters. If training tasks are too similar to each other, the model can learn a fixed routine instead of general decision-making ability. A search agent trained over and over on the same phrasing of questions and the same site structure may just learn a templated query, rather than learning to design a search strategy around the actual information gap.

**Interaction granularity** matters too. When granularity is too coarse, a single action bundles too many decisions together, so when something goes wrong it's hard to tell which part is at fault. When granularity is too fine, trajectories get very long, training cost rises, rewards get sparser, and the model can burn its budget on meaningless micro-actions.

```
too coarse:  one action = search + read + judge + answer
             problem: after a failure, hard to tell which step was wrong

too fine:    one action = click a button / scroll the page once
             problem: trajectories get very long, raising both training cost and
                       credit-assignment difficulty
```

**Sampling frequency** affects learning as well. Sample just one rollout per task, and the model has little way to know "what would other actions at this same state have done"; sample too many rollouts per task, and cost climbs fast. In practice, the number of rollouts, the max number of interaction turns, the sampling temperature, and whether environment caches get reused all directly affect training stability and sample efficiency.

### Challenge Four: Pure Outcome Reward Teaches Shallow Strategies

Final-answer reward is genuinely useful—it's simple, cheap, and verifiable. **But if reward comes only from the final outcome, the model can pick up shortcuts that "look effective" without necessarily learning the agent reasoning we actually want.**

In search QA, for instance, the model may learn to favor high-frequency answers, or to answer early even when the evidence is insufficient. In web tasks, it may learn a fixed clicking pattern. In tool tasks, it may learn the form of calling a tool without actually using the observation to revise its plan.

```
on the surface:  the model searches, reads, and answers
what may actually be happening:
    the search query is templated / conflicting evidence goes unverified
    the observation never really enters later reasoning / the final answer is
    just a prior-driven guess
```

This is exactly why methods like PRM, SPA-RL, and IGPO matter so much in the credit-assignment chapter—at bottom, they're all trying to make the training signal track "which step actually moved the task toward completion."

## A Minimal Agent Loop

Reading about a concept ten times is worth less than running it once. Below we build a working agent in a few dozen lines of code—no RL training involved, just "how does an agent interact with tools." Once this loop makes sense, adding RL on top follows naturally.

```python
import json, subprocess, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# ① Define tools and tell the model "what you can do"
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command and return output",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

# ② The actual execution logic for tools (the environment)
def execute_tool(name, args):
    if name == "execute_bash":
        r = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr
    elif name == "read_file":
        with open(args["path"]) as f:
            return f.read()
    return f"Unknown tool: {name}"

# ③ Agent Loop: perceive → reason → act → observe, repeat
def run_agent(task, max_turns=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": task},
    ]
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  [Turn {turn+1}] Tool call: {tc.function.name}({args})")
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "(Max turns reached; stopping.)"

print(run_agent("List the .md files in the current directory and tell me how many there are."))
```

Sample output:

```
  [Turn 1] Tool call: execute_bash({'command': 'ls *.md'})
  [Turn 2] Tool call: execute_bash({'command': 'ls *.md | wc -l'})
There are 12 .md files in the current directory.
```

Mapping these 50 lines back onto the earlier concepts: `tools = [...]` is the **action space** $A_{\text{action}}$; `execute_tool()` is the **environment**; `for turn in range(max_turns)` is the **agentic loop / rollout**; `client.chat.completions.create()` is the **policy $\pi_\theta$**; `messages.append(...)` is the **state $s_t$**.

How "smart" this agent is depends entirely on the policy $\pi_\theta$. How to add RL training on top—how reward is computed (ORM vs. PRM), how reward is distributed across a multi-step interaction (credit assignment), how training data is managed—is what the following sections unpack.

## Search-R1: The Smallest Runnable Agentic RL Case

The agent loop above has no training loop attached. If you want to see a **genuinely runnable, minimal Agentic RL system**, [Search-R1](https://github.com/PeterGriffinJin/Search-R1) is the best place to start.

Search-R1 confines the task to a very small agent environment: the model just has to learn "when to search, what to search for, when to answer." What sets it apart from traditional RAG isn't "whether retrieval happens" but "**who decides to retrieve**"—traditional RAG has the system retrieve first and then hand documents to the model, while Search-R1 lets the model issue its own search action mid-reasoning.

```
RAG (system decides retrieval):
  question → retriever → documents → model answer

Search-R1 (model decides retrieval):
  question → model emits <search>query</search>
          → retriever returns documents
          → model continues or answers
```

In the code, four tag types spell out this closed loop:

- `<think>…</think>`: the model's internal reasoning, model-generated, **trained on**
- `<search>query</search>`: the model's action, model-generated, **trained on**
- `<information>docs</information>`: environment observation, returned by the retriever, **masked, not trained on**
- `<answer>final answer</answer>`: the model's final answer, model-generated, **trained on**

The core logic of a single rollout is short: the model pauses generation when it hits `</search>`, the system parses the query, calls the retriever, wraps the returned documents in `<information>`, and splices them back into the context. The model reads that observation and keeps generating, until it emits `<answer>` or hits the max number of turns.

**The single most important training detail is the mask.** `<search>`, `<think>`, and `<answer>` are model-generated tokens and can be optimized; `<information>` is text returned by the retriever, meant purely as context—the model should never learn to "generate search results." That is exactly what `state_masking=true` in Search-R1 handles.

You don't need to get lost in an elaborate tool system to read Search-R1. Its core point is clear: **once the search query becomes a model action and the retrieved result becomes an environment observation, RL training is no longer just optimizing an answer—it's optimizing a whole trajectory that calls tools.**

## The Limits of SFT and Prompting

A natural question: ReAct, Toolformer, and similar methods already let an LLM call tools. Why do we still need RL?

The key distinction: what SFT and prompting teach the model is **imitation**—copying the pattern of "when to call which tool" out of human demonstrations. But in real agent tasks, the optimal tool-use strategy depends heavily on context:

- How should the search query be constructed? When should it open a page for details? When should it stop searching and start summarizing?
- If tests still fail after a code edit, should it keep debugging or switch approach?
- When multiple sources disagree, which one should it trust?

These are fundamentally **policy-learning problems**, not plain language-modeling problems. Demonstration data can't cover every possible decision path, while RL can shape behaviors like tool use, planning, and memory management by working backward from task outcomes.

The division of labor between SFT and RL in agentic settings:

- **SFT teaches format**: the syntax of tool calls, the basic interaction protocol.
- **RL teaches strategy**: when to call a tool, how to compose multi-step actions, how to recover after a failure.

The DeepSeek-R1-Zero experiments show that skipping SFT and going straight to RL can also give rise to reasoning ability—provided the base model is already strong enough. In practice, the two-stage recipe of SFT warmup followed by RL fine-tuning remains the dominant paradigm.

## Industrial Framework Landscape

Back to reality—when you actually want to train an agent, what framework do you use to run all of this?

This question wasn't sharp back in the PPO and GRPO chapters: the training loop was almost pure GPU compute, and either TRL or OpenRLHF handled it easily. But the Agentic RL training loop adds a "wait" into the picture—when the model calls a search engine, the GPU has to wait for the results; when the model runs code, the GPU has to wait for the sandbox to return. How do you keep the GPU from sitting idle? That is the core problem Agentic RL training frameworks exist to solve.

Between 2025 and 2026, a wave of open-source frameworks emerged around this problem:

| Framework    | Developer                         | One-line description                                                                                               | Native multi-turn agent support | GitHub                                                    |
| ------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------- |
| **OpenRLHF** | Open-source community             | Most concise code (~8k lines); algorithm decoupled from agent execution; one line to switch single-turn/multi-turn | Yes                             | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| **verl**     | ByteDance / open-source community | Highest throughput; training and inference dynamically share the same GPU group; richest ecosystem                 | Basic, community-extending      | [verl-project/verl](https://github.com/verl-project/verl) |
| **slime**    | THUDM / Z.ai ecosystem            | Megatron + SGLang post-training framework; strong MoE efficiency                                                   | Basic                           | [THUDM/slime](https://github.com/THUDM/slime)             |
| **AReaL**    | Ant Group / Tsinghua              | Fully asynchronous training—GPU never waits; 2.77× speedup                                                         | Yes                             | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) |
| **ROLL**     | Alibaba Taotian                   | Reasoning (RLVR) + Agent dual mode; native Qwen support                                                            | Yes                             | [alibaba/ROLL](https://github.com/alibaba/ROLL)           |
| **SkyRL**    | UC Berkeley                       | Modular full-stack—training, agent orchestration, and task environments each independent                           | Yes                             | [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)   |
| **Seer**     | Moonshot AI (Kimi)                | Pushed synchronous—eliminates rollout long-tail via in-context learning; 74–97% throughput gain                    | No                              | see arXiv:2511.14617                                      |
| **Relax**    | Xiaohongshu                       | Fully multimodal (text + image + audio) asynchronous training                                                      | Yes                             | see arXiv:2604.11554                                      |
| **TRL**      | HuggingFace                       | Lightweight and easy to use; seamless HF ecosystem integration, but no large-scale async support                   | Mostly single-turn              | [huggingface/trl](https://github.com/huggingface/trl)     |

The core difference between these frameworks comes down to a single trade-off: **synchronous vs. asynchronous**. Synchronous training is simple, controllable, and easy to debug, but GPU utilization is low. Asynchronous training doubles throughput, but the training data may be generated from stale weights, which requires extra algorithmic compensation.

AReaL's research shows async training can push speed up nearly 3× without sacrificing quality—provided training is already working reliably. Seer takes the opposite extreme: it sticks with a synchronous framework and leaves GRPO untouched, instead eliminating rollout long-tail latency through in-context learning (divided rollout, context-aware scheduling, adaptive grouped speculative decoding), lifting throughput 74–97% while preserving the on-policy guarantee.

Another key difference: was the framework originally designed for single-turn RL (reasoning tasks), or did it plan for multi-turn agent interaction from the start? In the former, the agent-execution module was bolted on afterward—usable, but not optimized for the job. In the latter, agent execution is a first-class architectural citizen, with native support for state management, heterogeneous trajectory lengths, and asynchronous tool-call returns. OpenRLHF, AReaL, ROLL, and SkyRL fall into this second camp.

Framework choice comes down to the specific scenario. Just getting started and want a demo running fast—OpenRLHF has the most concise code and the best documentation. Enterprise-scale training (70B+)—verl's throughput and ecosystem advantages are clear. Model is an MoE architecture (GLM-4.5, Qwen3-30B-A3B, DeepSeek-R1)—slime's Megatron + SGLang native architecture is specifically optimized for MoE fp8 rollout and DeepEP communication. Chasing maximum throughput—AReaL's fully asynchronous mode gets you close to a 3× speedup. More engineering detail—sandbox management, environment construction, distributed deployment—is covered in [22.4 Tool-Use RL](./tool-use-and-trajectory).

## Section Summary

Agentic RL expands the object of training from "a single response" to "a complete interaction trajectory." That expansion raises four core questions, which the rest of this chapter takes up one by one:

- **Formalization**—how are trajectory, state, and action precisely defined in the multi-turn setting? How does the POMDP view distinguish model-generated action tokens from environment-returned observation tokens? → [22.2 Multi-Turn RL Formalization](./formulation)
- **Credit assignment**—when a trajectory ultimately fails, how do you decompose the reward back onto each step? What trade-offs do the dozen-plus methods—ORM, PRM, SALT, GiGPO, HGPO, SPA-RL, AgentPRM, ARPO, IGPO, StepPO—each make? → [22.3 Trajectory Credit Assignment](./credit-assignment)
- **Tool and trajectory engineering**—where does training data come from, how is tool-use strategy learned, how are sandboxes managed? → [22.4 Tool-Use RL](./tool-use-and-trajectory)
- **Real training pitfalls**—what traps has industry actually fallen into? → [22.6 Code Interpreter RL in Industrial Practice](./industrial-practice)

Next, formalization: translating "multi-turn interaction" into a mathematical object RL can operate on—[22.2 Multi-Turn RL Formalization](./formulation).
