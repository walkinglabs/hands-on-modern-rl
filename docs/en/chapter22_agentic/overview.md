# 19.1 Foundations of Agentic RL

<ChapterNotebookLink chapter="agentic" locale="en" />

When models are connected to search, code execution, and external tools, a single decision can alter subsequent observations, thereby expanding the training objective to full trajectories. Part V begins with the fundamentals of Agentic RL, sequentially addressing multi-turn reinforcement learning, credit assignment, tool invocation, code and browser agents, and multi-agent collaboration.

**Core Content**

- Understand the fundamental differences between Agentic RL and single-turn RL: the training objective shifts from completion to trajectory, and rollout must be executed in the real environment.
- Master the formalization of multi-turn interaction as an MDP—joint state, structured actions, the POMDP perspective, and why it is not a simple extension of a single-turn MDP.
- Distinguish between ORM (outcome reward) and PRM (process reward), trajectory-level and step-level signals, and understand why credit assignment becomes more acute in multi-turn scenarios.
- Establish an engineering vision of Agentic RL training systems: asynchronous rollout, sandbox environments, heterogeneous trajectory lengths, and the core trade-offs between synchronous and asynchronous frameworks.

**Core Formulas**

$$
\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T) \quad \text{(trajectory of tokens, tool calls, and observations)}
$$

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle \quad \text{(POMDP: the agent observes only part of the state)}
$$

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right] \quad \text{(multi-turn objective over complete trajectories)}
$$

$$
A_t = R(\tau) - \bar{R}(s_t) \quad \text{(step-level advantage from a trajectory reward)}
$$

**The Role of Formulas in This Chapter**

Once a model is connected to tools, it is no longer just about writing a single response. It will first take an action, observe the environment's feedback, and then decide on the next step. Consequently, the object of reinforcement learning training also changes: **from generating a single response, it becomes completing a multi-step interaction process**.

The flight booking agent is an example of such a task. When a user says, "Help me book the cheapest early morning flight from Beijing to Shanghai tomorrow," the model cannot directly respond with "Your flight has been booked." Instead, it must first search for flights, then filter by time and price, confirm that the target flight still has available seats, and finally pass the correct passenger name, flight number, and cabin information to the booking interface. If an error is introduced in one of the earlier steps, the subsequent actions will continue to operate based on this incorrect state, potentially leading to a failed booking.

Therefore, during training, what we need to save is not the final response, but the complete process: what actions the model took in each state, and what observations the environment returned. This process is called a **trajectory**, denoted as:

$$
\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)
$$

Here, $s_t$ represents the state before the $t$-th step, $a_t$ is the action taken by the model, and $o_{t+1}$ is the observation returned by the environment after the action. It unifies dialogue, tool calls, and environment feedback into a single sequence. All subsequent discussions about trajectory probabilities, action masking, and step-by-step logging are built upon this representation.

A trajectory records what has happened, but it does not yet explain what the model can observe. When booking a flight, the model can read the list of flights from the search results page, but cannot access the full database of the airline; it can see the inventory information returned by the tool, but does not know whether the price will change in the next minute. The real environment has a complete state, but the model can only observe a part of it. This setting is represented by a **POMDP**:

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle
$$

Here, $S_{\text{agent}}$ is the state space jointly determined by the environment and the context. $A_{\text{agent}}$ contains the text and tool actions available to the model, $P_{\text{agent}}$ describes how an action changes the environment and produces the next observation, $R_{\text{agent}}$ gives the task reward, and $O$ denotes what the model can actually observe. The POMDP formulation makes one constraint explicit: **the policy acts on limited observations and interaction history, not on the complete state of the world**.

Once trajectories and partial observations enter the problem, the optimization objective must change as well. Single-turn RL maximizes the expected reward of one response:

$$\mathbb{E}_{a \sim \pi_\theta}[r(a)]$$

Agentic RL instead maximizes the cumulative return of the entire trajectory. What the model searches for at step $0$ determines which flights it observes at step $1$; the flight it selects at step $1$ determines which inventory it must inspect at step $2$. The objective is therefore

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$$

The notation $\tau \sim \pi_\theta$ means that the trajectory is sampled by letting the current policy interact with the environment. The factor $\gamma^t$ discounts delayed rewards, and $R(s_t,a_t)$ is the reward produced by the action at step $t$, including any final reward assigned back to that step. The training objective is now to **increase the total return of the interaction process**.

Trajectory return tells us what to optimize, but it does not yet tell us how each step should learn. A successful booking does not make every preceding action useful: the model may have taken a long detour and recovered only at the end. A failed booking does not make every action wrong either: the model may have found the correct flight and then supplied an invalid parameter when placing the order. Training needs to assign the final outcome back to individual steps so that it can strengthen useful actions and correct harmful ones.

This is the purpose of the **step-level advantage**:

$$
A_t = R(\tau) - \bar{R}(s_t)
$$

It compares the total reward $R(\tau)$ along the entire trajectory with the baseline return $\bar{R}(s_t)$ from the nearby state $s_t$, to determine whether the performance after the $t$-th step is above the usual level. The subsequent subsections discuss reward shaping, process rewards, round discounts, and intra-group advantage, all aiming to construct a more reliable $A_t$ so that the final reward can more accurately influence each action step.

This chapter continues to use the previously established MDP, policy gradient, GRPO, and verifiable reward, and re-examines them within the context of multi-round interaction tasks. When it comes to the Agent scenario, training is not only about updating the policy, but also involves handling asynchronous environments, sandbox management, inconsistent trajectory lengths, and long-term credit assignment in engineering issues.

## Chapter Outline

| Section                                                              | Central question                                                                                                       |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [19.1 Foundations of Agentic RL](./overview)                         | How does training change when a single response becomes a multi-step interaction?                                      |
| [19.2 Multi-Turn Reinforcement Learning](./formulation)              | How do we formalize joint states, structured actions, partial observations, step-level trajectories, and action masks? |
| [19.3 Trajectory Credit Assignment](./credit-assignment)             | When a multi-turn interaction fails, which actions should receive the blame?                                           |
| [19.4 Tool Use and Trajectory Generation](./tool-use-and-trajectory) | How are trajectories generated, tools invoked, sandboxes managed, and rollouts scheduled asynchronously?               |
| [19.5 Search-Augmented RL](./tool-use-agents)                        | How do systems such as DeepSeek-Researcher and Kimi-Researcher train search behavior?                                  |
| [19.6 Code Interpreter RL](./industrial-practice)                    | How do code agents handle instability, excessive trajectory length, and reward hacking?                                |
| [19.7 Multi-Agent Collaboration](./multi-agent-swarm)                | How do multiple agents divide roles, communicate, and assign group-level advantages?                                   |

## Learning Objectives

After reading this chapter, you should be able to:

- Formalize a multi-turn agent task using the POMDP six-tuple $\langle S, A, P, R, \gamma, O \rangle$, and identify the new elements introduced compared to a single-turn MDP (joint state, structured action, and external environment).
- Explain the trade-off between **ORM** and **PRM**: why verifiable tasks often work with outcome rewards, why complex reasoning may benefit from process rewards, and what intermediate approaches such as SALT and group-based methods try to solve.
- Understand the engineering skeleton of agentic RL training systems — the alternation between rollout and policy update, the cost and benefits of asynchronous training, and the design principles of sandbox environments — and be able to assess the applicability of frameworks such as OpenRLHF, verl, and AReaL accordingly.

The following concepts appear throughout the chapter. Review them first if needed:

- [GRPO and RLVR](../chapter18_grpo/rlvr) introduces verifiable rewards, a natural reward source for agentic tasks.
- [PPO and Reward Models](../chapter10_ppo/ppo-clip-objective) develops the policy-optimization framework used here.
- [The MDP Five-Tuple](../chapter03_mdp/mdp) provides the starting point for formalizing multi-turn interaction.

We now begin with the overall structure of an agentic RL system.

The previous sections provided the formal framework of Agentic RL, including trajectories, POMDPs, and credit assignment. This section brings these concepts down to the specific engineering landscape: how an Agent operates in a real-world environment, why training it is more challenging than training a single-turn LLM, and which frameworks are currently used in the industry for training.

After reading this section, you should be able to build a complete mental model of an "Agentic RL Training System," understanding where each subsequent section dives deeper into specific aspects.

## A Paradigm Shift from Single-turn to Multi-turn

All previous chapters of this textbook have discussed reinforcement learning in the context of **single-turn decision making**: the model receives a prompt, generates a complete response, the reward model provides a score, and the policy is updated once. Regardless of the underlying algorithm being PPO or GRPO, the "one question, one answer, one score" framework has remained unchanged.

But real-world agents do not operate this way.

Consider a flight booking agent. When a user says, "Help me book the cheapest early morning flight from Beijing to Shanghai tomorrow," the agent must take multiple steps: search for flights, compare prices and times, confirm seat availability, call the booking API, and wait for ticket confirmation. Any mistake during these steps—such as an overly broad search query, selecting the first option without price comparison, misjudging seat inventory, or incorrect booking parameters—will result in the entire task failing. The environment only provides a binary signal at the end: ticket success (reward = 1) or failure (reward = 0).

This shift from "one question, one answer" to "multi-step interaction with the environment" is precisely the core challenge that Agentic RL aims to address.

## Two Comparison Trajectories

The same flight booking task, the same model, two rollouts:

```
Trajectory A (success)                  Trajectory B (failure)
─────────────────────────              ─────────────────────────
T1 search("Beijing Shanghai early cheap") T1 search("Beijing Shanghai flights")
   obs: 12 relevant flights                obs: 200 mixed results

T2 filter(dep<9:00, sort=price)         T2 pick_first()
   obs: CA1501 6:30 ¥760                   obs: MU5101 9:30 ¥1280

T3 check_seat(CA1501)                   T3 order(MU5101)
   obs: seats available                     obs: order accepted

T4 order(CA1501, seat=window)
   obs: ticket issued

reward = 1                              reward = 0
```

Two trajectories end up with completely different rewards, but **where is the problem**? Does trajectory B fail because T1's query is too broad, or T2 directly selects the first option without comparison, or T3 places an order without confirmation? Only the final reward cannot answer this.

- [Section 19.2](./formulation) formalizes the distinction between a trajectory and a single-turn response.
- [Section 19.3](./credit-assignment) explains how to assign the final reward back to individual steps.

## Basic Components of an Agent

An agent is not just an LLM. Minimal definition: **LLM backbone + instruction + tool + environment**, the four components cycling within the agentic loop.

### LLM Backbone

The LLM is the agent's central decision-making component. It receives the current observation, reasons about the next step, and produces an action such as text or a tool call. In practice, reasoning-oriented models are often used because their training makes them better suited to multi-step decisions.

### Instructions

Tell the agent what problem to solve and what strategy to use. In addition to the task itself ("find the cheapest early morning flight"), it also includes solution strategy hints ("search first then filter," "consider both price and time," "retry if failed"). The quality of the instruction directly determines the lower bound of the agent's behavior.

### Tools and Environment

Tools are the interface through which an agent interacts with the environment: search API, code interpreter, CLI, MCP server, and order API. Tool calls are typically marked with a special token and embedded into the model's token stream:

```
<tool_call>{"name":"search_flights","args":{"from":"PEK","to":"SHA"}}</tool_call>
<tool_response>[CA1501 6:30 ¥760, CA1831 7:00 ¥690, ...]</tool_response>
```

The environment is stateful: search results vary, inventory changes, and placing an order modifies the database. The return of tool calls depends not only on the parameters but also on the current state of the environment. This ability to anchor outputs to the real world rather than parameter memory is called **grounding**—a significant advantage of agents over pure LLMs, and a core behavioral pattern that RL training can endow.

### Agent Loop

The four-step loop: **perception → reasoning → action → observation**, continuing until a termination condition is met (task completion, maximum step count, or a termination signal from the model).

A complete loop is referred to as a **rollout**; the full interaction record generated by a rollout is called a **trajectory**, denoted as $\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)$. Trajectories are not merely sequences of text; they combine model-generated tokens, tool calls, tool responses, and environmental state changes, structurally resembling a dialogue tree rather than a linear text.

## Four System-Level Challenges (from RAGEN)

The paper by RAGEN et al. reminds us that Agentic RL is not merely "applying GRPO to tool calls," but requires the simultaneous design of environment, sampling, reward, stable training, and evaluation.

[XiaoRed5's Introduction Materials](https://github.com/XiaoRed5/Agentic-RL-Most-Detailed-Intro)

The core challenges that distinguish Agentic RL from single-turn RL can be summarized into four points, each of which determines whether the training can truly learn effectively.

### Challenge 1: Long-Horizon Decision Making—Early Actions Shape Subsequent State Distributions

The "long-term" nature of Agentic RL is not merely about longer trajectories on the surface, but rather **earlier actions can alter the distribution of subsequent states**.

In the example of booking a flight, the search query at T1 determines which flights the model sees; the choice of a particular flight at T2 determines what inventory needs to be checked next; and whether the confirmation at T3 is successful determines whether an order can be placed at T4.

```
poor query → irrelevant results → misleading evidence → later reasoning drifts
good query → key source found   → verify the evidence → summarize the answer
```

An early small error may be amplified later; an early good decision may not lead to final success due to subsequent step errors. **Training signals often appear late, but the decisions that truly affect the outcome may occur at an early position.** This is the root of the credit assignment problem—see

[Section 19.3](./credit-assignment).

### Challenge 2: Environmental Randomness Causes Reward Variance to Surge

When an Agent interacts with an environment, the environment is not a completely stable text function. Search engine results may vary, web content may be updated, tool calls may fail, and simulated environments may have randomness; even if the environment is fixed, the sampling process itself can lead to multiple different trajectories for the same task.

This introduces a training challenge: **the final reward of different rollouts under the same prompt may vary significantly**. Some trajectories may just happen to find key evidence, while others navigate to irrelevant pages; some trajectories may answer correctly early, while others fail after taking more detours.

```
8 rollouts for the same task → 2 successes / 6 failures
another 8 rollouts          → 5 successes / 3 failures
```

This fluctuation does not necessarily indicate that the model has suddenly become stronger or weaker; it may simply be the result of sampling and environmental feedback variance. Therefore, Agentic RL cannot rely solely on the single reward curve; it must also pay attention to **reward variance, gradient spikes, whether the trajectory distribution has collapsed, and whether the model has fallen into a repetitive behavioral pattern**. Works such as AEM and RAGEN-2 have addressed this from the perspective of stability.

### Challenge 3: Rollout Design — Three Overlooked Dimensions

In Agentic RL, **rollout** is not simply "generating a few more answers" for the model. It determines what states the model can explore, which behaviors can be compared, and whether the reward signals are sufficiently informative.

**Initial state diversity** is crucial. If the training tasks are too similar, the model may learn a fixed pattern rather than developing general decision-making capabilities. For example, a search agent trained repeatedly on the same type of query and the same website structure may only learn a template-based query strategy, rather than learning how to design search strategies based on information gaps.

**Interaction granularity** is also important. When the granularity is too coarse, a single action may involve multiple decisions, making it difficult to identify where things went wrong. When the granularity is too fine, the trajectory becomes very long, increasing training costs, making reward signals sparser, and potentially causing the model to waste resources on meaningless micro-actions.

```
Too coarse: one action = search + read + judge + answer
            A failure cannot be assigned to a specific decision.

Too fine:   one action = one click or one scroll
            Trajectories become long and credit assignment becomes harder.
```

**Sampling frequency** also affects learning. If only one rollout is sampled per task, the model finds it difficult to understand "what would happen with other actions in the same state"; if too many rollouts are sampled per task, the cost will rise rapidly. In practice, the number of rollouts, maximum interaction steps, sampling temperature, and whether to reuse environment cache all directly impact training stability and sample efficiency.

### Challenge Four: Pure Outcome Reward Leads the Model to Learn Shallow Strategies

The final answer reward is very useful because it is simple, cheap, and verifiable. **However, if the reward only depends on the final outcome, the model may learn some "seemingly effective" shortcuts, rather than the reasoning we actually want the agent to learn**.

For example, in a search-qa task, the model may learn to prioritize generating high-frequency answers or to answer prematurely when evidence is insufficient; in a web task, it may learn certain fixed click patterns; and in a tool task, it may learn to invoke tools in a certain form without truly using observations to refine its own plan.

```
Observed behavior: the model searches, reads, and answers.
Hidden shortcut:   queries follow a fixed template; conflicting evidence is ignored;
                   observations do not affect later reasoning; the answer is guessed.
```

This is also why methods such as PRM, SPA-RL, and IGPO in the credit assignment chapter are so important—they essentially all aim to make the training signal more closely reflect "which step actually contributed to completing the task."

## A Minimal Agent Loop

We can now turn the components above into a small program. The following example implements a working agent loop without RL training, so the interaction between the model and its tools remains visible. The training loop introduced later builds on this same structure.

```python
import json, subprocess, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# Define Tools and Tell the Model "What You Can Do"

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

# The Actual Execution Logic of Tools (Environment)

def execute_tool(name, args):
    if name == "execute_bash":
        r = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr
    elif name == "read_file":
        with open(args["path"]) as f:
            return f.read()
    return f"Unknown tool: {name}"

# Agent Loop and the Cycle of Perception → Reasoning → Action → Observation

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

    return "Stopped after reaching the maximum number of turns."

print(run_agent("List the .md files in the current directory and count them."))
```

Running Effect:

```
  [Turn 1] Tool call: execute_bash({'command': 'ls *.md'})
  [Turn 2] Tool call: execute_bash({'command': 'ls *.md | wc -l'})
There are 12 .md files in the current directory.
```

## Minimal Agentic RL Example

The agent loop above does not include training. [Search-R1](https://github.com/PeterGriffinJin/Search-R1) provides a compact example of an agentic RL system that can actually run.

Search-R1 restricts the task to a very small agent environment: the model only needs to learn "when to search, what to search for, and when to answer." The difference between Search-R1 and traditional RAG is not "whether to retrieve," but rather "**who decides to retrieve**"—traditional RAG has the system retrieve documents first and then pass them to the model; Search-R1 allows the model to initiate a search action during the reasoning process.

```
RAG (the system decides when to retrieve):
  question → retriever → documents → model answer

Search-R1 (the model decides when to retrieve):
  question → model emits <search>query</search>
          → retriever returns documents
          → model continues or answers
```

The code implements this loop with four kinds of tags:

- `<think>...</think>` contains the model's reasoning.
- `<search>...</search>` requests a retrieval action.
- `<information>...</information>` contains documents returned by the environment.
- `<answer>...</answer>` marks the final answer.

The most important training detail is the action mask. Tokens emitted by the model, including the search request, reasoning, and final answer, participate in the policy loss. The retrieved text inside `<information>` is produced by the environment, so it must be excluded. Otherwise, the optimizer would treat an observation as though it were an action chosen by the policy.

## Why SFT and Prompting Are Not Enough

Methods such as ReAct and Toolformer already teach language models to call tools. This raises a natural question: what does reinforcement learning add?

Supervised fine-tuning and prompting teach the model to imitate patterns in demonstrations: when a tool was called and which tool was selected. In a real agent task, however, the best tool-use policy depends on the current state of the interaction:

- How specific should a search query be, and when should the model open a result or stop searching?
- If a test still fails after a code change, should the model keep debugging or try a different approach?
- When sources disagree, which evidence should the model trust?

These are policy-learning problems. Demonstrations cannot cover every possible decision path, whereas RL can use the final task outcome to shape tool use, planning, and memory management.

The two stages therefore have different jobs:

- **SFT teaches the protocol**: tool-call syntax and the basic interaction format.
- **RL teaches the policy**: when to call a tool, how to compose several actions, and how to recover after failure.

DeepSeek-R1-Zero shows that reasoning behavior can emerge from RL without an SFT stage when the base model is already strong. In practice, an SFT warm-up followed by RL fine-tuning remains the more common two-stage recipe.

## Industrial Framework Overview

The practical question is how to run this training loop efficiently. In PPO or GRPO reasoning tasks, most of the loop is GPU computation. Agentic RL adds external waiting time: a search request must wait for results, and a code action must wait for a sandbox to finish. A training framework must keep accelerators productive while these environments respond.

Several open-source frameworks now address this problem:

| Framework    | Developer                         | Main design point                                                              | Native multi-turn support                     | Repository                                                |
| ------------ | --------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------- | --------------------------------------------------------- |
| **OpenRLHF** | Open-source community             | Compact implementation with policy optimization separated from agent execution | Yes                                           | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| **verl**     | ByteDance and contributors        | High-throughput training and inference on a shared GPU pool                    | Basic support; ecosystem extensions available | [verl-project/verl](https://github.com/verl-project/verl) |
| **slime**    | THUDM and contributors            | Megatron plus SGLang, with efficient support for MoE post-training             | Basic support                                 | [THUDM/slime](https://github.com/THUDM/slime)             |
| **AReaL**    | Ant Group and Tsinghua University | Fully asynchronous training                                                    | Yes                                           | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) |
| **ROLL**     | Alibaba                           | Reasoning and agent training with native Qwen support                          | Yes                                           | [alibaba/ROLL](https://github.com/alibaba/ROLL)           |
| **SkyRL**    | UC Berkeley                       | Modular separation of training, agent orchestration, and task environments     | Yes                                           | [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)   |
| **Seer**     | Moonshot AI                       | Synchronous training with scheduling methods that reduce rollout tail latency  | No                                            | arXiv:2511.14617                                          |
| **Relax**    | Xiaohongshu                       | Asynchronous multimodal training for text, images, and audio                   | Yes                                           | arXiv:2604.11554                                          |
| **TRL**      | Hugging Face                      | Lightweight integration with the Hugging Face ecosystem                        | Primarily single-turn                         | [huggingface/trl](https://github.com/huggingface/trl)     |

The central design choice is synchronization. Synchronous training is easier to reason about and debug, but a slow environment can leave GPUs idle. Asynchronous training improves throughput by allowing rollout and optimization to proceed independently, although some trajectories may then be generated by older policy weights and require algorithmic compensation.

A second distinction is whether the framework was designed for multi-turn interaction from the beginning. A framework developed for single-turn reasoning can add an agent executor later, but state management, variable-length trajectories, and asynchronous tool responses remain secondary concerns. OpenRLHF, AReaL, ROLL, and SkyRL treat agent execution as a first-class part of the system.

Framework selection follows the workload. OpenRLHF offers a compact path for a first experiment. verl provides a broad ecosystem for large-scale training. For MoE models such as GLM-4.5, Qwen3-30B-A3B, and DeepSeek-R1, slime includes optimizations for FP8 rollouts and expert communication. AReaL is useful when asynchronous throughput is the primary constraint. [Section 19.4](./tool-use-and-trajectory) develops the corresponding engineering details, including sandbox management, environment construction, and distributed deployment.

## Summary of This Section

Agentic RL extends the training objective from "a single response" to "a complete interaction trajectory." This extension raises four core issues, which are addressed in the following subsections of this chapter:

- **Formalization** asks how trajectories, states, actions, and observations should be represented in a multi-turn setting. See [19.2 Multi-Turn Reinforcement Learning](./formulation).
- **Credit assignment** asks how a final outcome should influence individual actions. See [19.3 Trajectory Credit Assignment](./credit-assignment).
- **Tool and trajectory engineering** covers data generation, tool-use policies, sandboxes, and asynchronous rollouts. See [19.4 Tool Use and Trajectory Generation](./tool-use-and-trajectory).
- **Training failure modes** covers instability, excessive trajectory length, and reward hacking in code-agent training. See [19.6 Code Interpreter RL](./industrial-practice).

The next section turns multi-turn interaction into mathematical objects that RL can optimize: [19.2 Multi-Turn Reinforcement Learning](./formulation).
