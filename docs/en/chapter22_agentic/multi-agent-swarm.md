# 20.7 Multi-Agent Collaboration and Agent Swarms

> [22.6 Code Interpreter RL](./industrial-practice) trained a **single** agent to complete a programming task inside a tool-calling loop. But when the task escalates from "write a function" to "refactor the whole codebase + run tests + write docs + open a PR," a single agent's context window, attention bandwidth, and error-recovery capacity all get overwhelmed. **Multi-agent collaboration** is the key extension of agentic RL in 2025-2026: split a complex task across multiple agents, let each agent focus on one subtask, and coordinate them through an explicit communication protocol. This section covers three things: (1) the fundamental difference between LLM-era multi-agent systems and classical MARL; (2) the mainstream collaboration paradigms (Orchestrator-Worker, Debate, Swarm); (3) RL training methods for multi-agent systems.

## From Classical MARL to LLM-Era Multi-Agent Systems

[Chapter 12, Section 14.2](../chapter14_exploration_marl_hierarchical/marl) covered classical MARL: the CTDE framework, MADDPG, MAPPO. Those algorithms deal with **homogeneous** agents learning a **Nash equilibrium** in a **fixed** environment — think multiple robots chasing and evading each other, or multi-agent StarCraft micromanagement. LLM-era multi-agent systems are a different animal entirely:

| Dimension           | Classical MARL                      | LLM-era Multi-Agent                                         |
| ------------------- | ----------------------------------- | ----------------------------------------------------------- |
| Number of agents    | 2-20                                | 2-10 (cost-constrained)                                     |
| Agent heterogeneity | Homogeneous (shared policy)         | Highly heterogeneous (distinct roles: planner/coder/tester) |
| Communication       | Implicit, through environment state | Explicit, in natural language                               |
| Task type           | Zero-sum / cooperative games        | Long-horizon software tasks (PRs, research, operations)     |
| Training objective  | Nash equilibrium / team return      | Task completion rate (end-to-end verifiable)                |
| Training algorithm  | MAPPO / QMIX                        | GRPO + multi-trajectory reward assignment                   |

The most important difference is **heterogeneity** combined with **explicit communication**. In classical MARL, every agent shares the same policy $\pi_\theta(a \mid s)$ and agents only affect each other through the environment state. In LLM multi-agent systems, each agent runs a different system prompt ("you are a code reviewer," "you are a test engineer") and agents coordinate by exchanging **natural-language messages**. This blows up the communication bandwidth — a single round of coordination can burn thousands of tokens — but it also makes the semantics of collaboration far richer.

## Three Mainstream Architectures

### The Orchestrator-Worker Pattern

The **simplest and most widely used** collaboration paradigm. One **Orchestrator agent** handles task decomposition, subtask dispatch, and result aggregation; multiple **Worker agents** each execute one subtask.

```
[User: "Fix GitHub Issue #123"]

    ↓
[Orchestrator]
    ├── 1. Read the issue → call Worker-A: "locate the buggy file"
    ├── 2. Worker-A returns file.py:42
    ├── 3. Call Worker-B: "write a fix patch at file.py:42"
    ├── 4. Worker-B returns patch.diff
    ├── 5. Call Worker-C: "run tests + write changelog"
    └── 6. Aggregate → submit PR
```

Anthropic's internal 2025 research measured **the Orchestrator-Worker pattern accelerating SWE-bench Verified by 90.2% relative to a single agent, with success rate up 18-32%**. The key driver here is that **task decomposition keeps any single agent's context window from being overwhelmed**, more so than any inherent advantage of "two agents over one." A single agent handling the whole PR pipeline has its attention split across four things at once — finding the file, writing the code, running tests, writing docs. After decomposition, each worker only needs to focus on one of those.

The Orchestrator's policy can be formalized as a hierarchical MDP:

$$\pi_\theta^{\text{orch}}(w_t, m_t \mid q, h_{1:t})$$

where $w_t \in \{1, \ldots, K\}$ is which worker gets dispatched at step $t$, $m_t$ is the message sent to that worker, and $h_{1:t}$ is the interaction history so far.

### The Debate Pattern

Multiple agents **debate each other** to converge on a more reliable answer. Anthropic's AI Safety via Debate (Irving et al. 2018) is the theoretical basis for this paradigm; DeepMind's 2024 Scaling Inference paper validated LLM Debate's effectiveness on math problems.

The Debate MDP:

$$\pi_\theta(a_t^{(i)} \mid q, a_{1:t-1}^{(1)}, a_{1:t-1}^{(2)}, \ldots, a_{1:t-1}^{(K)})$$

The $i$-th agent sees the full history of every other agent's statements and outputs its response $a_t^{(i)}$ for this round. The final answer is chosen by an **external judge** — a human or another LLM.

The training objective of Debate is **convergence to the truth**: get the honest agent to win after multiple rounds of debate. This is much harder to train than Orchestrator-Worker — it requires **adversarial training**: deliberately train a "lying agent," then train an "honest agent" to defeat it.

### The Agent Swarm Pattern

**Kimi K2.5 (2026.01)** and **Step 3.7 Flash Advisor Mode** push multi-agent to the extreme: **dozens of heterogeneous agents** online simultaneously, dynamically scheduled by a meta-controller. This is essentially an A2A (Agent-to-Agent) protocol combined with an RL scheduler.

The key differences of Swarm:

- **An agent pool rather than a fixed worker set**: the meta-controller dynamically selects agents from the pool based on the task
- **An A2A communication protocol**: agents talk to each other through a structured protocol (e.g., Anthropic A2A, OpenAI Function Calls)
- **Cross-agent credit assignment**: which agent contributed the most? This requires SHAP or attention attribution

Formalized:

$$\pi_\theta^{\text{swarm}}(a_t \mid q, \text{pool}, h_{1:t})$$

where $a_t = (\text{select-agent}, \text{message}, \text{route-to})$.

::: warning Swarm's Cost Explosion
Swarm mode's token consumption runs 10-50x that of a single agent. The Kimi K2.5 paper reports an average of 280K tokens to process one SWE-bench task (the single-agent baseline is 18K). This is why Orchestrator-Worker remains the dominant industrial choice in 2026 — its cost is controllable and its results come close to Swarm's.
:::

## RL Training for Multi-Agent Systems

### From Team Return to Individual Attribution

The thorniest problem in multi-agent RL is **credit assignment**. The task succeeded — who deserves the reward?

**Option 1: Team-average return**

Every agent gets the same reward $r / K$ ($K$ is the number of agents):

$$R^{(i)} = \frac{1}{K} \sum_t r_t$$

Simple, but prone to the **free-rider** problem: a worker that slacks off can still see the team succeed and collect the same reward as everyone else.

**Option 2: Shapley value attribution**

The Shapley value from game theory measures each agent's marginal contribution:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(N - |S| - 1)!}{N!} [v(S \cup \{i\}) - v(S)]$$

where $v(S)$ is the success rate of subset $S$ completing the task on its own. The $N!$ term requires **counterfactual evaluation** — remove agent $i$ from the team and see whether the task can still be completed. This is expensive to compute, but it gives the fairest attribution.

**Option 3: Explicit Orchestrator allocation (heuristic)**

The Orchestrator outputs a weight $w_i$ against the final return, and agent $i$'s reward is $w_i \cdot R$:

$$R^{(i)} = w_i \cdot R^{\text{team}}, \quad \sum_i w_i = 1$$

This is the approach Kimi K2.5 actually uses in practice — it's cheap and interpretable, but it depends on the Orchestrator's ability to attribute credit correctly (which, in essence, means using RL to train the Orchestrator to learn attribution).

### Multi-Trajectory GRPO

Standard GRPO samples $G$ trajectories for the same prompt and normalizes the advantage:

$$\hat{A}_j = \frac{R_j - \text{mean}(R_{1:G})}{\text{std}(R_{1:G})}$$

The multi-agent version is called **Multi-Agent GRPO (MA-GRPO)**: each trajectory is generated by an **entire team collaborating**, rather than by a single agent. $G$ trajectories means $G$ rounds of team collaboration.

The key engineering implementation:

```python
def ma_grpo_step(prompts, team_size):
    # For each prompt, sample G team-collaboration trajectories
    trajectories = []
    for prompt in prompts:
        for g in range(G):
            # 1. Orchestrator decomposes the task
            subtasks = orchestrator.decompose(prompt)
            # 2. Workers execute in parallel
            worker_outputs = [workers[i](subtasks[i]) for i in range(team_size)]
            # 3. Orchestrator aggregates
            final_answer = orchestrator.aggregate(worker_outputs)
            # 4. Compute reward
            r = verifier(prompt, final_answer)
            trajectories.append({
                'prompt': prompt,
                'final': final_answer,
                'reward': r,
                'orch_logp': orchestrator.logp(...),
                'worker_logp': [w.logp(...) for w in workers]
            })

    # GRPO advantage normalization
    rewards = [t['reward'] for t in trajectories]
    advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

    # Compute loss separately for the orchestrator and the workers
    orch_loss = -mean(a * t['orch_logp'] for a, t in zip(advantages, trajectories))
    worker_losses = [-mean(a * lp for a, lp in zip(advantages, t['worker_logp']))
                     for t in trajectories]

    total_loss = orch_loss + sum(worker_losses)
    return total_loss
```

Three engineering details matter here:

1. **The Orchestrator and the Workers share the same advantage $a$** — team success or failure is a unified signal
2. **All three are updated together** (joint update), not alternately — this avoids non-stationarity problems
3. **Group-normalized advantages are essential** — otherwise one fast-learning agent's gradient can drown out the rest

## Kimi K2.5 and Step 3.7

### Kimi K2.5's Agent Swarm

Kimi K2.5 (2026.01, arXiv:2602.02276) is the first industrial model to publicly release training details for Swarm mode:

- **Agent pool**: 32 heterogeneous agents (coder, tester, planner, reviewer, debugger, etc.)
- **A2A protocol**: structured messages based on JSON Schema
- **Training data**: 12M team-collaboration trajectories, covering SWE / DeepResearch / Customer Service
- **Reward**: RLVR for verifiable tasks, LLM-as-Judge for open-ended tasks
- **Scheduling RL**: the meta-controller is trained with PPO, with the objective of minimizing token consumption while maximizing success rate

Reported metrics:

- SWE-bench Verified: 68.3% (single-agent baseline 49.1%)
- BrowseComp: 72.1% (single agent 51.4%)
- Average token consumption: 280K (baseline 18K, 15.6x)

### Step 3.7 Flash Advisor Mode

Step 3.7 Flash's Advisor Mode takes a different route: **a conservative Orchestrator-Worker setup**, plus an **Advisor agent** dedicated to "reflection and correction."

```
[Orchestrator] → [Worker-A: code] → [Advisor: review] → [Orchestrator] → [Worker-B: test]
```

The Advisor doesn't execute tasks directly — it only comments on Worker output. After seeing the Advisor's comments, the Orchestrator decides whether to send the work back for another pass. This "barbell-shaped" collaboration costs only 1/5 of Swarm, with results close to it.

Reported metrics:

- SWE-bench Verified: 62.4% (between single-agent and Swarm)
- Average token consumption: 52K (roughly 1/5 of Swarm)

## Echoes of [Chapter 30 Self-Play](../chapter32_selfplay/self-play-outlook/)

Multi-agent collaboration has a special case: **multiple agents that are just different instances of the same policy**, playing against each other. This is the core idea behind AlphaGo, AlphaZero, and Constitutional AI's Self-Critique. See [Chapter 30 Self-Play](../chapter32_selfplay/self-play-outlook/) for details.

The key distinction:

- **Multi-agent collaboration**: heterogeneous agents, explicit communication, team tasks
- **Self-play**: homogeneous agents (the same policy), interacting through the environment, zero-sum or cooperative games

The two are starting to converge in the LLM era — for example, Constitutional AI's Self-Critique can be seen as "two agents collaborating (one generates, one critiques), but sharing the same policy."

## Failure Modes of Multi-Agent Collaboration

With the theory out of the way, back to engineering — several typical failure modes multi-agent systems hit in production.

### Communication Amplifies Errors

When a single agent makes a mistake, it only affects itself. In a multi-agent system, one agent's erroneous output becomes another agent's input, and the error compounds exponentially.

```
Worker-A (wrong) → outputs "the bug is in file_X.py:42"
    ↓
Orchestrator dispatches Worker-B to fix file_X.py:42
    ↓
Worker-B fixes a bug that doesn't exist, introducing a new one
    ↓
Orchestrator dispatches Worker-C to test, finds the new bug
    ↓
...infinite loop...
```

Anthropic's internal data: the "cascading error rate" of multi-agent systems is 2.7x that of a single agent.

**Countermeasure**: attach a **confidence score** to every agent's output; low-confidence outputs trigger a second verification pass from the Orchestrator.

### Groupthink

When agents keep influencing each other, they can converge on a wrong consensus — especially in Debate mode. If one agent starts from a false premise, the other agents may accept it out of "politeness" or conformity.

**Countermeasure**: introduce a "Devil's Advocate" agent dedicated to pushing back on the mainstream view. Anthropic's Debate system forces at least one agent to hold a dissenting position.

### Free Riding

Under team-average reward allocation, a worker can learn to "contribute the bare minimum" — producing responses that look plausible but carry no real substance, while the team still succeeds overall.

**Countermeasures**:

- Shapley value attribution (expensive to compute)
- Explicit Orchestrator scoring (depends on the Orchestrator's competence)
- Evaluating each worker individually at test time (the strictest but the most costly)

### Context Duplication

In a multi-agent system, every worker needs to "understand the big picture" to do its job. But that global information — the task description, progress so far — gets repeated in every single worker's prompt, and the token cost explodes.

```
Task: "Fix GitHub Issue #123"
Context (seen by every worker):
  - Full issue description: 500 tokens
  - Relevant code files: 2000 tokens
  - Progress from other workers so far: 1500 tokens
Total: 4000 tokens × 5 workers = 20K tokens on context alone
```

**Countermeasure**: layered context — the Orchestrator maintains the full context, while each worker only sees a condensed summary.

## Open-Source Frameworks and Tools

If you want to reproduce multi-agent RL training, the following open-source tools are available:

| Framework        | Source      | Characteristics                                                          |
| ---------------- | ----------- | ------------------------------------------------------------------------ |
| **AutoGen**      | Microsoft   | Multi-agent conversation framework, supports several collaboration modes |
| **CrewAI**       | CrewAI Inc. | Role-based agents (planner/researcher/writer)                            |
| **MetaGPT**      | DeepWisdom  | SOP (standard operating procedure)-driven multi-agent framework          |
| **LangGraph**    | LangChain   | State-graph-based multi-agent orchestration                              |
| **Agency Swarm** | VRSEN       | An open-source implementation of "agent swarm" in the literal sense      |

Most of these frameworks are **inference-time** tools — they define how agents talk to each other, but don't touch RL training. **Open-source frameworks that can actually train multi-agent systems with RL are rare**; the main ones are:

- **OpenRLHF** (ByteDance-affiliated): supports multi-agent PPO/GRPO with customizable reward allocation
- **verl** (ByteDance-affiliated): a distributed RL framework supporting joint training of heterogeneous agents
- **OpenResearcher**: built for Deep Research, includes a simple Orchestrator-Worker setup

Industrial-grade Swarm training (like Kimi K2.5's) currently has **no complete open-source implementation** — this remains a core moat for top labs in China and the US.

## Summary

| Paradigm            | Communication       | Training objective            | Representative system | Cost   |
| ------------------- | ------------------- | ----------------------------- | --------------------- | ------ |
| Single agent        | N/A                 | Task completion rate          | baseline              | 1x     |
| Orchestrator-Worker | One-way dispatch    | Team return                   | Anthropic internal    | 3-5x   |
| Debate              | Two-way argument    | Convergence to truth          | Anthropic / DeepMind  | 5-10x  |
| Agent Swarm         | Fully connected A2A | Team + individual attribution | Kimi K2.5             | 15-30x |

The core RL training challenges for LLM-era multi-agent systems are **credit assignment** and **token cost**. The former determines whether training converges at all; the latter determines whether it can be commercialized. As of 2026, Orchestrator-Worker plus explicit attribution is the mainstream approach, while Swarm remains at the research stage.

The next chapter, [Chapter 21 Code Agent Reinforcement Learning](../chapter23_rl_based_swe/intro), applies this collaboration framework to SWE tasks — you'll see how SWE-Agent trains a single code agent with Orchestrator-Worker, and how DeepSWE trains multi-agent collaborative development with self-play.
