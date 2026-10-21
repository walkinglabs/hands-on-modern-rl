# 19.1 Agentic RL 基础

<ChapterNotebookLink chapter="agentic" />

Part IV 训练模型生成更符合偏好的单段回答。模型接入搜索、代码执行和外部工具后，一次决策会改变后续观测，训练对象随之扩展为完整轨迹。Part V 从 Agentic RL 基础开始，依次处理多轮强化学习、信用分配、工具调用、代码与浏览器智能体，以及多智能体协作。

**核心内容**

- 理解 Agentic RL 与单轮 RL 的根本差异：训练对象从 completion 变成 trajectory，rollout 必须在真实环境里执行。
- 掌握多轮交互的 MDP 形式化——联合状态、结构化动作、POMDP 视角，以及它为什么不是单轮 MDP 的简单扩展。
- 区分 ORM（结果奖励）与 PRM（过程奖励）、trajectory-level 与 step-level 信号，理解信用分配在多轮场景中为何变得尖锐。
- 建立 Agentic RL 训练系统的工程图景：异步 rollout、沙箱环境、异构轨迹长度，以及同步与异步框架的核心取舍。

**核心公式**

$$
\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T) \quad \text{（轨迹：模型生成 token、工具调用、环境观测的混合序列）}
$$

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle \quad \text{（POMDP：agent 只能观测部分状态）}
$$

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right] \quad \text{（多轮策略优化目标：从单步期望到轨迹期望）}
$$

$$
A_t = R(\tau) - \bar{R}(s_t) \quad \text{（step-level advantage：把轨迹奖励回拆到每一步）}
$$

**本章公式的作用**

模型一旦接入工具，就不再只是写完一段回答。它会先行动，看到环境返回，再决定下一步。强化学习要训练的对象也随之改变：**从生成一段回答，变成完成一条多步交互过程**。

订机票 Agent 就是这样的任务。用户说"帮我订一张明天北京到上海最便宜的早班机票"，模型不能直接写一句"已为你订好"。它要先把航班搜出来，再按时间和价格筛选，确认目标航班还有余票，最后把正确的乘机人、航班号和舱位信息传给下单接口。前面某一步如果带入了错误信息，后面的动作就会沿着错误状态继续执行，直到出票失败。

因此，训练时首先要保存的不是最终回答，而是完整过程：模型在什么状态下采取了什么动作，环境又返回了什么观测。这个过程称为**轨迹**，记作：

$$
\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)
$$

这里的 $s_t$ 表示第 $t$ 步前的状态，$a_t$ 表示模型采取的动作，$o_{t+1}$ 表示环境随后返回的观测。它把对话、工具调用和环境返回统一放进同一条序列里。后面讨论轨迹概率、动作掩码和逐步记录，都建立在这个表示之上。

轨迹记录了发生过什么，但它还没有说明模型能看到什么。订机票时，模型能读到搜索结果页的航班列表，却读不到航空公司的完整数据库；它能看到工具返回的库存提示，却不知道价格下一分钟会不会变化。真实环境有完整状态，模型只能看到其中一部分。这个设定用 **POMDP** 表示：

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle
$$

其中 $S_{\text{agent}}$ 是环境和上下文共同构成的状态空间，$A_{\text{agent}}$ 是模型可采取的文本动作或工具动作，$P_{\text{agent}}$ 描述动作如何改变环境和后续观测，$R_{\text{agent}}$ 给出任务奖励，$O$ 表示模型实际能看到的观测。POMDP 在这里强调一件事：**策略不是在完整世界状态上决策，而是在有限观测和历史上下文上决策**。

有了轨迹和观测设定，优化目标也要跟着改变。单轮 RL 最大化的是一段输出的期望奖励：

$$\mathbb{E}_{a \sim \pi_\theta}[r(a)]$$

Agentic RL 最大化的是整条轨迹的累积回报。模型在第 $0$ 步搜什么，会影响第 $1$ 步看到哪些航班；第 $1$ 步选哪趟航班，又会影响第 $2$ 步需要检查什么库存。目标函数因此写成：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$$

这里的 $\tau \sim \pi_\theta$ 表示轨迹由当前策略与环境交互采样得到，$\gamma^t$ 用来折扣远期奖励，$R(s_t,a_t)$ 表示第 $t$ 步动作带来的奖励或最终奖励回传后的贡献。这个公式把训练目标从"让回答得高分"改成了**让交互过程的总回报更高**。

轨迹回报解决了"优化什么"，还没有解决"每一步怎么学"。订票成功，不表示每个动作都值得强化；模型可能前面绕了远路，最后才补救回来。订票失败，也不表示每个动作都错；前几步可能已经找到正确航班，只是在最后下单时填错了参数。最终奖励必须拆回每一步，训练才知道该强化哪个动作、修正哪个动作。

这就是**逐步优势**要解决的问题：

$$
A_t = R(\tau) - \bar{R}(s_t)
$$

它把整条轨迹的结果 $R(\tau)$ 和同一状态附近的基准回报 $\bar{R}(s_t)$ 作比较，判断第 $t$ 步之后的表现是否高于通常水平。后续小节里的结果奖励、过程奖励、轮次折扣、组内优势，都是在构造更可靠的 $A_t$，让最终奖励能更准确地影响每一步动作。

本章沿用前面已经建立的 MDP、策略梯度、GRPO 和可验证奖励，把它们放进多轮交互任务里重新看。到了 Agent 场景，训练不只要更新策略，还要处理环境异步、沙箱管理、轨迹长度不一致和长时程信用分配等工程问题。

## 章节安排

| 小节                                                       | 核心问题                                                                          |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [19.1 Agentic RL 基础](./overview)                         | 单轮到多轮的范式转移、智能体四组件、最简 Agent Loop、工业框架全景                 |
| [19.2 多轮强化学习](./formulation)                         | 多轮 MDP 怎么写？联合状态、结构化动作、POMDP、step-level 轨迹、action mask        |
| [19.3 轨迹信用分配](./credit-assignment)                   | 多轮交互失败了，该怪谁？ORM vs PRM、turn-level discounting、group-based advantage |
| [19.4 工具调用与轨迹生成](./tool-use-and-trajectory)       | 训练数据从哪来？轨迹合成、工具调用策略、沙箱与异步 rollout                        |
| [19.5 Search-Augmented RL](./tool-use-agents)              | 搜索增强 Agent 怎么训？DeepSeek-Researcher、Kimi-Researcher 的训练范式            |
| [19.6 Code Interpreter RL](./industrial-practice)          | 代码 Agent 的真实训练陷阱：不稳定、长度失控、reward hacking 的工程对策            |
| [19.7 多智能体协作](./multi-agent-swarm)                   | 多个 Agent 怎么协作？角色分工、通信协议、群体优势分配                             |

## 学习目标

读完本章后，你应该能够：

- 用 POMDP 六元组 $\langle S, A, P, R, \gamma, O \rangle$ 形式化一个多轮 Agent 任务，并指出它相比单轮 MDP 新引入的要素（联合状态、结构化动作、外部环境）。
- 说清 **ORM** 与 **PRM** 的取舍——为什么可验证任务适合 ORM、为什么复杂推理需要 PRM、以及 SALT/group-based 这类介于两者之间的方案在解决什么问题。
- 理解 Agentic RL 训练系统的工程骨架——rollout 与 policy update 的交替、异步训练的代价与收益、沙箱环境的设计原则——并能据此判断 OpenRLHF、verl、AReaL 等框架的适用场景。

本章会频繁用到以下概念，建议先复习：

- [GRPO 与 RLVR](../chapter18_grpo/rlvr)——"可验证奖励"是 Agentic RL 的天然奖励来源
- [PPO 与奖励模型](../chapter10_ppo/ppo-bipedal-walker)——策略优化的基础框架
- [MDP 五元组](../chapter03_mdp/mdp)——形式化多轮交互的出发点

准备好后，先从 Agentic RL 的整体图景开始——[19.1 Agentic RL 基础](./overview)。

前文给出了 Agentic RL 的形式化骨架，包括轨迹、POMDP 和信用分配。本节把这些公式落回具体的工程图景：Agent 在真实环境里如何运转，为什么训练它比训练单轮 LLM 更难，以及当前工业界用哪些框架完成训练。

读完本节，你应该能建立一个完整的"Agentic RL 训练系统"心智模型，知道后续每一节在哪一块上深入。

## 从单轮到多轮的范式转移

前面所有章节讨论的强化学习，本质上都是**单轮决策**：模型接收一个 prompt，输出一段完整回答，奖励模型给出一个分数，策略据此更新一次。无论底层算法是 PPO 还是 GRPO，"一问一答一打分"的骨架始终未变。

但真实的智能体不这样工作。

考虑一个订机票 Agent。用户说"帮我订一张明天北京到上海最便宜的早班机票"，Agent 必须分步行动：先搜索航班，对比价格和时间，确认座位库存，调用下单 API，等待出票确认。中间任何一步出错——搜索 query 太宽、没比价直接选第一条、库存判断失误、下单参数错误——整个任务就失败。环境只在最后给出一个二元信号：出票成功（reward = 1）或失败（reward = 0）。

这种从"一问一答"到"多步与环境交互"的转变，正是 Agentic RL 要解决的核心问题。

## 两条对照轨迹

同一个订机票任务，同一个模型，两次 rollout：

```
轨迹 A（成功）                          轨迹 B（失败）
─────────────────────────              ─────────────────────────
T1 search("北京 上海 早班 低价机票")     T1 search("北京 上海 机票")
   obs: 12 条相关航班                      obs: 200 条混合结果

T2 filter(dep<9:00, sort=price)         T2 pick_first()
   obs: CA1501 6:30 ¥760                   obs: MU5101 9:30 ¥1280

T3 check_seat(CA1501)                   T3 order(MU5101)
   obs: 座位充足                            obs: 下单成功

T4 order(CA1501, seat=window)
   obs: 出票成功

reward = 1                              reward = 0
```

两条轨迹最终 reward 截然不同，但**问题出在哪一步**？轨迹 B 失败是因为 T1 的 query 太宽，还是 T2 没比价直接选第一条，还是 T3 没确认就下单？只看最终 reward 无法回答。

- 形式化地把"轨迹 vs 单轮 completion"写清楚，见 [19.2 多轮强化学习](./formulation)。
- 把最终 reward 回拆到每一步，见 [19.3 轨迹信用分配](./credit-assignment)。

## 智能体的基本组件

一个 agent 不只是 LLM。最小定义：**LLM 主干 + 指令 + 工具 + 环境**，四者在 agentic loop 中循环。

### LLM 主干

agent 的决策核心。接收当前观测，推理下一步，生成动作（文本或工具调用）。任何足够强的 LLM 都能充当主干，但工程上往往选用经过推理训练的模型——它们能在输出动作前先生成 thinking trace，对多步决策更友好。

### 指令

告诉 agent 要解决什么问题、用什么策略。除了任务本身（"订最便宜的早班机票"），还包含求解策略提示（"先搜索再筛选"、"价格和时间都要考虑"、"失败就重试"）。指令的质量直接决定 agent 行为的下限。

### 工具与环境

工具是 agent 操作环境的接口：搜索 API、代码解释器、CLI、MCP server、下单 API。工具调用通常以特殊 token 标记，嵌入到模型的 token 流中：

```
<tool_call>{"name":"search_flights","args":{"from":"PEK","to":"SHA"}}</tool_call>
<tool_response>[CA1501 6:30 ¥760, CA1831 7:00 ¥690, ...]</tool_response>
```

环境是有状态的：搜索结果会变化、库存会变动、下单会改变数据库。工具调用的返回不仅取决于参数，还取决于环境当前状态。这种把输出锚定到真实世界而非参数记忆的能力称为 **grounding**——是 agent 相对于纯 LLM 的一大优势，也是 RL 训练能赋予的核心行为模式。

### Agentic Loop

四者循环：**感知观测 → 推理下一步 → 执行动作 → 接收新观测**，直到满足终止条件（任务完成、达到最大步数、模型输出结束信号）。

一次完整的 loop 称为一次 **rollout**；rollout 产出的完整交互记录称为一条 **trajectory**，记作 $\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)$。轨迹不只是文本序列，它混合了模型生成的 token、工具调用、工具返回、环境状态变化，结构上更像一棵对话树而非线性文本。

## 四个系统级挑战（来自 RAGEN）

RAGEN 等系统论文提醒我们，Agentic RL 不只是"把 GRPO 套到 tool call 上"，而是要同时设计环境、采样、奖励、稳定训练和评估。[XiaoRed5 的入门资料](https://github.com/XiaoRed5/Agentic-RL-Most-Detailed-Intro) 把 Agentic RL 区别于单轮 RL 的核心挑战总结成四条，每一条都决定了训练能不能真正学起来。

### 挑战一：长程决策——早期动作塑造后续状态分布

Agentic RL 的"长程"不是表面上的"轨迹变长"，而是**早期的 action 会改变后续状态分布**。

订机票的例子里，T1 的搜索 query 决定了模型看到哪些航班；T2 选哪一班决定了后续要 check 什么库存；T3 是否确认决定了 T4 能不能下单。

```
query 写得差    →  搜索结果偏了  →  读到错误证据  →  后面推理全被带偏
query 写得好    →  找到关键来源  →  后续只需要验证和归纳
```

一个早期小错误可能在后面被放大；一个早期好决策也可能因为后续步骤失误而没有转化成最终成功。**训练信号往往很晚才出现，但真正影响结果的决策可能发生在很早的位置**。这是信用分配问题的根源——详见 [19.3](./credit-assignment)。

### 挑战二：环境随机性导致 reward variance 飙升

Agent 和环境交互时，环境不是一个完全稳定的文本函数。搜索引擎的结果可能变化、网页内容可能更新、工具调用可能失败、模拟环境可能有随机性；即使环境固定，模型采样本身也会让同一个任务产生多条不同轨迹。

这会带来一个训练问题：**同一个 prompt 下，不同 rollout 的最终 reward 可能差异很大**。有的轨迹刚好搜到关键证据，有的轨迹走到无关页面；有的轨迹早早答对，有的轨迹多绕几步后失败。

```
同一个问题采 8 条 rollout   →  2 条成功 (reward = 1) / 6 条失败 (reward = 0)
下一轮再采 8 条             →  5 条成功 (reward = 1) / 3 条失败 (reward = 0)
```

这种波动不一定说明模型突然变强或变弱，可能只是采样和环境反馈造成的方差。所以 Agentic RL 不能只看单次 reward 曲线，还要关注 **reward variance、梯度尖峰、轨迹分布是否塌缩、模型是不是陷入某种重复行为模式**。AEM、RAGEN-2 等工作就是从稳定性维度入手的。

### 挑战三：Rollout 设计——三个被忽视的维度

在 Agentic RL 里，rollout 不是简单的"让模型多生成几条答案"。它决定了模型能探索到什么状态、能比较哪些行为、reward 信号是否足够有信息量。

**初始状态多样性**很重要。如果训练任务过于相似，模型可能学会固定套路，而不是学会通用的决策能力。比如搜索 agent 如果总是在同一种问法、同一种网站结构上训练，就可能只学会某种模板化 query，而不是学会真正根据信息缺口设计搜索策略。

**交互粒度**也很重要。粒度太粗时，一次 action 包含太多决策，出了问题很难知道是哪一部分错了；粒度太细时，轨迹会变得很长，训练成本上升，reward 更稀疏，模型也可能在无意义的小动作上浪费预算。

```
粒度太粗：一次 action = 搜索 + 阅读 + 判断 + 回答
        问题：失败后很难判断哪一步错了

粒度太细：一次 action = 点击一个按钮 / 滚动一次页面
        问题：轨迹太长，训练成本和 credit assignment 难度都上升
```

**采样频率**也会影响学习。每个任务只采一条 rollout，模型很难知道"同一个状态下其他动作会怎样"；每个任务采太多 rollout，成本又会迅速变高。实际训练时，rollout 数量、最大交互轮数、采样温度、是否复用环境缓存，都会直接影响训练稳定性和样本效率。

### 挑战四：纯 outcome reward 让模型学到浅层策略

最终答案 reward 很有用，因为它简单、便宜、可验证。**但如果 reward 只有最终结果，模型可能会学到一些"看起来有效"的捷径，而不一定学到我们想要的 agent reasoning**。

比如搜索问答里，模型可能学会优先生成高频答案，或者学会在证据不足时也提前作答；网页任务里，模型可能学会某些固定点击模式；工具任务里，模型可能学会调用工具的形式，却没有真正利用 observation 修正自己的计划。

```
表面上看：模型会搜索、会阅读、会回答
实际可能：搜索 query 很模板化 / 读到冲突证据也不验证
        observation 没有真正进入后续推理 / 最后只是凭先验猜答案
```

这也是为什么信用分配章节里的 PRM、SPA-RL、IGPO 等方法如此重要——它们本质上都是在尝试让训练信号更接近"哪一步真的推动了任务完成"。

## 一个最简 Agent Loop

概念看十遍不如动手跑一遍。下面用几十行代码搭一个能跑的 Agent——不涉及 RL 训练，只看"一个 Agent 是怎么和工具交互的"。理解了这个循环，后面加上 RL 就水到渠成。

```python
import json, subprocess, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# ① 定义工具 与 告诉模型"你能做什么"
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

# ② 工具的实际执行逻辑（环境 Environment）
def execute_tool(name, args):
    if name == "execute_bash":
        r = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr
    elif name == "read_file":
        with open(args["path"]) as f:
            return f.read()
    return f"Unknown tool: {name}"

# ③ Agent Loop 与 感知→推理→行动→观测，循环往复
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
            print(f"  [Turn {turn+1}] 调用工具: {tc.function.name}({args})")
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "（达到最大轮次，强制停止）"

print(run_agent("查看当前目录下有哪些 .md 文件，告诉我一共有几个"))
```

运行效果：

```
  [Turn 1] 调用工具: execute_bash({'command': 'ls *.md'})
  [Turn 2] 调用工具: execute_bash({'command': 'ls *.md | wc -l'})
当前目录下有 12 个 .md 文件。
```

把这 50 行代码和前面的概念对应起来：`tools = [...]` 对应**动作空间** $A_{\text{action}}$；`execute_tool()` 对应**环境**；`for turn in range(max_turns)` 对应 **Agentic Loop / Rollout**；`client.chat.completions.create()` 对应**策略 $\pi_\theta$**；`messages.append(...)` 对应**状态 $s_t$**。

这个 Agent 的"聪明程度"完全取决于策略 $\pi_\theta$。怎么给它加上 RL 训练——奖励怎么算（ORM vs PRM）、多步交互中奖励怎么分配（信用分配）、训练数据怎么管理——是后续小节要展开的。

## Search-R1：最小可跑的 Agentic RL 案例

上面的 Agent Loop 没有训练循环。如果想看一个**真正能跑起来的最小 Agentic RL 系统**，[Search-R1](https://github.com/PeterGriffinJin/Search-R1) 是最好的起点。

Search-R1 把任务限制在一个很小的 agent 环境里：模型只需要学会"什么时候搜索、搜什么、什么时候回答"。它和传统 RAG 的差别不是"有没有检索"，而是"**谁来决定检索**"——传统 RAG 由系统先检索，再把文档交给模型；Search-R1 让模型在推理过程中自己发起 search action。

```
RAG（系统决定检索）：
  question → retriever → documents → model answer

Search-R1（模型决定检索）：
  question → model emits <search>query</search>
          → retriever returns documents
          → model continues or answers
```

代码里用四类标签把这个闭环写出来：

- `<think>…</think>`：模型内部推理，模型生成，**参与训练**
- `<search>query</search>`：模型 action，模型生成，**参与训练**
- `<information>docs</information>`：环境 observation，检索器返回，**masked 不训练**
- `<answer>final answer</answer>`：模型最终回答，模型生成，**参与训练**

一次 rollout 的核心逻辑很短：模型生成到 `</search>` 时暂停，系统解析 query，调用检索器，把返回文档包进 `<information>` 后拼回上下文；模型读到这些 observation 后继续生成，直到输出 `<answer>` 或达到最大轮数。

**最重要的训练细节是 mask**。`<search>`、`<think>`、`<answer>` 是模型生成的 token，可以被优化；`<information>` 是检索器返回的文本，只能作为上下文，不应该让模型学习"生成搜索结果"。Search-R1 里的 `state_masking=true` 对应的就是这件事。

读 Search-R1 时不需要先陷进复杂工具系统。它要表达的核心很清楚：**当搜索 query 变成模型 action，检索结果变成环境 observation，RL 训练就不再只是优化一段 answer，而是在优化一条会调用工具的轨迹**。

## SFT 与 Prompting 的局限

一个自然的疑问：ReAct、Toolformer 等方法已经能让 LLM 调用工具了，为何还需要 RL？

关键区别在于：SFT 和 prompting 教会模型的是**模仿**——复制人类演示中"何时调用工具、调用什么工具"的模式。但真实的 Agent 任务中，工具使用的最优策略高度依赖上下文：

- 搜索查询如何构造？何时打开网页详情？何时停止搜索开始总结？
- 代码修改后测试仍未通过，是继续调试还是切换方向？
- 多个来源的信息相互矛盾，应采信哪一个？

这些本质上是**策略学习问题**，而非单纯的语言建模问题。演示数据难以覆盖所有可能的决策路径，而 RL 可以根据任务结果反向塑造工具调用、规划和记忆管理等行为模式。

SFT 和 RL 在 Agentic 场景中的分工：

- **SFT 教格式**：教会模型工具调用的语法、基本的交互协议。
- **RL 教策略**：教会模型何时调用工具、如何组合多步行动、失败后如何恢复。

DeepSeek-R1-Zero 的实验表明，跳过 SFT 直接进行 RL 也能涌现出推理能力——但前提是基座模型足够强大。实践中，SFT warmup + RL fine-tuning 的两阶段方案仍是主流范式。

## 工业框架全景

回到现实——当你想真正训练一个 Agent 时，用什么框架把这套东西跑起来？

前面章节做 PPO、GRPO 时这个问题不尖锐：训练循环几乎全是 GPU 计算，TRL 或 OpenRLHF 都能轻松搞定。但 Agentic RL 的训练循环里多了一个"等"字——模型调用搜索引擎，GPU 就得等搜索结果；模型执行代码，GPU 就得等沙箱返回。怎么让 GPU 不空等？这就是 Agentic RL 训练框架要解决的核心问题。

2025–2026 年，围绕这个问题涌现了一批开源框架：

| 框架         | 开发方              | 一句话定位                                                        | 多轮 Agent 原生支持  | GitHub                                                    |
| ------------ | ------------------- | ----------------------------------------------------------------- | -------------------- | --------------------------------------------------------- |
| **OpenRLHF** | 开源社区            | 代码最简洁（8k 行），算法与 Agent 执行解耦，一行代码切换单轮/多轮 | 是                   | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| **verl**     | 字节跳动 / 开源社区 | 吞吐最高，训推在同一组 GPU 上动态切换，生态扩展最多               | 基础支持，社区扩展中 | [verl-project/verl](https://github.com/verl-project/verl) |
| **slime**    | THUDM / 智谱生态    | Megatron + SGLang 后训练框架，MoE 模型效率高                      | 基础支持             | [THUDM/slime](https://github.com/THUDM/slime)             |
| **AReaL**    | 蚂蚁 / 清华         | 全异步训练——GPU 完全不等，速度提升 2.77 倍                        | 是                   | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) |
| **ROLL**     | 阿里巴巴淘天        | 推理（RLVR）+ Agent 双模式，原生 Qwen 支持                        | 是                   | [alibaba/ROLL](https://github.com/alibaba/ROLL)           |
| **SkyRL**    | UC Berkeley         | 模块化全栈——训练、Agent 编排、任务环境各自独立                    | 是                   | [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)   |
| **Seer**     | Moonshot AI (Kimi)  | 极致同步——通过在线上下文学习消除 rollout 长尾，吞吐提升 74–97%    | 否                   | 详见 arXiv:2511.14617                                     |
| **Relax**    | 小红书              | 全模态（文本+图像+音频）异步训练                                  | 是                   | 详见 arXiv:2604.11554                                     |
| **TRL**      | HuggingFace         | 轻量易用，HuggingFace 生态无缝打通，但不支持大规模异步            | 单轮为主             | [huggingface/trl](https://github.com/huggingface/trl)     |

这些框架的核心差异可以归结为一个取舍：**同步 vs 异步**。同步训练简单、可控、容易调试，但 GPU 利用率低。异步训练吞吐翻倍，但训练数据可能基于旧权重生成，需要额外的算法补偿。

AReaL 的研究表明，异步训练可以在不损失效果的前提下将速度提升近 3 倍——但前提是训练已经调通。Seer 走另一个极端：坚持同步框架，不改变 GRPO 算法，而是通过在线上下文学习（divided rollout、context-aware scheduling、adaptive grouped speculative decoding）消除 rollout 长尾延迟，在保持 on-policy 保证的前提下将吞吐提升 74–97%。

另一个关键差异是：框架最初为单轮 RL（推理任务）设计，还是一开始就考虑了多轮 Agent 交互。前者的 Agent 执行模块是后加的，能用但不是为此优化；后者的 Agent 执行是架构一等公民，在状态管理、异构轨迹长度、工具调用异步返回等方面有原生支持。OpenRLHF、AReaL、ROLL、SkyRL 属于后者。

框架选型取决于具体场景。刚入门想快速跑通 demo，OpenRLHF 代码最简洁、文档最完善。企业级大规模训练（70B+），verl 的吞吐和生态优势明显。模型是 MoE 架构（如 GLM-4.5、Qwen3-30B-A3B、DeepSeek-R1），slime 的 Megatron + SGLang 原生架构对 MoE 的 fp8 rollout、DeepEP 通信等做了专门优化。极致吞吐，AReaL 的全异步模式能做到近 3 倍加速。更多工程细节——沙箱管理、环境构建、分布式部署——在 [19.4 工具调用与轨迹生成](./tool-use-and-trajectory) 展开。

## 本节总结

Agentic RL 把训练对象从"一段回答"扩展到"一条完整交互轨迹"。这一扩展引出四个核心议题，本章后续小节逐一深入：

- **形式化**——轨迹、状态、动作在多轮设定下如何精确定义？POMDP 视角如何区分模型生成的 action token 与环境返回的 observation token？→ [19.2 多轮强化学习](./formulation)
- **信用分配**——一条轨迹最终失败，reward 怎么回拆到每一步？ORM/PRM/SALT/GiGPO/HGPO/SPA-RL/AgentPRM/ARPO/IGPO/StepPO 等十多种方法各有何取舍？→ [19.3 轨迹信用分配](./credit-assignment)
- **工具与轨迹工程**——训练数据从哪来、工具策略怎么学、沙箱怎么管？→ [19.4 工具调用与轨迹生成](./tool-use-and-trajectory)
- **真实训练陷阱**——工业界在哪些坑里摔过？→ [19.6 Code Interpreter RL](./industrial-practice)

接下来先看形式化：把"多轮交互"翻译成 RL 能处理的数学对象——[19.2 多轮强化学习](./formulation)。
