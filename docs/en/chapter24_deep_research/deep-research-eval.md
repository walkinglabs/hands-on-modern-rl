# 22.2 Evaluation Benchmarks and Open-Source Projects

> [24.2](./browser-rl-harness) built the training harness. But how good is the Deep Research agent that comes out of it? That question needs an **evaluation benchmark**. This section covers two things: (1) the design philosophy and pitfalls of the mainstream Deep Research benchmarks (BrowseComp, xbench-DeepSearch, GAIA); (2) reproducible open-source projects (GPT-Researcher, STORM, OpenResearcher) so you don't have to build everything from scratch.

## Why Deep Research Evaluation Is Especially Hard

Traditional LLM benchmarks (MMLU, GSM8K) share two properties: (1) the **answer is unique** (a math problem has exactly one correct answer); (2) **no tools are needed** (the model answers directly). Deep Research breaks both of these:

- **The answer isn't unique**: ask "compare state management in React and Vue" and there are countless valid ways to phrase a correct answer
- **Tools are mandatory**: the model can't answer "what was the Bitcoin price in June 2026" from memory
- **The process matters**: did the model get the right answer in 5 steps or 50? Did it cite reliable sources?
- **Data contamination**: web content changes constantly, so today's answer may be stale tomorrow

Because of this, Deep Research evaluation needs benchmarks designed specifically for it.

## Mainstream Benchmarks

### BrowseComp (Meta, 2025)

**BrowseComp** is a browser-agent benchmark Meta released in 2025, built specifically to test an agent's ability to find information on the open web.

**Design philosophy**:

- **Hard enough to force browser use**: every question is designed so it cannot be answered from the model's parametric memory
- **Answers are unique and verifiable**: each question has one clear-cut answer that can be checked by exact string match
- **Anti-Google**: the answer isn't a single Google search away — it takes multi-step navigation

**Example**:

> Q: "The player who scored Argentina's only goal in the 1998 World Cup quarterfinal, which club did he serve as youth coach for after retiring?"
>
> A: "Argentinos Juniors" (exact string)

To solve this, the model has to: (1) look up who scored for Argentina in the 1998 World Cup quarterfinal → Batistuta; (2) look up what Batistuta did after retiring → became a youth coach; (3) look up which club. That's at least 3-5 steps of browser navigation.

**Metric**: Exact Match Accuracy.

**SOTA results** (as of June 2026):

| System              | BrowseComp | Notes                       |
| ------------------- | ---------- | --------------------------- |
| GPT-5 + browser     | 38.2%      | Upgraded OpenAI Operator    |
| Claude Opus 4.6     | 35.7%      | Anthropic internal          |
| Kimi K2.5 Swarm     | 72.1%      | Multi-agent collaboration   |
| Tongyi DeepResearch | 51.4%      | Alibaba, March 2026         |
| Human expert        | 87.5%      | Single person, 30-min limit |

Notice that Kimi K2.5 Swarm beats the single-agent systems by more than 30 percentage points — real-world evidence for the [22.6 multi-agent collaboration](../chapter22_agentic/multi-agent-swarm) discussion.

### xbench-DeepSearch (Tsinghua University, 2025)

**xbench-DeepSearch** is a Chinese-language Deep Research benchmark released in 2025 by Tsinghua and the University of Hong Kong, targeting several shortcomings of BrowseComp:

- **Chinese-first**: BrowseComp is English-only; xbench-DeepSearch covers both Chinese and English
- **Diverse task types**: BrowseComp is all single-entity Q&A; xbench-DeepSearch adds multi-document synthesis, comparative analysis, and temporal reasoning
- **Controllable difficulty**: each question is labeled with a difficulty rating (1-5 stars), so you can pick a subset matched to a model's capability

**Task types**:

| Type                     | Share | Example                                                                            |
| ------------------------ | ----- | ---------------------------------------------------------------------------------- |
| Single-entity Q&A        | 30%   | "Which university did the 2025 Turing Award winner get their bachelor's from?"     |
| Multi-document synthesis | 25%   | "Compare the training cost of DeepSeek V3 and Llama 4"                             |
| Comparative analysis     | 20%   | "How do React 19 and Vue 3.5 differ in SSR performance?"                           |
| Temporal reasoning       | 15%   | "When did the Vision Pro, announced at Apple WWDC 2024, launch in mainland China?" |
| Implicit reasoning       | 10%   | "Following the method in paper X, what accuracy would you expect on dataset Y?"    |

**Metrics**: besides EM, xbench-DeepSearch also reports:

- **Process Score**: correctness of intermediate steps
- **Efficiency**: average steps / minimum steps
- **Citation Quality**: whether the cited sources are reliable

### GAIA (Meta + HuggingFace, 2024)

**GAIA** (General AI Assistants) is an earlier benchmark, but it remains one of the standard Deep Research test sets. GAIA defines three difficulty levels:

| Level   | Task complexity | Average steps | Example                                       |
| ------- | --------------- | ------------- | --------------------------------------------- |
| Level 1 | Easy            | 5-10          | "Find an image matching a specific condition" |
| Level 2 | Medium          | 10-30         | "Extract a table from a PDF"                  |
| Level 3 | Hard            | 30-100        | "Plan a multi-city trip across Europe"        |

**Metric**: accuracy + average number of steps (fewer is better).

The key difference between GAIA and BrowseComp: GAIA tasks look more like "personal assistant" work, while BrowseComp tasks look more like "research" work.

## Four Pitfalls in Evaluation

Deep Research evaluation has several distinctive pitfalls — miss them and the reported numbers end up inflated.

### Data Contamination

The LLM's pretraining data may already contain the answer. Ask "who won the 2024 Nobel Prize in Physics" and the model can answer from memory, no browser required.

**Countermeasures**:

- Use **time-sensitive questions** (the answer was published after the training cutoff)
- Use **counterfactual questions** ("if event X hadn't happened, what would Y look like?" — the model has to look up the real facts about X)
- BrowseComp mitigates this by design, requiring multi-step navigation

### Diversity in Answer Phrasing

Ask an agent to "compare React and Vue," and both "React uses JSX, Vue uses templates" and "Vue uses templates, React uses JSX" are correct — but exact-match scoring would mark one of them wrong.

**Countermeasures**:

- Use an **LLM-as-Judge** (GPT-4 / Claude) to judge semantic equivalence
- Use **structured answers** (JSON, Markdown tables) to reduce phrasing variance
- xbench-DeepSearch calibrates with an LLM judge

### Gaming the Process

An agent might skip actually browsing and instead generate a plausible-looking answer directly, hallucinating the citation.

**Countermeasures**:

- **Citations must be clickable**: check at evaluation time whether the URLs the agent provides actually exist
- **Page snapshots**: save snapshots of the pages the agent visited during evaluation for later audit
- BrowseComp includes "reverse verification" questions — questions whose answers are random strings that no agent could guess

### Cost Contamination

Token cost varies 10-30× across agents ([22.6](../chapter22_agentic/multi-agent-swarm) notes that Kimi K2.5 Swarm costs 15× a single agent). A naive accuracy comparison ends up favoring the expensive system.

**Countermeasures**:

- Report **accuracy per token cost** as an efficiency metric
- Compare under a fixed budget (e.g., "at most 100K tokens per question")

## Reproducing with Open-Source Projects

You don't need to build a harness from scratch — the following open-source projects provide complete Deep Research training / inference pipelines.

### GPT-Researcher (assafelovic-gpt-researcher)

**The most popular open-source Deep Research framework.** 18K+ GitHub stars, actively maintained.

**Features**:

- **Python**, built on Playwright
- A built-in Planner / Researcher / Writer three-layer architecture (a typical Orchestrator-Worker pattern)
- Supports multiple search backends (Tavily, SerpAPI, Google CSE, Bing)
- Outputs Markdown reports with citations

**Good for**: quickly standing up a production-grade Deep Research service. **Not good for**: RL training (it's designed for inference-time use).

```bash
pip install gpt-researcher
```

```python
from gpt_researcher import GPTResearcher

async def research():
    researcher = GPTResearcher("Compare React 19 vs Vue 3.5 SSR performance")
    report = await researcher.conduct_research()
    print(report)
```

### Stanford STORM (stanford-omp-storm)

**An open-source research framework from Stanford's Oval group**, built specifically for "long-form structured article generation."

**Features**:

- Follows the Wikipedia writing workflow: first a "simulated conversation" (multiple personas interviewing each other), then an outline, then the final article
- Built-in Wikipedia retrieval and citation management
- Produces Wikipedia-style long-form articles (5K-20K words)

**Good for**: academic surveys, in-depth reports. **Strength**: high citation quality (Wikipedia standard).

```bash
pip install knowledge-storm
```

```python
from storm import STORMWikiRunner

runner = STORMWikiRunner(...)
runner.run("History of reinforcement learning")
```

### OpenResearcher (tjuloonkopen-researcher)

**A fully open-source Deep Research training pipeline**, RL training code included.

**Features**:

- **Reproducible training**: ships a 100K-trajectory dataset plus GRPO training scripts
- Built on the Search-R1 architecture
- A 7B model reaches 31.2% on BrowseComp
- Complete documentation (English)

**Good for**: training a Deep Research agent from scratch. **Strength**: a complete vLLM + veRL pipeline that's easy to extend.

```bash
git clone https://github.com/OPPO-PersonalAI/O-Researcher
cd open-researcher
bash train.sh --model qwen2.5-7b --algo grpo
```

### Other Projects Worth Watching

| Project                              | Institution                     | Highlights                                              |
| ------------------------------------ | ------------------------------- | ------------------------------------------------------- |
| **Search-R1**                        | UIUC                            | The earliest open-source Deep Research RL training code |
| **R1-Searcher**                      | Renmin University               | Multi-stage training (SFT → RL)                         |
| **Tongyi DeepResearch reproduction** | Alibaba DAMO Academy (official) | Industrial scale, needs an H100 cluster                 |
| **PokeeResearch**                    | Peking University               | A 7B model matching 70B-class performance               |
| **DeepResearcher**                   | Renmin University               | Open-source end-to-end RL training                      |

## End-to-End Experiment: Training a Deep Research Agent from Scratch

Now let's string everything in this section together into a full experimental pipeline.

### Step 1: Pick a Base Model

- Beginner: Qwen2.5-7B-Instruct (easy to get running)
- Intermediate: Llama-3.1-8B-Instruct
- Advanced: Qwen3-14B / DeepSeek-V2-Lite

### Step 2: Pick an Action Space

- Simple (API-based): the 3-action space from Search-R1
- Realistic (Playwright-based): the 7-action space from OpenResearcher

### Step 3: Pick Training Data

- xbench-DeepSearch training set (10K)
- HotpotQA + Natural Questions (needs adaptation)
- Synthesize your own: generate questions and answers with GPT-5 / Claude

### Step 4: Train

```bash
# Using OpenResearcher's training script
bash train.sh \
    --model qwen2.5-7b \
    --algo grpo \
    --env api \
    --data xbench-train.jsonl \
    --batch-size 256 \
    --lr 5e-7 \
    --epochs 3
```

### Step 5: Evaluate

```bash
# BrowseComp evaluation
python eval.py \
    --model checkpoint-final \
    --benchmark browsecomp \
    --max-steps 30
```

Expected outcome: after 3 epochs of training, Qwen2.5-7B's BrowseComp accuracy rises from 8% (the SFT baseline) to 25-30% — already close to GPT-4 + browser territory (35%).

## Section Summary

The four major Deep Research benchmarks (BrowseComp / xbench-DeepSearch / GAIA / your own) each emphasize something different — BrowseComp tests "forced multi-step navigation," xbench-DeepSearch tests Chinese-language coverage and task diversity, GAIA tests personal-assistant scenarios. **There's no silver bullet**; run at least two benchmarks.

On the open-source reproduction side, **GPT-Researcher is the pick for building a product, OpenResearcher is the pick for doing research**. The former is engineering-mature; the latter is training-transparent. If your goal is research or learning, start with OpenResearcher; if your goal is shipping a product, start with GPT-Researcher.

The next chapter, [Chapter 23: Computer Use and GUI Agents](../chapter25_computer_use/intro), moves from the browser to the entire desktop — the agent no longer just browses web pages, it operates arbitrary GUI applications (Excel, Photoshop, internal office tools).
