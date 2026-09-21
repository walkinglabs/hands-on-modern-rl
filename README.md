<div align="center">
  <img src="docs/public/readme/readmelogo.png" alt="Hands-On Modern RL" width="500" />
  <p><em>From Markov decision processes and policy optimization to reasoning models, agents, and multimodal systems</em></p>

  <p>
    <a href="https://walkinglabs.github.io/hands-on-modern-rl/"><img src="https://img.shields.io/badge/Course-Online-2563eb?style=flat-square" alt="Online Course" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/releases/latest"><img src="https://img.shields.io/badge/PDF-Download-e11d48?style=flat-square" alt="PDF Download" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-111827?style=flat-square" alt="CC BY-NC-SA 4.0 License" /></a>
    <img src="https://img.shields.io/badge/Node-%3E%3D18-16a34a?style=flat-square" alt="Node >= 18" />
    <img src="https://img.shields.io/badge/Docs-VitePress-646cff?style=flat-square" alt="VitePress" />
  </p>

  <p>
    <a href="README.md">English</a> ·
    <a href="README.zh.md">中文</a>
  </p>

  <p>
    <a href="https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole">Train on ModelScope</a> ·
    <a href="https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment01-cartpole.ipynb">Run the CartPole Notebook</a> ·
    <a href="https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole/file/view/master/train.py">ModelScope Script</a>
  </p>

  <p>
    <a href="#discussion-group-wechat">Discussion Group (WeChat)</a>
  </p>

  <p>
    <a href="#book-features">Book Features</a> ·
    <a href="#about-this-book">About This Book</a> ·
    <a href="#news">News</a> ·
    <a href="#contents">Contents</a> ·
    <a href="#structure-of-the-book">Book Structure</a> ·
    <a href="#experiment-code">Experiment Code</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#contributing">Contributing</a>
  </p>
</div>

> **📣 Announcement**
>
> We sincerely thank everyone for your support of this tutorial! A new version is coming soon. Many sections are still being organized and refined, so we appreciate your patience. Suggestions and feedback are always welcome!

## News

> **Note:** This course was created with AI assistance and has not yet been fully reviewed. It may contain factual mistakes or code that does not run as expected. Issues and pull requests are very welcome.

- **[2026-08-19]** 🎮 **Online Classic RL Environments and Scripts**: Over the past two weeks, we have added and refined a collection of online reinforcement learning environments, training scripts, and companion notebooks. Learners can now run classic reinforcement learning experiments online, inspect training logs and evaluation results, and study the algorithms more conveniently. We also fixed many previously reported bugs in the course content, links, and experiment code.
- **[2026-05-15]** 📖 **Full English Translation & PDF Release**: Complete English translation of all chapters is now available. PDF builds for both Chinese and English editions are released automatically via CI.
- **[2026-05-13]** 🚀 **Major Upgrade: LLM and Traditional RL Hands-on Labs**: Added reproducible training examples for **Agentic RL** (Deep Research / rLLM) and **Traditional RL** (Actor-Critic continuous control). Includes complete code and fine-tuning analysis for building an Agentic training system from scratch, along with new VLM RL (GeoQA geometry reasoning) hands-on experiments!
- **[2026-05-02]** Initial browsable open-source release for testing and feedback.

## Online Training Notebooks

WalkingLab is collaborating with ModelScope to provide online training environments for classic reinforcement learning experiments. A ModelScope Studio brings the experiment interface, runtime, and training entry point together on one page, so learners can start training in a browser and observe the agent without first configuring a local environment.

Every Studio has a companion notebook under [`code/online-experiments`](code/online-experiments/README.md). The notebook imports the same training runtime as the Studio, exposes the experiment parameters, prints the full training log, plots checkpoint evaluations, and displays the learned-policy replay or result artifact.

| Experiment                   | Resource | Companion notebook                                                                                                                                                                    | Live Studio                                                                                                |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 01 · CartPole PPO            | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment01-cartpole.ipynb)           | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole)           |
| Gymnasium Playground         | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment-gymnasium.ipynb)            | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium)            |
| 02 · ViZDoom                 | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment02-vizdoom.ipynb)            | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment02-vizdoom)            |
| 03 · Atari / ALE             | xGPU     | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb)              | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari)              |
| 04 · Board Games & Self-Play | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment04-board-selfplay.ipynb)     | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay)     |
| 05 · Multi-Agent Games       | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment05-multiagent-games.ipynb)   | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment05-multiagent-games)   |
| 06 · MiniGrid Adventures     | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment06-minigrid-adventure.ipynb) | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment06-minigrid-adventure) |
| 07 · JAX MinAtar             | CPU      | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment07-jax-games.ipynb)          | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games)          |
| 08 · ManiSkill               | xGPU     | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment08-maniskill.ipynb)          | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill)          |
| 10 · MineStudio / Minecraft  | xGPU     | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment10-minestudio.ipynb)         | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio)         |
| 11 · Unity ML-Agents         | xGPU     | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment11-unity-mlagents.ipynb)     | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents)     |
| 12 · AI2-THOR                | xGPU     | [Run Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment12-ai2thor-embodied.ipynb)   | [Open Studio](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied)   |

CPU entries run on an ordinary notebook instance. Experiments 03, 08, 10, 11, and 12 require a scheduled ModelScope xGPU Notebook and check CUDA before training.

## Book Features

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-learning-path.png" alt="Course learning map screenshot" width="100%" />
      <br />
      <strong>One continuous learning path</strong>
      <br />
      <sub>Begin with a CartPole trial and progress through value learning, policy optimization, and modern agents.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-code-focus.png" alt="PPO code focus screenshot" width="100%" />
      <br />
      <strong>Equations meet code</strong>
      <br />
      <sub>Key PPO, DPO, and GRPO derivations sit beside their implementations, with every tensor accounted for.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-training-metrics.png" alt="CartPole training metrics screenshot" width="100%" />
      <br />
      <strong>Claims tested by experiments</strong>
      <br />
      <sub>Real training curves, ablations, and failure signals show when an algorithm works and when it does not.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-rlhf-pipeline.png" alt="RLHF pipeline screenshot" width="100%" />
      <br />
      <strong>Classic RL to language models</strong>
      <br />
      <sub>Policy gradients and PPO lead naturally into RLHF, DPO, GRPO, and RLVR.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-agentic-rl.png" alt="Agentic RL experiment page screenshot" width="100%" />
      <br />
      <strong>Agents as sequential decisions</strong>
      <br />
      <sub>Tool use, browser interaction, and code repair become problems of states, actions, trajectories, and credit assignment.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-atari-game.png" alt="Atari Pong DQN experiment page screenshot" width="100%" />
      <br />
      <strong>Phenomena before abstractions</strong>
      <br />
      <sub>CartPole, LunarLander, Atari, and LLM experiments pose the problem before introducing the mathematics.</sub>
    </td>
  </tr>
</table>

---

> [!NOTE]
> We hope this open course gives more learners the courage to climb toward the frontier of intelligence and solve more of the hard problems on the path to AGI.
>
> The course is evolving quickly. We recommend focusing on chapters that are not marked as under construction; chapters still in progress may contain mistakes, and corrections or suggestions are welcome.

> **Help Wanted**
>
> Because compute resources are limited, we are seeking GPU support. If you can help with GPU access, please contact physicoada@gmail.com.

## Contents

- [Book Features](#book-features)
- [About This Book](#about-this-book)
- [Structure of the Book](#structure-of-the-book)
- [Experiment Code](#experiment-code)
- [Recommended Learning Path](#recommended-learning-path)
- [Quick Start](#quick-start)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## About This Book

Reinforcement learning studies a simple but difficult problem: a system acts, observes the consequences, and uses them to improve its next action. When rewards arrive late, observations are incomplete, and each update changes the distribution of future experience, the familiar supervised-learning model of inputs and labels is no longer enough. We need a language for interaction, a way to estimate long-term return, and methods that improve a policy stably from limited data.

**Hands-On Modern RL** follows that problem from beginning to end. CartPole and multi-armed bandits first make states, actions, rewards, and policies observable. Markov decision processes, value functions, and Bellman equations then provide a common language. From there, the book develops DQN, policy gradients, actor-critic methods, PPO, continuous control, and offline reinforcement learning. With those foundations in place, RLHF, DPO, GRPO, and RLVR become extensions of the same sequential-decision framework rather than an isolated collection of acronyms.

The second half expands the environment to tools, browsers, code repositories, vision, and audio. An action may be a passage of text, a function call, or a sequence of interface operations. A reward may come from human preference, a rule-based verifier, or a process reward model. The setting changes, but three questions run through the entire book: **How should the decision process be represented? How should an outcome be credited to earlier actions? How can we tell whether a policy has actually improved?**

### How the Material Is Taught

Each chapter follows a problem–method–experiment–reflection rhythm. A concrete task first exposes the difficulty. The concepts and equations needed to solve it come next. Runnable code, training curves, and evaluation metrics then test the argument. The chapter closes by examining assumptions, failure modes, and the limits of the method. Mathematics explains observed behavior; experiments check the mathematics.

Implementations retain the visible skeleton of each algorithm. Readers can trace trajectory collection, advantage estimation, loss construction, and metric changes—and see how reward hacking, KL drift, entropy collapse, distribution shift, or evaluation leakage can invalidate an apparently successful run.

### Who Should Read It

This book is for students, researchers, and engineers with basic machine-learning experience who want a systematic understanding of modern reinforcement learning. Readers should be comfortable with Python and basic PyTorch, and should know introductory linear algebra, probability, and calculus. The mathematical appendices rebuild these tools to the depth required by the chapters, so a separate advanced-mathematics sequence is not a prerequisite.

After completing the core chapters and labs, you should be able to:

- formulate a new decision problem using MDPs, value functions, Bellman equations, and credit assignment;
- implement, read, and diagnose DQN, REINFORCE, actor-critic methods, PPO, DPO, and GRPO;
- explain how SFT, reward modeling, preference optimization, RLHF, and RLVR fit together in LLM post-training;
- design trajectories, rewards, training loops, and evaluation protocols for tool-use, code, and multimodal agents;
- identify failure modes behind training curves and test proposed improvements with controlled experiments.

### Current Status

This repository is an active courseware project. Content is being expanded chapter by chapter, with emphasis on correctness, runnable examples, and a stable learning path.

- Course site: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- Source content: [`docs/`](docs/)
- Runnable examples: [`code/`](code/)
- Local verification: `npm run verify`
- License: [CC BY-NC-SA 4.0](LICENSE)

Issues and pull requests are welcome for typo fixes, conceptual corrections, reproducibility improvements, references, and focused course extensions.

## Roadmap

The course is under active development. Planned milestones:

- [x] **2026-05-02:** Initial open-source browsable release for community testing and feedback.
- [x] **2026-05-10:** Publish a first stable minor version, fix early typos, and stabilize Part 1 and Part 2 content and code.
- [x] **Late May 2026:** Improve reproducible LLM RL experiments and add a full RLVR hands-on module with evaluation.
- [ ] **Early June 2026:** Deliver Agentic RL projects step by step, from single-tool use to complex Deep Research trajectory synthesis.
- [ ] **Late June 2026:** Add Unity-based embodied RL environments and trainable project examples.
- [ ] **July 2026 and later:** Expand multimodal frontier content with full VLM RL or Diffusion RL hands-on cases.

## Structure of the Book

The book contains seven parts and twenty-six chapters. Parts I–III establish the common language and algorithmic foundations of reinforcement learning. Part IV brings those tools into LLM post-training. Parts V and VI study what changes when the action space expands to tools and multimodal environments. Part VII asks how to detect failures, build trustworthy evaluations, and move the research frontier forward. The appendices provide implementation, mathematics, and engineering references.

### Prologue: From Trial and Error to Modern Agents

| Reading                                                                    | Central question                                                                               |
| :------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| [Introduction to Reinforcement Learning](docs/en/preface/introduction.md)  | What does RL study, and how does the book connect classical methods to modern language models? |
| [A History of Reinforcement Learning](docs/preface/brief-history/index.md) | How did control, TD learning, DQN, AlphaGo, RLHF, and reasoning models develop?                |
| [Environment Setup](docs/preface/env-setup.md)                             | How do you prepare the environments for documentation, control tasks, and LLM experiments?     |

### Part I: The Language of Sequential Decisions

The book first makes an agent's failures and improvements observable, then develops the mathematical objects needed to describe long-term decisions.

| Ch. | Topic                                                                        | Through line                                                                                        |
| :-: | :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
|  1  | [Starting with CartPole](docs/chapter01_cartpole/principles.md)              | Use states, actions, rewards, policies, and training curves to see a complete RL loop.              |
|  2  | [RL Problems and Definitions](docs/chapter03_mdp/bandit.md)                  | Move from exploration and exploitation to MDPs, returns, trajectories, and partial observability.   |
|  3  | [Value Functions and Bellman Equations](docs/chapter03_mdp/value-bellman.md) | Express how a present action changes the future through state values, action values, and recursion. |
|  4  | [Classical RL Methods](docs/chapter03_mdp/dp-mc-td.md)                       | Compare dynamic programming, Monte Carlo, and temporal-difference learning.                         |

### Part II: Learning Values and Policies with Neural Networks

When the state space grows, tables no longer suffice. This part introduces function approximation and follows the value-learning and policy-learning routes into PPO and continuous control.

| Ch. | Topic                                                                             | Through line                                                                                           |
| :-: | :-------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
|  5  | [Deep Q-Networks](docs/chapter07_dqn/from-q-to-dqn.md)                            | Approximate action values with neural networks and stabilize learning with replay and target networks. |
|  6  | [Policy Gradient Methods](docs/chapter08_policy_gradient/policy-gradient.md)      | Optimize the policy directly, derive REINFORCE, and reduce variance with baselines.                    |
|  7  | [Actor-Critic Methods](docs/chapter09_actor_critic/advantage-function.md)         | Let policy and value estimation learn together, joined by the advantage function.                      |
|  8  | [TRPO and PPO](docs/chapter10_ppo/trust-region-clipping.md)                       | Limit each policy update and combine GAE with a clipped objective for stable learning.                 |
|  9  | [Continuous Control and World Models](docs/chapter11_continuous_control/intro.md) | Progress from DDPG, TD3, and SAC to model-based RL, MuZero, and Dreamer.                               |

### Part III: Expanding the Data, Task, and Agent

When interaction is expensive, expert demonstrations are available, or a task spans multiple agents and time scales, the object of learning changes.

| Ch. | Topic                                                                                                  | Through line                                                                             |
| :-: | :----------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| 10  | [Offline Reinforcement Learning](docs/chapter12_offline_rl/intro.md)                                   | Learn from a fixed dataset while controlling distribution shift and extrapolation error. |
| 11  | [Imitation, Inverse RL, and Meta-RL](docs/chapter13_imitation_meta_rl/bc-dagger.md)                    | Learn policies or rewards from experts and adapt to new tasks.                           |
| 12  | [Exploration, Multi-Agent, and Hierarchical RL](docs/chapter14_exploration_marl_hierarchical/intro.md) | Address sparse rewards, coordination, and the hierarchy of long-horizon tasks.           |

### Part IV: LLM Alignment and Post-Training

Language models expand an “action” into a passage of text. Policy optimization, distribution constraints, and credit assignment now lead into preference alignment, verifiable rewards, and inference-time computation.

| Ch. | Topic                                                                                        | Through line                                                                                                 |
| :-: | :------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| 13  | [The RLHF Training Pipeline](docs/chapter15_rlhf/base-model-to-assistant.md)                 | Move from SFT, AI feedback, and reward modeling to PPO-style RL fine-tuning and evaluation.                  |
| 14  | [Preference Alignment and the DPO Family](docs/chapter17_dpo/intro.md)                       | Derive DPO from a KL-constrained objective and compare preference-optimization assumptions.                  |
| 15  | [GRPO, RLVR, and Verifier Engineering](docs/chapter18_grpo/grpo-practice-and-mechanism.md)   | Train mathematical, coding, and tool-use capabilities with group-relative advantages and verifiable rewards. |
| 16  | [Reasoning Models and Inference-Time Compute](docs/chapter19_reasoning/emergence-and-o1.md)  | Explain long-reasoning training, compute-budget control, and chain-of-thought alignment.                     |
| 17  | [Process Rewards and Inference-Time Search](docs/chapter20_prm_search/outcome-vs-process.md) | Move supervision from final answers to intermediate steps and combine it with search.                        |
| 18  | [Industrial LLM RL](docs/chapter16_llm_rl_industrial/intro.md)                               | Scale a single-machine algorithm into a coordinated data, inference, training, and evaluation system.        |

### Part V: Tool Use and Agentic Reinforcement Learning

Once agents call tools across many environment steps, the unit of training becomes a trajectory. Credit assignment, environment construction, and safety boundaries become central.

| Ch. | Topic                                                                                         | Through line                                                                           |
| :-: | :-------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| 19  | [Tool Use, Multi-Turn Interaction, and Multi-Agent RL](docs/chapter22_agentic/overview.md)    | Formalize Agentic RL, synthesize tool trajectories, and run DeepCoder and FinQA labs.  |
| 20  | [Reinforcement Learning for Coding Agents](docs/chapter23_rl_based_swe/swe-bench-and-rlvr.md) | Study software-engineering agents through SWE-bench, code world models, and self-play. |
| 21  | [Deep Research and Browser Agents](docs/chapter24_deep_research/browser-rl-harness.md)        | Build trainable browser environments and evaluate deep-research systems.               |
| 22  | [Computer Use and GUI Agents](docs/chapter25_computer_use/training.md)                        | Train interface agents while handling instruction hierarchy and prompt injection.      |

### Part VI: Reinforcement Learning in Multimodal Worlds

Vision, audio, robot actions, and generative models introduce new state representations, reward sources, and evaluation criteria.

| Ch. | Topic                                                                                           | Through line                                                                        |
| :-: | :---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| 23  | [Vision-Language Model RL](docs/chapter26_vlm/vlm-challenges.md)                                | Design visual rewards and reflection, then run VLM-GRPO and GeoQA experiments.      |
| 24  | [Audio, Embodied Intelligence, and Visual Generation](docs/chapter27_audio_rl/reward-design.md) | Extend RLVR and RLHF to audio, VLA systems, image generation, and video generation. |

### Part VII: Safety, Evaluation, and Research Frontiers

A rising training reward only shows that the optimizer met its objective. This final part asks whether the objective was sound, whether the gain was real, and what new risks follow from broader capabilities.

| Ch. | Topic                                                                                            | Through line                                                                                   |
| :-: | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| 25  | [Reward Hacking and RL Evaluation](docs/chapter30_alignment_failures/classical-failures.md)      | Analyze specification gaming, spurious gains, sleeper behavior, and evaluation leakage.        |
| 26  | [Self-Play, Scaling, and Research Frontiers](docs/chapter32_selfplay/self-play-outlook/index.md) | Study self-play, RL scaling laws, multi-agent learning, and evolutionary scientific discovery. |

### Appendices: A Working Reference

| Appendix | Topic                                                                                         | Contents                                                                                            |
| :------: | :-------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
|    A     | [Training Debugging and Engineering](docs/appendix_industrial_training/training-debugging.md) | Training systems, parallelism, monitoring, agent sandboxes, and bad-case analysis.                  |
|    B     | [Core Algorithm Implementations](docs/appendix_code_cheatsheet/sft-kl.md)                     | Compact implementations of SFT, PPO, DPO, GRPO, DAPO, sampling, and attention.                      |
|    C     | [Learning and Reference Materials](docs/appendix_paper_reading/intro.md)                      | Paper roadmaps, GPU-hour estimates, a metrics glossary, and engineering exercises.                  |
|    D     | [Mathematical Foundations of RL](docs/appendix_math/linear-algebra-basics.md)                 | Progressive reviews of linear algebra, probability, calculus, optimization, and information theory. |

## Experiment Code

The [`code/`](code/) directory contains runnable examples aligned with course chapters. Each chapter's code is intentionally compact so it can be inspected, run, and modified independently.

| Area                   | Code Path                                                                                                          | Representative Experiments                                                                    |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| Classic control        | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                             | Train CartPole, inspect rewards and episode length, and compare PPO implementations.          |
| Preference fine-tuning | [`code/chapter17_dpo/`](code/chapter17_dpo/)                                                                       | Train a DPO model and inspect preference accuracy, reward margin, and KL drift.               |
| MDP and value learning | [`code/chapter03_mdp/`](code/chapter03_mdp/)                                                                       | Run bandit strategies, solve GridWorld, and verify Bellman updates numerically.               |
| Deep Q-learning        | [`code/chapter04_dqn/`](code/chapter04_dqn/)                                                                       | Implement replay buffers, target networks, and Double DQN variants.                           |
| Policy gradient        | [`code/chapter05_policy_gradient/`](code/chapter05_policy_gradient/)                                               | Compare REINFORCE, baseline variants, and Actor-Critic updates.                               |
| PPO                    | [`code/chapter07_ppo/`](code/chapter07_ppo/)                                                                       | Train LunarLander, inspect clipping, visualize GAE, and compare training stability.           |
| RLHF                   | [`code/chapter08_rlhf/`](code/chapter08_rlhf/)                                                                     | Walk through SFT, reward model training, PPO-style alignment, and veRL/GSM8K adapter scripts. |
| Alignment and RLVR     | [`code/chapter09_alignment/`](code/chapter09_alignment/), [`code/chapter09_grpo_rlvr/`](code/chapter09_grpo_rlvr/) | Explore DPO rewards, GRPO group advantages, and rule-based verifiable rewards.                |
| VLM and agents         | [`code/chapter10_agentic_rl/`](code/chapter10_agentic_rl/), [`code/chapter11_vlm_rl/`](code/chapter11_vlm_rl/)     | Build tool-use agent trajectory synthesis and implement multimodal model RL examples.         |
| Advanced topics        | [`code/chapter12_future_trends/`](code/chapter12_future_trends/)                                                   | Study frontier directions including multi-agent RL and model-based RL.                        |

See [`code/README.md`](code/README.md) for a code index and chapter-specific dependency notes.

## Recommended Learning Path

For a first systematic reading, follow the chapters in order. Chapters 1–4 establish the language and recursive ideas of RL; Chapters 5–9 develop the algorithmic core of deep RL; Chapters 10–12 expand the data and task settings. Together, these parts provide the foundation for the rest of the book.

Readers focused on LLM post-training can enter Chapters 13–18 after completing Chapters 6–8. Policy gradients, advantage estimation, PPO, and KL constraints directly explain the objectives used by RLHF, DPO, and GRPO. Then select the relevant topics from Chapters 19–24 for Agentic or multimodal RL. Chapters 25–26 are worth reading alongside any experiment because reward and evaluation errors affect every method in the book.

For each chapter: restate the problem it solves, derive the central equation, run at least one experiment, and change one important assumption to explain the resulting metrics. Use the appendices as references when mathematics or engineering details arise; they do not need to be read front to back first.

## Quick Start

### Read Online

Published course site:

```text
https://walkinglabs.github.io/hands-on-modern-rl/
```

### Run the Documentation Site Locally

Requirements:

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

Then open the local VitePress URL shown in the terminal, usually:

```text
http://localhost:5173
```

### Verify the Site

Before submitting a pull request that changes documentation structure, theme code, navigation, build scripts, or generated assets, run:

```bash
npm run verify
```

This checks formatting, lints the VitePress theme, builds the site, and verifies expected build artifacts.

### Run Course Code

Most code examples use Python and are organized by chapter.

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For smaller installs, use chapter-specific requirements files:

```bash
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

Some chapters may require additional system libraries, GPU support, model downloads, or environment-specific setup. Start with Chapter 01 before running examples that involve LLMs, VLMs, or heavy simulators.

## Repository Structure

```text
hands-on-modern-rl/
|-- docs/                      # VitePress course content
|   |-- .vitepress/            # Site config, navigation, and theme overrides
|   |-- public/                # Static assets copied into the built site
|   |-- preface/               # Course introduction and history
|   |-- chapter*/              # Main course chapters
|   |-- appendix*/             # Supplementary material and references
|   `-- summaries/             # Part-level review and summary notes
|-- code/                      # Runnable examples aligned with chapters
|-- scripts/                   # Maintenance and verification scripts
|-- package.json               # Site scripts and dependencies
|-- AGENTS.md                  # Repository maintenance guide
`-- README.md                  # Main project overview
```

## Development Commands

```bash
npm run dev           # Start the local documentation server
npm run build         # Build the static site
npm run preview       # Preview the built site locally
npm run format        # Format repository files with Prettier
npm run format:check  # Check formatting
npm run lint          # Lint VitePress theme code
npm run verify        # Run format check, lint, build, and artifact verification
```

## Contributing

Contributions should make the course clearer, more accurate, easier to reproduce, or easier to navigate.

Good contributions include:

- Fixing conceptual errors, formulas, diagrams, broken links, or typos.
- Improving explanations without changing the intended learning path.
- Adding small, reproducible experiments that clarify existing chapters.
- Improving scripts, build reliability, navigation, or accessibility.
- Adding high-quality references to papers, official documentation, or widely used open-source implementations.

Please keep pull requests focused. A good PR usually changes one chapter, one experiment, one group of diagrams, or one infrastructure issue at a time.

When adding content:

1. Put course material under [`docs/`](docs/).
2. Use kebab-case for new directories and files.
3. Prefer directory-based routes with `index.md`.
4. Update [`docs/.vitepress/config.mjs`](docs/.vitepress/config.mjs) when adding navigable pages.
5. Run `npm run verify` before requesting review if your change touches config, theme, scripts, or generated site output.
6. Use Conventional Commits, such as `docs: clarify ppo clipping` or `fix: repair chapter link`.

For repository-specific maintenance rules, see [`AGENTS.md`](AGENTS.md).

## Other Courses

Our team has also created other courses. Take a look:

- [**Learn Harness Engineering**](https://github.com/walkinglabs/learn-harness-engineering) — A course on Harness Engineering for AI coding agents. Through 12 lectures and 6 projects, it teaches you to build instructions, state management, verification, and control mechanisms that make model output reliable.
- [**Modern LLM Notebook**](https://github.com/walkinglabs/modern-llm-notebook) — Build modern LLMs from scratch through 23 runnable Jupyter Notebooks in PyTorch, covering Tokenizer, Transformer, training, inference, alignment, and frontier topics.

## Discussion Group (WeChat)

For suggestions or feedback, scan the QR code to join the discussion group (WeChat):

<img
  src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png"
  alt="Discussion Group"
  style="width: 100%; max-width: 520px; height: auto;"
/>

## Citation

If you use this course in teaching materials, study notes, or derivative non-commercial educational work, please cite the repository:

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

## Acknowledgements

We thank [OpenAI](https://openai.com/) for providing development resources and [AMD](https://www.amd.com/) for providing computing resources that support this project. Without their support, this course could not have evolved so quickly.

## License

This course is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).

You may share and adapt the material for non-commercial purposes, provided that you give appropriate credit and distribute derivative works under the same license.

---

<div align="center">
  <sub>Maintained by WalkingLabs and contributors.</sub>
</div>
