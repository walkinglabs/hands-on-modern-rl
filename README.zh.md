<div align="center">
  <img src="docs/public/readme/readmelogo.png" alt="Hands-On Modern RL" width="500" />
  <p><em>从马尔可夫决策过程与策略优化，到大模型推理、智能体和多模态系统</em></p>

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
    <a href="https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole">ModelScope 一键训练</a> ·
    <a href="https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment01-cartpole.ipynb">运行 CartPole Notebook</a> ·
    <a href="https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole/file/view/master/train.py">ModelScope 训练脚本</a>
  </p>

  <p>
    <a href="#读者交流群微信">读者交流群（微信）</a>
  </p>

  <p>
    <a href="#本书特色">本书特色</a> ·
    <a href="#本书介绍">本书介绍</a> ·
    <a href="#🔥-最新动态-news">最新动态</a> ·
    <a href="#目录">目录</a> ·
    <a href="#全书结构">全书结构</a> ·
    <a href="#实验代码">实验代码</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#参与贡献">参与贡献</a>
  </p>
</div>

> **📣 公告**
>
> 感谢大家对教程的支持！近期将会有版本更新，目前很多内容还在整理和完善中，请大家多点耐心。也欢迎大家多提建议。

## 🔥 最新动态 (News)

> ⚠️ **备注**：本教程由于有 AI 协助生成，目前尚未全面审稿结束，很有可能会有事实性或代码不可运行的错误。欢迎大家在阅读过程中提交 Issue 或 PR 帮助指正。

- **[2026-08-19]** 🎮 **经典强化学习在线环境与脚本更新**：过去两周，我们集中上线并完善了一批强化学习在线环境、训练脚本和配套 Notebook。现在可以直接在线运行经典强化学习实验，查看训练日志与评估结果，更方便地学习经典强化学习。同时，我们还修复了此前反馈的许多 Bug，涉及课程内容、链接和实验代码。
- **[2026-05-15]** 📖 **全量英文翻译与 PDF 发布**：全部章节英文翻译完成，中英文版 PDF 均通过 CI 自动构建发布。
- **[2026-05-13]** 🚀 **全面升级大模型与传统强化学习实战**：新增可复现的 **Agentic RL**（Deep Research / rLLM）与 **传统 RL**（Actor-Critic 连续控制）训练实例。包含从零构建 Agentic 训练系统的完整代码与微调过程解析，并同步上线 VLM 强化学习（GeoQA 几何推理）动手实验！
- **[2026-05-02]** 🎉 教程初期浏览版正式开源发布，开放测试与建议收集。

## 在线训练 Notebook

WalkingLab 与 ModelScope 合作，为经典强化学习实验提供在线训练环境。ModelScope 创空间将实验界面、运行环境和训练入口集中在一个页面中，读者无需先配置本地环境，即可通过浏览器启动训练并观察智能体的行为。

每个创空间都在 [`code/online-experiments`](code/online-experiments/README.md) 下配有实验 Notebook。Notebook 与创空间复用同一份训练运行时，可以调整实验参数、查看完整训练日志与检查点评估曲线，并显示本次训练生成的策略回放或结果文件。

| 实验                        | 资源 | 配套 Notebook                                                                                                                                                                          | 在线创空间                                                                                                |
| --------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 01 · CartPole PPO           | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment01-cartpole.ipynb)           | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole)           |
| Gymnasium 训练合集          | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment-gymnasium.ipynb)            | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium)            |
| 02 · ViZDoom                | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment02-vizdoom.ipynb)            | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment02-vizdoom)            |
| 03 · Atari / ALE            | xGPU | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb)              | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari)              |
| 04 · 棋盘游戏与自博弈       | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment04-board-selfplay.ipynb)     | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay)     |
| 05 · 多智能体游戏           | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment05-multiagent-games.ipynb)   | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment05-multiagent-games)   |
| 06 · MiniGrid 探索          | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment06-minigrid-adventure.ipynb) | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment06-minigrid-adventure) |
| 07 · JAX MinAtar            | CPU  | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment07-jax-games.ipynb)          | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games)          |
| 08 · ManiSkill              | xGPU | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment08-maniskill.ipynb)          | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill)          |
| 10 · MineStudio / Minecraft | xGPU | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment10-minestudio.ipynb)         | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio)         |
| 11 · Unity ML-Agents        | xGPU | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment11-unity-mlagents.ipynb)     | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents)     |
| 12 · AI2-THOR               | xGPU | [运行 Notebook](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment12-ai2thor-embodied.ipynb)   | [打开创空间](https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied)   |

CPU 实验可使用普通 Notebook 运行；实验 03、08、10、11、12 需要调度 ModelScope xGPU Notebook，训练单元会在开始前检查 CUDA。

## 本书特色

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-learning-path.png" alt="课程学习地图截图" width="100%" />
      <br />
      <strong>一条连续的知识主线</strong>
      <br />
      <sub>从一次 CartPole 试错出发，逐步走向价值学习、策略优化与现代智能体。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-code-focus.png" alt="PPO 代码聚焦截图" width="100%" />
      <br />
      <strong>公式与代码互相印证</strong>
      <br />
      <sub>PPO、DPO、GRPO 的关键推导紧邻实现，让每个张量都有明确含义。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-training-metrics.png" alt="CartPole 训练指标截图" width="100%" />
      <br />
      <strong>用实验检验判断</strong>
      <br />
      <sub>真实训练曲线、消融结果和失败信号帮助读者理解算法何时有效。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-rlhf-pipeline.png" alt="RLHF 流水线截图" width="100%" />
      <br />
      <strong>连接经典 RL 与大模型</strong>
      <br />
      <sub>从策略梯度和 PPO 出发，自然推导 RLHF、DPO、GRPO 与 RLVR。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-agentic-rl.png" alt="Agentic RL 实验页面截图" width="100%" />
      <br />
      <strong>把智能体还原为序贯决策</strong>
      <br />
      <sub>把工具调用、浏览器操作和代码修复写成状态、动作、轨迹与信用分配问题。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-atari-game.png" alt="Atari Pong DQN 实验页面截图" width="100%" />
      <br />
      <strong>从现象进入理论</strong>
      <br />
      <sub>CartPole、LunarLander、Atari 与大模型实验先提出问题，再引出所需数学工具。</sub>
    </td>
  </tr>
</table>

---

> [!NOTE]
> 希望本开源教程能够让更多人拥有向智能上限发起攀登的勇气，解决更多通往 AGI 道路上的问题。
>
> 当前教程快速迭代中。建议只看非 🚧 状态的章节，🚧状态的章节很可能有错误，也欢迎修正和建议 。

> **寻求帮助**
>
> 由于资源稀缺问题，我们正在寻求显卡支持，如果您有显卡使用方式愿意支持非常欢迎联系 physicoada@gmail.com。

## 目录

- [本书特色](#本书特色)
- [本书介绍](#本书介绍)
- [全书结构](#全书结构)
- [实验代码](#实验代码)
- [推荐学习路径](#推荐学习路径)
- [快速开始](#快速开始)
- [参与贡献](#参与贡献)
- [引用](#引用)
- [致谢](#致谢)
- [开源协议](#开源协议)

## 本书介绍

强化学习研究一个朴素而困难的问题：一个系统采取行动、观察结果，再根据结果改进下一次行动。当奖励会延迟出现、环境信息并不完整、一次更新还会改变后续采样分布时，监督学习中熟悉的“输入—标签”框架便不再够用。我们需要描述交互过程，估计长期回报，并在有限数据下稳定地改进策略。

**Hands-On Modern RL** 围绕这条问题主线展开。全书从 CartPole 和多臂老虎机开始，让状态、动作、奖励与策略先成为可以观察的对象；随后用马尔可夫决策过程、价值函数和贝尔曼方程建立统一语言，再依次进入 DQN、策略梯度、Actor-Critic、PPO、连续控制和离线强化学习。掌握这些工具后，大模型后训练中的 RLHF、DPO、GRPO 与 RLVR 就不再是一组孤立名词，而是同一套序贯决策思想在语言模型上的延伸。

后半部分把环境扩展到工具、浏览器、代码仓库、视觉与音频世界。动作可能是一段文字、一次函数调用或一组界面操作，奖励也可能来自人类偏好、规则验证器或过程奖励模型。问题的形式变得更丰富，但贯穿全书的三个问题保持一致：**如何表示决策过程，如何把结果归因给此前的动作，如何确认策略真的变得更好。**

### 如何讲解

每章遵循“问题—方法—实验—反思”的节奏。先用一个具体任务暴露困难，再引入解决困难所需的概念和公式；随后把公式落到可运行代码、训练曲线与评测指标上；最后检查方法的假设、失败模式和适用边界。数学用于解释现象，实验用于检验推理，两者共同组成完整论证。

书中的代码尽量保留算法骨架。你会看到轨迹如何收集、优势如何估计、损失如何组成、指标如何变化，也会看到奖励黑客、KL 漂移、熵坍缩、分布偏移与评测泄漏怎样使一个看似成功的训练失效。

### 适合谁读

本书适合学过基础机器学习、希望系统掌握现代强化学习的学生、研究者与工程师。读者应具备 Python 编程经验和基础 PyTorch 能力，并了解线性代数、概率论与微积分的基本概念。数学附录会在相关章节所需的深度上重新建立这些工具，因此无需先修完一整套高等数学课程。

读完并完成核心实验后，你将能够：

- 用 MDP、价值函数、贝尔曼方程和信用分配描述一个新的决策问题；
- 实现、阅读并诊断 DQN、REINFORCE、Actor-Critic、PPO、DPO 与 GRPO；
- 解释 SFT、奖励建模、偏好优化、RLHF 与 RLVR 在大模型后训练中的关系；
- 为工具调用、代码智能体和多模态系统设计轨迹、奖励、训练与评测流程；
- 识别训练曲线背后的失败模式，并用对照实验检验算法改进。

### 当前状态

本仓库是一个活跃的课件项目。课程内容正在逐章扩展和完善，重点关注正确性、可运行的示例和稳定的学习路径。

- 课程网站: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- 源码内容: [`docs/`](docs/)
- 可运行示例: [`code/`](code/)
- 本地验证: `npm run verify`
- 开源协议: [CC BY-NC-SA 4.0](LICENSE)

欢迎提交 Issue 和 Pull Request 来修复拼写错误、修正概念、改进可复现性、补充参考文献以及在合理范围内的课程扩展。

## 🗺️ 演进路线图 (Roadmap)

本课程正在持续迭代中，以下是接下来的开发计划：

- [x] **2026-05-02**：开源初始浏览版，用于收集社区测试反馈。
- [x] **2026-05-10**：发布正式小版本，修正初版笔误，稳定第一部分（基础）与第二部分（核心理论）的内容与代码。
- [x] **2026-05 下旬**：完善大模型强化学习可复现实验，补充完整的 RLVR（可验证奖励）实战与评估。
- [ ] **2026-06 上旬**：分步骤交付 Agentic RL 动手项目（从单工具调用到 Deep Research 复杂轨迹合成）。
- [ ] **2026-06 下旬**：增加基于 Unity 的具身强化学习（Embodied RL）可训练模型环境与项目。
- [ ] **2026-07 及以后**：扩展多模态前沿，补充 VLM 强化学习或 Diffusion RL 的完整实战案例。

## 全书结构

全书共七部分、二十六章。前三部分建立强化学习的统一语言和算法基础；第四部分把这些工具带入大语言模型后训练；第五、六部分研究动作空间扩展到工具和多模态世界后出现的新问题；第七部分讨论如何发现失败、建立可靠评测并继续推进研究。附录提供算法实现、数学基础与工程查阅资料。

### 序章：从试错学习到现代智能体

| 内容                                                  | 要解决的问题                                               |
| :---------------------------------------------------- | :--------------------------------------------------------- |
| [强化学习导论](docs/preface/introduction.md)          | 强化学习研究什么，本书如何把经典方法与现代大模型连接起来。 |
| [强化学习发展史](docs/preface/brief-history/index.md) | 从早期控制、TD 学习和 DQN，到 AlphaGo、RLHF 与推理模型。   |
| [环境配置](docs/preface/env-setup.md)                 | 搭建文档、经典控制和大模型实验所需的运行环境。             |

### 第一部分：决策问题的基本语言

先观察一个智能体如何失败与改进，再建立描述长期决策所需的数学对象。

| 章  | 主题                                                        | 本章主线                                                       |
| :-: | :---------------------------------------------------------- | :------------------------------------------------------------- |
|  1  | [CartPole 入门](docs/chapter01_cartpole/principles.md)      | 从倒立摆的状态、动作、奖励和训练曲线认识完整的强化学习循环。   |
|  2  | [强化学习问题与基本定义](docs/chapter03_mdp/bandit.md)      | 从探索与利用进入 MDP、策略、回报、轨迹与部分可观测性。         |
|  3  | [价值函数与贝尔曼方程](docs/chapter03_mdp/value-bellman.md) | 用状态价值、动作价值和递推关系表达“当前行动对未来有多好”。     |
|  4  | [经典强化学习方法](docs/chapter03_mdp/dp-mc-td.md)          | 比较动态规划、蒙特卡洛和时序差分，理解模型、采样与自举的取舍。 |

### 第二部分：用神经网络学习价值与策略

状态空间增大后，表格方法无法继续工作。这一部分引入函数近似，并沿着价值学习和策略学习两条路线走向 PPO 与连续控制。

| 章  | 主题                                                                   | 本章主线                                                      |
| :-: | :--------------------------------------------------------------------- | :------------------------------------------------------------ |
|  5  | [深度 Q 网络](docs/chapter07_dqn/from-q-to-dqn.md)                     | 用神经网络近似动作价值，并用经验回放和目标网络稳定训练。      |
|  6  | [策略梯度方法](docs/chapter08_policy_gradient/policy-gradient.md)      | 直接优化策略，推导 REINFORCE，并用基线降低梯度方差。          |
|  7  | [Actor-Critic 方法](docs/chapter09_actor_critic/advantage-function.md) | 让策略与价值估计协同学习，以优势函数连接两条算法路线。        |
|  8  | [TRPO 与 PPO](docs/chapter10_ppo/trust-region-clipping.md)             | 控制单次策略更新的幅度，用 GAE 与裁剪目标获得稳定训练。       |
|  9  | [连续控制与世界模型](docs/chapter11_continuous_control/intro.md)       | 从 DDPG、TD3、SAC 进入基于模型的强化学习、MuZero 与 Dreamer。 |

### 第三部分：数据、任务与智能体结构的扩展

当交互昂贵、专家数据可用，或任务需要多个主体和多层时间尺度时，训练对象随之改变。

| 章  | 主题                                                                                  | 本章主线                                                   |
| :-: | :------------------------------------------------------------------------------------ | :--------------------------------------------------------- |
| 10  | [离线强化学习](docs/chapter12_offline_rl/intro.md)                                    | 在固定数据集上学习策略，处理分布偏移、外推误差与序列建模。 |
| 11  | [模仿学习、逆强化学习与元强化学习](docs/chapter13_imitation_meta_rl/bc-dagger.md)     | 从专家行为学习策略或奖励，并让智能体适应新任务。           |
| 12  | [探索、多智能体与分层强化学习](docs/chapter14_exploration_marl_hierarchical/intro.md) | 研究稀疏奖励、主体协作和长时程任务的分层结构。             |

### 第四部分：大语言模型对齐与后训练

语言模型把“动作”扩展为一段文本。前面学到的策略优化、分布约束和信用分配由此进入偏好对齐、可验证奖励与推理时计算。

| 章  | 主题                                                                              | 本章主线                                                       |
| :-: | :-------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| 13  | [RLHF 训练流水线](docs/chapter15_rlhf/base-model-to-assistant.md)                 | 从 SFT、AI 反馈和奖励建模，走到 PPO 式强化学习微调与对齐评测。 |
| 14  | [偏好对齐与 DPO 家族](docs/chapter17_dpo/intro.md)                                | 从 KL 约束目标推导 DPO，并比较不同偏好优化方法的假设与指标。   |
| 15  | [GRPO、RLVR 与 Verifier 工程](docs/chapter18_grpo/grpo-practice-and-mechanism.md) | 用组相对优势和可验证奖励训练数学、代码与工具调用能力。         |
| 16  | [推理模型与推理时计算](docs/chapter19_reasoning/emergence-and-o1.md)              | 解释长推理的训练机制、计算预算控制与推理链对齐。               |
| 17  | [过程奖励与推理时搜索](docs/chapter20_prm_search/outcome-vs-process.md)           | 把监督信号从最终答案推进到中间步骤，并结合搜索提高解题可靠性。 |
| 18  | [大模型 RL 工业实践](docs/chapter16_llm_rl_industrial/intro.md)                   | 把单机算法扩展为数据、推理、训练、同步和评测协作的系统。       |

### 第五部分：工具调用与 Agentic 强化学习

智能体开始调用工具并跨越多个环境步骤后，训练单位从单段回答变成完整轨迹，信用分配、环境构造和安全边界成为核心问题。

| 章  | 主题                                                                               | 本章主线                                                              |
| :-: | :--------------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| 19  | [工具调用、多轮交互与多智能体 RL](docs/chapter22_agentic/overview.md)              | 形式化 Agentic RL，构造工具轨迹，并完成 DeepCoder 与 FinQA 训练实验。 |
| 20  | [代码智能体强化学习](docs/chapter23_rl_based_swe/swe-bench-and-rlvr.md)            | 用 SWE-bench、代码世界模型和自博弈研究软件工程智能体。                |
| 21  | [Deep Research 与浏览器智能体](docs/chapter24_deep_research/browser-rl-harness.md) | 构建可训练的浏览器环境，并建立深度研究任务的评测方法。                |
| 22  | [Computer Use 与 GUI Agent](docs/chapter25_computer_use/training.md)               | 训练界面操作智能体，处理指令层级与提示注入攻击。                      |

### 第六部分：多模态世界中的强化学习

视觉、音频、机器人动作和生成模型带来新的状态表示、奖励来源与评测标准。

| 章  | 主题                                                                 | 本章主线                                                |
| :-: | :------------------------------------------------------------------- | :------------------------------------------------------ |
| 23  | [视觉语言模型强化学习](docs/chapter26_vlm/vlm-challenges.md)         | 设计视觉奖励与反思机制，并完成 VLM-GRPO 和 GeoQA 实验。 |
| 24  | [音频、具身智能与视觉生成](docs/chapter27_audio_rl/reward-design.md) | 把 RLVR 与 RLHF 扩展到音频、VLA、图像生成和视频生成。   |

### 第七部分：安全、评估与研究前沿

训练奖励上升只说明优化器完成了目标。最后一部分检验目标是否正确、收益是否可信，以及智能体能力扩大后会出现哪些新风险。

| 章  | 主题                                                                           | 本章主线                                                     |
| :-: | :----------------------------------------------------------------------------- | :----------------------------------------------------------- |
| 25  | [奖励黑客与 RL 评估](docs/chapter30_alignment_failures/classical-failures.md)  | 分析规范博弈、假性收益、潜伏行为与评测泄漏，并建立防御机制。 |
| 26  | [自博弈、规模化与研究前沿](docs/chapter32_selfplay/self-play-outlook/index.md) | 研究自博弈、RL Scaling Laws、多智能体学习与进化式科学发现。  |

### 附录：随学随查的工具箱

| 附录 | 主题                                                                          | 内容                                                 |
| :--: | :---------------------------------------------------------------------------- | :--------------------------------------------------- |
|  A   | [训练调试与工程实践](docs/appendix_industrial_training/training-debugging.md) | 训练系统、并行策略、监控指标、Agent 沙箱与坏例分析。 |
|  B   | [核心算法实现](docs/appendix_code_cheatsheet/sft-kl.md)                       | SFT、PPO、DPO、GRPO、DAPO、采样与注意力的紧凑实现。  |
|  C   | [学习资源与参考资料](docs/appendix_paper_reading/intro.md)                    | 论文路线、GPU 小时估算、指标词典与工业实践练习。     |
|  D   | [强化学习的数学基础](docs/appendix_math/linear-algebra-basics.md)             | 线性代数、概率、微积分、优化与信息论的渐进式复习。   |

## 实验代码

[`code/`](code/) 目录包含与各章节对齐的可运行示例。每章的代码都设计得足够精简，以便独立检查、运行和修改。

| 领域           | 代码路径                                                                                                           | 代表性实验                                                     |
| :------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| 经典控制       | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                             | 训练 CartPole，检查奖励和回合长度，比较 PPO 实现。             |
| 偏好微调       | [`code/chapter17_dpo/`](code/chapter17_dpo/)                                                                       | 训练 DPO 模型，检查偏好准确率、奖励边际与 KL 漂移。            |
| MDP 与价值学习 | [`code/chapter03_mdp/`](code/chapter03_mdp/)                                                                       | 运行老虎机策略，求解网格世界，用数值方法验证贝尔曼更新。       |
| 深度 Q 学习    | [`code/chapter04_dqn/`](code/chapter04_dqn/)                                                                       | 实现经验回放、目标网络和 Double DQN 变体。                     |
| 策略梯度       | [`code/chapter05_policy_gradient/`](code/chapter05_policy_gradient/)                                               | 比较 REINFORCE、基线变体和 Actor-Critic 更新。                 |
| PPO            | [`code/chapter07_ppo/`](code/chapter07_ppo/)                                                                       | 训练 LunarLander，检查截断机制，可视化 GAE，并比较训练稳定性。 |
| RLHF           | [`code/chapter08_rlhf/`](code/chapter08_rlhf/)                                                                     | 走通 SFT、奖励模型训练、PPO 风格对齐和 veRL/GSM8K 适配脚本。   |
| 对齐与 RLVR    | [`code/chapter09_alignment/`](code/chapter09_alignment/), [`code/chapter09_grpo_rlvr/`](code/chapter09_grpo_rlvr/) | 探索 DPO 奖励、GRPO 组优势和基于规则的可验证奖励。             |
| VLM 与智能体   | [`code/chapter10_agentic_rl/`](code/chapter10_agentic_rl/), [`code/chapter11_vlm_rl/`](code/chapter11_vlm_rl/)     | 构建工具调用智能体轨迹综合，实现多模态模型强化学习等。         |
| 高级主题       | [`code/chapter12_future_trends/`](code/chapter12_future_trends/)                                                   | 学习前沿方向包括多智能体强化学习、Model-Based RL等。           |

参见 [`code/README.md`](code/README.md) 获取代码索引和各章节的依赖说明。

## 推荐学习路径

第一次系统学习建议按章顺序阅读。第 1—4 章建立术语与递推思想，第 5—9 章给出深度强化学习的算法骨架，第 10—12 章扩展数据与任务设定。这三部分是后续内容的共同基础。

如果主要目标是大模型后训练，可以在完成第 6—8 章后进入第 13—18 章。策略梯度、优势估计、PPO 与 KL 约束会直接解释 RLHF、DPO 和 GRPO 的目标函数。学习 Agentic RL 或多模态 RL 时，再选择第 19—24 章中的对应专题。第 25—26 章适合与任何实验并行阅读，因为奖励设计和评测错误会影响全书所有方法。

每章建议完成四件事：复述本章要解决的问题；手推核心公式；运行至少一个实验；改变一个关键假设并解释指标变化。遇到数学或工程细节时，按需查阅附录，无需先把附录顺序读完。

## 快速开始

### 在线阅读

发布的课程网站地址：

```text
https://walkinglabs.github.io/hands-on-modern-rl/
```

### 本地运行文档网站

环境要求：

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

然后在浏览器中打开终端显示的本地 VitePress 服务地址，通常是：

```text
http://localhost:5173
```

### 验证网站

在提交更改文档结构、主题代码、导航、构建脚本或生成资产的 Pull Request 之前，请运行：

```bash
npm run verify
```

这会检查代码格式，Lint VitePress 主题，构建网站，并验证预期的构建产物。

### 运行课程代码

大多数代码示例基于 Python，并按章节组织。

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

对于较小的安装需求，建议使用章节特定的 requirements 文件：

```bash
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

某些章节可能需要额外的系统库、GPU 支持、模型下载或特定环境的设置。建议在运行涉及 LLM、VLM 或重度仿真器的示例前，先从第 01 章开始。

## 仓库结构

```text
hands-on-modern-rl/
├── docs/                      # VitePress 课程内容
│   ├── .vitepress/            # 网站配置、导航、主题覆盖
│   ├── public/                # 复制到构建后网站的静态资产
│   ├── preface/               # 课程简介和历史背景
│   ├── chapter*/              # 主要课程章节
│   ├── appendix*/             # 补充材料和参考文献
│   └── summaries/             # 部分级别的回顾和总结笔记
├── code/                      # 与章节对齐的可运行示例
├── scripts/                   # 维护和验证脚本
├── package.json               # 网站脚本和依赖
├── AGENTS.md                  # 仓库维护指南
└── README.md                  # 项目总览
```

## 开发命令

```bash
npm run dev           # 启动本地文档服务器
npm run build         # 构建静态网站
npm run preview       # 在本地预览构建后的网站
npm run format        # 使用 Prettier 格式化仓库文件
npm run format:check  # 检查代码格式
npm run lint          # Lint VitePress 主题代码
npm run verify        # 运行格式检查、Lint、构建和产物验证
```

## 参与贡献

所有的贡献都应旨在让课程更清晰、更准确、更易于复现或更易于导航。

优秀的贡献包括：

- 修复概念错误、公式、图表、失效链接或拼写错误；
- 在不改变预期学习路径的情况下改进解释说明；
- 添加能够阐明现有章节的、可复现的小型实验；
- 改进脚本、构建可靠性、导航或可访问性；
- 添加高质量论文、官方文档或广泛使用的开源实现的参考文献。

请保持 Pull Request 的聚焦。一个好的 PR 通常一次只修改一个章节、一个实验、一组图表或一个基础设施问题。

添加内容时：

1. 将课程资料放在 [`docs/`](docs/) 目录下。
2. 为新目录和文件使用 kebab-case（短横线分隔）命名。
3. 优先使用基于目录的路由（即 `index.md`）。
4. 添加可导航页面时，更新 [`docs/.vitepress/config.mjs`](docs/.vitepress/config.mjs)。
5. 当更改涉及配置、主题、脚本或生成的网站输出时，在请求 Review 之前运行 `npm run verify`。
6. 使用 Conventional Commits 规范，例如 `docs: clarify ppo clipping` 或 `fix: repair chapter link`。

有关特定于本仓库的维护规则，请参阅 [`AGENTS.md`](AGENTS.md)。

## 其他课程

我们的团队还制作了其他课程！请查看：

- [**Learn Harness Engineering**](https://github.com/walkinglabs/learn-harness-engineering) — 面向 AI 编程智能体的 Harness Engineering 课程。通过 12 节讲座和 6 个实战项目，教你构建指令、状态管理、验证与控制机制，让模型输出真正可靠。
- [**Modern LLM Notebook**](https://github.com/walkinglabs/modern-llm-notebook) — 通过 23 个可运行的 Jupyter Notebook，从零用 PyTorch 实现现代 LLM 核心组件，涵盖 Tokenizer、Transformer、训练、推理、对齐与前沿主题。

## 读者交流群（微信）

有任何建议 / 反馈，欢迎扫码加入读者交流群（微信）：

<img
  src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png"
  alt="读者交流群"
  style="width: 100%; max-width: 520px; height: auto;"
/>

## 引用

如果您在教学材料、学习笔记或衍生非商业教育作品中使用本课程，请引用本仓库：

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

## 致谢

感谢 [OpenAI](https://openai.com/) 提供的开发资源支持，以及 [AMD](https://www.amd.com/) 提供的算力支持。没有他们的支持，本教程不可能迭代得如此之快。

## 开源协议

本课程资料在 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE) 下发布。

您可以出于非商业目的共享和修改本材料，前提是必须给出适当的署名，并且衍生作品也必须在相同的协议下分发。

---

<div align="center">
  <sub>由 WalkingLabs 及贡献者维护。</sub>
</div>
