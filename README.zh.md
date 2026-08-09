<div align="center">
  <img src="docs/public/readme/readmelogo.png" alt="Hands-On Modern RL" width="500" />
  <p><em>现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体。</em></p>

  <p>
    <a href="https://cdiddy77.github.io/hands-on-modern-rl/"><img src="https://img.shields.io/badge/Course-Online-2563eb?style=flat-square" alt="Online Course" /></a>
    <a href="https://github.com/cdiddy77/hands-on-modern-rl/releases/latest"><img src="https://img.shields.io/badge/PDF-Download-e11d48?style=flat-square" alt="PDF Download" /></a>
    <a href="https://github.com/cdiddy77/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-111827?style=flat-square" alt="CC BY-NC-SA 4.0 License" /></a>
    <img src="https://img.shields.io/badge/Node-%3E%3D18-16a34a?style=flat-square" alt="Node >= 18" />
    <img src="https://img.shields.io/badge/Docs-VitePress-646cff?style=flat-square" alt="VitePress" />
  </p>

  <p>
    <a href="README.md">English</a> ·
    <a href="README.zh.md">中文</a>
  </p>

  <p>
    <a href="#读者交流群微信">读者交流群（微信）</a>
  </p>

  <p>
    <a href="#课程简介">课程简介</a> ·
    <a href="#简介">简介</a> ·
    <a href="#🔥-最新动态-news">最新动态</a> ·
    <a href="#目录">目录</a> ·
    <a href="#课程大纲">课程大纲</a> ·
    <a href="#实验代码">实验代码</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#参与贡献">参与贡献</a>
  </p>
</div>

> **📣 公告**
>
> 感谢大家对教程的支持！近期将会有版本更新，目前很多内容还在整理和完善中，请大家多点耐心。也欢迎大家多提建议。

## 课程简介

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-learning-path.png" alt="课程学习地图截图" width="100%" />
      <br />
      <strong>一眼看懂的学习地图</strong>
      <br />
      <sub>从前言、基础导论到前沿专题，章节树和页内大纲帮助你快速定位。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-code-focus.png" alt="PPO 代码聚焦截图" width="100%" />
      <br />
      <strong>代码逐行聚焦</strong>
      <br />
      <sub>PPO、DPO、GRPO 关键实现配有代码地图，把公式落到可读代码。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-training-metrics.png" alt="CartPole 训练指标截图" width="100%" />
      <br />
      <strong>训练指标可视化</strong>
      <br />
      <sub>真实曲线、指标解释和失败信号放在一起，方便边跑实验边诊断。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-rlhf-pipeline.png" alt="RLHF 流水线截图" width="100%" />
      <br />
      <strong>LLM 后训练流水线</strong>
      <br />
      <sub>RLHF、DPO、GRPO、RLVR 等主题以流程、artifact 和案例串联起来。</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-agentic-rl.png" alt="Agentic RL 实验页面截图" width="100%" />
      <br />
      <strong>Agentic RL 实验</strong>
      <br />
      <sub>DeepCoder 风格的 GRPO 训练曲线，把工具调用智能体、回复长度和奖励变化放进可复现实验。</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-atari-game.png" alt="Atari Pong DQN 实验页面截图" width="100%" />
      <br />
      <strong>Atari 游戏实验</strong>
      <br />
      <sub>Atari Pong 游戏画面与 DQN 训练说明放在同一页，展示像素输入如何转化为动作决策。</sub>
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

- [课程简介](#课程简介)
- [简介](#简介)
- [课程大纲](#课程大纲)
- [实验代码](#实验代码)
- [快速开始](#快速开始)
- [参与贡献](#参与贡献)
- [引用](#引用)
- [开源协议](#开源协议)

## 简介

**Hands-On Modern RL** 是一门面向现代强化学习实践的开放课程。与传统的“先讲公式，再给黑盒 API”不同，本课程采用 **“实践优先”** 的路径：从一行行可运行的代码和直观的训练现象出发，让学习者先看到智能体如何在环境中试错并从奖励中改进行为，再回头深入剖析其背后的状态、价值函数、策略梯度、奖励建模与信用分配等核心数学结构。

课程内容跨越经典控制理论，直接连接到当前最前沿的 AI 进展，包括大语言模型（LLM）后训练、偏好对齐（DPO/GRPO）、可验证奖励（RLVR）、多轮工具调用的 Agentic RL 以及视觉语言模型（VLM）强化学习等核心主题。

我们希望为你铺设一条坚实的阶梯——从解出 CartPole 的第一步，一直通往构建大模型后训练与智能体系统的前沿实践。

### 设计原则

课程围绕以下工程和教学原则组织：

1. **实践先于形式化。** 每个主要主题都从实验、指标、失败案例或实现细节开始，然后再引入数学抽象。
2. **理论用于解释行为。** MDP、贝尔曼方程、策略梯度、GAE、PPO 截断、DPO 目标和 GRPO 风格的组优势，都是作为解释代码行为的工具引入的。
3. **现代强化学习，不止于经典强化学习。** 课程涵盖经典控制和深度强化学习，然后进入 RLHF、偏好优化、RLVR、VLM 强化学习和多轮智能体训练。
4. **将调试能力视为一等公民。** 训练崩溃、奖励破解（Reward hacking）、KL 漂移、熵衰减、OOM 故障和评估盲区被视为核心内容，而不是补充说明。
5. **可读的系统优于黑盒。** 代码示例倾向于显式的实现、可检查的指标和清晰的实验边界，以便学习者可以修改和扩展它们。

### 目标受众

本课程专为希望通过构建和检查工作系统来理解强化学习的学习者而设计。

它特别适合：

- 从监督学习转向强化学习的机器学习工程师；
- 准备阅读现代强化学习和对齐论文的研究人员和学生；
- 希望了解 RLHF、DPO、GRPO、RLVR 和后训练系统的大语言模型（LLM）从业者；
- 工具使用智能体、Web 智能体、代码智能体和评估流水线的构建者；
- 喜欢在密集的公式推导前先看代码、实验和直观可视化的自主学习者。

推荐背景：

- Python 编程经验；
- 基础的 PyTorch 熟练度；
- 了解入门机器学习级别的线性代数、概率论和微积分；
- 能够阅读论文并追踪开源训练脚本。

课程附带了数学基础复习附录，因此不要求从第一天起就完全精通数学。

### 学习目标

完成本课程后，学习者应能够：

- 实现并解释核心的强化学习循环：环境交互、轨迹收集、奖励反馈、策略更新和评估；
- 将 MDP、价值函数、贝尔曼方程、TD 学习、策略梯度和优势估计与具体的训练行为联系起来；
- 阅读并修改 DQN、REINFORCE、Actor-Critic、PPO、DPO、GRPO 及相关实现；
- 推理大模型（LLM）的后训练流水线，包括 SFT、奖励建模、PPO 风格的 RLHF、DPO 系列方法和可验证奖励（RLVR）训练；
- 理解多轮交互与信用分配，构建工具调用、轨迹合成与 Agentic RL 智能体系统；
- 将强化学习延伸到 VLM（视觉语言模型）、具身智能与多智能体自我博弈等前沿领域；
- 诊断常见的强化学习失败模式，为新的 RL 问题设计合理的算法、工程评测与调试方案。

### 当前状态

本仓库是一个活跃的课件项目。课程内容正在逐章扩展和完善，重点关注正确性、可运行的示例和稳定的学习路径。

- 课程网站: [cdiddy77.github.io/hands-on-modern-rl](https://cdiddy77.github.io/hands-on-modern-rl/)
- 源码内容: [`docs/`](docs/)
- 可运行示例: [`code/`](code/)
- 本地验证: `npm run verify`
- 开源协议: [CC BY-NC-SA 4.0](LICENSE)

欢迎提交 Issue 和 Pull Request 来修复拼写错误、修正概念、改进可复现性、补充参考文献以及在合理范围内的课程扩展。

## 🔥 最新动态 (News)

> ⚠️ **备注**：本教程由于有 AI 协助生成，目前尚未全面审稿结束，很有可能会有事实性或代码不可运行的错误。欢迎大家在阅读过程中提交 Issue 或 PR 帮助指正。

- **[2026-05-15]** 📖 **全量英文翻译与 PDF 发布**：全部章节英文翻译完成，中英文版 PDF 均通过 CI 自动构建发布。
- **[2026-05-13]** 🚀 **全面升级大模型与传统强化学习实战**：新增可复现的 **Agentic RL**（Deep Research / rLLM）与 **传统 RL**（Actor-Critic 连续控制）训练实例。包含从零构建 Agentic 训练系统的完整代码与微调过程解析，并同步上线 VLM 强化学习（GeoQA 几何推理）动手实验！
- **[2026-05-02]** 🎉 教程初期浏览版正式开源发布，开放测试与建议收集。

## 🗺️ 演进路线图 (Roadmap)

本课程正在持续迭代中，以下是接下来的开发计划：

- [x] **2026-05-02**：开源初始浏览版，用于收集社区测试反馈。
- [x] **2026-05-10**：发布正式小版本，修正初版笔误，稳定第一部分（基础）与第二部分（核心理论）的内容与代码。
- [x] **2026-05 下旬**：完善大模型强化学习可复现实验，补充完整的 RLVR（可验证奖励）实战与评估。
- [ ] **2026-06 上旬**：分步骤交付 Agentic RL 动手项目（从单工具调用到 Deep Research 复杂轨迹合成）。
- [ ] **2026-06 下旬**：增加基于 Unity 的具身强化学习（Embodied RL）可训练模型环境与项目。
- [ ] **2026-07 及以后**：扩展多模态前沿，补充 VLM 强化学习或 Diffusion RL 的完整实战案例。

## 课程大纲

课程分为七个部分以及附录。README 只保留主要模块；完整的小节树、图表、代码引用和详细导航请以在线网站为准。

### 前言

| 模块                                                | 内容                                     |
| :-------------------------------------------------- | :--------------------------------------- |
| [课程导读](docs/preface/intro.md)                   | 课程定位、学习路径及资料使用说明。       |
| [强化学习简史](docs/preface/brief-history/index.md) | 从试错学习到 AlphaGo、RLHF 和 LLM 对齐。 |
| [环境安装指南](docs/preface/env-setup.md)           | 课程所需环境与依赖的安装配置指南。       |

### 第一部分：基础与经典强化学习

| 章节 | 主要主题                                                       | 内容概览                                                                 |
| :--- | :------------------------------------------------------------- | :----------------------------------------------------------------------- |
| 1    | [CartPole 倒立摆](docs/chapter01_cartpole/intro.md)            | 通过第一个可运行控制任务理解状态、动作、奖励、策略、价值、熵和训练曲线。 |
| 2    | [强化学习过程的基本定义](docs/chapter03_mdp/intro.md)          | 多臂老虎机入门、MDP 五元组、策略/价值/回报、折扣轨迹与 POMDP。           |
| 3    | [价值函数与贝尔曼方程](docs/chapter03_mdp/value-bellman.md)    | V/Q 函数、贝尔曼期望/最优方程、压缩映射与数值实验。                      |
| 4    | [动态规划、蒙特卡洛与时序差分](docs/chapter03_mdp/dp-mc-td.md) | DP/MC/TD 价值估计、算法分类、奖励函数设计。                              |

### 第二部分：深度强化学习

| 章节 | 主要主题                                                        | 内容概览                                                                         |
| :--- | :-------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| 5    | [深度 Q 网络](docs/chapter07_dqn/intro.md)                      | 从表格 Q-learning 到 DQN，经验回放、目标网络、CNN 编码器、LunarLander 与 Atari。 |
| 6    | [策略梯度与 REINFORCE](docs/chapter08_policy_gradient/intro.md) | 直接优化策略、采样式梯度、baseline 与方差降低。                                  |
| 7    | [Actor-Critic](docs/chapter09_actor_critic/intro.md)            | Actor-Critic 架构、优势函数、基于 TD 误差的 Critic 训练与 Pendulum 实验。        |
| 8    | [PPO](docs/chapter10_ppo/intro.md)                              | 裁剪目标、信任域直觉、GAE、奖励模型与长时程规划。                                |
| 9    | [连续控制](docs/chapter11_continuous_control/intro.md)          | DDPG、TD3、SAC、Model-Based RL 与世界模型搜索。                                  |

### 第三部分：高级 RL 方法

| 章节 | 主要主题                                                                         | 内容概览                                   |
| :--- | :------------------------------------------------------------------------------- | :----------------------------------------- |
| 10   | [离线强化学习](docs/chapter12_offline_rl/intro.md)                               | 离策略数据、序列建模、CQL/IQL 与离线实验。 |
| 11   | [模仿与元强化学习](docs/chapter13_imitation_meta_rl/intro.md)                    | 行为克隆、DAgger、IRL、GAIL 与元强化学习。 |
| 12   | [探索、多智能体与分层 RL](docs/chapter14_exploration_marl_hierarchical/intro.md) | 好奇心驱动探索、多智能体 RL 与分层方法。   |

### 第四部分：大语言模型对齐与后训练

| 章节 | 主要主题                                                                | 内容概览                                                           |
| :--- | :---------------------------------------------------------------------- | :----------------------------------------------------------------- |
| 13   | [RLHF](docs/chapter15_rlhf/intro.md)                                    | SFT、奖励模型、PPO 风格 RLHF、评估、规模化与奖励破解。             |
| 14   | [大模型 RL 工业实践](docs/chapter16_llm_rl_industrial/intro.md)         | 分布式同步、现代后训练栈与工业流水线。                             |
| 15   | [DPO 家族](docs/chapter17_dpo/intro.md)                                 | DPO 推导、训练指标与 IPO/KTO/ORPO 变体。                           |
| 16   | [GRPO 与 RLVR](docs/chapter18_grpo/grpo-practice-and-mechanism.md)      | 组相对优势、DeepSeek-R1、DAPO、可验证奖励与沙箱训练。              |
| 17   | [推理模型](docs/chapter19_reasoning/intro.md)                           | O1/R1 风格推理涌现、Test-time Scaling、混合与自适应思考。          |
| 18   | [PRM 与推理期搜索](docs/chapter20_prm_search/outcome-vs-process.md)     | 结果 vs. 过程奖励模型、生成式 PRM 与并行推理。                     |
| 19   | [CAI 与 RLAIF](docs/chapter21_cai_rlvr/intro.md)                        | Constitutional AI、HHH 对齐与 RLAIF 工程。                         |

### 第五部分：Agentic 强化学习

| 章节 | 主要主题                                                      | 内容概览                                                |
| :--- | :------------------------------------------------------------ | :------------------------------------------------------ |
| 20   | [Agentic RL](docs/chapter22_agentic/intro.md)                 | 多轮信用分配、工具调用轨迹、多智能体协作与工业实践。    |
| 21   | [RL 驱动 SWE](docs/chapter23_rl_based_swe/intro.md)           | SWE-bench、DeepSWE、Code World Model 与 Self-play SSR。 |
| 22   | [Deep Research 智能体](docs/chapter24_deep_research/intro.md) | 浏览器 RL 框架与 Deep Research 评测。                   |
| 23   | [Computer Use](docs/chapter25_computer_use/intro.md)          | GUI 智能体、训练方案与安全协作。                        |

### 第六部分：多模态强化学习

| 章节 | 主要主题                                                            | 内容概览                                                |
| :--- | :------------------------------------------------------------------ | :------------------------------------------------------ |
| 24   | [VLM 强化学习](docs/chapter26_vlm/intro.md)                         | VLM GRPO、视觉奖励、Qwen3-VL 反思与 EasyR1 GeoQA 实战。 |
| 25   | [音频 RL](docs/chapter27_audio_rl/intro.md)                         | 音频奖励设计与未来方向。                                |
| 26   | [VLA 与具身智能](docs/chapter28_vla/embodied-intelligence/index.md) | 视觉-语言-动作模型与具身 RL。                           |
| 27   | [视觉生成 RL](docs/chapter29_visual_generation/intro.md)            | 图像与视频生成 RL。                                     |

### 第七部分：安全、评估与研究前沿

| 章节 | 主要主题                                                     | 内容概览                                                                |
| :--- | :----------------------------------------------------------- | :---------------------------------------------------------------------- |
| 28   | [对齐失败](docs/chapter30_alignment_failures/intro.md)       | 经典与现代失败模式、尺度定律、潜伏代理与防御。                          |
| 29   | [自博弈、规模化与未来方向](docs/chapter32_selfplay/intro.md) | 自博弈、RL 尺度定律、LLM 多智能体 RL 与进化式 LLM 搜索（AlphaEvolve）。 |

### 附录

| 附录 | 主要主题                                                         | 内容概览                                                             |
| :--- | :--------------------------------------------------------------- | :------------------------------------------------------------------- |
| A    | [训练调试与工程实践](docs/appendix_industrial_training/intro.md) | 训练调试指南、训练系统底座、Agent 沙箱与评测基准。                   |
| B    | [核心算法实现](docs/appendix_code_cheatsheet/intro.md)           | SFT、PPO、DPO、GRPO、采样、注意力与 DAPO 的核心代码速记。            |
| C    | [学习资源与参考资料](docs/appendix_game_projects/intro.md)       | 学习资料、论文阅读路线图、GPU 小时估算、训练指标词典与工业实战练习。 |
| D    | [强化学习的数学基础](docs/appendix_math/intro.md)                | 强化学习所需的线性代数、概率、微积分、优化与信息论。                 |

## 实验代码

[`code/`](code/) 目录包含与各章节对齐的可运行示例。每章的代码都设计得足够精简，以便独立检查、运行和修改。

| 领域           | 代码路径                                                                                                           | 代表性实验                                                     |
| :------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| 经典控制       | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                             | 训练 CartPole，检查奖励和回合长度，比较 PPO 实现。             |
| 偏好微调       | [`code/chapter02_dpo/`](code/chapter02_dpo/)                                                                       | 生成偏好数据，使用 DPO 进行训练，比较微调前后的模型行为。      |
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

本仓库的实用学习路径：

1. 阅读[课程简介](docs/preface/intro.md)并运行 CartPole 示例。
2. 尽早粗略阅读 [DPO 章节](docs/chapter17_dpo/intro.md)（甚至在完整的理论之前），以锚定大模型后训练的动机。
3. 按顺序学习第 01-11 章；这是概念核心（基础、Value-Based 与 Policy-Based 深度 RL、PPO、连续控制）。
4. 在掌握策略梯度和 PPO 机制后，返回第四部分（RLHF、DPO、GRPO、RLVR、推理、PRM、CAI）。
5. 每当训练运行出现异常时，使用调试和工程附录。
6. 将第五至第七部分作为扩展学习：Agentic RL、多模态 RL 与安全/前沿研究。

## 快速开始

### 在线阅读

发布的课程网站地址：

```text
https://cdiddy77.github.io/hands-on-modern-rl/
```

### 本地运行文档网站

环境要求：

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/cdiddy77/hands-on-modern-rl.git
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

## 开源协议

本课程资料在 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE) 下发布。

您可以出于非商业目的共享和修改本材料，前提是必须给出适当的署名，并且衍生作品也必须在相同的协议下分发。

---

<div align="center">
  <sub>由 WalkingLabs 及贡献者维护。</sub>
</div>
