---
title: Atari xGPU 在线训练街机厅
emoji: 🕹️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · Atari xGPU Training Arcade

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的 Atari/ALE 配套实验。使用 ALE 游戏画面训练 DQN，评估 checkpoint，并从保存的策略生成模拟器回放。默认快速实验可在 CPU 上运行；检测到 CUDA 时自动使用 GPU。

Studio 用两个参数定义一次训练：`Environment steps per epoch` 和 `Training epochs`。这里的 epoch 指一段固定步数的环境交互，不是遍历固定数据集；每个 epoch 结束时评估一次并保存一个独立模型。总训练预算由两者相乘得到，例如 `200,000 步/epoch × 5 epochs = 1,000,000 总步数 = 5 个模型`。即使训练中途停止，已经完成的 epoch 模型仍会保留。

Preview 区域的“Epoch 模型”选择框按运行 ID、epoch 序号和累计训练步数列出模型。训练结束时只立即录制本次评估最佳模型的 GIF；选择其他 epoch 时，页面会从对应权重生成真实 ALE 回放并缓存。每个模型只保存网络与优化器状态，不包含体积很大的经验回放缓冲区。

配置区默认载入每个游戏各自的 **Atari DQN xGPU baseline v4**。Pong 和 Breakout 默认为 `200,000 × 5`，Space Invaders 和 Q\*bert 为 `300,000 × 5`，Freeway 为 `50,000 × 6`，2,000,000 步任务为 `400,000 × 5`。该配方使用 4 帧堆叠、训练奖励裁剪、原始奖励评估、经验回放预热和 `ε=1.0→0.01` 探索调度。页面为实时日志而分段调用训练器，但探索调度始终覆盖所有 epochs 的总步数。Freeway 是最快的首次推荐训练。

总步数低于推荐 baseline 时，每个 epoch 只运行 1 个评估回合；达到推荐长度时，每个 epoch 运行 3 个评估回合。每个 epoch 最低支持 1,000 步，便于快速验证环境、日志、模型保存和 GIF 流程；该档位只验证工程链路，不代表策略已经学会游戏。Notebook 和 Python API 继续支持原有的总 `budget` 参数，不影响旧调用。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

WalkingLab：<https://modelscope.cn/organization/walkinglab>

## 配套实验 Notebook

[直接在 ModelScope Notebook 中运行 Atari 实验](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb)。Notebook 与当前创空间复用同一份 ALE/DQN 训练运行时，并显示完整日志、评估曲线和本次策略回放；普通 CPU Notebook 可完成默认的 2,000 步快速实验，也可以选择 GPU 实例增加训练预算。
