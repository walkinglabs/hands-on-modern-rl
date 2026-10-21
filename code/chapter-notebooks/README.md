# 章节实验源码与本地验收

这 10 个源文件补齐实验室原先没有 Notebook 的项目。每个文件用 `# %%` 分隔代码与讲解，生成的 Notebook 内含完整代码，可以独立运行，无需先克隆仓库。

正文提供[中文原版与完整英文版](../online-experiments/en/README.md)，实验室按页面语言选择对应文件；图表标签使用英文。默认配方均使用 CPU 和本地可生成数据；DPO 使用从零训练的微型语言模型，工具调用使用受限调用语言的微型自回归策略。它们的规模和能力边界在 Notebook 中说明，书中的 Qwen 实验继续保留。

每本依次包含环境安装、基线、训练或数值检查、独立评估、可视化、结果保存、练习和参考资料。练习默认关闭，填空后可自行开启；顺序“运行全部”不会被未填写的练习中断。

| 实验             | 默认预算                                  | 可检查的输出                                     |
| ---------------- | ----------------------------------------- | ------------------------------------------------ |
| Bandit           | 50 种子 × 2,000 步                        | 收益、遗憾、胜率改变后的表现                     |
| GridWorld        | 30 种子 × 500 回合                        | 价值表、策略箭头、最终路径                       |
| DQN / Double DQN | 每种算法 2 种子 × 15,000 步               | 独立评估、Q 估计、策略 GIF                       |
| REINFORCE        | 每种算法 2 种子 × 192 回合                | 固定策略梯度方差、奖励对照、策略 GIF             |
| Pendulum         | 100,000 步、8 个环境                      | 原始回报、归一化价值、模型与统计量重载、摆杆 GIF |
| DPO              | 400 步 SFT + 120 步 DPO                   | 偏好间隔、回答对照、主题保持率                   |
| GRPO 机制        | 6 个验证器用例 + 4 次更新                 | 零方差组、逐 token 裁剪、padding 梯度            |
| 工具调用 GRPO    | 500 步语法预训练 + 180 轮 GRPO            | mock 执行、参数正确率、留出组合失败              |
| 世界模型         | 140 条训练轨迹、40 条测试轨迹、900 次更新 | 一步误差、多步漂移、真实环境规划回报             |
| 离线 RL          | 3 种子，Q 600 步、策略 400 步             | 数据覆盖、分布外估计、策略约束对照               |

## 本地实际运行

[validation.json](./validation.json) 记录运行环境、每本 Notebook 的 SHA-256、运行耗时与实际指标。10 本都在独立的新 Python 内核中按默认预算顺序执行；依赖预先装在虚拟环境中，因此耗时不包含安装。全新内核能检验单元格顺序和隐含状态，但不等于全新操作系统或全新依赖环境。

实验保留了负结果：Pendulum 的短训练还不能稳定控制；DPO 的留出成对偏好率提高，同时主题保持率下降；工具策略的训练请求成功率提高，留出组合的成功率仍很低。图和回放来自这些实际运行，不能视为更大模型或其他平台的成绩。

Colab 与 ModelScope 提供直接打开入口，本次没有在这两个托管平台中启动内核，也没有验证 Qwen GPU 训练。原有 12 个 Studio Notebook 的模拟器与 GPU 适配状态仍分别说明。

## 重新生成与执行

在仓库根目录运行：

```bash
python scripts/build_chapter_notebooks.py
python scripts/build_chapter_notebooks.py --check
python scripts/build_localized_notebooks.py
python scripts/build_localized_notebooks.py --check
python scripts/verify_chapter_notebooks.py
```

安装执行器依赖后，在可用的 Python 虚拟环境中运行全部实验：

```bash
python -m pip install nbformat nbclient ipykernel
python scripts/execute_chapter_notebooks.py --output-dir /tmp/modern-rl-labs
```

Notebook 的第一个单元格会安装缺少的实验依赖。执行器为每本创建新的内核和工作目录，将带输出的 Notebook、图、GIF、checkpoint 和 `metrics.json` 保存在指定目录。首次联网安装的耗时受平台影响。

只执行部分实验可以添加 `--labs bandit gridworld`。修改源文件后必须重新生成 Notebook 并重新执行受影响的实验；只有对应 SHA-256 的新结果才能作为更新后的本地验收证据。

英文版的默认 CPU 配方也已在独立内核中实际运行，记录见 [validation.en.json](./validation.en.json)。可使用执行器的 `--locale en` 参数复现。
