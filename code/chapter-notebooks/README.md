# 章节实验源码与本地验收

这 14 个源文件补齐实验室原先没有 Notebook 的项目。每个文件用 `# %%` 分隔代码与讲解，生成的 Notebook 内含完整代码，可以独立运行，无需先克隆仓库。DeepResearch 首次运行另需下载一个固定版本的教材源码文件，并校验 SHA-256。

正文提供[中文原版与完整英文版](../online-experiments/en/README.md)，实验室按页面语言选择对应文件；图表标签使用英文。默认配方均使用 CPU 和本地可生成数据；DPO 使用从零训练的微型语言模型，工具调用使用受限调用语言的微型自回归策略。它们的规模和能力边界在 Notebook 中说明，书中的 Qwen 实验继续保留。

每本依次包含环境安装、基线、训练或数值检查、独立评估、可视化、结果保存、练习和参考资料。练习默认关闭，填空后可自行开启；顺序“运行全部”不会被未填写的练习中断。

| 实验              | 默认预算                                  | 可检查的输出                                     |
| ----------------- | ----------------------------------------- | ------------------------------------------------ |
| Bandit            | 50 种子 × 2,000 步                        | 收益、遗憾、胜率改变后的表现                     |
| GridWorld         | 30 种子 × 500 回合                        | 价值表、策略箭头、最终路径                       |
| DQN / Double DQN  | 每种算法 2 种子 × 15,000 步               | 独立评估、Q 估计、策略 GIF                       |
| REINFORCE         | 每种算法 2 种子 × 192 回合                | 固定策略梯度方差、奖励对照、策略 GIF             |
| Pendulum          | 100,000 步、8 个环境                      | 原始回报、归一化价值、模型与统计量重载、摆杆 GIF |
| DPO               | 400 步 SFT + 120 步 DPO                   | 偏好间隔、回答对照、主题保持率                   |
| GRPO 机制         | 6 个验证器用例 + 4 次更新                 | 零方差组、逐 token 裁剪、padding 梯度            |
| 工具调用 GRPO     | 500 步语法预训练 + 180 轮 GRPO            | mock 执行、参数正确率、留出组合失败              |
| 世界模型          | 140 条训练轨迹、40 条测试轨迹、900 次更新 | 一步误差、多步漂移、真实环境规划回报             |
| 离线 RL           | 3 种子，Q 600 步、策略 400 步             | 数据覆盖、分布外估计、策略约束对照               |
| 多轮工具 Agent    | 2 种子 × 160 次更新                       | 实际查询计算、动作梯度掩码与工具轨迹             |
| 代码修复 Agent    | 2 种子 × 160 次更新                       | Python 子进程测试、公开与独立样例、补丁轨迹      |
| SQL / FinQA Agent | 2 种子 × 160 次更新                       | 合成公司数据库、实际 SQLite 查询与计算轨迹       |
| Research Agent    | 3 种子 × 220 轮                           | 固定语料检索、证据选择、同条件前后评估           |

Agentic 的前三本训练有限动作神经网络，REINFORCE 使用同一请求下其他轨迹的平均回报作为基线。环境提供接口顺序约束；代码实验选择固定可信补丁，FinQA 使用固定 SQL 模板与合成公司数据。Research 复用教材线性特征策略基准。它们检验工具反馈、轨迹与策略更新，不代表 rLLM、Qwen 或开放网页 Agent 的性能。

## 运行结果

每本 Notebook 都已保存单元格输出、图表和本次运行信息。实验室的“查看运行结果”连接到独立结果页，可以直接浏览评估指标和策略回放。

[中文记录](./validation.json)与[英文记录](./validation.en.json)保存文件哈希、代码指纹、运行环境、耗时和实际指标。14 项章节实验的中英文版本均在新的 Python 内核中顺序运行。

实验室另有 7 项 CPU Studio 配套实验，运行记录位于 `code/online-experiments/validation.studio.zh.json` 和 `validation.studio.en.json`。结果页源码生成在 `docs/public/lab-results/`。

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

只执行部分实验可以添加 `--labs bandit gridworld`。修改训练代码后，重新执行对应实验并发布输出。只修改讲解文字时，生成器会保留代码相同的已执行输出。

英文版的默认 CPU 配方也已在独立内核中实际运行，记录见 [validation.en.json](./validation.en.json)。可使用执行器的 `--locale en` 参数复现。

## 保存输出与发布结果页

```bash
python scripts/execute_chapter_notebooks.py --output-dir /tmp/modern-rl-zh --locale zh
python scripts/publish_notebook_results.py --run-dir /tmp/modern-rl-zh --locale zh
python scripts/build_localized_notebooks.py
python scripts/execute_chapter_notebooks.py --output-dir /tmp/modern-rl-en --locale en
python scripts/publish_notebook_results.py --run-dir /tmp/modern-rl-en --locale en
npm run verify:notebooks
```

发布脚本将执行计数和输出写回 Notebook，同时生成结果页。生成器按代码指纹保留输出；训练代码改变后，需运行新版本并更新结果。
