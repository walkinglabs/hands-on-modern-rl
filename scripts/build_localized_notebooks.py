"""Build the missing language counterparts; retain all existing Notebook URLs.

Chapter originals are Chinese; Studio originals are English. Only their missing
counterparts are generated. Training code and budgets come from the originals.
"""

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from build_chapter_notebooks import LABS, ROOT
from build_online_experiment_notebooks import EXPERIMENTS

DIRECTORY = ROOT / "code/online-experiments"
COMMENTS = {
    "固定每一步每台机器的潜在奖励，便于同种子公平比较。":
        "Fix potential rewards for every step and arm to compare matching seeds fairly.",
    "移位后，第 len(prompt)-1 个预测位置对应回答的第一个词。":
        "After shifting, prediction position len(prompt)-1 targets the first response word.",
    "提示词长为 4，位置 3 的预测对应回答的第一个词。":
        "The prompt has four tokens; prediction position 3 targets the first response word.",
    "数据固定后，训练器只看到下面的数组，不调用真实奖励函数。":
        "After fixing the data, training uses these arrays without querying the reward function.",
    "均匀动作上的 log-mean-exp；常数不影响对 Q 的梯度。":
        "Uniform-action log-mean-exp; the constant has no effect on Q gradients.",
}
STUDIO_UI = {
    "Repository:": "代码仓库：",
    "Experiment runtime:": "实验运行目录：",
    "Starting the same training generator used by the live Studio...":
        "启动与在线 Studio 相同的训练流程…",
    "This experiment requires a scheduled ModelScope xGPU Notebook; CUDA is not visible.":
        "此实验需要已调度的 ModelScope xGPU Notebook，当前无法检测到 CUDA。",
    "The runtime returned no training events": "训练没有返回任何事件，请检查前面的日志。",
    "The playground returned no training result": "训练没有返回结果，请检查前面的日志。",
    "Unknown TASK_KEY=": "未知任务 TASK_KEY=",
    ". Choose one of ": "，请选择以下任务之一：",
    "Choose EXPERIMENT from ": "请从以下列表选择 EXPERIMENT：",
    "Epoch schedule: ": "轮次安排：",
    " total steps": " 总步数",
    "Epoch ": "轮次 ",
    " epochs × ": " 轮 × ",
    " steps · ": " 步 · ",
    " steps = ": " 步 = ",
    " · score ": " · 得分 ",
    "Saved ": "已保存 ",
    " selectable epoch models in this run.": " 个本次运行可选择的轮次模型。",
    " selectable epoch models for this task.": " 个此任务可选择的轮次模型。",
    "Fresh rollout unavailable (": "新的回放暂不可用（",
    "); using the saved epoch rollout.": "），显示此轮已保存的回放。",
    "reported in the log": "见训练日志",
    "Epoch model:": "训练轮次模型：",
    "Visualize selected policy": "回放所选策略",
    "No saved epoch model was found. Run the training cell above first.":
        "未找到已保存的轮次模型，请先运行上面的训练单元格。",
    "Loading saved model:": "加载已保存模型：",
    "Generated a fresh deterministic rollout.": "已生成新的确定性策略回放。",
    "Generated a fresh rollout from the selected saved policy.": "已从所选模型生成新的策略回放。",
    "No replay file is available for this checkpoint.": "此检查点没有可用的回放文件。",
    "The saved policy has no replay file.": "所选策略没有已保存的回放文件。",
    "The newest epoch is selected by default. Choose another saved model, then click":
        "默认选择最新轮次；也可以选择其他已保存模型，然后点击",
    "Available tasks:": "可选任务：",
    "Selected:": "所选任务：",
    "Algorithm:": "算法：",
    "Every epoch is evaluated and saved as a separately selectable model.":
        "每轮都会评估并保存一个可单独选择的模型。",
    "Curated experiments:": "可选实验：",
    "Final phase:": "最终阶段：",
    "Final score:": "最终得分：",
    "Model index:": "模型索引：",
    "Result artifact:": "结果文件：",
    "Evaluation score:": "评估得分：",
    "Training step:": "训练步数：",
    "Saved model:": "已保存模型：",
    "Model:": "模型：",
    "Epoch:": "轮次：",
    "This task reports its result through the artifact rather than a scalar learning curve.":
        "此任务通过结果文件展示实验结果。",
    "Run the inference cell below to choose an epoch model and visualize its exact learned policy.":
        "运行下面的推理单元格，选择轮次模型并查看该策略的回放。",
}
FULL_BUDGETS = {
    "cartpole": "30,000 个环境步；实际耗时取决于实例性能",
    "gymnasium": "使用各任务在 EXPERIMENTS 中配置的完整默认预算",
    "vizdoom": "Basic 场景建议 12,000 个以上环境步",
    "atari": "Freeway 基线建议 300,000 个环境步，使用 CUDA 实例",
    "selfplay": "Kuhn Poker 建议 2,000 轮以上 CFR 迭代",
    "multiagent": "合作导航建议 20,000 个以上环境步",
    "minigrid": "Empty-6x6 建议 12,000 个以上环境步",
    "jax": "首次 JAX 编译完成后建议训练 1,000 回合以上",
    "maniskill": "建议 100,000 条以上转移，使用可调度的 CUDA 实例",
    "minecraft": "引擎首次下载后建议训练 10,000 个以上模拟器步",
    "unity": "Huggy 场景首次下载后可扩展至 2,000,000 个 Unity 步",
    "ai2thor": "建议 10,000 个以上模拟器步，需要 Xvfb 与 CUDA",
}


def cell_text(cell):
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def localized_code(text, locale, chapter=False):
    if locale == "en":
        for chinese, english in COMMENTS.items():
            text = text.replace("# " + chinese, "# " + english)
        # The English verifier uses the same strict integer grammar and test cases.
        return text.replace("答案", "Answer")
    for english, chinese in sorted(STUDIO_UI.items(), key=lambda item: -len(item[0])):
        text = text.replace(english, chinese)
    text = text.replace("title.get('en', title)", "title.get('zh', title)")
    text = text.replace('.get("en", TASK_KEY)', '.get("zh", TASK_KEY)')
    text = text.replace('SEED, "English"', 'SEED, "中文"')
    return text


def studio_markdown(lab, spec):
    budget = FULL_BUDGETS[lab["id"]]
    gpu = ("需要调度支持 CUDA 的 ModelScope xGPU 实例，CUDA 不可用时训练会停止。"
           if spec["resource"] == "xGPU" else "使用普通 CPU Notebook 实例即可。")
    return [
        f"""# {lab['name']}

WalkingLab × Hands-On Modern RL 配套实验。

{lab['desc']}

- 运行资源：{spec['resource']}
- 默认快速预算：{spec['budget']:,} 个训练单位；具体单位由任务决定。
- 完整预算建议：{budget}。
- [在线 Studio](https://modelscope.cn/studios/walkinglab/{spec['slug']})
- [实验源码](https://github.com/walkinglabs/hands-on-modern-rl/tree/main/{lab['source']})
- [配套章节](https://walkinglabs.github.io/hands-on-modern-rl/{lab['lesson']})

Notebook 导入 Studio 使用的同一套训练实现。按顺序运行，比较检查点曲线和最终策略回放。
首次安装需要下载原生环境或模拟器资源；后续运行会复用工作目录缓存。
""",
        f"""## 1. 实验目标与运行边界

我们观察训练预算增加后，策略在任务评估指标上的变化。先用快速预算检查环境和日志，
确认结果单元格能产生曲线或结果文件，再延长训练。

{gpu}

短运行能确认流程可执行，不能证明收敛。比较算法或报告学习能力时，应使用更长预算。
本次语言适配没有新增托管平台或原生 GPU 环境的运行验收。
""",
        "## 2. 安装配套 Studio 运行环境\n\n安装当前实验所需的依赖，并加载与 Studio 相同的训练代码。",
        "## 3. 设置训练轮次\n\n选择任务、每轮步数和总轮次。每一轮都保留独立的模型与评估结果。",
        "## 4. 训练并保存每轮模型\n\n观察实际训练日志，等待模型索引、评估结果和结果文件写入工作目录。",
        """## 5. 加载已保存模型并回放

选择器重新读取磁盘中的模型索引。只要 Notebook 工作目录仍保留，内核重启后也能重新加载。
Studio 与 Notebook 使用独立容器，因此这里列出当前 Notebook 工作目录产生的模型。
选择一个轮次，再点击“回放所选策略”，检查该模型的实际行为。
""",
        f"""## 6. 先检查结果，再延长预算

比较第一个和最后一个检查点，再观察策略回放。曲线上升而回放行为异常时，
需要检查奖励、评估方式和渲染是否对应同一策略。快速预算下的平坦曲线也不足以下结论。

延长训练可参考：{budget}。比较方法时固定预算，并至少检查三个随机种子的结果。
""",
    ]


def build_counterpart(lab, specs):
    source_path = ROOT / lab["path"]
    original = json.loads(source_path.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(original)
    chapter = lab["id"] in LABS
    locale = "en" if chapter else "zh"
    if chapter:
        path = ROOT / "code/chapter-notebooks/en" / f"{lab['id']}.md"
        translations = path.read_text(encoding="utf-8").split("<!-- cell -->")
    else:
        translations = studio_markdown(lab, specs[source_path.stem])
    markdown = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    assert len(markdown) == len(translations), lab["id"]
    for cell, text in zip(markdown, translations):
        cell["source"] = text.strip() + "\n"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = localized_code(cell_text(cell), locale, chapter)
            ast.parse(cell["source"])
    target = DIRECTORY / locale / source_path.name
    metadata = notebook["metadata"].setdefault("homrl", {})
    metadata.update(locale=locale, source_notebook=lab["path"],
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    metadata["modelscope_notebook_url"] = (
        "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
        + target.relative_to(ROOT).as_posix())
    return target, notebook


def build_all(check=False):
    catalog = json.loads((ROOT / "docs/.vitepress/theme/components/lab-catalog.json").read_text(encoding="utf-8"))
    specs = {spec["slug"]: spec for spec in EXPERIMENTS}
    for lab in catalog["labs"]:
        path, notebook = build_counterpart(lab, specs)
        content = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
        if check:
            assert path.read_text(encoding="utf-8") == content, f"Regenerate {path}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(f"{'Checked' if check else 'Built'} {len(catalog['labs'])} language counterparts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build_all(parser.parse_args().check)
