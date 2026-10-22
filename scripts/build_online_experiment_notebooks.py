#!/usr/bin/env python3
"""Generate the ModelScope companion notebooks for every online experiment.

The Studio runtime remains the single source of truth.  Each notebook clones
this repository, installs the matching Studio dependencies once, imports the
same runtime module, and streams the same training events used by Gradio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

from notebook_execution import with_saved_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "code" / "online-experiments"
REPOSITORY = "walkinglabs/hands-on-modern-rl"
MODELSCOPE_PREFIX = "https://modelscope.cn/notebook/share/github"


EXPERIMENTS = [
    {
        "slug": "hands-on-modern-rl-experiment01-cartpole",
        "title": "Experiment 01 · CartPole PPO",
        "summary": "Train PPO on CartPole-v1 with CPU, inspect evaluation rewards, and render the learned policy.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "cartpole-ppo",
        "budget": 30_000,
        "full_budget": "30,000 environment steps (usually under one minute on a notebook CPU)",
    },
    {
        "slug": "hands-on-modern-rl-experiment-gymnasium",
        "title": "Gymnasium CPU Playground",
        "summary": "Run the same curated Gymnasium recipes as the Studio, from bandits and grids to control tasks.",
        "resource": "CPU",
        "kind": "gymnasium",
        "task": "Bandit · ε-greedy",
        "budget": 2_000,
        "full_budget": "Use each task's default budget from EXPERIMENTS for a full comparison",
    },
    {
        "slug": "hands-on-modern-rl-experiment02-vizdoom",
        "title": "Experiment 02 · ViZDoom CPU Arena",
        "summary": "Train a DQN agent in a real Doom-engine scenario and record the learned first-person policy.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "basic",
        "budget": 2_000,
        "full_budget": "12,000+ environment steps for the Basic scenario",
    },
    {
        "slug": "hands-on-modern-rl-experiment03-atari",
        "title": "Experiment 03 · Atari xGPU Arcade",
        "summary": "Train DQN from ALE pixels, evaluate checkpoints, and replay the learned Atari policy.",
        "resource": "CPU / GPU",
        "kind": "runtime",
        "task": "freeway",
        "budget": 2_000,
        "learning_rate": "1e-4",
        "epsilon": "1.0",
        "full_budget": "300,000 environment steps for the recommended Freeway xGPU baseline",
    },
    {
        "slug": "hands-on-modern-rl-experiment04-board-selfplay",
        "title": "Experiment 04 · Board Games & Self-Play",
        "summary": "Run CFR+ or tabular self-play in OpenSpiel and inspect the learned game policy.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "kuhn-poker",
        "budget": 500,
        "full_budget": "2,000+ CFR iterations for Kuhn Poker",
    },
    {
        "slug": "hands-on-modern-rl-experiment05-multiagent-games",
        "title": "Experiment 05 · Multi-Agent Games",
        "summary": "Train a shared PPO policy in PettingZoo and replay all agents in one synchronized result.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "simple-spread",
        "budget": 2_016,
        "full_budget": "20,000+ environment steps for cooperative navigation",
    },
    {
        "slug": "hands-on-modern-rl-experiment06-minigrid-adventure",
        "title": "Experiment 06 · MiniGrid Adventures",
        "summary": "Train an exploration policy in a partially observed maze and render its learned route.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "empty",
        "budget": 2_000,
        "full_budget": "12,000+ environment steps for Empty-6x6",
    },
    {
        "slug": "hands-on-modern-rl-experiment07-jax-games",
        "title": "Experiment 07 · JAX MinAtar Games",
        "summary": "JIT-compile a policy-gradient update on CPU and train inside a compact MinAtar game.",
        "resource": "CPU",
        "kind": "runtime",
        "task": "breakout",
        "budget": 100,
        "full_budget": "1,000+ episodes after the first JAX compilation",
    },
    {
        "slug": "hands-on-modern-rl-experiment08-maniskill",
        "title": "Experiment 08 · ManiSkill Robot Lab",
        "summary": "Train a robot policy with PPO and PhysX on CPU or CUDA, then replay its manipulation trajectory.",
        "resource": "CPU / GPU",
        "kind": "runtime",
        "task": "push-cube",
        "budget": 10_000,
        "full_budget": "100,000+ transitions with a scheduled CUDA notebook",
    },
    {
        "slug": "hands-on-modern-rl-experiment10-minestudio",
        "title": "Experiment 10 · MineStudio Minecraft Agent",
        "summary": "Start the Minecraft simulator on Linux, train visual PPO on CPU or CUDA, and record a first-person replay.",
        "resource": "CPU / GPU",
        "kind": "runtime",
        "task": "mine-dirt",
        "budget": 1_024,
        "full_budget": "10,000+ simulator steps after the one-time engine download",
    },
    {
        "slug": "hands-on-modern-rl-experiment11-unity-mlagents",
        "title": "Experiment 11 · Unity ML-Agents Arena",
        "summary": "Train Huggy the Dog to fetch a stick in a real Unity Linux scene, or switch to another ML-Agents task.",
        "resource": "xGPU",
        "kind": "runtime",
        "task": "unity-huggy",
        "budget": 100_000,
        "gamma": "0.995",
        "epsilon": "0.20",
        "full_budget": "up to 2,000,000 Unity steps after the one-time 39 MB Huggy scene download",
    },
    {
        "slug": "hands-on-modern-rl-experiment12-ai2thor-embodied",
        "title": "Experiment 12 · AI2-THOR Embodied Home",
        "summary": "Train a visual ObjectNav policy in an interactive AI2-THOR room and replay the learned route.",
        "resource": "xGPU",
        "kind": "runtime",
        "task": "find-mug",
        "budget": 2_000,
        "full_budget": "10,000+ simulator steps with Xvfb and CUDA",
    },
]


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook_url(slug: str) -> str:
    path = f"code/online-experiments/{slug}.ipynb"
    return f"{MODELSCOPE_PREFIX}/{REPOSITORY}/blob/main/{path}"


def intro_cell(spec: dict) -> dict:
    studio = f"https://modelscope.cn/studios/walkinglab/{spec['slug']}"
    source = f"https://github.com/{REPOSITORY}/tree/main/modelscope-space/{spec['slug']}"
    return markdown_cell(
        f"""
        # {spec['title']}

        **WalkingLab × Hands-On Modern RL companion experiment notebook**

        {spec['summary']}

        - Resource profile: **{spec['resource']}**
        - Quick run in this notebook: **{spec['budget']:,}** training units
        - Full experiment: {spec['full_budget']}
        - [Live ModelScope Studio]({studio})
        - [Experiment source]({source})
        - [Hands-On Modern RL](https://github.com/{REPOSITORY}) · [WalkingLab](https://modelscope.cn/organization/walkinglab)

        The notebook imports the exact runtime used by the Studio. Change the parameters below, run the cells in order,
        and compare the checkpoint curve with the final policy GIF or result image. The first setup can take longer because
        native environments and simulator assets are cached; later runs reuse `/mnt/workspace/hands-on-modern-rl-notebooks`.
        """
    )


def learning_cell(spec: dict) -> dict:
    gpu_note = (
        "A scheduled ModelScope **xGPU Notebook** is required. The run cell stops early if CUDA is unavailable."
        if spec["resource"] == "xGPU"
        else "A normal ModelScope **CPU Notebook** is sufficient; no GPU is required."
    )
    if spec["resource"] == "CPU / GPU":
        gpu_note = "Run the quick experiment on CPU; CUDA is selected automatically when available."
    return markdown_cell(
        f"""
        ## 1. Experiment and training schedule

        This experiment asks whether the selected policy improves on the task's evaluation metric as its training budget
        increases. Start with the quick budget to verify the environment and logs. Then increase the budget only after the
        complete result cell produces a curve and an artifact.

        {gpu_note}

        Compare the evaluation curve and replay at each checkpoint. Increase the budget above for a longer training run.
        """
    )


def setup_cell(spec: dict) -> dict:
    return code_cell(
        f'''
        from __future__ import annotations

        import hashlib
        import importlib.util
        import os
        import subprocess
        import sys
        from pathlib import Path

        REPOSITORY_URL = "https://github.com/{REPOSITORY}.git"
        SPACE_SLUG = "{spec['slug']}"
        INSTALL_DEPENDENCIES = os.getenv("HANDS_ON_LAB_PREINSTALLED") != "1"

        def locate_or_clone_repo() -> Path:
            here = Path.cwd().resolve()
            for candidate in (here, *here.parents):
                if (candidate / "modelscope-space" / SPACE_SLUG).is_dir():
                    return candidate
            workspace = Path("/mnt/workspace") if Path("/mnt/workspace").is_dir() else Path.cwd()
            target = workspace / "hands-on-modern-rl-notebooks" / "source"
            target.parent.mkdir(parents=True, exist_ok=True)
            if (target / ".git").is_dir():
                subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
            else:
                subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(target)], check=True)
            return target

        REPO_ROOT = locate_or_clone_repo()
        SPACE_DIR = REPO_ROOT / "modelscope-space" / SPACE_SLUG
        requirements = SPACE_DIR / "requirements.txt"
        packages = SPACE_DIR / "packages.txt"
        cache_root = Path("/mnt/workspace/hands-on-modern-rl-notebooks") if Path("/mnt/workspace").is_dir() else REPO_ROOT / ".cache" / "online-experiments"
        cache_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(requirements.read_bytes() + (packages.read_bytes() if packages.exists() else b"")).hexdigest()[:12]
        marker = cache_root / f"{{SPACE_SLUG}}-{{digest}}.ready"

        if INSTALL_DEPENDENCIES and not marker.exists():
            if packages.exists() and sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0:
                system_packages = [line.strip() for line in packages.read_text().splitlines() if line.strip() and not line.startswith("#")]
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", *system_packages], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
            marker.touch()
        else:
            print(f"Dependency cache ready: {{marker}}")

        if importlib.util.find_spec("ipywidgets") is None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "ipywidgets>=8,<9"],
                check=True,
            )

        os.chdir(SPACE_DIR)
        if str(SPACE_DIR) not in sys.path:
            sys.path.insert(0, str(SPACE_DIR))
        print(f"Repository: {{REPO_ROOT}}")
        print(f"Experiment runtime: {{SPACE_DIR}}")
        '''
    )


def runtime_parameter_cell(spec: dict) -> dict:
    epochs = int(spec.get("epochs", 2))
    steps_per_epoch = max(1, int(spec["budget"]) // epochs)
    return code_cell(
        f'''import importlib

if SPACE_SLUG.endswith("experiment10-minestudio"):
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "minestudio==1.1.6"], check=True)
if SPACE_SLUG.endswith("experiment11-unity-mlagents"):
    from bootstrap_mlagents import ensure_mlagents
    ensure_mlagents()

runtime = importlib.import_module("space_runtime")
tasks = {{item["key"]: item for item in runtime.TASKS}}
print("Available tasks:")
for key, item in tasks.items():
    title = item.get("title", {{}})
    print(f"  {{key:20s}} {{title.get('en', title)}} · {{item.get('environment', 'environment provided by runtime')}}")

TASK_KEY = "{spec['task']}"
STEPS_PER_EPOCH = {steps_per_epoch}
EPOCHS = {epochs}
TRAINING_BUDGET = STEPS_PER_EPOCH * EPOCHS
LEARNING_RATE = {spec.get('learning_rate', '3e-4')}
GAMMA = {spec.get('gamma', '0.99')}
EPSILON = {spec.get('epsilon', '0.10')}
SEED = 42

if TASK_KEY not in tasks:
    raise ValueError(f"Unknown TASK_KEY={{TASK_KEY!r}}. Choose one of {{list(tasks)}}")
selected_task = tasks[TASK_KEY]
print("\\nSelected:", selected_task.get("title", {{}}).get("en", TASK_KEY))
print("Algorithm:", selected_task.get("algorithm"))
print(f"Epoch schedule: {{EPOCHS}} epochs × {{STEPS_PER_EPOCH:,}} steps = {{TRAINING_BUDGET:,}} total steps")
print("Every epoch is evaluated and saved as a separately selectable model.")
'''
    )


def runtime_run_cell(spec: dict) -> dict:
    gpu_check = """import torch
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a scheduled ModelScope xGPU Notebook; CUDA is not visible.")
print("CUDA:", torch.cuda.get_device_name(0))""" if spec["resource"] == "xGPU" else "print('Device: CPU')"
    if spec["resource"] == "CPU / GPU":
        gpu_check = "import torch\nprint('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')"
    return code_cell(
        f'''from IPython.display import display
import json
import matplotlib.pyplot as plt
import time

{gpu_check}

print("Starting the same training generator used by the live Studio...\\n")
events = []
epoch_models = []
MODEL_INDEX = SPACE_DIR / "artifacts" / "notebook-models.json"
MODEL_INDEX.parent.mkdir(parents=True, exist_ok=True)

def persist_epoch_models():
    try:
        saved_index = json.loads(MODEL_INDEX.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        saved_index = {{"models": []}}
    previous = [item for item in saved_index.get("models", []) if isinstance(item, dict)]
    new_ids = {{item["model_id"] for item in epoch_models}}
    merged = epoch_models + [item for item in previous if item.get("model_id") not in new_ids]
    MODEL_INDEX.write_text(json.dumps({{"models": merged}}, indent=2, ensure_ascii=False), encoding="utf-8")

for event in runtime.run(TASK_KEY, TRAINING_BUDGET, LEARNING_RATE, GAMMA, EPSILON, SEED, checkpoints=EPOCHS):
    events.append(dict(event))
    if event.get("model") and event.get("checkpoint_index"):
        model_path = str(event["model"])
        model_file = Path(model_path)
        try:
            relative_model_id = model_file.relative_to(SPACE_DIR / "artifacts").as_posix()
        except ValueError:
            relative_model_id = model_file.name
        epoch_models.append({{
            "model_id": str(event.get("model_id") or relative_model_id),
            "task_key": TASK_KEY,
            "epoch": int(event["checkpoint_index"]),
            "epochs": int(event.get("checkpoint_count") or EPOCHS),
            "step": int(event.get("step") or 0),
            "score": event.get("score"),
            "model": model_path,
            "preview": str(event.get("preview") or ""),
            "created_ns": time.time_ns(),
        }})
        persist_epoch_models()
    message = event.get("log") or event.get("detail")
    if message:
        print(message, flush=True)

if not events:
    raise RuntimeError("The runtime returned no training events")
final_event = events[-1]
print("\\nFinal phase:", final_event.get("phase", "complete"))
print("Final score:", final_event.get("score", "reported in the log"))
print(f"Saved {{len(epoch_models)}} selectable epoch models in this run.")
print("Model index:", MODEL_INDEX)
'''
    )


def runtime_curve_cell() -> dict:
    return code_cell(
        '''
        x = final_event.get("x", [])
        y = final_event.get("y", [])
        if x and y:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(x, y, marker="o", color="#5b5ce2", linewidth=2)
            ax.set_title(f"{TASK_KEY} · checkpoint evaluation")
            ax.set_xlabel("Training progress")
            ax.set_ylabel("Evaluation score")
            ax.grid(alpha=0.25)
            plt.show()
        else:
            print("This task reports its result through the artifact rather than a scalar learning curve.")

        print("Run the inference cell below to choose an epoch model and visualize its exact learned policy.")
        '''
    )


def runtime_inference_cell() -> dict:
    return code_cell(
        '''
        import json
        from pathlib import Path
        from IPython.display import Image as NotebookImage, clear_output, display
        import ipywidgets as widgets

        MODEL_INDEX = SPACE_DIR / "artifacts" / "notebook-models.json"

        def load_notebook_models():
            try:
                payload = json.loads(MODEL_INDEX.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return []
            records = [
                item for item in payload.get("models", [])
                if isinstance(item, dict) and item.get("model_id") and item.get("task_key") == TASK_KEY
            ]
            return sorted(
                records,
                key=lambda item: (
                    int(item.get("created_ns", 0)),
                    int(item.get("step", 0)),
                    int(item.get("epoch", 0)),
                ),
                reverse=True,
            )

        trained_models = load_notebook_models()
        if not trained_models:
            raise RuntimeError("No saved epoch model was found. Run the training cell above first.")

        def model_label(record):
            score = record.get("score")
            score_text = "—" if score is None else f"{float(score):.2f}"
            return (
                f"Epoch {record.get('epoch')}/{record.get('epochs')} · "
                f"{int(record.get('step', 0)):,} steps · score {score_text} · {record['model_id']}"
            )

        model_by_id = {record["model_id"]: record for record in trained_models}
        model_picker = widgets.Dropdown(
            options=[(model_label(record), record["model_id"]) for record in trained_models],
            value=trained_models[0]["model_id"],
            description="Epoch model:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "110px"},
        )
        visualize_button = widgets.Button(
            description="Visualize selected policy",
            button_style="primary",
            icon="play",
            layout=widgets.Layout(width="240px"),
        )
        inference_output = widgets.Output()

        def visualize_selected(_=None):
            record = model_by_id[model_picker.value]
            with inference_output:
                clear_output(wait=True)
                print("Loading saved model:", record["model"])
                preview = record.get("preview")

                if hasattr(runtime, "render_preview"):
                    try:
                        rendered = runtime.render_preview(record["model_id"])
                        preview = rendered.get("preview", preview)
                        print("Generated a fresh deterministic rollout.")
                    except Exception as exc:
                        print(f"Fresh rollout unavailable ({type(exc).__name__}); using the saved epoch rollout.")
                elif hasattr(runtime, "render_saved_model"):
                    try:
                        rendered = runtime.render_saved_model(TASK_KEY, record["model"], record["model_id"])
                        preview = rendered.get("preview", preview)
                        print("Generated a fresh rollout from the selected saved policy.")
                    except Exception as exc:
                        print(f"Fresh rollout unavailable ({type(exc).__name__}); using the saved epoch rollout.")

                preview_path = Path(str(preview)) if preview else None
                if preview_path and preview_path.is_file():
                    display(NotebookImage(filename=str(preview_path)))
                else:
                    print("No replay file is available for this checkpoint.")
                print("Model:", record["model"])
                print("Epoch:", f"{record.get('epoch')}/{record.get('epochs')}")
                print("Training step:", f"{int(record.get('step', 0)):,}")
                print("Evaluation score:", record.get("score", "—"))

        visualize_button.on_click(visualize_selected)
        display(widgets.VBox([model_picker, visualize_button, inference_output]))
        print("The newest epoch is selected by default. Choose another saved model, then click Visualize selected policy.")
        '''
    )


def cartpole_cells(spec: dict) -> list[dict]:
    return [
        code_cell(
            f'''
            from train import train

            TIMESTEPS = {spec['budget']}
            OUTPUT_DIR = cache_root / "results" / SPACE_SLUG
            train(total_timesteps=TIMESTEPS, output_dir=OUTPUT_DIR)
            '''
        ),
        code_cell(
            '''
            from IPython.display import Image as NotebookImage, display

            curve = OUTPUT_DIR / "reward-curve.png"
            replay = OUTPUT_DIR / "cartpole-trained-policy.gif"
            model = OUTPUT_DIR / "ppo-cartpole.zip"
            for path in (curve, replay, model):
                if not path.exists():
                    raise FileNotFoundError(path)
            display(NotebookImage(filename=str(curve)))
            display(NotebookImage(filename=str(replay)))
            print("Saved model:", model)
            '''
        ),
    ]


def gymnasium_cells(spec: dict) -> list[dict]:
    return [
        code_cell(
            f'''
            import app as playground

            EXPERIMENT = "{spec['task']}"
            STEPS_PER_EPOCH = {max(1, int(spec['budget']) // 2)}
            EPOCHS = 2
            LEARNING_RATE = 0.1
            GAMMA = 0.99
            EPSILON = 0.10
            SEED = 42

            print("Curated experiments:")
            for name in playground.EXPERIMENTS:
                print(" ", name)
            if EXPERIMENT not in playground.EXPERIMENTS:
                raise ValueError(f"Choose EXPERIMENT from {{list(playground.EXPERIMENTS)}}")
            print(f"Epoch schedule: {{EPOCHS}} epochs × {{STEPS_PER_EPOCH:,}} steps")
            '''
        ),
        code_cell(
            '''
            import html as html_module
            import re
            from IPython.display import Image as NotebookImage, display

            results = []
            for result in playground.train(
                EXPERIMENT, STEPS_PER_EPOCH, EPOCHS, LEARNING_RATE, GAMMA, EPSILON, SEED, "English"
            ):
                results.append(result)
                console = result[-1]
                value = getattr(console, "value", str(console))
                text = html_module.unescape(re.sub(r"<[^>]+>", "", str(value)))
                print(text[-1800:], flush=True)

            if not results:
                raise RuntimeError("The playground returned no training result")
            status, metric, curve, preview, artifact, console = results[-1]
            display(curve)
            print("Result artifact:", getattr(artifact, "value", artifact))
            print(f"Saved {{len(playground.load_models(EXPERIMENT))}} selectable epoch models for this task.")
            '''
        ),
        code_cell(
            '''
            from pathlib import Path
            from IPython.display import Image as NotebookImage, clear_output, display
            import ipywidgets as widgets

            trained_models = playground.load_models(EXPERIMENT)
            if not trained_models:
                raise RuntimeError("No saved epoch model was found. Run the training cell above first.")

            def model_label(record):
                return (
                    f"Epoch {record['epoch']}/{record['epochs']} · {record['step']:,}/{record['total']:,} steps · "
                    f"score {record['score']:.2f} · {record['model_id']}"
                )

            model_by_id = {record["model_id"]: record for record in trained_models}
            model_picker = widgets.Dropdown(
                options=[(model_label(record), record["model_id"]) for record in trained_models],
                value=trained_models[0]["model_id"],
                description="Epoch model:",
                layout=widgets.Layout(width="95%"),
                style={"description_width": "110px"},
            )
            visualize_button = widgets.Button(
                description="Visualize selected policy", button_style="primary", icon="play",
                layout=widgets.Layout(width="240px"),
            )
            inference_output = widgets.Output()

            def visualize_selected(_=None):
                record = model_by_id[model_picker.value]
                with inference_output:
                    clear_output(wait=True)
                    print("Loading saved model:", record["model"])
                    preview = Path(str(record["preview"]))
                    if preview.is_file():
                        display(NotebookImage(filename=str(preview)))
                    else:
                        print("The saved policy has no replay file.")
                    print("Epoch:", f"{record['epoch']}/{record['epochs']}")
                    print("Training step:", f"{record['step']:,}")
                    print("Evaluation score:", record["score"])

            visualize_button.on_click(visualize_selected)
            display(widgets.VBox([model_picker, visualize_button, inference_output]))
            print("The newest epoch is selected by default. Choose another saved model, then click Visualize selected policy.")
            '''
        ),
    ]


def reflection_cell(spec: dict) -> dict:
    return markdown_cell(
        f"""
        ## 6. Read the result before increasing the budget

        Compare the first and last checkpoint values, then inspect the replay. A rising curve with an implausible replay can
        help locate differences in reward, evaluation and rendering. Continue with **{spec['full_budget']}**, keep the seed fixed,
        and compare at least three seeds before drawing a conclusion.
        """
    )



def saved_result_cell(spec):
    source = """
import json
from IPython.display import Image as NotebookImage, display

records = epoch_models if "epoch_models" in globals() else trained_models
assert records, "Training must produce a saved checkpoint"
latest = max(records, key=lambda record: (record.get("created_ns", 0), record.get("step", 0)))
preview = Path(str(latest.get("preview", "")))
assert preview.is_file(), "Training must produce a replay or result image"
display(NotebookImage(filename=str(preview)))
output_id = os.getenv("HANDS_ON_LAB_ID", SPACE_SLUG)
output = Path(os.getenv("HANDS_ON_LAB_OUTPUT", str(cache_root / "results"))) / output_id
output.mkdir(parents=True, exist_ok=True)
metrics = {"score": latest.get("score"), "steps": latest.get("step"),
           "epochs": EPOCHS, "checkpoints": len(records), "preview": preview.name}
(output / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
"""
    if spec["slug"].endswith("experiment03-atari"):
        source = source.replace(
            'preview = Path(str(latest.get("preview", "")))',
            'rendered = runtime.render_preview(latest["model_id"])\n'
            'latest["preview"] = str(rendered["preview"])\n'
            'persist_epoch_models()\n'
            'preview = Path(latest["preview"])')
    return code_cell(source)

def build_notebook(spec: dict) -> dict:
    cells = [intro_cell(spec), learning_cell(spec), markdown_cell("## 2. Prepare the matching Studio runtime"), setup_cell(spec)]
    # Load the training backend before MiniGrid initializes its renderer.
    if spec["slug"].endswith("experiment06-minigrid-adventure"):
        cells[3]["source"] += "\nimport stable_baselines3\n"
    if spec["kind"] == "runtime":
        cells.extend([
            markdown_cell("## 3. Configure the epoch schedule"),
            runtime_parameter_cell(spec),
            markdown_cell("## 4. Train and save one model per epoch"),
            runtime_run_cell(spec),
            runtime_curve_cell(),
            markdown_cell(
                "## 5. Load a saved epoch model and run inference\n\n"
                "This cell reloads the on-disk model index. It therefore works after training and after a kernel restart, "
                "as long as the ModelScope Notebook workspace is still available. Studio and Notebook run in separate "
                "containers, so this selector lists models produced in the current Notebook workspace."
            ),
            runtime_inference_cell(),
        ])
    else:
        cells.extend([
            markdown_cell("## 3. Configure the epoch schedule"),
            gymnasium_cells(spec)[0],
            markdown_cell("## 4. Train and save one model per epoch"),
            gymnasium_cells(spec)[1],
            markdown_cell(
                "## 5. Load a saved epoch model and run inference\n\n"
                "The selector reloads models produced in this Notebook workspace. The live Studio uses a separate container."
            ),
            gymnasium_cells(spec)[2],
        ])
    cells.append(reflection_cell(spec))
    cells.append(saved_result_cell(spec))
    notebook_id = hashlib.sha1(spec["slug"].encode("utf-8")).hexdigest()[:10]
    for index, cell in enumerate(cells):
        cell["id"] = f"{notebook_id}-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "homrl": {
                "studio": spec["slug"],
                "resource": spec["resource"],
                "modelscope_notebook_url": notebook_url(spec["slug"]),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return with_saved_outputs(notebook, OUTPUT / f"{spec['slug']}.ipynb")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for spec in EXPERIMENTS:
        path = OUTPUT / f"{spec['slug']}.ipynb"
        path.write_text(json.dumps(build_notebook(spec), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT), "->", notebook_url(spec["slug"]))


if __name__ == "__main__":
    main()
