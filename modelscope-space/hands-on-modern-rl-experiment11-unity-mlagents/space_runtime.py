from __future__ import annotations

import hashlib
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import yaml
import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
UNITY_DATASET_URL = "https://modelscope.cn/datasets/walkinglab/hands-on-modern-rl-unity-environments"
UNITY_DATASET_RESOLVE = f"{UNITY_DATASET_URL}/resolve/master"
UNITY_BUNDLE_URL = f"{UNITY_DATASET_RESOLVE}/linux/ml-agents-1.1.0/Startup.zip"
HUGGY_BUNDLE_URL = f"{UNITY_DATASET_RESOLVE}/linux/huggy/Huggy.zip"
UNITY_BUNDLE_SHA256 = "80e2322215fb7ff5c192e34bd67d63edc80d8cf24e66f8af858010b84a250a5d"
HUGGY_BUNDLE_SHA256 = "6b35692b1d867f74fdf8987a911700e06ff24d40b95b935460ccd175e3712d28"
UNITY_CACHE = Path(os.environ.get("UNITY_MLAGENTS_CACHE", "/mnt/workspace/hands-on-modern-rl/unity-mlagents-1.1.0"))
CAPTURE_FPS = 12
LIVE_PREVIEW_FPS = 8
REPLAY_FPS = 12

SPACE = {
    "title": {"en": "Unity ML-Agents xGPU Arena", "zh": "Unity ML-Agents xGPU 训练场"},
    "description": {
        "en": "Train PPO inside ready-to-run Unity ML-Agents Linux scenes, including Huggy the dog, and replay the rendered run.",
        "zh": "在可直接运行的 Unity ML-Agents Linux 场景中训练 PPO，包括小狗 Huggy，并回放本次训练画面。",
    },
    "badge": "EXPERIMENT 11 · UNITY",
    "training_guide": {
        "success": {"en": "The final mean reward should improve over the initial checkpoint, and the replay should show the intended coordinated behavior. Training complete confirms the pipeline, not that the task is solved.", "zh": "最终平均奖励应高于初始检查点，回放中应出现任务要求的协调行为；“训练完成”表示流程结束，不等于任务已经学会。"},
        "preview": {"en": "Before training it shows an authentic scene capture; during training it streams up to 8 sampled frames/s; after training it becomes this run's 12 fps replay GIF. The trainer runs independently of the sampled browser preview.", "zh": "训练前显示真实场景画面，训练中最多推送每秒 8 个采样帧，训练后变为本次运行的 12 FPS 回放 GIF；训练器运行速度不受网页采样帧率代表。"},
        "time": {"en": "Smoke runs finish in a few minutes. The learning baselines follow scene-scale ML-Agents budgets and take roughly 10 minutes to several hours on xGPU; Walker is intentionally a long curriculum run.", "zh": "短流程验证只需几分钟；学习配方采用与场景规模匹配的 ML-Agents 预算，在 xGPU 上约需 10 分钟到数小时，Walker 是有意保留的长程训练。"},
    },
    "device": "xGPU",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "dataset_url": UNITY_DATASET_URL,
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment11-unity-mlagents.ipynb",
}


def _task(key: str, title: str, title_zh: str, scene: str, behavior: str, description: str,
          description_zh: str, observation: str, action: str, preview: str,
          budget: tuple[int, int, int, int], trainer: dict[str, Any], *,
          bundle_url: str = UNITY_BUNDLE_URL, cache_subdir: str | None = None,
          bundle_sha256: str = UNITY_BUNDLE_SHA256,
          executable: str | None = None, env_args: list[str] | None = None,
          reference_url: str | None = None) -> dict[str, Any]:
    task = {
        "key": key, "title": {"en": title, "zh": title_zh}, "environment": f"Unity/{scene}",
        "scene": scene, "behavior": behavior,
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": observation, "zh": observation},
        "action": {"en": action, "zh": action}, "algorithm": "Unity PPO", "preview": preview,
        "budget": budget, "learning_rate": (1e-5, 0.001, 0.0003, 1e-5),
        "gamma": (0.8, 1.0, float(trainer["reward_signals"]["extrinsic"]["gamma"]), 0.005),
        "epsilon": (0.05, 0.35, float(trainer["hyperparameters"]["epsilon"]), 0.01),
        "checkpoints": 6, "trainer": trainer,
        "bundle_url": bundle_url,
        "bundle_sha256": bundle_sha256,
        "baseline_name": "Unity ML-Agents PPO learning baseline",
        "baseline_time": {"en": "about 10 minutes to several hours on xGPU, depending on the scene", "zh": "xGPU 上约 10 分钟到数小时，具体取决于场景"},
        "baseline_outcome": {"en": "Mean reward rises and the selected native checkpoint replay shows stable task-directed behavior.", "zh": "平均奖励上升，所选原生 checkpoint 的回放呈现稳定且面向任务的行为。"},
    }
    if cache_subdir:
        task["cache_subdir"] = cache_subdir
    if executable:
        task["executable"] = executable
    if env_args is not None:
        task["env_args"] = env_args
    if reference_url:
        task["reference_url"] = reference_url
    return task


TASKS = [
    _task("unity-huggy", "Huggy · Fetch the Stick", "Huggy · 小狗捡树枝", "Huggy", "Huggy",
          "Coordinate four articulated legs, run toward the randomly placed stick, and reach it without spinning out.",
          "协调四条腿的关节运动，跑向随机出现的树枝，并在不过度旋转的情况下抵达目标。",
          "Stick position, relative target direction, body state, and leg orientation",
          "Continuous joint-motor rotations for all four legs", "assets/unity-huggy.webp",
          (20_000, 5_000_000, 1_000_000, 20_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 2048, "buffer_size": 20480, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 512, "num_layers": 3, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.995, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000},
          bundle_url=HUGGY_BUNDLE_URL, bundle_sha256=HUGGY_BUNDLE_SHA256,
          cache_subdir="huggy", executable="Huggy/Huggy.x86_64", env_args=[],
          reference_url="https://huggingface.co/learn/deep-rl-course/unitbonus1/train"),
    _task("unity-basic", "Basic · Discrete PPO", "Basic · 离散 PPO", "Basic", "Basic",
          "Match the target value with a short sequence of discrete decisions.", "通过一小段离散决策使数值匹配目标。",
          "Vector target and current state", "Discrete left / stay / right", "assets/unity-basic.webp", (2_000, 1_000_000, 500_000, 2_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 32, "buffer_size": 256, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": False, "hidden_units": 20, "num_layers": 1, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.9, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 3}),
    _task("unity-3dball", "3D Ball · Continuous PPO", "3D Ball · 连续 PPO", "3DBall", "3DBall",
          "Tilt a platform in two axes and keep the ball from falling.", "控制平台在两个方向倾斜，使小球不掉落。",
          "Ball position/velocity and platform rotation", "Continuous platform tilt", "assets/unity-3dball.webp", (8_000, 1_000_000, 500_000, 4_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 64, "buffer_size": 12000, "learning_rate": 0.0003, "beta": 0.001, "epsilon": 0.2, "lambd": 0.99, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 128, "num_layers": 2, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.99, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000}),
    _task("unity-food", "Food Collector · Visual PPO", "Food Collector · 视觉 PPO", "FoodCollector", "GridFoodCollector",
          "Collect green food while avoiding red food and competing agents.", "收集绿色食物，同时避开红色食物与其他智能体。",
          "Ray sensors and local visual state", "Move, rotate, and fire", "assets/unity-food.webp", (10_000, 5_000_000, 2_000_000, 5_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 1024, "buffer_size": 10240, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": False, "hidden_units": 256, "num_layers": 1, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.99, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 64}),
    _task("unity-walker", "Walker · Locomotion PPO", "Walker · 运动控制 PPO", "Walker", "Walker",
          "Coordinate a many-jointed body to move toward the target direction.", "协调多关节身体，沿目标方向稳定行走。",
          "Joint rotations, velocities, contacts, and target direction", "Continuous joint targets", "assets/unity-walker.webp", (50_000, 30_000_000, 30_000_000, 50_000),
          {"trainer_type": "ppo", "hyperparameters": {"batch_size": 2048, "buffer_size": 20480, "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 3, "learning_rate_schedule": "linear"}, "network_settings": {"normalize": True, "hidden_units": 256, "num_layers": 3, "vis_encode_type": "simple"}, "reward_signals": {"extrinsic": {"gamma": 0.995, "strength": 1.0}}, "keep_checkpoints": 3, "time_horizon": 1000}),
]


def runtime_status() -> str:
    try:
        from importlib.metadata import version

        import torch
        accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback"
        renderer = "LIVE UNITY RENDER" if shutil.which("Xvfb") and shutil.which("ffmpeg") else "RENDERER MISSING"
        return f"ML-Agents {version('mlagents')} · {accelerator} · {renderer}"
    except Exception as exc:
        return f"installing ML-Agents runtime · {type(exc).__name__}"


def _find_unity_executable(cache: Path, relative_path: str | None) -> Path | None:
    if relative_path:
        exact = cache / relative_path
        return exact if exact.is_file() else None
    candidates = [path for path in cache.rglob("Startup.x86_64") if path.is_file()]
    if candidates:
        candidates[0].chmod(0o755)
        return candidates[0]
    return None


def _verify_bundle(archive: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"Unity scene bundle checksum failed: {archive.name}")


def _ensure_unity_bundle(task: dict[str, Any]) -> Path:
    cache = UNITY_CACHE / task["cache_subdir"] if task.get("cache_subdir") else UNITY_CACHE
    relative_path = task.get("executable")
    selected = _find_unity_executable(cache, relative_path)
    if selected:
        selected.chmod(0o755)
        return selected
    cache.mkdir(parents=True, exist_ok=True)
    bundle_url = str(task.get("bundle_url", UNITY_BUNDLE_URL))
    archive_name = bundle_url.rsplit("/", 1)[-1].split("?", 1)[0] or "UnityEnvironment.zip"
    bundled_archive = ROOT / "bundles" / archive_name
    archive, partial = cache / archive_name, cache / f"{archive_name}.part"
    if bundled_archive.is_file():
        archive = bundled_archive
    else:
        aria2 = shutil.which("aria2c")
        curl = shutil.which("curl")
        if aria2:
            command = [
                "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
                "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
                "--min-split-size=1M", "--console-log-level=warn", "--enable-color=false",
                "--dir", str(partial.parent), "--out", partial.name, bundle_url,
            ]
        elif curl:
            command = [
                "curl", "--location", "--fail", "--retry", "5", "--retry-all-errors",
                "--connect-timeout", "20", "--continue-at", "-", "--output", str(partial),
                bundle_url,
            ]
        else:
            raise RuntimeError("The Unity scene bundle requires aria2c or curl")
        subprocess.run(command, check=True, timeout=1800)
        partial.replace(archive)
    try:
        _verify_bundle(archive, str(task["bundle_sha256"]))
    except RuntimeError:
        if archive != bundled_archive:
            archive.unlink(missing_ok=True)
        raise
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(cache)
    selected = _find_unity_executable(cache, relative_path)
    if not selected:
        raise RuntimeError(f"The Unity bundle for {task['scene']} did not contain the expected Linux executable")
    selected.chmod(0o755)
    return selected


def _scaled_config(task: dict[str, Any], budget: int, learning_rate: float, gamma: float,
                   epsilon: float, seed: int, executable: Path, run_id: str,
                   graphics_available: bool, checkpoint_count: int) -> Path:
    import torch

    trainer = yaml.safe_load(yaml.safe_dump(task["trainer"]))
    hyper = trainer["hyperparameters"]
    hyper["learning_rate"], hyper["epsilon"] = learning_rate, epsilon
    trainer["reward_signals"]["extrinsic"]["gamma"] = gamma
    min_buffer = max(256, int(hyper["batch_size"]) * 4)
    hyper["buffer_size"] = max(min_buffer, min(int(hyper["buffer_size"]), max(min_buffer, budget // 4)))
    hyper["batch_size"] = max(16, min(int(hyper["batch_size"]), int(hyper["buffer_size"]) // 4))
    trainer.update(
        max_steps=budget,
        summary_freq=max(200, budget // max(8, checkpoint_count)),
        keep_checkpoints=checkpoint_count,
        checkpoint_interval=max(1, budget // checkpoint_count),
        even_checkpoints=True,
    )
    env_args = task.get("env_args")
    if env_args is None:
        env_args = ["--mlagents-scene-name", f"Assets/ML-Agents/Examples/{task['scene']}/Scenes/{task['scene']}.unity"]
    env_args = list(env_args) + [
        "-force-glcore", "-screen-fullscreen", "0",
        "-screen-width", "960", "-screen-height", "540",
    ]
    config = {
        "behaviors": {task["behavior"]: trainer},
        "env_settings": {"env_path": str(executable), "env_args": env_args, "num_envs": 1, "seed": seed, "timeout_wait": 180},
        "engine_settings": {"width": 960, "height": 540, "quality_level": 2, "time_scale": 20, "target_frame_rate": -1, "capture_frame_rate": 60, "no_graphics": not graphics_available},
        "checkpoint_settings": {"run_id": run_id, "results_dir": str(ARTIFACTS / "unity-results"), "force": True},
        "torch_settings": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if bundled_ffmpeg and Path(bundled_ffmpeg).is_file():
        return bundled_ffmpeg
    raise RuntimeError("ffmpeg is required to record the Unity window")


def _start_xvfb() -> subprocess.Popen[str]:
    display = os.environ.get("UNITY_DISPLAY", ":99")
    os.environ["DISPLAY"] = display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        raise RuntimeError(
            "Xvfb is missing. Restart this Studio so its xGPU startup bootstrap can install the Unity rendering stack."
        )
    process = subprocess.Popen(
        [
            xvfb, display, "-screen", "0", "960x540x24", "-ac", "-nolisten", "tcp",
            "+extension", "GLX", "+render", "-noreset",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    socket_path = Path("/tmp/.X11-unix") / f"X{display.lstrip(':').split('.', 1)[0]}"
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read().strip() if process.stdout else ""
            raise RuntimeError(f"Xvfb exited before Unity started: {output[-500:]}")
        if socket_path.exists():
            return process
        time.sleep(0.15)
    _stop_process(process)
    raise RuntimeError(f"Xvfb did not create display {display} within 12 seconds")


def _start_capture(target: Path, frames_dir: Path) -> subprocess.Popen[str]:
    ffmpeg = _ffmpeg_executable()
    frames_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([
        ffmpeg, "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(CAPTURE_FPS),
        "-video_size", "960x540", "-i", os.environ["DISPLAY"],
        "-filter_complex", f"[0:v]split=2[record][preview];[preview]fps={LIVE_PREVIEW_FPS},scale=420:-1:flags=lanczos[live]",
        "-map", "[record]", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(target),
        "-map", "[live]", "-q:v", "7", str(frames_dir / "frame-%06d.jpg"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, start_new_session=True)


def _latest_preview_frame(frames_dir: Path) -> np.ndarray | None:
    candidates = list(frames_dir.glob("frame-*.jpg"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.name)
    try:
        with Image.open(latest) as image:
            frame = np.asarray(image.convert("RGB"))
        # Live frames are transient. Consume every captured image once so a
        # quiet Unity scene does not resend the same frame to the browser.
        for stale in candidates:
            stale.unlink(missing_ok=True)
        # Xvfb is black before the Unity window is mapped. Keep the task card
        # visible until a real rendered frame is available.
        return frame if float(frame.mean()) > 2.0 else None
    except (OSError, ValueError):
        return None


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def _trim_blank_gif_tail(output: Path, replay_filter: str, fps: int) -> None:
    """Remove the Xvfb desktop left behind after the Unity window closes."""
    with Image.open(output) as replay:
        frame_count = int(getattr(replay, "n_frames", 1))
        tail_start = frame_count
        for frame_index in range(frame_count - 1, -1, -1):
            replay.seek(frame_index)
            frame = np.asarray(replay.convert("RGB"), dtype=np.float32)
            # The empty Xvfb desktop is a nearly uniform dark navy frame.
            # A real Unity scene may be dark, but it still has substantially
            # more spatial variation than this background.
            if float(frame.mean()) < 50.0 and float(frame.std()) < 12.0:
                tail_start = frame_index
            else:
                break
    blank_frames = frame_count - tail_start
    if blank_frames < 3 or tail_start < 12:
        return
    trimmed = output.with_name(f"{output.stem}.trimmed.gif")
    try:
        subprocess.run([
            _ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(output),
            "-t", f"{tail_start / fps:.3f}", "-filter_complex", replay_filter,
            "-loop", "0", str(trimmed),
        ], check=True, timeout=90)
        if not trimmed.exists() or trimmed.stat().st_size < 1_000:
            raise RuntimeError("Unity replay trimming produced an empty GIF")
        trimmed.replace(output)
    finally:
        trimmed.unlink(missing_ok=True)


def _make_gif(video: Path, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    replay_fps = REPLAY_FPS
    replay_filter = (
        f"[0:v]fps={replay_fps},scale=480:-1:flags=lanczos,split[palette_source][replay];"
        "[palette_source]palettegen=max_colors=96:stats_mode=diff[palette];"
        "[replay][palette]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    subprocess.run([
        _ffmpeg_executable(), "-y", "-loglevel", "error", "-sseof", "-6", "-i", str(video),
        "-filter_complex", replay_filter, "-loop", "0", str(output),
    ], check=True, timeout=90)
    _trim_blank_gif_tail(output, replay_filter, replay_fps)
    if not output.exists() or output.stat().st_size < 1_000:
        raise RuntimeError("Unity completed, but the replay capture was empty")
    with Image.open(output) as replay:
        frame_count = int(getattr(replay, "n_frames", 1))
        if frame_count < 12:
            raise RuntimeError(f"Unity replay contains only {frame_count} frame(s)")
        sample_indices = sorted({0, frame_count // 2, frame_count - 1})
        samples: list[np.ndarray] = []
        for frame_index in sample_indices:
            replay.seek(frame_index)
            samples.append(np.asarray(replay.convert("RGB"), dtype=np.float32))
        if max(float(frame.mean()) for frame in samples) <= 2.0:
            raise RuntimeError("Unity replay is black; the rendered window was not captured")
        if max(float(np.mean(np.abs(frame - samples[0]))) for frame in samples[1:]) < 0.35:
            raise RuntimeError("Unity replay is static; the rendered environment did not advance")
    return str(output)


def _checkpoint_step(path: Path) -> int | None:
    numbers = re.findall(r"(?:^|[-_])(\d+)(?=\D*$)", path.stem)
    return int(numbers[-1]) if numbers else None


def _unity_checkpoints(run_dir: Path, expected: int) -> list[tuple[int, Path]]:
    by_step: dict[int, Path] = {}
    for candidate in run_dir.rglob("*.pt"):
        step = _checkpoint_step(candidate)
        if step is not None:
            by_step[step] = candidate
    checkpoints = sorted(by_step.items())
    if len(checkpoints) < expected:
        found = ", ".join(path.name for _, path in checkpoints) or "none"
        raise RuntimeError(
            f"ML-Agents saved {len(checkpoints)}/{expected} resumable checkpoints ({found}); "
            "the run is not exposed as a complete epoch set"
        )
    return checkpoints[-expected:]


def _run_directory(model_path: Path) -> Path:
    for parent in model_path.parents:
        if parent.parent.name == "unity-results":
            return parent
    raise RuntimeError(f"Cannot locate the Unity results run for {model_path}")


def _score_near_step(x: list[float], y: list[float], step: int) -> float | None:
    candidates = [(sample_step, score) for sample_step, score in zip(x, y) if sample_step <= step]
    if candidates:
        return float(candidates[-1][1])
    return float(y[0]) if y else None


def _snapshot_checkpoint(source_run: Path, selected_model: Path, snapshot_id: str) -> tuple[Path, Path]:
    results_root = ARTIFACTS / "unity-inference-results"
    snapshot_run = results_root / snapshot_id
    if snapshot_run.exists():
        shutil.rmtree(snapshot_run)
    shutil.copytree(source_run, snapshot_run)
    selected_step = _checkpoint_step(selected_model)
    if selected_step is None:
        raise RuntimeError(f"The selected Unity checkpoint has no step number: {selected_model.name}")
    selected_relative = selected_model.relative_to(source_run)
    for candidate in snapshot_run.rglob("*.pt"):
        step = _checkpoint_step(candidate)
        if step is not None and step > selected_step:
            candidate.unlink(missing_ok=True)
    snapshot_model = snapshot_run / selected_relative
    if not snapshot_model.is_file():
        raise RuntimeError("The selected Unity checkpoint was not copied into the inference snapshot")
    return results_root, snapshot_model


def _inference_config(task: dict[str, Any], source_run: Path, model_path: Path, model_id: str) -> tuple[Path, Path]:
    source_run_id = source_run.name
    source_config = ARTIFACTS / f"{source_run_id}.yaml"
    if not source_config.is_file():
        raise RuntimeError(f"Training configuration is missing for {source_run_id}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id).strip("-")
    snapshot_id = f"replay-{safe_id}"
    results_root, snapshot_model = _snapshot_checkpoint(source_run, model_path, snapshot_id)
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["checkpoint_settings"] = {
        "run_id": snapshot_id,
        "results_dir": str(results_root),
        "resume": True,
        "load_model": True,
        "train_model": False,
        "inference": True,
        "force": False,
    }
    config["engine_settings"].update(
        time_scale=1,
        target_frame_rate=30,
        capture_frame_rate=30,
        no_graphics=False,
    )
    # The copied result directory retains all optimizer metadata, while newer
    # .pt files were removed. Resume therefore loads this exact epoch rather
    # than silently falling back to the final policy.
    config["behaviors"][task["behavior"]]["init_path"] = str(snapshot_model)
    path = ARTIFACTS / f"{snapshot_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, results_root / snapshot_id


def render_saved_model(key: str, model_path: str, model_id: str) -> dict[str, str]:
    task = next(item for item in TASKS if item["key"] == key)
    source_model = Path(model_path)
    if not source_model.is_file():
        raise RuntimeError(f"Saved Unity checkpoint no longer exists: {source_model}")
    cached = ARTIFACTS / "unity-policy-replays" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', model_id)}.gif"
    if cached.is_file() and cached.stat().st_size > 1_000:
        return {"preview": str(cached)}
    source_run = _run_directory(source_model)
    config, snapshot_run = _inference_config(task, source_run, source_model, model_id)
    video = cached.with_suffix(".mp4")
    frames_dir = cached.parent / f"{cached.stem}-live"
    cached.parent.mkdir(parents=True, exist_ok=True)
    executable = _ensure_unity_bundle(task)
    # The executable path is restored explicitly because inference snapshots
    # can outlive a rescheduled container's generated YAML.
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["env_settings"]["env_path"] = str(executable)
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    xvfb = capture = inference = None
    visible_since: float | None = None
    try:
        xvfb = _start_xvfb()
        capture = _start_capture(video, frames_dir)
        inference = subprocess.Popen(
            ["mlagents-learn", str(config), "--resume", "--inference"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        deadline = time.monotonic() + (120.0 if key == "unity-huggy" else 75.0)
        while time.monotonic() < deadline:
            if inference.poll() is not None:
                output = inference.stdout.read()[-2_000:] if inference.stdout else ""
                raise RuntimeError(f"Unity checkpoint inference exited early ({inference.returncode}): {output}")
            frame = _latest_preview_frame(frames_dir)
            if frame is not None and visible_since is None:
                visible_since = time.monotonic()
            if visible_since is not None and time.monotonic() - visible_since >= 8.0:
                break
            time.sleep(0.15)
        else:
            raise RuntimeError("The selected Unity checkpoint did not render a visible inference episode")
    finally:
        _stop_process(inference)
        _stop_process(capture)
        _stop_process(xvfb)
        if snapshot_run.exists():
            shutil.rmtree(snapshot_run)
    if not video.is_file() or video.stat().st_size <= 1_000:
        raise RuntimeError("Unity checkpoint inference produced no replay video")
    return {"preview": _make_gif(video, cached)}


def run(
    key: str,
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    checkpoints: int | None = None,
) -> Iterator[dict[str, Any]]:
    task = next(item for item in TASKS if item["key"] == key)
    checkpoint_count = max(1, min(12, int(checkpoints or task["checkpoints"])))
    run_id = f"{key}-{int(time.time())}"
    download_note = " · first run downloads a resumable 39 MB scene" if key == "unity-huggy" else ""
    yield {"phase": "initializing", "step": 0, "log": f"Checking the Unity ML-Agents 1.1.0 Linux environment cache{download_note}"}
    executable = _ensure_unity_bundle(task)
    yield {"phase": "initializing", "step": 0, "log": f"Unity executable ready: {executable}\nScene: {task['scene']} · behavior: {task['behavior']}"}
    xvfb_path = shutil.which("Xvfb")
    ffmpeg_path = _ffmpeg_executable()
    if not xvfb_path:
        raise RuntimeError(
            "Real Unity rendering is unavailable because Xvfb is missing. Restart the xGPU Studio and inspect its startup log."
        )
    yield {
        "phase": "initializing", "step": 0,
        "log": (
            f"Replay mode: real Unity window · live preview={LIVE_PREVIEW_FPS} sampled fps · "
            f"final replay={REPLAY_FPS} fps · Xvfb={xvfb_path} · ffmpeg={ffmpeg_path}"
        ),
    }
    config = _scaled_config(
        task,
        int(budget),
        float(learning_rate),
        float(gamma),
        float(epsilon),
        int(seed),
        executable,
        run_id,
        True,
        checkpoint_count,
    )
    video, training_replay = ARTIFACTS / f"{run_id}-training.mp4", ARTIFACTS / f"{run_id}-training.gif"
    frames_dir = ARTIFACTS / f"{run_id}-live"
    xvfb = capture = trainer = None
    x: list[float] = []
    y: list[float] = []
    last_step = 0
    number = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    step_re = re.compile(r"Step:\s*([0-9,]+)", re.IGNORECASE)
    score_re = re.compile(rf"Step:\s*([0-9,]+).*?Mean Reward:\s*({number})", re.IGNORECASE)
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    try:
        xvfb = _start_xvfb()
        capture = _start_capture(video, frames_dir)
        trainer = subprocess.Popen(["mlagents-learn", str(config)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        assert trainer.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_trainer_output() -> None:
            assert trainer is not None and trainer.stdout is not None
            for output_line in trainer.stdout:
                output_queue.put(output_line)
            output_queue.put(None)

        threading.Thread(target=read_trainer_output, daemon=True).start()
        pending: list[str] = []
        last_emit = time.monotonic()
        last_preview_emit = 0.0
        # Huggy is a larger standalone scene than the compact ML-Agents sample
        # launcher. On a freshly scheduled xGPU container it can need close to
        # a minute before the first Unity frame is mapped to Xvfb. Keep waiting
        # while the trainer is alive instead of misclassifying startup as a
        # renderer failure.
        render_timeout = 120.0 if key == "unity-huggy" else 75.0
        render_deadline = time.monotonic() + render_timeout
        rendered_frame_seen = False
        stream_finished = False
        while not stream_finished:
            step_match = match = None
            try:
                line = output_queue.get(timeout=0.125)
            except queue.Empty:
                line = ""
            if line is None:
                stream_finished = True
            elif line:
                clean = ansi_re.sub("", line).rstrip()
                # ML-Agents prints a large Unicode logo one row at a time. It
                # has no diagnostic content and would force many UI redraws.
                if clean and re.search(r"[A-Za-z0-9]{3}", clean):
                    pending.append(clean)
                    step_match = step_re.search(clean)
                    if step_match:
                        last_step = int(step_match.group(1).replace(",", ""))
                    match = score_re.search(clean)
                    if match:
                        last_step, score = int(match.group(1).replace(",", "")), float(match.group(2))
                        x.append(float(last_step))
                        y.append(score)

            now = time.monotonic()
            if capture is not None and capture.poll() is not None:
                raise RuntimeError("ffmpeg stopped before it captured the Unity window")
            live_frame = None
            if now - last_preview_emit >= 1.0 / LIVE_PREVIEW_FPS:
                live_frame = _latest_preview_frame(frames_dir)
                last_preview_emit = now
                rendered_frame_seen = rendered_frame_seen or live_frame is not None
            if not rendered_frame_seen and now >= render_deadline:
                raise RuntimeError(
                    f"The Unity window did not produce a visible frame within {render_timeout:.0f} seconds. "
                    "Training was stopped instead of returning a fake replay."
                )
            should_emit = bool(
                step_match or match or live_frame is not None
                or (pending and now - last_emit >= 0.8)
                or now - last_emit >= 2.0
            )
            if should_emit:
                preview_only = bool(
                    live_frame is not None and not step_match and not match and not pending
                    and now - last_emit < 2.0
                )
                event = {
                    "phase": "training", "step": last_step, "x": x, "y": y,
                    "detail": f"{last_step:,}/{int(budget):,} Unity steps",
                    "log": "\n".join(pending),
                    "preview_only": preview_only,
                }
                if match:
                    event.update(score=score, metric_detail="Unity trainer mean reward")
                if live_frame is not None:
                    event["preview"] = live_frame
                yield event
                pending.clear()
                last_emit = now
        if pending:
            yield {"phase": "training", "step": last_step, "x": x, "y": y, "detail": f"{last_step:,}/{int(budget):,} Unity steps", "log": "\n".join(pending)}
        if trainer.wait() != 0:
            raise RuntimeError(f"Unity ML-Agents trainer exited with code {trainer.returncode}")
    finally:
        _stop_process(trainer)
        _stop_process(capture)
        _stop_process(xvfb)
    if not rendered_frame_seen or not video.exists() or video.stat().st_size <= 1_000:
        raise RuntimeError("Unity training completed without a valid rendered recording")
    training_preview = _make_gif(video, training_replay)
    run_results = ARTIFACTS / "unity-results" / run_id
    saved_checkpoints = _unity_checkpoints(run_results, checkpoint_count)
    for checkpoint_index, (training_step, model_file) in enumerate(saved_checkpoints, start=1):
        model_id = f"{run_id}-epoch-{checkpoint_index:02d}"
        score = _score_near_step(x, y, training_step)
        is_latest = checkpoint_index == len(saved_checkpoints)
        preview = None
        if is_latest:
            yield {
                "phase": "finalizing",
                "step": training_step,
                "score": score,
                "x": x,
                "y": y,
                "detail": "Running the final saved checkpoint in Unity inference mode",
                "log": "Training capture is complete. Launching an exact-policy inference replay for the final epoch.",
            }
            preview = render_saved_model(key, str(model_file), model_id)["preview"]
        yield {
            "phase": "finalizing",
            "step": training_step,
            "score": score,
            "x": x,
            "y": y,
            "model": str(model_file),
            "model_id": model_id,
            "preview": preview,
            "checkpoint_index": checkpoint_index,
            "checkpoint_count": len(saved_checkpoints),
            "metric_detail": "Unity trainer mean reward",
            "detail": f"Registered epoch policy {checkpoint_index}/{len(saved_checkpoints)}",
            "log": (
                f"SAVE epoch={checkpoint_index}/{len(saved_checkpoints)} step={training_step:,} "
                f"model={model_file.name}"
                + (f" replay={Path(preview).name}" if preview else " replay=generated on first selection")
            ),
        }
    yield {
        "phase": "complete", "step": int(budget), "score": y[-1] if y else None,
        "x": x, "y": y, "preview": preview or training_preview,
        "log": (
            f"Unity PPO complete · saved {len(saved_checkpoints)} independently selectable checkpoints. "
            "The latest exact-policy replay is ready; earlier replays are generated once when selected."
        ),
    }
