from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
from PIL import Image

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
THOR_CACHE = Path("/mnt/workspace/hands-on-modern-rl/ai2thor")
os.environ.setdefault("DISPLAY", ":99")


def _configure_software_opengl() -> str:
    """Select Mesa llvmpipe before AI2-THOR starts its X11 Unity player."""
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
    os.environ["GALLIUM_DRIVER"] = "llvmpipe"
    os.environ["XLIB_SKIP_ARGB_VISUALS"] = "1"
    return "Mesa llvmpipe OpenGL"


SOFTWARE_RENDERER = _configure_software_opengl()

SPACE = {
    "title": {"en": "AI2-THOR xGPU Embodied Home", "zh": "AI2-THOR xGPU 具身家庭环境"},
    "description": {
        "en": "Train a visual ObjectNav policy in interactive kitchens, living rooms, and bedrooms rendered by AI2-THOR.",
        "zh": "在 AI2-THOR 渲染的可交互厨房、客厅和卧室中训练视觉目标导航策略。",
    },
    "badge": "EXPERIMENT 12 · AI2-THOR",
    "training_guide": {
        "success": {"en": "The evaluation log should report success=True or a clearly improved return, and the replay should bring the target object into a close, visible state.", "zh": "评估日志应出现 success=True 或明显更高的回报，回放中目标物体应变得清晰可见并处于较近距离。"},
        "preview": {"en": "Preview starts with an authentic AI2-THOR room capture and ends with this run's embodied navigation replay through the selected scene.", "zh": "Preview 起初显示真实 AI2-THOR 房间画面，训练结束后显示本次运行在所选场景中的具身导航回放。"},
        "time": {"en": "First-run Unity scene download and Xvfb startup usually take 5–15 minutes; warm runs are typically 2–8 minutes.", "zh": "首次下载 Unity 场景并启动 Xvfb 通常需要 5–15 分钟；预热后的运行一般为 2–8 分钟。"},
    },
    "device": "xGPU · CUDA PPO + Xvfb OpenGL",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment12-ai2thor-embodied/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment12-ai2thor-embodied.ipynb",
}


def _task(key: str, title: str, zh: str, scene: str, target: str, room: str, room_zh: str, preview: str) -> dict[str, Any]:
    return {
        "key": key, "title": {"en": title, "zh": zh}, "environment": f"AI2-THOR/{scene}",
        "scene": scene, "target": target,
        "description": {"en": f"Navigate through the {room} until the {target.lower()} is close and visible.", "zh": f"在{room_zh}中移动，直到靠近并看见 {target}。"},
        "observation": {"en": "84×84 egocentric RGB + shaped target distance", "zh": "84×84 第一人称 RGB 与目标距离塑形"},
        "action": {"en": "Move, rotate, and look", "zh": "移动、旋转与抬头/低头"},
        "algorithm": "ObjectNav CNN PPO", "preview": preview,
        "budget": (1_024, 1_000_000, 98_304, 256),
        "steps_per_epoch": (1_024, 1_000_000, 16_384, 256), "epochs": (1, 12, 6, 1),
        "learning_rate": (1e-5, 0.001, 0.00025, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005), "epsilon": (0.0, 0.2, 0.01, 0.005), "checkpoints": 6,
        "baseline_name": "AI2-THOR ObjectNav PPO learning baseline",
        "baseline_time": {"en": "about 20–60 minutes on xGPU after the scene cache is warm", "zh": "场景缓存就绪后，xGPU 上约 20–60 分钟"},
        "baseline_outcome": {"en": "Evaluation return rises, target distance falls, and the exact epoch replay finishes with the target close and visible.", "zh": "评估回报上升、目标距离缩短，对应 epoch 的回放最终能靠近并看见目标。"},
    }


TASKS = [
    _task("find-mug", "Find a Mug · Kitchen", "寻找杯子 · 厨房", "FloorPlan1", "Mug", "kitchen", "厨房", "assets/thor-kitchen.jpg"),
    _task("find-apple", "Find an Apple · Kitchen", "寻找苹果 · 厨房", "FloorPlan2", "Apple", "kitchen", "厨房", "assets/thor-apple.jpg"),
    _task("find-tv", "Find a Television · Living Room", "寻找电视 · 客厅", "FloorPlan201", "Television", "living room", "客厅", "assets/thor-living.jpg"),
    _task("find-bed", "Find a Bed · Bedroom", "寻找床 · 卧室", "FloorPlan301", "Bed", "bedroom", "卧室", "assets/thor-bedroom.jpg"),
]


def runtime_status() -> str:
    try:
        import ai2thor
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "waiting for xGPU"
        return f"AI2-THOR {getattr(ai2thor, '__version__', '5.0.0')} · {device} · {SOFTWARE_RENDERER}"
    except Exception as exc:
        return f"installing AI2-THOR runtime · {type(exc).__name__}"


def _prepare_cache() -> None:
    THOR_CACHE.mkdir(parents=True, exist_ok=True)
    default = Path.home() / ".ai2thor"
    if not default.exists():
        default.symlink_to(THOR_CACHE, target_is_directory=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in AI2-THOR archive: {member.filename}")
        bundle.extractall(destination)


def _prepare_thor_build() -> Iterator[str]:
    """Download the official Linux/X11 build, verify it, and persist it."""
    import ai2thor.build
    from ai2thor.platform import Linux64

    releases = Path.home() / ".ai2thor" / "releases"
    build = ai2thor.build.Build(
        platform=Linux64,
        commit_id=ai2thor.build.COMMIT_ID,
        include_private_scenes=False,
        releases_dir=str(releases),
    )
    executable = Path(build.executable_path)
    if executable.exists():
        yield f"AI2-THOR Linux64/Xvfb cache ready: {executable}"
        return

    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required to prepare the 797 MB AI2-THOR build efficiently")
    downloads = THOR_CACHE / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"{build.name}.zip"
    command = [
        "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
        "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
        "--min-split-size=1M", "--summary-interval=4", "--console-log-level=notice",
        "--enable-color=false",
        "--dir", str(downloads), "--out", archive.name, build.url,
    ]
    yield "Downloading the official AI2-THOR Linux64 build with 16 parallel ranges"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    last_progress = 0.0
    latest_progress = ""
    for line in process.stdout:
        clean = line.strip()
        if clean and ("Download complete" in clean or "NOTICE" in clean):
            yield clean
        elif "[#" in clean:
            latest_progress = clean
            now = time.monotonic()
            if now - last_progress >= 4.0:
                yield latest_progress
                last_progress = now
    if latest_progress and time.monotonic() - last_progress < 4.0:
        yield latest_progress
    if process.wait() != 0:
        raise RuntimeError(f"AI2-THOR parallel download exited with code {process.returncode}")

    expected = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", build.sha256_url],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip().split()[0].lower()
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected:
        raise RuntimeError("AI2-THOR build failed the official SHA-256 verification")

    yield "Official SHA-256 verified · extracting Linux64 into persistent storage"
    releases.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{build.name}-", dir=str(THOR_CACHE)) as temporary:
        extracted = Path(temporary) / build.name
        _safe_extract(archive, extracted)
        base_dir = Path(build.base_dir)
        if base_dir.exists():
            raise RuntimeError(f"Incomplete AI2-THOR cache already exists: {base_dir}")
        os.replace(extracted, base_dir)
    executable.chmod(0o755)
    archive.unlink(missing_ok=True)
    yield f"AI2-THOR Linux64/Xvfb cache ready: {executable}"


def _start_xvfb() -> subprocess.Popen[str]:
    os.environ["SDL_VIDEODRIVER"] = "x11"
    return subprocess.Popen(["Xvfb", os.environ["DISPLAY"], "-screen", "0", "1280x720x24", "-ac", "+extension", "GLX"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT); process.wait(timeout=6)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


class ThorObjectNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "MoveBack")

    def __init__(self, task: dict[str, Any], seed: int, max_steps: int = 256):
        from ai2thor.controller import Controller
        from ai2thor.platform import Linux64

        self.task, self.seed, self.max_steps, self.steps = task, seed, max_steps, 0
        # ModelScope xGPU exposes CUDA compute but no graphics device. The
        # official Linux64 player renders through Xvfb + Mesa llvmpipe. Hide
        # CUDA_VISIBLE_DEVICES only while AI2-THOR builds its Unity command so
        # it cannot append a Vulkan-only force-device-index argument; restore
        # it immediately for the already initialized Torch CUDA policy.
        cuda_visible_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            self.controller = Controller(
                scene=task["scene"], platform=Linux64, x_display=os.environ["DISPLAY"],
                width=320, height=240, fieldOfView=90, quality="Low",
                gridSize=0.25, rotateStepDegrees=90, visibilityDistance=1.5,
            )
        finally:
            if cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.target_id = ""
        self.previous_distance = 0.0
        self.last_event = self.controller.last_event

    def _image(self) -> np.ndarray:
        return np.asarray(Image.fromarray(self.last_event.frame).resize((84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)

    def _target(self) -> dict[str, Any]:
        targets = [obj for obj in self.last_event.metadata["objects"] if obj["objectType"] == self.task["target"]]
        if not targets:
            raise RuntimeError(f"{self.task['target']} is missing from {self.task['scene']}")
        agent = self.last_event.metadata["agent"]["position"]
        return min(targets, key=lambda obj: _distance(agent, obj["position"]))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.steps = 0
        self.last_event = self.controller.reset(scene=self.task["scene"])
        target = self._target()
        self.target_id = target["objectId"]
        self.previous_distance = _distance(self.last_event.metadata["agent"]["position"], target["position"])
        return self._image(), {"target": self.target_id, "distance": self.previous_distance}

    def step(self, action: int):
        action_name = self.ACTIONS[int(action)]
        action_arguments: dict[str, Any] = {"action": action_name}
        if action_name in {"MoveAhead", "MoveBack"}:
            action_arguments["moveMagnitude"] = 0.25
        self.last_event = self.controller.step(**action_arguments)
        self.steps += 1
        objects = {obj["objectId"]: obj for obj in self.last_event.metadata["objects"]}
        target = objects[self.target_id]
        distance = _distance(self.last_event.metadata["agent"]["position"], target["position"])
        success = bool(target.get("visible") and distance <= 1.25)
        reward = (self.previous_distance - distance) * 0.3 - 0.005
        if not self.last_event.metadata.get("lastActionSuccess", True):
            reward -= 0.02
        if success:
            reward += 5.0
        self.previous_distance = distance
        terminated, truncated = success, self.steps >= self.max_steps
        return self._image(), float(reward), terminated, truncated, {"success": success, "distance": distance, "target": self.target_id}

    def render(self) -> np.ndarray:
        return np.asarray(self.last_event.frame, dtype=np.uint8)

    def close(self) -> None:
        self.controller.stop()


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(sum((float(left[axis]) - float(right[axis])) ** 2 for axis in ("x", "y", "z")))


def _make_env(task: dict[str, Any], seed: int):
    return gym.wrappers.RecordEpisodeStatistics(ThorObjectNavEnv(task, seed))


def _record_episode(model: Any, env: Any, output_dir: Path, seed: int) -> tuple[str, float, bool]:
    frames: list[np.ndarray] = []
    total, success = 0.0, False
    observation, _ = env.reset(seed=seed)
    for _ in range(256):
        frames.append(env.render())
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total += float(reward); success = bool(info.get("success"))
        if terminated or truncated:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    path = save_gif(frames, output_dir / "learned-policy.gif", fps=12)
    return path, total, success


def run(
    key: str,
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    checkpoints: int | None = None,
) -> Iterator[dict[str, Any]]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if not torch.cuda.is_available():
        raise RuntimeError("This visual ObjectNav experiment requires a scheduled ModelScope xGPU")
    task = next(item for item in TASKS if item["key"] == key)
    _prepare_cache()
    yield {"phase": "initializing", "step": 0, "log": f"AI2-THOR persistent cache: {THOR_CACHE}"}
    for detail in _prepare_thor_build():
        yield {"phase": "initializing", "step": 0, "log": detail}
    yield {"phase": "initializing", "step": 0, "log": f"Preparing {task['scene']} with Linux64/Xvfb rendering; PPO device={torch.cuda.get_device_name(0)}"}
    xvfb: subprocess.Popen[str] | None = None
    env = None
    try:
        xvfb = _start_xvfb(); time.sleep(1.0)
        yield {"phase": "initializing", "step": 0,
               "detail": f"Launching {task['scene']}",
               "log": "Launching the AI2-THOR Unity scene through Xvfb; controls remain locked until the scene is ready."}
        env = _make_env(task, int(seed))

        class MetricsCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0); self.latest: dict[str, Any] = {}
            def _on_step(self) -> bool:
                self.latest = dict(self.logger.name_to_value); return True

        callback = MetricsCallback()
        model = PPO("CnnPolicy", env, learning_rate=float(learning_rate), gamma=float(gamma), ent_coef=float(epsilon),
                    n_steps=256, batch_size=64, n_epochs=4, device="cuda", seed=int(seed), verbose=0)
        x: list[float] = []
        y: list[float] = []
        completed = 0
        checkpoint_count = max(1, min(12, int(checkpoints or task["checkpoints"])))
        checkpoint_targets = [
            max(1, round(int(budget) * index / checkpoint_count))
            for index in range(1, checkpoint_count + 1)
        ]
        checkpoint_targets[-1] = int(budget)
        run_token = f"{int(time.time())}-{seed}"
        artifact_root = ROOT / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        saved_models: list[tuple[int, int, float, Path]] = []
        for checkpoint_index, target in enumerate(checkpoint_targets, start=1):
            model.learn(total_timesteps=max(1, target - completed), reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed = target
            score = float(callback.latest.get("rollout/ep_rew_mean") or 0.0)
            x.append(float(completed)); y.append(score)
            epoch_dir = artifact_root / f"{key}-{run_token}-epoch-{checkpoint_index:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir / "policy"))
            model_file = epoch_dir / "policy.zip"
            saved_models.append((checkpoint_index, completed, score, model_file))
            log = (f"ObjectNav PPO update · step={completed:,}\n"
                   f"scene={task['scene']}  target={task['target']}  device={model.device}\n"
                   f"episode_reward_mean={score:.4f}  entropy_loss={float(callback.latest.get('train/entropy_loss') or 0):.5f}\n"
                   f"policy_gradient_loss={float(callback.latest.get('train/policy_gradient_loss') or 0):.5f}\n"
                   f"SAVE epoch={checkpoint_index}/{checkpoint_count} model={model_file.name}")
            yield {"phase": "training", "step": completed, "score": score, "x": x, "y": y,
                   "detail": f"{completed:,}/{int(budget):,} simulator steps", "metric_detail": "episode reward mean", "log": log}
        env.close(); env = None
        yield {"phase": "finalizing", "step": completed, "x": x, "y": y,
               "detail": "Starting one replay scene for all saved epochs",
               "log": "Training is complete. Reusing one fresh AI2-THOR controller to replay every saved policy."}
        replay_env = _make_env(task, int(seed) + 10_000)
        try:
            for checkpoint_index, saved_step, score, model_file in saved_models:
                replay_model = PPO.load(str(model_file), device="cuda")
                epoch_dir = model_file.parent
                preview, evaluation, success = _record_episode(
                    replay_model,
                    replay_env,
                    epoch_dir,
                    int(seed) + 10_000 + checkpoint_index,
                )
                (epoch_dir / "metadata.json").write_text(json.dumps({
                    "scene": task["scene"],
                    "target": task["target"],
                    "algorithm": "CNN PPO",
                    "step": saved_step,
                    "epoch": checkpoint_index,
                    "epochs": len(saved_models),
                    "training_score": score,
                    "evaluation_return": evaluation,
                    "success": success,
                    "seed": int(seed),
                }, indent=2), encoding="utf-8")
                yield {
                    "phase": "finalizing",
                    "step": saved_step,
                    "score": evaluation,
                    "x": x,
                    "y": y,
                    "model": str(model_file),
                    "preview": preview,
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_count": len(saved_models),
                    "metric_detail": "deterministic ObjectNav return",
                    "detail": f"Rendered replay {checkpoint_index}/{len(saved_models)}",
                    "log": f"REPLAY epoch={checkpoint_index}/{len(saved_models)} model={model_file.name} return={evaluation:.3f} success={success}",
                }
        finally:
            replay_env.close()
        yield {"phase": "complete", "step": completed, "score": y[-1] if y else None, "x": x, "y": y,
               "log": f"Saved {len(saved_models)} independently selectable AI2-THOR policies and replays"}
    finally:
        if env is not None:
            env.close()
        _stop_process(xvfb)
