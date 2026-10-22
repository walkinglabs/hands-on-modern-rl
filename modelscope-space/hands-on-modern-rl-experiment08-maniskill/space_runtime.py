from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
BUNDLED_PHYSX = ROOT / "assets" / "physx-105.1-physx-5.3.1.patch0-linux-so.zip"
PERSISTENT_CACHE = Path(
    os.environ.get("HOMRL_PERSISTENT_CACHE", "/mnt/workspace/hands-on-modern-rl")
) / "maniskill"


def _configure_software_vulkan() -> str | None:
    """Select Mesa's CPU Vulkan ICD before SAPIEN is imported.

    ModelScope xGPU exposes CUDA compute, but its container runtime does not
    expose the host NVIDIA graphics/Vulkan capability. SAPIEN still creates a
    render system while building several ManiSkill tasks, including state-only
    tasks. Lavapipe gives that system a real Vulkan device while the PPO policy
    continues to run on CUDA.
    """
    candidates = sorted(
        list(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
        + list(Path("/usr/local/share/vulkan/icd.d").glob("lvp_icd*.json"))
    )
    if not candidates:
        return None
    icd = str(candidates[0])
    # Assign rather than setdefault: some base images publish an unusable host
    # ICD path, which must not take precedence over the bundled Mesa driver.
    os.environ["VK_ICD_FILENAMES"] = icd
    os.environ["VK_DRIVER_FILES"] = icd
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
    os.environ["GALLIUM_DRIVER"] = "llvmpipe"
    return icd


SOFTWARE_VULKAN_ICD = _configure_software_vulkan()
_WARMUP_LOCK = threading.Lock()
_WARMUP_DONE = threading.Event()
_WARMUP_THREAD: threading.Thread | None = None
_WARMUP_STATE: dict[str, Any] = {
    "phase": "pending",
    "detail": "GPU PhysX cache is queued for background preparation",
    "progress": 0.0,
    "error": None,
}
SPACE = {
    "title": {"en": "ManiSkill xGPU Robot Lab", "zh": "ManiSkill xGPU 机器人训练场"},
    "description": {
        "en": "Train ManiSkill robot policies with CUDA PPO and a verified PhysX/Vulkan runtime, then replay the learned policy.",
        "zh": "使用 CUDA PPO 与经过验证的 PhysX/Vulkan 运行时训练 ManiSkill 机器人策略，并回放学习结果。",
    },
    "badge": "EXPERIMENT 08 · MANISKILL",
    "training_guide": {
        "success": {"en": "The final evaluation should improve over the initial checkpoint and the replay should move the object toward the manipulation goal. Training complete alone means only that the pipeline finished.", "zh": "最终评估应高于初始检查点，回放中机器人应把物体推向操作目标；仅显示“训练完成”只说明流程结束。"},
        "preview": {"en": "Preview begins with an authentic task capture. After training it is replaced by this run's camera replay, or a task-space replay when the renderer cannot expose camera frames.", "zh": "Preview 起初显示真实任务画面；训练后会替换为本次相机回放，渲染器无法提供相机帧时则显示任务空间回放。"},
        "time": {"en": "Default xGPU recipes usually take 2–8 minutes. First-run PhysX and Vulkan preparation can add 1–4 minutes.", "zh": "默认 xGPU 配方通常需要 2–8 分钟；首次 PhysX 与 Vulkan 准备可能额外增加 1–4 分钟。"},
    },
    "device": "xGPU · CUDA PPO + PhysX",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment08-maniskill.ipynb",
}


def _task(key: str, env_id: str, title: str, zh: str, description: str, description_zh: str,
          action: str, preview: str, default_budget: int, gamma: float) -> dict[str, Any]:
    rollout_quantum = 16 * 50
    epoch_steps = max(rollout_quantum, round((default_budget / 6) / rollout_quantum) * rollout_quantum)
    return {
        "key": key, "title": {"en": title, "zh": zh}, "environment": env_id,
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": "Robot joint state + object pose", "zh": "机器人关节状态与物体位姿"},
        "action": {"en": action, "zh": action}, "algorithm": "PPO", "policy": "MlpPolicy",
        "device": "cuda", "preview": preview, "budget": (8_000, 5_000_000, epoch_steps * 6, 800),
        "steps_per_epoch": (8_000, 5_000_000, epoch_steps, rollout_quantum), "epochs": (1, 12, 6, 1),
        "learning_rate": (1e-5, 0.001, 0.0003, 1e-5), "gamma": (0.8, 1.0, gamma, 0.005),
        "epsilon": (0.0, 0.2, 0.02, 0.005), "checkpoints": 6,
        "baseline_name": "ManiSkill GPU PPO learning baseline",
        "baseline_time": {"en": "about 8–45 minutes on xGPU after simulator warm-up", "zh": "模拟器预热后，xGPU 上约 8–45 分钟"},
        "baseline_outcome": {"en": "Dense evaluation return rises and the exact epoch replay moves, grasps, stacks, or inserts the object toward success.", "zh": "密集评估回报上升，对应 epoch 的真实回放能将物体推、抓、叠放或插入到目标位置。"},
    }


TASKS = [
    _task("push-cube", "PushCube-v1", "PushCube · Panda", "PushCube · 熊猫机械臂", "Push the cube to the marked goal position.", "将方块推到标记的目标位置。", "7D end-effector delta pose", "assets/maniskill-push.jpg", 500_000, 0.8),
    _task("pick-cube", "PickCube-v1", "PickCube · Panda", "PickCube · 熊猫机械臂", "Grasp a cube and lift it to a target pose.", "抓住方块并将其抬升到目标位姿。", "7D end-effector delta pose + gripper", "assets/maniskill-pick.jpg", 1_000_000, 0.9),
    _task("stack-cube", "StackCube-v1", "StackCube · Panda", "StackCube · 熊猫机械臂", "Pick up one cube and stack it stably on another.", "拾取一个方块，并将它稳定叠放在另一个方块上。", "7D end-effector delta pose + gripper", "assets/maniskill-stack.jpg", 2_000_000, 0.95),
    _task("peg-insertion", "PegInsertionSide-v1", "PegInsertionSide · Panda", "PegInsertionSide · 熊猫机械臂", "Align a peg and insert it into a horizontal socket.", "对齐插销，并把它插入水平插座。", "7D end-effector delta pose + gripper", "assets/maniskill-peg.jpg", 3_000_000, 0.95),
]


def _set_warmup_state(phase: str, detail: str, progress: float = 0.0, error: str | None = None) -> None:
    with _WARMUP_LOCK:
        _WARMUP_STATE.update(phase=phase, detail=detail, progress=float(progress), error=error)


def _warmup_snapshot() -> dict[str, Any]:
    with _WARMUP_LOCK:
        return dict(_WARMUP_STATE)


def _prepare_persistent_sapien_home() -> Path:
    """Map SAPIEN's hard-coded ~/.sapien cache onto ModelScope persistent storage."""
    persistent_home = PERSISTENT_CACHE / "sapien-home"
    persistent_home.mkdir(parents=True, exist_ok=True)
    sapien_home = Path.home() / ".sapien"
    if sapien_home.is_symlink():
        return sapien_home
    if not sapien_home.exists():
        sapien_home.parent.mkdir(parents=True, exist_ok=True)
        sapien_home.symlink_to(persistent_home, target_is_directory=True)
        return sapien_home

    # A base image may have created ~/.sapien already. In that case persist the
    # large PhysX subtree without removing any image-owned files.
    persistent_physx = persistent_home / "physx"
    persistent_physx.mkdir(parents=True, exist_ok=True)
    physx_home = sapien_home / "physx"
    if not physx_home.exists():
        physx_home.symlink_to(persistent_physx, target_is_directory=True)
    return sapien_home


def _download_physx(url: str, archive: Path, target: Path, dll: Path) -> None:
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required to prepare the GPU PhysX cache efficiently")
    process = subprocess.Popen(
        [
            "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
            "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
            "--min-split-size=1M", "--summary-interval=2", "--console-log-level=notice",
            "--enable-color=false",
            "--dir", str(archive.parent), "--out", archive.name, url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    last_update = 0.0
    for line in process.stdout:
        clean = line.strip()
        now = time.monotonic()
        if clean and ("Download complete" in clean or ("[#" in clean and now - last_update >= 4.0)):
            _set_warmup_state("downloading", f"GPU PhysX · {clean}", 0.5)
            last_update = now
    if process.wait() != 0:
        raise RuntimeError(f"GPU PhysX parallel download exited with code {process.returncode}")
    _set_warmup_state("extracting", "Extracting the GPU PhysX runtime into persistent storage", 0.98)
    with ZipFile(archive) as bundle:
        bundle.extractall(target)
    archive.unlink(missing_ok=True)
    if not dll.exists():
        raise RuntimeError(f"PhysX archive extracted without the expected library: {dll.name}")


def _warmup_runtime() -> None:
    try:
        _set_warmup_state("preparing", "Connecting SAPIEN to persistent ModelScope storage", 0.01)
        _prepare_persistent_sapien_home()
        import sapien

        physx_version = sapien.physx.version()
        target = Path.home() / ".sapien" / "physx" / physx_version
        target.mkdir(parents=True, exist_ok=True)
        dll = target / "libPhysXGpu_64.so"
        if not dll.exists():
            if BUNDLED_PHYSX.exists() and BUNDLED_PHYSX.stat().st_size > 1_000_000:
                _set_warmup_state("extracting", "Loading the bundled BSD-licensed GPU PhysX runtime", 0.96)
                with ZipFile(BUNDLED_PHYSX) as bundle:
                    bundle.extractall(target)
                if not dll.exists():
                    raise RuntimeError("Bundled PhysX archive is missing libPhysXGpu_64.so")
            else:
                url = (
                    "https://github.com/sapien-sim/physx-precompiled/releases/download/"
                    f"{physx_version}/linux-so.zip"
                )
                _set_warmup_state("downloading", "Downloading the GPU PhysX runtime", 0.02)
                _download_physx(url, target / "linux-so.zip.part", target, dll)
        _set_warmup_state("loading", "Loading the cached GPU PhysX runtime", 0.99)
        sapien.physx.enable_gpu()
        _set_warmup_state("ready", f"GPU PhysX {physx_version} is cached and ready", 1.0)
    except Exception as exc:
        _set_warmup_state(
            "error",
            f"GPU PhysX preparation failed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _WARMUP_DONE.set()


def start_runtime_warmup() -> None:
    global _WARMUP_THREAD
    with _WARMUP_LOCK:
        if _WARMUP_THREAD is not None:
            return
        _WARMUP_THREAD = threading.Thread(target=_warmup_runtime, name="physx-warmup", daemon=True)
        _WARMUP_THREAD.start()


def _wait_for_runtime() -> Iterator[str]:
    start_runtime_warmup()
    last_detail = ""
    while not _WARMUP_DONE.wait(timeout=1.0):
        detail = str(_warmup_snapshot()["detail"])
        if detail != last_detail:
            yield detail
            last_detail = detail
    state = _warmup_snapshot()
    if state["error"]:
        raise RuntimeError(str(state["detail"]))
    if str(state["detail"]) != last_detail:
        yield str(state["detail"])


def runtime_status() -> str:
    try:
        import mani_skill
        import torch
        warmup = _warmup_snapshot()
        if torch.cuda.is_available():
            if warmup["phase"] == "ready":
                renderer = "Mesa software Vulkan" if SOFTWARE_VULKAN_ICD else "Vulkan device pending"
                return f"ManiSkill {mani_skill.__version__} · {torch.cuda.get_device_name(0)} · {renderer}"
            return f"ManiSkill {mani_skill.__version__} · {torch.cuda.get_device_name(0)} · {warmup['detail']}"
        return f"ManiSkill {mani_skill.__version__} · CPU PhysX"
    except Exception as exc:
        return f"installing ManiSkill runtime · {type(exc).__name__}"


def _make_vec_env(
    task: dict[str, Any],
    num_envs: int,
    render_mode: str | None = None,
    gpu_sim: bool = True,
):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

    # ModelScope xGPU exposes CUDA compute but not the host NVIDIA Vulkan ICD.
    # Lavapipe is selected before SAPIEN import so even state-only tasks can
    # finish scene construction. Rendering stays disabled during training.
    raw = gym.make(
        task["environment"],
        num_envs=num_envs,
        obs_mode="state",
        reward_mode="dense",
        render_mode=render_mode,
        render_backend="none" if render_mode is None else "cpu",
        sim_backend="physx_cuda" if gpu_sim and num_envs > 1 else "physx_cpu",
        # ManiSkillSB3VectorEnv auto-resets only the vector slots whose
        # episodes ended. Scene reconfiguration cannot be combined with that
        # partial reset, so build each slot once and reset state thereafter.
        reconfiguration_freq=0,
    )
    # Gymnasium 1.0 no longer forwards arbitrary attributes through wrappers.
    # ManiSkillSB3VectorEnv needs num_envs/single_*_space from the BaseEnv, so
    # give it the unwrapped vector environment while retaining ``raw`` for
    # rendering and lifecycle ownership.
    return raw, ManiSkillSB3VectorEnv(raw.unwrapped)


def _evaluate(
    model: Any,
    task: dict[str, Any],
    seed: int,
    episodes: int = 16,
    gpu_sim: bool = True,
) -> tuple[float, float]:
    environment_count = episodes if gpu_sim else 1
    raw, env = _make_vec_env(task, environment_count, gpu_sim=gpu_sim)
    try:
        observation = env.reset()
        returns = np.zeros(environment_count, dtype=np.float64)
        for _ in range(200):
            actions, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(actions)
            returns += np.asarray(rewards, dtype=np.float64).reshape(-1)[:environment_count]
        return float(returns.mean()), float(returns.std())
    finally:
        env.close()
        del raw


def _position(value: Any) -> np.ndarray:
    """Convert the first vectorized ManiSkill pose to a three-value NumPy position."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1, 3)[0].copy()


def _task_state(raw: Any, task: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read actual task-space state without invoking the Vulkan renderer."""
    base = raw.unwrapped
    tcp_pose = getattr(base.agent, "tcp_pose", None)
    if tcp_pose is None:
        tcp_pose = base.agent.tcp.pose
    state: dict[str, np.ndarray] = {"tcp": _position(tcp_pose.p)}
    if task["key"] == "push-cube":
        state.update(object=_position(base.obj.pose.p), target=_position(base.goal_region.pose.p))
    elif task["key"] == "pick-cube":
        state.update(object=_position(base.cube.pose.p), target=_position(base.goal_site.pose.p))
    elif task["key"] == "stack-cube":
        state.update(object=_position(base.cubeA.pose.p), support=_position(base.cubeB.pose.p))
    else:
        state.update(object=_position(base.peg.pose.p), target=_position(base.box_hole_pose.p))
    return state


def _task_state_frame(
    task: dict[str, Any],
    step: int,
    states: list[dict[str, np.ndarray]],
    rewards: list[float],
    action_norms: list[float],
    width: int = 720,
    height: int = 420,
) -> np.ndarray:
    """Draw a camera-free replay from the policy's real simulator poses."""
    image = Image.new("RGB", (width, height), "#f5f7ff")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=24, fill="#ffffff", outline="#d9def4", width=2)
    draw.text((42, 36), "MANISKILL · POLICY TRAJECTORY", fill="#5661e9", font=font)
    draw.text((42, 63), f"{task['environment']}  ·  learned PPO policy", fill="#151a38", font=font)
    draw.text((42, 86), f"Simulator step {step + 1:03d} / {len(states):03d}", fill="#59617a", font=font)
    draw.text((width - 270, 86), "Schematic projection of real poses", fill="#8a91a8", font=font)

    draw.rounded_rectangle((42, 116, width - 42, 340), radius=16, fill="#f8f9fd", outline="#e0e4f2")
    draw.rectangle((58, 292, width - 58, 320), fill="#dce1ef")
    draw.line((58, 292, width - 58, 292), fill="#aab3ca", width=3)

    def project(position: np.ndarray) -> tuple[int, int]:
        x, y, z = (float(value) for value in position)
        return int(360 + (x + 0.15) * 350 + y * 120), int(292 - z * 520 + y * 42)

    state = states[step]
    base_point = project(np.asarray([-0.615, 0.0, 0.02], dtype=np.float32))
    tcp_point = project(state["tcp"])
    elbow = ((base_point[0] + tcp_point[0]) // 2 - 12, min(base_point[1], tcp_point[1]) - 75)
    draw.line((base_point, elbow, tcp_point), fill="#505b79", width=13, joint="curve")
    for point in (base_point, elbow):
        draw.ellipse((point[0] - 12, point[1] - 12, point[0] + 12, point[1] + 12), fill="#ffffff", outline="#505b79", width=4)
    draw.line((tcp_point[0] - 15, tcp_point[1] - 7, tcp_point[0] + 15, tcp_point[1] - 7), fill="#151a38", width=5)
    draw.line((tcp_point[0] - 13, tcp_point[1] - 7, tcp_point[0] - 13, tcp_point[1] + 10), fill="#151a38", width=4)
    draw.line((tcp_point[0] + 13, tcp_point[1] - 7, tcp_point[0] + 13, tcp_point[1] + 10), fill="#151a38", width=4)

    object_point = project(state["object"])
    if task["key"] == "stack-cube":
        support_point = project(state["support"])
        draw.rectangle((support_point[0] - 15, support_point[1] - 15, support_point[0] + 15, support_point[1] + 15), fill="#e9a13b", outline="#9b5b0c", width=3)
        draw.rectangle((object_point[0] - 15, object_point[1] - 15, object_point[0] + 15, object_point[1] + 15), fill="#5661e9", outline="#30399b", width=3)
        draw.text((support_point[0] - 11, support_point[1] + 21), "B", fill="#6f440d", font=font)
        draw.text((object_point[0] - 11, object_point[1] - 32), "A", fill="#30399b", font=font)
    elif task["key"] == "peg-insertion":
        target_point = project(state["target"])
        draw.rounded_rectangle((target_point[0] - 22, target_point[1] - 27, target_point[0] + 22, target_point[1] + 27), radius=7, fill="#e7eaf4", outline="#6c7693", width=4)
        draw.ellipse((target_point[0] - 8, target_point[1] - 8, target_point[0] + 8, target_point[1] + 8), fill="#f8f9fd", outline="#343b54", width=3)
        draw.rounded_rectangle((object_point[0] - 28, object_point[1] - 8, object_point[0] + 28, object_point[1] + 8), radius=5, fill="#5661e9", outline="#30399b", width=3)
    else:
        target_point = project(state["target"])
        draw.ellipse((target_point[0] - 24, target_point[1] - 10, target_point[0] + 24, target_point[1] + 10), fill="#fff4f1", outline="#e35b45", width=4)
        draw.ellipse((target_point[0] - 8, target_point[1] - 4, target_point[0] + 8, target_point[1] + 4), fill="#e35b45")
        draw.rectangle((object_point[0] - 15, object_point[1] - 15, object_point[0] + 15, object_point[1] + 15), fill="#5661e9", outline="#30399b", width=3)

    cumulative = float(np.sum(rewards[: step + 1]))
    action_norm = action_norms[step]
    draw.text((48, 362), f"Cumulative dense reward  {cumulative:8.2f}", fill="#252b45", font=font)
    draw.text((width - 245, 362), f"Action norm  {action_norm:6.3f}", fill="#252b45", font=font)
    return np.asarray(image)


def _record_task_state(model: Any, task: dict[str, Any], seed: int, output_dir: Path) -> str:
    raw, env = _make_vec_env(task, 1)
    states: list[dict[str, np.ndarray]] = []
    rewards: list[float] = []
    action_norms: list[float] = []
    try:
        observation = env.reset()
        for _ in range(200):
            actions, _ = model.predict(observation, deterministic=True)
            observation, reward, dones, infos = env.step(actions)
            states.append(_task_state(raw, task))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            action_norms.append(float(np.linalg.norm(np.asarray(actions).reshape(-1))))
    finally:
        env.close()
        del raw
    frames = [
        _task_state_frame(task, index, states, rewards, action_norms)
        for index in range(0, len(states), 4)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    return save_gif(frames, output_dir / "learned-policy.gif", fps=12)


def _record(model: Any, task: dict[str, Any], seed: int, output_dir: Path) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = None
    env = None
    try:
        raw, env = _make_vec_env(task, 1, render_mode="rgb_array")
        frames: list[np.ndarray] = []
        observation = env.reset()
        for index in range(200):
            if index % 2 == 0:
                frame = raw.render()
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                frames.append(np.asarray(frame))
            actions, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(actions)
        preview = save_gif(frames, output_dir / "learned-policy.gif", fps=24)
        return preview, "Rendered the learned policy with the CPU Vulkan backend"
    except Exception as exc:
        if env is not None:
            env.close()
            env = None
        raw = None
        preview = _record_task_state(model, task, seed, output_dir)
        return preview, (
            f"CPU Vulkan camera replay unavailable ({type(exc).__name__}); "
            "generated a camera-free GIF from the learned policy's real TCP, object, and goal poses"
        )
    finally:
        if env is not None:
            env.close()
        del raw


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

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        for detail in _wait_for_runtime():
            yield {"phase": "initializing", "step": 0, "log": detail}
    task = next(item for item in TASKS if item["key"] == key)
    # Keep simulator topology independent from the requested training length.
    # Scaling a long run from 16 to 128 scenes caused SAPIEN to exhaust the
    # graphics/PhysX initialization resources on the ModelScope T4 container,
    # even though the same task is stable with 16 GPU scenes. A larger budget
    # should produce more PPO updates, not silently request a larger simulator.
    parallel_envs = 16 if use_cuda else 1
    yield {"phase": "initializing", "step": 0, "log": f"Creating {parallel_envs} parallel {task['environment']} simulations on {device}"}
    gpu_sim = use_cuda
    try:
        raw, train_env = _make_vec_env(task, parallel_envs, gpu_sim=gpu_sim)
    except RuntimeError as exc:
        message = str(exc).lower()
        if not any(marker in message for marker in ("rendering device", "vulkan", "vk::")):
            raise
        gpu_sim = False
        parallel_envs = 1
        yield {
            "phase": "initializing",
            "step": 0,
            "log": (
                f"GPU PhysX cannot enumerate a Vulkan device ({type(exc).__name__}).\n"
                "Falling back to one official ManiSkill CPU PhysX state environment; the PPO policy remains on CUDA."
            ),
        }
        raw, train_env = _make_vec_env(task, parallel_envs, gpu_sim=False)

    class MetricsCallback(BaseCallback):
        latest: dict[str, Any]
        def __init__(self) -> None:
            super().__init__(verbose=0); self.latest = {}
        def _on_step(self) -> bool:
            self.latest = dict(self.logger.name_to_value); return True

    n_steps = 50
    rollout = parallel_envs * n_steps
    batch_size = next(size for size in (512, 400, 320, 256, 200, 160, 128, 100, 80, 64, 50, 40, 32, 25, 20, 16) if size <= rollout and rollout % size == 0)
    callback = MetricsCallback()
    model = PPO("MlpPolicy", train_env, learning_rate=float(learning_rate), gamma=float(gamma),
                gae_lambda=0.9, ent_coef=max(0.0, float(epsilon)), n_steps=n_steps,
                batch_size=batch_size, n_epochs=6, device=device, seed=int(seed), verbose=0)
    checkpoint_count = max(1, min(12, int(checkpoints or task["checkpoints"])))
    checkpoint_targets = [
        max(1, round(int(budget) * index / checkpoint_count))
        for index in range(1, checkpoint_count + 1)
    ]
    checkpoint_targets[-1] = int(budget)
    run_token = f"{int(time.time())}-{seed}"
    artifact_root = ROOT / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    x: list[float] = []
    y: list[float] = []
    completed = 0
    saved_models: list[tuple[int, int, float, float, Path]] = []
    try:
        for checkpoint_index, target in enumerate(checkpoint_targets, start=1):
            current = max(1, target - completed)
            model.learn(total_timesteps=current, reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed = target
            score, spread = _evaluate(model, task, int(seed) + completed, gpu_sim=gpu_sim)
            x.append(float(completed)); y.append(score)
            metrics = callback.latest
            simulation_backend = "GPU PhysX" if gpu_sim else "CPU PhysX fallback"
            policy_loss = metrics.get("train/policy_gradient_loss")
            value_loss = metrics.get("train/value_loss")
            policy_loss_text = "n/a" if policy_loss is None else f"{float(policy_loss):.6g}"
            value_loss_text = "n/a" if value_loss is None else f"{float(value_loss):.6g}"
            epoch_dir = artifact_root / f"{key}-{run_token}-epoch-{checkpoint_index:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir / "policy"))
            model_file = epoch_dir / "policy.zip"
            saved_models.append((checkpoint_index, completed, score, spread, model_file))
            log = (f"PPO update · step={completed:,}\n"
                   f"parallel_envs={parallel_envs}  rollout={rollout}  policy_device={model.device}  simulation={simulation_backend}\n"
                   f"policy_loss={policy_loss_text}  value_loss={value_loss_text}\n"
                   f"EVAL mean_dense_return={score:.3f} std={spread:.3f}\n"
                   f"SAVE epoch={checkpoint_index}/{checkpoint_count} model={model_file.name}")
            yield {"phase": "training", "step": completed, "score": score, "x": x, "y": y,
                   "detail": f"{completed:,}/{int(budget):,} environment steps",
                   "metric_detail": f"mean dense return ± {spread:.2f}", "log": log}
        # Release the training scene before opening the replay scene. SAPIEN's
        # renderer and PhysX pools are process-global and keeping both alive can
        # make a later task fail with a misleading rendering-device error.
        train_env.close()
        train_env = None
        raw = None
        for checkpoint_index, saved_step, score, spread, model_file in saved_models:
            replay_model = PPO.load(str(model_file), device=device)
            epoch_dir = model_file.parent
            preview, preview_detail = _record(
                replay_model,
                task,
                int(seed) + 10_000 + checkpoint_index,
                epoch_dir,
            )
            (epoch_dir / "metadata.json").write_text(json.dumps({
                "environment": task["environment"],
                "algorithm": "PPO",
                "step": saved_step,
                "epoch": checkpoint_index,
                "epochs": len(saved_models),
                "evaluation_return": score,
                "evaluation_std": spread,
                "parallel_envs": parallel_envs,
                "simulation_backend": "gpu_physx" if gpu_sim else "cpu_physx_fallback",
                "policy_device": str(model.device),
                "seed": int(seed),
            }, indent=2), encoding="utf-8")
            yield {
                "phase": "finalizing",
                "step": saved_step,
                "score": score,
                "x": x,
                "y": y,
                "model": str(model_file),
                "preview": preview,
                "checkpoint_index": checkpoint_index,
                "checkpoint_count": len(saved_models),
                "metric_detail": f"mean dense return ± {spread:.2f}",
                "detail": f"Rendered replay {checkpoint_index}/{len(saved_models)}",
                "log": f"REPLAY epoch={checkpoint_index}/{len(saved_models)} model={model_file.name} · {preview_detail}",
            }
        yield {
            "phase": "complete",
            "step": completed,
            "score": y[-1] if y else None,
            "x": x,
            "y": y,
            "log": f"Saved {len(saved_models)} independently selectable ManiSkill policies and replays",
        }
    finally:
        if train_env is not None:
            train_env.close()
        raw = None
