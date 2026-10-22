from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import gymnasium as gym

from sb3_tools import save_gif


ROOT = Path(__file__).resolve().parent
BUNDLED_JAVA = ROOT / "assets" / "OpenJDK8U-jre_x64_linux_hotspot_8u502b07.tar.gz"
os.environ.setdefault("MINESTUDIO_DIR", "/mnt/workspace/hands-on-modern-rl/minestudio")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DISPLAY", ":99")
JAVA_CACHE = Path("/mnt/workspace/hands-on-modern-rl/temurin-jre8")
JAVA_URL = "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jre/hotspot/normal/eclipse"
ENGINE_SIZE = 458_106_630
ENGINE_SHA256 = "293fac6ac72245b3365dce0e8bfbb6396fb94df29b23b6538f3bd7e2eec13ec6"

SPACE = {
    "title": {"en": "MineStudio xGPU Minecraft Agent Lab", "zh": "MineStudio xGPU Minecraft 智能体训练场"},
    "description": {
        "en": "Start a real Minecraft simulator, train a compact visual PPO policy, inspect every update, and replay the learned run.",
        "zh": "启动真实 Minecraft 模拟器，训练紧凑的视觉 PPO 策略，查看每次更新，并回放学习后的运行过程。",
    },
    "badge": "EXPERIMENT 10 · MINESTUDIO",
    "training_guide": {
        "success": {"en": "The task reward event should occur and the final replay should perform the requested Minecraft interaction. Training complete alone only confirms that the engine and trainer exited normally.", "zh": "任务奖励事件应实际触发，最终回放应完成指定的 Minecraft 交互；仅显示“训练完成”只表示引擎和训练器正常退出。"},
        "preview": {"en": "Preview starts with a real MineStudio capture and ends with this run's first-person Minecraft replay. Inspect movement and object interaction, not just the reward curve.", "zh": "Preview 起初显示真实 MineStudio 画面，结束后显示本次运行的第一人称 Minecraft 回放；需要同时观察移动和物体交互。"},
        "time": {"en": "A first run usually takes 8–20 minutes for engine/JRE preparation; warm runs are typically 3–10 minutes.", "zh": "首次运行需要准备引擎和 JRE，通常为 8–20 分钟；预热后的运行一般需要 3–10 分钟。"},
    },
    "device": "xGPU · visual PPO",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment10-minestudio.ipynb",
}


def _task(key: str, title: str, zh: str, description: str, description_zh: str,
          commands: list[str], reward_event: str, objects: list[str], preview: str) -> dict[str, Any]:
    return {
        "key": key, "title": {"en": title, "zh": zh}, "environment": f"MineStudio/{key}",
        "description": {"en": description, "zh": description_zh},
        "observation": {"en": "84×84 RGB first-person frames", "zh": "84×84 RGB 第一人称画面"},
        "action": {"en": "10 reduced keyboard/mouse actions", "zh": "10 个简化键鼠动作"},
        "algorithm": "CNN PPO", "preview": preview, "commands": commands,
        "reward_event": reward_event, "reward_objects": objects,
        "budget": (1_024, 1_000_000, 98_304, 256),
        "steps_per_epoch": (1_024, 1_000_000, 16_384, 256), "epochs": (1, 12, 6, 1),
        "learning_rate": (1e-5, 0.001, 0.00025, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005), "epsilon": (0.0, 0.2, 0.01, 0.005), "checkpoints": 6,
        "baseline_name": "MineStudio CNN PPO learning baseline",
        "baseline_time": {"en": "about 20–90 minutes on xGPU after the Java world is ready", "zh": "Java 世界启动后，xGPU 上约 20–90 分钟"},
        "baseline_outcome": {"en": "The task event reward appears more often and the exact epoch replay mines, collects, tracks, or fights with purposeful actions.", "zh": "任务事件奖励出现得更频繁，对应 epoch 的回放能有目的地挖掘、收集、追踪或战斗。"},
    }


TASKS = [
    _task("mine-dirt", "Mine Dirt · Visual PPO", "挖掘泥土 · 视觉 PPO", "Find the nearby dirt blocks, aim, and mine them.", "找到附近的泥土方块，瞄准并挖掘。", ["/give @s minecraft:stone_shovel", "/execute as @p at @s run fill ~1 ~ ~1 ~4 ~2 ~4 minecraft:dirt"], "mine_block", ["dirt"], "assets/minecraft-mine.jpg"),
    _task("collect-wood", "Collect Wood · Visual PPO", "收集木材 · 视觉 PPO", "Approach the nearby oak logs and collect wood with an axe.", "走近附近的橡木原木，并使用斧头收集木材。", ["/give @s minecraft:wooden_axe", "/execute as @p at @s run fill ~1 ~ ~1 ~5 ~10 ~5 minecraft:oak_log"], "mine_block", ["oak_log", "log"], "assets/minecraft-wood.jpg"),
    _task("hunt-sheep", "Hunt a Sheep · Visual PPO", "猎取绵羊 · 视觉 PPO", "Track a nearby sheep and attack with a wooden sword.", "追踪附近的绵羊，并使用木剑攻击。", ["/replaceitem entity @s weapon.mainhand minecraft:wooden_sword", "/summon minecraft:sheep ~2 ~ ~", "/give @p minecraft:bread 10", "/give @p minecraft:wooden_sword 1"], "kill_entity", ["sheep"], "assets/minecraft-sheep.jpg"),
    _task("combat-zombie", "Combat a Zombie · Visual PPO", "对抗僵尸 · 视觉 PPO", "Face one nearby zombie at night and learn a short combat policy.", "在夜间面对附近的一只僵尸，学习一段短程战斗策略。", ["/replaceitem entity @s armor.head minecraft:diamond_helmet", "/replaceitem entity @s armor.chest minecraft:diamond_chestplate", "/replaceitem entity @s armor.legs minecraft:diamond_leggings", "/replaceitem entity @s armor.feet minecraft:diamond_boots", "/replaceitem entity @s weapon.mainhand minecraft:diamond_sword", "/summon minecraft:zombie ~3 ~ ~", "/time set night"], "kill_entity", ["zombie"], "assets/minecraft-zombie.jpg"),
]


def runtime_status() -> str:
    try:
        import minestudio
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return f"MineStudio {getattr(minestudio, '__version__', '1.1.6')} · {device} · ENGINE CACHE"
    except Exception as exc:
        return f"installing MineStudio runtime · {type(exc).__name__}"


def _start_xvfb() -> subprocess.Popen[str]:
    return subprocess.Popen(["Xvfb", os.environ["DISPLAY"], "-screen", "0", "1280x720x24", "-ac", "+extension", "GLX"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _ensure_java8() -> Path:
    java_candidates = list(JAVA_CACHE.glob("*/bin/java"))
    if not java_candidates:
        JAVA_CACHE.mkdir(parents=True, exist_ok=True)
        archive, partial = JAVA_CACHE / "temurin8.tar.gz", JAVA_CACHE / "temurin8.tar.gz.part"
        if BUNDLED_JAVA.exists() and BUNDLED_JAVA.stat().st_size > 10_000_000:
            source = BUNDLED_JAVA
        else:
            aria2 = shutil.which("aria2c")
            curl = shutil.which("curl")
            if aria2:
                command = [
                    "aria2c", "--allow-overwrite=true", "--auto-file-renaming=false", "--continue=true",
                    "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
                    "--min-split-size=1M", "--console-log-level=warn", "--enable-color=false",
                    "--dir", str(partial.parent), "--out", partial.name, JAVA_URL,
                ]
            elif curl:
                command = [
                    "curl", "--location", "--fail", "--retry", "5", "--retry-all-errors",
                    "--connect-timeout", "20", "--continue-at", "-", "--output", str(partial), JAVA_URL,
                ]
            else:
                raise RuntimeError("Temurin JRE 8 requires aria2c or curl")
            subprocess.run(command, check=True, timeout=1800)
            partial.replace(archive)
            source = archive
        with tarfile.open(source, "r:gz") as bundle:
            root = JAVA_CACHE.resolve()
            for member in bundle.getmembers():
                target = (JAVA_CACHE / member.name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"Unsafe path in Temurin archive: {member.name}")
            bundle.extractall(JAVA_CACHE)
        java_candidates = list(JAVA_CACHE.glob("*/bin/java"))
    if not java_candidates:
        raise RuntimeError("Temurin JRE 8 was downloaded but java was not found")
    java_home = java_candidates[0].parents[1]
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = f"{java_home / 'bin'}:{os.environ.get('PATH', '')}"
    return java_candidates[0]


def _ensure_engine() -> Iterator[str]:
    """Download, verify, and extract the official MineStudio engine visibly."""
    root = Path(os.environ["MINESTUDIO_DIR"])
    engine_jar = root / "engine" / "build" / "libs" / "mcprec-6.13.jar"
    if engine_jar.exists():
        yield f"MineStudio engine cache ready: {engine_jar}"
        return

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "engine.zip.part"
    completed_archive = root / "engine.zip"
    if completed_archive.exists() and not archive.exists():
        completed_archive.replace(archive)
    aria2 = shutil.which("aria2c")
    if aria2 is None:
        raise RuntimeError("aria2c is required for the resumable MineStudio engine download")
    endpoint = os.environ["HF_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/CraftJarvis/SimulatorEngine/resolve/main/engine.zip"
    process = subprocess.Popen(
        [
            aria2,
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--continue=true",
            "--file-allocation=none",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=1M",
            "--summary-interval=2",
            "--console-log-level=notice",
            "--enable-color=false",
            "--dir",
            str(root),
            "--out",
            archive.name,
            url,
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
        if clean and ("Download complete" in clean or ("[#" in clean and now - last_update >= 3.0)):
            yield f"MineStudio engine · {clean}"
            last_update = now
    if process.wait() != 0:
        raise RuntimeError(f"MineStudio engine download exited with code {process.returncode}")
    if not archive.exists() or archive.stat().st_size != ENGINE_SIZE:
        actual = archive.stat().st_size if archive.exists() else 0
        raise RuntimeError(f"MineStudio engine size mismatch: expected {ENGINE_SIZE}, received {actual}")

    yield "Verifying the official MineStudio engine archive (SHA-256)"
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ENGINE_SHA256:
        raise RuntimeError("MineStudio engine checksum mismatch; the resumable cache was not trusted")

    yield "Extracting the verified MineStudio engine into persistent storage"
    with zipfile.ZipFile(archive) as bundle:
        resolved_root = root.resolve()
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if resolved_root not in target.parents and target != resolved_root:
                raise RuntimeError(f"Unsafe path in MineStudio engine archive: {member.filename}")
        bundle.extractall(root)
    archive.unlink(missing_ok=True)
    if not engine_jar.exists():
        raise RuntimeError("MineStudio engine archive extracted without mcprec-6.13.jar")
    yield f"MineStudio engine cache ready: {engine_jar}"


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


class MinecraftDiscreteEnv(gym.Env):
    """A compact Gymnasium surface over MineStudio's full keyboard/mouse action dictionary."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: dict[str, Any], seed: int, max_steps: int = 256):
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import CommandsCallback, RewardsCallback
        self.task, self.max_steps, self.steps = task, max_steps, 0
        callbacks = [CommandsCallback(commands=task["commands"]), RewardsCallback([{
            "event": task["reward_event"], "objects": task["reward_objects"], "reward": 1.0,
            "identity": task["key"], "max_reward_times": 8,
        }])]
        self.sim = MinecraftSim(action_type="env", obs_size=(84, 84), render_size=(640, 360), seed=seed,
                                num_empty_frames=10, callbacks=callbacks)
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self._last_position: np.ndarray | None = None

    def _position(self, info: dict[str, Any]) -> np.ndarray | None:
        stats = info.get("location_stats") or info.get("location") or {}
        try:
            return np.asarray([stats["xpos"], stats["ypos"], stats["zpos"]], dtype=np.float32)
        except Exception:
            return None

    def _action(self, index: int) -> dict[str, Any]:
        action = copy.deepcopy(self.sim.env.action_space.no_op())
        choices = {
            1: {"forward": 1}, 2: {"back": 1}, 3: {"left": 1}, 4: {"right": 1},
            5: {"jump": 1, "forward": 1}, 6: {"sprint": 1, "forward": 1},
            7: {"attack": 1}, 8: {"attack": 1, "forward": 1},
            9: {"camera": np.asarray([0.0, 12.0], dtype=np.float32)},
        }
        action.update(choices.get(int(index), {}))
        return action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.steps = 0
        obs, info = self.sim.reset()
        self._last_position = self._position(info)
        return np.asarray(obs["image"], dtype=np.uint8), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.sim.step(self._action(int(action)))
        self.steps += 1
        current = self._position(info)
        if current is not None and self._last_position is not None:
            reward += min(0.02, float(np.linalg.norm(current - self._last_position)) * 0.01)
        self._last_position = current
        truncated = bool(truncated or self.steps >= self.max_steps)
        return np.asarray(obs["image"], dtype=np.uint8), float(reward), bool(terminated), truncated, info

    def render(self) -> np.ndarray:
        return np.asarray(self.sim.render(), dtype=np.uint8)

    def close(self) -> None:
        self.sim.close()


def _make_env(task: dict[str, Any], seed: int):
    # MineStudio already publishes an ``episode`` entry in its persistent info
    # dictionary. Gymnasium's RecordEpisodeStatistics asserts that this key is
    # absent at termination, while Stable-Baselines3 adds its own Monitor around
    # this plain environment and records the same statistics without that clash.
    return MinecraftDiscreteEnv(task, seed)


def _record_episode(model: Any, env: Any, output_dir: Path, seed: int) -> tuple[str, float]:
    frames: list[np.ndarray] = []
    total = 0.0
    observation, _ = env.reset(seed=seed)
    for _ in range(256):
        frames.append(env.render())
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    path = save_gif(frames, output_dir / "learned-policy.gif", fps=20)
    return path, total


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = next(item for item in TASKS if item["key"] == key)
    xvfb: subprocess.Popen[str] | None = None
    env = None
    yield {"phase": "initializing", "step": 0, "log": f"Preparing persistent MineStudio engine cache at {os.environ['MINESTUDIO_DIR']}"}
    try:
        java = _ensure_java8()
        yield {"phase": "initializing", "step": 0, "log": f"Temurin Java 8 ready: {java}\nMineStudio engine source: {os.environ['HF_ENDPOINT']}"}
        for detail in _ensure_engine():
            yield {"phase": "initializing", "step": 0, "log": detail}
        yield {"phase": "initializing", "step": 0, "log": "Starting the Minecraft renderer"}
        xvfb = _start_xvfb(); time.sleep(1.0)
        yield {
            "phase": "initializing",
            "step": 0,
            "detail": "Launching the Minecraft Java world",
            "log": (
                "Launching the MineStudio Java/Forge world. The first world connection "
                "normally takes several minutes; this run remains active while the controls stay locked."
            ),
        }
        env = _make_env(task, int(seed))

        class MetricsCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0); self.latest: dict[str, Any] = {}
            def _on_step(self) -> bool:
                self.latest = dict(self.logger.name_to_value); return True

        callback = MetricsCallback()
        model = PPO("CnnPolicy", env, learning_rate=float(learning_rate), gamma=float(gamma), ent_coef=float(epsilon),
                    n_steps=256, batch_size=64, n_epochs=4, device=device, seed=int(seed), verbose=0)
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
        # PPO uses 256-step rollouts internally. Each requested epoch becomes
        # one actual saved policy; rendering happens later from those exact
        # files after the training Java world has been released.
        for checkpoint_index, target in enumerate(checkpoint_targets, start=1):
            model.learn(total_timesteps=max(1, target - completed), reset_num_timesteps=False,
                        callback=callback, progress_bar=False)
            completed = target
            score = float(callback.latest.get("rollout/ep_rew_mean") or 0.0)
            x.append(float(completed)); y.append(score)
            epoch_dir = artifact_root / f"{key}-{run_token}-epoch-{checkpoint_index:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir / "policy"))
            model_file = epoch_dir / "policy.zip"
            saved_models.append((checkpoint_index, completed, score, model_file))
            log = (f"Minecraft PPO update · step={completed:,}\n"
                   f"device={model.device}  fps={callback.latest.get('time/fps', '—')}\n"
                   f"episode_reward_mean={score:.4f}  value_loss={float(callback.latest.get('train/value_loss') or 0):.5f}\n"
                   f"policy_gradient_loss={float(callback.latest.get('train/policy_gradient_loss') or 0):.5f}\n"
                   f"SAVE epoch={checkpoint_index}/{checkpoint_count} model={model_file.name}")
            yield {"phase": "training", "step": completed, "score": score, "x": x, "y": y,
                   "detail": f"{completed:,}/{int(budget):,} Minecraft steps", "metric_detail": "episode reward mean", "log": log}
        env.close(); env = None
        yield {"phase": "finalizing", "step": completed, "x": x, "y": y,
               "detail": "Starting one replay world for all saved epochs",
               "log": "Training is complete. Reusing one fresh Minecraft world to replay every saved policy."}
        replay_env = _make_env(task, int(seed) + 10_000)
        try:
            for checkpoint_index, saved_step, score, model_file in saved_models:
                replay_model = PPO.load(str(model_file), device=device)
                epoch_dir = model_file.parent
                preview, evaluation = _record_episode(
                    replay_model,
                    replay_env,
                    epoch_dir,
                    int(seed) + 10_000 + checkpoint_index,
                )
                (epoch_dir / "metadata.json").write_text(json.dumps({
                    "environment": task["environment"],
                    "algorithm": "CNN PPO",
                    "step": saved_step,
                    "epoch": checkpoint_index,
                    "epochs": len(saved_models),
                    "training_score": score,
                    "evaluation_return": evaluation,
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
                    "metric_detail": "deterministic replay return",
                    "detail": f"Rendered replay {checkpoint_index}/{len(saved_models)}",
                    "log": f"REPLAY epoch={checkpoint_index}/{len(saved_models)} model={model_file.name} return={evaluation:.3f}",
                }
        finally:
            replay_env.close()
        yield {"phase": "complete", "step": completed, "score": y[-1] if y else None, "x": x, "y": y,
               "log": f"Saved {len(saved_models)} independently selectable Minecraft policies and replays"}
    finally:
        if env is not None:
            env.close()
        _stop_process(xvfb)
