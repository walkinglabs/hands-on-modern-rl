from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sb3_tools import save_gif, train_sb3


ROOT = Path(__file__).resolve().parent

SPACE = {
    "title": {"en": "Atari xGPU Training Arcade", "zh": "Atari xGPU 在线训练街机厅"},
    "description": {
        "en": "Train DQN agents from Atari pixels, inspect checkpoint rewards, and render this run's learned policy inside the original ALE emulator.",
        "zh": "让 DQN 从 Atari 像素画面中学习，观察检查点评估，并在 ALE 模拟器中生成本次策略回放。",
    },
    "badge": "EXPERIMENT 03 · ARCADE",
    "training_guide": {
        "success": {"en": "The evaluation reward should rise above early checkpoints, and the replay should sustain useful play or improve the game score. Training complete only confirms the run ended normally.", "zh": "评估奖励应高于早期检查点，回放中应能持续做出有效动作或提高游戏得分；“训练完成”只表示运行正常结束。"},
        "preview": {"en": "The first clip shows the selected Atari game. The completed run replaces it with a replay rendered by this run's learned DQN policy.", "zh": "初始画面展示所选 Atari 游戏；训练完成后会替换为本次 DQN 策略在模拟器中生成的回放。"},
        "time": {"en": "The recommended baselines prioritize learned behavior over a short demo and can still take tens of minutes on xGPU because ALE environment stepping and replay evaluation remain CPU-bound. The estimate beside each game is a planning range, not a timeout.", "zh": "推荐 baseline 优先保证能学到行为。即使使用 xGPU，ALE 环境步进和回放评估仍受 CPU 限制，完整训练仍可能需要几十分钟。每个游戏旁的时间是规划区间，不是超时限制。"},
    },
    "device": "CPU / GPU · DQN",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter07_dqn/dqn-family",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment03-atari/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment03-atari.ipynb",
}


def task(
    key,
    title,
    environment,
    description,
    action,
    preview,
    *,
    baseline_budget,
    baseline_time,
    baseline_outcome,
    baseline_epochs=5,
    budget_range=(1_000, 5_000_000, 1_000),
):
    learning_starts = min(100_000, max(20_000, baseline_budget // 10))
    steps_per_epoch = max(1_000, baseline_budget // baseline_epochs)
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": {"en": "84×84 grayscale frame stack", "zh": "84×84 灰度帧堆叠"},
        "action": action,
        "algorithm": "DQN",
        "policy": "CnnPolicy",
        "preview": preview,
        "budget": (budget_range[0], budget_range[1], baseline_budget, budget_range[2]),
        "steps_per_epoch": (1_000, max(400_000, steps_per_epoch), steps_per_epoch, 1_000),
        "epochs": (1, 12, baseline_epochs, 1),
        "learning_rate": (1e-5, 0.0005, 0.0001, 1e-5),
        "gamma": (0.9, 1.0, 0.99, 0.005),
        "epsilon": (0.1, 1.0, 1.0, 0.05),
        "baseline_name": "Atari DQN xGPU baseline v4",
        "baseline_time": baseline_time,
        "baseline_outcome": baseline_outcome,
        "learning_starts": learning_starts,
        "buffer_size": 100_000,
        "batch_size": 32,
        "target_update_interval": 10_000,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.01,
        "optimize_memory_usage": True,
        "checkpoints": baseline_epochs,
        "device": "cuda",
    }


TASKS = [
    task("pong", "Pong", "ALE/Pong-v5", {"en": "Track the ball and move the paddle to outscore the opponent.", "zh": "跟踪球的位置并移动球拍，以更高比分击败对手。"}, {"en": "Paddle and fire controls", "zh": "球拍移动与发球"}, "assets/pong.gif", baseline_budget=1_000_000, baseline_time={"en": "about 30–90 xGPU minutes", "zh": "约 30–90 个 xGPU 分钟"}, baseline_outcome={"en": "Longer rallies and evaluation reward moving above the random-policy floor; strong positive scores may need several million steps.", "zh": "回合能够明显延长，评估奖励脱离随机策略下限；稳定正分通常还需要数百万步。"}),
    task("breakout", "Breakout", "ALE/Breakout-v5", {"en": "Bounce the ball, clear bricks, and preserve each life.", "zh": "反弹小球、清除砖块并尽量保住生命。"}, {"en": "Paddle and fire controls", "zh": "球拍移动与发球"}, "assets/breakout.gif", baseline_budget=1_000_000, baseline_time={"en": "about 30–90 xGPU minutes", "zh": "约 30–90 个 xGPU 分钟"}, baseline_outcome={"en": "The paddle begins tracking the ball and clears bricks more consistently than an untrained policy.", "zh": "挡板开始追踪小球，清砖表现明显优于未训练策略。"}),
    task("space-invaders", "Space Invaders", "ALE/SpaceInvaders-v5", {"en": "Move, shoot invading rows, and avoid incoming fire.", "zh": "移动并射击入侵队列，同时躲避敌方火力。"}, {"en": "Move / fire", "zh": "移动、射击"}, "assets/space-invaders.gif", baseline_budget=1_500_000, baseline_time={"en": "about 45–120 xGPU minutes", "zh": "约 45–120 个 xGPU 分钟"}, baseline_outcome={"en": "Sustained firing and useful horizontal control with a score above early checkpoints.", "zh": "能够持续射击并进行有效横向控制，得分高于早期检查点。"}),
    task("freeway", "Freeway", "ALE/Freeway-v5", {"en": "Time vertical movements to cross lanes of traffic safely.", "zh": "掌握上下移动的时机，安全穿过多条车道。"}, {"en": "Up / down", "zh": "向上、向下"}, "assets/freeway.png", baseline_budget=300_000, baseline_time={"en": "about 10–30 xGPU minutes", "zh": "约 10–30 个 xGPU 分钟"}, baseline_outcome={"en": "Repeated upward crossings and a clearly non-zero evaluation score. This is the fastest recommended first run.", "zh": "能够反复向上穿越车流，评估分数明显大于零；这是最适合作为第一次训练的游戏。"}, baseline_epochs=6, budget_range=(1_000, 2_000_000, 1_000)),
    task("seaquest", "Seaquest", "ALE/Seaquest-v5", {"en": "Rescue divers while managing oxygen, enemies, and ammunition.", "zh": "在管理氧气、敌人和弹药的同时营救潜水员。"}, {"en": "Move / fire", "zh": "移动、射击"}, "assets/seaquest.gif", baseline_budget=2_000_000, baseline_time={"en": "about 60–180 xGPU minutes", "zh": "约 60–180 个 xGPU 分钟"}, baseline_outcome={"en": "Useful movement and firing with improving score; oxygen management is a harder, longer-horizon behavior.", "zh": "移动和射击开始有效，分数逐步提高；氧气管理属于更难的长时程行为。"}),
    task("qbert", "Q*bert", "ALE/Qbert-v5", {"en": "Plan diagonal jumps to recolor the pyramid without colliding with enemies.", "zh": "规划斜向跳跃改变金字塔颜色，并避开敌人。"}, {"en": "Four diagonal jumps", "zh": "四个斜向跳跃动作"}, "assets/qbert.gif", baseline_budget=1_500_000, baseline_time={"en": "about 45–150 xGPU minutes", "zh": "约 45–150 个 xGPU 分钟"}, baseline_outcome={"en": "Purposeful diagonal movement and more tile changes than early checkpoints.", "zh": "出现有目的的斜向移动，改变的方块数超过早期检查点。"}),
    task("beam-rider", "Beam Rider", "ALE/BeamRider-v5", {"en": "Control horizontal movement and shooting in a fast scrolling arena.", "zh": "在快速滚动的竞技场中控制横向移动和射击。"}, {"en": "Move / fire", "zh": "移动、射击"}, "assets/beam-rider.gif", baseline_budget=2_000_000, baseline_time={"en": "about 60–180 xGPU minutes", "zh": "约 60–180 个 xGPU 分钟"}, baseline_outcome={"en": "Sustained shooting and horizontal control with reward improving across checkpoints.", "zh": "能够持续射击和横向控制，奖励随检查点逐步提高。"}),
    task("enduro", "Enduro", "ALE/Enduro-v5", {"en": "Steer and accelerate through traffic over a long racing horizon.", "zh": "在长时间赛车过程中控制方向、加速并穿过车流。"}, {"en": "Steer / accelerate / brake", "zh": "转向、加速、刹车"}, "assets/enduro.gif", baseline_budget=2_000_000, baseline_time={"en": "about 60–180 xGPU minutes", "zh": "约 60–180 个 xGPU 分钟"}, baseline_outcome={"en": "Forward driving with fewer immediate collisions; strong overtaking behavior needs a longer run.", "zh": "能够持续向前行驶并减少即时碰撞；稳定超车仍需要更长训练。"}),
]


def runtime_status():
    try:
        import ale_py
        import gymnasium as gym
        import torch

        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4)
        env.reset(seed=0)
        env.close()
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return f"ALE {ale_py.__version__} · {device} · ROM READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _make_vec_env(environment: str, seed: int, *, training: bool = False):
    import ale_py
    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    gym.register_envs(ale_py)

    def factory():
        base = gym.make(environment, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0, full_action_space=False)
        return AtariWrapper(
            base,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=training,
            clip_reward=training,
        )

    env = DummyVecEnv([factory])
    env.seed(seed)
    return VecFrameStack(env, n_stack=4)


def _record(model, env, artifacts: Path, seed: int, task, output_path: Path | None = None):
    env.seed(seed)
    observation = env.reset()
    frames: list[np.ndarray] = []
    for step in range(4_000):
        frame = env.render(mode="rgb_array")
        if frame is not None and (step % 2 == 0 or step < 20):
            frames.append(np.asarray(frame))
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _ = env.step(action)
        if bool(np.asarray(done).any()):
            break
        if len(frames) >= 500:
            break
    env.close()
    return save_gif(frames, output_path or artifacts / f"{task['key']}-learned-policy.gif", fps=20)


def _model_record(model_path: Path) -> dict:
    metadata_path = model_path.with_suffix(".json")
    metadata: dict = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    task_key = str(metadata.get("task_key") or "")
    if not task_key:
        for item in TASKS:
            if model_path.name == f"{item['key']}-model.zip":
                task_key = item["key"]
                break
    task = next((item for item in TASKS if item["key"] == task_key), None)
    if task is None:
        raise ValueError(f"Cannot identify the Atari task for {model_path.name}")

    preview_value = metadata.get("preview")
    preview_path = Path(str(preview_value)) if preview_value else None
    if preview_path is not None and not preview_path.is_absolute():
        preview_path = ROOT / preview_path
    if preview_path is not None and not preview_path.is_file():
        preview_path = None
    if preview_path is None:
        candidates = sorted(
            (ROOT / "artifacts").glob(f"{model_path.stem}*-preview.gif"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        legacy_preview = ROOT / "artifacts" / f"{task_key}-learned-policy.gif"
        if not candidates and model_path.name == f"{task_key}-model.zip" and legacy_preview.is_file():
            candidates = [legacy_preview]
        preview_path = candidates[0] if candidates else None

    return {
        **metadata,
        "model_id": model_path.name,
        "model_path": str(model_path),
        "task_key": task_key,
        "title": task["title"],
        "environment": task["environment"],
        "algorithm": str(metadata.get("algorithm") or task["algorithm"]),
        "budget": int(metadata.get("budget") or 0),
        "score": metadata.get("score"),
        "created_at": str(metadata.get("created_at") or ""),
        "preview": str(preview_path) if preview_path is not None else None,
        "modified_ns": model_path.stat().st_mtime_ns,
    }


def model_details(model_id: str) -> dict:
    """Return validated metadata for one model saved by this Studio."""
    artifacts = ROOT / "artifacts"
    safe_name = Path(str(model_id)).name
    if safe_name != str(model_id) or not safe_name.endswith(".zip") or "-model" not in safe_name:
        raise ValueError("Invalid saved-model identifier")
    model_path = artifacts / safe_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {safe_name}")
    return _model_record(model_path)


def list_trained_models(key: str | None = None) -> list[dict]:
    """List every saved policy checkpoint, newest first."""
    artifacts = ROOT / "artifacts"
    records: list[dict] = []
    for model_path in artifacts.glob("*-model*.zip"):
        try:
            record = _model_record(model_path)
        except (OSError, ValueError):
            continue
        if key is None or record["task_key"] == key:
            records.append(record)
    return sorted(records, key=lambda record: int(record["modified_ns"]), reverse=True)


def render_preview(model_id: str):
    """Run the selected saved model with its original deterministic replay setup."""
    record = model_details(model_id)
    task = next(item for item in TASKS if item["key"] == record["task_key"])
    model_path = Path(record["model_path"])

    from stable_baselines3 import DQN

    rollout_seed = max(0, min(int(record.get("seed", 42)), 2**32 - 1))
    model = DQN.load(str(model_path), device="cpu")
    environment = _make_vec_env(task["environment"], rollout_seed, training=False)
    artifacts = ROOT / "artifacts"
    output = artifacts / f"{model_path.stem}-preview.gif"
    preview = _record(model, environment, artifacts, rollout_seed, task, output)
    metadata_path = model_path.with_suffix(".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["preview"] = str(preview)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "preview": preview,
        "model": model_path.name,
        "model_id": model_path.name,
        "task_key": task["key"],
        "detail": f"{task['title']['en']} · {model_path.name} · selected saved policy",
    }


def run(
    key: str,
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    checkpoints: int | None = None,
):
    import torch

    task = dict(next(item for item in TASKS if item["key"] == key))
    task["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    environment = task["environment"]
    yield from train_sb3(
        root=ROOT,
        task=task,
        make_train_env=lambda: _make_vec_env(environment, seed, training=True),
        make_eval_env=lambda: _make_vec_env(environment, seed + 1_000, training=False),
        make_record_env=lambda: _make_vec_env(environment, seed, training=False),
        budget=budget,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
        checkpoint_count=checkpoints,
        record_episode=lambda model, env, artifacts, record_seed: _record(model, env, artifacts, record_seed, task),
    )
