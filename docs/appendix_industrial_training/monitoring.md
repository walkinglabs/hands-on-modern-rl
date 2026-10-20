---
search: false
---

# 训练监控与故障排查（已并入 A.4）

> 这一页保留为旧链接入口。核心内容已经合并到[附录 A.4 评测基准](./evaluation-badcase)的“训练监控与故障排查”部分。下面保留原文，方便从旧链接进入的读者对照。

> RL 训练不像监督学习那样稳定。reward 可能突然暴跌，策略可能坍缩，KL 散度可能飙升——而且这些变化往往在几步训练之内就会发生。
>
> 本节只回答一个问题：**训练出了问题怎么第一时间发现，怎么快速定位原因？**

## 实时监控仪表盘

训练时必须实时盯的 6 个指标：

| 指标                | 正常表现   | 异常意味着什么                        |
| ------------------- | ---------- | ------------------------------------- |
| **Training Reward** | 稳步上升   | 突然暴跌 = 策略崩溃                   |
| **KL Divergence**   | 缓慢增长   | 飙升 = 策略漂移过大                   |
| **Policy Entropy**  | 缓慢下降   | 骤降 = 过早收敛（策略坍缩到狭窄空间） |
| **Clip Fraction**   | < 0.3      | 持续 > 0.3 = 更新太激进               |
| **Value Loss**      | 持续下降   | 不降 = Critic 学不动                  |
| **Reward Margin**   | 稳定或增大 | 缩小 = Reward Model 失效              |

**最危险的信号**：Reward 在涨但 Entropy 在暴跌——模型可能在 reward hacking（学到了"骗取"奖励的捷径），详见 [附录 A](/appendix_industrial_training/training-debugging)。

## 常见问题速查表

| 现象                  | 可能原因       | 解决方案                           |
| --------------------- | -------------- | ---------------------------------- |
| Reward 暴跌           | 策略崩溃       | 降低学习率，增大 clip epsilon      |
| KL 飙升               | 更新步长太大   | 减小 batch size 或增大 KL 惩罚系数 |
| Entropy 骤降          | 探索不足       | 增大 entropy bonus                 |
| 评测指标不涨          | Reward hacking | 检查 RM 评分与真实质量的相关性     |
| OOM                   | 模型/数据太大  | 换用 FSDP/ZeRO-3，减少 batch size  |
| Reward 上升但质量下降 | RM 失效        | 回滚到最佳 checkpoint，重新训练 RM |

## 监控工具

- **Weights & Biases (wandb)**：最常用，一行代码接入，自动记录所有指标
- **TensorBoard**：PyTorch 原生集成，本地可视化
- **自定义仪表盘**：Grafana + Prometheus，适合生产环境

```python
# wandb 接入示例
import wandb
wandb.init(project="my-rl-training")
wandb.log({
    "reward": mean_reward,
    "kl_divergence": kl_div,
    "entropy": entropy,
    "clip_fraction": clip_frac,
})
```

## 什么时候该停下来

不要等训练跑完才看结果。以下情况应该立即停止：

1. **KL 散度超过阈值**（通常设 0.1-0.15）——策略已经偏离太远
2. **Reward 连续 N 步下降**（N 取决于你的容忍度，通常 50-100 步）
3. **Entropy 降到接近 0**——策略已经坍缩，继续训练没有意义
4. **评测 benchmark 分数连续 2 次下降**——模型在退化

最佳实践：每 100-500 步跑一次评测 benchmark，设置自动回滚——一旦评测分数低于历史最佳，自动回退到上一个 checkpoint。详见 [A.4 RL 后训练与 Agentic RL Benchmark](./evaluation-badcase) 中的自动化闭环。
