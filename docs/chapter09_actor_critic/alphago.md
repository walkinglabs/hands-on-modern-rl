# 7.5 动手：AlphaGo 复现

> **本节目标**：在 6×6 棋盘上实现策略网络、价值网络和蒙特卡洛树搜索，复现 AlphaGo 的核心决策流程。

> **学习路径**：[6.1 策略梯度与 REINFORCE](../chapter08_policy_gradient/reinforce) → [7.2 Actor-Critic](./actor-critic) → **7.5 AlphaGo 复现**

> **本节代码与资源**：本页按棋盘环境、策略网络、价值网络、MCTS 和自我对弈的顺序给出完整代码。

<OnlineTraining studios="board" compact />

本页代码按组件分块展示，需要依次放入同一个 Python 文件后再运行。它是 6×6 棋盘上的教学实现，验收目标是合法自我对弈和 MCTS 数据流，不代表标准 AlphaGo 的训练规模或棋力。

前面已经分别使用策略网络选择动作、使用价值网络评估状态。AlphaGo 在这两个网络之间加入蒙特卡洛树搜索：策略网络缩小候选落子范围，价值网络估计叶节点胜率，搜索再根据多次模拟决定真正的落子。

本节在 6×6 棋盘上实现这条完整链路。训练规模远小于标准围棋，目标也不是复现职业棋力；验收标准是程序能够完成合法自我对弈，MCTS 能利用网络先验搜索，训练后的策略在固定评测中优于随机落子。

::: tip 为什么是 6×6？
标准围棋是 19×19，状态空间约 $2 \times 10^{170}$，连遍历都做不到。6×6 棋盘把复杂度降到可以在笔记本上训练的程度，同时保留了围棋"围地、吃子、判断胜负"的核心机制。AlphaGo 的所有核心组件——策略网络、价值网络、MCTS——在 6×6 上一个不少。
:::

## 7.5.1 AlphaGo 的核心组件

AlphaGo 由三个核心组件构成：

| 组件           | 作用                       | 对应本章概念                                              |
| -------------- | -------------------------- | --------------------------------------------------------- |
| 策略网络       | 给出每个合法落子位置的概率 | [第 6 章策略梯度](../chapter08_policy_gradient/reinforce) |
| 价值网络       | 评估当前局面的胜率         | [补充阅读：Critic 训练](./critic-training)                |
| 蒙特卡洛树搜索 | 向前看若干步，找到最佳落子 | 这一节新引入                                              |

它们的关系是：MCTS 是"大脑"，策略网络提供"直觉"（优先搜索哪些分支），价值网络提供"判断"（不用搜到底就能评估局面）。

## 7.5.2 6×6 棋盘环境

我们用一个极简的围棋环境，只实现最基本的规则：落子、提子、判断胜负（数子法）。

```python
import numpy as np

BOARD_SIZE = 6
EMPTY, BLACK, WHITE = 0, 1, -1

class MiniGo:
    """6×6 极简围棋环境"""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.ko_point = None  # 劫的禁入点
        self.passes = 0       # 连续 pass 计数
        self.history = []     # 用于检测超级劫

    def copy(self):
        env = MiniGo()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.ko_point = self.ko_point
        env.passes = self.passes
        env.history = list(self.history)
        return env

    def get_opponent(self, player):
        return -player

    def on_board(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def get_neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self.on_board(nr, nc):
                yield nr, nc

    def get_group(self, r, c):
        """获取 (r,c) 所在的连通块及其气数"""
        color = self.board[r, c]
        if color == EMPTY:
            return set(), 0
        visited = set()
        liberties = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in self.get_neighbors(cr, cc):
                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return visited, len(liberties)

    def remove_group(self, group):
        for r, c in group:
            self.board[r, c] = EMPTY

    def play(self, r, c):
        """在 (r,c) 落子，返回是否合法"""
        if not self.on_board(r, c) or self.board[r, c] != EMPTY:
            return False
        if (r, c) == self.ko_point:
            return False

        self.board[r, c] = self.current_player
        opponent = self.get_opponent(self.current_player)
        captured = []

        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr, nc] == opponent:
                group, liberties = self.get_group(nr, nc)
                if liberties == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # 劫检测：恰好提一子且落子点只有一气
        if len(captured) == 1:
            _, my_liberties = self.get_group(r, c)
            if my_liberties == 1:
                self.ko_point = captured[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # 自杀检测
        _, my_liberties = self.get_group(r, c)
        if my_liberties == 0:
            self.board[r, c] = EMPTY
            return False

        self.passes = 0
        self.current_player = opponent
        return True

    def pass_turn(self):
        """跳过"""
        self.ko_point = None
        self.passes += 1
        self.current_player = self.get_opponent(self.current_player)

    def is_game_over(self):
        return self.passes >= 2

    def get_legal_moves(self):
        """返回所有合法落子位置"""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] == EMPTY and (r, c) != self.ko_point:
                    # 模拟落子检查合法性
                    env_copy = self.copy()
                    if env_copy.play(r, c):
                        moves.append((r, c))
        return moves

    def compute_score(self):
        """简单数子：棋子数 + 围住的空地"""
        score = {BLACK: 0, WHITE: 0}
        visited = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] != EMPTY:
                    score[self.board[r, c]] += 1
                elif (r, c) not in visited:
                    # BFS 找空地连通块
                    region = set()
                    borders = set()
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in region:
                            continue
                        if self.board[cr, cc] == EMPTY:
                            region.add((cr, cc))
                            visited.add((cr, cc))
                            for nr, nc in self.get_neighbors(cr, cc):
                                if self.board[nr, nc] == EMPTY:
                                    stack.append((nr, nc))
                                else:
                                    borders.add(self.board[nr, nc])
                    # 如果只被一方包围，算作该方领地
                    if len(borders) == 1:
                        score[list(borders)[0]] += len(region)

        # 贴目 3.75（6×6 棋盘通常贴 3.75 目）
        score[WHITE] += 3.75
        return score

    def get_winner(self):
        """返回胜者：BLACK 或 WHITE"""
        score = self.compute_score()
        return BLACK if score[BLACK] > score[WHITE] else WHITE
```

环境虽然简化，但保留了围棋的精髓：落子、提子、劫、围地、贴目。

## 7.5.3 策略网络与价值网络

AlphaGo 用了两个网络，输入都是棋盘状态，但输出不同：

- **策略网络**：输出每个位置的落子概率（Actor）
- **价值网络**：输出一个标量，表示当前玩家的胜率（Critic）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """基础卷积块：Conv3x3 + BatchNorm + ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class AlphaGoNet(nn.Module):
    """AlphaGo 风格的双头网络"""

    def __init__(self, board_size=BOARD_SIZE, num_blocks=4, channels=64):
        super().__init__()
        self.board_size = board_size

        # 输入：2 通道（黑子位置、白子位置）
        self.input_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)

        # 残差块
        self.blocks = nn.ModuleList([ConvBlock(channels) for _ in range(num_blocks)])

        # 策略头：输出 board_size × board_size 的 logits
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size,
                                   board_size * board_size)

        # 价值头：输出一个标量胜率
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board, current_player):
        """
        Args:
            board: (B, board_size, board_size) 棋盘状态
            current_player: (B,) 当前玩家 (1=黑, -1=白)
        Returns:
            policy_logits: (B, board_size * board_size)
            value: (B, 1) 当前玩家的胜率 [-1, 1]
        """
        # 编码棋盘：2 通道（当前玩家的子、对手的子）
        player_mask = current_player.view(-1, 1, 1).unsqueeze(1)  # (B,1,1,1)
        own = (board.unsqueeze(1) == player_mask).float()         # (B,1,H,W)
        opp = (board.unsqueeze(1) == -player_mask).float()        # (B,1,H,W)
        x = torch.cat([own, opp], dim=1)                          # (B,2,H,W)

        # 共享特征提取
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = x + block(x)  # 残差连接

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
```

这个双头网络和[第 7.2 节的 Actor-Critic](./actor-critic) 是同一个思路——共享特征提取，策略头做决策，价值头做评估。

## 7.5.4 蒙特卡洛树搜索

MCTS 是 AlphaGo 的"思考"过程。它在落子前向前模拟若干局，把搜索结果汇总成更靠谱的策略。核心思想：

1. **选择（Select）**：从根节点开始，用 UCB 公式选最有"潜力"的子节点
2. **扩展（Expand）**：到达叶子节点后，用策略网络生成子节点
3. **评估（Evaluate）**：用价值网络评估叶子节点（不再需要模拟到底）
4. **回传（Backpropagate）**：把评估结果沿路径更新回根节点

```python
import math

class MCTSNode:
    """MCTS 树节点"""

    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}       # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior       # 策略网络给出的先验概率

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct=1.5):
        """PUCT 公式：Q + U（探索 bonus）"""
        if self.visit_count == 0:
            return float('inf')
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) \
            / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self):
        """选 UCB 分数最高的子节点"""
        return max(self.children.items(),
                   key=lambda item: item[1].ucb_score())

    def expand(self, action_priors):
        """根据策略网络的输出扩展子节点"""
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def backpropagate(self, value):
        """回传价值（切换视角）"""
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            # 父节点是对方的回合，价值取反
            self.parent.backpropagate(-value)


class MCTS:
    """蒙特卡洛树搜索"""

    def __init__(self, model, c_puct=1.5, num_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, env):
        """从当前状态执行 MCTS，返回各动作的访问次数"""
        root = MCTSNode()

        # 用温度参数控制探索程度
        for _ in range(self.num_simulations):
            node = root
            sim_env = env.copy()

            # 1. 选择：沿树向下走到叶子
            while node.children:
                action, node = node.select_child()
                sim_env.play(*action)

            # 2. 评估：用网络预测策略和价值
            board_tensor = torch.tensor(sim_env.board, dtype=torch.float32).unsqueeze(0)
            player_tensor = torch.tensor([sim_env.current_player], dtype=torch.float32)

            with torch.no_grad():
                policy_logits, value = self.model(board_tensor, player_tensor)

            # 3. 扩展：只扩展合法动作
            legal_moves = sim_env.get_legal_moves()
            if legal_moves:
                # 把非法位置的 logits 设为 -inf
                mask = torch.full((BOARD_SIZE * BOARD_SIZE,), float('-inf'))
                for r, c in legal_moves:
                    mask[r * BOARD_SIZE + c] = policy_logits[0, r * BOARD_SIZE + c]
                probs = torch.softmax(mask, dim=0)

                action_priors = [
                    ((r, c), probs[r * BOARD_SIZE + c].item())
                    for r, c in legal_moves
                ]
                node.expand(action_priors)
            else:
                # 无合法落子，pass
                pass

            # 4. 回传
            node.backpropagate(value.item())

        # 根据访问次数计算最终策略
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts
```

注意 `backpropagate` 中的 `-value`——这是零和博弈的关键：对黑棋有利的情况，对白棋同样不利。价值在每一层都要翻转。

## 7.5.5 自我对弈训练

AlphaGo 最革命性的思路是**自我对弈（Self-Play）**：让 AI 和自己对弈，用对弈结果来训练自己。赢了就强化走的棋，输了就弱化。这正是策略梯度的思路——只不过样本来源从"人类棋谱"变成了"自己的对弈"。

```python
def self_play_game(model, mcts, temperature=1.0):
    """用 MCTS 自我对弈一局，返回 (states, policies, winner)"""
    env = MiniGo()
    states, players_list, policies = [], [], []
    max_moves = BOARD_SIZE * BOARD_SIZE * 2  # 防止无限对弈

    for _ in range(max_moves):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            env.pass_turn()
            if env.is_game_over():
                break
            continue

        # MCTS 搜索
        visit_counts = mcts.run(env)
        total_visits = sum(visit_counts.values())

        # 计算策略概率（访问次数的分布）
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for (r, c), visits in visit_counts.items():
            policy[r * BOARD_SIZE + c] = visits / total_visits

        # 训练初期加温度噪声鼓励探索
        if temperature > 0:
            noisy_policy = policy ** (1.0 / temperature)
            noisy_policy /= noisy_policy.sum() + 1e-8
            action_idx = np.random.choice(len(policy), p=noisy_policy)
        else:
            action_idx = policy.argmax()

        r, c = divmod(action_idx, BOARD_SIZE)

        states.append(env.board.copy())
        players_list.append(env.current_player)
        policies.append(policy)

        env.play(r, c)
        if env.is_game_over():
            break

    # 判定胜负
    winner = env.get_winner()

    # 将胜负转化为 +1/-1 的价值标签
    values = []
    for player in players_list:
        values.append(1.0 if player == winner else -1.0)

    return states, players_list, policies, values


def train_alphago(num_iterations=20, games_per_iter=10, num_epochs=5):
    """AlphaGo 训练主循环"""
    model = AlphaGoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mcts = MCTS(model, num_simulations=50)  # 训练时用少量模拟

    replay_buffer = []  # (state, player, policy, value)

    for iteration in range(num_iterations):
        # 阶段 1：自我对弈收集数据
        new_data = []
        for _ in range(games_per_iter):
            # 前期高温探索，后期低温利用
            temp = 1.0 if iteration < num_iterations // 2 else 0.5
            states, players, policies, values = self_play_game(model, mcts, temp)
            for s, p, pi, v in zip(states, players, policies, values):
                new_data.append((s, p, pi, v))

        replay_buffer.extend(new_data)
        # 只保留最近 5000 条
        if len(replay_buffer) > 5000:
            replay_buffer = replay_buffer[-5000:]

        # 阶段 2：用收集的数据训练网络
        model.train()
        for epoch in range(num_epochs):
            # 随机采样 mini-batch
            indices = np.random.choice(len(replay_buffer),
                                       size=min(64, len(replay_buffer)),
                                       replace=False)

            boards = torch.stack([
                torch.tensor(replay_buffer[i][0], dtype=torch.float32)
                for i in indices
            ])
            players = torch.tensor(
                [replay_buffer[i][1] for i in indices], dtype=torch.float32
            )
            target_policies = torch.stack([
                torch.tensor(replay_buffer[i][2], dtype=torch.float32)
                for i in indices
            ])
            target_values = torch.tensor(
                [replay_buffer[i][3] for i in indices], dtype=torch.float32
            ).unsqueeze(1)

            # 前向传播
            policy_logits, pred_values = model(boards, players)

            # 策略损失：交叉熵（MCTS 策略作为监督信号）
            policy_loss = F.cross_entropy(policy_logits, target_policies)

            # 价值损失：均方误差
            value_loss = F.mse_loss(pred_values, target_values)

            # 总损失
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration+1}/{num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Buffer: {len(replay_buffer)}")

    return model
```

训练的核心循环只有两个阶段：

1. **自我对弈**：用当前模型 + MCTS 下棋，收集 (局面, MCTS策略, 胜负) 三元组
2. **网络训练**：策略网络学习模仿 MCTS 的搜索结果，价值网络学习预测最终胜负

这和第 6 章的[策略梯度](../chapter08_policy_gradient/reinforce)有一个微妙但重要的区别：AlphaGo 的策略网络不是直接从对弈回报学习（像 REINFORCE），而是**学习模仿 MCTS 的搜索结果**。MCTS 做了大量模拟后给出的策略比单次采样可靠得多——这相当于一个天然的低方差基线。

## 7.5.6 人机对弈

```python
def human_vs_ai(model, mcts, human_color=BLACK):
    """人类 vs AI 的交互对弈"""
    env = MiniGo()
    print(f"你是{'黑棋(X)' if human_color == BLACK else '白棋(O)'}")
    print("输入格式：行 列（如 '2 3'），输入 'pass' 跳过\n")

    while not env.is_game_over():
        print(env_to_string(env.board))

        if env.current_player == human_color:
            # 人类回合
            legal = env.get_legal_moves()
            print(f"合法落子: {legal}")
            cmd = input("你的落子: ").strip()
            if cmd == 'pass':
                env.pass_turn()
            else:
                r, c = map(int, cmd.split())
                if not env.play(r, c):
                    print("非法落子，请重试")
                    continue
        else:
            # AI 回合
            visit_counts = mcts.run(env)
            if visit_counts:
                best_action = max(visit_counts, key=visit_counts.get)
                print(f"AI 落子: {best_action} "
                      f"(访问次数: {visit_counts[best_action]})")
                env.play(*best_action)
            else:
                print("AI: pass")
                env.pass_turn()
        print()

    # 游戏结束
    score = env.compute_score()
    print(env_to_string(env.board))
    print(f"黑棋: {score[BLACK]:.1f} 目 | 白棋: {score[WHITE]:.1f} 目")
    winner = "黑棋" if score[BLACK] > score[WHITE] else "白棋"
    print(f"{winner} 获胜！")


def env_to_string(board):
    symbols = {EMPTY: '·', BLACK: 'X', WHITE: 'O'}
    lines = ["   " + " ".join(str(i) for i in range(BOARD_SIZE))]
    for r in range(BOARD_SIZE):
        line = f"{r}: " + " ".join(symbols[board[r, c]] for c in range(BOARD_SIZE))
        lines.append(line)
    return "\n".join(lines)
```

## 7.5.7 AlphaGo 与本章概念

把 AlphaGo 的每个组件对应回本章学过的知识：

| AlphaGo 组件  | 对应概念                 | 出处                                                   |
| ------------- | ------------------------ | ------------------------------------------------------ |
| 策略网络      | Actor，输出动作概率      | [策略梯度定理](../chapter08_policy_gradient/reinforce) |
| 价值网络      | Critic，评估局面价值     | [Actor-Critic 架构](./actor-critic)                    |
| MCTS 策略监督 | 降低方差的"可靠策略信号" | [基线](../chapter08_policy_gradient/pg-improvements)   |
| 自我对弈      | 在线采样 + 策略改进      | REINFORCE 的采样思想                                   |
| $-v$ 回传     | 零和博弈的对称性         | [优势函数](./advantage-function)的符号翻转             |

这个简化实现展示了三部分的分工：策略网络提供候选落子的先验概率，价值网络评估叶节点，MCTS 汇总多次模拟后选择动作。标准 AlphaGo 的系统规模和训练数据远大于本节实验，本节只复现这条决策链路。

## 7.5.8 开源项目与数据集

上面的代码用于理解核心组件。若要继续运行更完整的围棋系统，可以使用下面的开源项目和数据集：

### 推荐开源项目

| 项目                                                                  | 说明                                                 | 适合场景                             |
| --------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------ |
| [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | PyTorch，游戏无关架构，自带 Othello/Gomoku/TicTacToe | **入门首选**，代码最简洁，笔记本可跑 |
| [michaelnny/alpha_zero](https://github.com/michaelnny/alpha_zero)     | PyTorch，9×9 围棋 + 15×15 五子棋                     | 想在 9×9 棋盘上跑真正的围棋          |
| [KataGo](https://github.com/lightvector/KataGo)                       | C++/Python，支持 7×7 到 19×19，有预训练模型          | 需要**预训练权重**做实验或对战       |
| [Leela Zero](https://github.com/leela-zero/leela-zero)                | C++，AlphaGo Zero 的忠实复现                         | 研究 AlphaGo Zero 的原始算法         |
| [MiniZero](https://github.com/rlglab/minizero)                        | C++/Python，支持 AlphaZero/MuZero/Gumbel 变体        | 对比不同 MCTS 算法变体               |

**最推荐的路径**：先跑 [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 的 Othello（自带环境，零配置），理解整个 pipeline 后，再切到围棋场景。

### 可用数据集

| 数据集                                                            | 规模              | 说明                                                            |
| ----------------------------------------------------------------- | ----------------- | --------------------------------------------------------------- |
| [JGDB](https://pjreddie.com/projects/jgdb/)                       | 53.5 万局，194 MB | **最佳选择**，已预分 train/val/test，公共领域，由 YOLO 作者制作 |
| [featurecat/go-dataset](https://github.com/featurecat/go-dataset) | 2110 万局         | 最大规模，来自 Fox 弈城，涵盖 18k 到 9p                         |
| [CWI 日本职业棋谱](https://homepages.cwi.nl/~aeb/go/games/games/) | 8.8 万局，45 MB   | 职业棋手对局，精校数据                                          |
| [KGS 棋谱档案](https://www.gokgs.com/)                            | 百万局以上        | KGS 围棋服务器存档，各段位混合                                  |

用 JGDB 做监督预训练的流程：下载 SGF → 解析为 (棋盘, 落子) 对 → 训练策略网络模仿人类走法 → 再用自我对弈强化。这正是 AlphaGo 论文中第一阶段的做法。

## 本节小结

- 策略网络为 MCTS 提供候选落子的先验概率，价值网络估计叶节点的胜率。
- MCTS 通过选择、扩展、评估和回传整合网络输出，再把搜索访问次数转成改进策略。
- 6×6 实验验证的是组件与数据流；标准围棋棋力还需要大规模棋谱、自我对弈和搜索计算。

## 进一步探索

1. **增加棋盘尺寸**：用 [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 跑 9×9 围棋，观察训练时间和棋力的变化。
2. **AlphaZero 化**：去掉人类棋谱预训练，完全从零开始自我对弈。在 6×6 上训练多少轮能收敛？
3. **MCTS 模拟次数**：对比 10 次、50 次、200 次模拟对棋力的影响。更多模拟一定更好吗？
4. **用 JGDB 做监督预训练**：下载 [JGDB 数据集](https://pjreddie.com/projects/jgdb/)，解析 SGF 格式棋谱，先让策略网络学会"模仿人类走法"，再用自我对弈强化——这就是完整的 AlphaGo pipeline。

AlphaGo 展示了 Actor-Critic 架构在大规模离散动作空间中的威力。下一节我们回到连续控制，用 A2C 训练一个更具挑战性的机器人 locomotion 任务：[7.6 动手：BipedalWalker 双足行走](./bipedalwalker)。

## 参考文献

[^1]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489. [DOI](https://doi.org/10.1038/nature16961)

[^2]: Silver, D., et al. (2017). Mastering the game of Go without human knowledge. _Nature_, 550(7676), 354-359. [DOI](https://doi.org/10.1038/nature24270)

[^3]: Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. _Science_, 362(6419), 1140-1144. [DOI](https://doi.org/10.1126/science.aar6404)
