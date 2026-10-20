---
title: '7.5 Hands-On: Reproducing AlphaGo'
---

# 7.5 Hands-on: Reproducing AlphaGo

> **Section goal**: Implement a policy network, value network, and Monte Carlo tree search on a 6×6 board to reproduce the central decision flow of AlphaGo.

> **Learning path**: [6.1 Policy Gradient and REINFORCE](../chapter08_policy_gradient/reinforce) → [7.2 Actor-Critic](./actor-critic) → **7.5 Reproducing AlphaGo**

> **Code and resources**: This page presents the board environment, policy network, value network, MCTS, and self-play code in execution order.

<OnlineTraining studios="board" compact />

After learning policy gradients and Actor-Critic, we already have two weapons in hand:
the **policy network** (deciding where to play next; review: [policy $\pi_\theta(a|s)$](../chapter08_policy_gradient/reinforce))
and the **value network** (judging which side has the better prospects; review: [Critic $V(s)$](./critic-training)).
In 2016, DeepMind's AlphaGo combined these two weapons with Monte Carlo Tree Search (MCTS) and defeated world champion Lee Sedol.
This was one of the most widely recognized moments in the history of reinforcement learning.

In this section, we reproduce AlphaGo's core idea with minimal code: we train an AI that can learn to play Go on a 6x6 board by self-play.

::: tip Why 6x6?
Standard Go is 19x19, with a state space on the order of $2 \times 10^{170}$, far beyond anything we can enumerate.
A 6x6 board reduces the complexity to a level that can be trained on a laptop, while still preserving the core mechanics of Go:
territory, captures, and win/loss judgment.
All key AlphaGo components, policy network, value network, and MCTS, are still present on 6x6.
:::

## 7.5.1 AlphaGo's Core Components

AlphaGo consists of three core components:

| Component               | Role                                                | Related Concept In This Chapter                                     |
| ----------------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| Policy network          | Outputs probabilities over legal moves              | [Chapter 6 Policy Gradient](../chapter08_policy_gradient/reinforce) |
| Value network           | Evaluates the win rate of the current position      | [Supplement: Training the Critic](./critic-training)                |
| Monte Carlo Tree Search | Looks ahead for several moves to find the best play | Newly introduced in this section                                    |

Their relationship is simple: MCTS is the "brain," the policy network provides "intuition" (which branches to prioritize),
and the value network provides "judgment" (so we do not have to search all the way to the end to evaluate a position).

## 7.5.2 The 6x6 Board Environment

We will use a minimal Go environment that implements only the most essential rules:
placing stones, capturing, and determining the winner (area scoring).

```python
import numpy as np

BOARD_SIZE = 6
EMPTY, BLACK, WHITE = 0, 1, -1

class MiniGo:
    """A minimal 6x6 Go environment."""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.ko_point = None  # Ko-prohibited point
        self.passes = 0       # consecutive pass count
        self.history = []     # for superko detection

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
        """Return the connected group containing (r,c) and its number of liberties."""
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
        """Place a stone at (r,c). Returns whether the move is legal."""
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

        # Ko detection: capture exactly one stone and the placed stone has exactly one liberty
        if len(captured) == 1:
            _, my_liberties = self.get_group(r, c)
            if my_liberties == 1:
                self.ko_point = captured[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Suicide detection
        _, my_liberties = self.get_group(r, c)
        if my_liberties == 0:
            self.board[r, c] = EMPTY
            return False

        self.passes = 0
        self.current_player = opponent
        return True

    def pass_turn(self):
        """Pass the turn."""
        self.ko_point = None
        self.passes += 1
        self.current_player = self.get_opponent(self.current_player)

    def is_game_over(self):
        return self.passes >= 2

    def get_legal_moves(self):
        """Return all legal moves."""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] == EMPTY and (r, c) != self.ko_point:
                    # Simulate the move to check legality
                    env_copy = self.copy()
                    if env_copy.play(r, c):
                        moves.append((r, c))
        return moves

    def compute_score(self):
        """Simple area scoring: stones + enclosed empty points."""
        score = {BLACK: 0, WHITE: 0}
        visited = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] != EMPTY:
                    score[self.board[r, c]] += 1
                elif (r, c) not in visited:
                    # BFS over a connected empty region
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
                    # If surrounded by a single color, count as that color's territory
                    if len(borders) == 1:
                        score[list(borders)[0]] += len(region)

        # Komi 3.75 (a common komi choice for 6x6)
        score[WHITE] += 3.75
        return score

    def get_winner(self):
        """Return the winner: BLACK or WHITE."""
        score = self.compute_score()
        return BLACK if score[BLACK] > score[WHITE] else WHITE
```

Even though this environment is simplified, it still keeps the essence of Go: moves, captures, ko, territory, and komi.

## 7.5.3 Policy Network and Value Network

AlphaGo uses two networks. They take the same input (the board state), but produce different outputs:

- **Policy network**: outputs a probability distribution over moves (Actor)
- **Value network**: outputs a scalar win probability for the current player (Critic)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolution block: Conv3x3 + BatchNorm + ReLU."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class AlphaGoNet(nn.Module):
    """An AlphaGo-style dual-head network."""

    def __init__(self, board_size=BOARD_SIZE, num_blocks=4, channels=64):
        super().__init__()
        self.board_size = board_size

        # Input: 2 channels (black stones, white stones)
        self.input_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual blocks
        self.blocks = nn.ModuleList([ConvBlock(channels) for _ in range(num_blocks)])

        # Policy head: output logits over board_size x board_size
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size,
                                   board_size * board_size)

        # Value head: output a scalar win rate
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board, current_player):
        """
        Args:
            board: (B, board_size, board_size) board state
            current_player: (B,) current player (1=black, -1=white)
        Returns:
            policy_logits: (B, board_size * board_size)
            value: (B, 1) win rate for the current player in [-1, 1]
        """
        # Encode the board as 2 channels (current player's stones, opponent's stones)
        player_mask = current_player.view(-1, 1, 1).unsqueeze(1)  # (B,1,1,1)
        own = (board.unsqueeze(1) == player_mask).float()         # (B,1,H,W)
        opp = (board.unsqueeze(1) == -player_mask).float()        # (B,1,H,W)
        x = torch.cat([own, opp], dim=1)                          # (B,2,H,W)

        # Shared feature extraction
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = x + block(x)  # residual connection

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
```

This dual-head design is the same idea as the [Actor-Critic](./actor-critic) model in Section 7.2:
shared feature extraction, a policy head for decisions, and a value head for evaluation.

## 7.5.4 Monte Carlo Tree Search

MCTS is AlphaGo's "thinking" process. Before making a move, it simulates many continuations and aggregates the results into a more reliable policy.
The core loop is:

1. **Select**: from the root, use a UCB-style formula to choose the most "promising" child
2. **Expand**: at a leaf node, use the policy network to create children
3. **Evaluate**: use the value network to evaluate the leaf (no need to roll out to the end)
4. **Backpropagate**: propagate the evaluation back up the path to update statistics

```python
import math

class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}       # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior       # prior probability from the policy network

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct=1.5):
        """PUCT formula: Q + U (exploration bonus)."""
        if self.visit_count == 0:
            return float('inf')
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) \
            / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self):
        """Select the child with the highest UCB score."""
        return max(self.children.items(),
                   key=lambda item: item[1].ucb_score())

    def expand(self, action_priors):
        """Expand children according to the policy network output."""
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def backpropagate(self, value):
        """Backpropagate the value (switch perspective along the path)."""
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            # The parent is the opponent's turn, so negate the value.
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search."""

    def __init__(self, model, c_puct=1.5, num_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, env):
        """Run MCTS from the current state and return visit counts per action."""
        root = MCTSNode()

        # The number of simulations controls how much search we do.
        for _ in range(self.num_simulations):
            node = root
            sim_env = env.copy()

            # 1. Select: move down the tree until reaching a leaf.
            while node.children:
                action, node = node.select_child()
                sim_env.play(*action)

            # 2. Evaluate: predict policy and value with the network.
            board_tensor = torch.tensor(sim_env.board, dtype=torch.float32).unsqueeze(0)
            player_tensor = torch.tensor([sim_env.current_player], dtype=torch.float32)

            with torch.no_grad():
                policy_logits, value = self.model(board_tensor, player_tensor)

            # 3. Expand: expand legal actions only.
            legal_moves = sim_env.get_legal_moves()
            if legal_moves:
                # Set illegal logits to -inf.
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
                # No legal move: pass
                pass

            # 4. Backpropagate
            node.backpropagate(value.item())

        # Compute the final policy from visit counts.
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts
```

Pay attention to `-value` in `backpropagate`. This is the key to zero-sum games:
what benefits Black harms White by the same amount. So as the perspective alternates each ply, the value must flip sign.

## 7.5.5 Self-Play Training

AlphaGo's most revolutionary idea is **self-play**: let the AI play against itself, and use the game outcomes to train itself.
If it wins, it reinforces the moves it played; if it loses, it weakens them.
This matches the spirit of policy gradients, except that the samples come not from human game records, but from the agent's own games.

```python
def self_play_game(model, mcts, temperature=1.0):
    """Play one self-play game with MCTS; return (states, policies, winner)."""
    env = MiniGo()
    states, players_list, policies = [], [], []
    max_moves = BOARD_SIZE * BOARD_SIZE * 2  # avoid infinite games

    for _ in range(max_moves):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            env.pass_turn()
            if env.is_game_over():
                break
            continue

        # MCTS search
        visit_counts = mcts.run(env)
        total_visits = sum(visit_counts.values())

        # Policy distribution (normalized visit counts)
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for (r, c), visits in visit_counts.items():
            policy[r * BOARD_SIZE + c] = visits / total_visits

        # Add temperature noise early in training to encourage exploration
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

    # Determine the winner
    winner = env.get_winner()

    # Convert the outcome into +1/-1 value labels
    values = []
    for player in players_list:
        values.append(1.0 if player == winner else -1.0)

    return states, players_list, policies, values


def train_alphago(num_iterations=20, games_per_iter=10, num_epochs=5):
    """Main training loop for AlphaGo."""
    model = AlphaGoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mcts = MCTS(model, num_simulations=50)  # fewer simulations during training

    replay_buffer = []  # (state, player, policy, value)

    for iteration in range(num_iterations):
        # Phase 1: self-play to collect data
        new_data = []
        for _ in range(games_per_iter):
            # higher temperature early, lower temperature later
            temp = 1.0 if iteration < num_iterations // 2 else 0.5
            states, players, policies, values = self_play_game(model, mcts, temp)
            for s, p, pi, v in zip(states, players, policies, values):
                new_data.append((s, p, pi, v))

        replay_buffer.extend(new_data)
        # Keep only the most recent 5000 samples
        if len(replay_buffer) > 5000:
            replay_buffer = replay_buffer[-5000:]

        # Phase 2: train the network on collected data
        model.train()
        for epoch in range(num_epochs):
            # sample a mini-batch
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

            # forward pass
            policy_logits, pred_values = model(boards, players)

            # policy loss: cross-entropy (MCTS policy as a supervision signal)
            policy_loss = F.cross_entropy(policy_logits, target_policies)

            # value loss: mean squared error
            value_loss = F.mse_loss(pred_values, target_values)

            # total loss
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

The training loop has only two phases:

1. **Self-play**: use the current model plus MCTS to play games, collecting triplets (position, MCTS policy, outcome)
2. **Network training**: train the policy network to imitate MCTS search results, and train the value network to predict the final outcome

There is a subtle but important difference from the [policy gradient](../chapter08_policy_gradient/reinforce) method in Chapter 6:
AlphaGo's policy network does not learn directly from game returns (as in REINFORCE).
Instead, it **learns to imitate the search policy produced by MCTS**.
Because MCTS aggregates many simulations, its policy signal is much more reliable than a single sampled trajectory.
You can view this as a naturally provided low-variance baseline.

## 7.5.6 Human vs. AI Play

```python
def human_vs_ai(model, mcts, human_color=BLACK):
    """Interactive match: human vs AI."""
    env = MiniGo()
    print(f"You are {'Black (X)' if human_color == BLACK else 'White (O)'}")
    print("Input format: row col (e.g. '2 3'); type 'pass' to pass.\n")

    while not env.is_game_over():
        print(env_to_string(env.board))

        if env.current_player == human_color:
            # Human turn
            legal = env.get_legal_moves()
            print(f"Legal moves: {legal}")
            cmd = input("Your move: ").strip()
            if cmd == 'pass':
                env.pass_turn()
            else:
                r, c = map(int, cmd.split())
                if not env.play(r, c):
                    print("Illegal move. Try again.")
                    continue
        else:
            # AI turn
            visit_counts = mcts.run(env)
            if visit_counts:
                best_action = max(visit_counts, key=visit_counts.get)
                print(f"AI plays: {best_action} "
                      f"(visits: {visit_counts[best_action]})")
                env.play(*best_action)
            else:
                print("AI: pass")
                env.pass_turn()
        print()

    # Game over
    score = env.compute_score()
    print(env_to_string(env.board))
    print(f"Black: {score[BLACK]:.1f} | White: {score[WHITE]:.1f}")
    winner = "Black" if score[BLACK] > score[WHITE] else "White"
    print(f"{winner} wins!")


def env_to_string(board):
    symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
    lines = ["   " + " ".join(str(i) for i in range(BOARD_SIZE))]
    for r in range(BOARD_SIZE):
        line = f"{r}: " + " ".join(symbols[board[r, c]] for c in range(BOARD_SIZE))
        lines.append(line)
    return "\n".join(lines)
```

For the interactive portion above, the prompts are kept in Chinese to match the original minimal demo.
When integrating it into your own project, it is straightforward to translate the CLI messages.

## 7.5.7 AlphaGo and the Concepts in This Chapter

Let's map each AlphaGo component back to what we have learned in this chapter:

| AlphaGo Component    | Related Concept                             | Where It Appears                                                  |
| -------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| Policy network       | Actor, outputs action probabilities         | [Policy Gradient Theorem](../chapter08_policy_gradient/reinforce) |
| Value network        | Critic, evaluates a position                | [Actor-Critic Architecture](./actor-critic)                       |
| MCTS policy targets  | "Reliable policy signal" to reduce variance | [Baselines](../chapter08_policy_gradient/pg-improvements)         |
| Self-play            | Online sampling + policy improvement        | The sampling spirit of REINFORCE                                  |
| $-v$ backpropagation | Symmetry in zero-sum games                  | Sign flip in [advantage functions](./advantage-function)          |

You can see the pattern: the core of AlphaGo is Actor-Critic plus MCTS search.
The policy network (Actor) provides search priors; the value network (Critic) evaluates leaf nodes; MCTS integrates both into a stronger decision rule.
This "Actor gives priors + Critic gives evaluations + search does the integration" template was later generalized by AlphaZero
to chess and shogi, and it influenced many subsequent RL algorithm designs.

## 7.5.8 Open-Source Projects and Datasets

The code above is intentionally minimal and written for understanding the ideas.
If you want a version that is truly practical, here are well-known open-source projects and datasets.

### Recommended Open-Source Projects

| Project                                                               | Notes                                                               | Best For                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | PyTorch, game-agnostic framework; includes Othello/Gomoku/TicTacToe | Best first choice; simplest code; runs on a laptop  |
| [michaelnny/alpha_zero](https://github.com/michaelnny/alpha_zero)     | PyTorch; 9x9 Go + 15x15 Gomoku                                      | Running Go on a real 9x9 board                      |
| [KataGo](https://github.com/lightvector/KataGo)                       | C++/Python; supports 7x7 to 19x19; has pretrained models            | Experiments or matches that need pretrained weights |
| [Leela Zero](https://github.com/leela-zero/leela-zero)                | C++; faithful reproduction of AlphaGo Zero                          | Studying the original AlphaGo Zero algorithm        |
| [MiniZero](https://github.com/rlglab/minizero)                        | C++/Python; supports AlphaZero/MuZero/Gumbel variants               | Comparing different MCTS variants                   |

The most recommended path is:
start with Othello in [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) (built-in environment, minimal setup),
understand the full pipeline end-to-end, then switch to the Go setting.

### Available Datasets

| Dataset                                                                          | Size                | Notes                                                                         |
| -------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------- |
| [JGDB](https://pjreddie.com/projects/jgdb/)                                      | 535k games, 194 MB  | Best choice; train/val/test split provided; public domain; by the YOLO author |
| [featurecat/go-dataset](https://github.com/featurecat/go-dataset)                | 21.10 million games | Largest scale; from Fox Go Server; covers 18k to 9p                           |
| [CWI Japanese Professional Games](https://homepages.cwi.nl/~aeb/go/games/games/) | 88k games, 45 MB    | Professional games, carefully curated                                         |
| [KGS Archives](https://www.gokgs.com/)                                           | Millions of games   | KGS server archives, mixed ranks                                              |

A standard supervised pretraining workflow with JGDB is:
download SGF files, parse them into (board, move) pairs, train the policy network to imitate human moves,
then switch to self-play for reinforcement. This matches the first stage described in the AlphaGo paper.

## Further Exploration

1. **Increase the board size**: run 9x9 Go with [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) and observe how training time and playing strength change.
2. **Make it AlphaZero-style**: remove human-game pretraining and start purely from self-play. On 6x6, how many iterations does it take to converge?
3. **MCTS simulation count**: compare 10, 50, and 200 simulations. How does it affect playing strength? Is "more simulations" always better?
4. **Supervised pretraining with JGDB**: download the [JGDB dataset](https://pjreddie.com/projects/jgdb/), parse SGF records, train the policy network to imitate human moves, then continue with self-play reinforcement: this forms a complete AlphaGo-style pipeline.

## References

[^1]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489. [DOI](https://doi.org/10.1038/nature16961)

[^2]: Silver, D., et al. (2017). Mastering the game of Go without human knowledge. _Nature_, 550(7676), 354-359. [DOI](https://doi.org/10.1038/nature24270)

[^3]: Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. _Science_, 362(6419), 1140-1144. [DOI](https://doi.org/10.1126/science.aar6404)

## Section Summary

This lab connects the runnable experiment to the main idea of the section. Use the reported metrics together with replay or task-level evaluation, and keep conclusions within the conditions that were actually tested.
