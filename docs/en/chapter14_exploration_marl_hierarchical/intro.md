---
title: 'Chapter 12: Exploration, Multi-Agent RL & Hierarchical RL'
---

# Chapter 12 · Exploration, Multi-Agent RL, and Hierarchical RL

> [Chapter 9, Continuous Control](../chapter11_continuous_control/intro) solved the sample-efficiency and stability problems of a single agent in continuous action spaces, and [Chapter 10, Offline RL](../chapter12_offline_rl/intro) solved the difficulty of "only historical data, no interaction." This chapter deals with three situations that were deliberately set aside until now: (1) the environment's reward is extremely sparse, so ε-greedy never stumbles onto a positive sample — this calls for exploration driven by **intrinsic motivation**; (2) multiple agents are learning in the environment at the same time, and non-stationarity breaks the MDP assumption — this calls for the **CTDE** paradigm and a centralized critic; (3) the task horizon is extremely long, and a single-layer policy cannot bridge the gap between subgoals — this calls for **hierarchical RL** to decompose long-horizon decisions into a sequence of options. All three point to the same engineering problem: once the classic assumptions of RL break down, how does structured inductive bias recover sample efficiency?

## 14.1 The Fundamental Tension Between Exploration and Exploitation

[Chapter 3's treatment of exploration and exploitation](../chapter03_mdp/bandit) already introduced the exploration-exploitation trade-off in the stateless setting: the expected return of each arm is unknown, and the agent must split its budget between "pulling the currently best arm" (exploitation) and "pulling an uncertain arm" (exploration). The UCB method introduced there uses an upper confidence bound $U_t(a) = \hat{\mu}_t(a) + c\sqrt{\ln t / N_t(a)}$, encoding uncertainty directly into the action value.

Deep RL amplifies this problem. In Atari's _Montezuma's Revenge_ or _Pitfall_, the agent has to execute dozens of meaningful actions before reaching the first reward from the initial state (jumping over traps, picking up a key, opening a door), and the probability that random $\epsilon$-greedy exploration stumbles onto the first reward is around $10^{-18}$. DQN's score on these **hard-exploration** games stayed at zero for a long time.

The core of the problem is the sparsity of the reward signal. Suppose the agent starts from $s_0$ and needs at least $H^\star$ steps to reach the rewarding state $s^\star$. Any update that depends only on the environment reward $r_t$ has to wait for the first successful trajectory to appear, and the density of successful trajectories decays exponentially with $H^\star$. **Intrinsic reward** sidesteps this bottleneck: the agent generates its own auxiliary reward $\tilde{r}_t$ that encourages it to visit "novel" or "unpredictable" states.

Formally, the total reward becomes

$$r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t$$

where $r^{\text{ext}}$ is the extrinsic reward given by the environment, $r^{\text{int}}$ is the intrinsic reward the agent computes for itself, and $\beta$ is a trade-off coefficient. The key constraint is that $r^{\text{int}}$ must satisfy two properties:

1. **Computable**: it depends only on observed data, with no need for external supervision
2. **Exhaustible**: once a state has been visited enough times, its intrinsic reward should decay to zero, so the agent doesn't get stuck "farming" a local pocket of reward

The next two subsections cover the two mainstream approaches: ICM, based on prediction error, and RND, based on random network distillation.

## 14.2 Intrinsic Curiosity (ICM) and Random Network Distillation (RND)

### Using forward prediction error as curiosity

The core idea of the Intrinsic Curiosity Module (Pathak et al., 2017): if the agent cannot predict its own next state, that region must be "unfamiliar" and worth exploring. The larger the prediction error, the higher the intrinsic reward.

Predicting directly in pixel space fails — the next frame has too much pixel-level detail, and the prediction error ends up dominated by irrelevant high-frequency noise. ICM first uses an **inverse model** $g_\phi$ to learn a feature space $\Phi(s)$: given $(s_t, s_{t+1})$ as input, it predicts the action $a_t$. This feature only retains "the part the action can affect," filtering out irrelevant variation like background flicker or camera shake.

Then it trains a **forward model** $f_\psi$:

$$\hat{\Phi}(s_{t+1}) = f_\psi(\Phi(s_t), a_t)$$

The intrinsic reward is defined as the forward prediction error:

$$r^{\text{int}}_t = \tfrac{1}{2}\|\Phi(s_{t+1}) - \hat{\Phi}(s_{t+1})\|^2$$

The overall loss:

$$\mathcal{L} = \mathcal{L}_{\text{policy}}(\theta) + \lambda_{\text{inv}}\,\mathcal{L}_{\text{inv}}(\phi) + \lambda_{\text{fwd}}\,\mathcal{L}_{\text{fwd}}(\psi)$$

```python
class ICM(nn.Module):
    def __init__(self, feat_dim=256, action_dim=6):
        self.encoder = CNNtoMLP(out=feat_dim)              # Φ(s)
        self.inverse = nn.Linear(feat_dim * 2, action_dim) # g_φ
        self.forward_net = MLP(feat_dim + action_dim, feat_dim)

    def intrinsic_reward(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return 0.5 * (phi_next - phi_pred).pow(2).sum(-1)

    def forward_loss(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return F.mse_loss(phi_pred, phi_next.detach()) + \
               F.cross_entropy(self.inverse(torch.cat([phi, phi_next], -1)), a)
```

In tasks like _Super Mario Bros_ that combine continuous control with visual input, ICM lets the agent traverse the whole map with no extrinsic reward at all. Its weakness is the **noisy-TV problem**: if the environment contains an unpredictable source of randomness (say, a TV in the corner of the screen randomly showing static), the forward model can never learn it, the intrinsic reward stays permanently high, and the agent gets stuck standing in front of the TV.

### Using a random network as an "unlearnable" prediction target

Random Network Distillation (Burda et al., 2018) sidesteps the noisy-TV problem with a cleverer mechanism. Fix a randomly initialized target network $\hat{f}(s)$ that is **never updated** (its parameters are frozen at their random initialization), and train a predictor network $f_\psi(s)$ to fit it:

$$\mathcal{L}_{\text{RND}}(\psi) = \mathbb{E}_s\bigl[\|f_\psi(s) - \hat{f}(s)\|^2\bigr]$$

$$r^{\text{int}}(s) = \|f_\psi(s) - \hat{f}(s)\|^2$$

The mechanism is simple: the predictor has already learned states it has visited, so the prediction error there is small; for states it hasn't seen, the prediction error is large. The random target network carries no semantic meaning of its own — its only job is to provide a **fixed but inexhaustible** learning signal.

RND's advantages:

- **No inverse model needed**, cutting the compute roughly in half
- **Action-independent**, so it can be bolted onto any model-free algorithm (PPO, A2C)
- **Naturally robust to the noisy-TV problem**: the random target has bounded complexity, so the prediction error has an upper bound and can't be driven up without limit

```python
class RND(nn.Module):
    def __init__(self, obs_shape, feat_dim=512):
        # Target network: frozen, never updated
        self.target = CNN(obs_shape, feat_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        # Predictor network: trained
        self.predictor = CNN(obs_shape, feat_dim)

    def intrinsic_reward(self, s):
        with torch.no_grad():
            target = self.target(s)
        pred = self.predictor(s)
        return (pred - target).pow(2).sum(-1)  # one scalar per state
```

In large-scale experiments, Burda et al. found that using RND intrinsic reward alone (with no extrinsic reward at all), a PPO agent could explore into complex behavior across several Atari games, and it broke past a score of zero on the extrinsically sparse _Montezuma's Revenge_ for the first time.

### ICM vs. RND

| Dimension                   | ICM                               | RND                    |
| --------------------------- | --------------------------------- | ---------------------- |
| Depends on actions          | Yes (forward model needs $a$)     | No                     |
| Submodules to train         | Encoder + inverse + forward       | Predictor only         |
| Robustness to noisy TV      | Weak                              | Strong                 |
| Compute overhead            | High                              | Medium                 |
| Representative applications | Visual exploration (Mario, DMLab) | Atari hard-exploration |

## 14.3 NGU and Agent57

ICM and RND each solve part of the problem, but they share a blind spot: **the absence of episodic memory**. A state can be novel within a single episode (short-term) while having already been visited ten million times across episodes (long-term). Prediction error fit by a neural network alone cannot distinguish these two notions of novelty. Never Give Up (Badia et al., 2020) and its successor Agent57 (Badia et al., 2020) model exploration at both timescales simultaneously, becoming the **first algorithm to exceed human-level performance** on the full suite of 57 Atari games.

### Dual-timescale intrinsic reward

NGU's intrinsic reward is composed of two parts multiplied together:

$$r^{\text{int}}_t(s) = r^{\text{episodic}}_t(s) \cdot r^{\text{life-long}}_t(s)$$

**Short-term (episodic) component** $r^{\text{episodic}}$: maintain a fixed-capacity table of controllable-state features, recording the state features visited within the current episode. If a new state is far from every state already in the table (a large kNN distance), its novelty is high; states visited frequently have their novelty decay:

$$r^{\text{episodic}}_t = \frac{1}{\sqrt{k} + c \sum_{i=1}^{k} \frac{1}{\sqrt{N(s_i)}}}$$

where $N(s_i)$ is the visit count of state $s_i$ and $c$ is a decay constant. This is a simplified kNN-style formula.

**Long-term (life-long) component** $r^{\text{life-long}}$: this is essentially RND. It operates across episodes, capturing states that "this episode hasn't visited, but other episodes have." Multiplying the two together guarantees that a state only gets a high intrinsic reward if it is both "unvisited this episode" and "unvisited globally."

### Retrace and distributed actors

NGU uses R2D2's distributed architecture (many actors sampling in parallel, plus an LSTM to handle partial observability), and estimates off-policy Q-values with Retrace($\lambda$) to carry the intrinsic reward signal stably across long horizons. The whole system is extremely expensive to train (billions of frames, hundreds of TPUs), but it was the first proof that Atari hard-exploration games can be cracked by end-to-end RL.

### Agent57 and adaptive exploration-exploitation switching

NGU still leaves one problem unresolved: the intrinsic reward weight $\beta$ is fixed. On simple games (_Pong_, _Space Invaders_), too high a $\beta$ sends the agent into frantic exploration instead of exploiting the known best policy; on hard-exploration games, too low a $\beta$ leaves it under-exploring. **Agent57** introduces an **adaptive policy scheduler**:

- Maintain a family of policies $\pi_i$, each with different exploration parameters $(\beta_i, \gamma_i, c_i)$, spread across the range from "pure exploitation" to "pure exploration"
- Use a meta-controller to estimate each policy's relative return online, and preferentially sample the policies that are performing well
- During training, all the policies share a replay buffer and a Q-network

This removes the need to hand-tune $\beta$ for every game. Agent57 is the **first algorithm to exceed human-level performance on every game** in the DeepMind Atari 57 suite, and it's widely regarded as the capstone of the classic Atari RL research program.

## Section Summary

Exploration driven by intrinsic motivation is the fundamental solution to the hard-exploration problem. From ICM's "prediction error as intrinsic reward," to RND's "random network distillation," to NGU's fusion of short-term and long-term uncertainty, to Agent57's adaptive exploration-exploitation balance — this line of work took DQN from a score of 0 to superhuman performance on hard-exploration games like Montezuma's Revenge.

The next section, [14.2 Multi-Agent RL: CTDE, MADDPG, MAPPO](./marl), turns to another challenge — when multiple agents learn simultaneously in the environment, non-stationarity breaks the MDP assumption.
