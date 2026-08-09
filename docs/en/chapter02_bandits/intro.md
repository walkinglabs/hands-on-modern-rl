# Chapter 2 · Multi-Armed Bandits and Exploration-Exploitation

> [Chapter 1](../chapter01_cartpole/principles) defined the RL problem with a single formula: find the optimal policy $\pi^*$ that maximizes cumulative return. One detail got glossed over — **how does the agent know which action is good?** This is the first fundamental difficulty that sets RL apart from supervised learning: **exploration vs. exploitation**. This chapter isolates that difficulty using the simplest possible RL problem — the multi-armed bandit.

## 2.1 The Multi-Armed Bandit Problem

The multi-armed bandit (MAB) is RL's "fruit fly" — the most stripped-down sequential decision problem there is, yet it keeps the full essence of the exploration-exploitation tension.

### Problem Definition

Picture a slot machine with $K$ arms. Pulling arm $a \in \{1, 2, \ldots, K\}$ returns a reward drawn from an unknown distribution. Your goal is to maximize cumulative reward over $T$ pulls.

Formally: each arm corresponds to an unknown reward distribution $R_a$, with mean $\mu_a = \mathbb{E}[R_a]$. At each round $t$, the agent picks $A_t \in \{1, \ldots, K\}$ and observes reward $R_t \sim R_{A_t}$. The objective: $\max \sum_{t=1}^T R_t$.

Notice the simplifications at work here:

- **No state transitions**: the choice made in one round doesn't change the "state of the environment" — the reward distribution depends only on which arm is pulled. This strips away the state dimension of an MDP entirely, leaving pure focus on action selection.
- **Rewards are immediate**: a pull gets an instant result, no delayed reward.
- **No discounting**: every round carries equal weight.

Even with all this stripped away, **the optimal action $a^* = \arg\max_a \mu_a$ is unknown** — the agent has to estimate every $\mu_a$ by trying arms out. This is where exploration-exploitation comes from.

### Regret

Because $\mu_a$ is unknown, the agent can't pick $a^*$ every round. We measure how good a policy is using **regret**:

$$\text{Regret}(T) = T \cdot \mu^* - \mathbb{E}\left[\sum_{t=1}^T R_t\right] = \sum_{t=1}^T \mathbb{E}[\mu^* - \mu_{A_t}]$$

where $\mu^* = \max_a \mu_a$. Regret is the expected gap between what the agent actually earned and what it would have earned by picking the optimal arm every single round.

The **regret bound** is the fundamental metric for evaluating an algorithm. A good policy's regret should grow **sublinearly** in $T$, i.e. $\text{Regret}(T) = o(T)$ — meaning that as $T \to \infty$, the agent keeps getting closer to optimal, and its average per-round loss goes to zero.

| Growth rate        | Meaning     | Assessment                                 |
| ------------------ | ----------- | ------------------------------------------ |
| $\Theta(T)$        | Linear      | The agent learned nothing — pure random    |
| $\Theta(\sqrt{T})$ | Sublinear   | Standard good policy (UCB, Thompson)       |
| $\Theta(\log T)$   | Logarithmic | Theoretical lower bound (Lai-Robbins 1985) |

## 2.2 ε-Greedy and Decay Schedules

The simplest possible policy: exploit the known best option most of the time, and occasionally explore at random.

### The ε-Greedy Algorithm

```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q = np.zeros(n_arms)         # estimated mean reward per arm
        self.n = np.zeros(n_arms)         # number of times each arm has been pulled

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)   # explore
        return np.argmax(self.q)                     # exploit

    def update(self, arm, reward):
        self.n[arm] += 1
        # Incremental mean update: avoids storing the full history
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
```

The core idea:

- **Exploit (probability $1-\epsilon$)**: pick the arm with the highest current estimate
- **Explore (probability $\epsilon$)**: pick uniformly at random, including arms currently believed to be worst

Why not exploit purely? Because the initial estimate $q_a = 0$ can be misleading. If arm 1's first pull returns 0, a pure-exploitation policy will conclude forever that it's bad — but its true mean might actually be 0.7, and that first pull was just unlucky.

A fixed $\epsilon$ has the virtue of simplicity and stability: no matter how confident the current estimates are, the agent always keeps a small margin for trying other arms. That guards against a run of early bad luck permanently ruling out a genuinely good arm.
The cost is just as direct: even once the agent basically knows which arm is best, it keeps exploring at the same fixed rate, wasting a fraction of its choices over the long run.

### Decay Schedules

A fixed $\epsilon$ always wastes an $\epsilon$-fraction of pulls on exploration. The smarter move: **explore heavily early on, exploit heavily later**.

```python
class EpsilonDecaying:
    def __init__(self, n_arms, epsilon_start=1.0, epsilon_end=0.01, decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = decay
        # ... rest same as above

    def select(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(self.n_arms)
        else:
            arm = np.argmax(self.q)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay)
        return arm
```

A fixed $\epsilon$ is simple and reliable, but it keeps wasting exploration opportunities indefinitely. In practice it's far more common to let $\epsilon$ decay over time, moving the agent gradually from "try a lot of things" to "trust what's already been learned." This idea shows up widely in deep RL too — Atari DQN's $\epsilon$ starts large and decays down to a small final value.

::: details Additional detail: theoretical properties of ε-greedy

A fixed ε-greedy policy keeps a permanent margin of random exploration, so even once it's clear which action is best, the policy still picks other actions with probability ε. That protects against early misjudgment, but it comes at a persistent long-run cost — so fixed-ε cumulative regret typically grows linearly.

If ε is instead decayed over time — starting large and shrinking toward a small value — the agent explores heavily early and exploits heavily later. A well-chosen decay schedule can push long-run regret down to sublinear, and under stricter settings can even approach a logarithmic regret bound.

PAC sample complexity asks a different question altogether. Instead of asking how much total loss accumulates over $T$ rounds, it asks: "how many pulls are needed to find, with probability at least $1-\delta$, an action within $\epsilon$ of optimal?" This framing is mainly used for theoretical analysis — for this course it's enough to know it measures "how many samples are needed to learn a good-enough policy."

:::

## 2.3 More Targeted Exploration

ε-greedy solves the problem of "can't just trust the current best guess," but its exploration is crude: whenever it enters the explore branch, it picks uniformly among _all_ actions — including ones that are already clearly bad.

Smarter exploration asks a sharper question: **which actions are still worth trying?**

### UCB: Adding a Bonus for Uncertainty

UCB (Upper Confidence Bound) works by adding an "uncertainty bonus" to each arm's estimated reward:

$$A_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

Here $\hat{\mu}_a$ is arm $a$'s current average reward, and $N_a$ is the number of times it has been tried so far. The fewer times an arm has been tried, the smaller $N_a$ is and the larger its uncertainty bonus; the more it has been tried, the smaller the bonus, and the choice leans more on the true estimate.

The intuition behind UCB is direct: if an action is currently performing well, keep exploiting it; if an action hasn't been tried enough to rule out being bad, give it some extra chances. Exploration is no longer spread evenly across all actions — it concentrates on actions that are "possibly good but still uncertain."

::: details Additional detail: UCB's regret bound

UCB1's regret can be shown to reach a logarithmic bound. A common form is:

$$\mathbb{E}[\text{Regret}(T)] \leq 8 \sum_{a: \mu_a < \mu^*} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \sum_{a: \mu_a < \mu^*} \Delta_a$$

where $\Delta_a = \mu^* - \mu_a$ is the gap between a suboptimal arm and the optimal one. On a first pass, there's no need to memorize this expression — just understand the mechanism it expresses: UCB gradually cuts back on trying clearly suboptimal actions, and concentrates its exploration on actions that are genuinely hard to distinguish from the best one.

:::

### Thompson Sampling: Choosing by the Odds of Being Optimal

Thompson sampling takes a different angle. Instead of adding an explicit uncertainty bonus, it maintains a posterior distribution for each arm — representing "what the true mean reward of this arm is likely to be." Each round, it draws one sample from every arm's posterior, and picks the arm with the highest sampled value.

If an arm has already been tried many times, its posterior is narrow, and its sampled value will typically land close to its current estimate. If an arm has barely been tried, its posterior is wide, giving it a real chance of sampling a high value and earning an exploration opportunity. Exploration naturally flows toward whichever actions still have a real chance of being optimal.

In a 0/1 reward setting, each arm's posterior can be represented with a Beta distribution:

$$\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$$

Pull arm $a$ and get reward 1: set $\alpha_a \leftarrow \alpha_a + 1$. Get reward 0: set $\beta_a \leftarrow \beta_a + 1$. This update rule is lightweight, which is why Thompson sampling is popular in recommendation systems, ad serving, and A/B testing infrastructure.

::: details Additional detail: the theoretical view of Thompson sampling

Theoretical analysis of Thompson sampling commonly uses Bayesian regret: assume the problem itself is drawn from a prior distribution, then compute the policy's expected loss under that prior. Under common stochastic bandit settings, it achieves logarithmic regret of the same order as UCB.

This level of detail isn't the main thread of this chapter. What matters is the core idea: Thompson sampling converts "how likely is this action to be optimal" directly into a selection probability, which is why it wastes less exploration than ε-greedy.

:::

### Contextual Bandits and RLHF

The ordinary multi-armed bandit assumes each arm's reward distribution is fixed. In real systems, though, whether an action is good often depends on the current input.

In a recommendation system, the same article appeals differently to different users. In an ad system, the same ad gets different click-through rates from different audiences. For a large language model, the quality of the same response has to be judged relative to the specific question being answered.

This gives us the **contextual bandit**:

- Each round, first observe a context $x_t$
- Then choose an action $A_t$
- Receive reward $R_t$
- The goal is to learn a policy $\pi(a \mid x)$ that makes the action depend on the current context

From this angle, RLHF can initially be understood as a contextual bandit problem: the prompt is the context, the model's generated response is the action, and the reward model's score is the reward. This abstraction doesn't yet capture full token-level state transitions, but it already explains a key fact: the model can't just learn "which response is more common overall" — it has to learn "given this prompt, which kind of response is more appropriate."

Once we move into MDPs later on, we'll extend this picture further: when a response is generated token by token, the action is no longer a single one-shot choice — each token changes the subsequent state and the actions available afterward. At that point, the bandit problem expands into a genuine sequential decision problem.

## Chapter Summary

The multi-armed bandit (MAB) is RL in its most stripped-down form — no state transitions, immediate rewards — yet it fully preserves the exploration-exploitation tension. ε-greedy keeps a fixed-probability margin for trying things out, making it the natural starting point for understanding exploration. UCB allocates trials toward actions with higher uncertainty. Thompson sampling uses posterior sampling to turn "might be optimal" directly into a selection probability.

Multi-armed bandits have no state transitions, and rewards appear instantly. The next chapter, [Markov Decision Processes](../chapter03_mdp/mdp), adds state transitions and long-run return, taking us into genuine sequential decision problems.

## Further Reading

- Sutton & Barto, _Reinforcement Learning: An Introduction_, Chapter 2
- [Auer et al. 2002, "Finite-time Analysis of the Multiarmed Bandit Problem"](https://link.springer.com/article/10.1023/A:1013689704352)
- [Russo et al. 2018, "A Tutorial on Thompson Sampling"](https://arxiv.org/abs/1707.02038)
- [Lattimore & Szepesvári, _Bandit Algorithms_](https://banditalgs.com/)
