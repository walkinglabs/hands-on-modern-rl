---
title: 7.1 The Advantage Function
---

# 7.1 Advantage Function

Chapter 5 followed the value-based route: learn $Q(s,a)$ and pick the action with the highest score (review: [Q-Values and the Greedy Policy](../chapter03_mdp/value-q)). Chapter 6 followed the policy-based route and optimized $J(\theta)$ directly (review: [Policy, Value, and Return](../chapter03_mdp/policy-value)). The first route uses explicit action scores; the second supports stochastic and continuous policies but produces noisy gradient estimates.

At the end of Chapter 6, we found a key clue: subtracting a baseline reduces variance (review: [Policy Gradient Improvements](../chapter08_policy_gradient/pg-improvements)), and the best baseline is $V(s)$ (review: [State-Value Function](../chapter03_mdp/value-bellman)). But $V(s)$ itself must be learned, which means we need a dedicated network to estimate it. This network is the **Critic**.

In this chapter, we will stitch the two lines together: train a Critic using the methods from Line 1 to evaluate how good an action is, and train an Actor using the methods from Line 2 to choose actions. This is the **Actor-Critic architecture**.

::: tip Prerequisites (Quick Review)
This chapter is a synthesis of everything we have built so far. The following concepts will appear repeatedly:

- [State-value $V(s)$ and the Bellman equation](../chapter03_mdp/value-bellman): the theoretical foundation of the Critic. $V^\pi(s)$ measures "starting from state $s$, how many points do we get on average?"
- [Action-value $Q(s,a)$](../chapter03_mdp/value-q): the difference between $Q$ and $V$ is the advantage function
- [DP / MC / TD: three ways to estimate values](../chapter03_mdp/dp-mc-td): three concrete strategies for training the Critic
- [TD Error $\delta = r + \gamma V(s') - V(s)$](../chapter03_mdp/dp-mc-td): the Critic's core training signal
- [Policy objective $J(\theta)$ and policy gradients](../chapter03_mdp/policy-value): the Actor's optimization target
- [REINFORCE and baselines](../chapter08_policy_gradient/pg-improvements): why we need $V(s)$ as a baseline
  :::

## Chapter Roadmap

| Section                                                      | Core Question                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| [Advantage Function](./advantage-function)                   | What is the advantage function $A(s,a)$? Why is it better than using $G_t$ directly? |
| [Training the Critic](./critic-training)                     | How do we train a Critic to estimate $V(s)$? Concrete implementations of DP/MC/TD    |
| [Actor-Critic Architecture](./actor-critic)                  | How do the Actor and Critic collaborate? How does TD Error replace $G_t$?            |
| [Frontier-Scale Applications of Actor-Critic](./ac-frontier) | AlphaStar, SAC robots, Isaac Lab: how AC lands in industrial-scale practice          |
| [Hands-on: Pendulum Balancing](./pendulum)                   | How does Actor-Critic handle continuous action spaces?                               |
| [Hands-on: BipedalWalker](./bipedalwalker)                   | Can Actor-Critic learn complex continuous control?                                   |

Let’s begin with the advantage function. It is the bridge that connects the Actor and the Critic: [Advantage Function](./advantage-function).

At the end of Chapter 6, we found that subtracting a baseline $V(s)$ reduces the variance of policy gradients without changing the gradient direction. This section deepens that insight and introduces the advantage function -- the bridge connecting the Actor and the Critic.

::: tip Prerequisites

- [REINFORCE policy gradient](../chapter08_policy_gradient/reinforce): $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$ -- where to insert the baseline
- [State value $V(s)$](../chapter03_mdp/value-bellman): what makes a good baseline
- [Action value $Q(s,a)$](../chapter03_mdp/value-q): the advantage is defined as the difference between $Q$ and $V$
- [TD error](../chapter03_mdp/dp-mc-td): $\delta = r + \gamma V(s') - V(s)$ -- a practical estimator of the advantage
  :::

## From Baseline to Advantage Function

Recall the REINFORCE [policy gradient](../chapter08_policy_gradient/reinforce):

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$$

$G_t$ is the total discounted return from the current step to the end of the episode (review: [discounted return](../chapter03_mdp/mdp)). The problem is that $G_t$ fluctuates wildly -- under the same policy, from the same state, two rollouts can yield completely different $G_t$ values.

After subtracting the baseline $V(s)$:

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot (G_t - V(s))$$

The quantity in parentheses, $G_t - V(s)$, is already an estimate of the **advantage function**. The formal definition is:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) \tag{7.1}$$

| Symbol       | Meaning                                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $A^\pi(s,a)$ | Advantage function: how much better taking action $a$ in state $s$ is compared to "average."                                                                 |
| $Q^\pi(s,a)$ | [Action-value function](../chapter03_mdp/value-q): expected discounted return starting from state $s$, taking action $a$ first, then following policy $\pi$. |
| $V^\pi(s)$   | [State-value function](../chapter03_mdp/value-bellman): expected discounted return starting from state $s$ and following policy $\pi$.                       |
| $\pi$        | The current policy, determining the probability of each action in each state.                                                                                |

Their difference captures exactly "how many extra points were earned because action $a$ was chosen."

In words, the advantage says:

**How much better is this action than what we would typically get in this state?**

- $A > 0$: the action is better than expected; we should choose it more often
- $A < 0$: the action is worse than expected; we should choose it less often
- $A \approx 0$: the action is about as good as expected

A chess analogy: $V(s)$ is "this position has a 60% win rate overall," while $Q(s, \text{play rook})$ is "after playing the rook move, the win rate becomes 75%." The advantage is $A = 75\% - 60\% = 15\%$, meaning the rook move is 15 percentage points better than the average outcome for the position -- a strong choice.

Let us work through a concrete 3-step episode to see how the advantage is computed. Suppose the discount factor is $\gamma = 0.9$, and a sampled trajectory yields:

$$s_0 \xrightarrow{r=+2} s_1 \xrightarrow{r=+3} s_2 \xrightarrow{r=+1} s_3\ (\text{terminal})$$

Computing the discounted return $G_t$ from each time step:

$$G_0 = r_1 + \gamma r_2 + \gamma^2 r_3 = 2 + 0.9 \times 3 + 0.9^2 \times 1 = 2 + 2.7 + 0.81 = 5.51$$

$$G_1 = r_2 + \gamma r_3 = 3 + 0.9 \times 1 = 3.9$$

$$G_2 = r_3 = 1$$

Now suppose the Critic provides value estimates for each state:

| State | $V(s)$ |
| ----- | ------ |
| $s_0$ | 3.0    |
| $s_1$ | 2.5    |
| $s_2$ | 0.8    |

Substituting $G_t$ and $V(s)$ into $A \approx G_t - V(s)$ yields the advantage estimate at each time step:

| Step $t$ | State | $G_t$  | $V(s_t)$ | $A = G_t - V(s_t)$  | Meaning                     |
| -------- | ----- | ------ | -------- | ------------------- | --------------------------- |
| 0        | $s_0$ | $5.51$ | $3.0$    | $5.51 - 3.0 = 2.51$ | $2.51$ better than expected |
| 1        | $s_1$ | $3.9$  | $2.5$    | $3.9 - 2.5 = 1.4$   | $1.4$ better than expected  |
| 2        | $s_2$ | $1$    | $0.8$    | $1 - 0.8 = 0.2$     | $0.2$ better than expected  |

All three advantages are positive, meaning every action along this trajectory performed better than average. $G_t - V(s)$ is an MC-return-based estimate of the advantage; it is unbiased but high-variance (different trajectories produce very different $G_t$ values).

## Advantage Versus Cumulative Return

The advantage reduces variance because it **subtracts the reward you would have gotten anyway**, retaining only the portion attributable to the specific action.

Consider a more complete example. Suppose that in some state $s$, the policy's average return is $V(s) = 10$. Four trajectories are sampled with returns $G_t^{(1)} = 18$, $G_t^{(2)} = 15$, $G_t^{(3)} = 7$, and $G_t^{(4)} = 4$.

First, using $G_t$ as the gradient signal:

| Episode | $G_t$ | Gradient signal    | Meaning                                |
| ------- | ----- | ------------------ | -------------------------------------- |
| 1       | 18    | $\nabla \times 18$ | Large positive, strongly pushes action |
| 2       | 15    | $\nabla \times 15$ | Positive, pushes action                |
| 3       | 7     | $\nabla \times 7$  | Positive, pushes action                |
| 4       | 4     | $\nabla \times 4$  | Positive, pushes action                |

All four are positive. The policy would conclude that "in this state, no matter what, this action is good" -- yet episodes 3 and 4 actually returned below average.

Now using $A = G_t - V(s)$:

| Episode | $G_t$ | $V(s)$ | $A = G_t - V(s)$ | Gradient signal      | Meaning                              |
| ------- | ----- | ------ | ---------------- | -------------------- | ------------------------------------ |
| 1       | 18    | 10     | $18 - 10 = +8$   | $\nabla \times (+8)$ | Far above average, strongly push     |
| 2       | 15    | 10     | $15 - 10 = +5$   | $\nabla \times (+5)$ | Above average, push                  |
| 3       | 7     | 10     | $7 - 10 = -3$    | $\nabla \times (-3)$ | Below average, suppress              |
| 4       | 4     | 10     | $4 - 10 = -6$    | $\nabla \times (-6)$ | Far below average, strongly suppress |

With $G_t$, all four episodes produce positive gradient signals -- the policy cannot distinguish "truly good" from "lucky high return." With $A$, the signal is calibrated: above-average returns get positive signals, below-average returns get negative signals.

To see the variance reduction quantitatively: using $G_t$, the four signals have mean $\frac{18+15+7+4}{4} = 11$ and variance $\frac{(18-11)^2+(15-11)^2+(7-11)^2+(4-11)^2}{4} = \frac{49+16+16+49}{4} = 32.5$. Using $A$, the four signals have mean $\frac{8+5-3-6}{4} = 1$ and variance $\frac{(8-1)^2+(5-1)^2+(-3-1)^2+(-6-1)^2}{4} = \frac{49+16+16+49}{4} = 32.5$.

The four-sample variance is the same, but $A$ has a mean much closer to zero. As sample size grows, $G_t$'s range is determined by the randomness of the entire trajectory (potentially spanning from 0 to dozens), while $A$'s range is centered by $V(s)$, with positive and negative values canceling out to produce a more stable expected gradient direction. This is exactly the mechanism by which "subtracting a baseline reduces variance."

## Estimating the Advantage with the TD Error

The theoretical definition of the advantage is $A = Q - V$, but in practice we rarely compute $Q$ directly. Starting from the definition and performing a one-step expansion yields a more practical form.

Begin with $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. The [action-value function](../chapter03_mdp/value-q) is defined as:

$$Q^\pi(s,a) = \mathbb{E}\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a\right]$$

This expectation represents: after taking action $a$ in state $s$, the immediate reward plus the value of the next state. If we take a single sample (without completing the entire episode or averaging over all possible transitions), we obtain a one-step estimate of $Q$:

$$Q(s,a) \approx r + \gamma V(s')$$

where $r$ is the actual reward received in this step and $s'$ is the actual next state reached. Substituting this approximation into the advantage definition:

$$A(s,a) = Q(s,a) - V(s) \approx r + \gamma V(s') - V(s)$$

The right-hand side is the [TD error](../chapter03_mdp/dp-mc-td):

$$A(s,a) \approx r + \gamma V(s') - V(s) = \delta \tag{7.2}$$

| Symbol   | Meaning                                                                     |
| -------- | --------------------------------------------------------------------------- |
| $r$      | The actual immediate reward received in this step.                          |
| $\gamma$ | Discount factor, controlling how much future value is discounted.           |
| $V(s')$  | The Critic's value estimate for the next state $s'$.                        |
| $V(s)$   | The Critic's value estimate for the current state $s$.                      |
| $\delta$ | TD error: how much better (or worse) the actual outcome was after one step. |

Replacing $G_t$ with the TD error as the policy gradient signal has two benefits:

1. **No need to wait for the episode to end** -- updates can happen after every step ($G_t$ requires a full episode, a limitation of [MC methods](../chapter03_mdp/dp-mc-td))
2. **Lower variance** -- $\delta$ involves randomness from only a single step ($G_t$ accumulates randomness over the entire trajectory)

Let us walk through a concrete numerical example. Suppose $\gamma = 0.9$, and at some step:

- Current state $s$, Critic estimates $V(s) = 5.0$
- The agent takes some action and receives immediate reward $r = +2$
- The next state is $s'$, Critic estimates $V(s') = 4.0$

Substituting into the TD error formula:

$$\delta = r + \gamma V(s') - V(s) = 2 + 0.9 \times 4.0 - 5.0 = 2 + 3.6 - 5.0 = +0.6$$

$\delta = +0.6$ means this step was $0.6$ better than the Critic predicted. Using this $\delta$ as the advantage estimate, the policy gradient will slightly increase the probability of this action.

Try different numbers. Suppose the same transition yields $r = -1$ instead:

$$\delta = -1 + 0.9 \times 4.0 - 5.0 = -1 + 3.6 - 5.0 = -2.4$$

$\delta = -2.4$ means this step performed far worse than predicted. The policy gradient will decrease the probability of this action.

Now consider the case $\delta = 0$. If $r = +1$, $V(s') = 5.0$, $V(s) = 5.5$:

$$\delta = 1 + 0.9 \times 5.0 - 5.5 = 1 + 4.5 - 5.5 = 0$$

$\delta = 0$: the actual outcome matches the Critic's prediction exactly. The policy gradient signal is zero, and the action's probability remains unchanged.

Now let us connect three time steps. Consider a 3-step episode with $\gamma = 0.9$:

| Step | State | Action | $r$  | Next state | $V(s)$ | $V(s')$ | $\delta = r + \gamma V(s') - V(s)$                |
| ---- | ----- | ------ | ---- | ---------- | ------ | ------- | ------------------------------------------------- |
| 0    | $s_0$ | $a_0$  | $+3$ | $s_1$      | 2.0    | 4.0     | $3 + 0.9 \times 4.0 - 2.0 = 3 + 3.6 - 2.0 = +4.6$ |
| 1    | $s_1$ | $a_1$  | $+1$ | $s_2$      | 4.0    | 1.0     | $1 + 0.9 \times 1.0 - 4.0 = 1 + 0.9 - 4.0 = -2.1$ |
| 2    | $s_2$ | $a_2$  | $+2$ | $s_3$      | 1.0    | 0.0     | $2 + 0.9 \times 0.0 - 1.0 = 2 + 0.0 - 1.0 = +1.0$ |

The three $\delta$ values are $+4.6$, $-2.1$, and $+1.0$. Step 0's action far exceeded expectations, so the policy should increase $a_0$'s probability; step 1's action fell short, so the policy should decrease $a_1$'s probability; step 2 slightly exceeded expectations, mildly encouraging $a_2$.

For comparison, the MC returns $G_t$ for the same trajectory are:

$$G_0 = 3 + 0.9 \times 1 + 0.9^2 \times 2 = 3 + 0.9 + 1.62 = 5.52$$

$$G_1 = 1 + 0.9 \times 2 = 2.8$$

$$G_2 = 2$$

The corresponding MC advantage estimates:

| Step | $G_t$ | $V(s)$ | $A_{\text{MC}} = G_t - V(s)$ |
| ---- | ----- | ------ | ---------------------------- |
| 0    | 5.52  | 2.0    | $5.52 - 2.0 = +3.52$         |
| 1    | 2.8   | 4.0    | $2.8 - 4.0 = -1.2$           |
| 2    | 2     | 1.0    | $2 - 1.0 = +1.0$             |

Both estimates give the same directional signals (positive, negative, positive), but different magnitudes. The TD advantage $\delta$ looks only one step ahead, while the MC advantage $G_t - V(s)$ sees to the end of the episode. $\delta$ has lower variance (only one step of randomness) but is biased (depends on the accuracy of $V(s')$); $G_t - V(s)$ is unbiased but high-variance (incorporating randomness from the entire trajectory).

This is the MC-to-TD transition replayed in the policy optimization setting: REINFORCE uses $G_t$ (MC), while Actor-Critic uses $\delta$ (TD).

|                    | **REINFORCE (MC)**                      | **Actor-Critic (TD)**                                      |
| ------------------ | --------------------------------------- | ---------------------------------------------------------- |
| Advantage estimate | $G_t - V(s)$ (requires full trajectory) | $r + \gamma V(s') - V(s) = \delta$ (update after one step) |
| Update timing      | after the episode ends                  | every step                                                 |
| Variance           | high                                    | low                                                        |
| Cost               | none                                    | requires training a Critic                                 |

## Implementing the Critic Network

To compute $\delta = r + \gamma V(s') - V(s)$, you need $V(s)$ and $V(s')$. In real problems $V$ is unknown -- a network is needed to approximate it. This network is the **Critic**.

```text
Actor (policy network)             Critic (value network)
  input:  state s                   input:  state s
  output: π_θ(a|s) distribution      output: V_φ(s) scalar
  role:   choose actions             role:   evaluate state value
  params: θ                          params: φ
```

The Actor and the Critic share the same input (the state $s$) but produce different outputs: the Actor outputs a probability distribution over actions, while the Critic outputs a scalar value estimate. They cooperate through the advantage estimate $A \approx \delta$: the Critic provides an evaluation signal, and the Actor adjusts its behavior based on that evaluation.

But how is the Critic trained? How does it learn to estimate $V(s)$ accurately? Section 7.3 expands on the three methods -- [DP, MC, and TD](../chapter03_mdp/dp-mc-td) -- briefly surveyed in Chapter 4, showing how they are applied concretely in Critic training. See: [Critic training methods](./critic-training)
