---
title: 2.1 Exploration and Exploitation
---

# 2.1 Exploration and Exploitation

<ChapterNotebookLink chapter="mdp" locale="en" />

**Core Content**

- Master the unified language of the MDP tuple, discounted cumulative return, value functions, and the Bellman equation.
- Understand how DP, MC, and TD estimate value functions under different assumptions.
- Distinguish the $Q(s,a)$ route from the $J(\theta)$ route, and understand how the reward function determines the optimization objective.

**Core Formulas**

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle \quad \text{(MDP tuple: defining the rules of the environment)}
$$

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \quad \text{(Discounted cumulative return: defining the long-term objective)}
$$

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \quad Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] \quad \text{(State value and action value: evaluating states and actions)}
$$

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right] \quad \text{(Policy optimization objective: measuring a policy's average return)}
$$

**Role of This Chapter's Formulas**

This part establishes a unified formulation of reinforcement learning through a set of foundational formulas. The MDP tuple is used to characterize the sequential decision-making environment the agent operates in; the discounted cumulative return $G_t$ defines the long-term optimization objective; the state value function and action value function evaluate the long-term return of states and actions; the policy objective $J(\theta)$ formulates the optimization problem for parameterized policies. Subsequent topics — DQN, policy gradient, Actor-Critic, and PPO — all build on these basic objects.

Chapter 1 trained a CartPole agent end to end. That run already contained the full reinforcement-learning loop, but action selection, state transitions, delayed return, and policy updates all appeared at once. Before formalizing the complete process, this section isolates the smallest decision problem: choosing among actions with unknown rewards.

The CartPole run let us use RL before defining its objects precisely. We now move from “how to run it” to “why it works”: what a reward measures, how actions acquire value, and how a sequence of decisions becomes an optimization problem.

Back to the most fundamental question: **What does reinforcement learning study?** The answer is **sequential decision-making** — the agent chooses an action at each step, the environment provides feedback and transitions to the next state, and so on. The key is that the agent pursues not the immediate reward at any single step, but the **cumulative return over the entire process**. Greedily taking the largest immediate reward is often not the optimal strategy.

To formally describe this process requires a unified formal framework — the **Markov Decision Process** (MDP). The MDP packages **states, actions, transition probabilities, reward functions, and discount factors** into a tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$. With it, value functions, the Bellman equation, Q-Learning, policy gradient, PPO — these seemingly diverse algorithms — all share a common language.

After defining the problem, the next step is to define a **measure of "goodness."** The **discounted cumulative return** $G_t$ describes the total reward obtainable from a trajectory starting at a given time, while the **value function** distributes the expectation of this reward to specific states or actions. Building on this, the **Bellman equation** reveals a key recursive structure: the value of a state equals "the immediate reward for the current step" plus "the discounted value of the next state." This seemingly simple equation is the common starting point for dynamic programming, Monte Carlo methods, and temporal difference methods.

Following the value function further naturally splits into **two algorithmic routes**:

- **Value-based route**: Learn $Q(s,a)$, score each action, then choose the action with the highest score — leading to Q-Learning and Deep Q-Networks.
- **Policy-based route**: Directly define a policy objective $J(\theta)$ and optimize the policy parameters through gradient methods — leading to policy gradient, Actor-Critic, and PPO.

This chapter serves as the **theoretical foundation** for the entire book. Chapter 5's Deep Q-Networks depend on $Q(s,a)$, Chapter 6's policy gradient depends on $J(\theta)$, and Chapters 7–8 use both value estimation and policy optimization in Actor-Critic and PPO. After understanding this chapter, many formulas in subsequent algorithms will no longer appear as isolated techniques, but as natural consequences derived from **the same decision modeling framework**.

## Section Outline

| Section                                                     | Core Content                                                                                  |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [Two-Armed Bandit](./bandit)                                | Understand exploration, exploitation, and expected return from the smallest decision problem  |
| [Markov Decision Process](./mdp)                            | Define the MDP tuple, discounted return, and policy                                           |
| [Value Functions and the Bellman Equation](./value-bellman) | Introduce the state value function and derive its recursive structure                         |
| [DP, MC, and TD](./dp-mc-td)                                | Compare three value estimation methods by assumptions, data requirements, and update rules    |
| [From Q to Q-Learning](./value-q)                           | Use GridWorld to illustrate action value, TD targets, exploration, and tabular boundaries     |
| [Policy, Value, and Return](./policy-value)                 | Define the objective function from the perspective of directly optimizing the policy          |
| [Where Does Data Come From](./algorithm-taxonomy)           | Discuss On-policy vs. Off-policy, Online vs. Offline                                          |
| [Reward Function Design](./reward-design)                   | Discuss how rewards express task objectives and the problems that incorrect rewards can cause |

## Learning Objectives

After reading this chapter, you should be able to:

- Formally describe a reinforcement learning problem using the MDP tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$;
- Understand how the **Bellman equation** writes long-term return as a recursive structure, and grasp the core differences among dynamic programming, Monte Carlo, and temporal difference methods for value estimation;
- Articulate the distinction between two algorithmic routes — the **$Q(s,a)$ route** scores actions and selects the best, while the **$J(\theta)$ route** directly optimizes policy parameters — leading respectively to Deep Q-Networks and PPO.

We begin with the multi-armed bandit, observe the exploration-exploitation tradeoff in a minimal environment, and then lift that intuition into the formal language of MDPs.

## What This Section Solves

**Core content**

- Master the multi-armed bandit problem as a stateless decision problem, and understand the basic tension between exploration and exploitation.
- Learn to compare policies (uniform random, oracle-optimal, explore-then-commit) using expected return.
- See why the **policy** is the core object in RL: in the same environment, different action-selection rules can lead to completely different long-term outcomes.

In Chapter 1, you ran CartPole end to end. But there is an even more basic question hiding underneath: **how does an agent know which action is better?** If we cannot answer "which of two options should I pick?", then dealing with continuously changing states in CartPole is even harder.

RL differs from supervised learning in two core ways. First, it is **trial-and-error**: nobody gives you the correct action, you must try. Second, it often involves **delayed reward**: the consequences of an action may only show up many steps later [^5]. Together, these create the characteristic difficulty of RL.

Trial-and-error immediately creates a problem that does not exist in ordinary supervised learning: **exploration vs. exploitation**. Imagine walking into a casino with only two slot machines and 100 coins. You do not know which machine pays out more often. The only way to learn is to try. But each try consumes a coin. **Do you keep testing the uncertain machine, or do you commit to the one that currently looks better?** Either extreme fails: explore without exploiting and you waste coins on bad machines; exploit without exploring and you may never discover the better one.

This scenario is called the **multi-armed bandit problem (MAB)** [^1][^2]. It shows up everywhere: choosing a restaurant (a familiar one or a new one), picking a movie (a known genre or a risky recommendation), selecting a research direction (go deeper or pivot). It is an ideal entry point to RL because it removes the complication of delayed reward: there is no changing state, so each round looks the same. That lets us isolate the two core tensions: trial-and-error and exploration vs. exploitation.

CartPole is not like this: every push changes the pole angle, and the whole situation evolves over time. Trial-and-error and delayed reward get mixed together. Slot machines deliberately remove state dynamics so we can focus on the two most basic questions:

1. **How do we evaluate whether an action is good?** → expected reward / expected return
2. **How do we trade off trying new options vs. exploiting what we already know?** → exploration strategies

::: info Core Idea
Exploration vs. exploitation is a **distinctive challenge** of RL. No matter whether the environment has state, or whether the action space is discrete or continuous, RL repeatedly asks two questions: **\"which is better\"** and **\"should I try something new\"**. The bandit problem is the simplest version of these questions. Once you understand it, later algorithms are mostly about adding structure on top.
:::

**Core formulas**

To answer \"which is better\", we need a quantitative notion. The two formulas below address two questions: **\"how good is a single machine\"** and **\"how good is an entire policy over T rounds\"**.

$$
\mathbb{E}[R_a] = p_a \cdot (+1) + (1-p_a)\cdot(-1) = 2p_a - 1 \quad \text{(Expected reward of a single arm: average payoff per pull)}
$$

> **Expected reward of an action:**
>
> - $\mathbb{E}$: expectation, the long-run average over repeated trials.
> - $R_a$: the reward obtained after choosing action $a$ (pulling a lever).
> - $p_a$: the true probability that action $a$ yields a positive payoff (a win).

$$
\mathbb{E}[R_T] = \mathbb{E}[R_{a_1}] + \mathbb{E}[R_{a_2}] + \cdots + \mathbb{E}[R_{a_T}] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}] \quad \text{(Expected total return over T rounds: policy-level performance)}
$$

> **Expected total return over T steps:**
>
> - $T$: the total number of rounds.
> - $a_t$: the action chosen at round $t$.
> - $R_T$: the cumulative reward after $T$ rounds.

The first formula says: if you repeatedly pull the same arm, what is the average profit per pull? The second formula sums across rounds so you can compare overall performance between different policies. With these tools, we can answer \"which policy is better\" with numbers instead of vibes.

For now, you can read a **policy** as a simple sentence: **given the current situation, what rule do I use to choose an action?** In bandits, it is the rule for choosing A vs. B each round; in CartPole, it is the rule for pushing left vs. right given the current pole angle; in language models, it is the rule for choosing the next token given the context.

## Two Slot Machines

You have two slot machines. Each play costs 1 coin. If you win, the machine returns 2 coins (net profit +1); if you lose, it takes the coin (net -1). You have 100 plays. You do not know the payout probabilities of the two machines.

::: info What is an episode?
In RL, a complete interaction from start to end is called an **episode**. For example, one run of Super Mario from start until game over is an episode; one CartPole run from upright until the pole falls is an episode.

Bandits are a special case: an episode can be a single step (choose an action, receive a reward). Later environments (CartPole, Atari) have many steps per episode. But regardless of episode length, we evaluate a policy by the **total return over an episode**.
:::

This simple setup already captures RL's core tension: **you need to try both machines early to learn which is better; once you have evidence, you should spend most of the remaining coins on the better machine.** Explore too much and you waste coins on the worse option; exploit too early and you may never learn that the other one is superior.

## Three Policies, Three Outcomes

To make the discussion concrete, suppose the true payout rates are: **machine A wins with probability 60%, machine B with probability 40%**. This is only for calculating expectations; the player does not know it and must estimate it by trying.

The point is not to do arithmetic for its own sake, but to see one fact clearly: **the environment does not change, and the number of coins does not change; what changes the outcome is the policy.** With the same 100 plays, \"uniform random\", \"always pick A\", and \"explore then commit\" can produce very different returns.

Before comparing policies, we need a tool to answer \"how much do we expect to earn on average?\" That tool is the **expectation**.

::: details What is an expectation?
An **expectation** is the long-run average of a random variable. Intuitively: **if you repeat the same experiment many, many times, the average outcome converges to the expectation.**

Formally, if an experiment has $n$ possible outcomes with values $x_1, x_2, \ldots, x_n$ and probabilities $p_1, p_2, \ldots, p_n$ (summing to 1), then the expectation is:

$$\mathbb{E}[X] = p_1 \cdot x_1 + p_2 \cdot x_2 + \cdots + p_n \cdot x_n = \sum_{i=1}^{n} p_i \cdot x_i$$

Sum up **(outcome value) × (its probability)** across outcomes.

Example: flip a fair coin. Heads wins $+1$, tails loses $-1$. Any single flip can lose money, but across 1000 flips you will see about 500 heads and 500 tails:

$$\text{total profit} = 500 \times (+1) + 500 \times (-1) = 0 \quad \Rightarrow \quad \text{average per trial} = \frac{0}{1000} = 0$$

Alternatively, compute it directly from probabilities:

$$\text{average per trial} = \frac{500 \times (+1) + 500 \times (-1)}{1000}$$

Since 500 is 0.5 × 1000 (50% probability times 1000 trials), we can rewrite:

$$= \frac{0.5 \times 1000 \times (+1) + 0.5 \times 1000 \times (-1)}{1000}$$

Cancel the 1000:

$$= 0.5 \times (+1) + 0.5 \times (-1) = 0$$

This matches the definition $\mathbb{E}[X] = 0.5 \times 1 + 0.5 \times (-1) = 0$.
:::

With this tool, we can compute the **expected payoff** for each machine per play:

$$\mathbb{E}[R_A] = 0.6 \times (+1) + 0.4 \times (-1) = +0.2$$

$$\mathbb{E}[R_B] = 0.4 \times (+1) + 0.6 \times (-1) = -0.2$$

That is, machine A earns about +0.2 per play on average, while machine B loses about -0.2. All policy comparisons below are based on these two numbers.

These numbers answer "is a single action good?" A policy answers "when should I choose which action?" RL is mainly about the latter: **how to turn value estimates into a reliable decision rule.**

### Policy 1: Uniform random

The simplest policy: flip a coin each round and choose A or B with 50% probability each.

First compute the expected reward **per round**. Each round has two stages: choose a machine (50/50), then win or lose. That yields four possible outcomes:

|       Event       | Machine | Outcome |   Probability   |
| :---------------: | :-----: | :-----: | :-------------: |
| Choose A and win  |    A    |   +1    | 0.5 × 0.6 = 0.3 |
| Choose A and lose |    A    |   -1    | 0.5 × 0.4 = 0.2 |
| Choose B and win  |    B    |   +1    | 0.5 × 0.4 = 0.2 |
| Choose B and lose |    B    |   -1    | 0.5 × 0.6 = 0.3 |

Apply the expectation formula: multiply each outcome by its probability and sum:

$$\mathbb{E}[R_{\text{per-round}}] = 0.3 \times (+1) + 0.2 \times (-1) + 0.2 \times (+1) + 0.3 \times (-1) = 0$$

Or take a cleaner view: instead of expanding all four cases, first condition on which machine you chose this round.

If you choose A, the expected payoff is +0.2; if you choose B, it is -0.2. Since each is chosen with probability 0.5:

$$\mathbb{E}[R_{\text{per-round}}] = 0.5 \times 0.2 + 0.5 \times (-0.2) = 0$$

This is the same computation as listing all four outcomes above, just grouped differently: above we expand by "what happened in the end"; here we first condition on "which machine was chosen", then use that machine's own expected payoff.

::: details Why is this regrouping valid?

Formally, this step uses the **law of total expectation**.

Let

$$C_A = \text{choose A}, \quad C_B = \text{choose B}$$

$C_A$ and $C_B$ are mutually exclusive and cover all cases in a round: you either choose A or choose B. Therefore:

$$\mathbb{E}[R_{\text{per-round}}] = P(C_A)\mathbb{E}[R_{\text{per-round}} \mid C_A] + P(C_B)\mathbb{E}[R_{\text{per-round}} \mid C_B]$$

The meaning is straightforward:

$$\text{overall mean} = \text{probability of group A} \times \text{mean within group A} + \text{probability of group B} \times \text{mean within group B}$$

Here:

$$\mathbb{E}[R_{\text{per-round}} \mid C_A] = 0.6 \times 1 + 0.4 \times (-1) = 0.2$$

$$\mathbb{E}[R_{\text{per-round}} \mid C_B] = 0.4 \times 1 + 0.6 \times (-1) = -0.2$$

So:

$$\mathbb{E}[R_{\text{per-round}}] = 0.5 \times 0.2 + 0.5 \times (-0.2) = 0$$

Why is this identical to enumerating the four outcomes? Because enumeration uses **joint probabilities**: the probability that multiple events happen together, such as "choose A and win" or "choose A and lose".

Each joint probability can be factored into two terms:

$$P(\text{choose A and win}) = P(\text{choose A})P(\text{win}\mid \text{choose A}) = 0.5 \times 0.6 = 0.3$$

$$P(\text{choose A and lose}) = P(\text{choose A})P(\text{lose}\mid \text{choose A}) = 0.5 \times 0.4 = 0.2$$

So this regrouping is simply re-parenthesizing the same four terms:

$$0.3 \times 1 + 0.2 \times (-1) + 0.2 \times 1 + 0.3 \times (-1)$$

$$= 0.5 \times \big(0.6 \times 1 + 0.4 \times (-1)\big) + 0.5 \times \big(0.4 \times 1 + 0.6 \times (-1)\big)$$

This is not a new assumption, and it is not claiming that outcomes can be added arbitrarily. It is only grouping the same random experiment by "which machine was chosen".
:::

The expected reward per round is 0. What about the expected total reward over 100 rounds? This uses a basic property of probability theory:

> **Theorem (Linearity of expectation)**: for any finite set of random variables $X_1, X_2, \ldots, X_n$, regardless of whether they are independent or correlated, we have:
>
> $$\mathbb{E}\!\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} \mathbb{E}[X_i]$$
>
> The only requirement is that each $\mathbb{E}[X_i]$ exists and is finite.

In words: **the expectation of a sum equals the sum of expectations**. This is the justification for the formula at the beginning of the section, $\mathbb{E}[R_T] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}]$. Here, the expected reward per round is 0, so adding 100 zeros is still 0:

$$\mathbb{E}[R_{100}] = \overbrace{0 + 0 + \cdots + 0}^{100} = 0$$

The total return fluctuates around 0. Gains from choosing A are canceled out by losses from choosing B. **The problem is that this policy does not learn at all.** No matter how many rounds you play, it never uses evidence about which machine is better; it stays 50/50 forever.

### Policy 2: Always choose A

Suppose a friend secretly tells you that A is better. Then you would play A for all 100 rounds. The per-round expectation is simply the expectation of machine A:

$$\mathbb{E}[R_{\text{per-round}}] = 0.6 \times (+1) + 0.4 \times (-1) = 0.2$$

Similarly, the expected total return over 100 rounds is the sum of the per-round expectations:

$$\mathbb{E}[R_{100}] = \overbrace{0.2 + 0.2 + \cdots + 0.2}^{100} = 100 \times 0.2 = 20$$

The expected profit is +20, which is the best possible result in expectation. But the problem is obvious: **in reality, nobody tells you which machine is better**. You have to discover it yourself. So this policy is an oracle baseline, not a usable strategy.

### Policy 3: Explore then commit

A more practical approach is: **spend a small number of rounds trying both machines to identify the winner, then spend the remaining rounds exploiting it.**

Concretely: for the first 20 rounds, alternate between A and B (10 pulls each) and record the observed win rates. For the remaining 80 rounds, always choose the machine with the higher observed win rate.

The first 20 rounds are the same as Policy 1: each round chooses A with probability 0.5 and B with probability 0.5. So the expected reward per round is still 0, and the expected total over 20 rounds is 0.

In the remaining 80 rounds we commit to A (assuming the exploration phase correctly identified the better machine). The per-round expectation is 0.2:

$$\mathbb{E}[R_{\text{last 80}}] = 80 \times 0.2 = 16$$

So the expected total return over 100 rounds is:

$$\mathbb{E}[R_{100}] = 0 + 16 = 16$$

This is lower than the oracle optimum of 20. The gap (4) is the cost of exploration.

At this point, "explore then commit" seems to capture the intuition of exploration vs. exploitation: spend 20 rounds exploring, then spend 80 rounds exploiting the better machine. But it hides an assumption: **the exploration phase must correctly identify which machine is better**.

This brings us back to the question at the start: how does an agent know which action is better? In real RL, the agent does not see the true payout rate of A (60%) or B (40%). It only sees samples from its own trials, and must estimate action values from those samples. Next we will see the key problem: **this estimate can be wrong**.

### Observations Can Mislead

The calculation above made an optimistic assumption: the exploration phase correctly identifies A as better. In reality, things are not so nice. With only a handful of trials, short-term luck can easily mislead you.

The easiest confusion here is this: the **true payout probability** is not the same thing as the **observed payout rate**.

- The true payout probability is the underlying long-run rule of the machine. A true rate of 60% means that if you play many, many times, the win fraction will approach 60%.
- The observed payout rate is what you see in a small experiment. With only 10 pulls, it can be skewed by short-term randomness.

So a true 60% rate does not mean "you must win 6 out of every 10". You might win 4 out of 10 in one batch, and 7 out of 10 in the next. Ten trials are too few for the observed rate to reliably reflect the truth.

For example, the exploration phase might look like this:

We use symbols to denote outcomes: ✅ for a win, ❌ for a loss.

| Machine | True win prob. | 10 actual outcomes            | Wins | Observed win rate |
| :-----: | :------------: | :---------------------------- | :--: | :---------------: |
|    A    |      60%       | ❌ ✅ ❌ ❌ ✅ ❌ ✅ ❌ ✅ ❌ |  4   |        40%        |
|    B    |      40%       | ✅ ❌ ✅ ✅ ❌ ❌ ✅ ❌ ✅ ❌ |  5   |        50%        |

These two rows do not change the true machine parameters. Each pull of A still has a 60% chance to win; it is just that in this batch of 10 trials, A happened to win fewer times. Each pull of B still has only a 40% chance to win; it is just that in this batch, B happened to win more often.

In truth, A is better than B. But if you only look at these 10 samples, B can appear better. This is not a contradiction; it is small-sample randomness.

If you decide purely based on observed win rate, you might incorrectly conclude that B is better and commit the remaining 80 rounds to B. This is exactly what "observations can mislead" means: **the truly better option may not look better in the short run.**

This is not to make the problem complicated for its own sake. It highlights a central difficulty in RL: **exploitation depends on estimates; estimates come from exploration; and exploration samples can be unreliable.**

::: details Mathematical note: why can short-term observations differ from the true probability?

Mathematically, the true win probability and the observed win rate are different quantities.

- **True win probability**: denote it by $p$. This is the machine's underlying parameter. For example, A has $p_A = 0.6$.
- **Number of wins in 10 trials**: denote it by $N$. This is a random variable because each trial outcome can differ.
- **Observed win rate**: denote it by $\hat p$ ("p-hat"), meaning the probability estimated from samples.

The observed win rate is:

$$\hat p = \frac{N}{10}$$

If in one run you pull A 10 times and only win 4 times, then:

$$N_A = 4,\quad \hat p_A = \frac{4}{10}=0.4$$

This does not mean A's true probability changed from $0.6$ to $0.4$. The true probability is still:

$$p_A = 0.6$$

It only means that in this small experiment, what you observed was:

$$\hat p_A = 0.4$$

More formally, for each pull of A, define a random variable:

$$
X_i =
\begin{cases}
1, & \text{win on trial } i\\
0, & \text{lose on trial } i
\end{cases}
$$

A true win probability of $p_A = 0.6$ means:

$$P(X_i=1)=0.6,\quad P(X_i=0)=0.4$$

Note: $0.6$ is the probability that the outcome is a win. It does not mean the outcome of each trial is "0.6". After a trial happens, $X_i$ can only be 1 or 0.

After 10 trials, the total number of wins is:

$$N_A = X_1 + X_2 + \cdots + X_{10}$$

Since each $X_i$ is random, $N_A$ is also random. It can be 6, or 4, or any other value. The observed win rate is simply:

$$\hat p_A = \frac{N_A}{10}$$

So the machine parameter and a particular realized outcome are not the same thing:

- $p_A=0.6$: the rule that generates outcomes.
- $N_A=4$: what happened in this specific run of 10 trials.
- $\hat p_A=0.4$: the temporary estimate computed from this run.

In expectation, $p_A$ controls the long-run average:

$$\mathbb{E}[N_A]=10\times 0.6=6,\quad \mathbb{E}[\hat p_A]=0.6$$

This does not mean "you will always win 6 out of every 10". It means that if you repeat 10-trial experiments many times, the average number of wins per experiment approaches 6.

This is exactly the point: **the number of wins in 10 trials is itself random**. With win probability 0.6, A might win 0, 1, ..., up to 10 times. It is not fixed at 6.

The probability of "winning k times" is given by the **binomial distribution**:

$$P(N=k)=\binom{n}{k}p^k(1-p)^{n-k}$$

Here $N$ is the number of wins in $n$ trials and $p$ is the true win probability per trial. For example, the probability that A wins exactly 4 times is:

$$P(n_A = 4) = \binom{10}{4} \times 0.6^4 \times 0.4^6 \approx 11.1\%$$

So even if A's true win probability is 60%, winning only 4 times out of 10 still happens with about 11.1% probability. It is not the most likely outcome, but it is not rare either.

The probability that B wins exactly 5 times is:

$$P(n_B = 5) = \binom{10}{5} \times 0.4^5 \times 0.6^5 \approx 19.1\%$$

Therefore, a misleading observation such as "A wins 4/10 while B wins 5/10" is not only possible, it has a nontrivial probability. For this specific pair of events, the joint probability is roughly:

$$11.1\% \times 19.1\% \approx 2.2\%$$

This is not large, but it is certainly possible. More importantly, misidentification is not limited to this one case: any time A's observed wins are fewer than B's, you will make the wrong call. Later we will account for all such cases.

What probability theory does guarantee is this: if you sample enough, the observed win rate converges toward the true probability. This is the **law of large numbers**. It does not guarantee that 10 trials are enough.

For the "A wins 4 times" formula above, the three factors have clear meanings:

- $\binom{10}{4}$: how many ways to choose which 4 out of 10 trials are wins.
- $0.6^4$: probability that those 4 selected trials are wins.
- $0.4^6$: probability that the remaining 6 trials are losses.

**How should we read the combination term?**

$\binom{n}{k}$ (also written $C_n^k$) counts the number of ways to choose $k$ positions out of $n$ positions. It is a **combinatorial coefficient**, not a probability by itself.

Its role here is to count all possible placements of the 4 wins among the 10 trials. Each specific placement has probability $0.6^4 \times 0.4^6$, so we multiply by the number of placements, $\binom{10}{4}$.

So $\binom{10}{4}$ is not a probability; it is a **counting factor**.

The key point: **combinations do not care about order**.

A tiny example: there are 4 positions $\{1,2,3,4\}$ and we want to choose 2 positions.

$$\binom{4}{2} = \frac{4!}{2! \times 2!} = \frac{4 \times 3 \times 2 \times 1}{(2 \times 1)(2 \times 1)} = 6$$

That is $C_4^2 = 6$. Explicitly, the combinations are:

$$\{1,2\}, \{1,3\}, \{1,4\}, \{2,3\}, \{2,4\}, \{3,4\}$$

This is not $A_4^2$. $A_4^2$ counts permutations and includes order: for example $\{1,2\}$ and $\{2,1\}$ are considered different. So $A_4^2 = 4 \times 3 = 12$.

Back to our case: $\binom{10}{4}$ (i.e. $C_{10}^4$) counts how many ways to choose which 4 out of 10 trials are wins. For instance, wins on trials 1,3,5,7 is one choice; wins on 2,4,6,8 is another.

This is also not $A_{10}^4$, because we do not care about the order among the 4 wins. The trial order is already fixed from 1 to 10; we only choose which positions are wins. So we use the combination count $C_{10}^4$.
:::

What really matters for the policy's return is: **with what probability does exploration mistakenly rank the worse machine B above the better machine A?**

The decision rule for explore-then-commit is simple: after 20 rounds, compare the number of wins for A and B. Whichever has more wins becomes the choice for the remaining 80 rounds.

- If A has more wins, the decision is correct and we exploit A.
- If B has more wins, the decision is wrong and we exploit B.
- If they tie, we will treat it separately (not counting it as a win for B).

So misidentification can be stated in one sentence:

> **B has more wins than A.**

Here $n_A$ is the number of wins of A in its 10 exploration pulls, and $n_B$ is the number of wins of B in its 10 pulls.

How do we compute this probability? Before formulas, think of it as a grid: the x-axis is how many times A wins, the y-axis is how many times B wins. Each cell is one possible exploration outcome.

| A wins | B wins | Choice afterwards | Wrong? |
| :----: | :----: | :---------------: | :----: |
|   4    |   5    |         B         |  yes   |
|   3    |   6    |         B         |  yes   |
|   5    |   5    |        tie        |   no   |
|   6    |   4    |         A         |   no   |

The rule is simple: whenever B's number is larger than A's, that cell is a mistake. Summing the probabilities of all mistake cells gives approximately:

$$P(\text{mistake}) \approx 12.8\%$$

Meaning: even though A is truly better, if you only try each machine 10 times, roughly 1 out of every 8 full experiments will be fooled by short-term noise and conclude that B is better.

::: details How do we sum up to 12.8%?

Mathematically, the set of mistake cells is:

$$n_A < n_B$$

That is, A wins fewer times than B.

We can condition on how many wins A gets:

| If A wins... | then B must win... | to be a mistake |
| :----------: | :----------------: | :-------------: |
|      0       |      1 to 10       |       yes       |
|      1       |      2 to 10       |       yes       |
|      2       |      3 to 10       |       yes       |
|     ...      |        ...         |       ...       |
|      9       |         10         |       yes       |
|      10      |     impossible     |       no        |

So the total mistake probability sums these cases:

$$P(\text{mistake})$$

$$=P(A\text{ wins }0)P(B\text{ wins at least }1)$$

$$+P(A\text{ wins }1)P(B\text{ wins at least }2)$$

$$+\cdots+P(A\text{ wins }9)P(B\text{ wins }10)$$

$$\approx 12.8\%$$

Two small pieces of logic are used here:

- Why multiply? Because the 10 trials of A and the 10 trials of B are independent experiments. For example, the probability of "A wins 4 and B wins 5" is $P(A\text{ wins }4)\times P(B\text{ wins }5)$.
- Why add? Because these cases are mutually exclusive. In one experiment A cannot both win 0 times and 1 time, so we can sum probabilities across cases.
  :::

If we make a mistake, we commit the last 80 rounds to B, and the expected total return becomes:

$$\mathbb{E}[R_{100} \mid \text{mistake}] = 0 + 80 \times (-0.2) = -16$$

So the outcome of Policy 3 depends on luck: with 87.2% probability you pick correctly (get +16 expected), and with 12.8% probability you pick wrongly (get -16 expected). The expectation is the probability-weighted average:

$$\mathbb{E}[R_{100}] = 87.2\% \times 16 + 12.8\% \times (-16) \approx 11.9$$

This is almost 4 points lower than the optimistic 16 that assumes no misidentification. The drop is not a detail; it is the point: **if you estimate action values incorrectly, exploitation becomes incorrect too.**

Note that this is for a relatively large gap (60% vs 40%). If the machines are closer, say 52% vs 48%, the misidentification probability can jump toward 50%, which is almost like flipping a coin to decide the fate of the last 80 rounds.

This shows that **the sample size in exploration directly controls the final return**. Too few samples leads to high misidentification probability and weakens the benefit of exploration; too many samples increases exploration cost and eats into profit. Balancing these is exactly the core of exploration strategy design in RL.

## Comparing Exploration Strategies

The example shows that the problem is not just "choose A or B". The real difficulty is: **you must estimate which action is better from limited samples, while also not wasting too many rounds on exploration.**

Explore-then-commit is only one exploration strategy. Different strategies are different answers to the same question: how do we trade off "reducing mistakes" versus "spending fewer samples on exploration"?

| **Policy**          | **How it acts**                                                | **Exploration mechanism** | **Downside**                                          |
| ------------------- | -------------------------------------------------------------- | ------------------------- | ----------------------------------------------------- |
| Uniform random      | sample uniformly                                               | none                      | never learns the best arm                             |
| Greedy              | always choose the currently estimated best                     | none                      | may get stuck on a suboptimal arm                     |
| $\epsilon$-greedy   | with prob. $\epsilon$ choose random, otherwise choose best     | fixed-rate exploration    | if $\epsilon$ does not decay, exploration never stops |
| Explore then commit | explore for the first $N$ steps, then exploit                  | budgeted exploration      | choosing $N$ is tricky                                |
| UCB                 | choose the arm with the largest "estimated mean + uncertainty" | uncertainty-driven        | must maintain confidence bounds                       |
| Thompson sampling   | sample from the posterior and pick the best draw               | probability matching      | requires Bayesian updates                             |

Among these, UCB (Upper Confidence Bound [^3]) and Thompson sampling [^1] are theoretically near-optimal strategies. But what does "optimal" mean here? The standard notion is **regret**, which we will briefly introduce at the end of the section.

## Building a Bandit Environment in Python

```python
import random

class TwoArmedBandit:
    """Two-armed bandit: a minimal RL environment."""

    def __init__(self, prob_a=0.6, prob_b=0.4):
        self.prob_a = prob_a
        self.prob_b = prob_b

    def step(self, action):
        """Choose a machine and return the reward."""
        if action == "A":
            return 1 if random.random() < self.prob_a else -1
        else:
            return 1 if random.random() < self.prob_b else -1
```

This environment has no state: regardless of what you chose last round, the situation in the next round is identical. That is the defining feature of bandits. You can view it as a **single-state MDP**. Later, CartPole and LLM settings will not be like this: their states evolve as actions are taken.

## Implementing the Three Policies in Python

Now implement the three policies above in code and compare empirical outcomes with theoretical expectations. The code below assumes you have already defined the `TwoArmedBandit` class.

### Policy 1: Uniform random

```python
from random import choice
env = TwoArmedBandit()
total = sum(env.step(choice(["A", "B"])) for _ in range(100))
print(f"Uniform random: total return over 100 rounds: {total}, average: {total/100:.2f}")
```

:::output
Uniform random: total return over 100 rounds: -2, average: -0.02
:::

The theoretical expectation is 0. The empirical result fluctuates around 0, as expected.

### Policy 2: Always choose A

```python
env = TwoArmedBandit()
total = sum(env.step("A") for _ in range(100))
print(f"Always choose A: total return over 100 rounds: {total}, average: {total/100:.2f}")
```

:::output
Always choose A: total return over 100 rounds: 18, average: 0.18
:::

The theoretical expectation is 20, while this run got 18. A single experiment has randomness, but the result is close to the expectation.

### Policy 3: Explore then commit

```python
env = TwoArmedBandit()

# Phase 1: exploration - try A and B 10 times each and record performance
rewards = {"A": [], "B": []}
for arm in ["A", "B"]:
    for _ in range(10):
        rewards[arm].append(env.step(arm))

# Based on exploration, choose the machine with the higher average reward
avg = {arm: sum(r) / len(r) for arm, r in rewards.items()}
best = max(avg, key=avg.get)  # e.g. avg["A"] = 0.2 > avg["B"] = -0.1 -> best = "A"

# Phase 2: exploitation - choose best for the remaining 80 rounds
explore_total = sum(sum(r) for r in rewards.values())
exploit_total = sum(env.step(best) for _ in range(80))
total = explore_total + exploit_total

print(f"Exploration results: A avg={avg['A']:.2f}, B avg={avg['B']:.2f} -> choose {best}")
print(f"Explore then commit: total return over 100 rounds: {total}, average: {total/100:.2f}")
```

:::output
Explore then commit: total return over 100 rounds: 14, average: 0.14
:::

The theoretical expectation is 16 (ignoring misidentification), while the empirical result is 14. This is lower than the 18 achieved by "always choose A" because the first 20 exploration rounds do not reliably produce positive reward.

### Comparing Results

| Policy              | Theoretical expectation | Empirical result | Notes                                                      |
| ------------------- | ----------------------- | ---------------- | ---------------------------------------------------------- |
| Uniform random      | 0                       | -2               | fluctuates around 0                                        |
| Always choose A     | 20                      | 18               | close to expectation, but requires knowing A is better     |
| Explore then commit | 16                      | 14               | about 4 points lower than always-A due to exploration cost |

Three policies, the same machines, drastically different outcomes. **The policy determines how much value you can extract from the environment.**

We can draw two conclusions:

- **Policies should be evaluated by expected return.** Any single run can be lucky or unlucky, but long-run averages reveal policy quality.
- **Whether a policy helps depends on exploitable structure in the environment.** If A and B are both 50%, even the smartest policy cannot gain an edge. If A is truly better, the value of a policy is discovering and exploiting that fact faster.

## Section Summary

Using the bandit as the simplest RL environment, this section established three foundational ideas in reinforcement learning.

**1. The policy is the key determinant of performance.** With the same environment, different policies can lead to dramatically different long-run returns:

- **Uniform random** (expected return 0): never uses observations and never converges to the best action.
- **Oracle: always choose the best** (expected return 20): optimal in theory, but it assumes the action values are known in advance, which is unrealistic.
- **Explore then commit** (expected return ≈ 12): explore for 20 rounds, then exploit for 80 rounds. This is closer to practical learning, but it exposes a key issue: **small-sample estimates can be wrong**. With 60% vs 40%, trying each arm 10 times still yields about a 12.8% chance of picking the wrong one; if the gap is smaller, the mistake rate rises sharply.

**2. Expected return is an objective way to compare policies.** We computed the expected reward per pull (A: +0.2, B: -0.2) and used linearity of expectation to sum per-step expectations into the expected total over $T$ rounds. Any single run is noisy, but long-run averages converge toward expectation.

**3. Exploration vs. exploitation is the signature tension of RL.** $\epsilon$-greedy mixes exploration and exploitation at a fixed rate; UCB chooses actions by an uncertainty upper bound; Thompson sampling performs probability matching via Bayesian posterior sampling.

The bandit problem deliberately removes delayed reward and state dynamics, so we can see trial-and-error and exploration-vs-exploitation in their simplest form. Real RL problems (CartPole, LLM post-training, etc.) add state transitions: the situation changes as actions are taken, and the policy expands from "which action" to "which action in which state". Next we will formalize these ideas with the MDP framework: [MDP tuple, discounted return, and policies](./mdp).

::: details What is Thompson sampling?
Thompson sampling is a way to explore **according to uncertainty**. Instead of keeping only one estimated win rate for each machine, it keeps a range of plausible win rates.

Imagine that machine A has won 6 of 10 pulls and machine B has won 1 of 2:

- A's observed win rate is 60%, backed by ten observations.
- B's observed win rate is 50%, backed by only two observations.

A currently looks better, but B is much less certain. A greedy policy always chooses A and may stop learning about B. Uniform random exploration chooses B half the time, even after strong evidence shows that B is worse. Thompson sampling takes a middle path: **choose each machine in proportion to how likely it currently seems to be the best.**

### The three-step idea

For every round:

1. Keep a probability distribution for each machine's unknown true win rate.
2. Draw one possible win rate from each distribution.
3. Pull the machine with the largest draw, then update its distribution using the result.

The draw is not treated as the true win rate. It is one plausible answer to the question, "Given everything I have seen, what might this machine's win rate be?"

Suppose one round produces:

$$
\tilde p_A = 0.58,\qquad \tilde p_B = 0.72
$$

Then Thompson sampling chooses B. On another round it might draw:

$$
\tilde p_A = 0.63,\qquad \tilde p_B = 0.41
$$

and choose A. B is explored because its uncertainty sometimes produces a high draw, not because the algorithm follows a fixed random-exploration schedule.

### Representing uncertainty with a Beta distribution

For a machine whose outcome is win ($1$) or loss ($0$), a convenient model is:

$$
p_a \sim \operatorname{Beta}(\alpha_a,\beta_a)
$$

You can read the parameters as accumulated evidence:

- $\alpha_a = 1 +$ number of observed wins;
- $\beta_a = 1 +$ number of observed losses.

Starting with $\operatorname{Beta}(1,1)$ means every win rate between 0 and 1 is initially plausible. After observing a win, add 1 to $\alpha_a$. After a loss, add 1 to $\beta_a$:

$$
\text{win: }\alpha_a \leftarrow \alpha_a+1,
\qquad
\text{loss: }\beta_a \leftarrow \beta_a+1
$$

For example, six wins and four losses produce:

$$
p_A \sim \operatorname{Beta}(7,5)
$$

Its center is near the observed 60% win rate, but it still has some width because ten observations are limited. After 600 wins and 400 losses, the center remains near 60%, while the distribution becomes much narrower. The estimate has not changed much; our **confidence** has.

### Why exploration happens automatically

An arm tried only a few times has a wide distribution. Its samples sometimes land very high, so it continues to receive occasional trials. An arm tried many times has a narrow distribution. If its results are consistently poor, it rarely produces the largest sample and is selected less often.

This gives Thompson sampling its useful behavior:

- uncertain arms receive opportunities to prove themselves;
- strong arms are exploited frequently;
- clearly weak arms gradually stop consuming many trials.

This behavior is called **probability matching**. If the current evidence says A has roughly an 80% chance of being the best arm, Thompson sampling will choose A roughly 80% of the time. It does not need a separate exploration probability such as $\epsilon$.

### Minimal pseudocode

```python
alpha = [1, 1]
beta = [1, 1]

for _ in range(num_rounds):
    samples = [
        random.betavariate(alpha[a], beta[a])
        for a in range(2)
    ]
    action = max(range(2), key=lambda a: samples[a])
    reward = pull(action)  # 1 for win, 0 for loss

    alpha[action] += reward
    beta[action] += 1 - reward
```

The essential idea is simple: **sample one believable world, act optimally in that world, then revise what worlds remain believable.**
:::

## Further Reading: Regret

This part is optional. Feel free to skip it on a first pass.

Above we compared policies using expected return. But expected return depends on the specific bandit parameters (e.g., A=60%, B=40%). Change the numbers and the comparison changes. We therefore want a **parameter-agnostic metric** that works for any number of arms and any payout rates. That metric is **regret**.

Regret is intuitive: if you knew the best arm from the beginning, you could play it for all 100 rounds and achieve the optimal return. But you do not know, so you follow some learning policy and get less. The gap is regret:

> Regret = (return of the optimal policy) - (return of your policy)

For our bandit example (optimal policy plays A for 100 rounds, expected return 20):

| Policy              | Expected return | Regret          |
| ------------------- | --------------- | --------------- |
| Uniform random      | 0               | 20 - 0 = **20** |
| Oracle: always A    | 20              | 20 - 20 = **0** |
| Explore then commit | ~12             | 20 - 12 ~ **8** |

The smaller the regret, the better the policy. Uniform random "wastes" about 20 points per 100 rounds. Explore-then-commit wastes about 8 points (including exploration cost and mistakes). The oracle policy wastes none, but it assumes you know A is better from the start, which is unrealistic.

**One goal of RL exploration is to design policies whose regret grows as slowly as possible.** UCB and Thompson sampling are classic near-optimal answers: their regret grows only logarithmically with time and matches theoretical lower bounds [^4]. If you want to go deeper, follow the references.

## References

For further reading, see Auer et al.'s [finite-time analysis of multi-armed bandits](https://link.springer.com/article/10.1023/A:1013689704352), Russo et al.'s [tutorial on Thompson sampling](https://arxiv.org/abs/1707.02038), and Lattimore and Szepesvári's open textbook [Bandit Algorithms](https://banditalgs.com/).

[^1]: Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. _Biometrika_, 25(3/4), 285-294.

[^2]: Robbins, H. (1952). Some aspects of the sequential design of experiments. _Bulletin of the American Mathematical Society_, 58(5), 527-535.

[^3]: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. _Machine Learning_, 47(2-3), 235-256.

[^4]: Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. _Advances in Applied Mathematics_, 6(1), 4-22.

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.
