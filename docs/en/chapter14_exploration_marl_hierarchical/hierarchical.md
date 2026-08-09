# 12.3 Hierarchical RL and a Primer on Generative World Models

> [14.2](./marl) dealt with the multi-agent setting. This section deals with a third case we've been deliberately dodging — **extremely long task horizons** (a robot cleaning an entire house, say, needs 1000+ steps of action). A single-layer policy learns almost nothing from a signal that thin, spread that far. **Hierarchical RL** decomposes long-horizon decision-making into a sequence of options: a high-level policy picks subgoals, a low-level policy carries them out.

## Hierarchical RL: Options, FeUdal Networks, and HIRO

Long-horizon tasks are another of RL's hard problems. Take _Atari Montezuma's Revenge_: the agent has to grab a key, open a door, then move to the next room. Train it directly with PPO and the gradient signal is buried within a few thousand steps. **Hierarchical RL** splits decision-making into two layers (or more):

- **High-level policy**: decides occasionally, outputting a "subgoal" or "option"
- **Low-level policy**: given the high-level policy's subgoal, executes atomic actions until that subgoal is complete

This way the high level only has to deal with a sparse sequence of subgoals — the long horizon gets cut into short horizons, and the gradient signal can propagate within each short segment.

### The options framework

**Options**, formalized by Sutton, Precup & Singh (1999), are the theoretical foundation of hierarchical RL. An option $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ has three parts:

- **Initiation set** $\mathcal{I}_\omega$: the set of states from which the option can be started
- **Intra-option policy** $\pi_\omega$: the policy followed while the option is executing
- **Termination function** $\beta_\omega(s)$: the probability of terminating the option upon reaching state $s$

The Bellman equation for a semi-Markov decision process (SMDP) extends the usual one by taking the expectation over options:

$$Q^\mu(s, \omega) = \mathbb{E}\left[\sum_{t=0}^{T-1}\gamma^t r_t + \gamma^T \max_{\omega'} Q^\mu(s_T, \omega')\right]$$

where $T$ is the timestep at which the option terminates. What makes the options framework elegant is the separation it buys you: the high level can be learned like an ordinary MDP (via SMDP-Q-learning), and the low level can be trained independently with any model-free algorithm.

### FeUdal Networks: the manager sets a direction, the worker executes it

FeUdal Networks (Vezhnevets et al. 2017) turn options into something end-to-end learnable. There are two networks:

- **Manager** $M_\theta$: every $c$ steps, outputs a direction vector $g_t \in \mathbb{R}^k$ in a latent space (not a subgoal directly)
- **Worker** $W_\phi$: within each $c$-step window, outputs an action $\pi_\phi(a \mid s; g_t)$ at every atomic step, with its target distribution modulated by $g_t$

The manager's training objective is the clever part: it makes $g_t$ predict **the direction of change in the hidden state over the next $c$ steps**:

$$\mathcal{L}_M = -\langle g_t,\ \hat{z}_{t+c} - \hat{z}_t\rangle$$

where $\hat{z}$ is the output of a shared encoder. This is a **self-supervised** objective — the manager needs no external reward at all to learn to "point toward directions with informative change." The worker is still trained on environment reward, but conditioned on $g_t$.

FeUdal was the first end-to-end deep RL method to score positively on _Montezuma's Revenge_, but training is unstable and hyperparameter-sensitive, which makes it hard to reproduce in practice.

### Off-policy hierarchical RL

Data-Efficient Hierarchical Reinforcement Learning (HIRO, Nachum et al. 2018) is a modern refinement of FeUdal, and its key innovation is **off-policy training combined with goal relabeling**:

- The high level outputs a continuous subgoal $g_t \in \mathbb{R}^d$ (directly a displacement in state space), switching every $c$ steps
- The low-level reward is intrinsic: $r^l_t = -\|s_{t+1} - (s_t + g_t)\|$, which rewards the low level for reaching the displacement the high level specified
- The high level is trained with an off-policy algorithm (such as TD3)

The main technical obstacle is **off-policy bias**: an old subgoal $g$ pulled from the replay buffer was generated with respect to whatever low-level policy was in effect at the time, but the low-level policy has since changed. HIRO resolves this with **goal relabeling**: it remaps the old subgoal $g$ into a new subgoal $g'$ — "the subgoal that would have been reached had the _current_ low-level policy been executed instead" — so the high level's training data stays consistent.

```python
# HIRO main loop skeleton
for step in range(total_steps):
    if step % c == 0:
        # High level samples a subgoal every c steps
        goal = high_level_policy(state)
    # Low-level conditioned policy
    a = low_level_policy(state, goal)
    s_next, r_ext, done = env.step(a)
    # Low-level intrinsic reward
    r_int = -np.linalg.norm(s_next - (state + goal))
    low_buffer.add(state, a, r_int, s_next)
    if step % c == 0:
        # High-level reward is the c-step cumulative external reward
        high_buffer.add(state, goal, ext_reward_sum, s_next_c)
    update(low_level_policy, low_buffer)
    update(high_level_policy, high_buffer, goal_transition=transition_fn)
```

### Comparing hierarchical RL algorithms

| Algorithm | High-level output    | Low-level target          | Training scheme       | Main issue              |
| --------- | -------------------- | ------------------------- | --------------------- | ----------------------- |
| Options   | option id            | fixed sub-policy          | SMDP-Q                | requires preset options |
| FeUdal    | latent direction $g$ | worker's intrinsic reward | on-policy, end-to-end | unstable training       |
| HIRO      | state displacement   | state matching            | off-policy            | goal-relabeling design  |

::: warning The practical trouble with hierarchical RL
Hierarchical RL sounds elegant, but industrial adoption has been limited. The reasons: (1) the hierarchy itself is a strong inductive bias, and a mismatched one hurts performance rather than helping it; (2) coupled training of the high and low levels easily collapses into a "mutual deception" local optimum — the manager emits meaningless directions and the worker learns to ignore them; (3) in the LLM era, "hierarchy" has largely moved from network architecture into the prompt layer (plan-then-act, ReAct), which is much easier to debug. Still, the underlying ideas run deep through agentic RL ([Chapter 23](../chapter22_agentic/tool-use-and-trajectory)) and [Chapter 38, Multi-Agent](../chapter32_selfplay/llm-multi-agent-rl/).
:::

## Generative world models as an RL environment

The first three sections of this chapter all worked on the same question — how to make an agent explore a _given_ environment more efficiently. This last section changes the angle: **what happens to exploration when the environment itself is a learned artifact?**

### From Dreamer to Genie

[Chapter 9, Dreamer V3](../chapter11_continuous_control/intro#_12-7-dreamer-v3-a-new-generation-of-world-models) already showed that "training an actor-critic inside a world model" works: first train an RSSM world model on real data, then optimize the policy on imagined trajectories. Dreamer's world model is still task-specific — trained on a particular Atari game or MuJoCo environment.

Genie (Bruce et al. 2024) pushes world models into a new, **generative, cross-task** stage. Given a video clip or a single image, Genie learns an interactive "game engine": feed in an action, and the model generates the next frame. That has three consequences:

- **Environment data can come straight from internet video**, with no dependence on a game engine or physics simulator
- **A single model can generate many different environments**, generalizing across tasks
- **RL training can happen inside the generated environment**, with no real physics engine required

Genie 3 goes a step further, introducing **latent action** learning: the model automatically discovers the latent control variables in the video that "cause the next frame to change," with no action labels of any kind. Formally:

$$z_t = \text{LatentAction}(x_t, x_{t+1}),\quad x_{t+1} = \text{Decoder}(x_t, z_t)$$

The learned $z_t$ can serve as the action space for RL, so that an agent trained inside a Genie-generated environment can transfer to real control tasks. This sits at the intersection of model-based RL ([Chapter 9](../chapter11_continuous_control/intro#_12-5-model-based-rl-learning-the-environment-model)), video generation models, and exploration-exploitation theory.

### Exploration, multi-agent, and hierarchy in the new paradigm

Once the world model is treated as a generatable environment, the three themes of this chapter recombine in new ways:

1. **Exploration**: intrinsic reward can act on the generated environment's latent space rather than pixel space — ICM's "forward-prediction error" is, at bottom, just the world model's own training loss
2. **Multi-agent**: Genie-style models can generate environments populated with NPCs, letting multiple agents run self-play inside a generated environment ([Chapter 38, self-play](../chapter32_selfplay/self-play-outlook/))
3. **Hierarchy**: a high-level policy can output "latent subgoals" directly, which the world model then decodes into environment state changes — effectively an implicit way of learning options

The industrial impact is already visible: DeepMind's SIMA (Scalable Instructable Multi-World Agent) trains a general-purpose agent inside Genie-generated multi-game environments; LLM agents like Tongyi DeepResearch are beginning to use LLM-self-generated "code world models" as training environments ([Chapter 37, LLM-driven scientific discovery](../chapter32_selfplay/llm-driven-discovery)). The world model is graduating from "training aid" to "the training environment itself" — one of the deepest shifts in RL over 2024-2026.

## Chapter summary

This chapter covered three rescue routes for when classical deep RL assumptions break down:

1. **Sparse reward → intrinsic reward**: ICM uses forward-prediction error, RND uses random network distillation; NGU combines a short-horizon episodic bonus with a long-horizon RND bonus, and Agent57 adaptively switches between exploration and exploitation, becoming the first algorithm to exceed human performance across all of Atari-57
2. **Multi-agent non-stationarity → CTDE**: MADDPG gives each agent its own centralized critic, MAPPO shares a critic with on-policy clipping, and both are the de facto standards on SMAC and Hanabi
3. **Long horizon → hierarchy**: the options framework, FeUdal Networks' end-to-end manager-worker design, and HIRO's off-policy goal relabeling all let the high level worry only about the sequence of subgoals

These three routes converge again in the LLM era: an agentic RL tool call is, at bottom, an option; multi-agent collaboration is, at bottom, CTDE; the world knowledge embedded in an LLM is, at bottom, a Genie-style generative environment. The next chapter, [Chapter 14, the RLHF training pipeline](../chapter15_rlhf/intro), moves into the main line of large-model alignment, where the "environment" is the LLM itself — but the ideas of exploration, multi-agent, and hierarchy from this chapter run through every LLM RL chapter that follows.

## Further reading

- [Pathak et al. 2017 "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)](https://arxiv.org/abs/1705.05363)
- [Burda et al. 2018 "Exploration by Random Network Distillation" (RND)](https://arxiv.org/abs/1810.12894)
- [Badia et al. 2020 "Agent57: Outperforming the Atari Human Benchmark"](https://arxiv.org/abs/2003.13350)
- [Lowe et al. 2017 "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)](https://arxiv.org/abs/1706.02275)
- [Yu et al. 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO)](https://arxiv.org/abs/2103.01955)
- [Vezhnevets et al. 2017 "FeUdal Networks for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1703.01161)
- [Nachum et al. 2018 "Data-Efficient Hierarchical Reinforcement Learning" (HIRO)](https://arxiv.org/abs/1805.08296)
- [Bruce et al. 2024 "Genie: Generative Interactive Environments"](https://arxiv.org/abs/2402.15391)
