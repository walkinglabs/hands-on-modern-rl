# Chapter 11 · Imitation Learning, Inverse RL, and Meta-RL

> [Chapter 10, Offline Reinforcement Learning](../chapter12_offline_rl/intro) deals with the setting of "only historical data, no interaction," but it still assumes the data carries an explicit reward signal. This chapter deals with two more extreme situations: (1) **there is no reward function at all** — only expert demonstration trajectories. What do you do? (2) **the environment itself keeps changing** — the agent has to learn to "adapt quickly to new tasks." The first leads to **Imitation Learning (IL)** and **Inverse RL**, the second leads to **Meta-RL**. The two ultimately converge in the LLM era: SFT is, at its core, behavior cloning; the three stages of InstructGPT can be rewritten as BC + RL + RL; and In-Context RL reveals that "the RL algorithm itself can be distilled into a transformer."

## 13.1 Behavior Cloning and DAgger

[Chapter 6, Policy Gradients](../chapter08_policy_gradient/reinforce) assumes the environment provides a reward. But in many real tasks, all we have are **expert demonstrations** — trajectories from a human driver, operation logs from a skilled worker, high-quality question-answer pairs. **Imitation learning** learns a policy directly from demonstrations, skipping the design of a reward function altogether.

### The supervised-learning view

The simplest approach treats imitation learning as supervised learning: treat $(s_t, a_t)$ as an (input, label) pair, and minimize the negative log-likelihood:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log \pi_\theta(a \mid s)\right]$$

where $\mathcal{D}_{\text{expert}} = \{(s_i, a_i)\}_{i=1}^N$ is the expert demonstration dataset. This is exactly the standard loss used for supervised fine-tuning (SFT) in LLMs.

```python
def behavior_cloning_step(policy_net, expert_batch):
    states, actions = expert_batch
    log_probs = policy_net.log_prob(states, actions)
    loss = -log_probs.mean()  # negative log-likelihood
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### The fatal flaw in BC

During BC training, the states come from the expert's state distribution $d_{\text{expert}}(s)$, but at deployment the agent visits states from its own distribution $d_{\pi_\theta}(s)$. As soon as the policy makes a small mistake at some step, the state drifts away from the expert's trajectory and lands in **territory the policy has never seen** — and the probability of a further mistake at the next step only goes up. The error **accumulates exponentially**.

Formalize it: suppose the per-step error probability is $\epsilon$. After $T$ steps, the probability of still being near the expert's distribution is roughly $(1-\epsilon)^T \to 0$. The DAgger paper (Ross et al. 2011) proves an upper bound on BC's expected error:

$$\mathbb{E}\left[\sum_{t=0}^T \mathbb{1}[\pi_\theta(s_t) \neq \pi^*(s_t)]\right] \leq O(T^2 \epsilon)$$

The error grows **quadratically** with the horizon. This is exactly why pure behavior-cloned self-driving is nearly unusable on long-horizon tasks.

### DAgger: iteratively collecting "failure states"

The core insight of Dataset Aggregation: instead of letting the agent struggle in states the expert has never seen, **actively collect those failure states and have the expert label them**.

```python
def dagger(env, expert, policy_net, n_iterations=20, n_traj_per_iter=50):
    dataset = []
    for it in range(n_iterations):
        # 1. Roll out with the CURRENT policy (note: not the expert!)
        trajectories = []
        for _ in range(n_traj_per_iter):
            s = env.reset()
            traj = []
            done = False
            while not done:
                # beta mixing: rely more on the expert early on for safety,
                # rely more on the policy as training progresses
                beta = max(0.0, 1.0 - it / 10)
                if np.random.rand() < beta:
                    a = expert(s)
                else:
                    a = policy_net.act(s)
                s_next, r, done, _ = env.step(a)
                traj.append((s, a))
                s = s_next
            trajectories.append(traj)

        # 2. Key step: have the expert re-label the states the POLICY visited
        # (including failure states)
        for traj in trajectories:
            for s, _ in traj:
                a_expert = expert(s)
                dataset.append((s, a_expert))

        # 3. Retrain the policy on the expanded dataset
        train_bc(policy_net, dataset)
```

DAgger lets the training distribution gradually approach $d_{\pi_\theta}$ starting from $d_{\text{expert}}$ — **addressing the root cause of the distribution-shift problem**. In theory, DAgger's error bound drops to $O(T \epsilon)$, growing linearly, which is far better than BC's $O(T^2 \epsilon)$.

| Method | Source of training data           | Fixes distribution shift | Needs online expert labeling       |
| ------ | --------------------------------- | ------------------------ | ---------------------------------- |
| BC     | Offline expert data only          | ❌                       | ❌                                 |
| DAgger | Expert + states the policy visits | ✅                       | ✅ (the key limitation)            |
| GAIL   | Expert + policy rollouts          | ✅ (implicitly)          | ❌ (needs only state-action pairs) |

DAgger's engineering bottleneck is that it **needs the expert online, interactively**. It's hard for a human driver to label the correct action, in real time, for the "weird states" an AI generates. This is exactly what motivates the next section's approach: inverse RL, which recovers a reward function from demonstrations instead.

## Section Summary

Behavior cloning (BC) is the simplest form of imitation learning — it trains a policy on expert trajectories treated as supervised data. But it suffers from **distribution shift**: it only ever learns on the expert's state distribution during training, and once the policy drifts away from that distribution at deployment, it can never recover. DAgger solves this by having the expert correct the agent's actual trajectories.

The next section, [13.2 Inverse RL and GAIL](./irl-gail), no longer imitates actions directly — instead it **recovers a reward function from expert behavior**. This is inverse reinforcement learning (IRL).
