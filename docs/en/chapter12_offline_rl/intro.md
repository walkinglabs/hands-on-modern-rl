---
title: 10. Offline Reinforcement Learning
---

# Chapter 10 · Offline Reinforcement Learning and the Decision Transformer

> [Chapter 9](../chapter11_continuous_control/intro) solved the problems of continuous actions and sample efficiency: DDPG/TD3/SAC reuse historical data through a replay buffer, and model-based RL cuts down on real interaction by learning an environment model. But all of these algorithms still let the agent **keep interacting with the environment** — the data in the replay buffer was collected by an older policy, and fresh data from the current policy keeps flowing in. This chapter takes on a stricter setting: **when the agent cannot interact at all, and can only learn from a fixed historical dataset, how do you train a reliable policy?** This is **Offline RL**, also called batch RL. It is the core paradigm behind LLM post-training, recommender systems, medical decision-making, and industrial robotics, and through one branch of it — **Decision Transformer** — it connects directly to modern sequence modeling (GPT).

## 12.1 Core Challenges of Offline RL and Distribution Shift

[Chapter 5 DQN](../chapter07_dqn/intro) and [Chapter 10 SAC](../chapter11_continuous_control/intro) both rely on the same mechanism: the Bellman backup. Whether on-policy or off-policy, the value function update is written as:

$$y = r + \gamma \cdot \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[V(s')\right]$$

In online RL, the $V(s')$ inside that target is backed by future exploration — even if the new policy wanders into a state it has never seen, the agent keeps interacting with the environment, collects new data, and corrects the estimate. **Offline RL has no such safety net.** The dataset $\mathcal{D} = \{(s, a, r, s')\}$ was collected by some behavior policy $\pi_\beta$, and during training it is **completely frozen**:

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}, \quad (s, a) \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$$

Once trained, the new policy $\pi_\theta$ gets deployed, but the action distribution it chooses, $\pi_\theta(a \mid s)$, differs from $\pi_\beta(a \mid s)$. This is where **distribution shift** comes from.

### A Formal Definition of Extrapolation Error

Fujimoto et al. (2019), in the BCQ paper, pinned down precisely where offline RL fails. Define the dataset's support as $\mathcal{D}_\mathcal{A}(s) = \{a : (s, a) \in \text{support}(\pi_\beta(\cdot \mid s))\}$. The Bellman operator's value at $a' \notin \mathcal{D}_\mathcal{A}(s')$ has no supervision signal behind it at all — the network **extrapolates** at these OOD (out-of-distribution) points, and the result is arbitrary.

Break the value estimation error down into three sources:

$$\underbrace{Q_\phi(s, a) - Q^\pi(s, a)}_{\text{total error}} = \underbrace{\epsilon_{\text{stat}}}_{\substack{\text{statistical error}\\\text{(finite samples)}}} + \underbrace{\epsilon_{\text{approx}}}_{\substack{\text{function approximation error}\\\text{(network capacity)}}} + \underbrace{\max_{a'} Q_\phi(s', a') - Q^\pi(s', \pi(s'))}_{\text{extrapolation error}}$$

The third term is the key one. Q-Learning's target uses $\max_{a'} Q(s', a')$, and on OOD actions, extrapolation can give $Q$ **inflated values**, which then pull the policy toward these "hallucinated" actions.

The accumulation of extrapolation error can be unrolled recursively. Let $Q_0$ be the initial estimate; after $T$ Bellman iterations, the error satisfies:

$$\|Q_T - Q^\pi\|_\infty \leq \gamma^T \|Q_0 - Q^\pi\|_\infty + \sum_{k=0}^{T-1} \gamma^k \|\mathcal{T} Q_k - \mathcal{T}^\pi Q_k\|_\infty$$

Here $\mathcal{T}$ is the data-constrained Bellman operator (with the max), and $\mathcal{T}^\pi$ is the true policy operator. If the max operator produces an error $\epsilon_{\text{ood}}$ on OOD actions at every step, the per-step error accumulates with a factor of $\sum \gamma^k \approx 1/(1-\gamma) \approx 100$ (at $\gamma = 0.99$). In online RL, the very next interaction immediately exposes this mistake (the actual reward comes back low), and $Q$ gets pulled back in line. Offline RL has no such correction mechanism, so the error **compounds exponentially** through the Bellman iterations.

::: warning Why More Data Doesn't Save You
Intuitively, expanding dataset coverage should ease the OOD problem. In continuous action spaces, though, no matter how much data you collect, $\mathcal{D}_\mathcal{A}(s)$ remains a sparse support inside a $|\mathcal{A}|$-dimensional space. The Euclidean distance from $a'$ to the nearest data point can be tiny, yet the $Q$ function's gradient in that direction can be arbitrarily large. **Extrapolation error is a structural flaw in combining Q-Learning's max operator with a function approximator — it has nothing to do with how much data you have.**
:::

### The Offline RL Objective

With this diagnosis in hand, we can formalize the offline RL objective: learn a policy $\pi_\theta$, supported on the dataset, whose expected return is as large as possible, while $\pi_\theta$ **stays close to $\pi_\beta$** — straying too far pushes it into OOD territory. Every modern offline RL algorithm is a balancing act between these two goals:

$$\max_\theta \; \mathbb{E}_{s \sim \mathcal{D}}\left[Q^\pi(s, \pi_\theta(s))\right] \quad \text{subject to} \quad D(\pi_\theta \| \pi_\beta) \leq \epsilon$$

The next three sections lay out three different routes to enforcing this constraint.

## 12.2 The Pessimism Route and CQL / IQL / BCQ

The most direct idea: **make the Q function pessimistic about OOD actions**. If $Q(s, a)$ assigns a low value to actions $a$ it has never seen, $\max_a Q(s, a)$ naturally won't pick a hallucinated action. Three classic algorithms — BCQ, CQL, and IQL — implement this principle from different angles.

### Constraining the Action Space

Batch-Constrained Q-Learning (Fujimoto et al., 2019) was the first deep algorithm proven to be stable on continuous-action offline data. Its core constraint: **the target action $a'$ must fall inside the support of $\pi_\beta$**.

BCQ trains a conditional VAE $\pi_\beta(a \mid s)$ to approximate the behavior policy, samples candidate actions $\{a_i\} \sim \pi_\beta$, and then maximizes over these candidates:

$$a' = \arg\max_{a \in \{a_i + \xi \Phi(s, a_i)\}} Q_\phi(s', a)$$

Here $\Phi(s, a)$ is a perturbation network that makes small corrections to the sampled action to approach a local optimum, and $\xi$ is the perturbation magnitude. This confines the "continuous-action argmax" to the high-density region of the behavior policy.

### Pessimism at the Value-Function Level

Conservative Q-Learning (Kumar et al., 2020) attacks the problem from a different angle: instead of constraining the action, it **directly penalizes the Q value on OOD actions**. It adds a regularization term on top of the standard Bellman error:

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]\right) + \mathcal{L}_{\text{Bellman}}(Q)$$

The first term, $\log \sum_a \exp(Q(s, a))$, is a logsumexp: a soft maximum taken over the Q values of **all actions** (OOD included). The only way to shrink it is to push down the Q values of every action. The second term pulls the Q values of $(s, a)$ pairs actually seen in the dataset back into a normal range. The gap between the two forms a "penalty gap" — OOD actions get their Q values systematically underestimated.

CQL's theoretical guarantee: the learned $\hat{Q}$ is a **lower bound** on the true $Q^\pi$, i.e., $\hat{Q}(s, a) \leq Q^\pi(s, a)$ holds for all $(s, a)$; it can further be shown that $\hat{Q}$ on OOD actions sits an $\mathcal{O}(\alpha)$ gap below its value on in-distribution actions. As a result, the policy derived from $\hat{Q}$ never overestimates the return of any action. In practice, $\alpha$ is tuned automatically via a Lagrangian, keeping the conservatism just right:

$$\mathcal{L}(\alpha) = -\alpha \cdot \left(\mathbb{E}_s\left[\log\sum_a \exp(\hat{Q}(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[\hat{Q}(s, a)] - \xi\right)$$

Here $\xi$ is the target gap (e.g., 5.0). When the actual gap falls below $\xi$, $\alpha$ increases; otherwise it decreases — this automatically stabilizes the gap near the target.

```python
class CQL(SAC):
    def critic_loss(self, batch):
        s, a, r, s_next, done = batch
        # Standard Bellman error (inherited from SAC)
        with torch.no_grad():
            a_next = self.actor(s_next)
            q_target = torch.min(self.critic_target1(s_next, a_next),
                                  self.critic_target2(s_next, a_next))
            y = r + self.gamma * (1 - done) * q_target
        bellman_loss = F.mse_loss(self.critic1(s, a), y) + \
                       F.mse_loss(self.critic2(s, a), y)

        # CQL conservative regularizer
        # First term: logsumexp over random actions (OOD)
        rand_a = torch.rand_like(a) * 2 - 1
        q_rand1 = self.critic1(s, rand_a).flatten()
        q_curr1 = self.critic1(s, a).flatten()  # in-dist
        q_next1 = self.critic1(s, a_next).flatten()
        cat_q1 = torch.cat([q_rand1, q_curr1, q_next1], dim=1)
        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()

        conservative_loss = \
            self.alpha * (logsumexp_q1 - q_curr1.mean()) \
            + self.alpha * (logsumexp_q2 - q_curr2.mean())

        return bellman_loss + conservative_loss
```

### Avoiding Explicit OOD Evaluation

Implicit Q-Learning (Kostrikov et al., 2022) pushes the insight one level deeper: **you don't need to evaluate the Q value of any OOD action at all**. It learns $V(s)$ with quantile regression, biasing $V$ toward the better actions present in the data:

$$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L_2^\tau(Q_{\bar{\theta}}(s, a) - V_\psi(s))\right]$$

Here $L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)| \cdot x^2$ is the quantile loss for expectile $\tau$ (typically $\tau = 0.7$). This trains $V$ to represent "the value of the better actions in the data," **without ever taking a max over anything**. The policy is then trained with advantage-weighted regression, using the advantage $A(s, a) = Q_{\bar{\theta}}(s, a) - V_\psi(s)$:

$$\mathcal{L}_\pi = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\theta(a \mid s)\right]$$

$\exp(\beta A)$ assigns a larger weight to actions in the data that performed well, pulling $\pi_\theta$ toward them; $\beta$ is the temperature. IQL sidesteps Q-Learning's max operator entirely, so **it never produces extrapolation error** — this is the essential difference between IQL and CQL.

### Comparing the Three Algorithms

| Dimension                            | BCQ                       | CQL                          | IQL                       |
| ------------------------------------ | ------------------------- | ---------------------------- | ------------------------- |
| Where the constraint applies         | Action space              | Value function               | Implicit (quantile + AWR) |
| Evaluates OOD actions?               | No (sampling constraint)  | Yes (logsumexp)              | No (fully avoided)        |
| Extra network                        | VAE $\pi_\beta$           | None                         | $V$ network               |
| Hyperparameter sensitivity           | High (perturbation scale) | Medium ($\alpha$ auto-tuned) | Low ($\tau, \beta$)       |
| Performance on medium-sized datasets | Medium                    | Strong                       | Strong                    |
| Stability on sparse datasets         | Medium                    | Occasionally unstable        | Strong                    |
| Implementation complexity            | High                      | Medium                       | Low                       |

**Practical advice**: start with IQL (most stable, least tuning); if the baseline comes in low, switch to CQL (more aggressive); BCQ is rarely used as a new baseline these days.

## 12.3 AWAC, TD3+BC, and Conservative Constraints via Behavior-Cloning Regularization

Another route is more engineering-driven: **keep the on-policy / off-policy actor-critic main loop, and add a behavior-cloning (BC) regularizer directly to the policy loss**. The advantage of this family of methods is compatibility with the PPO/SAC framework from [Chapter 9](../chapter11_continuous_control/intro) — the engineering changes required are minimal.

### TD3+BC and the Simplest Form of BC Regularization

TD3+BC, proposed by Fujimoto & Gu (2021), takes this idea to its logical extreme: it adds a BC term to TD3's actor loss, with an adaptively tuned weight $\lambda$:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q(s, \mu_\theta(s))\right] + \lambda \cdot \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[(\mu_\theta(s) - a)^2\right]$$

Here $\lambda = \frac{\alpha}{\frac{1}{N}\sum_i |Q(s_i, \mu_{\theta_{\text{old}}}(s_i))|}$. The denominator is the current scale of the Q values — it makes $\lambda$ automatically adapt to different environments' reward scales, with no tuning required. In the paper, $\alpha = 2.5$ is used as a single fixed setting across every D4RL MuJoCo task.

TD3+BC's simplicity makes it a strong offline RL baseline. Its performance points to a counterintuitive fact: **on many offline RL benchmarks, the plainest BC regularization gets you performance close to CQL/IQL**.

### Advantage-Weighted BC

Advantage-Weighted Actor-Critic (Nair et al., 2020) shares its policy loss's origin with IQL — advantage-weighted regression — but AWAC uses an explicit Q instead of a quantile-regressed V:

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\underbrace{\exp\left(\frac{A(s, a)}{\beta}\right)}_{\text{advantage weight}} \cdot \log \pi_\theta(a \mid s)\right]$$

Here $A(s, a) = Q(s, a) - V(s)$, and $\beta$ is the temperature. Intuitively: actions in the data that perform better than average get their weight amplified, and those below average get suppressed. AWAC generalizes BC into "weighted BC" — imitate only the good part of the data.

AWAC's engineering highlight is that it **supports a smooth offline-to-online transition**: pretrain purely offline, then fine-tune with a small amount of online interaction. This is highly practical for real robots, recommender systems, and similar settings.

### The Shared Origin of AWAC's and IQL's Policy Losses

Compare the two formulas closely:

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}\left[\exp\left(\frac{A(s, a)}{\beta}\right) \log \pi(a \mid s)\right], \quad \mathcal{L}_\pi^{\text{IQL}} = -\mathbb{E}\left[\exp\left(\beta \cdot A(s, a)\right) \log \pi(a \mid s)\right]$$

The two are almost identical in form ($\beta$ sits in a different place, but both function as a temperature). The difference is in how $A(s, a)$ is estimated:

- **AWAC**: $A = Q_\phi(s, a) - V_\psi(s)$, where $Q$ still goes through a standard Bellman backup (the target still has a max over $\pi$)
- **IQL**: $A = Q_\phi(s, a) - V_\psi(s)$, but $Q$ is backed up through $V$ (the target uses $V(s')$ instead of $\max_a Q(s', a)$), and $V$ is trained with quantile regression biased toward the better actions in the data

By replacing the Bellman target with $V(s')$ (no more max), IQL removes the source of extrapolation error at its root. AWAC keeps the standard Bellman target and relies on weighted BC to constrain the policy — a weaker constraint than IQL's implicit one, which is why AWAC is more prone to stepping into OOD territory when the dataset's Q values are noisy.

### AWAC vs TD3+BC vs IQL

| Method | Policy loss form                        | Needs $V$? | Online fine-tuning friendliness |
| ------ | --------------------------------------- | ---------- | ------------------------------- |
| TD3+BC | $-\!Q + \lambda \|\mu - a\|^2$          | No         | Medium                          |
| AWAC   | $-\!w(A) \log \pi$, $w = \exp(A/\beta)$ | Yes        | Strong                          |
| IQL    | $-\!\exp(\beta A) \log \pi$ (AWR)       | Yes        | Medium                          |

Notice how similar AWAC's and IQL's policy loss structures are — the difference lies in where $A$ comes from: AWAC uses the explicit Q-V difference, while IQL estimates it implicitly through quantile regression. This subtle difference has a large effect on stability with sparse data.

## Section Summary

This section laid out offline RL's core challenge — distribution shift and extrapolation error — and three conservative routes for handling it: BCQ constrains the action space, CQL penalizes OOD Q values, and IQL avoids the max operator entirely. All of these algorithms work within the Bellman framework.

The next section, [12.2 Decision Transformer, Trajectory Transformer, and Diffuser](./sequence-modeling), takes a different route entirely — dropping Bellman altogether and recasting RL as conditional sequence generation.
