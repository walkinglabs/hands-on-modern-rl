---
title: '6.4 Hands-On: Policy Gradient CartPole'
---

# 6.4 Hands-on: REINFORCE on CartPole

> **Section goal**: Train `CartPole-v1` with REINFORCE, observe policy-gradient training in a high-variance task, and connect returns to changes in action probability.

> **Learning path**: [6.1 Policy Gradient and REINFORCE](./reinforce) → [6.3 Dice-Game Bandit](./dice-game) → **6.4 REINFORCE on CartPole**

> **Code and resources**: [reinforce_cartpole.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_cartpole.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/requirements.txt)

In the previous section, we derived the policy gradient theorem and the REINFORCE algorithm. The bandit example showed the simplest possible setting: no state, a single step, and only two actions.

Now we switch to a more representative task: `CartPole-v1`. A cart can be pushed left or right; the goal is to keep the pole upright for as long as possible. For every time step you survive, the environment gives a `+1` reward. If the pole tilts too far or the cart goes out of bounds, the episode ends.

Unlike a bandit, CartPole has state (a 4D vector: cart position, cart velocity, pole angle, and pole angular velocity), multi-step decisions, and a clear notion of failure. The action space is still discrete (left/right), but it is already enough to expose the high-variance behavior of REINFORCE.

## 6.4.1 Run Training

Install dependencies first:

```bash
pip install -r code/chapter05_policy_gradient/requirements.txt
```

Then run training:

```bash
python code/chapter05_policy_gradient/reinforce_cartpole.py
```

This script trains a REINFORCE policy for 500 episodes. The core code has only three steps:

```python
# Step 1: Roll out one full episode with the current policy
states, actions, rewards, episode_reward = collect_episode(policy, env)

# Step 2: Compute discounted returns G_t from the end of the episode backwards
returns = compute_returns(rewards, gamma=0.99)

# Step 3: Policy gradient update
loss = -(log_probs * returns_tensor).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

That is the entirety of REINFORCE: finish an episode, compute $G_t$ at every step, and update the policy with `loss = -log_prob * G_t`.

After the run ends, the script will generate training curves under `output/`.

## 6.4.2 Read the Training Curve

![Training curve of REINFORCE on CartPole-v1: episodic return over training progress](../../chapter08_policy_gradient/images/reinforce-cartpole-reward.png)

In practice, the curve usually has the following characteristics:

**Early stage (episode 0–50)**: The policy is close to random. The pole falls quickly, and episodic return fluctuates between 10 and 30. At this point the policy network is nearly uniform: "try left and right and see what happens."

**Middle stage (episode 50–200)**: If you get lucky, one episode may survive for much longer than usual. Then $G_t$ becomes large for many time steps, and the policy will strongly reinforce the actions along that trajectory. The return begins to trend upward, but the variance is severe: after a good episode you may still see a streak of bad ones.

**Late stage (episode 200–500)**: The policy gradually converges toward a relatively stable balancing strategy, and episodic return may reach 100–200. But the curve still shows noticeable drops. That is the most direct symptom of high variance.

## 6.4.3 What High Variance Looks Like

CartPole reveals variance issues more clearly than a bandit because each episode lasts tens to hundreds of steps. The policy may make a good choice at some time step, but what happens later can still be dominated by sampling luck. The return $G_t$ compresses the randomness of the whole future trajectory into a single scalar. As a result, it reflects not only how good the current action was, but also how lucky (or unlucky) the later steps turned out to be.

On the training curve, this shows up as:

- **Sudden reward spikes**: You might sample one especially good trajectory and see episodic return jump to 200+, only to fall back to 30 in the next episode. The policy is pushed hard by one lucky episode, and then pulled back by the next unlucky one.
- **Unstable learning**: With the same hyperparameters, different random seeds can lead to very different outcomes. Sometimes 500 episodes is enough for a decent policy; sometimes even 1000 is not.
- **Sensitivity to learning rate**: If the learning rate is too large, the policy oscillates between "good" and "bad" behaviors. If it is too small, the policy barely changes. The workable window can be narrow.

These symptoms are all manifestations of the same underlying issue: REINFORCE uses $G_t$ to decide whether "this action was good," but $G_t$ is too noisy. A truly good action can be penalized because later steps went poorly by chance; a mediocre action that happened to appear in a lucky trajectory can be over-reinforced.

## 6.4.4 Key Implementation Details

**Computing discounted returns**: a backward recursion:

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G  # G_t = r_t + γ * G_{t+1}
        returns.insert(0, G)
    return returns
```

Start from the final time step: $G_T = r_T$. Step one time step earlier: $G_{T-1} = r_{T-1} + \gamma G_T$. Continue backward. This recursion ensures that each $G_t$ contains all discounted rewards from time $t$ to the end of the episode.

**Sampling by probability, not taking argmax**: a key difference between policy gradients and DQN:

```python
probs = policy(state_tensor)
dist = torch.distributions.Categorical(probs)
action = dist.sample()  # sample according to the probabilities
```

DQN chooses `argmax Q`, which makes the policy (mostly) deterministic. REINFORCE samples from a probability distribution, so exploration is built in. If the network believes an action is worth trying with probability 60%, then it will be tried about 60% of the time.

**What on-policy means**: REINFORCE must use data generated by the current policy, and then discard it:

```python
# For every episode, collect fresh data again
states, actions, rewards, episode_reward = collect_episode(policy, env)
```

DQN can reuse old data in a replay buffer. But in REINFORCE, the expectation $\mathbb{E}_{\pi_\theta}$ in the gradient estimator requires data generated by the current policy $\pi_\theta$. Once the policy is updated, old data no longer matches. This is one reason policy gradient methods are typically less data-efficient than DQN.

## 6.4.5 Back to the Variance Problem

The CartPole experiment shows that REINFORCE can learn, but it often learns unstably. The root cause is the large variance of $G_t$.

The policy gradient theorem has a useful property: you can subtract a baseline $b(s_t)$ that does not depend on the action from the gradient estimator, changing the update signal from $G_t$ to $G_t - b(s_t)$. This does not change the expected direction of the gradient, but it can drastically reduce variance.

The math behind this baseline is explained in [6.2 Policy Gradient Improvements](./pg-improvements); the next section runs the [value-baseline CartPole experiment](./baseline-experiment).

## Section Summary

This lab connects the runnable experiment to the main idea of the section. Use the reported metrics together with replay or task-level evaluation, and keep conclusions within the conditions that were actually tested.
