---
title: Brief History of RL
---

# A Brief History of Reinforcement Learning

If we asked an AI researcher in the early 2010s "what is reinforcement learning," they would probably sketch you a feedback loop of an agent interacting with an environment, and tell you it's mainly used for robot control and board games. But wind the clock back a century, or fast-forward to today's era of large models, and you'll find that reinforcement learning (RL) has gone through a sweeping evolution — starting from psychologists' animal experiments and growing, step by step, into the core engine driving today's most advanced AI systems.

Before we start writing code, it's worth spending a few minutes reviewing this century-spanning history. Knowing these milestones will help you understand why modern RL algorithms are designed the way they are.

## 1. Origins and Foundations: From Psychology to Mathematical Frameworks (1890s - 1950s)

The idea behind reinforcement learning didn't originate in computer science — it came from **psychology and neuroscience**.

In 1898, psychologist Edward Thorndike used his famous "puzzle box" experiments with cats to propose the **Law of Effect**: if a behavior produces a good outcome, that behavior is reinforced; if not, it's weakened. This is the origin of "trial-and-error learning."

![Thorndike's Puzzle Box](../../../preface/brief-history/images/puzzle_box.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: Thorndike's puzzle box. Source: <a href="https://commons.wikimedia.org/wiki/File:Original_%22Puzzle_Box%22_Apparatus_Design.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

More than half a century later, as cybernetics rose to prominence, this biological instinct began to be rigorously mathematized. In 1957, Richard Bellman introduced the **Markov Decision Process (MDP)** and the **Bellman Equation** [^1]. He used a five-tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ to abstract real-world sequential decision problems into a precise mathematical object — a state set $\mathcal{S}$, an action set $\mathcal{A}$, a transition probability $P(s'|s,a)$, a reward function $R(s,a)$, and a discount factor $\gamma$. Under this framework, the agent's goal is to find a policy $\pi(a|s)$ that maximizes the expected long-term cumulative discounted reward:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

To measure "how good a policy actually is," Bellman introduced the concept of the **value function** — $V^\pi(s)$ denotes the expected cumulative reward obtained by starting from state $s$ and always following policy $\pi$. The best policy among all possible policies corresponds to the **optimal value function** $V^*(s)$. Bellman proved that it satisfies an elegant recursive relationship — the **Bellman optimality equation**:

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \, V^*(s') \right]$$

The meaning of this equation runs deep: the optimal value of the current state equals the "immediate reward" plus the discounted expectation of the "optimal value of all possible future states." It turns what looks like an endless sequential decision problem into an equation that can be solved by recursion — this is the conceptual root of **dynamic programming**. This marked the moment reinforcement learning acquired a solid theoretical foundation.

## 2. Theory Takes Shape: Temporal-Difference Learning and Model-Free RL (1980s - 1990s)

Bellman's dynamic programming is mathematically airtight, but in practice it runs into two fatal limitations. **First, it requires complete knowledge of the environment's model** — the transition probability $P(s'|s,a)$ and the reward function $R(s,a)$ must be given in advance. In reality, a robot doesn't know how wide the hallway is behind a door it just pushed open, and an AI doesn't know what move its opponent will make next. **Second, it runs headlong into the "curse of dimensionality"** — the Bellman equation needs to be solved for every state individually, and the size of the state space grows exponentially with problem complexity. Take Go as an example: the number of board states is roughly $3^{361} \approx 10^{170}$, so far beyond reach that even every atom in the universe wouldn't be enough to store a lookup table. To let an agent learn in an **unknown environment**, **without relying on a complete state table**, pioneers began searching for a new way forward.

- **In 1988**, Richard Sutton, often called the "father of reinforcement learning," systematically proposed **Temporal-Difference (TD) learning** [^2]. It cleverly combines Monte Carlo sampling with the bootstrapping property of dynamic programming, letting an agent learn as it goes without needing a complete model of the environment. TD's core update rule is remarkably simple:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma V(s_{t+1}) - V(s_t)}_{\text{TD error } \delta_t} \right]$$

Here $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is called the **TD error**. Intuitively, it measures the gap between the "new estimate" and the "old estimate" — if things turn out better than expected after taking the next step ($\delta_t > 0$), the value of the current state gets nudged up; otherwise it gets nudged down. This "learn as you go" mechanism is one of the most central ideas in modern RL.

- **In 1989**, Chris Watkins introduced the now-famous **Q-learning** algorithm in his doctoral thesis [^3]. It's a model-free, off-policy algorithm, and it's still the first thing most people learn when starting RL. Its update rule is:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

What makes Q-learning elegant is that it directly learns the **action-value function** $Q(s,a)$ — exactly "how much it's worth" to take action $a$ in state $s$. Once you have this scoring table, the agent can make an optimal decision simply by greedily picking the highest-scoring action, $\arg\max_a Q(s,a)$, in every state.

- **In 1992**, IBM's Gerald Tesauro developed **TD-Gammon** [^4]. By combining TD learning with a shallow neural network, it reached the level of a human world champion at backgammon. This was an early landmark success of combining neural networks with RL.

![TD-Gammon / Backgammon](../../../preface/brief-history/images/backgammon.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 2: Backgammon, the classic game conquered by TD-Gammon. Source: <a href="https://commons.wikimedia.org/wiki/File:Backgammon_lg.jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

In 1998, Sutton and Barto published the field-defining textbook _Reinforcement Learning: An Introduction_ [^5], and the discipline of modern reinforcement learning was formally consolidated.

## 3. The Deep Learning Revolution: When RL Met Deep Learning (2013 - 2019)

Through the early 2000s, even as RL theory kept maturing, traditional tabular methods and linear function approximation simply couldn't handle the high-dimensional, complex inputs of the real world, like images. It took the explosion of deep learning for RL to have its true breakout moment.

- **In 2013**, DeepMind proposed the **Deep Q-Network (DQN)** [^6], combining deep neural networks with RL for the first time and letting an AI learn to beat human performance on multiple Atari arcade games from raw pixels alone. This formally opened the era of Deep RL. DQN's core idea is to approximate the Q-value function with a neural network $Q(s,a;\theta)$ parameterized by $\theta$, using the loss function:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right)^2 \right]$$

Here $\theta^{-}$ is the parameters of the **target network** (periodically copied from $\theta$ rather than updated every step), and $\mathcal{D}$ is the **experience replay buffer**. These two deceptively simple engineering tricks — the target network and experience replay — completely solved the training-instability problem that arises when you combine deep networks with Q-learning, and they were the key to DQN's success.

![DQN Atari Performance](../../../preface/brief-history/images/dqn_atari.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 3: DQN's performance across dozens of Atari games, surpassing human professional players on most of them. Source: <a href="https://research.google/blog/from-pixels-to-actions-human-level-control-through-deep-reinforcement-learning/" target="_blank" rel="noopener noreferrer">Google Research Blog</a></em>
</div>

- **2016** was a year destined for the history books. DeepMind's **AlphaGo** [^7] combined deep RL with Monte Carlo tree search and beat Go world champion Lee Sedol 4:1. The event didn't just stun the world — it also brought RL into public view for the first time in a truly spectacular way.

![AlphaGo](../../../preface/brief-history/images/alphago.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 4: A screenshot from the match where AlphaGo defeated European Go champion Fan Hui. Source: <a href="https://commons.wikimedia.org/wiki/File:AlphaGo_Fan_Huiren_aurka.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

- **In 2017**, OpenAI proposed **PPO (Proximal Policy Optimization)** [^8]. Compared with earlier policy-gradient methods, which suffered from high variance and fragility, PPO struck an excellent balance between training stability and sample efficiency. Its core idea is to use **clipping** to bound the size of each policy update, preventing training from collapsing because the "step size was too large":

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$

Here $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ is the **probability ratio between the new and old policy**, $\hat{A}_t$ is the **estimated advantage**, and $\epsilon$ is typically set between 0.1 and 0.2. The clipping mechanism guarantees that the policy never strays too far from the old policy after any single update — it's like bolting a "guardrail" onto the learning rate. Because it's easy to tune and remarkably robust, PPO quickly became the industry's default standard algorithm. OpenAI later used a massive distributed system built on PPO, **OpenAI Five**, to defeat the world champion team at Dota 2.

## 4. The LLM Era: New Paradigms for Alignment and Reasoning (2020s - Present)

Just as people started to assume RL's applications were mostly confined to games and robot control, the rise of large language models (LLMs) handed RL a brand-new mission — **alignment** and **reasoning**.

- **In 2022**, OpenAI released ChatGPT. The core technique behind it was **RLHF (Reinforcement Learning from Human Feedback)** [^9]. By training a reward model to mimic human preferences and then using PPO to optimize the language model, RL successfully turned LLMs from "statistical machines that can string together replies" into "assistants with a sense of judgment." RLHF training happens in two steps: first, train a reward model $r_\phi(x, y)$ on human preference data; then use it as the reward signal and optimize the language-model policy $\pi_\theta$ with PPO:

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \, \text{KL}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right) \right]$$

The KL-divergence penalty term $\beta \, \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ ensures the model doesn't drift too far from its original behavior in pursuit of a higher score — this is the key constraint in RLHF that guards against "reward hacking."

![Early ChatGPT UI](../../../preface/brief-history/images/chatgpt.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 5: An example of ChatGPT's early interface. Its 2022 release took RLHF out of post-training research papers and into a real product, marking the moment reinforcement learning began entering the alignment-and-reasoning phase of large models. Source: OpenAI, <a href="https://openai.com/index/chatgpt/" target="_blank" rel="noopener noreferrer">Introducing ChatGPT</a></em>
</div>

- **In 2023**, Stanford and other researchers introduced **DPO (Direct Preference Optimization)** [^10]. They discovered that you could skip the cumbersome step of training a reward model entirely and instead fine-tune a language model directly on human preference data with a simple classification loss. DPO's loss function is derived straight from the RLHF objective:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Here $y_w$ (winner) and $y_l$ (loser) are the human-labeled "good response" and "bad response," and $\sigma$ is the sigmoid function. This formula elegantly cancels out the reward model that RLHF implicitly relies on — the model just needs to learn that "the probability of the good response goes up relative to the bad one, and vice versa." DPO dramatically lowered the engineering barrier for RLHF-style training and swept through the open-source community.

- **From 2024 to 2025**, reasoning models like OpenAI o1 and DeepSeek-R1 [^11] made their stunning debut, and reinforcement learning evolved once again. In particular, **DeepSeek-R1-Zero proved that in settings with clear, objective rules — like math correctness or whether code compiles — you can throw out the traditional SFT (supervised fine-tuning) cold start entirely and run pure reinforcement learning (Pure RL) directly on a base model.** This didn't just break the stereotype that "you have to do SFT before you can do RL" — it also let the model spontaneously develop long chains of thought (CoT) and moments of sudden insight (the "a-ha moment"). The **GRPO (Group Relative Policy Optimization)** algorithm that DeepSeek used strips out the Critic network that traditional PPO relies on, which is extremely memory-hungry, and instead optimizes the policy directly using relative rewards computed within a group. GRPO's core idea: sample a group of responses $\{o_1, o_2, \ldots, o_G\}$ for the same prompt $q$, then normalize the rewards using the group's mean and standard deviation to get an advantage estimate:

$$\tilde{r}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

The policy is then optimized directly with a clipped objective:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_q \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \tilde{r}_i, \; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \tilde{r}_i \right) \right]$$

This lightweight architecture needs no additional Critic network — it drives learning purely from the **relative ranking within the same group of responses** — making it feasible to run pure RL for reasoning at large-cluster scale.

## 5. Industrial-Scale Explosion: The GRPO Family, Reasoning Models, and Agents (2025 - 2026)

If 2024 was the year RLHF and GRPO became well-known concepts, then 2025 to 2026 is when RL truly entered its industrial-scale explosion. Three things happened at once: **the GRPO algorithm family evolved rapidly**, **reasoning models became their own product category**, and **agentic RL moved into production**.

### 5.1 The GRPO Family: Four Independent Lines of Evolution

Within a year of the R1 paper's release, the open-source community and industrial labs proposed at least five influential GRPO variants, each addressing a specific pain point in training:

- **DAPO** (ByteDance and Tsinghua, March 2025, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) tackled the "thinking gets longer and longer" and "low sampling efficiency" problems seen in R1-Zero-style training, introducing four engineering changes: asymmetric clipping (Clip-Higher, setting $\epsilon_{\text{high}} > \epsilon_{\text{low}}$), Dynamic Sampling (filtering out samples that are either all-correct or all-wrong), token-level loss (so long responses don't dominate the gradient), and Overlong Filtering. It surpassed R1-Zero on AIME 2024 using only 50% of the training steps.
- **Dr.GRPO** (Liu et al., 2025, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) found that the standard-deviation normalization and length normalization inside GRPO introduce bias, leading to reward hacking and inflated response lengths. Removing both normalization terms made training more stable.
- **GSPO** (Zheng et al., the Qwen3 team, July 2025, [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) moved the importance-sampling ratio from the token level up to the **sequence level**, designed specifically for training stability under MoE architectures, and became the training foundation for the entire Qwen3 family.
- **CISPO** (MiniMax, June 2025, [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)) rewrote what gets clipped — **clipping the importance-sampling weight instead of the token update** — preserving the gradient contribution of every token, and pairing this with lightning attention for a 2x training speedup.
- **VAPO** (ByteDance Seed, April 2025, [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)) swam against the current — **reintroducing the value model** — showing that a critic network still has irreplaceable value on long-CoT reasoning tasks, reaching an AIME score of 60.4, ahead of every contemporary GRPO variant.

By early 2026, "which GRPO variant should we use" had gone from an open research question to a straightforward selection table.

### 5.2 Reasoning Models and Formal RL

OpenAI's o1 (September 2024), o3 (January 2025), and o4 (April 2025) series established "test-time compute scaling" as a new axis of scaling. **Anthropic's February 2025 paper, _Competitive Programming with Large Reasoning Models_** ([arXiv:2502.06807](https://arxiv.org/abs/2502.06807)), revealed a key fact: o3's sophisticated test-time strategies on IOI and Codeforces weren't hand-designed — they **emerged naturally** from end-to-end RL.

At the same time, DeepMind's **AlphaProof** and **AlphaGeometry 2** (July 2024) won a silver medal at the International Mathematical Olympiad (IMO) by combining the Lean formal language with AlphaZero-style self-play, opening up an entirely new path: **formal RL**. DeepSeek soon followed with **DeepSeek-Prover-V2** ([arXiv:2504.21801](https://arxiv.org/abs/2504.21801)), reaching 88.9% on the MiniF2F benchmark. Because Lean4 acts as a natural verifier, it gives a reward signal with zero false positives, making it a new frontier for PRM (process reward model) research.

### 5.3 Agentic RL Goes into Production

Another major storyline of 2025 was RL expanding from "single-turn question answering" to "long-horizon, multi-step tasks."

- **Anthropic invested $1 billion in RL environments in September 2025** (as [reported by The Information](https://www.theinformation.com/)); data from Wing VC shows the company spends tens of millions of dollars a year on coding and Computer Use environments, with plans to scale that 3-5x by 2026. Karpathy has called this "the new main stage of the LLM training pipeline."
- **Meta's SWE-RL** (February 2025, [arXiv:2502.18449](https://arxiv.org/abs/2502.18449)) trained Llama3-70B on 11 million GitHub pull requests, reaching 41% on SWE-bench Verified and observing an "aha moment" for the first time in this setting.
- **Anthropic's Claude Computer Use** (October 2024) and **OpenAI's Operator** (January 2025) let models directly operate a browser and desktop GUI.
- ByteDance's **UI-TARS-2** ([arXiv:2509.02544](https://arxiv.org/abs/2509.02544)) and Zhipu's **AutoGLM** introduced multi-turn GUI-agent RL and asynchronous rollout training pools.

### 5.4 The Rise of Chinese Labs

Chinese labs occupied a distinctive position in this wave of RL industrialization. **DeepSeek has been the most transparent** — publicly disclosing that V3 pretraining used 2.664M H800 GPU-hours and R1-Zero used 128K GPU-hours ([Stanford CRFM's transparency report](https://crfm.stanford.edu/fmti/)). **Qwen3 made GSPO the new standard, replacing PPO.** **Kimi K2 introduced the MuonClip optimizer** to solve RL training instability ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)). **ByteDance has been the largest contributor to the GRPO improvement family** (a full lineup spanning DAPO, VAPO, UI-TARS, DanceGRPO, and Seedance). **Zhipu's GLM-4.5/4.6/5 series** was the first to make "difficulty-curriculum RL" a mainstream training paradigm ([arXiv:2508.06471](https://arxiv.org/abs/2508.06471)). **StepFun's Step3-VL** proposed PaCoRe, parallel coordinated reasoning, opening up another path to test-time scaling.

In November 2025, **Anthropic's paper "Natural Emergent Misalignment from Reward Hacking"** ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)) pushed reward-hacking research into a new phase — misaligned behaviors that emerge naturally during RL training became a frontier safety topic. That same month, **Microsoft's Reinforcement Pre-Training (RPT)** ([arXiv:2506.08007](https://arxiv.org/abs/2506.08007)) challenged the boundary between pretraining and post-training by bringing RL directly into the pretraining stage. **DeepMind's AlphaEvolve** (May 2025) combined LLMs, evolutionary algorithms, and automated evaluators, discovering a 23% speedup for matrix multiplication — a new paradigm for search algorithms in the LLM era.

RL has traveled from the puzzle box of 1890 to the industrial clusters of 2026 — a span of one hundred thirty years. But its core has never changed: **let an agent try and fail within an environment, guided by cumulative reward alone, and let it work out the optimal policy for itself.**

## Summary

From Thorndike's puzzle box to Bellman's equation; from DQN inside an Atari console to DPO and GRPO iterating at breakneck speed on today's cloud clusters — the history of reinforcement learning is an epic of agents **"learning from the environment, evolving from feedback, and growing from a single machine into superintelligent models."**

Today, reinforcement learning is no longer a theoretical toy locked away in an ivory tower — it's the road that has to be traveled to reach artificial general intelligence (AGI). In the chapters that follow, we'll trace this history's thread and, starting from the very first line of code, build these landmark algorithms with our own hands.

## References

[^1]: Bellman, R. (1957). A Markovian Decision Process. _Journal of Mathematics and Mechanics_, 6(5), 679-684. [DOI](https://doi.org/10.1512/iumj.1957.6.56038)

[^2]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44. [PDF](http://incompleteideas.net/papers/sutton-88.pdf)

[^3]: Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. _PhD Thesis, King's College, Cambridge_. [PDF](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

[^4]: Tesauro, G. (1995). Temporal difference learning and TD-Gammon. _Communications of the ACM_, 38(3), 58-68. [DOI](https://doi.org/10.1145/203330.203343)

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [Read online](http://incompleteideas.net/book/the-book.html)

[^6]: Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. _arXiv preprint_. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

[^7]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489. [DOI](https://doi.org/10.1038/nature16961)

[^8]: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^9]: Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _arXiv preprint_. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[^10]: Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv preprint_. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

[^11]: DeepSeek-AI, et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _arXiv preprint_. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

[^12]: Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. _arXiv preprint_. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)

[^13]: Liu, Y., et al. (2025). Understanding r1-zero-like training. _arXiv preprint_. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)

[^14]: Zheng, C., et al. (2025). GSPO: Group Sequence Policy Optimization. _arXiv preprint_. [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)

[^15]: MiniMax, et al. (2025). MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention. _arXiv preprint_. [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)

[^16]: ByteDance Seed, et al. (2025). VAPO: Value-based Augmented PPO. _arXiv preprint_. [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)

[^17]: OpenAI (2025). Competitive Programming with Large Reasoning Models. [arXiv:2502.06807](https://arxiv.org/abs/2502.06807)

[^18]: Meta (2025). SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution. [arXiv:2502.18449](https://arxiv.org/abs/2502.18449)

[^19]: Anthropic (2025). Emergent Misalignment: Researching the impact of reward hacking. [arXiv:2511.18397](https://arxiv.org/abs/2511.18397)

[^20]: Microsoft Research (2025). Reinforcement Pre-Training. [arXiv:2506.08007](https://arxiv.org/abs/2506.08007)
