# Chapter 11 · Imitation Learning, Inverse RL, and Meta-RL

> [Chapter 10: Offline Reinforcement Learning](../chapter12_offline_rl/intro) handles the setting where you "only have historical data and cannot interact with the environment," but it still assumes that data carries an explicit reward signal. This chapter handles two more extreme situations: (1) **there is no reward function at all** — only expert demonstration trajectories, so what do you do? (2) **the environment itself keeps changing** — the agent has to learn to "adapt quickly to a new task." The first situation leads to **Imitation Learning (IL)** and **Inverse RL**; the second leads to **Meta-RL**. The two eventually converge in the LLM era: SFT is essentially behavior cloning, the three-stage InstructGPT pipeline can be rewritten as BC + RL + RL, and In-Context RL reveals that "the RL algorithm itself can be distilled into a transformer."

## Chapter Map

- [13.1 Behavior Cloning and DAgger](./bc-dagger): the supervised-learning view of imitation, BC's distribution-shift problem, and DAgger's fix for it
- [13.2 Inverse RL and GAIL](./irl-gail): recovering a reward function from expert behavior, and how GAIL uses a GAN framework to sidestep an explicit reward
- [13.3 Meta-RL: MAML, RL², PEARL, In-Context RL](./meta-rl): learning to adapt quickly to new tasks, and how Algorithm Distillation, in the LLM era, reveals "RL as in-context learning"

The next section, [13.1 Behavior Cloning and DAgger](./bc-dagger), starts with the most basic form of imitation learning.
