# From a value table to a path

We start in the top-left corner. Entering the bottom-right corner earns 1 point; entering the trap at (1, 1) earns −1 point and ends the episode. Every ordinary step costs 0.01 points. Moving into a wall leaves the agent in the same cell, so unnecessary movement has a cost.

First compute values with known transition rules, then let Q-learning learn through actual transitions. The default CPU experiment uses 30 seeds and 500 episodes per seed. Data is generated locally. The map, rewards and terminal-state convention match `code/chapter03_mdp/gridworld_q_learning.py`.

<!-- cell -->

## 1. Attach the reward to a transition

The trap or goal reward is paid on entry. A terminal transition has no successor value. For example, moving right from (3, 2) has a target of 1; adding a goal-state value would count the terminal reward twice.

<!-- cell -->

## 2. Learn without receiving the solution

The Q table has shape `[4, 4, 4]`. The last dimension represents up, down, left and right. Each step updates only the state and action just visited. The exploration rate decreases from 1 to 0.05, gradually replacing early exploration with exploitation.

<!-- cell -->

## 3. Inspect the policy and learning curve

The training curve includes exploratory actions. The displayed path comes from a separate greedy evaluation with exploration disabled, so the two returns need not match. The table reports success and evaluation return across 30 seeds.

<!-- cell -->

## Checklist

- [ ] Explain why terminal transitions have zero successor value.
- [ ] Read a successful path from the learned Q table.
- [ ] Distinguish exploratory training return from greedy evaluation return.

## Exercise

Fill in the TD target when moving right from (3, 2) into the goal. Hint: this transition terminates the episode. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [Companion value experiment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter03_mdp/value-experiment): the same map and reward convention.
- [Sutton & Barto, Chapters 4 and 6](http://incompleteideas.net/book/the-book-2nd.html): value iteration with a known model and learning through sampled transitions.
