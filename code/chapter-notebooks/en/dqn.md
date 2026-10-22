# Separate action selection from action evaluation

Two actions both have a true value of zero, but their estimates contain noise. Taking the larger estimate on every trial can produce a positive average. We first reproduce this effect, then compare DQN and Double DQN in CartPole.

The default CPU budget is 15,000 steps for each of two seeds per algorithm. Increase `STEPS` to 80,000 and use five seeds for a longer comparison. Companion source: `code/chapter04_dqn/double_dqn_cartpole.py`.

<!-- cell -->

## 1. Inspect the bootstrap target

The network maps observations of shape `[batch, 4]` to action values of shape `[batch, 2]`. DQN uses the target network both to select and to evaluate an action. Double DQN selects with the online network and evaluates that action with the target network.

`terminated` removes the successor value. A time-limit `truncated` signal ends the sampled episode while retaining the bootstrap term.

<!-- cell -->

## 2. Use matching budgets and independent evaluation

Both algorithms use the same network size, exploration schedule and evaluation seeds. The replay buffer stores actual transitions. The target network is synchronized every 250 steps.

Every 1,500 steps, we disable exploration and evaluate in a separate environment. Evaluation episodes never enter the training buffer.

<!-- cell -->

## 3. Inspect returns, values and an actual replay

A Q value predicts discounted return. Subtracting it from the undiscounted 500-point episode limit would not measure overestimation. The figures show evaluation rewards, Q estimates and a fixed-seed policy replay.

The replay always uses Double DQN with seed 0. We do not select the best-looking run from the comparison.

<!-- cell -->

## Checklist

- [ ] Locate action selection and evaluation in the two target calculations.
- [ ] Explain the bootstrap mask for task termination and time limits.
- [ ] Save independent evaluation curves and a fixed-seed replay, including weak policies.

## Exercise

The online network outputs `[3, 1]`, and the target network outputs `[1, 4]`. Fill in the Double DQN successor value. Hint: first find the action selected by the online network. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461): the separation of action selection and evaluation and the scope of the original experiments.
- [Gymnasium Env](https://gymnasium.farama.org/api/env/): the meanings of `terminated` and `truncated`.
