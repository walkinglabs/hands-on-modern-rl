# Subtract a state baseline from the return

Receiving 100 points after an action is not enough to decide whether that action deserves encouragement. If the same state normally leads to 120 points, the outcome is below expectation. If it normally leads to 60 points, the outcome is above expectation.

We learn this state-dependent expectation with a value network and compare updates based on raw returns with updates based on return minus expectation. The default CPU budget is two seeds and 192 episodes per method. Increase it to five seeds and 960 episodes for a longer experiment. Companion source: `code/chapter05_policy_gradient/reinforce_with_baseline.py`.

<!-- cell -->

## 1. Check the expected gradient and its variance

The baseline must not depend on the action sampled at the current state. At a fixed state, the policy score has zero expectation over actions. Subtracting a state baseline therefore leaves the expected gradient unchanged.

A useful baseline can reduce variance; an arbitrary baseline need not. We enumerate a two-action example to compare both the mean and variance of the gradient.

<!-- cell -->

## 2. Learn from complete episodes

The policy outputs logits for left and right, while the value network outputs one state value. Both methods divide returns by 100 to use the same scale. We update after eight complete episodes, summing the policy loss within each episode and averaging across episodes.

`detach()` prevents the policy loss from updating the critic through the advantage. The critic has its own regression objective.

<!-- cell -->

## 3. Compare curves and policy replays

During training, the policy, visited-state distribution and returns all change. Fluctuating gradient norms therefore do not isolate gradient variance under a fixed policy. The earlier enumerated example isolates the baseline's effect.

We retain both seeds' independent evaluation results even when a short training run shows no clear advantage.

<!-- cell -->

## Checklist

- [ ] Explain by enumeration why a state baseline preserves the expected gradient.
- [ ] Explain why an excessively large baseline can increase variance.
- [ ] Distinguish fixed-policy variance from changing gradient norms during training.

## Exercise

The return is 100 and the state baseline is 120. Fill in the unscaled advantage. Hint: subtract the baseline from the return. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [Sutton & Barto, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html): episodic REINFORCE updates and action-independent state baselines.
- [Companion baseline experiment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter08_policy_gradient/baseline-experiment): examine how value-fitting errors affect policy updates.
