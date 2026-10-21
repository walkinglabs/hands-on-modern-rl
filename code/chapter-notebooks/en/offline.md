# Evaluate actions beyond a fixed dataset

Suppose the dataset contains only actions near zero for a particular state. A value network may still assign a high value to action 1. Low training error cannot establish whether that unobserved action works.

We fix one dataset and compare direct Q maximization, a conservative penalty and a behavior constraint. The CPU defaults are three seeds, 600 updates per Q network and 400 updates per policy. All data is synthetic.

This is a single-step contextual bandit. It demonstrates a CQL-style penalty and behavior constraints without multi-step TD targets; it is not a complete CQL or TD3+BC benchmark. Companion source: `minimal_offline_rl_contextual_bandit.py`.

<!-- cell -->

## 1. Inspect coverage before fitting values

States cover −1 to 1, while actions occupy a narrow band. The conservative penalty compares Q values for random actions with Q values for observed actions, discouraging unsupported high estimates.

For each seed, both Q networks start from the same parameters and use the same training-sample indices.

<!-- cell -->

## 2. Separate fitting error from actual policy return

Held-out data MSE still measures only the behavior policy's neighborhood. We now freeze the models and use the known simulator to evaluate new policy actions. Evaluation values never feed back into the optimizer.

The oracle provides an upper bound only during evaluation; its actions are not supplied to the learner. Conservative methods can also be too restrictive, so we report both distance from data support and actual return.

<!-- cell -->

## 3. Inspect out-of-distribution estimates

The first plot shows dataset coverage. The second fixes a state and compares true rewards with both Q estimates. The third shows learned policies. Plots use seed 0, while the table retains all three seeds.

True reward curves are final diagnostics, never training inputs. If a run shows little extrapolation error, retain that result and test the explanation by changing data noise or initialization.

<!-- cell -->

## Checklist

- [ ] Avoid querying rewards for new actions after fixing the training dataset.
- [ ] Report behavior-data fitting error, support distance and independent policy return.
- [ ] Explain the boundary between this single-step experiment and full offline RL algorithms.

## Exercise

Observed actions are close to zero. Fill in whether these data directly validate `Q(state, action=1)`. Hint: producing a number does not establish observational support for it. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779): why suppressing unsupported high values can help offline policy learning.
- [TD3+BC](https://arxiv.org/abs/2106.06860): behavior constraints limit policy deviation from data. This experiment uses that constraint idea without implementing full TD3.
