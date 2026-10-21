# Continuous torque and value estimates

When a pendulum leans to the right, the action includes the magnitude of the applied torque. Pendulum accepts a single number between −2 and 2. We first measure a zero-torque policy, then train a Gaussian policy and a value network.

This experiment uses the book's A2C and VecNormalize recipe: CPU, eight environments and 100,000 steps by default. Set `TOTAL_STEPS=300000` for a longer budget. The short run does not guarantee a stable upright pendulum. Companion source: `code/chapter06_actor_critic/actor_critic_pendulum.py`.

<!-- cell -->

## 1. Preserve observation normalization during evaluation

Observations contain the cosine and sine of the angle and the angular velocity. Training normalizes observations and rewards, so the critic predicts normalized returns.

Evaluation uses the training observation statistics without updating them and accumulates raw environment rewards. Evaluation seeds 1,000–1,009 provide initial states that are excluded from training.

<!-- cell -->

## 2. Save and reload the policy and statistics

Saving only the network loses the way its inputs were normalized. We save both the model and the statistics, reload them and check that the same observation produces the same action.

The plots retain the actual short-run performance. If returns remain low, extend the training budget and compare against the same evaluation seeds.

<!-- cell -->

## 3. Inspect a fixed-seed replay

The replay uses a new seed, 2,000. Observations pass through frozen normalization statistics before reaching the policy. Torque, angular velocity and value predictions come from this saved policy. The critic values and raw environment rewards use different units.

<!-- cell -->

## Checklist

- [ ] Explain the output difference between Gaussian and discrete-action policies.
- [ ] Reload the model and observation statistics and pass the action-consistency check.
- [ ] Evaluate using raw environment rewards and retain unsuccessful control behavior.

## Exercise

Fill in whether observation statistics should be updated during evaluation. Hint: evaluation uses fixed training statistics. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [SB3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html): the implementation interface and continuous-policy configuration.
- [SB3 VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize): save, load and freeze normalization statistics.
- [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/): observations, torque bounds and rewards.
