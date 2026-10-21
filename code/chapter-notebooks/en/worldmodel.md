# Learn action consequences, then plan

A cart starts at position 2.5 with zero velocity. Applying force to the left moves it toward the origin. Continuing that force after reaching the origin sends it past the target.

We collect random-action trajectories, learn how state and force determine the next step, then compare candidate action sequences. The CPU defaults are 180 trajectories of 30 steps, 900 model updates and 256 candidate sequences with planning horizon 15. No external data is needed. Dynamics and rewards match the book's `minimal_mbrl_point_mass.py`; this experiment adds trajectory-level hold-out and drift checks.

<!-- cell -->

## 1. Compare a zero-change baseline with a learned model

The first two target columns are changes in position and velocity; the third is reward. Splitting complete trajectories keeps adjacent frames out of opposite sides of the split and avoids an overly optimistic test error.

Only training trajectories update the model. Test trajectories are used exclusively to measure prediction error.

<!-- cell -->

## 2. Inspect open-loop prediction drift

We keep the actual actions from a test trajectory and repeatedly feed predicted states back into the model. No new true position is supplied during this rollout, exposing how errors accumulate.

The planner in the next section receives a true observation after every executed action. That feedback makes it different from this open-loop prediction test.

<!-- cell -->

## 3. Plan inside the model and execute one real step

The planner accumulates predicted rewards for each candidate action sequence and executes only the first action of the highest-scoring sequence. It searches again at the next step.

Random control, zero torque and learned-model planning are evaluated from the same initial states. All reported returns come from the true dynamics. Planning can exploit model errors, so we retain the result for every initial state.

<!-- cell -->

## Checklist

- [ ] Split complete trajectories and report independent one-step error and multi-step drift.
- [ ] Distinguish the dynamics model, planner and real environment.
- [ ] Compare policies from matched initial states and measure returns with the true dynamics.

## Exercise

Fill in the number of actions executed after each planning search. Hint: execute one action and receive a new true state before planning again. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [Companion Model-Based RL chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter28_vla/embodied-intelligence/model-based-rl/): the roles of learned dynamics, model error and model predictive control.
- [PETS](https://arxiv.org/abs/1805.12114): further reading on dynamics uncertainty; this experiment uses one deterministic model.
