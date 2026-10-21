# Decide again after the tool responds

A request asks the agent to look up two numbers, add them and double the result. The initial lookup contributes to the final answer: subsequent calculations need its returned numbers. We implement that dependency as an actual environment and update the tool-selection policy using trajectory returns.

The default uses a CPU, two seeds and 160 policy updates, with no external dataset. A small neural network selects finite tool actions under an environment-provided protocol. This experiment tests multi-turn credit assignment and tool routing. General language understanding and freely generated parameters require the language-model recipes in the book.

<!-- cell -->

## 1. Observations from tools

One request type requires addition followed by doubling; the other requires subtraction followed by squaring. The request type is public. The lookup reveals the actual numbers. Legal-action constraints provide the interface order, while arithmetic choices remain policy decisions. Gold answers never enter the observation.

<!-- cell -->

## 2. A terminal reward can update earlier actions

With rewards `[0, 0, 0, 1]` and discount 0.9, the first action's return is 0.729. Early actions retain a learning signal; sparse rewards create exploration and variance problems. The optimizer uses REINFORCE with the mean return of the other trajectories for the same request as a baseline. Tool observations are inputs to the next decision.

Lookup and submission are fixed by the protocol; training mainly changes the two intermediate arithmetic choices. A nonzero return does not guarantee a nonzero gradient for every parameter. The discount above is a numerical example; training below uses the undiscounted trajectory return.

The gradient example applies a mask to selected action log probabilities. Observation positions have no direct policy-loss term. It illustrates loss masking, rather than a complete language-model training step.

<!-- cell -->

## 3. Inspect the trajectory

The curves evaluate unseen number combinations. Both before and after evaluation use greedy decisions, identical seeds and a fixed budget. Saved trajectories record lookup results, arithmetic calls and final answers.

The environment supplies legal tool order. This experiment does not measure autonomous discovery of that order. Removing the constraints is a separate difficulty experiment. Evaluation is logged without selecting checkpoints or changing the training budget.

<!-- cell -->

## Checklist

- [ ] Compute the nonzero return of an early action from terminal reward.
- [ ] Distinguish policy choices, legal-action constraints and tool observations.
- [ ] Save held-out curves, complete trajectories and policy parameters.

## Exercise

Fill in the first return for four actions and discount 0.9. Hint: discount the final unit reward three times.

<!-- cell -->

## References

- [Multi-turn RL](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/formulation): actions, tool observations, trajectories and action masks.
- [Credit assignment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/credit-assignment): how terminal rewards affect multi-turn policies.
