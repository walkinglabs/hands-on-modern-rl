# Repair code, then verify it with tests

We repair subtraction in `solve(x, y)` into addition or multiplication. The policy reads the specification, selects a patch, invokes Python to run public tests, then submits or revises. Submission triggers separate tests, so memorizing one public example cannot earn full reward.

The default uses a CPU, two seeds and 160 updates. The policy selects among three trusted patch templates to study multi-turn test feedback. The book's rLLM recipe covers freely generated code and language-model training.

<!-- cell -->

## 1. Execute the tests

The tool writes a candidate function into a temporary file and runs assertions in a separate Python process. Identical source/test pairs are cached to avoid repeated process startup; first results always come from execution.

Only our fixed trusted templates are executed here. Candidate code comes from three templates and runs with a two-second timeout.

<!-- cell -->

## 2. Public and private tests

The specification exposes whether the function should add or multiply. Public-test feedback enters the next observation; private answers remain in the terminal verifier. Legal actions provide the inspect, patch, test and submit protocol.

Training and evaluation use different numeric inputs. Evaluate the sum and product specifications separately.

<!-- cell -->

## 3. Train the patch policy

We sample four trajectories per request and use the other trajectories' mean return as a baseline. Patch and submission decisions contribute to trajectory loss. Test responses become inputs to the next decision.

Both before and after evaluation use greedy decisions on the same held-out requests. The budget is fixed and the held-out curve is used for reporting, without checkpoint selection.

<!-- cell -->

## Checklist

- [ ] Locate actual subprocess test inputs, exit codes and output.
- [ ] Explain why a patch that passes public tests can fail private tests.
- [ ] Save training curves, patch trajectories, parameters and invocation counts.

## Exercise

Fill in whether the example-memorizing patch passes the private case. Hint: it returns zero for other inputs.

<!-- cell -->

## References

- [rLLM DeepCoder lab](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/rllm-deepcoder-lab): the language-model code-agent training route.
- [Agentic training system](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/build-agentic-training-system): environment, trajectory and model-update interfaces.
