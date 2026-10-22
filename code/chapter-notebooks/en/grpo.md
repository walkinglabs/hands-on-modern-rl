# Turn a group of responses into inspectable advantages

Four responses to the same question receive rewards `[0, 0, 1, 1]`. The group mean is 0.5 and the population standard deviation is 0.5, giving advantages close to `[−1, −1, 1, 1]`. A group of entirely incorrect responses has no relative ordering: its advantages should be zero without dividing by zero.

This CPU mechanism experiment includes a verifier, token-level clipping and an actual gradient update. The policy is a small token-probability table. The DPO and tool-calling Notebooks provide language-model training. All examples are generated locally; companion source: `code/chapter09_grpo_rlvr`.

<!-- cell -->

## 1. Reject ambiguous answers

The verifier accepts exactly one integer in the form `Answer: 42`. It rejects conflicting answers and extra explanations. The cases below cover correct, incorrect, conflicting and non-integer answers.

<!-- cell -->

## 2. Clip each token before averaging

Consider token probability ratios 1.5 and 0.5, with clipping bounds `[0.8, 1.2]`. Multiplying the ratios first gives 0.75; clipping that sequence-level number changes the objective.

We preserve shape `[response, token]`, exclude padding from the mean and broadcast each response's advantage across its tokens.

<!-- cell -->

## 3. Inspect gradients and results

The table update checks three effects: better responses gain probability, worse responses lose probability and padding receives zero gradient.

The experiment uses population standard deviation, `correction=0`. Frameworks that use sample standard deviation will produce a different advantage scale.

<!-- cell -->

## Checklist

- [ ] Obtain finite advantages for all-correct and all-incorrect groups, with zero advantages at zero variance.
- [ ] Retain the token dimension through probability ratios and clipping; normalize response length afterward.
- [ ] Reject conflicting answers and verify zero gradients at padding positions.

## Exercise

Fill in the mean clipped token surrogate for ratios `[1.5, 0.5]` and positive advantage 1. Hint: compute `min(ratio, clip(ratio))` separately for each token before taking the mean. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [DeepSeekMath, Section 4.1](https://arxiv.org/html/2402.03300): token-level probability ratios, response-length averaging and the reference-policy KL term.
- [Companion RLVR chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter18_grpo/rlvr): connect a verifier to a trainable policy and examine reward-design limits.
