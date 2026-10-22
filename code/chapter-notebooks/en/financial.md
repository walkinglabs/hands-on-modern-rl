# Train a tool policy from mock execution results

A request asks for AAPL's price on day2. One candidate queries AAPL; another queries MSFT. Both JSON strings may parse successfully. We still need to check the arguments against the request and execute the mock tool to inspect its result.

We score format, schema, arguments and execution separately, then compare candidates for the same request. The CPU defaults are 500 syntax-pretraining steps and 180 GRPO updates, with eight candidates per group.

A tiny GRU generates three tool-language tokens and EOS; a deterministic renderer converts them to JSON. The policy learns tool names and arguments from synthetic requests and local mock-tool results.

<!-- cell -->

## 1. Fix the dataset, mock tools and held-out requests

The split groups requests by tool, entity and date or amount. The same request is not placed on both sides merely by rewording it. All call tokens may appear in syntax pretraining, but target arguments are randomly paired with request arguments during that stage.

Syntax pretraining supplies an initial policy that can attempt calls. GRPO rewards use training-side requests only.

<!-- cell -->

## 2. Learn a small autoregressive calling policy

The vocabulary contains tool names and available arguments. The GRU generates calls token by token, preserving log-probabilities with shape `[batch, token]`.

The renderer determines representation without reading the gold answer. Wrong arguments can still produce JSON, and the verifier assigns them a lower reward.

<!-- cell -->

## 3. Sample a group per request and update token probabilities

The reward weights are 0.2 for format, 0.2 for schema, 0.3 for arguments and 0.3 for successful execution. Rewards are standardized separately within each request's group. Equal-reward groups have zero advantages.

We save the old policy's sampled token probabilities, then perform two clipped updates. The reference policy stays fixed. KL is calculated per token before response-length averaging. Responses have exactly four tokens, so the mask is all ones; a variable-length implementation must exclude padding after EOS.

<!-- cell -->

## 4. Report formatting and execution separately

A training-reward increase can come from valid formatting alone. Complete held-out execution success requires the correct tool, arguments and result. Incorrect entities remain visible in the examples.

If training success improves but held-out success does not, the experiment exposes failure to generalize across argument combinations. Two possible explanations are insufficient request coverage and failure to copy request arguments. Adding training-side combinations and adding supervised argument-copying pretraining give separate follow-up comparisons.

<!-- cell -->

## Checklist

- [ ] Track format, schema, arguments and execution separately; wrong arguments receive no execution reward.
- [ ] Perform actual policy updates using old-policy token probabilities and a frozen reference.
- [ ] Isolate held-out semantic combinations and retain failed calls.

## Exercise

A call parses and has the correct schema, but its arguments and result are wrong. Fill in the total reward. Hint: add only the first two reward weights. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [DeepSeekMath](https://arxiv.org/html/2402.03300): the order of group-relative advantages, token-level ratios, clipping and KL aggregation.
- [Companion financial tool-calling chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter18_grpo/financial-tool-calling-grpo): connect the verifier to a Qwen language model.
