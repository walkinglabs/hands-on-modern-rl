# Update a language model using paired preferences

The request `help topic0` can be followed by `please read topic0 carefully` or by `go away topic0 now`. We first teach a small model to generate both responses, then use paired preferences to increase the relative probability of the first response.

This CPU experiment trains a tiny autoregressive GRU from scratch and performs actual DPO parameter updates. All data is synthetic. The defaults are 400 SFT steps and 120 DPO steps, with no model download. The experiment tests preference learning in a limited vocabulary; it does not establish general conversation ability or the removal of sycophancy. The book's Qwen 0.5B recipe remains in `code/chapter17_dpo`.

<!-- cell -->

## 1. Compute completion-only likelihood

The input contains a prompt, a separator and a response. Each position's logits predict the next word. DPO sums log probabilities over the complete response, excluding prompt predictions and the separator itself and including the response EOS.

Both responses here have equal lengths and require no padding. A variable-length extension must still sum only the valid response tokens.

<!-- cell -->

## 2. Train the language model and freeze a reference

SFT covers 12 topics with equal numbers of both response styles. DPO receives preferences only for topics 0–7; topics 8–11 are held out from preference training. Held-out topics have appeared during SFT, so this split tests preference transfer rather than generalization to entirely unseen topics.

After SFT, we copy the model to create a reference whose parameters remain frozen during DPO.

<!-- cell -->

## 3. Optimize changes relative to the reference

Suppose the chosen response gains 2 log-probability units relative to the reference, while the rejected response gains 0.5. Their relative gap is 1.5. After multiplication by β, the negative log-sigmoid loss encourages this gap to grow.

The reference contributes fixed likelihood values. Both chosen and rejected likelihoods from the current model contribute gradients.

<!-- cell -->

## 4. Evaluate preference and generated content separately

A growing preference margin establishes that the chosen responses become more preferred relative to the rejected responses. Generated answers may still lose their requested topic or repeat words, so we also inspect response content and topic retention.

A larger-model experiment needs new requests, human preferences and independent quality evaluation. This small experiment does not validate Qwen memory requirements or hosted execution.

<!-- cell -->

## Checklist

- [ ] Train an autoregressive model and verify that DPO changes its parameters.
- [ ] Keep the reference frozen and sum likelihoods only over valid response tokens.
- [ ] Evaluate held-out preference margins separately from generated content.

## Exercise

The chosen and rejected log probabilities change by 2 and 0.5 relative to the reference, with β=0.2. Fill in the reward margin. Hint: subtract first, then multiply by β. Enable `RUN_EXERCISE` after filling `answer`.

<!-- cell -->

## References

- [DPO paper](https://arxiv.org/abs/2305.18290): paired response log-probability differences relative to a fixed reference policy.
- [Companion Qwen DPO experiment](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter17_dpo/dpo-hands-on): use the same objective with a pretrained model and a different data and resource scale.
