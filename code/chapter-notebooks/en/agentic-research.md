# Retrieve, choose evidence and verify citations

The agent answers questions from the book's fixed teaching corpus. It selects a query, retrieves documents, chooses evidence and extracts an answer. Answer correctness and citation correctness are scored separately. We reuse the book's runnable benchmark and add matched before/after held-out evaluation.

The default uses a CPU, three seeds and 220 epochs. Queries are supplied candidates, answers come from document fields, and training updates a shared-feature linear policy.

<!-- cell -->

## 1. Pin source and data

The benchmark download is pinned to a repository commit and verified with SHA-256. Its first run requires the network to fetch one Python file. Subsequent runs reuse the verified local cache.

Each corpus document contains a title, text, answer and document identifier.

<!-- cell -->

## 2. Hold out questions

The first seven questions train the policy; the last three evaluate it. Their target document identifiers are disjoint. The retrieval corpus is shared because evaluation requires access to evidence for new questions.

Held-out labels do not enter training updates. Evaluation retains matching retrieval perturbation seeds, temperatures and action-selection modes.

<!-- cell -->

## 3. Matched evaluation

Training changes linear weights for query and evidence selection. We report greedy evaluation and repeated sampled evaluation separately, each using identical settings before and after training.

The curve shows training-question rewards. Held-out metrics are saved separately. The table separately reports answer accuracy, citation support and total reward.

<!-- cell -->

## Checklist

- [ ] Verify the source hash and train/evaluation split.
- [ ] Inspect answer accuracy, citation support and total reward separately.
- [ ] Save queries, retrieved documents, selected evidence and policy weights.

## Exercise

Fill in the reward when the answer is incorrect but the citation is correct. Hint: answers contribute 0.75 and citations contribute 0.25.

<!-- cell -->

## References

- [DeepResearch evaluation](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter24_deep_research/deep-research-eval): answers, evidence, process and cost.
- [Browser research trajectories](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter24_deep_research/browser-rl-harness): continue from a fixed corpus to web tools.
- [Pinned book benchmark](https://github.com/walkinglabs/hands-on-modern-rl/blob/4bce68267c0e17107ab3b42b35efb2dffa3f777d/docs/chapter22_agentic/code/deep_research_rl_benchmark.py): the exact source used here.
