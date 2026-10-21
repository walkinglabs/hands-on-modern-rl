# Exploration and regret

The first machine pays one point with probability 0.45; the second pays with probability 0.55. On the first attempt, the first machine may win while the second loses. Always choosing the machine with the highest observed average can therefore lock us into the worse choice.

We compare random selection, greedy selection, ε-greedy and UCB, then swap the machines' probabilities to examine the limits of old observations. The default CPU experiment uses 50 seeds and 2,000 steps per seed. All data is generated locally. Companion source: `code/chapter03_mdp/two_armed_bandit.py`.

<!-- cell -->

## 1. Inspect one decision

Only the environment and evaluator know the true payout probabilities. The policy sees the rewards it has collected. **Pseudo-regret** subtracts the chosen machine's expected reward from the best machine's expected reward. It separates the cost of a decision from random payout noise.

<!-- cell -->

## 2. Compare multiple seeds

ε-greedy reserves 10% of its decisions for further exploration. UCB adds an uncertainty bonus to actions with fewer observations. We plot mean cumulative reward and mean cumulative pseudo-regret; the shaded region is the standard deviation across seeds.

<!-- cell -->

## 3. Change the environment

After step 1,000, the machines exchange payout probabilities. Greedy selection and this UCB implementation use all past observations. Their regret after the switch shows how old evidence can slow adaptation.

A follow-up experiment can keep only a recent observation window. Use the same seeds and change point when comparing it with the original policy.

<!-- cell -->

## Checklist

- [ ] Distinguish a single payout, average reward and expected regret.
- [ ] Explain why greedy policies can behave differently across seeds.
- [ ] Identify the limitation of a full-history average after a distribution change.

## Exercise

Fill `answer` with the expected total rewards from choosing a machine with probability 0.55 ten times and a machine with probability 0.45 ten times. Hint: multiply the number of selections by the payout probability. Set `RUN_EXERCISE=True` after filling the answer.

<!-- cell -->

## References

- [Sutton & Barto, Chapter 2](http://incompleteideas.net/book/the-book-2nd.html): the reasoning behind ε-greedy, optimistic initialization and uncertainty-based exploration.
- [Companion bandit chapter](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter03_mdp/bandit): connect single-action rewards to the later policy-gradient experiments.
