# Query data, calculate, then submit

A request asks for a company's expense ratio or revenue growth. We create real SQLite tables and let the policy inspect schema, query rows, invoke a calculator and submit a numeric answer. Tool calls incur a small cost; calculations need retrieved data.

The default uses a CPU, two seeds and 160 updates. Companies are synthetic, and actions and SQL templates are fixed. This lab trains a finite tool-routing policy. The book's rLLM FinQA recipe covers language models generating tool parameters and answers.

<!-- cell -->

## 1. Inspect the tool data

Each task has its own in-memory database containing two years of data for one company. Company identifiers are disjoint between training and evaluation. Request types are public; gold answers remain in the terminal verifier.

Expense ratio is current expense divided by current revenue. Growth is the revenue change divided by previous revenue.

<!-- cell -->

## 2. Multi-turn observations

Legal actions enforce the tool protocol: discover tables and fields before querying, and obtain the required data before calculating. The policy chooses which year to query, which formula to use and when to submit.

Queries use bound parameters. Returned values enter subsequent observations. The saved trajectory includes actual query text, parameters and rows.

<!-- cell -->

## 3. Train and evaluate

Each update samples four requests and four trajectories per request. REINFORCE updates a small neural network. Terminal reward verifies the numeric answer, while each tool call costs 0.01.

Before and after evaluation both use greedy actions. We save success rates, mean turns and SQL/calculation traces. The supplied interface constraints already encode part of the workflow.

<!-- cell -->

## Checklist

- [ ] Locate actual SQL queries, bound parameters and returned rows.
- [ ] Identify the data and formulas needed for expense ratio and growth.
- [ ] Inspect success rates and tool costs on held-out companies.

## Exercise

Previous revenue is 100 and current revenue is 125. Fill in the growth rate. Hint: subtract first, then divide by previous revenue.

<!-- cell -->

## References

- [rLLM FinQA lab](https://walkinglabs.github.io/hands-on-modern-rl/en/chapter22_agentic/rllm-finqa-lab): language-model training with financial tools.
- [Python sqlite3](https://docs.python.org/3/library/sqlite3.html): the database interface executed in this notebook.
