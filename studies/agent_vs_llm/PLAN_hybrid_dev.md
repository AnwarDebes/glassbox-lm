# Plan Hybrid-dev: the readable agent shortlists, the LLM chooses (STAR and ABCD dev, fresh rows)

Dev only, exploratory. No test file is opened. Written and frozen before any hybrid answer.

## Why, and what was seen before this plan

Q1 of PLAN_llm_dev.md (500 dev rows per dataset, seed 1) showed, beside its registered reading:
- the LLM alone: STAR 0.544, ABCD 0.830; the readable C4 agent on the same rows: 0.666 and 0.850;
- the gold action is in the agent's top 3 on 84.6% (STAR) and 92.6% (ABCD) of those rows;
- when the LLM answered outside the agent's top 3 it was right on 9.3% and 6.7% of those rows; inside, on 70.1% and
  87.9%.
So the idea, and the choice k = 3, come from those 500 rows. They are therefore not used here: this plan draws 500
fresh dev rows per dataset, disjoint from Q1's rows.

## Arms (on the same fresh rows)

- Hybrid-k3 (primary): the Q1 prompt (actions, ABCD guidelines or STAR task specifications, 10 retrieved solved
  examples, the current dialogue), plus the readable agent's top 3 actions by class sum, most likely first, with the
  instruction to choose exactly one of them. An answer that is not one of the three is replaced by the agent's top 1
  (counted and reported).
- Hybrid-k5 (secondary, no rule): the same with the top 5.
- LLM alone: the Q1 prompt unchanged.
- The readable C4 agent alone (its top 1; runs/agent_<ds>_c4_tm_s1, seed 1).
- The best fine-tuned transformer of PLAN_ft_dev.md with saved dev predictions (by full-dev accuracy), and TF-IDF LR.

## Rows

`random.Random(2)` draws 500 rows from the dev rows that Q1 (seed 1) did not use; the same rows for every arm.

## Statistic and decision (fixed now)

Accuracy on the 500 rows; paired differences with a 95% bootstrap interval over rows (seed 20261026, 10,000 draws).

- The hybrid line goes on (to a registered test on the test splits, with fresh registrations and use counts) only if,
  on BOTH datasets, Hybrid-k3 minus the agent alone is at least +2.0 points and Hybrid-k3 minus the LLM alone is at
  least +2.0 points, at the point estimate, with both intervals above 0.
- Hybrid-k3 against the fine-tuned transformer is reported beside it with no rule; a claim against fine-tuning would
  need its own registration.
- Otherwise the hybrid is reported as measured and the line stops.

## Limits known now

- One 9B LLM, one prompt; the shortlist comes from the agent trained on the full train split (seed 1).
- 500 rows per dataset: about plus or minus 4 points on one accuracy, less on paired differences.
- k = 3 was chosen after seeing Q1's rows; k = 5 is reported to show the sensitivity.
