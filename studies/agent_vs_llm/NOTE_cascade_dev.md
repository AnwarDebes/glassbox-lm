# Note: the cascade of the readable agent and a neural model (dev only; Q2 of PLAN_llm_dev.md, widened)

Written and frozen before any cascade number. CPU only, on saved dev predictions.

## Question

How many dialogue turns can the readable network-free agent (C4 table, Tsetlin Machine) decide on its own, with no
loss of accuracy against a neural model, if it hands the turns where its clause votes are least confident to that
neural model? The neural partner is the best fine-tuned transformer of PLAN_ft_dev.md (by full-dev accuracy) and, when
its answers exist, the LLM of PLAN_llm_dev.md.

## Rule (fixed now)

- Confidence of the agent on a row: the margin between its two largest class sums (`agent_preds.py`, tm.npz).
- Split the dev rows into halves A and B by a seeded permutation (seed 20261026), whole dialogues together (by
  convo_id), so that turns of one dialogue never fall in both halves.
- On A, pick the largest coverage c (the share of rows the agent decides, in steps of 1%) at which the cascade's
  accuracy is at least the partner's accuracy on A minus 0.5 points. The margin threshold at that coverage on A is
  carried to B.
- On B, report the cascade's coverage and accuracy, the partner's accuracy, and their difference with a 95% interval
  (bootstrap over dialogues of B, seed 20261026, 10,000 draws). Then the roles of A and B are swapped and reported too.
- Reading: "the readable agent decides X% of turns at no loss" is stated only if on both B readings the difference is
  at least -0.5 points at its point estimate and the interval's lower end is above -1.5 points.
- Also reported, with no rule: the coverage-accuracy curve on all of dev, and the share of rows on which the agent is
  right and the partner wrong, and the reverse.
