# REG_agent_vs_llm: the registered test result (stored 2026-10-29T08:12:38Z)

The registered test of the readable agent (no neural network at any stage: a word table counted from C4 text, memory
bits of the actions taken, a Tsetlin machine) against LLM* (Qwen3.6-35B-A3B at 4-bit with 50 retrieved examples, the
dev-best of the local LLM configurations we ran), a fine-tuned bge-base (FT) and TF-IDF logistic regression (no
rule), for next-action prediction on STAR, ABCD and MultiWOZ 2.2. Independent verification: **MATCH**
(verification/VERDICT.md).

The numbers are in results.json (built by a script from registered_outputs/report_*.json) and in the registered table
below. Check the files: `cd studies/agent_vs_llm/results_REG_agent_vs_llm && sha256sum -c SHA256SUMS` (SHA256SUMS of
this published copy; the per-row prediction files are not distributed, see Contents).

## Every rule's outcome (registered statistic: seed 1, bootstrap over dialogues, 95%)

| Dataset | Rows | AGENT | LLM* | FT | AGENT - LLM* (95%) | NI (-3) | NI2 (-2) | SUP (+3) | FT - AGENT (95%) | FT | 5-seed mean AGENT - LLM* (two-level 95%) | Verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STAR | 4000 | 0.6490 | 0.6340 | 0.7350 | +1.50 [-0.02, +3.05] | met | met | not met | +8.60 [+7.24, +9.98] | met | +1.28 [-0.26, +2.84] | MATCH |
| ABCD | 3608 | 0.8379 | 0.8379 | 0.8786 | +0.00 [-1.17, +1.14] | met | met | not met | +4.07 [+3.11, +5.06] | met | -0.14 [-1.23, +0.91] | MATCH |
| MultiWOZ | 4000 | 0.3548 | 0.3660 | 0.4268 | -1.12 [-2.78, +0.57] | met (seed 1 only) | not met | not met | +7.20 [+5.51, +8.78] | met | -1.70 [-3.21, -0.14] | MATCH |

## No-rule readings

| Dataset | AGENT - LLM* 90% (inside +-3?) | TFIDF | TFIDF - LLM* (95%) | AGENT - TFIDF (95%) | AGENT - TFIDF two-level | FT - AGENT two-level | LLM* s/row (median) | LLM* invalid |
|---|---|---|---|---|---|---|---|---|
| STAR | [+0.21, +2.79] (yes) | 0.6455 | +1.15 [-0.47, +2.81] | +0.35 [-1.00, +1.72] | [-1.10, +1.38] | [+7.57, +10.14] | 10.6 | 1 |
| ABCD | [-0.98, +0.97] (yes) | 0.8473 | +0.94 [-0.14, +2.00] | -0.94 [-1.92, +0.03] | [-1.95, -0.23] | [+3.36, +5.08] | 11.13 | 1 |
| MultiWOZ | [-2.52, +0.30] (yes) | 0.3540 | -1.20 [-2.69, +0.27] | +0.07 [-1.32, +1.47] | [-1.76, +0.74] | [+6.27, +9.23] | 11.63 | 0 |

## Claim lines

- NI met on all three datasets: the readable agent is not worse than the best of the local LLM configurations we ran (chosen on dev; the 35B model at 4-bit), at a margin of 3 points (NI).
- NI2 (2 points) met on: STAR, ABCD; not on: MultiWOZ.

## Seed-1-only notes

- MultiWOZ NI: met on seed 1, not on the 5-seed mean

## What must travel with these numbers

- MultiWOZ NI is met on seed 1 only, not on the 5-seed mean (two-level lower end -3.21); seed 1 is the best of the 5
  seeds on these rows (+0.58 points over their mean), though it was fixed as the statistic before the run.
- The 3-point NI margin was set after a dev result (disclosed in the registration); NI2 at 2 points is met on STAR
  and ABCD only.
- SUP is met on no dataset: never write that the agent beats the LLM.
- FT is ahead of the agent on all three datasets (+8.60, +4.07, +7.20 points).
- The agent is level with TF-IDF logistic regression on STAR and MultiWOZ; on ABCD TF-IDF is 0.94 points ahead (5-seed
  interval excludes 0). The result is not specific to the Tsetlin machine.
- LLM* is the dev-best of 4-8 local configurations, not the best LLM that exists; the evaluation is offline (the
  logged next action is the label).
- Test uses: STAR 7th, ABCD 15th, MultiWOZ 2nd. The agent's test accuracy was known at the freeze; LLM*'s, FT's and
  TF-IDF's were not.

## Contents

- registration/: the frozen registration and its freeze record, the job lines, the queue log (copied after the
  report), the rows file and the LLM* choice record.
- registered_outputs/: the checks, the three reports, the verify outputs, the pin checks and the table. The LLM, FT and
  TF-IDF predictions on the test rows (about 18 MB) are not distributed.
- The agent's pinned test predictions, seeds 1-5 (about 28 MB), and the models are not distributed either; they are
  kept in local backup archives with their sha256.
- verification/: the independent recompute and comparison outputs and their tools.
- results.json: every registered number per dataset.
