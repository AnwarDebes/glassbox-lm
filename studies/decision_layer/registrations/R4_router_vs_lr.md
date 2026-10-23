# R4 registration (with R1): the TM decision layer against logistic regression on frozen LLM embeddings

Parts 3 and 4 of `FAILURE_MAP.md`. Frozen by `scripts/freeze.py`. Changes after freezing go into a dated amendment file.

## Questions

- **R4:** with the same frozen embedding, does the coalesced TM router reach the accuracy of logistic regression on three intent datasets and two encoders? Clause-Driven-LLM, upstream: CLINC150 94.2 against 95.6.
- **R1:** after one temperature, are the TM's vote sums calibrated? Does the top class sum detect out-of-scope queries as well as logistic regression's maximum softmax probability?

## Cells

- **Datasets:** CLINC150 (train 15,000 / val 3,000 / test 4,500; oos_test 1,000), Banking77 (train 9,000 / val 1,003 / test 3,080), HWU64 (train 8,954 / val 1,076 / test 1,076). Splits as in `src/dl/data.py`.
- **Encoders:** all-MiniLM-L6-v2 (the Clause-Driven-LLM encoder) and BAAI/bge-base-en-v1.5. Embeddings are L2-normalised (`scripts/embed.py`).
- That gives 6 cells.

## Arms

- **LR:** multinomial logistic regression on the raw embedding. C is chosen from {0.1, ..., 300} by val accuracy. Full-batch L-BFGS, `src/dl/linear.py`.
- **TM-single:** coalesced TM on the thermometer code `thermo:8` (7 quantile thresholds per dimension, fitted on train), with the best-validation snapshot. Its setting is chosen by a rule fixed now:
  - Candidates are the exploratory CLINC150 MiniLM runs (seed 1, val, `results/explore/tm.jsonl`): `repro_cdllm`, `e1_thermo16`, `e1_thermo8_c8k`, `e1_thermo8_e80`, `e5_ml4`, `e5_ml8`, `e5_ml16`, `e5_cd25`, `e5_cd50`, `e5_s4`, `e5_s16`.
  - The setting with the highest val accuracy is chosen.
  - If several are within 0.10 points of the best, the one with the lowest cost is chosen. Cost = bits × clauses × epochs.
  - The chosen setting is used unchanged in all 6 cells.
- **TM-composite:** two disjoint composites per cell, made from the TM-single machines of seeds 1 to 5 and of seeds 6 to 10. Each composite is the sum of the members' class sums, each sum divided by its standard deviation on val. No extra training.
- **TM-readout:** convex vote weights on the clause outputs of each seed's TM-single machine. These are multinomial logistic weights on [clause outputs] and on [clause outputs + input bits], C by val, and the better of the two is chosen by val. The decision is still a weighted vote of the same clauses.
  - This arm is included only if, in the exploratory run `e4_readout`, the better read-out exceeds that run's own TM val accuracy by at least 0.5 points.
  - If it is excluded, that is stated here in an amendment before the first confirmatory run.

## Runs

- TM arms: seeds 1 to 10 per cell; TM seed 100·k.
- `scripts/confirm_router.py --registration R4 --name <cell>`; reports by `scripts/report_router.py`.
- The test split is scored once per run.

## Statistic

For each arm, average each test query's correctness over the seeds (the two composites count as two seeds). Take the paired difference to LR over the test queries. Bootstrap it over queries: 10,000 resamples, seed 0, 95% percentile interval.

## Pass rules (test)

- **R4.1 (main):** TM-composite minus LR is at least -0.5 points in at least 5 of the 6 cells.
- **R4.2:** TM-single minus LR, reported per cell with its interval. "Gap closed" is said if it is at least -0.5 in at least 5 of the 6 cells.
- **R4.3:** in any cell where the interval of an arm minus LR lies entirely above 0, that arm is reported as ahead of LR in that cell.
- **R1.1 (calibration):** TM-single expected calibration error on test (15 equal-width bins; one temperature fitted on val by NLL) is at most 0.05 in every cell (mean over seeds).
- **R1.2 (out of scope, CLINC150, both encoders):**
  - The TM-single top-class-sum AUROC on test (4,500 in-scope against 1,000 oos_test; mean over seeds) is at least LR's maximum-softmax AUROC minus 0.5 points.
  - "Better than LR" is said only if all 10 seeds exceed LR.

Reported with no rule: macro averages over cells, clause statistics, TM-readout against LR, and training time.
