# R2 registration: the sparse-input fix on the lexical router

Part 1 of `FAILURE_MAP.md`. Frozen by `scripts/freeze.py` (record `R2_sparse_input_fix.frozen.json`: UTC time, SHA-256 of this text and of every source file). Changes after freezing go into a dated amendment file.

## Question

On sparse word-presence input, a coalesced TM grows long clauses that are mostly negated literals (F14, F19). Does a literal budget fix this? A fix means higher accuracy and short clauses, on the same input and machine.

## Data and input

- CLINC150, in-scope only: train 15,000, val 3,000, test 4,500. Out-of-scope queries are not used in the rules.
- Input `lex:2000`: binary presence of the 2,000 most frequent word 1- and 2-grams of the train texts (scikit-learn CountVectorizer, token pattern `\b\w+\b`). Negation is on, which gives 4,000 literals.

## Arms

- **A (default, shows the failure):** coalesced TM, `src/dl/tm.py`. 4,000 clauses, T 4,000, s 8, 256 states, 40 epochs, batch 128, hardest-negative sampling, no literal budget. The best-validation snapshot is taken (val scored every 2 epochs).
- **B (the fix):** A plus an exact literal budget L*. A clause that already includes L* literals cannot include another one.
  - L* is chosen by a rule fixed now, before those runs finished. It is the value among {8, 16, 32} with the highest val accuracy in the exploratory runs `e2b_lex_ml8`, `e2b_lex_ml16`, `e2b_lex_ml32` (seed 1, exact budget code, `results/explore/tm.jsonl`).
  - Ties go to the smaller L*.
- **C (linear control on the same bits):** multinomial logistic regression on the same 2,000 bits. C is chosen from {0.1, 0.3, 1, 3, 10, 30, 100, 300} by val accuracy.

## Runs

- A and B: seeds 1 to 10. The TM seed is 100·k. C is deterministic and fitted once.
- Script: `scripts/confirm_router.py --registration R2 --dataset clinc150 --encoder minilm --view lex:2000 ...`. The encoder argument is unused by `lex:2000`. Arm C is run with `--lr-input bits`.
- The test split is scored once per run.

## Metrics and statistic

- **Test accuracy.** For each arm, average each test query's correctness over the 10 seeds. Take the paired difference between two arms over the 4,500 test queries. Bootstrap it over test queries: 10,000 resamples, seed 0, 95% percentile interval.
- **Clause statistics of each snapshot:** median and maximum literals per clause, and the share of non-empty clauses that hold at least one plain (non-negated) literal.

## Pass rules (test)

- **R2.1 accuracy:** B minus A is at least +3.0 points, and its interval excludes 0.
- **R2.2 clause length:** in B, the median and the maximum literals per clause are at most L*, in every seed.
- **R2 is met** if R2.1 and R2.2 both hold.
- **R2.3 against the linear control, reported separately:**
  - B is non-inferior to C if B minus C is at least -0.5 points.
  - "B beats C" is said only if the interval of B minus C lies entirely above 0.

Reported with no rule: the plain-literal share in A and B, and the out-of-scope AUROC (top class sum) on oos_test.
