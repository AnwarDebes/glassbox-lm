# R2d and R7c registration: R2c's and R7b's budget arms again, with an exact budget selection

Written after the results of R2c and R7b. Both met their accuracy rules and failed only their length rules:

- R2c (literal budget 32 on the Banking77 and HWU64 word routers): B minus A +4.71 [+4.00, +5.44] and +4.09 [+2.79, +5.41] points, R2c.1 met; but Banking77 seed 6 holds two clauses of 33 literals, so R2c.2 and R2c are not met.
- R7b (the readable layer with budget 16): R7b.1, R7b.2 and R7b.4 met; but three clauses of 17 literals (CLINC150 seed 2, HWU64 seeds 6 and 10), so R7b.3 and R7b are not met.

**Cause, found and tested (`tests/test_budget_exact.py`):** the exact budget keeps the crossing literals whose random key is at least the k-th largest key. `torch.rand` on the GPU draws float32 values with ties (20 million draws gave 14.2 million distinct values), and a tie at the k-th place keeps k + 1. Every overshoot seen in R2, R2c and R7b is exactly one literal. The clause of 33 literals in R2 (seed 7) is most likely the same tie, not the start-state leftover named in R2's write-up.

**Fix:** `src/dl/tm.py` gained `budget_select`. `threshold` is the old rule and stays the default, so no earlier run changes. `rank` keeps exactly the k crossings with the largest keys, ties broken by position (`select_k`). Without ties both give the same selection (checked in the test). The draws of the random stream are the same; a run diverges from the old one only after its first tie.

Frozen by `scripts/freeze.py` before any R2d or R7c run.

## Runs

- **R2d:** arm B of R2c (`--max-literals 32 --init trim`) with `--budget-select rank`, Banking77 and HWU64, seeds 1 to 10. Arms A and C are R2c's files: same code path, deterministic for C, copied into `results/confirm/R2d/` before the runs.
- **R7c:** the TM arm of R7b (`--max-literals 16 --init trim`) with `--budget-select rank`, CLINC150, Banking77 and HWU64, seeds 1 to 10. The LR arms are R7's files, copied as in R7b.
- The test splits are the ones R2c and R7b scored; this is their second use, for a bug fix.

## Pass rules (test), unchanged

- **R2d:** R2c.1 and R2c.2 of `R2c_budget_trim_start.md`. R2d is met if both hold.
- **R7c:** R7b.1, R7b.2 and R7b.3 of `R7b_readable_budget.md`; R7c is met if all three hold. R7b.4 (out of scope) as a separate verdict.
- Applied by `scripts/rules_check.py R2d` and `R7c`.
