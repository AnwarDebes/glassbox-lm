# R7b registration: the readable decision layer with every clause bounded to 16 literals

Follow-up to R7, written after R7's test results (R7 met) and after exploratory val runs of the change. R7 left two weak points: about a quarter of its clauses are long exclusion lists (up to 755 literals; Part 1 of `FAILURE_MAP.md`), and its out-of-scope detection on CLINC150 is weak (AUROC 0.916). R7b adds the fix of Part 1, an exact literal budget with the start state that obeys it (`--max-literals L --init trim`, see R2c), to the R7 machine. Everything else is R7: encoder bge-large, view `proto:k=1:whiten=0.001+lex:2000`, 4,000 clauses, T 4,000, s 8, 40 epochs, best-val snapshot, CLINC150 / Banking77 / HWU64, 10 seeds, two 5-machine composites.

Frozen by `scripts/freeze.py` before any R7b run. Amendments go in a dated file.

## Exploration behind it (CLINC150 val, seed 1)

| Run | Budget | Start | Val accuracy | Out-of-scope AUROC (val, top sum) | Longest clause | Clauses with no plain literal |
|---|---|---|---|---|---|---|
| e19_rdw_bgelarge (= R7's setting) | none | half | 0.9700 | 0.9245 | 624 | 22% |
| e21_rdw_bgelarge_ml32_trim | 32 | trim | 0.9713 | 0.9475 | 32 | 87% |
| e21_rdw_bgelarge_ml16_trim | 16 | trim | 0.9707 | 0.9544 | 16 | 90% |

LR on the same bits: 0.9703. **L = 16** is chosen, after seeing both runs: its accuracy is within 0.1 points of L = 32 (2 of 3,000 queries), its clauses are half as long, and its out-of-scope AUROC is higher. No rule for this choice was fixed before the two runs; this is stated here.

## Arms

- **TM-readable-16:** the R7 machine with `--max-literals 16 --init trim`, seeds 1 to 10 per dataset (TM seed 100·k).
- **TM-readable-16 composite:** seeds 1 to 5 and 6 to 10, as in R7.
- **LR-readable, LR-raw:** the R7 arms. They are deterministic and their inputs are identical, so their files (`lr.json`, `lr_preds.npz`) are copied from `results/confirm/R7/` into the R7b folders before the runs start, instead of being fitted again.

## Pass rules (test)

- **R7b.1:** TM-readable-16 minus LR-readable is at least -0.5 points in at least 2 of the 3 datasets.
- **R7b.2:** TM-readable-16 composite minus LR-raw is at least -1.0 point in at least 2 of the 3 datasets.
- **R7b.3:** no clause of any seed holds more than 16 literals.
- **R7b.4 (out of scope, CLINC150):** the mean over the 10 seeds of TM-readable-16's out-of-scope AUROC (top class sum) is at least 0.936, that is, at least 0.02 above R7's 0.916.
- **R7b is met** if R7b.1, R7b.2 and R7b.3 hold. R7b.4 is a separate verdict.

Statistic as in R7. Reported with no rule: the readability report of `scripts/r7_readability.py --registration R7b --lr-base R7` (receipts, sufficiency, comparison with the one-rule nearest-centre router, marked post hoc where it is).
