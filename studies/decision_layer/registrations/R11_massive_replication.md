# R11 registration: does the readable decision layer (R7, R7c) hold on a dataset it never saw?

Every registered router test so far ran on CLINC150, Banking77 and HWU64, and their test splits have been used three times or more. R11 asks the readable layer's two met results to repeat, unchanged, on a fresh dataset with a test split never used in this study: **MASSIVE 1.1, en-US** (Amazon, CC BY 4.0; 60 intents; the file's own train / dev / test partitions, 11,514 / 2,033 / 2,974; `src/dl/data.py`, name `massive_en`; dev is the validation split).

Frozen by `scripts/freeze.py` before any model of any kind is trained or scored on MASSIVE (validation or test). The only thing done on MASSIVE before the freeze is the embedding of its texts (`scripts/embed.py`, bge-large, bge-m3, multilingual-e5-large; no labels involved). No setting below is tuned on MASSIVE: every setting is the frozen setting of R7 or R7c.

## Arms (all on bge-large, `BAAI/bge-large-en-v1.5`, as R7)

- **TM-readable (R7's setting):** view `proto:k=1:whiten=0.001+lex:2000`, 4,000 clauses, T 4,000, s 8, 40 epochs, best-validation snapshot, start state half. 10 seeds (1 to 10).
- **TM-readable-16 (R7c's setting):** the same with `--max-literals 16 --init trim --budget-select rank`. 10 seeds.
- **Composites:** two disjoint 5-machine composites per TM arm (seeds 1 to 5 and 6 to 10), as in R4 and R7.
- **LR-readable:** logistic regression on the same readable bits, C from the validation grid.
- **LR-raw:** logistic regression on the raw bge-large embedding, C from the validation grid.

Code path: `scripts/confirm_router.py` exactly as R7 and R7c ran it (job file `logs/jobs_R11.txt`), reports by `scripts/report_router.py`. Since R7 was frozen, the source changed only in ways that leave these arms' computation unchanged: `--init` and `--budget-select` options with the old behaviour as default (R7c used them), `src/dl/tm_init.py`, a MASSIVE branch in `src/dl/data.py`, two encoders added to `scripts/embed.py`.

Statistic as in R7: per test query the mean correctness over seeds of each arm; paired bootstrap over test queries, 10,000 resamples, seed 0, 95% percentile interval. The test split is scored once per run.

## Pass rules (test)

- **R11.1 (R7 replicates):** TM-readable minus LR-readable is at least -0.5 points, **and** TM-readable composite minus LR-raw is at least -1.0 point.
- **R11.2 (R7c replicates):** TM-readable-16 minus LR-readable is at least -0.5 points, **and** TM-readable-16 composite minus LR-raw is at least -1.0 point, **and** no clause of any TM-readable-16 model holds more than 16 literals.
- **R11.3 (R1.1 replicates):** the expected calibration error of TM-readable after one temperature (fitted on validation by NLL; 15 equal-width bins) is at most 0.05, mean over seeds.

Each rule is its own verdict. R11 as a whole is met if R11.1 and R11.2 both hold.

## Reported with no rule

- TM-readable (single machine) minus LR-raw, for both TM arms.
- The one-rule router "route to the intent whose whitened centre is nearest" (the view's own "nearest = X" bit), and the TM's agreement with it (it was post hoc in R7; here it is declared in advance).
- Clause statistics (share of clauses with no plain literal, lengths) and the share of decisions whose 3 strongest firing clauses alone give the same decision.

## What a failure would mean

R7.2 was met with little room on two of three datasets (-0.98, -0.99). If R11.1 fails on the composite rule, the cost of readability on fresh data is more than one point, and the R7 claim must be read as "within about one point on the datasets it was built on".
