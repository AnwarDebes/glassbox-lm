# R2b registration: the sparse-input fix with tmu's literal budget

Companion to R2 (`R2_sparse_input_fix.md`, frozen 2026-10-15T22:18:42Z). It is written before any confirmatory run of R2. Frozen by `scripts/freeze.py`.

## Why a second registration

R2 fixed an "exact" budget: crossings into include are capped in every step, and Type II inclusions count too. In the first exploratory run of that variant (`e2b_lex_ml8`, val), the accuracy barely moved: 82.13 against 81.23 for the default. 3,922 of 4,000 clauses were still negation-only.

tmu, the library of CAIR, enforces its `max_included_literals` differently (`tmu/clause_bank/cuda/clause_feedback.cu`, read today). A clause that already includes more than the budget gets Type Ib feedback (erosion only) instead of Type Ia when it fires. Type II is unchanged. R2b tests that variant, `budget_mode="tmu"` in `src/dl/tm.py`. R2 stays as frozen and is run and reported as frozen.

## Data, input, arms, runs, statistic

The same as R2, with arm B replaced:

- **B' (tmu budget):** arm A plus `max_literals = L*`, `budget_mode = "tmu"`. L* is chosen by a rule fixed now, before those runs finished. It is the value among {8, 16, 32} with the highest val accuracy in the exploratory runs `e9_lex_tmu8`, `e9_lex_tmu16`, `e9_lex_tmu32` (seed 1). Ties go to the smaller L*.
- Arms A (default) and C (logistic regression on the same 2,000 bits) are the R2 runs themselves (identical settings and seeds). They are not run twice.
- B' runs on seeds 1 to 10 (TM seed 100·k), test scored once per run.

## Pass rules (test)

- **R2b.1 accuracy:** B' minus A is at least +3.0 points, and its interval excludes 0.
- **R2b.2 clause length:** in B', the median literals per clause is at most L*, and the 95th percentile is at most 2·L*, in every seed. Type II can push single clauses over the budget, so the maximum is reported but has no rule.
- **R2b is met** if R2b.1 and R2b.2 both hold.
- **R2b.3 against the linear control, reported separately:** B' minus C is at least -0.5 points (non-inferior). "B' beats C" only if its interval lies entirely above 0.

Reported with no rule: the share of non-empty clauses with at least one plain literal, the maximum literals, and the out-of-scope AUROC (top class sum) on oos_test.

## Code changes after the R2 and R4 freezes (before any confirmatory run)

- `src/dl/tm.py`: new field `budget_mode` (default "exact"). The default code path is unchanged: the exact branch now also checks `budget_mode == "exact"`, which is its default. `stats()` gains `p95_literals`.
- `scripts/tm_explore.py`, `scripts/confirm_router.py`: new option `--budget-mode`, default "exact".
- `scripts/confirm_router.py` also gained `--lr-input bits` and `--readout` (R4's TM-readout arm), and stores the normalised class sums per seed (`S_test`, `S_val`, `S_oos`) for the composites of R4.
