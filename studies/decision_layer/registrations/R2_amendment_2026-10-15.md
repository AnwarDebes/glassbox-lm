# R2 and R2b: amendment written 2026-10-15 between 22:44:09Z and 22:44:30Z (before the first confirmatory run, launched 22:44:30Z)

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

## Selections applied (rules of the frozen texts)

- **R2, L***: exploratory val accuracy of the exact budget was 0.8213 at 8 (`e2b_lex_ml8`), 0.8500 at 16 (`e2b_lex_ml16`) and 0.8717 at 32 (`e2b_lex_ml32`). **L* = 32.**
- **R2b, L***: the tmu-style budget gave 0.5597 at 8 (`e9_lex_tmu8`), 0.5567 at 16 (`e9_lex_tmu16`) and 0.5853 at 32 (`e9_lex_tmu32`). **L* = 32.** On val this variant is far below the default (0.8123, `e2_lex_default`). R2b is run as registered all the same.

## Source changes since the freezes, and why they do not touch the registered arms

`scripts/diff_frozen.py` lists `src/dl/tm.py`, `src/dl/linear.py`, `src/dl/mdp.py`, `scripts/confirm_router.py`, `scripts/tm_explore.py`, `scripts/r6_run.py` as changed, plus new scripts.

- **`src/dl/tm.py`:**
  - added `budget_mode` (default "exact", whose branch is the frozen code);
  - added the "tmu" and "block_full" branches, which run only when selected;
  - `stats()` gained `p95_literals`.
  - Arm A (no budget) and arm B (exact) run the frozen code paths.
- **`src/dl/linear.py`:** added `recall_at_precision` (a new metric, reported without a rule).
- **`src/dl/mdp.py`, `scripts/r6_run.py`:** the R6 environment, not used by R2.
- **`scripts/confirm_router.py`:**
  - options `--budget-mode`, `--readout`, `--lr-input` (bits for arm C);
  - stores class sums per seed;
  - also writes the out-of-scope recall at 95% precision.
- **`scripts/tm_explore.py`:** exploration only.

## Run plan

- A: `--name A_default` (no budget).
- B: `--name B_exact32 --max-literals 32`.
- B': `--name Bp_tmu32 --max-literals 32 --budget-mode tmu`.
- C: `--name C_lr_bits --lr-only --lr-input bits`.
- The TM arms use seeds 1 to 10.

## Note added 2026-10-16 (internal audit 1)

- **Timing of this file.** It was first written before the launch of 22:44:30Z: the job file `logs/jobs_R2.txt`, built in the same command, already held L* = 32 at 22:44:22Z. Its header line was then edited with `sed -i` at 22:44:36Z to state the clock time, and that edit recreated the file. So the file system shows a creation time 6 s after the launch. The first R2 result came at 22:55:49Z.
- **Code-change list.** It is complete. R2's runs used the `src/dl/tm.py` from before the chunked update, which they never reach anyway (4,000 x 4,000 literals).

## Note added 2026-10-16T11:34:06Z (correction of the cause of the R2.2 failure)

The write-up called the one clause of 33 literals (seed 7) a leftover of the start state. That is most likely wrong. R2c and R7b, whose start state can never exceed the budget, also ended with clauses exactly one literal over it. The cause is a tie of two float32 random keys in the exact budget's selection, which keeps k + 1 crossings (`tests/test_budget_exact.py`, `R2d_R7c_exact_budget_reruns.md`). The verdict of R2 does not change.
