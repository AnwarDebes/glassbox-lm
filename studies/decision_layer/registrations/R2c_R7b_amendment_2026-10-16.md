# R2c and R7b amendment, written 2026-10-16T11:34:06Z, after both verdicts

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

No change to the frozen texts, the runs or the verdicts. This note records the verdicts, their cause and what follows.

- **R2c: not met.** R2c.1 met in both datasets (Banking77 B minus A +4.71 [+4.00, +5.44] points, HWU64 +4.09 [+2.79, +5.41]). R2c.2 fails: Banking77 seed 6 holds two clauses of 33 literals (limit 32). HWU64: none over 32.
- **R7b: not met.** R7b.1 met in 3 of 3, R7b.2 met in 3 of 3, R7b.4 met (CLINC150 out-of-scope AUROC 0.950 against the bar 0.936). R7b.3 fails: one clause of 17 literals in CLINC150 seed 2 and one each in HWU64 seeds 6 and 10 (limit 16).
- **Cause:** a tie of two float32 random keys at the k-th place makes the exact budget keep k + 1 crossings (`tests/test_budget_exact.py`; registration `R2d_R7c_exact_budget_reruns.md`). The "trim" start state did what it was meant to do, but the start state was not the cause. Every overshoot, here and in R2, is exactly one literal.
- **What follows:** `budget_select="rank"` in `src/dl/tm.py` (the default stays the old rule, so earlier runs are unchanged), and the re-runs R2d and R7c, registered and frozen at 2026-10-16T11:33:29Z.

## Source changes since the R2c (09:42:34Z) and R7b (09:45:25Z) freezes, none used by their runs

- src/dl/mdp_h.py, scripts/r8_run.py, scripts/report_r8.py, scripts/demo_agent_h.py, scripts/report_agent.py: R8 and R5c.
- scripts/rules_check.py: `r2c` and `r7b` take the registration name, for the re-runs.
- src/dl/tm.py: `budget_select` and `select_k` (11:30, after all R2c and R7b runs had started). scripts/confirm_router.py and scripts/tm_explore.py: option --budget-select.

## Corrections added 2026-10-16T12:05:17Z (internal audit 3)

- scripts/demo_agent_h.py is listed above as changed since the freezes; it is not (its hash equals the frozen one).
- **R7b's test splits:** R7b scored the same CLINC150, Banking77 and HWU64 test splits as R7 (their second use with this view), and it was designed after R7's test results (out-of-scope AUROC 0.916, the long clauses). Its budget L was chosen on val only. The R7b text did not say that the splits are the same; it says so here.
- R2c's text says R7 scored the Banking77 and HWU64 test splits "with embedding inputs". R7's view also holds the same lex:2000 word bits, inside the readable view; no purely lexical run had scored them.
