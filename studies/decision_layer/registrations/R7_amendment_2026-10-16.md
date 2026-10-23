# R7 amendment, written 2026-10-16T09:08:28Z, while R7's last jobs run and before rules_check.py was run on R7

No change to the arms, the runs or the pass rules.

## Source changes since the freeze (2026-10-16T08:47:13Z), from scripts/diff_frozen.py

None of them is used by an R7 run. R7's runs use scripts/confirm_router.py and src/dl/{tm,boolean,data,linear,calib}.py, all unchanged.

- scripts/rules_check.py: gained `r7()`, the frozen R7 rules (R7.1 TM-readable minus LR-readable >= -0.5 in at least 2 of 3 datasets; R7.2 composite minus LR-raw >= -1.0 in at least 2 of 3). It also keeps the report against LR-readable as report_<ds>_vs_lrbits.json.
- scripts/tm_explore.py: option --init (start state, exploration only).
- New, not used by R7: src/dl/tm_init.py (start states), src/dl/mdp_h.py and scripts/r8_run.py (a 4-step clarify-or-route task), scripts/demo_agent_h.py, scripts/r7_readability.py (the readability numbers that R7 reports with no rule; the parts marked post hoc in its output are not registered).

## What was seen before this note

The R7 output files of the first seeds were opened while the rest ran: CLINC150 seeds 1 and 3 (test 0.9647 and 0.9622), Banking77 seeds 3 and 4 (0.9315 and 0.9308), and the LR arms. They were opened to check that the runs worked. Nothing was changed after they were opened.
