# Note: versions of llm_next_action.py during the dev runs (written 2026-10-26T23:59:41Z)

- At 23:59Z llm_next_action.py gained a guarded --test-registration option (it refuses unless the
  registration's .frozen.json exists; the test split is never opened otherwise). With the option absent, the dev path is
  the same code as before (dv = rows_of(dataset, "dev")), so prompts and answers of dev runs do not change.
- The previous file is kept as llm_next_action.v6_fc7eae6d.py (the hash pinned by PLAN_mwoz_dev.md and
  PLAN_ablation_dev.md). The file was saved at 23:58:56Z; every run that started after it loads the new file: both
  MultiWOZ arms (L9 started 00:06:34Z, after the edit), the ablation and the thinking arms. (Corrected 2026-10-27:
  an earlier line said L9 on MultiWOZ used the pinned file; the start times show it did not.)
- agent_preds.py gained the same kind of guarded test path (--test-registration); the previous file is kept as
  agent_preds.v2_*.py. No dev run of the agent is pending.
- (2026-10-27T01:23:59Z) ft_baseline.py gained a guarded --test-registration (test predictions at the best dev
  epoch) and --dry-run-dev (dev stands in for test); without them the dev path is unchanged. Previous files kept as
  ft_baseline.v1_8c328b8a.py (the PLAN_ft_dev.md pin) and ft_baseline.v2_*.py. agent_preds.py gained --dry-run-dev
  (previous file agent_preds.v3_*.py).
