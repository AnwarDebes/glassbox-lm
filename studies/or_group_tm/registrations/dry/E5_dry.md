# E5 dry: dry run of the E5 pipeline on the DEV pairs (throwaway plan, no rule, no claim)

This text exists so that the job script, the guards and the checker of E5 can be run end to end before the
registration `registrations/E5_wikipedia_test.md` is fixed. It is not a registration of a result.

- Job: `registrations/E5_job.sh dry`. Same code and same setting as E5 (arm TB of E4, scored once after the last
  pass), on the DEV pairs of `sa`, `np` and `ns_all`, model seed 1 for each (the seed of the design rows and of the
  first DEV seed of the arm), runs under `runs_dry/`, run names `e5dry_<data>_s1`. The stock model enters with the
  five folders in which the study scored its saved models on the DEV pairs.
- What it shows: that the job runs from the tests to the report, that every check of `src/e5_report.py` is carried
  out (none is skipped in dry mode), that the three models are the models of the DEV rows of seed 1 (same numbers),
  and that the report reproduces the study's DEV numbers of the baselines and of the stock seeds. The numbers of this
  dry run are DEV numbers that were known before; they go into the list "What I knew" of E5.
- What it does not show: anything about the test pairs. The record of this plan does not open them
  (`src/run_wiki.py` and `src/wiki_report.py` refuse `--split test` with it).
- A dry run that is repeated after a change keeps its old folders under `runs_dry/` with the suffix `_SUPERSEDED`,
  and the reason is written into `logs/RUNS_wiki.md`.
