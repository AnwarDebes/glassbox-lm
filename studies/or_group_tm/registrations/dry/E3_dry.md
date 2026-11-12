# E3 dry: dry run of the E3 pipeline on the DEV sentences (throwaway plan, no rule, no claim)

This text exists so that the job script, the guards and the checker of E3 can be run end to end before the
registration `registrations/E3_slide10_text.md` is fixed. It is not a registration of a result.

- Job: `registrations/E3_job.sh dry`. Same code and same setting as E3 (arm A6 of E2), on the DEV sentences, model
  seeds 1 to 5 (the seeds of the design rows) and 6 and 7 (in the place of E3's further seeds), runs under
  `runs_dry/`, run names `e3dry_text_s<seed>`.
- What it shows: that the job runs from the tests to the report, that every check of `src/e3_report.py` is carried
  out (none is skipped in dry mode), and what the chosen arm gives on the DEV sentences with the files that E3
  will pin. Seeds 1 to 5 repeat design rows, so the numbers of this dry run are DEV numbers that were known
  before; they go into the list "What was known" of E3.
- What it does not show: anything about the EVAL sentences. The record of this plan does not open them
  (`src/run_slide10.py` refuses `--split eval` with it).
- A dry run that is repeated after a change keeps its old folders under `runs_dry/` with the suffix `_SUPERSEDED`,
  and the reason is written into `logs/RUNS_slide10.md`.
