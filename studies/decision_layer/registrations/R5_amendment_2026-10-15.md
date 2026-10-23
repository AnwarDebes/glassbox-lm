# R5: amendment written 2026-10-15 at 23:42Z, before any confirmatory run of R5

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

## Selections applied (rule of the frozen text: highest final val accuracy on CLINC150 domains among the e3 and e6 runs, per learner family)

**TM bandit.** Candidates (final val accuracy):

| Setting | Val accuracy |
|---|---|
| epsilon 0.05 | 0.9267 |
| epsilon 0.2 | 0.9347 |
| epsilon 0.5, decay 0.0002 | 0.9297 |
| epsilon 0.1, top-3 | 0.9347 |
| epsilon 0.1, top-3, replay 4 | 0.9517 |

**Chosen: epsilon 0.1, top-3 exploration, replay 4.**

**LinUCB.** Candidates: alpha 0.2 (0.9350), alpha 1.0 (0.9443), alpha 0.5 (0.9417). **Chosen: alpha 1.0.**

**Epsilon-greedy logistic.** Candidates (final val accuracy):

| Setting | Val accuracy |
|---|---|
| epsilon 0.05, lr 0.05 | 0.9320 |
| epsilon 0.1, lr 0.01 | 0.9327 |
| epsilon 0.1, top-3, lr 0.05 | 0.9437 |
| epsilon 0.1, top-3, lr 0.05, replay 4 | 0.9367 |

**Chosen: epsilon 0.1, top-3, lr 0.05, no replay.**

## Source changes since the freeze (`scripts/diff_frozen.py`)

- `src/dl/tm.py`: the clause-chunked update, added after the freeze (only above 50,000,000 clause-literal pairs; R5 uses 4,000 x 5,376 = 21,504,000, so the unchunked path). The literal-budget modes were already in the file at the freeze (corrected after internal audit 2); R5 uses no budget.
- `src/dl/explain.py`: printing only. `scripts/r6_run.py`: R6 only.
- New helper scripts.
- `src/dl/bandit.py` and `scripts/bandit_run.py` are unchanged since the freeze.

## Runs

The jobs in `logs/jobs_R5.txt` (built by `scripts/build_r5_jobs.py`), 123 runs:
- 3 tasks x 10 seeds x {TM, LinUCB, epsilon-greedy logistic, sup-TM};
- plus sup-LR once per task.
