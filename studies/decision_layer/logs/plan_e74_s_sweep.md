# Dev plan e74: the TM's s for the agent on the C4 table (STAR dev; rules fixed before the first e74 job)

Proposed at 11:45 UTC. Frozen with `scripts/freeze.py` before the first e74 job. Dev only: no test
split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Why

- Every agent run so far used s 8. e47 varied only batch size and T.
- The N8 study (stage 1) shows that s moves TM accuracy a lot on sparse bits, and in opposite directions:
  - TREC: s 2 gives 79.97 against 75.06 at s 9;
  - MR: s 2 gives 51.4 to 52.1, near chance, against 61 to 64 at s 9 to 15.
- So the agent's best s is measured on STAR dev, not guessed.
- The gap to close is LR's lead on the same bits: 0.6558 (wide grid) against 0.6317 for the TM at s 8 on STAR dev.
  On test, LR is 2.79 points ahead (R21, no rule).

## Runs (GPU, one at a time, after the W2 study's part 2; cpu_load at most 4.5 cores before each start)

**Pattern:** `table_run.py --table c4words-S1000-v1 scripts/abcd_explore.py --encoder none --parts mem,words,xcust,xrecent
--arms tm --seed 1 --s <s> ...`. Everything but s is R21's: T = clauses, 40 epochs, init half, the best-dev snapshot,
seed 1. The jobs are in `logs/jobs_e74a.txt`, and `logs/jobs_e74b.txt` is written by `scripts/e74_pick.py`.

- **e74a, the screen:** STAR, 4,000 clauses (T 4,000), s in {2, 3, 4, 6, 8, 12}. Six jobs, in that order; s 8 is the
  in-sweep reference.
- **The pick** (`scripts/e74_pick.py`, recomputed from the rows):
  - the s with the highest 4k dev score;
  - if that score is less than 0.3 points above s 8's 4k score, the pick is s 8;
  - an exact tie goes to the s that comes first in the list.
- **e74b, only if the pick is not s 8:**
  - STAR at 16,000 clauses (T 16,000) with the pick, against s 8's 0.6317 (e72_star_Bc4_c16000 / R21's seed 1) and
    wide-grid LR's 0.6558;
  - ABCD at 4,000 clauses with the pick, against s 8's 0.8317 (e72_abcd_Bc4_c4000), so that a STAR gain is not an
    ABCD loss.
- **If the pick is s 8,** e74b does not run, and the result is "s 8 stays".

## The bar for a registration (written now; any registration is decided after the dev numbers)

A STAR test registration (the 6th use of STAR's test split) is considered only if both hold:
- **STAR:** the pick at 16,000 clauses reaches at least 0.6417 on dev (s 8's 0.6317 plus 1.0 point);
- **ABCD:** the pick at 4,000 clauses reaches at least 0.8267 on dev (s 8's 0.8317 minus 0.5 point).

Otherwise the six screen rows and any e74b rows go into the failure map as a dev result, and no test split is read.

## Reported, with no rule

- all six 4k rows (dev accuracy, best epoch, train time);
- the 16k and ABCD rows if they run;
- the gaps to s 8 and to LR.

**Selection is optimistic:** the pick is chosen on STAR dev among 6 values, and dev numbers are best-epoch numbers.
At 4,000 clauses STAR's view B curve was still rising at epoch 40 (e60_star_B_c4000), so the screen also measures
speed of learning.

## Files at the freeze

- scripts/e74_pick.py 1e4900b38101ba6c (the pick, recomputed from the rows).
- logs/jobs_e74a.txt 730223e8441207e6 (the six screen jobs).
- The run path is e72's: table_run.py, count_text.py and seq_queue.sh, all unchanged, plus the frozen abcd_explore.py (its --s option).
- **Start:** after the W2 study's part 2 is done (run log in the project notes, not published) and on a recorded decision.
