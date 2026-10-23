# Dev plan e74c and e74d (2026-10-19): s above the grid's edge, and the same tuning for the bge-large agent (rules fixed before the first job)

Decided at 15:19 UTC, after e74 (`logs/plan_e74_s_sweep.md`, frozen 11:46:21Z). Frozen with
`scripts/freeze.py` before the first e74c or e74d job. Dev only: no test split is read. R22 (STAR test) is written and
frozen after these runs.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Why

- **e74c:** in e74's screen accuracy rose with s at every step up to the grid's edge (s 12: 0.6306 at 4k, +2.81 over
  s 8). So the best s may lie above 12. The 6th use of STAR's test split should register the best s, not an edge value.
- **e74d:** the C4 agent at s 12 is +1.25 over the bge-large agent on STAR dev. That compares a tuned agent with an
  untuned one (bge-large at s 8), so it cannot be claimed as is. The bge-large agent gets the same tuning.

## e74c: the C4 agent, s 16 and 24 (STAR, 4,000 clauses, seed 1)

- **Jobs:** `logs/jobs_e74c.txt`. table_run.py with the C4 table, view B, as e74a; only s changes.
- **The pick** (`scripts/e74_pick2.py --stage c`):
  - candidates: s 12 (e74a's row, 0.6306), 16 and 24;
  - the highest 4k dev score, but it must be at least 0.3 points above s 12's, otherwise s 12;
  - an exact tie goes to the first of [12, 16, 24].
- **If the pick is not s 12:**
  - its STAR 16k run and ABCD 4k run follow (`logs/jobs_e74c2.txt`, written by the script);
  - s* (the s R22 registers) is the pick, unless its STAR 16k dev is below s 12's 0.6519 or its ABCD 4k dev is
    below 0.8267 (e74's ABCD bar). In either case s* = 12.
- **If the pick is s 12,** then s* = 12 and nothing more runs.

## e74d: the bge-large agent, s 8, 12, 16 and 24 (STAR, 4,000 clauses, seed 1)

- **The agent:** R18's view (`--encoder bge-large --parts mem,intent,step,words`, 2,715 bits), R18's settings but s.
- **Jobs:** `logs/jobs_e74d.txt`, through `scripts/abcd_explore.py` directly, as the R18 dev rows were.
  - s 8 is the in-sweep reference. e45_star_mem, the old 4k row at s 8, was 0.6131.
- **The pick** (`scripts/e74_pick2.py --stage d`):
  - the highest 4k dev score, but it must be at least 0.3 points above s 8's in-sweep row, otherwise s 8;
  - an exact tie goes to the first of [8, 12, 16, 24].
- **If the pick is not s 8,** its STAR 16k run follows (`logs/jobs_e74d2.txt`). Otherwise the 16k number is
  e46_star_mem_c16000's 0.6394.
- **Reported:** the tuned dev gap at 16,000 clauses (the C4 agent at s* minus the bge-large agent at its pick). After
  e74d, it is decided whether R22 gets R22.3 ("tuned C4 minus tuned bge-large at least -2.0", with
  bge-large's test runs at its own s) or only reports the tuned dev numbers beside it.

## Order and CPU

- One job at a time, after a recorded decision to start: e74c, its follow-ups if any, then e74d and its follow-up if any.
- Each queue has its own log: logs/e74c_queue.log, e74c2_queue.log, e74d_queue.log, e74d2_queue.log.
- The cpu_load gate of 4.5 cores applies.

## Selection caveat

s is now chosen on STAR dev among up to 8 values for the C4 agent and 4 for bge-large. Dev numbers are best-epoch
numbers and are optimistic. R22's test decides.

## Files at the freeze

- scripts/e74_pick2.py 2d9406ebc8f040ad (the picks and the final s*, recomputed from the rows).
- logs/jobs_e74c.txt cfdb3d019ce1144a (2 jobs).
- logs/jobs_e74d.txt aed9b8021488d224 (4 jobs).
- The follow-up job files (logs/jobs_e74c2.txt, logs/jobs_e74d2.txt) are written by the script, only when a pick changes.
- **Start:** on a recorded decision, after the W3 study's GPU part.
