# Dev plan e75 (2026-10-19): the TM's s on ABCD dev for both agents (rules fixed before the first job)

Proposed at 19:24 UTC to give R22's equal-tuning comparison a second dataset. Frozen with
`scripts/freeze.py` before the first e75 job. Dev only: no test split is read. The test is R23 (ABCD's 14th use),
written and frozen after these runs.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Why

R22.3 found the C4 agent not worse than the equally tuned bge-large agent on STAR test (+1.64 [+0.94, +2.35]), but on
one dataset. On ABCD at s 8 (R21) the C4 agent was 1.29 points below the bge-large agent (R9b). That comparison was
untuned on both sides. ABCD's own dev should pick each agent's s.

## The screens (ABCD dev, 4,000 clauses, T 4,000, seed 1; R9b's and R21's settings but s)

- **The C4 agent** (view B on the C4 table, through table_run.py, `--encoder none`).
  - Its ABCD rows at s 8, 12 and 16 exist, with identical settings: e72_abcd_Bc4_c4000 0.8317, e74_abcd_Bc4_c4000_s12
    0.8398, e74c_abcd_Bc4_c4000_s16 0.8417.
  - They were run under earlier plans, as checks, not to pick an ABCD s. This plan adds s 24 (1 job) so that both
    agents are screened on the same four values.
- **The bge-large agent** (R9b's view: `--encoder bge-large --parts mem,intent,step,words`, 1,581 bits, as R9b's and e25_full's):
  s 8, 12, 16 and 24 (4 jobs). s 8 is the in-sweep reference; e25_full, the old row, was 0.8398.
- **Jobs:** `logs/jobs_e75.txt`, one at a time, own log `logs/e75_queue.log`, cpu_load gate 4.5 cores.

## The picks (`scripts/e75_pick.py`, recomputed from the rows)

- For each agent: the highest dev score among s 8, 12, 16 and 24.
- It must be at least 0.3 points above that agent's own s 8 row, otherwise s 8.
- An exact tie goes to the first of [8, 12, 16, 24].
- There is no 16k step: ABCD's registered size is 4,000 clauses.

## R23 (sketch; written and frozen after e75)

- **Test use:** ABCD's 14th, declared.
- **Arms:** both agents at their ABCD dev picks, 4k, seeds 1 to 5.
- **R23.1:** C4 minus bge-large is at least -2.0 on the point estimate; strong if the lower end is at least -2.0.
- **No rule:** each agent against its own s 8 reference (R21abcd tm_mem_c4; R9b tm_mem_short, 10 seeds).
- **Cost:** about 1 GPU-hour.

## Selection caveat

s is chosen on ABCD dev among 4 values for each agent, by the same rule. The C4 agent's rows at s 8 to 16 were
already known when this plan was written.

## Files at the freeze

- scripts/e75_pick.py 3c9d764e6a842d82.
- logs/jobs_e75.txt a6a5a27d02d90f3c (5 jobs).
- **Start:** on a recorded decision, after the text-understanding line's G1.
