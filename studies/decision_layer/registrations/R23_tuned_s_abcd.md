# R23 registration: equal tuning of the TM's s on ABCD, the agent with no neural network at any stage against the bge-large agent

**Status:** the design was fixed (19:24-19:27 UTC); the picks are e75's (`logs/e75_pick.out`, 2c8f3f6d). Frozen with `scripts/freeze.py` after the dry runs and before any R23 test run.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

Follows R22 (STAR test: the C4 agent at s 16 not worse than the equally tuned bge-large agent, +1.64 [+0.94, +2.35]).
Dev rules: `logs/plan_e75_abcd_s.md` (frozen 2026-10-19T19:25:09Z). Every gate is checked a second time, independently.

**The question.** Is the agent with no neural network at any stage also not worse than the bge-large agent on ABCD,
when each agent's s is tuned on ABCD's own dev by the same rule? On ABCD at s 8 (R21) it was -1.29 [-1.94, -0.66]; that
comparison was untuned on both sides.

**Use of the test splits:** ABCD's 14th, declared.
- Why it is worth it: R22.3 rests on STAR alone, and ABCD jobs cost minutes.
- Every earlier ABCD use was pre-registered, and the picks use ABCD dev only.
- The untuned gap (-1.29) is close to the -2.0 margin, so this is a real test.
- Every reuse weakens trust in the split a little, and that is stated.

**What was known at the freeze:**
- R21's and R22's test numbers;
- R9b's;
- every e72, e74 and e75 dev row;
- the C4 agent's ABCD dev rows at s 8, 12 and 16 (known before e75 was planned);
- the dry runs'.

## Dev rows behind it (ABCD dev, 4,000 clauses, seed 1; `results/explore/abcd.jsonl`)

| Agent | s 8 | s 12 | s 16 | s 24 | Pick |
|---|---|---|---|---|---|
| C4 view B | 0.8317 | 0.8398 | **0.8417** | 0.8341 | **s 16** (+1.00) |
| bge-large, R9b's view | 0.8398 | 0.8445 | 0.8447 | **0.8472** | **s 24** (+0.73) |

- **The tuned ABCD dev gap** (C4 s 16 minus bge-large s 24) is -0.55.
- **The bge-large pick is the grid's edge** (still rising at s 24). The frozen plan has no larger s, so a larger one could only make R23 harder for the C4 agent.
- **Rows:** the C4 rows at s 8, 12 and 16 come from e72a, e74b and e74c2 (identical settings, known when e75 was planned); s 24 and every bge-large row are e75's.

## Arms (ABCD test, `results/confirm/R23abcd/`; 4,000 clauses, T 4,000, 40 epochs, start state half, best-dev snapshot)

- `tm_mem_c4_s16`: the C4 agent at its ABCD pick s 16, seeds 1 to 5 (table_run.py, `--encoder none`).
- `tm_mem_bge_s24`: the bge-large agent (R9b's view, `--encoder bge-large --parts mem,intent,step,words`) at its ABCD pick s 24, seeds 1 to 5.
- References copied by `scripts/r23_refs.py`: R21abcd's tm_mem_c4 (s 8, 5 seeds) as `tm_mem_c4_s8_R21`, and R9b's
  tm_mem_short (s 8, 10 seeds) as `tm_mem_bge_s8_R9b`.

## Pass rule (test)

- **R23.1 (no neural network, equal tuning, ABCD):** `tm_mem_c4_s16` minus `tm_mem_bge_s24` is at least -2.0 on the
  point estimate. It is "strong" if the interval's lower end is also at least -2.0.

**Statistic** as R21: a bootstrap over the 1,004 test dialogues, 10,000 resamples, seed 0, 95% interval
(`scripts/report_abcd.py`).

## Reported with no rule

- Each agent against its own s 8 reference: `tm_mem_c4_s16` minus `tm_mem_c4_s8_R21`, and `tm_mem_bge_s24` minus
  `tm_mem_bge_s8_R9b`.
- AST-weighted accuracy (30 buttons).

## Wording, if met

- "On ABCD too, the agent with no neural network at any stage is not worse than the equally tuned bge-large agent".
- Never "better than", unless a future registered superiority test says so.

## Commands, checks, dry runs and hashes

- **Jobs:** logs/jobs_R23.txt (5b9317c84aa9ec1f), 10 jobs, one at a time through scripts/seq_queue.sh with its own log
  logs/R23_queue.log and cpu_load at most 4.5 cores. The C4 seeds run first. The report command is the file's last
  comment.
- **Dry runs** (logs/jobs_R23_dry.txt, logs/R23_dry.out ddf940aa63206c3a; tiny TM, 200 clauses, 2 epochs; dev stands in for test):
  both passed at 20:31 UTC, and their bits equal the dev rows':
  - C4: d7b0c2c7 / 56297e10, equal to e72a, R21abcd and e74c2;
  - bge-large: ee92ffd9 / f5d86956, equal to e75.
- **Checker:** scripts/r23_check.py (7fc7c14180fbb1ef), adapted from r22_check.py and written before any test number.
- **Copied references:** scripts/r23_refs.py (6e45ba26e4f03537) copied 32 files; every SHA-256 is in logs/r23_refs.out (b92779a8ca074d99).
- **R23's path** is R22's (src/dl/count_text.py, scripts/table_run.py, seq_queue.sh, abcd_confirm.py, report_abcd.py and
  the unedited src/dl files).
