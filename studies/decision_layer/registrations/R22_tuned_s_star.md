# R22 registration: the TM's s, tuned on dev, for the agent with no neural network at any stage (STAR)

**Status:** written after e74, e74c and e74d. The design, with R22.3, was fixed at 16:28 UTC. Frozen with `scripts/freeze.py` after the dry runs and before any R22 test run.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

Follows R21 (met: the C4-table agent within 0.10 of the potion agent on STAR test at s 8). Dev rules:
`logs/plan_e74_s_sweep.md` (frozen 2026-10-19T11:46:21Z) and `logs/plan_e74cd.md` (frozen 15:20:48Z). Every gate is
checked a second time, independently.

**The question.** Every agent run up to R21 used the TM's s 8. On STAR dev a larger s helps both agents. Three
questions follow:
1. Does the tuned s (s* = 16) raise the C4 agent on STAR test?
2. Does it close the TM-LR gap on the same bits? On test, R21 found LR 2.79 points ahead.
3. Is the C4 agent level with the bge-large agent when both are tuned the same way?

**What was known at the freeze:**
- the test numbers of R21 (STAR: tm_mem_c4 0.6261, tm_nomem_c4 0.4788, lr_mem_c4 0.6540), R19, R18 and R20;
- every e74, e74c and e74d dev number (below) and the dry runs';
- e72's dev numbers.

**Use of the test splits:** STAR's 6th (after R18, R19, R19's post-hoc LR refit, R20 arm 1, R21). ABCD's test split
is not read.

## Dev rows behind it (seed 1; `results/explore/abcd.jsonl`)

| Agent | s | 4k dev | 16k dev |
|---|---|---|---|
| C4 view B | 2 / 3 / 4 / 6 | 0.5138 / 0.5450 / 0.5662 / 0.5860 | |
| C4 view B | 8 | 0.6025 | 0.6317 (e72 / R21's seed 1) |
| C4 view B | 12 | 0.6306 | 0.6519 |
| C4 view B | **16 (s\*)** | **0.6349** | **0.6545** |
| C4 view B | 24 | 0.6310 | |
| bge-large, R18's view | 8 | 0.6131 | 0.6394 (e46) |
| bge-large, R18's view | 12 / 16 / 24 | 0.6372 / **0.6401** / 0.6375 | **0.6507** at s 16 |
| LR on C4 view B's bits (wide grid, C 0.01) | | | 0.6558 |

- **ABCD dev with the C4 agent at 4k:** s 8 0.8317, s 12 0.8398, s 16 0.8417. ABCD's test split is not read; this is
  the check that the STAR gain is not an ABCD loss.
- **Picks:** by the frozen rules (+0.3 at 4k over the reference, then the 16k and ABCD checks), s* = 16 for the C4
  agent and s 16 for bge-large.
- **The tuned dev gap at 16k:** C4 minus bge-large, both at s 16, is +0.38.
- **Selection is optimistic:** s was chosen on STAR dev by the same rule for both agents: among 8 values for the C4
  agent (2, 3, 4, 6, 8, 12, 16, 24) and 4 for bge-large (8, 12, 16, 24). The values above 12 were added for both after
  e74 showed the C4 curve still rising at 12. Dev numbers are best-epoch numbers.

## Arms (STAR test, `results/confirm/R22star/`; 16,000 clauses, T 16,000, 40 epochs, start state half, best-dev snapshot)

- `tm_mem_c4_s16`: the C4 agent (view B on the C4 table, through table_run.py, `--encoder none`), s 16, seeds 1 to 5.
- `tm_mem_bge_s16`: the bge-large agent (R18's view, `--encoder bge-large --parts mem,intent,step,words`), s 16,
  seeds 1 to 5.
- References copied by `scripts/r22_refs.py` (seed json and npz, bit_names; no models):
  - from R21star: `tm_mem_c4_s8_R21` (tm_mem_c4) and `lr_mem_c4_R21` (lr_mem_c4);
  - from R18: `tm_mem_bge_s8_R18` (tm_mem).

## Pass rules (test)

- **R22.1 (the tuned s helps):** `tm_mem_c4_s16` minus `tm_mem_c4_s8_R21` is at least +1.0 points on the point
  estimate, with the interval above 0.
- **R22.2 (the TM within 1 point of a tuned LR on the same bits):** `tm_mem_c4_s16` minus `lr_mem_c4_R21` is at least
  -1.0 on the point estimate. It is "strong" if the interval's lower end is also at least -1.0.
- **R22.3 (no neural network, equal tuning):** `tm_mem_c4_s16` minus `tm_mem_bge_s16` is at least -2.0 on the point
  estimate. It is "strong" if the interval's lower end is also at least -2.0.

Each rule is reported on its own; there is no combined "R22 met" line.

**Statistic** as R19 and R21: per test action turn, the mean correctness over seeds. Differences use a bootstrap over
the 611 test dialogues, 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`).

## Reported with no rule

- `tm_mem_bge_s16` minus `tm_mem_bge_s8_R18`: the bge-large agent's own gain from the tuned s.
- `tm_mem_c4_s16` minus `tm_mem_bge_s8_R18`, labelled "unequal tuning" (the C4 agent tuned, bge-large at s 8).
- Accuracy by position (first action, later actions).

## Commands, checks and dry runs

- **Jobs:** `logs/jobs_R22.txt` (10 jobs), one at a time through `scripts/seq_queue.sh` (`logs/R22_queue.log`, its own
  log), with cpu_load at most 4.5 cores. Order: the C4 seeds, then the bge-large seeds. The report command is the file's
  last comment.
- **Dry runs before the freeze** (`logs/jobs_R22_dry.txt`, own log `logs/R22_dry_queue.log`; output `logs/R22_dry.out`;
  `--dry-run`, tiny TM with 200 clauses and 2 epochs at s 16; dev stands in for test). Both passed (16:29:44-16:30:42 UTC),
  and their bits equal the e74 rows':
  - C4: train ea5abe1d, dev 3fc00b17 (= e72, R21, e74c);
  - bge-large: train fc79ea05, dev 65b2bb54 (= e74d).
- **The gate checker** `scripts/r22_check.py` is written before any test number is read.

## Wording, if met

- R22.1: "tuning the TM's s on dev (s 16) raises the agent with no neural network at any stage by X points on STAR
  test".
- R22.2: "within X points of a tuned logistic regression on the same bits".
- R22.3: "level with the bge-large agent when both are tuned the same way".
- Never: "beats the transformer" from a non-inferiority rule; "state of the art"; "parity with published TMs".

## Hashes and code

- **Jobs:** logs/jobs_R22.txt 4dc2e9437a635df9.
- **Checker:** scripts/r22_check.py 55a6658cbc038e19, written before any test number.
- **Refs script:** scripts/r22_refs.py 7d0df17af93a179e. Copied references: 25 files, every SHA-256 in logs/r22_refs.out (ed50de3c48908cd4).
- **Dry runs:** logs/R22_dry.out 53641e8d75793b4e.
- **Table:** data/tables/c4words_S1000_v1/table.npz e1771d9e. Example files as R18's and R21's.
- **R22's path** is R21's (src/dl/count_text.py, scripts/table_run.py, seq_queue.sh, abcd_confirm.py, report_abcd.py and
  the unedited src/dl files). The only new knob is abcd_confirm.py's existing --s option.
