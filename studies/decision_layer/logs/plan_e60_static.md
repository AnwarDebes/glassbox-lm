# Dev plan e60/e61: the decision agent with no neural network at run time (rules fixed before the first dev job)

Source plan: the plan in the project notes (not published), fixed at about 20:43 UTC on 2026-10-17; every gate is checked a second time, independently. This file fixes the dev rules S1 to S3. It is frozen with `scripts/freeze.py` before the first e60 job. Dev splits only: no `abcd_confirm.py` call and no test read in e60 and e61.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## The three network-free views

All use `--encoder potion-32m` (data/emb/<ds>__potion-32m.npz from `scripts/static_prepare.py`: the normalised average of the token vectors of potion-base-32M). No bge-large data is loaded.

| View | `--parts` | What runs per decision |
|---|---|---|
| A, static mean vectors | `mem,intent,step,words` | table lookups, an average, a fixed linear map (the whitened centres) and comparisons |
| B, lookup + OR | `mem,words,xcust,xrecent` | table lookups of precomputed per-token bit codes and a bitwise OR (existsT) |
| A+B | `mem,intent,step,xrecent,words` | both; it leaves out xcust (long customer texts may saturate) |

Block order is fixed in code: mem, intent, step, words, mem2, fstep, fwords, xcust, xrecent.

## Runs

Every job runs with `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2`, at most 3 at a time through `scripts/master_queue.sh`. Command pattern:

`.venv-dl/bin/python scripts/abcd_explore.py --dataset <star|abcd> --encoder potion-32m --parts <view> --arms tm,lr_bits --clauses <n> --seed 1 --tag e60_<ds>_<A|B|AB>_c<n>`

- **e60a:** A, B and A+B at 4,000 clauses on STAR and on ABCD (6 jobs).
- **e60b:** the S1 survivors at 16,000 clauses on STAR, with `--save`, and DL_GPU_FRAC sized from the 4,000-clause peak.
- **e61 (twins, reported with no rule):** the chosen view without mem on both datasets (arms tm, lr_bits; STAR at 16,000 clauses, ABCD at 4,000).

## Rules

- **S1, screening:** at 4,000 clauses on STAR dev, a view more than 3.0 points below the best of the three is dropped.
- **S2, choice:** at 16,000 clauses on STAR dev, take the first of B, A, A+B (in that order) within 0.5 points of the best. B comes first because it is pure logic; A has fewer bits. The same view is used for both datasets, so it is one agent design.
- **S3, gate, per dataset:** a dataset goes to the registered test only if the chosen view on its dev split is at least its bge-large dev TM row minus 3.0 points:
  - STAR: 0.6394 (e46_star_mem_c16000, 16,000 clauses) gives 0.6094, read at 16,000 clauses;
  - ABCD: 0.8398 (e25_full, 4,000 clauses) gives 0.8098, read from the e60a row of the chosen view.
  - Otherwise its costs (every view, TM and LR) go into FAILURE_MAP.md and docs/results.md, and its test split stays unread.
- **Crashes:** one rerun with the same command, logged. A second crash drops the view.
- A dev loss between 2 and 3 points passes the gate but would likely fail R19.1 (at most 2.0 points below bge-large on test). The registration will say so.

## What is checked at G2

- Every job has its row in results/explore/abcd.jsonl.
- The freeze time of this file is earlier than the first job.
- S1 to S3 are recomputed from the rows.
- bits_sha256 is equal across runs of the same view and dataset.
