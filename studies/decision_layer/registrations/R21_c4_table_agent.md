# R21 registration: the agent on a word table made with no neural network at any stage (STAR and ABCD)

**Status:** both datasets passed their dev gate G-c4 (e72a ABCD at 05:10, e72b STAR at 05:35 UTC). This text was checked independently (05:4x UTC; judged by an AI model, not a person); the LR grid change below came from that check. Frozen with `scripts/freeze.py` after the dry runs and before any R21 test run.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

Follows R19 (met: the agent with no neural network at run time). Dev rules: `logs/plan_e72_count_table.md` (frozen
2026-10-19T05:05:14Z, sha256 1f800fad11299f26). Every gate is checked a second time, independently.

**The question.** R19's agent reads its text through potion-base-32M, a table distilled once, offline, from a
transformer (bge-base-en-v1.5). It replaces that table with the C4 word table. Does the agent keep R19's test
accuracy? How the table was made: counting + SVD over C4 text, with a C4 word list. Nothing from potion or any
neural model went into it.

**The agent:** view B, `--parts mem,words,xcust,xrecent --encoder none`, with the C4 table through
`scripts/table_run.py --table c4words-S1000-v1`. What runs per decision:
- a rule-based word tokenizer (the rule of the text-understanding line; a word not in the list is dropped);
- lookups in a fixed 63,091 x 500 table of C4 word vectors (data/tables/c4words_S1000_v1, table.npz e1771d9e);
- a bitwise OR of precomputed per-word bit codes (existsT: "some word of the customer's words so far / of the last
  3 turns is close to concept k at level j"; 512 concepts per field, fitted on train);
- the memory bits and the 1-2-gram bits;
- Boolean clauses and an integer vote.

**How each part was made:**
- the table: C4 text, then a C4 word list, then counting + SVD (by the W1 study; registered as W1-T);
- the concepts and levels: k-means + thresholds on the train texts;
- the decision: a Tsetlin machine.

No neural network is used at any stage, and nothing comes from potion or any transformer. The data are
human-written dialogues; ABCD's text was delexicalised by its authors, by a method not checked here. The code path
is checked:
- the run scripts and src/dl/static_text.py, abcd.py are unedited (diff 0 against the e70 and e71 freezes);
- in a C4-table run, potion's table cannot load (it raises), no encoder vector is loaded (`--encoder none`, empty
  arrays), and sentence_transformers, model2vec and transformers are not imported (`tests/test_count_text.py`).

**What was known at the freeze:**
- the test numbers of:
  - R19: STAR TM-mem 0.6271, TM-nomem 0.4766, LR-mem 0.6392; ABCD 0.8301, 0.6222, 0.8063;
  - R18 (STAR, bge-large): 0.6265, 0.4722, 0.6591; R9b (ABCD, bge-large): 0.8401, 0.6537, 0.8390;
  - R20, and R19's post-hoc numbers;
- every e72 dev number (below) and the dry runs', including the two LR dev numbers on the wide grid (STAR 0.6558 at C 0.01, ABCD 0.8119 at C 0.03);
- the share of tokens with no vector (step 1 of the plan);
- the registered W1-T result of the text-understanding line: on N7's test splits, within 1.0 of potion's table on 3 of 5 sets, below it on
  sentiment;
- e70 and e71 (Line B) as they stood.

**Use of the test splits:** STAR's 5th (after R18, R19, R19's post-hoc LR refit, R20 arm 1); ABCD's 13th.

## Dev rows behind it (seed 1, `results/explore/abcd.jsonl`; gate G-c4: potion's row minus 2.0)

| Run | Dataset | Clauses | C4 TM dev | potion TM dev (R19's seed-1 model) | Difference | Gate |
|---|---|---|---|---|---|---|
| e72_abcd_Bc4_c4000 | ABCD | 4,000 | 0.8317 | 0.8325 | -0.08 | passed (at least 0.8125) |
| e72_star_Bc4_c16000 | STAR | 16,000 | 0.6317 | 0.6343 | -0.26 | passed (at least 0.6143) |

- **ABCD:** -0.81 against bge-large dev (0.8398). Bits: 6,684 (potion 7,317), best epoch 14 (potion 26).
  First action 0.9064 (0.9044), later actions 0.8037 (0.8056).
- **STAR:** -0.77 against bge-large dev (0.6394). Bits: 7,227 (potion 7,938), best epoch 16 (potion 22).
  First action 0.7599 (0.7748, -1.49), later actions 0.6194 (0.6209).
  - The dev curve swings 1 to 2 points between epochs, as potion's does, so -0.26 is within one model's noise.
  - Every 2 epochs: 0.585, 0.612, 0.623, 0.632, 0.620, 0.618, 0.625, 0.617, 0.618, 0.613.
- **Selection:** the view and settings are R19's, fixed before e72. Only the table changed, and there was no
  choice among tables, so the dev numbers carry no selection beyond the best-epoch snapshot.

## Arms (test)

**STAR** (`results/confirm/R21star/`), at R19's settings: 16,000 clauses, T 16,000, s 8, 40 epochs, start state
half, best-dev snapshot, one seed per job. Each runs through table_run.py with `abcd_confirm.py --encoder none`.
- `tm_mem_c4`: view B on the C4 table, seeds 1 to 5.
- `tm_nomem_c4`: view B without mem (`words,xcust,xrecent`), seeds 1 to 5.
- `lr_mem_c4`: LR on tm_mem_c4's bits, seed 0, no rule. C is chosen on dev from 0.001, 0.003, 0.01, 0.03, 0.1,
  0.3, 1, 3, 10, 30, 100 (`--grid-c`; ties go to the first, the smaller C), R20's wider grid. This grid was chosen
  at 05:4x UTC: R19's default grid sat at its lowest C (0.1) on STAR, and tuning LR moved the TM-LR gap by
  1 point there (e67). An under-tuned baseline is not registered again.
- References copied by `scripts/r21_refs.py` (seed json and npz, bit_names; no models):
  - from R19: `tm_mem_static_R19`, `tm_nomem_static_R19`, `lr_mem_static_R19`;
  - from R18: `tm_mem_R18`, `tm_nomem_R18`, `lr_mem_R18`.

**ABCD** (`results/confirm/R21abcd/`), at R19's ABCD settings (4,000 clauses, T 4,000, otherwise as above):
- the same three arms, seeds 1 to 5 (LR seed 0);
- references copied from R19abcd (5 seeds) and from R9b (`tm_mem_R9b`, `tm_nomem_R9b`, `lr_mem_R9b`, 10 seeds).

## Pass rules (test, per dataset; both must hold for the dataset)

- **R21.1 (no neural network at any stage, little cost):** `tm_mem_c4` minus `tm_mem_static_R19` is at least -2.0
  points on the point estimate. It is "strong" if the interval's lower end is also at least -2.0.
- **R21.2 (the memory still pays):** `tm_mem_c4` minus `tm_nomem_c4` is at least +5.0 points, with the interval
  above 0.

**Statistic** as R19: per test action turn, the mean correctness over seeds. Differences use a bootstrap over test
dialogues (611 STAR, 1,004 ABCD), 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`). The
references enter with their own seeds (R19: 5, R18: 5, R9b: 10).

**The headline** needs both datasets met; one dataset met is reported as that dataset only.

## Reported with no rule

- `tm_mem_c4` minus the bge-large reference (`tm_mem_R18`, `tm_mem_R9b`).
- `tm_nomem_c4` minus `tm_nomem_static_R19`: the table's cost without memory, where the text carries everything.
- `tm_mem_c4` minus `lr_mem_c4`: the fair TM-LR number on the same bits.
- `lr_mem_c4` minus `lr_mem_static_R19`, labelled "grid differs; R19's LR was C 0.1, under-tuned on STAR". It mixes
  the grid with the table.
- Accuracy by position. The AST-weighted accuracy for ABCD (30 buttons; not the paper's 231-way B-Slot).
- e70's checks on the five `tm_mem_c4` ABCD models, in a dated note of their own: the 50 guards with M recomputed,
  the unguarded counterexamples, and the guards' cost on dev. They read the test split nowhere.

## Commands, checks and dry runs

- **Jobs:** `logs/jobs_R21.txt` (e130683ebb063142; 22 jobs), one at a time through `scripts/seq_queue.sh`, with
  cpu_load at most 4.5 cores before each start.
  - Order: the STAR TM jobs (mem and nomem, by seed), the ABCD TM jobs, then the two LR jobs on the wide grid.
  - A job that exits non-zero is run once more with the same command (seq_queue.sh), and that is logged.
  - The file was edited once before any use, to give the LR lines the wide grid. The old copy is in `logs/archive/`.
- **Dry runs before the freeze** (`--dry-run`: dev stands in for test; output in `results/explore/dryrun/`; log
  `logs/R21_dry.out`).
  - **What ran:** tiny TM (200 clauses, 2 epochs; the accuracy is meaningless) and LR on the wide grid.
  - **Order:** `logs/jobs_R21_dry.txt` ran its 2 STAR TM dry runs. Its queue was then stopped by PID at 05:40:32 UTC,
    after the grid change, before its ABCD lines and its one-C LR lines ran; those never ran.
    `logs/jobs_R21_dry2.txt` ran the 2 ABCD TM dry runs and the 2 LR dry runs on the wide grid.
  - **Results:** every bits_sha256 equals the e72 row of the same view and dataset.

| Dry run | Bits (train / dev) | Matches |
|---|---|---|
| STAR, with mem | ea5abe1d / 3fc00b17 | e72b (and the smoke) |
| STAR, without mem | b8b574ec / 9c90659d | no e72 row (new; recorded here) |
| ABCD, with mem | d7b0c2c7 / 56297e10 | e72a (and the smoke) |
| ABCD, without mem | 69c697e0 / 5f0963d5 | e72c |
| STAR LR, wide grid | ea5abe1d / 3fc00b17 | e72b; dev 0.6558 at C 0.01 |
| ABCD LR, wide grid | d7b0c2c7 / 56297e10 | e72a; dev 0.8119 at C 0.03 |

- On dev, the tuned LR is 2.41 points ahead of the TM on STAR (0.6558 against 0.6317) and 1.98 points behind it on ABCD (0.8119 against 0.8317). Potion's view B showed the same pattern: on STAR test, LR at its tuned C was 2.22 points ahead of the TM (R19, post hoc), while on ABCD the TM was ahead.
- **Report commands** (after all runs): report_abcd.py with the arms and pairs above, `--out report_R21<ds>.json`.

## Wording, if met

- "A dialogue agent with no neural network at any stage (C4 text, a C4 word list, counting + SVD, k-means +
  thresholds, a Tsetlin machine), within X points of the potion agent."
- After e70's guard rerun on these models, add: "made to follow its 50 guideline-derived order rules by veto
  clauses".
- **Never write:**
  - "first network-free TM text pipeline" (Kadhim et al. 2025);
  - "state of the art";
  - "parity with published TMs";
  - "the guidelines are verified";
  - "proved for all possible texts".
- e71's word-edit proofs are about the potion agent and are not claimed for this one.
- "No neural network at any stage" stays banned for every potion result (R19, R20, e70, e71).

## Hashes

- **Table:** data/tables/c4words_S1000_v1/table.npz e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a. It is the text-understanding line's
  export, copied 2026-10-19 04:50 UTC, and its MANIFEST checks.
- **Tokenizer reference:** tokenize_ref.py 2c3c1aff44ed34e0.
- **Empty encoder caches:** data/emb/abcd__none.npz and star__none.npz, fe366dc1df93b26f.
- **Example files:** as R19's.
  - STAR: train 47735157f3e0035c, dev e562736dd9afa1b7, test f9cc8f150b25d94d.
  - ABCD: train c9e0f9b58a13a3dd, dev 5ccc737f5be90860, test 190312c67b2258b4.
- **Copied references:** 120 files, every SHA-256 in logs/r21_refs.out (that list: f7ff6bd4aa068032).

## Code since the e72 freeze (`scripts/diff_frozen.py logs/plan_e72_count_table.frozen.json`)

- **New:** scripts/r21_refs.py (73ba39bf2d46bc92).
- **Changed:** none.
- **R21's path:**
  - the e72 files (src/dl/count_text.py, scripts/table_run.py, seq_queue.sh), as frozen there;
  - the unedited scripts/abcd_confirm.py (0ae66772) and report_abcd.py;
  - src/dl/static_text.py, abcd.py, tm.py, tm_init.py, boolean.py and linear.py, all unedited.
