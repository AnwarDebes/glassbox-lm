# Dev plan e72: the agent on a word table made with no neural network at any stage (rules fixed before the first e72 job)

Source: a proposal in the project notes (not published; 2026-10-19, 04:4x and 04:47 UTC) and the word table's README
(`data/tables/c4words_S1000_v1/README.md`, a verified copy; not published). This file is frozen with `scripts/freeze.py` after the
pre-checks and before the first e72 job. Dev splits only: no `abcd_confirm.py` call and no test read in e72.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## The question

R19's agent (view B) reads its text through potion-base-32M. That table was distilled once, offline, from a
transformer (bge-base-en-v1.5), so R19 has no neural network at run time but one at build time. Does the agent keep
its dev accuracy when its word table is the C4 word table, where no neural network was used at any stage?

## What changes, and what does not

Only the inside of xcust and xrecent changes:
- **Table:** potion's 63,091 WordPiece pieces x 512 are replaced by c4words_S1000_v1. That table holds the 63,091
  most frequent words of 1B tokens of C4 en, 500 dimensions, made by counting + SVD (PPMI-SVD). The corpus was
  deduped against ABCD and STAR dev/test turns. table.npz e1771d9eaf4a07f7 is byte-identical to the W1-T file.
- **Tokenizer:** WordPiece, which covers every text with pieces, is replaced by the table's word rule: lower-case; split
  n't, 's, 're, 've, 'll, 'd, 'm; tokens are letter/digit runs and single punctuation marks. A token not in the list
  gets no vector and is dropped.

Unchanged:
- the existsT recipe: K 512, 25 iterations, a CPU generator seeded 0 on one thread, as in R19 (not the text-understanding line's CUDA
  draw), and Thermo(8) fitted on train;
- the 66 memory bits and the 1,000 1-2-grams;
- the TM settings (R19's), seed 1, and the best-dev snapshot.

## Step 1 (done, 04:51 UTC): tokens without a vector, train/dev

`scripts/count_table_coverage.py` (598e381c), counted per example row, with the export's reference tokenizer.
Output in `results/explore/count_table_coverage.json` (3dc7cb91) and `logs/e72_coverage.out`.

| Dataset | cust (xcust) | recent (xrecent) | customer's last 3 (xcrecent) |
|---|---|---|---|
| ABCD train / dev | 0.65% / 0.66% | 0.46% / 0.49% | 0.72% / 0.79% |
| STAR train / dev | 0.85% / 0.80% | 0.70% / 0.73% | 0.86% / 0.87% |

- **ABCD drops** names and usernames (afzal, aphoenix, jbanter), typos (acount, subcription), brands (acmebrands) and
  "covid". The placeholders keep their vectors: `<order_id>` becomes "<", "order", "_", "id", ">". No text is left
  with no known token.
- **STAR drops** place names (shadyside), long numbers and quoted words such as 'hilton. The rule's `'[a-z]+` makes
  "'hilton" one token, and it has no vector. Texts with no known token: 8 train and 1 dev customer texts. Their
  field's existsT bits are all off.
- **The port of the rule equals the reference** on all 284,085 texts (0 mismatches).
- **For comparison**, the intent val sets of the text-understanding line drop 0.11% to 0.80%; potion drops nothing, because of its subword
  fallback.

## Code: new files only

**Why new files:** the e70 and e71 freezes hash `src/dl/static_text.py`, `dl/abcd.py`, `abcd_explore.py` and
`abcd_confirm.py`, and frozen steps still start new processes (e70's MLP-64; e71's k = 3 and 5, seeds 2 to 5,
e71b). So e72 edits none of them. The default path is unchanged by construction (diff 0), and no pristine copy is
needed.

- **`src/dl/count_text.py` (new):**
  - `CountTable`: loads table.npz with numpy only, after checking its SHA-256 against the constant and the MANIFEST.
    The float16 rows are cast to float32 and L2-normalised. It provides `ids()` (the ported rule, with
    "(nothing yet)" for an empty text, as in static_text), `id_to_token` and the hashes.
  - `ExistsTC(ExistsT)`: the same fit, transform and transform_max as ExistsT, with all table access through one
    method. `info` adds the table's name, its file and array SHA-256 and the tokenizer.
  - `patch(name)`: makes `dl.static_text.ExistsT` build ExistsTC on that table, so ABCDView's xcust, xrecent and
    xcrecent read it. It also makes `dl.static_text.table` and `StaticTable` raise, so potion cannot be loaded in
    such a run.
- **`scripts/table_run.py` (new):** `table_run.py --table c4words-S1000-v1 scripts/abcd_explore.py <args>`.
  - It applies `patch`, then runs the unedited target with runpy. It logs the target's SHA-256.
  - It requires `--encoder none` and parts that read no encoder vector (mem, words, xcust, xrecent, cwords,
    xcrecent).
- **`scripts/none_prepare.py` (new):** `data/emb/<ds>__none.npz` with empty arrays under the 9 cache keys. A run then
  loads no encoder vector; a part that needs one fails.
- **`tests/test_count_text.py` (new):** the pre-checks below.

## Pre-checks (before the freeze; CPU, one core, nice; train/dev only)

1. **The subclass changes only the table.** On ABCD train/dev texts, both fields, ExistsTC given potion's own table
   must give the same concept, threshold and code hashes, the same names and the same bits as ExistsT.
2. **Tokenizer:** count_text's port equals the reference on every train/dev text of ABCD and STAR. Step 1 checked
   the same rule: 0 mismatches.
3. **Lookup + OR equals the max-cosine reference** (transform = transform_max) on the C4 table: train/dev, both
   fields, both datasets.
4. **No potion:** a C4-table fit + transform runs while static_text.table and StaticTable raise, and
   sentence_transformers is not in sys.modules afterwards.
5. **Smoke, end to end** (GPU, seconds; the accuracy is meaningless):
   - `table_run.py ... abcd_explore.py --arms tm --clauses 200 --epochs 2`, tags e72_smoke_abcd and e72_smoke_star;
   - their bits_sha256 and static_info are recorded, and the real runs must reproduce them.

### Results of pre-checks 0 to 4 (all passed, 05:01:29 to 05:04:40 UTC, before this freeze)

`tests/test_count_text.py` (0f5a66995b2d2882), output `logs/e72_precheck.out` (59e530c643355fa9), one CPU thread.
- **0 (an added check):** the four frozen files equal both freeze records, before and after the checks:
  static_text.py ec89f28c, abcd.py a66c0c63, abcd_explore.py d3c53f57, abcd_confirm.py 0ae66772.
- **1:** ExistsTC given potion's table equals ExistsT on ABCD, for cust (3,218 bits) and recent (3,033 bits):
  - the same info (concept, threshold and code hashes), names, concepts, thresholds and codes;
  - the same transform and transform_max bits on all 29,190 train and 3,684 dev texts.
  - These bit counts are R19's.
- **2:** the port equals the reference, tokens and row ids, on 284,085 texts.
- **3:** OR equals max on every train and dev text, both datasets, three fields, and two fits are identical.
  - Bits per field on the C4 table (potion's in brackets): ABCD cust 2,865 (3,218), recent 2,753 (3,033), crecent
    2,973; STAR cust 3,045 (3,374), recent 2,802 (3,184), crecent 3,134.
  - The C4 table has fewer bits because fewer thresholds stay distinct.
  - About 0.53 to 0.56 of the dev bits are on.
- **4:** with the patch and the empty "none" cache, views B (6,684 bits on ABCD dev) and C (6,904) fit and transform.
  static_text.table raises, and sentence_transformers, model2vec and transformers are not imported.

### Files at this freeze (first 16 hex of SHA-256)

- **Code:**
  - src/dl/count_text.py 5f201106593c065f;
  - scripts/table_run.py 0af00bd556741c8b;
  - scripts/none_prepare.py 2d017bf22a928912;
  - scripts/seq_queue.sh ca41d9181b4c9ae6 (one job at a time, with the load check and one rerun);
  - scripts/e72_gate.py d1bed82836508518 (the gate, recomputed from the rows);
  - scripts/count_table_coverage.py 598e381c6eea25fe.
- **Data:**
  - data/tables/c4words_S1000_v1/table.npz e1771d9eaf4a07f7;
  - data/emb/abcd__none.npz and star__none.npz fe366dc1df93b26f (the same empty arrays).
- **Jobs:**
  - `logs/jobs_e72_smoke.txt` 62a8c3e5b30f0243: the smoke, run first and checked;
  - `logs/jobs_e72.txt` a4384a579ff28a0c: e72a, e72b and e72c, in that order.
  - Both are run by `scripts/seq_queue.sh <jobs> logs/e72_queue.log`.

## Runs (GPU; after a recorded decision and the freeze)

**Settings:**
- `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25`;
- one job at a time, ABCD first;
- before each start, `scripts/cpu_load.py 10` must show at most 4.5 cores. A job adds about one core, and e71 keeps
  its slot.

**Pattern:** `.venv-dl/bin/python scripts/table_run.py --table c4words-S1000-v1 scripts/abcd_explore.py
--dataset <abcd|star> --encoder none --parts mem,words,xcust,xrecent --arms tm --clauses <4000|16000> --seed 1 --save
--tag e72_<ds>_Bc4_c<n>`. The other settings are the script's defaults, which are R19's: T = clauses, s 8,
40 epochs, init half and the best-dev snapshot.

- **e72a:** ABCD, 4,000 clauses (the potion row took 770 s and 2.0 GB).
- **e72b:** STAR, 16,000 clauses (the potion row took 7,943 s and 4.4 GB, with 3 jobs sharing the GPU).
- **e72c (no rule, after e72a and e72b):** the no-memory twin on ABCD (`--parts words,xcust,xrecent`, 4,000 clauses).
  Without memory the text carries everything, so it shows what the table itself costs (potion twin: 0.6200).
  - STAR's twin (16,000 clauses, about 2 h) and the LR twins (`--arms lr_bits`, CPU-heavy) run only if the CPU
    allows.
- **Crashes:** one rerun with the same command, logged. A second crash means "not run", and that dataset does not go
  to test.

## Rule: dev gate G-c4, per dataset

The e72 TM's dev accuracy (best epoch, seed 1) must be at least the potion view-B row minus 2.0 points (same seed,
settings, view and splits):
- STAR: 0.6343 (e60_star_B_c16000), so at least **0.6143**;
- ABCD: 0.8325 (e60_abcd_B_c4000), so at least **0.8125**.

**A dataset that passes** may go to a registered test (R21, below).

**A dataset that fails:**
- its rows and costs go into FAILURE_MAP.md and docs/results.md;
- its test split stays unread for this line;
- the agent stays "no neural network at run time" (R19).

**Notes:**
- The 2.0 margin is R19.1's test margin. A dev loss between 1.5 and 2.0 passes the gate but puts the test at risk.
  R19's dev-to-test shifts against bge-large were +0.57 on STAR and -0.27 on ABCD.
- One seed only: STAR's dev curve swings 1 to 2 points between epochs (R19's B curve).

**Reported with no rule:**
- the cost against bge-large dev (STAR 0.6394, ABCD 0.8398);
- n_bits per field, the dev curve, the train time and the GPU peak;
- the dropped-token shares above.

## If a dataset passes: R21 (written and frozen separately, after the gate)

A sketch only; not frozen by this plan.
- **Test-split use:** STAR's 5th and ABCD's 13th, declared.
- **Arms**, following R19:
  - `tm_mem_c4`, seeds 1 to 5;
  - `tm_nomem_c4`, seeds 1 to 5;
  - `lr_mem_c4`, seed 0, with no rule;
  - references copied from R19 (tm_mem_static, tm_nomem_static, lr_mem_static) and from R18/R9b (bge-large).
- **R21.1:** tm_mem_c4 minus tm_mem_static must be at least -2.0 on the point estimate. It is "strong" if the lower
  end of the interval is at least -2.0.
- **R21.2:** tm_mem_c4 minus tm_nomem_c4 must be at least +5.0, with the interval above 0.
- **Statistic:** R19's (scripts/report_abcd.py).
- **abcd_confirm.py** runs through table_run.py. A dry run (`--dry-run`, tiny TM) must reproduce the e72 bits.
- **Cost:** about 10 STAR jobs at 1.5 to 2.2 h each and 10 ABCD jobs at about 15 min, 2 at a time when the CPU allows,
  so about 12 h.

## The verification link (no rule; ABCD; after the gate; its own dated note and pre-check)

- e70's veto clauses read only memory bits. So the same 50 guards apply to an e72 model. They prove the guarded
  policy follows the 50 order rules once its M is recomputed.
- e70's unguarded checks (counterexamples, guard cost) would be rerun on the e72 ABCD model by a new script that
  imports verify_policy's functions, with no edit. Its pre-check: the encoding reproduces the model's dev
  decisions.
- e71's word-edit proofs are specific to potion's tokenizer. A port is a separate step.

## Wording

**If R21 is met,** the table's wording: "no neural network at any stage: C4 text, a C4 word list, counting + SVD,
k-means + thresholds, then a Tsetlin machine".
- It is used only for results on this table. It stays banned for R19, R20, e70 and e71, which use potion.
- The datasets are human-written dialogues; ABCD was delexicalised by its authors, a method not checked here.

**Never write:**
- "first network-free TM text pipeline" (Kadhim et al. 2025);
- "state of the art";
- "parity with published TMs";
- "the guidelines are verified" (the rules come from guidelines.json and kb.json through a hand-written name
  mapping);
- "proved for all possible texts".

## Prior (a guess, not a rule)

**Costs against potion on dev:** ABCD -0.3 to -1.5, STAR -0.5 to -2.5.
- Memory and the 1-2-grams carry most of view B: potion against bge-large cost only -0.51 and -0.73 on dev.
- The table is 1.8 to 3.6 below potion on intent val, and STAR has the most free text.
- A guess at the chance of passing: about 3 in 4 for ABCD and about 2 in 3 for STAR.
