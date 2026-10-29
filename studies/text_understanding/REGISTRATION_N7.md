# REGISTRATION N7: existsT against published Tsetlin machine text results (TEST; written 2026-10-18T19:47:00Z; corrected at 2026-10-18T19:47:21Z and 2026-10-18T19:47:47Z, before any check: R8 length, the
bars heading, the bag-of-words size and GloVe table in "Arms", and the time of the commitment; after a separate
check, at 2026-10-18T19:50:32Z, one wording fix: "clean" became "with a stated averaging protocol" in bar C, and the verdict hash)

Why: CAIR's published Tsetlin machine (TM) text results use a bag of words, or a bag of words widened with GloVe
neighbour words (Yadav et al. 2021). Our existsT encoder (a word-table lookup plus an OR; the study log, tests 2-3) may lift
TM accuracy on the TM community's own benchmarks. The GloVe arm (val, the study log 19:10:27Z) showed that the potion table
matters (+2.7 to +4.1 points over GloVe on the intent sets), so both tables are run; the GloVe rows are the
same-table comparison with Yadav et al.

This is the FIRST use of these five test splits in this project. 5 seeds. Every dataset below is reported whatever it
shows (committed in the study log at 19:16:19Z, after the five 10-epoch timing rows and before the 100-epoch smoke rows).

## Data (pinned; sha256 of every file in data_untrusted/*/SOURCE.txt)

| Set | Source (revision) | Published split used | Classes | Median words |
|---|---|---|---|---|
| TREC-6 | SetFit/TREC-QC (0a34640b), label_coarse | train 5,452 / test 500 | 6 | 10 |
| MR | dxgp/MR (c51035cd), the PTE split (Tang et al. 2015) | train = train.csv + val.csv = 7,108 / test 3,554 | 2 | 20 |
| SST-2 | SetFit/sst2 (00ea8ccb) | train 6,920 / test 1,821 (dev not used) | 2 | 19 |
| R52 | dxgp/R52 (cdf48fcb), the TextGCN split | train = train.csv + val.csv = 6,532 / test 2,568 | 52 | 73 |
| R8 | dxgp/R8 (97c9e126), the TextGCN split | train = train.csv + val.csv = 5,485 / test 2,189 | 8 | 66 |

Our split: the published train texts (file order) permuted by numpy default_rng(2026); the last 10% is our val split
and the first 90% our train split. Concepts, thermometer thresholds and the bag-of-words vocabulary are fitted on our
train split only. DEVIATION from Yadav et al.: we train on 90% of the published train set (they used all of it); this
is conservative for us.

## Arms (125 runs = 5 sets x 5 views x 5 seeds)

Views (n7_tm.py): bow; existsT_potion; existsT_glove; bow+existsT_potion; bow+existsT_glove.
Seeds 1-5. Sets TREC-6, MR, SST-2, R52, R8.
- bow: binary presence of the N words with the highest train document frequency (ties alphabetical), N = Yadav et
  al.'s stated vocabulary size (TREC 6,000, R52 6,000, MR 5,000, R8 5,000; SST-2 5,000), words by
  glove_feats.glove_tokens. This is the calibration arm against Yadav's bag-of-words TM.
- existsT_*: the test 2 existsT encoder (512 spherical k-means concepts of the distinct train tokens, per text and
  concept the max cosine over its tokens, Thermo(8) on the train texts) with potion-base-32M (1e5a03f8, its own
  tokenizer) or GloVe 42B 300d (the table Yadav et al. used; hashes under "TM setting"; word tokens).

## TM setting

The decision-layer CoTM (dl/tm.py, read only) with 4,000 shared clauses, batch 128, hardest negative, negation, init
half, 100 epochs, and Yadav et al.'s reported T 80 and s 9 ("Yadav's reported T and s, on our CoTM with 4,000
clauses"). Changes from test 2's setting: epochs 100 instead of 40 (the statistic needs 100), T 80 instead of 4000,
s 9 instead of 8. Their machine differs: a standard multi-class TM with 2,000 (TREC) / 3,000 (MR) / 1,500 (R52) /
2,500 (R8) clauses, per class as far as stated, each voting +-1; ours is a coalesced machine whose 4,000 shared
clauses vote with integer weights per class. So T does not carry over exactly. Bag-of-words size: Yadav et al.'s
stated vocabulary sizes (TREC 6,000, R52 6,000, MR 5,000, R8 5,000; SST-2, not in their paper, 5,000).
Why the setting changed from test 2's (decided before any test read; the study log 19:30Z-19:38Z): with T 4000 / s 8 our
bag-of-words anchor was about 11 points below Yadav's bag-of-words TM on TREC and MR on our val split, so bars B and C
would have measured TM settings rather than features. The paper's own T and s were chosen "whatever the val check
shows" before any T 80 row existed. The T 80 / s 9 val check did not fix the anchor
(rows below). No further search, whatever bar A shows. These were the last setting changes.
GloVe table of N7: GloVe 42B 300d (Common Crawl, uncased, 1,917,495 words; the table Yadav et al. used), from the
Stanford NLP group's mirror stanfordnlp/glove on Hugging Face at revision 1db2080b2d94def6e5b0386a523102f9d8849e9d
(the Stanford server ran at 0.3 MB/s; that download was stopped and its partial file removed): zip sha256
357baac33090f645e71e253b3295ee1b767c98a0336e9a1d99c77e9e33b43c4a (1,877,802,108 bytes; Stanford's zip is 1,877,800,501
bytes, a packaging difference), txt member sha256 46cc21a4c7d1d340219d9f871372c3dd8b031b75e97ab13784a2deb51f7cafd5,
parsed by glove42_prep.py (7e98b1b7ac9472db; 0 lines skipped) into glove.42B.300d.vectors.npy (027b311967f4315b) and
glove.42B.300d.words.json (823e7a96e30d163e). The intent-set GloVe arm (the study log 19:10:27Z) used GloVe 6B and stays a
separate record.

## Statistics

- S1 (Yadav et al. 2021): "Reported accuracy of TM is the mean of last 50 epochs of 5 independent experiments with
  their standard deviation." (captions of Tables 3 and 4). Ours: per run the mean test accuracy over epochs 51-100;
  then mean and sd over the 5 seeds. This scores the test split after every epoch (100 reads per run). It is accepted
  only as a FIXED statistic for the like-for-like comparison: no selection, no early stopping, nothing is decided from
  the test split.
- S2 (single read): test accuracy at the epoch with the best accuracy on our val split (ties: the earliest epoch);
  mean and sd over the 5 seeds.

## Published numbers compared with (all accuracy %)

Tsetlin machines, stated averaging protocol, no neural network:
- Yadav, Jiao, Granmo, Goodwin, BlackboxNLP 2021 (ACL Anthology 2021.blackboxnlp-1.19; pages 265-274 per a search
  snippet, unverified). Table 3 (R8 / R52 / MR): TM 96.16 / 84.62 / 75.14; TM with k 97.50 / 88.59 / 77.51; TM with phi
  96.39 / 89.14 / 76.55. Table 4 (TREC): TM 88.05 +- 1.52; TM with k 89.82 +- 1.18; TM with phi 90.04 +- 0.94. Their
  setting: T 80, s 9, binary bag of words of 5k-6k words; clauses and epochs not stated. Caveats (N7 log): k and phi
  were chosen on the test set; R8 97.50 does not appear in their own Table 1 (best 97.08); the TREC split is not
  stated; R8 / R52 / MR are the TextGCN / PTE splits (their baseline rows equal TextGCN's Table 2).
- Contracting TM (Bhattarai et al. 2023, ISTM, arXiv 2310.11481, Table I, mean of the last 25 epochs): TREC 91.61,
  R8 95.84, R52 89.53.
- Sparse TM (Østby, Brambo, Glimsdal 2024, arXiv 2405.02375v2, Table I, mean of the last 25 epochs; split not stated):
  MR 77.18, SST-2 78.93.
Tsetlin machines with weaker protocols (reported next to ours, no bar): Drop Clause (Sharma et al. 2023): R8 98.94,
R52 92.75, MR 78.67, TREC 90.5 (protocol not stated); Abeyrathna et al. 2023 (IJCAI): TREC 93.2 (best epoch on test);
HVTM (Halenka et al. 2024): TREC 95.6 (best on test, split not stated).
Not Tsetlin machines (references; protocols differ): TextGCN (Yao et al. 2019, Table 2, same R8 / R52 / MR splits,
mean of 10 runs): CNN-non-static 95.71 / 87.59 / 77.75, Bi-LSTM 96.31 / 90.54 / 77.68, Text GCN 97.07 / 93.56 / 76.74.
Kim 2014 (Table 2): CNN-non-static TREC 93.6, SST-2 87.2. SWEM (Shen et al. 2018, Table 8): SWEM-max TREC 89.0, SST-2
83.6. Fine-tuned BERT-base: TREC 97.20 (Sun et al. 2019, 5,452 / 500); R8 / R52 / MR 97.8 / 96.4 / 85.7 (BertGCN,
Lin et al. 2021, same splits); SST-2 93.5 (Devlin et al. 2019, GLUE test, trained on 67k phrases: not like-for-like).
Must be cited: "Clusters are All You Need" (arXiv 2606.19815, June 2026): k-means on BERT document embeddings fed to a
TM, the closest published cousin of our concept bits.

## Bars (A, B, C on S1; D, E on S2; means of 5 seeds; verdict_n7.py)

- A calibration (two-sided; the direction is printed per set): our bow TM within 2.0 of Yadav's bag-of-words TM
  (TREC 88.05, MR 75.14, R52 84.62, R8 96.16) on at least 3 of the 4. On our val split R52's bag of words was already
  about 7 points ABOVE Yadav's number, so A may fail on the high side there. If A does not hold, B and C are reported
  with that caveat (they then mix features and TM settings) and the published numbers are context only.
- B same table: bow+existsT_glove above Yadav's best TM+GloVe row (TREC 90.04, MR 77.51, R52 89.14, R8 97.50) on at
  least 3 of the 4.
- C potion: bow+existsT_potion above the same rows on at least 3 of the 4. Strong: also above the best published TM
  number with a stated averaging protocol of each set (TREC 91.61, MR 77.51, R52 89.53, R8 97.50) on at least 3 of
  the 4. MR 77.51 and R8 97.50 are Yadav's k rows, and k was chosen on the test set: optimistic published numbers.
- SST-2: reported against the Sparse TM's 78.93 (no bar for the published comparison: its split is not stated).
- D (internal, stands even if A fails; same setting, splits and seeds): on S2 (single read at the best val epoch),
  with S1 printed beside it. On a set, bow+existsT_X - bow "adds" when the mean over the 5 seeds is > 0 and at least
  4 of the 5 paired seed differences are > 0. D_potion and D_glove are met when it adds on at least 3 of the 5 sets
  (SST-2 included). A weak per-set test: write "adds on N of 5 sets", never "significant".
- E (table, internal): bow+existsT_potion - bow+existsT_glove, the same paired rule on S2, met on at least 3 of 5.
- Reported, no bar: S2 of every view; existsT alone (both tables; pure lookup + OR); bit counts; best val epochs.

## Wording

"Against published TM results", never "state of the art"; the best published non-TM numbers are printed next to ours.
GloVe rows: "transformer-free", never "no neural network". Potion rows are not transformer-free (potion is distilled
from bge-base). A bar passing on a set where Yadav chose k or phi on the test set is a comparison with an optimistic
published number, and is said so.

## What was known at the freeze (train and carved val only; no test file was opened before the registered runs)

- GloVe arm on the intent sets (val, 3 seeds, GloVe 6B): GloVe-existsT 91.50 / 87.34 / 88.51 against potion-existsT
  94.60 / 91.46 / 91.23 (CLINC150 / Banking77 / HWU64): potion matters by 2.7-4.1 points.
- N7 literature (agent log novelty/N7_tm_text_benchmarks.md; quotes and tables cited above).
- Smoke, T 4000 / s 8 (test 2's setting), seed 1, 100 epochs, our carved val split (mean of the last 50 / best epoch):
  TREC bow 76.80 / 81.28, bow+existsT_potion 82.45 / 84.40; MR bow 64.25 / 67.89 (59.86 at epoch 100),
  bow+existsT_potion 70.84 / 74.51; SST-2 bow 65.39 / 68.79, bow+existsT_potion 77.86 / 80.78; R52 bow 92.09 / 92.80,
  bow+existsT_potion 92.95 / 94.03; R8 bow 95.94 / 97.08, bow+existsT_potion 95.95 / 96.72. Earlier 10-epoch timing
  rows (bow+existsT_potion, best val): TREC 77.61, MR 74.51 (epoch 3), R52 82.70, R8 92.70, SST-2 78.47. These used
  GloVe 6B and a 5,000-word bag of words for every set.
- Val check, T 80 / s 9, seed 1, 100 epochs (mean of the last 50 / best): TREC bow 74.67 / 77.61, bow+existsT_potion
  80.68 / 82.75; MR bow 63.69 / 66.48, bow+existsT_potion 71.70 / 73.10; R52 bow 91.51 / 92.50, bow+existsT_potion
  90.89 / 92.04 (same GloVe 6B / 5,000-word caches).
- Saturation of the max over tokens (n7_saturation.py, train split; median over concepts of the interquartile range
  of the max cosine; GloVe 6B): potion CLINC150 0.065, TREC 0.067, MR 0.082, SST-2 0.086, R52 0.117, R8 0.115; GloVe
  0.070 / 0.059 / 0.094 / 0.098 / 0.107 / 0.102. The spread grows with length: no saturation at 65-80 tokens. Share of
  concepts with an IQR below 0.02: potion 0-0.03; GloVe TREC 0.19 (short questions leave many concepts unmatched).
  The bits-on share of existsT is 0.51-0.53 on every set by construction (quantile thermometer), so it shows nothing.
- Frozen-code smoke (n7_tm.py 3d715305b119b57b, GloVe 42B, Yadav's vocabulary sizes; explore caches only): GloVe 42B
  out-of-vocabulary share of the train tokens TREC 0.18%, MR 0.28%, SST-2 0.12%, R52 0.49%, R8 0.52%; no text
  without a token; cache build 10-13 s per set; one 10-epoch timing run (TREC, bow+existsT_glove, seed 1): best val
  76.70 at epoch 8, 10.5 s training. Nothing changed after it.
- Committed in the study log (19:16:19Z), after the five 10-epoch timing rows and before the 100-epoch smoke rows: all five
  sets are registered and reported whatever the smoke shows. No test file was opened before this registration (emb_n7 holds explore caches only; results_n7_test.jsonl
  does not exist; a separate check of results_n7.jsonl and emb_n7 at 19:33Z confirmed it).

## Run

- n7_test_chain.sh (c7acd1a54e1077b5), lanes A and B, at most 2 jobs of this line at once, shared GPU lock: lane A checks
  the data files (data_untrusted/*/SOURCE.txt, glove42B/PARSED.sha256) once and builds the five test-mode caches,
  then both lanes take the 125 runs (mkdir claims, each run once), checking before every build and run: n7_tm.py
  3d715305b119b57b, glove_feats.py 7b5e8760e1f5e107, exists_tm.py 7da0aee9f4eef796, dl/tm.py f649043f5801605b,
  dl/tm_init.py 557ff69706529738, dl/boolean.py b5f5917f390da8a6. On a mismatch the lane stops.
- Rows: results_n7_test.jsonl (per run: both curves, S1, S2, epoch-100 accuracy, bits, on-shares, hashes, cache meta).
  Log runs_n7_test.log.
- Verdict: verdict_n7.py (a6efd305983c668f; ef5477f9de34eb25 until the wording fix of 19:50:32Z, which changed printed
  labels and comments only), written and self-tested on synthetic rows before any test read; bars D and E on S2 with
  S1 beside it.
- Starts only after a separate check of this file.

## Verdict (2026-10-19T00:36:59Z; verdict_n7.py a6efd305983c668f, output saved in verdict_n7.txt, sha256 db0855c1de8fbaa3)

125 test rows, one per (set, view, seed), every row with the registered code hashes; both lanes ended with exit 0
on every run and no lane stopped (runs_n7_test.log, 19:50:45Z to 00:35:32Z). Test splits read once, by these runs.

- A calibration DOES NOT HOLD (within 2.0 on 1 of 4). Our bag-of-words TM against Yadav et al.'s: TREC 79.93 vs 88.05
  (-8.12), MR 63.55 vs 75.14 (-11.59), R52 88.87 vs 84.62 (+4.25, above), R8 94.97 vs 96.16 (-1.19). So B and C mix
  features with TM settings, and the published numbers are context only.
- B (GloVe 42B, Yadav et al.'s table) NOT MET, 0 of 4: bow+existsT_glove 83.36 / 72.24 / 88.87 / 95.49 against
  Yadav et al.'s best TM+GloVe 90.04 / 77.51 / 89.14 / 97.50 (-6.68 / -5.27 / -0.27 / -2.01; TREC / MR / R52 / R8, S1).
- C (potion) NOT MET, 0 of 4, strong NOT MET: bow+existsT_potion 83.77 / 71.41 / 88.75 / 95.22 (-6.27 / -6.10 / -0.39 /
  -2.28 against the same numbers; the MR and R8 references are Yadav et al.'s k rows, k chosen on the test set, so
  they are optimistic published numbers).
- D_potion MET: adds on 3 of 5 sets (S2, paired over 5 seeds: TREC +2.64, MR +7.39, SST-2 +14.18, 5 of 5 seeds each;
  R52 -0.19 and R8 +0.26 do not add). S1 agrees (+3.84 / +7.86 / +12.97).
- D_glove MET: adds on 4 of 5 sets (S2: TREC +3.68, MR +8.33, R8 +0.54, SST-2 +11.59, 5 of 5 seeds each; R52 -0.02 does
  not add).
- E NOT MET, adds on 0 of 5 sets: potion's table does not add over GloVe 42B (S2 -1.04 / -0.94 / -0.17 / -0.28 / +2.59
  on TREC / MR / R52 / R8 / SST-2; on SST-2 only 2 of 5 seeds are above).
- SST-2 (no bar): S1 bow+existsT_potion 75.71, bow+existsT_glove 73.54, existsT_potion alone 77.84, against the Sparse
  TM's 78.93 (its split is not stated).
- Plainly: against published TM results N7 is NEGATIVE. Our TM setting does not reproduce Yadav et al.'s bag-of-words
  TM (8-12 points below on TREC and MR, 4 above on R52), and no view is above a published TM+GloVe number on any of
  the 4 sets. Inside our own setting the meaning bits add to the words on 3 (potion) and 4 (GloVe 42B) of 5 sets, by
  up to 14 points (SST-2), and not on R52. GloVe 42B, a transformer-free table (counting at 42B web tokens), is as good
  as potion on these five sets. Wording: "against published TM results", "adds on N of 5 sets" (never
  "significant"), "transformer-free" for GloVe. No reruns on these test splits.

## Follow-ups (added 2026-10-19T04:46:52Z; exploratory unless marked)

- N7-F2: GloVe 42B on the intent sets (val). NOT CLOSED: 2.45-3.59 below potion, only 6-13% of the GloVe-6B gap
  closed (the study log 00:5xZ). Potion's edge on intents is not corpus size; on these five sets GloVe 42B equals potion.
- N7-F1: tmu's standard TM with Yadav et al.'s settings on our TREC bag of words (carved val). S1-val 70.91, against
  88.05 published and our CoTM's 74.67. MR is running; the words of the rule follow when it ends.
- W1-T (REGISTERED, REGISTRATION_W1T.md; the second use of these test splits, declared):
  - existsT on a C4-word counting table, with no neural network at any stage, REACHES potion's table in our TM setting
    (within 1.0 on 3 of 5). Paired S2: +1.48 / -2.03 / -4.00 / +1.29 / +0.50 (TREC / MR / SST-2 / R52 / R8).
  - It is still below the published TM numbers, as is all of our TM setting (bar A).
