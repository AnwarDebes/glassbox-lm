# REGISTRATION W1-T: a Tsetlin text pipeline with no neural network at any stage, on N7's test splits (second use, declared)

Written 2026-10-19T04:25:52Z (UTC), before any test read for this registration. A separate check of this file comes
before the first run.

## Why

W1-S (the study log 04:2xZ, exploratory, carved val) found that a table built from C4 alone reaches potion's level on three of
N7's five sets: C4 text, a C4 word list, counting + SVD. Through existsT and N7's TM setting it is within 1.0 of potion
on TREC, R52 and R8, 2.25-2.74 below on MR and SST-2. On the intent sets it stays 1.8-3.6 below potion; nothing is
claimed there (W1-S rule 6). This registration tests the suite-2 result once on the test splits.

## Second use of the test splits, declared

- N7 (REGISTRATION_N7.md, 125 registered runs, 2026-10-18 19:50Z to 2026-10-19 00:35Z) read the published test
  splits of TREC-6, MR, SST-2, R52 and R8 once. This is the second use.
- N7's test numbers for the reference arms are known; they are listed below under "What was known".
- The only unknown here is the test result of the C4-word arm.

## Arms (N7's TM setting and statistics, seeds 1-5)

1. **existsT on the C4-word table (new runs).**
   - The table is w1/tables/ppw_S1000_d500_e0.5.npz (sha256 e1771d9eaf4a07f7). It is exactly the file of the
     W1-S val runs, used as it is, with no rebuild and no new tuning.
   - Provenance:
     - Text: C4 en shards 00000-00007 (allenai/c4 @1588ec45, sha256 in data_untrusted/c4_en/SOURCE.txt and
       SOURCE2.txt), deduped against every eval split.
     - Slice S1000: the nested slice, 2,243,398 documents (w1/corpus/c4_ext_slices.json, sha256 3c42c0465f97e089).
     - Words: by glove_feats.glove_tokens. The vocabulary is the 63,091 most frequent words
       (w1/corpus/c4_words_vocab.json, c7d396d7c3b24106).
     - Table: built by w1/ppmi_svd2.py (fe37efe876686391) with 20,000 contexts, window 5 with expected weights,
       subsampling t 1e-4 (seed 0), context smoothing 0.75, no shift, 500 dims, eigenvalue weight 0.5, unit rows.
   - Features:
     - Built by w1/w1_n7_test.py (91906d0248803783), using w1/w1_n7w.py's encoder (b54688135df9f106).
     - The encoder: words by glove_tokens; k-means of the train tokens (K 512, 25 iterations, CUDA generator
       seeded 0); max cosine over the text's tokens; Thermo(8) fitted on train.
     - Check at build: the train and val bits must equal the W1-S val-run cache
       (w1/feats_n7/ppw_S1000_d500_e0.5/{ds}.npz) exactly, or the build stops.
2. **existsT on potion-base-32M:** N7's registered test rows (results_n7_test.jsonl, view existsT_potion).
3. **existsT on GloVe 42B, the published count table:** N7's rows, view existsT_glove.
4. **bow alone, the anchor:** N7's rows, view bow.

Arms 2-4 are not re-run. Their N7 rows come from the same TM setting, the same seeds and the same test splits, so a
re-run would read the test splits again for numbers that already exist.

**TM:** n7_tm.py's run loop in test mode:
- dl/tm.py CoTM (f649043f5801605b), dl/tm_init.py (557ff69706529738), dl/boolean.py (b5f5917f390da8a6);
- 4,000 clauses, T 80, s 9, 100 epochs, batch 128, hardest negative, negation, init half;
- val and test scored after every epoch, no best-state restore.

## Statistics and bars (verdict: w1/verdict_w1t.py, 8a383c6431df5e24, self-tested on synthetic rows)

- **Primary statistic, S2:** test accuracy at the best carved-val epoch (ties: the earliest), a single read.
  S1 (mean test accuracy over epochs 51-100) is printed beside it.
- **Paired by seed:** per set, the mean over seeds 1-5 of the C4-word arm minus the reference arm, on S2.
- **Word minus potion (the bar):**
  - REACHES potion: at most 1.0 below on at least 3 of 5 sets;
  - NEAR potion: at most 2.0 below on at least 3 of 5;
  - else BELOW.
- **Word minus GloVe 42B:** the same words, reported.
- **Word minus bow:** N7's D rule. On a set it "adds" when the mean paired S2 difference is > 0 and at least 4 of the
  5 seed differences are > 0. Met when it adds on at least 3 of 5 sets. Write "adds on N of 5 sets", never
  "significant".
- A set counts only with all 5 seeds in both arms, and a repeated row is ignored. Comparisons allow 1e-9.

## Wording, fixed now

- **For the pipeline:** "no neural network at any stage: C4 text, a C4 word list, counting + SVD, k-means +
  thresholds, then a Tsetlin machine."
- **The comparison is to potion's table in OUR TM setting.** Our TM setting is below the published TM numbers (N7 bar
  A did not hold: our bag-of-words TM is 8-12 points below Yadav et al.'s on TREC and MR). So never write "state of the
  art" or "parity with published TMs".
- **On any test result:** this is the second use of these test splits; say so.

## What was known at the freeze

**N7 test results** (REGISTRATION_N7.md verdict; S2 / S1, mean of seeds 1-5):

| View | TREC | MR | SST-2 | R52 | R8 |
|---|---|---|---|---|---|
| bow | 81.00 / 79.93 | 64.10 / 63.55 | 62.36 / 62.74 | 88.97 / 88.87 | 94.95 / 94.97 |
| existsT_potion | 84.24 / 83.32 | 70.03 / 70.18 | 77.98 / 77.84 | 85.63 / 85.58 | 94.42 / 94.40 |
| existsT_glove (42B) | 84.00 / 83.18 | 70.97 / 70.95 | 74.05 / 73.15 | 86.73 / 86.65 | 94.25 / 94.33 |

- Bar A failed, B and C were 0 of 4, D_potion and D_glove met, E 0 of 5.

**W1-S, N7 carved val** (exploratory, S1-val, seeds 1-3; TREC / MR / SST-2 / R52 / R8):

| Table | TREC | MR | SST-2 | R52 | R8 |
|---|---|---|---|---|---|
| C4-word table at S1000 | 80.71 | 66.52 | 73.83 | 88.92 | 93.24 |
| potion | 80.30 | 68.77 | 76.57 | 87.92 | 94.06 |
| GloVe 42B | 81.75 | 69.28 | 72.82 | 88.91 | 93.11 |
| bow | 74.57 | 63.31 | 58.97 | 91.35 | 94.77 |
| potion-string tables, S10 | 78.24 | 56.83 | 56.11 | 85.98 | 91.28 |
| potion-string tables, S100 | 80.37 | 61.14 | 67.06 | 87.57 | 92.68 |
| potion-string tables, S300 | 79.32 | 64.28 | 70.46 | 87.88 | 92.52 |
| potion-string tables, S1000 | 79.86 | 66.86 | 71.18 | 87.99 | 92.64 |

The val numbers of the word arm's own seeds 1-3 (S1-val and best val) are in w1/results_w1_n7val.jsonl.

**W1-P / W1-S, intent val** (test 2 harness, seeds 1-3; CLINC150 / Banking77 / HWU64):
- potion-string tables: S10 89.84 / 85.91 / 86.31, S100 92.58 / 88.50 / 88.75, S300 92.78 / 88.30 / 88.63,
  S1000 92.22 / 88.76 / 89.47;
- the C4-word table: 92.29 / 87.91 / 89.34;
- references: potion 94.60 / 91.46 / 91.23, GloVe 42B 91.70 / 87.87 / 88.78.

**F1 calibration (CAIR's machine type):**
- tmu 0.8.3's standard multi-class TM with Yadav et al.'s settings on our TREC bag of words (carved val, seed 1):
  S1-val 70.91, against Yadav et al.'s published 88.05 and our CoTM's 74.67.
- MR is still running at the freeze. Its result does not change this registration.

**W1 step 0** (negative): a TM-learned word table loses to counting and gains less from data (the study log 02:0xZ).

## Run

- w1/w1t_chain.sh (c17e745a31981da9): build the five test-mode caches (with the check), then 25 runs (5 sets x 5 seeds), at most 2 jobs
  of this line at once, under the shared GPU lock.
  - Before every build and run it checks these code hashes: w1_n7_test.py, w1_n7w.py, w1_n7.py, n7_tm.py,
    exists_tm.py, glove_feats.py, dl/tm.py, dl/tm_init.py, dl/boolean.py; and the table e1771d9eaf4a07f7.
  - On a mismatch the lane stops.
- Rows go to w1/results_w1t_test.jsonl, the log to w1/runs_w1t.log. Then verdict_w1t.py, with its output saved as
  w1/verdict_w1t.txt.
- Starts only after a separate check of this file.

## Verdict (2026-10-19T04:43:55Z; w1/verdict_w1t.py 8a383c6431df5e24, output w1/verdict_w1t.txt 8f4c2f6de69f36cd; addendum w1/verdict_w1t_extra.py a5a39dd8b0857d47, output w1/verdict_w1t_extra.txt f2bd1996e3aa4bf6)

This is the second use of N7's five test splits, declared.

**Runs and checks**
- 25 test runs of the C4-word arm (seeds 1-5), one row per (set, seed). Every row carries the registered code hashes
  and has train and val bits equal to the W1-S val-run cache. Both lanes ended with exit 0; no lane stopped.
- Reference arms: N7's 75 registered test rows (results_n7_test.jsonl, sha256 77b65abed26cf59a). They were run by
  n7_tm.py 3d715305b119b57b, with seeds 1-5 in every set and view. REGISTRATION_N7.md now holds its own verdict
  section (f849d4975ef5cd33); verdict_n7.txt is db0855c1de8fbaa3.
- Drift check: the word arm's carved-val S1 in the test runs equals the W1-S val runs exactly (difference 0.000) for
  seeds 1-3 on all 5 sets.

**Test results.** S2 / S1, mean of seeds 1-5, TREC / MR / SST-2 / R52 / R8:

| Arm | TREC | MR | SST-2 | R52 | R8 |
|---|---|---|---|---|---|
| C4-word table | 85.72 / 84.72 | 68.01 / 68.05 | 73.98 / 73.28 | 86.92 / 86.56 | 94.92 / 94.76 |
| potion | 84.24 | 70.03 | 77.98 | 85.63 | 94.42 |
| GloVe 42B | 84.00 | 70.97 | 74.05 | 86.73 | 94.25 |
| bow | 81.00 | 64.10 | 62.36 | 88.97 | 94.95 |

(Reference arms: S2 only.)

**The bar: word minus potion** (paired S2)

| | TREC | MR | SST-2 | R52 | R8 |
|---|---|---|---|---|---|
| Difference | +1.48 | -2.03 | -4.00 | +1.29 | +0.50 |
| Seeds above zero | 4 of 5 | 0 of 5 | 0 of 5 | 5 of 5 | 5 of 5 |

- Within 1.0 on 3 of 5 sets: **REACHES POTION**. The word table is above potion on TREC, R52 and R8.

**Word minus GloVe 42B:** +1.72 / -2.96 / -0.06 / +0.19 / +0.66. Within 1.0 on 4 of 5: **REACHES GLOVE 42B**.

**Word minus bow (N7's D rule):** +4.72 / +3.91 / +11.62 / -2.05 / -0.03. It adds on 3 of 5 sets (TREC, MR, SST-2):
**MET**.

**Plainly**
- On N7's five test splits, a Tsetlin text pipeline with no neural network at any stage (C4 text, a C4 word list,
  counting + SVD, k-means + thresholds, then a Tsetlin machine) reaches the level of the transformer-distilled potion
  table in our TM setting: above it on TREC (+1.5), R52 (+1.3) and R8 (+0.5), below it on the sentiment sets MR (-2.0)
  and SST-2 (-4.0).
- The comparison is to potion's table in OUR TM setting, which is below the published TM numbers (N7 bar A failed).
  So it is not "state of the art" and not "parity with published TMs".
- The intent sets did not reach potion (W1-S: 1.8-3.6 below on val). Nothing is claimed there.
- This was the second use of these test splits. No reruns.
