# Plan M2: the text-understanding stack on a fresh intent test set, MTOP English (registered, frozen 2026-10-25T15:03:02Z)

Status: **registered, frozen 2026-10-25T15:03:02Z**, after two rounds of pre-freeze review (no blocker). This published
copy leaves out the pre-freeze decisions, the review answers and the draft history, and is edited in wording only; every
rule, bar and number is as registered. The registered text's sha256 is in `docs/hash-table.md`.
- At the freeze nothing of M2 had run on the GPU. No MTOP label had been used and no model had been scored on any MTOP
  split; the stand-in loaders split en/train and en/eval rows into their columns in memory and use only the ID and the
  utterance. MTOP's use count was 0 for every split.
- Order: the GPU dry run, the freeze (it writes the final pins block, the freeze line and tu/m2/pins.sh), then the
  registered chain.
- Every command in this plan runs from studies/text_understanding.

## Why

- M1 (registered, frozen 2026-10-22T10:51:33Z, verified MATCH) confirmed H1 and H2 on CLINC150, Banking77 and HWU64.
  But those test splits had been read 13 to 19 times across the project (tu/m1/census_test_reads_2026-10-21.md). Every
  M1 choice was made on val, but the pipeline grew up on those splits, so M1 is not fresh-test evidence.
- MTOP English was accepted on 2026-10-23 as a fresh intent test set (official source, CC BY-SA 4.0, use count 0).
- M2 asks M1's H1 and H2 again on a set this project has never scored.

## What was known before writing this (no MTOP number of any M2 arm exists)

- **M1, test (tu/m1/REPORT_M1.md; TM = mean of seeds 7-9 at s 8; LR = one fit, C on val).** Splits reused (above).

  | Arm | CLINC150 TM / LR | Banking77 TM / LR | HWU64 TM / LR |
  |---|---|---|---|
  | C4 | 91.99 / 91.78 | 89.17 / 88.99 | 86.56 / 87.36 |
  | a3B | 93.59 / 93.78 | 90.89 / 90.68 | 88.72 / 89.59 |
  | potion | 93.53 / 93.78 | 91.36 / 90.75 | 88.14 / 89.03 |

  - H1 a3B - C4: TM +1.60 [+1.05, +2.13] / +1.72 [+1.03, +2.40] / +2.17 [+0.93, +3.44]; LR +2.00 [+1.33, +2.64] /
    +1.69 [+0.75, +2.60] / +2.23 [+0.65, +3.81]. MET 3/3, STRONG.
  - H2 a3B - potion: TM +0.06 [-0.40, +0.50] / -0.48 [-1.00, +0.05] / +0.59 [-0.46, +1.64]; LR +0.00 [-0.56, +0.56] /
    -0.06 [-0.81, +0.68] / +0.56 [-0.84, +1.95]. MET 3/3, not strong. Banking77's TM seed 8 was -0.94.
  - Paired seed SEs of the TM differences (tu/m1/verdict_m1.json): H1 0.12 / 0.25 / 0.57; H2 0.05 / 0.24 / 0.24.
  - Pooled seed SDs over M1's 7 configurations per set (14 df; the power script's output, kept outside the repository): VAL (best-epoch) 0.187 /
    0.356 / 0.328; TEST 0.179 / 0.282 / 0.595 points.
- **M1, val (seeds 7-9, the same runs):** TM C4 / a3B / potion 92.20 / 94.57 / 94.45 (CLINC150), 87.77 / 91.59 / 91.66
  (Banking77), 89.06 / 90.18 / 90.80 (HWU64); LR 92.73 / 95.03 / 94.73, 87.34 / 91.53 / 91.03, 88.85 / 90.33 / 91.17.
- **T2 (2026-10-17, results_confirm.jsonl):** potion's existsT configuration on the old test splits, seeds 1-3: 93.56 /
  91.09 / 88.39. Its val-to-test gaps: CLINC150 -1.04, Banking77 -0.37, HWU64 -2.85 (M1 addendum, item 2).
- **MTOP itself (fresh/NOTE_2026-10-23_mtop.md):** 11 domains, 117 intents over 6 languages; English train 15,667,
  eval 2,235, test 4,386 utterances. The number of English train intents and of unseen eval or test intents needs the
  label column, so it is not known.
- **MTOP overlap with our texts** (fresh/mtop_overlap.json; test texts of 4,386 matched by exact / order-free /
  one-edit keys): our intent train and val (CLINC150, Banking77, HWU64, MASSIVE en; 51,780 texts) 42 / 46 / 419; their
  test texts 15 / 15 / 228; 3c's paraphrases (599,045) 55 / 76 / 578; other evaluation texts 9 / 9 / 169; 3a's C4 fit
  sentences 2 / 2 / 108; MTOP's own train and eval 115 / 125 / 1,135 (26% of its test texts within one word).
  - The frozen secondary list (fresh/mtop_test_matched_v1.json): 676 left out, 3,710 kept (by pool: intent train/val
    427, paraphrases 589, C4 sentences 108; the pools overlap); 55 paraphrase-exact items listed apart.
  - Which M2 arm saw which pool: c4_fit (108) = 3a's C4 fit sentences, the label-free input of a3B's refinement (the
    one pool an M2 arm's training touched); ours_train_val (427) and paraphrases (589): no M2 arm trained on them.
  - MTOP is close in domain to HWU64 and MASSIVE (assistant commands: alarms, weather, music, calls, reminders).
- **W1's C4 filter, the precedent for the unscanned C4 stream** (w1/corpus/c4_slices.json, filter.per_eval_split):
  on its S100 scan of 4,911,009 sentences it dropped, by exact key, 1,266 sentences equal to a CLINC150 test text, 82
  to an HWU64 test text and 1,244 to a MASSIVE en test text (counts of C4 sentences, not of distinct test texts). MTOP's
  commands are of the same kind, so MTOP test texts very likely occur in the 1B-token S1000 stream that the C4 table
  (and so a3B) was counted on; W1's filter predates MTOP.
- **Published MTOP numbers of the teacher family** (novelty/n3d_2026-10-21/report.md, MTEB results files read
  2026-10-21): MTEB MTOPIntentClassification (en) bge-base-en-v1.5 67.14, potion-base-8M 57.00; potion-base-32M has no
  MTOPIntent file. MTEB's protocol (logistic regression on embeddings with few training examples per label; the
  8-per-label setting not re-verified for mteb 2.x) differs from M2's, so these numbers decide nothing here. No number
  of a3B or C4 on MTOP exists.
- **potion's own card** (local snapshot 1e5a03f8, README.md): MTEB averages only (classification average 71.70); the
  per-task numbers are not in it. MTEB's English classification tasks include MTOP intent and domain, so bge-base's and
  potion's developers may have looked at MTOP's test split while choosing models (UNVERIFIED; MTEB's MTOP split being
  equal to the official en/test is also UNVERIFIED). Whether MTOP texts entered bge-base's or potion's training cannot
  be checked. Teacher contamination would inflate H1 (a3B has the teacher, C4 does not); it affects H2 less, since a3B
  and potion share the teacher.
- **The software is M1's.** The imported code, tables, code-check inputs and potion files equal M1's pins
  (tu/m1/pins.sh); the decision_layer venv holds 73 packages, none newer than 2026-10-15T21:44 (find -newermt
  2026-10-21 found no dist-info), so the library versions are the ones M1 ran with (torch 2.6.0+cu124). The
  decision_layer src/dl files carry file times of 2026-10-23 11:09 with contents equal to M1's pins (cause not checked).
- **3d** is NOT READY (above). **Power** (below) is an estimate from M1's published numbers.
- **Code and tests:** written and run on the CPU on 2026-10-25 (sections "Code" and "Tests"); no GPU step of M2 has run.

## Data: MTOP English

- The official zip (fresh/NOTE_2026-10-23_mtop.md): data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip, sha256 in the
  pins; read in memory with its sha256 checked first, never extracted. Licence CC BY-SA 4.0 (its LICENSE.txt).
- Splits: train = en/train.txt (15,667), val = en/eval.txt (2,235), test = en/test.txt (4,386). Columns used: 1 (ID),
  2 (the top-level intent), 4 (the utterance).
- Classes: the sorted distinct top-level intents of en/train. Fit on en/train; the TM's epoch and LR's C on en/eval;
  en/test is read once, by the registered run.
- en/test.txt's structure was already checked without its labels: fresh/mtop_overlap.py and fresh/mtop_freeze_matched.py
  asserted 8 columns on every row and 4,386 rows on 2026-10-23, with the same row parsing as m2_data._rows. No new
  structure check is added, since it would be one more read of the file.
- Before the freeze, only the stand-in loaders are used (m2_data.load_standin, m2_standin.mtop_unseen and
  mtop_small_unseen: labels = sha256(text) mod 20, every 40th en/eval label set to -1 in the unseen variants, the intent
  column never used, en/test.txt never opened, test := val; the audit wrapper tu/m2_audit_run.py raises if
  en/test.txt is opened). Like fresh/mtop_overlap.py, they split each en/train and en/eval row into its 8 columns in
  memory and use only columns 1 and 4 (tests/test_m2_data.py: another intent column gives the same stand-in labels).

## Rules this plan applies (PLAN_text_understanding.md section 0)

- **R1.** The C4 counting table stays the reference with no neural network at any stage, reported beside the teacher
  table.
- **R3.** s 8 for every arm, as in M1. No re-tune on MTOP: the reader's settings came from the old sets' val splits.
- **R5.** No new bit block, so no new guard proofs.
- **R6.** Provenance on every line (section "Provenance and wording").
- **R7.** This registration, with the use count below.

## Arms (one set: mtop_en)

| Arm | Table | Provenance | Probes |
|---|---|---|---|
| C4 | ppw_S1000_d500_e0.5 (w1/tables) | counting table, no neural network at any stage | TM, LR |
| a3B | a3B_d768 (tu/a3/tables) | the C4 table refined toward bge-base-en-v1.5@a5beb1e3; network-free at run time, the teacher at training time | TM, LR |
| potion | potion-base-32M at revision 1e5a03f8 | the published static table, distilled from bge-base-en-v1.5 by its authors (Tokenlearn on C4; the teacher revision they used is not known) | TM, LR |

- **Reader (M1's).** CoTM, 4,000 clauses, T 4,000, s 8, 40 epochs, batch 128, hardest negative sampling, negation
  on, eval every 2 epochs. dl.tm keeps the best en/eval state (the first strictly highest) and restores it; en/test is
  predicted once with that state. Seeds 7, 8 and 9.
- **LR (M1's).** dl.linear.fit_logreg on the same existsT bits; C in {1, 10, 100}, the first strictly highest en/eval
  accuracy; en/test predicted once with that fit; every C's en/eval accuracy kept.
- **Features (M1's).** One CUDA spherical k-means per table (K 512, 25 iterations, seed 0) on the distinct en/train
  tokens; thermometer levels from the members' cosines; the existsT bits (Thermo 8 on the per-concept maximum cosine,
  fit on en/train). train, val and test are pooled in the same run; the centroids are saved. potion uses its own
  tokenizer, read offline from the snapshot.
- **View and selection caveats.** existsT, K 512, Thermo 8, the reader's size and s 8 were chosen on the old sets' val
  splits, never on MTOP. On MTOP only the epoch and C are chosen, on en/eval; the en/eval numbers are best-epoch
  numbers and so optimistic, but en/test is read once at the chosen epoch.

## Hypotheses (fixed before any MTOP label is read; no rule changed in a later revision)

**Met** means the bar holds on the point estimates with BOTH probes. TM = the mean over seeds 7-9 of the test accuracy;
LR = its one fit. Points throughout.
- **H1** (M1's H1 on a fresh set): a3B - C4 >= +0.5 with TM and with LR.
- **H2** (M1's H2, one-sided): a3B - potion >= -0.5 with TM and with LR. There is no upper bound.

**Strong** means the lower end of the registered 95% interval clears the same bar with both probes (M1's definition).
- The registered interval: a paired bootstrap over the 4,386 test items, 10,000 resamples, numpy
  default_rng([20261025, 0]), percentiles 2.5 and 97.5. One index draw per resample is shared by every comparison and
  by seeds 7-9; the TM statistic is the seed-averaged paired difference on the drawn items (M1's method, M2's seed).
- **Beside it, with no rule: a seed-inclusive two-level interval** for every comparison with a TM side.
  default_rng([20261025, 1]): per resample one item draw shared by every comparison, then each TM configuration's
  three seeds drawn with replacement, independently per configuration and shared by the comparisons that use it. LR
  (one fit) has no seed level and is printed "no seed level". With three seeds this level is coarse (a bootstrap over
  three values keeps about two thirds of their variance; only 10 distinct seed multisets), so it is never a rule.
  Also given: the per-seed differences, the paired seed SE and each configuration's seed SD.
- **The fixed qualifier:** STRONG with a TM probe whose seed-inclusive lower end misses the bar reads "MET (STRONG on
  the item-level interval only)" (decision 6).
- Every interval is reported, met or not.

**The claim sentences, fixed now** (m2_verdict.py writes them from the numbers; the review's design MAJOR 1 and M1's
addendum item 4):
- H1 met: "a3B - C4 is at least +0.5 points on MTOP English with the TM (+x [l, u]) and with LR (+y [l, u]) (point
  estimates)"; only when STRONG: "; the lower ends of the item-level 95% intervals clear +0.5" (plus the qualifier);
  when MET but not STRONG, the fixed clause "; not STRONG: a point-estimate result, weak evidence (tu/PLAN_M2.md Power:
  at no true gain H1 is MET in 0.4-10.5% of runs)".
- H2 met: "a3B is no more than 0.5 points below potion-base-32M on MTOP English with the TM and with LR (point
  estimates; one-sided; TM +x [l, u], LR +y [l, u])"; only when STRONG: the same interval clause at -0.5; when MET but
  not STRONG, the fixed clause ending "at a true -1.0 H2 is MET in 0.1-7.8% of runs)".
- Not met, per probe: "H1 NOT MET (the registered rule ... was not met): with the TM, a3B - C4 is +x [l, u]: holds;
  with LR, a3B - C4 is +y [l, u]: below the bar. A true difference inside the interval is not ruled out."
- Every claim sentence, met or not, ends with its arms' provenance (R6). H1: "Provenance: a3B is network-free at run
  time, with bge-base-en-v1.5@a5beb1e3 as teacher at training time; C4 uses no neural network at any stage." H2:
  "Provenance: a3B is network-free at run time, with bge-base-en-v1.5@a5beb1e3 as teacher at training time;
  potion-base-32M@1e5a03f8 is distilled from bge-base-en-v1.5 by its authors, so the two share the teacher family."
- Never: "within 0.5 points" (H2 has no upper bound), "beats", "statistically non-inferior", "failed replication" as a
  bare phrase, or "better than" without a superiority rule.
- Under the per-probe table the text prints each rule part with its point, both intervals, its per-seed differences and
  seed SE (TM) or "no seed level" (LR), and its en/eval difference, as M1's verdict text did.

**Headline, fixed now.** H1 (a3B - C4) carries the new information; H2 is the second claim. One dataset, so every
claim reads "on MTOP English". No combined line; no claim rests on any one rule being met; every rule part's outcome
is shown in one per-probe table in verdict_m2.txt (H1 TM, H1 LR, H2 TM, H2 LR: point, bar, holds, item-level interval
and whether its lower end clears, seed-inclusive interval or "no seed level", the en/eval difference). No line pools
MTOP with M1's three reused splits (no "4 of 4 sets"); M1's numbers stand beside M2's, labelled "reused splits".

**Multiplicity.** No multiplicity correction is applied; each rule stands alone. The claims are H1 (TM and LR, both
needed) and H2 (TM and LR, both needed). Requiring both probes is an intersection, which does not raise the chance of a
false "met".

**Descriptive (no rule).**
- TM minus LR per arm (C4, a3B, potion): the seed-averaged TM correctness minus LR's on the same items, with both
  intervals and the per-seed values.
- Each comparison's en/eval difference (best-epoch val accuracies of the same rows, optimistic) beside its test
  difference; each configuration's en/eval mean, best epochs and the picked C.
- **Secondary:** every comparison and arm without the 676 items of fresh/mtop_test_matched_v1.json (n 3,710), item-only
  interval default_rng([20261025, 2]). It is not a contamination control for most of its pools: only c4_fit (108)
  touched an M2 arm's training (a3B's teacher-fit input); no M2 arm trained on ours_train_val (427) or paraphrases
  (589). The list's positions are checked against the en/test IDs first. A list (or, for the c4_fit line, an overlap
  file) that does not fit marks the lines that use it NOT EVALUABLE with the reasons; H1 and H2 are written all the
  same, since no rule uses those files.
- **Without c4_fit only** (fresh/mtop_overlap.json; n 4,278), item-only interval default_rng([20261025, 3]): the one
  sensitivity line for an M2 arm's own exposure (a3B, through the teacher-fit sentences).
- **The 55 paraphrase-exact items:** each arm's accuracy on them, on their own line. No M2 arm trains on paraphrases.
- **Unseen intents:** the number of en/eval and en/test items whose intent is not in en/train (each an error for every
  arm), and the number of train intents.

## Power (an estimate from M1's published numbers, not a recompute)

A power script (kept outside the repository) reads tu/m1/verdict_m1.json and tu/m1/results_m1_tm.jsonl only (sha256 in the
pins block). Per M1 set: the item SE of each difference = half-width / 1.96 scaled to n 4,386; the TM seed SE of a
difference of two 3-seed means = sqrt(2/3) x the pooled seed SD, either the VAL SD (best-epoch) or the TEST SD scaled
to n 4,386 (0.181 / 0.236 / 0.295); normal approximation; both probes lie between the product and the minimum of the
per-probe chances. The k-means and the C pick are not modelled (the k-means variance was never measured).

| Quantity (range over the three M1 sets and both seed-SD sources) | H1 TM | H1 LR | H2 TM | H2 LR |
|---|---|---|---|---|
| Half-width of the registered interval at n 4,386 (M1's H1 TM: 0.54 / 0.69 / 1.25) | 0.55-0.62 | 0.66-0.78 | 0.44-0.52 | 0.56-0.69 |
| Point needed for "strong" | +1.05 to +1.12 | +1.16 to +1.28 | -0.06 to +0.02 | +0.06 to +0.19 |

- H1 at a true gain of +1.75 (near M1's means, TM +1.83, LR +1.97): P(STRONG) 0.82-0.96. At +1.0: P(MET) 0.79-0.93.
- H2 at a true difference of 0: P(MET) 0.84-0.96, P(STRONG) 0.14-0.41.
- **At no true gain** (H1 true 0): P(MET) 0.4-10.5% (TM alone 5.7-11.4%, LR alone 7.0-10.5%); P(STRONG) at most
  0.07%. H2 at a true -1.0: P(MET) 0.1-7.8%. So "MET (not strong)" is weak evidence: the verdict's sentence says so
  (the fixed clause above), and every report keeps it.
- **At the bars** (H1 +0.5, H2 -0.5): "met" 25-50% of the time with both probes. "Strong" for the TM probe alone comes
  out 4.2-11.6% of the time (per set: 4.2-8.2% for H1, 4.9-11.6% for H2), not 2.5%, because the registered interval
  leaves out seed noise; LR's 2.5% caps the two-probe rule. Both 2.5% statements hold only given the one k-means per
  table and the C pick, whose variance was not measured and moves TM and LR together (both read the same bits).
- If MTOP is easier than M1's sets (accuracy near 95% or more), the gains and the noise both shrink; these numbers are
  then rough.

## What each outcome leads to

- **H1 met:** the H1 sentence above (with the interval clause only when STRONG). It is a replication of potion's recipe
  on our own word list with bge-base-en-v1.5@a5beb1e3 as teacher, not a new method.
- **H1 not met:** "the registered replication rule was not met on MTOP English", with the per-probe sentence. M1's H1
  then holds only "on reused splits".
- **H2 met:** the H2 sentence above. **H2 not met:** the per-probe sentence; M1's H2 then holds only on the reused
  splits.
- **A run that fails after its test read:** it is not rerun; H1 and H2 are NOT EVALUABLE (the verdict refuses to write
  without every row and says so). A failure before a test read may be rerun once (logged).
- **A matched list or overlap file that does not fit the test split** (another size, IDs that differ at the positions,
  a c4_fit union outside the list or of another count, an unreadable file): the lines that use it are NOT EVALUABLE,
  with the reasons in verdict_m2.txt; H1 and H2 stand, since no rule uses those files.
- Either way: no rerun, no tuning on MTOP test, no second test read for these hypotheses. MTOP en/test's use count
  becomes 1; any later test read of it (J or anything else) is its 2nd use and must say so.

## Test use, declared

| Split | Earlier reads (exactly what was read) | This plan |
|---|---|---|
| MTOP en/test | no label used, nothing scored. fresh/mtop_overlap.py (2026-10-23) split every row into its 8 columns in memory and used column 4 (the utterance); fresh/mtop_freeze_matched.py (2026-10-23) split every row likewise and used column 1 (the ID). | 1st use: the three feature builds parse en/test with its label column (m2_data.load) and store y_test in each cache, unused by the build; the 9 TM runs and 3 LR fits each reload the labels for an equality assert with the cache and read the test split once (12 test scorings); then the verdict; the verify-result recompute reads the same 12 prediction files again (part of use 1; no model is scored again) |
| MTOP en/train, en/eval | the overlap step (column 4); the stand-in dry runs of 2026-10-25 and the GPU dry run to come (each row split in memory; columns 1 and 4 used) | the label column first used by the registered run, for fitting and selection |

- The code enforces one read per run: the prediction file is written exclusively, and a run whose file exists stops
  before training.
- The code check uses CLINC150: m2_standin.load loads CLINC150's test split into memory and replaces it with val before
  any use; no CLINC150 test number is produced.

## Provenance and wording

- C4: "no neural network at any stage". a3B: "network-free at run time, bge-base-en-v1.5@a5beb1e3 as teacher at
  training time". potion-base-32M@1e5a03f8: "the published static table, distilled from bge-base-en-v1.5 by its
  authors" (the teacher revision they used is not known). bge-base is a3B's teacher and potion's; its training data
  cannot be checked for MTOP texts, and MTEB (on which it and potion were evaluated) includes MTOP. These lines are in
  m2_verdict.py's PROVENANCE, printed into verdict_m2.txt, and each claim sentence carries the short tail above.
- Every report says "MTOP English, first use" and gives TM and LR apart (LR is not a Tsetlin machine).
- Network-free claim, tested: in the CPU dry run the C4 and a3B steps imported no sentence_transformers, transformers
  or model2vec (tu/m2_audit_run.py's record); only the potion feature build loads the potion snapshot through
  sentence_transformers.
- **Share-alike.** MTOP is CC BY-SA 4.0. Any data derived from it that is ever released (feature caches, prediction
  files tied to MTOP items, cleaned splits, paraphrases) must carry CC BY-SA 4.0 too. Nothing derived from MTOP is
  published now; M2's caches and predictions stay outside the repository (or under git-ignored data folders).

## Code (each a declared copy of its M1 parent; the exact diffs are saved and pinned)

- **The diffs, exactly:** tu/m2_code_diff_vs_m1.txt, written by tu/m2_code_diff.sh: for each pair m1_X -> m2_X (feats,
  tm, lr, verdict, chain, write_pins, check_cache, standin) the parent's and the copy's full sha256 and `diff -u` with
  fixed labels, then m2_data.py against its first version (tu/archive/m2_data_draft0_f4d2a44b.py: a byte-identical
  rebuild of the 2026-10-23 file, whose sha256 was read at 11:28:16Z into a record kept outside the repository before the
  file was overwritten). The parents' full
  sha256 are in the pins block (their 16-hex prefixes equal M1's frozen pins). tests/test_m2.py regenerates the file and
  compares it byte for byte; so does the verify step.
- **m2_data.py** (changed on 2026-10-25 from f4d2a44b): the pin check compares resolved paths (a relative Path to the
  pinned zip skipped the sha256 check and a str path raised AttributeError; tests/test_m2_data.py showed the failing run
  before the fix); the utterance IDs per split.
- **m2_feats.py, m2_lr.py, m2_check_cache.py:** dataset changes only. **m2_tm.py:** the hard-set rule (M1's code raised
  KeyError 'mtop_en' after writing its test predictions); -1 labels named in the descriptive examples.
- **m2_verdict.py:** one set; H1 and H2 with M1's MET and STRONG; the claim sentences with the weak-evidence clause and
  the provenance tail; the per-probe table and the rule parts with their per-seed values; the seed-inclusive interval
  (none for LR) and the fixed qualifier; TM minus LR; the en/eval differences; the secondary analyses (ID check; c4_fit
  line; NOT EVALUABLE when the list or the overlap file does not fit); refusals: rows (exit 3, with "NOT EVALUABLE"),
  including a table that does not fit its arm (C4 ppw_S1000 e1771d9e, a3B a3B_d768 4a1ed2ae, potion at 1e5a03f8) and an
  arm whose TM and LR rows come from two feature caches; predictions (4); an overwrite (2).
- **m2_chain.sh:** one set, three arms; the code check FIRST, before any step that reads MTOP; nice -n 19 on every step
  and on the code check; tu/m2_freeze_check.py before the go (and in --dry); full sha256 pins checked before every step.
- **m2_codecheck.py** (new): the code check's comparison, tested on M1's own check rows: val accuracy, test accuracy =
  val, best epoch (40), the full 20-point curve, LR val, LR test = val, C (1.0).
- **m2_freeze_check.py** (new): the literal comparison of the plan's pins block with tu/m2/pins.sh (below). Every
  block line is kept, and a path with two different sha256 is refused wherever the stale line stands; its command-line
  paths are read from the working directory; a relative pins line is refused.
- **m2_write_pins.sh:** 58 files (the plan, the M1 parents, the diff, the venv's version files of torch, numpy,
  sentence_transformers, tokenizers, transformers, safetensors and huggingface_hub among them) plus the dry run's
  summary, which is required; modes --list and --block; it writes pins.sh only when the freeze check passes; the dry
  summary, the plan and the output path are resolved with realpath, so the block and the pins hold absolute paths only.
- **m2_standin.py:** M1's code-check stand-ins, plus mtop_small, mtop_unseen and mtop_small_unseen. **m2_audit_run.py,
  m2_dry_gpu.sh** (dry runs only): the audit of zip reads and imports, and the dry run with the -1 stand-in, which
  records the full sha256 of every listed file at its start.
- Imported unchanged: exists_tm.py, w1/wordtab_feats.py, and decision_layer src/dl tm.py, tm_init.py, boolean.py,
  data.py, linear.py, explain.py (read-only; pinned).

## Tests (CPU only, 2 threads, nice 19; outputs and their sha256 in the block below)

- tests/test_m2_data.py: 18/18 (12/18 before the 2026-10-25 fix of the pin check).
- tests/test_m2.py: 84/84 without --feats; 98/98 with --feats. Unit part: the verdict on synthetic
  rows (MET, STRONG, NOT MET; both probes; the bars at the boundary; H2 one-sided; the claim sentences, their
  weak-evidence clause and provenance tail, and the words they never use; the provenance; the per-probe table and the
  rule parts with per-seed values; "no seed level" for LR; the fixed qualifier; -1 labels; the secondary, c4_fit and
  paraphrase lines; the en/eval differences; 13 refusals and the overwrite; 6 cases of a matched list or overlap file
  that does not fit, each with H1 and H2 written and equal to the unbroken case and only the lines that use the file
  NOT EVALUABLE); a real-size verdict (n 4,386, the frozen list and the real overlap file); the code check against M1's
  own rows (pass, and 8 failures); the freeze check (2 passes, one with command-line paths relative to the working
  directory, and 12 refusals, among them a path with two hashes and a relative pins line); the diff file regenerated
  byte for byte. --feats: the CPU dry run on the real en/train and
  en/eval texts with the -1 stand-in (56 labels -1; its own check passed; only en/train.txt and en/eval.txt opened; the
  C4 and a3B steps network-free; tu/m2 untouched; the summary names all 58 file hashes), M1's m1_tm.py failing with
  KeyError 'mtop_en' after its test write and m2_tm.py writing its row, the one-read refusals, and the verdict on the dry
  run's own predictions under the audit (the 56 -1 labels counted as errors).
- The first runs of tests/test_m2.py on the second revision's code failed one check each, both errors in the tests, fixed
  there and not in the code: the unit part's qualifier scenario had too little seed spread (its seed-inclusive lower end
  cleared the bar), and --feats compared the verdict's exact accuracies with the rows' 4-decimal rounding (the verdict
  had checked every row itself and exited 0).
- tests/test_m2_chain.sh: 94/94 (stubs that write minimal outputs: 19 steps in order, all under nice 19; a
  second run skips all 19; a failing step, then a restart that runs only the 12 steps not done; TERM during a step and
  while waiting for the go; a refused pin; a pinned file changed between two steps; no pins file; no matched list; a
  16-character pin; the freeze check refusing a differing block and pins that do not pin the plan; the dry listing;
  m2_write_pins.sh's modes, its required dry summary and its refusals; relative dry-summary, plan and output paths
  written as absolute paths and accepted by the chain from another working directory).
- The freeze-check fix of 2026-10-25, shown on small files before and after: the earlier
  checker passed a stale zero line placed before the right one ("freeze check ok", exit 0) and refused a --plan given
  relative to the working directory; the fixed checker refuses the first ("2 different sha256") and accepts the second.

```
# the pin-check fix of 2026-10-25: before (12/18, by design) and after (18/18)
# the freeze-check fix of 2026-10-25, on small files: before (the stale line passed) and after
# the registered code: loader 18/18, chain 94/94, unit 84/84, --feats 98/98
# the CPU dry run inside the passing --feats run
02dc6a1ddebec81cc8e4039d2b7a45285ab320a29fb7fdffcb758f4886ea106c  tu/tests/runs/20261025T135550Z_m2/dry/20261025T135635Z_cpu/dry_summary.json
321c5311e84a78a5d1db180888556140250940107fb50c9d9415ff2e5c291b4e  tu/tests/runs/20261025T135550Z_m2/dry/20261025T135635Z_cpu/dry.log
94fa9baf0bbb73850df0130f5847207bcf146cd7f60644911647fee5a2aaa561  tu/tests/runs/20261025T135550Z_m2/dry/20261025T135635Z_cpu/audit.jsonl
0d312ec960a3bf39ea0a13fed9ae97f88dba1810395c1c730cfefa34241f2755  tu/tests/runs/20261025T135550Z_m2/dry/20261025T135635Z_cpu/code_sha256.txt
```

## Dry run (GPU request 1, before the freeze)

`nice -n 19 bash tu/m2_dry_gpu.sh` (about 4-5 GPU-minutes; cap 10): the full sha256 of every listed file first; the
code check against 3c's rows (m2_codecheck.py); the three feature builds on the full MTOP stand-in with -1 labels; a
smoke TM run per arm and one full-size TM run (a lower bound on the time of one registered run: 20 stand-in classes,
not MTOP's intent count); LR per arm; its own check (the -1
accounting row by row). It reads no MTOP label column and never opens en/test.txt. Its dry_summary.json must say
"pass": true, and every pinned file must still have the sha256 it recorded, or the freeze check refuses.

## Freeze procedure (after the dry run; every time read from the clock in the command that writes it)

All of it runs from studies/text_understanding; m2_write_pins.sh resolves its path arguments with
realpath, so the block and pins.sh hold absolute paths only.

1. `bash tu/m2_write_pins.sh --block tu/m2_dry/<UTC>/dry_summary.json` gives the final sha256sum block (every listed
   file except this plan, plus the dry summary); it replaces the code and data part of the block in the section that
   starts at the heading line "## Pins" (cited records may stay; every hash line in that section is checked against its
   file, and hash lines anywhere else are not read, so a block placed outside the section is refused).
2. The status line becomes "REGISTERED, frozen <UTC>" (date -u in the writing command).
3. `bash tu/m2_write_pins.sh tu/m2_dry/<UTC>/dry_summary.json` writes tu/m2/pins.sh only if m2_freeze_check.py passes:
   every pins line except the plan equals the plan's block literally, the pins pin the plan itself, and the dry summary
   passed with the same code hashes. If step 3 refuses, the plan is not frozen, the cause is logged and fixed, and
   steps 1-3 run again.
4. Into the freeze record (the study log and the research log): the freeze time, pins.sh's sha256, and the output of
   `bash tu/m2_chain.sh --dry` (the freeze check again and the 19 steps) with its sha256 (the dry listing creates the
   empty folder tu/m2 if it is missing, as M1's chain did).
5. After the run, the verify step compares: the chain log's "pins <sha256>" line with pins.sh's recorded sha256;
   pins.sh with the plan's block (m2_freeze_check.py again); and m2_code_diff.sh's output with the pinned diff file.

## The report step (fixed now)

- The code check must reproduce, exactly, 3c's CLINC150 val rows: 3c-ref seed 4, s 8, val accuracy 0.945, best epoch
  40 and its full 20-point curve, on val and again on its "test" (a copy of val); and ref_rebuilt's LR val accuracy
  0.9503 at C 1.0, likewise (m2_codecheck.py; M1's own check rows pass it).
- The verdict: `m2_verdict.py --m2 tu/m2 --matched fresh/mtop_test_matched_v1.json --overlap fresh/mtop_overlap.json`
  (the chain's last step), writing tu/m2/verdict_m2.json and verdict_m2.txt once (it refuses to overwrite).
- No other report command reads MTOP's test labels, apart from the verify-result recompute of the same 12 prediction
  files (part of use 1; no model is scored again).

## Cost and order

- **GPU request 2, the registered chain:** `setsid nohup nice -n 19 bash tu/m2_chain.sh > /dev/null 2>&1 &` after the
  freeze, on the go file tu/.m2/gpu_go (one line with its time read by date -u in the writing command, and who gave the
  go): about 12 minutes from M1's log (code check 1.1, three feature builds 1, nine TM runs about 49 s each 7.5, three
  LR fits 1, verdict 0.5, gate and start-up 1); cap 25. Alone on the GPU.
- **After the run:** an independent recompute with its own bootstrap seed and a check for leakage, selection and
  integrity, then the research log, before any number counts.
- **Constraint until "m2 chain end":** the pinned files must not change; no git operation or sync may touch
  studies/decision_layer/src/dl or scripts/cpu_load.py (shared with E2, E4, C step 9 and S4). A change stops the chain before its next step.

## Priors (honest; judgement, not computed)

- H1 met about 85%, strong about 65%: M1's gains were +1.6 to +2.2 on three sets, HWU64 (the closest domain) +2.2;
  a ceiling near 95% on MTOP could shrink the gain.
- H2 met about 65%, strong about 20%: M1's points were -0.48 to +0.59, and at MTOP's size "strong" needs a3B level with
  potion or better.

## Limits

- One fresh set; the claims are about MTOP English only.
- Teacher contamination (bge-base, potion) is not checkable; MTEB includes MTOP, so model selection by the teacher's
  developers on MTOP test cannot be ruled out.
- The registered interval leaves out seed, k-means and C variance; the seed-inclusive interval covers seeds coarsely;
  the k-means variance was never measured.
- 26% of MTOP's test texts are within one word of its own train or eval texts, which helps every arm alike.
- **The full C4 S1000 stream was not scanned for MTOP test texts** (decision 6). By W1's precedent such a scan would
  almost surely find hits, but it could change no rule or wording: the C4 table and a3B share the stream (a3B was
  refined from the C4 table, tu/a3/fit_meta.json), so in H1 the exposure mostly cancels; the only teacher-mediated path
  into a3B, 3a's fit sentences, is already scanned (2 exact, 108 one-edit) and has its own line; potion was distilled on
  its own C4 sample, which cannot be scanned here, and any exposure there helps potion, against a3B in H2. A scan would
  be one more text-only read of en/test to declare.

## Pins

sha256sum output; the code and data part was replaced at the freeze by the output of `bash tu/m2_write_pins.sh --block tu/m2_dry/20261025T140731Z/dry_summary.json`, read 2026-10-25T15:03:02Z in the command that wrote it here; the cited records were read 2026-10-25T13:09:16Z. The freeze procedure replaces the code and data
part with the final block and writes tu/m2/pins.sh, which also pins this file and the GPU dry run's summary.

```
# every file of tu/m2_write_pins.sh --list except this plan (code, tests, M1 parents, the diff, imported code, tables, data, code-check inputs, potion, venv version files)
cf8ac5f81dc461ded4228b0303eb8663ae77153f3f30a7c580d1a69e447eb992  studies/text_understanding/tu/m2_chain.sh
323d772e0e9532d64f64d364603f9bbc2e98f2378946a53f268999a4ceb35b71  studies/text_understanding/tu/m2_feats.py
379409332c185ef91e23a8da7f5c9ca734daeb7391929e006a7d7c7d08f6b37a  studies/text_understanding/tu/m2_tm.py
7cd95e963dba4fcc47d6c87df7b10a078bc67d3d3edac18333425cd75cb57d33  studies/text_understanding/tu/m2_lr.py
a67bbdb1526a098e33c3be1d093c3302b1062bdddedbd0e70ca4d88feb7f1a96  studies/text_understanding/tu/m2_verdict.py
265eb3113ad6ca06d2d9706ae2c89bba2c08d45176b5b373d96a2aa0fdd0ef11  studies/text_understanding/tu/m2_standin.py
d2804841d943ecf7587a5203c406fdb6792da4584e33efc83070ac244ae18247  studies/text_understanding/tu/m2_check_cache.py
d65b9c3ec952284c35fa6c23e4d01ddd42f043219feebcfe1c8ba6932968cd20  studies/text_understanding/tu/m2_data.py
e9774234c095f7d92475220205706d2f9e867c76bdd9fce55da29c75ad285534  studies/text_understanding/tu/m2_write_pins.sh
4a5b9a03ad0d3129496bcaed5448ad22d1370186a8740cefbab0afc8738c6727  studies/text_understanding/tu/m2_dry_gpu.sh
5a769fcfd92c9847371064c798fb1cfd1247092448d71db3547a2b7b5f81f04e  studies/text_understanding/tu/m2_audit_run.py
a21518b492858286d01a0dbb1c5f397621c12de94f372eda5430372b5c705663  studies/text_understanding/tu/m2_codecheck.py
155ce8e9c449e0c9f84b63b34782a18cce33217d03a2b2b03add18a81c44e891  studies/text_understanding/tu/m2_freeze_check.py
c60bc2ad39d0b030074ef8fa5b1450e0218c7eb9268771815a9693f65156f588  studies/text_understanding/tu/m2_code_diff.sh
35383ea407f9003de5b8d186f28df7a48e72fe15a3e0fc7d4b6bdf1eee5c841c  studies/text_understanding/tu/m2_code_diff_vs_m1.txt
a126f39a767764a4abc1179f7a078017ef05495682d72ee7e41ae4a5b051432c  studies/text_understanding/tu/tests/test_m2.py
957b992de0fa46efb76661ddc4fdf747e76407d70712be17143f3cb1edebb808  studies/text_understanding/tu/tests/test_m2_chain.sh
e6b2b22643ec6b35982e6af7586702fe5c9253ba834d117b49d817300f82a01d  studies/text_understanding/tu/tests/test_m2_data.py
bb93c24f72edfe67465c5f2b102d415763d9ddbc29fb6e7ea54056f8082ebef9  studies/text_understanding/tu/m1_feats.py
06ccf8eba7505de7f0266131d06e063ee950a5754254ff6b874bcf18208b765f  studies/text_understanding/tu/m1_tm.py
d2a41815a2ccc5688b30227a8a58c21ef2f24a120bbbae83cabe1c65ae4db127  studies/text_understanding/tu/m1_lr.py
d32188aaf86a3c8d2db11ac27c55d50f63973b9bfffe7fa9c0eb7fb7f067c442  studies/text_understanding/tu/m1_verdict.py
1db3b6461cfe3bab728d6b1089fb660c4b050aba6f22d8558ee72cd9b107b193  studies/text_understanding/tu/m1_chain.sh
33d289fe82d1a62c9ca6729c95ac392a5d5f9f3443181c75c748560ac2bf2eb3  studies/text_understanding/tu/m1_write_pins.sh
6eccc2b88763d531fd41819264b0f2311c1893149e41cd6f12e33bab7653a5bb  studies/text_understanding/tu/m1_check_cache.py
d295f9ddba944c4d4e781da50589e4825846fff80ad659b3482d05eafbbf1138  studies/text_understanding/tu/m1_standin.py
f4d2a44b1120b8781e93dfb4477c5a4db74abef0c72b11de20903dddd5e21612  studies/text_understanding/tu/archive/m2_data_draft0_f4d2a44b.py
7da0aee9f4eef7969976463f7a8a65319a14466e0235bfcc2500c632107d3942  studies/text_understanding/exists_tm.py
001a93b259a1b34f677d317794c3c00c121eb04b0dbf58951cf8b4139c461776  studies/text_understanding/w1/wordtab_feats.py
f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424  studies/decision_layer/src/dl/tm.py
557ff697065297387c0b698ec77ae1159496e10fd9c1547dc79dbf2ee67ef577  studies/decision_layer/src/dl/tm_init.py
b5f5917f390da8a6f73a215b08fe713f2fd0de24d7c950abbf5b31e8af36cc57  studies/decision_layer/src/dl/boolean.py
0c4f8d22576e817c6cb676582c496d4b5e3ee613208f878e1176161799159a1f  studies/decision_layer/src/dl/data.py
e3ef06b96ef60740af7757c9f8c0672d7c6258404d67936bd4106723d3a0a65d  studies/decision_layer/src/dl/linear.py
1b6f79a8965f31851f59a7f70121886e9a1d8133fc2a11a4c65a15c32576e780  studies/decision_layer/src/dl/explain.py
2e300054a9284ef4bb0d395048fb5988d3176c100aa184384d22a3cc5eb04328  studies/decision_layer/scripts/cpu_load.py
4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855  studies/text_understanding/tu/a3/tables/a3B_d768.npz
e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a  studies/text_understanding/w1/tables/ppw_S1000_d500_e0.5.npz
0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b  studies/text_understanding/data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip
1ea621f8c8a67ff05c76c4b30a595bdc2f650ddfc85586c4bf3d1462b8d98405  studies/text_understanding/fresh/mtop_test_matched_v1.json
aba34b7e710108552d01b9123b8c6230f786852985cd634e1e6e867212a4b8ff  studies/text_understanding/fresh/mtop_overlap.json
a59544c7915b2367d6f18e59530a11749ae072124f1f50a8a36482501ffe4d32  studies/text_understanding/tu/c3/feats/orig/clinc150__p32tok_K512.npz
b15ea06a26e99cabec44b77cc92b40197a4781a0ba33ea9cb53c97eff23aec9a  studies/text_understanding/tu/c3/feats/orig/clinc150__p32tok_K512.meta.json
4c7cda604f41e0c66b52f08145119bf1d5d588dd302b83d1fc2c2c9770cdddcf  studies/text_understanding/tu/c3/results_c3_val.jsonl
de04f93a52a076feda437249bef5a4a76da0988265c48f1d5ea0f8d91e75d653  studies/text_understanding/tu/c3/results_c3_lr.jsonl
42959c198eabb2b29b8283b866d41508f26f57018288676c75bc5a2aee3d02fa  studies/text_understanding/hardset.json
36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0  studies/decision_layer/data/raw/clinc150_data_full.json
99f6c33204c9231a7391871b7a3c91409b532c8f587a9ea44fc282303d8dec28  ~/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots/1e5a03f8eeb2c98b928fbbd846f22f816360919f/model.safetensors
a68dcbed0429dcdd5bfdca92b0b03cc30d09122c0a3fcf4758787d4b244e45b2  ~/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots/1e5a03f8eeb2c98b928fbbd846f22f816360919f/modules.json
7d75cbc54318138807c401b0f0c9721117c628b39de8e8e0edb6cb17e0ee7d18  ~/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots/1e5a03f8eeb2c98b928fbbd846f22f816360919f/tokenizer.json
fb37823f3a4c3c29cd008fac7d3ce5aeb8e7da6f883b17aa9ea6d73ce6eecbfd  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/torch/version.py
833eb1051581de6bf4d23cd6f090e0fa4cbab6c548463dc718b5be62920fa2d7  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/numpy/version.py
03873bdff225ec7e9fc70778b1335d6dd82959ac56bf1afd7ac70b14e2271900  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/sentence_transformers/__init__.py
b1cee924695651af5c3dab98e8c34406ab355a2cc1d633228646c470210e649e  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/tokenizers/__init__.py
17651056183b3f8bcba527684aa92307b0edc5ba1ff887a7b92539d84424005d  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/transformers/__init__.py
92c988289b8c225a92f25e487ad5d7a1eb7e326eec0fedfc6b75e6f04d85e32f  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/safetensors/__init__.py
284be6e001a015f9933e756c6f6552b7266d811aff7b5124c960b5f1fc66b27e  studies/decision_layer/.venv-dl/lib/python3.11/site-packages/huggingface_hub/__init__.py
2fce02b04c7d4e954bc3e52db26926f2939b63e6b0730ff6c195b130e380106e  studies/text_understanding/tu/m2_dry/20261025T140731Z/dry_summary.json
# cited records (not read by the chain)
b2998726b4050ca514313b0cbb9126432550ef2f6bbb60f4dfe1308356435398  fresh/NOTE_2026-10-23_mtop.md
8726804cb5a7f8d8fdd5a6538f74c2a2a321cbdd6bf1ba598907b47c203139dc  fresh/mtop_overlap.py
f1cf7744e4a76c177b40cedfb3d831d40db66bbfa7dc417e76511ada150edeeb  fresh/mtop_freeze_matched.py
af5afc9f11f8e1f246b15a6decbb55256b9d1874478d1285d4a012b66a8cf6ff  tu/PLAN_M1.md
a8b451eda7ef6a8173e33c1fcfcae3220e5d5173cb50ce07c40f1e84ffd8fd9b  tu/m1/REPORT_M1.md
f451a3c05f95b9acf8128c42b463f853a52eee97c64d9a3c1c4c0866657034ae  tu/m1/ADDENDUM_2026-10-22.md
09ee22e3ba99a3374f582e1fe09c8d0331525dc90ccf4b4f3d9ad2f513e6390d  tu/m1/verdict_m1.json
3063dad47ded0a85d21f374c1bf0b556b4fc956bd9ae73ee3f429095e3719540  tu/m1/results_m1_tm.jsonl
77059a486dcb0ebb1fcc35dac4738b4073a7aba49aaa8e27bc9880e9ac821e8a  tu/m1/pins.sh
7a486389f8cf1629ac0b1ca057f2297e8676f84b884c43d757dd27600a2d1331  verify_m1/VERDICT.md
47325d24255c6977eb880cc7da249fee332b6b0103ee12a33626032c23ec78ca  novelty/n3d_2026-10-21/report.md
09062883ecdd5319671b5c94ecbb32369d96cedfb766b3a6af1ebbe511d35acc  review_3d_2026-10-23/reviewer_design.md
4ef13fdff37fa88a3980f178fa80604869aee8addf781430985284b33c06839d  review_3d_2026-10-23/reviewer_data.md
a107c9c12664bde0cf2bbb084b6fcaaae2dad7bcca4dbaccf29c67c07dd916bf  w1/corpus/c4_slices.json
9ca2c41298c63c9dda5d9bb02f15cc7830387d96f8855c808fbf29ef2761fb83  tu/a3/fit_meta.json
1e1d1fdb9479ce0ba271cfa4dc64e8c90e7770699bd1a481f8c09d658d34fe9c  tu/a3/teacher_meta.json
a68a0333e9ef2d4fbb46f50919b64fbb0f259e666cb6cf7b17ecf2a1ae96bf18  ~/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots/1e5a03f8eeb2c98b928fbbd846f22f816360919f/README.md
1c4313cb4e8d47e356d7077835b5ffb9617934a7450a323eb4a98fbc6a8b1e3c  tu/archive/m2_test_m2_data_draft0_1c4313cb.py
5a769fcfd92c9847371064c798fb1cfd1247092448d71db3547a2b7b5f81f04e  tu/archive/m2_draft1_code/m2_audit_run_5a769fcf.py
524c8b1293c99ff1c23b04675f79083ff23fd390d93f9da3789acf48569351f6  tu/archive/m2_draft1_code/m2_chain_524c8b12.sh
d2804841d943ecf7587a5203c406fdb6792da4584e33efc83070ac244ae18247  tu/archive/m2_draft1_code/m2_check_cache_d2804841.py
d65b9c3ec952284c35fa6c23e4d01ddd42f043219feebcfe1c8ba6932968cd20  tu/archive/m2_draft1_code/m2_data_d65b9c3e.py
edb77eda0ac7c4b95c6efbf70da40db884f8a29af6e7fb7a4e03e2b0b1d9d9f0  tu/archive/m2_draft1_code/m2_dry_gpu_edb77eda.sh
323d772e0e9532d64f64d364603f9bbc2e98f2378946a53f268999a4ceb35b71  tu/archive/m2_draft1_code/m2_feats_323d772e.py
7cd95e963dba4fcc47d6c87df7b10a078bc67d3d3edac18333425cd75cb57d33  tu/archive/m2_draft1_code/m2_lr_7cd95e96.py
e387de8b08b6007a3e038d66731df5d8d6ea1dd6dee1fa8802c790ad6f3ebf3a  tu/archive/m2_draft1_code/m2_standin_e387de8b.py
379409332c185ef91e23a8da7f5c9ca734daeb7391929e006a7d7c7d08f6b37a  tu/archive/m2_draft1_code/m2_tm_37940933.py
02d41713e207341bf1b6c4e2ea9944ee1f44c4c2c3b55af69c5583b905d9e46d  tu/archive/m2_draft1_code/m2_verdict_02d41713.py
33f48124897f6f8715d051b5a6d8d0407b88ee9b517afb350e54057e8e673427  tu/archive/m2_draft1_code/m2_write_pins_33f48124.sh
1bdecec7498c892756374d70523ae95c87bba4084fa358122c8eecdbd9f811da  tu/archive/m2_draft1_code/test_m2_1bdecec7.py
ac0e3491eeee89bcfaee87020648f27701df7904c7413f6658d5d775eb6d2203  tu/archive/m2_draft1_code/test_m2_chain_ac0e3491.sh
e6b2b22643ec6b35982e6af7586702fe5c9253ba834d117b49d817300f82a01d  tu/archive/m2_draft1_code/test_m2_data_e6b2b226.py
5a769fcfd92c9847371064c798fb1cfd1247092448d71db3547a2b7b5f81f04e  tu/archive/m2_draft2_code/m2_audit_run_5a769fcf.py
cf8ac5f81dc461ded4228b0303eb8663ae77153f3f30a7c580d1a69e447eb992  tu/archive/m2_draft2_code/m2_chain_cf8ac5f8.sh
d2804841d943ecf7587a5203c406fdb6792da4584e33efc83070ac244ae18247  tu/archive/m2_draft2_code/m2_check_cache_d2804841.py
a21518b492858286d01a0dbb1c5f397621c12de94f372eda5430372b5c705663  tu/archive/m2_draft2_code/m2_codecheck_a21518b4.py
c60bc2ad39d0b030074ef8fa5b1450e0218c7eb9268771815a9693f65156f588  tu/archive/m2_draft2_code/m2_code_diff_c60bc2ad.sh
caecf316c66f3133389d9c9c6901b079cc70b3259306d0fa8f6b60a5012a1198  tu/archive/m2_draft2_code/m2_code_diff_vs_m1_caecf316.txt
d65b9c3ec952284c35fa6c23e4d01ddd42f043219feebcfe1c8ba6932968cd20  tu/archive/m2_draft2_code/m2_data_d65b9c3e.py
a19786756c054831c6bb14f9865b95a5a59a871cbca08e712faee8a438424b70  tu/archive/m2_draft2_code/m2_dry_gpu_a1978675.sh
323d772e0e9532d64f64d364603f9bbc2e98f2378946a53f268999a4ceb35b71  tu/archive/m2_draft2_code/m2_feats_323d772e.py
1f879d60ba48c9dbddc8346f00e9391b138da0e026252a2ea821b9f6760741f9  tu/archive/m2_draft2_code/m2_freeze_check_1f879d60.py
7cd95e963dba4fcc47d6c87df7b10a078bc67d3d3edac18333425cd75cb57d33  tu/archive/m2_draft2_code/m2_lr_7cd95e96.py
265eb3113ad6ca06d2d9706ae2c89bba2c08d45176b5b373d96a2aa0fdd0ef11  tu/archive/m2_draft2_code/m2_standin_265eb311.py
379409332c185ef91e23a8da7f5c9ca734daeb7391929e006a7d7c7d08f6b37a  tu/archive/m2_draft2_code/m2_tm_37940933.py
4ed45e4a21e44d79d1a42560ed824eb700d615f82dbc2cdcce8acb53f1442598  tu/archive/m2_draft2_code/m2_verdict_4ed45e4a.py
c2cd5c106b82db56251585e854937f415447c06fc08e51de6c905b85642e4dd6  tu/archive/m2_draft2_code/m2_write_pins_c2cd5c10.sh
8fde92ae169b7a6805a601bd26f0994bf7a36c5f39ae9d3ffdb3841e2f134ce3  tu/archive/m2_draft2_code/test_m2_8fde92ae.py
8889e9395d15aa497be2a340e5a5da4a9864aaeb705fe0e85037bba81852d1b9  tu/archive/m2_draft2_code/test_m2_chain_8889e939.sh
e6b2b22643ec6b35982e6af7586702fe5c9253ba834d117b49d817300f82a01d  tu/archive/m2_draft2_code/test_m2_data_e6b2b226.py
```

