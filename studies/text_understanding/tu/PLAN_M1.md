# PLAN M1: the text-understanding milestone, one registered test of the 3a and 3c stack (REGISTERED)

Status: REGISTERED, frozen 2026-10-22T10:51:33Z (UTC, read from the clock when this line was written).
- Drafts: draft 1 ff04610df5c2ef7a (2026-10-21 13:05Z); draft 2 written 2026-10-21 13:25Z with the changes of
  13:10Z / 13:15Z.
- Draft ce995b93f8767d18 was accepted at 10:54Z on 2026-10-22 with one rule fix: s 8 for every arm, and R3c's picks
  descriptive only. That fix is in this text.
- c3x's report was read before the freeze; its numbers are in "What was known". Step 0 ran before the freeze; its output
  is pinned.
- No M1 model has been scored on a test split. Every code and input file is pinned in tu/m1/pins.sh, written after
  this line, and the chain checks the pins before every step.
- Runs after decision_layer's e84, on e84's end line, once the GPU is empty.

## Why now

- **The plan reached its milestone.** PLAN_text_understanding.md, order of work, item 5: after 3a, 3c and 3b comes
  "the agent check (R4), the guards (R5), then one registered test (R7)".
- **3a.** PASS on val, 3 of 3 sets (tu/a3/verdict_a3.txt).
  - R4: TAKEN in the agent (decision_layer, e82, 03:21Z).
  - R5: 250 of 250 guarded proofs (05:26Z).
- **3c.** PASS on val, 2 of 3 sets, with narrow margins.
  - Its TM gains are largely explained by about 5x more training updates.
  - The LR gains are the cleaner signal.
  - It was not run in the agent.
- **3b** is closed, negative. 2a, 2b and 2c were not run (step 1's shares).
- **Every main-line number so far is val.** The thesis needs test numbers for the stack. The agent-side registered
  test stays blocked: e82's go-bars failed at 05:09Z. This plan uses only the intent sets.

## What was known before writing this (all val; no test number of M1's arms exists)

- **3a** (tu/a3/verdict_a3.txt; TM = the mean of seeds 4-6 at s 8):

  | Set | a3B TM | C4 TM | potion TM | a3B LR | C4 LR | potion LR |
  |---|---|---|---|---|---|---|
  | CLINC150 | 94.49 | 92.29 | 94.45 | 95.03 | 92.73 | 94.73 |
  | Banking77 | 91.56 | 88.37 | 91.69 | 91.53 | 87.34 | 91.03 |
  | HWU64 | 90.27 | 89.03 | 91.17 | 90.24 | 88.85 | 91.17 |

- **3c** (tu/c3/verdict_c3.json).
  - Registered gains of 3c-F over the reference: CLINC150 TM +0.79 / LR +0.77; Banking77 +1.00 / +0.69; HWU64 +1.08
    / -0.37.
  - At matched updates (3c-F at epoch 8 against the reference at epoch 40, read from the stored curves): TM +0.21 /
    +0.23 / -0.25.
  - Banking77's LR gain depends on C: +0.39 at C 10, +0.89 at C 1.
- **R3, the reader's s.** The single-seed screen picked s 16 on HWU64. Over seeds 4-6, s 16 gives 89.28 against s
  8's 90.27, on val only. 0b picked s 8 for the C4 table and for potion on all 3 sets.
- **c3x** (one k-means per set; F2 = 3c's kept_F rebuilt; twin = the EDA copies; rep5 = the originals repeated 5x).
  Source: tu/c3x/report_c3x.txt 07a87cbea14e54fa; the independent recompute matches (verify_3c/recompute_c3x_20261021T143624Z.json, 132 checks).
  All are val numbers for CLINC150 / Banking77 / HWU64.
  - F2 - orig2: TM +1.01 / +1.00 / +1.39; LR +0.77 / +0.69 / -0.37. 3c replicates.
  - F2 - twin: TM +0.88 / +0.96 / +0.71; LR +2.43 / +2.69 / +1.76. twin - orig2: LR -1.66 / -2.00 / -2.13.
  - F2 - rep5 (matched updates): TM +0.64 / +0.89 / +0.25. rep5 - orig2: +0.37 / +0.10 / +1.14.
  - CLINC150 F2 - F_rk (reader-identical copies of eval texts removed): TM -0.00, LR +0.00.
  - These were read after H3a and H3b were registered in draft 2 (13:25Z on 2026-10-21). The hypotheses did not
    change.
- **R3c** (the R3 re-tune on the 3c stack; tu/r3c/NOTE_2026-10-22_R3c.md) had not run when this plan was frozen.
  - **Rule:** s 8 stays for every M1 arm, whatever R3c picks. R3c's picks are descriptive only for M1, reported
    beside M1's results, so the plan has no val-driven fork.
  - If R3c picks another s for a set, the limit is stated: "the 3c stack at its own tuned s was not tested".
- **T2's test rows ran below val for every view** (results_confirm.jsonl, seeds 1-3; potion existsT -0.37 / -1.04 /
  -2.85).

## Rules this plan applies (PLAN_text_understanding.md section 0)

- **R3.** s 8 for every arm and set, declared here with the val evidence above, before any test read.
- **R4.** 3c is "kept for classification only, not tested in the agent", as R4 allows. An agent version would need
  paraphrases of the agent's own dialogues, which is its own plan in the decision_layer study.
- **R5.** No new bit block, so no new guard proofs.
- **R2, the ablation.** C4+3c gives "the stack minus a3B". a3B gives "the stack minus 3c".
- **R7.** This registration, with the use count below.

## Arms (each set: CLINC150, Banking77, HWU64)

| Arm (code name) | Table | Training rows | Updates per epoch | Probes |
|---|---|---|---|---|
| C4 | ppw_S1000_d500_e0.5 (e1771d9eaf4a07f7; no neural network at any stage) | originals | 1x | TM, LR |
| a3B | a3B_d768 (4a1ed2aed7aa4ce7; bge-base teacher at training time) | originals | 1x | TM, LR |
| potion | potion-base-32M at revision 1e5a03f8 (the published static table, distilled from bge-base) | originals | 1x | TM, LR |
| a3B_3c | a3B_d768 | originals plus 3c's kept_F paraphrases (tu/c3/kept_F.jsonl, d359ea5aad34dfea) | about 5x | TM, LR |
| C4_3c | ppw_S1000 | originals plus kept_F | about 5x | TM, LR |
| a3B_twin | a3B_d768 | originals plus the EDA copies (tu/c3x/twin_kept.jsonl, 37326506ae233a3f; PLAN_3c lines 113-116) | about 6x | TM, LR |
| a3B @5 | a3B_d768 | originals repeated 5x (--repeat 5) | 5x | TM only (the updates-matched control) |

- **Reader.** As in 3a, 3c and c3x: CoTM, 4,000 clauses, T 4,000, 40 epochs, batch 128, s 8.
  - The epoch is chosen on val. dl.tm keeps the best-val state and restores it after the fit. The test is predicted
    once with that state (confirm_tm.py's protocol).
  - Seeds 7, 8 and 9 are fresh; val used seeds 1-6.
- **LR.** dl.linear.fit_logreg on the same existsT bits. C in {1, 10, 100} is picked on val (the first strictly
  highest, as exists_tm.best_logreg does). The test is predicted once with that fit. Every C's val accuracy is kept.
- **Features (m1_feats.py).**
  - One CUDA k-means per set and table: K 512, 25 iterations, seed 0, on the ORIGINAL training tokens, with the
    thresholds fit on the original training rows (c3x_feats.py's steps).
  - The a3B arms share one k-means per set, and the two C4 arms share another.
  - train, val and test are pooled in the same run. The centroids are saved.
  - potion uses exists_tm.features' tokenizer path, with the cached snapshot read offline.
- **Code check, before any M1 run.**
  - 3c's CLINC150 rebuilt-original cache is copied with "test" replaced by a copy of val (m1_check_cache.py, the
    stand-in dataset m1_standin.py). No test text or label is used.
  - m1_tm.py (seed 4, s 8) and m1_lr.py must give 3c's 3c-ref and ref_rebuilt rows exactly, and the same number again
    on that "test". Otherwise the chain stops.

## Hypotheses, fixed before any test read

**Met** means that the bar holds on the point estimates (TM = the mean over seeds 7-9; LR = its one fit) on at least
2 of 3 sets.
- H1, 3a holds on test: a3B - C4 >= +0.5, with TM and with LR.
- H2, 3a within a fixed margin of potion: a3B - potion >= -0.5, with TM and with LR. The wording if met is "within 0.5
  points of potion-base-32M"; never "beats", never "statistically non-inferior".
- H3a, the augmentation: a3B_3c - a3B >= +0.5, with LR. The TM leg is confounded by the extra updates, so it is
  descriptive.
- H3b, the paraphrase claim: a3B_3c - a3B_twin >= +0.5, with LR. It asks whether Qwen's paraphrases beat cheap EDA
  copies of the same rows.

**Strong.**
- On a set, a hypothesis is strong when the lower end of its 95% interval clears the same bar with every probe the
  hypothesis uses (for H2, the lower end is at least -0.5).
- The hypothesis is STRONG when that holds on at least 2 of 3 sets.
- The interval is a paired bootstrap over test examples:
  - 10,000 resamples per set, numpy default_rng([20261021, set index]), percentiles 2.5 and 97.5;
  - one index draw per resample, shared by every comparison of that set and by seeds 7-9. The TM statistic is the
    seed-averaged paired difference on the drawn examples.
- **Limit:** the interval leaves out the variance between training seeds. So every TM number is given with its three
  per-seed differences, the paired SE over seeds, and each configuration's seed SD. A reader can then see when one
  seed carries a gain.
- Every interval is reported, met or not.

**Multiplicity.** H1, H2, H3a and H3b are separate claims, with no correction for multiplicity. Meeting one says
nothing about another.

**Descriptive (no bar).**
- TM: a3B_3c - a3B (as registered on val), a3B_3c - a3B @5 (matched updates), a3B @5 - a3B (the reader's budget),
  a3B_3c - a3B_twin, a3B_twin - a3B.
- The R2 ablation: a3B_3c - C4_3c, TM and LR.
- LR: a3B_twin - a3B.
- Each run's own val accuracy, beside the earlier val rows. Small differences are expected from the new k-means.
- Each arm's test accuracy without the step-0 matched test texts.

**Wording.** Always all 3 sets. "Network-free at run time; bge-base (and, for 3c, Qwen3.5-9B) at training time". The C4
arm is "no neural network at any stage".

## Step 0 (CPU, BEFORE the freeze; reads test TEXTS, never test labels; no model is scored)

- **This step reads the test texts before the freeze, and that is declared here.** Its output file and its sha256 go
  into the registration and the pins.
- **m1_overlap.py.** Each test text gets three keys: exact (build_corpus_c4.norm), order-free (sorted words) and
  reader (the set of in-table tokens). Matches are counted against:
  - 3a's 1,999,932 C4 fit sentences and 39,995 held-out sentences (exact and order-free; exact should be 0, since
    a3_sentences.py drops evaluation keys);
  - kept_F rows of the same set (exact, order-free and reader).
- **a3_teacher.py's 64 check sentences** are checked against every evaluation text.
- **Rule, fixed now.** No rows are removed. The counts are reported, and the verdict gives each arm's test accuracy
  without the matched test texts (descriptive).
- **a3_teacher.py.** Every check sentence that is an evaluation text ("please turn off the lights in the kitchen" is
  expected to be one) is replaced by a sentence that is in no evaluation split, and the wrong comment ("no evaluation
  text") is fixed.
  - The check sentences are a numerics check only and never entered the table fit.
  - The version that ran is kept under its hash. The swap is logged with the old and new hashes.
- **Not checkable.** Whether bge-base, potion or Qwen3.5-9B saw these test sets in their own training is stated as a
  limitation.
- **Step 0 result.** Run 2026-10-21 13:35:16Z-13:35:49Z: tu/m1/overlap_m1.json 9b0c5ef3fbcb5d7b, log
  tu/m1/step0_20261021T133426Z.log 78670d250934be11. Nothing is removed, by the rule above.

  | Set (n_test) | C4 exact | C4 order-free | kept_F exact | kept_F order-free | kept_F reader key | Test texts matched (union) |
  |---|---|---|---|---|---|---|
  | CLINC150 (4,500) | 0 | 5 | 0 | 13 | 22 | 30 |
  | Banking77 (3,080) | 0 | 0 | 0 | 6 | 0 | 6 |
  | HWU64 (1,076) | 0 | 0 | 0 | 10 | 5 | 15 |

  - Three of a3_teacher.py's check sentences are evaluation texts:
    - "what is the weather like tomorrow" (MASSIVE en val, STAR dev);
    - "will it rain tomorrow" (MASSIVE en test);
    - "please turn off the lights in the kitchen" (HWU64 test).
  - All three are swapped, as described above.

## Test use, declared (project-wide)

**These test splits are not fresh.** A read-only census (tu/m1/census_test_reads_2026-10-21.md, c7e0cc1379d8d352)
lists every earlier read with its date and evidence file. It covered this study, the decision-layer study,
clause_lm and the research notes; the archived old working folders were not searched.

| Split | Earlier read events | Earlier test scorings (rows) |
|---|---|---|
| CLINC150 in-scope test | 19 | 329 |
| CLINC150 oos_test (M1 does not score it) | 16 | 244 |
| clinc_oos plus test (P1) | 1 | 25 |
| Banking77 test | 15 | 242 |
| HWU64 test | 13 | 202 |

- **Who read them.**
  - Most reads are the decision-layer study's registered runs, 2026-10-15/16: R2, R2b, R4, R4b, R5, R5b-d, R6, R7,
    R7b, R7c, R2c-e and R8, with their readability reports.
  - The rest: clause_lm M5 (CLINC150; 2026-10-02 and 2026-10-14), T2 (2026-10-17) and P1 (2026-10-18).
  - Decontamination filters also touched the test texts, without labels (the census lists them).
- **MASSIVE.** MASSIVE train holds 470 of the 1,076 HWU64 test texts. No M1 arm trains on MASSIVE. Whether bge-base,
  potion or Qwen3.5-9B saw MASSIVE in their own training is not checkable, so that is stated as a limitation.
- **What this means: adaptive reuse.**
  - M1's own choices (tables, s, arms, bars, seeds) come from val only.
  - But the pipeline was built in a project whose earlier steps read these test splits. T2 (2026-10-17) confirmed the
    existsT view on them, and every later step builds on existsT.
  - So M1's numbers are test-split numbers on splits read many times before, with no M1 choice made on them. They
    are not fresh-test evidence.
  - Every report says: "test splits reused across the project (census in tu/m1/)".
- **A fresh test.** It would need an intent set never read in this project. Whether one is held locally was not
  checked. That is a limitation, and a possible later step.
- **Reads in M1.** 7 TM configurations x 3 sets x 3 seeds = 63 TM runs, plus 6 LR arms x 3 sets = 18 LR fits. That is
  81 test scorings, each read once.
  - The code enforces one read per run: the predictions file is written exclusively, and a run whose file exists
    stops before training.
- **No reruns.** A run that fails before its test read may be rerun once (logged). A run that fails after it is
  reported as failed.

## Cost and order

- **GPU, about 100-110 min:**
  - features about 5 min;
  - C4, a3B and potion runs: 27 x about 35 s;
  - the 3c arms: 18 x about 120 s;
  - the twin arm: 9 x about 150 s;
  - a3B @5: 9 x about 140 s;
  - LR: 18 x about 20 s.
- **Queue.** After c3x and the speed block's first part (decided 13:10Z). The speed block's B3+B4 come after M1.
- **Code.** m1_feats.py, m1_tm.py, m1_lr.py, m1_verdict.py, m1_overlap.py, m1_standin.py, m1_check_cache.py and
  m1_chain.sh, each a declared copy of its c3x parent with its changes listed in its docstring.
  - Tests: tests/test_m1.py (with --feats for the CPU runs on stand-in data) and tests/test_m1_chain.sh (stubs: done,
    a failing step, TERM in a step, TERM while waiting, a refused pin, no step-0 file, dry run). Then a dry run.
- **The freeze.** The registration (this plan, final) with the hashes of every code and input file in tu/m1/pins.sh.
  The chain checks them before every step, as in c3x.
- **After the runs.** An independent recompute and a check for leakage, selection and integrity run before any
  number counts.

## Priors (honest)

- H1: about 85%. The val margins are large on every set.
- H2: about 50%. On val it holds on 2 sets by small margins. In T2, test ran below val for every view.
- H3a: about 35%. The val margins are narrow, and Banking77's depended on C.
- H3b: about 70%. On val (c3x) 3c beat the twin by LR +2.43 / +2.69 / +1.76, and EDA copies lowered LR on every set.
