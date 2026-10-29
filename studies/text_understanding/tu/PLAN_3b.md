# Step 3b: a TM student of the teacher's bits (plan; CLOSED 2026-10-21 11:34Z as a NEGATIVE result at P0-T: the TM
# student stayed near chance, 52.3 balanced at best, against ridge 63.7; never registered, no GPU run)

Step 3b of the text-understanding plan ("a TM student of the teacher's bits"), the next step after 3a and 3c.
Drafted 2026-10-21 from 07:12Z, with three changes of 07:15Z (the no-teacher control 3b-I, the teacher numerics, the
+0.3 wording) built in. It was to be registered only after the CPU pilot P0 (section "Tests before the GPU job"),
and its GPU job was to run at its turn in the queue, after 3c, decision_layer's e81 and e81x, and the speed block.

## Question

Can a multi-output Tsetlin machine, trained on unlabeled C4 sentences to reproduce a frozen sentence encoder's bits
from word-concept bits, give the TM reader bits that help on intents on top of the current best stack, and help more
than a linear student of the same bits?

**How to say it, in every report:**
- Provenance: "network-free at run time, teacher (bge-base-en-v1.5) at training time". At run time a 3b classifier is
  word-table lookups, concept bits, the student's clauses and the TM reader: lookups, bits and TMs only.
- The ceiling arm (the teacher's own bits, below) runs the teacher at run time. It is descriptive and is labelled
  "a neural network at run time" in every table.
- Novelty, from N2 (2026-10-17) and its dated update (novelty/n2b_2026-10-21/report.md, searched 2026-10-21 07:28
  UTC): "we found no published study that trains TM clauses, or any logic or rule model, to predict a transformer
  sentence encoder's binarised embedding bits for text, and none that then drops the transformer and feeds the
  predicted bits to a TM classifier". Never "first".
- The direct precedent, cited in every report: CRISP (Chan, Chowdhury, Erkus, Granmo, Yakovlev, Shafik, arXiv
  2610.02431, 1 Oct 2026, accepted at ISTM 2026). It "reconstructs the last-layer activation vector (LLAV) of binary
  neural teachers as Tsetlin Machine (TM) clauses" (abstract, checked 2026-10-21): one TM per sign-binarised hidden
  unit, on images (MNIST to CIFAR10), with binary-network teachers, 256 separate TMs, and the teacher's own head on
  the predicted bits. So 3b's design of one student output per teacher bit is NOT new. What 3b adds is narrower: text,
  a transformer sentence-encoder teacher, an unlabeled corpus, concept-bit inputs, one shared clause pool for all
  outputs, a run time with no network (a TM reader instead of the teacher's head), and the linear, counting and
  no-teacher controls.
- Other closest work: Self-Taught Hashing (SIGIR 2010: one linear classifier per code bit); "Clusters are All You
  Need" (arXiv 2606.19815: a TM learns BERT cluster ids, one 200-way label, BERT still assigns clusters); Kinateder
  2025 and its journal version (TM-to-TM distillation); Gao et al., Findings of ACL 2026 (an LLM writes synthetic
  training data for a TM: labels, not teacher bits); Tokenlearn / Model2Vec (a lookup student, no logic).
- TM word embeddings are different prior art: Bhattarai et al. 2023's TM autoencoder (our T5: no better than
  counting) learns word codes from raw text without a teacher; Yadav et al. 2021 add "some word means X" bits from
  pretrained word vectors. Neither has a sentence-level teacher target.
- Hand checks are "judged by an AI model, not a person".

## Base stack (R2), fixed now

- **If 3c FAILS:** the base is the a3B table (tu/a3/tables/a3B_d768.npz) with existsT at R3's picks, s 8 / 8 / 16
  (tu/r3/r3_picks.json). The reference TM rows are R3's seed rows (tu/r3/results_r3_val.jsonl, tag r3-seeds, seeds
  4-6), the LR reference is a3's a3B row (tu/a3/results_a3_lr.jsonl).
- **If 3c PASSES:** the base is a3B plus 3c-F's kept paraphrases, after the R3 re-tune that PLAN_3c.md runs on a PASS,
  with that re-tune's pick and rows. The student's bits are computed for the paraphrases too (they are texts).

## Teacher, texts and the leakage record

- **Teacher:** bge-base-en-v1.5 at revision a5beb1e3 (3a's teacher; tu/a3_teacher.py 7539e26231b77a17;
  [CLS], L2-normalised, at most 128 tokens, fp16 on the GPU; its verify_files() checks the model files).
- **Training texts, reused from 3a, by hash:** 1,999,932 C4 sentences (tu/a3/c4_sent_train_f.txt
  00481f8a27d5a08f) and a held-out set of 39,995 (tu/a3/c4_sent_held_f.txt
  3207323aca2a59e9), with their teacher vectors (tu/a3/T_train.f16.npy
  e180c79f7d08830e, tu/a3/T_held.f16.npy 91dcfc74f842c476; tu/a3/teacher_meta.json
  1e1d1fdb9479ce0b). No new C4 text and no new teacher pass over C4.
- **The filter's earlier reads, declared (as e82 did).** Building these sentences read the TEXTS, never the labels,
  of evaluation splits, only to drop C4 sentences that copy them (tu/a3/sentences_meta.json
  308692bd3754a652; of the 21,575,699 sentences of the pool, 228,122 were dropped by key and 3,912 by 8-gram):
  - every split of CLINC150, Banking77 and HWU64 (dl.data, including val and test), MASSIVE en-US and nb-NO,
    CLINC OOS-plus, TREC, SST-2, MR, R52 and R8;
  - the ABCD and STAR dev and test examples;
  - then tu/a3_mwoz_filter.py (a95146874ebe8c6c) read the MultiWOZ 2.2 dev and test utterances and
    dropped 68 training and 5 held-out sentences more (tu/a3/mwoz_filter_meta.json
    9b5d5ca8a58cf9a3).
  - So the 2M texts exclude every val and test text of the intent sets and every STAR, ABCD and MultiWOZ dev and test
    text, by key and by 8-gram. 3b's training side reads nothing else of an evaluation split.
- **New reads in 3b:** the teacher encodes the intent sets' TRAINING texts (for the target-bit choice below, on a
  carved val of the training split) and, for the descriptive ceiling arm only, their val texts. Test is never read.

## Teacher numerics: the device and dtype of every vector set (change 2 of 07:15Z)

- C4 targets: T_train and T_held were made by 3a on the GPU in fp16 (teacher_meta.json: "float16 on cuda"; its own
  check gave a minimum cosine of 0.99993 between fp16 on the GPU and fp32 on the CPU on 64 sentences).
- The code choice: the intent sets' TRAINING texts are encoded on the CPU, computed in fp32 and stored as fp16 by
  a3_teacher.encode (2 threads, nice 19), with the pinned revision and verify_files().
- The ceiling arm: the intent sets' training and val texts on the CPU in fp32, the same path as the code choice.
- Check, before the GPU job and without the GPU: the first 500 sentences of c4_sent_held_f.txt are encoded on the CPU
  in fp32 and compared with their fp16 GPU rows of T_held. Reported: the minimum and mean cosine, and the code-bit
  disagreement rate of each candidate code (SH, PT, ITQ). The student learns fp16 targets while the ceiling arm
  reads fp32 bits, so this rate is the size of that mismatch. If it is above 1% of the bits for the chosen code, the
  report says so beside the ceiling arm.
- RESULT (2026-10-21 07:35:40Z, tu/b3/numerics.json): minimum cosine 0.999998, mean 1.0; bit disagreement SH 0.037%,
  PT 0.029%, ITQ 0.039% of the bits. The mismatch is negligible, so the two paths are interchangeable for the codes.

## Target bits (fixed on C4 and on a carved val of the training splits only)

- Three candidate codes, each 768 bits (the teacher's width), each fit on T_train only:
  - **SH:** the signs of 768 random hyperplanes on centred vectors, x - mu (SimHash, Charikar 2002), Gaussian,
    numpy default_rng(20261025);
  - **PT:** PCA of T_train (centred; the covariance over all rows, eigenvector signs fixed so that each one's largest
    entry is positive), the top 256 axes, a thermometer at the 25 / 50 / 75% quantiles of each axis on a fixed
    subsample of 200,000 rows (numpy default_rng(20261026));
  - **ITQ:** all 768 PCA axes, then the ITQ rotation (Gong and Lazebnik 2011) fit on the same subsample: a random
    orthogonal start from numpy default_rng(20261027), 50 rounds, signs.
  - Code: tu/b3_targets.py (fit, numerics, choose); tests on synthetic vectors in tu/tests/test_b3_targets.py.
- **Choice (before any student is trained):** the teacher-bits probe, logistic regression (C 1 / 10 / 100) fit on a
  carved 90% of each intent set's TRAINING split and scored on the other 10% (a numpy default_rng(20261025)
  permutation per set). The code with the best mean carved-val accuracy over the 3 sets is taken; within 0.1 points
  SH is taken (the simplest). The official val is not read for this. The result is logged before the GPU job.
- The chosen code sets the ceiling. Its probes on the official val (TM at the base's s, seeds 4-6, and LR) are
  reported beside the students, as the descriptive ceiling arm.
- RESULT (2026-10-21 07:43:33Z, tu/b3/choice.json): mean carved-val LR accuracy PT 93.67, SH 93.44, ITQ 93.255, so
  the code is **PT** (0.23 above SH, outside the 0.1 band). Per set, SH / PT / ITQ: CLINC150 98.27 / 98.07 / 98.27,
  Banking77 92.67 / 93.00 / 91.89, HWU64 89.39 / 89.94 / 89.61 (carved TRAINING data; the official val was not read).

## Student input: one set of concept bits for every text

- **C4 concepts:** the distinct tokens of the C4 training sentences under the a3B table (w1/wordtab_feats.py's
  tokens), spherical k-means with K 512 (25 iterations, generator seed 0), exists_tm.features' clustering.
- **Bits:** the reader's existsT rule (exists2_tm.py, kept in c3_exists_tm.py): per concept, the max cosine of a
  text's tokens to the concept, binarised by dl.boolean.Thermo(8), i.e. the 7 quantiles of those max cosines, here
  over the first 200,000 C4 training sentences (the file's order is 3a's random sample order). On-share about 0.5, as
  the reader's existsT (0.52 on CLINC150). Code: tu/b3_c4_feats.py.
- They are computed once for the C4 sentences and for every intent text (train, val; paraphrases if 3c passed).
- They are NOT the reader's existsT bits, which use each set's own concepts. One student serves all three sets.
- K 512 keeps the input size known (about 3,500 bits). A larger K is a later variation, not part of 3b.

## Students (all on the same input bits, the same 1,999,932 sentences and the same target bits)

- **TM student** (new tu/b3_student.py on dl/tm.py's CoTM, read only and unchanged): 768 outputs over the C4 concept
  bits, one shared pool of clauses with an integer weight per (clause, output).
  - Rule: R feedback calls per minibatch. In each call, every example draws one output j uniformly; it gets
    target-class feedback for j when its target bit j is 1, and negative-class feedback for j when it is 0
    (CoTM.feedback with pos = neg = j and complementary masks). This is the CoTM rule for 768 binary outputs on one
    clause pool. A predicted bit is S_j > 0.
  - Settings (clauses, T, s, R, epochs, batch) are fixed by P0 and by a dev check on C4 held-out only, never on an
    intent split. The starting point is the reader's: 4,000 clauses, T 4,000, batch 1,024, s in {4, 8, 16}, R 4.
  - Metrics on the C4 held-out set: the mean over outputs of the balanced bit accuracy; the fidelity (CRISP's measure,
    the bitwise agreement with the teacher's code), reported beside it for every student; for SH codes also the
    implied cosine, cos(pi x Hamming / 768).
- **Linear student (twin):** one ridge (least-squares) classifier per output bit on +1 / -1 targets with an
  intercept, in closed form from streamed sums over all training rows (exact and deterministic at 2M rows, with no
  optimiser settings); a predicted bit is x W + b > 0; lam = f x n with f from {1e-4, 1e-3, 1e-2, 1e-1}, chosen on C4
  held-out only (tu/b3_twins.py).
- **Counting student (twin):** m_ij = the mean of target bit j over the training sentences in which input bit i is on;
  a text's score_j is the mean of m_ij over its on bits; bit j is on when score_j > c_j, where c_j makes the on-rate
  on C4 train equal to the target's. Counts only, no learning; "no neural network at any stage" except the teacher's
  targets, so it is also "teacher at training time".

## Arms and readers (intent val; the test split is never read)

- base (the reference rows), base + TM-student bits (3b-T, the registered arm), base + linear-student bits (3b-L),
  base + the student's INPUT bits with no student (3b-I: the C4 concept existsT bits; the no-teacher control),
  base + counting-student bits (3b-C, descriptive); the ceiling arm base + teacher bits (3b-ceil, descriptive, a
  network at run time).
- 3b-I is in the bar: without it, a gain of 3b-T could come from a second set of concept bits and not from the
  teacher (change 1 of 07:15Z).
- Reader: b3_exists_tm.py, the 3c reader (c3_exists_tm.py) with one change: a bit block appended after existsT. With
  an empty block its rows must equal the reference rows (the code check, as in 3c). 4,000 clauses, T 4,000, 40 epochs,
  batch 128, val at the best epoch, at the base's s, seeds 4-6. The existsT thermometer stays fit on the original
  training rows.
- LR: c3_lr.py's path with the same one change (C 1 / 10 / 100, best on val).

## Bar (fixed now)

- **PASS:** on at least 2 of the 3 sets, BOTH of these hold with BOTH probes (TM: the mean of the seed-paired
  differences over seeds 4-6; LR: val):
  1. 3b-T minus base is at least +0.5 (R2's bar);
  2. 3b-T minus 3b-L is at least +0.3 (the TM must beat a linear student of the same bits);
  3. 3b-T minus 3b-I is at least +0.3 (the teacher must add to the student's own input bits).
  - At R3's seed spread (seed SD 0.27-0.37 points) the standard error of a difference of two 3-seed means is
    0.22-0.30 points, so 0.3 is about one standard error. It is a minimum margin for a val gate, not a significance
    test; a registered test decides later (R7). The report gives the seed SD seen in the 3b rows and the SE of each
    paired difference next to it.
  - Unrounded gains are compared with 0.5 - 1e-9 and 0.3 - 1e-9.
- **Otherwise FAIL.** When part 1 holds and part 2 does not, the report says that the gain comes from re-encoding the
  bits toward the teacher, which a linear student does as well. When part 1 holds and part 3 does not, it says that
  the gain comes from the second set of concept bits, not from the teacher. In both cases 3b's question is answered no.
- **Descriptive:** 3b-L's and 3b-C's own gains over base; the ceiling arm; each student's C4 held-out metrics; the
  on-share of each predicted block; the share of the gap to potion that is closed; training seconds; 20 C4 held-out
  sentences with the TM student's most and least teacher-like codes.
- **On PASS:** R3 (re-tune the reader on the new stack), R4 (the decision_layer agent on STAR, ABCD and MultiWOZ dev at its
  tuned s, no dataset more than -0.5; a separate frozen plan in that study), R5 (the student's bits enter the guard domain and
  the guard proofs are run again), before any test split is read (R7).

## Tests before the GPU job

- **P0, the CPU pilot** (dev, C4 only; torch on the CPU with 2 threads, nice 19; CPU gate at most 3.5 cores until
  3c's generation ends, then at most 4.0):
  - 100,000 C4 training sentences and 10,000 held-out ones, the chosen code (PT);
  - the TM student and both twins on 64 of its 768 bits, fixed by numpy default_rng(20261031).choice(768, 64,
    replace=False). A feedback call costs 1.2 s on 2 CPU threads at 1,000 clauses whatever the number of outputs (measured
    07:48Z), and each call trains one output per row. All 768 outputs would get about 1,560 row-feedbacks each on the
    CPU; 64 outputs get about 18,750. The GPU job's 2M rows give each of the 768 outputs about 104,000 (R 4, 10 epochs),
    so the 64-bit pilot is the conservative reading;
  - the TM student at 1,000 clauses, T 1,000, R 4, batch 1,024, for 3 epochs, s in {4, 8, 16}, against the linear
    and counting twins at the same size;
  - P0's C4 concepts are fit on the CPU, whose torch generator stream differs from CUDA's, so they are not the GPU
    job's concepts: P0 measures feasibility only;
  - GO for the GPU job when the best TM student's mean balanced bit accuracy on the held-out sentences is at most
    2.0 points below the linear twin's, or above it. The GPU run has 20 times the data and 4 times the clauses, so a
    small deficit may close; a larger one is unlikely to turn into +0.3 on intents. Otherwise the plan is revised
    before any GPU time.
- **P0 RESULT (2026-10-21 09:13:43Z, tu/b3/p0/p0_verdict.json): NO-GO.** On the 64 bits, the TM student at T = clauses
  = 1,000 reaches a mean balanced bit accuracy of 52.15 / 52.33 / 52.12 at s 4 / 8 / 16 (flat over the 3 epochs). Ridge
  reaches 63.66 and counting 58.65. Best TM minus ridge: -11.33, against the GO line of -2.0. No GPU time.
- **P0-T, the one declared revision round (decided 09:42Z; post hoc after the P0 NO-GO, dev only, C4
  only; its own frozen note, tu/b3/NOTE_2026-10-21_P0T.md):**
  - T in {22, 60, 200}, with T 1,000 read from P0's s 8 row. 22 is about sqrt(C/2), the rule of Tarasyuk et al.
    (Pattern Recognition 174, 113028, 2026, quoting their ISTM 2023 paper), which is for the per-class vanilla TM, so
    using C = 1,000 shared clauses extrapolates it. 200 is the planted check's T/C of 0.15.
  - Everything else is P0's s 8 setting. The pick is by C4 held-out balanced accuracy, and the GO rule is unchanged.
  - If still NO-GO: 3b closes as a negative result, with no GPU and no further knobs without a new plan.
  - If GO: the 333-clause capacity run uses the same T/C ratio, P0b reruns at the chosen T, and the GPU job takes that T
    by a dated note before the freeze.
- **P0b, descriptive (a note of 07:32Z; not part of the GO rule; its first run used T = C and is
  probably uninformative):** the shared pool is one of 3b's
  claimed differences from CRISP, so on 32 code bits fixed by numpy default_rng(20261029).choice(768, 32,
  replace=False), at P0's best s and matched total clauses, one shared pool (1,024 clauses, T 1,024, R 4) is compared
  with 32 CRISP-style per-bit TMs (32 clauses and T 32 each, R 1) on the same rows and epochs (b3_p0.py --pool-check).
  Feedback is not matched: a per-bit TM gets feedback on every row. If the per-bit TMs are clearly better, the
  shared-pool choice gets a stated reason, or is changed, before the freeze.
- **Code checks:**
  - b3_student.py on planted targets (each output an AND or an OR of 2 input bits): at least 99% held-out bit accuracy;
  - the twins on the same planted targets;
  - the C4 concept features: the same code run on an intent set's own training tokens must give the cached a3B
    features bit for bit (tu/a3/feats/a3B_d768/; the largest absolute difference is reported);
  - b3_exists_tm.py and b3_lr.py with an empty block: rows equal to the reference rows on the same features and seed.

## Budget (one big GPU job at a time; CPU gate cpu_load.py at most 4.5 cores)

1. Teacher on the intent sets' training and val texts (about 42,000 short texts): about 1-2 minutes.
2. C4 concept bits for 2.04M sentences (k-means plus pooling): about 10-20 minutes.
3. TM student: about 2M examples per epoch. W1's multi-output TM ran about 7,200 examples/s at 20,000 features and
   8,585 outputs; here about 3,500 features and 768 outputs, with R 4 calls: an assumed 3,000-6,000 examples/s, so
   about 6-11 minutes per epoch, 1-2 hours for 10 epochs. Measured on the first epoch: if the projected training time
   is over 3 hours, the job stops and the plan is revised.
4. Twins: about 10-20 minutes.
5. Readers: 3 sets x 5 arms (3b-T, 3b-L, 3b-I, 3b-C and the ceiling) x 3 seeds = 45 TM runs, about 50-75 minutes;
   LR about 10 minutes.

In all, about 2.5-4 GPU-hours. If the speed block accepts the faster TM code (FastCoTM, identical
results), this plan may select it by file hash in a dated note; the saving is not counted until the GPU check runs.

## Prior

About 10-15% (the main plan). W1's multi-output TM lost to counting on a word-table job, and our T5 found Bhattarai et
al.'s TM autoencoder no better than counting. What differs here: a dense teacher target for every sentence, instead
of self-supervision on raw text, and a linear twin that makes the TM's own contribution visible.

## Newest work (searched 2026-10-21 07:28 UTC; report kept outside the repository)

- Distillation into TMs: CRISP (above) is the new direct precedent [read]. Kinateder, Anjum, Zhan, "Distribution-
  enhanced knowledge distillation", Neurocomputing 685 (2026) 133605: metadata read, content unverified (HTTP 403);
  probably the journal version of arXiv 2504.01798 (TM to TM). Gao et al., Findings of ACL 2026, pp. 30210-30222
  [abstract]. "Clusters are All You Need": still v1, no venue, 0 citations in OpenAlex.
- Recurrent TMs (for the later token variant): Xu, Duan, Shafik, Yakovlev, ISTM 2025 (clause outputs of earlier steps
  as the state; no state target in the abstract) [abstract]; Kumar et al., ISTM 2026 (the state has no target; a
  no-feedback control does as well) [abstract]; an ISTM 2026 poster "A Recurrent Multi-Output Tsetlin Machine for
  Sequential Adaptive Sparse Sampling" [title only]. No per-step teacher target for a TM state found.
- Multi-output TMs and TM embeddings: FastOmniTMAE (arXiv 2605.06982, 2026), Omni TM-AE (2025), the multi-phase TM word
  embedding (ISTM 2025), all without a teacher [abstract].
- Binary-code students of text encoders: neural students only (BitNet text embeddings 2026, SemHash-LLM 2026); a
  non-neural student of a text encoder's bits: not found.
- Static-embedding baselines [read, MTEB result files]: Banking77 test accuracy potion-base-32M 0.7478, potion-base-8M
  0.7264, bge-base-en-v1.5 (the teacher) 0.8695; MTEB(eng) classification average potion-base-32M 71.70 against
  all-MiniLM-L6-v2 69.25. No newer English static model with a higher published classification average was found.
  These are MTEB's test splits and settings, not our val splits; they are context, not our bar.
- Not checked: the Neurocomputing and Xu et al. full texts (paywalls), three ISTM 2026 papers that are not online.
