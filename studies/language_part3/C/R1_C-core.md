# R1 [C-core v2]: registration of experiment 2 of Part 3 (the Part 3 plan, section 4)

Written 2026-10-21 (UTC), from 03:09:23Z (study-log start line) to the freeze; this published copy is edited in wording
only. Frozen by `scripts/freeze.py` into
`R1_C-core.frozen.json` (UTC time, SHA-256 of this text and of every tracked file). Binding together with the line
"R1 [C-core]" in the study log (not distributed). Paths are relative to `..` unless they start
with `C/` (= `studies/language_part3/C/`).

**Status at the freeze.** No stage of C has run on calib, valid, dev-test or test. Only self-tests (synthetic data, and
tiny pseudo splits made of whole train blocks) and dry runs of every stage on pseudo splits of train blocks ran
(section 12). No GPU was used. Draft history: the first full draft was saved at 2026-10-21T03:49:45Z (stat); it was
revised before the freeze (the final text is the one hashed in the frozen record); no rule changed after any number
of C on a real split was seen, because none exists. Pre-freeze review: `C/review/R1_checklist_review.md` (the
16-point registration checklist, applied as a self-review; verdict READY, one blocker found and fixed, the kn.py check
of section 4).

## 0. Sources (SHA-256, full)

- Plan: `the Part 3 plan (not distributed)` e21fb927b9f28e860df9ee309a16e17af75e6371eb0c6a19b49e4d23401a6f3f
  (section 4 and the review amendments of section 11 are binding; this text adds only implementation choices, listed
  in section 13).
- Proposal: `glassbox-lm/studies/language_part3/proposal_C-core.md` 914f4fd262736828f4d3fea859e9ca9ef0dfc99a46fc9c9ddf8b89e35439befa.
- Code: every file under `C/ccore/`, `C/scripts/`, `C/tests/` and the imported tally code (`src/tally/__init__.py`
  e3b0c442..., `ptb.py` ab635615bf9713efa93f66d8bd5ea3994c7168525aabfcd16d91baffc38245cf, `floor/kn.py`
  fcdf25e5b7f080737aac5295cb0c87993ce4bcaaaa72a14d411143ea2d3ccd80, `floor/cache.py`
  286109bc3147db4d6be49f21891a508e02630d886a56a25c36600d695819bc45, `floor/mix.py`
  29f05bb4c4edaec96aa15ecf590b78729aa8a336b0c5fa9dad8e748ddc631b8a, `floor/exchange.py`
  bb3157cec12d3de6c3c9005198046dec2b3b91a5aa84ad0f3dc1a5afa694f165, `floor/stats.py`
  64a4efd9f2a83ab94dd1e07611ac340fac66af29e4f3c67d6ffbe1eb1e005b05, `floor/aa.py`
  044022c4074d6355ffeb91ad4573cd0fff054b11bbf235dc62b41f0eabaf1f8c, `floor/classlm.py`
  792229578ebb0d9f19c5bb4744a93c986b6a9b68f56676b5fae82bbf0535bdce; the other floor files are hashed in the record).
  All of `glassbox-lm/studies/tally/src/tally/` is imported read only and never edited.
- Data (read only, nothing downloaded):
  - PTB (Mikolov) `glassbox-lm/studies/tm_language_model/data/ptb/raw/train/text.utf8`
    5145926136ee9aef6f359b267ac09cc8a920879cd71725de17c490dd111d2998 and `.../validation/text.utf8`
    fadf6277290823f881b7bf39b84e88536c87a859cd629dcfd1d2dfdef06097cf. The test file is not hashed: no code here opens it.
  - C4 counting table `glassbox-lm/studies/decision_layer/data/tables/c4words_S1000_v1/`: `words.txt`
    49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29, `vectors.npy`
    9263bf09383de22806e320a1988de57816dbd62bfa1cc692b0fb7077e44d267c, `MANIFEST.sha256`
    f49b934ad1fe8d520af9f87ef37f145aff5483f1c55068e2c5309747f8bd734e (`table.npz` e1771d9e... and `tokenize_ref.py`
    2c3c1aff... are not used: the mapping is a direct string lookup, section 3).
  - T0's frozen outputs: `bits.npz` and `metrics.json` of the 30 runs `glassbox-lm/studies/tally/results/
    *_t0_final_coal_c2000_s20_seed{1..10}`, `*_t0_final_flat_c100_s5_seed{1..10}` and
    `*_t0_logreg_of_final_coal_c2000_s20_seed{1..10}`; their 60 full hashes are in the frozen record.
- Recorded in the frozen record but not enforced at stage start (so an edit of the plan for another arm does not
  block C): the plan, proposal C, the CPU gate script, and the code for the later steps (read only): dl `src/dl/tm.py`
  f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424, `src/dl/cpu_tm_fast.py`
  bf1c4a9086d76448d9d77cf7775e0bb0e5f4bffc926d777b867f89148279423e. CPU gate script
  `glassbox-lm/studies/decision_layer/scripts/cpu_load.py` 2e300054a9284ef4bb0d395048fb5988d3176c100aa184384d22a3cc5eb04328.
- Environments: `.venv-tm` (python 3.11.16, numpy 2.4.6, numba 0.67.0, scipy 1.17.1, scikit-learn 1.9.1) for every
  counting stage; `.venv-dl` (numpy 2.4.6, scikit-learn 1.9.1, LightGBM 4.7.0) for LR and LightGBM.

## 1. Question and the comparison that carries the claim

Cut the TM's job down to the next word's class (65 classes) from 60 dense code bits, and let counts choose the word
inside the class. Is the TM class model then as good as a counted class model on the same codes, inside KN5 + cache,
on PTB? The claim-carrying comparison, fixed now: **S_TM minus S_twin** on the PTB test file (bar LEVEL, section 10).
One dataset, one headline; BEATS and ADDS are the only other named claims.

## 2. Data, splits and split uses

- PTB word level, Mikolov preprocessing, V = 10,000 including `<unk>` and `<eos>`; every sentence ends with `<eos>`,
  which is counted.
- The tally splits (`src/tally/ptb.py`, blocks of 100 sentences of the standard training file, permutation seed
  20261015): train 836,253 tokens (counts, codes, every model), calib 46,925 (every weight, read-out and early stop),
  valid 73,760 (the standard validation file: every choice, 8 looks per arm), dev-test 46,411 (only in Step 0).
- Test = the standard test file, read once at the end with every arm (step 10), by a frozen test script that does not
  exist yet. Earlier PTB test uses: one line in `glassbox-lm/studies/tm_language_model/data/test_access_log.jsonl`
  (2026-09-22T16:56:09Z, "M2 build: PTB raw store and tokenization"); no model scored on it was found. C's read would
  be the 2nd logged access. No other Part 3 registration uses PTB test (A reads the MultiWOZ dev split, B its own
  splits), so there is no either/or.
- Dev-test was read by T0 (10 seeds of each machine). Step 0 reads it again from T0's frozen outputs, exploratory.

## 3. Codes, inputs and targets (train only; `ccore/codes.py`, `ccore/layout.py`)

- **Tree E.** Exchange clustering of the train bigrams into 64 word classes (tally `floor/exchange.py` `tree()` with
  levels (64,): sentence mode, `<eos>` and BOS fixed in classes of their own, Martin et al. start, at most 20 passes,
  stop when a pass moves fewer than 0.1% of the words). Then a balanced binary merge tree over the 64 classes: at each
  of 6 levels the current clusters are paired greedily, always merging the unpaired pair with the least loss of
  class-bigram log-likelihood (losses recomputed after every merge; ties to the lowest index pair). Each class gets a
  6-bit path, top split first; the child with the lower lowest class id gets bit 0.
- **Tree B.** Vectors: a PTB word found as a string in `words.txt` gets its row; `N` gets the mean of the rows "0" to
  "9"; `<unk>` and every other word without a row get the train-count-weighted mean of the rows of the PTB words that
  have a row and are seen 1 to 5 times in train (proposal C's fixed rule); every vector is L2-normalised. Coverage on
  the train split (computed in the self-tests, train only): 9,039 types with a row (84.25% of train tokens), `N` (3.50%),
  959 types on the shared fallback vector (7.73%, `<unk>` among them), `<eos>` (4.53%). Recursive bisection to depth 8:
  each node is split at the train-count-weighted median of its words' projections on the node's first principal
  direction (weighted by train counts; uniform weights in a node whose words all have count 0); identical vectors get
  one projection and stay together; the cut is the boundary between distinct projections whose left weight is closest
  to half (ties to the first). `<eos>` gets no B path (it is never a previous word in sentence mode).
- **Inputs (60 bits).** For each previous word t-1 .. t-4 inside the sentence of t: E path (6 bits), B path (8 bits),
  pad flag; a position before the sentence start has pad 1 and all its code bits 0. Bit index (k - 1) x 15 + offset.
- **Target.** The exchange class of the token: 64 word classes + `<eos>` = 65 classes.
- **Emission.** e(w) = (n(w) + 0.5) / (n(c) + 0.5 |c|), n = train counts (tally `floor/classlm.py`'s rule).

## 4. Floor F (`ccore/floor.py`)

- KN5 (tally `floor/kn.py`, interpolated modified Kneser-Ney, sentence mode, counts from train).
- Decayed unigram cache (tally `floor/cache.py`): half-life 100, reset per 100-sentence block, `<eos>` not added (A1's
  valid choice, `results/20261015T072513Z_a1_baseline_check`); where the cache is empty, its member is KN5.
- Bucketed MAP-EM weights on calib (tally `floor/mix.py`: longest KN5 match x count bucket, 40 buckets, tau 20, prior
  = the global EM weights on calib).
- **Floor-code check (a stop rule).** The current `kn.py` is not the file A1 hashed (A1 used e40909db...; A2 and T0 used
  the current fcdf25e5...), and A1's KenLM match (3.84e-5 in relative perplexity) was made with the older file. Step 0
  and prep therefore assert that the current code reproduces A1's own valid numbers within 1e-9 bits per word: KN5
  7.255057512770932, KN3 7.34256972151113, KN5 + cache with the global calib weights 7.054522637538526. These numbers
  were known before R1. A mismatch stops the stage.

## 5. Members and systems (`ccore/systems.py`; the same for every arm X)

- factored member f_X(w | h) = P_X(c(w) | h) F(w | h) / F(c(w) | h); standard member s_X(w | h) = P_X(c(w) | h) e(w);
- system S_X = bucketed MAP-EM mixture of {F, f_X, s_X} on calib (F's 40 buckets, tau 20, prior = the global EM
  weights on calib);
- for ADDS: S_twin+X = the same EM over {F, f_twin, s_twin, f_X, s_X};
- a mixture of class models (the twin's chains) uses the same bucketed EM on the class likelihood on calib.

## 6. Arms and configuration lists (8 trials each, all selected by the lowest valid bpw of their system; ties to
the lower trial index)

| Arm | What | 8 trials |
|---|---|---|
| Count twin (claim-carrying comparator; `ccore/twin.py`, `ccore/chains.py`) | Bucketed-EM mixture on calib of counted class chains over the codes. T-E: interpolated modified KN on the class stream (tally `kn.py`, sentence mode, 65 classes, fallback discounts 0.5/1/1.5). T-B: (B8 t-1, B6 t-2, B4 t-3) -> (B8 t-1, B6 t-2) -> (B8 t-1) -> (B6 t-1) -> (). T-EB: (E6 t-1, B8 t-1, E6 t-2) -> (E6 t-1, B8 t-1) -> (E6 t-1, B6 t-1) -> (E6 t-1) -> (). T-B4 [review 7]: (B8 t-1..t-4) -> (t-1..t-3) -> (t-1, t-2) -> (t-1) -> (). T-EB2 [review 7]: (E6 and B8 t-1, E6 and B8 t-2) -> (E6 and B8 t-1, E6 t-2) -> (E6 and B8 t-1) -> (E6 t-1) -> (). These four: interpolated modified absolute discounting with raw counts, D1/D2/D3+ per level from its count-of-counts (fallback 0.5/1/1.5), the empty key interpolated with the uniform over 65 classes. A position before the sentence start has the PAD value (E6 64, B8 256, B6 64, B4 16). B6 = B8 >> 2, B4 = B8 >> 4. | TE3; TE4; TE5; TE4+TB+TEB; TE5+TB+TEB; TE5+TEB; T-ALL (TE4+TB+TEB+TB4+TEB2); T-ALL (TE5+TB+TEB+TB4+TEB2). With T-ALL at T-E order 5 every input bit is in a counted chain at full depth (self-test); TALL4 leaves the 6 E bits of t-4 out. |
| LR (gates via C-K0b; `ccore/learners.py`) | sklearn multinomial logistic regression, lbfgs, L2, intercept, on the one-hot tree nodes of the same bits: E at depths 2/4/6, B at depths 2/4/6/8, pad flag (425 per position, 1,700 in all, 28 on per row without pads). Fitted in ascending C with warm start, max_iter 200, tol 1e-4; convergence recorded per C. | C in {0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10} |
| LightGBM 4.7.0 (gates via C-K0b) | multiclass (65) on the 60 bits, at most 400 rounds, early stopping on calib (multi_logloss, patience 20), predictions at the best iteration; num_threads 2, deterministic, force_col_wise, seed 0, max_bin 255. | num_leaves {15, 63} x learning_rate {0.05, 0.2} x min_data_in_leaf {20, 200} |
| TM (GPU, step 9; code written later under an amendment) | our CoTM (`src/dl/tm.py` f649043f..., workspace code, not a CAIR library), 65 classes, the same 60 bits with negation (120 literals), 8,000 clauses, neg_sampling "random", batch 128, at most 20 epochs, other TMConfig defaults; its own training loop (fit() calls torch.cuda.synchronize). Read-out softmax(S / tau + b), tau and 65 biases fitted on calib by log-loss after each epoch; the epoch is chosen on calib by that class log-loss, per seed; valid scores only that checkpoint (8 looks in the grid). Grid at seed 1; the pick then runs seeds 2-10 (10 seeds); per-position bits averaged over seeds. | T {8,000, 16,000} x s {3, 10, 30, 90} |
| Tree twin, count leaves (step 10) | sklearn DecisionTreeClassifier (criterion entropy, random_state 0) on the 60 bits; each node's class distribution = (N_node + alpha P_parent) / (n_node + alpha) down the path from the root (root: P_parent = uniform over 65). | min_samples_leaf {20, 100, 500, 2,000} x alpha {1, 10} |
| Untrained banks (step 10) | Two kinds, 10 draws each (draw seeds 1-10): uniform (each clause gets as many plain and negated literals as the seed-1 TM clause in the same place, drawn uniformly without replacement) and data-anchored (the same lengths; plain literals from the 1-bits and negated literals from the 0-bits of a uniformly drawn train position, as many as it has). T0's count read-out: vote(j, k) = log((N_jk + a) / (n_j + 65 a)) counted on train, logits = sum over firing clauses / tau + b; a from {0.01, 0.1, 0.5, 1, 2, 5}, tau and b on calib (not a valid trial grid). Clause outputs from `cpu_tm_fast.py` (CPU). | none |
| C0 (step 9) | the picked TM setting trained on a seeded 1% of the train positions, the same epoch rule | 10 seeds |
| Early checkpoint (step 9) | each seed's state after epoch 1, with its own calib read-out | 10 seeds |
| Floor F and references | section 4; printed beside: A1's KenLM KN5 152.747 and KN3 162.299 (valid) | none |

## 7. Checks before any number is logged (each stops the stage)

- The contract (tally rule) on 1,000 fixed positions of calib and of valid (seeds 31, 32) for every system: the vector
  over V sums to 1 within 1e-6, its minimum is above 0, the fast path equals the vector entry (|log difference| <=
  1e-9). F's own contract also in Step 0 (seeds 21, 22, 23 on calib, valid, dev-test) and prep.
- At every calib and valid position: F's full vector sums to 1 within 1e-6 and is above 0 (in the class-marginal pass);
  every class row sums to 1 within 1e-6; the weights of every bucket sum to 1 and F's weight is above 0; the emission
  sums to 1 inside every class.
- Future-change test (with the self-tests, and again in prep on 6 cuts of calib, seed 33): changing the tokens at and
  after t leaves the inputs, buckets, F's vectors and T-E's vectors at t unchanged.
- PairKN equals tally KN on the true targets (T-E orders 3-5, calib and valid, max abs difference 1e-12) before it
  carries the shuffled T-E.
- The floor-code check of section 4; T0's arrays equal T0's logged means (|difference| <= 1e-9) and its block ids equal
  the tally splits' (Step 0).
- No neural-network library (torch, tensorflow, jax, transformers, sentence_transformers, model2vec) is imported in a
  CPU stage; the run fails otherwise.

## 8. Step 0 (free, exploratory, no claim, no gate; `ccore/step0.py`)

Each T0 arm is mixed with F per bucket (weights on calib): coalesced learned votes, counted read-out and the 10
untrained draws per seed, LR on the clauses, flat learned and flat counted; 10 seeds each. Scored on valid and dev-test;
per arm the per-position bits are averaged over seeds (the untrained arm over its 100 seed-draw systems) and compared
with F by the paired block bootstrap (1,000 resamples, seed 0, 95% percentile). Contrasts reported: learned minus
untrained, counted minus untrained, learned minus counted, LR minus learned. Every line is labelled exploratory.

## 9. Gates (CPU, valid, before any GPU job) and stop rules

Point estimates of valid bits per word of the selected systems; comparisons allow 1e-12 for float representation only
(`ccore/gates.py`).
- **C-K0 (sanity; review 13):** bpw(F) - bpw(best twin system) >= 0.0439 (3% perplexity). A stop points to a bug or
  poor codes: no TM run, no test read.
- **C-K0b:** min(bpw(best LR system), bpw(best LightGBM system)) - bpw(best twin system) <= 0.0072, else STOP.
- **C-A/A (review 6):** the best twin trial rebuilt on 95% of the train blocks (tally `floor/aa.py`
  `subsample_train`, seed 1): its chains, its T-E and the emission are recounted; F and the codes stay fixed (they
  are shared by every arm); weights refitted on calib. If |bpw(rebuilt) - bpw(twin)| > 0.0036, STOP. The interval is
  logged (no rule), before any TM run.
- **C-shuffled (review 6):** every learned class model retrained on class targets permuted across its train positions
  must give a system no more than 0.0010 bpw below F, else STOP and find the leak: the twin (the selected trial, all
  train positions, permutation seed 1, T-E through PairKN on the true contexts); LR and LightGBM (their selected
  settings, a seeded 10% of the train blocks, subset seed 0, permutation seed 1); the tree twin (step 10, its selected
  setting, all train positions) and the TM at smoke scale (step 9) later, by the same rule. The class-unigram system
  (one class distribution for every position) is reported beside it as a diagnostic, with no rule.
- Order of the reported stop reason: shuffled, A/A, C-K0, C-K0b (all four are reported whatever the order).
- **C-K1 (after the TM grid, valid):** bpw(best TM system) - bpw(best twin system) > 0.0218 -> STOP, "FAIL at K1
  (exploratory)", no test read.
- **C-K2 (smoke: 10% of the train blocks, subset seed 1, 2 epochs, T 8,000, s 10, TM seed 1; logged, not a result):** stop and
  diagnose if more than 50% of clauses are empty, the median clause has more than 30 literals, more than 50% of
  clauses have no positive literal, or the calibrated valid class bits are worse than the class unigram's. A fix is a
  dated amendment.
- **C-K3 (review 10):** measured GPU cost more than twice the 50-115 GPU-minute estimate, or no GPU slot within the
  time box: C's claim waits, or C stops. The CPU path cannot carry the claim; the only CPU option is exploratory (4
  trials x 8 epochs, 1 seed, valid only, settings fixed in an amendment first); the smoke times one CPU epoch first.
- **C-K4 (review 10):** no C-K0/C-K0b decision within 72 gate-open hours of starting the code (2026-10-21T03:09:23Z)
  or by 2026-10-31T03:09:23Z, whichever comes first: STOP and report. C's runner logs every gate check
  (`C/runs/gate_log.tsv`) and every job (`C/runs/jobs_log.tsv`); gate-open time with no C job queued is the
  scheduler's record.

## 10. Bars on the PTB test file (step 10), statistic and wording

- Metric: system bits per word on the test file with `<eos>` counted; perplexity and token counts beside it.
- Statistic (registered): per-position bits of every multi-seed arm averaged over its seeds or draws (bits, never
  probabilities); paired block bootstrap over 100-sentence test blocks (file order), 1,000 resamples, seed 0, 95%
  percentile interval (tally `stats.block_bootstrap`). "upper" = its upper end. Per-seed results are printed. Beside
  every pair with a multi-seed arm: the seed-inclusive two-level bootstrap (`ccore/boot.py`: seeds resampled within each
  arm, then blocks; 1,000 resamples, seed 0), reported, no rule.
- Before any test number is written, the test script asserts at every test position that the true word gets a
  probability above 0 and that the vector over V sums to 1 within 1e-6 (in chunks).
- **LEVEL** = P1 and P2 and P3 (`gates.level`):
  - P1: upper(S_TM - S_twin) < +0.0072 and every seed's point estimate of S_TM(seed) - S_twin < +0.0072;
  - P2: upper(S_TM - S_uniform), upper(S_TM - S_anchored), upper(S_TM - S_C0), upper(S_TM - S_epoch1) all < -0.0072,
    upper(S_TM - S_tree) < 0, and at least one untrained bank non-vacuous: its clauses fire (any clause) on at least
    1% of valid positions and its two members' EM weight, averaged over valid positions, is at least 0.01;
  - P3: upper(S_TM - F) < -0.0291.
- **BEATS:** LEVEL, upper(S_TM - S_twin) < -0.0145, and mean bpw(S_TM) <= mean bpw(S_LightGBM).
- **ADDS:** upper(S_twin+TM - S_twin) < -0.0072; upper(S_twin+TM - S_twin+uniform) < -0.0036 and
  upper(S_twin+TM - S_twin+anchored) < -0.0036; upper(S_twin+TM - S_twin+tree) < -0.0036; S_twin+LR and
  S_twin+LightGBM printed beside it.
- FAIL: anything else. No multiplicity correction is applied; each rule stands alone. The claims are LEVEL, BEATS and
  ADDS; P2's conditions are all required (no "k of n").
- Reported with no rule: class bits per token on valid and test for every arm; the standalone class LM s_X in bpw;
  rare-context split (longest KN5 match <= 2); the duplicate share (test tokens inside a train match of 8 or more
  tokens) and every arm's result with them removed; clause-length histograms; LR and LightGBM beside every arm; the
  valid boot_sd of S_LR - S_twin and S_LightGBM - S_twin (a power reading for P1).
- Wording, fixed: LEVEL reads "level with a counted class twin over the same codes", never "beats counting"; every
  line names "our CoTM (dl/tm.py, workspace code, not a CAIR library)"; never "first", never "a TM writes language";
  where LR or LightGBM are ahead: "best gradient-free learner here, x behind logistic regression". Provenance: no
  neural network at any stage (counts, PPMI-SVD vectors from C4, trees, LR, LightGBM, a Tsetlin machine); T0's
  outputs in Step 0 come from CAIR's tmu library. Selection is optimistic: every valid number is the best of 8.
- If C reaches LEVEL or BEATS, the replication adds a twin trial with generalized parallel back-off over all 8 code
  factors (FLM-GPB) before any write-up (review 7).
- Line R2 (before the test read): the picks, the frozen best counting system, the row hashes, and the test script's hash.

## 11. Runs and resources

- Order (the Part 3 plan, section 6): Step 0 (step 2) -> prep, twin, learners, gates (step 6: C-K0, C-K0b) -> GPU TM (step 9, only
  if both gates pass) -> tree twin, banks, R2, test (step 10). Commands in `C/README.md`.
- Every job through `C/scripts/run_stage.sh`: `python -I -B`, nice 19, OMP/MKL/OPENBLAS/NUMBA/RAYON threads 2 (and
  threadpoolctl 2; LightGBM num_threads 2; no process pools), CUDA_VISIBLE_DEVICES empty, NUMBA_CACHE_DIR, TMPDIR,
  JOBLIB_TEMP_FOLDER, MPLCONFIGDIR and XDG_CACHE_HOME inside `C/tmp/`, the imported folders listed before and after
  with `ls -la --time-style=full-iso` and compared (exit 4 on any change). Real stages wait for the CPU gate
  (`cpu_load.py 10`: at most 4.0 cores and 0 throttled periods, checked every 2 minutes). One Part 3 CPU job at a time
  is the scheduler's rule.
- A real stage refuses to start without the frozen record, when a tracked file changed or disappeared since the freeze,
  when a finished real run of the same stage exists, or when an earlier stage's output differs from what its run wrote.
  A crashed stage may be rerun unchanged (deterministic code); any change is a dated amendment.
- Estimated CPU at 2 threads: Step 0 a few minutes; prep 2-5 min; twin 5-15 min; learners 2-6 h (LR about 1-1.5 h;
  LightGBM about 1-4 h, measured on synthetic rows at about 0.8-1 s per round at 80,000 rows); gates 1-2 min. LightGBM
  may exceed the plan's 1-3 h line; the plan's 6-10 h total holds.

## 12. What was known at the freeze

- Valid numbers on these splits (A1, `results/20261015T072513Z_a1_baseline_check`): own KN5 152.753 ppl (7.255058
  bpw), KN3 162.306, KN5 + cache with block reset (global weights) 132.930 (7.054523); KenLM KN5 152.747, KN3 162.299;
  6.43% of valid tokens inside train matches of 8 or more tokens (KN5 195.07 without them); 34 bootstrap blocks, SD
  0.0142 for KN5 + cache minus KN5.
- A2 smoke (88,702 train and 9,156 valid tokens): exchange classes 5.15% below KN5 + cache (a smoke, not these sizes).
- T0 (valid, 10 seeds; the Part 3 plan, section 9): learned votes 9.0266-9.0986 bpw, counted read-out 9.3262-9.3880, untrained
  banks 9.2173-9.2525, LR on its clauses 8.2284-8.2479; flat machine (seed 1) learned 10.634, counted 9.790; its
  dev-test numbers are in its metrics files. T0's mixture with F (Step 0) is not known.
- Published PTB test (Mikolov et al. 2011, Table 1, via the Part 3 plan, section 9): KN5 141.2, KN5 + cache 125.7, random
  clusterings class LM + KN5 + cache 115.6. Local E3 (logical-llm): KN-3 161.72 -> 157.24 with k-means-200 class clauses.
- From this work: C4 coverage of the train vocabulary (section 3; train only); LR and LightGBM timings on synthetic
  rows; dry-run numbers on pseudo splits of train blocks only (meaningless for C's question). No number of C's floor
  with bucketed weights, of any twin trial, LR or LightGBM on calib, valid, dev-test or test is known.
- What the known numbers predict: C-K0 passes for any working class model (Mikolov's class LM gave 8% on test; the
  A2 smoke 5.15%); C-K0b has no prediction (G1's LR and trees beat its count twin, but on count facts, not codes);
  the PLAN's priors stand (LEVEL about 0.04, ADDS about 0.08, BEATS about 0.02; the most likely result is a stop at
  C-K0b or C-K1).

## 13. Implementation choices this text adds to the plan (fixed before any run)

1. Twin chain mixtures use F's buckets (longest KN5 match x count; 40, tau 20), like every other mixture here.
2. C-A/A rule as the plan words it for C (mean outside +-0.0036 stops; the interval is logged, no rule); F and the
   codes are not rebuilt.
3. Shuffled labels: the twin on all train positions (T-E through PairKN, which equals tally KN on true targets); LR and
   LightGBM on 10% of the train blocks; the class-unigram system as a diagnostic beside the rule.
4. LR: warm start along ascending C, max_iter 200, tol 1e-4. LightGBM: patience 20, the base parameters above.
5. Tree E's balanced pairing and bit rule; tree B's weighted PCA, tie rule and normalisation; the PAD values.
6. The non-vacuous rule of P2 is read as "any clause fires on >= 1% of valid positions" and "the bank's f and s
   weights, averaged over valid positions, >= 0.01".
7. Step 0 checks F's contract; T0's vectors over V were checked by T0's own contract.
8. The floor-code check against A1 (section 4).
9. K2's smoke setting (T 8,000, s 10, seed 1).

Not covered by this line: the code of steps 9-10 (TM loop, read-out, smoke, C0, epoch-1, tree twin, banks, test
script). It is written later and hashed in a dated amendment line before it runs; the grids, rules and bars above stay
binding and cannot change after any valid number of C is seen.
