# e83: the TM's learning dynamics on MultiWOZ: the negative-class rule, clause dropout, a literal budget (dev only)

Draft rev 1, 2026-10-21 (started 13:35 UTC). Not frozen. No e83 number exists yet. It is reviewed before any
freeze.
- **GPU:** free at the freeze; e83 runs before step M1 of the text-understanding line (after the review at 14:4x UTC,
  rev 1 57056086 was kept, with the one line added under the pick).
- **Data:** no test split is read. Every job scores on dev.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**The question (fixed now):** does the negative-class rule, clause dropout or a literal budget lift the refined
agent's TM reader on MultiWOZ dev? There the TM reads the same bits 5.35 points worse than LR.

**Why now:**
- e81 (verified MATCH) found that T = clauses/2 and clauses/4 do not close the gap, at each agent's s tuned at
  T = clauses.
- e81x found that LR is not under-tuned: C 0.001 is its best over C 1e-4 to 100.
- On dev, these are now ruled out on MultiWOZ: capacity (4k is level with 16k), T at those values, s (8 to 24), and
  LR's tuning.
- What the earlier rows show (descriptive; a diagnostic in the project notes, not published; no rule):
  - the TM's dev curve is flat after epoch 4;
  - its clauses stay at about 2 literals whatever s is;
  - on the count-table bits, 12 to 26% of clauses hold no positive literal.
- So the next candidates act on how the TM learns, not on its size or margin.

## What was known at this draft (dev only)

MultiWOZ's test split has never been scored. Its texts were read twice, text only: the overlap check of the
text-understanding line, and the a3 fit filter, which only removed sentences (logs/plan_e82_note2_2026-10-21.md).

| Row (refined agent: c4words_a3B_v1, view B, MultiWOZ dev) | Best-dev | Last-10 mean |
|---|---|---|
| e82r3_multiwoz_Bt_c4000_s8, seed 1 (s 8, the s screen's reference and pick) | 0.3123 | 0.2901 |
| e82r3 s 12 / s 16 / s 24, seed 1 | 0.3007 / 0.3037 / 0.3068 | 0.2883 / 0.2929 / 0.2903 |
| e81 T 2,000 / T 1,000 at 4k, seed 1 | 0.2976 / 0.2950 | 0.2781 / 0.2747 |
| e82_multiwoz_Bt_c16000_s8, seeds 1 / 2 | 0.3123 / 0.3093 | 0.2857 / 0.2824 |
| LR on the same bits, C 0.001 (its best over C 1e-4 to 100) | 0.3643 | - |

- **MultiWOZ dev noise:**
  - the median swing between successive evaluations is 0.77 to 1.18 points (0.30 to 0.47 on STAR and ABCD);
  - best-dev minus the last-10 mean has a median of 1.97 points over 21 rows;
  - seed 1 against seed 2 at 16k differs by 0.30 (best-dev) and 0.33 (last-10).
- **Where the TM loses** (e77's breakdown, C4 bits, TM 16k against LR; logs/e77_lr_vs_tm.out): the largest losses
  are labels that extend a frequent label:
  - general-bye+general-welcome: 365 dev turns, TM 0.8%, LR 27.1%;
  - Train-Inform+Train-OfferBook: 297 turns, TM 36.0%, LR 63.6%;
  - Booking-Book+general-reqmore: 278 turns, TM 43.9%, LR 69.4%.
  - Many other labels lose too, for example Attraction-Inform 38.7% against 55.8% and Booking-Request 13.0% against
    49.3%.
- **The negative class:** every run so far used tm.py's default "hardest" negative (the top-scoring wrong class).
  - The published multi-class TM draws a random other class (Granmo 2018; as described in the Regression TM paper,
    arXiv 1905.04206).
  - Focused negative sampling (Glimsdal et al., ISTM 2022) draws it in proportion to (T + clip(S_i)) / 2T. Its
    abstract reports comparable accuracy in about 42% fewer epochs.
  - tm.py has "hardest", "random" (uniform) and "mix:q" (hardest with probability q, else uniform). Focused negative
    sampling as published is not in tm.py and is not run.
- **In-study priors:**
  - the exact literal budget lifted the sparse word routers by 4 to 6 points (R2e, R2d: registered, met);
  - on STAR at 4k, budgets 16, 32 and 64 gave 0.637 to 0.639 against 0.613 without one (e47, dev, one seed: the
    level of 16k without a budget);
  - clause drop did not help on CLINC150's dense MiniLM bits (94.2 to 94.5 against 94.6 to 94.9; FAILURE_MAP Part 3).
- **Published priors** (each picked on test data or a peak over epochs, so priors only;
  the literature report of 2026-10-21 in the project notes, not published):
  - Drop Clause (Sharma et al., AAAI 2023): R52 89.14 to 92.75 at p 0.75, with the mask drawn once per epoch;
  - CSC-TM (Abeyrathna et al., IJCAI 2023): TREC 85.8 to 93.2 at budget 32.
- **A prior from LR:** stronger regularisation did not help LR on these bits (e81x). So a regularising knob for
  the TM has to clear its bar on dev.

## The arms

**Screen e83a.** The refined agent (a3B table, view B: mem, words, xcust, xrecent) on MultiWOZ dev, at 4,000
clauses, T 4,000, s 8, 40 epochs and seed 2.
- All arms use seed 2. Seed 1's reference row won its s screen on the same dev split (the e81 review's selection
  part, finding m3), and a fresh seed removes that tilt.
- Every arm runs through scripts/knob_run.py (below), which also saves its dev predictions.

| Arm | Change from REF | Tag |
|---|---|---|
| REF | none: "hardest" negatives, no dropout, no budget, start "half" | e83_multiwoz_a3B_c4000_s8_ref |
| NR | neg_sampling "random" (uniform over the other classes) | e83_multiwoz_a3B_c4000_s8_nrand |
| NM | neg_sampling "mix:0.5" (hardest with probability 0.5, else uniform) | e83_multiwoz_a3B_c4000_s8_nmix50 |
| D50 | clause_drop 0.5 | e83_multiwoz_a3B_c4000_s8_drop50 |
| L8 | exact literal budget 8 (--max-literals 8; budget_select rank) with start "trim" | e83_multiwoz_a3B_c4000_s8_b8 |

**Design choices, declared now:**
- **Clause dropout is tm.py's own rule:** each clause is left out of each minibatch of 128 with probability p,
  redrawn every minibatch. The paper (and tmu) draw it once per epoch. That needs new TM code and is not run here. One
  p (0.5) is screened.
- **The budget arm uses start "trim"**, so the budget holds from the first step (R2c, e47). Budget 8 binds on
  MultiWOZ. Its clauses have a mean of about 4 to 5 literals and a 95th percentile of 12 to 18 (the 4k and 16k rows at
  s 8). Budget 32 would hardly bind.
- **"random" stands in for the published multi-class rule.** Focused negative sampling is not run.

## Statistics and the pick

- **Two statistics per row, both from results/explore/abcd.jsonl:**
  - best-dev: the row's acc, the best of its 20 evaluations;
  - the last-10 mean: the mean of its last 10 evaluations.
- **An arm qualifies** if, against REF, it gains at least +0.5 points at best-dev AND at least +0.5 on the last-10
  mean.
- **The pick** is the qualifying arm with the largest last-10 gain. Ties go to the earlier arm in the table.
- **No qualifier:** no follow-up, and e83 ends with "none of the four knobs lifts the reader at 4k (one seed)".
- **Power:** a one-seed +0.5 at 4k is about 1.6 seed SDs (0.30), so e83b is the real check.

## Follow-ups (only with a pick)

- **e83b:** the pick on MultiWOZ at 16,000 clauses, T 16,000, seeds 1 and 2.
  - The reference is e82_multiwoz_Bt_c16000_s8, seeds 1-2: e82 Stage A rows, not chosen by any screen at 16k.
- **e83c** (only if C1 is met): the pick on STAR at 16k (s 16, T 16,000, seeds 1-2) and on ABCD at 4k (s 16, T 4,000,
  seeds 1-2).
  - The references are e82 Stage A: e82_star_Bt_c16000_s16 and e82_abcd_Bt_c4000_s16, seeds 1-2.

## Bars (fixed now, before any e83 number)

- **C1, "the knob lifts the MultiWOZ reader":** the pick's 16k mean of seeds 1-2 minus the reference's is at least
  +0.5 at best-dev AND at least +0.5 on the last-10 mean.
- **C2, "the knob closes half the gap":** the pick's 16k best-dev mean minus LR (0.3643158037776872) is at least
  -2.677. That is e81's B2 bar. The last-10 reading is reported beside it, with no rule.
- **C3, "no harm on STAR and ABCD"** (e83c): for each dataset, the pick's 2-seed best-dev mean minus the reference's
  is at least -0.3.
- **A claim beyond MultiWOZ** needs C1 and C3. Otherwise any claim is MultiWOZ-only.
- **P1, the near-duplicate prediction** (stated now; it can fail; screen rows only).
  - Pooled set: the dev turns whose gold label is one of the 6 labels of three pairs: general-bye and
    general-bye+general-welcome; Train-Inform and Train-Inform+Train-OfferBook; Booking-Book and
    Booking-Book+general-reqmore.
  - Prediction: under NR or NM, the TM's pooled accuracy on these turns rises by at least +2.0 points over REF (each
    arm judged alone).
  - Reported beside it, with no rule:
    - each of the 6 labels;
    - the pooled three "+" labels;
    - the 20 most frequent gold labels for every arm.
  - Source: the saved dev predictions. Their final accuracy must equal the row's acc exactly.

**What a miss means:**
- **No qualifier:** none of these knobs, at these values and at 4k with seed 2, lifts the MultiWOZ reader. FAILURE_MAP
  Part 20 gets the line.
  - Left open: combinations, a per-epoch dropout mask, focused negative sampling, other p and budget values.
- **C1 missed:** the screen's gain did not hold at full size on 2 seeds.
- **C1 met, C2 missed:** the knob lifts the reader, but by less than half the gap to LR.
- **P1 missed with C1 met:** the gain does not come from the near-duplicate labels. P1 met with no qualifier: the knob
  moves the pairs but not the total.

**How a pass is written:** C1's own words, "on MultiWOZ dev, 16k, 2 seeds", with the final-epoch reading beside it.
- No "beats", "better than", "gains" or "first".
- No registered test follows from e83 alone. Any test needs its own registration and use count.

## Code (all NEW files; tm.py, abcd_explore.py, newtable_run.py and table_run.py stay byte for byte)

- **scripts/knob_run.py** runs a target script in this process (runpy) after patching dl.tm.CoTM.
  - `__init__` sets the given knobs on the TMConfig the script built: --neg-sampling, --clause-drop. The row's cfg
    records them.
  - `fit` notes the dev labels it is given.
  - `predict` appends each prediction it returns: the 20 evaluations, then abcd_explore's final call on the best-dev
    snapshot.
  - --preds-out writes them, with the dev labels, to one .npz at exit. It refuses to overwrite.
  - With no knob and no --preds-out, it changes nothing.
- **Knob-off identity**, in the GPU smoke stage (a condition of the review).
  - Small ABCD jobs: refined agent, 200 clauses, T 50, 2 epochs, s 16, seed 1.
  - The plain row (no wrapper) and the wrapper's row with no knob must be equal in every field except tag, time,
    train_s, peak_gpu_mb and each evaluation's clock reading "t" (as scripts/e83_pick.py checks).
  - A third job with neg_sampling random and clause_drop 0.5 must show both in its cfg and differ from the plain
    row.
  - A fourth with budget 8 and start trim must show no clause over 8 literals.
- **scripts/e83_pick.py** has four stages: smoke, screen (writes jobs_e83b.txt), mid (C1; writes jobs_e83c.txt only
  if C1 is met) and final (C2, C3, P1 and the per-label table). Every row it reads is checked:
  - the settings above;
  - TM settings otherwise equal to the reference rows' (n_states 256, epochs 40, batch_size 128, negation, half,
    eval_every 2);
  - the a3B table;
  - one dev bits hash per dataset, equal to the reference rows';
  - e83 rows written after the freeze, exactly one per tag and seed.
  - It refuses on a count above 1 (e81's integrity m3) and checks each LR row's bits and time (m4).
- **logs/e83_driver.sh is the top-level launcher, pinned by this plan** (e81's lesson):
  - it runs smoke, the smoke check, e83a, the screen, e83b, mid, e83c and final through seq_queue.sh, and stops when
    a check fails;
  - every way out appends one end line, "e83 end: <branch> <UTC> exit <code>".
- **Tests (plain scripts):**
  - tests/test_knob_run.py: knob parsing and refusals; CPU path tests through a stub target that builds a CoTM on the
    CPU;
  - tests/test_e83_pick.py: made-up rows; every stage, the pick rule at its edges, the bars at their edges, P1 met
    and not met, and the refusals;
  - tests/test_e83_driver.py: a stub queue (done path, failed smoke, no pick, C1 missed, SIGTERM, an existing log).

## GPU estimate (from e81's and e82's queue logs)

| Stage | Jobs | Minutes |
|---|---|---|
| Smoke | 4 tiny ABCD jobs (about 0.5 each) | 3-5 |
| e83a | 5 MultiWOZ 4k jobs (e81: 641-710 s each) | 55-60 |
| e83b (if a pick) | 2 MultiWOZ 16k jobs (e82: about 29 each) | about 58 |
| e83c (if C1 met) | 2 STAR 16k (about 23 each), 2 ABCD 4k (about 4 each) | about 54 |
| **In all** | | **about 1.0 h (no pick) to 3.0 h** |

## Hashes (sha256sum at 14:31:18Z; re-read at the freeze)

The 16 data files are byte-identical to e81's hash table (the a3B table, the example files, labels.json and the empty
*__none.npz). scripts/ is also pinned by the freeze record. logs/ and tests/ files are pinned only by this table.

| File | SHA-256 |
|---|---|
| data/tables/c4words_a3B_v1/table.npz | 4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855 |
| data/tables/c4words_a3B_v1/vectors.npy | 21d0c4fb148498531a6523f9635401a6215ef4dd5c8cd4b7ca8ce6e7b80f8e14 |
| data/tables/c4words_a3B_v1/words.txt | 49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29 |
| data/tables/c4words_a3B_v1/tokenize_ref.py | 2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803 |
| data/tables/c4words_a3B_v1/README.md | 35ae923e408084ff8d9512a06eea5f481aff0037da0e23d06b111286e6dbd85f |
| data/tables/c4words_a3B_v1/MANIFEST.sha256 | 1d1a00ad74634f77d38894e4a476c1717494c94a76450210518ec9bc8ed267f3 |
| data/multiwoz/examples_train.json.gz | 0e20a31434de10a5248f0b10ab01e7474ceb5e020cd5cbc7048bfe8f5db69e9b |
| data/multiwoz/examples_dev.json.gz | 3414cc8a6ecc64aafa0df070748b9d502a12e7996420b84a12df1b094a89fcd2 |
| data/multiwoz/labels.json | ed3379da965e969daa1f5e7982c42bb04dc61ee0e184a93327dc2e8bf02dede7 |
| data/star/examples_train.json.gz | 47735157f3e0035c59c8f88005ebe5457678083d60e653fbdca2f4df45c023df |
| data/star/examples_dev.json.gz | e562736dd9afa1b781d48bf78f43f23b1b8ceda17b3ddbf01557c59f0c8efb18 |
| data/abcd/examples_train.json.gz | c9e0f9b58a13a3dd9eff5821177cd74373cd1816465445a32bb6cc131933835f |
| data/abcd/examples_dev.json.gz | 5ccc737f5be9086060de4c28f560f1aa9b94566ed9599a98f44212c0e6cf2f17 |
| data/emb/star__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| data/emb/abcd__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| data/emb/multiwoz__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| scripts/knob_run.py | 113cb74eca36d1f9c3fbea889a2fe8aad2c1ad91878af9f17d93cffa14d65e4e |
| scripts/e83_pick.py | 50210db4df926a338925738cc845075260c69bf2baa77836a88ae4b8eb89eccd |
| logs/e83_driver.sh | f75ec266da30f90d751ecb4cde4fb27d9b7f636ee0c4e3848a3f02aeccfe320b |
| logs/jobs_e83_smoke.txt | 0c62c57f4afd72d75d31c9f9b0bb1971977e9c6488e4cad507f310f0c022fd0c |
| logs/jobs_e83a.txt | 31b7bbf0cbc73a1a836ee1310ef4568c9d58af424f46da0842914fe923c57e4d |
| tests/test_knob_run.py | 93fa20eab0c6cfd9b4f768ff6f4be345896c5f6b3a50400d4ad67d5d0b011cd5 |
| tests/test_e83_pick.py | 940dda881fc4feb7a44f48f7fb0dd331aacf86f966c4c60809da5d4218541840 |
| tests/test_e83_driver.py | fc1e7c3a5e80b06afd82e7e7417abac7e99497c0fb5f18b5b6c5539f2f95f30c |
| logs/test_knob_run.out | 12984948e816bdd124be501f7aa3947a12295a9dbed66317b8df1777171793ac |
| logs/test_e83_pick.out | efc36a6a768b39d8c1d4eb65200733f115b2f6c63adf727fae5619831403c72e |
| logs/test_e83_driver.out | 827b174a60e49e561cd4e43297c67e0cfb9de8041355e5163098afee84dea854 |

**Dev bits of the refined agent** (abcd_explore.py's bits_sha256["dev"]; e81's table; checked on every row by e83_pick.py):

| Dataset | Dev bits SHA-256 |
|---|---|
| STAR | 5d3cd86eb18788b28165baaaaeb9c87e4323eae90b56be62820725ebe4566254 |
| ABCD | 3222f78c87af48579d5e214eb2ebf75b1a177e633851b8417985ed6bfb44d110 |
| MultiWOZ | 612b8f0b75e32a30b50493eaedc1eec9d6bac6893d48075144a1f48fad2e23a3 |

**Reference rows** (results/explore/abcd.jsonl, written before this draft; e83_pick.py checks their settings and bits):
- e82_multiwoz_Bt_c16000_s8 seeds 1-2: 0.3122706889523033 and 0.3092811523304797 (the constant TODAY);
- e82_star_Bt_c16000_s16 and e82_abcd_Bt_c4000_s16, seeds 1-2;
- LR: results/explore/dryrun/dry_e82mwoz_lr_mem_a3B/seed0.json, dev_acc 0.3643158037776872 at C 0.001.

## Newest work (three web searches on 2026-10-21, between the clock reads at 13:28 and 13:35 UTC; snippets only, not read in full)

- No 2025-2026 paper was found on negative-class selection for multi-class or coalesced TMs.
- Found and checked only as snippets:
  - Uncertainty quantification in the TM (arXiv 2507.04175, 2025): one-vs-rest for multi-class;
  - Fuzzy-Pattern TM (arXiv 2508.08350, 2025);
  - ETHEREAL (arXiv 2502.05640, 2025);
  - a 2026 ChemRxiv QSAR study using tmu.
  None discusses negative sampling.
- tmu's README lists focused negative sampling, drop clause and the literal budget as features (tmu main 5605ff0,
  2026-08-03, from the earlier report).
- The earlier report found no 2025-2026 arXiv paper on TM dropout, clause budgets or over-fitting either (newest-work
  section dated 06:34 UTC).
