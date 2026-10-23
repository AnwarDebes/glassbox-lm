# Plan e79 (2026-10-20; design fixed 18:57 UTC with two additions; frozen before any run): view A+B on the C4 table

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

The lead (18:51 UTC) from e77a. On MultiWOZ dev, LR on view B's bits gets 0.3586, which is 5.0 points
above the C4 TM (0.3085 at s 8, 4k) and 1.9 above the bge-large TM (0.3400 at s 24, 4k). So the C4 bits hold more than
the TM reads (the map's P5). The bge-large view also has two dense-similarity parts that view B lacks. They are
`intent`, the whitened similarity of the customer's words so far to each subflow's centre, and `step`, the same for
the last 3 turns against each next button's centre. This plan adds those two parts with no neural network at any stage:
they are built from C4 mean vectors. All of view B is kept (additive, R1), and the result is called "view A+B on the C4
table". The idea was tried once, with potion, in e60 (STAR dev 16k, R19 era): A+B 0.6378 against B 0.6343, +0.35.

## What is new (new files only; no frozen file changes)

- **`scripts/c4mean_prepare.py`:** data/emb/&lt;ds&gt;__c4mean.npz, in the layout of the encoder caches
  ({train,dev,test}__{cust,recent,full}, float32, rows in record order).
  - Each vector is the L2-normalised average of the L2-normalised C4 word vectors of the text's known tokens, with
    "(nothing yet)" for an empty text. These are lookups, an average and a division, from the counting table
    (c4words_S1000_v1).
  - Train and dev only. The test arrays are empty (0, 0) until a registration adds them, as multiwoz_embed.py
    --add-test does, so MultiWOZ's test texts stay unread.
  - Provenance in the file: the table's file SHA-256, the tokenizer, the empty-text rule and the examples' SHA-256s.
  - It refuses to overwrite.
- **`scripts/mean_run.py`:** table_run.py's way (dl.count_text.patch, so potion's table cannot load; runpy of the
  unedited target).
  - It requires `--encoder c4mean` and a cache whose provenance names the C4 table file e1771d9e.
  - It allows the parts mem, intent, step, words, xcust and xrecent.
- **Self-test before the freeze (CPU):**
  - the cache's rows equal a direct recomputation on 1,000 random texts per split (max gap 1e-6);
  - view A+B's bits, with intent and step left out, equal view B's dev bits (e74c's hashes).
  - Note: ABCDView's Proto draws its centres with a torch generator on the GPU (src/dl/abcd.py, device "cuda"), as for
    bge-large. So the A+B bits are fixed by the GPU run, and the dev rows' hashes are their record.

## Built and self-tested before the freeze (CPU; 2026-10-20 18:58-18:59 UTC)

- `scripts/c4mean_prepare.py` (2a18bee9) -> logs/c4mean_prepare.out (a56cbb9c):
  - data/emb/star__c4mean.npz (3699d21c), abcd__c4mean.npz (6a57f25a) and multiwoz__c4mean.npz (adbb86cd);
  - train and dev only, with the test arrays (0, 0);
  - zero rows (texts with no known C4 token): STAR train cust 8 and dev cust 1, none elsewhere.
- `scripts/c4mean_selftest.py` (0b56fe2c) -> logs/c4mean_selftest.out (3eb6e954). All 60 checks pass:
  - provenance and empty test arrays;
  - row counts;
  - norms of 1 or 0;
  - 1,000 random rows per split and field recomputed in float64 from the raw table file, max gap 2.4e-7 or less.
- `scripts/mean_run.py` (df72618e): needs `--encoder c4mean` and a cache whose provenance names the C4 table file. It
  applies count_text.patch and count_bits.patch, then runs the unedited target.
- **The A+B bits are first built in the GPU runs,** since Proto draws its centres on the GPU. e79's check: within a
  dataset, every candidate row has the same train and dev bits, as each bge-large view did across its runs.
  Candidate n_bits = view B's + the intent and step bits.

## Arms (dev; R21's settings; the C4 agent's tuned s)

| Dataset | Clauses, s | Reference: view B | Candidate: view A+B on C4 |
|---|---|---|---|
| MultiWOZ | 16,000, the C4 s* from e77 | seed 1 = e77b's row; seed 2 new | seeds 1, 2 |
| STAR | 16,000, s 16 | e74c seed 1; e78's seed 2 | seeds 1, 2 |
| ABCD | 4,000, s 16 | e74c seed 1; e78's seed 2 | seeds 1, 2 |

## The rule (fixed before any run)

- **Taken** only if all three hold, on the mean of seeds 1 and 2, candidate minus reference:
  - MultiWOZ dev at least +1.0;
  - STAR dev no worse than -0.5;
  - ABCD dev no worse than -0.5.
- **If taken:**
  - R3: s is re-tuned for the new view by the e74 rule on each dataset.
  - R5: the guards. Intent and step bits are free in the level-B domain, a superset, and the 50 proofs are re-run on
    the new view's ABCD models.
  - Then the R24 gate is checked again on MultiWOZ dev at 16k: the tuned C4 A+B agent minus the tuned bge-large agent
    must be at least -1.0, and "what was known" lists every dev try.
- **Provenance:** "no neural network at any stage" (C4 counting table, means, whitened prototype thresholds, k-means,
  a TM).
- **Reported with no rule:**
  - LR on the A+B bits;
  - e77_breakdown.py (first against later actions, domain-right against act-right) for the candidate against view B.

## Two additions made before the freeze (18:57 UTC)

- **(a) Stacking (R2).**
  - If e78 takes 2d-a, e79 runs on top of it: the reference is view B + xcustn and the candidate is A+B + xcustn.
  - Both run at the s that e78's R3 picks, and only after e78's R3 and R5 are done.
  - If e78 does not take 2d-a, e79 runs as written above.
  - mean_run.py applies dl.count_bits.patch too, so that "xcustn" is available for (a).
- **(b) R24 wording, if A+B reaches R24.** The registration says that the C4 agent's view was chosen on MultiWOZ dev
  among the views tried (B, B + xcustn, A+B). The bge-large agent keeps R18's fixed view, and s is tuned for both. Any
  "better than" line carries that caveat.
- **Order:** e77b, then e78, then whichever of e79 and the text-understanding line's step 0b is frozen and ready first. e79 also waits for
  e78's verdict, because of (a). One big job at a time.

## Budget

- GPU, alone: MultiWOZ 16k is about 25 min a run (3 runs), STAR 16k about 25 min (2 runs), ABCD 4k about 3 min
  (2 runs). About 2.2 h.
- CPU: the caches take minutes.
- **Order:** after e78 (2d-a), whose seed-2 references this plan reuses.
