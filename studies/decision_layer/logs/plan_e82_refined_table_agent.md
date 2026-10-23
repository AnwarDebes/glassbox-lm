# Plan e82 (2026-10-21): the agent on the teacher-refined table (step 3a of the text-understanding line): R4, e82-R3, the go-bars, R5, and which one test

**Status:** revision 4. Not frozen. It is frozen with `scripts/freeze.py` before any e82 job. GPU slot: after step 3c
of the text-understanding line (fixed at 01:00 UTC).

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**Draft history** (times read with date -u):
- **Rev. 1 (638a5e27, about 00:42 UTC):** the shelf-ready draft.
- **Rev. 2 (41008b88, 01:03 UTC):** the export and its self-test.
- **Rev. 3 (31a04c81, 01:20 UTC):** the export's README/MANIFEST update and its caveat.
- **Rev. 4 (this, from 01:24 UTC):** the decisions of 01:21 UTC on the findings of a two-part review of the R25 draft
  and of this plan (judged by an AI model, not a person; project notes, not published; 2026-10-21, fa111d9d).
- The earlier revisions are in logs/archive/ (plan_e82_draft3_31a04c81.md).
- No e82 job had run and no e82 dev row existed at any revision. So no rule here was shaped by an e82 number.

**Gate:** step 3a of the text-understanding line passed (3 of 3 intent sets with both probes, val; checked independently,
01:00 UTC). That is the gate for Part 1. The order of the work is: teacher table, then paraphrases, then the agent and
rule checks, then one registered test.

## The table and its caveats (in every report line)

- **c4words_a3B_v1** (the text-understanding line's export, copied byte for byte to data/tables/c4words_a3B_v1/):
  - the same 63,091 C4 words, row order and tokenizer as c4words_S1000_v1;
  - 768 dimensions, unit rows.
  - Each row was refitted by least squares, so that a C4 sentence's mean row matches a teacher's sentence embedding.
  - **The teacher:** bge-base-en-v1.5@a5beb1e3 ([CLS], L2-normalised, MIT).
  - **The fit:** 1,999,932 C4 sentences, 39,995 held out, lambda* 0.01; held-out student-teacher cosine 0.8116.
- **Provenance:** "network-free at run time, teacher at training time". Never "no neural network at any stage"; that
  phrase stays with c4words_S1000_v1, whose agent is reported beside every refined-table number.
- **Same family:**
  - The teacher is the smaller sibling of the reference agent's encoder, bge-large-en-v1.5 (same v1.5 family and
    recipe).
  - It is also the teacher that potion-base-32M was distilled from (R19's table).
  - With lambda* 0.01, the rows are mostly set by the teacher. The counting table survives as the word list, the
    tokenizer and the start of the fit.
  - So this is a replication of potion's recipe on the C4 word list, not a new method (the export's README). R19's
    potion agent is shown beside every result: STAR 0.6271, ABCD 0.8301 on test, at s 8, unequal tuning.
- **The teacher's training data cannot be checked.** It may include the intent sets and STAR, ABCD or MultiWOZ text. So
  the teacher could inflate the refined table's gain over the counting table; it does not inflate a comparison with
  potion.
- **The view-choice caveat, on any comparison with bge-large:** view B was chosen and developed on STAR, ABCD and
  MultiWOZ dev (e60, e78, e79), while the bge-large agent keeps R18's or R9b's fixed view. Only s is tuned equally.
- **Only 3a enters the agent.** 3c (LLM paraphrases of intent-set training texts) and 3b are not part of the agent's
  view.

## Hashes (full SHA-256; checked at the freeze and by the checkers)

| File | SHA-256 |
|---|---|
| data/tables/c4words_a3B_v1/table.npz | 4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855 |
| data/tables/c4words_a3B_v1/words.txt | 49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29 |
| data/tables/c4words_a3B_v1/tokenize_ref.py | 2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803 |
| data/tables/c4words_a3B_v1/vectors.npy | 21d0c4fb148498531a6523f9635401a6215ef4dd5c8cd4b7ca8ce6e7b80f8e14 |
| data/tables/c4words_a3B_v1/README.md | 35ae923e408084ff8d9512a06eea5f481aff0037da0e23d06b111286e6dbd85f |
| data/tables/c4words_a3B_v1/MANIFEST.sha256 | 1d1a00ad74634f77d38894e4a476c1717494c94a76450210518ec9bc8ed267f3 |
| data/star/examples_{train,dev}.json.gz | 47735157f3e0035c59c8f88005ebe5457678083d60e653fbdca2f4df45c023df, e562736dd9afa1b781d48bf78f43f23b1b8ceda17b3ddbf01557c59f0c8efb18 |
| data/abcd/examples_{train,dev}.json.gz | c9e0f9b58a13a3dd9eff5821177cd74373cd1816465445a32bb6cc131933835f, 5ccc737f5be9086060de4c28f560f1aa9b94566ed9599a98f44212c0e6cf2f17 |
| data/emb/{star,abcd,multiwoz}__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |

- The first README and MANIFEST (3ab5795b and 3a8cc442) are kept beside these. table.npz did not change.
- **`scripts/newtable_run.py`** (f98ce98e) takes the expected hash from the folder's own MANIFEST, so it only proves that
  the folder agrees with itself. Every e82 checker therefore also compares each row's
  static_info[xcust/xrecent].table_file_sha256 with the literal table.npz hash above, and checks the MANIFEST's own hash
  before job 1.

## Code (new files only; no frozen file changes)

- **scripts/newtable_run.py (f98ce98e):** it registers the table in-process, then runs table_run.py's main unchanged.
  - CountTable's checks apply: the file hash, and tokenize_ref.py being the C4 reference.
  - So do table_run.py's: --encoder none, the no-encoder parts, and potion's table cannot load.
- **scripts/e82_selftest.py (0f8ec781):**
  - The plumbing test on the C4 table under a test name passed (logs/e82_selftest_plumbing.out e01c27e7).
  - The run on c4words_a3B_v1 passed (logs/e82_selftest_a3B.out 031606be): view B builds on STAR, ABCD and MultiWOZ,
    and mem and words are byte-identical to the C4 view's.
- **The network-free test on the newtable path** (tests/test_newtable_path.py, before the freeze): view B on
  c4words_a3B_v1 is built through newtable_run's registration, followed by a CPU TM decision. It asserts that:
  - sentence_transformers, model2vec and transformers are not in sys.modules;
  - no encoder cache other than *__none.npz is opened.
- **scripts/e82_check.py (14457252):** R4's checks and rule.
- **scripts/e82_pick.py (before the freeze):** e82-R3's picks below. It writes the follow-up job files and refuses to
  overwrite.

## Stage A, R4 (dev; view B on c4words_a3B_v1 against view B on the C4 table; the agent's own tuned s; seeds 1 and 2)

| Dataset | Clauses, s | Reference rows (C4 table) | Candidate tag |
|---|---|---|---|
| STAR | 16,000, s 16 | e74c_star_Bc4_c16000_s16 (seed 1, 0.6545), e78_star_B_c16000_s16 (seed 2, 0.6579) | e82_star_Bt_c16000_s16 |
| ABCD | 4,000, s 16 | e74c_abcd_Bc4_c4000_s16 (seed 1, 0.8417), e78_abcd_B_c4000_s16 (seed 2, 0.8423) | e82_abcd_Bt_c4000_s16 |
| MultiWOZ | 16,000, s 8 | e77_mwoz_Bc4_c16000_s8 (seed 1, 0.3038), e79_mwoz_B_c16000_s8 (seed 2, 0.3079) | e82_multiwoz_Bt_c16000_s8 |

- **Jobs:** logs/jobs_e82.txt (868edd44), 6 jobs, one at a time through seq_queue.sh with cpu_load at most 4.5.
- **The R4 rule** (the map's stack rule): the table is taken into the agent only if no dataset loses more than 0.5
  points (candidate minus reference, mean of seeds 1 and 2). If it is not taken, e82 ends there: a dev-only line, and
  no test.

## Stage B, e82-R3 (only if R4 takes it): the refined agent's own s, each dataset by its earlier rule, copied exactly

The name "e82-R3" keeps it apart from R3 of the text-understanding line. The grid is {8, 12, 16, 24}, with no
extension; an edge pick is kept. Everything is seed 1. The candidate tags are e82r3_<ds>_Bt_c<clauses>_s<s>.

- **STAR** (e74d's rule, the one bge-large got for R22):
  - a 4k screen at s 8, 12, 16 and 24;
  - the pick: the highest 4k dev, which must be at least 0.3 points above s 8's in-sweep row, otherwise s 8; an exact
    tie goes to the first of [8, 12, 16, 24];
  - if the pick is not 8, 16k runs at s 8 and at the pick follow;
  - s*: the pick, unless its 16k dev is below s 8's 16k dev, then s 8. That is the 16k confirmation and fallback, as
    in e77.
- **ABCD** (e75's rule, the one bge-large got for R23):
  - a 4k screen at s 8, 12, 16 and 24, with the same pick rule;
  - no 16k step; s* is the pick.
- **MultiWOZ** (e77's rule, for the R24 gate):
  - a 4k screen at s 8, 12, 16 and 24, with the same pick rule;
  - 16k at s 8, and at the pick if it is not 8;
  - s* is the pick, unless its 16k dev is below s 8's 16k, then s 8.
- **scripts/e82_pick.py** applies these rules from the rows, writes logs/jobs_e82r3b.txt (the 16k jobs), and prints s*
  per dataset. Its output fills R25's and R24's s.
- **Noise note:** the text-understanding line's intent-set R3 picked s 16 on HWU64 from one seed, and seeds 4-6 then
  came out about 1 point lower. A single-seed screen is noisy, so the 0.3 margin rule is kept exactly.

## Stage C, the seeds at s* (only after Stage B)

- **Seed 2 at each dataset's s*:**
  - STAR at 16k, MultiWOZ at 16k, ABCD at 4k;
  - seed 1 is Stage B's row at s* (16k for STAR and MultiWOZ, 4k for ABCD).
  - If s* equals Stage A's s, Stage A's seed-2 row is used and nothing new runs.
- **ABCD seeds 3, 4 and 5 at s*** (with --save): R5's five models. They are the same models a registered test run
  reproduces: same settings and seeds, best-dev snapshot. The checker requires seed 1's dev accuracy to match exactly.
- **Jobs:** logs/jobs_e82c.txt, written by e82_pick.py.

## The dev go-bars (fixed now, before any e82 dev number)

- **STAR's test split is read** only if the refined agent at its e82-R3 s* minus the C4 agent at its own tuned s
  (s 16; reference rows e74c and e78) is at least **+1.0** on STAR dev, on the mean of seeds 1 and 2.
- **ABCD's test split is read** only if the same difference is at least **+0.7** on ABCD dev (C4 at s 16; reference
  rows e74c and e78).
- **Why these bars:** power. Potion, with the same teacher, beat the C4 table in this agent by only +0.10 (STAR) and
  +0.29 (ABCD) in R21. An interval above 0 on test needs about +0.55 / +0.46. 80% power needs about +0.8 / +0.65, or
  about +1.0 / +0.8 with seed noise. These bars come from that review (fa111d9d) and were fixed at 01:21 UTC.
- A dataset below its bar becomes a dev-only failure-map line, and its test split stays unread.

## The R24 gate for the refined agent (MultiWOZ dev)

- The refined agent at its MultiWOZ s* (16k, seed 1) minus the bge-large agent at its s* (e77_mwoz_bge_c16000_s8,
  0.3359, seed 1) must be at least **-1.0**. That is the same gate as e77's (logs/plan_e77_note_2026-10-20.md).
- The mean over seeds 1 and 2 is reported beside it.
- "What was known" then lists every MultiWOZ dev try: B, B + xcustn, A+B and the refined table.

## Which one registered test (the precedence, fixed now; decided from dev alone, before any test read)

1. **If the refined agent passes the R24 gate:** R24-refined is the one registered test, because MultiWOZ is the fresh
   split. R25 does not run.
2. **Otherwise:** R25, on the datasets whose go-bars pass.
3. **Otherwise:** no registered test.

Running both needs a recorded decision.

## R5, the guards (before any test freeze; ABCD)

- **The wrapper** (scripts/verify_policy_a3B.py, with its note frozen first; e72e's way):
  - it registers c4words_a3B_v1 through newtable_run.register;
  - it sets verify_policy_c4's TABLE and SRC to the a3B table and to the 5 ABCD models at s* (Stage C);
  - so the bits are built on the a3B table, never on the C4 table.
- **The 50 rules** are proved raw and guarded, with M recomputed per model.
  - Pass: 250 of 250 guarded proofs, which is a precondition of any registered test.
  - Raw counterexamples are expected and reported.
- xcust and xrecent keep their names, so verify_policy's thermometer chains apply.
- The guards' cost on dev is measured open loop and closed loop.

## Budget (GPU, one job at a time)

- Stage A: 6 runs, about 1.8 h.
- Stage B: 12 screens at 4k (STAR about 6 min, ABCD about 3 min, MultiWOZ about 7 min each), plus up to 4 runs at 16k
  (about 26 min each). About 2.6 h.
- Stage C: up to 3 runs at 16k and 4 at 4k, about 1.5 h.
- R5 is CPU.

## Selection caveat

- The table passed a dev gate.
- s is picked on the same dev splits among 4 values.
- Dev numbers are best-epoch numbers, so they are optimistic.
- A registered test decides.
