# Plan e81 (2026-10-21): does the TM's vote threshold T close the reader's gap to LR on the same bits? (dev only)

**Status:** revision 3. Not frozen. It is frozen with `scripts/freeze.py` before any e81 job, the smoke run included. The
design was fixed at 05:23 UTC, after e82.
- **Earlier drafts:** archived, never frozen, nothing run from them.
  - Rev. 1 (c19f3637): the design of 2026-10-20 21:28 UTC, set to plan-only at 22:21 UTC. It is in
    logs/archive/plan_e81_draft1_c19f3637.md.
  - Rev. 2 (2ac63ac2): written 05:29 UTC today. It is in logs/archive/plan_e81_draft2_2ac63ac2.md.
- **At this revision:** no e81 job had run. results/explore/abcd.jsonl had 0 rows tagged e81, and abcd_models had 0
  e81 files.
- **GPU:** after step 3c of the text-understanding line (about 09:30 UTC), alone, one job at a time through seq_queue.sh.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**The question (fixed now):** does the vote threshold T close the TM's gap to LR on the same bits? In every run so far
T equals the number of clauses.

**Why now:** e82 found that the reader is the bottleneck on MultiWOZ.
- **Refined bits:** LR on the refined agent's bits is 0.3643 (C 0.001). The refined TM is 0.3108 (mean of e82's
  seeds 1-2, 16k), which is 5.35 points below.
- **C4 bits:** LR is 0.3586 and the TM is 0.3059, which is 5.27 points below.
- **STAR and ABCD:** on these the C4 TM is level with LR or above it (table below). T may help MultiWOZ and hurt the
  others, so a general claim needs all three datasets.

**Provenance in every line:**
- refined agent (table c4words_a3B_v1): "network-free at run time, teacher at training time (bge-base-en-v1.5)";
- C4 agent (c4words-S1000-v1): "no neural network at any stage";
- bge-large agent: R18's view on bge-large sentence vectors.

## What was known at the freeze

These are dev numbers only, and none is from a test split. They were read through e81_pick.py's own checks by a
read-only script.

T is the number of clauses everywhere. The full-size columns use 16k clauses for STAR and MultiWOZ and 4k for ABCD.

| Agent | Dataset | 4k seed 1 (screen reference) | Full size: best-dev, final, last-10 (seeds) | LR on the same bits | TM - LR |
|---|---|---|---|---|---|
| refined | STAR | 0.6375 | 0.6576, 0.6564, 0.6534 (2) | not run | - |
| refined | ABCD | 0.8461 | 0.8476, 0.8415, 0.8433 (2) | not run | - |
| refined | MultiWOZ | 0.3123 (best epoch 4) | 0.3108, 0.2961, 0.2840 (2) | 0.3643 | -5.35 |
| bge-large | STAR | 0.6401 | 0.6507, 0.6481, 0.6468 (1) | not run | - |
| bge-large | ABCD | 0.8472 | 0.8472, 0.8436, 0.8423 (1) | not run | - |
| bge-large | MultiWOZ | 0.3325 | 0.3359, 0.2856, 0.3120 (1) | not run | - |
| C4 | STAR | 0.6349 | 0.6562, 0.6540, 0.6507 (2) | 0.6558 | +0.04 |
| C4 | ABCD | 0.8417 | 0.8420, 0.8373, 0.8367 (2) | 0.8119 | +3.01 |
| C4 | MultiWOZ | 0.3085 | 0.3059, 0.2963, 0.2839 (2) | 0.3586 | -5.27 |

- **MultiWOZ's dev curves are noisy.** The median change between two logged evaluations is 0.77-1.18 points on
  MultiWOZ and 0.30-0.47 on STAR and ABCD.
  - On MultiWOZ the last-10 mean lies 2.2-2.7 points below the best-dev snapshot.
  - So a +0.3 or +0.5 margin on best-dev is within one step's swing there.
- **The test use counts are unchanged:** e81 reads no test split.

## Arms (dev; R21's other settings; each agent at its own tuned s)

| Agent | Runner | s: STAR / ABCD / MultiWOZ | T = clauses rows at 4k, seed 1 |
|---|---|---|---|
| refined | newtable_run.py on c4words_a3B_v1, view B (mem, words, xcust, xrecent) | 16 / 16 / 8 (e82-R3) | e82r3_star_Bt_c4000_s16, e82_abcd_Bt_c4000_s16, e82r3_multiwoz_Bt_c4000_s8 |
| bge-large | abcd_explore.py, R18's view (mem, intent, step, words) | 16 / 24 / 8 (e74d, e75, e77) | e74d_star_bge_c4000_s16, e75_abcd_bge_c4000_s24, e77_mwoz_bge_c4000_s8 |
| C4 (no rule) | table_run.py on c4words-S1000-v1, view B | 16 / 16 / 8 (e74c, e74c, e77) | e74c_star_Bc4_c4000_s16, e74c_abcd_Bc4_c4000_s16, e77_mwoz_Bc4_c4000_s8 |

- **The smoke run** (logs/jobs_e81_smoke.txt, 3 jobs): each runner path on ABCD with a tiny TM (200 clauses, T 50, 2
  epochs, seed 1), tagged e81smoke_<agent>.
  - Check: `scripts/e81_pick.py --stage smoke`. Each row must show 200 clauses, T 50, 2 epochs and the tuned s, and must
    be written after the freeze.
  - Its dev bits must equal the agent's ABCD reference row's bits, and the refined row must be on the a3B table.
  - Nothing else starts unless all 3 pass.
- **The screen** (logs/jobs_e81a.txt, its first 18 jobs): T = 2,000 and T = 1,000 at 4,000 clauses, seed 1, for the 3
  agents on STAR, ABCD and MultiWOZ. The tags are e81_<ds>_<agent>_c4000_s<s>_T<T>.
- **LR on the same bits** (the same file, its last 5 jobs; no rule):
  - The runs are refined STAR and ABCD, and bge-large STAR, ABCD and MultiWOZ.
  - They use abcd_confirm.py --dry-run with the wide grid C in {0.001, ..., 100}: C is chosen on dev and scored on
    dev, as for the 4 LR rows that already exist.
- **The pick** (`scripts/e81_pick.py --stage screen`), per agent and dataset:
  - take the highest 4k dev over the T ratio r in [1, 1/2, 1/4];
  - keep it only if it is at least 0.3 points above r = 1, otherwise r = 1;
  - a tie goes to the earlier ratio in the list, so an exact tie with r = 1 stays at r = 1.
  - The script writes logs/jobs_e81b.txt and refuses to overwrite it.
- **The follow-ups** (logs/jobs_e81b.txt), only for a pick r != 1 of the refined and bge-large agents:
  - STAR and MultiWOZ: the pick at 16,000 clauses (T = r x 16,000), at seeds 1 and 2;
  - ABCD: the 4k pick stands, run at seed 2;
  - bge-large: also its seed-2 row at T = clauses, since none exists on dev.
  - The C4 agent gets no follow-up.
- **The T = clauses references** at full size, seeds 1-2:
  - refined: e82's Stage A rows;
  - bge-large: e74d (STAR 16k), e75 (ABCD 4k) and e77 (MultiWOZ 16k) at seed 1, plus the new seed 2;
  - C4: e74c and e78, and e77 and e79.

## Bars (fixed now, before any e81 number)

- **B1, "T helps the reader"** for one (agent, dataset): the pick's full-size mean of seeds 1-2, minus T = clauses'
  mean, is at least +0.5. The statistic is the best-dev snapshot, as in e74, e75 and e82.
  - It is judged per dataset. The claim "T helps this agent's reader" needs B1 met on all three datasets.
  - A pick of r = 1 means "T does not help" on that dataset.
- **B2, "T closes the gap"** for the refined agent on MultiWOZ: the TM at its pick (16k, mean of seeds 1-2) minus LR
  on the same bits (0.3643) must be at least half of today's gap.
  - Today's gap is -5.354, so the bar is -2.677, the exact half that e81_pick.py computes. The design's -2.675 is
    the same rule on the rounded gap.
  - If the pick is r = 1, today's TM stands and B2 is not met.
- **Reported beside every B1 and B2 line, with no rule:**
  - the same difference at the final epoch and as the mean of the last 10 logged evaluations;
  - TM minus LR on the same bits, for every agent and dataset where LR exists.
- **How a pass is written (an addition to the design).** This applies when B1
  or B2 is met on the best-dev statistic but not on the last-10 mean:
  - for B1, the last-10 difference printed beside it is below +0.5;
  - for B2, the script's "B2 beside it" line says "not steady". That means the TM's last-10 mean minus LR is below half
    of today's last-10 gap; today the gap is -8.029, so below -4.014.
  - In either case the write-up says "met on the best-dev snapshot only, not steady across epochs", and no registered
    test is planned from it without a fresh dev check.
- **What a miss means:**
  - T is not what holds the reader back.
  - The TM-LR gap on MultiWOZ stays as a failure-map line, with today's numbers.
  - The next reader step is planned separately.

## Checks and tests (run before the freeze; CPU)

- **scripts/e81_pick.py:** has the stages smoke, screen and final. On every row it reads, it checks:
  - the dataset, encoder, parts, clauses, T, s and seed;
  - every other TM setting, which must equal BASE_CFG (the settings of all 21 reference rows: 40 epochs, batch 128, 256
    states, eval every 2, and so on), plus lex_k 1000 and init half;
  - the a3B table hash on refined rows;
  - e81 rows written after this plan's freeze;
  - exactly one row per tag and seed;
  - one dev bits hash per (agent, dataset);
  - LR rows on those same bits, with the wide grid;
  - B2's constants against the rows.
- **tests/test_e81_pick.py** runs it on made-up rows in temporary folders and covers:
  - the smoke stage (pass, a lost --T flag, other bits);
  - the 0.3 margin (exactly +0.3 taken, +0.29 not), both ties, and no follow-up for the C4 agent;
  - the exact follow-up job lines and the refusal to overwrite;
  - B1 and B2 (met, not met, and a pick that stays at T = clauses), with the epoch columns, the B2 last-10 line and
    TM - LR, all checked against values the test computes;
  - six refusals.
  - Result: PASSED. The output is in logs/test_e81_pick.out.
- **The real reference rows today:** all 24 reads of the 21 reference rows pass every check of e81_pick.py, and B2's constants equal the rows.
  This was a read-only script outside the repository.
- **Comparability of the references:**
  - Every reference row has the same TM settings apart from clauses, T, s and seed.
  - Every row of one (agent, dataset) has one dev bits hash, the existing LR rows included. The hashes are listed
    below.
  - Since e74's freeze (2026-10-19 11:46:21 UTC), src/ has changed only by the new file src/dl/count_bits.py, which
    these views do not use.
  - abcd_explore.py and abcd_confirm.py changed after e74's freeze. Their help text says MultiWOZ was added on 10-19
    for e77. They are unchanged since e82's freeze (2026-10-21 01:31:09 UTC).

## Hashes (SHA-256, from sha256sum at 05:40 and 05:45 UTC; the a3B and C4 MANIFEST files check OK)

| File | SHA-256 |
|---|---|
| scripts/e81_pick.py | bc9fdcd91924cb31e667d34183b252e082ed28a9e4fb327063217341378ebda4 |
| tests/test_e81_pick.py | e56004afb0dbb945079a16bb6508599a371eee98ea815d9ff3106cae3118b3da |
| logs/test_e81_pick.out | 08ccbb6a4c5bfeb044cb0d10a57364356063aa21bebbbabece437fdf89085a81 |
| logs/jobs_e81_smoke.txt | 0cbd896ce00ef1348a87f8b4b848f802a4b6a852b898de6626e685dfb1ed1753 |
| logs/jobs_e81a.txt | dc599b1a178332403cd35e1188242f46aceeb67869b2f59dd72c41b30491b9cc |
| data/tables/c4words_a3B_v1/table.npz | 4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855 |
| data/tables/c4words_a3B_v1/vectors.npy | 21d0c4fb148498531a6523f9635401a6215ef4dd5c8cd4b7ca8ce6e7b80f8e14 |
| data/tables/c4words_a3B_v1/words.txt | 49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29 |
| data/tables/c4words_a3B_v1/tokenize_ref.py | 2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803 |
| data/tables/c4words_a3B_v1/README.md | 35ae923e408084ff8d9512a06eea5f481aff0037da0e23d06b111286e6dbd85f |
| data/tables/c4words_a3B_v1/MANIFEST.sha256 | 1d1a00ad74634f77d38894e4a476c1717494c94a76450210518ec9bc8ed267f3 |
| data/tables/c4words_S1000_v1/table.npz | e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a |
| data/tables/c4words_S1000_v1/vectors.npy | 9263bf09383de22806e320a1988de57816dbd62bfa1cc692b0fb7077e44d267c |
| data/tables/c4words_S1000_v1/words.txt | 49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29 |
| data/tables/c4words_S1000_v1/tokenize_ref.py | 2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803 |
| data/tables/c4words_S1000_v1/README.md | fafc3a213c666f08e52177911732f4c53973085f6c81e0c9276d7ec43daf9454 |
| data/tables/c4words_S1000_v1/MANIFEST.sha256 | f49b934ad1fe8d520af9f87ef37f145aff5483f1c55068e2c5309747f8bd734e |
| data/star/examples_train.json.gz | 47735157f3e0035c59c8f88005ebe5457678083d60e653fbdca2f4df45c023df |
| data/star/examples_dev.json.gz | e562736dd9afa1b781d48bf78f43f23b1b8ceda17b3ddbf01557c59f0c8efb18 |
| data/abcd/examples_train.json.gz | c9e0f9b58a13a3dd9eff5821177cd74373cd1816465445a32bb6cc131933835f |
| data/abcd/examples_dev.json.gz | 5ccc737f5be9086060de4c28f560f1aa9b94566ed9599a98f44212c0e6cf2f17 |
| data/multiwoz/examples_train.json.gz | 0e20a31434de10a5248f0b10ab01e7474ceb5e020cd5cbc7048bfe8f5db69e9b |
| data/multiwoz/examples_dev.json.gz | 3414cc8a6ecc64aafa0df070748b9d502a12e7996420b84a12df1b094a89fcd2 |
| data/multiwoz/labels.json | ed3379da965e969daa1f5e7982c42bb04dc61ee0e184a93327dc2e8bf02dede7 |
| data/emb/{star,abcd,multiwoz}__none.npz (all three) | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| data/emb/star__bge-large.npz | d3a0b1b952c20444d9fffe142193bda409ecf02d5ae3bd9a5e9f7842229e27bb |
| data/emb/abcd__bge-large.npz | 21c6243324a7152165281f9da6b277e7adeda522986b860e5a1b6d909333b134 |
| data/emb/multiwoz__bge-large.npz | 324f6d626675c5a3600125561e12784d8b8df19519893644a74b4f3cc5ebe421 |

**The dev bits hash every row of an (agent, dataset) must have.** T and the TM seed do not change the bits, since the
view's seed is fixed at 0.

| Agent | STAR | ABCD | MultiWOZ |
|---|---|---|---|
| refined | 5d3cd86eb18788b28165baaaaeb9c87e4323eae90b56be62820725ebe4566254 | 3222f78c87af48579d5e214eb2ebf75b1a177e633851b8417985ed6bfb44d110 | 612b8f0b75e32a30b50493eaedc1eec9d6bac6893d48075144a1f48fad2e23a3 |
| bge-large | 65b2bb544e9ccc7454f2c0efc5a9c81d92539034fccc51588117c3c94c54c521 | f5d8695629502e6d112e69dcf0be970022336f969a43f733e94334c09192136b | b068b64787582fc8c00dd8a446cb415fcb9737d3069e7111f092503bf7bf2797 |
| C4 | 3fc00b173cf402a56f152dfcea2a5f9e26b02a2d2100ba7dec76fa063851d4f7 | 56297e10edd0532f15823b602508976eaa28fbeb5fd0f14893eb862620014d33 | 311e8933d2724bf8fc3ee1a5cef412010bd3a580c076217e3a964bcd286d68ba |

## GPU time (alone, one job at a time)

The estimates come from the queue logs of the same job types at T = clauses (e74c, e74d, e75, e77a, e77b, e82, e82c,
e82r3a, e82r3b). A lower T may change the speed.

| Stage | Jobs | Minutes |
|---|---|---|
| Smoke | 3 tiny jobs | about 3-5 |
| Screens, refined | 6 jobs: STAR 6.1-7.3, ABCD 3.1-4.5, MultiWOZ 5.8-8.5 min each | 30-41 |
| Screens, C4 | 6 jobs: STAR 5.7-6.1, ABCD 2.6-3.5, MultiWOZ 5.3-7.6 min each | 27-34 |
| Screens, bge-large | 6 jobs: STAR 2.1-2.5, ABCD 0.7-0.9, MultiWOZ 1.9-2.9 min each | 9-13 |
| LR | 5 jobs, 2.3-2.5 min each on MultiWOZ | about 12-15 |
| Follow-ups, worst case (every pick moves) | refined STAR 2 x 23-27, MultiWOZ 2 x about 27, ABCD 1 x 3-4.5; bge-large STAR 3 x 8, MultiWOZ 3 x 7-9, ABCD 2 x 1 | about 150-165 |
| Follow-ups, best case (no pick moves) | none | 0 |
| **In all** | | **about 1.4 h to 4.5 h** |

## Report, verification and logging

1. `scripts/e81_pick.py --stage smoke | tee logs/e81_smoke_check.out`, after jobs_e81_smoke.txt.
2. `scripts/e81_pick.py --stage screen | tee logs/e81_pick_screen.out`, after jobs_e81a.txt.
3. `scripts/e81_pick.py --stage final | tee logs/e81_pick_final.out`, after jobs_e81b.txt, or straight after the screen
   if no pick moved.
4. Each output is saved with its sha256. Then the verification procedure, then the entry in the record of results. The
   write-up goes to docs/results.md, and a miss also goes to FAILURE_MAP.md.

## Selection caveats

- T is picked among 3 values on the same dev splits.
- The TM's numbers are best-dev snapshots, and LR's C is chosen on dev, so both sides are optimistic.
- On MultiWOZ, LR picked the grid's lowest C (0.001) on both tables. That is an edge pick, so LR may be under-tuned
  there and the gap understated. The grid is not extended now.
- MultiWOZ's best-dev margins are within one evaluation's swing (see "What was known"), hence the epoch columns and the
  "how a pass is written" rule.
- e81 reads no test split.
