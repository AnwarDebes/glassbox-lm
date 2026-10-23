# e86: clause dropout 0.5 for the C4 agent on MultiWOZ, STAR and ABCD, and one negative-sampling setting (dev only)

Written 2026-10-23, before any e86 number, after R24 ended (logs/R24_end.log). No test split is read: MultiWOZ test
stays at its 1st use (R24), and STAR and ABCD test are not touched.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**The question (fixed now):** does clause dropout 0.5 lift the C4 agent's reader on each dataset's dev split, and so
narrow the TM's gap to LR on the same bits? And can random negatives with dropout serve as one negative-sampling
setting for all three datasets?

**Why:**
- On MultiWOZ test the TM is below LR on the same bits for all three agents: C4 -1.55, refined -2.49, bge-large -1.89
  (R24, no rule).
- e83 left combinations open (logs/plan_e83_learning_dynamics.md). On MultiWOZ dev (refined agent, 4k, seed 2, hardest
  negatives), drop 0.5 gave +2.02 at best-dev and +2.01 on the last-10 mean (logs/e83_pick_screen_noL8.out). Random
  negatives gave +3.89, and the 50/50 mix only +0.96.
- Random negatives cost STAR -11.91 and ABCD -4.00 on dev (e83, refined agent), so R24's rule is MultiWOZ-only.

## Cells (the C4 agent, no neural network at any stage)

- **The agent:** table c4words_S1000_v1 and view mem, words, xcust, xrecent; 40 epochs; start half; seeds 1-2.
- **Each dataset's size and s, fixed now from earlier dev work:**
  - MultiWOZ: 16k, T 16,000, s 8 (e85, under random negatives);
  - STAR: 16k, T 16,000, s 16 (e74c, as R22);
  - ABCD: 4k, T 4,000, s 16 (e75, as R23).
- **The cells:**

| Dataset | Reference (no dropout) | Dropout cell | One-setting cell |
|---|---|---|---|
| MultiWOZ | random negatives: e84's rows e84_multiwoz_C4_c16000_s8_nrand, seeds 1-2 | random + drop 0.5 | (the dropout cell) |
| STAR | hardest: e86_star_C4_c16000_s16_ref (fresh) | hardest + drop 0.5 | random + drop 0.5 |
| ABCD | hardest: e86_abcd_C4_c4000_s16_ref (fresh) | hardest + drop 0.5 | random + drop 0.5 |

- **The MultiWOZ reference is not rerun.** e84's rows come from the same code path. R24's checker found that seed 1
  reproduces them exactly: 0.35004756080989263, the R24 C4 arm's seed-1 dev accuracy, in logs/r24_check.out. The
  integrity part of its review found the same for all 12 seeds that have a dev row.
- **STAR and ABCD get fresh references.** Their earlier rows ran under R22 (2026-10-19) and R23. Each fresh seed 1 is
  reported beside R22's or R23's seed-1 dev row, with no rule. A difference of 0.00 shows that the path is unchanged.
  - Any other difference means the path changed. It is reported and explained before any e86 result is used.
- **Dropout** is tm.py's clause_drop: each clause is left out of each minibatch of 128 with probability 0.5, during
  training only. It is set through scripts/knob_run.py --clause-drop 0.5, the path e83 tested.
- **The jobs:** logs/jobs_e86.txt, 14 jobs: MultiWOZ 2, then ABCD 6, then STAR 6. Each saves its dev predictions to
  results/explore/e86_preds/.

## Rules (fixed now, before any e86 row)

- **The measure.** In points, for every cell, use the mean of seeds 1-2 at best-dev and on the last-10 mean (the last
  10 of the 20 evaluations).
- **Dropout per dataset.** Drop 0.5 is kept on a dataset if the dropout cell minus the reference is at least +0.5 on
  both readings.
  - A kept dataset's later C4 dev work uses drop 0.5, written "chosen on <dataset> dev".
  - On the other datasets it stays off.
- **E86.1:** drop 0.5 is kept on all three datasets. Then it is one dropout setting for the C4 agent.
- **E86.3, one negative-sampling setting:** random + drop 0.5 minus each dataset's best setting is at least -0.5 on
  both readings, on all three datasets.
  - The best setting is the one after e86's own keep rule: the dropout cell where dropout is kept, else the
    reference. So a setting that loses more than 0.5 against a kept dropout cell does not pass. (A fix made
    to the reviewed draft fc2873c6 before the freeze.)
  - On MultiWOZ, the one-setting cell is the dropout cell. Where dropout is kept there, its difference is 0.
  - Met: the C4 agent can run with one setting (random negatives + drop 0.5) on every dataset. Not one is chosen per
    dataset.
  - Not met: the negative-sampling rule stays per dataset (random on MultiWOZ, hardest on STAR and ABCD).
  - **Caveat:** on STAR and ABCD the random + drop cells use s 16, which was tuned under hardest negatives. A "not met"
    may partly reflect an untuned s.
- **Closed as negative:** if no dataset keeps dropout and E86.3 is not met, e86 is a dead end. It is logged in
  FAILURE_MAP.md.
- **What a met rule does not give:**
  - No test claim and no "better than".
  - Before any agent comparison, the bge-large and refined agents get the same setting check (e84's lesson: a rule
    tuned on one agent flatters it).
  - A test would need its own frozen registration, with MultiWOZ test's 2nd use declared.
- **Reported with no rule:**
  - every seed, and the final epoch;
  - TM minus LR on the same bits per cell (the existing LR dev rows, which dropout does not change):
    - MultiWOZ: dry_e77mwoz_lr_mem_c4, 0.3586, C 0.001 (LR's best over 1e-4 to 100, e81x);
    - STAR: dry_R21star_lr_mem_c4, 0.6558, C 0.01;
    - ABCD: dry_R21abcd_lr_mem_c4, 0.8119, C 0.03;
  - the fresh references' seed 1 beside R22's and R23's seed-1 dev rows.

## What was known at writing

- **The C4 reference rows (e84, MultiWOZ, random, 16k, s 8):** best-dev 0.35005 and 0.34991 (mean 0.3500); last-10
  0.34263 and 0.34359 (mean 0.3431); TM minus LR -0.86 (logs/e84_check.out). e84b's seeds 3-4 gave 0.34339 and
  0.35018.
- **Seed noise:** over seeds 1-4, best-dev spans 0.68 points, so the +0.5 bar on 2-seed means is about 1.5 standard
  deviations of a difference. A kept dataset is a dev direction, not a claim.
- **STAR and ABCD, the C4 agent:**
  - R22's and R23's seed-1 dev rows: 0.6545 (STAR, 16k, s 16) and 0.8417 (ABCD, 4k, s 16);
  - the TM minus LR on test: STAR -0.09 (R22); ABCD +1.76 (R21, s 8);
  - on dev against the LR rows above: STAR about -0.13, ABCD about +3.0, from seed 1.
  - There is little headroom to LR on STAR and none on ABCD.
- **Clause dropout before:**
  - On CLINC150's dense MiniLM bits it did not help (94.2 to 94.5 against 94.6 to 94.9; FAILURE_MAP Part 3).
  - Drop Clause (Sharma et al., AAAI 2023): +3.6 on R52 at p 0.75, with a per-epoch mask (from the paper as quoted
    in plan_e83_learning_dynamics.md:62, a line number of that text as registered; not rechecked here). tm.py's is per minibatch.
- **What the numbers predict (a guess, not a rule):**
  - MultiWOZ: some gain, perhaps less than e83's +2.02, because random negatives may cover part of the same effect.
  - STAR and ABCD: unclear.
  - E86.3: unlikely, since dropout would have to recover STAR's -11.9.

## Checks, commands and GPU

- **The checker:** scripts/e86_check.py. Its test, tests/test_e86_check.py, PASSED (logs/test_e86_check_2.out; the
  first run, logs/test_e86_check.out, was before the comparator fix). It covers:
  - kept on all three at exactly +0.5;
  - kept on MultiWOZ only;
  - a gain at best-dev only, which is not kept;
  - E86.3 at exactly -0.5 against STAR's kept dropout cell, and failing against the reference;
  - the kept dropout cell as E86.3's comparator (+0.5 over the reference, -1.5 under the kept cell: fails);
  - the dry mode;
  - seven refusals.
- **The run** (e86_chain.sh, in the project notes, outside the repository) is started by hand, on a recorded decision. It handles every stage:
  - before each start, the start rule: the given process gone (ps -p), then 3 empty-GPU checks in a row;
  - `dry`: logs/jobs_e86_dry.txt (7 tiny jobs, 200 clauses, T 50, 2 epochs, seed 1, dev), then `e86_check.py --dry`;
  - `run`: needs the dry check passed; the 14 jobs through scripts/seq_queue.sh, then `e86_check.py >
    logs/e86_check.out`;
  - `both`: dry, then run.
  - End lines go to logs/e86_dry_end.log and logs/e86_end.log ("e86 end: <branch> <UTC> exit <code>").
  - Its stub test (test_e86_chain.sh) PASSED (test_e86_chain.out).
- **GPU estimate**, from earlier queue logs:
  - MultiWOZ C4 16k: about 19 min a job (R24_queue.log);
  - STAR 16k: about 20 min (e83c_queue.log);
  - ABCD 4k: about 3 min (R23_queue.log, e83c_queue.log).
  - So about 2 x 19 + 6 x 20 + 6 x 3 = 176 min, plus the dry run (about 5 min).
  - Dropout may change the speed.
- **Resources:** one job at a time, alone on the GPU, never beside a Qwen server; 2 threads, DL_GPU_FRAC 0.25 (the job
  lines).

## Hashes (sha256sum, read 2026-10-23 between 00:50:08Z and 00:57:29Z)

| File | SHA-256 |
|---|---|
| scripts/e86_check.py | c9828c1bef7dbc0af5d9a76572af4e6d9dce2ed0030742aa342c4bdec7396670 |
| tests/test_e86_check.py | 202ccda8580cc097ed6fa2fc3ff2cba4c63ad0f66e9c55a7223ee7f5b9ac1bad |
| logs/test_e86_check_2.out | 81d743e32a7aaabdad49a8f5c7d830108e89eee5f142823d9e7b8a8da44690ac |
| logs/test_e86_check.out (before the fix) | aa79a3d505712e7c61172d0ad2162f39bfa9f7772c72bb09bc150fae53c46a2c |
| logs/jobs_e86.txt | e3a36fa7d6483b9e33eedb2aa4998fb4df8f2d94a7802595e2168ba0ec30bc00 |
| logs/jobs_e86_dry.txt | 6869f43d281972810650f01d389a1a726b10fd1964b1581c6b48628a3187fc75 |
| scripts/knob_run.py | 113cb74eca36d1f9c3fbea889a2fe8aad2c1ad91878af9f17d93cffa14d65e4e |
| scripts/abcd_explore.py | ca6b017c128a087bceac7bd606d96d602c49fdead3081c7ceb43a405882d3505 |
| scripts/table_run.py | 0af00bd556741c8b1e89aa03fa047372e4636f1d4ab91e0e089401217112fb29 |
| src/dl/tm.py | f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424 |
| data/tables/c4words_S1000_v1/table.npz | e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a |
| data/tables/c4words_S1000_v1/MANIFEST.sha256 | f49b934ad1fe8d520af9f87ef37f145aff5483f1c55068e2c5309747f8bd734e |
| data/multiwoz/examples_train.json.gz | 0e20a31434de10a5248f0b10ab01e7474ceb5e020cd5cbc7048bfe8f5db69e9b |
| data/multiwoz/examples_dev.json.gz | 3414cc8a6ecc64aafa0df070748b9d502a12e7996420b84a12df1b094a89fcd2 |
| data/star/examples_train.json.gz | 47735157f3e0035c59c8f88005ebe5457678083d60e653fbdca2f4df45c023df |
| data/star/examples_dev.json.gz | e562736dd9afa1b781d48bf78f43f23b1b8ceda17b3ddbf01557c59f0c8efb18 |
| data/abcd/examples_train.json.gz | c9e0f9b58a13a3dd9eff5821177cd74373cd1816465445a32bb6cc131933835f |
| data/abcd/examples_dev.json.gz | 5ccc737f5be9086060de4c28f560f1aa9b94566ed9599a98f44212c0e6cf2f17 |
| results/explore/dryrun/dry_e77mwoz_lr_mem_c4/seed0.json | f3abfcb8761ebb2a0d202a109731df4cddabb62916d43bdbfe2be25340b587cb |
| results/explore/dryrun/dry_R21star_lr_mem_c4/seed0.json | d64cbb4b8d4e2e83fafcaef9df43242ccd4e32b03b4a30004b72f48ccf213e8c |
| results/explore/dryrun/dry_R21abcd_lr_mem_c4/seed0.json | 98646174173b77ad533139f4dffded944eb4ca7782dbf278a904725f931e7f78 |
| results/confirm/R22star/tm_mem_c4_s16/seed1.json | dc637ac1d48a11fc1deeec50af2bb878175893bc4946dfe184407f537163318f |
| results/confirm/R23abcd/tm_mem_c4_s16/seed1.json | 4ba1dc75f8366a7aea3ed8a1283addaec4e3fc4fb3d3390b7f667e0c1707ced4 |
| results/explore/e84_preds/e84_multiwoz_C4_c16000_s8_nrand_seed1.npz | 20a3325f3b25101d3d4ca93d81b30c86560189d4dd606b24ef421f39058e059a |
| results/explore/e84_preds/e84_multiwoz_C4_c16000_s8_nrand_seed2.npz | 5abc12e1a21b9704775e05f3b72c94b15220390fba2a2ee1fc95ec3b6bcb4a32 |
| logs/e84_check.out | b0716f5424655bbd337e5104240d05c8d89e552bc5a1499c95ea8b821223a165 |
| e86_chain.sh (project notes, not published) | 90e169dab06aab657e4e82de3cd4fec35a992e0870a2856cb79431343328cb9a |
| test_e86_chain.sh (project notes, not published) | 6029b16b72b3b0a0340e5aaabc62a01541238917efa931c2ef314aea48f34dbd |
| test_e86_chain.out (project notes, not published) | 59cb83893e3beb772e9fa7ea5c115307f9370460737159661aa96a9d424ba075 |

The checker also checks every row's dev bits (the C4 agent's on each dataset, as in its DS table) and the table hash.
