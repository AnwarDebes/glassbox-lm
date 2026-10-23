# R25 registration (draft, revision 2): the agent on the teacher-refined table (3a) against the agent on the C4 counting table (STAR and ABCD test)

**Status:** draft, not frozen.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged. This draft was never frozen.

**Draft history** (times read with date -u):
- **Rev. 1 (d9eb6ef9, written 01:04:32 UTC):** the first draft. Kept in logs/archive/R25_draft1_d9eb6ef9.md.
- **Rev. 2 (this, from 01:28 UTC):** the decisions of 01:21 UTC on the findings of two reviews
  (judged by an AI model, not a person; R25_review_2026-10-21.md fa111d9d, in the project notes, not published).
- **Rev. 2a (03:51 UTC):** one correction after the verification of e82's R4 (logs/plan_e82_note2_2026-10-21.md):
  "no labels" was wrong for ABCD's full field. No rule changed.
- No e82 dev row existed at either revision. So no rule here was shaped by an e82 number.

## Preconditions (all before the freeze; dev only; plan logs/plan_e82_refined_table_agent.md)

1. **R4 takes the table:** no dataset loses more than 0.5 on dev, on the mean of seeds 1-2.
2. **e82-R3's picks are fixed** (`scripts/e82_pick.py --stage final`). They give s* for STAR and for ABCD.
3. **The precedence:** R25 runs only if the refined agent FAILS R24's MultiWOZ dev gate, which needs at least -1.0
   against the bge-large agent at 16k, seed 1. If the gate passes, R24-refined is the one registered test, and R25 does
   not run. This is decided from dev alone, before any test read. Running both needs a recorded decision.
4. **The dev go-bars** (fixed 2026-10-21 01:21 UTC, before any e82 dev number): the refined agent at s* minus the C4
   agent at s 16, mean of seeds 1-2.
   - **STAR** is tested only if this is at least +1.0 on STAR dev.
   - **ABCD** is tested only if it is at least +0.7 on ABCD dev.
   - A dataset below its bar is a dev-only failure-map line, and its test split stays unread. If neither passes, R25
     does not run.
5. **R5:** 250 of 250 guarded proofs on the five refined ABCD models at s*. These are e82's Stage C models; the test
   runs reproduce them, and seed 1's dev accuracy is checked equal.
6. **The dry runs and the checks** below pass.

## The question

Step 3a of the text-understanding line refitted the C4-word table's rows so that a C4 sentence's mean row matches a teacher's sentence
embedding (c4words_a3B_v1). Does this better text reading help the agent? The comparison is the agent on the refined
table against the agent on the C4 counting table, each at its own dev-tuned s, on STAR test and ABCD test.

## Rules (test; fixed 2026-10-21 01:21 UTC)

- **HEADLINE.** Each difference is the refined agent minus the C4 agent, with each agent at its own dev-tuned s.
  - **R25.3, STAR:** `tm_mem_a3B_s<sSTAR>` minus `tm_mem_c4_s16_R22` is above 0, with the whole 95% interval above 0.
  - **R25.4, ABCD:** `tm_mem_a3B_s<sABCD>` minus `tm_mem_c4_s16_R23` is above 0, with the whole 95% interval above 0.
- **The general claim** "the refined table helps the agent" needs BOTH R25.3 and R25.4 met. If only one is met, it is
  reported for that dataset alone, with the other's interval beside it.
- There is no combined "R25 met" line, and no claim rests on any one rule being met.
- If s* differs from s 16, the wording is "each agent tuned by the same rule", not "the table helps" at equal s.
- **R25.1 and R25.2 are withdrawn as rules** (01:21 UTC). The bge-large comparisons are decided largely by
  R22 and R23 on these same splits, so they are readings with no rule. See "Readings".
- **Statistic,** as R21-R23: per test turn, each arm's correctness is averaged over its seeds. The difference's 95%
  interval comes from a bootstrap over the test dialogues (STAR 611, ABCD 1,004), 10,000 resamples, seed 0
  (`scripts/report_abcd.py`, 7d0ca5ca).
  - Beside every pair, a two-level interval with no rule: seeds resampled within each arm, then dialogues; 10,000
    resamples, seed 0 (`scripts/report_twolevel.py`, written before the freeze).
  - A rule that is met only on the registered interval is reported with the seed-inclusive interval beside it.
- **Multiplicity:** no correction is applied, and each rule stands alone.

## Readings (no rule; labelled "informed by R22 / R23")

- **The refined agent minus the equally s-tuned bge-large agent** (`tm_mem_bge_s16_R22` on STAR, `tm_mem_bge_s24_R23` on
  ABCD). Each is reported both as a non-inferiority reading (at least -2.0) and as a superiority reading (interval above
  0).
- R25 gives no "better than bge-large" wording. A registered "better than" stays with R24 on MultiWOZ, behind its dev
  gate.
- **Predicted by the known test numbers** (so these are not independent):
  - STAR: (refined minus bge-large) = (refined minus C4) + 1.64. The superiority reading is met whenever refined minus C4
    is above about -0.8 on test.
  - ABCD: (refined minus bge-large) = (refined minus C4) - 0.92. The non-inferiority reading is met unless refined minus
    C4 is below -1.08.
- **R19's potion agent** (the same teacher, s 8, unequal tuning; STAR 0.6271, ABCD 0.8301 on test) is shown beside.
- **The guards** on the five test-run ABCD models (the same models as R5's), and their open-loop and closed-loop cost on
  dev.
- **AST-weighted accuracy** on ABCD.

## Arms (test; R21's settings; best-dev snapshot; the results go to results/confirm/R25star/ and R25abcd/)

| Dataset | New arm (view B on c4words_a3B_v1; newtable_run.py wrapping abcd_confirm.py) | References, copied (no new test run) |
|---|---|---|
| STAR | `tm_mem_a3B_s<sSTAR>`, 16,000 clauses, seeds 1 to 5 | `tm_mem_c4_s16_R22`, `tm_mem_bge_s16_R22` (from R22star) |
| ABCD | `tm_mem_a3B_s<sABCD>`, 4,000 clauses, seeds 1 to 5 | `tm_mem_c4_s16_R23`, `tm_mem_bge_s24_R23` (from R23abcd) |

- Seed 1 must reproduce e82-R3's dev row at s* exactly: the 16k STAR row and the 4k ABCD row.
- A dataset whose go-bar fails gets no arm.

## Provenance and caveats (in every report line)

- **The refined table** is "network-free at run time, teacher at training time".
  - The teacher is bge-base-en-v1.5@a5beb1e3 ([CLS], L2-normalised, MIT). It was fitted on 1,999,932 C4 sentences, with
    a held-out cosine of 0.8116.
  - Never write "no neural network at any stage" for it; that phrase stays with the C4 agent, which is reported beside
    it.
- **Same family:** the teacher is the smaller sibling of the reference agent's encoder (bge-large-en-v1.5). It is also
  potion's teacher.
  - With lambda* 0.01, the rows are mostly set by the teacher.
  - So this is "a replication of potion's recipe on the C4 word list, not a new method".
- **The teacher's training data cannot be checked.** It may include STAR, ABCD, MultiWOZ or intent-set text. That could
  inflate the refined table's gain over the counting table (R25.3/R25.4). It does not inflate a comparison with potion.
- **The view-choice caveat (on the readings):** view B was chosen and developed on these datasets' dev splits:
  - e60/S2: B against A against A+B;
  - e78: B + xcustn, not taken;
  - e79: A+B on the C4 table, not taken.
  - The bge-large agent keeps R18's / R9b's fixed view. Only s is tuned equally.
- **Only 3a enters the agent.** 3c's paraphrases and 3b are not in it.
- **The network-free check on the new path:** tests/test_newtable_path.py (logs/test_newtable_path.out). No
  sentence_transformers, model2vec or transformers module is imported; the only encoder cache opened is *__none.npz;
  the only table opened is c4words_a3B_v1.

## Use of the test splits (declared)

- **STAR test, 7th use.** The six earlier uses are R18, R19, R19's post-hoc LR refit at C 0.01, R20 arm 1, R21 and R22.
  All were pre-registered except R19's post-hoc LR refit.
- **ABCD test, 15th use.** The 14 earlier uses run through R23, and all were pre-registered.
- **A text-only read, declared.** The refined table's fit filter (a3_sentences.py, through W1's eval_texts) read the STAR and ABCD
  dev and test example files, text fields only (ABCD's full field carries the earlier actions as "action: <button>"
  strings, which were read too, for removal only). It dropped matching C4 sentences: 228,122 by key and 3,912
  by 8-gram, pooled over all evaluation sets. This is removal only, so it is conservative.
- Each reuse weakens trust in a split a little, and that is stated.

## What was known at the freeze

- **Every earlier test number on both splits:**
  - R9b; R18;
  - R19, with its post-hoc numbers, among them the potion agent's 0.6271 / 0.8301;
  - R20;
  - R21: potion minus C4 was +0.10 [-0.45, +0.65] (STAR) and +0.29 [-0.17, +0.74] (ABCD). That is the prior for
    R25.3/R25.4;
  - R22: C4 minus bge-large +1.64; tuning +2.70; TM against LR -0.09;
  - R23: C4 minus bge-large -0.92.
- **That the references' per-turn test predictions are known,** and the readings' predicted outcomes (above).
- **Other dev and val results:**
  - W1-T's test result (the text-understanding line);
  - the r3 picks of the text-understanding line and the 3c verdict (val);
  - every MultiWOZ dev row and the R24 gate's outcome (-3.21 for the C4 agent);
  - every e72-e82 dev number, including e82's R4, e82-R3 and Stage C;
  - the dry runs'.
- **Every view tried:** B, A and A+B in e60; B + xcustn in e78; A+B on the C4 table in e79; B on the refined table.
- **Selection is optimistic:**
  - the table passed a dev gate (R4 and the go-bars);
  - s was picked on the same dev splits among 4 values;
  - dev numbers are best-epoch numbers.

## Code since R23's freeze (diff_frozen.py against registrations/R23_tuned_s_abcd.frozen.json; checked at the freeze)

- **scripts/abcd_confirm.py:** 0ae66772 to 8c86665a, one line, "multiwoz" added to the --dataset choices (e77; the
  pristine copy and diff are logs/pre_e77/abcd_confirm.py and diff_abcd_confirm.txt 26e0f832).
  - abcd_explore.py changed the same way.
  - r25_check asserts that the diff is exactly that line.
- **New on the path:** scripts/newtable_run.py (f98ce98e), and tests/test_newtable_path.py.
- **src/dl:** the files on the path are unchanged. count_bits.py is new but not on R25's path.
- **The venv:** no site-packages entry newer than 2026-10-19T20:30Z (rechecked at the freeze). torch 2.6.0+cu124 and
  a pip freeze hash are recorded at the freeze, with the GPU driver.
- **The C4 references reproduce:** e78 reproduced the seed-2 dev accuracy of both C4 reference arms exactly (R22star
  0.6579328314997105, R23abcd 0.8422909880564604).
- **A bge-large dev reproduction before the freeze** (ABCD s 24 seed 1 must give 0.8471769815418024), with the bge-large
  caches' hashes unchanged (star d3a0b1b9, abcd 21c62433).

## Hashes (full SHA-256; checked at the freeze and by scripts/r25_check.py)

- The a3B table:
  - table.npz 4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855;
  - words.txt 49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29;
  - tokenize_ref.py 2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803;
  - vectors.npy 21d0c4fb148498531a6523f9635401a6215ef4dd5c8cd4b7ca8ce6e7b80f8e14;
  - README.md 35ae923e408084ff8d9512a06eea5f481aff0037da0e23d06b111286e6dbd85f;
  - MANIFEST.sha256 1d1a00ad74634f77d38894e4a476c1717494c94a76450210518ec9bc8ed267f3.
  - r25_check compares every new seed's static_info[xcust/xrecent].table_file_sha256 with the literal table.npz hash.
- The examples:
  - STAR train 47735157f3e0035c59c8f88005ebe5457678083d60e653fbdca2f4df45c023df, dev
    e562736dd9afa1b781d48bf78f43f23b1b8ceda17b3ddbf01557c59f0c8efb18, test
    f9cc8f150b25d94da38930e55e4316f68cac3420da58d07fc67f9990853ed556;
  - ABCD train c9e0f9b58a13a3dd9eff5821177cd74373cd1816465445a32bb6cc131933835f, dev
    5ccc737f5be9086060de4c28f560f1aa9b94566ed9599a98f44212c0e6cf2f17, test
    190312c67b2258b4353b4f7a95bcd8edff6ddc1798e81ece9a3fa0d13d930c4d.
- The empty caches: fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335.
- The reports to reproduce:
  - report_R22star.json ca2f182805401a9db27f5aa3ae47d46936f342764b1871555e314d6f52b5e7a6;
  - report_R23abcd.json 3c6c0c1d1015bc7eb778ed5f0ff74e8f6ba4fa57c3a8bd390acb7345ddd56921.
- The references' test-bit hashes: C4 d3f682f2 (STAR) and fb13341b (ABCD); bge-large 05f28f95 (STAR) and 0fc386cd
  (ABCD). The full values are in r25_check.py.
- Filled at the freeze: jobs_R25.txt, the dry-run output, the refs list, r25_refs.py, r25_check.py and
  report_twolevel.py.

## Checks (scripts/r25_check.py, adapted from r23_check.py and written before any test number)

- **The freeze:** the registration text is unchanged; the files on the path are unchanged except the declared
  abcd_confirm.py line; newtable_run.py f98ce98e is on the path.
- **The jobs:** the seed files are exactly jobs_R25.txt's, written after the freeze. The new arms' static_info names the
  a3B table file.
- **The cfg:** the new arm's equals the C4 reference's except s, seed and the table. The mem and words bit_names are
  identical.
- **The bits:** train and dev equal the dry runs' and e82-R3's rows; one test hash per arm.
- **Reproduction:** seed 1's dev accuracy equals e82-R3's row exactly.
- **The test rows:** all five meta arrays (y, convo_id, k, w_bert, w_roberta) are equal in every arm and reference, and
  the test example hashes are as above.
- **The copies** (`scripts/r25_refs.py`):
  - every file's SHA-256 equals its source;
  - the reference folders hold exactly the listed files;
  - their test-bit hashes are as above;
  - rerunning report_abcd.py on R22star and R23abcd with the original pairs, to a temporary path, gives byte-identical
    reports.

## Commands

- **Jobs:** logs/jobs_R25.txt. STAR first if its go-bar passed, then ABCD, one at a time through seq_queue.sh with
  cpu_load at most 4.5. They run alone on the GPU.
- **The report commands** are the file's last comment lines:
  - report_abcd.py per dataset, with the rule pair (refined against C4) and the no-rule pairs (refined against
    bge-large, C4 against bge-large);
  - report_twolevel.py on the same pairs.
- **Dry runs:** tiny TM (200 clauses, 2 epochs), dev standing in for test, through newtable_run.py and abcd_confirm.py.
  Their train and dev bits must equal e82-R3's rows.

## Wording

- **The headline comes first:**
  - "On STAR test and ABCD test, the agent on a C4-word table refined by a teacher (bge-base-en-v1.5) at training time
    (network-free at run time) is better than the agent on the C4 counting table: +X [lo, hi] / +Y [lo, hi]".
  - That is only if both R25.3 and R25.4 are met. Otherwise it is per dataset, as above.
  - Beside it go the potion line (R19) and the caveats.
- **The bge-large readings** follow, labelled "informed by R22 / R23", with no "better than".
- **Never:**
  - "no neural network at any stage" for the refined agent;
  - "a new method";
  - "distilled from the reference";
  - "beats the transformer";
  - "state of the art".
- A rule that is not met is reported as it is.
