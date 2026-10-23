# R24 registration: on MultiWOZ 2.2 test, the network-free agents against the bge-large agent, all with the negative-class rule chosen on MultiWOZ dev

**Status:** rev 2, for the freeze. The freeze record (registrations/R24_multiwoz_random_negatives.frozen.json) gives its
time and hash. It supersedes the unfrozen 2026-10-19 draft registrations/R24_multiwoz_superiority.md (b19d8c4a), which
is kept unchanged. That draft was written for the "hardest" negative rule, before e83 and e84.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**Draft history** (times from entries of the log in the project notes, not published, and from `stat`; hashes from sha256sum):
- **Rev 0,** 2026-10-19 (registrations/R24_multiwoz_superiority.md, b19d8c4a): "hardest" negatives. Headline:
  superiority (C4 minus bge-large, the whole interval above 0). Non-inferiority at -2.0 beside it.
- **Rev 1,** log 2026-10-22T18:05:50Z (its hash was not recorded): random negatives. Headline: non-inferiority at
  -1.0; superiority secondary.
- **3e6ac886,** log 18:10:17Z: two changes made before the freeze (no switch to FastCoTM; "chosen on MultiWOZ dev" and the
  STAR / ABCD costs said plainly) and the dated re-search.
- **f416c19f,** log 18:17:59Z: the caveats of the e83 / e84 reviews (judged by an AI model, not a person).
- **bdb2d603,** log 18:23:51Z (copy kept in the project notes, not published): the arms move from s 8 to each
  agent's e85 s*, for equal tuning.
- **Rev 2 (this text),** edited from 19:38Z: e84b's result, the dry runs and the hashes (6c099db0, 19:45:00Z; copy
  kept), then the fixes from two read-only reviews, judged by AI models, not people
  (project notes, not published): the headline named,
  the gate at s*, the two-level interval, the start guards, the network-free test, the checker's fixes and test, and
  the sections on provenance, selection, dev rows and power.
- **Rules set after dev numbers were seen, said plainly:**
  - Rev 1's rules were written after e84's seeds 1-2 dev numbers (logged 13:09:44Z: G-ref +0.45, G-C4 +0.14). The
    headline moved from superiority to non-inferiority, and the margin from -2.0 to -1.0, the gate's margin.
  - Rev 2's rule changes came after e84b's numbers were read (19:36Z): the C4 agent's R24.1 named as the headline (the
    lower of the two gaps, and the cleaner gate), and the gate at s*. Both make a pass harder, not easier.
  - No e85 number was read before this freeze (below). 3e6ac886 and f416c19f were not kept, so what they changed rests
    on the log alone.

**The gate** (logs/plan_e77_note_2026-10-20.md: "the tuned 16k dev gap is at least -1.0 ... at its s*, seed 1"; "R24
returns only when fresh dev runs of a later stack pass the same gate, with every dev try listed"):
- **e84** (plan frozen 2026-10-22T10:37:12Z): random negatives for every agent, MultiWOZ dev, 16k, s 8, seeds 1-2.
  - G-ref MET, steady: refined minus bge-large +0.45 at best-dev, +0.77 on the last-10 mean.
  - G-C4 MET, steady: C4 minus bge-large +0.14 and +0.71.
- **e84b,** the e84 note's robustness rule (frozen 18:03:34Z, before seeds 3-4 existed): an agent's arm stays only if
  its gate also holds on the mean of seeds 1-4. Result (logs/e84b_report.out, 5ccf78ed; e84b ended
  2026-10-22T19:36:09Z): at best-dev, G-ref +0.73 and G-C4 +0.23 (last-10 +0.75 and +0.44, no rule). Both hold.
- **G-s*, the gate at the s the arms run at** (rev 2): e85 may move an agent's s, and the gate is defined at s*. So
  after e85 and before any test file is written, logs/R24_chain.sh runs `scripts/r24_check.py --gate`:
  - for each network-free agent, its 16k seed-1 dev row at s* minus bge-large's 16k seed-1 dev row at s* must be at
    least -1.0;
  - an agent below it keeps its arm, with no rule ("gate missed at s*");
  - if both agents are below it, R24 stops and MultiWOZ test stays unread (the gate note's rule).
- **Every dev try at the gate:** e77b (C4, hardest) -3.21; e82 (refined, hardest, seed 1) -2.36; e84 (random, seeds
  1-2) +0.14 (C4) and +0.45 (refined); e84b (random, seeds 1-4) +0.23 (C4) and +0.73 (refined); G-s* (after e85, in
  logs/R24_gate.out).

**The question.** On a dataset whose test split this study has never used, are the network-free agents not worse
than the bge-large agent by more than 1.0 point, when all three use the same reader settings, chosen on MultiWOZ dev?
- The C4 agent has no neural network at any stage. It carries the headline.
- The refined agent is network-free at run time, with a teacher at training time (bge-base-en-v1.5@a5beb1e3).
- The reader settings: random negatives (chosen on MultiWOZ dev), 16k, and each agent's s tuned under that rule on
  MultiWOZ dev (e85).

**One registered test at this milestone.** R25 (registrations/R25_refined_table_agent.md, STAR and ABCD) does not run:
its dev go-bars failed (logs/e82_gobars.out: STAR +0.14 against +1.0, ABCD +0.56 against +0.7; "PRECEDENCE: no
registered test"). So R24 is the only registered test here. Running another would need a recorded decision.

**Use of the test split:** MultiWOZ 2.2 test, 1st use. Its labels have never been used. Its texts were read twice
before, text only, and both reads took every turn's text, the wizard's included (whose words put the labels into words):
- The overlap check of the text-understanding line (2026-10-19; in the project notes, not published: mwoz_c4_overlap.py 1d5ab407, overlap.json 52cb74eb,
  overlap.txt cd7f7c43). overlap.txt prints 5 test utterances.
- The a3 table's fit filter (removal only; in the project notes, not published: a3_mwoz_filter.py a9514687, mwoz_filter_meta.json
  9b5d5ca8; logs/plan_e82_note2_2026-10-21.md). It read 29,492 dev and test utterances and removed 68 training and 5
  held-out fit sentences, counted over dev and test together.
- multiwoz_prepare.py parses all of the shared dialog_acts.json, test entries included, and keeps only the requested
  splits' entries (its docstring says less; a note from the leakage part of the review).

**What was known at the freeze** (all dev; the numbers are in "Dev rows behind it"):
- e77, e78, e79, e81, e82, e83, e84 and e84b (seeds 3-4);
- the earlier tests on other datasets: R22 (STAR, +1.64 [+0.94, +2.35]) and R23 (ABCD, -0.92 [-1.61, -0.26]);
- **The test-text path differs between the arms.** The a3B table's fit filter read MultiWOZ dev and test texts (above).
  The C4 table was not deduplicated against MultiWOZ: its README names ABCD and STAR only. bge-large's training data
  is unknown.
- **The gates' freshness:** G-ref's refined side (e83b) was known when e84 was frozen, so only its bge-large side was
  fresh. G-C4 is the cleaner gate, and e84b added fresh seeds for all three agents.
- **Equal tuning under the rule: e85.** The rule was screened on the refined agent only. So e85 (plan
  logs/plan_e85_s_under_random.md, frozen 2026-10-22T18:22:19Z, df0e0507) tunes every agent's s under random negatives
  by e77's rule, and each arm here runs at its agent's s*.
  - e85 started at 2026-10-22T19:44:56Z (logs/e85_chain.log, line 1), before this freeze: e85_chain.sh (outside the repository) waits for
    the dry check, not the freeze. Its first row was written at 19:50:27Z. Its rule was frozen before any e85 number,
    so s* is mechanical.
  - No e85 log, row or preds file was opened before the freeze. Only the e85 rows' times were read, to write this.
    r24_check.py prints how many e85 rows predate the freeze.
  - Its frozen checker writes logs/jobs_R24.txt, and r24_check.py reads s* from logs/e85_pick_final.out.
- **The rule was chosen on MultiWOZ dev.** Random negatives were picked on MultiWOZ dev in e83, and on dev they hurt
  STAR (-11.9) and ABCD (-4.0). The rule is MultiWOZ-specific, and every line that names it says "chosen on MultiWOZ
  dev".
- **Selection is optimistic.** The rule (picked among e83's arms; logs/e83_pick_final.out), the gate (met on its 3rd
  and 4th tries, after -3.21 and -2.36) and s (e85) are all chosen on MultiWOZ dev, and the dev numbers are best-epoch
  numbers.
  - Under random negatives (e84b, seeds 1-4), best-dev exceeds the last-10 mean by 0.86 (refined), 0.87 (bge-large)
    and 0.67 (C4) points. Under the hardest rule it was about 2 to 2.7 points (logs/plan_e82_note2_2026-10-21.md),
    more than the 1.0 margin.
  - The test numbers use the snapshot chosen on dev, so this selection inflates the dev gaps, not the test numbers.

## Dev rows behind it (MultiWOZ dev, 16k, random negatives, s 8)

- **e84b, the mean of seeds 1-4** (logs/e84b_report.out), best-dev / final epoch / last-10 mean:
  - refined 0.3534 / 0.3447 / 0.3448; bge-large 0.3460 / 0.3348 / 0.3373; C4 0.3484 / 0.3407 / 0.3417.
  - TM minus LR on the same bits (no rule): refined -1.09, bge-large -1.99, C4 -1.02.
  - Random minus hardest negatives (no rule), best-dev: refined +4.26 (2 hardest seeds), bge-large +1.01 (1 seed), C4
    +4.25 (2 seeds).
- **Seeds 3-4 alone** (fresh after e84's freeze; computed in the design review): refined minus bge-large +1.02, C4
  minus bge-large +0.33.
- **The seed-1 rows each TM arm's seed 1 must reproduce exactly** (results/explore/abcd.jsonl; dev accuracy, write time,
  train / dev bits):
  - e83_multiwoz_a3B_c16000_s8_nrand 0.34977578475336324 (2026-10-21T15:30:05Z), 10fd2411 / 612b8f0b;
  - e84_multiwoz_C4_c16000_s8_nrand 0.35004756080989263 (2026-10-22T12:47:53Z), a8c92a30 / 311e8933;
  - e84_multiwoz_bge_c16000_s8_nrand 0.34855279249898086 (2026-10-22T12:22:46Z), ca57e35d / b068b647;
  - for an agent whose s* is not 8: e85_multiwoz_{a3B|C4|bge}_c16000_s{s*}_nrand, seed 1 (e85b).
- **The LR dev rows** (each LR arm's dev accuracy and C must equal them; results/explore/dryrun/<row>/seed0.json):
  dry_e82mwoz_lr_mem_a3B 0.364316 at C 0.001 (db61ee5b), dry_e77mwoz_lr_mem_c4 0.358609 at C 0.001 (f3abfcb8),
  dry_e81multiwoz_lr_bge 0.365946 at C 0.01 (82292c54). Two of the three sit at the grid's lowest C.

## Power and predicted outcomes (from dev, at s 8; e85 may move s)

Computed in the design review on MultiWOZ dev (1,000 dialogues, best-dev predictions of seeds 1-4), with
report_abcd.cluster_boot and report_twolevel.two_level (findings of the design review, in the project notes, not published):
- C4 minus bge-large +0.23: registered interval [-0.49, +0.96] (half-width 0.73), two-level [-0.70, +1.15].
- Refined minus bge-large +0.73: registered [+0.01, +1.46] (half-width 0.72), two-level [-0.16, +1.66].
- An interval above 0 needs a test point estimate of about +0.73 (C4) / +0.72 (refined); "strong" needs at least
  about -0.27 / -0.28.
- If the test effect equals dev's, the predicted chances, C4 / refined: R24.1 met 0.99 / 1.00 (dev has in effect
  decided it); strong 0.83 / 0.98; R24.2 0.17 / 0.51. The open questions are the C4 agent's strong form and the
  refined agent's superiority.

## Arms (MultiWOZ test, results/confirm/R24mwoz/; 16,000 clauses, T 16,000, each agent's e85 s*, 40 epochs, start half, best-dev snapshot, seeds 1-5; random negatives, chosen on MultiWOZ dev)

| Arm | Agent | Command (through scripts/knob_run.py --neg-sampling random) |
|---|---|---|
| tm_mem_c4_nrand | C4 (view B on the C4 table) | table_run.py --table c4words-S1000-v1, abcd_confirm.py --encoder none --parts mem,words,xcust,xrecent |
| tm_mem_ref_nrand | refined (a3B table, view B) | newtable_run.py on c4words_a3B_v1, abcd_confirm.py --encoder none --parts mem,words,xcust,xrecent |
| tm_mem_bge_nrand | bge-large (R18's view) | abcd_confirm.py --encoder bge-large --parts mem,intent,step,words |
| lr_mem_ref, lr_mem_c4, lr_mem_bge (no rule) | LR on each arm's bits | abcd_confirm.py, C chosen on dev from the wide grid (seed 0) |

The exact lines are scripts/e85_pick.py's r24_jobs(s*), written to logs/jobs_R24.txt by e85's final stage.

## Pass rules (test)

- **The headline: R24.1 for the C4 agent.** tm_mem_c4_nrand minus tm_mem_bge_nrand is at least -1.0 on the point
  estimate. It is "strong" if the 95% interval's lower end is also at least -1.0.
- **Secondary: R24.1 for the refined agent,** the same rule on tm_mem_ref_nrand minus tm_mem_bge_nrand. A claim about
  "the network-free agents" needs both.
- **Secondary: R24.2, superiority, for each agent:** the same difference, met only if the whole 95% interval is above
  0, as in the 2026-10-19 draft (no minimum point estimate; R22.1's +1.0 minimum was for another contrast, s 16
  against s 8).
- **The margin:** -1.0 is the gate's margin. It is stricter than the -2.0 that R22.3 and R23 used for the same kind of
  contrast; the -2.0 reading is reported beside it, with no rule.
- **An agent that missed G-s*** keeps its arm, and its lines are reported with no rule ("gate missed at s*").
- **No combined line.** Each rule is reported on its own; there is no combined "R24 met" line, and no claim rests on
  either agent meeting a rule. One table shows, per agent: R24.1, R24.1 strong, R24.2, the registered interval and the
  two-level interval.
- **Multiplicity:** no multiplicity correction is applied; each rule stands alone. The superiority claims are R24.2 for
  the C4 agent and R24.2 for the refined agent.
- **Statistic, as in R21-R23:** per test turn, each arm's correctness is averaged over its 5 seeds. The 95% interval
  comes from a bootstrap over the test dialogues, 10,000 resamples, seed 0 (scripts/report_abcd.py).
- **The two-level interval, beside every pair** (scripts/report_twolevel.py, no rule): each resample draws each arm's
  seeds with replacement, then the test dialogues; 10,000 resamples, numpy default_rng(0). A rule met only on the
  registered interval is reported with the two-level interval beside it.
- **If r24_check fails** after the test jobs: MultiWOZ test's 1st use is spent, no rule is reported met, the failure is
  logged, and any fix is a dated amendment that names the failing check.

## Reported with no rule

- TM minus LR on each agent's bits. For lr_mem_ref and lr_mem_c4 it is labelled "LR's C at the grid's edge".
- The -2.0 reading of each R24.1 pair.
- The "hardest" rule is not run on test. Its dev contrast (e83, e84, e84b) is reported as dev only.
- Not run: the per-label, majority-label and first-against-later readings of the 2026-10-19 draft. No script for them
  was written before the freeze.

## Provenance and caveats (in every report line)

- **The C4 agent** has "no neural network at any stage" (table c4words_S1000_v1, counted from C4 text).
- **The refined agent** is "network-free at run time, teacher at training time".
  - The teacher is bge-base-en-v1.5@a5beb1e3 ([CLS], L2-normalised, MIT; registrations/R25_refined_table_agent.md:85,
    logs/plan_e82_refined_table_agent.md:25; line numbers of the texts as registered). It was fitted on 1,999,932 C4 sentences.
  - Never write "no neural network at any stage" for it; that phrase stays with the C4 agent, reported beside it.
- **Same family:** the teacher is the smaller sibling of the comparison agent's encoder (bge-large-en-v1.5), and it is
  also potion's teacher. With lambda* 0.01, the table's rows are mostly set by the teacher. So the refined table is "a
  replication of potion's recipe on the C4 word list, not a new method".
- **The teacher's training data cannot be checked.** It may include MultiWOZ text. That could inflate the refined
  agent's numbers, here against bge-large too, since both encoders are from one family. bge-large's training data
  cannot be checked either.
- **The view-choice caveat:** view B was chosen and developed on STAR and ABCD dev (e60/S2; e78; e79). The bge-large
  agent keeps R18's fixed view. Only s is tuned equally (e85).
- **Only 3a enters the agent.** 3c's paraphrases and 3b are not in it.
- **The network-free check on R24's path:** tests/test_r24_network_free.py. Its self-test passed on the CPU
  (logs/test_r24_network_free_selftest.out). Its --run is step 0b of logs/R24_chain.sh: the refined and C4 dry
  commands, on the GPU, at the dry size, on dev, before any test file is written. It asserts: no sentence_transformers,
  model2vec or transformers module; the only file opened under data/emb/ is multiwoz__none.npz. A failure stops R24
  with the test unread.

## Wording

- **R24.1 met (the headline):** "On MultiWOZ 2.2 test (its 1st use), the agent with no neural network at any stage
  (the C4 counting table) is not worse than the bge-large agent by more than 1.0 point: X [lo, hi] (non-inferiority
  registered), with all three agents using random negatives, chosen on MultiWOZ dev."
  - The refined agent's line carries "network-free at run time, teacher at training time (bge-base-en-v1.5)".
  - Beside each line: the rule was chosen on MultiWOZ dev, and on dev it costs STAR (-11.9) and ABCD (-4.0); the
    two-level interval; selection is optimistic.
- **R24.2 met:** "better than the bge-large agent on MultiWOZ" may be said for that agent only, with R22 and R23
  beside it, so the claim is MultiWOZ-only.
- **Never:** "beats the transformer" in general; "state of the art"; "first"; "a new method"; "distilled from
  bge-large" or "from the reference"; "better than" for an agent whose R24.2 is not met.

## Data, commands, checks, dry runs and hashes

- **The start:** r24_wait_start.sh (outside the repository) runs logs/R24_chain.sh once the
  freeze record exists, a line "e85 end: done" exists and logs/jobs_R24.txt exists, and the GPU shows no compute app.
  Its 12-hour wait ends 2026-10-23T06:23:38Z; after that R24 does not start.
- **The chain's own guards** (logs/R24_chain.sh): it refuses to start unless the freeze record holds this text's hash,
  the last line of logs/e85_end.log starts "e85 end: done ", logs/R24_dry_check.out ends "all dry checks passed", and
  data/untrusted/multiwoz22/MANIFEST.sha256 has its pinned hash and `sha256sum -c` passes. It also refuses if any of its
  outputs exists (logs/R24_end.log, R24_gate.out, R24_network_free.out, R24_queue.log, r24_check.out,
  data/multiwoz/examples_test.json.gz, labels_test.json, data/emb/multiwoz__bge-large.devstage.npz).
- **The chain's steps:**
  0. G-s*: `scripts/r24_check.py --gate > logs/R24_gate.out` (dev). Both agents missed: stop, exit 3, test unread.
  0b. `tests/test_r24_network_free.py --run > logs/R24_network_free.out` (GPU, dry size, dev). A failure: stop, test unread.
  1. `scripts/multiwoz_prepare.py --with-test` (CPU), then 2. `scripts/multiwoz_embed.py --add-test` (GPU, minutes;
     train and dev arrays copied byte for byte from the dev-stage file, which is kept as multiwoz__bge-large.devstage.npz).
  3. logs/jobs_R24.txt through scripts/seq_queue.sh, one job at a time. Estimate: refined 5 x about 29 min, C4 5 x
     about 27, bge-large 5 x about 10, LR 3 x about 3, so about 5.5 h. "Alone on the GPU" is kept by the queue in
     the project notes (not published) and by the start waiter; gpu_run.sh shared does not enforce it.
  4. `scripts/r24_check.py > logs/r24_check.out`, before the report is read.
  5. The registered report: `scripts/report_abcd.py --registration R24mwoz --arms tm_mem_ref_nrand tm_mem_c4_nrand
     tm_mem_bge_nrand lr_mem_ref lr_mem_c4 lr_mem_bge --pairs tm_mem_ref_nrand:tm_mem_bge_nrand
     tm_mem_c4_nrand:tm_mem_bge_nrand tm_mem_ref_nrand:lr_mem_ref tm_mem_c4_nrand:lr_mem_c4 tm_mem_bge_nrand:lr_mem_bge
     --out report_R24mwoz.json > logs/R24_report.out`.
  6. The two-level intervals: `scripts/report_twolevel.py --registration R24mwoz --pairs <the same 5 pairs> --out
     <study>/results/confirm/R24mwoz/twolevel_R24mwoz.json > logs/R24_twolevel.out`.
  - Every way out appends one line "R24 end: <branch> <UTC> exit <code>" to logs/R24_end.log.
- **The checker:** scripts/r24_check.py, adapted from r23_check.py, written before any test number. It reads s* from
  logs/e85_pick_final.out (written by the frozen e85_pick.py); the verify step recomputes s* from the e85 rows by
  e77's rule.
  - Full mode: the freeze, the sources (no file under src/ or scripts/ may be added, changed or removed after the
    freeze, the checker included), s* and logs/jobs_R24.txt equal to e85_pick.py's r24_jobs(s*) line for line, the 18
    seed files, cfg and init (half) equal to the s* dev row's, the arm's table, train and dev bits, seed 1's dev
    accuracy exactly, each npz's test accuracy, the five meta arrays (y, convo_id, k, w_bert, w_roberta) in every arm,
    every test turn and dialogue of examples_test.json.gz in the arms (MultiWOZ 2.2 test has 1,000 dialogues; the
    count comes from the examples file, not a constant), examples_test written after the freeze with train's label
    list, the bge-large dev-stage file's literal hash and its train and dev arrays.
  - Tested: tests/test_r24_check.py (e06e6027bdae2002f2cf296e95c1ef1ee58fdc2a67da3e535001045d191429ec) PASSED on
    made-up files (full mode passes; 26 refusals, each on its own check's line; --gate's three cases):
    logs/test_r24_check_2.out 201266ae731e6ce1c256cf12ce2ba1b077adac41fd7e57c3d710fa334a9f3301 (an earlier run before
    the dialogue-count change: logs/test_r24_check.out 733f4f5c). Two bugs found by the review (the first --name in a
    refined job line; the rewritten provenance entry) are fixed and covered by it.
  - The chain: logs/test_R24_chain.sh PASSED with stubs (every guard and branch): logs/test_R24_chain_rev2.out
    e4ce47bc604beed2228c6fd0965950eccc3efa8ce1e8d0b6ca0db80db0c4f845.
- **Code since the dev rows** (`scripts/diff_frozen.py`): against e83's freeze (2026-10-21T14:36:33Z, before the e83b
  refined rows) only NEW files: e83_pick_noL8.py, e84_check.py, e84b_report.py, e85_pick.py, r24_check.py; against
  e84's freeze, NEW e84b_report.py, e85_pick.py, r24_check.py. Nothing on R24's path changed. Against e85's freeze,
  only r24_check.py CHANGED (not on e85's path). torch 2.6.0+cu124, numpy 2.4.6, NVIDIA driver 570.172.08.
- **Dry runs** (logs/jobs_R24_dry.txt; 200 clauses, 2 epochs, dev standing in for test): all six passed.
  logs/R24_dry_queue.log (6 x exit 0, queue done 2026-10-22T19:44:36Z) and logs/R24_dry_check.out ("all dry checks
  passed").
  - Train / dev bits: refined 10fd2411 / 612b8f0b, C4 a8c92a30 / 311e8933, bge-large ca57e35d / b068b647, equal to the
    dev rows'. The LR dry runs' dev accuracy and C equal the dev LR rows' (0.364316 at C 0.001, 0.358609 at C 0.001,
    0.365946 at C 0.01).
- **Overlap** with the table texts (the text-understanding line, overlap.json 52cb74eb): 2.61% of test utterances, stock phrases. It is
  declared, and the tables are not rebuilt.
- **Hashes, full SHA-256, read with sha256sum** (freeze.py hashes only the text and src/ and scripts/):
  - read at 2026-10-22T19:38:31Z:
    - r24_wait_start.sh (outside the repository) ca9f159974d617eeccfcfa17bcf92d8884f1ec026b1c50769ec09e1f9d566d39;
    - logs/jobs_R24_dry.txt 39945d40c9c0b77718441e61d06b0ad70f79d791bf17ed96d1d5cb67a7082f8b;
    - data/multiwoz/examples_train.json.gz 0e20a31434de10a5248f0b10ab01e7474ceb5e020cd5cbc7048bfe8f5db69e9b;
    - data/multiwoz/examples_dev.json.gz 3414cc8a6ecc64aafa0df070748b9d502a12e7996420b84a12df1b094a89fcd2;
    - data/multiwoz/labels.json ed3379da965e969daa1f5e7982c42bb04dc61ee0e184a93327dc2e8bf02dede7;
    - data/emb/multiwoz__bge-large.npz 324f6d626675c5a3600125561e12784d8b8df19519893644a74b4f3cc5ebe421 (the dev stage);
    - data/emb/multiwoz__none.npz fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 (empty arrays for
      every split; the network-free arms read no vectors);
    - data/tables/c4words_a3B_v1/table.npz 4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855;
    - data/tables/c4words_S1000_v1/table.npz e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a;
    - data/untrusted/multiwoz22/dialog_acts.json 328f392165e7826db9f827731b14b5cc04e79e9e3c6332bfb192a1ea17f8e9b6;
    - data/untrusted/multiwoz22/test/dialogues_001.json d6f43876cf130fdb2dfa8f96bc056b0995354137f02122e004925d01264ed386
      and dialogues_002.json 89af95d8f596a448e733d59b31be78f1dd1632eddd99d5cb298a3fcb1ac9d185 (hashed as bytes, not
      parsed).
  - read at about 19:44:41Z, after the dry queue ended:
    - logs/R24_dry_queue.log 8a705c6418617e3ead0297cd7ff264202ffc16347d55667a91d4da107014b208;
    - logs/R24_dry_check.out e37f77023823627a9e45969aa0af7cdf27a7349f8fa7ec332c859b7316832c62;
    - results/explore/dryrun/: dry_R24mwoz_tm_mem_ref_nrand_s8/seed1.json
      1a1b7eea2876404a7e02181ad6a3385fd545a80d85c2c1dc7e5259c1443ea93a, dry_R24mwoz_tm_mem_c4_nrand_s8/seed1.json
      a13e5fbf345422598e1b693b6990f2f6c1c2475c7d4398e3fa98a8180b0159fa, dry_R24mwoz_tm_mem_bge_nrand_s8/seed1.json
      a8941c460f5f6af09f6eec636c31680ca748029ddfad2528de60ab86f8f7b741, dry_R24mwoz_lr_mem_ref/seed0.json
      b8ef3ead77958efdc53d7e70ebdc75def3745c995f786d68793f59bed9bf4bfc, dry_R24mwoz_lr_mem_c4/seed0.json
      97acc4205350759a5ffb2e97c6c741eb3abace9ffc78afcbf80bac4a45e452c4, dry_R24mwoz_lr_mem_bge/seed0.json
      d539f12471ccbaf03bbf79a580c10e70c2101a8add2c67d8cad2056358e71164.
  - read at 2026-10-22T20:11:19Z:
    - logs/R24_chain.sh 0b5925549ab45cbdbcc8d5e6882a777419ee4e3f7ae36779f29907016e93c159;
    - logs/test_R24_chain.sh dc9f0d593dbae886662b1915b1fe99008bb373819d50062e47430dd6189cd131;
    - tests/test_r24_network_free.py d3bc4426741390a983b914bdffdd482b5705bba5fa299c201dd572a49c236a20, its self-test
      output logs/test_r24_network_free_selftest.out fdc414d714c550e1b0a5d6175fa881bcca70467e065658e267e944d1e6328796;
    - logs/R24_dry.out e2874297ca7c8080e4c8270a525077cc84bd24f9d93a34ba857a36b39a7e2b0e;
    - the LR dev rows: dry_e82mwoz_lr_mem_a3B/seed0.json db61ee5b03601589384517e5a3613119c29ee22bc55d5e06076599b9824333cf,
      dry_e77mwoz_lr_mem_c4/seed0.json f3abfcb8761ebb2a0d202a109731df4cddabb62916d43bdbfe2be25340b587cb,
      dry_e81multiwoz_lr_bge/seed0.json 82292c545af6d9cd15df1b91abee9264db39606695ff0afa6b26257f24920f5b;
    - data/tables/c4words_a3B_v1/: MANIFEST.sha256 1d1a00ad74634f77d38894e4a476c1717494c94a76450210518ec9bc8ed267f3,
      README.md 35ae923e408084ff8d9512a06eea5f481aff0037da0e23d06b111286e6dbd85f, vectors.npy
      21d0c4fb148498531a6523f9635401a6215ef4dd5c8cd4b7ca8ce6e7b80f8e14;
    - data/tables/c4words_S1000_v1/: MANIFEST.sha256 f49b934ad1fe8d520af9f87ef37f145aff5483f1c55068e2c5309747f8bd734e,
      README.md fafc3a213c666f08e52177911732f4c53973085f6c81e0c9276d7ec43daf9454, vectors.npy
      9263bf09383de22806e320a1988de57816dbd62bfa1cc692b0fb7077e44d267c;
    - both tables: tokenize_ref.py 2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803, words.txt
      49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29. The run reads table.npz, not vectors.npy;
    - data/untrusted/multiwoz22/MANIFEST.sha256 87d8b2cd220b8f2cd2f9527f2a223b658151a5cb3f3bd810bbcd4c50209e58cc
      (`sha256sum -c`: all 24 entries OK);
    - ../tm_language_model/scripts/gpu_run.sh 1e24a8c8bb9441ae90bd077669ff2e7e620ded244a491307efcb6719f35aaeac (wraps
      every job);
    - r24_dry_after_e84b.sh (outside the repository) 34fb132435b798ad5566c8c73e8cd20ca33a80d650ea0f5de01e3fdb15b51e95
      and e85_chain.sh (outside the repository) 46556c25cda4370f2a6d4e9f7f38758549235b59130dcf3842c6a2262f7ceb7c;
    - the bge-large weights: the only local snapshot is d4aa6901d3a41ba39fb536a557fa166f842b0e09 (embed.py pins no
      revision).
  - scripts/r24_check.py, scripts/report_twolevel.py, scripts/e85_pick.py and the rest of src/ and scripts/ are in the
    freeze record.
- **Newest work:** the 2026-10-19 baseline search, multiwoz_newest.md in the project notes (not published).
  - It found no published result for this task (the next system act set as one of 128 classes, accuracy), so no
    published number is claimed.
  - **The dated re-search** (two web searches, finished by 2026-10-22T18:09:13Z; snippets only): no 2025-2026 paper reports this task.
    - The newest MultiWOZ 2.2 paper found, "Proactive Dialogue Model with Intent Prediction" (arXiv 2604.27379, 2026),
      predicts intents, not the next act set.
    - Logic-based prior work on MultiWOZ policies exists: differentiable inductive logic programming (arXiv
      2011.05457). It is cited as related, not as a comparable number.
    - Next-action scores across papers are not comparable (candidate sets differ; arXiv 1910.00486).
    - So no published number is claimed beaten.
