# Registration REG_agent_vs_llm (v11): a readable agent with no neural network against local LLM agents and a fine-tuned transformer

The frozen copy is REG_agent_vs_llm.md, byte for byte this text; freeze_reg.py writes its record
REG_agent_vs_llm.frozen.json once (the freeze time read from the clock and the full SHA-256 of this text and of every
pinned file; the hashes live in that record, not in this text). Conditions of the freeze, all met before it: L35k50 on
MultiWOZ, L35k100 on STAR (AMEND_q3_k100.md's rule did not send it to ABCD or MultiWOZ), L35t16 (STAR re-ask,
MultiWOZ, ABCD) and RoBERTa on ABCD ended on dev; select_llm_star.py, rows_rule.py and make_reg_jobs.py wrote
runs/llm_star_choice.json, runs/reg_rows.json, jobs_reg.txt and reg_report_all.sh; a last read-only review checked
v11's first text (f8417ba4; judged by an AI model, not a person: READY, 2 majors, 23 minors); the same review
checked the diff of the answers (1849df0e; 2 blockers, both text, and 9 minors); this text puts in its two replacement texts (the first one narrowed, see section 6) and the wording of its
minors m2-m5, m7 and m8, and that last step was not reviewed again; no test file of STAR, ABCD or MultiWOZ was opened
in this line (freeze_reg.py hashes them; nothing reads their rows).

## 1. Question and headline

For choosing the next action in task-oriented dialogues (STAR, ABCD, MultiWOZ 2.2), how does a readable agent with no
neural network at any stage (a word table counted from C4 text, memory bits of the actions taken, a Tsetlin machine
whose decision is a vote of clauses over named bits) compare with (a) LLM*, the best on dev of the prompted
configurations of two local Qwen models that we ran (Qwen3.5-9B at Q8_0, Qwen3.6-35B-A3B at Q4_K_M; eight on STAR, six
on ABCD, four on MultiWOZ), and (b) a transformer fine-tuned end to end on the same examples?

Headline contrast, the same on every dataset: AGENT minus LLM*. The claim "the readable agent is not worse than the
best of the local LLM configurations we ran" (chosen on dev; the 35B model at 4-bit) needs NI met on all three datasets; with fewer, it is stated only for the
datasets that meet it, with the other intervals beside it, and it always names the margin met (NI: 3 points; NI2: 2
points). No combined line; no claim rests on any one rule being met. Every rule's outcome is shown in one table, built from
the three report files (runs/reg/report_*.json) and nothing else.

## 2. Arms

- AGENT: the C4 agent at its registered settings (parts mem, words, xcust, xrecent; table c4words-S1000-v1; STAR
  16,000 clauses s 16; ABCD 4,000 clauses s 16; MultiWOZ 16,000 clauses s 8 with random negatives). No agent is trained
  in this registration: AGENT's test predictions are the registered, verified seed files of R22 (STAR), R23 (ABCD) and
  R24 (MultiWOZ), seeds 1-5 (results/confirm/R22star/tm_mem_c4_s16, R23abcd/tm_mem_c4_s16, R24mwoz/tm_mem_c4_nrand;
  written by decision_layer's table_run.py -> abcd_confirm.py; pinned). Seed 1 carries the registered statistic; seeds
  1-5 enter the two-level interval. Their seed-1 dev accuracies are this line's dev references (STAR
  0.6544585987261147, ABCD 0.8417480998914224, MultiWOZ 0.35004756080989263); check_reg.py asserts them. So no new agent
  code reads a test split. The four decision_layer source files that changed since R24's freeze changed in comments and
  docstrings only (code_since_R24.diff). Network-free: R24's network-free path test (tests/test_r24_network_free.py,
  an audit hook on the run path table_run.py -> abcd_confirm.py, the path R22 and R23 also took) found no
  sentence_transformers, transformers or model2vec module.
- LLM*: per dataset, the configuration with the highest accuracy on the 500 fresh dev rows (random.Random(2); STAR and
  ABCD without Q1's rows) among these: L9 (9B, policy text, 10 retrieved examples), L35 (35B, the same), L35t (35B,
  reasoning, 4,096-token cap; STAR only), L35t16 (35B, reasoning, 16,384-token cap), L35k50 (35B, 50 examples), L35k100
  (35B, 100 examples; STAR only: AMEND_q3_k100.md's rule, at least 1.0 point above L35k50 on STAR, was not met, 0.622
  against 0.638), L9-noG and L35-noG (no policy text; STAR and ABCD). So eight on STAR, six on ABCD and four on
  MultiWOZ (no policy text exists for MultiWOZ: L9, L35, L35t16, L35k50). L35t16 on STAR is the composite that
  NOTE_review_fixes.md specified: L35t's 449 rows with a valid answer, and its 51 rows without one (50 of them at
  the 4,096-token cap) asked again with the 16,384-token cap (combine_reask.py). Eligible: 500 answered rows, a result.json with n 500 and no request error;
  ties to the faster configuration (select_llm_star.py, dev files only). Its prompt, model file and settings are used
  unchanged on test (reg_llm_job.sh -> llm_next_action.py --test-registration). The L35k50 dev runs used
  llm_next_action.py 4fca5acb (archive/llm_next_action.v7_4fca5acb.py); the test uses the pinned current version, whose
  prompt code is the same: the diff touches only failure handling (retries, re-asks on restart, --only-rows, a stop
  after 10 rows in a row with a request error, and a 300 s request timeout without reasoning, where the slowest L35k50
  dev request took 41.3 s).
  - The candidate set grew after dev results at least three times (the no-policy arms were already candidates in v3,
    after the policy-text ablation; v1 and v2 are not recorded), each recorded before the new arm's first answer: k = 50
    after L35's STAR result (AMEND_q3_k50.md), k = 100 after k = 50's STAR result (AMEND_q3_k100.md), and L35t16 after
    L35t's 4,096-token cap was found to leave 51 of 500 STAR answers invalid, 50 of them empty at the cap
    (NOTE_review_fixes.md). L9t (the 9B with
    reasoning) was dropped (NOTE_review_fixes.md: on the same 42 ABCD rows it had 30 right against 34 for L9, 35 for
    L35 and 36 for L35k50, with 5 answers cut at the cap).
- FT: bge-base-en-v1.5, one model type fine-tuned separately on each dataset; seed 1, lr 5e-5, 4 epochs, test predictions at the
  best dev epoch (ft_baseline.py --test-registration). On full dev it was the best of the models run on STAR (0.7409;
  RoBERTa 0.7399, BERT 0.7380) and on ABCD (0.8727; BERT 0.8716; RoBERTa 0.8697); on MultiWOZ only
  bge-base was run (0.4327). GPU training is not bit-for-bit reproducible: its dev accuracy must be within 0.5 points
  of these, or the result is reported with that flag (the ABCD dry run gave 0.8716). One seed: FT's seed variance is
  not in any interval.
- TFIDF (no rule): TF-IDF word 1-2 grams and logistic regression on the same rendered text as FT, at the C that was
  best on dev (STAR 4, ABCD 16, MultiWOZ 1; NOTE_tfidf_dev.md), trained on train, random_state 0 (tfidf_reg.py
  --test-registration). Dry runs on dev reproduced the earlier dev accuracy exactly on STAR and ABCD; on MultiWOZ
  0.35073 against 0.35059 (one row; tfidf_lr.py had left random_state unset). Reported beside the rules: TFIDF minus
  LLM* and AGENT minus TFIDF; no claim rests on it.

## 3. Rows, statistic, rules

- Test rows: per dataset, n rows drawn by random.Random(20261027).sample over the test rows; LLM* answers them, and
  AGENT, FT and TFIDF are restricted to them. n comes from rows_rule.py on dev only: a test split of at most 4,000 rows
  is used whole (ABCD: 3,608); otherwise n is the smallest of 2,000, 3,000 and 4,000 with NI power of at least 0.90 at
  the planning difference D = min(mean AGENT minus LLM* on LLM*'s 500 dev rows, 0) (no credit for a dev lead), with
  power(n) = Phi((D + 0.03) / (sd * 1.1 / sqrt(n)) - 1.96); if none reaches 0.90, 4,000 rows and the power is stated.
  Split sizes from R19 and R24's reports: STAR 6,988, ABCD 3,608, MultiWOZ 7,355 rows. The rule was fixed in v8
  (07:50:00Z) before L35k100 and L35t16 had any answer, while L35k50 on MultiWOZ had answered about 98 of its 500 rows (by the per-row times)
  (10 of them, 5 right, had been looked at for timing at 07:33Z); ABCD's whole split is from v10 (the same test use,
  about 2 GPU-hours more, power 0.95 instead of 0.91). With the final LLM* (runs/reg_rows.json): STAR 4,000 rows (NI
  power 0.95 at D = 0), ABCD 3,608 (0.95 at D = -0.6), MultiWOZ 4,000 (0.86 at D = -0.2). MultiWOZ misses the 0.90
  target, so the rule's cap of 4,000 rows applies and its power is stated; its whole split (7,355 rows) would give about
  0.98, but the rule was fixed in v8 and is not changed after the final dev numbers. NI2 power at D: STAR 0.67, ABCD
  0.56, MultiWOZ 0.49. There is no dev go-bar: every dev result is in before the freeze, and the test is run with a
  planned NI power of at least 0.86 on every dataset.
- The 1.1 inflates standard errors for dialogue clustering; on 2,000-row draws of full dev the measured ratio for the
  agent-FT pair was 0.96 to 1.08 (design_effect_dev.py, design_effect_dev.out). At 4,000 rows or a whole split more
  turns of each dialogue enter than in those draws, so the ratio may be larger and the stated powers slightly
  optimistic; the registered interval resamples dialogues either way.
- Registered statistic (reg_report.py): accuracy differences on those rows; 95% percentile interval from a bootstrap
  over dialogues (convo_id), seed 20261027, 10,000 draws. The registered interval decides every rule.
- Rules, the same on every dataset, each standing alone:
  - NI: AGENT minus LLM* has the lower end of its interval above -3.0 points.
  - NI2: the same above -2.0 points.
  - SUP: AGENT minus LLM* at least +3.0 points with the lower end above 0.
  - FT: FT minus AGENT with the lower end above 0.
- The margin's history, stated with the rule: drafts v3-v5 had one NI rule at -2.0. v6 (07:31:34Z) raised it to -3.0,
  66 seconds after L35k50's ABCD dev run ended (07:30:28Z) with the LLM 0.6 points ahead of the agent (0.864 against
  0.858). At -2.0, NI on ABCD had power of about 0.23 at that dev estimate with v3-v5's 1,200 rows (about 0.35 with
  v6's 2,000; v6 changed the rows and the margin together); at -3.0 with ABCD's whole split it has 0.95. The reason then written for 3 points (about 0.5 ms per decision on one CPU thread against about
  10-12 s on the V100; measured later: 0.5 to 3.0 ms) was written after that dev result. So the margin is dev-informed; NI2 at the earlier -2.0 is
  registered beside it, and every claim line names the margin it meets.
- No rule, beside every pair: the 90% interval of AGENT minus LLM* and whether it lies inside (-3, +3) (equivalence at
  3 points); a two-level bootstrap (agent seeds 1-5, then dialogues; seed 20261027, 10,000 draws) for AGENT minus
  LLM*, FT minus AGENT and AGENT minus TFIDF (FT, LLM* and TFIDF are one run each: their run-to-run variance is not
  included); TFIDF minus LLM*.
- Seed 1 and the 5-seed mean: every rule met on the registered (seed-1) interval is reported with the 5-seed-mean
  difference (reg_report.py's two_level.agent_minus_llm_mean) and its two-level interval beside it; where the
  two-level interval does not meet the same bound, the claim line says so ("met on seed 1, not on the 5-seed mean").
  Seed 1's known test edge over the 5-seed mean is stated in section 5.
- No multiplicity correction is applied; each rule stands alone. Claims registered: NI, NI2, SUP and FT on each of STAR,
  ABCD and MultiWOZ.
- Pins during the test (freeze_reg.py, make_reg_jobs.py):
  - Right after the record is written, freeze_reg.py --snapshot copies this text, the record and every pin except the
    model files, the llama.cpp build, the bge-base snapshot folder and the decision_layer data folder into
    frozen_snapshot/, read-only, each copy checked against the record's hash. The data folder also holds small files
    the LLM prompt reads (the ABCD guidelines, the STAR task specifications) and the C4 table; none of the skipped
    files can be restored. The bge-base weights blob lies outside that folder (the cache's shared blobs) and is copied.
    freeze_reg.py --lock then turns off the write bits of every pin in this folder; they are turned back on by hand
    after the report.
  - Every registered job starts only after freeze_reg.py --check finds every pin unchanged (the first command of its
    queue line; runs/reg/pincheck_<job>.out), so a changed file on the test path is found before a test split is read.
  - Every job's pin check and check_reg.py's global check cover every pin, so any CHANGED pin stops all registered
    jobs and closes every dataset's gate until it is restored. A CHANGED pin in this folder or in the R22-R24 seed
    folders is restored from the snapshot (freeze_reg.py --restore keeps the changed file as <path>.changed_<utc>; the
    report names it and gives its diff, written by hand), and the check runs again. A CHANGED pin in another study's
    folder (decision_layer src, scripts, registrations, logs and tests; gpu_run.sh; the C4 corpus records; the review
    files) is restored the same way only on the author's explicit decision, and that study's version is put back after the report.
    A CHANGED pin with no snapshot copy (the model files, the llama.cpp build, the bge-base snapshot, the decision_layer
    data) cannot be restored: a dataset whose run or report reads it is reported as not completed; the others are
    reported after check_reg.py --skip-pins (all its other hard checks still apply) and a by-hand freeze_reg.py
    --check that shows no other changed pin, with the pin named.
- Checks and report (reg_report_all.sh, written by make_reg_jobs.py and pinned; the last queue line):
  - check_reg.py first (it reads queue.log, so the registered jobs run in this queue: jobs_ft.txt and run_queue.sh).
    Global hard checks: the freeze record exists and every pin is unchanged. Per dataset: AGENT's
    seed files present and seed 1's dev accuracy equal to the reference; LLM*'s test rows exactly the registered draw,
    with no request error and a result.json; LLM*'s run with its registered settings (k, reasoning, policy text, max
    tokens, row seed and rows, from its argv and result.json); FT's and TFIDF's test files written under
    --test-registration and not as dry runs; FT with bge-base, lr 5e-5, seed 1, 4 epochs, batch 32 and 256 tokens;
    TFIDF with its registered C; every queue start of the dataset's three job lines equal to its pinned line in
    jobs_reg.txt. A dataset is reported when the global checks and its own pass (check_reg.py's gate); a dataset whose
    checks fail is reported as not completed, with the failed check named, and the others are reported. Flags (not
    fatal): FT or TFIDF dev accuracy more than 0.5 points from the earlier dev run; LLM*'s invalid answers above 0.5%
    of its rows (L35k50 had 0 of 1,500 on dev), after which the server log is read and the finding reported.
  - Then reg_report.py per open dataset, then verify_reg.py (an independent recompute: standard library only,
    bootstrap seed 7, 4,000 draws). Where the registered lower end lies within 0.003 of a bound and verify_reg.py's
    falls on the other side, or a difference is +3.0 points up to rounding, it reports NEAR-BOUND, not MISMATCH; the
    registered interval decides.
  - A failed LLM request is re-sent twice (30 s apart); rows still failing are recorded as errors and asked again on a
    restart. A run stops after 10 rows in a row with a request error (a dead or hung server; a request times out after
    300 s) and exits 4, as does a finished run with errors left; reg_llm_job.sh makes a second attempt (a new server)
    after exit 4, which resumes the run; a job that still fails is re-queued once. No row is ever counted wrong
    because a request failed: if a dataset's LLM* run cannot answer every registered row after that, its result is
    reported as not completed, with no rule outcome.
  - Each test job runs once. A job is re-queued with its identical line, appended by hand to jobs_ft.txt under the
    same name (so check_reg.py's queue-line check covers it), only after a failure that left its outputs incomplete
    (FT, TFIDF and LLM alike), at most once; a server refusal before any row is answered, and a job stopped by its own
    pin check (which has read nothing), do not count toward that limit. No arm is re-run after it completed. Nothing in
    runs/reg/ is read before reg_report_all.sh runs, apart from exit codes, row counts, error counts and the pin-check
    outputs runs/reg/pincheck_*.out.
- Cost (no rule): decision time of AGENT's seed-1 model on one CPU thread (agent_speed.py on dev rows, measured before
  the freeze on the R22, R23 and R24 seed-1 models, whose CPU class sums reproduce their dev accuracy exactly: median
  2.71 ms STAR, 0.50 ms ABCD, 2.98 ms MultiWOZ; runs/cost/agent_*_seed1_cpu.json); the FT network on one CPU thread
  (ft_speed.py, dev rows: 256, 195 and 263 ms; NOTE_cost_cpu.md); LLM*'s median seconds per row on the V100 (its test
  rows).
- Predicted outcomes from dev, with the final LLM* (L35k50 on all three datasets; AGENT minus LLM* on its 500 dev
  rows: STAR +2.8, ABCD -0.6, MultiWOZ -0.2). NI: likely met on STAR and ABCD (power 0.95) and on MultiWOZ (0.86).
  NI2: likely on STAR (1.00 at the dev estimate), uncertain on ABCD (0.56) and MultiWOZ (0.49). SUP: possible on STAR
  only (about 0.4 at the dev estimate of +2.8, just under the +3.0 it needs), not predicted on ABCD or MultiWOZ. FT:
  predicted met on all three (FT minus AGENT on the 500 dev rows +8.6, +4.0, +8.0; on full dev +8.6, +3.1, +8.3).

## 4. Caveats fixed now

- Selection is optimistic for both sides: the agent's view, s and clause budget were chosen on these datasets' dev
  splits (e74, e75, e85; MultiWOZ's random negatives on MultiWOZ dev, e83-e84), and its dev numbers are best-epoch
  numbers; LLM* is the best of up to eight configurations on the same dev rows, and the LLM grid grew after dev
  results.
- Contamination, both ways. The pretraining data of the LLMs and of bge-base cannot be checked for these public
  dialogues; that would favour the LLM and FT arms. The C4 table's text was not deduplicated against MultiWOZ and
  overlaps 2.47% of its dev and 2.61% of its test utterances (R24), which can favour the agent there. For STAR and
  ABCD, the C4 corpus filter read the dev and test turns as text only, to drop C4 sentences equal to them or sharing 8
  consecutive words (studies/text_understanding/w1/build_corpus_c4.py; counts in corpus/c4_slices.json for shards 00000-00001:
  ABCD test 65,045 texts, 14,655 sentences dropped by key and 254 by 8-gram; STAR test 50,982 texts, 36,405 and 113).
  Shards 00002-00007, the rest of the table's eight, went through the same filter, which read the same evaluation
  texts; for them only totals over all evaluation sets exist: 505,667 sentences dropped by key and 8,306 by 8-gram
  (corpus/c4_ext_slices.json).
  No label was read; the filter removes test text from the table's corpus.
- The Tsetlin machine is not why the agent is level with the LLM agents. On the same bits, logistic regression is
  level with it on STAR test (R22.2: -0.09 [-0.90, +0.73]) and 1.6 points above it on MultiWOZ test (R24, 5-seed means
  0.3592 against 0.3437); on full dev, TF-IDF logistic regression is level with the agent on all three datasets
  (0.6529, 0.8415, 0.3506 against 0.6545, 0.8417, 0.3500). What the Tsetlin machine adds is the readable form of its
  decisions, not accuracy. Prior work already reports prompted LLMs below trained task-specific models on dialogue
  action prediction and belief tracking (lit_headline_2026-10-27.md).
- The LLM arm's limits: two Qwen models only, the 35B quantized to 4 bits and the 9B to 8 bits; prompted, never
  fine-tuned; at most 100 retrieved examples; reasoning capped at 16,384 tokens, where 44 of 500 STAR rows,
  4 of 500 MultiWOZ rows and 2 of 500 ABCD rows still had no valid answer (counted wrong; on STAR 43 at the cap and
  one unmappable answer, "query"; L35t's 4,096-token cap had left 51 STAR answers invalid, 50 of them empty at the
  cap); one run each at temperature 0 on llama.cpp b11742 (commit 4c3c581) on one V100;
  no hosted model. "The best of the local LLM configurations we ran", never "LLMs".
- Accuracy on MultiWOZ is exact match of the whole act set, strict for every arm alike.
- The agent has no guard (veto) clauses in these runs.
- FT's training length: its best dev epoch was its last (4 of 4) in every seed-1 run and in seed 2 on STAR and ABCD
  (MultiWOZ's seed 2 peaked at epoch 2), so FT might gain with more epochs; FT minus AGENT is measured at this 4-epoch
  budget.

## 5. Test-split uses and what was known at the freeze

- STAR test: this is its 7th use (R18; R19; R19's post-hoc LR refit; R20; R21; R22). ABCD test: 15th use (14 before,
  through R23). MultiWOZ test: 2nd use (R24). All earlier uses but the LR refit were registered. No other registration
  is pending on these splits: R24's superiority draft (registrations/R24_multiwoz_superiority.md) and R25
  (R25_refined_table_agent.md) were never frozen.
- Known test numbers of this agent at these settings: STAR 65.3% (R22, 5-seed mean; seed 1 0.6564), ABCD 83.6% (R23,
  5-seed mean; seed 1 0.8379), MultiWOZ 0.3437 (R24, 5-seed mean; seed 1 0.3460). Its per-turn test predictions are the
  files AGENT uses; their accuracy on the registered rows was not computed before the freeze. No LLM, FT or TFIDF test
  number on any of these splits was ever seen.
- Seed 1's known test edge: its test accuracy is above the 5-seed mean on all three splits, STAR 0.65641 against
  0.65312 (+0.33 points; 2nd of 5), ABCD 0.83786 against 0.83642 (+0.14; 3rd), MultiWOZ 0.34602 against 0.34368
  (+0.23; 2nd); on dev it sits at the mean (-0.08, -0.01, +0.18). Seed 1 has carried the statistic since v3
  (NOTE_agent_seeds.md), so this is no pick of the best seed, but every AGENT minus LLM* difference carries this known
  tilt of 0.1 to 0.3 points toward the agent (R19, R21, R22, R23 and R24 registered the per-row mean over seeds
  instead). Section 3
  reports every met rule beside the 5-seed mean.
- Other test numbers known on these splits (decision_layer docs/results.md): the bge-large agent 63.7% on STAR (R22,
  5-seed mean), 0.92 points above the C4 agent on ABCD (R23.1: C4 minus bge-large -0.92 [-1.61, -0.26]) and 0.3384 on
  MultiWOZ (R24); logistic regression on the C4 agent's own bits 65.4% on STAR (R21, one seed; R22.2 above) and
  0.3592 on MultiWOZ (R24), and 0.6493 on STAR on R19's agent's bits (the potion word table; R19's post-hoc refit); on
  ABCD's AST measure (the paper's own example filter; R19's amendment) R19's agent (the potion word table, no network
  at run time; not the C4 agent) 84.78 against the
  published fine-tuned pipeline BERT 86.7, BERT 89.9, AlBERT 90.9 and RoBERTa 93.6 (the ABCD paper's Table 3). So
  fine-tuned transformers were known to lead on ABCD, which informs the FT rule there.
- What these numbers predict: the agent's test accuracy is known and the LLM's is not, so no rule's outcome is fixed
  by a seen number. Read with LLM*'s dev accuracy as a guess for its test accuracy, AGENT minus LLM* would be about
  STAR 65.6 - 63.8 = +1.8, ABCD 83.8 - 86.4 = -2.6, MultiWOZ 34.6 - 37.6 = -3.0 (seed 1; the final LLM*). These are
  cross-split guesses, biased against the agent: the 500 dev rows were easier than full dev for every arm with full-dev
  predictions (points above full dev: agent +1.2, +1.6, +2.4; FT +1.1, +2.5, +2.1; TF-IDF +1.9, +2.5, +3.3), so LLM*'s
  500-row accuracy likely overstates its level on a whole split, while the agent's side is a test number. Taken at
  face value they would give NI only about 0.09 on ABCD and 0.03 on MultiWOZ. The paired dev differences (+2.8, -0.6,
  -0.2) are the like-for-like estimates. NI2 on ABCD and MultiWOZ, NI on MultiWOZ and SUP on STAR are the most
  uncertain rules.
- Dev numbers known at the freeze (all in pinned files): runs/results_table.md (every arm on the 500 fresh rows), the
  policy-text ablation (runs/abl_table.json), the hybrid and cascade tables, Q1, the Zipf synthetic study, the costs
  (NOTE_cost_cpu.md), the readability measure on ABCD (runs/readability/abcd_s1.json), the TF-IDF dry runs (runs/dry/),
  the design effect, the 42-row L9t comparison and the L35t cap finding (NOTE_review_fixes.md), the partial
  L35k50 MultiWOZ rows (NOTE_review_fixes_add1.md), the agent's CPU times and receipts on all three dev splits
  (runs/cost/agent_*_seed1_cpu.json, runs/readability/*_s1.json), and the last dev runs: L35k50 on MultiWOZ (0.376),
  RoBERTa on ABCD (0.8697 on full dev), L35k100 on STAR (0.622), L35t16 on STAR (0.558, the composite), MultiWOZ
  (0.356) and ABCD (0.822). Two more dev sets of runs, both pinned: the bge-large raw-embedding classifiers (frozen
  encoder plus memory bits, agent_preds.py, full dev; ended 17:49:56Z and 17:51:05Z, before v11's first text): STAR
  LR 0.6618 and MLP 0.6807, ABCD LR 0.8480 and MLP 0.8561; and FT seed 2 (queued at 17:52Z, after v11's first text;
  ended 18:30:52Z, before the freeze): STAR 0.7373, ABCD 0.8713, MultiWOZ 0.4257 (seed 1: 0.7409, 0.8727, 0.4327).
  Neither can change FT (seed 1 is fixed) or any rule.

## 6. Draft history, with the dev numbers seen before each version

- v1, v2 (2026-10-26/27, overwritten in place; times and hashes not recorded): SUP on STAR only; 500 rows.
- v3 6c528d29 (2026-10-27T02:55:13Z): the same rules on every dataset (NI at -2.0), 1,200 rows, agent seeds, the
  module check, the contamination caveat. Seen before: FT dev, TF-IDF dev, LLM Q1, hybrid, L9 and L35 on all three
  datasets (thinking off), the ablation, the Zipf study.
- v4 c07d8fab (04:15:23Z), v5 d9adeb68 (04:25:00Z): the L9t clause, exact dev checks, the FT tolerance, use counts,
  known test numbers, predicted outcomes. Seen before: L35t on STAR (0.552, 04:13Z).
- v6 a0403ad4 (07:31:34Z): NI margin -3.0 and 2,000 rows. Seen before: L35k50 on STAR (0.638) and ABCD (0.864, ended
  07:30:28Z).
- v7 9eafe888 (07:47:00Z): L35k100 and L9t in the LLM* set, the freeze tool, the FT CPU cost. Seen before: 10 rows of
  L35k50 on MultiWOZ (5 right), the FT CPU timings.
- v8 e1be081d (07:50:00Z): the row rule (rows_rule.py).
- v9 28bba62e (07:57:35Z): the TFIDF arm (no rule), the TM-versus-LR caveat, the literature check. Seen before: the
  TF-IDF ABCD dry run.
- v10 a49cb5f8 (08:44:38Z): after the review of v9 (judged by an AI model, not a person;
  2 blockers, 4 majors, 16 minors): NI2 registered beside NI with the margin's
  history; AGENT from R22-R24's registered files (no retraining); L35t16 and the dropped L9t; check_reg.py,
  reg_report_all.sh and the LLM retry and re-ask rules; ABCD's whole split; two-level intervals for every AGENT pair;
  the declared C4 filter read; the claim narrowed to the LLM configurations we ran. Seen before: the L35t cap finding,
  the 42-row L9t comparison, the TF-IDF STAR and MultiWOZ dry runs and their 500-row numbers (0.672, 0.384), the ABCD
  readability measure, L35k50 MultiWOZ at 343 of 500 rows (0.364).
- v10b 3f3aa318 (08:57:08Z): the measured agent CPU times in the cost line. Seen before: those times. No rule changed.
- v10c d919a3f3 (09:14:55Z): RoBERTa's ABCD dev accuracy (0.8697). Ended before it: L35k50 on MultiWOZ (0.376,
  09:06:49Z) and RoBERTa on ABCD (09:14:06Z). No rule changed.
- v11's first text f8417ba4 (17:51:50Z): the final LLM*, rows and powers (MultiWOZ's 0.86 stated; v10c's "power of at
  least 0.88 on every dataset" became "at least 0.86", because MultiWOZ's final LLM* lowered it), the predicted
  outcomes, the configuration counts per dataset, the reasoning-cap counts, the freeze conditions, section 7 filled.
  Seen before: L35k100 on STAR (0.622, 12:23:12Z), L35t16 on STAR (the re-ask ended 14:50:34Z; composite 0.558), on
  MultiWOZ (0.356, 16:20:00Z) and on ABCD (0.822, 17:49:17Z); ended before it, the bge-large raw classifiers
  (17:49:56Z and 17:51:05Z; ABCD's numbers seen at 17:50Z).
- v11 (this text, the frozen one): the answers to the review of f8417ba4 (READY; 2 majors, 23 minors). M1: seed 1's
  known test edge stated and every met rule reported beside the 5-seed mean. M2: the snapshot, the lock, a pin check
  before every job, the restore rule and a gate per dataset in check_reg.py. Minors: the settings and queue-line
  checks, the invalid-answer flag, the client's stop after 10 request errors in a row and its 300 s timeout without
  reasoning, verify_reg.py's +3.0 rounding case, an absolute registration path in the job lines, more pins (the
  evidence files named here, the dev job scripts, the archived drafts, a copy of queue.log taken at the freeze), and
  the wording fixes in sections 1-6 (1849df0e, archived). Then the diff check's two blockers: B1, the two numbers of
  R19's potion-table agent first given as this agent's (its replacement used, with "R19 to R24" narrowed to R19, R21,
  R22, R23 and R24, whose statistic was checked; R20's was not), and B2, the restore rule (its replacement used
  verbatim); and its minors m2-m5, m7 and m8 in the wording. verify_reg.py's version before the sup_agrees() change
  was not archived (diff check m6). Seen before: FT seed 2 on dev (above). No outcome rule changed: NI, NI2, SUP, FT,
  the statistic, the rows and LLM* are as in v10c, and the retry rule too, plus the stop and the timeout, which change
  no answer. No test number has been seen at any revision.

## 7. Filled in at the freeze

- LLM* per dataset (runs/llm_star_choice.json): L35k50 on all three (Qwen3.6-35B-A3B Q4_K_M, reasoning off, policy
  text where one exists, 50 retrieved examples). Dev accuracy on the 500 fresh rows: STAR 0.638 (next best L35-noG
  0.628), ABCD 0.864 (next L35 0.840), MultiWOZ 0.376 (next L35t16 0.356); mean seconds per row on dev 10.57, 11.54
  and 11.49. The choice file's policy_text field is the configuration's setting: MultiWOZ has no policy text, so its
  runs had none (their result.json says guidelines false).
- Rows per dataset (runs/reg_rows.json, drawn by random.Random(20261027).sample): STAR 4,000 of 6,988; ABCD 3,608 of
  3,608 (the whole split); MultiWOZ 4,000 of 7,355.
- The predicted outcomes with the final LLM* are in section 3. The date and time of the freeze are in the record,
  with every hash.
