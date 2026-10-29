# Registrations after the study

This page carries the registrations written after the study ended (it ran from 1 to 3 Oct 2026), for runs of 25 Oct 2026; each was frozen before its first run. docs/registrations.md keeps the registrations of 1 to 3 Oct 2026 with their own conventions. On 25 Oct 2026 Norwegian time is CET (UTC+1): summer time ended at 01:00 UTC that day. The UTC time is added in brackets in each heading. Both texts were frozen in the same step; the reproduction check comes first because the other's pins depend on its list.

## 25 Oct 2026, 17:31 CET (16:31 UTC): the reproduction check of the freeze rule, after the study ended

*Publication note.* The text below is the frozen file (sha256 `7b6deef110d4847a5611abf1c66cae8b4a5b20c124e1258ad0a81bb686b4fa2e`, frozen 2026-10-25T16:31:59Z; the frozen file itself is not distributed), edited for publication in wording only: the notes on its review and drafting are shortened and private paths are left out. Every rule and number is as frozen. workstation paths are shortened (the finish pass's folder is written `<finish-pass records>`, paths in this repository are relative to it, and the home folder is dropped from the others), and the headings move down two levels to fit this page. 25 of its 668 lines differ; nothing else changed.

### Frozen 2026-10-25T16:31:59Z (draft 4): the reproduction check of the freeze rule of gates.md, after the study ended (HTM-01)

Written in the finish pass of 2026-10-25. Draft 3 answered two pre-freeze reviews (one NOT READY with 1 blocker, 9 majors
and 12 minors; one READY AFTER FIXES with 5 majors and 8 minors), and draft 4 a re-check (READY AFTER FIXES: 1 major, 12
minors); all three reviews were judged by an AI model, not a person. Paths without a folder are in <finish-pass records>
(the records of the finish pass, not distributed); `htm01/...` is a folder there.

**Times.** On 25 Oct 2026 Norwegian time is CET = UTC+1: summer time ended at 01:00 UTC that day, the last Sunday of
October (the EU rule; converted with tzdata 2026.4 in a review, judged by an AI model, not a person, which found the same
offset).

#### 1. Question, headline and due date

gates.md, Day 5: "every number is frozen. A number that does not reproduce within 0.02 moves to the list of claims that
are not made, in the study README" (lines 133-134). The addendum of 2 Oct 2026 keeps the rule when the freeze moves
(line 153: "The freeze moves from Monday 18:00 to Sunday"), and the run plan of that night says "the **freeze moves to 4
Oct 2026**" (docs/registrations.md line 61). 4 Oct 2026 was a Sunday. The check never ran (docs/results.md line 309;
README line 278). It runs now, 21 days after its due date and after the study ended: a deviation.

Question: which numbers of the results page reproduce within 0.02 when the published code runs again on the same inputs?

**Headline, fixed now:** "The freeze rule was applied to N of M page numbers (per class); this is not a general claim
that the study's numbers reproduce." M = the 683 page tokens of docs/results.md at the version mapped here (sha256
e9ca0667..., section 2); the later R4 edit of line 309 does not change M. N = the tokens with at least one candidate
leaf that step (i) or step (ii) compares; at most 124 (section 11). The readings of section 4 are counted separately and
not in N. The outcome table has fixed columns (section 9). The verdict words and paired
intervals of the final test's main table ("below" word overlap, "below" the baseline that was best on dev; results.md
lines 299-307) are not re-checked (class A'). Step (iv) checks the arithmetic of the final report from its saved
per-pair scores; the scoring path itself is checked on dev only (step i).

#### 2. The decision unit: page numbers and the leaves that carry them

- **Page numbers** are the tokens of docs/results.md that `htm01/page_map.py` finds: every number with a decimal point,
  and every percentage with a decimal point. Integers, integer percentages, dates, times, version strings, run ids and
  numbers after a setting name (s, T, A, K, F, C) are not tokens; counts go to the outcome table as "not mapped".
  - Built on docs/results.md sha256 e9ca0667dd9ec08e24663a30f3248598ac328f7db2ed603f548e5dc4ebd51562, the current page
    (rebuilt in round 3: the round-2 map predated the line-47 pointer, which added 6 tokens with no run id).
  - 683 tokens; 609 are mapped to at least one leaf; 74 are not mapped. 36 of those 74 are on lines with no run id:
    rule thresholds such as 0.95, 0.90 and 0.02, the six line-47 values of the 2026-10-25 re-run, and numbers whose run
    is named on another line. The other 38 have a run on their line but no leaf of that value.
  - Files: `htm01/page_tokens.tsv`, `htm01/page_map.tsv` (4,069 candidate leaves; 158 runs searched, 139 with a
    candidate) and the counts in `htm01/page_map_counts.txt`.
- **Leaves**:
  - Runs searched: the runs written on the token's line. A table row with no run id takes those of the row above it.
    Every table row also takes those of the sentence that leads into the table.
  - Leaves of a run: every numeric leaf of its JSON files (run folder and direct sub-folders, except run.json, env.json,
    config.json and MANIFEST.json), and of its log.txt lines tagged SCORE, FINAL, COMPARE, GATE and EPOCH.
  - A leaf maps to a token when its value, rounded to the token's decimals, equals the token.
  - Leaves of per-pass trajectories (EPOCH lines, "history" entries) are kept only when the token has no other match.
    Then only the passes the line names are kept, if it names any.
  - Matching is by value, so a token can carry coincidental candidates. Example: line 268's 0.737 is the test AUC of
    seed 2, and it also matches a baseline interval end of report.json (0.7374). The outcome table lists every
    candidate of a failing token.
- **Scope of the rules.**
  - R1 and R2 act on a page number only through those of its mapped leaves that a GPU step compares.
  - A page number is reproduced when every compared leaf of it is. It is not reproduced when any compared leaf of it
    is not; this is conservative, since a coincidental candidate that fails withdraws the number, and the table names
    the leaf.
  - Every compared leaf outside 0.02, every missing leaf and every changed string or flag goes into the outcome table
    with its path, logged value and new value, whether or not it carries a page number. Only those that carry one move
    anything.
  - The README has few numbers of its own (`htm01/readme_tokens.txt`). The study README has 8 tokens; 2 are on the
    results page, and the other 6 are version numbers (WordNet 3.0, Python 3.11 twice, compute capability 7.0) and the
    two maxima of the HTM-B2 re-run (0.0011, 0.0028). All 9 tokens of the hierarchical_tm rows of the root README are
    results-page tokens. A README number takes the outcome of the equal page number.
- **Excluded leaf types** (`SKIP_KEYS` of `page_map.py`; the step-(ii) comparison also drops the timing fields):
  - timing: fit_s, ms_per_pair, eval_s, wall_s, runtime_s, wall_time_s, poll_s;
  - settings and counts: s, T, A, K, F, passes, pass, seed, clauses, batch, n_train, n_dev, final_pass, resamples,
    n_resamples;
  - pos_share, the share of "yes" predictions of the Check 1b classifiers, which the page never reports.
  - **Written after step (iii) was seen**: this list, the scope rule above and the page map all date from round 2, after
    `length_checks_rerun.json` and step (iii) had been seen. Draft 1 dropped timing fields in step (ii) only.
    `runtime_s` and `pos_share` are the two types that step (iii) had already shown outside 0.02.
- **The share flag of gates.md** (line 138: "Flag predicted-positive shares below 0.05 or above 0.95") is unchanged by
  the HTM-B2 fix: the new shares 0.496, 0.533, 0.6743 and 0.6265 all lie between 0.05 and 0.95, as did the logged ones.

#### 3. Classes of the runs (`htm01/class_map.tsv`; counts in `htm01/class_counts.txt`)

All 167 run ids that the page cites, plus 23 runs that the steps compare although the page does not cite them (190
rows, each with its class, route and the jobs that read it as an input). Rebuilt in round 3 on the current page: line
319 cites `20261002T200228Z_slide6_screen_s25` since the edit of this pass (HTM-06); it is class E.

| class | what | cited | compared, not cited | route |
|---|---|---|---|---|
| A | the 20 `*_testscore` runs and the final table | 21 | 0 | step (iv), a reading (section 4) |
| A' | baseline side, paired intervals and verdict words of the final table | (in A's final table) | | not re-checked: the baseline test scores were not saved (R5) |
| B | dev re-scores of saved models | 19 | 17 | step (i) |
| C | CPU scorers | 26 | 1 | step (iii), a reading (section 4) |
| D-sampled | training runs of the retraining sample | 11 | 5 | step (ii) |
| D-not-sampled | other training runs | 37 | 0 | not re-checked (R5); 21 of them are read as inputs by step (i) or (iii) (column read_as_input_by) |
| E | slide-6 and slide-10 tests | 45 | 0 | not re-checked (R5): held-back inputs; evaluation splits scored once |
| G | earlier data-check builds made with the HTM-B2 bug (`20261001T092321Z_data_checks_exploratory`, `20261001T093113Z_data_checks_v2_exploratory`, `20261001T094221Z_data_checks_ns_all_exploratory`) | 3 | 0 | not re-checked (R5): their data were rebuilt and are gone; their kept code copies have the unsigned `length_feats` line, so their length-only values were computed with the wrapped feature |
| H | two dry runs (fake input) and one failed dev check (`20261002T134702Z`, exit 65) of the final-test report | 3 | 0 | not results (R5) |
| X | failed runs that were repeated | 2 | 0 | not results (R5) |

**The sampled retrains and the page numbers they feed** (`htm01/facts_round2.txt`, part 3).
- Only one sampled retrain carries a page token in its own files: `20261001T212959Z_e4_sa_granmo_C50_A64_s100`, with
  line 171's dev AUC 0.745.
- The page numbers of the others come from the jobs that read their folders:
  - step (i) re-scores of the saved models of the 8 Wikipedia-set retrains (the seed-1 retrains on next-paragraph
    pairs and on the synonym data, and the 6 retrains of tier 2);
  - step (iii) scorers: `crossfit_acc` and `crossfit_splits` for L1 to L3, `L2_s_compare` for L2, `L3_s800` for L3,
    `e5a_npmi` for the hierarchical news model, and `headline_s_rule`, `repro_compare` and `rules_recheck` for the
    free-check run.
  - No job reads the two `e1_verify` runs or the flat news model.
- **Rule.** When a retrain is exact (every compared leaf exact, include bits and dataset hashes equal), the jobs that
  read its folder read files equal to the logged ones, so their outcomes stand for it. When it is not exact:
  - (a) R2 marks the claims of its setting;
  - (b) those jobs are re-run with the logged folder replaced by the new run's folder (a path-only change named now:
    step (i) on the GPU as the conditional request R2c, step (iii) on the CPU with `run_iii.py --subst RUN_ID=FOLDER`,
    which presents the new folder under the logged run's name) and compared under R1 with their logged outputs.
    `test_run_iii_subst_r3.txt`: 15 of 15 (a byte-identical copy gives identical outputs, plain and mirror job; a
    real substitution changes the comparison; a missing folder and other usage errors stop with exit 2); the same
    test fails 7 of 15 on run_iii.py 20e747a1 (`test_run_iii_subst_r3_before.txt`). Readers that glob results/
    (`recovery_all_e2`) take no folder argument: their rows for such a run are "not re-checked".
  - Page numbers of training runs outside the sample are "not re-checked".

#### 4. Readings already made before the freeze (no rule): steps (iv) and (iii)

**Label, for this plan and for every page text:** "readings under the unreviewed draft 1, run before the freeze; their
outcomes were known when the plan was frozen; no rule". Rules R1 to R4 bind steps (i) and (ii) only. Their full outcomes
are in section 11.

- **Order of events.** Draft 1 was written at 11:52:10Z (sha256 b5be827a1133f58fc45b1e9c240e14a24bdef108fb626998eab5cc0ae5a9b174,
  kept in `archive/`). Its hash is first recorded in log.md at 11:56:18Z.
  - Step (iv) ran 11:52:31Z to 11:55:11Z (its output; the log says 11:52:30Z to 11:55:12Z). That draft 1 came first is
    therefore shown only from the later record.
  - For step (iii), which first started at 11:56:35Z, the order is shown.
- **Step (iv)** (`htm01/recompute_test_from_scores.py`, output `htm01/out_iv/`). Interpreter, from the output's summary:
  `studies/tm_language_model/.venv-tm/bin/python`. The shell line itself was not
  saved, so its thread caps and nice level are not on record; the CPU gate read 3.48 cores at 11:52:29Z (log.md line 9).
- **Step (iii)** (`htm01/run_iii.py`, 26 jobs).
  - First run, 11:56:35Z to 12:06:29Z (`htm01/out_iii`, run_iii.py 5b8c3b62..., which was not kept and is lost). 23 of
    the 26 jobs ran. `crossfit_acc`, `crossfit_splits` and `final_test_report_dev` stopped before computing any number:
    the run logs name the data under `htm-poc/`, the folder of the workstation of the recorded runs.
  - A path-only change followed, `mirror()`: those three jobs read copies of the run folders in which log.txt and the
    .json files have that folder replaced by the study folder; every other file is a symbolic link to the original.
    The three jobs ran again from 12:08:50Z to 12:11:10Z (`htm01/out_iii_b`).
  - **Both runs are superseded** by a re-run of all 26 jobs with the kept script (run_iii.py 20e747a1..., copy in
    `archive/`) and comparator version 2, 13:08:29Z to 13:21:12Z (`htm01/out_iii_r2`; command
    `htm01/run_iii_r2.sh`, which waits for the CPU gate, 3.32 cores at 13:08:29Z, and runs
    `timeout 2700 nice -n 19 studies/tm_language_model/.venv-tm/bin/python -B run_iii.py <finish-pass records>/htm01/out_iii_r2`
    with 2 threads and CUDA_VISIBLE_DEVICES empty; log `htm01/out_iii_r2.log`). Its counts equal those of the first runs.
  - The Check 1b replica (`htm01/check1b_dev_replica.py`) repeats the bodies of `data_checks.check_version` and
    `check_ns_all.main` without the lines that open a test file. Its diff against them is `htm01/check1b_replica.diff`:
    - the module prefix `dc.`;
    - the left-out test lines (n_test, the test overlaps, the test histogram of shared words);
    - one renamed variable (`z` to `zz` for the paragraph file);
    - the shared setup (paragraph matrix, idf weights and PPMI vectors, computed once as in `data_checks.main`);
    - the output writing.
    The left-out lines draw no random numbers (the data review, judged by an AI model, not a person, compared the two
    and found the same).
- Optional after the freeze: re-run steps (iv) and (iii) into new folders with the frozen code and require identical
  output files (a record that the frozen code gives the same files, not a test).

#### 5. What runs: steps (i) and (ii)

All job scripts are generated from the run records by `htm01/make_gpu_jobs.py` (version 2) and run through the
serial GPU queue, one GPU job at a time.
- **Each script:**
  - `set -euo pipefail`, so it stops at the first job that fails;
  - the study folder; `export HTM_PY=studies/decision_layer/.venv-htm/bin/python`,
    `HTM_TOOL_PY` and `PYTHONDONTWRITEBYTECODE=1`; 2 threads; `nice -n 19`;
  - first the pre-flight check `htm01/preflight.py htm01/preflight_pins.sha256`, which stops on any pinned file or core
    package that differs and reports the GPU environment.
- **Absolute driver paths** (`"$PWD/src/..."`): `src/launch.sh` changes to `../tm_language_model` before it starts a job.
- **The job-line test** (`htm01/test_job_lines.py`) checks the header of every script and, for every line, the driver's
  `--help` run from that folder with the job interpreter, the parsed arguments, and that the data and model files exist.
  224 checks pass on the version-2 scripts. On the round-1 scripts 64 fail and 1 passes; the failures include exit 2
  "can't open file" for the relative driver path.
- **Step (i), dev re-scores of saved models.**
  - `htm01/gpu_jobs/gpu_htm01_i_smoke.sh` runs job 1, the re-score of the model of
    `20261002T030724Z_evalsched_best_every1_s1_exploratory`. `htm01/gpu_jobs/gpu_htm01_i.sh` then runs jobs 2 to 41.
  - Job 1's logged reference, `20261002T031952Z_saveload_check_exploratory`, has exit status 65 in run.json. Its job
    printed "exit status: 0" and a SCORE line, but the logger rejects a metrics.json without labels: it moved the file
    to metrics_rejected.json and returns 65. The repeat `20261002T032557Z_saveload_check_v2_exploratory` (job 2) wrote
    labelled metrics and exit 0. The SCORE line is compared. A new attempt counts only with exit 0.
  - 36 jobs repeat every dev re-score that the study wrote, with the argument tail of the logged command. 34 of them
    carry the logged `--max-minutes 10`, which `score_saved.py` accepts and ignores ("accepted from the queue runner").
  - 5 jobs re-score the five shuffled-label models of the final table, which had no dev re-score; they carry the same
    tail.
  - The SCORE lines of the two saveload checks have no bootstrap block (22 fields against 23 later), so only the fields
    they hold are compared.
  - Logged wall times: 0.27 min for job 1 and 14.969 min for the other 35 logged jobs; the 5 shuffled-label jobs have no
    logged time.
- **Step (ii), the retraining sample**, only if every step-(i) job is reproduced exactly.
  - `gpu_htm01_ii_smoke.sh`: `e1_verify_default`, logged 19.465 s.
  - `gpu_htm01_ii_tier1.sh`: 7 runs, 107.555 logged minutes:
    - seed 1 of the headline setting on same-article pairs, i.e. the s-screen run
      `20261001T212959Z_e4_sa_granmo_C50_A64_s100`. It has no saved model and is tied to the main-table seed-1 model
      (model_p100 of the curve run `20261002T035511Z_e4_curve_sa_granmo_s100_p200`) only through the free check;
    - seed 1 on next-paragraph pairs and on the synonym data (`ns_all`);
    - `e1_verify_binary`;
    - L1, L2 and L3 seed 1 at the settings of their verdict rows.
  - Then **always** `gpu_htm01_ii_tier2.sh`: 8 runs, 196.041 logged minutes. These are seeds 2 and 3 of the three
    Wikipedia sets and seed 1 of the two news-topic models, hierarchical and flat. Tier 2 is fixed now as "always". If
    the GPU budget cannot take it, the plan must be changed to "never" before the freeze, not after tier 1.
  - Outer caps per job: 10 to 89 min (jobs.json, `cap_min`). The headline and synonym runs also carry their logged
    `--max-minutes 75`, the news runs `--max-minutes 35`.
- **R2c, conditional** (only if a retrain is not exact; section 3): the step-(i) line of each model of that retrain with
  the logged folder replaced by the new run's folder, in the same queue.
- **Comparison**: `htm01/compare_after_gpu.py htm01/gpu_jobs/jobs.json htm01/out_gpu` (CPU, version 2).
  - It first checks the 266 logged files of `htm01/reference_files.sha256` and stops on a difference.
  - It lists every attempt of every job and uses the first complete one.
  - Step (i):
    - every SCORE field (missing fields count), rules.txt and the include bits;
    - the 5 shuffled-label jobs against the EPOCH line of their training log at pass 100, with the flag
      rescore_equals_in_run.
  - Step (ii):
    - `src/compare_runs.py` in both modes;
    - every EPOCH line and the FINAL line, without the timing fields; a logged pass missing from the new run counts as
      missing;
    - the include bits and dataset_hashes.json;
    - for the two Check 1a runs, the VERIFY line: every flag and value must be equal (exactly, not within 0.02).
  - Every new run must name the plan's interpreter (`job_python`) and hold code copies equal to the pinned files.
  - Attempts are classified as R3 says (version 3 of the comparator).
- **Its tests** (comparator version 3, round 3).
  - `test_compare_after_gpu_r3.txt`, 23 of 23, under `.venv-tm` on the final jobs.json and reference list: every logged
    run equals itself (both VERIFY lines included), and the shuffled-label branch works. Negative controls caught:
    - seed 2 presented as the retrain of seed 1; one EPOCH value raised by 0.05; a SCORE value raised by 0.03;
    - another job interpreter; a changed code copy;
    - a VERIFY value moved by 0.001, and a VERIFY flag turned false;
    - a reference list with one wrong hash.
    Each R3 case gives its status: a cap by the driver and one by the launcher; a cap with a complete re-run; a
    second cap; a crash; a crash followed by a complete attempt that is not counted; exit 143 long before the cap; a
    re-score that stopped before any number.
  - The same test on comparator version 2 fails 13 of 23 (`test_compare_after_gpu_r3_before.txt`).
  - `test_compare_json_after.txt`, 13 of 13, covers a missing field, a shorter list, a slash in a non-path string,
    exactly 0.02, NaN and later rows. Version 1 failed 3 of them.

#### 6. Rules (bind steps (i) and (ii) only)

- **R1, tolerance.** A compared leaf reproduces when |new - logged| <= 0.02 + 1e-12 on the value stored in the logged
  file; exact equality is reported for every leaf. A count reproduces only when equal. A string or flag must be equal:
  a verdict word, a pass flag, rules.txt, the include bits. A leaf of the logged file that is missing from the new one
  does not reproduce, and neither does a logged pass that the new run lacks.
- **R2, what follows.**
  - A page number with a compared leaf that does not reproduce moves to the list "Claims that are not made" of the
    study README (gates.md line 134). The entry gives its run id, leaf, logged value and new value.
  - A changed verdict word or pass flag moves the claim it decides to that list, with the check named.
  - A sampled retrain that does not reproduce marks every claim that rests on its setting "not reproduced in the
    sample" on the results page:
    - Check 1a for the verify runs;
    - Check 3 and the final test's model side for the Wikipedia seeds;
    - Check 4b for the synonym seeds;
    - Check 2a for L1 to L3;
    - Check 2c for the news models.
    Its readers are re-run (section 3). The unsampled seeds of that setting stay "not re-checked".
- **R3, failure cases.**
  - A job that stops before computing any number (a smoke or path failure) allows one path-only fix, named now: the
    mirror rule of section 4, or the correction of a path in a job line. The fix is reported and the job runs again.
  - A cap is the driver's own `--max-minutes` (a "STOP: --max-minutes" line in log.txt) or the launcher's timeout:
    `src/launch.sh` exits 124 or 137, and the logger records exit status 124, 137 or 143 (128 plus the signal) at a
    wall time of at least the cap minus 2 minutes. A capped run is reported with its compared passes and started once
    more with the same arguments; the first complete attempt is compared. A second cap leaves every logged pass it
    lacks not reproduced.
  - Any other attempt that ends before it is complete is a crash: "not reproduced (crash)", with no re-run. A later
    attempt is reported but never counted.
  - A job with no complete attempt after these allowances is not reproduced. No further attempt is made: a complete
    attempt is compared whatever its result.
  - Step (ii) starts only if every step-(i) job is reproduced exactly (every leaf exact). Otherwise step (ii) does not
    start, and the environment difference is reported as the outcome.
  - Tiers 1 and 2 start only if the step-(ii) smoke job (`e1_verify_default`) is reproduced exactly, its VERIFY line
    included. Otherwise neither tier runs, the smoke job counts as not reproduced, and the difference is the reported
    outcome.
  - Tier 2 always runs after tier 1 (section 5).
- **R4, page wording**, chosen by the outcome (section 9 gives the numbers):
  - all reproduce: "The freeze rule was applied after the study ended (due 4 Oct 2026, run on 25 Oct and <date>), with
    the published code: N of M page numbers were re-checked, all within 0.02 (n exactly); the others were not
    re-checked (by class)."
  - some fail: the same, followed by "these numbers did not reproduce and are on the list of claims that are not made:
    ...", each one named.
  - GPU steps not run: "Only the readings of 25 Oct 2026 exist (steps (iii) and (iv), made before the plan was frozen);
    the dev re-scores and retrains of the plan did not run; the freeze rule has no result for the other page numbers."
- **R5, not re-checked**, each listed with its reason: classes A', E, G, H, X and D-not-sampled (apart from the rule of
  section 3), the 74 unmapped page tokens, the 19 test-side fields that the Check 1b replica leaves out, and the counts.
  Such a number is neither withdrawn nor called reproduced.
- **R6, environment.**
  - **Logged runs:**
    - job interpreter `tm-to-llm/.venv-htm` (gone). Its versions are known only from
      `../tm_language_model/env/htm.lock`, written 2026-10-10 17:36:38 UTC, eight days after the runs; it pins the
      package by git commit 5b4a223, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1 and pycuda 2025.1.3;
    - logger `.venv-tm`, Python 3.11.16. Its env.json lists the logger's packages, among them
      PyHierarchicalTsetlinMachineCUDA 0.3.1, not the job interpreter's;
    - Tesla V100-SXM3-32GB, driver 570.172.08, nvcc "release 12.9, V12.9.86"; 6 threads;
    - `gpu_run.sh shared` for 34 of the 36 dev re-scores and 10 of the 16 sampled retrains, `exclusive` for the
      others.
  - **Planned runs:**
    - `../decision_layer/.venv-htm`, built 2026-10-17 about 04:41 UTC (dist-info times): PyHierarchicalTsetlinMachineCUDA
      0.3.2 at commit 5b4a223 (direct_url.json), with the core versions of htm.lock. imageio 2.38.1, markupsafe 3.0.4,
      narwhals 2.27.1, pillow 13.0.0 and platformdirs 4.13.3 differ from it;
    - the same GPU, driver and nvcc (pre-flight, `test_preflight_r3.txt`); 2 threads; `exclusive` for every job.
  - Prediction for the time caps: the training runs on the GPU and the logged runs mostly shared it, so 2 threads alone
    on the GPU should not slow them. The headline and synonym retrains keep `--max-minutes 75` against at most 27.93
    logged minutes, which would need a slowdown of 2.7 times; the news runs keep `--max-minutes 35` against 18.034,
    1.9 times.
  - A leaf that differs is not reproduced under R1 whatever the cause; the report names the environment as a possible
    cause.
- **Multiplicity.** No multiplicity correction is applied; each number stands alone.

#### 7. Predicted outcome of each step

- **Step (i): exact**, in an environment equivalent to the logged one. Evidence: every one of the 36 logged dev
  re-scores has rescore_equals_in_run true; the re-score of the saved model gave the in-run dev AUC and accuracy
  exactly. The code on this path is unchanged as code (section 8).
- **Step (ii): exact**, in an equivalent environment. Evidence: the two free reproducibility checks of 2 Oct 2026 found
  identical training.
  - `20261002T043040Z_e4_curve_repro_check_posthoc` compared `20261001T212959Z_e4_sa_granmo_C50_A64_s100` with the curve
    run: dev AUC, accuracy and pred_pos identical at all 100 passes.
  - `20261002T131023Z_e4_second_repro_check_posthoc` compared `20261001T175704Z_e4_sa_best_C2000_s40_p50` with
    `20261002T123929Z_e4_second_C2000_s40_sa_s1`: identical at 14 passes.
  - docs/registrations.md lines 67-74 tie the first check to this rule: "Different: record the first differing pass and
    the size; that becomes the tolerance for the reload check and for the reproduce-within-0.02 rule of 4 Oct 2026". The
    check was identical, so 0.02 stands.
- If a trajectory diverges, a number moves like a new seed would. The seed spread of the reported dev numbers
  (`htm01/facts_round2.txt`, sample SD over seeds) gives the chance of a move above 0.02, by the normal approximation
  with the SD times the square root of 2:
  - same-article AUC 0.01; next-paragraph AUC 0.24; synonym-data AUC 0.03;
  - next-paragraph plain accuracy 0.64; L2 dev accuracy 0.62; L2 tuned accuracy 0.40.
  So R1 at 0.02 mostly tests that the environment reproduces bit for bit. It is applied as registered (gates.md is not
  changed). The report puts the first differing pass and the seed SD of the number next to each number that fails.

#### 8. Code since the logged runs (`htm01/code_since/table.tsv`, `summary.json` and the diffs there)

Definition: "reproduce" means the **published** code runs again on the same inputs. It differs from the logged copies
only as listed here. Counts are folders that hold a copy (run folders and their sub-folders).
`frozen_settings_hashes.json` is the published record. Its README section ("The hash record") says that 14 of its 21
script values were replaced by the hashes of the published files, that six scripts keep their registered pins, and that
the pin of text_driver.py (8ee7063b..., the version of the news-topic runs) does not match the published file
(6f5a2f3a..., which runs now).

- `src/htm_driver.py` (steps i, ii, iii and iv):
  - 152 folders hold dc31407e..., the version of the final test (report.json `inputs_sha256`): 4 changed lines,
    docstring and comment, equal as code.
  - 29 folders hold ab8c6233..., among them the tier-1 runs `e4_sa_granmo_C50_A64_s100`, `e2_L1_hier_s1` and
    `e2_L2_hier_s1`. 8 hold 5e183986..., among them both `e1_verify` runs. The other sampled retrains hold dc31407e...
    (hashes of the kept copies). Against the published file these two older versions have 30 and 32 changed lines:
    - new options `--eval-every` (default 1, which evaluates every pass as those runs did), the early stop (off by
      default), `--save-at` and `--save-model` (off unless given; none of these runs gives them);
    - a reordered time-cap check, inactive here because none of these runs sets `--max-minutes`;
    - `best_pass_by_dev_auc` read from the history, which gives the same value when every pass is evaluated;
    - for 5e183986, an ALWAYS-TRUE text for A = 1; this is printing only, and those runs have A = 8.
  - Predicted effect at the logged arguments: none.
- `src/pairdata.py`: 14 folders hold 062141da... (8 lines). The change is a refusal of `ns` at a vocabulary other than
  1,024, which no job of the plan uses. 163 folders hold the current file.
- `src/text_driver.py` (tier 2): 12 folders hold 8ee7063b... (10 lines: `--class-names`, printing only, default
  unchanged); 10 hold 51e339cf... (2 lines, equal as code).
- `src/data_checks.py` (step iii):
  - 15 folders hold 72b0d717..., the registered and final-test version. The diff is a docstring line and the HTM-B2
    fix, the declared change.
  - db78c8a2... is in 1 folder, `20261001T092321Z_data_checks_exploratory` (class G).
  - a198fd40... is in 2: `20261001T093113Z_data_checks_v2_exploratory` (class G) and
    `20261001T093837Z_data_checks_v3_exploratory` (not cited).
- `src/check_ns_all.py` (step iii): its 2 copies equal the current file.
- `src/final_test_report.py` (step iii):
  - 5 folders hold eff80607..., the registered version: `20261002T134938Z_final_test_report_devcheck_exploratory` and its
    sub-folder, `20261002T135204Z` (class H), `20261002T164442Z` (class C) and the final test. The diff is two comments
    and one output string (an upper-case note on the frozen settings became `"(frozen settings)"`) in
    dev_report.json, a field the step does not compare.
  - Older copies (a242f69c..., 6b617bcc..., 5ccab03e...) are in the dev checks and the code-path test of 2 Oct, 13:35 to
    13:47 UTC: `20261002T133519Z`, `20261002T134017Z`, `20261002T134414Z` (class H) and `20261002T134702Z` (class H).
- No copy in any run folder, so no diff can be shown:
  - `score_saved.py`, `pcp_recovery.py`, `pcp_purity.py`, `apply_rules.py` and `compare_runs.py`: registered and
    published hashes are equal;
  - the earlier `score_saved.py` that wrote the two saveload references (`20261002T031952Z`, `20261002T032557Z`; SCORE
    lines without the bootstrap block): not kept; no diff can be shown;
  - `crossfit_acc.py`, `crossfit_splits.py`, `e5a_choice.py`, `e5a_check2c.py`, `npmi_groups.py` and `wiki_groups.py`:
    registered and published hashes differ by the publication edits of comments, docstrings, messages and path
    constants. One such edit changed the printed note of `wiki_groups.py` (section 11);
  - `pcp_world.py` and `lr_agnews.py`: not in the hash table.
- The launch chain: `src/launch.sh` and `src/interpreters.sh` changed on 2026-10-25 (HTM-B1; the fixes note).
  `../tm_language_model/env/env.sh` was changed on 2026-10-25 in the same pass (2bca37b6... against
  fe72f61c... at HEAD). The launcher of 2 Oct (`src/launch.sh` of the original working folder) is not kept, so its diff cannot be
  shown.

#### 9. Outcome table (form fixed now)

One table, one row per class. Columns:
- class;
- page numbers (mapped tokens on the class's runs);
- re-checked;
- exact;
- within 0.02;
- not reproduced: outside 0.02, missing, or a changed string or flag. Each is listed below the table with its run,
  source, leaf, logged value, new value and the page tokens it carries;
- not re-checked, with the reason.

Rows A and C carry the readings of section 4 with their label. Below the table: every compared leaf outside 0.02,
missing or changed, whether it carries a page number or not, and the 74 unmapped tokens by line.

**Join keys, fixed now** (`htm01/page_map.tsv` columns page_line, token, run, source, leaf, value; job records of
`out_gpu/compare_after_gpu.json` in its lists "i" and "ii"):
- a candidate row is compared by a step-(i) job when `run` equals the job's `logged` and `source` is SCORE. It is also
  compared when `run` is the `source_run` of a shuffled-label job, `source` is EPOCH:100 and `leaf` is dev_auc,
  dev_acc, dev_tuned_acc or pred_pos;
- a candidate row is compared by a step-(ii) job when `run` equals the job's `logged` and `source` is FINAL or
  EPOCH:<pass>, the timing fields excepted;
- its status: not reproduced when the job's `reproduced` is false and either its `leaf` (last path element) is in
  `not_within_0.02[].path`, `missing[]` or `other_mismatch[].path`, or the job failed on a check that covers every
  number (status not "reproduced" for an attempt, job_python, code copies, include bits, VERIFY or dataset hashes).
  Otherwise it is exact when the job's `exact` equals its `numbers`, and else within 0.02;
- a token takes the worst status of its compared rows. A token with none is "not re-checked", with the class of its
  candidates' runs in class_map.tsv.

#### 10. Use of the test split

- **Final-test split: 1 use** (registered, 2 Oct 2026): the 20 test re-scores at 19:28 to 19:35 UTC (first run id
  `20261002T192818Z`, last finished 19:34:55Z), then `20261002T193513Z_final_test_main_table`.
- **The re-read of 2026-10-25** (11:52:31Z to 11:55:11Z): step (iv) re-read the saved outputs of that use before this
  plan was frozen, under the finish-pass assignment. It is a recomputation of those saved outputs, not a new use.
  - It opened report.json, the 20 `scores_test.npz` files (arrays score, pred, ps_score, ps_pred, y, art_a and par_a) and
    the SCORE lines of the 20 test re-score logs. The sha256 of each scores file equals the value that report.json
    recorded.
  - That it opens only these files is what keeps the step to saved outputs; a guard also refuses any file named
    test.npz.
  - No model or baseline was scored and no pair text was opened.
  - No statistic beyond the registered entries and the cross-checks of section 11 was computed from these files; the
    intervals with bootstrap seed 12345 are descriptive and support no claim. Any further statistic from these files
    would be a new use.
- Steps (i) to (iii) read train and dev data only.
- Both first-round reviews, judged by an AI model, not a person, found this reading acceptable. Whether this
  recomputation counts inside the one registered use is the decision of whoever freezes the plan, not of the reviews.
  If it is not accepted, step (iv) is withdrawn and class A moves to R5.

#### 11. What was known when this was written (as of the freeze)

- **Step (iv)** (`htm01/out_iv/iv_recompute.json`):
  - 529 of 529 numbers equal the logged ones exactly: per seed AUC, tuned accuracy and their interval ends; means and
    SDs; shuffled-label AUCs and intervals; the model side of each paired comparison; every field of each SCORE line
    with its five intervals.
  - Cross-checks: AUC from scikit-learn and tuned accuracy from a separate scan agree with the study's functions to at
    most 1.1e-16 (35 values). Intervals with bootstrap seed 12345 move the 70 interval ends by at most 0.0015 (median
    0.00045).
- **Step (iii)**, round 2 (`htm01/iii_r2_counts.txt`; every non-exact value in `htm01/out_iii_r2_non_exact.txt`):
  - 26 jobs, all exit 0; 4,337 numbers compared; 4,298 exact; 4,330 within 0.02.
  - Outside 0.02 (7), all in excluded leaf types:
    - three Check 1b shares moved by the HTM-B2 fix: `check1b_dev` `ns.length_gbt.pos_share` 0.5688 -> 0.533,
      `ns_all.length_lr.pos_share` 0.4999 -> 0.6743, `ns_all.length_gbt.pos_share` 0.5891 -> 0.6265;
    - four wall times: `runtime_s` of `check4a` 125.1 -> 123.3, `check4b_s100` 90.3 -> 88.0, `check4b_s400`
      144.4 -> 163.4 and `check4b_s200` 120.7 -> 115.0.
  - Not exact but within 0.02 (32), all moved by the HTM-B2 fix: 14 values of Check 1b (ns and ns_all: the accuracy,
    interval ends and AUC of the two length-only classifiers, and `ns.length_lr.pos_share` 0.4812 -> 0.496), and 18 of
    Check 4b part 1 (accuracy, AUC and their interval ends, 6 values in each of the runs at s 100, 200 and 400).
    - Largest moves: 0.0011 in an accuracy, 0.0012 in an accuracy interval end, 0.0028 in an AUC and 0.0025 in an AUC
      interval end.
    - At the three decimals of the results table, the fixed function gives 0.499 [0.492, 0.506] for the boosted trees
      on ns (logged 0.500 [0.493, 0.507]). The upper end for the boosted trees on ns_all moves from 0.510 to 0.509; the
      other length-only cells are unchanged.
    - Every length-only interval still includes 0.50, so no verdict changes, and the share flag of gates.md line 138 is
      unchanged (section 2).
    - The version before the fix (896f2bdc...) reproduced the 7 logged records exactly (`length_checks_rerun.json`).
  - Missing, by design (19): the leaves that need the test split, which the Check 1b replica leaves out:
    - `n_test` of sa, np, ns and ns_all;
    - the test overlaps of ns and ns_all at 1,024 and 3,072 words;
    - the 11 bins of the test histogram of shared words of ns.
  - Changed string (4): the printed note on the gate method in `check4a` and the three `check4b` jobs.
    - Its last sentence deferred the choice of method to a later decision, with the bootstrap deciding meanwhile and the
      calibrated result printed next to it. It is now "Under the frozen settings the bootstrap decides ...".
    - That is more than wording. gates.md (decisions of 2 Oct 21:28) and decision 1 of 21:27 CEST say that the
      calibrated test now decides, so the published note in `src/wiki_groups.py` conflicts with decision 1.
    - The wording fix is made in the documentation, not in this plan.
  - Not compared:
    - 216 paths found only in the new recovery.json of `recovery_all_e2`: rows for 19 runs that finished after its
      logged file was written (that run started 2026-10-01T23:43:30Z; the earliest of the 19,
      `20261001T232651Z_e2_L3_hier_s3_p1000`, finished 2026-10-02T00:10:26Z);
    - 3 path values.
- **The record of the earlier steps**: both free reproducibility checks and their registration (section 7), the seed
  spreads (section 7), and the code since the runs (section 8).
- **Every number of the results page**, report.json of the final test, and the run records (wall times, commands, exit
  statuses) used to write the job lists.
- **The pins against the logged records** (`htm01/pins_vs_logged.txt`).
  - 120 of the 130 pinned files equal the record the study kept: data files against `frozen_settings_hashes.json`, the
    L1 to L3 pair files against the dataset_hashes.json of their runs, the 40 models and their logs against their
    MANIFEST.json, and the borrowed scripts against git HEAD.
  - 2 differ, as declared: text_driver.py (section 8) and env.sh.
  - 8 have no record: the vocabulary files of the L1 to L3 pairs, and `src/launch.sh` and `src/interpreters.sh`, which
    were changed on 2026-10-25.
- **The pre-flight result** on the final pin lists (`test_preflight_r3.txt`): 130 pinned files equal, the same GPU,
  driver and nvcc as logged, the five package differences of R6, and PREFLIGHT PASSED. Its two negative controls stop
  with exit 1: a changed pin (`test_preflight_r3_neg_hash.txt`) and HTM_PY unset (`test_preflight_r3_neg_env.txt`).
  The tests of section 5.
- **The ceiling of N** (page_map.tsv with jobs.json, the join of section 9): 124 page tokens can be re-checked at all.
  123 of them through SCORE leaves of the logged dev re-scores (step i); 1 directly through step (ii), line 171's
  0.745, the FINAL dev_auc of the free-check run; none through the shuffled-label EPOCH lines. Step (ii) matters
  mainly through bit-identity and the readers of section 3.
- **Not known:** any number of steps (i) and (ii). No GPU job of this plan has run.

#### 12. Selection

The retraining sample is purposive, fixed in `htm01/gpu_jobs/jobs.json` before any GPU job.
- **In it:** one seed per setting on which a check rests, plus seeds 2 and 3 of the three Wikipedia sets:
  - the two Check 1a verify runs;
  - seed 1 of the three Wikipedia sets;
  - L1 to L3 at their verdict rows;
  - the hierarchical and flat news models.
- It includes the run already known to reproduce (the free check's `20261001T212959Z`), so tier 1 starts with a predicted
  success.
- **Not in it** (the 37 D-not-sampled rows of class_map.tsv):
  - seeds 4 and 5 of the Wikipedia sets and the shuffled-label models;
  - the curve run, and the s-screen runs at s 21, 40, 200 and 400;
  - the 1,000-pass L3 runs, L3 at s 400, the unseen-pair runs and the flat L3 controls;
  - the four switch runs, the demo runs, the evaluation-schedule runs and `e1_p1word_hier`.
- Not cited, so not in the class map: the training runs of the second setting (`e4_second_C2000_s40_*`); their dev
  re-scores are in step (i).

#### 13. Times, pins and history

Times are read from the clock (`date -u`) in the command that appended them; hashes are `sha256sum` output pasted by that
command. Drafts 1 to 3 (2026-10-25T11:52:10Z to 14:00:01Z) were reviewed before the freeze; draft 4 is this text. No
rule was changed after a result of a step that it binds was seen.

##### Draft 4: pin block, pasted by command at 2026-10-25T16:24:09Z

Base path of block 1: <finish-pass records> (sha256sum output):

```
d0b8d374cb4b81da8fe7be9c4bfd77d89371b4839774c7f61c13b5f5b55176da  htm01/preflight_pins.sha256
ec7e775085132763b6299517ef0054e79d93b9100bd39faac58e210f84d601a7  htm03/preflight_pins_htm03.sha256
8e9c9801435a700cd509b626165acd8b84de7c8dbcfb58a685eacf6ec1449861  htm01/reference_files.sha256
3cb39cea384b12aebf51cd89ccf0c9a78adf5c9239dfe89dacc5e130b8a3ab30  htm01/gpu_jobs/jobs.json
d5fe127764f8c535bb55c643c5489ddecd5b0a6bc13bb8ec5405264f8134e3e1  htm01/gpu_jobs/gpu_htm01_i_smoke.sh
9e84a2d52c478a1bdbcb815c1ea605b88256065448c1c0b0d14e7d17370921d9  htm01/gpu_jobs/gpu_htm01_i.sh
b31f4c1f35b0a26e8c12a713b04e81b3f40931727f01c46c827f47668f9074f2  htm01/gpu_jobs/gpu_htm01_ii_smoke.sh
3ea1dfac2b58056990ea96b0dc78480327aa47cca18b3cb9de816a995c8b9703  htm01/gpu_jobs/gpu_htm01_ii_tier1.sh
3d0ddce761decec3111bfc1fe022dc738c6c73ac4401704b6a9b950f8a06932c  htm01/gpu_jobs/gpu_htm01_ii_tier2.sh
db8b6a028c62df3616a19394788457b50a373d2c79929034d1b66f26a753ddac  htm01/compare_after_gpu.py
7311953e756e10a61201cd336dbcfc736f562217eefcb64bf8e78dd2206b8cec  htm01/compare_json.py
c13775489184c83c7a91d5cf65064438db18d67aeee204e06a54297d0d54f8b5  htm01/preflight.py
43c9c71037567bfde6528902bf9ccdd2bf82fce0135ee382a86dd1ea71c24a97  htm01/make_gpu_jobs.py
b4d3cea4252ba1db0508c07a482ffbede79da1dc9c38dfef5e4410247c8462d2  htm01/make_pins.py
50b44125390a45f9973aa816a84abb0b0213db19af7592e7a7d4150dd4da5a9e  htm01/run_iii.py
bac37d8d7b9cda6b071f41bfe10066a6d181d73083e7c49f373c06daccfdeaf7  htm01/run_iii_r2.sh
d55c84230664daba40773ba67638ab19df3e53ebfaa366730c821c24de92685a  htm01/check1b_dev_replica.py
0a4dd1ddf0509c5d1bd8570494e37b3e1259cae47a189c4c48d14ad5f398d1b7  htm01/recompute_test_from_scores.py
f6a62239d5e8219e562701faee6800cb293503a180c0af42c2a061d59c77b344  htm01/page_map.py
70b51728223954c0780bdddd5c5bfeb81782cbc8f63e1093737a818791091441  htm01/page_map.tsv
707e460557f3c3eb1641b32b9a73bfbd9a8ec2e9854923c61b4822348dd1d6c0  htm01/page_tokens.tsv
6535c31f218924b8eaf58c2356bc5d00c852b5e0aabbac249b3451d2d283705e  htm01/page_map_counts.txt
452c4f90267f5d7ba92ba7243f3739e07d705f99b6366035a0bd0341045983bf  htm01/cited_runs.py
3d71cd048bad7e9f0b67d7dd1ee18a731edfb8166928ec3ee6783ab45c5f31b8  htm01/cited_runs.tsv
10a79d0cf17b30f0d5ff8cdd913a1aa5067fbc31b579f0764d291a19c7538b01  htm01/class_map.py
6f4848a0b4d039a613ffbae36775d711580f141dbb1f1c414b95ee4adfcc7dd7  htm01/class_map.tsv
7244435342df8d6916a0a77118559f3859202fd5981783d6b12eeae0c16db3f3  htm01/class_counts.txt
3303b050812444df9087a925045431af1efc482e7de89eb820b25996d8576d25  htm01/code_since.py
4236c9fa2b83ac2fd1e0de696f38ed5568eabbf2b9068946a7ecd5768963cb15  htm01/code_since/data_checks_72b0d717.diff
375ffc99dd1e17102df1f634435e1a2e14b463cfafe54430d5eff875dd93b237  htm01/code_since/data_checks_a198fd40.diff
6307abb402923686673c5725b1a622e769fe93a9381a9b5aba7ea3626b4debdc  htm01/code_since/data_checks_db78c8a2.diff
7d6214db3146135d5d7dca79af06992a7b3d0a6e3f35cf0b43fd1b2d75578cd0  htm01/code_since/final_test_report_5ccab03e.diff
8341407b21c54f9aac4a0ced09759becfdeb7f12ca1977e358b4b0d593a65421  htm01/code_since/final_test_report_6b617bcc.diff
3a5e5514ebd6f961f4cce8f70977994b1317074beab6f762a5c37d4889de3735  htm01/code_since/final_test_report_a242f69c.diff
360d1e2dc7d963d6f099e43bee063c17205fd85d7c6bb91bd1fef56265608b7b  htm01/code_since/final_test_report_eff80607.diff
c3cd5f4d22eab31921726ac4e36c76b1b979ae75d22491147b391e8801bacb82  htm01/code_since/htm_driver_5e183986.diff
f3bdbb4f816343f554ea82d725874bd5e441d676ecf9f89b6c0c5885c250454c  htm01/code_since/htm_driver_ab8c6233.diff
7bd3cab257940da66ae94bac22d709e1a76a549d41cd4e646850f6bb0e76a25a  htm01/code_since/htm_driver_dc31407e.diff
fbae3b1f15626232e8bd550aea08d64a7ec70b7e0a5745c522c84eb4c4196144  htm01/code_since/pairdata_062141da.diff
e9defd11ca3fd627a09c157c206843f7d0412e55a474099545609d73cf2845e2  htm01/code_since/summary.json
f3fb6877aa2029f7222eb81f005222a1b847a5a02a4fe03827c30239453df883  htm01/code_since/table.tsv
deccb03642eb8bc8971f41e87a2cd99801f27cd0c9399c8c638cb0a218a016d0  htm01/code_since/text_driver_51e339cf.diff
85802e1d3a969fa091a2a06bbe90fc80a260911e96698a4ceffbed40353fafff  htm01/code_since/text_driver_8ee7063b.diff
b44ca64a1e948fbc58439dbca872d9bf9316b7d1e6c231edafafd463708d9993  htm01/facts_round2.py
b337f356ffa694db0c1fa4b1a27d9e4f7f0dc7e917b8116e677ce732ada558c7  htm01/facts_round2.txt
4e6438424a42e6c0031ee2cb7b0be6f46927add0f7471b34209fd3fc2dbb03b6  htm01/out_iv/iv_recompute.json
f8e30bc405c981329663e8ca1dedb59b652e2b1cd9a49f728c33283fedce21ca  htm01/out_iv/iv_recompute.md
bc47f737772950210d25f647fbd67ee6b3bfdfcb7e414af4d28a4581c45cdb0c  htm01/out_iv.log
ef3ac35994c2b80865e35ebede4905881110bf428950beabdb20bc58f8311036  htm01/out_iii_r2/iii_results.json
85bd3c9978ecc5fe74bd6497558769617e4b98907c2461607a05926328fead97  htm01/out_iii_r2/iii_results.md
683477f92c07e04ab2b8e70aa5cf0996177942703c61065bb152a610fe103e6f  htm01/out_iii_r2.log
daf6cdc723a0d23cb78955dfb9424c69c66b3af76812c1eadf1b9b3a00daf1cd  htm01/iii_r2_counts.txt
37832b85a6fb1a238850798c788ea1da0b803a01b6b8f71451b6ebcc5b5c299a  htm01/list_non_exact.py
c07a547a04c4da3d4b73445fa39bb5fe040c0c3fecf60e6a7d54903e74351fda  htm01/out_iii_r2_non_exact.txt
0bd6b0b22193f252b0d038f8adaca384d8983ea6642c594d4d608c9e9656c9cf  htm01/check_pins_vs_logged.py
175ddec08bc0e6fc1e3c3dee8ae3c83bc2bd4c1c900f009aacbbd50b179a7112  htm01/pins_vs_logged.txt
138b93d8cf57e7ef77712652ad872aacc5dcb220c2917d85fb7ec82bb0b1fca1  htm01/replica_diff.py
5b58deda5eae950f58149c21bd30cc4f2ecb2705c8a7b12865a68d4a46973982  htm01/check1b_replica.diff
8a65431d1859413fb11daddf88190c63e523e72785021d08419b4dde7b737af2  htm01/readme_tokens.py
cc7e7f028eb42026a543eb2d0e8f338f8c7720e87b6d03630b606b2d6fcedfe5  htm01/readme_tokens.txt
54cb3e8e2032987e527e9b293c78f98ede973e835e75b2c52d7c6994cf15ebba  htm01/test_compare_after_gpu.py
743861b0e4db7cd2ff8e1f7ce7528da6b59aa094a031a017eca29d73b8e2b7d3  htm01/test_compare_json.py
56001ee94403f20a727d088f68d29099b74186761c13dd41b976b271fd559ce3  htm01/test_job_lines.py
d1f415c31bbd7da469b3cd46b12ff05bb4a90df09b3a6e0dfb3d4f9ecd9ad5ab  htm01/test_run_iii_subst.py
4cfabd7cca27f93ce33f36a719e3db143173bcfb20e7f64d4779819650cca1cb  htm01/test_round2_helpers.py
6ebb8dafe8aff5a727f6098eb173d11bc72f29b0d986db22b5e102c3dfa67f37  test_compare_after_gpu_r3.txt
cd63444acfcda7b0cc90f1eb1aaf7896a5e059d64075093f5d98f72870d9448c  test_compare_after_gpu_r3_before.txt
90a0ba73014cfbe07a550f22695d084b232fea9fc677c425e23dad41747aba98  test_run_iii_subst_r3.txt
20783c2b5235c86c267c9ad23ab228a90db3301b6de94cd2458a1d26a84a646f  test_run_iii_subst_r3_before.txt
dd6238a57c7ed56c2bdd4c69081a1cff6de0df71eb7d4c4af05205c533a48b4e  test_preflight_r3.txt
12619334a00448b4029a46b66e450f9c8bca87f049cce927c7ec71f50f1299f6  test_preflight_r3_neg_hash.txt
2e912d4963795e2082171ab2124c8af80a34664cb9bf6441c0ebfa174f470dfd  test_preflight_r3_neg_env.txt
a5edd009ef7d9ed45d137acc1ba2654eeb7d7ad1526623912dff157495605867  test_compare_json_after.txt
3e9d45fd26be64b5e0b370c7b80239bd5553bc88e0afe97be32c22a0880e84d6  test_compare_json_before.txt
42d131bb6cd4bbd1a3156a14a6da5ace57ffa17cc0325678f5b6a6d85114ed2f  test_job_lines_after.txt
f6e4fdde4649b5f278bed93f06152a88cb94d45a5745e9776e12dbe5d91c060f  test_job_lines_before.txt
f454440e3d92f9402196ba162fbe2e4c24abda5af2eb5e5947af1d9c943c69bf  length_checks_rerun.json
9eedda8ccdd6e72d5da2277d832909339797acbffa2e7364425871d4e37af6e3  HTM01_results_steps_iv_iii.md
b2c0ee74fa6ea255c4b3c6aca7b9d99fe75fcbd06389c664a4c02609c7d0f816  GPU_REQUESTS.md
```

Base path of block 2: studies/hierarchical_tm (sha256sum output):

```
e9ca0667dd9ec08e24663a30f3248598ac328f7db2ed603f548e5dc4ebd51562  docs/results.md
5f09ef7b5e1a99345f24fcc52e62571a3e650f9962962af015f3396c3e291596  gates.md
ef0eea12985626b7d899509be11c5bb7eb2db0291e3a9ee2bd6fab328fa54cc8  docs/registrations.md
537c35b65c793a2663a5d137e3e24ce4d82e94ab4933d0e3f538d6d9e1e34950  frozen_settings_hashes.json
```

##### How the GPU stages are queued (added before the freeze, 2026-10-25T16:31:45Z, date -u)

R1 and then R2 with R3 run as two jobs of the serial GPU queue, `bash htm01/run_htm01_chain.sh r1` and
`bash htm01/run_htm01_chain.sh r2r3`. The script runs the GPU scripts and the check command of GPU_REQUESTS.md in that order, and
`htm01/gate_htm01.py` applies this plan's conditions between them, reading `htm01/out_gpu/compare_after_gpu.json`: job 1 of step (i) exact before
jobs 2 to 41, every step-(i) job exact before R2, and the step-(ii) smoke exact, its VERIFY line included, before tier 1. Tier 2
follows tier 1. A failing job stops the stage; a capped job's one re-run, R2c and any path-only fix are done by hand, as the
rules say. Tests: `test_gate_htm01.txt` (10 of 10, on a self-test of compare_after_gpu.py v3 and edited copies) and `test_chains.txt`
(12 of 12, on a stub tree). The freeze tool and its test are listed too. sha256sum output, base ws/htm:

```
507935c150d3b0e2f4328d77d9debf9b9310769b79e0404a53e5f636daab989c  htm01/run_htm01_chain.sh
f746ca66504360e210a768f82f61a81f5448d2ad4ce5a399a8cd23f166efdeb9  htm01/gate_htm01.py
0ddff66df5a4ff085339743120d4ecb2de79fb2569f3c9f3a2877f76be4235c6  test_gate_htm01.txt
e1eb33d1f1f3db8753f9e4c7d6e67426ba0eb927801cab8d9f73d1eeeaf39528  test_chains.sh
e1cca2813d11a1013aedd70506d7384b38e79acccbbdb8f86ff8e42d48efdb7b  test_chains.txt
166ae1d6117e55648072bf20944e00ed09ed6d45a879008d70cc0026577ef01e  freeze_htm.py
19d03ca3b8f4f2e29c0d7b27cdb5b0120d818ab49906c47859b71a9a53f416fa  test_freeze_htm.txt
```

## 25 Oct 2026, 17:31 CET (16:31 UTC): L2 at `s` 100 with A = 8, exploratory, after the study ended

*Publication note.* The text below is the frozen file (sha256 `cc17decaa4243765a53d9ddc5a5c6074b9e9587649fd3d8ecfbfd324b4fba02e`, frozen 2026-10-25T16:31:59Z; the frozen file itself is not distributed), edited for publication in wording only: the notes on its review and drafting are shortened and private paths are left out. Every rule and number is as frozen. workstation paths are shortened (the finish pass's folder is written `<finish-pass records>`, paths in this repository are relative to it, and the home folder is dropped from the others), and the headings move down two levels to fit this page. 24 of its 363 lines differ; nothing else changed.

### Frozen 2026-10-25T16:31:59Z (draft 4): L2 at `s` 100 with A = 8, the registered "cheap morning test" that never ran (HTM-03)

Written in the finish pass of 2026-10-25. Exploratory, dev only, run after the study ended. It cannot change any verdict:
Check 2a stays not met for L2 at 100 passes, whatever this test shows. Drafts 3 and 4 answer two pre-freeze reviews (one
NOT READY with one blocker and five majors; a re-check READY AFTER FIXES with one major and eight minors; both judged by
an AI model, not a person).

**Decision to run.** The registration asks for "a recorded decision" (docs/registrations.md line 95). The record is kept
outside this frozen text, so that it does not change the text's hash: a file `htm03/DECISION_TO_RUN.txt` that names the
sha256 of the frozen note and the time read by `date -u` in the command that writes it, and a research-log row, both
written after the freeze and before the first GPU job.

**Times.** On 25 Oct 2026 Norwegian time is CET = UTC+1 (summer time ended at 01:00 UTC that day, the last Sunday of
October).

#### Question and the registered prediction

The registration of 2 Oct 2026, 03:05 CEST, in its undated notes (docs/registrations.md lines 94-96), says:

> **Capacity mismatch (hypothesis).** An L2 hidden group has 8 words, but each OR-group has A = 32 options. At `s` 100,
> coverage 0.92 with purity 0.49 means about 15 words per group: about 7 right and 8 intruders.
>   - **Cheap morning test, needing a recorded decision:** L2 at `s` 100 with A = 8, all else equal (about 4 min per
>     seed). Prediction: purity and recovery rise, and AUC stays similar.
>   - This separates "too many slots" from "forgetting too slowly"; the `s` schedule tests the second.

The results page records it as never tested (docs/results.md line 366). Question: at L2, `s` 100, 100 passes, does
cutting the options per OR-group from 32 to 8 raise purity and group recovery while dev AUC stays about the same?

#### What was known when this was written (all values read from files by the scripts named)

- **The comparison arm, A = 32 at `s` 100** (`htm03/paste_a32_values.txt`, pasted at the end): dev AUC 0.964373125 /
  0.967543625 / 0.950257125, recovery 0.0 / 0.25 / 0.0, purity of all groups 0.487 / 0.513 / 0.467, coverage 0.919 / 0.934
  / 0.897. Seed ranges: AUC 0.0173, recovery 0.25, purity 0.046, coverage 0.037.
- **A = 32 can hold pure, recovered groups at a lower `s`.** At `s` 21.1 after 100 passes: recovery 0.625 / 0.5 / 0.75,
  purity 0.68 / 0.66 / 0.68 (results.md lines 62 and 71). After 1,000 passes: recovery 0.875 / 1.0 / 0.875, purity 0.92 /
  0.935 / 0.92 (results.md lines 64 and 69). So "too many slots" is not the only reading of the `s` 100 result: the same
  32 options give recovered groups when `s` is lower and training longer.
- **L3, purity against `s`** (registrations.md line 81; results.md line 364): purity fell as `s` rose (one seed).
- **The slot-count reference** (`htm03/slot_reference.py`, output `slot_reference.json`, run 2026-10-25T13:12:42Z to
  13:12:55Z with the study's own `pcp_purity` and `pcp_world.recovery`). Step 1 reproduced the logged A = 32 values
  exactly on all three seeds. Step 2 kept 8 of the 32 options of every OR-group at random (2,000 draws per seed, numpy
  seed 20261025) and scored them with A = 8: purity of all groups 0.5946 [0.561, 0.630] / 0.6152 [0.583, 0.647] / 0.5693
  [0.539, 0.603] (mean [2.5%, 97.5%]), coverage 0.4825 / 0.4935 / 0.4666, recovery mean 0.0141 / 0.0230 / 0.0053 (above 0
  in 10.85% / 17.65% / 4.15% of the draws; 97.5% point 0.125 on every seed). Purity counts a group's distinct words, which
  the number of options bounds, so fewer options raise purity by themselves, with nothing relearned. A pre-freeze
  review (judged by an AI model, not a person) found nearly the same values with an independent pure-Python check
  (HTM03__both_checks/slot_null.py: 0.594 [0.56, 0.63] / 0.615 [0.583, 0.648] / 0.570 [0.536, 0.604]).
- **The size of clause values changes with A** (a check of the same review, judged by an AI model, not a person,
  HTM03__both_checks/or_scale.py, sha256 e6d3b3e2..., not recomputed here): at A = 32, `s` 100 the best word's OR-sum per group averages about 4.8 and the best clause value
  about 23 (maximum 42); a recovered A = 8 group gives OR-sum 1 and clause value 1. T stays 128, so A = 8 is also a
  different feedback regime, not only fewer slots.
- **Chance of a sign rule** (`htm03/sign_rule_chance.py`, seeded): with no effect and independent normal noise, "higher
  on at least 2 of 3 seeds" holds with probability 0.4989, and together with "higher mean" with 0.4185 (400,000 draws).
- **Selection.** The hypothesis was formed from these three A = 32 runs ("coverage 0.92 with purity 0.49"), and `s` 100 is
  the headline value chosen on seed 1's dev data (registrations.md lines 104-107). The comparison values are not
  independent of the question; regression toward the mean favours a rise.
- No L2 run with A = 8 exists (the run index has none; results.md line 366). No rule of this note was changed after an
  A = 8 number was seen, because none exists.

#### Predicted outcome of each part (from what was known)

- **Purity rises (registered reading):** predicted to hold from the slot count alone (reference means 0.5946 / 0.6152 /
  0.5693 against 0.487 / 0.513 / 0.467). A held purity part therefore cannot separate "too many slots" from "forgetting
  too slowly".
- **Purity beyond the slot-count reference:** no prediction.
- **Recovery rises:** no confident prediction. The reference stays near 0 (mean 0.014 / 0.023 / 0.005); recovery needs
  purity and coverage together (Jaccard at least 0.8 with a hidden group), so it does separate the readings.
- **AUC stays similar:** no prediction; fewer options with T unchanged change the size of the class sums.

#### What runs (study folder; through the serial GPU queue, one GPU job at a time)

The job scripts are generated from the run records by `htm01/make_gpu_jobs.py` (version 2) and set, before the first job:
`set -euo pipefail`, `cd` to the study folder by its absolute path,
`HTM_PY=studies/decision_layer/.venv-htm/bin/python`,
`HTM_TOOL_PY=studies/tm_language_model/.venv-tm/bin/python`,
`PYTHONDONTWRITEBYTECODE=1`, 2 threads, and the pre-flight check (`htm01/preflight.py` on two pin lists; it stops on any
changed file or package). The two lists: `htm01/preflight_pins.sha256`, shared with HTM-01 (the code, interpreter,
environment and data files that both GPU parts use), and `htm03/preflight_pins_htm03.sha256` (17 files of this note).
HTM-01's plan, and with it its list, is frozen before this note, and this note pins the sha256 of both list files. The
queue line of each HTM-03 job first checks those two hashes (`sha256sum -c`) and does not start the script on a
difference, so a later change of HTM-01's list cannot change what this note's jobs check. Every job line passes the driver as an absolute path
(`"$PWD/src/htm_driver.py"`), because `src/launch.sh` changes to `../tm_language_model` before it starts the job;
`htm01/test_job_lines.py` runs each line's driver with `--help` from that folder and parses each line's arguments (224
checks, all passing; it fails on the round-1 lines with the relative path, exit 2 "can't open file").

1. `htm03/gpu_jobs/gpu_htm03_smoke.sh`: a 2-pass A = 32 run of seed 1 with the logged arguments of
   `20261002T005202Z_e2_L2_hier_s1_s100`, then a 2-pass A = 8 smoke of seed 1.
2. Check: `studies/tm_language_model/.venv-tm/bin/python -I <finish-pass records>/htm03/score_htm03.py
   <finish-pass records>/htm03/gpu_jobs/jobs_htm03.json
   <finish-pass records>/htm03/out_repro --repro`. The A = 32
   run must reproduce EPOCH lines 1 and 2 of the logged run exactly in every field except the timing fields (log.txt
   lines 12-13: dev AUC 0.54189225 and 0.624357125, dev accuracy 0.52125 and 0.53475, and all other fields); the smoke
   must end with exit status 0 and two EPOCH lines.
3. If the reproduction is exact: `htm03/gpu_jobs/gpu_htm03.sh` (the three A = 8 seeds, the logged arguments of the three
   A = 32 runs with `--A 8` as the only change). If it is not exact: first `htm03/gpu_jobs/gpu_htm03_a32_fallback.sh` (the
   three A = 32 seeds again with their logged arguments), then `gpu_htm03.sh`, and the comparison uses the new A = 32 runs
   (`--arm32 fallback`); the report says so.
4. Scoring (CPU, about a minute): `studies/tm_language_model/.venv-tm/bin/python -I <finish-pass records>/htm03/score_htm03.py
   <finish-pass records>/htm03/gpu_jobs/jobs_htm03.json
   <finish-pass records>/htm03/out_score [--arm32 fallback]`. One command per scorer
   (`src/pcp_recovery.py`, `src/pcp_purity.py`, `src/crossfit_acc.py`) on both arms; the logged A = 32 folders are read
   through path mirrors (their CONFIG lines name the old workstation folder); the logged A = 32 rows must equal the
   pasted values (`a32_rows_equal_logged`). The scorer's tests are a script, `htm03/test_score_htm03.py` (sha256
   25af7e26...), run once on the pinned scorer `score_htm03.py` c8e62ba7... from 2026-10-25T16:03:02Z to 16:03:04Z after
   the CPU gate (1.17 cores), CPU only, 2 threads, nice 19: 11 of 11 checks pass, exit status 0 (output
   `test_score_htm03_r3.txt`, sha256 in the pin block). It replaces the hand-run `test_score_htm03.txt` of 13:16:02Z,
   which tested an earlier version of the scorer (6ac4132a...); the planned batch run (`test_score_htm03_r2b.txt`) never
   ran. The checks:
   - the reproduction check is exact on a scratch copy of the logged run cut to 2 passes, and catches a change in the
     ninth decimal;
   - a missing A = 8 smoke stops the chain;
   - the self-test with A = 32 as both arms reads "not higher", "not higher", "similar", "not held", and the A = 32 rows
     equal the logged ones;
   - a seed without a complete attempt gives "not judged";
   - nothing is written into the study's `src/` (no bytecode).

Estimates: the logged A = 32 runs at `s` 100 took 258.27, 232.718 and 319.639 s (run.json; 810.6 s together). Caps: 10
min for each smoke job, 20 min for each seed (they include the wait for the GPU lock); 80 min in all, 140 with the
fallback. Dev only: the three pair files hold only Xp, P_train, Y_train, P_held and Y_held (zip listing), the driver reads
P_train and P_held; test-split uses are 0 and stay 0.

#### Reading rule (fixed before any run)

Seeds are paired (seed k of A = 8 with seed k of A = 32). Values are those the scorers print (purity and coverage
rounded to 3 decimals by `pcp_purity.py`, recovery in steps of 1/8, dev AUC from the FINAL line).

- **Registered reading (direction only).** Purity rises: A = 8 purity of all groups higher on at least 2 of 3 seeds and
  a higher 3-seed mean. Recovery rises: the same for recovery. An equal value (purity at 3 decimals, recovery in steps of
  1/8) is not higher. AUC stays similar: |3-seed mean AUC (A = 8) - 3-seed mean AUC (A = 32)| <= 0.02, inclusive (the
  A = 32 mean 0.960724625 gives [0.940724625, 0.980724625]); 0.02 is the tolerance of the freeze rule of gates.md, used
  here as the meaning of "similar". The prediction holds when all three parts hold, and fails otherwise. With no effect,
  each sign part holds with probability about 0.42, so a held part is weak evidence; the purity part is expected to hold
  from the slot count alone.
- **Purity beyond the slot-count reference.** A = 8 purity above the 97.5% point of its seed's 8-of-32 draws (0.630 /
  0.647 / 0.603) on at least 2 of 3 seeds.
- **Recovery margin.** A = 8 recovery at least 0.25 (2 of the 8 hidden groups) on at least 2 of 3 seeds; 0.25 is above
  the 97.5% point (0.125) of the reference on every seed.
- Reported next to these, without a rule: coverage and the number of groups (purity counts only groups of at least 2
  distinct words, so the count can fall at A = 8), counted groups, the always-true share, tuned and cross-fitted
  accuracy, and each seed's values.
- No multiplicity correction is applied; each part stands alone, and the conjunction is reported as it is.
- **The comparison arm and the reference with the fallback.** With the logged arm (the reproduction is exact), the
  scorers must give the pasted A = 32 values; if `a32_rows_equal_logged` is false, no part is judged, and the report
  gives the values that differ. With the fallback arm, A = 8 is compared with the new A = 32 runs (the check above
  does not apply: those runs have no logged values), and the slot-count reference and the recovery margin stay the ones
  drawn from the logged A = 32 runs (`slot_reference.py` reads the logged runs only; it is not re-run). The report then
  says that the reference comes from runs the new environment did not reproduce exactly, and gives the new A = 32 runs'
  purity, coverage and recovery next to it.

#### What follows

Either way the result goes into the row of the registered predictions (docs/results.md line 366) as "tested after the
study ended, exploratory, 3 seeds", with the outcome of each part and the reference next to the purity part.

- Only a recovery rise that passes the recovery margin, with a 3-seed mean coverage (all groups) of A = 8 of at least
  0.897 (the lowest of the three A = 32 seeds; inclusive), is read as consistent with
  "too many slots" at `s` 100 and T 128 (T not rescaled; A also changes the size of clause values); and even then the
  `s` 21.1 results above show that 32 options can recover the groups.
- A purity rise alone is read as the slot-count effect of the metric, unless it passes the reference.
- A failed prediction is reported as a measured negative.
- No check, verdict or threshold changes.

#### Failure rules (fixed now)

- The smoke script fails (a job ends with a non-zero exit status): stop; report; no seed runs. One path-only fix is
  allowed before any number exists (a wrong path in a job line, as the relative driver path of draft 2), and is
  reported.
- The A = 32 reproduction is not exact: the fallback in step 3; the report gives the first differing field.
- A seed fails or hits its cap: it is re-run once with the same arguments. The first complete attempt (exit status 0 and
  a FINAL line) counts and is never replaced; every attempt is reported. A seed without a complete attempt after one
  re-run leaves the reading "not judged", reported with the reason.
- A scorer fails: the error is reported, and the reading is "not judged" for the parts that need that scorer.

#### Code since the logged runs, and provenance

- `src/htm_driver.py`: the three logged runs hold the copy dc31407e1057b2ecbf85de469d59d5bd645d8778d605dd7550fdd5c52707df01,
  the file that runs now is 0073b2a4bd61a3b86bdac07f7a3022efeaa401a8fec1e09f65927f910fd0f478. The diff
  (`htm01/code_since/htm_driver_dc31407e.diff`) changes 4 lines, a docstring line and a comment, and the two are equal as
  code once docstrings are removed (`htm01/code_since/table.tsv`). `--A 8` is the only change in the job arguments.
- `src/launch.sh`: 778525b68fcea06aca53a807bfb4c32e7b6ab344d0703dfe813af9220b3d6afb at HEAD, faff0e7a... now (the HTM-B1
  change: a usage check, `src/interpreters.sh`, `job_python` in the run config, the helper scripts started through
  `HTM_TOOL_PY`, the job through `HTM_PY`; diff in `diffs_round2/src_launch.sh.diff`). `src/interpreters.sh` is new. The
  launcher of 2 Oct 2026 (`src/launch.sh` of the original working folder) is not kept in the run folders, so the diff to it cannot be
  shown.
- The logged runs' job interpreter was `tm-to-llm/.venv-htm` (paths in the package warnings of log.txt lines
  4 and 8 of the seed-1 run), which no longer exists. The new runs use `../decision_layer/.venv-htm`. The evidence that
  both hold PyHierarchicalTsetlinMachineCUDA at commit 5b4a223: the driver docstring of the run copies,
  `frozen_settings_hashes.json` line 3, `../tm_language_model/env/htm.lock` (written 2026-10-10 17:36:38 UTC, eight days
  after the runs) for the old one, and `direct_url.json` and METADATA (version 0.3.2) of the new one. The env.json of the
  runs lists 0.3.1 for the logger's `.venv-tm`, which is not the job's interpreter. GPU, driver and nvcc equal the logged
  values (pre-flight: Tesla V100-SXM3-32GB, 570.172.08, release 12.9, V12.9.86). The logged runs used 6 threads, the new
  ones 2 (predicted to make no difference: the training is on the GPU).

#### Times, pins and history

Times are read from the clock (`date -u`) in the command that appended them; hashes are `sha256sum` output pasted by that
command. Drafts 1 to 3 (2026-10-25T11:58:25Z to 13:58:53Z) were reviewed before the freeze; draft 4 is this text. No
rule was changed after an A = 8 number was seen; none existed.

##### Blocks pasted by command, 2026-10-25T16:26:25Z (date -u; htm03/paste_blocks_htm03.sh)

Job lines (grep of the generated scripts gpu_htm03.sh, gpu_htm03_smoke.sh and gpu_htm03_a32_fallback.sh):

```
nice -n 19 src/launch.sh htm03_e2_L2_hier_s1_s100_A8_exploratory 20 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s1.npz" --F 64 --clauses 40 --A 8 --K 1 --T 128 --s 100 --passes 100 --seed 1
nice -n 19 src/launch.sh htm03_e2_L2_hier_s2_s100_A8_exploratory 20 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s2.npz" --F 64 --clauses 40 --A 8 --K 1 --T 128 --s 100 --passes 100 --seed 2
nice -n 19 src/launch.sh htm03_e2_L2_hier_s3_s100_A8_exploratory 20 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s3.npz" --F 64 --clauses 40 --A 8 --K 1 --T 128 --s 100 --passes 100 --seed 3
nice -n 19 src/launch.sh htm03_repro_L2_A32_s100_s1_p2_exploratory 10 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s1.npz" --F 64 --clauses 40 --A 32 --K 1 --T 128 --s 100 --passes 2 --seed 1
nice -n 19 src/launch.sh htm03_smoke_L2_A8_s100_s1_p2_exploratory 10 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s1.npz" --F 64 --clauses 40 --A 8 --K 1 --T 128 --s 100 --passes 2 --seed 1
nice -n 19 src/launch.sh htm03_e2_L2_hier_s1_s100_A32_rerun_exploratory 20 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s1.npz" --F 64 --clauses 40 --A 32 --K 1 --T 128 --s 100 --passes 100 --seed 1
nice -n 19 src/launch.sh htm03_e2_L2_hier_s2_s100_A32_rerun_exploratory 20 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s2.npz" --F 64 --clauses 40 --A 32 --K 1 --T 128 --s 100 --passes 100 --seed 2
nice -n 19 src/launch.sh htm03_e2_L2_hier_s3_s100_A32_rerun_exploratory 20 "$PWD/src/htm_driver.py" --data "$PWD/data/pcp/pairs_L2_s3.npz" --F 64 --clauses 40 --A 32 --K 1 --T 128 --s 100 --passes 100 --seed 3
```

Comparison values (`htm03/paste_a32_values.txt`):

```
A = 32, s 100, 100 passes (the comparison arm; FINAL line of each run; recovery.json and purity.json of 20261002T010559Z_e2_L2_s_compare_posthoc_exploratory):
  seed 1: dev_auc 0.964373125, dev_tuned_acc 0.903, dev_acc 0.883, recovery 0.0, n_counted 38, always_true_share 0.05, purity(all) 0.487, coverage(all) 0.919, n 40
  seed 2: dev_auc 0.967543625, dev_tuned_acc 0.908, dev_acc 0.89775, recovery 0.25, n_counted 39, always_true_share 0.025, purity(all) 0.513, coverage(all) 0.934, n 40
  seed 3: dev_auc 0.950257125, dev_tuned_acc 0.874, dev_acc 0.8495, recovery 0.0, n_counted 39, always_true_share 0.025, purity(all) 0.467, coverage(all) 0.897, n 40
  3-seed means: dev_auc 0.9607246250000001, recovery 0.08333333333333333, purity 0.48900000000000005, coverage 0.9166666666666666
  seed range: dev_auc 0.0173 (0.950257125 to 0.967543625), recovery 0.2500 (0.0 to 0.25), purity 0.0460 (0.467 to 0.513), coverage 0.0370 (0.897 to 0.934)
A = 32, s 21.1, 100 passes (same scoring run):
  seed 1: recovery 0.625, purity(all) 0.68, coverage(all) 0.959
  seed 2: recovery 0.5, purity(all) 0.66, coverage(all) 0.947
  seed 3: recovery 0.75, purity(all) 0.68, coverage(all) 0.966
A = 32, s 21.1, 1,000 passes (recovery and purity runs of the fallback):
  e2_L2_hier_s1_p1000 (recovery.json): recovery 0.875
  e2_L2_hier_s1_p1000 (purity.json): purity(all) 0.92, coverage(all) 1.0
  e2_L2_hier_s2_p1000 (recovery.json): recovery 1.0
  e2_L2_hier_s2_p1000 (purity.json): purity(all) 0.935, coverage(all) 1.0
  e2_L2_hier_s3_p1000 (recovery.json): recovery 0.875
  e2_L2_hier_s3_p1000 (purity.json): purity(all) 0.92, coverage(all) 0.988
```

Slot-count reference (`htm03/slot_reference.log`) and sign-rule chance (`htm03/sign_rule_chance.txt`):

```
# slot_reference.py (sha256 21f91a83f41f44ed47cc6b0d97ba0ba9bd12bd53038a53f330ca5e56fedb441b), 2026-10-25T13:12:42Z
1 step 1 equal: True | 8 of 32: {"purity_all": {"mean": 0.5946, "p2.5": 0.561, "p97.5": 0.63}, "coverage_all": {"mean": 0.4825, "p2.5": 0.453, "p97.5": 0.512}, "n_groups_all": {"mean": 40.0, "p2.5": 40.0, "p97.5": 40.0}, "recovery": {"mean": 0.0141, "p2.5": 0.0, "p97.5": 0.125, "share_above_0": 0.1085}}
2 step 1 equal: True | 8 of 32: {"purity_all": {"mean": 0.6152, "p2.5": 0.583, "p97.5": 0.647}, "coverage_all": {"mean": 0.4935, "p2.5": 0.463, "p97.5": 0.525}, "n_groups_all": {"mean": 40.0, "p2.5": 40.0, "p97.5": 40.0}, "recovery": {"mean": 0.023, "p2.5": 0.0, "p97.5": 0.125, "share_above_0": 0.1765}}
3 step 1 equal: True | 8 of 32: {"purity_all": {"mean": 0.5693, "p2.5": 0.539, "p97.5": 0.603}, "coverage_all": {"mean": 0.4666, "p2.5": 0.438, "p97.5": 0.497}, "n_groups_all": {"mean": 40.0, "p2.5": 40.0, "p97.5": 40.0}, "recovery": {"mean": 0.0053, "p2.5": 0.0, "p97.5": 0.125, "share_above_0": 0.0415}}
all step-1 values equal the logged ones: True
exit status 0
# ended 2026-10-25T13:12:55Z
draws 400000; P(at least 2 of 3 above 0) = 0.4989; P(at least 2 of 3 above 0 and mean above 0) = 0.4185
```

Pinned files, base 1: the study folder studies/hierarchical_tm (sha256sum output):

```
ef0eea12985626b7d899509be11c5bb7eb2db0291e3a9ee2bd6fab328fa54cc8  docs/registrations.md
e9ca0667dd9ec08e24663a30f3248598ac328f7db2ed603f548e5dc4ebd51562  docs/results.md
0073b2a4bd61a3b86bdac07f7a3022efeaa401a8fec1e09f65927f910fd0f478  src/htm_driver.py
e1f0f0fdf7b5f831e0f2f393f273ffb37861d60d6db6ff1cbe493bf8eb1fe6f9  src/pcp_recovery.py
77bcba68faf47f8458c274e66bef3eec1738c8a3639525d855ed4a4969a57d03  src/pcp_purity.py
bd50daa1f0dcce59a8a30d77a8c7efc33d5b9a39bd16b22e116f910ebe8bc6cb  src/pcp_world.py
5f9063f03e5a528d44790bb5b0eeb0cdb996d4e3ed173905587cf920bb419ac8  src/crossfit_acc.py
faff0e7a7dcfad57733d6e792aa7003a460528f155f0f48a0e8345be8624c1f8  src/launch.sh
206f3017527cf5556580e5e3c43d8aa8e679d719139ad5764d735b06f4ee5f4e  src/interpreters.sh
f0964d681a945a5a6b45b65876426a662db23ffb2c4db53f354ee1f9a6ec7bd3  data/pcp/pairs_L2_s1.npz
b8c66cefd840309bcd799b51d234adac4c209a405e8b8a5b372d6d9aec4812f9  data/pcp/pairs_L2_s2.npz
e51fdb07336db043bc765908c8ec91edb34b697ec3cc24519a867351d8f7e127  data/pcp/pairs_L2_s3.npz
0e57a5463770fcf9ec74b4530f62a7d1ccd5bc338a83eed33a084766c76e0ffd  data/pcp/vocab_L2_s1.json
0e57a5463770fcf9ec74b4530f62a7d1ccd5bc338a83eed33a084766c76e0ffd  data/pcp/vocab_L2_s2.json
0e57a5463770fcf9ec74b4530f62a7d1ccd5bc338a83eed33a084766c76e0ffd  data/pcp/vocab_L2_s3.json
72c84beb56d341881938769e846ce1c3fc3521247d7043fd70dd32e69d1e01a5  results/20261002T005202Z_e2_L2_hier_s1_s100/log.txt
43339e8e72131f0fd652593a960ee22f185141f73507aeb091467f73de1fec49  results/20261002T005621Z_e2_L2_hier_s2_s100/log.txt
5cf4fb9915bfee7adff8944d8704d6c0c35ccffbd8d29103c513db9f797f1043  results/20261002T010014Z_e2_L2_hier_s3_s100/log.txt
0698c6ca0a01516342be82a03f802e08c4379f7e82bae93f3342d1205f865b83  results/20261002T010559Z_e2_L2_s_compare_posthoc_exploratory/recovery.json
16b693777e36e660e245eed38596c0c8bb89ad651a03fc0a6c0b7b27932782df  results/20261002T010559Z_e2_L2_s_compare_posthoc_exploratory/purity.json
2bca37b6d8c0277be2084b258e11b0de88e103119c320350893bd2f4275d9ab9  ../tm_language_model/env/env.sh
e9ee13f8df42c9c6d66ae2c5ed1eee35589449ef49216807bbe9887ae281ad8f  ../tm_language_model/env/htm.lock
ba88c5c1cd420babad282e7f03a471a7fefbfe71c5c391f65ba4c0e876eb6c5f  ../tm_language_model/scripts/run_logged.py
33e7c2cd34143fc0d647584bb5dc7d1ea2d743ba6fb4eb98facd43f391dbd00c  ../tm_language_model/scripts/mem_guard.py
1e24a8c8bb9441ae90bd077669ff2e7e620ded244a491307efcb6719f35aaeac  ../tm_language_model/scripts/gpu_run.sh
9587d0bfe6905f6320936debf9fee286e27c33c9f2782b3ad13426bafc535faa  ../decision_layer/.venv-htm/lib/python3.11/site-packages/pyhierarchicaltsetlinmachinecuda-0.3.2.dist-info/direct_url.json
```

Pinned files, base 2: the folder of the finish pass's hierarchical_tm work, <finish-pass records> (sha256sum output; the two pin
lists of the pre-flight are the 6th and 7th lines):

```
6f236b3f123b4b282ceea73fa33dd5561dee389b1b17020fd5a4c9cf8e33f388  htm03/gpu_jobs/gpu_htm03_smoke.sh
5ec694945f4b083c0cfb68a64a7292c0d03dd725830ed610081380152fed4acc  htm03/gpu_jobs/gpu_htm03.sh
c17cf1fd3797aea4833bda66849953904f560017040a7d9a8c3dc5766a9f8c54  htm03/gpu_jobs/gpu_htm03_a32_fallback.sh
54f4436f59562cd34399a2f4aa17a4bc89615d74e4bbb59d2c3d99a06081d17b  htm03/gpu_jobs/jobs_htm03.json
71fa472da777455fe0bec8c0832a1e91a84b3398c758996948224cce08723a15  htm03/paste_blocks_htm03.sh
ec7e775085132763b6299517ef0054e79d93b9100bd39faac58e210f84d601a7  htm03/preflight_pins_htm03.sha256
d0b8d374cb4b81da8fe7be9c4bfd77d89371b4839774c7f61c13b5f5b55176da  htm01/preflight_pins.sha256
c13775489184c83c7a91d5cf65064438db18d67aeee204e06a54297d0d54f8b5  htm01/preflight.py
43c9c71037567bfde6528902bf9ccdd2bf82fce0135ee382a86dd1ea71c24a97  htm01/make_gpu_jobs.py
b4d3cea4252ba1db0508c07a482ffbede79da1dc9c38dfef5e4410247c8462d2  htm01/make_pins.py
56001ee94403f20a727d088f68d29099b74186761c13dd41b976b271fd559ce3  htm01/test_job_lines.py
50b44125390a45f9973aa816a84abb0b0213db19af7592e7a7d4150dd4da5a9e  htm01/run_iii.py
7311953e756e10a61201cd336dbcfc736f562217eefcb64bf8e78dd2206b8cec  htm01/compare_json.py
c8e62ba7ad6c8d106027c9a0a0a4b2a626f5633896c30b5d1b20619465145566  htm03/score_htm03.py
25af7e2687dba5704fab32ad57bce2adebd133a08a0a80e390216e6d1cfca649  htm03/test_score_htm03.py
21f91a83f41f44ed47cc6b0d97ba0ba9bd12bd53038a53f330ca5e56fedb441b  htm03/slot_reference.py
60c00357df8de682b012eec9f6b9c80edbf80b022876ae2614477af46281f181  htm03/slot_reference.json
8b2c1398a20209cddc83d3093a9ac8e35278092ed12b7d4201be4b1912a0c806  htm03/slot_reference.log
acc994f46c7762ebf17833f8c5c143ca847f70274faa68830f97deb7ee8109ae  htm03/sign_rule_chance.py
1a7a56024a164f2bb41f809a9a8fe699a9de31edeb5bf94a534fab076738e208  htm03/sign_rule_chance.txt
c88ca1475db335987f18dfda91d985c3ae3e7714ef093b4b1256683de4818229  htm03/paste_a32_values.py
2cd1767ffd75add97a17b36f9bf0b1ffa40d5de65408c12a9f61d9ab712878b7  htm03/paste_a32_values.txt
42d131bb6cd4bbd1a3156a14a6da5ace57ffa17cc0325678f5b6a6d85114ed2f  test_job_lines_after.txt
f6e4fdde4649b5f278bed93f06152a88cb94d45a5745e9776e12dbe5d91c060f  test_job_lines_before.txt
cc7804ebb24ef791cc5c440819db856e7e8f17089513d029c62b5a5ef689c41f  test_score_htm03.txt
8118ce3042916ab6daa91e4020c5034173444a09700f1ac1be06c188641f5486  test_score_htm03_r3.txt
2e912d4963795e2082171ab2124c8af80a34664cb9bf6441c0ebfa174f470dfd  test_preflight_r3_neg_env.txt
12619334a00448b4029a46b66e450f9c8bca87f049cce927c7ec71f50f1299f6  test_preflight_r3_neg_hash.txt
dd6238a57c7ed56c2bdd4c69081a1cff6de0df71eb7d4c4af05205c533a48b4e  test_preflight_r3.txt
8e7e8548ff70ae6260c0b9337f591965202d4ed7c714faf4408267b49645f3e1  test_preflight.txt
```

`test_score_htm03.txt` tested the earlier scorer 6ac4132a... and is kept as a record only; `test_score_htm03_r3.txt` is
the test of the pinned scorer.

##### How steps 1 to 4 are queued (added before the freeze, 2026-10-25T16:31:45Z, date -u)

Steps 1 to 4 run as one job of the serial GPU queue, `bash htm03/run_htm03_chain.sh`. The script first checks the
two pin lists of the pre-flight against the hashes above (`htm03/list_pins_at_freeze.sha256`, written at the freeze from them)
and that the recorded decision exists (`htm03/DECISION_TO_RUN.txt`). Then it runs the smoke script and the check, runs the fallback
when the reproduction is not exact, runs the three A = 8 seeds, and scores with the arm the check chose. A failing job stops it.
Tests: `test_chains.txt` (12 of 12, on a stub tree). sha256sum output, base ws/htm:

```
f2120eaf42f1637c18e426c4137fe3538d7f532ba9568b9d4128f19a23683d10  htm03/run_htm03_chain.sh
e1eb33d1f1f3db8753f9e4c7d6e67426ba0eb927801cab8d9f73d1eeeaf39528  test_chains.sh
e1cca2813d11a1013aedd70506d7384b38e79acccbbdb8f86ff8e42d48efdb7b  test_chains.txt
166ae1d6117e55648072bf20944e00ed09ed6d45a879008d70cc0026577ef01e  freeze_htm.py
19d03ca3b8f4f2e29c0d7b27cdb5b0120d818ab49906c47859b71a9a53f416fa  test_freeze_htm.txt
```
