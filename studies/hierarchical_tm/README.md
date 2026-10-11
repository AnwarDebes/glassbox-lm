# hierarchical_tm: a test of the hierarchical Tsetlin Machine of the project proposal

## Question

Does the hierarchical Tsetlin Machine procedure of the supervisor's project proposal (unpublished) learn what the proposal expects? The project title of the proposal is "Logical self-supervised learning of language with hierarchical Tsetlin machines". In this procedure a Tsetlin Machine whose clauses are conjunctions of OR-groups of words is trained to say whether two paragraphs belong to the same document, and the OR-groups it has learned are then read as word groups.

The study ran the procedure with the package PyHierarchicalTsetlinMachineCUDA (public repository `cair/PyHierarchicalTsetlinMachineCUDA`), starting from Granmo's MultiConceptDemo script, on artificial pair data with hidden word groups, on two sentence families built from slides 6 and 10 of the project proposal, on news topics (AG News) and on Wikipedia paragraph pairs. Each expectation was turned into a check with a pass rule. The pass rules were frozen in [gates.md](gates.md) on 1 Oct 2026 at 11:15 Norwegian time (09:15 UTC), before any new run. Rules, settings and predictions that came later are collected in [docs/registrations.md](docs/registrations.md) with the times at which they were written down. Most were written down before the runs they govern; the results page says where an entry was not kept, records a deviation or follows a run. Choices that were made after results were known are labelled post hoc on the results page.

## Status

Finished. The study ran from 1 to 3 Oct 2026 and is frozen: the [run index](../../docs/run-index.md) holds no run of this study after 3 Oct 2026, and the pass rules and registrations are published with their registered content (each carries its own publication note). The code is published as it stood at the end of the study, apart from edits for publication: the working copies held references to unpublished working notes and plans, paths of the workstation and remarks on day-to-day work, and these were removed or reworded. The sections below name the files concerned.

## What the study found

The study ended mostly negative. The model learns relations between Wikipedia paragraphs, and the word groups it forms there are topically coherent. Its single final test is nevertheless below word overlap on both Wikipedia tasks and below the best simple baseline on all three data sets. Of the three artificial tests only the easiest met its check. No learned rule captured either of the two sentence families of the project proposal, in the pair version, in the direct-label fallback or in the arms that were added after the results were known. On news topics the hierarchical model stayed further below a flat Tsetlin Machine than the check allows, while its word groups met the coherence part of the check. The synonym test was met on one arm with a small effect; for that arm the wording is "slightly more synonym-like than chance" and nothing stronger.

The table gives one status per entry of gates.md (eight checks and the final test). The statuses and qualifiers are those of [docs/results.md](docs/results.md), which gives every number with its run id and the file it was read from.

| id | status | qualifier |
|---|---|---|
| Check 1a (G0): the package demo is reproduced | met | the demo ran with 1 repeat per seed, not its default of 10 |
| Check 1b (data): no shortcuts | met | the next-paragraph data passed after a rebuild; the main synonym data is borderline on the skewed-word count and passes only by the decision of 1 Oct 2026, 20:10 CEST (addendum of gates.md) |
| Check 2a (G1): artificial tests L1 to L3 | not met | L1 met; L2 not met, borderline, under an accuracy that was chosen post hoc; L3 not met, at an `s` fixed by a rule that was written after the first `s` 400 result |
| Check 2b (G3): slide 10 | not met | model shaped to the known template (the number of OR-groups and of options per group was set to it); not met in the pair version and in the direct-label fallback; the fallback is judged under one of two readings of its time cap, and under the other reading it would be "not judged within the cap" |
| Check 2c (news): AG News word groups | not met | accuracy part not met, NPMI part met; the word groups are shown as examples only |
| Check 3 (G2): Wikipedia learns (A) or not (B) | met | branch A; `s` and the number of passes were chosen on the dev data of seed 1 |
| Check 4a (G4): Wikipedia word groups | met | scored before the decision between branch A and branch B (2 Oct 2026, 21:27 CEST) that it was to follow; topical coherence, not synonymy |
| Check 4b (synonyms): the synonym test | met | on the arm at `s` = 100 only, with a small effect; not met on the arms at `s` = 400 and `s` = 200; the arm at `s` = 200 was allowed after the results of the other two arms were known |
| The final test | not met | status of the registered comparison "beats X": the outcome is "below" word overlap on both Wikipedia tasks and "below" the baseline that was best on dev on all three data sets |

The slide-6 paraphrase test is not a check of gates.md. It was registered on 2 Oct 2026 as a descriptive test, its model was shaped to the known template as for slide 10, and its claim was not met. The robustness arms R1, R2 and S6-R1 were registered after the results they relate to were known; the results page labels them post hoc, and they change no verdict. The results page also reports four runs that have no registration and no verdict, and it collects the disclosures that belong to single results.

## What a clone allows

From a clone a reader can read the pass rules, the registrations, the code and the results page, and can place every cited run id in the run index. A reader cannot rerun the study or recompute a number from a clone alone. The run folders (logs, metrics, saved models), the datasets, the sentence files of the slide-6 and slide-10 tests, the job lists of the queues and the virtualenvs are not distributed, and every launcher of this study starts through `studies/tm_language_model`.

## Dependence on studies/tm_language_model

This study is not self-contained. Its jobs start through the run logger, the environment script, the GPU lock and the virtualenvs of [studies/tm_language_model](../tm_language_model/README.md), and several scripts read data that was built there. The code finds that study as the sibling folder `../tm_language_model`, derived from the location of each file: the shell scripts set `STUDY` from their own path and change to `$STUDY/../tm_language_model`, and the Python files build the same path from `__file__`. The two study folders therefore have to stay next to each other under `studies/`.

| what is used | where it is in `../tm_language_model` | file of this study that makes the link |
|---|---|---|
| run logger (creates the run folder and its records) | `scripts/run_logged.py` | `src/launch.sh`, `queue/run_queue_mps.sh`, `queue/gpu_share_test.sh`, `src/feas_slide11/launch.sh` |
| memory guard | `scripts/mem_guard.py` | the same four scripts |
| GPU lock, modes `exclusive` and `shared` | `scripts/gpu_run.sh` | the same four scripts |
| environment script (CUDA paths, thread caps, hash seed) | `env/env.sh` | the same four scripts, which source it |
| virtualenv of the GPU jobs | `.venv-htm`, built by `env/htm_install.sh` and frozen in `env/htm.lock` | the same four scripts; the shell scripts in `src/recovered/` call `.venv-htm/bin/python` by relative path |
| virtualenv of the CPU scripts | `.venv-tm` | `queue/gpu_share_test.sh` (its analysis step), `src/test_wiki_groups.py` and `src/test_e5a.py` (the command in their docstrings), `src/build_agnews.py` and `src/lr_agnews.py` (named in their docstrings) |
| package `tmlm` (`tmlm.common.runlog.validate_metrics`) | `src/` | `src/test_wiki_groups.py`, which puts that folder on the import path |
| WikiText-103 raw text | `data/wikitext103/raw` | `src/build_pairs.py`, `src/feas_slide11/prep_data.py` |
| WordNet 3.0 | `data/concept_eval/nltk_data` | `src/wiki_groups.py` |
| AG News train file (Hugging Face cache) | `data/hf_cache` | `src/build_agnews.py` |

`queue/run_queue.sh` reaches the same tools through `src/launch.sh`. All launchers except one pass `--results-dir` so that the run folders are written to `results/` of this study; `src/feas_slide11/launch.sh` passes none, so the run folders of the probe runs are in `results/` of the tm_language_model study.

Because the runs were logged through that study's run logger, their run records carry the hashes of that study's pre-registration and source tree and of none of this study's files. The section on code revisions in docs/results.md says what follows from this.

## Layout

```
hierarchical_tm/
  README.md
  gates.md                      pass rules of the checks, frozen on 1 Oct 2026, with dated addenda
  frozen_settings_hashes.json   SHA-256 pins of the final test: 21 scripts and 17 data files
  src/                          drivers, data builders, checks and scorers
    feas_slide11/               the eight probe scripts of 2026-10-01
    recovered/                  thirteen analysis scripts recovered from run folders
  queue/                        the two GPU queues and the GPU-sharing test
  docs/                         registrations, results, two data cards
```

The folders `results/` (run folders) and `data/` (datasets) belong to the study and are local: they are not part of the repository.

### src/

- Training drivers: `src/htm_driver.py` (pair tasks, hierarchical and flat model) and `src/text_driver.py` (one text to one class: the news topics and the direct-label fallback of slide 10). `src/launch.sh` starts one GPU job.
- Data: `src/build_pairs.py` and `src/build_ns_all.py` (Wikipedia pairs), `src/pairdata.py` (loader), `src/build_agnews.py` (news topics), `src/pcp_world.py` and `src/export_pcp.py` (the artificial tests L1 to L3), `src/slide10.py` (generator of the slide-10 sentence family and scorer of Check 2b).
- Checks of the data and of the package: `src/data_checks.py` and `src/check_ns_all.py` (Check 1b and the baselines), `src/run_demo_seeded.py` (Check 1a).
- Registered selection rules: `src/apply_rules.py` (headline `s` and number of passes), `src/e5a_choice.py` (the `s` of the news-topic test).
- Scoring: `src/score_saved.py` (re-scores a saved model; the one final test), `src/final_test_report.py` (main table of the final test and its dry run on dev), `src/crossfit_acc.py` and `src/crossfit_splits.py` (the accuracies of Check 2a), `src/pcp_recovery.py` and `src/pcp_purity.py` (group recovery and purity), `src/npmi_groups.py` and `src/wiki_groups.py` (word-group scores of Checks 2c, 4a and 4b), `src/e5a_check2c.py` (accuracy part of Check 2c), `src/lr_agnews.py` (logistic-regression baseline), `src/compare_runs.py` (two runs that must train identically).
- CPU tests: `src/test_e5a.py` and `src/test_wiki_groups.py`.

### src/feas_slide11/

The eight scripts of the feasibility probe of 2026-10-01, which came before this study: `train_tm.py`, `prep_data.py`, `baselines.py`, `posthoc.py`, `launch.sh`, `queue2.sh`, `queue3.sh` and `queue4.sh`. The 27 runs of these scripts (run ids that contain `feas_slide11`) used pair data with a sampling bias. They are flagged as biased exploratory in the run index, and no number from them is used as a result. The pair data of this study was built anew by `src/build_pairs.py`.

The launcher copies `train_tm.py`, `prep_data.py` and `launch.sh` into the run folder after the run has ended, and none of the copies of the probe scripts is listed in the manifest of its run. Before the edits for publication, the retained scripts compared with the copies in the 27 run folders as follows:

| script | distinct versions in the run folders | published version |
|---|---|---|
| `train_tm.py` | 5 | the version of the last probe run, `20261001T080935Z_feas_slide11_posthoc_exploratory` |
| `prep_data.py`, `baselines.py`, `posthoc.py`, `launch.sh` | 1 each | that version |
| `queue2.sh`, `queue3.sh`, `queue4.sh` | none (no run folder holds a copy) | the retained copy |

The published files differ from the retained scripts by the edits for publication (path constants and wording). The scripts read `pairs_F<F>.npz` from their own folder and the launcher writes a config and a log file next to itself; none of these files is distributed.

### src/recovered/

Thirteen analysis scripts of this study survived only as copies inside run folders (most of them copy themselves into the folder of the run they produce). They were recovered from there, one copy each, and then edited for publication (path constants and wording), so the published files are not byte-identical to the copies in the run folders. Each recovered copy is listed in the manifest of its run.

| script | recovered from run folder | distinct versions (run folders that hold a copy) |
|---|---|---|
| `compare_data.py` | `20261001T094115Z_data_build_v3_repro_exploratory` | 1 (1) |
| `slide10_lr_null.py` | `20261002T040216Z_slide10_lr_null_exploratory` | 1 (1) |
| `devcheck.sh` | `20261002T134938Z_final_test_report_devcheck_exploratory` | 2 (4) |
| `codepath.sh` | `20261002T135204Z_final_test_report_codepath_test_exploratory` | 1 (2) |
| `devcheck5.sh` | `20261002T164442Z_final_test_report_devcheck_5seeds_exploratory` | 1 (1) |
| `e3check.sh` | `20261002T202633Z_e3_screen_check2b_dev` | 1 (1) |
| `e3_gate.py` | `20261002T213043Z_e3_check2b_eval_hier_s150` | 1 (3) |
| `e3_overlap_diag.py` | `20261002T213109Z_e3_pair_overlap_diagnostic_posthoc_exploratory` | 1 (1) |
| `e3fb_choice.py` | `20261002T225455Z_e3fb_s_choice` | 1 (1) |
| `e3fb_report.py` | `20261003T001719Z_e3fb_check2b_eval_s21` | 1 (1), and an identical copy in a sub-folder of `20261003T015449Z_e3fb_flat_C768_check2b_eval_and_probe_v2_exploratory` |
| `e3fb_deletion_probe.py` | `20261003T001721Z_e3fb_deletion_probe_posthoc_exploratory` | 2: the second one is in a sub-folder of `20261003T015449Z_e3fb_flat_C768_check2b_eval_and_probe_v2_exploratory` |
| `diag_report.py` | `20261003T004307Z_e3diag_R1_report_posthoc_exploratory` | 1 (3) |
| `flat_both.sh` | `20261003T015449Z_e3fb_flat_C768_check2b_eval_and_probe_v2_exploratory` | 2 (2) |

Where two versions exist, the published one is the version of the run folder named in the table. For `devcheck.sh` and `flat_both.sh` that is the last run that used the script. For `e3fb_deletion_probe.py` it is the version of the run that docs/results.md cites for the deletion probe. The registrations state that this script was changed afterwards to read its result by a two-part rule; the published copy does not contain that change.

`diag_report.py` has two branches: its slide-6 branch imports the module `slide6`, which is not distributed, so that branch cannot be started from the repository; its slide-10 branch imports only the published `src/slide10.py`. The five shell scripts name run folders of 1 to 3 Oct 2026 by id, call `.venv-htm/bin/python` by relative path and write into `$TMLM_RUN_DIR`, so they expect to be started through the run logger from the tm_language_model folder. Two scripts that only call `src/slide6.py` are not distributed, and the script of the run `20261003T001723Z_e3fb_gpu_time` was not retained anywhere.

### queue/

`queue/run_queue.sh` is the serial queue and `queue/run_queue_mps.sh` the queue with two slots under NVIDIA MPS. `queue/gpu_share_test.sh` is the GPU-sharing test with the check of the evaluation schedule, and `queue/analyse_share_test.py` evaluates it. The job lists, the logs and the state files that the queues write next to themselves (`queue/jobs.txt`, `queue/queue.log`, `queue/done.txt`, `queue/started.txt`) are not distributed.

### docs/

[docs/registrations.md](docs/registrations.md), [docs/results.md](docs/results.md) and the two data cards named below.

## Inputs that are not distributed

Sentences of the project proposal are not part of this repository. Files that carry such sentences are held back, and the code that expects them names these paths:

- `slide10_templates.txt` in the study folder: the input of `src/slide10.py build`. Its lines 1 and 2 are the two sentences of slide 10 of the project proposal, and the lines after them are the near-miss sentences. Without the file the build stops with a message.
- `src/tests/slide10_fixture.txt`: the test copy that `src/slide10.py` names for a dry run. The unit tests of the generator, `src/tests/test_slide10.py`, are held back with it.
- `deck.txt` in the study folder: the text of the project proposal. When the file exists, `src/slide10.py` compares lines 1 and 2 of the templates file with it; when it is missing, the comparison is skipped.
- `src/slide6.py`: generator and scorer of the slide-6 paraphrase test, which holds sentences of slide 6 as string constants. `src/recovered/diag_report.py` imports it as the module `slide6`. Its unit tests, `src/tests/test_slide6.py`, and the data card of the slide-6 data are held back with it.
- `data/slide10/` and `data/slide6/`: the built sentence data (docs/registrations.md names `data/slide6/pairs_slide6.npz`; `src/slide10.py` writes `pairs_slide10.npz`, `vocab_slide10.json`, `slide10_meta.json` and the folder `text/`).

The other inputs that a run needs and that are not distributed:

- The datasets in `data/`: the Wikipedia pair versions `data/sa`, `data/np`, `data/ns` and `data/ns_all` with the vocabulary, split and paragraph files, the news-topic data in `data/agnews`, and the generated files of the artificial tests in `data/pcp`. The raw data they are built from sits in `../tm_language_model/data` (table above). [docs/data.md](../../docs/data.md) at the repository root says how each dataset can be obtained.
- The run folders in `results/`: logs, metrics, rule printouts, include bits and saved models. Scoring scripts such as `src/score_saved.py`, `src/wiki_groups.py`, `src/final_test_report.py` and everything in `src/recovered/` take run folders as input.
- The demo script that Check 1a runs (Granmo's MultiConceptDemo). `src/run_demo_seeded.py` takes its path through `--demo`; the docstring of `src/text_driver.py` refers to a copy under `imported/`, a folder that is not distributed.
- The job lists of the queues (see the section on `queue/`).

## Environment

The GPU jobs ran in the virtualenv `.venv-htm` of the tm_language_model study. Its install script is `../tm_language_model/env/htm_install.sh` and its freeze is `../tm_language_model/env/htm.lock`. The versions below are taken from these two files only.

| component | version |
|---|---|
| Python | 3.11 (`uv venv --python 3.11` in the install script) |
| PyHierarchicalTsetlinMachineCUDA | from `github.com/cair/PyHierarchicalTsetlinMachineCUDA` at commit `5b4a223075c1e621577d508bc7c3c061230e29d5` |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| pycuda | 2025.1.3, built from source against that numpy |
| scikit-image, networkx (pulled in by the package) | 0.26.0, 3.6.1 |

The lock file lists the remaining packages. It also holds the tm_language_model package itself as an editable install. The install script expects a GPU of CUDA compute capability 7.0 and ends with an error otherwise. The CPU scripts (data builds, data checks, group scorers, tests) used the virtualenv `.venv-tm`, whose versions are in `../tm_language_model/env/tm.lock`.

The environment record that the run logger writes into a run folder (`env.json`) describes the interpreter of the logger. For the GPU jobs of this study that was the interpreter of `.venv-tm`, while the job itself ran under `.venv-htm/bin/python`. The environment record of a GPU job therefore names the logging interpreter and not the environment of the job, and the package versions of the GPU jobs are those of `env/htm.lock`.

The launchers start `scripts/mem_guard.py` and `scripts/run_logged.py` of the tm_language_model study directly. In the published files both scripts begin with `#!/usr/bin/env python3`, so a `python3` with the environment of that study has to be on the `PATH` when a launcher is started.

## How the jobs were started

Paths are relative to this folder. A single GPU job was started through `src/launch.sh` (usage line from its header):

```
src/launch.sh NAME MINUTES SCRIPT.py [script args...]
```

`NAME` is the run name (names of exploratory runs end in `_exploratory`) and `MINUTES` the wall-time cap, which includes the wait for the GPU lock. The script changes to `../tm_language_model`, sources `env/env.sh` and runs the chain of study rule 4 with a memory limit of 12 GiB. The run folder is written to `results/` of this study, and the job finds it in `$TMLM_RUN_DIR`.

Queued jobs are lines of `queue/jobs.txt`. For the serial queue `queue/run_queue.sh` a line is `NAME|MINUTES|htm_driver.py args...`, and each job runs through `src/launch.sh`. `queue/gpu_share_test.sh` refuses to start while a process started as `bash ./run_queue.sh` is running, so its guard covers the serial queue only when that queue is started in this form from inside `queue/`. For the MPS queue `queue/run_queue_mps.sh` a line is `NAME|MINUTES|ARGS`, where `ARGS` are arguments of `src/htm_driver.py` or, written as `@DRIVER.py args`, of another driver in `src/`; every driver has to accept `--max-minutes`. Whenever a slot is free, the first line that has not started yet starts. Creating the file `queue/STOP` pauses either queue: running jobs finish and nothing new starts.

`queue/run_queue_mps.sh` and `queue/gpu_share_test.sh` start an NVIDIA MPS control daemon (`nvidia-cuda-mps-control -d`) and stop it when they end; the sharing test also calls `nvidia-smi`. The queue scripts encode the workstation of the recorded runs: one GPU, two MPS slots and a memory limit of 12 GiB per job.

The CPU tests are started from this folder (commands from their docstrings):

```
../tm_language_model/.venv-tm/bin/python src/test_e5a.py
../tm_language_model/.venv-tm/bin/python src/test_wiki_groups.py
```

The one final test consisted of two steps (commands from the docstrings of the two scripts and from the frozen settings of docs/registrations.md):

```
python score_saved.py RUN_DIR MODEL_FILE test --final-test
final_test_report.py --split test --final-test --best-from DEVRUN/dev_report.json --group ... --shuffled ...
```

Both scripts refuse the test split without `--final-test`.

## Study rules

The code cites two rules of the study by number; the numbers are those of the rule list that the study used.

### Study rule 4: GPU jobs

One GPU job runs at a time, or at most two at once under NVIDIA MPS. A single job is wrapped in this order: `timeout`, then `scripts/mem_guard.py --scope group --limit-gib N`, then `scripts/run_logged.py`, then `scripts/gpu_run.sh exclusive`, then the Python interpreter of `.venv-htm`. The scripts are those of `studies/tm_language_model`. `src/launch.sh` builds this chain, and the serial queue `queue/run_queue.sh` runs a job list through it.

Under MPS the chain is the same with three differences. The lock mode is `gpu_run.sh shared`, which lets the two jobs overlap. The MPS environment variables are set. A job stops only through its own `--max-minutes` option: the outer `timeout` is a distant last resort, and a running MPS client is never killed, because on the Volta GPU of the recorded runs one killed client can take down the others. `queue/run_queue_mps.sh` builds this chain.

The rule had its one-job form until 2 Oct 2026. The condition for sharing had been registered beforehand (docs/registrations.md, entry of 1 Oct 2026, about 20:40 CEST: the same results alone and in parallel, and a summed throughput of at least 1.3 times the single job), and the rule was changed on 2 Oct 2026 after the sharing test `queue/gpu_share_test.sh`. Two jobs under MPS met both conditions, and sharing without MPS was slower than one job at a time; both statements rest on a queue report outside the run folders, and the measured ratios are not given in this repository. Since the change the following holds:

- at most two GPU jobs run at once, and only under MPS through `queue/run_queue_mps.sh`;
- a run whose timing is itself a reported result runs alone (`gpu_run.sh exclusive`, no MPS);
- speed numbers from shared runs are not reported and not compared;
- three jobs at once would need a measured three-job MPS test that beats two jobs by a clear margin.

The rule is cited by `src/launch.sh`, by `queue/run_queue_mps.sh` and by the entry of docs/registrations.md named above.

### Study rule 7: empty options

An option of an OR-group that includes no word is true, in training and at inference, so an OR-group that has an empty option is always true. Such a group is printed as `ALWAYS-TRUE` and is never counted as a word group. gates.md fixes the same in its definitions "Always-true OR-group" and "Counted OR-groups".

The published drivers and scorers follow the rule without citing its number (`src/htm_driver.py`, `src/text_driver.py`, `src/slide10.py`, `src/wiki_groups.py`). Only `src/slide6.py`, which is not distributed, cites it as study rule 7.

## Registrations, frozen records and data cards

- [gates.md](gates.md): the pass rules of the eight checks and of the final test, frozen on 1 Oct 2026, with dated addenda. It carries its own publication note.
- [docs/registrations.md](docs/registrations.md): the registrations of 1 to 3 Oct 2026 in time order, among them the frozen settings of the final test, the slide-6 test, the direct-label fallback of Check 2b and the post hoc robustness arms.
- [frozen_settings_hashes.json](frozen_settings_hashes.json): the hash record of the final test, described in the next section.
- [docs/data_card_wikipedia_pairs.md](docs/data_card_wikipedia_pairs.md): the pair versions `sa`, `np` and `ns`. It does not cover `data/ns_all`, which is described in the docstring of `src/build_ns_all.py`.
- [docs/data_card_agnews.md](docs/data_card_agnews.md): the news-topic data.

The two data cards were written next to the data, the first in `data/` and the second in `data/agnews/`. The files that they name without a folder (`splits.json`, `vocab_F1024.json`, `vocab.json`, `build_stats.json` and the `.npz` files, among others) sat in those two folders and are not distributed. No card is published for the artificial tests or for the two sentence families; the docstrings of `src/pcp_world.py` and `src/slide10.py` describe that data.

Times in gates.md and in docs/registrations.md are Norwegian time (CEST, UTC+2) as registered. Run ids are UTC.

The frozen settings of the final test were drafted on 2 Oct 2026 at 15:19 CEST. Decisions of the same day: the settings are binding, and the final test is run. docs/registrations.md times these decisions at 21:27 CEST (19:27 UTC) and the addendum of gates.md at 21:28 (19:28 UTC). The first test re-score has the run id `20261002T192818Z_e4_head_sa_s1_testscore`, which is 19:28:18 UTC.

## The hash record frozen_settings_hashes.json

The record pins the SHA-256 of 21 scripts (key `scripts_sha256`) and of 17 data files (key `data_sha256`), with paths relative to this folder, and names the commit of the package.

Publication note: the record was edited for publication on 2026-10-10 and 2026-10-11, and no key was added.

- The text of the value `prepared` was edited. It keeps that the record was prepared on 2 Oct 2026 at 15:19 Norwegian time as a draft of the frozen settings, and it refers to gates.md for the decision that made the settings binding (2 Oct 2026, 21:28 Norwegian time there).
- 14 of the 21 values under `scripts_sha256` were replaced by the hashes of the published files, because these 14 scripts were edited for publication (wording of comments, docstrings and messages, and path constants). Six scripts are byte-identical to the registered files and keep their pins.
- The pin of `src/text_driver.py` is published as registered and does not match the published file. It disagreed with the file before publication already: the script was changed on 2 Oct 2026 at 22:28 CEST, after the freeze, to add a printing option (disclosed in docs/registrations.md, entry of 22:28). The pin refers to the version that the news-topic runs used.
- The 17 values under `data_sha256` are unchanged. They refer to data files that are not distributed.

The [hash table](../../docs/hash-table.md) at the repository root gives the SHA-256 as registered and as published for each pinned script. The hash check that the final test ran on 2 Oct 2026 refers to the files as they stood then.

## Claims that are not made

gates.md refers to this list. On the strength of this study the repository does not claim:

- that the model beats word overlap. Its single final test is below word overlap.
- that the model learns synonyms from Wikipedia. A task that asks whether two paragraphs belong together teaches topics. For the one arm of Check 4b that met the check, the wording is "slightly more synonym-like than chance" and nothing stronger; the two other arms did not meet the check.
- that the model understands negation.
- that the results show anything about large language models, scale, energy or the state of the art.
- that the package has a bug. Where the code works around a behaviour of the package (an empty option counts as true; the `load()` of the package does not restore four options, as the docstring of `src/score_saved.py` describes), whether that behaviour is intended is a question about the design of the package.
- that anything here is a "first" or "unique". gates.md fixes the wording for novelty.

gates.md also sends to this list any number that does not reproduce within 0.02. The reproduction was not run, so this rule has put no number on the list: the run index holds no run of this study after 3 Oct 2026.
