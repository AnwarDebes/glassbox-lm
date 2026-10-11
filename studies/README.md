# Studies

Each study of the programme is one folder in `studies/`. A study keeps its own code, registration, results page and data cards; what spans the studies is in the root [docs](../docs/README.md) folder, and the entry point of the repository is the [root README](../README.md). File paths on this page are written from the repository root.

## The studies

| Study | Period | Status | Question | Headline result |
|---|---|---|---|---|
| [tm_language_model](tm_language_model/README.md) | 19 Sep to 1 Oct 2026 | archived | Can a token-level Tsetlin Machine language model, trained with Tsetlin feedback on next-token prediction, reach the accuracy of an n-gram baseline at useful speed, and what do its clauses represent? | Of 91 registered hypotheses, 12 are met, 11 are not met, 7 are not judged and 61 were not run. The reproducibility hypotheses were met at the scale tested. The main accuracy hypothesis H1.1 is not met on seed 1, and the same run, `20260923T042350Z_CP_anchor`, triggered the kill criterion KB7. A count-vote read-out met H8.1, which compares it with the signed-vote read-out of the same clause bank and not with the n-gram baseline. In the self-supervised family, 7 of the 8 hypotheses that were run are not met. [Results page](tm_language_model/docs/results.md). |
| [hierarchical_tm](hierarchical_tm/README.md) | 1 to 3 Oct 2026 | finished | Does the hierarchical Tsetlin Machine procedure of the supervisor's project proposal (unpublished) learn what the proposal expects? | Mostly negative. Of the nine entries of the pass rules, Checks 1a, 1b, 3 and 4a are met, Check 4b is met on one of its three arms with a small effect, and Checks 2a, 2b and 2c and the final test are not met. No learned rule captured either of the two sentence families of the proposal (Check 2b and the slide-6 paraphrase test). The model learns relations between Wikipedia paragraphs, and its single final test, `20261002T193513Z_final_test_main_table`, is below word overlap on both Wikipedia tasks and below the best simple baseline on all three data sets. [Results page](hierarchical_tm/docs/results.md). |
| [clause_lm](clause_lm/README.md) | from 2 Oct 2026 | running, as preparation | Do clauses with learned OR-groups of words add predictive value over count statistics, first on word pairs and then inside a language model, and does the same kind of clause help a readable router? | N-gram baselines, pair data, statistical pair baselines and one clause evaluator ran once each on Penn Treebank; no Tsetlin Machine has been trained on Penn Treebank. Gate 0 of the one registered step, the reproduction of the Clause-Driven-LLM router on CLINC150, was not met in its one completed run, `20261002T195255Z_m5_gate0_s8_seed42_sorted`. The cause is not confirmed, and the registered question has no verdict. [Results page](clause_lm/docs/results.md). |

The status of a study is one of four words. Planned: the study has a folder and a question, and no run has been made. Running: runs are being made or prepared. Finished: the study has ended and its records are frozen. Archived: the study was stopped and is not continued; its row says what it found. The statuses of the registered ids of each study and its headline numbers are collected in [docs/results-overview.md](../docs/results-overview.md).

## Dependencies between the studies

tm_language_model needs neither of the other studies. hierarchical_tm and clause_lm are not self-contained. Both start jobs through scripts and virtual environments of tm_language_model and read data that was built there, and clause_lm also waits on queue files of hierarchical_tm. The code finds the other study as a sibling folder, by a relative path that each file derives from its own location, so the three study folders have to stay next to each other in `studies/`.

### hierarchical_tm uses tm_language_model

Four launchers make most of the links: `studies/hierarchical_tm/src/launch.sh`, `studies/hierarchical_tm/queue/run_queue_mps.sh`, `studies/hierarchical_tm/queue/gpu_share_test.sh` and `studies/hierarchical_tm/src/feas_slide11/launch.sh`. `studies/hierarchical_tm/queue/run_queue.sh` reaches the same tools through the first of them.

| What is used | Where it is | File of hierarchical_tm that makes the link |
|---|---|---|
| Run logger, which creates the run folder and its records | `studies/tm_language_model/scripts/run_logged.py` | the four launchers |
| Memory guard | `studies/tm_language_model/scripts/mem_guard.py` | the four launchers |
| GPU lock, modes `exclusive` and `shared` | `studies/tm_language_model/scripts/gpu_run.sh` | the four launchers |
| Environment script (CUDA paths, thread caps, hash seed) | `studies/tm_language_model/env/env.sh` | the four launchers, which source it |
| Virtual environment of the GPU jobs, `.venv-htm` | built inside the tm_language_model folder by `studies/tm_language_model/env/htm_install.sh`, frozen in `studies/tm_language_model/env/htm.lock` | the four launchers; the shell scripts in `studies/hierarchical_tm/src/recovered/` call its interpreter by relative path |
| Virtual environment of the CPU scripts, `.venv-tm` | built inside the tm_language_model folder by `studies/tm_language_model/env/bootstrap.sh`, frozen in `studies/tm_language_model/env/tm.lock` | `studies/hierarchical_tm/queue/gpu_share_test.sh` (its analysis step); `studies/hierarchical_tm/src/test_wiki_groups.py` and `studies/hierarchical_tm/src/test_e5a.py` (the command in their docstrings); `studies/hierarchical_tm/src/build_agnews.py` and `studies/hierarchical_tm/src/lr_agnews.py` (named in their docstrings) |
| Package `tmlm` | `studies/tm_language_model/src/` | `studies/hierarchical_tm/src/test_wiki_groups.py`, which puts that folder on the import path |
| WikiText-103 raw text | `studies/tm_language_model/data/wikitext103/raw`, not distributed | `studies/hierarchical_tm/src/build_pairs.py`, `studies/hierarchical_tm/src/feas_slide11/prep_data.py` |
| WordNet 3.0 | `studies/tm_language_model/data/concept_eval/nltk_data`, not distributed | `studies/hierarchical_tm/src/wiki_groups.py` |
| AG News train file (Hugging Face cache) | `studies/tm_language_model/data/hf_cache`, not distributed | `studies/hierarchical_tm/src/build_agnews.py` |

Two consequences reach back into tm_language_model. The run records of hierarchical_tm were written by that study's run logger, so they carry the hashes of its pre-registration and of its source tree and of no file of hierarchical_tm. And `studies/hierarchical_tm/src/feas_slide11/launch.sh` passes no results folder to the run logger, so the 27 exploratory probe runs of 2026-10-01 were written to the `results/` folder of tm_language_model. Those runs used pair data with a sampling bias, are flagged in the [run index](../docs/run-index.md) and are not used as results.

### clause_lm uses tm_language_model and hierarchical_tm

| What is used | Where it is | File of clause_lm that makes the link |
|---|---|---|
| Penn Treebank text | `studies/tm_language_model/data/ptb/raw`, not distributed | `studies/clause_lm/src/ptb.py`, through which every Penn Treebank script of the study loads its data |
| Environment script | `studies/tm_language_model/env/env.sh` | `studies/clause_lm/src/launch_gpu.sh`, which changes into the tm_language_model folder and sources it |
| Memory guard | `studies/tm_language_model/scripts/mem_guard.py` | `studies/clause_lm/src/launch_gpu.sh`; `studies/clause_lm/src/m5_run.py` when it is called with `--backend torch_gpu` |
| GPU lock, mode `exclusive` | `studies/tm_language_model/scripts/gpu_run.sh` | `studies/clause_lm/src/launch_gpu.sh`; `studies/clause_lm/src/m5_run.py` with `--backend torch_gpu`; `studies/clause_lm/src/m5_gpu_all.sh` |
| Virtual environment `.venv-htm`, which provides the package PyHierarchicalTsetlinMachineCUDA | built inside the tm_language_model folder, frozen in `studies/tm_language_model/env/htm.lock` | `studies/clause_lm/src/launch_gpu.sh` (interpreter of the job); `studies/clause_lm/src/pair_tm.py` (imports the package) |
| Queue state of hierarchical_tm | the files `jobs.txt`, `started.txt` and `queue.log` in `studies/hierarchical_tm/queue/`, local and not distributed | `studies/clause_lm/src/m5_gpu_all.sh` |

`studies/clause_lm/src/m5_gpu_all.sh` starts its GPU steps only when the queue of hierarchical_tm is idle: no job of `jobs.txt` is missing from `started.txt`, no process with the name of one of that study's drivers or queue runners is alive, and the last line of `queue.log` contains `QUEUE EMPTY` or `MPS daemon stopped`. clause_lm writes its run records with its own `studies/clause_lm/src/runlog.py`; the three launchers named in the table do not call the run logger of tm_language_model. For its router runs the study has two virtual environments of its own, one of which was prepared and not used; its README describes them.

### What follows for a clone

The launchers of both studies start `studies/tm_language_model/scripts/mem_guard.py` by name, and those of hierarchical_tm also `studies/tm_language_model/scripts/run_logged.py`. Both scripts begin with `#!/usr/bin/env python3`, so a `python3` with the environment of tm_language_model has to come first on the `PATH` when a launcher is started. The relative paths between the studies replaced absolute ones when the files were prepared for publication, and the replacement was not exercised by a run. The borrowed tools were not moved into a shared place, because that step needs a test run ([known limitations](../docs/known-limitations.md)). The launchers and queue scripts describe one workstation with a single GPU that the studies share through a lock; they record how the jobs were started and are not portable tools.

## Adding a study

A new study is one folder `studies/<short_snake_case_name>/`. It holds a `README.md` that states the question, the status and how the study is run, the folders `src/` and `tests/`, and a folder `docs/` with the registration of the study, its results page and its data cards.

The folders `results/` (run folders) and `data/` (datasets) of a study stay local. The root `.gitignore` ignores `/studies/*/results` and `/studies/*/data`, as it ignores `/studies/*/work`, `/studies/*/references`, `/studies/*/third_party/ref` and every `.venv*`, so run folders, datasets, working copies and virtual environments do not enter the repository. A results page therefore cites a run by its id, exactly as the run folder is named.

The study gets one row in the table at the top of this page and one in the root README: name, question, status and headline result. The status is one of the four words defined above.

Code is shared between studies only when a second study needs it. Until then a new study is self-contained: it reaches no code, data or queue file of another study by path. hierarchical_tm and clause_lm are the exceptions that the previous section describes.

The existing studies differ from this layout in places. tm_language_model has further folders for its experiments, scripts, environments, launch chains and a vendored package. hierarchical_tm keeps its pass rules, `studies/hierarchical_tm/gates.md`, and its hash record, `studies/hierarchical_tm/frozen_settings_hashes.json`, at the study root, because the keys of the hash record are paths relative to that folder, and its two CPU tests are in `studies/hierarchical_tm/src/`. clause_lm has no tests.
