# Known limitations

This page collects what limits the use of this repository and of its results: what a clone can and cannot reproduce, how the published files differ from the files that produced the runs, what is known about the data, the scientific limits of each study, and the runs that are flagged in the run index. It is built from the run records, from the registrations and results pages of the three studies, and from an internal audit of the whole workspace of 2026-10-03. A count taken from that audit is marked "from an internal audit of 2026-10-03, not recomputed". Counts of run records were computed from the run records on 2026-10-10. Measured values are not repeated here. They are on the results pages, each with its run id: [tm_language_model](../studies/tm_language_model/docs/results.md), [hierarchical_tm](../studies/hierarchical_tm/docs/results.md) and [clause_lm](../studies/clause_lm/docs/results.md).

Paths are given from the repository root. The short names docs/11, docs/12 and docs/13 stand for the engine specification, the log of design changes and the pre-registration of tm_language_model, and gates.md for the pass rules of hierarchical_tm.

## 1. Reproducibility

### What is not distributed

A clone holds code, configs, registrations and documents. It holds no run folder, no dataset, no saved model, no compiled engine binary, no virtual environment and no clone of a third-party reference repository. No number on a results page can be recomputed or inspected from a clone alone. A cited run id can be looked up in the [run index](run-index.md), which lists the 971 run folders of the three studies with date, exit status and flag. Nothing more of a run is published: the raw run records carry paths and host names of one machine and stay out.

Reproducing a result means obtaining the data as the [data page](data.md) describes, building an environment from the lock files, and running the experiment again on comparable hardware. No such rerun was made from the published tree, so whether one returns a recorded value is not known.

Some inputs are missing for other reasons.

- In hierarchical_tm the sentence files of the slide-6 and slide-10 tests, the script `studies/hierarchical_tm/src/slide6.py` and the test files of both tests are not distributed, because they carry sentences of the supervisor's project proposal (unpublished). The checks that use them cannot be rerun from a clone.
- clause_lm needs two local copies of Clause-Driven-LLM and the CLINC150 file. The copy that the runs used was a source archive without version history. Its upstream commit is recorded in project notes and was not checked against upstream ([upstream repositories](upstream-repositories.md)).
- Saved models are pickle files. An internal audit of 2026-10-03 notes that they are tied to the library versions of the environment that wrote them.

### Which code produced a run

No run record carries a commit id. In tm_language_model no commit was made while the study ran ([docs/12](../studies/tm_language_model/docs/12-design-changes.md) A8); the run logger writes a content hash of the study folders `src`, `experiments`, `scripts`, `tests` and `env` into each record as `source_tree_hash`. The commit field is empty in all 643 run records of tm_language_model and in all 319 of hierarchical_tm, and the records of clause_lm have no such field. The published layout does not change this: the run logger looks for `.git` in the study folder, next to `src/`, and the repository root lies two levels above it, so a run started from a clone records no commit either.

Most runs were produced by earlier revisions of the code. The 643 run records of tm_language_model carry 136 distinct source tree hashes. The most frequent one covers 209 records. 40 records carry the hash that all 319 records of hierarchical_tm also carry, which is the last recorded state of that tree. Only these records matched the code as it stood on 2026-10-03 (40 of 643, from an internal audit of 2026-10-03, not recomputed). The earlier states of the tree were not retained. The published tree is one more state: files in the five hashed folders were edited for publication (section 2), so a hash computed on a clone equals none of the recorded ones.

The same holds for the registration. `studies/tm_language_model/scripts/run_logged.py` writes the SHA-256 of the pre-registration, [docs/13](../studies/tm_language_model/docs/13-preregistration.md), into each record. Of the 643 records, 384 carry the hash of the last registered revision, 249 carry the hashes of ten earlier revisions, and 10 have no such field. The ten earlier revisions were not retained, so for those 249 runs the text that was in force at the start of the run cannot be shown. What remains is the amendment section of the file, in which every change is dated. The last registered revision itself is not part of the repository either: the published docs/13 is a copy whose wording was edited, and the [hash table](hash-table.md) gives the hash of the last registered revision next to that of the published copy.

The 319 run records of hierarchical_tm were written by the run logger of tm_language_model. Each carries the pre-registration hash of that study and one source tree hash of that study, and nothing of its own study: no record hashes [gates.md](../studies/hierarchical_tm/gates.md), the [registrations](../studies/hierarchical_tm/docs/registrations.md) or the code of hierarchical_tm. The code revision of a hierarchical_tm run is therefore known only from the script copies inside its run folder, and for the 21 scripts of the frozen settings from the hashes that were registered on 2 Oct 2026 (the "as registered" values of the hash table). Three consequences follow.

- Thirteen analysis scripts existed only as copies inside run folders. They were recovered from there into `studies/hierarchical_tm/src/recovered/` and then edited for paths and wording, so the published files are not byte identical to the copies that ran. Three of them (`devcheck.sh`, `flat_both.sh`, `e3fb_deletion_probe.py`) exist in two versions in the run folders, and one version of each is published.
- The script of the run `20261003T001723Z_e3fb_gpu_time` was not retained anywhere. The two GPU-time totals of the fallback of Check 2b cannot be traced to code.
- The eight scripts in `studies/hierarchical_tm/src/feas_slide11/` belong to the 27 exploratory probe runs of 2026-10-01. One of them, `train_tm.py`, exists in five versions across those run folders; the published file derives from the version in the last of them. None of the script copies in those run folders is listed in the manifest of its run, and in at least one folder the copies were added after the run.

The environment record of the GPU jobs of hierarchical_tm names the logging interpreter and not the interpreter that ran the job. `studies/hierarchical_tm/src/launch.sh` starts a job as `scripts/run_logged.py ... -- scripts/gpu_run.sh exclusive -- .venv-htm/bin/python ...`. The run logger records the environment of the interpreter that runs the logger itself, which was `.venv-tm` in the recorded runs, while the job ran in `.venv-htm`. The two environments pin different commits of PyHierarchicalTsetlinMachineCUDA (`studies/tm_language_model/env/tm.lock` and `studies/tm_language_model/env/htm.lock`). Package versions of GPU jobs are therefore taken from [frozen_settings_hashes.json](../studies/hierarchical_tm/frozen_settings_hashes.json) and from the second of these lock files, never from the environment record of a run.

clause_lm keeps a simpler record format of its own. Its nine run folders have no `run.json`, so they record no source tree hash, no registration hash and no exit status. Each holds its config, its metrics, the hashes of its data files and a copy of the script that ran. The published `studies/clause_lm/src/m5_run.py` is a later revision than the copy in the two Gate 0 run folders. The four Penn Treebank runs record no environment. No run record hashes the registration of M5 step 1: the records store its path only, and no version of it without the amendment of 21:11 UTC was retained. The keys of `data_hashes.json` in the existing run folders name the data files by the folder layout at the time of the runs, while a run started from the published tree writes keys that begin with `clause_lm/` and `tm_language_model/`.

The three studies thus use three record formats, and the run index leaves a cell empty where a record does not hold the value.

### One workstation, and studies that depend on each other

All runs come from one workstation with a single Tesla V100 and a quota of 6 CPUs. The code encodes that machine: the build target of the CUDA kernels, the thread caps and CPU slots, the GPU lock, the memory guard that reads control-group files, and the queue of hierarchical_tm, which lets two jobs share the GPU. Launchers, queue scripts and launch chains are published as the record of how the runs were started. They are not portable tools.

hierarchical_tm and clause_lm are not self-contained. hierarchical_tm starts its jobs through the run logger, the environment file, the GPU lock and the virtual environments of tm_language_model. clause_lm starts its GPU steps through the environment file, the memory guard and the GPU lock of tm_language_model and logs its runs with its own `studies/clause_lm/src/runlog.py`. Both read data from the `data/` folder of tm_language_model and reach that study as the sibling folder `../tm_language_model`. hierarchical_tm has no environment file of its own. These relative paths replace absolute ones, and the replacement was not exercised by a run (section 2).

Of the fourteen files in `studies/tm_language_model/launch/`, four are copies that were rebuilt on 2026-09-28, after the originals had been lost in a host reboot of 2026-09-27, and three scripts that the published files name were not retained ([launch/README.md](../studies/tm_language_model/launch/README.md)). The queue scripts of hierarchical_tm leave shell variables unquoted on many lines, so a clone path that contains a space breaks them. The internal audit also notes that the job list of that queue was edited while the queue ran.

### Tests and checks

The published tree was not run: no test, no experiment, no build and no install was run on it. The last runs of the tests and experiments were made on the working copies, before the edits for publication.

- The tests of tm_language_model were run during the internal audit of 2026-10-03, on the CPU only and on a copy of the code as it then stood. The GPU tests were not run then. The counts of that run describe an earlier state of the code and are not repeated here.
- The GPU tests of tm_language_model are not marked as such (33 tests, from an internal audit of 2026-10-03, not recomputed), so `make test` uses the GPU without taking the GPU lock.
- The numbers check of tm_language_model needs the run folders. `make lint` runs `studies/tm_language_model/scripts/check_numbers_have_logs.py`, which accepts a label of the form `[measured: results/<run id>/]` only when that folder exists. Without the run folders `make lint`, `make smoke` and the test that runs the check on the real tree fail. The check tests that the cited folder exists. It does not compare the number with the content of the folder.
- `make smoke` is the lint followed by the tests. The smoke run that a comment in the Makefile describes (a build, a short training, parity with the GPU, receipts) is not implemented.
- The training driver of hierarchical_tm has no unit test beyond its numpy verification mode (from the internal audit), and the test files of the slide-6 and slide-10 tests are not distributed.
- clause_lm has no tests, and neither have the launch chains of tm_language_model.

## 2. The published copies

The files in this repository are copies that were edited for publication on 2026-10-10 and 2026-10-11. The edits concern wording, comments, docstrings and paths. This section says where a published file is no longer the file that a registration or a run record refers to.

### Frozen records

The wording of the frozen records was edited. Their registered content is unchanged: hypotheses, thresholds, seeds, analysis plans, amendments and dates stand as registered. This applies to the pre-registration of tm_language_model (docs/13) and to four of its fifteen data hash records ([docs/notes/README.md](../studies/tm_language_model/docs/notes/README.md)), to gates.md, the registrations page and the `prepared` text of frozen_settings_hashes.json in hierarchical_tm, and to the [registration of M5 step 1](../studies/clause_lm/docs/prereg_M5_step1.md) in clause_lm. Each carries a publication note; for a JSON record the note stands in a README next to it.

The registrations page of hierarchical_tm was put together for publication from registrations that were written into the lab notebook of the study, which is not published. The registration times of hierarchical_tm and clause_lm are the times written in those records. With no commit history, and with no hash of these files in the run records, the order of a registration and a run rests on the recorded times and on the UTC start time in each run id. Four time entries of 2 Oct 2026 in hierarchical_tm were first written ahead of the clock and later corrected from file times; the results page lists this among its disclosures.

### Hash pins

Some files are pinned by a SHA-256 that a config or a record holds. Where such a file had to change, its pin was replaced in the same step. The [hash table](hash-table.md) gives the hash as registered and the hash as published for each of them.

- tm_language_model: nine source files under `studies/tm_language_model/src/tmlm/cvtm/` are pinned by seven configs named `cvtm_a3_H8_*.json`. Three of the nine (`__init__.py`, `doc_clauses.py` and `mixture.py`) were reworded in a docstring or a comment and re-pinned in the seven configs. The other six are byte identical to the files that the runs used.
- hierarchical_tm: frozen_settings_hashes.json pins 21 scripts. Fourteen were edited and re-pinned: comments and docstrings, absolute paths that are now derived from the location of the file, and two error messages in `studies/hierarchical_tm/src/slide10.py`. For these fourteen the file now holds the published hashes, and the registered values are in the hash table. Six scripts are byte identical.
- One pin disagreed before publication. `studies/hierarchical_tm/src/text_driver.py` was changed after the settings were frozen (a printing option, disclosed on the results page). Its pin is published as registered and was not repaired. One docstring line of the file was reworded for publication on top of that, so the published file matches neither its pin nor the file as it stood before publication.
- Pins on files outside the repository keep their registered values and cannot be checked from a clone: the 17 data files of hierarchical_tm, the dataset hashes in the data hash records of tm_language_model, the hashes of files inside run folders, and the hashes of data files that the run records of clause_lm hold.

### As-run configs

A run copies its config into its run folder, and its record holds the SHA-256 of the config. The text of 23 configs in `studies/tm_language_model/experiments/cvtm/configs/` was edited for publication. Only string values changed, as the README of that folder says, which also lists the files: descriptive text, an absolute folder prefix at the start of two command strings in `ssl_r4_controls.json`, and the three replaced pin values in the seven `cvtm_a3_H8_*.json` files. These published configs no longer have the config hashes that the run records of their runs hold. The two scripts that compare a config with a hash stored in a run folder, `studies/tm_language_model/experiments/ssl_r4_controls.py` and `studies/tm_language_model/experiments/cvtm_transformer_tuned.py`, can therefore report a difference for them.

### What needs a run to be verified

The edits were not exercised by a run. The following points need one.

- The helper scripts of tm_language_model that began with a fixed interpreter path now begin with `#!/usr/bin/env python3`. Started by name, as the launchers of hierarchical_tm start `scripts/mem_guard.py` and `scripts/run_logged.py` of tm_language_model and those of clause_lm start `scripts/mem_guard.py`, they run under whichever `python3` comes first on `PATH`.
- Comment edits in the engine sources of tm_language_model change their modification times, and the build compares modification times, so the engine binaries are rebuilt at first use. Bit-exact parity of the three engines after such a rebuild was not re-tested.
- hierarchical_tm and clause_lm now reach tm_language_model through the relative path `../tm_language_model`.
- In `studies/hierarchical_tm/src/slide10.py` the wording of two error messages changed. The test file of the slide-10 test, which is not distributed, matches the earlier texts.
- `studies/hierarchical_tm/src/build_pairs.py` now derives the location of its source data from its own location and writes that path into the split file `data/splits.json` of the study. A rebuild in a clone therefore gives a split file whose hash differs from the pinned one, even when the splits are the same.
- `studies/hierarchical_tm/src/final_test_report.py` records the hash of `studies/hierarchical_tm/src/data_checks.py` at each run. Both files were edited and re-pinned, so a new report records another hash than the existing report.
- The recovered scripts of hierarchical_tm copy themselves into their run folder. A copy started from a run folder no longer finds the study folder.
- The unquoted variables in the queue scripts of hierarchical_tm and the record keys of `studies/clause_lm/src/runlog.py`, both described in section 1.
- Several passages are known to be stale and were left as written: the first paragraph of the docstring of `studies/clause_lm/src/m5_run.py`; the header line of the Makefile of tm_language_model that says nothing is installed, although the target `bootstrap` runs the installer; the docstring of `studies/tm_language_model/src/tmlm/ssl/__init__.py`, which names amendment A6 where A7 is meant; the launch-command comments of `chain_hao_diag2.sh` and `chain_hao_diag3.sh` in `studies/tm_language_model/launch/`, which quote the command of `chain_hao_diag.sh`; and the sentence of the engine specification (docs/11, section 8c) that the engine doubled its accuracy with the negatives switched off or drawn uniformly, which rests on exploratory runs without a run folder, while the one logged diagnosis run does not show a doubling (results page of tm_language_model).
- Refactors that need a test run were not made: regrouping the experiments, moving shared helpers into the package, making the study folders installable, marking the GPU tests, and moving the tools that two studies borrow from tm_language_model to a shared place.

## 3. Data

No dataset is distributed. The [data page](data.md) gives the source and the version of each one and says how to obtain it.

- Duplicate articles cross the splits of the Wikipedia pair data of hierarchical_tm. The splits are made by whole article and no article is in two splits, but WikiText-103 contains repeated articles. About 2.1% of the test paragraphs are word for word identical to training paragraphs, in 235 of the 2,847 test articles (from an internal audit of 2026-10-03, not recomputed). The final test was not scored again without these articles, and the [data card](../studies/hierarchical_tm/docs/data_card_wikipedia_pairs.md) does not mention the duplicates.
- The 27 probe runs of 2026-10-01 whose names contain `feas_slide11` used pair data with a sampling bias. They came before hierarchical_tm, sit among the run folders of tm_language_model and are flagged in the run index. No number from them is used.
- The main synonym data of hierarchical_tm (`data/ns_all` of the study) is borderline on the registered bound on skewed words and passes Check 1b only by a recorded decision of 1 Oct 2026. As a safeguard, the verdict of Check 4b leaves out every word group that contains one of the skewed words. The data card of the Wikipedia pairs does not cover this data set.
- The pair data of hierarchical_tm reduces a paragraph to the set of its words among the 1,024 most frequent ones, without counts and without order.
- The export step of the artificial data (L1 to L3) of hierarchical_tm ran outside the run logger (from the internal audit).
- The built data of tm_language_model departs from its plan in points that docs/12 F16 lists: merged headings in the train split, fewer validation and test pairs than planned, clustering levels that did not converge within their cap, and a measured n-gram overlap between the test split and the training split.
- clause_lm reads the Penn Treebank text in the form prepared by tm_language_model and has used its validation split only. Its pair data keeps the 1000 most frequent words.

## 4. Scientific limits

The statements below are limits of the evidence. The status of every registered id and the measured values are on the results page of each study.

### tm_language_model

The core result of this study is negative: its main accuracy goal was missed, 11 of the 30 hypotheses that were run are not met, and 61 of the 91 registered hypotheses were not run. Results page: [studies/tm_language_model/docs/results.md](../studies/tm_language_model/docs/results.md).

- The main accuracy comparison, H1.1, was run on seed 1 only (docs/13 A3).
- The count-vote read-out is credited through comparisons whose baseline is the model's own read-out. H8.1 compares the count-vote read-out with the signed votes of the same clause bank, not with an n-gram model. The count-vote core contains the Kneser-Ney 5-gram model as a component, so its comparison with that model (H8.3) says nothing about Tsetlin learning. docs/13 A3 states this itself and credits Tsetlin learning only through H8.1 and H8.10.
- The hypothesis families from H8 on were registered by dated amendments after the main accuracy goal had been missed. A3 lists the exploratory runs that had been seen before its thresholds were set. A run id that ends in `_exploratory` marks an exploratory run, and choices made after results were seen are labelled as such on the results page.
- The internal audit notes that the small masked model of 2026-09-22 (the tiny LM) and the count-vote results it examined were measured on validation data only, and that the comparison with the stock GraphTsetlinMachine model (E1) covers the first epoch only.
- In the confirmatory report of the self-supervised family, a comparison with a random rule bank was computed without a registered threshold and has no status. The notes of the study did not report it; the results page does.
- H17.5 was run exactly as registered, on the package commit on which no configuration tried in the tuning pilot of 2026-09-25 had trained (docs/12 F27 and F29). Its outcome therefore says little about the hierarchical mechanism as such.
- Many registered hypotheses never ran. The internal audit gives a tally, which it marks as not recounted, so the tally is not given here; the results page lists every id with its status. Fifteen configs in `studies/tm_language_model/experiments/configs` name scripts that were never written, the subpackages `featurizer`, `graphs` and `heads` are placeholders, the kernel patches planned for milestone M5 were not applied (`studies/tm_language_model/src/tmlm/kernels/graphtm_patched/kernels.py` is a byte-identical copy of the stock kernel file), and the evaluators that `studies/tm_language_model/src/tmlm/eval/__init__.py` lists as planned were not written. No recorded decision cut these items (docs/12 F4). The study was archived before they ran.
- The standalone receipt checker, `studies/tm_language_model/src/tmlm/certify/checker_standalone.py`, recomputes the random draw of a sampled token from the seed, the stream and the counters that the receipt itself records. The internal audit reports that the checker does not tie these inputs to the seed of the generation run or to the position of the token, and that a receipt with a forged draw was accepted. On that reading a receipt of a sampled token shows that the recorded draw selects the claimed token, not that the draw was the one the generator had to make. The internal audit also describes the "SAT receipts" of the study as mostly brute-force re-checks; this was not examined again.
- The internal audit reports that the C++ engine does work proportional to the vocabulary size for every example, which the engine specification (docs/11) does not intend. This was not examined again.
- The receipt and replay hypotheses H2.2 and H2.4 were judged at the scale of the tiny language model, and the training half of H2.2 and the whole of H2.3 from a test run that left no run folder (results page).
- That the three engine implementations give identical results is asserted by tests. The GPU tests were not run in the audit, and nothing was run for the publication.

### hierarchical_tm

The core result of this study is negative: no learned rule captured either of the two sentence families of the proposal, and the single final test is below word overlap on both Wikipedia tasks. Results page: [studies/hierarchical_tm/docs/results.md](../studies/hierarchical_tm/docs/results.md).

- Every number is a dev number except those of the one final test. Both `s` and the number of passes were chosen on the dev data of seed 1 on same-article pairs, so the dev numbers of that seed are slightly optimistic.
- The results concern one package version and small settings: on the Wikipedia tasks a vocabulary of 1,024 words and 40,000 training pairs per data set. They say nothing about larger settings.
- Four caveats apply to the final test. The duplicate articles of section 3 are in its test split. The tuned accuracy of the final-test table has its cut-off chosen on the test labels, although gates.md defines tuned accuracy with a cut-off chosen on dev; the registered comparisons use AUC and do not depend on it (from the internal audit). The article bootstrap resamples only the article of paragraph A; the audit estimates that resampling both articles would give wider intervals, and it did not verify that estimate. And the seeds are not fully independent repeats: `studies/hierarchical_tm/src/htm_driver.py` seeds the global numpy generator with the seed plus the pass number at the start of every pass, so seed 2 at pass 1 draws the same stream for batch order and for the shuffling inside the package as seed 1 at pass 2.
- Some results rest on one seed: each value of the `s` screen, and the four runs with one package option switched. The notes of the study did not report those four runs; the results page does. They have no registration and no status.
- Choices made after results were seen: the accuracy that counts in Check 2a, on which the borderline result for L2 depends; the rule that selects `s` for the later L3 runs, which is dated about 02:58 CEST on 2 Oct 2026 and was written after the L3 result at `s` 400 on seed 1 existed (its `metrics.json` is of 02:52 CEST); the third arm of Check 4b at `s` 200; the reading of the time cap of the Check 2b fallback; and the robustness arms R1, R2 and S6-R1. One registration was not kept: Check 4a was scored before the branch decision that it was to follow. The results page labels each of these.
- The models of the slide-6 and slide-10 tests were shaped to the known template, and the near-miss sentences of the slide-10 test follow a fixed one-word substitution rule and were not written independently as planned. The near-miss part of Check 2b is weak evidence.
- Check 4b is met on the `s` 100 arm only, on 2 of 3 seeds and with a small effect, and most counted word groups of that arm are left out of the verdict by the safeguard of section 3. It is not met on the arms at `s` 200 and `s` 400. The results page fixes the wording that this result allows.
- The freeze rule of gates.md, under which a number that does not reproduce within 0.02 is withdrawn, has no result: the study has no run after 3 Oct 2026.
- The length-only part of Check 1b applies to the data with shared words removed only. The internal audit reports that a classifier on the two paragraph lengths alone is above chance on the dev pairs of the same-article and next-paragraph data, a value it did not verify, and recommends checking how much of the model's score paragraph length explains. That check was not made.
- No speed number is reported, because most runs shared the GPU.
- In `length_feats` of `studies/hierarchical_tm/src/data_checks.py` the absolute difference of the two paragraph lengths is taken before the conversion to float. The internal audit reports that it wraps around for unsigned counts and found no effect on the reported results. The same function also passes the smaller and the larger length, from which the difference follows.
- The extraction of OR-groups from a model exists in several copies across the scripts of the study (from the internal audit).

### clause_lm

This study has no result on its own question yet: no Tsetlin Machine was trained on its language data, and the reproduction of an existing Tsetlin Machine router did not meet its first gate. Results page: [studies/clause_lm/docs/results.md](../studies/clause_lm/docs/results.md).

- The study is preparation. Every result comes from a single run, and none was repeated with a second seed. The Penn Treebank results are validation results.
- No Tsetlin Machine has been trained on Penn Treebank in this study. Tsetlin Machine training took place only in the Gate 0 runs of M5 step 1, which train the coalesced Tsetlin Machine of the Clause-Driven-LLM router. The clause evaluator ran once with k-means word groups, and the pair model was written and not run.
- Gate 0 of M5 step 1 was not met in its one completed run, and the cause is not confirmed. The explanation in the working notes of 2026-10-02, a difference between the two training backends of Clause-Driven-LLM, rests on reading the code and is tested by no run. A retry was stopped without a score, the GPU rerun of the amended gate did not take place, and the registered question has no verdict.
- The target of Gate 0 is one recorded run of the upstream repository with one seed. The embedding results of the upstream repository were not rerun in this programme.
- The data loader that the runs used is a rebuild of a module that is absent upstream, not the original module, and its Banking77 part is an unverified reconstruction that M5 step 1 does not use.

## 5. Flagged runs

The [run index](run-index.md) flags 125 of the 971 runs: 120 of tm_language_model and 5 of hierarchical_tm. None of the nine runs of clause_lm is flagged. No number from a flagged run is given as a result anywhere in this repository; a flagged run may still be named as a fact. Three runs carry two flags, so the counts below add up to 128.

| Flag | Runs | Meaning |
|---|---|---|
| never cite | 2 | Withdrawn by the log of design changes of tm_language_model. `20260922T111529Z_envcheck` was made while the container was attached to another GPU model (docs/12 F9). `20260922T220251Z_s4` was stopped before completion on a cost analysis that was wrong, and holds no result (docs/12 F18). |
| stale (recorded as running) | 5 | The record still says "running" and has no end time. Four are runs of the night of the host reboot of 2026-09-27; runs with the same names were started again on 2026-09-28. The fifth is `20260919T203409Z_p0_tmu_ae_run_seed3`. |
| killed | 50 | The record holds a terminating signal or the exit status 143. |
| smoke | 44 | A plumbing test, named so in its run id: 39 of tm_language_model and 5 of hierarchical_tm. |
| biased exploratory | 27 | The `feas_slide11` probe runs of 2026-10-01, whose pair data had a sampling bias (section 3). |

A run without a flag is not always a clean run. In tm_language_model 545 of the 643 records have exit status 0; 50 have exit status 143, 5 are the stale records, and 43 ended with exit status 1, an error of their own, for which the run index has no flag of its own; a few of them are smoke runs and carry that flag. In hierarchical_tm 315 of the 319 records have exit status 0, and the results page names the other four.
