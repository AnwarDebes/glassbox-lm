# tally: a count floor and a test bench for Tsetlin learning

This study holds the code and the run records of the first steps of the design described on the page [research/design.md](../../research/design.md), whose working name is Tally. The design has two parts that are kept apart: a count floor, a language model built from counts with no Tsetlin Machine in it, and a test bench, places in that model where a Tsetlin Machine is compared on the same bits with a twin that only counts and with standard learners. The study covers three lines of that design: line T (T0, B1 and B6, Tsetlin tests that share no code with the language model), line A (steps A1 and A2 of the count floor) and line C (C3, the neural comparison as a curve over training budgets). Its run folders date from 2026-10-15 to 2026-10-17.

Two tests are registered. A2's registered look on dev-test, on 2026-10-25, met its success line: the class members of an exchange-class code lowered the perplexity of a Kneser-Ney 5-gram with a cache by 7.59% (95% interval over dev-test blocks 6.51% to 8.58%; without train-side noise), confirming the development result; it is exploratory (one build, one split, one benchmark) and a result about count models, not about Tsetlin learning ([results](docs/results.md#a2-the-registered-look-on-dev-test-2026-10-25)). T0 has two amendments that fix the settings of its two machines. Its verdict was computed on 2026-10-25: on the coalesced machine of tmu, the learned votes beat the counted read-out of the same clauses, +0.251 bits per word (+0.222 to +0.280) for counted minus learned on the dev-test split of Penn Treebank, so the H8.1 finding of tm_language_model is not shown with this library, and the line "Tsetlin learning helps the counted read-out" is not met. The other lines have code and smoke, timing, baseline, noise or exploratory runs, and no frozen registration; A2 has a draft registration that has not been reviewed.

The status of every registered id and every run folder is on the results page, [docs/results.md](docs/results.md). The datasets are described in the data card, [docs/data.md](docs/data.md).

## Lines of work

| Id | Line | Question | Code |
|---|---|---|---|
| T0 | T | On Penn Treebank next-word prediction, with the tmu library (commit 5605ff0), do counted votes on the same clauses beat the learned votes? T0 repeats hypothesis H8.1 of tm_language_model, which was tested with that study's own engine on WikiText-103, with the library of the Tsetlin Machine group. Registered in [registrations/T0.md](registrations/T0.md). | `src/tally/t0/`, `scripts/t0_auto.py`, `scripts/t0_final.sh` |
| B1 | T | Does marginal-bit feedback, which pays each member the bits the team saved because it spoke, find a planted rule set with fewer samples than a shared reward, and does the gap grow with team size? Second question: does the dual-state trial, in which the old and the changed clause are both kept and counted until a sequential test decides, beat applying a change directly? | `src/tally/b1/`, with its own [README](src/tally/b1/README.md); `scripts/b1_launch.sh` |
| B6 | T | Does rule T, under which a member learns from an example with a probability equal to the out-of-fold error of the rest of the team, beat a blind composite of Tsetlin Machines at an equal number of clauses, on Fashion-MNIST and AG News? | `src/tally/b6/`, with its own [README](src/tally/b6/README.md) |
| A1 | A | Protocol, noise, and a fair measure of the room an arbiter has. The code measures interpolated modified Kneser-Ney 3-gram and 5-gram models and a 5-gram model with a cache on the validation split, with the study's own code and with KenLM. | `src/tally/floor/`, `scripts/a1_baseline_check.py`, `scripts/a1_selftest.py` |
| A2 | A | Which category code helps? Each arm adds the class members of one code to a Kneser-Ney 5-gram model with a cache; the comparator is the same system without class members. | `src/tally/floor/codes.py`, `classlm.py`, `codeeval.py` and `exchange.py`; `scripts/a2_run.py`, `scripts/a2_selftest.py` |
| C3 | C | A neural comparison as a curve over training budgets. The code is a word-level LSTM language model with the settings of Zaremba et al. 2014 that records the validation perplexity after every epoch together with the number of training tokens seen; its docstring calls it exploratory, a preparation of the C3 comparison with no claim. | `src/tally/neural/lstm.py` |

## Status

Running. A2 was frozen and its one look ran on 2026-10-25: success line met (see the A2 line below). T0 was frozen on 2026-10-15 at 07:27:25 UTC (`registrations/T0.frozen.json`). Its tuning rule fixed the settings of the two machines in two amendments: s = 20 and 9 epochs for the coalesced machine, frozen at 18:24:29 UTC, and s = 5 and 1 epoch for the flat machine, frozen at 21:14:27 UTC. The final runs of seeds 1 to 10 of both machines and the ten runs of the logistic-regression arm have finished, and `src/tally/t0/report.py` computed the verdict from them once, at 10:43:15 UTC on 2026-10-25 (run `20261025T104315Z_t0_report_devtest`): T0 is not met, the learned votes beat the counted read-out. The Kneser-Ney 3-gram and 5-gram models that the registration reports beside the arms were scored on the same dev-test by `scripts/t0_kn_beside.py` (run `20261025T104406Z_t0_kn_beside_devtest`). An independent recompute in `verification/` matches the report. The final runs had logged their per-seed dev-test means, which were read before the report; none was used for a choice. The results page gives the numbers, the checks and their limits.

B1 and B6 have code, self-checks and smoke or timing runs, and no registration. `src/tally/b1/README.md` calls a registration in `registrations/` the binding text, but no B1 registration is in that folder. `src/tally/b6/README.md` runs its grid after a registration in the same folder is frozen, and there is no B6 registration there either. A1 has four runs of its baseline check and one A/A noise run, and no registration. A2 has five smoke runs and three development runs on the validation split (counts of 2026-10-25, 13:24 UTC; an earlier version of this sentence gave three and one), and a draft registration of its one look at dev-test, `registrations/A2.md`, which has been reviewed twice and revised and is not frozen. C3 has 20 LSTM runs on the validation split, two sizes with seeds 1 to 10, which are under-trained (below). No code of the study opens the test file of Penn Treebank.

### The lines other than T0, as of 2026-10-25

- **A1** (count floor, no Tsetlin part). The baseline check matches KenLM on valid within 3.8e-5 in perplexity; its fourth run, `20261025T113821Z_a1_baseline_check`, reproduces the third exactly and records the KenLM binaries. The A/A noise runs (`scripts/a1_aa_noise.py`, run `20261025T113715Z_a1_aa_noise`, valid only, no rule) rebuild the systems of the baseline check ten times, each time without a seeded 5% of the train blocks: the valid perplexity of KN5 + cache (reset per block) has a mean of 134.569 and a standard deviation of 0.304 over the ten repeats (KN3 164.252 and 0.407, KN5 154.769 and 0.447), two repeats differ by at most 0.59% (KN3 0.66%, KN5 0.82%), and for 8 of the 45 pairs of repeats (KN3 7, KN5 9) the 95% interval from resampling the valid blocks excludes 0. That interval covers the evaluation text and not the train side. The third part of A1, a fair measure of the room an arbiter has, has no code.
- **A2** (count floor; no Tsetlin part except the code `orgroups`, which comes from the OR-groups of a hierarchical_tm run). The development run on valid (`20261025T114718Z_a2_valid_dev`) gives KN5 + cache 132.930, and with the class members of the exchange code 122.150 (8.11% lower; 35% lower on the positions whose bigram is unseen in train); k-means-200 1.02%, the predefined code 0.93%, the topic code 0.10% and the OR-groups 0.00% lower; the ten random codes get mixing weights near 0 and stay at 132.928 to 132.935. That is a development reading, not a verdict; two later development runs with later versions of the script (`20261025T123159Z_a2_valid_dev`, `20261025T132230Z_a2_valid_dev`) reproduce it exactly. The registration names the exchange code as the claim-carrying arm of the dev-test look (2nd use of dev-test, after T0). It was frozen at 13:38:45 UTC on 2026-10-25 and the look ran once (`20261025T133912Z_a2_devtest_final`): KN5 + cache 124.251, with the exchange members 114.821, a gain of 7.59% (95% block interval 6.51% to 8.58%), so the success line is met, as the dev work on calib and on train subsamples had predicted; the registered recompute matches on every check.
- **B1** (line T). Code, self-checks, 12 smoke runs at team size 8 and three smoke calibrations; no timing run and no registration. Four problems have to be settled before a registration (`src/tally/b1/README.md`): the shared reward never reached the target in the calibration that ran the current code (60 of 60 runs censored at 1,000,000 samples), so a primary ratio shared over marginal would rest on a comparator that does not learn on this game; the team constants of `team.default_params()` (s 3.9, RHO 0.25) were not chosen by a calibration, and RHO 0.25 is in no calibration grid; the caps are provisional (`run.CAPS_FROM` is unset, and a registered run refuses to start until a timing run fixes them); the dual-state success line has no interval condition. Fixed on 2026-10-25: censored runs no longer read as a dual-state kill in `src/tally/b1/report.py`, and the launcher follows the machine rule (nice 19, thread pools at 1, the CPU gate before every start). The grid of 637 runs is estimated at about 320 CPU-hours for the team and search arms plus 40 to 100 for the flat arm, by a linear extrapolation in team size from the smoke speeds (UNVERIFIED).
- **B6** (line T). Prepared data with manifests and nine timing runs of single members; the arms ruleT, nocomm, coupled, seqboost, fusion and gate have never run, not even as a smoke run, no tuning run exists, and `src/tally/b6/config.py` leaves open the T multiplier (`T_SCALE`), the epochs and the base of the single machine, which the tuning is to fix. Its report already holds the lines of the design (success at least 0.5 points with the interval above 0 on one dataset and the other not entirely below 0; kill under 0.2 points on both). The grid is estimated at about 145 CPU-hours for Fashion-MNIST and 86 for AG News, plus 30 to 80 for the tuning, from the timing runs (UNVERIFIED). A registration needs the smoke runs of every arm and the tuning on valid first.
- **C3** (line C). The 20 runs of 2026-10-15 are under-trained, so they are no reference for the published values. They used the mean cross-entropy over all tokens of a chunk with the learning rate 1 of Zaremba et al. 2014, whose reference code adds the loss of the 35 unrolled steps (each averaged over the 20 rows of the batch), a gradient 35 times larger, and they clipped the large model at 5 where the paper uses 10. Their best valid perplexities are 147.8 to 148.5 (medium) and 113.4 to 113.8 (large), against the published 86.2 and 82.2. `src/tally/neural/lstm.py` now follows the paper by default (`--loss sum_steps`, clip 5 medium and 10 large), keeps the old setting behind `--loss mean_tokens --clip 5`, and records the gradient norm and the share of clipped steps per epoch; a CPU test checks the loss scaling (`tests/test_lstm_loss.py`). The rerun with the loss and the clips of the paper ran on 2026-10-25 and 26 (validation split only, exploratory, no claim). The medium model reaches a best valid perplexity of 90.99 to 92.12 on 10 of 10 seeds (paper 86.2). The large model reaches 86.22 to 87.05 on 6 of 10 seeds (paper 82.2); the other 4 collapsed in their first epoch to about the perplexity of a unigram model (686.60 to 688.61). The runs train on the study's training split, 90.0% of the standard training file; the gap to the paper is not explained ([results page](docs/results.md#c3)). Sources: arXiv 1409.2329v5, section 4.1 and Table 1; `main.lua` of github.com/wojzaremba/lstm at commit 76870253; `ClassNLLCriterion` of torch/nn, whose `sizeAverage` is true by default.
- A3 and A4 have no code.

## How the study is run

Every command runs from the study folder with `PYTHONPATH=src`, except the scripts in `scripts/`, which set the import path themselves. Every run writes a run folder `results/<UTC time>_<name>/` through `src/tally/runlog.py`.

T0:

- Tuning, seed 1, one run per value of s: `python -m tally.t0.run --mode tune --seed 1 --s 10 --epochs 10 --patience 3` for the coalesced machine, and the same with `--machine flat --clauses 100` for the flat machine. The tuning runs record these arguments (`run.json`, key `argv`).
- `python scripts/t0_auto.py coalesced|flat` reads the run records of the three tuning runs of one machine every 300 s until none of them is recorded as running, applies the registered selection rule, writes the dated amendment `registrations/T0_amendment_<machine>_settings.md`, freezes it with `scripts/freeze.py` and starts the ten final runs, each in one CPU slot at nice 10. For the coalesced machine, each final run is followed by its logistic-regression run under the GPU lock. Both amendments name `scripts/t0_auto.py` as the script that wrote them.
- `scripts/t0_final.sh S_COAL E_COAL S_FLAT E_FLAT` starts the same final runs with the settings given on the command line.
- The verdict: `python -m tally.t0.report <coalesced final runs> --logreg <logistic-regression runs> --flat <flat final runs>`. The script reads the per-position bits of the runs (`bits.npz`), scores the dev-test split unless `--split` names another, and writes its result to `metrics.json` of a run folder of its own.

B1: the commands are in `src/tally/b1/README.md`. The full grid is `scripts/b1_launch.sh registrations/<file>.md [JOBS]`, which stops when the freeze record of the registration it is given is missing. It starts each run at nice 19 with every thread pool capped at 1, one after the other, each only when the CPU gate (`../decision_layer/scripts/cpu_load.py 10`) reads at most 4.5 cores; a gate reading that fails counts as closed (since 26 Oct 2026). `B1_DRY_RUN=1` prints the jobs and starts nothing.

B6: the commands are in `src/tally/b6/README.md`: the data preparation in the environment `.venv-b6`, smoke, timing and tuning runs, the grid through `python -m tally.b6.launch` and the verdict through `python -m tally.b6.report`. The launcher starts its jobs at nice 19 with every thread pool capped at 1, each after the same CPU gate. A gate reading that fails still stops it while its started jobs go on. The change made in B1's launcher on 26 Oct 2026 was not made here, because the launcher lies under `src/`, which the freeze record of A2 pins and A2's scripts check; it belongs before B6's own freeze.

A1: `python scripts/a1_selftest.py` tests the Kneser-Ney model, the cache, the duplicate mask, EM and the block bootstrap on small synthetic streams; `python scripts/a1_baseline_check.py` runs the baseline check and needs the KenLM binaries. `python scripts/a1_aa_noise.py [--seeds 1-10] [--drop 0.05]` runs the A/A noise runs on train, calib and valid, and `python scripts/a1_aa_selftest.py` tests their helpers (`src/tally/floor/aa.py`) on synthetic streams.

A2: `python scripts/a2_selftest.py` tests the A2 code on small synthetic streams. `python scripts/a2_run.py --split valid` is the development run and `--split valid --smoke` the same with small sizes. `--split devtest --final --claim-arm exchange --registration registrations/A2.md` is the one dev-test look. Before it makes a run folder or reads any data, the script refuses that start unless the registration's freeze record matches every pinned file and no earlier look exists; in the look it builds no code and stops before it scores dev-test if the cache or the mixing weights differ from the development run (`tests/test_a2_args.py`). `python scripts/a2_recompute.py --run <run folder> --out <file> [--rescore]` recomputes a run's reading from its `logp.npz` with a bootstrap of its own and, with `--rescore`, scores its split again (for dev-test only the finished registered look; `tests/test_a2_recompute.py`).

C3: `python -m tally.neural.lstm --size medium|large --seed S [--epochs N] [--loss sum_steps|mean_tokens] [--clip C]` needs torch and a CUDA device. The defaults follow Zaremba et al. 2014; `--loss mean_tokens --clip 5` is the setting of the 20 runs of 2026-10-15, and `--epochs` stops early for a smoke run.

A registration is frozen with `python scripts/freeze.py registrations/<file>.md [extra files...]`. The record pins the registration, every file under `src/` and the extra files, so a registration names the scripts and data files of its job as extra files.

## What the study needs from other studies

The study is not self-contained. It reaches scripts, environments, data and one run folder of other studies by relative paths and expects them as sibling folders under `studies/`.

| What is used | Where it is | File of tally that makes the link |
|---|---|---|
| CPU slot, one slot per job | `../tm_language_model/scripts/cpu_slot.sh` | `scripts/t0_auto.py`, `scripts/t0_final.sh`, `scripts/b1_launch.sh`, `src/tally/b6/launch.py`; the commands in `src/tally/b1/README.md` and `src/tally/b6/README.md` |
| CPU gate: the cores in use by the cgroup over 10 s, at most 4.5 before a start | `../decision_layer/scripts/cpu_load.py` | `scripts/b1_launch.sh`, `src/tally/b6/launch.py` (added 2026-10-25) |
| GPU lock, mode `exclusive` | `../tm_language_model/scripts/gpu_run.sh` | `scripts/t0_auto.py`, `scripts/t0_final.sh` (the logistic-regression arm of T0) |
| Virtual environment `.venv-tm`, which provides tmu at commit 5605ff0 | built inside the tm_language_model folder, frozen in `../tm_language_model/env/tm.lock` | `scripts/t0_auto.py` (also for `scripts/freeze.py`), `scripts/t0_final.sh`, `scripts/b1_launch.sh`, `src/tally/b6/launch.py` |
| Virtual environment `.venv-cdllm-gpu` (torch on the GPU) | built inside the clause_lm folder, described in its README | `scripts/t0_auto.py`, `scripts/t0_final.sh`; the CPU test of the C3 LSTM, `tests/test_lstm_loss.py` |
| Penn Treebank text | `../tm_language_model/data/ptb/raw/{train,validation}/text.utf8`, not distributed | `src/tally/ptb.py` |
| WordNet 3.0 (nltk_data) | `../tm_language_model/data/concept_eval/nltk_data`, not distributed | `src/tally/floor/codes.py`, `scripts/a2_run.py` |
| Fashion-MNIST train file, AG News train and test files | the Hugging Face cache `../tm_language_model/data/hf_cache/hub`, not distributed | `src/tally/b6/prepare.py` |
| OR-groups of a hierarchical_tm run | `../hierarchical_tm/results/20261002T114726Z_check4a_word_groups/groups.tsv`, the groups of run `20261002T050235Z_e4_head_sa_s1_scoreP100`; local and not distributed | `src/tally/floor/codes.py` (the code `orgroups` of A2) |

`cpu_slot.sh` takes one of six slot locks, caps the thread pools at the number of slots it holds and starts the command; the run records give the number of slots in `run.json` (key `slots`). The KenLM binaries `lmplz`, `build_binary` and `query` are outside the repository: `src/tally/floor/kenlm.py` looks for them in `tools/kenlm/bin` under the folder that contains the repository folder (three levels above the study folder), or in the folder that the environment variable `KENLM_BIN` names. The launchers describe one workstation with six CPU slots and one GPU that the studies share through a lock; they record how the jobs were started and are not portable tools.

## Environment

Every run record gives Python 3.11.16 (`run.json`, key `python`).

- `.venv-tm` of tm_language_model is the environment in which the launchers start the Tsetlin jobs of T0, B1 and B6. Its lock file `../tm_language_model/env/tm.lock` pins tmu to commit `5605ff070a18549328028c907a9acf68e063346e` of `https://github.com/cair/tmu`, and lists nltk 3.10.3, numba 0.67.0, numpy 2.4.6, scikit-learn 1.9.1 and scipy 1.17.1. The smoke, tuning and final runs of the two T0 machines record their backend as tmu 5605ff0 (`config.json`, key `backend`). The scripts of A1 and A2 name no interpreter; the A2 smoke run records nltk 3.10.3 for its part-of-speech tagger (`metrics.json`).
- `.venv-cdllm-gpu` of clause_lm runs the logistic-regression arm of T0 on the GPU. Those runs record torch 2.6.0+cu124 and the name of the GPU (`config.json`, keys `torch` and `device`).
- The LSTM runs of C3 record torch 2.6.0+cu124 (`config.json`, key `torch`). `.venv-tm` has no torch, and the run records do not name the environment.
- `.venv-b6`, inside the study folder, runs only the data preparation of B6, which needs OpenCV 4. `src/tally/b6/README.md` gives the commands that build it. The two data manifests record cv2 4.10.0, numpy 2.4.6, pyarrow 26.0.0, scipy 1.17.1 and sklearn 1.9.1 (key `versions`).
- KenLM has no version number, and its binaries hold no version string. `kenlm.info()` in `src/tally/floor/kenlm.py` (added 2026-10-25) records what can be read: the SHA-256 of `lmplz` (`cdc75fa471da385d4dffa58836fe4068a60bfe1ebc8bdda76aed645f92f61d75`), `build_binary` (`86d0c8432007d3e63490872a3bba89efdb957b15ec96942f49c0530a02ca553f`) and `query` (`97374c0f9586b0e830090a4f0e21edbfbd03bb6d9596aeeff5a174fe8ec74c45`), the maximum order that `query` prints (6), and the commit that the README of the tools folder states for them, `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5` of `https://github.com/kpu/kenlm`. The baseline check records this from 2026-10-25 on (`config.json` of `20261025T113821Z_a1_baseline_check`, key `kenlm`); the three earlier runs do not.

## Records and hashes

`src/tally/runlog.py` writes into each run folder `config.json`, `metrics.json`, `log.txt` (a copy of every printed line), `source_hashes.json` (the SHA-256 of every `.py`, `.sh`, `.md`, `.json` and `.toml` file under `src/` and of the registration the run belongs to), `data_hashes.json` (the SHA-256 of the data files, keyed by their path relative to the study folder; until 2026-10-17 keyed by file name) and `run.json` (start, end, wall time and state). A run that ends with an exception is recorded as crashed. Since 2026-10-25 a run that gets SIGTERM or SIGHUP is recorded as killed, with the signal, and then ends by that signal; a signal that was ignored when the run started (SIGHUP under nohup) stays ignored, and SIGKILL still leaves a run recorded as running (`tests/test_runlog.py`). The run folders hold no copy of the code and no commit of the repository.

`scripts/freeze.py` writes `<name>.frozen.json` next to a registration, with the UTC time, the SHA-256 of the registration, of every file under `src/` and of extra files, and makes both files read-only. The record of T0 also pins the three test scripts. It holds 37 files under `src/`. Six of them changed after the freeze: `src/tally/t0/run.py`, whose change both amendments describe as log text only, and `src/tally/b1/flat.py`, `src/tally/b1/report.py`, `src/tally/b1/team.py`, `src/tally/b6/run.py` and `src/tally/floor/kn.py`, which the code of T0 does not import; six more files were added after the freeze. The two amendment records hold the same 43 files, and every final run of T0 records the same values (`source_hashes.json`).

The wording of six files was edited for publication on 2026-10-23: `registrations/T0.md`, which carries a publication note; the comments and docstrings of `src/tally/t0/data.py`, `src/tally/floor/kenlm.py`, `src/tally/floor/codes.py` and `src/tally/b6/config.py`; and `scripts/a2_run.py`, whose docstring and one exit message changed. The SHA-256 of the registration and of the four files under `src/` now differs from the freeze records and from the source records of the runs; `scripts/` is not pinned. The code of the four files under `src/` is unchanged, and of the code of T0 only `src/tally/t0/data.py` is among them, with a changed docstring.

The Penn Treebank runs pass the training and the validation file to the run logger, and both files are named `text.utf8`, so the `data_hashes.json` of the runs up to 2026-10-17 holds one value, that of the validation text. The runs of 2026-10-25 hold both, keyed by path. The data card gives both hashes and where they are recorded.

Nine files under `src/` were changed on 2026-10-25, after every run that the T0 records cover had finished and after the report of T0: `src/tally/runlog.py` (the killed state and the keys of the data hashes), `src/tally/floor/aa.py` (the helpers of the A/A runs), `src/tally/floor/kenlm.py` (`info()`), `src/tally/b1/report.py` (censored dual-state runs), `src/tally/b1/run.py` (the caps comment and `CAPS_FROM`), `src/tally/b1/README.md`, `src/tally/b6/launch.py` (nice 19, thread caps, CPU gate), `src/tally/b6/README.md` and `src/tally/neural/lstm.py` (the loss and the clip). They differ from all three freeze records. Of the code of the T0 report only `src/tally/runlog.py` is among them; the report run of T0 used and recorded its frozen version. `scripts/freeze.py` got a usage message, and its policy is unchanged.

## Rules of the design that the code cites

The code cites three rules by number, as "design rule N" or "rule N". The numbers are those of the numbered list of rules for every experiment in the longer design document of the project notes (not published). The design page `research/design.md` gives a shorter list of these rules without numbers, in which the third item is a different rule.

| Rule | Statement | Cited in |
|---|---|---|
| 1 | Before the run, the data, the split, the number of tokens, the metric with its unit, the comparator, the statistic and the success and kill lines are written down and hashed. Between success and kill there is no claim. | `scripts/freeze.py` |
| 2 | Five splits, cut by document: train, calibration, validation (for tuning), dev-test (for the verdict, one look) and test (once, for all arms, with its use recorded). | `src/tally/ptb.py`, `scripts/a1_baseline_check.py` |
| 3 | Everything built from text, such as the vocabulary, a category code, counts, clauses or calibration tables, comes from the training or the calibration split only, and each built file carries the hash of its source. | `src/tally/floor/codes.py`, `src/tally/t0/data.py` |

For rule 2, `src/tally/ptb.py` cuts the calibration and the dev-test split from the standard training file in blocks of 100 sentences, uses the standard validation file as the tuning split, and has no function that reads the standard test file.

Other references in the code point to the same design document. "Section 4" (in `src/tally/floor/cache.py`, `mix.py` and `stats.py` and in `scripts/a1_baseline_check.py`) is its section on the output formula of the count floor: the members are mixed by weights fitted by EM on the calibration split, a member that proposes only a few words falls back on another member, and a check runs before any number is logged. "Module A" and "member C" are lettered parts of the floor in it: A codes words into categories, and C is the context memory, which holds the cache. "The week-0 check" in `src/tally/floor/kenlm.py` belongs to the first step of the design page, which installs a reference n-gram toolkit.

## Layout

```
tally/
  README.md
  docs/
    results.md                 results page: statuses and run folders
    data.md                    data card
  registrations/
    T0.md                                         registration of T0
    T0.frozen.json                                its freeze record
    T0_amendment_coalesced_settings.md            amendment: settings of the coalesced machine
    T0_amendment_coalesced_settings.frozen.json   its freeze record
    T0_amendment_flat_settings.md                 amendment: settings of the flat machine
    T0_amendment_flat_settings.frozen.json        its freeze record
    A2.md                                         registration of A2, frozen 2026-10-25 (record A2.frozen.json)
  scripts/
    freeze.py                  freezes a registration (design rule 1)
    t0_auto.py                 T0: selection rule, amendment, freeze, final runs of one machine
    t0_final.sh                T0: final runs of both machines and the logistic-regression arm
    t0_kn_beside.py            T0: Kneser-Ney 3-gram and 5-gram beside the arms (written 2026-10-25)
    b1_launch.sh               B1: the full grid, after a registration is frozen
    a1_baseline_check.py       A1: baseline check, own Kneser-Ney and cache, and KenLM
    a1_selftest.py             A1: tests on synthetic streams
    a1_aa_noise.py             A1: A/A noise runs (train blocks dropped by seed)
    a1_aa_selftest.py          A1: tests of the A/A helpers on synthetic streams
    a2_run.py                  A2: one arm per category code
    a2_selftest.py             A2: tests on synthetic streams
    a2_recompute.py            A2: independent recompute of a run's reading (the verify step)
  src/tally/
    ptb.py                     Penn Treebank in the splits of the study
    runlog.py                  run records
    t0/                        T0: inputs, FastCoalesced, clause evaluator, read-outs, runs,
                               logistic regression, report
    b1/                        B1: planted team game, team, search arms, flat machine,
                               calibration, runs, self-checks, report
    b6/                        B6: preparation, settings, loading, engine, composites, runs,
                               timing, launcher, report
    floor/                     line A: Kneser-Ney, KenLM wrapper, cache, mixture, statistics,
                               category codes, class model, reports on codes, exchange
                               clustering, helpers for A/A noise runs
    neural/lstm.py             C3: word LSTM, exploratory
  tests/
    test_clauses.py            compiled clause evaluation against the predict path of tmu
    test_fast_coalesced.py     FastCoalesced against TMCoalescedClassifier of tmu
    test_flat_sums.py          class sums of the flat machine against tmu
    test_runlog.py             run records: data hashes by path, runs ended by a signal
    test_freeze_usage.py       freeze.py: usage message, nothing frozen
    test_launchers.py          B1 and B6 launchers with stubs: nice 19, thread caps, CPU gate
    test_b1_report.py          B1 report: censored runs and the dual-state line; caps guard
    test_lstm_loss.py          C3 LSTM: loss scaling and clip on a tiny model (needs torch)
    test_kenlm_info.py         what is recorded about the KenLM binaries
    test_a2_args.py            a2_run.py refuses every dev-test start the registration does not allow
    test_a2_recompute.py       a2_recompute.py on a synthetic run folder; its dev-test refusal
  verification/                independent recompute of the T0 verdict, post hoc checks on valid,
                               their recorded outputs and a README; the freeze checks of A2
                               (a2_freeze_*.txt, a2_devtest_env.txt), the output of its look
                               (a2_devtest_look.out), its recompute (a2_devtest_recompute.json)
                               and its three-lens check (a2_devtest_lenses.md)
    a2_dev/                    development evidence of the A2 registration (power and noise checks
                               on valid and calib, recompute outputs, drafts), with a README
```

The code also reads and writes local folders inside the study folder that the repository ignores and that are not distributed: `results/` (the 125 run folders), `data/` (the KenLM files of A1, the category codes of A2, the word-similarity sets and nltk data of A2, the prepared data of B6 and the Fashion-MNIST test file), `logs/` (the run logs of the launched jobs, which stay local) and the environment `.venv-b6`.

## Tests

The three scripts in `tests/` are plain scripts, run as their docstrings say, for example `PYTHONPATH=src python tests/test_fast_coalesced.py`, with the interpreter of `.venv-tm`, which provides tmu. They train small machines and compare the result with tmu: `tests/test_fast_coalesced.py` checks on random inputs that FastCoalesced reaches the same automaton states and weights as the library, and `tests/test_clauses.py` and `tests/test_flat_sums.py` check on real T0 inputs that the compiled clause evaluator gives the clause outputs and class sums of the library, so these two need the Penn Treebank text. The registration cites them for these equalities. All three were run on the published files on 2026-10-23, on the CPU with two threads, and passed. No output of these tests is kept in the study folder.

Eight test files were added on 2026-10-25, plain scripts as well, run from the study folder as their docstrings say: `tests/test_runlog.py`, `tests/test_freeze_usage.py`, `tests/test_launchers.py`, `tests/test_b1_report.py`, `tests/test_kenlm_info.py`, `tests/test_a2_args.py` and `tests/test_a2_recompute.py` with the interpreter of `.venv-tm`, and `tests/test_lstm_loss.py` with the interpreter of `.venv-cdllm-gpu` on the CPU (`CUDA_VISIBLE_DEVICES=`). Each test of a bug fix was run on the code before the fix, where it fails, and after it, where it passes. They write nothing into `results/`; `tests/test_launchers.py` runs the launchers with stub interpreters and reads the CPU gate once for real. On 2026-10-25 all eight passed, as did `scripts/a1_selftest.py`, `scripts/a2_selftest.py`, `scripts/a1_aa_selftest.py` and `python -m tally.b1.selfcheck`; their outputs are kept with their SHA-256 in `logs/tests_2026-10-25/` (local, not distributed).

## What a clone contains

A clone holds the code, the registration with its two amendments and their freeze records, this README, the results page and the data card. It does not hold the run folders, the run logs, the data, the virtual environments, the KenLM binaries or the run folder of hierarchical_tm that A2 reads. No number on the results page can therefore be checked from a clone alone; the run ids can be compared with the [run index](../../docs/run-index.md).
