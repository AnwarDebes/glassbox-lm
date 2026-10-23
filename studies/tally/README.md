# tally: a count floor and a test bench for Tsetlin learning

This study holds the code and the run records of the first steps of the design described on the page [research/design.md](../../research/design.md), whose working name is Tally. The design has two parts that are kept apart: a count floor, a language model built from counts with no Tsetlin Machine in it, and a test bench, places in that model where a Tsetlin Machine is compared on the same bits with a twin that only counts and with standard learners. The study covers three lines of that design: line T (T0, B1 and B6, Tsetlin tests that share no code with the language model), line A (steps A1 and A2 of the count floor) and line C (C3, the neural comparison as a curve over training budgets). Its run folders date from 2026-10-15 to 2026-10-17.

One test is registered, T0, with two amendments that fix the settings of its two machines. All its final runs have finished, and its verdict has not been drawn. The other lines have code and smoke, timing, baseline or exploratory runs, and no registration in the folder.

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

Running. T0 was frozen on 2026-10-15 at 07:27:25 UTC (`registrations/T0.frozen.json`). Its tuning rule fixed the settings of the two machines in two amendments: s = 20 and 9 epochs for the coalesced machine, frozen at 18:24:29 UTC, and s = 5 and 1 epoch for the flat machine, frozen at 21:14:27 UTC. The final runs of seeds 1 to 10 of both machines and the ten runs of the logistic-regression arm have finished. The verdict, which `src/tally/t0/report.py` computes from these runs, has not been drawn: the only run of the report script is a smoke test on the validation split of a smoke run.

B1 and B6 have code, self-checks and smoke or timing runs, and no registration. `src/tally/b1/README.md` calls a registration in `registrations/` the binding text, but no B1 registration is in that folder. `src/tally/b6/README.md` runs its grid after a registration in the same folder is frozen, and there is no B6 registration there either. A1 has three runs of its baseline check and A2 one smoke run, and neither has a registration in the folder. C3 has 20 LSTM runs on the validation split, two sizes with seeds 1 to 10. No code of the study opens the test file of Penn Treebank.

## How the study is run

Every command runs from the study folder with `PYTHONPATH=src`, except the scripts in `scripts/`, which set the import path themselves. Every run writes a run folder `results/<UTC time>_<name>/` through `src/tally/runlog.py`.

T0:

- Tuning, seed 1, one run per value of s: `python -m tally.t0.run --mode tune --seed 1 --s 10 --epochs 10 --patience 3` for the coalesced machine, and the same with `--machine flat --clauses 100` for the flat machine. The tuning runs record these arguments (`run.json`, key `argv`).
- `python scripts/t0_auto.py coalesced|flat` reads the run records of the three tuning runs of one machine every 300 s until none of them is recorded as running, applies the registered selection rule, writes the dated amendment `registrations/T0_amendment_<machine>_settings.md`, freezes it with `scripts/freeze.py` and starts the ten final runs, each in one CPU slot at nice 10. For the coalesced machine, each final run is followed by its logistic-regression run under the GPU lock. Both amendments name `scripts/t0_auto.py` as the script that wrote them.
- `scripts/t0_final.sh S_COAL E_COAL S_FLAT E_FLAT` starts the same final runs with the settings given on the command line.
- The verdict: `python -m tally.t0.report <coalesced final runs> --logreg <logistic-regression runs> --flat <flat final runs>`. The script reads the per-position bits of the runs (`bits.npz`), scores the dev-test split unless `--split` names another, and writes its result to `metrics.json` of a run folder of its own.

B1: the commands are in `src/tally/b1/README.md`. The full grid is `scripts/b1_launch.sh registrations/<file>.md [JOBS]`, which stops when the freeze record of the registration it is given is missing.

B6: the commands are in `src/tally/b6/README.md`: the data preparation in the environment `.venv-b6`, smoke, timing and tuning runs, the grid through `python -m tally.b6.launch` and the verdict through `python -m tally.b6.report`.

A1: `python scripts/a1_selftest.py` tests the Kneser-Ney model, the cache, the duplicate mask, EM and the block bootstrap on small synthetic streams; `python scripts/a1_baseline_check.py` runs the baseline check and needs the KenLM binaries.

A2: `python scripts/a2_selftest.py` tests the A2 code on small synthetic streams. `python scripts/a2_run.py --split valid` is the development run and `--split valid --smoke` the same with small sizes; `--split devtest` scores the dev-test split, which the script does only together with `--final`.

C3: `python -m tally.neural.lstm --size medium|large --seed S` needs torch and a CUDA device.

A registration is frozen with `python scripts/freeze.py registrations/<file>.md [extra files...]`.

## What the study needs from other studies

The study is not self-contained. It reaches scripts, environments, data and one run folder of other studies by relative paths and expects them as sibling folders under `studies/`.

| What is used | Where it is | File of tally that makes the link |
|---|---|---|
| CPU slot, one slot per job | `../tm_language_model/scripts/cpu_slot.sh` | `scripts/t0_auto.py`, `scripts/t0_final.sh`, `scripts/b1_launch.sh`, `src/tally/b6/launch.py`; the commands in `src/tally/b1/README.md` and `src/tally/b6/README.md` |
| GPU lock, mode `exclusive` | `../tm_language_model/scripts/gpu_run.sh` | `scripts/t0_auto.py`, `scripts/t0_final.sh` (the logistic-regression arm of T0) |
| Virtual environment `.venv-tm`, which provides tmu at commit 5605ff0 | built inside the tm_language_model folder, frozen in `../tm_language_model/env/tm.lock` | `scripts/t0_auto.py` (also for `scripts/freeze.py`), `scripts/t0_final.sh`, `scripts/b1_launch.sh`, `src/tally/b6/launch.py` |
| Virtual environment `.venv-cdllm-gpu` (torch on the GPU) | built inside the clause_lm folder, described in its README | `scripts/t0_auto.py`, `scripts/t0_final.sh` |
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
- The KenLM version is not recorded.

## Records and hashes

`src/tally/runlog.py` writes into each run folder `config.json`, `metrics.json`, `log.txt` (a copy of every printed line), `source_hashes.json` (the SHA-256 of every `.py`, `.sh`, `.md`, `.json` and `.toml` file under `src/` and of the registration the run belongs to), `data_hashes.json` (the SHA-256 of the data files, keyed by file name) and `run.json` (start, end, wall time and state). A run that ends with an exception is recorded as crashed. The run folders hold no copy of the code and no commit of the repository.

`scripts/freeze.py` writes `<name>.frozen.json` next to a registration, with the UTC time, the SHA-256 of the registration, of every file under `src/` and of extra files, and makes both files read-only. The record of T0 also pins the three test scripts. It holds 37 files under `src/`. Six of them changed after the freeze: `src/tally/t0/run.py`, whose change both amendments describe as log text only, and `src/tally/b1/flat.py`, `src/tally/b1/report.py`, `src/tally/b1/team.py`, `src/tally/b6/run.py` and `src/tally/floor/kn.py`, which the code of T0 does not import; six more files were added after the freeze. The two amendment records hold the same 43 files, and every final run of T0 records the same values (`source_hashes.json`).

The wording of six files was edited for publication on 2026-10-23: `registrations/T0.md`, which carries a publication note; the comments and docstrings of `src/tally/t0/data.py`, `src/tally/floor/kenlm.py`, `src/tally/floor/codes.py` and `src/tally/b6/config.py`; and `scripts/a2_run.py`, whose docstring and one exit message changed. The SHA-256 of the registration and of the four files under `src/` now differs from the freeze records and from the source records of the runs; `scripts/` is not pinned. The code of the four files under `src/` is unchanged, and of the code of T0 only `src/tally/t0/data.py` is among them, with a changed docstring.

The Penn Treebank runs pass the training and the validation file to the run logger, and both files are named `text.utf8`, so their `data_hashes.json` holds one value, that of the validation text. The data card gives both hashes and where they are recorded.

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
  scripts/
    freeze.py                  freezes a registration (design rule 1)
    t0_auto.py                 T0: selection rule, amendment, freeze, final runs of one machine
    t0_final.sh                T0: final runs of both machines and the logistic-regression arm
    b1_launch.sh               B1: the full grid, after a registration is frozen
    a1_baseline_check.py       A1: baseline check, own Kneser-Ney and cache, and KenLM
    a1_selftest.py             A1: tests on synthetic streams
    a2_run.py                  A2: one arm per category code
    a2_selftest.py             A2: tests on synthetic streams
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
```

The code also reads and writes local folders inside the study folder that the repository ignores and that are not distributed: `results/` (the 91 run folders), `data/` (the KenLM files of A1, the category codes of A2, the word-similarity sets and nltk data of A2, the prepared data of B6 and the Fashion-MNIST test file), `logs/` (the run logs of the launched jobs, which stay local) and the environment `.venv-b6`.

## Tests

The three scripts in `tests/` are plain scripts, run as their docstrings say, for example `PYTHONPATH=src python tests/test_fast_coalesced.py`, with the interpreter of `.venv-tm`, which provides tmu. They train small machines and compare the result with tmu: `tests/test_fast_coalesced.py` checks on random inputs that FastCoalesced reaches the same automaton states and weights as the library, and `tests/test_clauses.py` and `tests/test_flat_sums.py` check on real T0 inputs that the compiled clause evaluator gives the clause outputs and class sums of the library, so these two need the Penn Treebank text. The registration cites them for these equalities. All three were run on the published files on 2026-10-23, on the CPU with two threads, and passed. No output of these tests is kept in the study folder.

## What a clone contains

A clone holds the code, the registration with its two amendments and their freeze records, this README, the results page and the data card. It does not hold the run folders, the run logs, the data, the virtual environments, the KenLM binaries or the run folder of hierarchical_tm that A2 reads. No number on the results page can therefore be checked from a clone alone; the run ids can be compared with the [run index](../../docs/run-index.md).
