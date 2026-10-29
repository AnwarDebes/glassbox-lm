# tm_language_model: a token-level Tsetlin Machine language model

This study asks whether a token-level Tsetlin Machine language model, trained with Tsetlin feedback on next-token prediction, can reach the accuracy of an n-gram baseline at useful speed, and what its clauses represent. The package `tmlm` in `src/` holds the engines, the baselines, the read-outs and the tooling that logs every run; `experiments/` holds the scripts that produced the runs.

Status: archived. The study ran from 19 Sep to 1 Oct 2026 and was archived on 2026-10-01 with nothing cut (docs/12 F4 and F23). In the finish pass of 2026-10-25 a few rows prepared before the archive were run under amendments A11 and A12 and job requests of that pass (H2.3 and H2.4 re-checked, H3.5 at 200 steps, H8.3 and H8.4 on the test split, H8.8 part (c), H8.9, H9.6); the results page gives each, and nothing else of the study changed. Code, registration and design documents are published as they stood at the end of the study, apart from the edits for publication that the page of [known limitations](../../docs/known-limitations.md) describes. This README, the index, the research protocol, the log of design changes and the results page were written for the publication from the records of the study. The run folders and the datasets are not part of the repository.

## What the study found

The results page, [docs/results.md](docs/results.md), lists every registered hypothesis, gate and kill criterion with its status, its run ids and its measured values. This section gives the outcome in words only.

The reproducibility hypotheses of the pre-registration (H2.1 to H2.4) were met at the scale tested; H2.4 was re-checked on 2026-10-25 with the checker that binds every decision to its generation run, and it holds. Of the 91 registered hypotheses, 15 are met, 14 are not met, 5 are not judged and 57 were never run; the results page gives the status and the runs of each. Of the 15, H8.11 was read on 2026-10-25 from a complete run file for which no verdict had been recorded at the time, and H8.3 and H8.4 were read on the test split on the same day, once: the count-vote core, which holds no Tsetlin-learned clause, scores 1.2478 bits per byte, 0.124 below KN-5 and 0.105 below KN-5 plus a document unigram cache, so the credit goes to counting and in-context copying, not to Tsetlin learning. The main accuracy goal was missed: hypothesis H1.1 of the pre-registration ([docs/13-preregistration.md](docs/13-preregistration.md)), the causal model against an in-house Kneser-Ney 5-gram on identical tokens, is not met on seed 1, and the same run, `results/20260923T042350Z_CP_anchor`, triggered the kill criterion KB7 of docs/08. The pre-registration has ten dated amendments (A1 to A10), several of which add hypothesis families; after the miss these were, among others, the count-vote read-out of amendment A3 (family H8) and the self-supervised family of amendment A7 (family H17). The confirmatory tests of the self-supervised family were negative: gate G-R1 was read as "fail and below bag of words" on the confirmatory report `results/20260930T163619Z_ssl_r1cf_report_confirmatory` (docs/12 F29), H17.5 is not met (`results/20260930T203322Z_ssl_r2_hao_report_confirmatory`), and H17.10 is not met (`results/20260930T203322Z_ssl_r4_report_confirmatory`). Of the eight ids of the family that were run, one, H17.12, is met; the results page lists all twelve.

The log of design changes, [docs/12-design-changes.md](docs/12-design-changes.md), records how the design moved between these steps and why.

## Layout

| Path | Content |
|---|---|
| `src/tmlm/` | the package (version in `tmlm.__version__`), see below |
| `experiments/` | 62 experiment scripts in one flat folder and their configs (three of them, `cvtm_test_read.py`, `cvtm_test_read_verify.py` and `h2_4_recheck.py`, added on 2026-10-25 and not yet run as registered runs); [experiments/README.md](experiments/README.md) |
| `tests/` | the pytest suite: tooling, engine parity, configs, experiment helpers |
| `scripts/` | `run_logged.py` (runs a command as a logged run), `gpu_run.sh` and `cpu_slot.sh` (GPU lock and CPU slots), `mem_guard.py`, `env_check.py`, `check_numbers_have_logs.py`, `lib_locks.sh` |
| `env/` | `env.sh`, `bootstrap.sh`, `htm_install.sh` and the lock files `tm.lock`, `dl.lock`, `htm.lock` |
| `launch/` | the launch chains of the overnight runs; [launch/README.md](launch/README.md) |
| `docs/` | the docs index (00), the design documents 01 to 10, the engine specification (11), the log of design changes (12), the pre-registration (13), the research protocol, the results page, `docs/notes/` with the data hash records, and `docs/drafts/` with two draft amendments of 2026-10-25 that are not frozen; reading order in [docs/00-README.md](docs/00-README.md) |
| `third_party/` | the vendored GraphTsetlinMachine package; `third_party/README.md` |
| `Makefile`, `pyproject.toml` | targets and package metadata; `pyproject.toml` reads this file as the package description |

The subpackages of `src/tmlm/`:

| Subpackage | Content |
|---|---|
| `common` | run logging, seeding, hashing, locks, thread caps, and the model interface |
| `data` | article and token stores, byte-level BPE, cluster codes, frozen evaluation sets with the test-access counter, paragraph pairs, the contamination report, synthetic corpora |
| `engine` | the sparse event-driven engine of docs/11 in three implementations: the Python oracle (`oracle/`), the C++ engine (`cpu/`) and the CUDA engine (`cuda/`); the last two share the header `core/tm_core.h` |
| `kernels` | `graphtm_patched/kernels.py`, a byte-identical copy of the stock GraphTsetlinMachine kernel file with its recorded hash; the patches planned for milestone M5 are not in it |
| `eval` | the in-house baselines in `eval/baselines` (n-gram and Kneser-Ney models, count probe, CBOW-kNN, a small transformer, a BERT reference); the evaluators listed as planned in `eval/__init__.py` were not written |
| `decode` | clause bank export, calibration, the any-order and masked-diffusion samplers, the elimination decoder, incremental re-evaluation, per-token traces |
| `certify` | receipts for generated tokens and a standalone checker that imports nothing from the package |
| `cvtm` | the count-vote read-out of docs/11 section 11: Kneser-Ney count fit, suffix clauses, document clauses, the mixture, the masked variant, the copy selector |
| `ssl` | the self-supervised family: pretext arms, document features, few-label probes, controls, transfer, word concepts, the H-AO world, image patch tokens |
| `featurizer`, `graphs`, `heads` | placeholders: each holds a docstring that lists planned modules, and no code |

## Setup

All commands run from this folder. From the repository root the same targets are reached with `make -C studies/tm_language_model <target>`.

```bash
env/bootstrap.sh       # or: make bootstrap. Builds .venv-tm and .venv-dl, refuses when either exists
source env/env.sh      # .venv-tm/bin first on PATH, CUDA paths, thread caps, PYTHONHASHSEED
make env-check         # machine limits, GPU, imports, API signatures, written to results/<timestamp>_envcheck/
make test              # pytest -q -m "not gpu and not slow" tests, every CUDA device hidden
make test-slow         # the three slow tests that need no GPU, every CUDA device hidden (about 3 min)
make test-gpu          # the gpu-marked tests under the shared GPU lock (scripts/gpu_run.sh)
make lint              # every number labelled as measured in docs/ cites a log under results/
make smoke             # lint, test, then scripts/smoke_e2e.py (end to end on tiny data)
make configs-check     # tests/test_configs.py and tests/test_tm_core_constants.py
```

`env/bootstrap.sh` works in userspace through `uv` (it installs the standalone `uv` binary when none is found) and needs `nvcc` and `g++`. It creates both environments with Python 3.11, builds pycuda from source, installs tmu and the hierarchical package from GitHub at pinned commits, installs the vendored GraphTsetlinMachine and this package in editable mode, imports thirteen of the installed packages once, and asserts that the GPU has compute capability 7.0 (a Tesla V100). `--skip-dl` builds `.venv-tm` only; `--fresh` removes both environments first, after a typed confirmation. The script writes its log and a manifest to `results/_bootstrap/<UTC stamp>_bootstrap/` and ends by overwriting `env/tm.lock` and, unless `--skip-dl` is given, `env/dl.lock` with the new freeze.

There are three virtual environments, all created inside this folder and ignored by git. `.venv-tm` and `.venv-dl` are never used in one process (docs/12 A3).

| Environment | Built by | Lock file | Content |
|---|---|---|---|
| `.venv-tm` | `env/bootstrap.sh` | `env/tm.lock` | the Tsetlin stack; its interpreter is the one the Makefile targets use |
| `.venv-dl` | `env/bootstrap.sh` | `env/dl.lock` | the neural baselines (torch, transformers, spaCy); versions in the lock file |
| `.venv-htm` | `env/htm_install.sh` | `env/htm.lock` | the hierarchical package at a later upstream commit, for the exploratory pilot of docs/12 F27 |

`env/tm.lock` records, among others: numpy 2.4.6, scipy 1.17.1, pycuda 2025.1.3, cupy-cuda12x 14.2.0, numba 0.67.0, scikit-learn 1.9.1, tokenizers 0.23.2, datasets 5.0.1, nltk 3.10.3, z3-solver 5.1.0.0, python-sat 1.8.dev17, codecarbon 3.3.1, nvidia-ml-py 13.610.43, pytest 9.1.1, tmu at git commit 5605ff070a18549328028c907a9acf68e063346e, and PyHierarchicalTsetlinMachineCUDA at git commit c85ca4e70af90511d6b4f8e8eedccd23a7d7d0ec. `env/htm.lock` records numpy 2.4.6, scipy 1.17.1, pycuda 2025.1.3, scikit-learn 1.9.1, scikit-image 0.26.0, networkx 3.6.1, and PyHierarchicalTsetlinMachineCUDA at git commit 5b4a223075c1e621577d508bc7c3c061230e29d5. The editable installs are written as paths relative to this folder: `-e ./third_party/GraphTsetlinMachine` and `-e .` in `env/tm.lock`, and `-e .` in `env/htm.lock`.

`env/htm_install.sh` is written to be started through the run logger (its header gives the command), refuses to run when `.venv-htm` exists, and writes `env/htm.lock`.

### The vendored GraphTsetlinMachine

`third_party/GraphTsetlinMachine` is the upstream package at commit c6129e1466d87ebafe411c8a334276ad53c10763 (MIT, licence file kept), with its files unmodified. Three upstream items are not included: `.devcontainer/`, `.gitignore` and `.gitattributes`. The bootstrap installs the package in editable mode from `./third_party/GraphTsetlinMachine`. The package is the stock reference of the study and is not edited; the copy that the study may patch is `src/tmlm/kernels/graphtm_patched/kernels.py`.

## What a clone does and does not contain

A clone contains the code, the configs, the registration and the documents. It contains no run folder and no dataset: `results/`, `data/`, the virtual environments and the reference clones under `third_party/ref` are ignored by git and are not distributed. Every result on the results page names its run id, and the [run index](../../docs/run-index.md) at the repository root lists each run folder with its date, exit status and flag, but the folders themselves cannot be inspected from a clone. Reproducing a number means building the data with `experiments/m2_build_data.py` (sources and versions are on the [data page](../../docs/data.md); the script reads the Hugging Face cache under `data/hf_cache` in offline mode unless told otherwise) and running the experiment again on comparable hardware. Nothing in this repository states that the tests pass in a clone or that a rerun returns a recorded value.

The following facts of the code matter for anyone who tries:

- `make lint` runs `scripts/check_numbers_have_logs.py`. The check accepts a label of the form `[measured: results/<run id>/]` in `docs/` only when the cited path exists under `results/`. Without the run folders it reports every such label, so `make lint` and `make smoke` fail until the run folders are present. `tests/test_tooling_checks.py` runs the same check on the real tree, and `tests/test_ssl_r4_controls.py` opens `run.json` in run folders that a config names, so `make test` depends on the run folders in these two places as well.
- The compiled engine binaries are not included. The CPU engine is built with `g++` and the CUDA kernels with `nvcc` for `sm_70` when they are first used, and again whenever a source file is newer than the binary.
- The code encodes the workstation of the recorded runs: one Tesla V100 (the bootstrap assertion and the `sm_70` build), a quota of 6 CPUs (the thread caps of `env/env.sh` and the six slots of `scripts/cpu_slot.sh`), a fixed location of the CUDA toolkit in `env/env.sh`, and the control-group memory files that `scripts/mem_guard.py` reads. docs/12 A1 and A2 describe the machine.
- Run records carry no commit id. No commit was made while the study ran (docs/12 A8), and the run logger looks for `.git` in the study folder, next to `src/`, where this layout has none. A run started from a clone is therefore identified by its `source_tree_hash` only, a content hash of `src`, `experiments`, `scripts`, `tests` and `env`.
- The helper scripts in `scripts/` and `experiments/p0_tmu_ae_repro.py` start with `#!/usr/bin/env python3`. Started by name, as in the recorded launch commands (`scripts/run_logged.py ...`, `scripts/mem_guard.py ...`), they run under whichever `python3` comes first on `PATH`. On the machine of the study no `python3` is on `PATH`, so until 2026-10-25 every such start failed with exit status 127. Since then `env/env.sh` puts `.venv-tm/bin` first on `PATH` once that environment exists, and `scripts/cpu_slot.sh` sources `env/env.sh`, so the recorded launch lines work after `source env/env.sh` (`tests/test_tooling_locks.py`). The studies `hierarchical_tm` and `clause_lm`, which start these helpers too, get the same `python3` when they source this file. Calling a helper as `.venv-tm/bin/python scripts/<name>.py` does not depend on `PATH` at all.
- `make test` deselects the tests marked `gpu` or `slow` and hides every CUDA device; `tests/conftest.py` also hides the devices unless the GPU lock of `scripts/gpu_run.sh` is held, and skips the gpu-marked tests then. `make test-slow` runs the three slow tests that need no GPU: the data pipeline end to end on synthetic rows, and the two KB1 parity tests against tmu. The GPU tests run through `make test-gpu`. Until 2026-10-25, 31 test functions that need a CUDA device carried no marker (`tests/test_tooling_gpu_isolation.py` scans for them).
- Known skips. On the workstation of the study `make test` reports 14 skipped items, all expected (2026-10-25):
  - the 5 test modules that need a CUDA device, whose device probe runs at import;
  - 8 tests that need torch, which `.venv-tm` does not hold (`tests/test_baselines_smoke.py` three times, `tests/test_cvtm_gx_transformer.py` five times); their docstrings name `.venv-dl/bin/python`, this study has no `.venv-dl`, and nothing is installed into a study environment;
  - 1 opt-in test that reads dataset metadata over the network (`TMLM_HF_METADATA_TEST=1`).

  Until 2026-10-25 the list held 3 more, the tests of `tests/test_ssl_r2_hao_diagnostics.py` that need the package sources at commits c85ca4e7 and 5b4a2230 and a reference clone under `third_party/ref`, which is not distributed. They are gpu-marked since then, so `make test` deselects them and `make test-gpu` runs them; without the clone they skip.
- Fifteen configs in `experiments/configs` name experiment scripts that were never written (E2, E3_trackA, E4 to E10, H1, H3, TB, TK, s1, s2). They stay because `tests/test_configs.py` requires a config for every experiment heading of docs/05.

The page of [known limitations](../../docs/known-limitations.md) at the repository root carries the limits that concern all studies.

## Relation to the other studies

The studies `hierarchical_tm` and `clause_lm` are not self-contained. `hierarchical_tm` starts its jobs through this study's run logger, environment file, GPU lock and virtual environments; `clause_lm` uses this study's environment file, memory guard and GPU lock for its GPU steps and logs its runs with a logger of its own. Both read data from this study's `data/` folder; in their code this folder is the sibling `../tm_language_model`. The run records of `hierarchical_tm` carry the pre-registration hash of this study for that reason (see the publication note at the top of docs/13-preregistration.md). The run index also lists the 27 exploratory `feas_slide11` probe runs of 2026-10-01 under this study, because they were written to its `results/` folder; they prepared the `hierarchical_tm` study, used pair data with a sampling bias, and are not used as results. The list of studies is in [studies/README.md](../README.md).
