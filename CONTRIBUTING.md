# Contributing

This page describes how work is added to the repository and which conventions the three existing studies follow. The repository is a research record: each study keeps its code, its registration and its results page, while the run folders and datasets behind the results stay local. The conventions below keep a number, a registered rule and a pinned file traceable when something is added or changed.

## Studies

A new study is one folder `studies/<short_snake_case_name>/`. It holds a `README.md` with the question, the status and how the study is run, the folders `src/` and `tests/`, and a folder `docs/` with the registration of the study, its results page and its data cards. The study gets one row in [studies/README.md](studies/README.md) and one in the root [README](README.md), with a status that is one of planned, running, finished and archived. Code is shared between studies only when a second study needs it; until then a new study reaches no code, data or queue file of another study by path. The section "Adding a study" of studies/README.md has the full description and names the places where the existing studies differ from it.

## Runs

Every run is started through the run logger of its study and writes one run folder, `results/<UTC timestamp>_<name>/`, in the folder of the study. In tm_language_model the logger is `studies/tm_language_model/scripts/run_logged.py` for a command and `tmlm.common.runlog.RunLog` inside Python; rule 5 of the [research protocol](studies/tm_language_model/docs/research-protocol.md) lists what the run folder holds. hierarchical_tm starts its jobs through the same logger, and its launchers, except the one of the probe scripts in `studies/hierarchical_tm/src/feas_slide11/`, pass `--results-dir`, so that the run folders are written to its own `results/`. clause_lm has its own logger, `studies/clause_lm/src/runlog.py`, which writes the config, the metrics, the log and the hashes of the data files and copies the calling script into the run folder.

Run folders are not committed. The root `.gitignore` ignores `/studies/*/results` and `/studies/*/data`, and with them `/studies/*/work`, `/studies/*/references`, `/studies/*/third_party/ref` and every `.venv*`. A run is cited by its run id, which is the name of its folder, exactly as written, and a run folder is not renamed. In tm_language_model data and run folders are kept, and deleting one needs a recorded decision (protocol rule 9). The [run index](docs/run-index.md) lists the run folders of the three studies with date, exit status and flag. A run that is flagged there (never cite, stale, killed, smoke, biased exploratory) may be named as a fact; no number is taken from it.

## Numbers in documents

A measurement in a document carries the run id of the run that produced it. Its source is the metrics or report file of that run, not text that was written about the run. In tm_language_model a measured number is followed by a label of the form `[measured: results/<run id>/<file>]`, with the path relative to the study folder; the labels measured, derived and unverified are defined in the [docs index](studies/tm_language_model/docs/00-README.md) of the study. `studies/tm_language_model/scripts/check_numbers_have_logs.py` looks for the token `[measured` in the Markdown files under `docs/` of the study, and in a `CHANGELOG.md` in the study folder if there is one, and requires a path under `results/` that exists, on the same line or in the same paragraph. The check therefore works only when the run folders are present. `make lint` runs it and `make smoke` starts with `make lint`, so both need the run folders and report every label without them. The pages at the repository root are not scanned. The results pages of hierarchical_tm and clause_lm name the run and the file next to each number, without this label.

A results page lists every id of the registration of its study once, with one status: met, not met, not judged or not run, and triggered or not triggered for a kill criterion. A qualifier that belongs to a verdict (on seed 1 only, borderline, exploratory) stays next to the status wherever the result is repeated. A choice that was made after results were known is labelled post hoc, and a negative result is reported like any other.

## Registrations

The registrations are frozen records: the [pre-registration](studies/tm_language_model/docs/13-preregistration.md) of tm_language_model, [gates.md](studies/hierarchical_tm/gates.md) and the [registrations page](studies/hierarchical_tm/docs/registrations.md) of hierarchical_tm, and the [registration of M5 step 1](studies/clause_lm/docs/prereg_M5_step1.md) of clause_lm. A change to a registration is an amendment, appended with its date and its reason; the registered text is not edited. The pre-registration of tm_language_model carries the amendments A1 to A10 in this form, gates.md has dated addenda, and the registration of M5 step 1 has one amendment. An amendment that was written after the result it concerns says so, and the results page labels it post hoc. The run logger of tm_language_model writes the SHA-256 of the pre-registration into every run record, which ties a run to the revision that was in force when it started; an amendment therefore changes the value that later runs record. The published copies of the registrations carry a publication note that says how their text relates to the registered revisions.

## Pinned and vendored files

Some files are pinned by a SHA-256 that a record holds. `studies/hierarchical_tm/frozen_settings_hashes.json` pins 21 scripts and 17 data files of that study, the run records of clause_lm hold the hash of the loader `studies/clause_lm/src/clause_driven_llm/cdllm_data__init__.py`, and `studies/tm_language_model/src/tmlm/kernels/graphtm_patched/kernels.py` is a byte-identical copy of the stock kernel file with a recorded hash. The [hash table](docs/hash-table.md) lists the pinned files with their SHA-256 as registered and as published. A pinned file is changed only together with its pin, in the same change, and the hash table follows. A pin whose target is not in the repository, such as a data file or a file in a run folder, keeps its registered value. One pin of the hash record of hierarchical_tm, that of `studies/hierarchical_tm/src/text_driver.py`, has disagreed with its file since 2 Oct 2026; it is published as registered and is not repaired, and the [README](studies/hierarchical_tm/README.md) of the study explains it.

The `.editorconfig` at the root sets UTF-8, line feeds, a final newline and the trimming of trailing whitespace, the last of these not in Markdown and patch files. For the vendored package, for the stock kernel copy, for the JSON files under `studies/tm_language_model/experiments/`, for all of `studies/hierarchical_tm/` and for `studies/clause_lm/src/clause_driven_llm/` it switches the final newline and the trimming off, so that an editor leaves vendored and pinned files byte for byte.

The vendored package `studies/tm_language_model/third_party/GraphTsetlinMachine` is the upstream package at the commit recorded in its `UPSTREAM_COMMIT` file. It is the stock reference of tm_language_model and is not edited: it is installed in editable mode, so an edit there would change the reference itself. Kernel patches belong in `studies/tm_language_model/src/tmlm/kernels/graphtm_patched/`.

## Layout that code depends on

In tm_language_model the folder names `src`, `experiments`, `scripts`, `tests` and `env` enter the run records: the run logger hashes the paths and contents of these five folders into the `source_tree_hash` of every run. Code and documents cite the documents of that study by number (document 13 is the pre-registration) and their items by id (a hypothesis id, an item of the log of design changes, a protocol rule), so numbers and ids are kept when a document changes. `studies/tm_language_model/tests/test_configs.py` requires a config in `studies/tm_language_model/experiments/configs` for every experiment heading of [document 05](studies/tm_language_model/docs/05-experiments.md). The make targets of the study are run from its folder, or from the root with `make -C studies/tm_language_model <target>`. hierarchical_tm and clause_lm find tm_language_model as a sibling folder by relative path, so the study folders stay next to each other under `studies/`, and the keys of the hash record of hierarchical_tm are paths relative to its folder.

## Text

Text in the repository is ASCII, apart from math symbols in formulas and letters with diacritics in names. Dates are written as 2026-09-22 or 22 Sep 2026. A clock time is given with its time zone, and only where the order against a run matters, because run ids are UTC. Inside a study a path is written relative to the study folder; a page at the root writes paths from the repository root.

## Open points

Open points that a contributor meets, one line each:

- hierarchical_tm and clause_lm are folders of scripts and not installable packages; only tm_language_model has a `pyproject.toml`, for its package `tmlm`.
- The run logger, the memory guard, the GPU lock, the environment script and the virtual environments that hierarchical_tm and clause_lm use still live in tm_language_model; moving them to a shared place needs a test run.
- The relative paths between the studies replaced absolute ones for publication and have not been exercised by a run.
- The experiment scripts of tm_language_model sit in one flat folder; they have not been regrouped, and helpers that they share have not been moved into the package.
- Marking the tests that need a GPU is an open refactor; `studies/tm_language_model/pyproject.toml` declares the markers `gpu` and `slow`.
- Run records carry no commit id: the run logger of tm_language_model looks for `.git` in the study folder, where this layout has none, so `source_tree_hash` identifies the code of a run.
- The helper scripts in `studies/tm_language_model/scripts/` start with `#!/usr/bin/env python3`, so a launcher that starts them by name runs them under the first `python3` on `PATH`.
- `make lint`, `make smoke` and two tests of tm_language_model depend on the run folders, which a clone does not have.
- The compiled engine binaries are not included and are rebuilt when a source file is newer than the binary; the parity tests of the three engines have not been run on the published sources.
- The queue scripts of hierarchical_tm use path variables without quotes, so a clone path that contains a space breaks them.
- New run records of clause_lm name data files by the layout of this repository, while the existing records use the folder layout at the time of the runs.
- As-run configs that were edited for publication no longer have the `config_sha256` that their runs recorded.
- Two launch chains that experiment scripts of tm_language_model cite, `chain_config_arms.sh` and `chain_controls.sh`, were not retained.
- A few comments are stale and were left as written, for example line 2 of `studies/tm_language_model/Makefile`, which says that nothing is installed although the target `bootstrap` runs the installer.

## Licence

The repository is released under CC BY-NC 4.0 ([LICENSE](LICENSE)). Contributions are accepted under the same licence.
