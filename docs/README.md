# Documentation

The pages in this folder span the three studies of the programme. Each study keeps its own documents, that is its registration, its results page and its data cards, in its folder under `studies/`; the second section lists them. The entry point of the repository is the [root README](../README.md), and the list of studies with the dependencies between them is in [studies/README.md](../studies/README.md).

## Pages in this folder

| Page | Content |
|---|---|
| [results-overview.md](results-overview.md) | For each study, the statuses of what was registered (counts per hypothesis family for tm_language_model, one row per check or id for the other two studies) and the headline measurements with their run ids, copied from the three results pages. |
| [known-limitations.md](known-limitations.md) | What limits the use of the repository and of its results: what a clone can and cannot reproduce, how the published files differ from the files that produced the runs, what is known about the data, the scientific limits of each study, and the flagged runs. |
| [data.md](data.md) | The datasets of the three studies: source, recorded version, the script that obtains or builds each one, where the code expects it, the record that pins its hashes, and what is known about its licence. No dataset is distributed. |
| [references.md](references.md) | The bibliography of the programme. |
| [upstream-repositories.md](upstream-repositories.md) | The third-party repositories and packages that the studies use, with the commit that is recorded for each. |
| [run-index.md](run-index.md) | One row per run folder of the three studies: run id, study, experiment, date, status, exit status, wall time, size and flag. The flags mark the runs whose numbers are not used as results. The run folders themselves are not distributed. |
| [hash-table.md](hash-table.md) | The files that a config or a record pins by SHA-256, with the hash as registered and the hash as published. |

## Documents of the studies

tm_language_model keeps its documents in `studies/tm_language_model/docs/`. The [index](../studies/tm_language_model/docs/00-README.md) of that folder gives a reading order and explains the evidence labels. The main entries are the [pre-registration](../studies/tm_language_model/docs/13-preregistration.md) with its amendments, the [results page](../studies/tm_language_model/docs/results.md), the [log of design changes](../studies/tm_language_model/docs/12-design-changes.md) and the [research protocol](../studies/tm_language_model/docs/research-protocol.md). The [study README](../studies/tm_language_model/README.md) describes the code and the setup.

hierarchical_tm keeps its pass rules, [gates.md](../studies/hierarchical_tm/gates.md), and its hash record, [frozen_settings_hashes.json](../studies/hierarchical_tm/frozen_settings_hashes.json), at the study root, and the other documents in `studies/hierarchical_tm/docs/`: the [registrations](../studies/hierarchical_tm/docs/registrations.md), the [results page](../studies/hierarchical_tm/docs/results.md), and the data cards of the [Wikipedia pairs](../studies/hierarchical_tm/docs/data_card_wikipedia_pairs.md) and of the [news-topic data](../studies/hierarchical_tm/docs/data_card_agnews.md). The [study README](../studies/hierarchical_tm/README.md) describes the code, the inputs that are not distributed and the claims that are not made.

clause_lm keeps its documents in `studies/clause_lm/docs/`: the [registration of M5 step 1](../studies/clause_lm/docs/prereg_M5_step1.md) and the [results page](../studies/clause_lm/docs/results.md). The [study README](../studies/clause_lm/README.md) describes the milestones, the data and the upstream dependency.
