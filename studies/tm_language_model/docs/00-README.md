# 00 Index of the design and record documents

This folder holds the documents of the tm_language_model study: the design documents, the log of design changes, the pre-registration, the research protocol, the results page and the data hash records. The study README ([../README.md](../README.md)) gives the status of the study and its headline results. This page says what each document is, in which order the documents are best read, and what the evidence labels in them mean.

Documents 00 to 10 are the design documents of the initial design. Documents 01 to 10 record the design as it stood on 2026-09-22. Apart from additions that carry their own date, such as the section of docs/09 added on 2026-09-25, they were not brought up to date when the work moved on, so they contain statements that later documents supersede: docs/02 and docs/08 say that `env/bootstrap.sh` does not exist yet, and docs/08 says that no training run has been logged. What changed is in [12-design-changes.md](12-design-changes.md), what was registered is in [13-preregistration.md](13-preregistration.md), and what came out is in [results.md](results.md).

Code and documents cite these files as docs/NN, an item inside one of them by its id (docs/12 F23, docs/13 A3, docs/05 E1), and a rule of the research protocol as "protocol rule N". Two file names are fixed, because code opens the files by name: `tests/test_configs.py` reads `05-experiments.md` and `13-preregistration.md`, and the run logger hashes `13-preregistration.md`.

## Documents

| Document | Content |
|---|---|
| [01-state-of-the-art.md](01-state-of-the-art.md) | State of the art: published Tsetlin Machine work on language and embeddings, convergence theory, masked diffusion language models, logic-native competitors, and the prior-art corrections with the list of work that has to be cited next to the claims of the study. |
| [02-verified-code-facts.md](02-verified-code-facts.md) | Verified code facts: what was read in the source of the CAIR packages (GraphTsetlinMachine, tmu, PyHierarchicalTsetlinMachineCUDA), the kernel edit points, the blockers in the stock package, the installation (V4) and the datasets (V5). |
| [03-architecture.md](03-architecture.md) | Architecture: one interface and three tracks. Track A is the reduced GraphTM reference, Track B the sparse event-driven engine and main model, Track C everything written once against the interface. The component specifications and the memory and compute budgets follow. |
| [04-implementation-plan.md](04-implementation-plan.md) | Implementation plan: the milestones M0 to M10 with their acceptance criteria, the schedule, the contingency order, and the mapping from the task ids of the initial design to milestones. |
| [05-experiments.md](05-experiments.md) | Experiments: one specification per experiment id (E0 to E11, s1 to s5, CP, TK, CM, TB, H1 to H3). |
| [06-evaluation-protocol.md](06-evaluation-protocol.md) | Evaluation protocol: metrics, the causal perplexity protocol, baselines, the equal-cost comparison, contamination and leakage rules, reporting conventions and required ablations. |
| [07-novelty-impact-and-signature-demos.md](07-novelty-impact-and-signature-demos.md) | Novelty, impact and signature demos: the qualified novelty statement, the three proof-of-concept pillars, the answer to the supervisor's project proposal (unpublished), the targets, and the eight signature demos G1 to G8. |
| [08-risks-kill-criteria-and-unverified.md](08-risks-kill-criteria-and-unverified.md) | Risks, kill criteria and unverified items: the kill criteria of the initial design, the Track B criteria KB1 to KB7, the risks, and the list of items that are not verified. |
| [09-bibliography.md](09-bibliography.md) | Bibliography, with the evidence label of each entry. |
| [10-glossary.md](10-glossary.md) | Glossary of Tsetlin Machine terms and of the terms that the design adds. |
| [11-engine-spec.md](11-engine-spec.md) | Engine specification of Track B, normative, v0.3. Sections 9 (clause sharding) and 10 (cost bounds) were added on 2026-09-22, section 11 (the count-vote read-out) on 2026-09-24. The Python oracle, the C++ engine and the CUDA engine follow this text. |
| [12-design-changes.md](12-design-changes.md) | Log of design changes: every point at which the study departs from the initial design or from one of its own earlier design decisions, one row per id in sections A to F, with reason and evidence. |
| [13-preregistration.md](13-preregistration.md) | Pre-registration: hypotheses, thresholds, seeds, splits and analysis plan, frozen on 2026-09-22, followed by the dated amendments A1 to A10. The copy carries a publication note. |
| [research-protocol.md](research-protocol.md) | Research protocol: project purpose, hardware, environment setup, and the numbered rules under which runs are made and numbers are reported. |
| [results.md](results.md) | Results: what came out, per registered id, with run ids. |
| [notes/README.md](notes/README.md) | Note on the fifteen data hash records `notes/preregistration-data-hashes*.json` of the pre-registration. |

## Reading order

The order below follows the study from its rules to its outcome. For the outcome alone, the study README and [results.md](results.md) are enough, with this page for the labels.

1. [research-protocol.md](research-protocol.md): purpose, hardware, environment and the rules.
2. [12-design-changes.md](12-design-changes.md): what changed relative to the initial design, and why. It comes before 01 to 10, because those documents were not brought up to date.
3. docs/02: the facts read from the source of the CAIR packages, with the kernel edit points and the blockers in the stock package. It comes before any reading of the code.
4. docs/03: the architecture (one interface, Tracks A, B and C) and the reasoning behind it.
5. docs/11: the engine specification of Track B, normative.
6. docs/04: the milestones M0 to M10 with their acceptance criteria.
7. docs/05: the experiment specifications.
8. docs/06: the evaluation protocol, the baselines, and the contamination and leakage rules.
9. docs/07: the novelty statement, the three pillars and the signature demos.
10. docs/08: risks, kill criteria and the list of unverified items.
11. docs/13: the pre-registration and its amendments.
12. [results.md](results.md): what came out.
13. docs/01, docs/09 and docs/10: state of the art, bibliography and glossary, as references.
14. [notes/README.md](notes/README.md): the data hash records.

## Evidence labels

Numbers and facts taken from a source carry a label in square brackets. The labels are used in the documents of this folder and in comments in the code.

- [measured]: produced by a run on the machine of the recorded runs; always carries a log path under `results/`.
- [verified in source]: read from the actual GitHub or PyPI source or from the official documentation, with the source named. For the vendored GraphTM (`third_party/GraphTsetlinMachine`, commit c6129e14) the label carries `file:line`.
- [verified in paper]: read from the cited paper.
- [verified: exists]: the reference was confirmed to exist; nothing else about it was checked.
- [derived]: arithmetic from verified quantities, not measured.
- [stated]: taken from an earlier repository README, the project proposal, an earlier reading of a source or the design notes; not re-verified.
- [unverified]: not confirmed; flagged for confirmation.

A label may name its source or carry a qualifier after the label word, for example `[verified in source: tm.py:961]`, `[stated: design notes]`, `[derived, unverified]` or `[measured, exploratory: results/<run id>/]`. The design notes are not part of the repository, so a `[stated: design notes]` label cannot be followed further than the statement itself.

A label of the form `[measured: results/<run id>/...]` cites a run folder of this study by its run id. Run folders are not distributed with the repository, so the cited logs cannot be opened from a clone; the [run index](../../../docs/run-index.md) lists the runs. `scripts/check_numbers_have_logs.py` (the `lint` target of the Makefile) checks these labels: every line that carries one has to cite a path under `results/` that exists in the study folder, on the same line or in the same paragraph. The check can therefore be made only where the run folders are present. The script skips a label that is written as inline code, and the leading label of a list item that defines it, as the first item of the list above does. For that reason this page writes a label in running text as inline code.
