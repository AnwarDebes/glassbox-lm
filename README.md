<p align="center"><img src="docs/assets/banner.svg" alt="Logic-based language modelling with Tsetlin Machines" width="100%"></p>

<p align="center">
  <a href="#status-per-study">Studies</a> |
  <a href="#results">Results</a> |
  <a href="docs/results-overview.md">All registered ids</a> |
  <a href="docs/known-limitations.md">Limitations</a> |
  <a href="research/README.md">Ideas and roadmap</a> |
  <a href="docs/data.md">Data</a> |
  <a href="CITATION.cff">Cite</a>
</p>

# Measured steps toward a logic-based language model with Tsetlin Machines

**At a glance**

- Five studies (19 Sep to 25 Oct 2026) toward a language model whose knowledge is a set of logical clauses that can be read; the fourth tests a Tsetlin Machine as the readable decision layer of dialogue agents.
- 91 hypotheses were registered in the first study: 15 are met (13 of them with a qualifier on the results page), 14 are not met, 5 are not judged and 57 were not run. Six of these statuses were set in the finish pass of 2026-10-25 (H8.3, H8.4 and H8.11 met; H8.8, H8.9 and H9.6 not met), and H2.3, H2.4 and H3.5 were checked again there and hold. The [results overview](docs/results-overview.md) lists every id.
- In the fourth study, decision_layer, 107 ids carry a status: 72 are met, 32 are not met and 3 were not run. With the same tuning, its agent with no neural network at any stage is within the registered margin of an agent that runs a transformer encoder per decision on three dialogue test splits, two of them read many times before; the one registered test of "better than" in that comparison, R24.2 on MultiWOZ, is not met.
- The first three studies are mostly negative, the fourth met most of its registered tests with narrow claims, and the fifth, tally, has two verdicts: T0 is not met (on its clauses, the learned votes of a Tsetlin Machine beat the counted read-out) and A2 is met (an exchange-class code lowers the perplexity of a Kneser-Ney 5-gram with a cache by 7.59% on dev-test, a result about count models). Everything is reported as measured: see [Results](#results) and the [known limitations](docs/known-limitations.md).
- A [glass box page](https://anwardebes.github.io/glassbox-lm/) runs the CVTM-core of the first study, the count model that study measured its Tsetlin Machine against, in the browser and shows every vote. It holds no Tsetlin Machine ([live/](live/README.md)).
- A clone holds the code, the configs, the registrations and the results pages. Run folders, result files, datasets and saved models are not distributed; the glass box page is the one exception (see [Data](#data)).
- The ideas behind the programme, what the literature says about each of them and the open work are in [research/](research/README.md).

This repository, `glassbox-lm`, holds a research programme that takes measured steps toward a language model built from Tsetlin Machines, in which what the model knows is a set of readable logical clauses. Its five studies date from 19 Sep to 25 Oct 2026; on 25 Oct a finish pass ran the registered and frozen work that was left and fixed the bugs it found. The first built a token-level Tsetlin Machine language model on a sparse engine written for it and compared it with an in-house Kneser-Ney 5-gram model on the same tokens. The second tested the hierarchical Tsetlin Machine procedure of the supervisor's project proposal (unpublished) on artificial data, on two sentence families, on news topics and on Wikipedia paragraph pairs. The third prepares a word-level clause language model; so far it has produced count baselines, one evaluator run without a Tsetlin Machine, a reproduction of the author's earlier Tsetlin Machine router, Clause-Driven-LLM, that met its first gate only under an amendment written after the gate had failed, and a registered test of the specificity s on that router whose verdict is negative. The fourth, decision_layer, asks where Tsetlin Machines fail on language and what works when an encoder does the language and a Tsetlin Machine makes the decision: as an intent router, as a readable decision layer, and as an agent that chooses the next action in customer-service and task dialogues. The fifth, tally, holds the first steps of a language model built from counts, with places where a Tsetlin Machine must beat a twin that only counts; both of its registered tests have verdicts: in T0 the learned votes of a Tsetlin Machine beat the counted read-out on its clauses (not met), and in A2 an exchange-class code lowers the perplexity of a Kneser-Ney 5-gram with a cache by 7.59% on dev-test (met; a result about count models). The first three studies are mostly negative: the main accuracy goal of the first study was missed, no learned rule of the second study captured either of the two sentence families of the proposal, its single final test is below word overlap, the third study has not trained a Tsetlin Machine on its language data, and its registered test on the router is negative. The fourth met most of its registered tests, and its claims are narrow: a single Tsetlin Machine router trails logistic regression on the same embedding, and the Tsetlin Machine stays below logistic regression on the same bits in several comparisons; an agent with no neural network at any stage is not worse, within the registered margin, than an agent that runs a transformer encoder per decision, and the one registered test of "better than" in that comparison, R24.2 on MultiWOZ, is not met. The first two studies registered hypotheses or pass rules with thresholds before the runs they govern; in the third study one step has a registration so far; decision_layer froze each registration that it ran, with a record of its time and hash, before its confirmatory runs, and tally froze T0 before its final runs, after four of its six tuning runs had started, and A2 before its one look on dev-test. The results pages say where a rule was added, not kept or read after results were known. The name of the repository states the goal of the programme, a language model that can be read like a glass box. It is not a claim about a model in the repository, and no model built here is called a large language model.

## The glass box

**[Open the glass box](https://anwardebes.github.io/glassbox-lm/)**: the CVTM-core of the first study, counted in the browser from ten public-domain books. It is the word part of the count floor of [research/design.md](research/design.md), without the design's word-category codes.

https://github.com/user-attachments/assets/5b06241d-a0be-4f2a-89bf-f9d824464df2

The page counts the books in the browser tab and predicts the next word with the CVTM-core of tm_language_model (section 11 of its [engine specification](studies/tm_language_model/docs/11-engine-spec.md)): Kneser-Ney 5-gram suffix clauses in their exact vote form, the document clauses, the pointer clause and the bucketed mixture, with weights fitted by the study's MAP-EM. For every candidate word it shows the clauses that vote, the memory of the text typed so far and the passages where the voting pattern was counted, and it scores 19 held-out chapters when it loads. No Tsetlin Machine runs on the page. A test checks the JavaScript against the study's own Python code on the same tokens: at 2,500 sampled held-out positions every probability agrees to within 4e-15, and over all 58,266 positions the totals agree to within 1e-9. The page, its data and the test are in [live/](live/README.md); its numbers belong to its corpus and are not comparable with those of the studies.

## Status per study

| Study | Question | Status | Headline result |
|---|---|---|---|
| [tm_language_model](studies/tm_language_model/README.md) | Can a token-level Tsetlin Machine language model, trained with Tsetlin feedback on next-token prediction, reach the accuracy of an n-gram baseline at useful speed, and what do its clauses represent? | archived (ran 19 Sep to 1 Oct 2026) | The reproducibility hypotheses were met at the scale tested; the main accuracy hypothesis H1.1 is not met on seed 1 and the registered kill criterion KB7 triggered; 61 of the 91 registered hypotheses were not run. |
| [hierarchical_tm](studies/hierarchical_tm/README.md) | Does the hierarchical Tsetlin Machine procedure of the project proposal learn what the proposal expects? | finished (ran 1 to 3 Oct 2026) | Mostly negative: no learned rule captured either of the two sentence families of the proposal; the model learns relations between Wikipedia paragraphs, and its single final test is below word overlap on both Wikipedia tasks. |
| [clause_lm](studies/clause_lm/README.md) | Do clauses with learned OR-groups of words add predictive value over count statistics, first on word pairs and then inside a language model, and does the same kind of clause help a readable router? | running as preparation (started 2 Oct 2026) | Count baselines and one evaluator ran, without a Tsetlin Machine. The router reproduction missed its first gate on the CPU backend and met it on a GPU under a post hoc amendment; the registered test of s on the router that followed is negative. |
| [decision_layer](studies/decision_layer/README.md) | Where do Tsetlin Machines fail on language, part by part, and what works instead when an LLM encoder does the language and a Tsetlin Machine makes the decision: as a router, as a readable decision layer, and as an agent that chooses the next action in a dialogue and learns from reward? | running (started 15 Oct 2026; its last registered test so far, R24, ended on 23 Oct 2026) | As a decision layer on frozen LLM embeddings the Tsetlin Machine is calibrated, its decisions can be read as votes of named clauses (post hoc, the readable layer restates the rule "route to the nearest whitened centre"), and it learns a dialogue workflow from labels or from reward alone (memory literals add +18.65 points on ABCD, R9b), but in routing accuracy with labels it never beats a linear model on the same information in the language it was trained in. With the same tuning, an agent with no neural network at any stage is within the registered margin of an agent that runs bge-large-en-v1.5 per decision on the STAR, ABCD and MultiWOZ 2.2 test splits (R22.3, R23.1, R24.1); the registered superiority test of that comparison on MultiWOZ (R24.2) is not met, and the Tsetlin Machine stays below logistic regression on the same bits in several comparisons. |
| [tally](studies/tally/README.md) | In the design called Tally, a language model built from counts with no Tsetlin Machine in it, and places in it where a Tsetlin Machine is compared on the same bits with a twin that only counts: on Penn Treebank, do counted votes on the clauses of a Tsetlin Machine beat its learned votes (T0), and does an exchange-class code help the count floor (A2)? | running (started 15 Oct 2026; run folders from 15 to 26 Oct 2026) | T0 not met: on the clauses of a Tsetlin Machine the learned votes beat the counted read-out by 0.251 bits per word on dev-test, on every seed. A2 met: an exchange-class code lowers the perplexity of a Kneser-Ney 5-gram with a cache by 7.59% on dev-test, a result about count models. The other lines of the study (B1, B6, A1 and C3) have smoke, timing, baseline or exploratory runs and no registration. |

The list of studies with the dependencies between them is in [studies/README.md](studies/README.md).

## Results

The table gives the headline measurements of each study with the run that produced them, or, for decision_layer, which keeps no run folders, with the result file; neither is distributed. Each value is copied from the results page of its study, which names the file of the run folder or the result file and gives the remaining conditions. The statuses of all registered ids are collected in [docs/results-overview.md](docs/results-overview.md).

| Study | Measurement | Value and comparison | Registered id and status | Run id or result file |
|---|---|---|---|---|
| tm_language_model | Causal model at V = 8000 with 1,048,576 clauses, seed 1: bits per byte on the WikiText-103 test split | 2.2193, against 1.3719 for the in-house Kneser-Ney 5-gram on the same tokens (ratio 1.618). Single seed, one read of the test split, and 0.114 of the test 8-grams also occur in the training split. | H1.1 not met, on seed 1 only (seeds 2 and 3 not run); kill criterion KB7 triggered | `20260923T042350Z_CP_anchor` |
| tm_language_model | Count-vote read-out of the clause bank of that run (causal arm of H8.1): bits per byte on the validation split | 1.8280, against 2.1749 for the signed votes of the same bank. The comparison is with the model's own read-out, not with the n-gram baseline; the results page places this read-out at the level of a Kneser-Ney bigram. | H8.1 met; compared with the model's own signed votes; the causal arm uses a bank seen in the exploratory evaluation | `20260924T062704Z_cvtm_H8_1_causal_cp20M` |
| tm_language_model | Confirmatory self-supervised text runs, seeds 1 to 3: Q_text (accuracy of a few-label probe on three text classification sets, in points) of the winning arm minus the stronger control, its own young bank | +0.09 (-1.51 to +1.78) on the all-clauses code, which decides, and -4.57 (-6.16 to -2.47) on the dimension-matched code. Registered threshold: at least 1.0 point, interval excluding 0. | H17.1 not met | `20260930T163619Z_ssl_r1cf_report_confirmatory` |
| tm_language_model | The same report: Q_text of the winning arm minus bag of words | -5.85 (-7.31 to -4.23) on the all-clauses code, which decides, and -12.41 (-14.32 to -10.44) on the dimension-matched code. Registered threshold: at least 2.0 points, interval excluding 0. | H17.2 not met | `20260930T163619Z_ssl_r1cf_report_confirmatory` |
| hierarchical_tm | The one final test, mean of 5 seeds: test AUC on same-article pairs | 0.735, against 0.817 for word overlap and 0.869 for the baseline that was best on dev, a logistic regression | The final test: not met, in the sense of the registered comparison "beats X"; the registered outcome is "below" | `20261002T193513Z_final_test_main_table` |
| hierarchical_tm | The same test: test AUC on next-paragraph pairs | 0.750, against 0.857 for word overlap and 0.903 for the baseline that was best on dev, a logistic regression | as above | `20261002T193513Z_final_test_main_table` |
| hierarchical_tm | The same test: test AUC on next-paragraph pairs with every shared word removed | 0.614, against 0.811 for the baseline that was best on dev, a PPMI word-vector cosine. Word overlap is 0.500 on this data by construction, which is not a comparison. | as above | `20261002T193513Z_final_test_main_table` |
| clause_lm | Gate 0 of M5 step 1, the reproduction of the lexical Clause-Driven-LLM router on CLINC150 (s 8, seed 42): test accuracy | 0.5407; the registered rule is a test accuracy within 0.005 of 0.82778. One completed run with one seed on the numba CPU backend. The published `studies/clause_lm/src/m5_run.py` is a later revision than the one that ran, and the commit of the upstream copy is recorded in project notes, not checked against upstream. | Gate 0 not met; cause not confirmed; a retry was stopped without a score | `20261002T195255Z_m5_gate0_s8_seed42_sorted` |
| clause_lm | M5 step 1, the registered test of s for the same router on CLINC150, seeds 42, 123 and 456: mean test accuracy at s*, the s with the highest mean validation accuracy | 0.8201 at s* = 8, against the 0.848 that the success rule needs, with seeds 123 and 456 below 0.828; the mean test accuracy falls as s grows. The grid rests on the amended Gate 0 (test accuracy 0.8293 with the torch backend on a GPU), which is post hoc. | M5 step 1 not met; negative-result rule met | `20261014T222730Z_m5_summary_sorted_torchgpu` |
| clause_lm | Perplexity on the Penn Treebank validation split: Kneser-Ney trigram model alone, and with the count votes of k-means word groups (no Tsetlin Machine) | 161.72 and 157.24, from a single run | no registered id; the bar that milestone M2 (not run) has to beat | `20261002T174323Z_clause_lm_kmeans200` |
| decision_layer | Agent with no neural network at run time (the word table potion-base-32M, distilled once, offline, from bge-base-en-v1.5, read by lookup and OR, then a Tsetlin Machine: the potion agent): next-action accuracy on the STAR and ABCD test splits, against an agent that runs bge-large-en-v1.5 per decision | STAR +0.06 [-0.63, +0.74] points (62.7% against 62.7%), ABCD -1.00 [-1.62, -0.40] (83.0% against 84.0%); registered margin 2.0 points. Open loop; the 2nd use of STAR's test split and the 11th of ABCD's. | R19 met; R19.1 in the strong form on both datasets | `studies/decision_layer/results/confirm/R19star/report_R19star.json`, `studies/decision_layer/results/confirm/R19abcd/report_R19abcd.json` |
| decision_layer | Agent with no neural network at any stage (a word table counted from C4 text): the same accuracy, against the potion agent of R19 | STAR -0.10 [-0.65, +0.45], ABCD -0.29 [-0.74, +0.17] (62.6% and 82.7%). With no rule: -0.04 and -1.29 against the bge-large agent; on the same bits logistic regression is 2.79 points ahead on STAR, and the Tsetlin Machine is 1.76 ahead on ABCD. Open loop; the 5th use of STAR's test split and the 13th of ABCD's. | R21 met; R21.1 in the strong form on both datasets | `studies/decision_layer/results/confirm/R21star/`, `studies/decision_layer/results/confirm/R21abcd/` |
| decision_layer | The same agent with the Tsetlin Machine's s tuned on dev, against the bge-large agent tuned the same way (s 16 for both on STAR; on ABCD s 16 for the C4 agent and s 24 for bge-large) | STAR +1.64 [+0.94, +2.35] (65.3% against 63.7%), ABCD -0.92 [-1.61, -0.26] (83.6% against 84.6%); registered margin 2.0 points, superiority not registered. On STAR the C4 agent is level with a tuned logistic regression on the same bits (-0.09 [-0.90, +0.73], R22.2). The 6th use of STAR's test split (R22) and the 14th of ABCD's (R23). | R22.3 met, strong form; R23.1 met, strong form | `studies/decision_layer/results/confirm/R22star/`, `studies/decision_layer/results/confirm/R23abcd/` |
| decision_layer | MultiWOZ 2.2 test (its 1st use), all three agents with random negatives, a rule chosen on MultiWOZ dev on the refined agent: the C4 agent (no neural network at any stage) against the bge-large agent | +0.53 [-0.18, +1.23] (accuracy 0.3437 against 0.3384; registered margin 1.0 point); the refined-table agent (network-free at run time, with bge-base-en-v1.5 as teacher at training time) +0.05 [-0.65, +0.72]. On dev the rule costs STAR -11.9 and ABCD -4.0 points (measured on the refined agent, e83). With no rule, the Tsetlin Machine is below logistic regression on the same bits for all three agents (C4 -1.55, refined -2.49, bge-large -1.89). | R24.1 met, strong form, for both agents; R24.2 (superiority) not met for either | `studies/decision_layer/logs/R24_report.out` (a run log, not distributed) |
| tally | T0: the counted read-out against the learned votes on the clauses of the coalesced Tsetlin Machine of tmu, in bits per word on the dev-test split of Penn Treebank (mean of seeds 1 to 10) | 9.4110 for the counted read-out against 9.1598 for the learned votes: counted minus learned +0.2512 (+0.2220 to +0.2801), with the same sign on every seed, so the learned votes are better; logistic regression on the same clause outputs reaches 8.2797 (no verdict) | T0 not met; "Tsetlin learning helps the counted read-out" not met either (+0.1126, +0.1041 to +0.1205) | `20261025T104315Z_t0_report_devtest` |
| tally | A2: perplexity of a Kneser-Ney 5-gram with a cache on the dev-test split of Penn Treebank, without and with the class members of an exchange-class code | 124.251 and 114.821, 7.59% lower (95% interval over dev-test blocks 6.51% to 8.58%). Exploratory (one build, one split, one benchmark), and a result about count models, not about Tsetlin learning | A2 met | `20261025T133912Z_a2_devtest_final` |

The limits of these numbers, among them duplicate articles in the test split of the Wikipedia pair data, are on the page of [known limitations](docs/known-limitations.md). The README of hierarchical_tm lists the claims that are not made on the strength of that study.

## Ideas, design and open work

The folder [research/](research/README.md) holds what comes before the next study, and nothing in it has been run. [ideas.md](research/ideas.md) records the ideas the programme started from, with the [pages of the writing board](research/board/README.md) they were written on. [findings.md](research/findings.md) gives what a literature search found about each idea, negative verdicts included. [design.md](research/design.md) is the design that follows from it, a count floor without any Tsetlin Machine and a test bench in which Tsetlin learning is measured against a twin that only counts. [roadmap.md](research/roadmap.md) lists what the studies registered or prepared and did not run, and what comes next.

## Layout

```
README.md               this page
CHANGELOG.md            history of the repository
CONTRIBUTING.md         conventions for changes
CITATION.cff            citation metadata
LICENSE                 CC BY-NC 4.0
docs/                   what spans the studies
  README.md             index of the folder
  results-overview.md   statuses of the registered ids and headline numbers
  known-limitations.md  what limits the use of the repository and of its results
  data.md               the datasets and how to obtain them
  references.md         bibliography
  upstream-repositories.md
  run-index.md          one row per run folder of the studies that keep run folders
  hash-table.md         SHA-256 pins, as registered and as published
  assets/               banner and logo
research/               ideas, literature findings, design and roadmap; nothing run
  README.md             index of the folder and how far to trust it
  ideas.md  findings.md  design.md  roadmap.md
  board/                the pages of the writing board, photos and readings
studies/
  README.md             list of studies, their dependencies, how a study is added
  tm_language_model/    README.md, pyproject.toml, Makefile, src/tmlm/, experiments/, tests/,
                        scripts/, env/, launch/, docs/, third_party/
  hierarchical_tm/      README.md, gates.md, frozen_settings_hashes.json, src/, queue/, docs/
  clause_lm/            README.md, src/, docs/
  decision_layer/       README.md, FAILURE_MAP.md, registrations/, logs/ (dev plans, notes, drivers),
                        src/dl/, scripts/, tests/, verification/, docs/, env_dl_freeze.txt, env_htm_freeze.txt
  tally/                README.md, registrations/, src/tally/, scripts/, tests/, docs/
live/                   the glass box page: the CVTM-core in the browser, its data and its tests
.github/workflows/      the test and the deployment of the page
```

Each study keeps its registration, its results page and its data cards in its own folder. The folders `results/` (run folders or result files) and `data/` (datasets) of a study are local and are ignored by git, as are the virtual environments and the run logs of decision_layer and tally.

## Setup

Nothing is built at the repository root. Each study README gives the commands of its study, and they are run from the study folder. The tm_language_model study has a Makefile, whose targets (`help`, `env-check`, `lint`, `test`, `smoke`, `bootstrap`, `configs-check`) are reached from the root with

```bash
make -C studies/tm_language_model <target>
```

`studies/tm_language_model/env/bootstrap.sh` builds two virtual environments inside the study folder and `studies/tm_language_model/env/htm_install.sh` a third; the lock files in `studies/tm_language_model/env/` record the package versions. The bootstrap needs `nvcc` and `g++` and asserts a GPU of CUDA compute capability 7.0. It installs the vendored GraphTsetlinMachine package in editable mode. hierarchical_tm and clause_lm have no environment script of their own: their launchers start jobs through scripts of tm_language_model. decision_layer builds its own environments, `.venv-dl` and `.venv-htm`, as its README describes, and takes only the GPU lock of tm_language_model; tally runs in environments of tm_language_model and clause_lm and builds one of its own, `.venv-b6`, for the data preparation of B6. The study folders therefore have to stay next to each other ([studies/README.md](studies/README.md)). The code encodes the workstation of the recorded runs, one Tesla V100 and a quota of 6 CPUs.

## Data

No dataset is distributed, and neither are derived data or saved models. [docs/data.md](docs/data.md) names the source and the version of each dataset and says how to obtain it. The data cards of hierarchical_tm are in `studies/hierarchical_tm/docs/`, and those of decision_layer and tally in their own `docs/` folders.

The one exception is the glass box page: [live/](live/README.md) holds the token ids of ten public-domain books (`tokens.u16.gz`) and the mixture weights fitted on them (`core.json`), which the page needs. `live/export.py` downloads the books and rebuilds both; the SHA-256 of the ten book files that reproduce them are in [live/README.md](live/README.md) and [docs/data.md](docs/data.md).

## What a clone allows

A clone holds the code, the configs, the registrations, the results pages and the [run index](docs/run-index.md). The registrations are [docs/13](studies/tm_language_model/docs/13-preregistration.md) of tm_language_model, [gates.md](studies/hierarchical_tm/gates.md) and the [registrations page](studies/hierarchical_tm/docs/registrations.md) of hierarchical_tm, the [registration of M5 step 1](studies/clause_lm/docs/prereg_M5_step1.md) of clause_lm, the registrations of decision_layer in `studies/decision_layer/registrations/` with the dev plans and notes in its `logs/`, and the registrations T0 and A2 of tally in `studies/tally/registrations/`. A reader can compare a status or a number on a results page with the registered rule, read the code that the page names, look up a cited run id in the run index, and read the independent recomputations of decision_layer in `studies/decision_layer/verification/`.

A clone holds no run folder, no result file, no dataset, no saved model, no compiled engine binary and no virtual environment, so no number of the studies can be recomputed from the clone alone, and the source file of a number can be inspected only for the recomputations in `studies/decision_layer/verification/`, whose recorded outputs are included. Reproducing a number means obtaining the data, building an environment from the lock files and running the experiment again on comparable hardware. The sentence files of the slide-6 and slide-10 tests of hierarchical_tm are not distributed either, because they carry sentences of the project proposal, so those checks cannot be rerun from a clone.

The numbers check of tm_language_model needs the run folders: `make lint` accepts a number that is labelled as measured in the study's documents only when the cited folder exists under `results/`, so without the run folders `make lint` and `make smoke` fail, and two tests of `make test` depend on run folders as well.

The glass box page is the part that a clone can rerun: `studies/tm_language_model/.venv-tm/bin/python live/export.py --cache <folder>` rebuilds its data with the study's code on a CPU in under half a minute, and `npm ci && npm test` in `live/` checks the page against it (`npm ci` installs jsdom, the one test tool).

The published files are copies of the working files, edited for publication: references to unpublished working notes and plans, paths of the workstation and remarks on day-to-day work were removed or reworded in documents, comments and docstrings. The published tree of tm_language_model and hierarchical_tm was not run as a whole: their tests, experiments and installs were last run on the working copies, before the edits for publication. clause_lm ran its amended Gate 0 and its grid on 2026-10-14 with copies of `studies/clause_lm/src/m5_run.py`, `studies/clause_lm/src/m5_report.py` and `studies/clause_lm/src/runlog.py` that are byte identical to the published files. The tests of decision_layer and tally were run on the published files on 2026-10-23, on the CPU; the tests that need the GPU were not run, and two test files of decision_layer stop at a check that failed before publication as well. The page of [known limitations](docs/known-limitations.md) says what follows from this.

## Author and supervision

The work is by Anwar Debes, master's student at the University of Agder, Centre for Artificial Intelligence Research (CAIR). The supervisor is Ole-Christoffer Granmo, and the hierarchical_tm study tests the procedure of his project proposal (unpublished). He has not reviewed the code, the registrations or the results of this repository. The decisions that the records call recorded decisions are the author's, and so is any error.

## Licence

Copyright (c) 2026 Anwar Debes. The code and the documents of this repository are released under the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0, [LICENSE](LICENSE)): they may be used, shared and adapted for non-commercial purposes with credit to the author. The vendored package in `studies/tm_language_model/third_party/GraphTsetlinMachine` keeps its upstream MIT licence file. The datasets are not distributed and keep their own licences ([docs/data.md](docs/data.md)).

## Citation

See [CITATION.cff](CITATION.cff).
