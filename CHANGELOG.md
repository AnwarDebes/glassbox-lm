# Changelog

History of the programme, oldest first: one entry per week of the tm_language_model study, one for
each of the two later studies and one for the publication, which also covers the research notes. The three studies are listed in
[studies/README.md](studies/README.md). A run id names a run folder and starts with the UTC time of
the run; run folders are not distributed. Every number and status below is repeated from the results
page that its entry links, where the file of the run and the full qualifiers are given.

## 19 to 25 Sep 2026: tm_language_model, from the pre-registration to the count-vote read-out

The pre-registration of the study was frozen on 2026-09-22; amendments A1 and A2 followed the same
day. Kill criterion KB1 was not triggered (`20260922T155509Z_parity_tmu`), on a statistic fixed
before that run, after a first comparison without a run folder had fallen outside the allowed band.
The planted-rules line of milestone M3 was met (`20260922T165439Z_planted_rules`) on the mean over
seeds, under a superset definition adopted post hoc, not under the strict one. The tiny language
model met H1.4 (`20260922T190301Z_tiny_lm`, `20260922T193047Z_tiny_lm`) on the validation split,
with settings chosen post hoc after two logged attempts had failed.

The anchor run `20260923T042350Z_CP_anchor` (causal, V = 8000, 1,048,576 clauses, seed 1) scores
2.2193 bits per byte on the WikiText-103 test split against 1.3719 for the in-house Kneser-Ney
5-gram on the same tokens: one seed, one read of the test split, and 0.114 of the test 8-grams also
occur in the training split. H1.1 is not met on seed 1 only, and kill criterion KB7 triggered;
seeds 2 and 3 were not run.

After the miss, amendments A3 to A10 (24 and 25 Sep) registered the families H8 to H17. The
count-vote read-out of A3 met H8.1 (`20260924T062704Z_cvtm_H8_1_causal_cp20M` and the masked runs
of seeds 2 to 5). Its threshold compares it with the signed votes of the same clause bank, not with
the n-gram baseline; it is at the level of a Kneser-Ney bigram, and the causal arm uses a bank
seen in the exploratory evaluation. The copy selector met H9.1, trails its best control (H9.2 not
met) and has no learning credit (`20260925T063437Z_cvtm_copy_selector_H9_verdicts`).
[Results page](studies/tm_language_model/docs/results.md).

## 26 Sep to 1 Oct 2026: tm_language_model, the self-supervised runs and their controls

The exploratory control study of the self-supervised family
(`20260926T204018Z_ssl_r1c_report_exploratory`) read "training does not add value". The
confirmatory text runs (seeds 1 to 3, 26 to 30 Sep) leave H17.1 to H17.4 and H17.11 not met
and H17.12 met (`20260930T163619Z_ssl_r1cf_report_confirmatory`); gate G-R1 was read on 2026-09-30
as "fail and below bag of words". The mechanism test (H17.5, both models at chance,
`20260930T203322Z_ssl_r2_hao_report_confirmatory`) and the image test (H17.10, trained code below
its young bank, `20260930T203322Z_ssl_r4_report_confirmatory`) are not met; the image controls of
1 Oct (`20261001T202442Z_ssl_r4c_report`) have no threshold. The study ended on 1 Oct and is
archived, with 12 of its 91 registered hypotheses met, 11 not met, 7 not judged and 61 not run.
[Results page](studies/tm_language_model/docs/results.md).

## 1 to 3 Oct 2026: the hierarchical_tm study

The study tested the hierarchical Tsetlin Machine procedure of the supervisor's project proposal
(unpublished). Its pass rules, [gates.md](studies/hierarchical_tm/gates.md), were frozen on 1 Oct
2026 before its runs; 27 exploratory probe runs of that day used pair data with a sampling bias
and are not used. Checks 1a, 1b, 3 and 4a are met. Check 4b is met on the s = 100 arm only, with a
small effect, and not met at s = 400 and at s = 200, an arm that was allowed after the results of
the other two were known. Checks 2a, 2b and 2c are not met, and no learned rule captured either of
the two sentence families of the proposal (Check 2b and the slide-6 paraphrase test); the results
page gives the qualifier of each. The one final test, `20261002T193513Z_final_test_main_table`
(mean of 5 seeds), has test AUC 0.735 on same-article and 0.750 on next-paragraph pairs, below word
overlap (0.817 and 0.857). [Results page](studies/hierarchical_tm/docs/results.md).

## 2 Oct 2026: preparation of the clause_lm study

The clause_lm study prepares a word-level clause language model. On Penn Treebank the n-gram
baselines, the pair data, the statistical pair baselines and one clause evaluator with k-means word
groups ran once each, without a Tsetlin Machine. For the one registered step,
[M5 step 1](studies/clause_lm/docs/prereg_M5_step1.md), Gate 0 (the reproduction of the lexical
Clause-Driven-LLM router on CLINC150) was not met in its one completed run,
`20261002T195255Z_m5_gate0_s8_seed42_sorted`; the cause is not confirmed. A retry was stopped
without a score, the registration was amended post hoc for a GPU rerun that did not take place,
and the grid and the pair model were not run. [Results page](studies/clause_lm/docs/results.md).

## 10 and 11 Oct 2026: preparation of the publication

The three studies were brought into one repository, each as a folder under `studies/`, with the
pages that span them in `docs/`. The wording of documents, comments and docstrings was edited,
absolute paths were replaced by paths relative to the study folders, and the registrations were
published as copies with a publication note; their registered hypotheses, thresholds, seeds and
dates are unchanged. Files that a record pins by hash and whose text changed were re-pinned, as the
[hash table](docs/hash-table.md) lists. Run folders, datasets and the files that carry sentences
of the project proposal are not distributed, and the edits were not exercised by a run
([known limitations](docs/known-limitations.md)).

On 11 Oct 2026 the folder [research/](research/README.md) was added: the ideas of the programme with
the pages of the writing board, the findings of a literature search of 2026-10-10 on each idea, the
design that follows from them and the list of open work. Nothing in that folder has been run. The
repository was given the licence CC BY-NC 4.0, and the upstream `setup.py` of the vendored package
was added to it.
