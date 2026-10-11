# Results overview

This page gives, for each of the three studies, the status of what the study registered and its headline measurements. Every number is copied from the results page of its study and stands next to the id of the run that produced it. The results pages name the file of the run folder for each value and give the conditions that belong to it; where this page and a results page differ, the results page is right. Run folders and datasets are not distributed, so no number can be recomputed from a clone. A run id can be placed in the [run index](run-index.md), and the limits of the evidence are on the page of [known limitations](known-limitations.md).

The statuses are those of the results pages: met, not met, not judged (runs exist, and no verdict was drawn from them) and not run. A kill criterion is triggered or not triggered.

## tm_language_model

Status of the study: archived; it ran from 19 Sep to 1 Oct 2026. [Results page](../studies/tm_language_model/docs/results.md), [pre-registration](../studies/tm_language_model/docs/13-preregistration.md) (docs/13 of the study).

The study registered 91 hypotheses in 16 families. H1 to H5 belong to the pre-registration that was frozen on 2026-09-22. H6 was added by amendment A1 on the same day, and H8 to H17 by amendments of 2026-09-24, after the anchor run of 2026-09-23 had missed the main accuracy goal. Of the 91 ids, 12 are met, 11 are not met, 7 are not judged and 61 were not run. The four reproducibility hypotheses (H2) are met at the scale tested. Six of the met ids need their qualifier to be read correctly. H1.4 was met with settings chosen after two failed logged attempts (post hoc) and on the validation split. H2.2 was judged in its training half, and H2.3 as a whole, from a test run that left no run folder. H3.5 was judged on 100 steps where the registration names 200. H8.1 compares a count-vote read-out with the signed-vote read-out of the same clause bank and not with the n-gram baseline, and H9.1 is met without learning credit, which by the registered rule means that the gain comes from the literals and the read-out and not from Tsetlin learning beyond 1 percent of the pairs. In the self-supervised family H17, 7 of the 8 ids that were run are not met.

| Family | Subject | Registered by | Ids | Met | Not met | Not judged | Not run |
|---|---|---|---|---|---|---|---|
| H1 | accuracy | pre-registration, 2026-09-22 | 4 | 1 | 1 | 1 | 1 |
| H2 | reproducibility and extensibility | pre-registration, 2026-09-22 | 4 | 4 | 0 | 0 | 0 |
| H3 | scaling with compute | pre-registration, 2026-09-22 | 5 | 1 | 0 | 1 | 3 |
| H4 | the rows on the hierarchical package and the task of the supervisor's project proposal (unpublished) | pre-registration, 2026-09-22 | 3 | 0 | 0 | 1 | 2 |
| H5 | kill-criterion checks on the stock GraphTsetlinMachine package | pre-registration, 2026-09-22 | 3 | 1 | 1 | 0 | 1 |
| H6 | speed | amendment A1, 2026-09-22 | 5 | 0 | 1 | 0 | 4 |
| H8 | count-vote read-out | amendment A3, 2026-09-24 | 11 | 3 | 0 | 4 | 4 |
| H9 | learned residual clauses, transformer frontier, white-box properties | amendment A4, 2026-09-24 | 11 | 1 | 1 | 0 | 9 |
| H10 | kinship reasoning task | amendment A4, 2026-09-24 | 9 | 0 | 0 | 0 | 9 |
| H11 | second corpus (PG-19) | amendment A4, 2026-09-24 | 3 | 0 | 0 | 0 | 3 |
| H12 | data scale | amendment A6, 2026-09-24 | 7 | 0 | 0 | 0 | 7 |
| H13 | Tsetlin-learned generalisation | amendment A6, 2026-09-24 | 1 | 0 | 0 | 0 | 1 |
| H14 | deep Tsetlin Machine pilot | amendment A6, 2026-09-24 | 3 | 0 | 0 | 0 | 3 |
| H15 | efficiency of the count-vote rows | amendment A6, 2026-09-24 | 7 | 0 | 0 | 0 | 7 |
| H16 | distillation pilot | amendment A6, 2026-09-24 | 3 | 0 | 0 | 0 | 3 |
| H17 | logical self-supervised representations | amendment A7, 2026-09-24, with A8 to A10 of 2026-09-25 | 12 | 1 | 7 | 0 | 4 |
| all | | | 91 | 12 | 11 | 7 | 61 |

The rows were counted on the results page id by id and agree with the count table of that page.

The study has seven kill criteria for its engine, KB1 to KB7 (docs/08 of the study). One is triggered: KB7, sparse-only causal perplexity above 1.5 times that of the Kneser-Ney 5-gram, in the anchor run `20260923T042350Z_CP_anchor`. Four are not triggered (KB1, KB2, KB3 and KB6); of these, KB2 failed on the first two logged runs and was met on the third. Two are not judged (KB4 and KB5), because the experiments that define them were not run. Of the three gates that the implementation plan of the study keeps (its docs/04), Gate 1 is not judged, Gate 2 is not met (calibration part, at tiny-LM scale; no verdict on the 48-hour part) and Gate 3 is met (at tiny-LM scale). The six gates of amendment A4 and the decision rows of amendment A5 are branching rules without a pass condition and carry no status of this kind; one gate (G1 T2) and one decision row (D7) were applied, both on 2026-09-25.

Headline measurements:

| Id and status | Measurement | Value and comparison | Run id |
|---|---|---|---|
| H1.1 not met, on seed 1 only (seeds 2 and 3 not run); KB7 triggered | Causal model at V = 8000 with 1,048,576 clauses, seed 1: bits per byte on the WikiText-103 test split | 2.2193, against 1.3719 for the in-house Kneser-Ney 5-gram on the same tokens, a ratio of 1.618. Single seed, one read of the test split, and 0.114 of the test 8-grams also occur in the training split. KB7 is written in terms of perplexity, and the run evaluates it on bits per byte. | `20260923T042350Z_CP_anchor` |
| H8.1 met; compared with the model's own signed votes; the causal arm uses a bank seen in the exploratory evaluation | Count-vote read-out of the clause bank of the anchor run: bits per byte on the validation split | 1.8280, against 2.1749 for the signed votes of the same bank. The results page places this read-out at the level of a Kneser-Ney bigram and far worse than the 5-gram. | `20260924T062704Z_cvtm_H8_1_causal_cp20M` |
| H17.1 not met | Confirmatory self-supervised text runs, seeds 1 to 3: Q_text (accuracy of a few-label probe on three text classification sets, in points) of the winning arm minus the stronger control, its own young bank | +0.09 (-1.51 to +1.78) on the all-clauses code, which decides, and -4.57 (-6.16 to -2.47) on the dimension-matched code. Registered threshold: at least 1.0 point, interval excluding 0. | `20260930T163619Z_ssl_r1cf_report_confirmatory` |
| H17.2 not met | The same report: Q_text of the winning arm minus bag of words | -5.85 (-7.31 to -4.23) on the all-clauses code, which decides, and -12.41 (-14.32 to -10.44) on the dimension-matched code. Registered threshold: at least 2.0 points, interval excluding 0. | `20260930T163619Z_ssl_r1cf_report_confirmatory` |

With H17.1 not met and the winning arm below bag of words, gate G-R1 of amendment A7 took its third branch, "fail and below bag of words". The verdict on H9.1 carries a disclosure on the results page about the seed runs that the verdict configuration lists. Every count-vote number of the study is a validation number: the count-vote core was never scored on the test split.

## hierarchical_tm

Status of the study: finished; it ran from 1 to 3 Oct 2026. [Results page](../studies/hierarchical_tm/docs/results.md), [pass rules](../studies/hierarchical_tm/gates.md) (gates.md of the study), [registrations](../studies/hierarchical_tm/docs/registrations.md).

The study tested the hierarchical Tsetlin Machine procedure of the project proposal: a model whose rules are conjunctions of OR-groups of words is trained to say whether two texts belong together, and its OR-groups are then read as word groups. The pass rules were frozen on 1 Oct 2026, before the runs of the study, and have nine entries, eight checks and the final test. The study ended mostly negative. The model learns relations between Wikipedia paragraphs and forms topically coherent word groups there, and its single final test is below word overlap on both Wikipedia tasks and below the baseline that was best on dev on all three data sets.

| Entry of the pass rules | Status | Qualifier |
|---|---|---|
| Check 1a (G0): the package demo is reproduced | met | the demo ran with 1 repeat per seed, not its default of 10 |
| Check 1b (data): no shortcuts | met | the next-paragraph data passed after a rebuild; the main synonym data is borderline on the skewed-word count and passes only by the decision of 1 Oct 2026, 20:10 CEST (addendum of gates.md) |
| Check 2a (G1): artificial tests L1 to L3 | not met | L1 met; L2 not met, borderline, under an accuracy that was chosen post hoc; L3 not met, at an `s` fixed by a rule that was written after the first `s` 400 result |
| Check 2b (G3): slide 10 | not met | model shaped to the known template (the number of OR-groups and of options per group was set to it); not met in the pair version and in the direct-label fallback; the fallback is judged under one of two readings of its time cap, and under the other reading it would be "not judged within the cap" |
| Check 2c (news): AG News word groups | not met | accuracy part not met, NPMI part met; the word groups are shown as examples only |
| Check 3 (G2): Wikipedia learns (A) or not (B) | met | branch A; `s` and the number of passes were chosen on the dev data of seed 1 |
| Check 4a (G4): Wikipedia word groups | met | scored before the decision between branch A and branch B was recorded; topical coherence, not synonymy |
| Check 4b (synonyms): the synonym test | met | on the arm at `s` = 100 only, with a small effect; not met on the arms at `s` = 400 and `s` = 200; the arm at `s` = 200 was allowed after the results of the other two arms were known |
| The final test | not met | status of the registered comparison "beats X": the outcome is "below" word overlap on both Wikipedia tasks and "below" the baseline that was best on dev on all three data sets |

For Check 4b the records of the study allow the wording "slightly more synonym-like than chance" for the one arm that met the check, and nothing stronger. The arm at `s` = 100 met the check on 2 of 3 seeds (`20261002T093158Z_check4b_synonym_groups_s100`); the arms at `s` = 400 and `s` = 200 met it on 0 of 3 seeds each (`20261002T130624Z_check4b_synonym_groups_s400`, `20261002T231829Z_check4b_synonym_groups_s200`).

The final test was run once, on 5 seeds per data set. The numbers are test AUC values from the file `report.json` of the run `20261002T193513Z_final_test_main_table`; the bracketed intervals are the 95% intervals of the paired article bootstrap that the registration fixes.

| Final test | Same-article pairs | Next-paragraph pairs | Next-paragraph pairs with every shared word removed |
|---|---|---|---|
| Model, mean of 5 seeds | 0.735 | 0.750 | 0.614 |
| Word overlap (Jaccard) | 0.817 | 0.857 | 0.500 by construction |
| Baseline that was best on dev | 0.869 (a logistic regression) | 0.903 (a logistic regression) | 0.811 (PPMI word-vector cosine) |
| Model minus word overlap, paired | -0.082 [-0.093, -0.071]: below | -0.106 [-0.116, -0.096]: below | by construction, not a comparison |
| Model minus the baseline that was best on dev, paired | -0.135 [-0.144, -0.125]: below | -0.153 [-0.162, -0.144]: below | -0.197 [-0.210, -0.185]: below |

Outside the nine entries the results page reports three further items. The slide-6 paraphrase test was registered on 2 Oct 2026 as a descriptive test and is not a check of gates.md; its model was shaped to the known template as for slide 10, and its claim was not met. The robustness arms R1, R2 and S6-R1 were registered after the results they relate to were known, are labelled post hoc and change no verdict. Four runs with one option of the package switched have no registration and no verdict.

The 27 exploratory probe runs of 2026-10-01 that came before the study used pair data with a sampling bias and are not used. The claims that are not made on the strength of this study are listed in the [README of the study](../studies/hierarchical_tm/README.md).

## clause_lm

Status of the study: running, as preparation; every run dates from 2026-10-02. [Results page](../studies/clause_lm/docs/results.md), [registration of M5 step 1](../studies/clause_lm/docs/prereg_M5_step1.md).

The study prepares a word-level language model whose clauses are AND-rules over OR-groups of words with count votes. Twelve ids carry a status: the milestones of the study design, with M5 counted as its two steps, and six items of the registration of M5 step 1, the only step that has a registration. No milestone has been met or missed so far. The tests of M1 and M2 have not been run; the runs that exist for them set the bars that the later models have to beat. M5 step 1 stopped at its first gate.

| Id | Status | Qualifier |
|---|---|---|
| M1, pair model on adjacent word pairs of Penn Treebank | not run | preparation only: the pair data and the statistical baselines ran, the pair model was written and not run |
| M2, clause-count language model with the OR-groups of M1 | not run | preparation only: the evaluator ran once with k-means word groups and no Tsetlin Machine |
| M3, full clause language model | not run | no code, no run |
| M4, scale and efficiency | not run | no code, no run |
| M5 step 1, the registered question about s for the Clause-Driven-LLM router on CLINC150 | not judged | stopped at Gate 0, which was not met; the s-test was not run, so the registered question has no verdict |
| M5 step 2, OR-group word concepts as router features | not run | no code, no run |
| Gate 0, the loader is valid | not met | one completed run, numba CPU backend, seed 42; cause not confirmed; a retry was stopped without a score |
| Amended Gate 0 (amendment of 21:11 UTC on 2026-10-02) | not run | post hoc: the amendment was written after the Gate 0 result was known; the GPU rerun did not take place |
| s-test grid | not run | Gate 0 not met; the registration then stops before the s-test |
| Choice of s* | not run | no grid |
| Success rule | not run | no grid |
| Negative-result rule | not run | no grid |

Gate 0 was run once as registered, in `20261002T195255Z_m5_gate0_s8_seed42_sorted`: the test accuracy is 0.5407, and the registered rule is a test accuracy within 0.005 of 0.82778. Four limits go with this number. It comes from one run with one seed. The run used the numba CPU backend. The published `studies/clause_lm/src/m5_run.py` is a later revision than the one that ran. The commit of the upstream copy of Clause-Driven-LLM is recorded in project notes, not checked against upstream. The working notes of 2026-10-02 attribute the miss to a difference between two training backends of Clause-Driven-LLM on the basis of reading the code, and no run of the study tests that explanation.

The bars from the Penn Treebank validation split, each from a single run: the Kneser-Ney trigram model has perplexity 161.72, and 157.24 when the count votes of k-means word groups are added, with no Tsetlin Machine involved (`20261002T174323Z_clause_lm_kmeans200`); on word pairs that do not occur in the training text, the best statistical baseline, a class bigram model with k = 200, has AUC 0.7239 (`20261002T171904Z_pair_baselines_K1000`).
