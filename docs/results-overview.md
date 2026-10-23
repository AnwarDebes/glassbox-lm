# Results overview

This page gives, for each of the five studies, the status of what the study registered and its headline measurements. Every number is copied from the results page of its study and stands next to the id of the run that produced it; decision_layer keeps no run folders; its numbers stand next to the id of the registered test or rule, and the paragraphs below its table name the result files of its headline results. The results pages name the file of the run folder or the result file for each value and give the conditions that belong to it; where this page and a results page differ, the results page is right. Run folders, result files and datasets are not distributed, so no number can be recomputed from a clone. A run id can be placed in the [run index](run-index.md), and the limits of the evidence are on the page of [known limitations](known-limitations.md).

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

Status of the study: running, as preparation; the runs date from 2026-10-02 and 2026-10-14. [Results page](../studies/clause_lm/docs/results.md), [registration of M5 step 1](../studies/clause_lm/docs/prereg_M5_step1.md).

The study prepares a word-level language model whose clauses are AND-rules over OR-groups of words with count votes. Twelve ids carry a status: the milestones of the study design, with M5 counted as its two steps, and six items of the registration of M5 step 1, the only step that has a registration. Of the twelve, four are met, three are not met and five are not run. The tests of M1 and M2 have not been run; the runs that exist for them set the bars that the later models have to beat. M5 step 1 is not met: its registered question about s has a negative answer, which rests on an amended Gate 0 that is post hoc.

| Id | Status | Qualifier |
|---|---|---|
| M1, pair model on adjacent word pairs of Penn Treebank | not run | preparation only: the pair data and the statistical baselines ran, the pair model was written and not run |
| M2, clause-count language model with the OR-groups of M1 | not run | preparation only: the evaluator ran once with k-means word groups and no Tsetlin Machine |
| M3, full clause language model | not run | no code, no run |
| M4, scale and efficiency | not run | no code, no run |
| M5 step 1, the registered question about s for the Clause-Driven-LLM router on CLINC150 | not met | negative result by the registered rules: s* = 8 and the success rule not met; it rests on the amended Gate 0, which is post hoc, and on 3 seeds per value of s; one grid run did not have the GPU to itself for about 40 s |
| M5 step 2, OR-group word concepts as router features | not run | no code, no run |
| Gate 0, the loader is valid | not met | one completed run, numba CPU backend, seed 42; cause not confirmed; a retry was stopped without a score |
| Amended Gate 0 (amendment of 21:11 UTC on 2026-10-02) | met | post hoc: the amendment was written after the Gate 0 result was known; one run on 2026-10-14, torch backend on a GPU, seed 42, test accuracy 0.82933, inside the band but not the exact match that the registration expected; a first attempt ended at its start without a score |
| s-test grid | met | s in {8, 25, 50, 100} x seeds {42, 123, 456}, 12 runs including the amended Gate 0 run, all completed with the torch backend on a GPU; one run did not have the GPU to itself for about 40 s |
| Choice of s* | met | s* = 8, with no tie |
| Success rule | not met | mean test accuracy at s* 0.8201 against the required 0.848; seeds 123 and 456 below 0.828 |
| Negative-result rule | met | s* = 8; the gain over 0.828 is -0.79 points |

Gate 0 was run once as registered, in `20261002T195255Z_m5_gate0_s8_seed42_sorted`: the test accuracy is 0.5407, and the registered rule is a test accuracy within 0.005 of 0.82778. Four limits go with this number. It comes from one run with one seed. The run used the numba CPU backend. The published `studies/clause_lm/src/m5_run.py` is a later revision than the one that ran. The commit of the upstream copy of Clause-Driven-LLM is recorded in project notes, not checked against upstream. The working notes of 2026-10-02 attribute the miss to a difference between two training backends of Clause-Driven-LLM on the basis of reading the code.

The amended Gate 0 ran on 2026-10-14 with the torch backend on a GPU, in `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu`: test accuracy 0.8293, inside the band of 0.005 around 0.82778, with a median of 1 literal per clause against 1,021 in the numba run. This is consistent with the backend explanation, and the runs do not show which part of the two kernels makes the difference. The grid that followed gives, by the registered rules, s* = 8 and a mean test accuracy at s* of 0.8201, below the 0.848 of the success rule; the mean test accuracy falls from 0.8201 at s = 8 to 0.7274 at s = 100 (`20261014T222730Z_m5_summary_sorted_torchgpu`).

The bars from the Penn Treebank validation split, each from a single run: the Kneser-Ney trigram model has perplexity 161.72, and 157.24 when the count votes of k-means word groups are added, with no Tsetlin Machine involved (`20261002T174323Z_clause_lm_kmeans200`); on word pairs that do not occur in the training text, the best statistical baseline, a class bigram model with k = 200, has AUC 0.7239 (`20261002T171904Z_pair_baselines_K1000`).

## decision_layer

Status of the study: running; it started on 15 Oct 2026, and its last registered test so far, R24, ended on 23 Oct 2026. [Results page](../studies/decision_layer/docs/results.md), [registrations](../studies/decision_layer/registrations/) (each frozen one with its freeze record), [failure map](../studies/decision_layer/FAILURE_MAP.md).

The study asks where Tsetlin Machines fail on language, part by part, and what works instead when an LLM encoder does the language and a Tsetlin Machine makes the decision. Its registered tests cover a literal budget for a word router, an intent router on frozen LLM embeddings, a readable decision layer whose inputs have names, routing and a dialogue workflow learned from reward alone, clarify-or-route policies learned from delayed reward, and agents that choose the next action in customer-service and task dialogues (ABCD, STAR and MultiWOZ 2.2), at the end with no neural network at any stage. Each registered test was frozen before its confirmatory runs. The table has one row for each registered test and one for each numbered rule with a pass condition; a test with a single rule has one row, a re-run under the rules of an earlier registration (R2d, R2e, R7c) has one row, and the two registration texts that were never frozen have one row each. R20.4 has a second row for STAR, the headline question that stopped at its dev gate before the freeze, and R24.1 and R24.2 have one row per agent, as the registration of R24 states them. R20, R22 and R24 have no combined verdict by their registrations, so only their rules carry a status. Of the 107 ids, 72 are met, 31 are not met, 1 is not judged (R2c.3, whose verdict the results page does not draw) and 3 were not run. At the level of whole tests, phase 1 has 17 verdicts, 9 met and 8 not met (R1, inside the registration of R4, counted as one); in phase 2 every test is met as a whole except R9, R11 and R12, each of which has rules that are met. Several met ids need their qualifier to be read correctly: R5c.2 is met just; R14 and R17.2 are met on the point estimate with an interval that reaches past the margin; R6.1 and R8.1 are nearly built in, and the gains of R6.2 and R8.2 come from the out-of-scope episodes; R6.2, R12.1 and R15.2 are non-inferiority rules, so an interval above 0 there is not a registered superiority; R20.4 on ABCD is supporting evidence only; and every met comparison of an agent without a neural network with the bge-large agent is a non-inferiority rule (R19.1, R20.3, R22.3, R23.1, R24.1), as is the comparison of the C4 table with the potion table (R21.1); the one registered superiority rule of that comparison, R24.2 on MultiWOZ, is not met.

| Id | Status | Qualifier |
|---|---|---|
| R1, calibration and out-of-scope detection of the router (rules in the registration of R4) | met | R1.1 and R1.2 met; the registration has no combined rule, and the study README counts R1 among the nine met tests of phase 1 |
| R1.1, calibration | met | expected calibration error after one temperature fitted on val 0.013 to 0.025 in every cell (bar 0.05); 0.215 to 0.336 before the temperature |
| R1.2, out-of-scope detection on CLINC150 | met | the top class sum is non-inferior to the maximum softmax of LR with both encoders (AUROC 0.9623 against 0.9625 with MiniLM, 0.9728 against 0.9694 with bge-base); above LR in all 10 seeds with bge-base only |
| R2, the literal budget on the CLINC150 word router | not met | R2.1 met, R2.2 failed by one clause; the cause found later is most likely a tie in the budget's selection, shown in R2c and R7b; repeated as R2e, met |
| R2.1, accuracy | met | budget minus default TM +5.86 [+5.28, +6.45] points |
| R2.2, clause length | not met | seed 7 has one clause of 33 literals among 40,000 clauses over 10 seeds; the rule allows at most 32 |
| R2.3, against LR on the same bits, reported separately | met | -0.26 [-0.72, +0.20]: non-inferior (bar -0.5), not ahead |
| R2b, the budget form of the tmu library | not met | R2b.1 and R2b.3 not met, R2b.2 met |
| R2b.1, accuracy | not met | -25.51 [-26.51, -24.53] points against the default TM |
| R2b.2, clause length | met | median literals 1 to 2, 95th percentile at most 40 (2 x L* = 64) |
| R2b.3, against LR on the same bits | not met | -31.64 [-32.81, -30.46]: not non-inferior |
| R2c, the budget with a start state that obeys it (Banking77 and HWU64 word routers) | not met | R2c.1 met, R2c.2 failed by two clauses of 33 literals (the same tie); repeated as R2d, met |
| R2c.1, accuracy | met | +4.71 [+4.00, +5.44] (Banking77) and +4.09 [+2.79, +5.41] (HWU64) points over the default TM |
| R2c.2, clause length | not met | Banking77 seed 6 holds two clauses of 33 literals |
| R2c.3, against LR on the same bits, reported separately | not judged | the results page gives B minus C, -1.43 [-2.05, -0.80] on Banking77 and -5.19 [-6.50, -3.93] on HWU64, and draws no verdict under this id; the registration's bar is -0.5, and it did not expect non-inferiority |
| R2d, the budget arm of R2c again, with the exact selection | met | R2c.1 met in both datasets (+4.81 [+4.08, +5.53] and +3.96 [+2.69, +5.27]) and R2c.2 met (longest clause 32); it checks the length fix: 16 of its 20 models are bit-identical to those of R2c |
| R2e, R2 again with the fixed budget (CLINC150) | met | R2.1 met (+5.82 [+5.24, +6.41]) and R2.2 met (longest clause 32); R2.3 non-inferior, not ahead (-0.30 [-0.76, +0.17]); second use of the CLINC150 test split by a lexical router |
| R4, the TM router against LR on frozen embeddings, 3 datasets x 2 encoders | not met | R4.1 and R4.2 not met |
| R4.1, the composite | not met | the composite is within 0.5 points of LR in 3 of 6 cells; the rule needs 5 |
| R4.2, the single machine | not met | it trails LR in every cell, by 0.56 to 2.41 points: the gap is not closed |
| R4b, the same with 16,000 clauses | not met | R4b.1 and R4b.2 not met, R4b.4 met |
| R4b.1, the composite | not met | within 0.5 points of LR in 3 of 6 cells; the rule needs 5 |
| R4b.2, the single machine | not met | within 0.5 points in 3 of 6 cells: the gap is not closed |
| R4b.4, the rules of R1 on the 16,000-clause machine | met | calibration error 0.013 to 0.027 after one temperature; out-of-scope AUROC above LR in all 10 seeds with both encoders |
| R5, routing learned from reward alone (bandit) | not met | R5.1 and R5.2 hold on 1 of 3 tasks; the rules need 2 |
| R5.1, against the better of LinUCB and epsilon-greedy logistic | not met | met on CLINC150 domains only (+0.77 [+0.54, +0.99]); HWU64 -6.96 [-9.06, -5.11], Banking77 -42.8 |
| R5.2, against the supervised TM | not met | 1.004, 0.930 and 0.551 times the supervised TM on CLINC150 domains, HWU64 and Banking77 (bar 0.97) |
| R5b, Boltzmann exploration and 15 passes | not met | R5b.1 not met on Banking77; registered after the results of R5 and after exploratory val runs of the fix |
| R5b.1, against LinUCB | not met | HWU64 +1.69 [+1.23, +2.03] (met), Banking77 -3.16 [-3.52, -2.83] (not met) |
| R5b.2, against the supervised TM | met | 0.999 (HWU64) and 0.975 (Banking77) |
| R5c (its rule R5c.1), 150 actions | met | 0.998 times the supervised TM |
| R5c.2, against LinUCB, a separate verdict | met | -0.98 [-1.20, -0.80] (bar -1.0): met, just; 2 of the 5 seeds are more than a point behind; the rule was moved out of "met" after the val run had shown -1.03 |
| R5d, Banking77 with 30 passes | met | third use of the Banking77 test split by a bandit |
| R5d.1, against LinUCB | met | -0.71 [-0.96, -0.33] (bootstrap over seeds); one of five seeds is just over a point behind |
| R5d.2, against the supervised TM | met | 0.996 |
| R6, clarify-or-route with delayed reward (TD) | met | the chosen TM-Q setting came from a candidate queued after the freeze; a sensitivity run (post hoc) gives the same conclusion |
| R6.1, credit assignment back in time | met | TM-Q minus the myopic TM +0.399 [+0.392, +0.405]; nearly built in |
| R6.2, against the tuned rule | met | +0.042 [+0.035, +0.049], a non-inferiority rule (bar -0.01); all from out-of-scope episodes; in scope the rule is better (-0.014 [-0.020, -0.010]); the linear Q-learner gets the same return |
| R6.4, against linear Q | met | +0.0011 [-0.0035, +0.0049]: non-inferior |
| R7, the readable decision layer (bge-large; CLINC150, Banking77, HWU64) | met | post hoc, it agrees with the one-rule router "route to the nearest whitened centre" on 99.2 to 99.7% of test queries |
| R7.1, against LR on the same readable bits | met | -0.22 [-0.40, -0.03], -0.12 [-0.43, +0.19], -0.29 [-0.85, +0.24]; met in 3 of 3 (the rule needs 2) |
| R7.2, the composite against LR on the raw embedding | met | -0.98 [-1.34, -0.61], -0.99 [-1.69, -0.29], -0.88 [-2.14, +0.37] (bar -1.0); met in 3 of 3, two of them just inside the margin |
| R7b, the readable layer with every clause at most 16 literals | not met | R7b.3 failed by three clauses of 17 literals (the tie of R2c); designed after the test results of R7, on the same test splits; L = 16 was chosen after both val runs, with no rule fixed before them |
| R7b.1, against LR on the same readable bits | met | -0.00 [-0.16, +0.15], +0.03 [-0.19, +0.25], -0.13 [-0.61, +0.33]; met in 3 of 3 |
| R7b.2, the composite against LR on the raw embedding | met | -0.80 [-1.14, -0.44], -0.88 [-1.57, -0.19], -0.65 [-1.86, +0.51]; met in 3 of 3 |
| R7b.3, clause length | not met | three clauses of 17 literals (CLINC150 seed 2, HWU64 seeds 6 and 10) |
| R7b.4, out-of-scope detection, a separate verdict | met | CLINC150 AUROC 0.950 against 0.916 for R7 (bar 0.936) |
| R7c, the machine of R7b again, with the exact selection | met | R7b.1, R7b.2 and R7b.3 met (longest clause 16) and R7b.4 met (0.950); third use of these test splits with this view; 14 of the 30 models are bit-identical to those of R7b |
| R8, a TM policy over four steps | met | with no rule, TM-Q is 0.022 below the tuned rule and 0.021 below linear Q |
| R8.1, credit assignment over four steps | met | +0.369 [+0.356, +0.383] over the myopic TM; nearly built in |
| R8.2, against always asking | met | +0.164 [+0.152, +0.176], from the out-of-scope episodes; in scope +0.006 [-0.003, +0.014] |
| R9, the ABCD next action with memory literals | not met | R9.1 not met; R9.2 and R9.3 met |
| R9.1, memory literals | not met | +1.09 [+0.72, +1.47]: the control read the history as words, a design error; R9b repeats the test |
| R9.2, against LR on the same bits | met | +0.36 [-0.29, +1.02] (bar -0.5); LR chose the smallest C of its grid; on an extended grid checked afterwards (post hoc) the conclusion does not depend on the grid edge |
| R9.3, against an MLP on the same bits | met | +0.31 [-0.23, +0.83] (bar -1.0) |
| R9.4, own verdict: the composite against the opaque MLP on raw embeddings | met | +0.10 [-0.62, +0.82] (bar -1.0) |
| R9.5, own verdict: the TM beats LR on the same bits | not met | +0.36 [-0.29, +1.02]: ahead, with the interval through 0 |
| R9b (its rule R9b.1), memory literals against a control with no record of the earlier actions | met | +18.65 [+17.20, +20.10] points (84.0% against 65.4%); frozen after the first seeds of R9; second use of the ABCD test split |
| R10, the workflow learned from a 0/1 reward alone | met | logged conversations are replayed: the memory bits are the human agent's logged presses |
| R10.1, against the supervised TM | met | 0.988 (difference -1.04 [-1.49, -0.59]) |
| R10.2, memory, with reward alone | met | +22.87 [+21.37, +24.39] points |
| R10c, the TM bandit against LinUCB | met | LinUCB is ahead for the first 7 to 11 passes and earns more reward while learning; one task, one input view, 15 passes |
| R10c.1, at the dev setting (alpha 0.5) | met | +5.51 [+4.66, +6.39] points |
| R10c.2, at the best alpha on test | met | +3.00 [+2.18, +3.81] against alpha 1.0, chosen on test |
| R11, R7 and R7c on MASSIVE en-US, a related dataset | not met | R11.1 and R11.3 not met, R11.2 met; about a third of the MASSIVE test texts are in HWU64 train, with different labels |
| R11.1, R7 replicates | not met | -0.65 [-1.07, -0.22] against LR on the same bits (bar -0.5); its composite part holds (-0.92 [-1.80, -0.03], bar -1.0) |
| R11.2, R7c replicates | met | -0.35 [-0.65, -0.06], composite -0.44 [-1.31, +0.45], no clause over 16 literals; with less room: the rules judge point estimates, and neither interval lies inside its margin |
| R11.3, calibration | not met | 0.060 against the bar 0.05 |
| R12, trained in English, scored in Norwegian (MASSIVE nb-NO) | not met | R12.2 not met; one encoder, one view, one language pair |
| R12.1, against LR on the same bits | met | +0.72 [+0.30, +1.16]; a non-inferiority rule whose interval also lies above 0 |
| R12.2, the cost of readability | not met | composite minus LR on the raw embedding -1.80 [-2.91, -0.67] (bar -1.0); in English the same gap is -1.66, so most of the cost comes with the readable view rather than with the language (post hoc) |
| R12.3, the share of English accuracy kept | met | 0.949 (82.2% in Norwegian) |
| R13, the closed loop: the memory holds the model's own presses | met | only the memory bits are the model's own; the text, including the human agent's utterances, is replayed from the log, so a fully closed loop probably costs more |
| R13.1, memory in the closed loop | met | 80.7%, +15.34 [+13.80, +16.91] over no memory |
| R13.2, the cost of its own mistakes | met | -3.31 [-3.95, -2.71] (bar -4.0) |
| R14 (its rule R14.1), the TM bandit against a neural bandit | met | -0.45 [-1.04, +0.16], judged on the point estimate (bar -1.0); the interval reaches just past the margin, and is [-1.33, +0.44] with the training seeds also resampled (post hoc) |
| R15, replay for the TM bandit | met | uneven comparison: the TM gets 5 updates per round (replay 4), the neural bandit 1 and no replay |
| R15.1, online accuracy | met | +6.35 [+5.87, +6.81] points (bootstrap over seeds) |
| R15.2, against the neural bandit at the end | met | +0.52 [+0.03, +1.02], a non-inferiority rule; [-0.26, +1.28] with the training seeds also resampled (post hoc), so "ahead" is not established |
| R15.3, own verdict: online gap to the neural bandit | met | -1.90 [-2.34, -1.49] (bar -3.0) |
| R16 (its rule R16.1), the hierarchical TM against the flat machine of the same package | met | +9.40 [+8.35, +10.41] points (81.7% against 72.3%), 3 seeds each; size and settings differ as well as the structure, so the gain is not the AND-OR structure's alone; 2.28 points below this study's coalesced flat TM (no rule) |
| R16b (its rule R16b.1), against a flat machine of matched size | met | +7.02 [+6.01, +8.04] points (81.7% against 74.7%); one flat setting tried; T, s and the vote still differ |
| R17, the reward-learned TM in the closed loop | met | memory bits only; the text is replayed from the log |
| R17.1, memory in the closed loop | met | 79.1%, +18.96 [+17.30, +20.61] over no memory |
| R17.2, the cost of its own mistakes | met | -3.70 [-4.44, -2.99]: met on the point estimate (bar -4.0); the interval reaches past the margin |
| R18 (its rule R18.1), memory literals on STAR | met | +15.43 [+14.38, +16.51] points (62.7% against 47.2%); first use of STAR's test split, open loop; the control still reads the wizard's last message; with no rule, the TM is 3.26 points behind LR and 3.15 behind an MLP on the same bits |
| R19, an agent with no neural network at run time (STAR, ABCD) | met | the word table was distilled once, offline, from a transformer; only the run time is free of neural networks; open loop; second use of STAR's test split, eleventh of ABCD's |
| R19.1, against the bge-large agent | met | strong form on both datasets: STAR +0.06 [-0.63, +0.74], ABCD -1.00 [-1.62, -0.40] (margin 2.0 points) |
| R19.2, memory | met | +15.05 [+14.07, +16.05] (STAR) and +20.79 [+19.32, +22.29] (ABCD) |
| R20.1, STAR: memory in the closed loop | met | +7.53 [+6.51, +8.58]; 4th use of STAR's test split |
| R20.2, STAR: its own mistakes cost at most 4 points | not met | -7.51 [-8.34, -6.72], as expected from dev (-7.57); the bar of R13 was kept |
| R20.3, STAR: the word table against bge-large in the same loop | met | strong form: +1.96 [+1.07, +2.85]; superiority was not a registered rule |
| R20.4, ABCD: memory for the customer-only agent in the closed loop | met | supporting evidence only: +23.94 [+22.20, +25.67], at 72.6%; no text written by the human agent; 12th use of ABCD's test split |
| R20.4 on STAR, the question named as the headline | not run | stopped at its dev gate S4 (+4.50 against +5.0, seed 1); STAR's test split was not spent on it; a dev-only line of the failure map |
| R21, an agent with no neural network at any stage (a word table counted from C4 text) | met | met on both datasets, as the headline needs; STAR's 5th use, ABCD's 13th; open loop only |
| R21.1, against the potion agent of R19 | met | strong form on both: STAR -0.10 [-0.65, +0.45], ABCD -0.29 [-0.74, +0.17]; against the bge-large agent, with no rule, -0.04 and -1.29 |
| R21.2, memory | met | +14.73 [+13.73, +15.74] (STAR) and +20.63 [+19.17, +22.11] (ABCD) |
| R22.1, STAR: the TM's s tuned on dev (16) against s 8 | met | +2.70 [+2.17, +3.24]; s chosen on STAR dev; 6th use of STAR's test split |
| R22.2, against a tuned LR on the same bits | met | strong form: -0.09 [-0.90, +0.73] (bar -1.0) |
| R22.3, against the equally tuned bge-large agent | met | strong form: +1.64 [+0.94, +2.35] (bar -2.0); superiority was not registered |
| R23 (its rule R23.1), ABCD: equal tuning | met | strong form: C4 agent at s 16 minus bge-large agent at s 24 -0.92 [-1.61, -0.26] (bar -2.0); the interval lies below 0; 14th use of ABCD's test split; the bge-large pick is at the edge of the grid |
| R24.1 for the C4 agent (the headline) | met | strong form: +0.53 [-0.18, +1.23] against the bge-large agent on MultiWOZ 2.2 test, its 1st use (margin 1.0); two-level interval (no rule) [-0.32, +1.40]; all three agents use random negatives, chosen on MultiWOZ dev |
| R24.1 for the refined agent (secondary) | met | strong form: +0.05 [-0.65, +0.72]; two-level [-0.83, +0.95]; network-free at run time, teacher at training time (bge-base-en-v1.5) |
| R24.2 for the C4 agent: superiority | not met | the interval [-0.18, +1.23] is not above 0, so no "better than" |
| R24.2 for the refined agent: superiority | not met | [-0.65, +0.72] |
| R25, an agent on the teacher-refined table (STAR, ABCD) | not run | never frozen; its dev go-bars failed (e82: STAR +0.14 against +1.0, ABCD +0.56 against +0.7), so no test split was read for it |
| The superiority test on MultiWOZ written on 2026-10-19 (R24_multiwoz_superiority.md, rev 0 of R24) | not run | never frozen; superseded by the frozen R24, whose R24.2 asks the superiority question under random negatives and is not met |

The rows were counted id by id from the results page, and from the registrations where the results page is silent. Some registered items are reporting rules without a pass condition and carry no status of this kind: R4.3 and R4b.3, which name an arm "ahead" of LR in a cell where its interval lies above 0 (in R4 the read-out arm in one cell, Banking77 with bge-base, +0.45 [+0.02, +0.90], which the results page calls fragile; in R4b no arm), and R6.3, the effect of the memory literals in R6, +0.0006 [-0.0019, +0.0034], reported with no threshold. The encoder ladder that the failure map calls R3 is part of R4 and has no rule of its own.

decision_layer keeps no run folders and no run ids. Its registered results are files in `results/confirm/<registration>/` of the study folder, and its run logs are in `logs/`; neither is distributed, and the files below are named relative to `studies/decision_layer/`. R23 was the 14th use of ABCD's test split and R22 the 6th use of STAR's, so the results of R22 and R23 rest on splits that had been read many times.

On frozen LLM embeddings a single 4,000-clause Tsetlin Machine router trails LR on the same embedding in every cell of R4, by 0.56 to 2.41 points; after one temperature its calibration error is 0.013 to 0.025 (R1.1), and with 16,000 clauses it flags out-of-scope queries better than LR in every seed with both encoders (R4b.4). The readable layer of R7 comes within 0.3 points of LR on the same named inputs, and post hoc it agrees with the one-rule nearest-centre router on 99.2 to 99.7% of test queries: its accuracy comes from the encoder and the whitened centres (run files in `results/confirm/R7/`). From reward alone the TM bandit reaches 0.998 and 0.996 of the supervised TM with 150 and 77 actions (R5c, R5d), and online it makes 2.6 to 3.4 times the errors of LinUCB (online accuracy 78.1% against 93.6% in R5c, 84.5% against 94.0% in R5d).

On the ABCD workflow memory literals lift the next-action accuracy of the TM from 65.4% to 84.0% (R9b; run files in `results/confirm/R9b/`); with labels the TM is level with LR and an MLP on the same named inputs there (R9.2, R9.3), and on STAR the memory result replicates (+15.43, R18) while the TM is 3.26 points behind LR on the same bits. From a 0/1 reward per press the TM reaches 0.988 of the supervised TM (R10), and the supervised TM keeps 80.7% with its own presses as memory (R13).

An agent with no neural network at run time is level with the bge-large agent on STAR test (+0.06 [-0.63, +0.74]) and 1.0 point below it on ABCD (-1.00 [-1.62, -0.40]), inside the registered 2-point margin (R19; `results/confirm/R19star/report_R19star.json`, `results/confirm/R19abcd/report_R19abcd.json`). One of its decisions takes a median of 29.5 ms on one CPU thread against 806 ms for the bge-large agent, a rough timing on a shared machine; the bge-large agent would be faster on a GPU, which was not measured, and the speed-up of the text step is that of Model2Vec (`results/explore/static_speed_R19.json`). With a word table counted from C4 text, so with no neural network at any stage, an agent stays within 0.10 (STAR) and 0.29 (ABCD) points of the potion agent (R21; `results/confirm/R21star/`, `results/confirm/R21abcd/`). With the TM's s tuned on dev for both agents in the same way, the C4 agent is +1.64 [+0.94, +2.35] against the bge-large agent on STAR (R22.3; `results/confirm/R22star/`) and -0.92 [-1.61, -0.26] on ABCD (R23.1; `results/confirm/R23abcd/`). On MultiWOZ 2.2 test, with random negatives for all three agents, a rule chosen on MultiWOZ dev on the refined agent, the C4 agent is +0.53 [-0.18, +1.23] against the bge-large agent (R24.1, strong form), R24.2 is not met, and the TM is below LR on the same bits for all three agents: C4 -1.55 [-2.32, -0.79], refined -2.49 [-3.33, -1.64], bge-large -1.89 [-2.69, -1.10], with no rule (`logs/R24_report.out`). On dev the same rule costs STAR -11.9 and ABCD -4.0 points (measured on the refined agent, e83), so it is a MultiWOZ setting.

## tally

Status of the study: running; its run folders date from 2026-10-15 to 2026-10-17. [Results page](../studies/tally/docs/results.md), [registration T0](../studies/tally/registrations/T0.md).

The study holds the first steps of the design called Tally ([research/design.md](../research/design.md)): a count floor, a language model built from counts with no Tsetlin Machine in it, and a test bench, places in that model where a Tsetlin Machine is compared on the same bits with a twin that only counts and with standard learners. One test is registered, T0: on Penn Treebank next-word prediction with the tmu library, do counted votes on the clauses of a Tsetlin Machine beat its learned votes? It repeats H8.1 of tm_language_model with the library of the Tsetlin Machine group in place of that study's own engine. Three ids carry a status: T0 and its two amendments, which fix the settings of the two machines by the tuning rule of T0. Of the three, two are met and one is not judged; for an amendment, which is a step of the procedure, met means that the step was carried out as registered. The other lines of the study (B1, B6, A1, A2 and C3) have code and smoke, timing, baseline or exploratory runs, and no registration.

| Id | Status | Qualifier |
|---|---|---|
| T0, the counted read-out against the learned votes on the clauses of the coalesced machine, in bits per word on the dev-test split of Penn Treebank | not judged | the 20 final runs of the two machines (seeds 1 to 10) and the 10 logistic-regression runs have finished; the registered report, `src/tally/t0/report.py`, has not been run on them, so no number of the dev-test split exists |
| Amendment of the coalesced settings, frozen at 18:24:29 UTC on 2026-10-15 | met | s = 20 and 9 epochs by the tuning rule, frozen after the three coalesced tuning runs had ended and before the first coalesced final run; the chosen s is the largest value of the grid {5, 10, 20}, and its best epoch is one below the cap of 10 |
| Amendment of the flat settings, frozen at 21:14:27 UTC on 2026-10-15 | met | s = 5 and 1 epoch by the tuning rule, frozen after the three flat tuning runs had ended and before the first flat final run; the chosen s is the smallest value of the grid, and all three flat tuning runs had their best score after the first epoch; the flat tuning run with s = 10 started before T0 was frozen, the other two after it |

No measured value of tally stands on this page. T0 has no verdict yet, and the results page gives no number of the baseline check of A1 or of the LSTM runs of C3, which score the validation split and carry no claim. The tuning scores that fixed the settings of T0 are on the [results page](../studies/tally/docs/results.md#tuning-runs).
