# Results of the hierarchical_tm study

The study ran from 1 to 3 Oct 2026 and tested the procedure of the supervisor's project proposal (unpublished), whose project title is "Logical self-supervised learning of language with hierarchical Tsetlin machines". A hierarchical Tsetlin Machine, whose rules are conjunctions of OR-groups of words, is trained to say whether two texts belong together, and its OR-groups are then read as word groups. The procedure was run on artificial pair data with hidden word groups (L1 to L3), on two sentence families built from slides 6 and 10 of the project proposal, on news topics (AG News) and on Wikipedia paragraph pairs (same article, next paragraph, and next paragraph with every shared word removed). The pass rules were frozen in [gates.md](../gates.md) on 1 Oct 2026, before these runs.

The study ended mostly negative. Checks 1a, 1b, 3 and 4a were met, Check 4b was met on one of its three arms with a small effect, and Checks 2a, 2b and 2c were not met: of the three artificial rungs only the easiest met its check (the second is borderline and missed the accuracy bar in its 1,000-pass fallback, and the third recovered no hidden group), no learned rule captured either sentence family, and on news topics the hierarchical model was more than the allowed 2 points below a flat model. The model does learn relations between Wikipedia paragraphs, but its single final test (5 seeds, `report.json` of `20261002T193513Z_final_test_main_table`: test AUC 0.735 on same-article pairs and 0.750 on next-paragraph pairs) is below word overlap (0.817 and 0.857) and below the best simple baseline on all three data sets.

## How to read this page

Each check of gates.md has one section below, in the order of that file, with its pass rule in short, the measurements and one status: met, not met, not judged or not run. A qualifier (for example "borderline") stands next to the status. Qualifiers, quoted wordings and readings of a result are those noted when the result was judged during the study; where a verdict rests on such a note and not on a file of a run, the section says so. The section "All checks at a glance" repeats the statuses in one table, and "Registered rules and predictions" says what came out of every registration of [registrations.md](registrations.md). The slide-6 paraphrase test is not a check of gates.md and has its own section.

Every measurement names its run and is read from a file of that run; `metrics.json` is meant where no file is named. Values are rounded to three decimals unless more digits matter. Scores are those of gates.md: AUC is computed on the difference of the two class sums, and tuned accuracy is defined there, with a cut-off chosen on dev data. The one exception is the tuned accuracy of the final-test table, whose cut-off is chosen on the test pairs. Intervals in square brackets are 95% intervals from the article bootstrap of gates.md unless stated otherwise. All numbers are dev numbers except in the section on the final test.

Times of registrations and decisions are given in Norwegian time (CEST) as registered, with UTC in brackets. A run id starts with the UTC time at which the run started: `20261002T192818Z` is 19:28:18 UTC on 2 Oct 2026, which is 21:28:18 CEST.

Run folders and datasets are not distributed with the repository, and neither are the sentence files of the slide-6 and slide-10 tests. No number on this page can therefore be recomputed from a clone; each number can only be compared with the file of the run folder named next to it. The [run index](../../../docs/run-index.md) lists the 319 run folders of this study. 315 of them ended with exit status 0. The other four (`20261002T010551Z_e2_L2_s_compare_posthoc_exploratory`, `20261002T031952Z_saveload_check_exploratory`, `20261002T134702Z_final_test_report_devcheck_exploratory`, `20261003T015436Z_e3fb_flat_C768_check2b_eval_and_probe_exploratory`) ended with a non-zero exit status, and each was followed by a repeat run. Five runs of this study are flagged as smoke runs in the run index, and no number is taken from them. The 27 probe runs of 2026-10-01 whose names contain `feas_slide11` came before this study and sit in the run folder of the tm_language_model study. They used pair data with a sampling bias and are not used.

## Check 1a (G0): the package demo is reproduced

Rule in short: the MultiConcept demo of the package PyHierarchicalTsetlinMachineCUDA (Granmo), with its defaults, reaches at least 0.95 test accuracy on each of 3 seeds; a "one word decides" test, in which the label says whether paragraph A contains one planted word, reaches AUC of at least 0.98; and a numpy recomputation of the clause outputs agrees with the package, exactly under binary inference and clause by clause under the default inference.

| part | run | measurement |
|---|---|---|
| demo, seed 1 | `20261001T091807Z_e1_demo_seed1` | final accuracy 1.0 after 100 epochs, 1 repeat |
| demo, seed 2 | `20261001T092250Z_e1_demo_seed2` | final accuracy 1.0 |
| demo, seed 3 | `20261001T092751Z_e1_demo_seed3` | final accuracy 1.0 |
| one word decides | `20261001T092825Z_e1_p1word_hier` | dev AUC 1.0, accuracy 0.9999; always answering "no" gives 0.9352 |
| recomputation, binary inference | `20261001T093345Z_e1_verify_binary` | class sums equal on all of 500 pairs for both classes, and every clause output agrees |
| recomputation, default inference | `20261001T093407Z_e1_verify_default` | class sums equal on all of 500 pairs for both classes, and every clause output agrees |

Status: met. The demo ran with one repeat per seed. The demo's own default is 10 repeats per call; the single repeat was kept, because the rule requires 3 seeds.

## Check 1b (data): no shortcuts

Rule in short, for each pair version: at most about 3 vocabulary words show abs(z) > 3 in a test of single-paragraph word frequency between positives and negatives; a logistic regression that sees one paragraph only has an accuracy whose 95% interval includes 0.50; the same holds for a logistic regression on partner-shuffled pairs; and for the version with shared words removed, word overlap is 0 in every pair and a classifier on the two paragraph lengths alone stays at 0.50.

The table gives the checks of the data that the study used. The intervals here are those printed by the check script.

| pair version | words with abs(z) > 3 | one paragraph alone | partner-shuffle | length only (logistic regression / boosted trees) | verdict printed by the check script | run and file |
|---|---|---|---|---|---|---|
| same article (`sa`) | 2 | 0.497 [0.490, 0.504] | 0.502 [0.495, 0.509] | not applicable | pass | `20261001T094249Z_data_checks_final_exploratory`, `checks.md` |
| next paragraph (`np`) | 3 | 0.499 [0.492, 0.506] | 0.493 [0.487, 0.500] | not applicable | pass | same run and file |
| shared words removed, swap design (`ns`), a side variant | 0 | 0.499 [0.492, 0.506] | 0.498 [0.492, 0.505] | 0.502 [0.495, 0.509] / 0.500 [0.493, 0.507] | pass | same run and file |
| shared words removed, all pairs (`ns_all`), the main synonym data | 4 | 0.503 [0.496, 0.510] | 0.504 [0.497, 0.511] | 0.500 [0.493, 0.507] / 0.503 [0.496, 0.510] | FAIL | `20261001T094347Z_data_checks_ns_all_v2_exploratory`, `checks.md` |

The first build did not pass. In `20261001T092321Z_data_checks_exploratory` (`checks.md`) the next-paragraph version had 5 skewed words and a one-paragraph accuracy of 0.507 [0.500, 0.514], and the first version with shared words removed had 690 skewed words, a one-paragraph accuracy of 0.661 [0.654, 0.668] and a length-only accuracy of 0.679. Both were rebuilt; the [data card](data_card_wikipedia_pairs.md) describes the rebuilds. Two later builds did not pass either. The second build of the version with shared words removed (`ns`) still printed FAIL, with 15 skewed words, a one-paragraph accuracy of 0.516 [0.509, 0.523] and a partner-shuffle accuracy of 0.513 [0.506, 0.520] (`20261001T093113Z_data_checks_v2_exploratory`, `checks.md`); the build that passed is the one in the table. The first build of `ns_all` printed FAIL on the length-only check, with 0.530 [0.523, 0.537] for the logistic regression and 0.532 [0.525, 0.539] for the boosted trees, and had 1 skewed word (`20261001T094221Z_data_checks_ns_all_exploratory`, `checks.md`); the table gives the check of its second build.

For `ns_all` the check script prints FAIL because 4 skewed words are more than 3. Decision of 1 Oct 2026, 20:10 CEST (18:10 UTC), an addendum of gates.md: 4 counts as "at most about 3", and `ns_all` is the main synonym data, with the safeguard that the verdict of Check 4b leaves out every word group that contains one of the four words. The count of 4 was known when the decision was taken. Word overlap is 0 in `ns_all`: every baseline that uses shared words has AUC 0.500 there (same file).

Status: met. Qualifiers: the next-paragraph data passed after one rebuild, and the main synonym data is borderline on the skewed-word count and passes only by the decision of 1 Oct 2026.

## Check 2a (G1): artificial tests L1 to L3

Rule in short: for each of L1, L2 and L3, on 2 of 3 seeds, hierarchical accuracy is at least 0.90 and group recovery is at least 0.80; at L3, accuracy on held-out same-group word pairs is at least 0.85 on 2 of 3 seeds; the one-paragraph check passes on at least 20,000 fresh pairs per rung; hierarchical against flat is reported whatever it shows. On a fail the rung is run for 1,000 passes (cap 3 GPU-hours) and is then reported as "not learned at this size".

gates.md does not say which accuracy counts. The decision of 2 Oct 2026, 21:27 CEST (19:27 UTC) fixes the split-averaged cross-fitted accuracy, the mean over 200 stratified splits. This measure was added after the result of the one pre-registered cross-fit split was known, so the choice is post hoc and the table shows every reading. Only the verdict on L2 depends on it.

The readings "plain", "tuned" and "cross-fitted" are from `20261002T053510Z_e2_crossfit_acc_posthoc`, "split-averaged" from `20261002T053708Z_e2_crossfit_split_avg_all_posthoc_exploratory`, and group recovery from the file `recovery.json` of the run named in the last column. Each cell gives seeds 1 / 2 / 3. The training runs of each row are named in the two report files.

| rung and setting | dev AUC | plain accuracy | tuned accuracy | cross-fitted, the one registered split | split-averaged cross-fitted | group recovery | recovery from |
|---|---|---|---|---|---|---|---|
| L1, `s` 21.1, 100 passes | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | `20261001T234330Z_e2_recovery_posthoc_exploratory` |
| L2, `s` 21.1, 100 passes | 0.915 / 0.904 / 0.930 | 0.523 / 0.522 / 0.572 | 0.847 / 0.846 / 0.8755 | 0.8455 / 0.846 / 0.8755 | 0.845 / 0.845 / 0.873 | 0.625 / 0.5 / 0.75 | `20261001T234330Z_e2_recovery_posthoc_exploratory` |
| L2, `s` 100, 100 passes | 0.964 / 0.968 / 0.950 | 0.883 / 0.898 / 0.8495 | 0.903 / 0.908 / 0.874 | 0.903 / 0.908 / 0.874 | 0.902 / 0.907 / 0.871 | 0.0 / 0.25 / 0.0 | `20261002T010559Z_e2_L2_s_compare_posthoc_exploratory` |
| L2, `s` 21.1, 1,000 passes (the fallback) | 0.981 / 0.962 / 0.960 | 0.911 / 0.5025 / 0.634 | 0.9215 / 0.897 / 0.9015 | 0.918 / 0.896 / 0.900 | 0.9185 / 0.8943 / 0.8985 | 0.875 / 1.0 / 0.875 | `20261002T043039Z_e2_L2_p1000_recovery_posthoc_exploratory`, `20261002T050238Z_e2_L2_p1000_s2_recovery_purity_posthoc_exploratory`, `20261002T053216Z_e2_L2_p1000_s3_recovery_purity_posthoc_exploratory` |
| L3, `s` 800, 300 passes | 0.916 / 0.908 / 0.908 | 0.8365 / 0.821 / 0.821 | 0.8365 / 0.821 / 0.821 | 0.8365 / 0.821 / 0.821 | 0.836 / 0.819 / 0.821 | 0.0 / 0.0 / 0.0 | `20261002T023042Z_e2_L3_s800_posthoc_exploratory` |

L1 is met on 3 of 3 seeds.

L2 is not met; its qualifier is "borderline". At 100 passes it fails at both values of `s`: at `s` 21.1 on accuracy and on recovery, at `s` 100 on recovery. It comes close only through the 1,000-pass fallback, with the cap of that fallback read per rung (the reading has no clock time of its own and is first cited in the entry that reports seed 1 at 1,000 passes; under a reading as one total, seeds 2 and 3 would not count). Under the split-averaged accuracy 1 of 3 seeds reaches 0.90. Seed 3 lies exactly on the bar on the one pre-registered split: its two halves of 2,000 pairs give 0.9035 and 0.8965, which is 3,600 of 4,000 pairs, and the file stores 0.8999999999999999. Over the 200 splits seed 3 reaches 0.90 in a share of 0.305 of them, seed 1 in all and seed 2 in none. The fixed wording is: "L2 at 1,000 passes is borderline. It meets Check 2a's accuracy bar on tuned accuracy and on the pre-registered cross-fit split (seed 3 exactly 0.900), but not robustly: averaged over random splits, seed 3 is 0.8985, and plain accuracy fails. Group recovery passes on all 3 seeds (0.875, 1.0, 0.875), and purity is 0.92 to 0.94." The purity of all groups is 0.92 / 0.935 / 0.92 (`purity.json` of `20261002T043139Z_e2_L2_p1000_purity_posthoc_exploratory` and of the two runs for seeds 2 and 3 named in the table). On seed 1, 17 OR-groups are counted and a share of 0.575 of the OR-groups contains an empty option and is not counted.

Raising `s` from 21.1 to 100 at L2 raised the AUC and lowered recovery and purity on all three seeds. Tuned accuracy rose on seeds 1 and 2 and not on seed 3 (0.8755 against 0.874). The purity of all groups is 0.68 / 0.66 / 0.68 at `s` 21.1 and 0.487 / 0.513 / 0.467 at `s` 100 (`purity.json` of `20261002T010559Z_e2_L2_s_compare_posthoc_exploratory`). The effect on AUC, purity and recovery is a finding; its mechanism is a hypothesis.

L3 is not met. At `s` 21.1 the dev AUC is 0.514 / 0.494 / 0.516 after 200 passes (`20261001T234330Z_e2_recovery_posthoc_exploratory`, `recovery.json`) and 0.493 / 0.504 / 0.508 after 1,000 passes (same file for seeds 1 and 2; `20261001T232651Z_e2_L3_hier_s3_p1000` for seed 3), which the fail path of the rule reports as "not learned at this size". The value `s` 800 was fixed by a rule of gates.md that is dated 2 Oct 2026, about 02:58 CEST: seeds 2 and 3 and the unseen-pair runs use the better of `s` 400 and `s` 800 by the final-pass dev AUC, which on seed 1 is 0.866 at `s` 400 (`20261002T003819Z_e2_L3_hier_s1_s400_p300`) against 0.916 at `s` 800. The rule was written after the `s` 400 result of seed 1 existed (its `metrics.json` is of 02:52 CEST), so the choice of `s` is post hoc. At `s` 800 the model learns, but tuned accuracy stays below 0.90, group recovery is 0 on every seed, and the purity of all groups is 0.479 / 0.491 / 0.516 (`purity.json` of `20261002T023042Z_e2_L3_s800_posthoc_exploratory`). On word pairs held out of training the AUC is 0.859 / 0.819 / 0.841 and the tuned accuracy 0.788 / 0.74 / 0.752 against the required 0.85 (`recovery.json` of the same run, rows of the runs `20261002T014618Z_e2_L3_hier_unseen_s1_s800_p300`, `20261002T015954Z_e2_L3_hier_unseen_s2_s800_p300` and `20261002T021328Z_e2_L3_hier_unseen_s3_s800_p300`). The fixed wording is: "learned at `s` scaled to word frequency, and generalises to unseen word pairs; the groups are partial mixtures, not recovered groups".

The one-paragraph check of the artificial data gives accuracies between 0.4963 and 0.5054 over the 12 data sets (20,000 fresh pairs each), and the run records that all its checks passed (`20261001T100004Z_pcp_world_checks`).

Hierarchical against flat, reported as the rule requires. The flat runs of 1 Oct 2026 follow the old convention (400 flat rules at `s` 21.1, not the leaf budget of the hierarchical model): their dev AUC is 0.547 / 0.546 / 0.551 at L1, 0.485 / 0.496 / 0.503 at L2 and 0.488 / 0.492 / 0.500 at L3 (`20261001T234330Z_e2_recovery_posthoc_exploratory`, `recovery.json`). The leaf-matched comparison was registered on 2 Oct 2026, 10:27 CEST (08:27 UTC): a flat model with 1,280 rules at `s` 800, 300 passes, scored on the unseen word pairs of L3. Its AUC is 0.506 / 0.510 / 0.503 and its tuned accuracy 0.512 / 0.510 / 0.5075 (`20261002T091027Z_e2_L3_flat_unseen_s1_s800_p300_c1280`, `20261002T091917Z_e2_L3_flat_unseen_s2_s800_p300_c1280`, `20261002T092238Z_e2_L3_flat_unseen_s3_s800_p300_c1280`), against 0.859 / 0.819 / 0.841 for the hierarchical model. Both models use `s` 800, which was not tuned for the flat model, the same leaf budget and 300 passes. This is a check of the counting argument in Granmo's talk, not a finding of the study; the hierarchical groups are about half pure, and recovery is 0.

Status of Check 2a: not met. L1 met; L2 not met, borderline; L3 not met.

## Check 2b (G3): slide 10

Rule in short: on at least 3 of 5 seeds the best single class-1 rule covers at least 90% of the unseen sentences of the family, fires on at most 5% of negated twins, near-misses and broken sentences, and recovers at least 4 of the 6 brackets (Jaccard of at least 0.8 with the true bracket). Coverage is reported for 288 and for 216 sentences. On a fail, 2 rules with 64 options are run for 1,000 passes (cap 2 h), and a measured negative is reported.

The sentence family is built from the two sentences of slide 10 of the project proposal. The templates file and the built data are not distributed. The build `20261002T201024Z_e3_slide10_data_build` has 5,856 sentences in 21 families and a vocabulary of 221 words: 288 slide-10 sentences (216 in the view of the project proposal), 288 negated twins, 624 near-miss sentences, 2,784 broken sentences and 1,872 sentences of other families. One sentence alone carries no label (logistic regression 0.498, interval [0.491, 0.505] as printed by the build), and word overlap alone ranks the held pairs at AUC 0.863.

The following applies to every number below. First, the number of OR-groups and of options was set to the known bracket structure (K 6, A 4); this model is called "a hierarchy shaped to the template". Second, the five near-miss sentences of the templates file are rule-generated: they follow a fixed one-word substitution rule (one word outside its bracket in each of brackets 1 to 5, the word "only" kept), they were not written independently as the study design had planned, they were written before any run on the slide-10 data, and their replacement words occur in the training data, so a rule can avoid them by memorising five words. The near-miss part of the check is therefore weak evidence. Third, `s` = 150 was chosen by the registered tie-break on dev AUC between three runs that were all at chance, so the choice is noise. Fourth, seed 1 is the screening run.

### Pair version

The screen ran seed 1 at `s` 12, 40 and 150. Each of the three runs was stopped early by rule at pass 90, with dev AUC 0.500, 0.476 and 0.518 (`20261002T201414Z_e3_slide10_hier_C50_A4_K6_s12_seed1`, `20261002T201749Z_e3_slide10_hier_C50_A4_K6_s40_seed1`, `20261002T202109Z_e3_slide10_hier_C50_A4_K6_s150_seed1`). Seeds 2 to 5 ran 300 passes at `s` 150. Check 2b on the evaluation split is in `20261002T213043Z_e3_check2b_eval_hier_s150`.

| seed | training run | dev AUC at the last pass | coverage of unseen sentences, 288 / 216 | firing on negated / near-miss (rule-generated) / broken | brackets recovered | Check 2b |
|---|---|---|---|---|---|---|
| 1 | `20261002T202109Z_e3_slide10_hier_C50_A4_K6_s150_seed1` (stopped early by rule at pass 90) | 0.518 | 0% / 0% | 0% / 0% / 0% | 0 of 6 | fails |
| 2 | `20261002T210206Z_e3_slide10_hier_C50_A4_K6_s150_seed2` | 0.497 | 0% / 0% | 0% / 0% / 0% | 0 of 6 | fails |
| 3 | `20261002T210312Z_e3_slide10_hier_C50_A4_K6_s150_seed3` | 0.540 | 0% / 0% | 0% / 0% / 0% | 0 of 6 | fails |
| 4 | `20261002T211332Z_e3_slide10_hier_C50_A4_K6_s150_seed4` | 0.516 | 0% / 0% | 0% / 0% / 0% | 0 of 6 | fails |
| 5 | `20261002T211437Z_e3_slide10_hier_C50_A4_K6_s150_seed5` | 0.521 | 0% / 0% | 0% / 0% / 0% | 0 of 6 | fails |

0 of 5 seeds pass. No best rule accepts any unseen slide-10 sentence, so the limit of 5% on firing is met only trivially.

A diagnostic without training (`20261002T213109Z_e3_pair_overlap_diagnostic_posthoc_exploratory`, post hoc) shows that the pair data is learnable from word overlap: the Jaccard AUC on the held pairs is 0.863 (0.867 on the training pairs), 0.979 for the 2,600 pairs whose sentences both come from the other families, 0.665 for the 4,915 pairs whose sentences both come from the slide-10 variant families, and 0.696 for the 698 pairs that involve a slide-10 sentence. The result of the pair version is therefore read as a learning failure of this hierarchical setting, not as missing signal.

Two comparison arms have no verdict of their own. The unshaped hierarchy (K 8, A 8, `s` 150, 3 seeds) has dev AUC 0.474 / 0.521 / 0.502 (`20261002T213459Z_e3_slide10_hier_C50_A8_K8_s150_seed1`, `20261002T214319Z_e3_slide10_hier_C50_A8_K8_s150_seed2`, `20261002T214610Z_e3_slide10_hier_C50_A8_K8_s150_seed3`), and 0 of its 3 seeds pass Check 2b (`20261002T215759Z_e3_check2b_eval_hier_A8K8_s150`). The leaf-matched flat model (1,200 rules, `s` 150, 3 seeds) learns the pair task, with dev AUC 0.891 / 0.899 / 0.916 (`20261002T212458Z_e3_slide10_flat_C1200_s150_seed1`, `20261002T212548Z_e3_slide10_flat_C1200_s150_seed2`, `20261002T213409Z_e3_slide10_flat_C1200_s150_seed3`), and 0 of its 3 seeds pass Check 2b (`20261002T215757Z_e3_check2b_eval_flat_C1200_s150`): the best rule of seed 2 covers 9.9% of the unseen sentences and fires on 4.6% of the near-miss sentences (rule-generated) and on 5.1% of the broken ones, and the best rules of seeds 1 and 3 cover none. The flat model is not called better at slide 10: Check 2b requires one rule, and the flat model fails it too.

### Direct-label fallback

The fallback of the rule was registered on 2 Oct 2026, 22:28 CEST (20:28 UTC), after the screen of the pair version had ended at chance. The model sees one sentence and the label says whether it is a slide-10 sentence or a negated twin, a near-miss or a broken sentence. The setting is 2 rules, 64 options, T 128, 1,000 passes, with K 6 (one OR-group per bracket, so again shaped to the template).

The `s` screen on seed 1 is in `20261002T225455Z_e3fb_s_choice`. The run at `s` 5 stopped at pass 959 of 1,000 by the 30-minute cap of its job and has dev accuracy 0.5; its best rule covers 0% and fires on 0%. The runs at `s` 21.1 and `s` 100 completed 1,000 passes with dev accuracy 1.0; their best rule covers 100% of the slide-10 dev sentences, and its highest firing rate on the negatives is 100%. All three meet 1 of 3 criteria, and the registered rule gives `s` = 21.1 through its last tie-break, the lower `s`.

Check 2b on the evaluation split, scored once after the choice, is in `20261003T001719Z_e3fb_check2b_eval_s21`: 0 of 5 seeds pass, in the 288 view and in the 216 view. For seed 1 (`20261002T215720Z_e3fb_slide10_text_C2_A64_K6_s21_seed1`) the file gives coverage 100% / 100%, firing of 100% on negated twins, on near-miss sentences (rule-generated) and on broken sentences, and 0 of 6 brackets. All five seeds completed 1,000 passes with dev accuracy 1.0 (`20261003T001721Z_e3fb_deletion_probe_posthoc_exploratory`).

The fixed wording for the `s` 21.1 screen run, and with it for this verdict, is: "perfect classification, achieved by a negative-class word blacklist; the slide10-class rule is always true; no single learned rule captures the family". The mechanism is this: the rule of the slide-10 class fires on everything, and the sentences are told apart by the other class, whose rule lists words that only the negatives contain. The data allows this shortcut because every kind of negative adds a word.

The deletion probe was registered on 3 Oct 2026, 00:29 CEST (22:29 UTC on 2 Oct), before any further fallback seed existed and after the rules of the seed-1 run had been read. It scores slide-10 evaluation sentences from which the words of one bracket are deleted and nothing is added, and it predicts that a template rule rejects at least 95% of them and that a blacklist accepts more than 50%. In `20261003T001721Z_e3fb_deletion_probe_posthoc_exploratory` all five seeds accept 100% of the deletion texts, on every bracket, next to 100% of the intact sentences; the outcome is the blacklist, as predicted. The `s` 100 screen run does the same. The `s` 5 screen run accepts 0% of the deletion texts and 0% of the intact sentences; the file labels it "template-like", which is wrong; the correct label is "reject-all (degenerate)".

The cap of the fallback carries a fixed disclosure: "cap '2 h' written before GPU sharing; read as GPU time (A) after the screen results were seen. Under reading B (summed job time), seeds 2 to 5 exceed the cap, so the verdict would be 'not judged within the cap'." The run `20261003T001723Z_e3fb_gpu_time` gives 103.9 GPU-minutes under reading A, inside the 120, and 206.9 minutes under reading B. Decision of 3 Oct 2026, 00:02 CEST (22:02 UTC on 2 Oct): the check is judged under reading A. The last summary of the study (3 Oct 2026, 03:55 CEST) still listed the choice between the two readings as open. Under either reading no seed passes.

The flat comparison of the fallback (768 flat rules, `s` 21.1, 3 seeds) lies beyond the 2-hour cap under both readings and is not part of the verdict; it ran because the schedule had slack. Its dev accuracy is 0.915 / 0.933 / 0.903 (`20261003T005424Z_e3fb_slide10_text_flat_C768_s21_seed1`, `20261003T010445Z_e3fb_slide10_text_flat_C768_s21_seed2`, `20261003T012450Z_e3fb_slide10_text_flat_C768_s21_seed3`). Its Check 2b scores and deletion-probe scores are in sub-folders of `20261003T015449Z_e3fb_flat_C768_check2b_eval_and_probe_v2_exploratory`, whose `metrics.json` is empty; the scores are not reproduced here. No flat seed passed: this outcome is the one noted at the time, and no file of the run is cited for it.

What this result supports: with direct labels the model separates slide-10 sentences from their negated, near-miss and broken variants on 5 of 5 seeds, and it does so with a word blacklist, not with one clause for the family. Whether a design without the shortcut, with negatives that only remove or reorder words, would yield such a clause is untested.

### Robustness arms R1 and R2 (post hoc)

Both arms were registered on 2 Oct 2026, 23:35 CEST (21:35 UTC), after the results of the pair version were known. They are robustness diagnostics, they cannot change the verdict, and they are scored on dev only. R1 is the hierarchy at the Wikipedia headline setting (50 rules, A 64, K 2, T 3200, `s` 100) on the slide-10 pairs. R2 is the template-shaped model at `s` 25. The numbers are from `20261003T004307Z_e3diag_R1_report_posthoc_exploratory` and `20261003T004309Z_e3diag_R2_report_posthoc_exploratory`.

| arm | seed | training run | train accuracy | dev AUC | best rule on dev: coverage 288 | firing on negated / near-miss (rule-generated) / broken | brackets |
|---|---|---|---|---|---|---|---|
| R1 | 1 | `20261002T225427Z_e3diag_R1_hier_C50_A64_K2_T3200_s100_seed1_exploratory` | 0.851 | 0.915 | 53.5% | 0% / 60.6% / 55.1% | 0 |
| R1 | 2 | `20261003T000813Z_e3diag_R1_hier_C50_A64_K2_T3200_s100_seed2_exploratory` | 0.840 | 0.915 | 47.5% | 0% / 66.1% / 46.9% | 0 |
| R1 | 3 | `20261003T001647Z_e3diag_R1_hier_C50_A64_K2_T3200_s100_seed3_exploratory` | 0.842 | 0.915 | 72.3% | 0% / 58.4% / 62.5% | 0 |
| R2 | 1 | `20261003T002022Z_e3diag_R2_hier_C50_A4_K6_s25_seed1_exploratory` | 0.515 | 0.519 | 0% | 0% / 0% / 0% | 0 |
| R2 | 2 | `20261003T002827Z_e3diag_R2_hier_C50_A4_K6_s25_seed2_exploratory` | 0.5195 | 0.537 | 0% | 0% / 0% / 0% | 0 |
| R2 | 3 | `20261003T003133Z_e3diag_R2_hier_C50_A4_K6_s25_seed3_exploratory` | 0.518 | 0.537 | 0% | 0% / 0% / 0% | 0 |

The reading was fixed in the registration: if R1 or R2 reaches dev AUC of at least 0.70 on at least 2 of its 3 seeds, the write-up says "the failure is specific to our configuration" and names the setting that works. R1 reaches it on 3 of 3 seeds and R2 on none, so this is the reading: the template-shaped settings at T 128 failed, and the hierarchy at the Wikipedia headline setting learns the slide-10 pairs. Which of A, K or T matters was not separated. The registered verdict of the pair version, 0 of 5, stands, and R1 is reported next to it as configuration sensitivity. R1 does not meet Check 2b either: its best rules recover no bracket and fire on about half of the near-miss and broken sentences. These rules fire on no negated twin because they require a fixed word of the template that the twins lack; no claim about negation follows from that.

Status of Check 2b: not met, in both versions (0 of 5 seeds each). The fallback is judged under reading A of its cap.

## Check 2c (news): AG News word groups

Rule in short: hierarchical accuracy is within 2 points of a flat Tsetlin Machine while using fewer rules (mean of 3 seeds), and the top OR-groups beat random groups on NPMI. The registration of 2 Oct 2026, 06:06 CEST (04:06 UTC) reads "within 2 points" one-sidedly, and the registration of 07:38 CEST (05:38 UTC) counts it in whole texts: over 3 seeds of 10,000 dev texts the hierarchical total of correct texts must be at least the flat total minus 600. On a fail the groups are shown as examples only.

The `s` screen on seed 1 gives the mean dev accuracy over passes 46 to 50 (`20261002T072322Z_e5a_s_choice_rule`): for the hierarchical model 0.718 at `s` 21.1, 0.785 at `s` 60 and 0.775 at `s` 230; for the flat model 0.729, 0.818 and 0.803. The registered rule gives `s` = 60 for both models. No run was stopped early. The screen stops at `s` = 230, a limit that the registration states; accuracy was lower there than at `s` 60 for both models.

| model | rules per topic | correct dev texts of 10,000, seeds 1 / 2 / 3 | total of 30,000 | mean accuracy |
|---|---|---|---|---|
| hierarchical | 50 | 7,884 / 7,805 / 7,947 | 23,636 | 0.788 |
| flat | 6,400 | 8,197 / 8,172 / 8,227 | 24,596 | 0.820 |

The numbers are from `20261002T075715Z_e5a_check2c_accuracy`, which names the six training runs. The hierarchical model is 960 texts short of the flat one and 600 are allowed, so the accuracy part is not met.

The NPMI part is met under both methods (`20261002T074043Z_e5a_npmi_check2c`). The 195 counted OR-groups of the top 10 positive rules per topic, pooled over the 3 hierarchical seeds, hold 52.7 words on average. Their NPMI is 0.027 against -0.115 for frequency-matched random groups, a difference of 0.143 with a group-bootstrap interval of [0.140, 0.145]; the calibrated test gives p = 0.0005 and a t-interval of [0.140, 0.145].

Status: not met, because both parts are needed. By the rule the news-topic word groups are shown as examples only, and the result is a measured negative. Each model's `s` was chosen on its own seed-1 dev curve; seeds 2 and 3 are not selected. As context, and not as a rescue: the hierarchical model reaches 96% of the flat model's accuracy with 1/128 of its rules, a logistic regression reaches 0.862 on the same dev texts (`20261002T035139Z_e5a_lr_baseline`), and both Tsetlin models were tuned in the same way.

## Check 3 (G2): Wikipedia learns (A) or not (B)

Rule in short: the best arm passes on 2 of 3 seeds. A seed passes if dev AUC is at least 0.65, tuned accuracy is at least 0.60, the lowest accuracy over the last 5 passes is at least 0.58, the share of "same article" answers is between 0.2 and 0.8 in each of those passes, the lower bound of the article-bootstrap interval of tuned accuracy is above 0.55, and the interval of the partner-shuffle accuracy includes 0.50. A pass gives branch A, a fail branch B.

The headline setting is 50 rules, 64 options, 2 OR-groups per paragraph, T 3,200, with `s` and the number of passes P fixed by two rules of gates.md. The rule for P was registered on 2 Oct 2026 at about 02:30 CEST (00:30 UTC), before the learning-curve run existed. The rule for `s` was registered at about 02:58 CEST (00:58 UTC), before the screen runs at `s` 200 and `s` 400 existed and after the runs at `s` 21, 40 and 100 of 1 Oct 2026.

The rule for `s` takes the value with the highest dev AUC at pass 100 among the stable runs on same-article pairs, seed 1 (`20261002T035442Z_e4_headline_s_rule`). Every value tried is reported:

| `s` | run | dev AUC at pass 100 | lowest accuracy, passes 96 to 100 | stable |
|---|---|---|---|---|
| 21 | `20261001T202954Z_e4_sa_granmo_C50_A64_s21` | 0.490 | not a candidate of the rule | not a candidate of the rule |
| 40 | `20261001T205856Z_e4_sa_granmo_C50_A64_s40` | 0.488 | not a candidate of the rule | not a candidate of the rule |
| 100 | `20261001T212959Z_e4_sa_granmo_C50_A64_s100` | 0.745 | 0.642 | yes |
| 200 | `20261002T032820Z_e4_screen_sa_granmo_s200` | 0.648 | 0.585 | yes |
| 400 | `20261002T032840Z_e4_screen_sa_granmo_s400` | 0.600 | 0.554 | no |

The headline `s` is 100. The rule chooses among 100, 200 and 400; at `s` 21 and `s` 40 the model did not learn, and both runs are listed because gates.md requires every value tried to be reported. The rule for P keeps P = 100 unless pass 150 or pass 200 gains at least 0.01 in dev AUC (`20261002T050204Z_e4_pass_count_rule`, on the learning-curve run `20261002T035511Z_e4_curve_sa_granmo_s100_p200`): the AUC is 0.7452 at pass 100, 0.7542 at pass 150 (a gain of 0.0090) and 0.7476 at pass 200 (a gain of 0.0024), so P = 100. After a tolerance for rounding was added to the comparison, the rule was applied again and gave P = 100 again (`20261002T053822Z_e4_rules_recheck_after_float_tolerance`). The learning-curve run and the earlier run at `s` 100 are two runs of seed 1 with the same settings, and their dev AUC at pass 100 is the same, 0.745230375 (`20261002T035442Z_e4_headline_s_rule` for the earlier run, `20261002T050235Z_e4_head_sa_s1_scoreP100` for the learning-curve run).

Both `s` and P were chosen on the dev data of seed 1 on same-article pairs. The dev numbers of that seed are therefore slightly optimistic; the other seeds are not selected, and the one final test decides the reported number.

| data | seed | scoring run | dev AUC | tuned accuracy | lowest accuracy, last 5 passes | share of "same" answers, last 5 passes | partner-shuffle accuracy | per seed |
|---|---|---|---|---|---|---|---|---|
| same article | 1 (chose `s` and P) | `20261002T050235Z_e4_head_sa_s1_scoreP100` | 0.745 [0.737, 0.753] | 0.667 [0.660, 0.675] | 0.642 | 0.561 to 0.710 | 0.506 [0.499, 0.513] | passes |
| same article | 2 | `20261002T052930Z_e4_head_sa_s2_score` | 0.739 [0.731, 0.747] | 0.666 [0.660, 0.674] | 0.654 | 0.5855 to 0.675 | 0.506 [0.498, 0.512] | passes |
| same article | 3 | `20261002T060727Z_e4_head_sa_s3_score` | 0.733 [0.725, 0.741] | 0.661 [0.654, 0.669] | 0.645 | 0.474 to 0.671 | 0.503 [0.496, 0.510] | passes |
| next paragraph | 1 | `20261002T062612Z_e4_head_np_s1_score` | 0.752 [0.743, 0.760] | 0.682 [0.675, 0.689] | 0.630 | 0.350 to 0.756 | 0.502 [0.495, 0.509] | passes |
| next paragraph | 2 | `20261002T070833Z_e4_head_np_s2_score` | 0.728 [0.719, 0.737] | 0.657 [0.650, 0.666] | 0.608 | 0.376 to 0.842 | 0.498 [0.492, 0.505] | fails on stability |
| next paragraph | 3 | `20261002T071028Z_e4_head_np_s3_score` | 0.751 [0.743, 0.759] | 0.675 [0.669, 0.684] | 0.638 | 0.441 to 0.774 | 0.498 [0.491, 0.505] | passes |

Same-article pairs pass on 3 of 3 seeds and next-paragraph pairs on 2 of 3. Seed 2 on next-paragraph pairs fails only on stability: at pass 100 it answers "same" for a share of 0.842 of the pairs.

Status: met. Branch A was fixed by the decision of 2 Oct 2026, 21:27 CEST (19:27 UTC).

Further results belong next to this status. First, the simple baselines are ahead on the same dev pairs. Word overlap (Jaccard) has dev AUC 0.819 on same-article pairs and 0.864 on next-paragraph pairs, and the best logistic regression 0.871 and 0.907 (`20261001T094249Z_data_checks_final_exploratory`, `checks.md`; these are AUCs, and the accuracies of the same file are not comparable with them).

Second, the flat control registered on 2 Oct 2026, 06:20 CEST (04:20 UTC): 6,400 flat rules, the leaf budget of the hierarchical model, with the same T, `s` and P. Its `s` is not tuned separately.

| data | seed | scoring run | flat dev AUC |
|---|---|---|---|
| same article | 1 | `20261002T114436Z_e4_flat_sa_s1_score` | 0.680 [0.670, 0.688] |
| same article | 2 | `20261002T114457Z_e4_flat_sa_s2_score` | 0.678 [0.669, 0.687] |
| same article | 3 | `20261002T114517Z_e4_flat_sa_s3_score` | 0.679 [0.670, 0.688] |
| next paragraph | 1 | `20261002T114537Z_e4_flat_np_s1_score` | 0.704 [0.696, 0.713] |
| next paragraph | 2 | `20261002T114557Z_e4_flat_np_s2_score` | 0.703 [0.694, 0.712] |
| next paragraph | 3 | `20261002T114617Z_e4_flat_np_s3_score` | 0.701 [0.692, 0.710] |

The hierarchical model is ahead of the flat model on every seed, and every flat interval lies below every hierarchical interval on the same data. This is the opposite of the news-topic test, where the flat model was ahead.

Third, the second setting (2,000 rules, 8 options, 2 OR-groups, T 4,000, `s` 40, 50 passes), reported from dev only so that its higher AUC stays visible next to the headline setting.

| data | seed | scoring run | dev AUC | tuned accuracy | share of "same" answers, last 5 passes | stable |
|---|---|---|---|---|---|---|
| same article | 1 | `20261002T145433Z_e4_second_sa_s1_score` | 0.768 | 0.688 | 0.131 to 0.698 | no |
| same article | 2 | `20261002T145523Z_e4_second_sa_s2_score` | 0.763 | 0.684 | 0.148 to 0.748 | no |
| same article | 3 | `20261002T145613Z_e4_second_sa_s3_score` | 0.771 | 0.692 | 0.179 to 0.721 | no |
| next paragraph | 1 | `20261002T145703Z_e4_second_np_s1_score` | 0.796 | 0.7115 | 0.143 to 0.823 | no |
| next paragraph | 2 | `20261002T145753Z_e4_second_np_s2_score` | 0.792 | 0.7105 | 0.130 to 0.843 | no |
| next paragraph | 3 | `20261002T145843Z_e4_second_np_s3_score` | 0.778 | 0.697 | 0.125 to 0.851 | no |

This setting ranks pairs better than the headline setting, but its yes/no cut-off swings from pass to pass, so it passes Check 3 on 0 of 3 seeds for both data sets, failing the stability part on every seed. It is also below word overlap on dev.

## Check 4a (G4): Wikipedia word groups (branch A only)

Rule in short: the counted OR-groups beat both random groups and the groups of the shuffled-label model on NPMI. The registration of 2 Oct 2026, about 07:00 CEST (05:00 UTC) fixes that NPMI is computed on whole articles and that the verdict pools the 3 seeds. Until the decision of 21:27 CEST the frozen group bootstrap decided; since then the calibrated test decides and the bootstrap is shown next to it. The two methods agree on every line of this check.

The scoring run is `20261002T114726Z_check4a_word_groups`, on the three same-article headline seeds and three shuffled-label runs (`20261002T092713Z_e4_head_sa_shuf_s1`, `20261002T093048Z_e4_head_sa_shuf_s2`, `20261002T095639Z_e4_head_sa_shuf_s3`, whose dev AUC is 0.503 / 0.504 / 0.501).

| groups | counted groups | NPMI difference to random groups | NPMI difference to shuffled-label groups | frozen gate / calibrated test |
|---|---|---|---|---|
| seed 1 | 72 | +0.032 | +0.035 | yes / yes |
| seed 2 | 67 | +0.033 | +0.035 | yes / yes |
| seed 3 | 59 | +0.030 | +0.032 | yes / yes |
| pooled (the registered verdict) | 198 | +0.032 [0.029, 0.034] | +0.034 [0.032, 0.037] | yes / yes |

The pooled intervals are group-bootstrap intervals. Without the groups that contain one of the four skewed words (161 groups pooled) the differences are +0.033 and +0.035. The groups of the shuffled-label model score below random groups (-0.009).

Status: met. The check was scored before the decision between branch A and branch B: the run started at 11:47 UTC (13:47 CEST) on 2 Oct 2026, the decision is of 21:27 CEST, and the registration of 12:24 CEST had placed the scoring after that decision. The result is topical coherence, not synonymy: on the WordNet score, which is not part of this check, the pooled groups are +0.005 above random groups and -0.002 against the shuffled-label groups (permutation p 0.83), so they are not more synonym-like than the groups of a model trained on shuffled labels.

## Check 4b (synonyms): the synonym test

Rule in short: on next-paragraph pairs with every shared word removed, the check passes on 2 of 3 seeds if (1) the model is above guessing, with a dev AUC whose interval is above 0.5 while partner-shuffle and the length-only check stay at 0.50, and (2) the counted OR-groups score higher than random groups and higher than the groups of the shuffled-label model on the WordNet score and on NPMI. By the addendum of 1 Oct 2026 the verdict leaves out every group that contains one of the four skewed words. A pass allows a claim about synonym-like groups; a fail is reported as a measured negative.

gates.md registers two arms, `s` 100 and `s` 400, each with 3 seeds and 3 shuffled-label seeds. A third arm at `s` 200 was proposed before any synonym run; the decision to run it is of 2 Oct 2026, 21:27 CEST, after the results of the other two arms were known, so it is post hoc in that sense. The registration of about 07:00 CEST (05:00 UTC) fixes that a seed passes only if part 1 and all four comparisons pass.

| arm | scoring run | dev AUC, seeds 1 / 2 / 3 | part 1 | WordNet difference to random / to shuffled-label groups, pooled, verdict version | seeds that pass | frozen gate and calibrated test |
|---|---|---|---|---|---|---|
| `s` 100 (the arm of the final test) | `20261002T093158Z_check4b_synonym_groups_s100` | 0.622 / 0.622 / 0.611 | 3 of 3 | +0.011 / +0.011 (54 groups) | 2 of 3 | both pass |
| `s` 400 (the registered contrast) | `20261002T130624Z_check4b_synonym_groups_s400` | 0.563 / 0.569 / 0.558 | 3 of 3 | +0.003 / +0.001 (86 groups) | 0 of 3 | both fail |
| `s` 200 (allowed after the other results) | `20261002T231829Z_check4b_synonym_groups_s200` | 0.577 / 0.572 / 0.573 | 3 of 3 | -0.0002 / -0.0009 (59 groups) | 0 of 3 | both fail |

On every seed of every arm the NPMI comparisons pass and part 1 passes; the arms differ on the WordNet comparisons only. For the `s` 100 arm the verdict version has 21 / 20 / 13 groups per seed. The WordNet difference to random groups is +0.014 / +0.013 / +0.003 and to the shuffled-label groups +0.015 / +0.014 / +0.001; seeds 1 and 2 pass and seed 3 fails both WordNet comparisons (p 0.29 and 0.39). The pooled NPMI differences are +0.032 and +0.035. Without pairs that are two forms of one word, the seeds pass and fail in the same way. The length-only accuracy is 0.500 [0.493, 0.507], and the shuffled-label runs of this arm end at dev AUC 0.507 / 0.508 / 0.499 (`20261002T075700Z_e4s_nsall_s100_shuf_s1`, `20261002T082531Z_e4s_nsall_s100_shuf_s2`, `20261002T085502Z_e4s_nsall_s100_shuf_s3`).

Status: met on the `s` 100 arm only, on 2 of 3 seeds and with a small effect; not met on the `s` 400 arm and on the `s` 200 arm, 0 of 3 seeds each.

The effect is about one percentage point: the word pairs inside the groups of the `s` 100 arm are WordNet-related more often than those of random groups by 0.011. The groups are large topical mixtures, with 58.6 words on average in the verdict version, not synonym sets. The wording is "slightly more synonym-like than chance"; the result does not show that the model learns synonyms. A note that follows the registration of 2 Oct 2026, 03:05 CEST had expected topic-like mixtures and a modest effect, because an OR-group has 64 options while real synonym sets are small.

The result is fragile. Of the 204 counted groups of the three seeds, 150 contain one of the four skewed words and are left out of the verdict, which leaves 13 to 21 groups per seed, close to the minimum of 12 that the calibrated test needs. The arm was scored at 09:31 UTC (11:31 CEST) on 2 Oct 2026, before the decision of 21:27 CEST that made the calibrated test the deciding method; both methods agree on every line, so that decision changes no verdict. The note that the scorer prints in the `s` 200 run still says that the bootstrap decides; it was out of date after the decision of 21:27 CEST, and both methods agree in that run. Raising `s` above 100 made the groups less synonym-like, not more, on both of the other arms; the explanations noted for this are readings, not established mechanisms.

## The final test

Rule in short: the final-test split is used once, for the entries of the main table. The registration of the frozen settings (2 Oct 2026, 15:19 CEST, completed by 18:47 CEST) fixes the main table as the Wikipedia headline setting on same-article and next-paragraph pairs and the `s` 100 arm of the synonym test, each on seeds 1 to 5, with shuffled-label seeds 1 to 5 for the synonym data. It also fixes the comparison: over a paired article bootstrap of 1,000 resamples, the model "beats X" only if the whole 95% interval of the mean difference in AUC is above 0, is "below X" if it is all below 0, and is otherwise "not distinguishable". X is word overlap and the baseline that was best on dev.

The decisions by which the frozen settings became binding and the final test was run are timed 21:27 CEST (19:27 UTC) on 2 Oct 2026 in docs/registrations.md and 21:28 CEST (19:28 UTC) in the addendum of gates.md; the two times differ by one minute. The first test re-score started at 19:28:18 UTC.

The test was run once. Twenty run folders named `*_testscore` each score one saved model on the test split, and `20261002T193513Z_final_test_main_table` builds the table from them (file `report.json`). The hashes of the frozen settings are in [frozen_settings_hashes.json](../frozen_settings_hashes.json).

| data | seed | run folder of the test re-score | model trained in run | test AUC |
|---|---|---|---|---|
| same article | 1 | `20261002T192818Z_e4_head_sa_s1_testscore` | `20261002T035511Z_e4_curve_sa_granmo_s100_p200` (model at pass 100) | 0.743 |
| same article | 2 | `20261002T192838Z_e4_head_sa_s2_testscore` | `20261002T050310Z_e4_head_sa_s2` | 0.737 |
| same article | 3 | `20261002T192858Z_e4_head_sa_s3_testscore` | `20261002T053211Z_e4_head_sa_s3` | 0.736 |
| same article | 4 | `20261002T192918Z_e4_head_sa_s4_testscore` | `20261002T142452Z_e4_head_sa_s4` | 0.731 |
| same article | 5 | `20261002T192938Z_e4_head_sa_s5_testscore` | `20261002T145158Z_e4_head_sa_s5` | 0.727 |
| next paragraph | 1 | `20261002T192959Z_e4_head_np_s1_testscore` | `20261002T055851Z_e4_head_np_s1` | 0.755 |
| next paragraph | 2 | `20261002T193019Z_e4_head_np_s2_testscore` | `20261002T062537Z_e4_head_np_s2` | 0.730 |
| next paragraph | 3 | `20261002T193039Z_e4_head_np_s3_testscore` | `20261002T064353Z_e4_head_np_s3` | 0.756 |
| next paragraph | 4 | `20261002T193059Z_e4_head_np_s4_testscore` | `20261002T150009Z_e4_head_np_s4` | 0.756 |
| next paragraph | 5 | `20261002T193119Z_e4_head_np_s5_testscore` | `20261002T151944Z_e4_head_np_s5` | 0.754 |
| shared words removed | 1 | `20261002T193140Z_e4s_nsall_s100_s1_testscore` | `20261002T074540Z_e4s_nsall_s100_s1` | 0.611 |
| shared words removed | 2 | `20261002T193200Z_e4s_nsall_s100_s2_testscore` | `20261002T081320Z_e4s_nsall_s100_s2` | 0.613 |
| shared words removed | 3 | `20261002T193220Z_e4s_nsall_s100_s3_testscore` | `20261002T084141Z_e4s_nsall_s100_s3` | 0.609 |
| shared words removed | 4 | `20261002T193240Z_e4s_nsall_s100_s4_testscore` | `20261002T152830Z_e4s_nsall_s100_s4` | 0.616 |
| shared words removed | 5 | `20261002T193301Z_e4s_nsall_s100_s5_testscore` | `20261002T155721Z_e4s_nsall_s100_s5` | 0.618 |
| shared words removed, shuffled labels | 1 | `20261002T193321Z_e4s_nsall_s100_shuf_s1_testscore` | not named in the report | 0.506 [0.494, 0.517] |
| shared words removed, shuffled labels | 2 | `20261002T193341Z_e4s_nsall_s100_shuf_s2_testscore` | not named in the report | 0.492 [0.480, 0.503] |
| shared words removed, shuffled labels | 3 | `20261002T193401Z_e4s_nsall_s100_shuf_s3_testscore` | not named in the report | 0.507 [0.496, 0.518] |
| shared words removed, shuffled labels | 4 | `20261002T193422Z_e4s_nsall_s100_shuf_s4_testscore` | not named in the report | 0.511 [0.4996, 0.523] |
| shared words removed, shuffled labels | 5 | `20261002T193442Z_e4s_nsall_s100_shuf_s5_testscore` | not named in the report | 0.500 [0.489, 0.512] |

The summary table is taken from `report.json` of `20261002T193513Z_final_test_main_table`. Each data set has 10,000 test pairs. Four limits apply to the table. The tuned accuracy of the model has its cut-off chosen on the test pairs, where gates.md names dev data; the registered comparisons use AUC and do not depend on it. The test split contains duplicate articles. The article bootstrap resamples only the article of paragraph A. And the seeds are not fully independent repeats, because they share parts of their random streams. Section 4 of the [known limitations](../../../docs/known-limitations.md) gives each of the four in full.

| final test | same article | next paragraph | shared words removed |
|---|---|---|---|
| test articles | 2,763 | 2,767 | 2,743 |
| model AUC, mean of 5 seeds (SD) | 0.735 (0.006) | 0.750 (0.011) | 0.614 (0.004) |
| model tuned accuracy, mean (SD) | 0.665 (0.008) | 0.679 (0.010) | 0.576 (0.005) |
| word overlap (Jaccard) AUC | 0.817 [0.808, 0.826] | 0.857 [0.849, 0.864] | 0.500 by construction |
| PPMI word-vector cosine AUC | 0.852 | 0.886 | 0.811 [0.802, 0.820] |
| logistic regression on both paragraphs and their shared words, AUC | 0.869 [0.862, 0.877] | 0.903 [0.897, 0.910] | 0.494 |
| baseline that was best on dev | the logistic regression | the logistic regression | PPMI cosine |
| model minus word overlap, paired | -0.082 [-0.093, -0.071]: below | -0.106 [-0.116, -0.096]: below | by construction, not a comparison |
| model minus best-on-dev baseline, paired | -0.135 [-0.144, -0.125]: below | -0.153 [-0.162, -0.144]: below | -0.197 [-0.210, -0.185]: below |
| model minus the word-matching classifier, paired | | | +0.119 [+0.109, +0.129]: above |

The comparison with word overlap, as the report gives it, is "below" on same-article pairs and on next-paragraph pairs. On the data with shared words removed, word overlap is 0.500 by construction; the report states that this line is not a comparison, and the registration states that it can never support a claim against word overlap. Against the baseline that was best on dev the model is "below" on all three data sets. In all of these comparisons the paired interval of every single seed has the same sign as the interval of the mean.

The last line carries its registered label: the logistic regression on both paragraphs and their shared words is a word-matching classifier that is helpless without shared words; on this data its AUC is 0.494. That the model is above it (the report's verdict word is "beats") says that the model uses something other than shared words. It does not compare the model with a working baseline: the working baseline on this data is PPMI cosine at 0.811, and the model is far below it. The shuffled-label controls are at chance, and each of their intervals includes 0.50.

Status: not met, in the sense of the registered comparison "beats X". The registered outcome is "below" word overlap on both Wikipedia tasks and "below" the best-on-dev baseline on all three data sets. gates.md sets no pass threshold for the final test, so the status refers to that comparison only.

The freeze rule of gates.md, under which a number that does not reproduce within 0.02 moves to the list of claims that are not made, has the status not run: the study ended on 3 Oct 2026, and the run index holds no later run of this study.

## The slide-6 paraphrase test

This test is not a check of gates.md. It was registered on 2 Oct 2026, 21:29 CEST (19:29 UTC), before any code or run, as a descriptive test, and amended at 21:39 CEST (19:39 UTC), before any slide-6 run. The question is whether one learned rule covers the five paraphrases of slide 6 of the project proposal, which are kept out of training and dev. The claim "one rule covers all 5" is made only if, on at least 2 of 3 seeds, the best single class-1 rule fires on all five, fires on at most 5% of the sentences of other families and of the broken sentences, and recovers at least 4 of 5 brackets. As for slide 10, K 5 and A 5 were set to the known template, and the sentences are not distributed. Bracket recovery uses the rule of Check 2b, with the limits that the registration states: an alternative that is missing while its words appear in another alternative is not visible to it, and an OR-group must contain nearly all words of each alternative.

The build `20261002T194614Z_slide6_data_build` has 3,300 sentences in 14 families and a vocabulary of 208 words. 96 broken sentences were dropped because no word-based rule can reject them. One sentence alone carries no label (0.505, interval [0.498, 0.512] as printed by the build), and word overlap alone ranks the held pairs at AUC 0.880.

The `s` screen on seed 1 is in `20261002T202450Z_slide6_s_choice`. At `s` 25 and `s` 50 the best rule is always true: it covers 100% of the family's dev sentences and fires on 100% of the broken and of the other-family sentences. At `s` 100 it fires on nothing. All three score 0.0 on the registered criterion, and the tie goes to the higher dev AUC: 0.687 at `s` 25 against 0.570 and 0.515. Seeds 2 and 3 ran at `s` 25 and reached dev AUC 0.672 (`20261002T205056Z_slide6_s25_seed2`) and 0.617 (run `20261002T205201Z_slide6_s25_seed3`, value from the claim run named below).

The verdict noted at the time is that the claim fails on all 3 seeds, under the original rule of 21:29 and under the amendment of 21:39 alike. The file `metrics.json` of the claim run `20261002T210327Z_slide6_eval_claim` holds only the entry of the last seed scored, seed 3: its best rule fires on 0 of the 5 paraphrases, on 0% of the broken sentences and on 11.5% of the other-family sentences, and it recovers 0 of 5 brackets. For seed 1 the screen file above shows the always-true rule on dev. The scores of seeds 1 and 2 on the evaluation split are not in that file; an always-true rule on both seeds was noted at the time, and no file of the run is cited for it.

Result: the claim is not met, on 3 of 3 seeds. Its reach: in this setting (50 rules, 14 families, pair task) no single learned rule captures the paraphrase family; it is a measured negative for slide 6 at this size, and it does not show that a hierarchical clause cannot represent the family, only that the training did not find one.

S6-R1 is the slide-6 version of the R1 arm, registered on 3 Oct 2026, 01:08 CEST (23:08 UTC on 2 Oct), after the slide-6 results were known and after seed 1 of R1 had learned the slide-10 pairs. It is a post hoc robustness diagnostic and cannot change the verdict. The numbers are from `20261003T010506Z_s6diag_R1_report_posthoc_exploratory`.

| seed | training run | train accuracy | dev AUC | best rule on dev: coverage of the family | firing on broken / other families | brackets | paraphrases covered, of 5 |
|---|---|---|---|---|---|---|---|
| 1 | `20261003T003953Z_s6diag_R1_hier_C50_A64_K2_T3200_s100_seed1_exploratory` | 0.9185 | 0.958 | 71.0% | 66.0% / 38.3% | 0 of 5 | 4 |
| 2 | `20261003T004244Z_s6diag_R1_hier_C50_A64_K2_T3200_s100_seed2_exploratory` | 0.9025 | 0.961 | 79.0% | 73.1% / 15.3% | 0 of 5 | 3 |
| 3 | `20261003T005134Z_s6diag_R1_hier_C50_A64_K2_T3200_s100_seed3_exploratory` | 0.907 | 0.962 | 51.6% | 69.9% / 42.1% | 0 of 5 | 2 |

The registered reading holds: dev AUC is at least 0.70 on 3 of 3 seeds, so the slide-6 setting (K 5, A 5, T 128) was configuration-limited too. The claim is still not met on any seed: no seed covers all five paraphrases, firing on broken and other-family sentences is far above 5%, and no bracket is recovered, which is impossible by construction at K 2, as the registration says. The scorer prints its fixed line that K 5 and A 5 were set to the known template for these runs as well; that line is wrong for them (K 2, A 64, not shaped), and the logged run keeps it.

For slides 6 and 10 alike, the hierarchy at the Wikipedia headline setting learns the pair task, and in none of these runs does one learned clause capture a sentence family in the way the project proposal describes.

## All checks at a glance

gates.md has nine entries: eight checks and the final test. Each has one status.

| id | pass rule, in short | status | qualifier | evidence (run ids) |
|---|---|---|---|---|
| Check 1a (G0) | package demo at least 0.95 on 3 seeds; "one word decides" AUC at least 0.98; recomputation agrees with the package | met | demo run with 1 repeat per seed | `20261001T091807Z_e1_demo_seed1`, `20261001T092250Z_e1_demo_seed2`, `20261001T092751Z_e1_demo_seed3`, `20261001T092825Z_e1_p1word_hier`, `20261001T093345Z_e1_verify_binary`, `20261001T093407Z_e1_verify_default` |
| Check 1b (data) | per pair version: at most about 3 skewed words; one paragraph alone, partner-shuffle and length alone at 0.50; no word overlap where shared words are removed | met | next-paragraph data after one rebuild; main synonym data borderline (4 skewed words), passed by the decision of 1 Oct 2026 | `20261001T094249Z_data_checks_final_exploratory`, `20261001T094347Z_data_checks_ns_all_v2_exploratory`, `20261001T092321Z_data_checks_exploratory` |
| Check 2a (G1) | L1, L2, L3: accuracy at least 0.90 and group recovery at least 0.80 on 2 of 3 seeds; L3 unseen-pair accuracy at least 0.85 | not met | L1 met; L2 not met, borderline, under an accuracy chosen post hoc; L3 not met, at an `s` fixed by a rule that was written after the first `s` 400 result | `20261002T053510Z_e2_crossfit_acc_posthoc`, `20261002T053708Z_e2_crossfit_split_avg_all_posthoc_exploratory`, `20261001T234330Z_e2_recovery_posthoc_exploratory`, `20261002T023042Z_e2_L3_s800_posthoc_exploratory` |
| Check 2b (G3) | one rule covers at least 90% of unseen slide-10 sentences, fires on at most 5% of negatives, recovers at least 4 of 6 brackets, on 3 of 5 seeds | not met | model shaped to the template; 0 of 5 seeds in the pair version and 0 of 5 in the direct-label fallback; fallback judged under reading A of its cap, under reading B "not judged within the cap" | `20261002T213043Z_e3_check2b_eval_hier_s150`, `20261003T001719Z_e3fb_check2b_eval_s21`, `20261003T001721Z_e3fb_deletion_probe_posthoc_exploratory`, `20261003T001723Z_e3fb_gpu_time` |
| Check 2c (news) | hierarchical accuracy within 2 points of flat with fewer rules; top OR-groups beat random groups on NPMI | not met | accuracy part not met (960 texts short, 600 allowed); NPMI part met; groups shown as examples only | `20261002T075715Z_e5a_check2c_accuracy`, `20261002T074043Z_e5a_npmi_check2c`, `20261002T072322Z_e5a_s_choice_rule` |
| Check 3 (G2) | on 2 of 3 seeds: dev AUC at least 0.65, tuned accuracy at least 0.60, stable last 5 passes, bootstrap bound above 0.55, partner-shuffle at 0.50 | met | same article 3 of 3 seeds, next paragraph 2 of 3; seed 1 chose `s` and P; branch A | `20261002T050235Z_e4_head_sa_s1_scoreP100`, `20261002T052930Z_e4_head_sa_s2_score`, `20261002T060727Z_e4_head_sa_s3_score`, `20261002T062612Z_e4_head_np_s1_score`, `20261002T070833Z_e4_head_np_s2_score`, `20261002T071028Z_e4_head_np_s3_score` |
| Check 4a (G4) | counted OR-groups beat random groups and shuffled-label groups on NPMI | met | scored before the decision between branch A and B; topical coherence, not synonymy | `20261002T114726Z_check4a_word_groups` |
| Check 4b (synonyms) | on data without shared words, 2 of 3 seeds: above guessing, and groups beat random and shuffled-label groups on WordNet score and NPMI | met | on the `s` 100 arm only (2 of 3 seeds), small effect, fragile; not met on the `s` 400 arm and on the `s` 200 arm (0 of 3 each); the `s` 200 arm was allowed after the other results were known | `20261002T093158Z_check4b_synonym_groups_s100`, `20261002T130624Z_check4b_synonym_groups_s400`, `20261002T231829Z_check4b_synonym_groups_s200` |
| The final test | test split used once; registered comparison "beats X" only if the whole paired interval is above 0 | not met | "below" word overlap on same-article and next-paragraph pairs; "below" the best-on-dev baseline on all three data sets; no comparison with word overlap on the data without shared words | `20261002T193513Z_final_test_main_table` and the 20 runs named `*_testscore` |

## Registered rules and predictions

The table follows the sections of [registrations.md](registrations.md) in their order and says what came out of each. Details and run ids are in the sections above where a section is named.

| registered (CEST, UTC in brackets) | what it fixed | what came out |
|---|---|---|
| 1 Oct 2026, 11:55 (09:55) | decisions on work that had already run: the changed next-paragraph negatives, `ns_all` as the main synonym data, the demo check with 1 repeat per seed | recorded after the builds and runs it concerns; see Checks 1a and 1b |
| 1 Oct 2026, 20:10 (18:10) | `ns_all` is the main synonym data; the verdict of Check 4b leaves out groups with the four skewed words | applied in all three arms of Check 4b; 150 of the 204 counted groups of the `s` 100 arm are left out |
| 1 Oct 2026, about 20:40 (18:40) | the queue runs 2 or 3 jobs at once only if summed throughput is at least 1.3 times the single job and results are identical | 2 jobs at once under NVIDIA MPS met both conditions, and the queue ran in that form from 2 Oct 2026; this rests on a queue report that is not part of a run folder, and the measured ratios are not given here. The test runs are the 18 runs of 2026-10-02 whose names start with `share_`. No speed number is reported on this page, because most runs shared the GPU |
| 1 Oct 2026, after 20:56 (18:56) | open decision on the accuracy of Check 2a, default tuned accuracy | decided on 2 Oct 2026, 21:27: split-averaged cross-fitted accuracy, chosen post hoc; see Check 2a |
| 1 Oct 2026, 21:23 (19:23) | the 1,000-pass fallback of L3 is queued for seeds 1 to 3 and is cancelled if seeds 2 and 3 learn at 200 passes | not cancelled: the fallback ran for all three seeds (`20261001T215937Z_e2_L3_hier_s1_p1000`, `20261001T224330Z_e2_L3_hier_s2_p1000`, `20261001T232651Z_e2_L3_hier_s3_p1000`); see Check 2a |
| 2 Oct 2026, about 01:45 (23:45 on 1 Oct) | L3 at `s` 100, 200 and 400 on seed 1; early stop for screening runs; evaluation every 5th pass after one run confirms identical results | L3 dev AUC 0.630, 0.701 and 0.866 (`recovery.json` of `20261002T003832Z_e2_recovery_L3_stest_exploratory`; `20261002T003819Z_e2_L3_hier_s1_s400_p300`). The early-stop rule stopped the three slide-10 screen runs at pass 90. The two schedule runs give the same dev AUC, 0.675747035, and the same accuracy (`20261002T030724Z_evalsched_best_every1_s1_exploratory`, `20261002T031354Z_evalsched_best_every5_s1_exploratory`) |
| 2 Oct 2026, about 03:00 (01:00) | what is recorded if the learning-curve run and the earlier 100-pass run agree, and what if they differ | the check ran as `20261002T043040Z_e4_curve_repro_check_posthoc`; its metrics file is empty, so its outcome cannot be read from that file |
| 2 Oct 2026, 03:05 (01:05) | prediction for L3 at `s` 800: AUC at least as high as at `s` 400, purity lower | AUC 0.916 at `s` 800 against 0.866 at `s` 400, so the AUC part holds. The purity of all groups is 0.479 at both values, and the purity of the top-8 rules is 0.523 against 0.554 (`purity.json` of `20261002T023042Z_e2_L3_s800_posthoc_exploratory` and of `20261002T005432Z_e2_purity_L3_equal_footing_exploratory`). One seed; no trade-off is stated |
| same entry, undated notes | two synonym arms, `s` 100 and `s` 400 | both ran; see Check 4b |
| same entry, undated notes | prediction for L2 at `s` 100 with A = 8: purity and recovery rise, AUC stays similar | never tested: the run index has no L2 run with 8 options, and no result was noted |
| same entry, undated notes | the decisive L3 test: unseen-pair scores next to seen-pair scores | AUC 0.859 / 0.819 / 0.841 on unseen pairs against 0.916 / 0.908 / 0.908 on seen pairs; see Check 2a |
| 2 Oct 2026, 05:55 (03:55) | headline `s` by the rule of gates.md | `s` = 100; see Check 3 |
| 2 Oct 2026, 06:00 (04:00) | two proposals: the calibrated test for group comparisons; `s` 200 for the second synonym arm | both decided on 2 Oct 2026, 21:27. The calibrated test decides and changes no verdict: both methods agree on every verdict of Checks 2c, 4a and 4b. The second arm had started at `s` 400 at 13:18 (11:18), and `s` 200 ran as an extra arm, decided after the results of the other two arms were known |
| 2 Oct 2026, 06:06 (04:06) | news topics: `s` screen, choice rule, one-sided reading of "within 2 points" | `s` = 60 for both models; accuracy part not met; see Check 2c |
| 2 Oct 2026, 06:15 (04:15) | slide 10: the shaped model and the `s` choice | `s` = 150 by a tie-break at chance level; see Check 2b |
| 2 Oct 2026, 06:20 (04:20) | Wikipedia flat control with 6,400 flat rules | flat dev AUC 0.678 to 0.680 on same-article pairs and 0.701 to 0.704 on next-paragraph pairs, below the hierarchical model on every seed; see Check 3 |
| 2 Oct 2026, 06:25 (04:25) | unshaped arm K 8 / A 8 next to the main slide-10 arm; shuffled-label runs for Check 4a only after a pass of Check 3 | unshaped arm at chance, 0 of 3 seeds; the shuffled-label runs ran after Check 3 had passed |
| 2 Oct 2026, about 06:35 (04:35) | the cap of the Check 2a fallback is read per rung | applied; under a reading as one total, L2 seeds 2 and 3 would not count |
| 2 Oct 2026, about 07:00 (05:00) | scorer rules for Checks 4a and 4b: NPMI on whole articles, verdict without skewed-word groups, word forms reported both ways, pooled seeds for Check 4a | applied: `metrics.json` of `20261002T093158Z_check4b_synonym_groups_s100` names 2,847 reference articles as the text unit; WordNet covers 957 of the 1,024 words; the verdict of the `s` 100 arm is the same without word-form pairs |
| 2 Oct 2026, 07:02 (05:02) | P by the rule of gates.md | P = 100; see Check 3 |
| 2 Oct 2026, 07:03 (05:03) | the verdict of the Check 2a fallback uses the final model of 1,000 passes and not an earlier pass | applied: the L2 fallback is judged on its final model; see Check 2a. The reading was written down after the result of seed 2 was known |
| 2 Oct 2026, 07:34 (05:34) | cross-fitted tuned accuracy on one fixed split, as a measurement for the decision on Check 2a | L2 at 1,000 passes, which is borderline and not met under the deciding accuracy: 0.918 / 0.896 / 0.900 on the one registered split, seed 3 exactly on the bar; see Check 2a. The logged configuration of that run carries the time 07:40, which was written first and later corrected to 07:34 from file times |
| 2 Oct 2026, 07:38 (05:38) | averaged accuracies and the Check 2c comparison are counted in whole pairs or texts | used: 3,600 of 4,000 pairs for L2 seed 3; 960 texts short against 600 allowed in Check 2c |
| 2 Oct 2026, 10:27 (08:27) | flat L3 at `s` 800 on unseen pairs, 1,280 flat rules | flat AUC 0.506 / 0.510 / 0.503; see Check 2a |
| 2 Oct 2026, 12:24 (10:24) | Check 4a is scored only after the branch decision | not kept: Check 4a was scored at 13:47 (11:47), before the decision of 21:27 |
| 2 Oct 2026, 15:19 (13:19) | frozen settings, baselines, the comparison "beats X" and the 5-seed reporting of the final test | the final test ran once; see the section on the final test. The registration expected "below word overlap" from the dev numbers, and the test gave the same outcome |
| 2 Oct 2026, 21:27 (19:27) | the four open decisions, the slide-6 test and the final test | calibrated test decides; `s` 200 arm runs as an extra arm; branch A; split-averaged accuracy for Check 2a; slide-6 test and final test are run |
| 2 Oct 2026, 21:29 (19:29) | slide-6 paraphrase test, amended 21:39 (19:39) | claim not met on 3 of 3 seeds; see the slide-6 section |
| 2 Oct 2026, 22:11 (20:11) | inputs of the slide-10 test: templates file written at 22:09 (20:09), data, `s` values | the first run on the slide-10 data, the data build, has run id `20261002T201024Z_e3_slide10_data_build`; see Check 2b |
| 2 Oct 2026, 22:28 (20:28) | direct-label fallback of Check 2b | `s` = 21.1; 0 of 5 seeds, judged under reading A of the cap (under reading B the verdict would be "not judged within the cap"); see Check 2b |
| 2 Oct 2026, 23:35 (21:35) | robustness arms R1 and R2, with a fixed reading | R1 learns on 3 of 3 seeds, R2 on none: "the failure is specific to our configuration"; the verdict stands |
| 3 Oct 2026, 00:00 (22:00 on 2 Oct) | two deviations of the fallback: screen runs capped at 30 minutes; two readings of the 2-hour cap | the `s` 5 run stopped at pass 959 and the two other screen runs completed 1,000 passes; reading A 103.9 GPU-minutes, reading B 206.9 minutes (`20261003T001723Z_e3fb_gpu_time`) |
| 3 Oct 2026, 00:02 (22:02 on 2 Oct) | decision: full fallback design, judged under reading A, with a fixed disclosure | applied; the disclosure is in Check 2b |
| 3 Oct 2026, 00:29 (22:29 on 2 Oct) | deletion probe, with a fixed prediction | all 5 seeds accept 100% of the deletion texts: a blacklist, as predicted |
| 3 Oct 2026, 01:08 (23:08 on 2 Oct) | S6-R1, with a fixed reading | dev AUC at least 0.70 on 3 of 3 seeds; the claim is still not met |
| reporting rules, no time of their own | flat controls match the leaf budget, old-convention runs are labelled; both Wikipedia settings shown with AUC and tuned accuracy; early-stopped runs labelled; selection effect stated | applied: see Check 2a, Check 2b and Check 3 |

## Results that were not reported at the time

Four runs of 2 Oct 2026 train the headline setting on same-article pairs, seed 1, 100 passes, each with one option of the package switched: binary inference, AND-group normalisation, a constant update probability, and no clipping of the class sum. The study design lists them as an extra for the case that GPU time is left. Their run ids give start times between 16:25 and 17:13 UTC, before the final test, and no result was reported for them at the time. They have no registration and no verdict. The numbers are dev numbers at pass 100 from the `metrics.json` of each run.

| option switched | run | dev AUC | tuned accuracy | plain accuracy | train accuracy | partner-shuffle AUC |
|---|---|---|---|---|---|---|
| binary inference | `20261002T162502Z_e4_switch_binary_inference_sa_s1` | 0.564 | 0.543 | 0.540 | 0.581 | 0.506 |
| AND-group normalisation | `20261002T164428Z_e4_switch_and_group_norm_sa_s1` | 0.581 | 0.555 | 0.544 | 0.5805 | 0.506 |
| constant update probability | `20261002T165303Z_e4_switch_constant_update_p_sa_s1` | 0.726 | 0.652 | 0.637 | 0.686 | 0.508 |
| no clipping | `20261002T171309Z_e4_switch_no_clipping_sa_s1` | 0.745 | 0.667 | 0.6565 | 0.711 | 0.505 |
| none (the seed-1 headline model) | `20261002T050235Z_e4_head_sa_s1_scoreP100` | 0.745 | 0.667 | 0.6565 | 0.711 | 0.505 |

With binary inference and with AND-group normalisation the dev AUC is far below that of the default run on this seed. With the constant update probability it is about 0.02 lower, and the run without clipping differs from the default run only in the fourth decimal of its AUC. This is one seed per option on dev data, and no conclusion about the options is drawn from it. Seed 1 is the seed whose dev data fixed `s` and P, so the dev numbers of the reference row are slightly optimistic.

## Disclosures

The following disclosures belong to single results. Each stands next to its number in the section named, and they are collected here.

- Check 1a: the demo ran one repeat per seed, not the demo's default of 10.
- Check 1b: the main synonym data has 4 skewed words against "at most about 3" and passes by a decision taken when that count was known. The four words are flagged, and the verdict of Check 4b leaves out the groups that contain them.
- Check 2a, L2: "borderline"; the deciding accuracy was added after the pre-registered result was known, and every reading is shown; verdicts count pairs (3,600 of 4,000), not printed decimals; a share of 0.575 of the OR-groups of seed 1 contains an empty option and is not counted.
- Check 2a, L2: the effect of `s` on purity and recovery is a finding, its mechanism a hypothesis.
- Check 2a, L3: the fixed wording quoted in the section; no trade-off between AUC and purity is stated from one seed.
- Check 2a, flat runs: the flat L1 to L3 runs of 1 Oct 2026 use 400 rules at `s` 21.1 and are labelled "old convention"; the leaf-matched flat L3 runs use an `s` that was not tuned for the flat model.
- Check 2b: A and K were set to the known bracket structure; the model is "a hierarchy shaped to the template"; the unshaped arm is reported next to it and the verdict stays on the main arm.
- Check 2b: `s` = 150 was chosen by a tie-break at chance level, which is noise.
- Check 2b: seed 1 of the pair version, and each of the three screen runs, was stopped early by rule at pass 90.
- Check 2b: the near-miss sentences are rule-generated (a fixed one-word substitution rule, not written independently as planned, written before any run, replacement words present in training), so the near-miss part is weak evidence.
- Check 2b: the limit on firing is met only trivially in the pair version.
- Check 2b: the flat model fails the check too and is not called better at slide 10.
- Check 2b, fallback: the fixed wording about the word blacklist; the disclosure about the two readings of the cap; the final pass of each screen run next to its numbers; the flat 768 runs are beyond the cap under both readings and the verdict does not depend on them; the label "template-like" of the `s` 5 run in the probe file is wrong and reads "reject-all (degenerate)".
- Check 2b, R1 and R2: post hoc; the verdict is unchanged; which of A, K or T matters was not separated.
- Check 2c: the groups are examples only; a measured negative; the context facts are not a rescue; each model's `s` was chosen on its seed-1 dev curve; the screen stops at `s` = 230.
- Check 3: `s` and P were chosen on seed 1's dev data, so its dev numbers are slightly optimistic; the flat control's `s` is not tuned separately; both Wikipedia settings are shown with AUC and tuned accuracy; baseline AUCs are compared with model AUCs only.
- Check 4a: scored before the decision between branch A and branch B; topical coherence, not synonymy.
- Check 4b: "slightly more synonym-like than chance" and nothing stronger; scored before the decision on the method; fragile because most groups are left out; the same verdict without word-form pairs; the `s` 200 arm was decided after the results of the other arms were known; the note that the scorer prints in the `s` 200 run still says that the bootstrap decides, which was out of date after the decision of 21:27 CEST, and both methods agree in that run.
- Final test: "below word overlap"; on the data without shared words the line for word overlap is "by construction, not a comparison"; the logistic regression there is a "word-matching classifier, helpless without shared words".
- Final test, dry runs: the two runs `20261002T134414Z_final_test_report_codepath_test_exploratory` and `20261002T135204Z_final_test_report_codepath_test_exploratory` use a labelled fake shuffled-label input to exercise a code branch. Their numbers are not results.
- Slide 6: K 5 and A 5 were set to the known template; the limits of bracket recovery; the verdict is the same under the original rule and under the amendment; the fixed line that the scorer prints is wrong for the S6-R1 runs.
- Timing: four entries of 2 Oct 2026 were first written with times ahead of the clock and corrected from file times; the logged configuration of the cross-fit run keeps the first-written time.

## Code revisions

No run record of this study hashes gates.md or the study's code. All 319 records carry the hash of the pre-registration of the tm_language_model study and one source tree hash of that study, because the runs were logged through that study's run logger. The code revision of a run is therefore known only from the copies of the scripts inside its run folder. One hashed script, `src/text_driver.py`, was changed after the settings were frozen (2 Oct 2026, 22:28 CEST, a printing option); its entry in frozen_settings_hashes.json refers to the version that the news-topic runs used (see the [hash table](../../../docs/hash-table.md)).

Thirteen analysis scripts existed only as copies inside run folders and were recovered from there; they are in `src/recovered/`. The script of one run, `20261003T001723Z_e3fb_gpu_time`, was not retained, so the two GPU-time totals of the fallback cannot be traced to code. The scripts `src/slide6.py` and the test files of the two sentence families are not distributed, because they carry sentences of the project proposal.

The environment record of the GPU jobs names the interpreter of the run logger, not the interpreter that ran the job. Package versions of GPU jobs are therefore not taken from it. The frozen settings name the package as PyHierarchicalTsetlinMachineCUDA at commit 5b4a223. The [known limitations](../../../docs/known-limitations.md) page lists these points for the whole repository.
