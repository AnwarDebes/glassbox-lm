# or_group_tm: a learning rule for Tsetlin machine clauses made of OR-groups of words

## Question

Can a Tsetlin machine learn clauses that are ANDs of OR-groups of single words, the clause form of the supervisor's project proposal (unpublished), so that its groups become the word groups that the data hold? The study [hierarchical_tm](../hierarchical_tm/README.md) tested the procedure of the proposal with the package PyHierarchicalTsetlinMachineCUDA and ended mostly negative. This study keeps the clause form, the data and the scorers of that study and replaces the way the clauses are learned: it writes a learning rule for such clauses, with one Tsetlin automaton per clause, group and word, and tests it on the planted-group pair tests, on the slide-10 sentence family and on the Wikipedia paragraph pairs of hierarchical_tm, and on a synthetic sequence task. All runs were made on 2026-11-12. Every number on this page is read from the file named next to it; [docs/results.md](docs/results.md) gives all of them per plan.

## What the study found

- **Planted groups (E1, registered): met.** On worlds of five new seeds the rule meets the three bars of Check 2a of hierarchical_tm on 5 of 5 seeds each: plain dev accuracy 0.9998 to 1.0000, group recovery 1.0, plain accuracy on unseen word pairs 0.9990 to 1.0000 (`logs/RUNS.md`). A separate implementation trained the ten models again and gave every state and number back (`verification/recompute_e1.out`). The first form of the rule (E0) missed the bars; the change that made the difference, one seed event per example (E0b), was made after the results of E0 on the same files were seen.
- **Slide 10 with direct labels (E3, registered): R1 met on 5 of 5 registered seeds** (both flags of the scorer; exact 95% interval of the share 0.478 to 1.000). On 20 further seeds of the same pass, which carry no rule, both flags held on 8 of 20 (0.191 to 0.639), so on 13 of 25 seeds pooled (0.313 to 0.722); the 288 reading held on 23 of 25. The model as a whole (any of its 20 clauses) is not template-like on any of the 25 seeds (`logs/e3_report.json`). An independent recomputation matched every number of the report. The wording that E3 fixed for this result, the tie-break that moves its counts and a correction to its text are in the section on slide 10 below.
- **Slide 10 as pairs, the procedure of the proposal (E2, design): not met.** No arm formed a clause for the family on the DEV sentences, and no registered run was made (`logs/RUNS_slide10.md`).
- **Wikipedia pairs (E5, registered): the three rules are met, as predicted from DEV.** On the test pairs of `sa`, `np` and `ns_all` the OR-group rule (arm TB, chosen on DEV) is above the stock model in its registered setting by +0.091, +0.101 and +0.098 AUC of the vote sum, each with both intervals above 0 (`logs/e5_report.json`). These are the only rules of E5, and it is not a comparison at matched size or cost. Readings with no rule: the vote sum minus word overlap is +0.009 [+0.002, +0.016] on `sa` (the interval includes 0 with the articles of both paragraphs resampled and without the listed duplicate articles) and -0.005 [-0.011, +0.002] on `np`; the vote sum minus PPMI cosine has its interval below 0 on all three versions, and minus the logistic regression on `sa` and `np`. An independent recomputation matched the report. A correction to the text of E5 (its 15 models had been trained once before the freeze, in a rehearsal scored on DEV pairs only) is in the section on the Wikipedia pairs.
- **Wikipedia pairs (E4, design on DEV).** Tied groups raise the DEV AUC of the vote sum on same-article pairs from 0.7774 to 0.8255 (seed 1); the go-bar G2 of E4 for a test against word overlap was missed in the vote sum and passed only in the ratio reading, a reading defined after the miss (`logs/RUNS_wiki.md`), so E5 has no rule against word overlap.
- **Synthetic sequences (E6, exploratory): not met.** Test accuracy at most 0.7983 against a bar of 0.95 (`logs/RUNS_seq.md`).

## Why it was built

hierarchical_tm tested the procedure of the proposal from 1 to 3 Oct 2026 with one version of the package, commit 5b4a223 ([its README](../hierarchical_tm/README.md), section Environment). The procedure did not hold there ([its docs/results.md](../hierarchical_tm/docs/results.md)): of the artificial tests only L1 met Check 2a (L2 borderline, and at L3 no hidden group was recovered on any seed); no learned rule captured the slide-10 family (0 of 5 seeds in the pair version and 0 of 5 in the direct-label fallback); and its one final test on Wikipedia pairs was below word overlap (test AUC 0.735 and 0.750 against 0.817 and 0.857). Three days with one version of the package left open whether the misses came from the clause form or from the way the clauses were learned. E0 gives three reasons, read from the source of the package, why its feedback was expected to struggle on the planted tests; they are a hypothesis, and this study does not test the package, in that version or in a later one (E6 reads an example of the package at commit f40b2a3, read on 2026-11-12). It tests whether a rule written for this clause form meets the bars of the same checks.

## The OR-group rule in plain words

A clause is an AND of OR-groups, and a group is an OR of single words: the clause fires on a text when every one of its groups finds at least one of its words in the text. For a pair of texts a clause has K groups for each text; with tied groups (below) the two texts share one set of groups.

- Each group holds one Tsetlin automaton per word of the vocabulary, so there is one automaton for every clause, group and word. The state is an integer from -32 to 31 and starts below 0 (at -1 in the planted and slide-10 settings, at -3 in the Wikipedia setting); the word is a member of the group when its state is 0 or more.
- A group is true on a text when one of its members occurs in the text. A group with no member is false, so a clause that has learned nothing does not fire. In the package that hierarchical_tm tested, an option of a group that includes no word is true, so a group with an empty option is always true (its README, study rule 7).
- The vote sum is the number of clauses that fire; all clauses vote for the positive class. One design arm with half of the clauses voting against reached a DEV AUC of 0.5047 and was dropped (`logs/RUNS_wiki.md`).

Learning goes example by example. Each clause takes part with the vote gate of a Tsetlin machine: on a positive example with probability (T - v) / 2T and on a negative one with (T + v) / 2T, where v is the vote sum clipped to the range from -T to T, so that clauses stop learning from examples that the vote gets right by a margin. A clause that takes part gets one of five events:

- **seed**: the clause should fire and every group that is false has no member (for a new clause, all of its groups). The words of the text become members of the empty groups: each word goes to one empty group of its text at random, or, with seed-all, to every empty group. In the registered settings at most one clause is seeded per example.
- **reinforce**: the clause should fire and fires. Its members that occur in the text gain 1: all of them; or, with reinforce critical, only a member that is the one present member of its group; or, with reinforce share, one present member per group, picked at random.
- **recruit**: the clause should fire, does not fire, and the mode names some of its false groups: mode near when exactly one group is false, mode side when every group of the other text is true, mode atleast when at least a given number of groups is true. The words of the text that are not yet members of the named groups gain 1, so that a word whose state reaches 0 joins the group.
- **reject**: the clause should not fire but fires. The present members of one group lose r (2 in most settings): a group picked at random, or, with reject weakest, the firing group whose strongest present member is weakest.
- **repress**: the clause should not fire and does not. The present non-members of the named false groups lose 1.

A word can become a member only after it occurred in an example on which the clause should fire. A clause therefore cannot learn a list of words that only the negative examples contain, which is how the direct-label fallback of hierarchical_tm separated the slide-10 sentences (a negative-class word blacklist, its docs/results.md, Check 2b).

**Tied groups** (pairs of texts): group k of the first text and group k of the second are one group; every change is made to both, and a new clause is seeded with the words of both texts. With one group per text (K 1) a tied clause is a single OR of words, a concept in the sense of slide 8 of the proposal, and it fires on a pair when both texts contain one of its words.

**The ratio reading** (pairs of texts): besides the vote sum, a pair can be scored by the number of clauses that fire divided by the number of clauses whose groups are all true on at least one of the two texts. With tied groups and K 1 this is the Jaccard overlap of the two texts counted in concepts instead of words. It was defined on the DEV pairs after the tied runs had been seen (`logs/RUNS_wiki.md`, note of 16:51:28Z).

The engine is `src/orgroup_tm.py` (numba, CPU only). Its docstring states the rule in full, including an optional decay step that only the slide-10 design arms used, and `tests/test_orgroup_tm.py` tests each event.

## Plans and registrations

Each plan was fixed, with the SHA-256 of its text and of the code, before the first run it governs; [registrations/INDEX.md](registrations/INDEX.md) lists them with their times and hashes. Times below are UTC on 2026-11-12.

| plan | kind, time fixed | data | what it found |
|---|---|---|---|
| [E0](registrations/E0_planted_feasibility.md) | design, 15:53:26Z | planted L2 and L3, the files of seeds 1 to 3 of hierarchical_tm | bar not met: with modes near and side the groups are pure (purity 1.0), but plain dev accuracy is 0.6817 to 0.8150 and recovery 0.125 to 0.625, and 3 to 5 of the 8 hidden groups have no clause; mode all gives mixed groups (`logs/RUNS.md`, `logs/e0_diag_allocation.out`) |
| [E0b](registrations/E0b_one_seed_per_example.md) | post hoc revision of E0, 15:56:18Z | the same files | one seed event per example: the bar of E0 is met on these files, post hoc (`logs/RUNS.md`) |
| [E1](registrations/E1_planted_confirmation.md) | registered confirmation, 15:58:27Z | worlds of the new seeds 11 to 15 | met: R1, R2 and R3 on 5 of 5 seeds each; recomputed by a separate implementation (`verification/recompute_e1.out`) |
| [E2](registrations/E2_slide10_design.md) and its [note](registrations/E2_note_decay.md) | design, 16:04:09Z and 16:06:22Z | slide-10 DEV sentences | direct labels: arm A6 (seed-all) is the only arm that met the go-bar, at exactly 3 of 5 seeds; pairs: no clause for the family (`logs/RUNS_slide10.md`) |
| [E4](registrations/E4_wikipedia_design.md) | design, 16:20:17Z | Wikipedia DEV pairs | tied groups and the ratio reading; G1 met in both readings; G2 missed in the vote sum and passed only in the ratio reading, a reading defined after the miss (`logs/RUNS_wiki.md`) |
| [E6](registrations/E6_sequence_demo.md) | exploratory, 17:03:46Z | synthetic sequences, the task of the package's sequence example | bar not met (`logs/RUNS_seq.md`) |
| [E3](registrations/E3_slide10_text.md) | registered confirmation, 17:48:56Z | slide-10 EVAL sentences, direct labels | met: R1 (both flags) on 5 of 5 registered seeds (0.478 to 1.000); both flags on 8 of 20 further seeds (0.191 to 0.639) and 13 of 25 pooled (0.313 to 0.722); every number recomputed independently (`logs/e3_report.json`) |
| [E5](registrations/E5_wikipedia_test.md) | registered test, 21:05:33Z | Wikipedia test pairs (10,000 per version; their second registered use) | met, as predicted from DEV: R1 on `sa`, `np` and `ns_all` (above the stock model in its registered setting by +0.091, +0.101 and +0.098 AUC of the vote sum, both intervals above 0); no rule against word overlap; recomputed independently (`logs/e5_report.json`) |


## Results

| test | rule or bar | this study | reference in hierarchical_tm (other model, seeds or budget) | source |
|---|---|---|---|---|
| planted L2, worlds of seeds 11 to 15 (E1) | plain dev accuracy at least 0.90 and recovery at least 0.80 on 4 of 5 seeds | met on 5 of 5: accuracy 1.0000 and recovery 1.0 | the package on seeds 1 to 3, 100 passes: accuracy 0.523 / 0.522 / 0.572, recovery 0.625 / 0.5 / 0.75 | `logs/RUNS.md`, `verification/recompute_e1.out` |
| planted L3 (E1) | as at L2, and plain accuracy on the unseen pairs at least 0.85 | met on 5 of 5: accuracy 0.9998 to 1.0000, recovery 1.0, unseen pairs 0.9990 to 1.0000 | the package at `s` 800, 300 passes: accuracy 0.8365 / 0.821 / 0.821, recovery 0 / 0 / 0, unseen pairs 0.788 / 0.74 / 0.752 (tuned) | the same |
| slide 10, direct labels, EVAL (E3) | R1: both flags of the scorer on at least 3 of 5 seeds | met, 5 of 5 [0.478, 1.000]; further seeds, no rule: 8 of 20 [0.191, 0.639]; pooled 13 of 25 [0.313, 0.722] | the direct-label fallback, shaped to the template: 0 of 5 | `logs/e3_report.json` |
| slide 10, the model as a whole (E3, no rule) | template-like: accepts at least 90% of the intact sentences and at most 5% of the deletion texts | 0 of 25 seeds; it accepts 0.117 to 0.304 of the deletion texts on the registered seeds | the fallback accepted 100% of the deletion texts on 5 of 5 seeds | `logs/e3_report.json` |
| slide 10, pairs, DEV (E2) | go-bar: both flags on 3 of 5 DEV seeds | not met: in 12 of 15 runs the best rule covers no DEV family sentence; the best of the other three covers 0.693 and fires on 0.369 of the broken sentences | the pair version: 0 of 5 seeds on EVAL | `logs/RUNS_slide10.md` |
| Wikipedia, sa and np, DEV (E4, design) | G1: mean DEV AUC of seeds 1 to 3 at least 0.759 (sa) and 0.764 (np) | met: vote sum 0.8272 and 0.8567, ratio reading 0.8470 and 0.8787 | the package, headline setting, DEV AUC: sa 0.745 / 0.739 / 0.733, np 0.752 / 0.728 / 0.751 | `logs/RUNS_wiki.md`, note of 18:12:43Z |
| the same (E4, design) | G2: at least 0.830 (sa) and 0.875 (np), the go-bar for a test against word overlap | missed in the vote sum; passed only in the ratio reading, a reading defined after the miss | word overlap, DEV AUC: sa 0.819, np 0.864 | the same |
| synthetic sequences, setting (a) (E6) | test accuracy at least 0.95 on 3 of 3 data seeds | not met: 0.7971 / 0.7865 / 0.7983 | no run of the package; memoriser 0.5133 / 0.5090 / 0.5164 and logistic regression 0.4992 / 0.4969 / 0.5034 on the same data | `logs/RUNS_seq.md` |

The column of hierarchical_tm is context, not a head-to-head comparison: the package ran on other seeds (the planted tests), was shaped to the template (slide 10) and ran in the settings that hierarchical_tm registered (for the Wikipedia pairs its `s` and its number of passes were chosen by its rules on the DEV pairs of one seed), while the rule of this study was designed on the DEV data of each test. The references are read from [../hierarchical_tm/docs/results.md](../hierarchical_tm/docs/results.md) and, for Wikipedia, from the table of E4.

## Slide 10 in the wording of E3

E3 fixed the only wording for its result and two companion sentences that go with it. The report `logs/e3_report.json` prints them as follows:

> With direct family labels and a setting chosen on the DEV sentences, on 5 of 5 new seeds the best single clause of the model (picked by the study's scorer on the training sentences) accepts at least 90% of the 101 held-back sentences of the slide-10 template family (and of the 80 among them that are in the 216 view), fires on at most 5% of the negated, near-miss and broken sentences, and recovers at least 4 of the 6 brackets in the 288 reading; 5 of 5 seeds when both flags of the scorer are required.

> The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911).

> The pair version (the procedure of the proposal) met none of the bars on the DEV sentences.

Held-back sentences are combinations of the alternatives of the six brackets that are in neither the training nor the DEV sentences, while every alternative and every word of them occurs in the training sentences of the family.

The counts, which are never to be read without each other (`logs/e3_report.json`): both flags of the scorer on 5 of 5 registered seeds (exact 95% interval of the share 0.478 to 1.000); on 20 further seeds of the same pass, which carry no rule, on 8 of 20 (0.191 to 0.639); pooled, on 13 of 25 (0.313 to 0.722). The 288 reading holds on 23 of 25 (0.740 to 0.990). On 10 of the 12 seeds without both flags only the flag of the 216 view fails, on its bracket count, and seeds 21 and 25 fail both flags ([docs/results.md](docs/results.md), section E3). The registered 5 of 5 lies at the lucky end of what the 25 seeds show: with a share of 13 of 25, five of five comes up with probability 0.038 (a post hoc diagnostic of the verification below, not a registered test).

The tie-break. The best rule is the first of the clauses that tie at the top score on the training sentences, as E3 fixed in advance. Other choices among the tied clauses move the counts: if any tied clause counted, both flags would hold on 5 of 5 and 12 of 20 seeds; if every tied clause had to pass, on 3 of 5 and 4 of 20 (verification below).

Independent verification of E3 (2026-11-12, verdict MATCH; its record is not distributed). Separate code, using the standard library and numpy and importing nothing from this study, recomputed from the 25 stored models 41 fields per seed, the counts and exact intervals of the three sets of seeds, the ranges and the four printed sentences, with 0 mismatches. The pinned runner made the 25 models again on the DEV sentences only, and they came out identical. The record was fixed before every run (the first started 1 second after the freeze), and no model was scored on the EVAL sentences outside the 25 registered runs. The text that was frozen, draft 2 of E3, had no independent check: draft 1 had been checked from two sides and found not ready, and draft 2 was checked once more, not independently, 27 seconds before the freeze. None of these checks is a review by another person or by the supervisor.

The record of E3 (`registrations/E3_slide10_text.frozen.json`) holds the hash of every file that was under `src/` and `tests/` at its freeze. The work on E5 added files there and changed `src/run_wiki.py` and `src/wiki_report.py` after the E3 report was written, so a check of the whole tree against the E3 record (`src/e3_common.py --check`) now lists those files. None of them is on the path of E3: the engine, the E3 runner, probe, report and common module, `src/metrics.py`, the job script and every pinned data and scorer file still have the hashes of the record.

Correction to the text of E3 (written 2026-11-12T18:23:39Z; the registration itself stays as it was fixed). E3 says, in its section on the data and in its limits, that the data allow at most 5 of the 6 brackets to be recovered. That holds only for a clause with one group per bracket (`logs/slide10_facts.json`). The verification above found a hand-written clause of 8 groups, the clause with one group per bracket and one group for a fixed word of the template plus one blocking group of 4 words, that accepts 86 of 86 TRAIN and 101 of 101 DEV family sentences, fires on no TRAIN or DEV negative of any kind, and recovers 6 of 6 brackets in the 288 reading (5 of 6 in the 216 reading); the model of E3 has 12 groups. Bracket 5 is therefore not out of reach by the data: that no seed recovered it (its Jaccard is at most 0.667 on all 25 seeds, `logs/e3_report.json`) is a property of this learner. No number of E3 changes.

The model of E3 is a different model from the one that hierarchical_tm tested, and that study's verdict on Check 2b stands.

## Wikipedia pairs

Design numbers, on the DEV pairs only (`logs/RUNS_wiki.md`; all in [docs/results.md](docs/results.md)). The data are the pair versions of hierarchical_tm with 1,024 words: `sa` (same article or not), `np` (next paragraph or not) and `ns_all` (next paragraph with every shared word removed). The arm chosen in E4, TB, has 2,000 clauses with tied groups (K 1), T 40, mode side, reinforce share, reject weakest, r 2, initial state -3 and 10 passes. On seeds 1 to 3 its mean DEV AUC is 0.8272 (sa), 0.8567 (np) and 0.7031 (ns_all) with the vote sum, and 0.8470, 0.8787 and 0.7336 with the ratio reading (note of 18:12:43Z), against the means 0.739, 0.744 and 0.618 of the headline setting of the package and 0.819 and 0.864 for word overlap on sa and np (E4). On the DEV pairs of sa, seed 1, the vote sum of the tied model is 0.0064 [0.0004, 0.0121] above word overlap and below PPMI word-vector cosine (0.8538) and the best logistic regression of hierarchical_tm (0.8713) (`logs/design/wiki_report_dev_sa_tied_s1_design.json`). The OR-groups of that design run hold 2.97 words on average and score above frequency-matched random groups on NPMI (+0.140 [0.134, 0.147]) and on the WordNet score (+0.071 [0.060, 0.083]), against +0.032 and +0.013 for the groups of the package, which hold 63.21 words on average (`logs/design/score_groups_sa_tied_s1_vs_stock.json`); this is topical coherence and WordNet relatedness of one run, not synonym sets. None of these numbers is a registered result.

### The registered test (E5)

Fixed on 2026-11-12T21:05:33Z (`registrations/E5_wikipedia_test.md`, its record next to it). On the Wikipedia pairs the three rules of E5 are met, as predicted from DEV: on the test pairs of `sa`, `np` and `ns_all` the OR-group rule (arm TB, chosen on DEV) is above the stock model in its registered setting by +0.091, +0.101 and +0.098 AUC of the vote sum, each with both intervals above 0 (lower ends at least +0.080). These are the only rules of E5. All 199 checks of the report passed (`logs/e5_report.json`).

| test pairs (10,000 per version) | OR-group rule, arm TB, vote sum (mean AUC of 5 seeds) | stock model in its registered setting (mean AUC of its 5 seeds) | difference, paired article bootstrap (95%) | with the seeds resampled | rule |
|---|---|---|---|---|---|
| `sa` | 0.8255 | 0.7348 | +0.0907 [+0.0815, +0.0993] | [+0.0803, +0.1009] | R1-sa met, as predicted from DEV |
| `np` | 0.8519 | 0.7504 | +0.1015 [+0.0934, +0.1096] | [+0.0895, +0.1145] | R1-np met, as predicted from DEV |
| `ns_all` | 0.7114 | 0.6136 | +0.0978 [+0.0850, +0.1098] | [+0.0842, +0.1107] | R1-ns_all met, as predicted from DEV |

What this comparison is. The stock model is the setting that hierarchical_tm registered (its `s` and its number of passes were chosen by its rules on the DEV pairs of one seed). Its second setting (2,000 rules) was about 0.03 (`sa`) and 0.05 (`np`) higher on DEV (the reference table of E4) and was never scored on the test pairs, so on DEV the gap to it is about +0.06 and +0.07. Arm TB uses tied groups, which were not in the plan of E4. It is not a comparison at matched size or cost (the stock model: 50 rules of 64 options, 100 passes, another package, on a GPU; arm TB: 2,000 clauses, 10 passes of 17.2 to 21.0 s on one CPU thread), and "above the stock model" is about these two models as they were set. A met R1 was the expected outcome and is not news: the differences on DEV were +0.090 / +0.108 / +0.081 (seeds 1 to 3) and on the test pairs they are +0.091 / +0.101 / +0.098 (seeds 11 to 15). Each of the 5 registered seeds is above each of the 5 stock seeds in AUC of the vote sum; the smallest gap is +0.081 (`sa`), +0.094 (`np`) and +0.092 (`ns_all`) (arithmetic on the per-seed AUCs of the report; no rule).

Readings with no rule (each registered with its prediction; the 16 sentences of the report are printed whole in [docs/results.md](docs/results.md)). Each difference is given with the paired article bootstrap interval and then the interval with the seeds resampled.

- **Word overlap.** G2 was missed in the vote sum and passed only in the ratio reading, a reading defined after the miss, so the comparison with word overlap carries no rule. The vote sum minus word overlap: +0.009 [+0.002, +0.016] [+0.001, +0.016] on `sa` and -0.005 [-0.011, +0.002] [-0.012, +0.002] on `np`; for the vote sum on `sa` the interval includes 0 with the articles of both paragraphs resampled ([-0.001, +0.018]) and without the pairs that touch the 242 listed articles (+0.006 [-0.003, +0.013] [-0.002, +0.014]). The ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after G2 was missed in the vote sum; no rule) minus word overlap: +0.030 [+0.023, +0.036] [+0.022, +0.037] on `sa` and +0.017 [+0.010, +0.023] [+0.010, +0.023] on `np`; these two differences include the pairs that touch the listed articles, and E5 registered no reading without them for the ratio reading.
- **tf-idf cosine, PPMI cosine and the logistic regression** (first the vote sum; then, in brackets, the ratio reading, defined on DEV after G2 was missed, no rule). Minus tf-idf cosine: +0.004 [-0.003, +0.011] [-0.003, +0.011], interval includes 0 (+0.025 [+0.019, +0.031] [+0.019, +0.032], interval above 0) on `sa`; -0.010 [-0.016, -0.004] [-0.017, -0.004], interval below 0 (+0.011 [+0.005, +0.017] [+0.005, +0.017], interval above 0) on `np`. Minus PPMI cosine: -0.026 [-0.034, -0.018] [-0.034, -0.018], interval below 0 (-0.005 [-0.013, +0.002] [-0.013, +0.003], interval includes 0) on `sa`; -0.034 [-0.040, -0.027] [-0.041, -0.026], interval below 0 (-0.012 [-0.019, -0.006] [-0.020, -0.005], interval below 0) on `np`; -0.100 [-0.110, -0.090] [-0.110, -0.090], interval below 0 (-0.069 [-0.079, -0.060] [-0.079, -0.060], interval below 0) on `ns_all`. Minus the logistic regression: -0.044 [-0.052, -0.036] [-0.052, -0.035], interval below 0 (-0.023 [-0.031, -0.015] [-0.031, -0.015], interval below 0) on `sa`; -0.051 [-0.059, -0.044] [-0.059, -0.044], interval below 0 (-0.030 [-0.037, -0.023] [-0.037, -0.023], interval below 0) on `np`.
- **`ns_all`**, where no pair shares a word and word overlap is 0.500 by construction: the mean test AUC is 0.711 in the vote sum and 0.742 in the ratio reading (defined on DEV; no rule), against 0.614 for the stock model in its registered setting and 0.811 for PPMI cosine.
- **Without the listed duplicate articles** (the test pairs that touch one of the 242 test articles with a paragraph whose 3,072-word set equals a training paragraph; 1,545 of the 71,639 test paragraphs are such paragraphs): the vote sum minus the stock model is +0.091 [+0.081, +0.101] [+0.080, +0.102] (`sa`), +0.103 [+0.094, +0.112] [+0.091, +0.116] (`np`) and +0.101 [+0.089, +0.115] [+0.087, +0.115] (`ns_all`).
- **Word groups** (`sa`, the five registered models; not independently recomputed): 8,972 groups with at least two words (of 10,000), 2.97 words on average; NPMI over reference articles that are in no split 0.219 against 0.078 for random groups of words of the same frequencies. The stock model's 316 counted groups hold 63.5 words on average (0.120 against 0.089); the two sets of groups are not matched in size.
- **Cost.** Each registered run took 192 to 230 s on one CPU thread (10 passes of 17.2 to 21.0 s); 10,516 to 16,810 stored members per model ([docs/cost.md](docs/cost.md)).

Correction to the text of E5 (written 2026-11-12T22:56:45Z; the registration itself stays as it was fixed). Its sentence "Seeds 11 to 15 were never run" is wrong. The same 15 models had been trained once before the freeze (2026-11-12T20:20:40Z to 20:43:59Z) in a scratch rehearsal of the registered job and scored there on 10,000 DEV pairs per data version (vote sum minus the stock model on those DEV pairs: +0.087 [+0.078, +0.096], +0.106 [+0.097, +0.115] and +0.082 [+0.069, +0.094]). These numbers had been seen before the freeze, and the list "What I knew" of the registration lacks them. No test pair was read in the rehearsal; the registered runs gave the 15 models back bit for bit; seeds, code and setting were fixed before the first rehearsal number. "5 new seeds" in the sentences of the report therefore means seeds that were not used in the design (seeds 1 to 3). The rehearsal's record is not distributed.

How the text was checked. The first draft of E5 was checked from two sides and found not ready (1 blocker, 10 majors, 19 minors); the comparison with word overlap was taken out of the rules after that. The text that was frozen, draft 2, was checked only by its author, 44 seconds before the freeze, and had no independent check. None of these checks is a review by another person or by the supervisor. The job ran three one-thread streams at once, as the frozen text declares.

Verification of E5 (2026-11-12, verdict MATCH; its record is not distributed). Separate code gave 837 of the 1,098 numeric fields of the report back to the last digit, with no mismatch; the record was fixed at 21:05:33Z, 12 s before the first run started; every pinned file matches the record; no model of this study was scored on a test pair outside the 15 registered runs, as far as run folders and score files show. Not recomputed: the word-group scores. The verification read the test pairs again; it reproduced the registered numbers and scored no new model.

Uses of the test pairs. E5 is their second registered use and the first by this study: hierarchical_tm scored them in its one final test and read its saved outputs again on 2026-10-25. The verification of 2026-11-12 read them once more. Any further scoring needs a new registration that lists the numbers of E5.

## Limits

- One day of runs. The rule, its settings and two readings were chosen on development data of the same tests: E0b changed the rule after E0's results on the same files; the slide-10 refinements, seed-all, K 12 and the 288 reading were chosen on the DEV sentences; the ratio reading was defined after the tied Wikipedia rows had been seen; min true 4 in E6 was set from the structure of the task.
- The planted worlds are artificial, and E1 confirms the rule on worlds of the same generator it was designed on.
- Slide 10 is one template family with 86 training, 101 DEV and 101 EVAL sentences, and the registered result uses direct family labels, not the self-supervised pairs of the proposal (`logs/slide10_facts.json`). The near-miss sentences are rule-generated. A clause with one group per bracket can recover at most 5 of the 6 brackets (same file); a clause with more groups can recover all 6 (the correction in the section on slide 10), and no seed of this learner recovered bracket 5.
- A group is an OR of single words, not of phrases: the best rule accepts 0.861 to 0.907 of the family sentences from which one word of a multi-word alternative was removed (registered seeds, `logs/e3_report.json`). The best rule is picked from 20 clauses by the scorer with the labels of the training sentences; only 1 to 4 of the 20 clauses are template-like by themselves.
- The tie-break fixed in E3 (the first of the tied clauses) moves the counts: other choices among the tied clauses give both flags on 3 to 5 of the 5 registered seeds and on 4 to 12 of the 20 further seeds.
- R1 is met on 5 of 5 registered seeds (0.478 to 1.000), but both flags held on 8 of the 20 further seeds (0.191 to 0.639) and on 13 of 25 pooled (0.313 to 0.722): over 25 seeds the share is about one half.
- The frozen text of E3, draft 2, had no independent check before the freeze (section on slide 10).
- Wikipedia: design numbers on one set of 40,000 training pairs and 1,024 mostly frequent words, with duplicate articles across the splits (E4).
- No comparison with the package on equal terms was run: other seeds, budgets and shaping. The engine and the runner were revised during the day; the runs of E0 to E1 used earlier revisions of the engine. A rerun of the ten E1 runs with the engine now in `src/` gave the same numbers in every row (note at the end of `logs/RUNS.md`).
- No convergence analysis exists for this rule.
- The run folders and the data of hierarchical_tm are not distributed; see the section on running the tests.

## What the study does not claim

- That this rule is the hierarchical Tsetlin machine or the model of the proposal. It is a different model, flat clauses of OR-groups with feedback of its own, and the package was not run in this study.
- That any verdict of hierarchical_tm is reversed or corrected. Its Check 2a, Check 2b and final test stand as registered there.
- That the rule is better or worse than the package, or anything about a fault of the package.
- That a clause learned the template or is exactly the family: the best rule accepts more than the family (the one-word reading), no seed recovered more than 5 of the 6 brackets in the 288 reading, and the model as a whole accepts 0.117 to 0.304 of the deletion texts on the registered seeds.
- That the family is learned self-supervised: the pair version met none of the bars on the DEV sentences.
- That the brackets or the Wikipedia groups are word classes or synonyms, or that the model understands negation.
- Anything about real text beyond the Wikipedia design numbers, about large language models, scale, energy or the state of the art.
- That anything here is a first (see the section on prior art).

## Relation to the project proposal, slide by slide

| slide | what it shows | what this study did with it |
|---|---|---|
| 4 | reasoning by elimination: of two options, when one is ruled out, the other holds | the recruit event of mode near has this shape: the clause should fire, every group but one is true, so the missing word is looked for in that one group, and the words present join it. An analogy for the feedback, not a test of reasoning |
| 6 | paraphrases of one statement | not used; no slide-6 run in this study |
| 7 | the answer of deep learning: representations learned by projection | no neural network at any stage of this study, and no comparison with one |
| 8 | a logical answer: a concept as an OR of words | the clause form of this study; with tied groups and K 1 every clause of the Wikipedia runs is one such OR of words |
| 10 | one clause for a sentence family: a template with six brackets of alternatives (288 sentences, 216 in the view of the slide), next to a twin of opposite meaning that differs in a few words | E2 (design) and E3 (registered, direct labels): R1 met on 5 of 5 registered seeds; both flags on 8 of 20 further seeds and 13 of 25 pooled; the pair version met no bar on DEV |
| 11 | the self-supervised procedure: two paragraphs or sentences from the same document or not, learned with the hierarchical package | E4 (design) and E5 (registered) run this procedure on the Wikipedia pairs of hierarchical_tm with the OR-group rule: on the test pairs arm TB is above the stock model in its registered setting by +0.091, +0.101 and +0.098 AUC of the vote sum, as predicted from DEV (not a comparison at matched size or cost); the comparisons with word overlap, PPMI cosine and the logistic regression are readings with no rule (section on the Wikipedia pairs); the pair version of E2 runs the procedure on sentences and met no bar |
| 12 | the vision of the project | nothing in this study bears on it |

Slides 1 to 3 and 5 give the background (the Tsetlin automaton, propositional logic, a convergence result for Tsetlin machines), and slide 9 holds figures only; this study tests none of them.

## Prior art

Neither the clause form nor the training signal is new. A conjunction of disjunctions of words is a monotone formula in conjunctive normal form. Valiant ("A Theory of the Learnable", Communications of the ACM 27(11): 1134-1142, 1984) showed that k-CNF formulas, ANDs of ORs of at most k literals, can be learned from positive examples in the probabilistic model of learning that the paper introduced (PAC learning), by deleting the clauses that a positive example makes false. Littlestone's Winnow ("Learning Quickly When Irrelevant Attributes Abound: A New Linear-threshold Algorithm", Machine Learning 2(4): 285-318, 1988) learns disjunctions with a bounded number of mistakes, raising the weights of the attributes present after a missed positive and lowering or removing them after a false positive, much as the recruit and reject events do here, group by group. That words which fill the same slot in the same contexts form a class, and that such classes can be found in raw text, underlies the class-based models of Brown, Della Pietra, de Souza, Lai and Mercer ("Class-Based n-gram Models of Natural Language", Computational Linguistics 18(4): 467-479, 1992) and ADIOS (Solan, Horn, Ruppin and Edelman, "Unsupervised learning of natural languages", PNAS 102(33): 11629-11634, 2005), whose equivalence classes are the words that fill one slot of a learned pattern, as the alternatives of a bracket of slide 10 do. What this study adds is a learning rule of this kind inside a Tsetlin machine, with one automaton per clause, group and word under the vote gate of the machine, and its tests on the data and with the scorers of hierarchical_tm. No claim of a first is made.

## Files

```
or_group_tm/
  README.md
  docs/results.md        every number per plan, with the file it is read from
  registrations/         the plans in the order they were fixed (INDEX.md), the record of E3 and its job script
  src/                   the engine, the runners, the probes and reports of E3, the Wikipedia readings, the freeze tool
  tests/                 unit, smoke and pipeline tests (plain scripts)
  verification/          the independent recomputation of E1 and its output
  logs/                  run logs (RUNS*.md), the reports of E3, data facts, design outputs, the outputs of the E3 jobs
  env/requirements.txt   the package versions of the virtualenv
```

- Engine and scores: `src/orgroup_tm.py`, `src/metrics.py`.
- Runners, each appending one row per run to its log: `src/run_pcp.py` (planted tests, `logs/RUNS.md`), `src/run_slide10.py` (slide 10, `logs/RUNS_slide10.md`), `src/run_wiki.py` (Wikipedia pairs, `logs/RUNS_wiki.md`), `src/run_seq.py` (synthetic sequences, `logs/RUNS_seq.md`).
- E3: `src/probe_slide10.py` (deletion and one-word texts), `src/e3_common.py` (what E3 fixes and the guards), `src/e3_report.py` (checks and report), `src/slide10_facts.py` (data facts), `registrations/E3_job.sh` (the whole job, dry or registered).
- Wikipedia readings: `src/rescore_wiki.py` (ratio reading from a saved model), `src/wiki_report.py` (baselines and the paired article bootstrap of hierarchical_tm), `src/score_groups.py` (NPMI and WordNet scores of the groups).
- Diagnostics: `src/diag_allocation.py`, `src/diag_slide10_pairs.py`. Freezing a plan: `src/freeze.py`.
- E5, the registered test on the Wikipedia test pairs, adds its own files (`src/e5_*.py`, `tests/test_e5_*.py`), described by its registration.
- The run logs hold rows made in a working folder before the code was moved here; each log says so in a dated publication note, and `registrations/INDEX.md` says which plans were fixed there.

## How to run the tests

The code runs on the CPU. The virtualenv `.venv` of this folder, which is not distributed, was made with uv 0.13.7 on CPython 3.14 (`.venv/bin/python` reports 3.14.10) and holds the packages pinned in `env/requirements.txt` (among them numpy 2.5.4, numba 0.68.1, scipy 1.18.1, scikit-learn 1.9.1 and nltk 3.10.4). The code imports from the sibling folder `../hierarchical_tm`, so the two folders have to stay next to each other under `studies/`. The tests are plain scripts (no pytest), run from this folder with the commands of their docstrings:

```
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_orgroup_tm.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_smoke_pcp.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_smoke_real.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_e3_units.py
CUDA_VISIBLE_DEVICES= .venv/bin/python -I tests/test_e3_pipeline.py
```

Each prints one line per test and a last line such as `ALL 23 TESTS PASSED`, or stops with exit status 1 at the first failure.

- `tests/test_orgroup_tm.py` tests the engine and the scores and needs no data.
- `tests/test_smoke_pcp.py` runs the planted-test runner on worlds of seed 990 that `../hierarchical_tm/src/pcp_world.py` generates on the spot.
- `tests/test_smoke_real.py` runs the slide-10 and Wikipedia runners on tiny settings and needs the data of hierarchical_tm, which are not distributed.
- `tests/test_e3_units.py` (the guards of E3) and `tests/test_e3_pipeline.py` (the dry job of E3 in a scratch copy, breaking one thing at a time; about three minutes) skip the parts that need the slide-10 data when they are missing.
- The tests of E5 (`tests/test_e5_*.py`) are run the same way; their docstrings give the commands.

The smoke tests write their run folders under `logs/tests/smoke_runs/`, which is not distributed. The last recorded outputs are `logs/e3/test_orgroup_tm.out` (`ALL 23 TESTS PASSED`) and `logs/e3/test_e3_units.out` (`ALL 11 TESTS PASSED`), written by the registered job of E3 just before its freeze.

What a clone allows: the registrations, the run logs, the reports and the code can be read, and the unit and planted smoke tests can be run. The worlds of E1 are generated from their seeds by the distributed `../hierarchical_tm/src/pcp_world.py`, so E1 can be run again with the flags of its registration and then recomputed with `verification/recompute_e1.py`, which looks for the E1 run folders under `runs/`; this has not been done with the engine now in `src/`. E6 generates its data itself (`src/run_seq.py`) and can be run again from a clone as well. The slide-10 sentences and the Wikipedia pairs belong to hierarchical_tm and are not distributed (the slide-10 data are built from sentences of the unpublished proposal), and neither are the run folders in `runs/` and `runs_dry/`, so E2 to E5 cannot be run again from a clone.
