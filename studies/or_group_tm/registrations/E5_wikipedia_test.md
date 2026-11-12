# E5: registered test of the OR-group rule on the Wikipedia paragraph pairs of the hierarchical_tm study (test split)

Fixed when `registrations/E5_wikipedia_test.frozen.json` exists; `registrations/E5_job.sh registered` writes it with `src/freeze.py` and then runs every job below without a pause. Before the freeze no model of this study has been scored on a test pair: `src/e5_report.py` reads every run configuration under `runs/`, `runs_dry/` and `logs/tests/` and fails if one shows the test split, or holds a `scores_test.npz`, outside the registered runs; the runner (`src/run_wiki.py`) and the report tool (`src/wiki_report.py`) refuse the test pairs without the record of this text, before any pair is read (tests: `tests/test_e5_units.py`, `tests/test_e5_checks.py`).

## Question

On the Wikipedia paragraph pairs of the hierarchical_tm study (its files, its 1,024-word vocabulary, its held-back test pairs), where the stock model of that study ended below word overlap in its one final test: is the OR-group rule of this study (`src/orgroup_tm.py`, arm TB of E4; not the package that study tested), in the score that E4 named (the vote sum), above the stock model on the same test pairs (R1)? And, with no rule, where does it stand against word overlap, tf-idf cosine, PPMI cosine and the logistic regression, in the vote sum and in the ratio reading?

The study's verdict on its final test (its model "below" word overlap on `sa` and `np`, and "below" the baseline that was best on dev on all three data versions) is a closed registered result about its model, and E5 does not change it.

## Headline, fixed now

The report prints these sentences, one per data version, with the numbers filled in (`src/e5_report.py`, function `sentences`); they are the only wording used for the result. For sa:

"Same-article pairs (sa), 10,000 held-back test pairs: the mean AUC of 5 new seeds of the OR-group rule in the vote sum (the score E4 named) is x, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is y; the difference is d [lo, hi] (paired article bootstrap, 95%) and [lo2, hi2] with the seeds resampled, so R1-sa (above the stock model) is met | not met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is x_r; the ratio reading minus word overlap (AUC j) is d_r [lo, hi] [lo2, hi2], and the vote sum minus word overlap is d_u [lo, hi] [lo2, hi2]."

The same for "Next-paragraph pairs (np)", and for "Next-paragraph pairs with every shared word removed (ns_all), 10,000 held-back test pairs" with the ratio reading minus the stock model in place of the two differences to word overlap. x is the mean test AUC of the 5 registered seeds, y that of the 5 stock seeds, d their difference; [lo, hi] is the study's article bootstrap and [lo2, hi2] the bootstrap that also resamples the seeds. They are always printed with this sentence: "The ratio reading was defined on the DEV pairs at 16:51:28Z, after the tied rows of seed 1 were below the bar G2 of E4 in the vote sum that E4 named (sa 0.8255 against 0.830, np 0.8567 against 0.875). G2 is missed in the vote sum (means of seeds 1 to 3: 0.8272 and 0.8567) and passed only in the ratio reading (0.8470 and 0.8787); by E4, the comparison with word overlap carries no rule."

and with one sentence of readings per data version and reading (no rule): the differences to word overlap, tf-idf cosine, PPMI cosine, the baseline that was best on dev in the study and the stock model, each with both intervals and the words "interval above 0", "interval below 0" or "interval includes 0", then the intervals with the articles of paragraphs A and B resampled, and the sentences of the reading without the pairs that touch a duplicate.

Words: "above the stock model in its registered setting on the test pairs of <data>" only for a met R1 of that data version, and with "as predicted from DEV". A sentence about "the Wikipedia pairs" in general is used only if R1 is met on sa, np and ns_all, and then as a list of the three results. Every comparison with word overlap, tf-idf cosine, PPMI cosine or the logistic regression is a reading: "the vote sum minus word overlap: d [lo, hi]" or "the ratio reading minus word overlap: d_r [lo, hi]", the ratio reading always named and always with the vote sum's difference in the same sentence, never "above word overlap" for the model or the rule, never a verdict word. The JSON of `src/wiki_report.py` keeps the study's field `verdict` ("beats X", "below X", "not distinguishable", copied unchanged from its final-test report, for the study's interval alone); in E5 that field decides nothing and is never quoted. The rules are the field `met` of `rules` in `logs/e5_report.json`.

## Rules (three, each standing alone)

Every rule is read in the score that E4 named, the AUC of the vote sum (unit reading). A rule is met only when the whole 95% interval of the difference is above 0 in both intervals of `src/wiki_report.py`: the study's paired article bootstrap, unchanged (1,000 resamples of the articles of paragraph A, numpy seed 0; the mean AUC of the five seeds of each side in each resample), and the bootstrap that also resamples the seeds (10,000 resamples of the articles of paragraph A, numpy seed 0, in each of which the seeds of both sides are drawn with replacement). The second is the interval of a claim about a learning rule whose seeds are random; on DEV it was 0.1% to 8.5% wider than the first for the comparisons with the baselines and 13.1% to 79.4% wider for those with the stock model, whose seeds vary more (computed by a review from the DEV readings of `logs/e5_dev_readings/`, which hold both intervals).

- **R1-sa, R1-np, R1-ns_all (above the stock model).** Mean AUC of the 5 registered seeds in the vote sum minus the mean AUC of the 5 stock seeds of the study on the same test pairs (their saved test scores, which are also vote sums). Predicted from DEV: met on all three, each with probability 1.0000 (to four decimals) for the study's interval; the second interval leaves the DEV differences 7 to 14 half-widths above 0. E4 set no bar for ns_all and called it "reported"; R1-ns_all is added by E5, with its DEV difference known (+0.081 [+0.071, +0.090] in the vote sum).
- The stock model is the setting that the study registered for its final test: on sa and np its headline setting (50 rules, 64 options, 2 OR-groups per paragraph, T 3,200, `s` 100, 100 passes; `s` and the number of passes chosen by rules on the DEV pairs of seed 1 of sa), on ns_all the `s` 100 arm of its synonym test. Its second setting (2,000 rules, 8 options, `s` 40, 50 passes) had higher DEV AUC (sa 0.763 to 0.771, np 0.778 to 0.796), failed the stability part of the study's Check 3 and was never scored on the test pairs. "The stock model" in E5 means the registered setting. On this side, arm TB uses tied groups, which were not among the single changes that E4 planned; it is outside E4's plan.
- **No rule against word overlap.** E4 fixed G2 as the go-bar for such a rule, in the vote sum. G2 was missed there (means of seeds 1 to 3: sa 0.8272 against 0.830, np 0.8567 against 0.875; note of 18:12:43Z in `logs/RUNS_wiki.md`), and E4's consequence is "a registered test of 'above the stock model' only, with word overlap, PPMI cosine and logistic regression shown beside it". G2 is passed only in the ratio reading (sa 0.8470, np 0.8787). That reading was defined at 16:51:28Z, after the tied rows of seed 1 were below the G2 bars in the vote sum (sa 0.8255, np 0.8567) and after a DEV report had put the tied sa model's vote sum at +0.0064 [+0.0004, +0.0121] above word overlap. A bar is not met in a reading defined after it was seen to be missed. The comparisons with word overlap are registered readings with a prediction (next section), not rules, and no sentence of E5 says that the model or the rule is above word overlap.

Multiplicity: three rules, no correction; each stands alone and none stands for another. With the nominal interval each would be met by chance with probability at most 0.025, at least one of the three at most 0.075; if the spread from the article of paragraph B is as large as a review found on DEV, about 0.0433 to 0.0584 each and about 0.18 for three. The readings (about 30 paired comparisons in two readings, the subsets and the group scores) are not corrected and carry no claim. The DEV selection (22 design rows; four readings, one defined after the rows) does not change these error rates; it makes the DEV effect sizes optimistic.

## Registered readings with a prediction fixed now (no rule)

Predicted outcomes on the test pairs, computed by a review before this text was written (a normal approximation from the DEV readings; each range covers three scenarios: only the article of paragraph A resampled; the spread of both articles found on DEV; that spread plus a DEV-to-test shift of 0.004). t is the smallest test difference whose interval of the study's kind is above 0. The predictions are for the study's interval; on DEV the second interval was 0.1% to 8.5% wider for the comparisons with the baselines.

| data | reading | minus | DEV difference | t | predicted |
|---|---|---|---|---|---|
| sa | ratio reading | the stock model | +0.1103 | 0.0090 | interval above 0, probability 1.0000 |
| sa | ratio reading | word overlap | +0.0280 | 0.0071 | interval above 0, probability 0.9989 to 1.0000 |
| sa | ratio reading | tf-idf cosine | +0.0230 | 0.0069 | interval above 0, probability 0.9917 to 0.9999 |
| sa | ratio reading | PPMI cosine | -0.0068 | 0.0084 | above 0 with 0.0019 to 0.0240, below 0 with 0.3804 to 0.4176, otherwise the interval includes 0 |
| sa | ratio reading | the logistic regression (best on dev) | -0.0243 | 0.0085 | interval below 0, probability 0.9797 to 0.9986 |
| sa | vote sum | the stock model (R1) | +0.0904 | 0.0092 | interval above 0, probability 1.0000 |
| sa | vote sum | word overlap | +0.0081 | 0.0074 | interval above 0, probability 0.5432 to 0.5653 |
| sa | vote sum | tf-idf cosine | +0.0032 | 0.0073 | above 0 with 0.1830 to 0.2766, below 0 with 0.0108 to 0.0660, otherwise the interval includes 0 |
| sa | vote sum | PPMI cosine | -0.0266 | 0.0085 | interval below 0, probability 0.9905 to 0.9997 |
| sa | vote sum | the logistic regression (best on dev) | -0.0441 | 0.0093 | interval below 0, probability 1.0000 |
| np | ratio reading | the stock model | +0.1297 | 0.0096 | interval above 0, probability 1.0000 |
| np | ratio reading | word overlap | +0.0145 | 0.0061 | interval above 0, probability 0.9189 to 0.9846 |
| np | ratio reading | tf-idf cosine | +0.0100 | 0.0060 | interval above 0, probability 0.7511 to 0.8556 |
| np | ratio reading | PPMI cosine | -0.0104 | 0.0071 | interval below 0, probability 0.6992 to 0.7769 |
| np | ratio reading | the logistic regression (best on dev) | -0.0287 | 0.0074 | interval below 0, probability 0.9993 to 1.0000 |
| np | vote sum | the stock model (R1) | +0.1078 | 0.0093 | interval above 0, probability 1.0000 |
| np | vote sum | word overlap | -0.0075 | 0.0066 | interval below 0, probability 0.5579 to 0.5865 |
| np | vote sum | tf-idf cosine | -0.0120 | 0.0063 | interval below 0, probability 0.8260 to 0.9245 |
| np | vote sum | PPMI cosine | -0.0324 | 0.0074 | interval below 0, probability 0.9999 to 1.0000 |
| np | vote sum | the logistic regression (best on dev) | -0.0506 | 0.0079 | interval below 0, probability 1.0000 |
| ns_all | ratio reading | the stock model | +0.1114 | 0.0132 | interval above 0, probability 1.0000 |
| ns_all | ratio reading | PPMI cosine | -0.0748 | 0.0112 | interval below 0, probability 1.0000 |
| ns_all | vote sum | the stock model (R1) | +0.0809 | 0.0135 | interval above 0, probability 1.0000 |
| ns_all | vote sum | PPMI cosine | -0.1053 | 0.0111 | interval below 0, probability 1.0000 |

- **The ratio reading against word overlap (sa, np).** The ratio reading is a Jaccard index over the clauses' word groups: the clauses whose group is true on both paragraphs, divided by those whose group is true on at least one. It was defined on the DEV pairs after the tied rows were seen to be below G2 in the vote sum. Printed: the mean AUC of the 5 seeds in the ratio reading minus word overlap, with both intervals; written "the ratio reading minus word overlap: d [lo, hi]", never "above word overlap", and always with the vote sum's difference in the same sentence.
- **The vote sum against word overlap (sa, np)**, the score that E4 named: predicted as in the table (on sa above 0 about as often as not; on np below 0 slightly more often than not).
- **tf-idf cosine, PPMI cosine and the logistic regression**, in both readings, as in the table; on DEV the rule is below PPMI cosine on all three data versions and below the logistic regression on sa and np, in both readings.
- **R1 in the ratio reading** (the same three differences): predicted above 0, probability 1.0000.

What E5 adds is the size of each test difference and the readings whose outcome is open: the ratio reading against word overlap on np, the vote sum against word overlap, the ratio reading against PPMI cosine on sa and against tf-idf cosine on np. A met R1 is the expected outcome: it shows that the DEV distance holds on held-back pairs, and it is not news.

## Further readings with no rule

- **The article of paragraph B.** The intervals of the rules resample the article of paragraph A only, as the study's final test did (one of its known limitations). Paragraph B of every negative pair comes from another article. A third interval, `ci95_two_way` (1,000 resamples, numpy seed 0), resamples all articles of the split and weights each pair by the copies of its article, or by the product of the copies of its two articles, with the seeds fixed; it is printed for every comparison and decides nothing (product weights may overstate the spread). On DEV a review found that weights on the articles of A and B give a 14% (np) and 25% (sa) larger spread of the ratio reading minus word overlap than weights on A's article alone.
- **Without the pairs that touch a duplicate** (`registrations/E5_duplicate_articles.json`, made by `src/e5_duplicates.py` from the word ids of the paragraphs alone): the vote sum's comparisons, with both intervals, recomputed without the test pairs whose paragraph A or B comes from one of the 36 test articles that are whole copies of a training article, and again without those that touch one of the 242 test articles with a paragraph equal to a training paragraph. The same for every method; the article lists are fixed now, with no score.
- Per seed: AUC of the unit, ratio, weight and log-odds readings, partner-shuffle AUC, members, seconds per pass, wall time; the tuned accuracy (its cut-off is chosen on the scored pairs themselves, as in the study's table, so it supports no claim).
- The baselines and the stock AUCs recomputed by the report, next to the numbers of the study's final-test report (equal to its printed digits is expected).
- Word groups of the registered models (`src/score_groups.py`, the study's measures): NPMI over the 2,847 reference articles that are in no split, and the WordNet score, against frequency-matched random groups and against the groups of the five stock models, pooled over the five seeds of a data version. It reads no test pair. The groups of this rule hold about 3 words (dry run: 2.97, 2.85, 4.61), those of the stock model about 63 (63.49, 63.26, 61.82); the comparison with the stock groups is not matched on size and supports no statement that one set of groups is better. Both scores measure topical coherence and WordNet relatedness, not synonym sets.
- `groups.txt` of each run lists the 40 clauses with the largest difference of firing rates between the classes on the first 4,000 test pairs, in words: a display chosen with the test labels, never a reading.

## What the known numbers predict, and what five seeds can show

DEV, arm TB, seeds 1 to 3 (each 20,000 DEV pairs; `logs/e5_dev_readings/`, made with `src/wiki_report.py` of draft 1, whose study interval is unchanged, and the five stock seeds' DEV scores):

| | sa | np | ns_all |
|---|---|---|---|
| vote sum, mean AUC | 0.8272 | 0.8567 | 0.7031 |
| ratio reading, mean AUC | 0.8470 | 0.8787 | 0.7336 |
| SD over the 3 seeds, vote sum | 0.0015 | 0.0007 | 0.0020 |
| SD over the 3 seeds, ratio | 0.0013 | 0.0007 | 0.0013 |
| stock model, mean DEV AUC of seeds 1 to 5 | 0.737 | 0.749 | 0.622 |
| vote sum minus stock (R1) | +0.090 [+0.084, +0.097] | +0.108 [+0.101, +0.114] | +0.081 [+0.071, +0.090] |
| ratio minus stock | +0.110 [+0.104, +0.117] | +0.130 [+0.123, +0.136] | +0.111 [+0.101, +0.120] |
| vote sum minus word overlap | +0.008 [+0.003, +0.014] | -0.007 [-0.012, -0.003] | - |
| ratio minus word overlap | +0.028 [+0.023, +0.033] | +0.014 [+0.010, +0.019] | - |
| vote sum minus tf-idf cosine | +0.003 [-0.002, +0.008] | -0.012 [-0.016, -0.007] | - |
| ratio minus tf-idf cosine | +0.023 [+0.018, +0.028] | +0.010 [+0.006, +0.014] | - |
| ratio minus PPMI cosine | -0.007 [-0.013, -0.001] | -0.010 [-0.015, -0.005] | -0.075 [-0.083, -0.067] |
| ratio minus the baseline best on dev | -0.024 [-0.030, -0.018] | -0.029 [-0.034, -0.023] | -0.075 [-0.083, -0.067] |

Pooled over the three data versions (6 degrees of freedom) the spread between seeds is 0.0015 in the vote sum and 0.0011 in the ratio reading; 95% upper bounds 0.0033 and 0.0025. Five seeds are enough: at the upper bound the seeds widen the interval of the closest comparison (np, ratio reading minus word overlap) by about 6% (a review's computation), and the rules are means, not counts of seeds. No further seeds are run.

The test split has 10,000 pairs per data version, half the DEV pairs, from 2,763 / 2,767 / 2,743 articles of paragraph A. From DEV to test the study's own numbers moved little: the largest move of a baseline or of a stock mean is 0.0115, of a single stock seed 0.0125 (ns_all seed 4). A miss of R1 needs a loss of about 0.1 AUC from DEV to test. In the study's final test the half-width of the paired interval against word overlap was 0.011 (sa) and 0.010 (np).

## Data and scorer (the hierarchical_tm study's, read only)

- Pairs: `data/{sa,np,ns_all}/{train,test}.npz`, 40,000 training and 10,000 test pairs per data version, half of them positive. `sa`: two paragraphs of the same article (positive) or of two articles (negative). `np`: a paragraph and the next one (positive), or one paragraph from each of two articles, drawn so that first and last paragraphs are as frequent in both classes (negative). `ns_all`: a separately sampled set of next-paragraph pairs from which every word that the two paragraphs share is removed (symmetric removal, lengths matched between the classes; `src/build_ns_all.py` of the study), 2,743 test articles of paragraph A. A paragraph is the set of its words among the 1,024 most frequent ones (`vocab_F1024.json`). Training reads `train.npz` only.
- Provenance: built by the study from the train part of WikiText-103 (raw), split by whole article (seed 20261001; train 19,930 articles, dev 2,848, final test 2,847, NPMI reference 2,847), on 1 Oct 2026 (`data/ns_all/` separately). The word lists rank words by the number of training paragraphs that hold them: over the training paragraphs the order of `vocab_F3072.json` has 0 violations, and counting the paragraphs of every split would change 8 of the 1,024 words (the word ids of the paragraphs, counted for this text). `vocab_F1024.json` is the first 1,024 words of `vocab_F3072.json`; its sha256 equals the study's `frozen_settings_hashes.json`, and `vocab_F3072.json` is in no record of the study.
- Duplicates (`src/e5_duplicates.py`, word sets of 3,072 words, an upper bound for paragraphs equal word for word): 1,545 of the 71,639 test paragraphs (2.16%) equal a paragraph of a training article, in 242 of the 2,847 test articles; 36 test articles are whole copies of a training article. DEV: 1,320 of 69,113 (1.91%), 232 articles, 32 whole copies. Test against DEV: 142 test paragraphs (0.20%) in 56 test articles, 2 whole copies. The study's data card gives 2.1% of the test paragraphs in 235 articles (word for word, an internal audit of 2026-10-03). Word overlap and the shared-word count learn nothing and cannot gain from a duplicate; tf-idf cosine only through its document frequencies; PPMI cosine through vectors fitted on all training paragraphs; the two logistic regressions, the stock model and the OR-group rule learn from training pairs, among them pairs of articles that recur in the test split. So the duplicates can favour the trained methods over word overlap; in R1 both sides are trained, though not with the same capacity to memorise. DEV had the same rates when the arm and the reading were chosen. The reading without the pairs that touch a duplicate (above) measures it.
- Reference copies: the sha256 of the three `train.npz` and `test.npz` files, of `paragraphs_F3072.npz` and of `splits.json` equal the values that the study's final-test report recorded (`inputs_sha256`), and the sha256 of the 15 stock test score files equal its `scores_sha256` (checked by `tests/test_e5_units.py`, which hashes the test files and never loads them). Each stock seed is the same model on both sides: the DEV score folders that give the groups scored the training run that the final-test report names.
- Baselines and bootstrap: `src/wiki_report.py` imports the study's `data_checks.py` (Jaccard, tf-idf cosine, shared-word count, PPMI-SVD cosine, the two logistic regressions), `htm_driver.auc` and `pairdata.py`, and copies the article bootstrap and the verdict rule of its `final_test_report.py`; it compares y, art_a and par_a of every stock score file with the pairs, as that script did. `data_checks.py` (7523133e) and `htm_driver.py` (0073b2a4) differ from the copies in the final test's run folder (72b0d717, dc31407e): a docstring line and the 2026-10-25 fix of `length_feats`, which E5 does not use, and a docstring line and a comment; `pairdata.py` is equal (82d98e94). With the pinned versions the dry report reproduces the study's DEV numbers of all six baselines and of the five stock seeds to their printed digits.

## Uses of the test pairs so far

- 2026-10-02, registered: the study scored them in 20 test re-score runs (`*_testscore`: the 15 stock seeds and 5 shuffled-label ns_all seeds) and built `20261002T193513Z_final_test_main_table`. Its two `final_test_report_codepath_test_exploratory` runs read DEV data only.
- 2026-10-03, post hoc, not listed by the study: an internal audit (not distributed) read the test texts to count duplicated paragraphs, recomputed the 20 test scores from the saved per-pair files, and records two statements on these pairs: that removing the duplicated articles "lowers the TM's test AUC by at most 0.003" (a statement about the stock model on a subset of the test pairs; how it was obtained is not recorded) and that a two-way resampling "might make them 18-37% wider" (an unverified estimate).
- 2026-10-25, post hoc: step (iv) of the study's later reproduction check re-read its final-test report, the 20 saved test score files and their score lines (11:52:31 to 11:55:11 UTC): 529 numbers recomputed equal, cross-checks with other code, and descriptive intervals with bootstrap seed 12345. The study counts it inside its one use and left that decision to whoever froze that check.
- 2026-11-12, before this text: a read-only review read the test arrays as data (word ids, articles, labels, for counts only) and hashed the stock test score files; `tests/test_e5_units.py` hashes the test files; this text's duplicate and vocabulary counts read the word ids of the paragraphs of every article (no pair, no label, no model, no score).
- Counts: by the study's count E5 is the second registered use of the test split and the first by this study; if the re-read of 2026-10-25 is counted, the third. After the final test, statistics of the stock model on these pairs were computed on 2026-10-03 and 2026-10-25. The study's records say that any further statistic from its saved test scores is a new use; E5 computes paired differences against those scores, inside its own use. E5 is the first time new models are scored on these pairs.
- This study: no run configuration under its `runs/`, `runs_dry/` or `logs/tests/`, nor under the `runs/` and `runs_dry/` of the working folder where E0 to E4 were run, shows the test split (44 Wikipedia run configurations read when this text was written), and none uses a registered seed (11 to 15).
- Known before this text was written, from the study's final-test report (pinned): the test AUCs of the stock model (mean of 5 seeds: sa 0.735, np 0.750, ns_all 0.614; per seed sa 0.7427, 0.7366, 0.7363, 0.7315, 0.7270; np 0.7552, 0.7303, 0.7564, 0.7558, 0.7543; ns_all 0.6108, 0.6134, 0.6090, 0.6165, 0.6183) and of the six baselines (sa: jaccard 0.8169, tfidf_cosine 0.8212, shared_count 0.7898, ppmi_svd_cosine 0.8517, lr_A_AND_B 0.8184, lr_A_B_AandB 0.8694; np: jaccard 0.8565, tfidf_cosine 0.8620, shared_count 0.8253, ppmi_svd_cosine 0.8855, lr_A_AND_B 0.8564, lr_A_B_AandB 0.9031; ns_all: jaccard 0.5000, tfidf_cosine 0.5000, shared_count 0.5000, ppmi_svd_cosine 0.8110, lr_A_AND_B 0.5000, lr_A_B_AandB 0.4944), and its paired intervals of the stock model minus word overlap (sa -0.082 [-0.093, -0.071], np -0.106 [-0.116, -0.096]). No test number of the OR-group rule exists. These known numbers and the DEV rows predict every rule and reading (tables above).

## Model and setting (arm TB of E4), fixed now

Every flag written out, as in `registrations/E5_job.sh` (D = sa, np, ns_all; S = 11, 12, 13, 14, 15):

    src/run_wiki.py --data D --seed S --F 1024 --clauses 2000 --K 1 --T 40 --mode side --init-state -3 \
    --seed-state 0 --r 2 --p-rec 1 --p-rep 1 --q-seed 1 --max-seed 1 --reject weakest --reinforce share \
    --p-decay 0 --tied --wmax 64 --neg-frac 0 --smin -32 --smax 31 --passes 10 --eval-every 10 --n-train 0 \
    --n-dev 0 --top 40 --split test --registered registrations/E5_wikipedia_test.frozen.json --name e5_D_sS \
    --out-root runs

2,000 clauses of positive polarity; one OR-group per paragraph (K 1), tied: one set of groups serves both paragraphs, so a clause fires when both paragraphs hold a word of its group; T 40; a group learns from a paragraph when the clause's other paragraph matched (mode `side`); one present member per group reinforced, chosen at random (`share`); on a false firing the weakest firing group is hit (r 2); initial state -3; one seed event per example; 10 passes; clause weights up to 64 kept as extra readings. The model after pass 10 is applied to the test pairs, and no earlier pass is: for the logged numbers (`src/run_wiki.py`, the evaluation of the last pass and the final numbers), for the partner-shuffle control, and for the display of `groups.txt`. The engine settings that the runner cannot set are fixed by the engine file, which is pinned.

Readings: unit = the number of positive clauses that fire (the vote sum, the model's output); ratio = that number divided by the number of positive clauses whose group is true on at least one of the two paragraphs (a Jaccard index over the learned groups); weight and log-odds = the two weighted readings. Registered seeds 11 to 15 for each data version, 15 runs.

## Code since the design rows

- The DEV rows of seed 1 (run with engine b94db364) are the design rows of E4 byte for byte (state, polarity and weight arrays, stored scores, every final number; note of 18:12:43Z), although the design rows were made with engine files 53cc7b4b (sa, np) and 32062302 (ns_all). The dry job of this folder ran the registered command on the DEV pairs (seed 1): its three models are the DEV seed-1 models byte for byte, and its AUCs are their AUCs (sa vote sum 0.8255, ratio 0.8456, np vote sum 0.8567, ratio 0.8787, ns_all vote sum 0.7023, ratio 0.7323).
- `src/run_wiki.py` checks the record before any pair is read (the plan, every file under `src/` and `tests/`, every pinned file, the registered arguments, a second run of a seed), takes the engine hash before the engine is imported and again at the end, and writes the start time, the interpreter, the Python version and the version of every package of `env/requirements.txt`, the word-list hash and paths relative to the study folder. A design run on the DEV pairs needs no record, as before.
- `src/wiki_report.py`: the same check before the test pairs are read; with `--split test` only the registered runs and the study's five test score folders are accepted; the ratio reading; the paired comparison with tf-idf cosine; the bootstrap that also resamples the seeds; the interval with the articles of paragraphs A and B; the subsets; y, art_a and par_a compared. The baselines, the study's bootstrap and its verdict words are unchanged.
- New: `src/e5_common.py` (what E5 fixes), `src/e5_report.py` (checks first; the numbers step by step, each written as soon as it is done; every number kept when a check fails), `src/e5_duplicates.py`, `registrations/E5_job.sh`, `tests/test_e5_units.py` (ALL 15 TESTS PASSED) and `tests/test_e5_checks.py` (CHECK TEST PASSED (53 cases); it copies the dry job's run folders into a scratch copy of the study folder, makes the report again with every opened file recorded, and breaks one thing at a time). The older tests pass: `tests/test_orgroup_tm.py` (ALL 23 TESTS PASSED), `tests/test_e3_units.py` (ALL 11 TESTS PASSED), `tests/test_smoke_real.py` and `tests/test_smoke_pcp.py`.
- The branches that only the registered run takes (the test branch of `src/wiki_report.py`, the record handed to it, the registered names, the final-test report in the report, five seeds per data version) ran end to end before this text in a scratch copy of the study whose three test files were DEV subsets under test names, with the stock DEV scores of the same pairs as stock test scores and a placeholder final-test report (no test pair read); the unit tests cover the same branches with made-up data.
- After the E5 code was added, the record of E3 (which holds every file under `src/` and `tests/`) no longer matches this folder; E3 had ended (its report at 2026-11-12T17:51:57Z) before the first change. The two versions that E3 pins of the changed files are kept as `verification/e3_sources/run_wiki.py` and `verification/e3_sources/wiki_report.py` (sha256 equal to E3's record; pinned here).

## What I knew at the time of writing (TRAIN and DEV, and the study's published test numbers)

From `logs/RUNS_wiki.md` (its sha256 when this text was written is in "Hashes"; the registered runs append to it, and the report checks its text up to the freeze), `logs/e5_dev_readings/`, `logs/design/` and `logs/e5_dry_report.json`:
- Order of events: the tied sa model of seed 1 ended at 16:40:52Z (vote sum 0.8255, below the sa bar of G2); a DEV report of that model against the baselines followed (`logs/design/wiki_report_dev_sa_tied_s1_design.json`: minus word overlap +0.0064 [+0.0004, +0.0121], minus PPMI cosine -0.0284 [-0.0347, -0.0223], minus the logistic regression -0.0459 [-0.0526, -0.0393], minus the stock model +0.0887 [+0.0820, +0.0952]; and the word groups of the same run, `logs/design/score_groups_sa_tied_s1_vs_stock.json`); TB on np ended at 16:47:42Z (vote sum 0.8567, below the np bar); the ratio reading was defined at 16:51:28Z and rescored both (sa 0.8456, np 0.8787); the ns_all row of TB ended after that note, at 16:52:16Z.
- E4's arms: untied arms on sa, vote sum 0.7346 to 0.7861; half the clauses voting against: 0.5047, dropped. Not run or not finished (note of 16:47:07Z): K 2 untied, mode near, reinforce all + reject random, 30 passes (all untied); T 20 and 500 clauses were never started. Run outside E4's list of single changes: tied groups, clause weights (wmax 64), `q_seed` 0.25 (two rows), half the clauses voting against, initial state -5 and every tied variant. TB was run on np and ns_all before it was chosen, so being "the only arm run on all three data versions" followed from that choice. TB variants on sa (vote sum / ratio): critical 0.8243 / 0.8445; init1 0.8220 / 0.8422; init5 0.8257 / 0.8458; T100 0.8227 / 0.8515; K2 0.8006 / 0.8368; r4 0.8224 / 0.8404; c4000 0.8295 / 0.8517; the 8,000-clause row `e4_sa_B_tied_w64_c8000_s1` shows the vote sum 0.8304 only, with neither a ratio reading nor the tied flag (named tied, its folder is not in this repository).
- DEV rows of TB, seeds 1 to 3 (vote sum / ratio): sa 0.8255 / 0.8456, 0.8278 / 0.8474, 0.8283 / 0.8481; np 0.8567 / 0.8787, 0.8561 / 0.8779, 0.8574 / 0.8793; ns_all 0.7023 / 0.7323, 0.7053 / 0.7349, 0.7016 / 0.7336. Partner-shuffle AUC 0.4949 to 0.5064; 10,448 to 16,830 members; 17.2 to 22.4 s per pass on one thread.
- The bars of E4 (note of 18:12:43Z): G1 (sa at least 0.759, np at least 0.764) met in both readings; G2 (sa at least 0.830, np at least 0.875) missed in the vote sum, passed in the ratio reading only (np by 0.0037), a reading defined after the seed-1 rows were below the bar.
- The DEV comparisons in the tables above, among them tf-idf cosine (vote sum sa +0.003 [-0.002, +0.008], np -0.012 [-0.016, -0.007]); on ns_all the logistic regression, a word-matching classifier, is near chance (DEV AUC 0.506).
- Dry job (DEV, seed 1), as its report printed it:
  - Same-article pairs (sa), 20,000 DEV pairs (dry run): the mean AUC of 1 dry seed of the OR-group rule in the vote sum (the score E4 named) is 0.825, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is 0.737; the difference is +0.089 [+0.082, +0.095] (paired article bootstrap, 95%) and [+0.080, +0.097] with the seeds resampled, so R1-sa (above the stock model) is met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is 0.846; the ratio reading minus word overlap (AUC 0.819) is +0.027 [+0.021, +0.032] [+0.021, +0.032], and the vote sum minus word overlap is +0.006 [+0.000, +0.012] [+0.001, +0.012].
  - Next-paragraph pairs (np), 20,000 DEV pairs (dry run): the mean AUC of 1 dry seed of the OR-group rule in the vote sum (the score E4 named) is 0.857, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is 0.749; the difference is +0.108 [+0.101, +0.114] (paired article bootstrap, 95%) and [+0.097, +0.120] with the seeds resampled, so R1-np (above the stock model) is met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is 0.879; the ratio reading minus word overlap (AUC 0.864) is +0.015 [+0.010, +0.019] [+0.010, +0.019], and the vote sum minus word overlap is -0.008 [-0.012, -0.002] [-0.013, -0.002].
  - Next-paragraph pairs with every shared word removed (ns_all), 20,000 DEV pairs (dry run): the mean AUC of 1 dry seed of the OR-group rule in the vote sum (the score E4 named) is 0.702, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is 0.622; the difference is +0.080 [+0.070, +0.090] (paired article bootstrap, 95%) and [+0.069, +0.091] with the seeds resampled, so R1-ns_all (above the stock model) is met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is 0.732; the ratio reading minus the stock model is +0.110 [+0.100, +0.120] [+0.099, +0.121].
- The study's test numbers listed under "Uses of the test pairs so far"; the study's DEV numbers of the stock seeds: sa 0.745, 0.739, 0.733, 0.732, 0.733; np 0.752, 0.728, 0.751, 0.755, 0.759; ns_all 0.622, 0.622, 0.611, 0.629, 0.626; the audit's two statements on the test pairs (same section); the duplicate counts ("Data"); the predictions ("Registered readings").

## Limits, stated before the run

- Selection on DEV: arm TB was picked after 22 design rows on seed 1 (most on sa), with tied groups, which E4's list of single changes did not contain; it was run on np and ns_all before it was chosen. The ratio reading was defined after the tied rows were seen to miss G2 in the vote sum. The stock model had one registered setting, chosen by a rule on the DEV pairs of seed 1, and no reading was chosen for it. Both make the DEV numbers of this rule optimistic and the comparison unequal in design effort; neither reads a test pair.
- In the score that E4 named (the vote sum), G2 was missed; by E4 there is no rule against word overlap. The ratio reading, in which G2 passes, was defined after the miss was visible on seed 1.
- The intervals of the rules resample the article of paragraph A only, as the study's final test did (its known limitation). On the DEV pairs, weights on the articles of A and B gave a 14% (np) and 25% (sa) larger spread for the ratio reading minus word overlap (a review's DEV check, an upper estimate); if that holds, the one-sided error of each rule is about 0.0433 to 0.0584, not 0.025. The audit of 2026-10-03 estimated 18-37% wider two-way intervals for the study's comparisons, unverified. The two-way interval is printed beside every comparison.
- The seed-inclusive interval resamples five stock seeds that are not fully independent repeats: the study's driver seeds its random stream with the seed plus the pass, so its seeds share streams (its known limitation). The seeds of this rule draw distinct streams for each seed and pass.
- Five seeds of one arm. On DEV the spread between seeds is at most 0.0025 (95% upper bound, ratio reading; 0.0033 in the vote sum). Seeds 11 to 15 were never run.
- Not a comparison at matched size or cost: the stock model is the study's registered setting (50 rules of 64 options, 100 passes, another package, on a GPU); this rule has 2,000 clauses and 10 passes on one CPU thread. "Above the stock model" is about these two models as they were set.
- A small world: 1,024 frequent words, 40,000 training pairs, words as sets. On DEV, PPMI cosine (all three data versions) and the logistic regression (sa, np) are above the rule; nothing here says that the rule is a good text model.
- Duplicate articles across splits ("Data"); they can favour the trained methods over word overlap. The test articles are not new kinds of text; "held-back" means held-back pairs of articles of the same source.
- The data versions share articles and the same 1,024 words; the three R1 rules are not independent pieces of evidence.

## What a miss means, and the other cases

- R1-d met: "On the test pairs of d, the OR-group rule (arm TB, chosen on DEV) is above the stock model in its registered setting, as predicted from DEV", in the vote sum, with the difference and both intervals.
- R1-d not met: a loss of about 0.1 AUC from DEV, eight times the largest DEV-to-test move of any number of the study (0.0125); it is reported as measured under the registered words, the checks and the per-seed rows are examined for a fault, and nothing is rerun.
- The ratio reading's interval against word overlap above 0 and the vote sum's not above it (the most likely outcome on np): both are written in one sentence; no sentence says that the model or the rule is above word overlap. If the ratio reading's interval is not above 0, the DEV prediction failed, and that is reported as a measured result.
- R1 is never pooled into one verdict over the data versions, and a met rule on one data version says nothing about another.
- A check of `src/e5_report.py` fails after the runs: the use of the test pairs is spent; the report still computes every number it can (when the record passed) and writes it under "registered run void: <the check>" to a file of its own; nothing is rerun. If the report stops before its file is complete, it is run again (it is deterministic), and every attempt keeps its file.
- Whatever E5 shows, this study does not score the test pairs again without a new registration that lists E5's numbers.

## Never written, whatever the outcome

"the hierarchical Tsetlin machine" or "the model of the proposal" as a name for `orgroup_tm.py`; "reverses", "corrects" or "rescues" the study's verdict; "the stock rule failed" beyond the study's own registered words; "better learning rule" or any comparison at matched size; "G2 met" or "the go-bar was met"; "beats word overlap", "above word overlap" or "registered result" for any comparison with word overlap, and "the model", "the Tsetlin machine" or "the OR-group rule" is above word overlap (the ratio reading is a reading, named every time); "above the stock model" without "in its registered setting"; "confirms", "shows" or "demonstrates" for R1 without "as predicted from DEV"; "beats the baselines"; "reaches", "matches" or "on par with" PPMI cosine or the logistic regression; "understands", "meaning", "semantic" or "synonyms" for the clauses or groups; "generalises" without "to held-back pairs of articles of the same source"; anything about real text beyond these pairs; "reliably", "robustly" or "on every data version" unless R1 is met on all three; "first" or "new" before a literature check; anything about the supervisor beyond "the project proposal".

## Jobs, in this order, in one go

    registrations/E5_job.sh registered

It runs the unit tests (stop on a failure), stores the length and sha256 of `logs/RUNS_wiki.md` (`logs/e5/runs_wiki_prefix.json`), fixes this text with `src/freeze.py` and every file that `src/e5_common.py --extras` lists (the files in "Hashes"), checks the record (`src/e5_common.py --check`, stop on a problem), runs seeds 11 to 15 in three parallel streams of one thread, one per data version, and ends with `src/e5_report.py --frozen registrations/E5_wikipedia_test.frozen.json --out logs/e5_report.json` and the sha256 of every output (`logs/e5/SHA256SUMS_<UTC>`). `nice -n 19`, no GPU, a new empty cache folder for compiled code, `TMLM_RUN_DIR` unset; the interpreter is `.venv/bin/python` of this study and the job takes no setting from the environment; before each run and before the report it applies the CPU rule of the machine itself (at most 4.5 cores in use over 10 s, `../decision_layer/scripts/cpu_load.py`, pinned), which changes only when they start. All seeds are run whatever the earlier ones show; no seed is added or replaced. From the freeze to the end of the report no file of this study folder is edited by hand.

The report runs its checks first (record and plan, sources, pinned files; no test scoring outside the registered runs; its own interpreter and packages; the run log up to the freeze; per data version and seed: one complete run folder, the registered setting, data, seed, name, split and folder, its row in the run log, the record named, the engine, data and word list of the record, the interpreter, Python version and packages of the run, a start after the freeze, 40,000 training and 10,000 scored pairs, the pairs scored after the last pass only, partner-shuffle AUC within 0.47 to 0.53, a score file with the labels of the split that gives the logged AUCs). Then the numbers, each step written to its file as soon as it is done: the rules and the vote sum's readings, the ratio reading, the reading without the duplicates, the word groups, the study's printed numbers; a failure in a later step is recorded and leaves the rules in place.

Restart rule (as the job does it): a run whose folder holds the five files that the report requires is skipped; a folder without them is renamed `<folder>_CRASHED` by the job, logged, and the run is repeated; a run that stopped before writing its folder is repeated. Every run output, report output and hash list of every attempt is kept under its own UTC stamp.

## Relation to other registrations

E4 is the design plan; its bars G1 and G2 were computed on DEV before this text (note of 18:12:43Z). E5 follows E4's branch "G1 met, G2 missed"; the ratio reading is its one addition to E4, as a reading, and R1-ns_all is added with its DEV difference known. E3 (slide 10) is a separate registration on other data; no claim joins the two. The study's final test stays the registered result about its own model.

## Hashes (sha256sum, computed by the command that wrote this section at 2026-11-12T21:04:18Z; the freeze record holds the values at the freeze and the report compares every run with it)

Pinned files outside `src/` and `tests/` (paths relative to this study folder; `logs/e5/runs_wiki_prefix.json` is written by the job at the freeze):

    75c6cf8308429eccb4897528cdf70b524f487366ae631518fed707a6f77cbe85  ../hierarchical_tm/data/sa/train.npz
    bc50b044f4fa15baef8889786e19820d51f996df819ed0182ee99a9307a8a7a7  ../hierarchical_tm/data/sa/test.npz
    ae9a5de7366dab43101bdeab40f7905721a40203f2b2ed730e655e4fc092a807  ../hierarchical_tm/data/np/train.npz
    f83bc4b9b8de2cb024442b62e4c44379085ed601408a6f728327d62eefc72518  ../hierarchical_tm/data/np/test.npz
    9577e1e360f913c07481cd45887af786198a339dc11f75e8ca84b2f3250b33b1  ../hierarchical_tm/data/ns_all/train.npz
    39dc9e36244dbae00c7e6cdf7d72cc5dfaebdadd6e7d269994eb4bcf82bec39d  ../hierarchical_tm/data/ns_all/test.npz
    250ba4454e8ff1ed81a8764da132c7fb98dc7020c4b03606f78ec16ff82076ae  ../hierarchical_tm/data/vocab_F1024.json
    543806996ac67a4d13fa7109694eeadb3953efec5fa88843545426e853de18c3  ../hierarchical_tm/data/vocab_F3072.json
    6bdb5d91bac3591c6970661ca38edd30efd555c6880a08d1e9d2cc10667839c5  ../hierarchical_tm/data/paragraphs_F3072.npz
    685f6390c5aed880bc48a6e40766c20808d94405dbf8d65d641b06dafa86079e  ../hierarchical_tm/data/splits.json
    7523133e933459d552a355a28c0ac4967bcc9685be8d3c08d3a00fa42cbe6fe5  ../hierarchical_tm/src/data_checks.py
    0073b2a4bd61a3b86bdac07f7a3022efeaa401a8fec1e09f65927f910fd0f478  ../hierarchical_tm/src/htm_driver.py
    82d98e94f8546c46740c42456557fc45ec270e82befada4ac75a68b38bd4b1e0  ../hierarchical_tm/src/pairdata.py
    4c93f9a10040c183a677baf8743f1b9814272361bf2d96ed2ab84345ae1130b1  ../hierarchical_tm/src/npmi_groups.py
    7ce4a9fc34313e6753f8d51c6cf2632ba2dc2e37318414d33e5b3ac9f4246a6d  ../hierarchical_tm/src/wiki_groups.py
    cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59  ../tm_language_model/data/concept_eval/nltk_data/corpora/wordnet.zip
    8a52c269237c77008227d2d09bea8f4f8003ac62fdd90f48c72c62ae18d4c758  ../hierarchical_tm/results/20261002T192818Z_e4_head_sa_s1_testscore/scores_test.npz
    1acdfc6e953165b3e641b8419d81530a30476e141bc3a83da8a1e7ce41ffeb4c  ../hierarchical_tm/results/20261002T192838Z_e4_head_sa_s2_testscore/scores_test.npz
    7ae28053831600dd40257407dc39a78a544d43d8c9b32b5d53af762674525bde  ../hierarchical_tm/results/20261002T192858Z_e4_head_sa_s3_testscore/scores_test.npz
    4f6a3d99b7b6d45dd025e60759ce8edddcd188089e4aa17aff940122860368ea  ../hierarchical_tm/results/20261002T192918Z_e4_head_sa_s4_testscore/scores_test.npz
    5838c7a07ae657c1ab4ae1c97fdae9d4f4ef627dcbb712a6c39f3413ace8642f  ../hierarchical_tm/results/20261002T192938Z_e4_head_sa_s5_testscore/scores_test.npz
    2164b168b20eb014c95af2dbc02a31805d4c7fbb31c46c602d506c3b51bf213e  ../hierarchical_tm/results/20261002T192959Z_e4_head_np_s1_testscore/scores_test.npz
    02a83e089d514393d301d870fd0ad97a58f7ee4673664f23449b9ffffc86119e  ../hierarchical_tm/results/20261002T193019Z_e4_head_np_s2_testscore/scores_test.npz
    6fc3343c7a9e580e14bde3feeac368f82c4e05f8ff00f5fc5b0ce1f9c4e934b9  ../hierarchical_tm/results/20261002T193039Z_e4_head_np_s3_testscore/scores_test.npz
    f5fabdb1a8bae8b334604953fd9d4b5551652242cf7f217257f5df61d4466862  ../hierarchical_tm/results/20261002T193059Z_e4_head_np_s4_testscore/scores_test.npz
    9c2d7517d25a77975e803d660067f0ba8ae45bcdd2ba1f386d0dee342a098ef8  ../hierarchical_tm/results/20261002T193119Z_e4_head_np_s5_testscore/scores_test.npz
    45a3413f470cb388debac3ece8f444a38aef3019aca9d578a7ddf327bffec384  ../hierarchical_tm/results/20261002T193140Z_e4s_nsall_s100_s1_testscore/scores_test.npz
    6401d9501f691d96e57ca5d094e9c1df983d6af19f3809c8382817fb07f0f71e  ../hierarchical_tm/results/20261002T193200Z_e4s_nsall_s100_s2_testscore/scores_test.npz
    31fb025e02118b5512aa7e9ea0b052c237b100693d9ccbb74f626205843fcd84  ../hierarchical_tm/results/20261002T193220Z_e4s_nsall_s100_s3_testscore/scores_test.npz
    6bf494c0c1447cb2b7ad1c7f87b59c30e123bcd42390ad351dd04b0dd3b525b4  ../hierarchical_tm/results/20261002T193240Z_e4s_nsall_s100_s4_testscore/scores_test.npz
    d7ff9c871516350f81e98e40388a797d8ba1ff8ab9c977bb37c8ad14a370ddea  ../hierarchical_tm/results/20261002T193301Z_e4s_nsall_s100_s5_testscore/scores_test.npz
    726e9f47e1d818b2df2ac9cd2a9b851abe4b14b2c14690fbeac1633d37911bef  ../hierarchical_tm/results/20261002T050235Z_e4_head_sa_s1_scoreP100/include_bits.npz
    87bb364953360ba842ade483b2a7231d5593cbd947138e74d142e017c7b662de  ../hierarchical_tm/results/20261002T050235Z_e4_head_sa_s1_scoreP100/log.txt
    e7a80e7cf615a82fa77ecf566500c884b3a89cf618f20a53252d27a68e339715  ../hierarchical_tm/results/20261002T050235Z_e4_head_sa_s1_scoreP100/metrics.json
    9ba5562a8dc95cf452818397e1f1251269fe9885be6f0b273f4675ea1cdbbaff  ../hierarchical_tm/results/20261002T035511Z_e4_curve_sa_granmo_s100_p200/log.txt
    4c0bb95af02df30a38bfd446684b737f99cadad3a80a9d77e8b3dc7ca3018a9c  ../hierarchical_tm/results/20261002T052930Z_e4_head_sa_s2_score/include_bits.npz
    4c50b1ef12eeac0b553a490f865b1c09e2ad4ee4ef8b57394b9d39225b88ca7a  ../hierarchical_tm/results/20261002T052930Z_e4_head_sa_s2_score/log.txt
    e2da34f40d6addeb6a423dd50ff7c6decd6de5cad4ef0f7e5dec6322d03b9b4f  ../hierarchical_tm/results/20261002T052930Z_e4_head_sa_s2_score/metrics.json
    3e2e433e81dd6b6e888cbf58a14d3521b357808811607bb3d8d34eeddd26ad56  ../hierarchical_tm/results/20261002T050310Z_e4_head_sa_s2/log.txt
    e242a1ffb260177e68fd6172745c4b56ac72f46cab39d5b0ca7e1e203df88c50  ../hierarchical_tm/results/20261002T060727Z_e4_head_sa_s3_score/include_bits.npz
    09201d5cd00e05565959cab7021091c60590ed96aeac6091bb67fe9765eadba4  ../hierarchical_tm/results/20261002T060727Z_e4_head_sa_s3_score/log.txt
    12478200f92fcb88fb9d185172fd12dabffcc65cf2333ecf73e22ecaf56d9226  ../hierarchical_tm/results/20261002T060727Z_e4_head_sa_s3_score/metrics.json
    f1561e3e6325dbadb213f0b699b42ffabb04bde788639ec6763a49591faabb6d  ../hierarchical_tm/results/20261002T053211Z_e4_head_sa_s3/log.txt
    ef7c6ba9bc08c7feb8a6d770d40dc109df034da758dcfadd23d52795d5183652  ../hierarchical_tm/results/20261002T145933Z_e4_head_sa_s4_score/include_bits.npz
    27724912fa95bce0a2d38763c6f359269261d2d4dcd958edf79177acc705784c  ../hierarchical_tm/results/20261002T145933Z_e4_head_sa_s4_score/log.txt
    dc2da12b567b32b7304f307313c200f58214674b7b088e8fe739c0106f1347cd  ../hierarchical_tm/results/20261002T145933Z_e4_head_sa_s4_score/metrics.json
    1e8ae96f01db4a1ff52c3c759fc8f9ed31966cb134391f35a65df1cde8375f10  ../hierarchical_tm/results/20261002T142452Z_e4_head_sa_s4/log.txt
    17c97b7e485f4ca9be7575f85d1448b6e6b2ace06fe0ec2ae0ee61e2dcee84b6  ../hierarchical_tm/results/20261002T152719Z_e4_head_sa_s5_score/include_bits.npz
    63ec959ddfbe27a5b669e61a0cf9ecf47829f4b3da0ac93d509cae964e00f34b  ../hierarchical_tm/results/20261002T152719Z_e4_head_sa_s5_score/log.txt
    ea8aa3ab5be7222bbbc71fe3403450a086206a5fa72bb4bfed09faa5a31cd0db  ../hierarchical_tm/results/20261002T152719Z_e4_head_sa_s5_score/metrics.json
    b8ad8236fb0d5c7bd70d6990a0c0bf278935c62b240a1bb6bf9f5bf577de295c  ../hierarchical_tm/results/20261002T145158Z_e4_head_sa_s5/log.txt
    87d1802d4e5dc6a6773f2a2908b1531ba5e8192a2b2e083b2b3b8bb804738d01  ../hierarchical_tm/results/20261002T062612Z_e4_head_np_s1_score/include_bits.npz
    28a8b3faf6b671fa8baf475460c72b871b4a65fa841dc1196ec66ad8855cdd0b  ../hierarchical_tm/results/20261002T062612Z_e4_head_np_s1_score/log.txt
    73fa7581baf812bfbde8d31d1c0456d0c1d2f82f85d895e7405b8a30fcd7cd66  ../hierarchical_tm/results/20261002T062612Z_e4_head_np_s1_score/metrics.json
    0579b50bccaf66ffd0dfc4c07abe7aee8fe86737b2db17659ef9ceef03586ba8  ../hierarchical_tm/results/20261002T055851Z_e4_head_np_s1/log.txt
    f7fe101bf8fb93dd97dde9897928b938fc366b936633a7ada0633be5a00fa1ff  ../hierarchical_tm/results/20261002T070833Z_e4_head_np_s2_score/include_bits.npz
    b5a88770815b60a1877a97fb6e555d0a069bc597ea93c03223c57c7f86c55432  ../hierarchical_tm/results/20261002T070833Z_e4_head_np_s2_score/log.txt
    aefc9b5ec4b16a1e3346cbc5477a896d5e8f8d8787730a235070657c1ab46d63  ../hierarchical_tm/results/20261002T070833Z_e4_head_np_s2_score/metrics.json
    9a3e024e10191ddc5936e9d731550904dd228ced909974796d77bfc1f01d7909  ../hierarchical_tm/results/20261002T062537Z_e4_head_np_s2/log.txt
    edf32b7570b90c3a2cd4fe8855d9c8b8825637fb91a6f386d5b7dac3ac1f99dd  ../hierarchical_tm/results/20261002T071028Z_e4_head_np_s3_score/include_bits.npz
    a789830c790a635b65b1255a44c4de6c91c831ca726ffeff38690cd4ba554f11  ../hierarchical_tm/results/20261002T071028Z_e4_head_np_s3_score/log.txt
    355583c84004cbe7b4d7d0e4635606508802e6f915f2f8bf986dfa245a8554a6  ../hierarchical_tm/results/20261002T071028Z_e4_head_np_s3_score/metrics.json
    44c421a0f0bfeb98bb395a58f2da58be65474414cd572aebd7aeeb026e16d38e  ../hierarchical_tm/results/20261002T064353Z_e4_head_np_s3/log.txt
    462c91e21af79e489dd386afbcbb7dd1d9d29c9b00e8f808ec7fbe0659074d9e  ../hierarchical_tm/results/20261002T152755Z_e4_head_np_s4_score/include_bits.npz
    abcbfe28e50114c818a0a6d5cbe95f59c6b9a16e581fca8798b8001ebd0462f0  ../hierarchical_tm/results/20261002T152755Z_e4_head_np_s4_score/log.txt
    a2f8bd1fa1dd5c28ace3a2029cb3bc60fa3f6134764e62f0cd800c6d726ec1d2  ../hierarchical_tm/results/20261002T152755Z_e4_head_np_s4_score/metrics.json
    2ded33cbb064f483fbaf44a51d5ae1965882cf046b3ee3f4dc932a89b2d0fcd3  ../hierarchical_tm/results/20261002T150009Z_e4_head_np_s4/log.txt
    e001f9b04097b9012de1585a50a90c0f2a72e25c50df176ea9e975e031030328  ../hierarchical_tm/results/20261002T155611Z_e4_head_np_s5_score/include_bits.npz
    adec1be2248fa0f66b0aa802f00d2ebd8653fa497a56c473202061af0efc45bc  ../hierarchical_tm/results/20261002T155611Z_e4_head_np_s5_score/log.txt
    e1f5e51df28289046a17f571ca8a44ade9d72c14a51ba0172adad8a62722b7e8  ../hierarchical_tm/results/20261002T155611Z_e4_head_np_s5_score/metrics.json
    59cdcd9ad4af9307fdbc589dfc3efa83f12237bcbccbc960df807655c6b669c9  ../hierarchical_tm/results/20261002T151944Z_e4_head_np_s5/log.txt
    cd8d06ddf6676cb7e35ca0c1febe209bcc88494308eb15d758f8ffe075f0b695  ../hierarchical_tm/results/20261002T082456Z_e4s_nsall_s100_s1_score/include_bits.npz
    d36cbc7eb20ae452b29bc2dca96058a1ea3c2ac2e0dc6de7bd7123fd7405bce7  ../hierarchical_tm/results/20261002T082456Z_e4s_nsall_s100_s1_score/log.txt
    52046b59cef0079881440656f84b86f5181bb9c4eb334adbc82c6b1832a6a5c1  ../hierarchical_tm/results/20261002T082456Z_e4s_nsall_s100_s1_score/metrics.json
    889ff5a8ef54f82568b794dd307e2d50fe734f93e228fe98b81ae36504434c08  ../hierarchical_tm/results/20261002T074540Z_e4s_nsall_s100_s1/log.txt
    9a9d0a86def32813f9555cebd5e6dea1b6340ad1caa0fd5e9a31b3e4e515a265  ../hierarchical_tm/results/20261002T084106Z_e4s_nsall_s100_s2_score/include_bits.npz
    a5d4baefa501e938f1bb143ecb1131af6d5679f16ab95abe2279bc36267c8f1c  ../hierarchical_tm/results/20261002T084106Z_e4s_nsall_s100_s2_score/log.txt
    01718d5c18465ae1bfec915f93fe48eb07b4b4e354e4b46cff7b5e40309fba08  ../hierarchical_tm/results/20261002T084106Z_e4s_nsall_s100_s2_score/metrics.json
    eff9f8ada6b01675e2d77ef4adcb5f1fafd19e137c08b5a3447f9a217e05a7ec  ../hierarchical_tm/results/20261002T081320Z_e4s_nsall_s100_s2/log.txt
    a825a1176789352fd61b02237fe243e6fe33edc7c489a8cdc909b71230d647bc  ../hierarchical_tm/results/20261002T090952Z_e4s_nsall_s100_s3_score/include_bits.npz
    bde8969a350be3f1ec98c67e8e7834f6b6f018aa68d03566db88254664ff9452  ../hierarchical_tm/results/20261002T090952Z_e4s_nsall_s100_s3_score/log.txt
    1c4b94074821679250194cd1c5ed4401a5d221afdf9f2a8da480dc55cbc97298  ../hierarchical_tm/results/20261002T090952Z_e4s_nsall_s100_s3_score/metrics.json
    a10a64777ea93c20619ab079c560bea03b306b7ab378c2999d330611856cde29  ../hierarchical_tm/results/20261002T084141Z_e4s_nsall_s100_s3/log.txt
    4b32abdbde0a03758201042830b187ebe88b3cf05673d8d70968258c75804bca  ../hierarchical_tm/results/20261002T155646Z_e4s_nsall_s100_s4_score/include_bits.npz
    3273c38b8598790907e96631e2de790e5d1f1d03dd202c3d9b2e92ec38644a53  ../hierarchical_tm/results/20261002T155646Z_e4s_nsall_s100_s4_score/log.txt
    e8b50b5f08bca697a27370347cf01f765b06a16eaba7517f62be7ea8b102d8d5  ../hierarchical_tm/results/20261002T155646Z_e4s_nsall_s100_s4_score/metrics.json
    07cec926beab516f9ecc8edab75a509c54bf10b6b965bbe1a422368ba9825ac1  ../hierarchical_tm/results/20261002T152830Z_e4s_nsall_s100_s4/log.txt
    4c95a9733ef07aa40d86b81d137c0c50043e29b8861f0cf07147786cce3b80f0  ../hierarchical_tm/results/20261002T164352Z_e4s_nsall_s100_s5_score/include_bits.npz
    47ba767d058b6fa7270aefaea2d0007d26e68fd8786bfa22e2ef8054fd86a7ff  ../hierarchical_tm/results/20261002T164352Z_e4s_nsall_s100_s5_score/log.txt
    f05dc3047e1826e4f4fbe0f2a30129f0abf7047b08c4aa822c767e61704c071d  ../hierarchical_tm/results/20261002T164352Z_e4s_nsall_s100_s5_score/metrics.json
    2aba3b718a0a06a6db154320ce86cdbf1ee3a784f0106223e3b61f1509b70396  ../hierarchical_tm/results/20261002T155721Z_e4s_nsall_s100_s5/log.txt
    0905618ea455ec002d75c6662fcf7fb36b742ac54c075b7e8f20ecc3fdfdef0f  ../hierarchical_tm/results/20261002T193513Z_final_test_main_table/report.json
    ab991c57b3eae494ff9a8b00fb5b8f6b3cb41531786bd541cdaf7fa5000adff3  registrations/E5_job.sh
    350d2ee929c65283a486c7cd0a8f127ec86ac4bedda1d7149bbdb474b8e79690  env/requirements.txt
    557c2deb6c27a4e5471a8db5d42a0057bcd9b12a7e107cbc7a017f0697d70bef  registrations/E5_duplicate_articles.json
    2e300054a9284ef4bb0d395048fb5988d3176c100aa184384d22a3cc5eb04328  ../decision_layer/scripts/cpu_load.py
    0979f52238dc8106fb2d4364927736a7dd5e7f873f962805921ab69a6ab698ec  verification/e3_sources/run_wiki.py
    cbfd67dfd79b41b5ced6d92143f754caddba99ef5cdb0cace22d086734d3acb9  verification/e3_sources/wiki_report.py
    d5edd7fdae2bc844ca8e75bbe018e746b1c12e29736039382b55db80f0fddf33  logs/e5_dry_report.json

Files under `src/` and `tests/` (the record holds every file of both folders):

    c1b8eb3e1661ea7f36d87a127956cadb52ac2186f02ffc0efca6c705a0fbba19  src/diag_allocation.py
    4f6d6a35fb52a72b771010548f70e5b23f652fe7cc1bd68610904e30a6b3c50e  src/diag_slide10_pairs.py
    5c9fe39e960bb339b9ad6922384a0c5b0fdf3a9e9d51088679b02f455e716762  src/e3_common.py
    07fe5b9d2e144714d3241eee4d2fc8cab9007a2034ca29aacbbf185bc3c1f020  src/e3_report.py
    cdd4203d58c5b4aa0af54db06d3ff775dd24d7bb25b740b65704cffd19ea3c2e  src/e5_common.py
    8b98009bad83a402a3c7a932d68fa951c9b04e24c6aa9d11be6e4c893b4cabfc  src/e5_duplicates.py
    2a2b67a94ab6f120c3a2fb71429b9de101fda709c95411290518df465139914c  src/e5_report.py
    1737db98afb3920a15c733109825c869e68987e38d087f56c8b8e78c4d02b9be  src/freeze.py
    bf614ee5e02bd10525380641e66b02cbfe9af9c238bc00e9b7c0714cf22232c1  src/metrics.py
    b94db364700f41125dc9c1a12577e91a597c6d5476924330ca7d33b2e728999e  src/orgroup_tm.py
    34481dbab669029b8c7baf8dba8d90c6cc35b74878133229a8f038c379926b4c  src/probe_slide10.py
    b43625bf9e4bf941ca03033db67bc5dc462cac5556be8b99c317c7e65d1eaf89  src/rescore_wiki.py
    8a6b1f29aa8401a7f1cec31481287a2ba250c0d3765fef4ab5869e731160ec9b  src/run_pcp.py
    f92eddc34ddafc65e84ee3469c563d305daf71cf952fa55ffbda8dbd091ec850  src/run_seq.py
    6292c8010ac9075e26bc77824434ff3121940160682b1061d398af4f3e5c2ad5  src/run_slide10.py
    46cc4253197a4057638248504f832cc15a9a090d14398f3ac8e215b0afe95820  src/run_wiki.py
    84c8c6b3da327a15d856dbd84d3744af66743da964f72fc72b179219f9d9142e  src/score_groups.py
    5cfb7f66408bf2438dc60535cbe761a0a867b19899edcd1a011c02316389c677  src/slide10_facts.py
    bc18a566b95acd4fc01ad4590d0516b30e02119781f363ecd9d6fd0133be0cac  src/wiki_report.py
    dd729252c63057f6f9cf9866c04e3095b9414d9983c2031990ec5c3566ef6df3  tests/test_e3_pipeline.py
    da02ffeffa8ed05fb01e35fbf2b106ea67e1e85f5d6af3ed2e72ca3a15f018f5  tests/test_e3_units.py
    d5a265c6bcdaf2e72a2cdebf15de8ab1cf42b2e3fa8105347800faca17b8d699  tests/test_e5_checks.py
    a8b623543e995fdca69367524b01502cef9af39ee3cfba8df5b41c417077a4f0  tests/test_e5_units.py
    1d883a4377e76c42ea9306705672a71692c0f9b45d3efc397d29b26a23f978a7  tests/test_orgroup_tm.py
    3a45c3fbcc685ccc8b457c63fbda5bec4e1548e8697204d87305d16074581506  tests/test_smoke_pcp.py
    bdc473b7962ff0ba276766038740fa8d12bfaba2ad4071270df83901ab98bb16  tests/test_smoke_real.py

Not pinned, quoted (the registered runs append to the run log):

    715150b61eda3e4cbcc8e9e00b5e3d6d036d4a4918717b92d3bc82ee2202290a  logs/RUNS_wiki.md
    e1f5407f86bacc679fc9d1b95a328ef252594914b01902f41b8122023622718d  registrations/dry/E5_dry.frozen.json
    d80616560734c4f328d9c9f25b45ffe58e6b8dfe4e45032383d0bb9e26d089f4  logs/design/wiki_report_dev_sa_tied_s1_design.json
    8592f0026ead3eea4523cbf1162f259b1f0d2a763521620476e0e10e17ba1bbe  logs/design/score_groups_sa_tied_s1_vs_stock.json
    66a7033f8347323f3c56d2487b2e8bb2fdc77859585b635651deedf66f4645d0  logs/e5_dev_readings/np_ratio.json
    c16c0b09163a0789d9bee49275cd4bda778e006b46265ac8474ab10a8ab03e71  logs/e5_dev_readings/np_unit.json
    1379e6db91750587fd40fca2fda8ef314a13704f7b27c78392899ad517191016  logs/e5_dev_readings/ns_all_ratio.json
    e2314b9e3b63efc1bb9565bfb57fe63f3701c001604a2b11cecc42970e61d7e9  logs/e5_dev_readings/ns_all_unit.json
    ac8b866077c0d61bc2c6c3977b640ded0fe51697b8b19c126a68c40657fba6d0  logs/e5_dev_readings/sa_ratio.json
    a194ef6cd4604966b17bece8e02c35daa8d68836ac978f76b6fdb50d9a8b8118  logs/e5_dev_readings/sa_unit.json

The newest outputs of the tests (`logs/tests/`, not distributed):

    47ec20eaf3e5a75fe68046da5eb73876ffbab9dee196c5d8e3068b8900f1c664  logs/tests/test_e5_units_20261112T201755Z.out  (ALL 15 TESTS PASSED)
    b7694f393837eb8e9425bdaaf2f19e79664c43ad5dcdde1309a7bf32162f36c5  logs/tests/test_e5_checks_20261112T204200Z.out  (CHECK TEST PASSED (53 cases))
    be6c1f1aab4df017220a815d30e120f7d7f9800a2dda3aec0c226be70dffe127  logs/tests/test_orgroup_tm_20261112T201755Z.out  (ALL 23 TESTS PASSED)
    19f2a4801af524709d782bbdababc71afe7179709211ff979500082211523a93  logs/tests/test_e3_units_20261112T201755Z.out  (ALL 11 TESTS PASSED)
    b533123999f3f42ef1fb89188ff5415e5ce53bb2c1c2e7d9dcbabf73ea383d69  logs/tests/test_smoke_real_20261112T201755Z.out  (SMOKE TEST PASSED (slide10 text, slide10 pairs, wiki sa, 3 refusals))
    54b5a443e2259bd0566c2faa1cd7e979cd84524f3d45f7b4f89121ec49e3cdfd  logs/tests/test_smoke_pcp_20261112T201755Z.out  (SMOKE TEST PASSED (2 runs))

## Draft history

- Draft 1: 2026-11-12T19:21:22Z, sha256 3fcb3f8af29d6d31062c49761ace24343a053e2f6a151a8d6c783f3e5b74a565. Before any freeze it was checked from two sides (design and statistics; data, provenance and code path), read-only, between 19:22:57Z and 19:54:31Z: not ready, with 1 blocker, 10 majors and 19 minors. A review is not an approval.
- Draft 2 (this text): 2026-11-12T21:04:18Z. Changes: R2 is no rule (E4's branch "G1 met, G2 missed"); the rules are read in the vote sum and need both intervals, the second with the seeds resampled (10,000 resamples); the ratio reading, the two-way interval and the reading without duplicates are readings, and every reading carries the prediction of a review; the fixed sentence names the reading and gives the vote sum's difference and both intervals; the stock setting, its untested second setting and the arm outside E4's plan are named; "What I knew" holds the order of events, the design report, the plan arms, all published test numbers, the moves from DEV to test; the use history names the audit of 2026-10-03 and the re-read of 2026-10-25; the duplicates are counted here; the report keeps every number when a check fails, writes each step as soon as it is done, and a failed reading cannot remove the rules; the job applies the CPU rule itself, takes no setting from the environment, and its restart rule is the report's completeness test; the package check covers every package and the interpreter; the run log is checked up to the freeze; the versions that E3 pins are kept; the registered branch was rehearsed. The dry run of draft 1 is superseded (note of 20:18:34Z in `logs/RUNS_wiki.md`).
- Minors left, with the reason: the job runs three streams of one thread at once (the machine's rule asks for at most two threads per job): kept as the job was specified, each run gated by the CPU rule; further seeds as a reading: not added, five are enough by the review's own computation.
- Every rule of E5 was written after the DEV rows of E4 and of seeds 1 to 3 were seen. The ratio reading was defined after the tied rows of seed 1 were below G2 in the vote sum; the comparison with word overlap is therefore a reading with no rule. No rule was written or changed after any test number of the OR-group rule: none exists before the freeze.
