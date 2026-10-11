# gates.md: the pass rules, frozen before any new run

Frozen: Thu 1 Oct 2026, 11:15 Norwegian time (09:15 UTC).

Source: the study design.

**Do not edit.** A rule that looks wrong is recorded as an open decision; the rule is not changed.

Publication note: the wording of this copy was edited for publication on 2026-10-10 and 2026-10-11. The registered checks, thresholds and seeds, and the dates and times of the rules and addenda, are unchanged; in the rules and addenda, conditions on how a decision reaches the record were shortened to "a recorded decision". References to the notebook and to the plan of the study were redirected to [docs/registrations.md](docs/registrations.md) and to the [README](README.md) of the study. Times are Norwegian time (CEST, UTC+2) as registered; in the headings of the addenda the UTC time was added in square brackets, because run ids are UTC. The file as it stood during the study is kept by the author.

## Definitions

- **Score and AUC.** The score of a `MultiClassTsetlinMachine` is `class_sum[1] - class_sum[0]`, where class 1 = same article (or same sentence family). AUC is computed on this score.
- **Always-"no" accuracy:** the share of negatives in the evaluated split.
- **Tuned accuracy:** accuracy at the best cut-off on the score, chosen on dev data.
- **Pass:** one full pass over the training data.
- **Partner-shuffle:**
  - keep paragraph A and replace paragraph B with a random paragraph B from the same split, keeping the label;
  - a model that really relates the two paragraphs must drop to about 0.50 (95% CI includes 0.50).
- **Shuffled-label model:** the same model and settings, trained on the same pairs with the labels shuffled. It is the "no signal" control for word groups. A truly untrained TM has only empty rules, so it has no groups to score.
- **Always-true OR-group:** an OR-group with at least one empty option. It is never counted as a word group.
- **Counted OR-groups:**
  - all OR-groups in the rules that vote for class 1;
  - not always-true;
  - repeated words removed;
  - at least 2 distinct words.
- **Article bootstrap:**
  - resample whole articles (with all their pairs), with replacement;
  - 1,000 resamples;
  - 95% percentile interval.
- **Group bootstrap:** the same, resampling OR-groups.
- **NPMI reference text:** articles that are in no split. They are reserved by the Day-1 data build and used only for NPMI.
- **Random groups:**
  - the same sizes as the counted OR-groups;
  - words drawn from the same vocabulary, with frequency matched by frequency bins;
  - 1,000 draws.

## Check 1a (G0), Thu: our code reproduces his demo

- Granmo's MultiConceptDemo with his defaults, run through a seed wrapper (his file unchanged), reaches at least 0.95 test accuracy on each of 3 seeds.
- The "one word decides" test (label = paragraph A contains one planted word) with `MultiClassTsetlinMachine` reaches AUC of at least 0.98.
- Our numpy recomputation of the clause outputs agrees with the package:
  - exactly under `binary_inference=True`;
  - clause by clause (fires or not) under the default inference.
- **Fail:** fix our code (cap 3 h). If it still fails, stop all text runs; the next step is recorded as an open decision.

## Check 1b (data), Thu: no shortcuts

This check runs for each pair version:
- same article;
- next paragraph;
- next paragraph with shared words removed.

Rules:
- Positives and negatives have the same single-paragraph distribution: sample the article pair first, then the paragraphs.
- The order within a pair is random.
- Splits are article-disjoint: train, dev and final test. The final test is used once, on Day 5.
- At most about 3 vocabulary words show |z| > 3 in a two-proportion z-test of single-paragraph word frequency, positives vs negatives.
- A logistic regression that sees only one paragraph scores accuracy with a 95% CI that includes 0.50, on at least 10,000 dev pairs.
- Partner-shuffle: a logistic regression on [A, B, A AND B], trained and tested on partner-shuffled pairs, scores accuracy with a 95% CI that includes 0.50.
- Shared-words-removed version only:
  - word overlap is 0 for every pair;
  - negatives come from two different articles, with shared-word counts matched to the positives' counts before removal;
  - a classifier on the two paragraph lengths alone scores accuracy with a 95% CI that includes 0.50. If it does not, match the lengths by subsampling (cap 2 h).
- **Fail:** rebuild (cap 2 h), then use the versions that pass and say so. If none passes, stop all Wikipedia runs; the next step is recorded as an open decision. No Wikipedia number is quoted before its version passes.

## Check 2a (G1), Fri: artificial tests L1 to L3

- For each of L1, L2 and L3, on 2 of 3 seeds:
  - hierarchical accuracy is at least 0.90;
  - group recovery is at least 0.80. Group recovery = the share of hidden groups matched by a counted OR-group with Jaccard of at least 0.8.
- At L3, accuracy on the 25% of same-group word pairs held out of training is at least 0.85, on 2 of 3 seeds.
- The one-paragraph check passes (95% CI includes 0.50) on at least 20,000 fresh pairs per rung.
- Hierarchical vs flat at the same rule budget is reported whatever it shows. It is a check of Granmo's counting argument, not a pass rule.
- **Fail:** run 1,000 passes (cap 3 GPU-hours), then report "not learned at this size" for that rung.

## Check 2b (G3), Fri: slide 10

- On at least 3 of 5 seeds, the best single class-1 rule:
  - covers at least 90% of the unseen family sentences;
  - fires on at most 5% of negated twins, near-misses and broken sentences;
  - recovers at least 4 of the 6 brackets (Jaccard of at least 0.8 with the true bracket).
- Coverage is reported for both 216 and 288 sentences.
- The generator keeps every function word, and a unit test checks that every negative lacks at least one word of its source sentence.
- **Fail:** try 2 rules x 64 options x 1,000 passes (cap 2 h), then report a measured negative.

## Check 2c (news), Fri: AG News word groups

- Hierarchical accuracy is within 2 points of a flat TM while using fewer rules (mean of 3 seeds).
- The top OR-groups beat random groups on NPMI: the group-bootstrap 95% CI of the difference excludes 0.
- **Fail:** show the groups as examples only.

## Check 3 (G2), Sat: Wikipedia learns (A) or not (B)

The best arm passes on 2 of 3 seeds (or on both seeds, if only 2 were run):
- dev AUC is at least 0.65;
- tuned accuracy is at least 0.60;
- over the last 5 passes:
  - the minimum accuracy is at least 0.58;
  - the share of "same article" answers stays between 0.2 and 0.8 in each pass;
- the lower bound of the article-bootstrap 95% CI of tuned dev accuracy is above 0.55;
- partner-shuffle accuracy has a 95% CI that includes 0.50.

Outcome:
- Pass = branch A. Fail = branch B.
- The branch is fixed by a recorded decision: B may be chosen after a pass, but never A after a fail.

## Check 4a (G4), Sun: Wikipedia word groups (branch A only)

- The counted OR-groups beat both random groups and the shuffled-label model's groups on NPMI. Each group-bootstrap 95% CI of the difference excludes 0.
- **Fail:** show the groups as examples only.

## Check 4b (synonyms), Sun: the synonym test (E4s), both branches

The test uses "next paragraph" pairs with every word that appears in both paragraphs removed. It passes on 2 of 3 seeds if all of the following hold:
1. **The model is above guessing.** Dev AUC has an article-bootstrap 95% CI above 0.5. Partner-shuffle and the length-only check both stay about 0.50 (95% CIs include 0.50).
2. **The groups are synonym-like.** The counted OR-groups score higher than random groups and higher than the shuffled-label model's groups (3 seeds), on the WordNet score AND on NPMI. Each group-bootstrap 95% CI of the difference must exclude 0.

The WordNet score:
- It is the share of word pairs inside a group that are:
  - synonyms (they share a synset); or
  - hypernym-close (one is a hypernym of the other within 2 steps, or they share a direct hypernym).
- It uses nouns, verbs and adjectives.
- Pairs with a word missing from WordNet are skipped, and coverage is reported.
- WordNet 3.0 is loaded from `../tm_language_model/data/concept_eval/nltk_data` in `.venv-tm`.

Outcome:
- **Pass:** the claim "his model learns synonym-like groups from real text" is allowed.
- **Fail:** report a measured negative.

## Day 5: the final test and the freeze

- The final-test split is used once, for the main table entries.
- Monday 18:00 Norwegian time: every number is frozen. A number that does not reproduce within 0.02 moves to the list of claims that are not made, in the study README.

## On every run

- Flag predicted-positive shares below 0.05 or above 0.95.
- Report AUC and the always-"no" accuracy.
- Choose passes on dev, never on test.
- Print always-true OR-groups as ALWAYS-TRUE.

## Decision of Thu 1 Oct, about 20:10 [18:10 UTC]

It is an addendum: no threshold above was changed.

- **Check 1b, synonym data.** The all-pairs version `data/ns_all` passes. Its skewed-word count of 4 is accepted as "at most about 3", because the one-paragraph check is clean (0.503) and 4 is within chance. `data/ns_all` is the main synonym data. `data/ns` (the swap version) stays a side variant.
- **Safeguard for Check 4b.** The 4 skewed words (began, big, since, called) are flagged in every printed group. The WordNet score and NPMI are reported both with and without the groups that contain any of them. **The Check 4b verdict uses the version without them**, so they cannot create a false pass.

## Decisions of Fri 2 Oct, about 01:45 [1 Oct 23:45 UTC]

These are addenda; no pass threshold above changed.
- **Final test and freeze.** The final seeds and the one final test run as soon as all settings are frozen (target: Saturday night). The freeze moves from Monday 18:00 to Sunday. A number that does not reproduce within 0.02 still moves to the list of claims that are not made, in the study README.
- **Early stop, screening runs only.** A screening run stops if its dev AUC is below 0.55 at 30% of its planned passes. This never applies to controls, final seeds or the artificial tests (L1 to L3).
- **Evaluation schedule.** Dev is evaluated every 5th pass, at every pass of the last 5, at the 30% mark of screening runs, and always at the final pass. The stability rules of Check 3 use the last 5 passes, which are all evaluated. Before first use, one short run confirms that results are identical to evaluating every pass.
- **Wording.** All runs use fixed plus/minus 1 clause weights (`VANILLA_TM`). Never call them "weighted".

## Pre-registered rule for the number of passes P (Fri 2 Oct, about 02:30 [00:30 UTC], before the learning-curve run exists)

- The learning-curve run uses the headline setting (Granmo's regime: 50 clauses, A 64, K 2, T 3200, s 100), seed 1, same-article pairs, 200 passes, with dev evaluated at every pass. It saves models at passes 100, 150 and 200.
- **Rule:** P = 100, unless the dev AUC at pass 150 or 200 is at least 0.01 above the dev AUC at pass 100 AND passes P-4 to P each meet Check 3's per-pass stability conditions (accuracy at least 0.58; share of "same article" answers between 0.2 and 0.8). In that case, P is the smallest such pass count.
- P applies to the headline E4 seeds on both sa and np, to E4s at the headline setting, and to the final seeds.
- The curve run's saved model at pass P serves as the seed-1 sa headline run, since it uses the same seed, data and settings.
- Saved models are re-scored with `src/score_saved.py`. Seeds 1-3 are therefore not retrained for the final test; only seeds 4-5 are trained.

## Trust safeguards (requirement of Fri 2 Oct, about 02:45 [00:45 UTC]: the time reduction in computation should not cost quality or trust)

- **Early-stopped runs are labelled.** Any run stopped by the early-stop rule is shown as "stopped early by rule at pass X" wherever it appears: tables, the status summary, the note and the chapter.
- **Final-test gate.** `--final-test` runs only after two things exist:
  1. a "FROZEN SETTINGS" block (published in docs/registrations.md) listing every reported configuration, the chosen P, the seeds, the data hashes (`dataset_hashes.json`) and the script hashes;
  2. a recorded decision that makes the frozen settings binding for this one-shot step.

  Target: Saturday evening, Sunday morning at the latest.
- **Novelty wording.** Say: "we found no published Tsetlin-machine work on paragraph pairs or zero-overlap matching (searched: OpenAlex, ISTM 2022-25, arXiv to Sep 2026, ACL Anthology)". Never "first" or "unique".

## Pre-registered rule for the headline s (Fri 2 Oct, about 02:58 [00:58 UTC], before the s = 200 / 400 Wikipedia screen exists)

This is the study design's extra ("s = 400 only if s = 100 beat s = 40"), pulled forward; its condition was met.
- **Screen:** Granmo's regime (50 clauses, A 64, K 2, T 3200), same-article pairs, seed 1, 100 passes, at s = 200 and at s = 400. The screening early-stop applies (AUC < 0.55 at 30% of passes), and a stopped run is labelled.
- **Rule:** the headline s is the value in {100, 200, 400} with the highest dev AUC at pass 100 whose passes 96-100 meet Check 3's per-pass stability conditions (accuracy at least 0.58; share of "same article" answers between 0.2 and 0.8). Ties go to the lower s.
- The learning-curve run, the headline seeds and the main E4s arm all use that s. If it equals the winning L3 s, the separate "L3-s" E4s arm is the same arm and is not run twice.
- **L3:** seeds 2-3 and the unseen-pair runs use the better of s = 400 and s = 800 by final-pass dev AUC (tie: the lower s).
- **Trust:** every s tried (21, 40, 100, 200, 400; and for L3 21.1, 100, 200, 400, 800) is reported, failures included, with Granmo's demo s first. Both s and P are chosen on dev, so the selection effect is noted, and the one final test decides the reported number.

## Superseding note (Fri 2 Oct, about 03:10 [01:10 UTC])

- **Supersedes** the line in the headline-s addendum that reads "If it equals the winning L3 s, the separate 'L3-s' E4s arm is the same arm and is not run twice." E4s always runs **two arms, s = 100 and s = 400**, each with 3 seeds + 3 shuffled-label seeds, whatever the headline s.
  - Reason: the design priority on group quality, and the purity-vs-s trend.
- **Pass count:** both E4s arms use the headline P from the pre-registered P rule. Neither arm gets its own dev-tuned pass count.

## Decisions of Fri 2 Oct, 21:28 [19:28 UTC]

- **Group comparisons (Checks 2c, 4a, 4b): the calibrated test now decides.**
  - Groups vs their frequency-matched random groups: an exact Monte Carlo randomization test (2,000 replicates), one-sided p.
  - The model's groups vs the shuffled-label model's groups: a one-sided permutation test (2,000 permutations).
  - "Beats" means p < 0.025 AND the 95% t-interval over groups (Welch's version for two sets) is above 0. A verdict needs at least 12 counted groups per set.
  - The frozen group bootstrap is still printed next to it.
  - Reason: in no-effect tests, the group bootstrap said "beats random" 20% (6 groups) and 8.7% (30 groups) of the time on toy data, and 1.0% to 5.0% on the real matrices, against 2.5% intended.
  - All group verdicts computed so far were identical under both methods.
- **Synonym test: an extra arm at s = 200** (3 seeds + 3 shuffled-label seeds, the headline P). It is allowed after the s = 100 and s = 400 arms' results were known, and is reported with that label. The s = 400 arm stays reported as the pre-registered contrast.
- **Branch A** (Check 3 passed).
- **Check 2a's accuracy is the split-averaged cross-fitted accuracy:** the mean over 200 stratified splits of the accuracy with the cut-off chosen on one half and scored on the other.
- **The slide-6 paraphrase test** (new, small) is allowed; its design is pre-registered before any run (published in docs/registrations.md).
- **The one final test** is allowed (FROZEN SETTINGS block, published in docs/registrations.md).
