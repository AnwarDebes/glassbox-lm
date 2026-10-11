# Registrations of the hierarchical_tm study

This page carries the registrations of the study (1 to 3 Oct 2026): the rules, settings, predictions and analysis plans that were written into its lab notebook before the runs they govern, each with its amendments, and the frozen settings of the one final test. The pass rules of the checks are in `../gates.md`, together with the addenda that were written into that file (among them the rules for the headline `s` and for the pass count P). Results are on `results.md`. Where a registration itself states what was already known when it was written, that statement stays with it here.

All times are as registered, in Norwegian time (CEST, UTC+2). The UTC time is added in brackets in each heading and at each amendment. Run ids are UTC. The sections stand in time order. Where the notebook gives a registration no clock time of its own, the section says so.

Names used in the registered text: E3 is the slide-10 test (Check 2b), E4 the Wikipedia pair runs (Checks 3 and 4a), E4s the synonym test (Check 4b), E5a the news-topic test (Check 2c), and L1 to L3 are the artificial tests (Check 2a). A is the number of options in an OR-group, K the number of OR-groups (per paragraph or per sentence), P the number of passes. Slide numbers refer to the supervisor's project proposal (unpublished). "The note" is the short write-up that the study design planned for 3 Oct 2026. Decisions 1 to 4 are the four open decisions that were settled on 2 Oct 2026 at 21:27 CEST: the method for group comparisons (1), the `s` of the second synonym arm (2), branch A or B after Check 3 (3), and the accuracy that counts in Check 2a (4).

Publication note: this page was put together for publication on 2026-10-10 and 2026-10-11 from the registrations in the study's lab notebook, which is not published and is kept by the author. The wording was edited; the registered rules, settings, thresholds, seeds, predictions, analysis plans and times are unchanged. The sentences of the project proposal that some registrations list are not reproduced: the page says so at each such place. Five word lists that the registrations give are left out as well and are described in words: the 9 shared words of slide 10 and an example bracket (06:15), a two-word group (amendment of 21:39), the 5 replacement words of the near-misses (22:11) and the words of one learned group (00:29).

## 1 Oct 2026, 11:55 CEST (09:55 UTC): decisions on the pair data and on the demo check

The entry records decisions on work that had already run: the rebuild with the changed next-paragraph negatives has the run id `20261001T092928Z_data_build_v2_exploratory`, and the three demo runs are `20261001T091807Z_e1_demo_seed1`, `20261001T092250Z_e1_demo_seed2` and `20261001T092751Z_e1_demo_seed3`. `np` is the next-paragraph pair version. The decision that the second point leaves open is the one of 20:10 (next section).

- Decisions of 11:55:
  - **np negatives** now take one side of a random consecutive pair, the same position law as the positives. This deviates from the task text of the study design but is the fix Check 1b called for.
  - **Main synonym data:** `data/ns_all` (all pairs), pending the decision on the skewed-word count (section of 20:10). `data/ns` (swap, strict) is a side variant. Both are valid at F=1024 only, so do not train them with the 3,072-word bags.
  - The demo check ran 1 repeat per seed (the default of Granmo's MultiConcept demo file is 10 repeats per call). The model settings were his defaults. The single repeat is kept: the study design asks for 3 seeds.

## 1 Oct 2026, 20:10 CEST (18:10 UTC): the synonym data and the safeguard for Check 4b

- Decision of 20:10: "about 3" accepts 4, so `data/ns_all` is the main synonym data. Safeguard: Check 4b's verdict excludes groups that contain the 4 skewed words. Recorded in the gates.md addendum.
- The synonym test uses `data/ns_all`. Check 4b reports its scores with and without groups that contain the 4 skewed words (began, big, since, called), and the verdict uses the version without them (gates.md addendum).
- What was known (Check 1b on `data/ns_all`): one paragraph alone 0.503, partner-shuffle 0.504, length alone 0.500 / 0.503, word overlap 0 everywhere.
  - Only the skewed-word count is borderline: 4 words (began, big, since, called) against "at most about 3". Chance alone gives 2.75 on average and up to 6 (95th percentile).
- The question as it stood before the decision: does "about 3" accept 4? **Default: yes.** 4 is within chance, the words are generic, and the direct one-paragraph test is clean. Otherwise the synonym test falls back to the strict `data/ns` variant.

## 1 Oct 2026, about 20:40 CEST (18:40 UTC): the decision rule of the GPU-sharing test

- **GPU-sharing test, on the morning of 2 Oct 2026 after the queue ends** (logged, 10 to 15 min):
  1. Run 1 short job alone, then 2 copies at once, then 3, and record ms per pair.
  2. Switch the queue to 2 (or 3) jobs at once only if both conditions hold:
     - summed throughput is at least 1.3x the single-job figure;
     - the same seed gives identical results alone and in parallel (same accuracy, AUC and printed rules).
  3. Runs whose timing is itself a reported result keep the GPU to themselves, and speeds measured under different concurrency are never compared.
  4. CPU-only work (baselines, scoring, data builds) may run alongside the GPU queue.

  Decision of 1 Oct 2026, about 20:40 CEST: sharing is used only if it delivers the same result and cuts the time. So if both conditions hold, the queue switches and study rule 4 is updated. If not, it stays at 1 job at a time.

## 1 Oct 2026, after 20:56 CEST (18:56 UTC): the open decision on the accuracy of Check 2a, and its default

The notebook gives this entry no clock time of its own. It stands between the entries of 20:56 (L2 seed 1) and 21:15 (L2, 3 seeds). It is decision 4.

- **Open decision (for the morning of 2 Oct 2026): which accuracy counts in Check 2a?** gates.md says "hierarchical accuracy at least 0.90" but not which accuracy. **Default until the decision: tuned accuracy** (best cut-off chosen on dev, exactly as Check 3 defines it), with plain accuracy reported next to it. Reason: the plain cut-off of `MultiClassTsetlinMachine` flips between all-yes and all-no even when the model ranks pairs almost perfectly. If L2 or L3 are still rising at their last pass, the study design's own fallback applies (up to 1,000 passes, cap 3 GPU-hours).

## 1 Oct 2026, 21:23 CEST (19:23 UTC): the condition for the 1,000-pass fallback of L3

The condition was written down when the 200-pass result of seed 1 was known and before the 200-pass runs of seeds 2 and 3 had started (`20261001T193947Z_e2_L3_hier_s2` and `20261001T200422Z_e2_L3_hier_s3`). The fallback is the fail path of Check 2a in gates.md.

- **L3 hierarchical seed 1 did not learn:** AUC 0.51 after 200 passes, with the yes/no answer stuck on one class.
  - The study design's Check 2a fallback (1,000 passes, cap 3 GPU-hours) was added to the end of the queue of that night for seeds 1 to 3, about 45 min each. It is cancelled if seeds 2 and 3 learn at 200 passes.

## 2 Oct 2026, about 01:45 CEST (23:45 UTC on 1 Oct): changes to the run plan

- **Changes decided on 2 Oct 2026, about 01:45 CEST:**
  - **The same night, after the last L3 rerun:** L3 hierarchical at `s` = 100, 200 and 400 (seed 1, 300 passes, all else unchanged). These are queued. If any reaches AUC of at least 0.65, that `s` runs on seeds 2 and 3, with group recovery and the unseen-pair score.
  - **On the morning of 2 Oct 2026:**
    1. If L3 learned at some `s`, add a synonym-test (E4s) arm at that `s` (3 seeds plus the shuffled-label control) next to the planned best setting.
    2. Driver: evaluate dev every 5th pass and at every pass of the last 5, after a short run confirms identical results.
    3. GPU-sharing test: add an MPS arm (2 jobs first), confirm MPS is really active, and under MPS stop jobs only through `--max-minutes`, never by killing.
    4. Start the Wikipedia runs (E4 and E4s) right after the sharing test, by day. The final seeds and the final test run as soon as settings are frozen (target: the night of 3 Oct 2026), and the **freeze moves to 4 Oct 2026**.
    5. Screening runs stop early if AUC is below 0.55 at 30% of their passes (never for controls or final seeds).
    6. Runs use fixed plus/minus 1 weights; never call them "weighted".

  These are recorded in the gates.md addendum (no threshold changed).

## 2 Oct 2026, about 03:00 CEST (01:00 UTC): the reading of the free reproducibility check

The notebook gives this reading no clock time of its own. It stands in the run plan of that night, which is dated about 03:00 and whose progress marks go on to 07:02. Its wording still treats the headline `s` as open. The rule of gates.md fixed `s` = 100 at 05:55 (03:55 UTC, section below; run `20261002T035442Z_e4_headline_s_rule`), and the learning-curve run that the check concerns has the later run id `20261002T035511Z_e4_curve_sa_granmo_s100_p200`. "Granmo's setting" in the fixed wording is the headline setting, which gates.md calls Granmo's regime (50 clauses, A 64, K 2, T 3200). The reload check is the check that a saved model, loaded again, gives the score of its run, and the reproduce-within-0.02 rule is the freeze rule of gates.md.

- **Free reproducibility check:** compare the curve run's dev AUC, accuracy and pred_pos for passes 1 to 100 with the earlier 100-pass run at the same `s` and seed. That is `e4_sa_granmo_C50_A64_s100` if the headline `s` stays 100, otherwise the new screen run at that `s`.
  - Command: `src/compare_runs.py A B --dev --upto 100`.
  - Identical: record "Granmo's setting is bit-reproducible over 100 passes".
  - Different: record the first differing pass and the size; that becomes the tolerance for the reload check and for the reproduce-within-0.02 rule of 4 Oct 2026.

## 2 Oct 2026, 03:05 CEST (01:05 UTC): a prediction for L3 at `s` 800, and the undated notes that follow it

Recorded with the L3 group purity diagnostic of 03:05 (seed 1, `s` 21.1 to 400):

- **Hypothesis:** 1/`s` is also the forget rate (Granmo's book, p. 49, "Forget Value 1/s"). A high `s` lets rare words in, but forgets intruder words slowly, which would cost purity.
- **Data so far (one seed; suggestive, NOT a finding):** from `s` 100 to 400, AUC rises from 0.63 to 0.87, while all-groups purity falls from 0.57 to 0.48 (top-8 only from 0.59 to 0.55).
  - Do not state a trade-off until there is more than one seed. The `s` 400 purity for seeds 2 to 3 is added when they run.
  - Prediction for `s` 800: AUC at least as high, purity lower.

The notes below follow the 03:05 entry in the notebook and have no clock time of their own. gates.md carries the first of them as its superseding note of about 03:10.

- **Decided:**
  - the synonym test (E4s) runs **two arms, `s` = 100 and `s` = 400**, each with 3 seeds plus 3 shuffled-label seeds, whatever the headline `s`. Both report AUC and the WordNet/NPMI group scores, because group quality is a design priority and `s` may trade it against accuracy;
  - **no recovery-fix runs in that night.**
- **Options to decide later** (recovery fixes; no code changes now):
  - more clauses at `s` = 400 (are 40 too few to specialise?);
  - an `s` schedule, high then low, by loading the model state into a lower-`s` model;
  - a different `s` per clause (the s_range idea, arXiv 1909.07310).
- **Capacity mismatch (hypothesis).** An L2 hidden group has 8 words, but each OR-group has A = 32 options. At `s` 100, coverage 0.92 with purity 0.49 means about 15 words per group: about 7 right and 8 intruders.
  - **Cheap morning test, needing a recorded decision:** L2 at `s` 100 with A = 8, all else equal (about 4 min per seed). Prediction: purity and recovery rise, and AUC stays similar.
  - This separates "too many slots" from "forgetting too slowly"; the `s` schedule tests the second.
- **Implication for reading the synonym test (no change to the runs):** the regime of Granmo's MultiConcept demo has A = 64 options per OR-group, while real synonym sets have 2 to 5 words, so expect topic-like mixtures. Check 4b compares against random groups of the same size, so it stays fair, but expect a modest effect. Say this in the note.
- **The decisive L3 test:** the unseen-pair runs at the winning `s`. Unseen-pair AUC and accuracy go next to the seen-pair numbers, per seed. Close numbers mean it generalises from words to groups; a big drop means it memorised word pairs.

## 2 Oct 2026, 05:55 CEST (03:55 UTC): the headline `s`, fixed by the registered rule of gates.md

The rule itself is in `../gates.md`. This section carries what the notebook fixes about its application; the measured values of the screen are results and are not repeated here. The times are as recorded: the notebook gives 02:55 for the last change to gates.md, and the headings of gates.md date this rule about 02:58 and its superseding note about 03:10; all three are before the start of the screen at 05:28.

- **Headline `s` = 100 by the pre-registered rule.**
  - The rule was applied by `src/apply_rules.py`, logged as `results/*_e4_headline_s_rule`.
  - The rule was in gates.md before the screen existed: gates.md last changed at 02:55 (00:55 UTC), and the screen started at 05:28 (03:28 UTC).
  - **Selection effect:** `s` was chosen on seed 1's dev data. The one final test decides the reported number.

## 2 Oct 2026, 06:00 CEST (04:00 UTC): two proposals, and the rule in force until they are decided

Both proposals were decided on 2 Oct 2026 at 21:27 CEST (decisions 1 and 2, section below). They were made between about 05:50 and 06:00.

- **Two proposed changes, both checked.** Both are open until decided.
  - **Group-bootstrap calibration.** In the logged toy test (`e5a_cpu_tests_exploratory`), the interval contained 0 in 78% (6 groups) and 91% (30 groups) of no-effect replicates, instead of 95%.
  - **Frequencies in the synonym data** (F = 1,024). 20.6% of the words have frequency at or below 1/100 (21.8% on the first paragraph of positive pairs), and none are at or below 1/200 (the lowest is 0.0057).
  - Changes to the frozen gates need a recorded decision. Until then, the frozen rules apply, and the bootstrap's measured undercoverage (0.78 / 0.91) is printed next to every group verdict.

The text of the two open decisions, as the notebook kept it as the background to each decision:

1. **How group comparisons are decided (Checks 2c, 4a and 4b).**
   - The problem: the frozen "group bootstrap" interval is too narrow when there are few groups. In 1,000 toy cases with no real effect, it wrongly said "beats random" 20% of the time with 6 groups and 8.7% with 30 groups, against the intended 2.5%. The calibrated test below did so 1.1% and 3.5% of the time (`e5a_cpu_tests_v2_exploratory`).
   - On the real Wikipedia NPMI and WordNet matrices, with 30 no-effect groups, the bootstrap's rate was 1.0% to 5.0% and the calibrated test's 0.1% to 2.0% (entry of about 07:00).
   - The proposal: decide with a calibrated test instead. That means an exact randomization test against the matched random groups (p < 0.025), plus a t-interval whose lower bound is above 0, with at least 12 groups.
   - **Until a decision is recorded:** the frozen method decides, and the calibrated test is printed next to it.
2. **Synonym test (E4s): run the second arm at `s` = 200 instead of 400.**
   - Reasons: on real paragraphs, `s` 400 was unstable (AUC 0.600), while `s` 200 was stable (AUC 0.648). In the synonym data, 21% of the words are too rare to enter a rule at `s` 100, and `s` 200 is the smallest `s` tried that admits every word.
   - **Status (13:18; 11:18 UTC):** the point was still open, so the second arm started at 13:18 at `s` 400, the frozen design. If `s` 200 is added later, it runs as an extra arm afterwards: 6 more runs, about 1.5 GPU-hours. The `s` 400 runs are kept either way.

## 2 Oct 2026, 06:06 CEST (04:06 UTC): E5a, news topics (Check 2c)

- **E5a (news topics, Check 2c): built on the CPU, and pre-registered here before any E5a run.**
  - **Data:** AG News (its train file only).
    - Splits: 60,000 train, 10,000 dev, 10,000 test (untouched) and 40,000 NPMI-reference texts.
    - Words: the top 1,024 by train document frequency, stop words removed; 11.5 words per text on average.
    - Logistic regression on dev (best C = 0.1): accuracy 0.862 (always answering the biggest topic gives 0.251).
  - **Model:** Granmo's MultiConcept setup (A = 64, K = 2, positive words only, 8 state bits, boost 0), with 50 rules per topic and T 3,200 (his T per rule, 64), for 50 passes.
    - **Flat model:** his demo's `--vanilla` arm, checked in his code: 6,400 flat rules per topic (50 × 64 × 2), with the same T and `s`.
  - **The `s` screen (seed 1):** hierarchical and flat, each at three values:
    - 21.1, his demo value (at this value only 10% of the words can enter a rule);
    - 60, the median word's 1/p in its main topic;
    - 230, where every word can enter.

    A screening run stops early at pass 15 if dev accuracy is below the biggest-topic share plus 0.05.
  - **Choice, separately for hierarchical and flat:** the `s` with the highest mean dev accuracy over passes 46 to 50.
    - Differences below 0.005 count as ties (about 1.5 standard errors on 10,000 texts), and ties go to the lower `s`.
    - Seed 1 is the screen run at the chosen `s`. Seeds 2 and 3 are new runs without early stop.
  - **Check 2c, accuracy part:** mean dev accuracy at pass 50 over seeds 1 to 3.
    - It passes if the hierarchical mean is at least the flat mean minus 0.02. The hierarchical model uses 50 rules per topic, the flat 6,400.
    - **"Within 2 points" is read one-sidedly here.** A two-sided reading would fail a hierarchical model that is more than 2 points better.
  - **Check 2c, NPMI part:** the counted OR-groups of each topic's top 10 positive rules, pooled over the 3 hierarchical seeds, compared with frequency-matched random groups (`src/npmi_groups.py`). The gate is the frozen group bootstrap; the calibrated test is printed next to it (decision 1).
  - **Limit:** the screen stops at `s` = 230. If accuracy is still rising there, that is a limit of this screen.
  - **Queued at 06:06:** 2 short smoke runs first (the driver has not yet run on the real GPU package), then the 6-run screen, ahead of the second-setting seeds.

## 2 Oct 2026, 06:15 CEST (04:15 UTC): E3, slide 10: the model and the `s` choice

- **E3 (slide 10): generator, tests and Check 2b scorer ready (CPU only). Not started: `slide10_templates.txt` does not exist yet.**
  - `src/slide10.py build` makes `data/slide10/` from the templates file only; it refuses to run without it. `src/slide10.py check2b RUN` scores Check 2b. All 10 tests pass (`slide10_tests_v2_exploratory`), on a test copy of the two sentences of slide 10 of the project proposal plus 5 test-only near-misses.
  - **Counts:** 288 sentences in the family (216 in the project proposal's view, reported for both), 288 negated twins, expanded near-misses, "broken" sentences (one bracket replaced by a filler word), and 8 other sentence families for the "different family" pairs. Every negative lacks at least one word of every slide-10 sentence, so a positive-word rule can reject all of them.
  - **One sentence alone carries no label** (on the test copy): logistic regression 0.499, 95% CI [0.492, 0.506], on 20,000 fresh pairs.
    - An earlier version with only 4,000 held pairs once gave 0.5175 [0.502, 0.533].
    - A logged repeat study (160 repeats) shows that was the usual 5% chance of a 95% interval, not a leak. The real data gets the 20,000-pair check.
  - **Split:** each family uses 30% of its word combinations for training, 35% for dev (choices) and 35% for the Check 2b evaluation, so the evaluation sentences are unseen combinations.
  - **Model (proposed):** the slide-11 hierarchy with K = 6 OR-groups (one per bracket) and A = 4 options (the largest bracket), 50 rules, T 128, 300 passes.
    - A and K are set to the known bracket structure. That is said wherever results appear.
    - 3 `s` values: 12, 40 and 150. Bracket words have frequency at least 0.117, so 1/p is about 8.5; every word has at least 0.0084, so 1/p is about 119.
    - Then 4 more seeds at the chosen `s`, and a flat model (3 seeds) for the study design's comparison. About 1.5 hours on 2 GPU slots.
    - **Amended at 10:29 (08:29 UTC), before any E3 run:** the flat model gets 1,200 rules, leaf-matched by Granmo's `--vanilla` convention (50 × 4 × 6), instead of 500. That is the same rule as every other flat control.
  - **Pre-registered here, before any E3 run: the `s` choice.**
    - Pick the `s` whose seed-1 run meets the most Check 2b criteria on dev (288 view).
    - Ties go first to the higher "coverage minus the highest firing rate", then to the higher dev AUC, then to the lower `s`.
    - The evaluation split is scored only after the choice.
  - **Open point (the default below holds until it is decided).** gates.md does not say how the 9 words shared by every slide-10 sentence count in the bracket Jaccard.
    - Default: remove them from both sets before computing the Jaccard, and also report the strict Jaccard.
    - Without the removal, a perfect group (all alternatives of one bracket) whose options also contain the shared words scores only about 0.3.
  - **Known issue:** `src/text_driver.py` prints the AG News topic names. It must be fixed before any direct-label fallback run for slide 10. The model itself is unaffected.

## 2 Oct 2026, 06:20 CEST (04:20 UTC): the Wikipedia flat-TM control

- **Pre-registered: the Wikipedia flat-TM control** (the study design: "with the partner-shuffle and flat-TM controls"; no setting had been fixed).
  - Granmo's own convention, from his demo's `--vanilla` arm (the same one E5a uses): the flat model gets clauses × options × groups = 50 × 64 × 2 = **6,400 flat rules**, with the same T (3,200) and `s` (the headline `s`, 100) and the same P.
    - This gives it the same number of leaves (word conjunctions) as the hierarchical model: 12,800.
    - Each flat rule is one word conjunction per paragraph.
  - Seeds 1 to 3 on same-article and next-paragraph pairs, models saved.
  - Queued after the headline seeds and the `s` 100 synonym arm (the Check 4a controls run only if Check 3 passes; see 06:25).
  - Its `s` is not tuned separately, as in his demo. That is said next to its numbers.

## 2 Oct 2026, 06:25 CEST (04:25 UTC): additions to the registrations of 06:06, 06:15 and 06:20

- **These pre-registrations were checked a second time.** Before any of these runs:
  1. **E5a:** `s` is chosen separately for the hierarchical and the flat model, each by its own best mean dev accuracy over passes 46 to 50 (already the rule above). One shared `s` tuned on one model would bias the comparison.
  2. **E3:** the model with K = 6 and A = 4 is "a hierarchy shaped to the template". It is reported under that name.
     - **Added:** an unshaped robustness arm, K = 8 and A = 8 (spare slots), at the chosen `s`, 3 seeds, about 45 GPU-minutes on 2 slots.
     - It is reported next to the main arm. The Check 2b verdict stays on the main arm.
     - The reason: at L2, spare slots filled with intruder words. So the reader should see whether one rule still recovers the family without the exact shape.
  3. **Check 4a's shuffled-label runs are queued only if Check 3 passes on the headline seeds.** Check 4a is for branch A only, so after a Check 3 fail they would feed nothing.
  4. **The Wikipedia flat controls:** fine as specified.

## 2 Oct 2026, about 06:35 CEST (04:35 UTC): how the cap of the Check 2a fallback is read

The notebook gives this reading no clock time of its own. Its log entry of 06:35, which reports L2 seed 1 at 1,000 passes, already cites it.

- **How the Check 2a fallback cap is applied (without a separate decision):** gates.md says "run 1,000 passes (cap 3 GPU-hours), then report ... for that rung". The cap is read **per rung**.
  - L3 used 2.18 h (3 runs alone).
  - L2 uses about 1.7 h (3 runs of about 0.57 h, shared under MPS).
  - The cap was first read as one total for all rungs; after the text was checked, the per-rung reading was kept. L2 seed 3 was held from 06:35 to 06:40 meanwhile. Under a reading as one total, L2 seeds 2 and 3 would not count.

## 2 Oct 2026, about 07:00 CEST (05:00 UTC): the scorer rules for Checks 4a and 4b

- **The Check 4a/4b word-group scorer is ready** (`src/wiki_groups.py`, on the CPU; tests `check4_scorer_tests_v3_exploratory`: 10 of 10 passed, on synthetic groups only; no real run's groups were scored).
  - **What it does:**
    - It reads the counted OR-groups from a run's include bits and scores them with NPMI and with the gates.md WordNet score.
    - It compares them with frequency-matched random groups and with the shuffled-label model's groups. Each comparison shows the frozen bootstrap gate with the calibrated test next to it, once with all groups and once without the 4 skewed words.
    - It assembles Check 4b part 1, including a length-only check on `ns_all`: accuracy 0.500 [0.493, 0.507].
    - Its WordNet code agrees with an independent version on all 3,024 test pairs. WordNet covers 957 of the 1,024 words.
  - **Pre-registered here, before any real scoring:**
    1. **The NPMI unit is the whole article** (2,847 untouched articles). The study design defines NPMI as "how often a group's words appear together in new articles". The scorer had first defaulted to paragraphs; that default was wrong, and the default is now `article` (tests re-run as `check4_scorer_tests_v4_exploratory`). Paragraph NPMI may be shown as a secondary view only.
    2. **Two forms of one word** (film / films) count as synonyms under gates.md's literal WordNet definition, and the verdict uses that definition. The score without such pairs is printed next to it.
       - If the verdict changes without them, the note and the status summary say so explicitly, for example: "Check 4b passes only when plural pairs count; without them it does not."
       - Word forms must never be read as synonyms.
    3. **Check 4a seeds:** gates.md names no seed rule. The verdict pools the 3 seeds (as in E5a's Check 2c), and per-seed results are printed. Each seed's groups are compared with the pooled 3 shuffled-label seeds.
    4. **Check 4b:** a seed passes only if part 1 and all 4 comparisons pass, and the check needs 2 of 3 seeds. "About 0.50" means that both the accuracy and the AUC intervals include 0.50.
  - **New evidence for decision 1** (no-effect groups drawn on the real Wikipedia matrices, 30 groups per case): the frozen bootstrap wrongly said "beats random" 5.0% (NPMI, 2 to 6 words), 4.0% (NPMI, 10 to 40 words), 1.0% and 4.0% (WordNet) of the time, against the intended 2.5%. The calibrated test: 1.4%, 2.0%, 0.1% and 1.7%.

Status recorded at 11:34 (09:34 UTC), when Check 4b was scored on the `s` 100 arm:

- **Timing:** run on 2 Oct 2026, before decisions 1 and 2. If those decisions are recorded after this result is known, that is said with them. The entries of 10:25 and 11:10 had planned part 2 for 4 Oct 2026, once both arms are done.
- **Fixed before scoring:** NPMI on whole articles; the frozen gate with the calibrated test beside it; the verdict without groups that contain the 4 skewed words; plural pairs reported both ways.

## 2 Oct 2026, 07:02 CEST (05:02 UTC): the pass count P, fixed by the registered rule of gates.md

The rule itself is in `../gates.md`. This section carries what the notebook fixes about its application; the measured values of the learning-curve run are results and are not repeated here.

- **P = 100 by the pre-registered pass-count rule** (`results/*_e4_pass_count_rule`, `src/apply_rules.py`; the rule was written before the curve run existed).
  - The seed-1 same-article headline is the curve run's model at pass 100, an exact reproduction of the run of 1 Oct 2026 (entry of 06:30). It is re-scored first (`src/score_saved.py`: article bootstrap, rules, include bits).
  - **Selection effect:** P and `s` were chosen on seed 1's dev data, so seed 1's dev numbers are slightly optimistic. Seeds 2 and 3 are not selected, and the one final test decides the reported number.
  - **Queued at 07:02 (30 jobs), in this order:**
    1. the seed-1 re-score;
    2. the headline seeds (same-article 2 and 3; next-paragraph 1 to 3), alternating with the 6 news-topic screen runs, so neither waits hours for the other;
    3. the synonym test at `s` 100 (3 seeds + 3 shuffled-label);
    4. the Wikipedia flat controls (6);
    5. the second synonym arm (still `s` 400 until decision 2 is recorded; switchable while it waits);
    6. then the second-setting seeds.

    All runs use P = 100, eval every 5 passes, and saved models.

## 2 Oct 2026, 07:03 CEST (05:03 UTC): the Check 2a fallback is judged on the final model

This reading was written down together with the result of seed 2 of the L2 fallback at 1,000 passes, so after that result was known. Seed 3 was still running (`20261002T050200Z_e2_L2_hier_s3_p1000`, started at 05:02 UTC). The other measurements of the entry are results and are not repeated here.

- **L2 fallback, seed 2 at 1,000 passes: near miss on accuracy** (`e2_L2_p1000_s2_recovery_purity_posthoc_exploratory`).
  - Accuracy misses: tuned accuracy 0.897 (needs 0.90).
  - Its best pass by dev AUC was 210. The verdict uses the final model (1,000 passes), as the fallback says; choosing a better pass afterwards would be selection on the same dev data.

## 2 Oct 2026, 07:34 CEST (05:34 UTC): cross-fitted tuned accuracy for Check 2a

- **Pre-registered before computing: cross-fitted tuned accuracy for Check 2a.** It is a measurement for decision 4, not a gate change.
  - **Why:** tuned accuracy picks its cut-off on the same dev pairs it is scored on, which favours it slightly, by about the size of L2 seed 3's 0.0015 margin. Cross-fitting removes that, without punishing a misplaced default cut-off. A misplaced cut-off is a calibration problem, not a learning problem.
  - **Scores:**
    - Each run's final model is rebuilt in numpy from its saved include bits (both class machines; counting inference; clipped at ±T), on the driver's own dev pairs.
    - A run is used only if the rebuilt dev AUC and plain accuracy equal the run's logged values exactly.
  - **Split:** the dev pairs are split into two halves, stratified by label, with a fixed numpy seed (2026).
  - **Cut-off:** on one half, the rule "say same if score ≥ c" is tried for every observed score value, plus "never say same".
    - The c with the highest accuracy is chosen; ties go to the largest c.
    - That c is applied unchanged to the other half, and then the halves swap.
  - **Cross-fitted accuracy** is the mean of the two held-out accuracies.
  - **Runs:** L1 (3 seeds); L2 at `s` 21.1 (100 and 1,000 passes, 3 seeds each); L2 at `s` 100 (3 seeds); L3 at `s` 800 (3 seeds). Each is reported next to plain and tuned accuracy.
  - **Whatever it shows is the result.** Decision 4 then offers three options: plain, tuned or cross-fitted.

Timestamp correction, recorded at 07:39 (05:39 UTC):

- Four entries were first written with times ahead of the clock (07:40, 07:50, 08:00 and 08:05). Norwegian time had been mis-tracked.
  - From the results folders (UTC) and file times, they really happened at 07:34 (the cross-fit pre-registration, written before `src/crossfit_acc.py` existed at 07:34:45), 07:35, 07:37 and 07:38. The entries were corrected.
  - The order was always right: the pre-registration came before the computation.
  - The logged config of `e2_crossfit_acc_posthoc` still carries the first-written time, 07:40, in its label. Logged results are never edited, so that label stays wrong there.

## 2 Oct 2026, 07:38 CEST (05:38 UTC): counting in whole pairs or texts (rounding check of the verdict code)

This entry is one of the four whose time was corrected from file times (see the correction under 07:34).

- **Rounding-trap check of the verdict code** (after the 0.8999999999999999 case).
  - **Safe as written.** These compare a single ratio of whole numbers with a threshold, and such a ratio rounds the same way as the threshold:
    - Check 3's accuracy, AUC and window tests (`score_saved.py`, `apply_rules.py`);
    - Check 2b's coverage, firing and Jaccard (`slide10.py`).

    The Monte Carlo p-value in `npmi_groups.py` cannot equal 0.025 exactly.
  - **Fixed:** the P rule compared a difference of two AUCs with 0.01, so a true gain of exactly 0.01 could have failed by rounding. It now allows 1e-12. The gains of that day were 0.0090 and 0.0024.
    - Both rules were re-applied after the fix (`e4_rules_recheck_after_float_tolerance`), and the decisions are unchanged: `s` 100, P 100.
  - **Counted in whole pairs or texts from now on:**
    - averaged accuracies, such as the cross-fit (L2 seed 3: 3,600 of 4,000);
    - E5a's Check 2c test "hierarchical mean at least flat mean minus 0.02". Over 3 seeds × 10,000 dev texts, that means the hierarchical total of correct texts is at least the flat total minus 600.

## 2 Oct 2026, 10:27 CEST (08:27 UTC): flat L3 at `s` 800 on unseen word pairs

- **Pre-registered: flat L3 at `s` 800 on unseen word pairs, for figure 3 of the note planned for 3 Oct 2026** (the study design: "hierarchical vs flat on unseen word pairs (L3)").
  - **Why:** the hierarchical L3 model learns only at `s` 800, but every flat L3 run so far used `s` 21.1, where nothing learns. A figure from those runs would mix up "hierarchy" with "`s`".
  - **The runs:** `--model flat --clauses 1280 --T 128 --s 800 --passes 300`, seeds 1 to 3, on the unseen-pair files, with models saved.
    - **Amended at 10:29 (08:29 UTC), before any run, for consistency:** 400 flat clauses became 1,280. That follows the rule every other flat control uses, Granmo's `--vanilla` convention: flat clauses = clauses × options × groups = 40 × 32 × 1. So the flat model gets the same 2,560 leaves (word conjunctions) as the hierarchical one, where 400 clauses gave it only 800.
    - The earlier flat L1 to L3 runs (at `s` 21.1) used 400 clauses, the old convention. That is said next to the figure.
    - Otherwise these are the earlier flat L3 settings, except `s` (800, the hierarchical L3 value) and passes (300, as the hierarchical `s` 800 runs).
    - The seen and unseen files share identical training pairs (checked), so one run per seed suffices. The saved model can be re-scored on seen pairs later.
  - **Fixed-`s` comparison:** the flat model's `s` is not tuned separately. That is said next to the figure.
  - **Reported whatever it shows** (Check 2a: "hierarchical vs flat ... is reported whatever it shows").
  - Queued after the `s` 100 synonym arm, about 45 GPU-minutes on 2 slots.

## 2 Oct 2026, 12:24 CEST (10:24 UTC): Check 4a is scored only after the branch decision

- **Check 4a inputs are ready, but scoring is held until the decision between branch A and branch B (decision 3) is recorded.**
  - The 3 shuffled-label same-article runs ended at chance (AUC 0.503 / 0.504 / 0.501).
  - The study design bases A or B on Check 3 alone. If Check 4a's result were known first, it could sway that choice. So Check 4a is scored only after branch A is decided, or on 4 Oct 2026 as planned.
  - The command is ready: `src/wiki_groups.py --data sa`, with the 3 same-article headline re-scores and the 3 shuffled-label runs.

Status recorded at 13:49 (11:49 UTC), when Check 4a was scored:

- **Timing:** run **before the decision between A and B** (decision 3).
  - Under gates.md, A or B rests on Check 3 alone. Check 4a then tests branch A's groups.
  - The bias risk is small: a Check 4a fail under A would only demote the groups to examples, while B gives up the whole Wikipedia headline.

## 2 Oct 2026, 15:19 CEST (13:19 UTC): the frozen settings of the final test (completed 15:21 to 18:47, binding from 21:27 CEST)

- **Frozen settings (the block that gates.md calls "FROZEN SETTINGS"): drafted 15:19 (13:19 UTC) and completed 15:21 to 18:47 (13:21 to 16:47 UTC); binding, with the final test decided, from 2 Oct 2026, 21:27 CEST (19:27 UTC).** gates.md requires this block and a recorded decision for the one `--final-test`.
  - **Model, for every run:**
    - PyHierarchicalTsetlinMachineCUDA at commit 5b4a223 (`.venv-htm`), `MultiClassTsetlinMachine`;
    - positive words only, 8 state bits, boost 0, default (counting) inference, clipping on;
    - fixed ±1 rule weights (never "weighted");
    - training batches of 5,000 with both labels, and the numpy generator seeded before every pass.
  - **Main table: these get the one final test.**
    - **Wikipedia headline:** the slide-11 hierarchy (1,024 words, 64 options, 2 groups per paragraph), 50 rules, T 3,200, `s` 100 and P 100 (both by pre-registered rules).
      - Data: same-article and next-paragraph pairs, seeds 1 to 5.
      - Seed 1 same-article is the curve run's model at pass 100. Seeds 4 and 5 are queued now.
    - **Synonym test (E4s):** the same setting at `s` 100 on `ns_all`, seeds 1 to 5, with shuffled-label seeds 1 to 5.
  - **Reported from dev only (3 seeds). No final test of these without a separate decision:**
    - the 2,000-rule second setting (A 8, K 2, T 4,000, `s` 40, 50 passes; same article and next paragraph);
    - the Wikipedia flat controls (6,400 flat rules, `s` 100, P 100);
    - the `s` 400 synonym arm;
    - news topics (E5a: hierarchical 50 rules × 64 × 2 at `s` 60, flat 6,400 at `s` 60, 50 passes; logistic regression C 0.1);
    - the artificial tests L1 to L3 (scored on their own held-out pairs);
    - slide 10 (E3) starts when its templates file exists.
  - **What the final test does:**
    - `src/score_saved.py RUN MODEL test --final-test`, once, on each main-table run;
    - article-bootstrap 95% CIs (whole articles, 1,000 resamples);
    - the verdicts use the frozen gates plus the pre-registered rules of this page.
  - **Baselines on the same test pairs, in the same session** (added 15:21 (13:21 UTC) after a second check; definitions exactly as the dev baselines of 1 Oct 2026 in `src/data_checks.py`, at the model's 1,024 words):
    - **Similarity scores, no training:** word overlap (Jaccard), tf-idf cosine, shared-word count, and PPMI word-vector cosine. The idf weights and PPMI vectors come from train paragraphs only.
    - **Logistic regressions:** one on the pair's shared words, and one on both paragraphs plus their shared words. C is chosen on an 80/20 split inside train, then refitted on all of train.
    - **Synonym test (`ns_all`):** the same set. Word overlap is 0 by construction, and PPMI cosine is the key meaning-based baseline.
    - **The "best baseline"** for each data set is the one with the highest dev AUC, fixed before the test and never chosen on test.
  - **Pre-registered comparison: "beats X".**
    - Paired article bootstrap: 1,000 resamples of whole test articles (a pair belongs to its first paragraph's article). The same resamples are used for every seed and every baseline.
    - In each resample, the statistic is the mean over the 5 seeds of [AUC(seed) - AUC(X)], with a 95% percentile interval.
    - **"Beats X"** only if the whole interval is above 0. **"Below X"** if it is all below 0. Otherwise "not distinguishable".
    - It is reported against word overlap (Jaccard) and against the best-on-dev baseline. Each seed's own paired interval is shown too.
    - **Synonym data (`ns_all`), added at 15:45 (13:45 UTC) after a second check:**
      - Word overlap is 0.50 there by construction (shared words removed), so its line reads "by construction, not a comparison" and can never support "beats word overlap".
      - The comparisons there are against PPMI cosine (the best on dev) and against the logistic regression on both paragraphs and shared words, labelled "word-matching classifier, helpless without shared words" (AUC about 0.506).
  - **Pre-registered 5-seed reporting:**
    - each seed's test AUC and tuned accuracy, with article-bootstrap 95% CIs;
    - the mean and SD over the 5 seeds;
    - the verdict rows use the 5-seed mean;
    - the shuffled-label seeds' test AUC is reported as a no-signal check.
  - **Expect it honestly** (corrected at 15:38 (13:38 UTC): this line first set baseline *accuracies* 0.761 / 0.804 and 0.792 / 0.829 against TM *AUCs*, which understated the gap). With matched metrics, on dev (`final_test_report_devcheck_exploratory`, which reproduces the logged values of 1 Oct 2026):
    - **AUC vs AUC** (same article / next paragraph):
      - TM 0.739 / 0.744 (mean of seeds 1 to 3);
      - word overlap (Jaccard) 0.819 / 0.864; PPMI cosine 0.854 / 0.889; the best logistic regression 0.871 / 0.907.
      - Paired difference TM minus Jaccard: -0.080 [-0.088, -0.072] and -0.121 [-0.128, -0.113], so **below word overlap**.
    - **Accuracy vs accuracy:** TM tuned accuracy 0.665 / 0.671, against Jaccard 0.761 / 0.804 (cut-off chosen on train).
    - **Synonym data (`ns_all`), AUC:** TM 0.619 against PPMI cosine 0.808, so below PPMI (-0.190). Word overlap is 0.500 by construction there.
    - So "beats word overlap" is very unlikely on test for the Wikipedia tasks, and the note must say so plainly.
  - **The final-test report script, `src/final_test_report.py`, is ready** (dev check `final_test_report_devcheck_exploratory`, 15:49; 13:49 UTC):
    - each seed's dev AUC and tuned accuracy equal its own scoring log exactly;
    - the baselines of 1 Oct 2026 are reproduced exactly, in every field;
    - a seed compared with itself gives [0, 0];
    - it refuses the test split without `--final-test`, and on test it reads the best-on-dev baseline from the dev report.
  - **The final-test procedure, only after the recorded decision:**
    1. Re-check `frozen_settings_hashes.json`: 17 data files and 21 scripts, now including `data_checks.py` and `final_test_report.py` (15:54; 13:54 UTC).
    2. Run 20 test re-scores through the queue (`score_saved.py SRC MODEL test --final-test`, named `<run>_testscore`, seconds each): same article 1 to 5 (seed 1 is the curve run's `model_p100.pkl`), next paragraph 1 to 5, synonym test 1 to 5, and its shuffled-label 1 to 5.
    3. Run `src/final_test_report.py --split test --final-test` once over them. The command, as the notebook keeps it in its entry dated 15:49 (the line names the 5-seed dev check, whose run id is 16:44 UTC, so it was completed after that run started):
       `final_test_report.py --split test --final-test --best-from results/20261002T164442Z_final_test_report_devcheck_5seeds_exploratory/dev_report.json --hashes frozen_settings_hashes.json --group sa sa <5 sa testscores> --group np np <5 np testscores> --group ns_all ns_all <5 testscores> --shuffled ns_all <5 shuffled testscores>`
  - **Done (18:47; 16:47 UTC): all 20 main-table runs are trained and pass the same dev checks** (`final_test_report_devcheck_5seeds_exploratory`).
    - Test without `--final-test` is refused, and the hashes are unchanged.
    - All 15 seeds' AUC, tuned accuracy and CIs equal their own scoring logs exactly, and each seed against itself gives [0, 0].
    - The baselines are identical to those of 1 Oct 2026.
    - The final test reads the best-on-dev baselines from `results/20261002T164442Z_final_test_report_devcheck_5seeds_exploratory/dev_report.json` (logistic regression for same article and next paragraph; PPMI cosine for the synonym data).

    **No model has been scored on any test split yet:** the drivers never load it, and `score_saved.py` refuses without `--final-test`.
  - **Hashes:** `frozen_settings_hashes.json` (SHA-256 of 17 data files and 16 scripts).
    - All 92 data hashes recorded by the Wikipedia and news-topic runs equal the current files (checked 15:19).
    - Before the final test, the hashes are checked again. Any change is listed here.
  - **Still open, and none of them changes the settings above:** decision 1 (which group-comparison method is the verdict), decision 3 (A or B framing) and decision 4 (Check 2a accuracy).

Status recorded at 21:37 (19:37 UTC): the one final test ran as `final_test_main_table`, with 20 test re-scores `*_testscore` at 21:28 to 21:35 (19:28 to 19:35 UTC), all exit 0; hashes re-checked unchanged; each model scored on test exactly once. A later change to one hashed script, `src/text_driver.py`, is recorded under 22:28.

The block gives two script counts, both as registered: the 16 scripts of the line on hashes are the state of the draft of 15:19, and the 21 scripts of the first step of the procedure are the state from 15:54 (13:54 UTC), which was re-checked unchanged for the final test and is the state that became binding at 21:27.

## 2 Oct 2026, 21:27 CEST (19:27 UTC): the four open decisions, the slide-6 test and the final test

- **Decisions of 2 Oct 2026, 21:27 CEST:**
  1. **Group comparisons:** the calibrated test now decides (randomization test p < 0.025 plus a t-interval above 0, with at least 12 groups). The frozen bootstrap is shown next to it.
     - It changes no verdict so far: both methods agreed on every line of Checks 2c, 4a and 4b.
  2. **Synonym test, `s` 200 arm:** it runs as an extra arm (3 seeds + 3 shuffled-label seeds, P 100). Its design dates from before any synonym run, but **the decision to run it came after the `s` 100 and `s` 400 results were known**, which is said wherever it appears.
  3. **Branch A** (Check 3 passed).
  4. **Check 2a accuracy = split-averaged cross-fitted accuracy.** So **L2 fails Check 2a** (1 of 3 seeds at 0.90 or more), borderline. Check 2a overall: L1 passes, L2 and L3 fail.
  5. **The slide-6 paraphrase test:** it runs. It is pre-registered before any run (section of 21:29).
  6. **The final test:** it runs. The frozen settings (section of 15:19) are now binding.

What the notebook fixes about the accuracy chosen in decision 4:

- Split-averaged cross-fitted accuracy uses the cut-off procedure of the registration of 07:34, averaged over 200 random splits (`e2_crossfit_split_avg_all_posthoc_exploratory`: mean over 200 stratified splits). It was computed at 07:37 (05:37 UTC; one of the four corrected times, see under 07:34) and was proposed after the 07:35 result of the one pre-registered split was seen. It is a lower-variance version of the same measure, and stricter here.
- It was added after the pre-registered result was known, which is why every reading is shown next to it: plain, tuned, cross-fitted on the one pre-registered split, and split-averaged.

## 2 Oct 2026, 21:29 CEST (19:29 UTC): the slide-6 paraphrase test, with its amendment of 21:39

- **Pre-registered before any code or run: the slide-6 paraphrase test** (decision of 2 Oct 2026, 21:27 CEST).
  - **Question:** does one learned rule cover all 5 paraphrases of slide 6 of the project proposal, E1 to E5 (taken verbatim from the proposal)?
  - **Family**, built from E1 to E5: a template of five brackets (it holds the words of the sentences of slide 6 of the supervisor's project proposal, unpublished, and is not reproduced here).
    - That makes 3 × 2 × 3 × 2 × 5 = 180 sentences. E1 to E5 are 5 of them; each fits the template exactly.
    - **E1 to E5 are kept out of training and dev**, and are scored only at the end.
  - **Data and task:** E3's pair task (same family vs different family) and machinery (`src/slide10.py` functions, all function words kept), with the 8 other families E3 uses as "different family".
    - The other 175 sentences are split by word combination, as in E3: 30% train (every alternative seen at least once), 35% dev, 35% evaluation.
    - The one-sentence check must be about 0.50.
  - **Model, the same kind as E3:**
    - the slide-11 hierarchy shaped to this template: K 5 (one OR-group per bracket) and A 5 (the largest bracket);
    - 50 rules, T 128, 300 passes, models saved;
    - `s` screen at 3 values derived from the build's word frequencies (bracket words and the rarest word), as in E3.
  - **Choice of `s`:** the seed-1 run with the highest dev "coverage of family sentences minus the highest firing rate on other families". Ties go to the higher dev AUC, then the lower `s`. Then seeds 2 and 3 at that `s`.
  - **Reported per seed:**
    - whether the best single class-1 rule (chosen on train sentences, as in E3) fires on all 5 of E1 to E5, and on how many;
    - its coverage of the family's evaluation sentences;
    - its firing rate on the other families;
    - its OR-groups printed next to the 5 brackets.
  - **This is a new, descriptive test, not a frozen gate.** "One rule covers all 5" is claimed only if it holds on at least 2 of 3 seeds, and the rule fires on at most 5% of other-family sentences.
  - **Amended at 21:39 (19:39 UTC), before any slide-6 run** (no slide-6 GPU run exists). The reason, a design weakness:
    - The 8 other families share none of slide 6's content words. So a trivial rule with one OR-group that holds only the two alternatives of one bracket would fire on all 180 family sentences, E1 to E5 included, and on none of the negatives. It would pass the claim while capturing no paraphrase structure.
    - **(a) Hard negatives:** add E3's automatic "broken" sentences for this family (one bracket replaced by a neutral filler word, from the same generator). The claimed rule must fire on at most 5% of them.
    - **(b) Structure:** report bracket recovery for each of the 5 OR-groups (Jaccard of at least 0.8 with the true bracket, the fixed words handled as in E3). "One rule covers all 5" is claimed only with **at least 4 of 5 brackets recovered**, mirroring Check 2b.
    - **(c) Disclosure:** wherever results appear, say that K 5 and A 5 were set to the known template, as for E3.
  - **Built and fixed at 21:50 (19:50 UTC), before any slide-6 run** (`src/slide6.py`; `data/slide6/`; logs `slide6_data_build` and `slide6_tests_v2_exploratory`: 21 tests pass; `src/slide10.py` unchanged, its hash equal to the frozen one).
    - **Data:** 3,300 sentences in 14 families (the slide-6 family 52 / 62 / 61 + E1 to E5; broken 1,248; others 1,872); F = 208.
      - The one-sentence check is 0.505 [0.498, 0.512] on 20,000 fresh pairs, so it passes.
      - Word overlap alone ranks the dev pairs at AUC 0.880.
    - **Decisions on 3 points of the build:**
      1. 96 broken sentences are dropped: those whose filler replaced the determiner and left a sentence containing every word of a family sentence. No word-based rule can reject them, and E3's build refuses such negatives too.
      2. "Highest firing rate", in both the rule choice and the `s` choice, means the maximum over the two kinds of negatives (broken and other families, each pooled). That is the stricter reading, and it follows from the amendment.
      3. Bracket recovery uses E3's Check 2b rule unchanged. Its limits are stated next to the results: a missing alternative whose words appear in another alternative is not visible to it, and an OR-group must contain nearly all words of each alternative.
    - **`s` screen (seed 1):** 25, 50 and 100. The rarest slide-6 bracket word "entire" has 1/p = 19.6, and the rarest word overall has 1/p = 70.4. That is the same rule that gave E3's 12, 40 and 150. 300 passes, no early stop.
    - Then `slide6.py choose` picks `s`, seeds 2 and 3 run at that `s`, and only then are E1 to E5 and the evaluation split scored.

## 2 Oct 2026, 22:11 CEST (20:11 UTC): the inputs of E3 as it started (templates file, data, `s` values)

- **E3 (slide 10) started.**
  - **The templates file `slide10_templates.txt`** (not distributed):
    - It was written at 22:09 (20:09 UTC).
    - Lines 1 and 2 are verbatim from slide 10 of the project proposal (checked against the text of the proposal).
    - **The 5 near-misses follow a fixed one-word substitution rule. The study design had planned independently written near-misses; these were not written independently** (one word outside its bracket in brackets 1 to 5, "only" kept). They were written before any E3 run. This is stated wherever Check 2b's near-miss numbers appear.
  - **Data** (`e3_slide10_data_build`):
    - 5,856 sentences in 21 families, F = 221: 288 slide-10 sentences (216 in the project proposal's view), 288 negated twins, 624 near-misses, 2,784 broken, 1,872 other.
    - The one-sentence check is 0.498 [0.491, 0.505] on 20,000 fresh pairs, so it passes.
  - **The pre-registered `s` values stand:** the rarest bracket word "for" has 1/p 8.5, and the rarest word has 1/p 125 (119 on the test copy: within the 20% band). The screen is `s` 12 / 40 / 150, seed 1, K 6 / A 4 (shaped to the template), queued after the slide-6 screen.

## 2 Oct 2026, 22:28 CEST (20:28 UTC): E3's direct-label fallback

This registration follows the notebook entry of 22:27 (20:27 UTC) on the screen of E3's pair version.

- **Pre-registered before any run: E3's direct-label fallback** (the study design: "The fallback uses direct labels: sentence vs negated twin vs broken sentence"; gates.md Check 2b fail path: "try 2 rules x 64 options x 1,000 passes (cap 2 h)"; it runs after the main E3 runs).
  - **Why both versions:** the pair version (the slide-11 way, the main result) asks whether the TM finds the family by itself. The direct-label version asks the question of slide 10 head on: can one hierarchical clause capture the family and reject the "not" twins?
  - **Data:** `data/slide10/text/`, built by `slide10.py` from the same templates file.
    - Label 1 is a slide-10 sentence. Label 0 is a negated twin, a near-miss (from the fixed one-word substitution rule, not written independently) or a broken sentence.
    - Split by word combination: train and dev 10,000 balanced texts each (drawn from the 30% / 35% combination splits); evaluation 1,390 texts (101 unseen slide-10 sentences).
  - **Model:** Granmo's own MultiConcept demo setting, the gate's "2 rules × 64 options": `text_driver.py --model hier --clauses 2 --A 64 --T 128`, 2 classes, positive words only, 8 state bits, boost 0, **K 6** (one OR-group per bracket, so shaped to the template; disclosed), 1,000 passes, models saved, dev every 5 passes.
  - **`s` screen (seed 1):**
    - 5: just above 1/p of the rarest bracket alternative among positives (one of 4, so p is about 0.25);
    - 21.1: his demo value;
    - 100: high.
  - **Choice:** the 06:15 E3 rule, applied with `slide10.py check2b --split dev`: the most Check 2b criteria met on dev; ties go to the higher coverage minus highest firing, then to the higher dev accuracy, then to the lower `s`. Then seeds 2 to 5 at that `s`, because Check 2b needs 3 of 5 seeds.
  - **Flat comparison** (the study design: "compare with a flat TM"): leaf-matched by Granmo's `--vanilla` convention, 2 × 64 × 6 = 768 flat rules, the same `s` and T, 3 seeds.
  - **Check 2b** is scored on the evaluation split only after the choice. Coverage is reported for 288 and for 216 sentences. Near-miss numbers carry the label that the near-miss sentences are rule-generated (not written independently as planned).
  - **The gate's cap is 2 GPU-hours for this fallback.** If it is reached, the rest is not run and that is reported.
  - **Driver change (22:28):** `src/text_driver.py` gained a backward-compatible `--class-names` option, because it printed the news-topic names. It is used for printing only.
    - The frozen hash of `text_driver.py` refers to the version the E5a runs used.
    - Every run dir keeps its own copy of the driver.
    - The CPU tests are re-run as `e5a_cpu_tests_v3_after_classnames_exploratory`.

## 2 Oct 2026, 23:35 CEST (21:35 UTC): two robustness arms for E3's pair version (R1, R2)

- **Pre-registered before any run: 2 robustness arms for E3's pair version.**
  - **Status:** these are **post-hoc robustness diagnostics, registered after the pair-version results were known. They cannot change the pre-registered Check 2b verdict (fails, 0 of 5) and are reported as configuration sensitivity.** All runs are named `..._exploratory`.
  - **Already established:** the pipeline works. The leaf-matched flat model on the same pairs learns (seed 1: dev AUC 0.891, partner-shuffle 0.500). The shaped hierarchy cannot even fit train (train accuracy about 0.49 at pass 300 on all 5 seeds). The open question is whether one of the setting choices of this study (the template shape K 6 / A 4, T 128, the `s` grid 12 / 40 / 150) caused the failure.
  - **R1, the known-good regime:** the hierarchy at the Wikipedia headline setting on `data/slide10` pairs: 50 rules, A 64, K 2, T 3200, `s` 100, 300 passes, seeds 1 to 3, no early stop, models saved.
  - **R2, the gap in the `s` screen** (the grid jumped from 12 to 40 to 150, while slide 6's best was 25): the template-shaped model at `s` 25: 50 rules, A 4, K 6, T 128, 300 passes, seeds 1 to 3, no early stop.
  - **Reported for both:** dev pair AUC and train accuracy per seed at the final pass, and Check 2b on **dev only** (`slide10.py check2b --split dev`) with the best rule's OR-groups. The evaluation split is not used.
    - For R1, the bracket part of Check 2b is nearly impossible by construction (4 OR-groups across the two sides for 6 brackets, 64 options each), so coverage and firing are the informative parts there.
  - **Reading, fixed now** (final-pass dev AUC):
    - if R1 or R2 learns (dev AUC at least 0.70 on at least 2 of its 3 seeds), the write-up says "the failure is specific to our configuration" and names the setting that works;
    - if neither learns, it says "the hierarchical TM failed on slide-10 pairs across 3 configurations while a flat TM learned" (with the flat model's final numbers).
  - **Order:** queued after the `s` 200 runs. The direct-label fallback's seeds and flat runs (the gate's own fail path) go above them once its `s` is chosen. About 1 GPU-hour on 2 slots.

## 3 Oct 2026, 00:00 CEST (22:00 UTC on 2 Oct): two deviations of the direct-label fallback

- **E3 direct-label fallback: two deviations, recorded before the `s` choice is applied.**
  - **What was already seen when this was written:** the `s` 21.1 screen run had train and dev accuracy 1.0 at pass 45; the `s` 5 run was at chance (dev accuracy 0.50 at pass 125). The `s` 100 run has not started.
  - **1. The screen runs will stop before 1,000 passes.** Each screen job was given a 30-minute cap from a wrong runtime guess. They run at about 2 s per pass, so 1,000 passes need about 34 minutes. Each run stops by its cap at roughly pass 870 to 900 (the exact pass is reported). Under MPS a running job is never stopped or restarted. The `s` choice uses each run's final model as it stands. Seeds 2 to 5 get a 45-minute job cap so that they finish 1,000 passes.
  - **2. How the gate's cap of "2 GPU-hours" is counted under GPU sharing.** gates.md and the 22:28 entry do not say. At 23:32 (21:32 UTC on 2 Oct) the cap had been described as "summed job wall-minutes". That counts the one GPU twice while 2 jobs share it.
    - **Reading A (GPU time):** each fallback job counts its wall time divided by the number of jobs on the GPU at that moment. Under this reading the screen uses roughly 45 to 50 minutes, and seeds 2 to 5 (2 at a time, about 70 minutes) fit inside the 120.
    - **Reading B (summed job time, the description of 23:32):** the screen alone uses about 90 minutes, so seeds 2 to 5 would be beyond the cap and Check 2b could not be judged (it needs 3 of 5 seeds).
    - **Because the choice between A and B was made after the `s` 21.1 run was seen learning, both are reported, and which one counts is decided separately.** Seeds 2 to 5 run (they are inside the cap under A). Under B, the verdict is "cap reached after the screen; not judged". The flat comparison runs only if it still fits under A. Both GPU-time totals are computed from the queue log (UTC) and reported with the verdict.

## 3 Oct 2026, 00:02 CEST (22:02 UTC on 2 Oct): the decision on the two deviations

- **Decision on the 2 fallback deviations** (of 00:01; it was left open to revision).
  - (1) The 30-minute screen cap stands as disclosed. Each screen run's actual final pass is stated next to its numbers.
  - (2) The cap reading: gates.md's "cap 2 h" was written before GPU sharing, when GPU time, summed job time and wall time were the same. **Decision: run the full fallback design (seeds 2 to 5), and judge Check 2b on the 5 seeds under reading A (GPU time).** The reasons: the cap protects the schedule, which has slack; the gate's "report a measured negative" presupposes a judged result; and the decision would be the same if the screen had stayed at chance.
  - **Disclosed wherever the fallback verdict appears:** "cap '2 h' written before GPU sharing; read as GPU time (A) after the screen results were seen. Under reading B (summed job time), seeds 2 to 5 exceed the cap, so the verdict would be 'not judged within the cap'."
  - **Flat 768 × 3 runs last, after the R1 / R2 arms,** labelled "beyond the 2-hour cap under both readings; run because the schedule has slack; the gate verdict does not depend on it". It gives the study design's "compare with a flat TM" for the fallback.
  - **Added to the fallback's Check 2b report:** the best rule's OR-groups in slide-8 style next to the 6 true brackets, and whether one and the same clause both accepts the slide-10 sentences and rejects their negated twins (with the required words that the twins lack).

The notebook's last summary (3 Oct 2026, 03:55 CEST; 01:55 UTC) still lists the choice between reading A and reading B as open.

## 3 Oct 2026, 00:29 CEST (22:29 UTC on 2 Oct): the deletion probe of the direct-label fallback

- **E3 direct-label fallback: the `s` 21.1 screen run's 100% comes from a word blacklist, not from the rule of slide 10** (noted at 00:27; checked in the run's own rules section: `20261002T215720Z_e3fb_slide10_text_C2_A64_K6_s21_seed1`, 1,000 passes completed, train and dev accuracy 1.0).
  - **The slide10-class rule is always true:** all 6 OR-groups have empty options (2 to 4 of 64). It fires on 100% of slide-10 texts and 100% of the others, so it captures nothing.
  - **The not_slide10-class rule:** groups 1 to 5 are always true. Group 6 is a list of words that are not in the template (plus words of no slide-10 text, included at random).
  - So the model learned "reject a text that contains a non-template word". It did not learn slide 10's "accept iff every bracket has an allowed alternative" (an AND of ORs). This data allows that shortcut, because **every negative kind adds a word**: the negated twin adds not / already, a broken sentence adds a filler, and a near-miss adds its replacement word.
  - **Limitation, recorded now:** the 5 near-miss replacement words (from the fixed one-word substitution rule) occur in training. So "fires on at most 5% of near-misses" can be met by memorising 5 words, and the near-miss part of Check 2b is weak evidence in this version.
  - **Wording, fixed now for wherever this run appears:** "perfect classification, achieved by a negative-class word blacklist; the slide10-class rule is always true; no single learned rule captures the family". Check 2b scores the class-1 (slide10) rule, so for this run it fails on firing and brackets.
  - The `s` choice rule (22:28) is applied unchanged when the `s` 100 screen run ends.
  - **Pre-registered now, before any fallback seed exists: the deletion probe** (a post-hoc diagnostic; no training; it cannot change Check 2b).
    - **Texts:** each of the 101 slide-10 EVAL sentences with one bracket's words deleted and nothing added: 6 × 101 = 606 texts. The deleted words are exactly the bracket alternative that the sentence uses, found by its word sequence. A word that also occurs elsewhere in the sentence (for example "the") stays present as a word.
    - **Scored with:** the saved fallback models at the chosen `s` (5 seeds), the other 2 screen runs for information, and the flat 768 runs if they run. Scoring is on the CPU: each model's class sums are rebuilt from its include bits (counting inference, clipped to ±T, ties to class 0, as the package predicts). The rebuild must reproduce the run's logged final dev accuracy exactly first.
    - **Reported:** the share of deletion texts each model accepts as slide 10, per bracket and overall, next to its acceptance of the intact eval sentences.
    - **Prediction, fixed now:** a template rule (whitelist) rejects at least 95% of the deletion texts; a blacklist accepts most of them (more than 50%).

Amended at 02:18 (00:18 UTC on 3 Oct), after the probe results were known, in the entry that reports the probe on the 5 seeds and on the screen runs:

- The `s` 5 screen run accepts 0% of the deletion texts, but it also accepts 0% of the intact slide-10 sentences: it says "no" to everything (dev accuracy 0.50). The script's label "template-like" for it is wrong (the logged run keeps it; results are never edited). **Read it as "reject-all (degenerate)".** Rejecting deletion texts means something only together with accepting at least 90% of intact sentences. The probe script now uses that two-part rule for later runs (the flat 768 models).

## 3 Oct 2026, 01:08 CEST (23:08 UTC on 2 Oct): S6-R1, the slide-6 version of the R1 arm

- **Pre-registered before any run: S6-R1, the slide-6 version of the R1 robustness arm** (registered after R1 seed 1 learned the slide-10 pairs: dev AUC 0.915, train accuracy 0.851, partner-shuffle 0.505, checked in the queue log).
  - **Status: a post-hoc robustness diagnostic, registered after the slide-6 results were known. It cannot change the slide-6 verdict (the claim fails on 3 of 3; recorded on 2 Oct 2026 at 23:03).** Runs are named `..._exploratory`.
  - **Runs:** the hierarchy at the Wikipedia headline setting on `data/slide6/pairs_slide6.npz`: F 208, 50 rules, A 64, K 2, T 3200, `s` 100, 300 passes, seeds 1 to 3, no early stop, models saved.
  - **Reported:** dev pair AUC and train accuracy per seed at the final pass; `slide6.py score --split dev` (coverage, firing on broken and other families, the best rule's OR-groups next to the 5 brackets); and `--split eval` only for E1 to E5 firing (the verdict is final and nothing is chosen from it).
    - Bracket recovery is impossible by construction at K 2 (4 OR-groups across the two sides for 5 brackets, 64 options each), as for R1.
  - **Reading, fixed now:** if dev AUC is at least 0.70 on at least 2 of 3 seeds, the write-up says the slide-6 setting (K 5 / A 5, T 128) was configuration-limited too.
  - **Order:** after R2, before the flat 768 runs. About 12 minutes per run.
  - **Dropped if** R1 seeds 2 and 3 are both below 0.70 dev AUC before S6-R1 starts. That would be recorded.

## Reporting rules of the notebook's summary (no clock time of their own)

gates.md dates the trust safeguards among them at 2 Oct 2026, about 02:45 CEST.

- **Reporting rules:**
  - **Flat controls match the hierarchical model's leaf budget,** Granmo's `--vanilla` convention: flat rules = rules × options × groups. That covers E4, E5a, L3 at `s` 800 and E3. The L1 to L3 flat runs of 1 Oct 2026 (400 rules at `s` 21.1) are the one exception and are labelled "old convention" wherever they appear.
  - If seed 1 run alone twice differs by more than 0.01 AUC, say so before any headline number, and report mean plus or minus spread.
  - Tables give both AUC and tuned accuracy for both Wikipedia settings, so the 2,000-rule setting's higher AUC (0.768) stays visible.
  - **Trust safeguards** (requirement: the time reduction in computation should not cost quality or trust):
  - early-stopped runs are labelled "stopped early by rule at pass X";
  - **the final test runs only after a frozen-settings block exists and the decision to run it is recorded;**
  - novelty is worded as "we found no published TM work on paragraph pairs or zero-overlap matching (searched: OpenAlex, ISTM 2022-25, arXiv to Sep 2026, ACL Anthology)".

  Details are in the gates.md addendum.
  - **Selection effect:** P is chosen on seed 1's own dev curve (same-article pairs), so seed 1's dev numbers at P are slightly optimistic. Say so in every table that shows them. Seeds 2 to 3 are not selected, and the one final test decides the reported number.
  - Seed 1's headline outputs at pass P come from `src/score_saved.py` on `model_pP.pkl`: partner-shuffle, rules, include bits, the history up to P, the best pass and the stability window. It also checks that the re-score equals the run's own dev AUC at P.
