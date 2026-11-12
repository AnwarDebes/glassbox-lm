# E3: registered confirmation of the slide-10 clause on the EVAL sentences (direct-label version)

Fixed when `registrations/E3_slide10_text.frozen.json` exists; `registrations/E3_job.sh registered` writes it with `src/freeze.py` and then runs every job below without a pause. Before the freeze no model of this study has been scored on an EVAL sentence: `src/e3_report.py` reads every run configuration under `runs/`, `runs_dry/` and `logs/tests/` and fails if one shows the EVAL split outside the registered runs, and the runner and the probe refuse the EVAL sentences without the record of this text (tests: `tests/test_e3_units.py`, `tests/test_e3_pipeline.py`). The data facts quoted below were computed on all splits, EVAL included, from the arrays alone, with no learned model (`src/slide10_facts.py`).

## Question

The bars of Check 2b of the hierarchical_tm study, applied to a different model (the OR-group rule of this study, `src/orgroup_tm.py`; not the package that study tested): does ONE learned clause capture the sentence family of slide 10 of the project proposal? The study's own verdict on Check 2b (not met, 0 of 5 seeds in the pair version and in the direct-label fallback) is a closed registered result, and E3 does not change it.

E3 covers the direct-label version only: one sentence in, label = a sentence of the family, or a negated twin, near-miss or broken sentence. The pair version, which is the procedure of the proposal (two sentences, same family or not), is not covered: no arm of it met the bars on the DEV sentences (see "What I knew").

## Headline, fixed now

The report prints this sentence with k, j and the numbers filled in, and it is the only wording used for the result:

"With direct family labels and a setting chosen on the DEV sentences, on k of 5 new seeds the best single clause of the model (picked by the study's scorer on the training sentences) accepts at least 90% of the 101 held-back sentences of the slide-10 template family (and of the 80 among them that are in the 216 view), fires on at most 5% of the negated, near-miss and broken sentences, and recovers at least 4 of the 6 brackets in the 288 reading; j of 5 seeds when both flags of the scorer are required."

It is always printed with two companion sentences: "The model as a whole (at least one of its 20 clauses fires) accepts x to y of the deletion texts (bracket 6: u to v)." and "The pair version (the procedure of the proposal) met none of the bars on the DEV sentences."

"Held-back sentences" means: combinations of the six brackets' alternatives that are in neither the training nor the DEV sentences, while every alternative and every word of them occurs in the training sentences of the family.

The words "the bars of Check 2b are met" are used only if R1 is met, and always with "direct labels" and "a different model". If R1 is not met and the 288 reading is met on at least 3 seeds, the wording is: "R1 is not met (j of 5 with both flags); in the 288 reading of the brackets, which was fixed after the DEV rows were seen, the bars are met on k of 5 seeds."

## Rule (one)

**R1 (both flags of the scorer, the reading of E2's go-bar).** A seed meets R1 when its best rule has both flags `check2b_eval_pass_288` and `check2b_eval_pass_216` of the scorer true, that is:
- (a) it accepts at least 0.90 of the 101 EVAL family sentences (at least 91) and at least 0.90 of the 80 of them in the 216 view (at least 72);
- (b) it fires on at most 0.05 of each kind of EVAL negative: at most 5 of the 101 negated twins, 10 of the 216 near-miss sentences and 48 of the 972 broken sentences;
- (c) it recovers at least 4 of the 6 brackets in the 288 reading AND at least 4 of 6 in the 216 reading (a counted OR-group with Jaccard at least 0.8 against the bracket's words, the fixed words of the family removed from both).

R1 is met when at least 3 of the 5 registered seeds (11, 12, 13, 14, 15) meet it.

The best rule is chosen by the scorer on the TRAIN sentences (family coverage minus the highest firing rate on the three kinds of negatives); among tied clauses it takes the one with the lowest index (`np.argmax`). This was so in every design run; the runner asserts it.

No multiplicity correction is applied: there is one rule. The readings below carry no rule, and there is no either-of claim.

## Readings with no rule, printed for every seed

- **Reading 288:** (a), (b) and (c) with the brackets judged in the 288 reading only. This reading was fixed on 2026-11-12 after the DEV rows of E2 were seen (arm A6: 5 of 5 DEV seeds in this reading, 3 of 5 with both flags); gates.md of the study names the two views for coverage only and gives none for the bracket rule, and the go-bar of E2, fixed before any run, asked for both flags. It is a relaxation of this study's own earlier criterion and is never printed without the count of R1.
- **Each flag alone:** the counts of `pass_288` and of `pass_216`, the two numbers the study's own gate script printed.
- **Whole-model deletion reading (the unit of the study's probe):** each EVAL family sentence with the words of one bracket's alternative deleted and nothing added (6 x 101 = 606 texts, the construction of the study's probe, checked to be identical to its function), scored on the whole model (at least one of the 20 clauses fires), per bracket, with the study's two-part reading (template-like: at least 90% of the intact sentences and at most 5% of the deletion texts accepted). Predicted: not template-like on any seed (DEV, seeds 1 to 5: 0.139 to 0.284 accepted; bracket 6: 0.485 to 0.990).
- **Best-rule deletion check (a consistency check, not a second piece of evidence):** the same texts on the best rule. For a clause of positive words a deletion text holds a subset of the words of the matching broken sentences, so this outcome follows from (b); 428 of the 606 EVAL deletion texts are, once the filler word is dropped, broken sentences of the training file. Also printed on the 178 that are not. Predicted: at or near 0 (DEV seeds 1 to 5: 0.000).
- **One-word reading:** each EVAL family sentence with ONE word of a multi-word alternative removed (569 texts: bracket 3 62, bracket 5 303, bracket 6 204), scored on the best rule. It measures the limit "a group is an OR of single words". Predicted: about 0.86 to 0.92 accepted (DEV: 0.863 to 0.918).
- **Ties:** the number of clauses that share the top selection score on the TRAIN sentences, the brackets each recovers, and how many of them meet both flags (DEV seeds 1 to 5: 3, 3, 4, 1, 3 tied clauses).
- The six bracket Jaccards in both readings and the strict Jaccards (fixed words kept), firing per kind, the DEV accuracy of the vote sum, members, time.
- **Further seeds 16 to 35** (20 seeds, same setting, same pass of the EVAL sentences): every reading above as a count with its exact 95% interval, and pooled with seeds 11 to 15. They carry no rule. If one of them fails a check of the report, all further seeds are left out and the reason is printed.

## What the known numbers predict, and what five seeds can show

From the DEV sentences (the five design seeds, re-made byte for byte by the dry job of this folder, and the two dry seeds 6 and 7): both flags on 3 of 5 design seeds (4 of 7 with seeds 6 and 7), the 288 reading on 5 of 5 (7 of 7). With the two nearest variants of the arm (K 16, r 4) both flags held on 6 of 15 design runs. So R1 is close to an even chance and the 288 reading is likely. Part (c) does not depend on the split: it compares the groups of the best rule with the template. The new information of E3 is five new training seeds, scored on held-back combinations.

If the true share of seeds that meet a rule is 0.5, "at least 3 of 5" is met with probability 0.50; at 0.6, 0.68; at 0.7, 0.84; at 0.8, 0.94. Exact 95% intervals of the share: 3 of 5 [0.147, 0.947], 4 of 5 [0.284, 0.995], 5 of 5 [0.478, 1.000]. A met R1 at 3 of 5 therefore does not say that the DEV result held; the count is always given with its interval, and the further seeds turn it into a rate.

## Data and scorer (the hierarchical_tm study's, read only)

- Training: `data/slide10/text/train.npz`, 10,000 rows drawn with replacement, all from the TRAIN split: 5,000 family rows (all 86 training family sentences), 1,666 negated twins (86 distinct), 1,666 near-miss rows (187 distinct), 1,668 broken rows (729 of the 835 training broken sentences). No other family. DEV scores per pass come from `text/dev.npz` (DEV split only).
- Scoring: `slide10.check2b(run_dir, data=pairs_slide10.npz, split_name="eval")` of the study, unchanged, on include bits exported from the model. The export is checked on every TRAIN and DEV sentence (the study's `accept()` on the exported bits equals the model's own firing, clause by clause), and the report checks that the scorer saw 101 family, 101 negated, 216 near-miss and 972 broken EVAL sentences.
- What "held-back" means here: no bag of words is shared between the splits; every one of the 16 alternatives occurs in 20 to 48 training family sentences; the family has 35 words in all, and every word of the EVAL family sentences occurs in training family sentences; 99 of the 101 EVAL family sentences differ from the nearest training family sentence in one bracket and 2 in two. The EVAL negatives make the same kinds of change with the same replacement words as the training negatives.
- A ceiling set by the data: a hand-written clause with one group per bracket (plus the fixed word that no negated twin holds) accepts every family sentence but fires on 43 of the 187 training near-miss sentences (50 of the 216 EVAL ones): all of one near-miss family, which keeps two of the three words of one alternative of bracket 5. A clause that fires on at most 5% of them must drop words from its bracket-5 group, whose Jaccard is then at most 0.667. So at most 5 of the 6 brackets can be recovered, in both readings. In the 216 reading a bracket-6 group that holds the words of all four time phrases scores 0.556; one that holds the words of the three phrases of the 216 view and one word of the fourth reaches 0.833. (`logs/slide10_facts.json`.)
- Provenance: the data were built once by the study on 2026-10-02 (its run `20261002T201024Z_e3_slide10_data_build`, seed 1, train share 0.3) from a templates file whose first two lines are the two sentences of slide 10 of the project proposal. Neither the templates nor the data are distributed; the run folders of E3 hold the words of the best rules and are not distributed either.
- Uses of the EVAL sentences so far: the study scored them in six completed runs and one that ended with an error: the pair version (registered verdict, 0 of 5 seeds), its unshaped and flat comparison arms (0 of 3 each, no verdict of their own), the direct-label fallback (registered verdict, 0 of 5), the deletion probe (a post hoc diagnostic: 100% of the deletion texts accepted on 5 of 5 seeds), and the flat 768-rule arm with its probe (exploratory, beyond the cap; its first attempt ended with an error). E3 is the first use by this study and the seventh completed scoring of these sentences. The DEV sentences were used for every choice below.
- The study's EVAL numbers (its `docs/results.md`, section Check 2b): pair version, no best rule accepts any held-back family sentence (0% on 5 seeds); flat 1,200 rules, seed 2: 9.9% coverage, firing 4.6% on near-miss and 5.1% on broken sentences; fallback, seed 1: coverage 100%, firing 100% on all three kinds, 0 of 6 brackets (a negative-class word blacklist); deletion probe of the fallback: 100% of the deletion texts accepted on all five seeds; flat 768 rules: coverage 10.9% / 7.9% / 12.9%, firing at most 4.6% / 3.6% / 4.4%, 0 brackets, deletion texts accepted 78.5% / 78.7% / 50.3%.

## Model and setting (arm A6 of E2), fixed now

Every flag written out, as in `registrations/E3_job.sh`:

    src/run_slide10.py --version text --seed S --clauses 20 --K 12 --T 8 --mode near --init-state -1 --seed-state 0 \
        --r 2 --p-rec 1 --p-rep 1 --q-seed 1 --max-seed 1 --reject weakest --reinforce critical --neg-frac 0 \
        --p-decay 0 --seed-all --passes 20 --n-train 0 --split eval --registered registrations/E3_slide10_text.frozen.json \
        --name e3_text_sS --out-root runs

20 clauses of positive polarity, 12 groups per sentence (not set to the 6 brackets of the template), every empty group seeded with every word of the seed sentence, one seed event per example, a member reinforced only when it is the one present member of its group, the weakest firing group hit on a false firing, no decay. Registered model seeds 11 to 15; further seeds 16 to 35. The engine settings the runner cannot set (p_reinf 1.0, smax 31, smin -32, no weights, min_true 1) are fixed by the engine file, which is pinned.

## Code since the design rows

The engine versions in the run configurations of the working folder where E0 to E4 were run: 38afa6cd (A0, A1), 49b6c8d2 (A5, invalid), d23a248c (A5b), 67b93f64 (A6 and its variants, then the pair arms up to P4 with T 2), 53cc7b4b (P4 with T 3 and a first dry run), 32062302 (two P5 runs), and b94db364 (the file pinned here, sha256 below). The dry job of this folder ran the A6 setting with b94db364 on DEV seeds 1 to 7: the five models of seeds 1 to 5 are byte for byte the models of the design rows A6 (state and polarity arrays equal), so those rows hold for the pinned engine.

After draft 1 the runner, the probe and the report were rewritten: guards before any model is trained or scored on EVAL, the engine hash taken before the engine is imported and checked again at the end, the export check, the tie list, the one-word reading, the whole-model reading in the report, and checks that cannot pass with an empty list. `tests/test_e3_units.py` (11 tests) checks each guard with a case in which it must fail, and `tests/test_e3_pipeline.py` (37 cases) runs the dry job in a scratch copy and breaks one thing at a time; both pass.

## What I knew at the time of writing (TRAIN and DEV only)

From `logs/RUNS_slide10.md` (its sha256 when this text was written is in "Hashes"; the registered runs append to it) and `logs/e3_dry_report.json`:
- A0 (the rule of E1: reinforce all, reject random): coverage 0.41 to 0.68, firing up to 0.39, 0 or 1 brackets.
- A1 (reinforce critical, reject weakest): coverage 0.96 to 1.00 (288; down to 0.943 in the 216 view), firing 0.000 on 5 of 5 seeds; brackets 5, 2, 2, 5, 3; both flags on 2 of 5 seeds, the 216 flag alone on 3 of 5.
- A5 (10 rows) invalid, run while a unit test failed; the numbers it showed (both flags on 1 of 5 and on 3 of 5) are not used. A5b (decay 0.1 and 0.5, corrected code): coverage 1.000 and firing 0.000 on 10 of 10 runs, brackets 3 or 4; both flags on 1 of 5 and 2 of 5; the 288 flag on 2 of 5 and 2 of 5.
- A6 (seed-all; this arm): coverage 1.000, 1.000, 0.960, 0.941, 1.000 (288) and 1.000, 1.000, 1.000, 0.914, 1.000 (216); firing 0.000; brackets 5/4, 4/3, 5/4, 4/3, 5/4; the 288 flag on 5 of 5, both flags on 3 of 5. It is the only arm that met E2's go-bar, at exactly 3 of 5, of nine variants (A1 2, A5b 1 and 2, A6 with decay 0.1 2, K 16 1, r 4 2, K 8 2; 50 passes repeats A6's best rules and is not a second sample). A6 was added by a note in the run log (16:11:08Z), not by a fixed note. Arms A2, A3 and A4 and the initial state -2 of E2 were never run.
- Bracket 5 was recovered by no text run of any arm; no run recovered all 6 brackets.
- Dry job of this folder (2026-11-12, 17:38Z, DEV): seeds 1 to 5 as A6 above; seed 6: both flags; seed 7: the 288 flag only, the best rule fires on 0.027 of one kind and accepts 0.036 of the deletion texts. Whole-model deletion shares 0.139 to 0.284 on seeds 1 to 5 (0.234 and 0.350 on seeds 6 and 7), not template-like on any seed; best-rule deletion shares 0.000 on seeds 1 to 6; one-word shares 0.863 to 0.918; 1 to 5 of the 20 clauses template-like by themselves; 1 to 4 clauses tied at the top TRAIN score.
- Pair version (not registered): P1 to P4 (60 or 100 clauses, T 2, 3 or 8, 20 to 300 passes): pair accuracy 0.82 to 0.89, AUC 0.92 to 0.95, coverage 0.000 on 11 runs. P5 (tied groups, 4 runs): accuracy 0.90 to 0.92, AUC 0.93 to 0.97, coverage 0.34, 0.17, 0.69 and 0.00, firing up to 0.37, 0 brackets. No arm met the go-bar.
- E0, E0b and E1 (planted groups) are in `logs/RUNS.md`; E1 was met on 5 of 5 new world seeds and recomputed independently.
- No slide-10 run with a model seed from 11 to 35 exists (design seeds 1 to 5, dry seeds 1 to 7, test seeds 7 and 990).

## Limits, stated before the run

- Direct labels: the family label is given. This is the study's fallback version, not the self-supervised procedure of the proposal.
- One template family of 288 sentences: 86 in training, 101 in DEV, 101 in EVAL. The rule was designed on this family's DEV sentences and is confirmed on the same family; no second family was tried. Nothing follows for real text.
- A group is an OR of single words, not of phrases. A multi-word alternative is matched by any one of its words, so the clause accepts more than the family (DEV: about 0.9 of the family sentences with one word of a multi-word alternative removed). Check 2b has no negative of that kind; the one-word reading measures it.
- The best rule is the best of 20 clauses, picked by the scorer with the labels of the TRAIN sentences. On DEV, 1 to 5 of the 20 clauses behave like the template; the model as a whole accepts 0.14 to 0.28 of the deletion texts. Both are reported.
- Bracket 5 can reach a Jaccard of at most 0.667 by the data (see the ceiling above); at most 5 of 6 brackets can be recovered.
- The near-miss sentences are rule-generated (one fixed substitution per bracket); the best-rule deletion check follows from part (b) and adds no evidence.
- K = 12 is not set to the 6 brackets, but it was kept after K 8 and K 16 did worse on DEV, and the rule's two refinements and seed-all were chosen on this family's DEV sentences. Selection on DEV is optimistic.
- No other method runs in E3. Nothing is said about better or worse than any learner. The stock model of the study had one registered setting and no design on DEV.

## What a miss means, and the other cases

- R1 met (j at least 3): the headline with both counts and the interval of j.
- R1 met with j = 3: "the bars are met on fewer seeds than the 288 reading on DEV (5 of 5)" is added.
- R1 not met, the 288 reading met on at least 3 seeds: the second wording of the headline section; not "Check 2b met".
- R1 not met, the 288 reading not met: "the registered confirmation failed; the DEV numbers of the chosen arm did not hold on new seeds". No claim that a clause captures the family is made for any version; the DEV rows stay in the log as design numbers. The deletion and one-word readings are not a fallback headline.
- A check of `src/e3_report.py` fails after the runs: the use of the EVAL sentences is spent; the numbers are reported in full under "registered run void: <the check>"; nothing is rerun.
- Whatever E3 shows, this study does not score the EVAL sentences again without a new registration that lists E3's numbers.

## Never written, whatever the outcome

"Check 2b is met" or "passed" without "the bars of", "direct labels", "a different model" and the reading; "reverses" or "corrects" the study's verdict; "the hierarchical Tsetlin machine" or "the model of the proposal" as a name for `orgroup_tm.py`; "better than" or "where the stock rule failed"; "the model learned the template", "the clause is the template" or "exactly the family"; "generalises" or "unseen sentences" without "combinations of alternatives that were all in training"; "word classes" or "synonyms" for the brackets; "learns the family self-supervised" (the pair version did not); "understands negation" (the clause holds the word "only"; no claim about negation follows); "reliably", "robustly" or "every seed" unless 5 of 5; "first" or "new" before a literature check; anything about real text; anything about a fault of the stock package; anything about the supervisor beyond "slide 10 of the project proposal".

## Jobs, in this order, in one go

    registrations/E3_job.sh registered

It runs the unit tests (stop on a failure), fixes this text with `src/freeze.py` and every file in "Hashes", checks the record (`src/e3_common.py --check`, stop on a problem), runs seeds 11 to 15 and then 16 to 35, each followed by its probe (`src/probe_slide10.py <run> --split eval`), and ends with `src/e3_report.py --frozen registrations/E3_slide10_text.frozen.json --out logs/e3_report.json` and the sha256 of every output (`logs/e3/SHA256SUMS`). One thread, `nice -n 19`, no GPU, a new empty cache folder for compiled code, `TMLM_RUN_DIR` unset. All seeds are run whatever the earlier ones show; no seed is added or replaced. No file under `src/` or `tests/` and no pinned file is edited between the freeze and the report.

The report runs its checks first (record and plan, sources, pinned files; no EVAL scoring outside the registered runs; per seed: one complete run folder, started after the freeze, the registered setting, name, split and folder, written to the run log, the engine, data and scorer of the record, the expected numbers of sentences, the export check, a probe file of that run whose intact share equals the scorer's coverage) and prints R1 and the readings only when all pass; a failed report goes to a file of its own.

Crash rule: a run that crashes is logged with its reason, its folder is kept and renamed `<folder>_CRASHED`, and the job is started again; it skips every seed whose complete run exists.

## Relation to other registrations

The pair version met no go-bar and has no registration; if one is ever written it is a further use of the EVAL sentences and lists E3's numbers. E5 (Wikipedia pairs) is a separate registration on other held-back data; no claim joins the two.

## Hashes (sha256sum, computed by the command that wrote this section at 2026-11-12T17:47:52Z; the freeze record holds the values at the freeze and the report compares every run with it)

Pinned files outside `src/` and `tests/` (paths relative to this study folder):

    d15b0d5a65994d7de8fa8507d635b3d64368d5dc8ebe9dba46c9f88459c01a32  ../hierarchical_tm/data/slide10/pairs_slide10.npz
    05a2bbcf03656aa031d72f4f7b9b787b60ab5621a843d5b96a40da059af98925  ../hierarchical_tm/data/slide10/slide10_meta.json
    9ff404f6729a39ccf0090437d83f645f8369b2a2775062922acc8dabbed5a018  ../hierarchical_tm/data/slide10/vocab_slide10.json
    d6bd557d415a978dd175cc1d9de8ffe8b67adc56fd1c32fa1c70b91f1a37a0cc  ../hierarchical_tm/data/slide10/text/train.npz
    c54c6a50969a35fbfea62be7350a9bf30d8b624311866dbea299d1bb7b39730d  ../hierarchical_tm/data/slide10/text/dev.npz
    00c0d0812b472f0b38532bd3296f8944956b7fc1e7e8c07d5111a0a7fd81b6f8  ../hierarchical_tm/src/slide10.py
    0073b2a4bd61a3b86bdac07f7a3022efeaa401a8fec1e09f65927f910fd0f478  ../hierarchical_tm/src/htm_driver.py
    1228a2a02610bf5a6e58e02018bd2c08f9ea717d06069f6b91610e72dd3b2976  ../hierarchical_tm/src/recovered/e3fb_deletion_probe.py
    1a3f4089a5e4088b6d734a27b7b401795f85e2c78a14fb5c01576489f2326658  registrations/E3_job.sh
    350d2ee929c65283a486c7cd0a8f127ec86ac4bedda1d7149bbdb474b8e79690  env/requirements.txt
    b912ae4d3dfc399a2168b5ffe5b77872e59df906446437982358e352cfd63aca  logs/e3_dry_report.json

Files of the path under `src/` and `tests/` (the record holds every file of both folders):

    b94db364700f41125dc9c1a12577e91a597c6d5476924330ca7d33b2e728999e  src/orgroup_tm.py
    6292c8010ac9075e26bc77824434ff3121940160682b1061d398af4f3e5c2ad5  src/run_slide10.py
    34481dbab669029b8c7baf8dba8d90c6cc35b74878133229a8f038c379926b4c  src/probe_slide10.py
    07fe5b9d2e144714d3241eee4d2fc8cab9007a2034ca29aacbbf185bc3c1f020  src/e3_report.py
    5c9fe39e960bb339b9ad6922384a0c5b0fdf3a9e9d51088679b02f455e716762  src/e3_common.py
    bf614ee5e02bd10525380641e66b02cbfe9af9c238bc00e9b7c0714cf22232c1  src/metrics.py
    1737db98afb3920a15c733109825c869e68987e38d087f56c8b8e78c4d02b9be  src/freeze.py
    5cfb7f66408bf2438dc60535cbe761a0a867b19899edcd1a011c02316389c677  src/slide10_facts.py
    1d883a4377e76c42ea9306705672a71692c0f9b45d3efc397d29b26a23f978a7  tests/test_orgroup_tm.py
    da02ffeffa8ed05fb01e35fbf2b106ea67e1e85f5d6af3ed2e72ca3a15f018f5  tests/test_e3_units.py
    dd729252c63057f6f9cf9866c04e3095b9414d9983c2031990ec5c3566ef6df3  tests/test_e3_pipeline.py

Not pinned, quoted: `logs/RUNS_slide10.md` 621c4e8de49ee201a80b8a73d8de6ced298a3b32f18c219886ac04b954bd6895 (the registered runs append to it), `logs/slide10_facts.json` b966a817e66d8d6313c079a42d76b7e5e1109b8b6e1c4d5cb6de0c7bb5a7c5ba, `registrations/dry/E3_dry.frozen.json` 7ac904f2d495529c2716f250d9361060caf67e478006e1b553750ca96b77a16a, and the last output of `tests/test_e3_pipeline.py` (PIPELINE TEST PASSED (37 cases)) 47e06efab7663818361070819fd75757b73b0000e820a797556dc3cfd05a5f16. The data file `text/eval.npz` is opened by no code of E3 (the scorer reads the EVAL rows of `pairs_slide10.npz`).

## Draft history

- Draft 1: 2026-11-12T16:35:59Z, sha256 7d91ceb270f22871f13ca4eb481ca1aeb7e60b4afc3cbc8639981c7d5249377e, written in the working folder before the code was moved here. Its rule R1 took coverage from both views and the brackets from the 288 reading, and its rule R2 was a deletion rule on the best rule. Before any freeze it was checked line by line against the data, the code and the run log, from two sides (design and statistics; data, provenance and code path), between 16:37Z and 17:10Z: not ready, with 4 blockers, 17 majors and 24 minors.
- Draft 2 (this text): 2026-11-12T17:47:52Z. Changes: R1 is now both flags of the scorer (the reading of E2's go-bar) and the 288 reading is a labelled reading with no rule; the deletion rule is dropped (it follows from part (b)) and replaced by three readings with no rule (whole model, best rule, one word); the counts are corrected (80 family sentences in the 216 view, not 75; the negatives counted); the use count and the study's EVAL numbers are added; "What I knew" is rewritten from the log; the data ceiling, the tie-break and the meaning of "held-back" are stated; the freeze command, the pinned files and their hashes are written; the runner, the probe and the report check the record before any model is scored on EVAL, take the engine hash before the import, check the export, and fail on an empty list; one job script runs everything from the tests to the report; the dry job re-made the five design models byte for byte and passed every check; the failing paths are tested.
- Every rule of E3 was written after the DEV rows of E2 were seen; the reading of the brackets and the dropping of the deletion rule were decided with those rows known. No rule was changed after any EVAL number: none exists before the freeze.
