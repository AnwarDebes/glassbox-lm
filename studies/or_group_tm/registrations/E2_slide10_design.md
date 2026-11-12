# E2: the slide-10 sentence family with the OR-group rule (design work on the DEV sentences)

Publication note: the text below is the plan as it was fixed on 2026-11-12T16:04:09Z (SHA-256 of that file: `64d265c87bf4e39e9463463f9b9c54bcc500fa3dfa6b237d89fe460db0c17167`), with folder and file names changed to those of this repository. Rules, bars, settings and numbers are unchanged.

Written before any slide-10 run of this study. Exploratory design work; the EVAL sentences are not read in E2.

## Question

Check 2b of the hierarchical_tm study: does ONE learned clause capture the sentence family of slide 10 of the project proposal? Bars of gates.md, unchanged: on at least 3 of 5 seeds the best single rule (chosen on the TRAIN sentences by the study's scorer) covers at least 90% of the unseen family sentences, fires on at most 5% of the negated twins, near-miss and broken sentences, and recovers at least 4 of the 6 brackets (Jaccard at least 0.8, fixed words removed). The study's stock runs: 0 of 5 seeds in the pair version and 0 of 5 in the direct-label fallback (its results page, Check 2b).

## Data and scorer (the study's, read only)

`data/slide10/text/{train,dev}.npz` (direct labels: 10,000 sentences each, half of them family sentences), `data/slide10/pairs_slide10.npz` (pairs; 10,000 train and 10,000 held pairs), `slide10_meta.json`; scorer `slide10.check2b(run_dir, data, split_name="dev")`, unchanged, on include bits exported from the model. The DEV sentences are for choosing settings (the study's rule); the EVAL sentences are kept for one registered confirmation (E3, their 2nd use after the study's own).

## Model

Unshaped: K = 12 groups per sentence (the family has 6 brackets; K is not set to the template), 20 clauses, positive polarity only, one seed event per example. A word is a member of a group only after it was present in a sentence on which the clause should fire, so a list of words that only negatives contain (the blacklist of the study's fallback) cannot be expressed.

## What I expect, written now

From the mechanics: the rule of E1 (reinforce every present member, reject a random group) should NOT clean the groups, because a stray word of another bracket is reinforced on every positive that contains it. Two refinements are therefore tested as arms: `reinforce critical` (a member is reinforced only when it is the one present member of its group) and `reject weakest` (the group whose strongest present member is weakest is hit).

## Arms (text version first), seeds 1 to 5, 20 passes, T 8, r 2, initial state -1

- A0: reinforce all, reject random (the rule of E1).
- A1: reinforce critical, reject weakest (the first look).
- A2: reinforce critical, reject random. A3: reinforce all, reject weakest. A4: reinforce share, reject weakest.
- Then single changes on the best arm: K in {8, 16}, r = 4, initial state -2, 50 passes. Every arm that runs is logged in `logs/RUNS_slide10.md`.
- Pair version: the best text arm with mode `near` and `side`, K = 12 per sentence.

## Go-bar for a registered confirmation (E3)

Text version: an arm with at least 3 of 5 seeds passing Check 2b on DEV in BOTH views (288 and 216) is frozen as E3 and scored once on EVAL with 5 new model seeds. The same for the pair version, separately. Below the go-bar the result is a dev-only negative and EVAL is not read.

## Known limits, stated before the runs

- The near-miss sentences of the study are rule-generated (one fixed substitution per bracket), so the near-miss part is weak evidence; the deletion probe of the study (the words of one bracket deleted, nothing added) is added to E3 as a second check.
- The direct-label version is supervised by family labels; only the pair version is the self-supervised procedure of the proposal.
- A pass says that the search finds one CNF clause for this template family; it says nothing about real text.
