# Results of or_group_tm

All runs of this study were made on 2026-11-12. This page gives the numbers of each plan of [registrations/INDEX.md](../registrations/INDEX.md), each with the file it is read from; paths are relative to the study folder. The run logs in `logs/` (one row per run, rows never edited, notes dated), the reports `logs/e3_report.json` and `logs/e3_dry_report.json`, the data facts `logs/slide10_facts.json`, the design outputs in `logs/design/` and the recomputation `verification/recompute_e1.out` are part of the repository; the run folders and the data of hierarchical_tm are not. Design numbers are numbers on development data that were used to choose a rule, a setting or a reading. E1 and E3 are the registered confirmations among the plans below; E5, the registered test on the Wikipedia test pairs, is reported in the [README](../README.md) of this study.

Page written at 2026-11-12T18:27:10Z (UTC). The tables are filled from the files named next to them.

## Planted groups: E0, E0b, the design arms and E1

The tests are the artificial pair tests L2 and L3 of hierarchical_tm, built by its `src/pcp_world.py` (its docstring): at L2, 64 words in 8 hidden groups and one word per paragraph; at L3, 256 words in 8 groups, two words per paragraph, no word shared by the two paragraphs of a pair, and a quarter of the same-group word pairs held out of training, which the unseen pairs test. A pair is positive when both paragraphs come from the same hidden group. The bars are those of Check 2a of hierarchical_tm: plain dev accuracy (the vote sum is at least 1) of at least 0.90 and group recovery (a learned group of at least 2 words with Jaccard at least 0.8 against a hidden group) of at least 0.80, and at L3 a plain accuracy on the unseen pairs of at least 0.85. Every model has 40 clauses of positive polarity and one group per paragraph (K 1). Source of every number in the tables: `logs/RUNS.md`. Cells give seeds 1 / 2 / 3, or 11 / 12 / 13 / 14 / 15 for E1.

### E0 (design data: the files of seeds 1 to 3 of hierarchical_tm; T 4; 5 passes unless named)

| rung, arm | plain dev accuracy | group recovery | purity | coverage | plain accuracy, unseen pairs |
|---|---|---|---|---|---|
| L2, mode near | 0.8150 / 0.8047 / 0.7520 | 0.625 / 0.625 / 0.500 | 1.0 / 1.0 / 1.0 | 0.6522 / 0.7127 / 0.6802 | - |
| L2, mode side | 0.8150 / 0.8047 / 0.7520 | 0.625 / 0.625 / 0.500 | 1.0 / 1.0 / 1.0 | 0.6522 / 0.7127 / 0.6802 | - |
| L2, mode all | 0.6455 / 0.6815 / 0.7000 | 0.000 / 0.000 / 0.000 | 0.4656 / 0.4848 / 0.5039 | 0.33 / 0.3157 / 0.3253 | - |
| L3, mode near | 0.7512 / 0.6817 / 0.7465 | 0.500 / 0.125 / 0.250 | 1.0 / 1.0 / 1.0 | 0.5375 / 0.4984 / 0.4938 | 0.7455 / 0.6845 / 0.7522 |
| L3, mode side | 0.7512 / 0.6817 / 0.7465 | 0.500 / 0.125 / 0.250 | 1.0 / 1.0 / 1.0 | 0.5375 / 0.4984 / 0.4938 | 0.7455 / 0.6845 / 0.7522 |
| L3, mode all | 0.6245 / 0.6492 / 0.6302 | 0.000 / 0.000 / 0.000 | 0.3717 / 0.4022 / 0.3703 | 0.1812 / 0.1906 / 0.175 | 0.5825 / 0.5787 / 0.5730 |
| L2, mode near, 100 passes | 0.8150 / 0.8047 / 0.7520 | 0.625 / 0.625 / 0.500 | 1.0 / 1.0 / 1.0 | 0.6522 / 0.7127 / 0.6802 | - |
| L3, mode near, 100 passes | 0.7512 / 0.6817 / 0.7465 | 0.500 / 0.125 / 0.250 | 1.0 / 1.0 / 1.0 | 0.543 / 0.4984 / 0.4953 | 0.7455 / 0.6845 / 0.7522 |

Modes near and side give the same numbers at K 1. With 100 passes, mode near gives the same accuracies and recoveries as with 5 passes. The bar of E0 is met by no mode. `logs/e0_diag_allocation.out` counts the clauses that hold members of each hidden group: in the runs of mode near, 3 / 3 / 4 of the 8 hidden groups have no clause at L2 and 4 / 5 / 4 at L3 (the same with 100 passes), and no clause is blank or mixed. The early examples had seeded all 40 clauses with the first few hidden groups (E0b, section on what E0 showed).

### E0b (post hoc revision of E0 on the same files: at most one seed event per example)

| rung, arm | plain dev accuracy | group recovery | purity | coverage | plain accuracy, unseen pairs |
|---|---|---|---|---|---|
| L2, 5 passes | 1.0000 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 | 0.875 / 0.8877 / 0.85 | - |
| L2, 100 passes | 1.0000 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 | 0.875 / 0.8877 / 0.85 | - |
| L3, 5 passes | 0.9998 / 1.0000 / 1.0000 | 1.000 / 1.000 / 0.875 | 1.0 / 1.0 / 1.0 | 0.7391 / 0.7398 / 0.7375 | 0.9998 / 1.0000 / 1.0000 |
| L3, 100 passes | 0.9998 / 1.0000 / 1.0000 | 1.000 / 1.000 / 0.875 | 1.0 / 1.0 / 1.0 | 0.7477 / 0.7477 / 0.743 | 0.9998 / 1.0000 / 1.0000 |

The bar of E0 is met on these files at L2 and at L3, with 5 passes and with 100. These numbers are post hoc: the rule was changed after the results of E0 on the same files were seen (E0b, section Label). `logs/e0b_diag_allocation.out`: in the 6 runs with 100 passes every hidden group has a clause (0 / 0 / 0 / 0 / 0 / 0 groups without one).

### The design arms T 8 and T 16 (the files of E0b, 5 passes; in no fixed plan)

| rung, arm | plain dev accuracy | group recovery | purity | coverage | plain accuracy, unseen pairs |
|---|---|---|---|---|---|
| L2, T 8 | 1.0000 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 | 0.9797 / 0.9873 / 0.9719 | - |
| L3, T 8 | 0.9998 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 | 0.9102 / 0.9242 / 0.8586 | 0.9998 / 0.9990 / 1.0000 |
| L2, T 16 | 1.0000 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | - |
| L3, T 16 | 0.9998 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 | 0.9648 / 0.9609 / 0.9555 | 0.9998 / 0.9990 / 1.0000 |

E1 took T 16 on these numbers (E1, section on what was known at its freeze).

### E1 (registered confirmation: worlds of the new seeds 11 to 15; T 16, 5 passes)

| rung, arm | plain dev accuracy | group recovery | purity | coverage | plain accuracy, unseen pairs |
|---|---|---|---|---|---|
| L2 | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 | - |
| L3 | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 0.9998 | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 | 0.9586 / 0.9594 / 0.9617 / 0.9641 / 0.9695 | 0.9992 / 1.0000 / 1.0000 / 1.0000 / 0.9990 |

R1 (L2) is met on 5 of 5 seeds, R2 (L3) on 5 of 5 and R3 (L3, unseen pairs) on 5 of 5; each needed 4 of 5, so E1 is met. `verification/recompute_e1.py` generated the ten worlds again, trained the ten models again with a separate plain-Python implementation of the rule (no numba, no import of `src/orgroup_tm.py`) and computed accuracy, recovery and unseen accuracy from the stored states: in `verification/recompute_e1.out` every retrained state equals the stored one and every number equals the logged one, and its verdict line gives R1, R2 and R3 met. The publication note of `logs/RUNS.md` names the engine and runner versions of these runs; the E1 setting has not been run with the engine now in `src/`.

### Reference: the package tested by hierarchical_tm

In hierarchical_tm Check 2a was not met (its [docs/results.md](../../hierarchical_tm/docs/results.md), Check 2a). The package gave there, at L2 with 100 passes plain accuracy 0.523 / 0.522 / 0.572 and recovery 0.625 / 0.5 / 0.75, with 1,000 passes 0.911 / 0.5025 / 0.634 and 0.875 / 1.0 / 0.875; at L3 (`s` 800, 300 passes) 0.8365 / 0.821 / 0.821 and recovery 0.0 / 0.0 / 0.0, and on the unseen pairs a tuned accuracy of 0.788 / 0.74 / 0.752. These are seeds 1 to 3, the files of E0 and E0b, with other budgets; E1 states that the package had not been run on the worlds of seeds 11 to 15. The two are not compared here.

## Slide 10: E2 (design) and E3 (registered)

The data are those of hierarchical_tm, built there on 2026-10-02 from a templates file whose first two lines are the two sentences of slide 10 of the project proposal; neither the templates nor the data nor the run folders of this section are distributed. From `logs/slide10_facts.json`: 5,856 sentences of 21 families over 221 words. The family of slide 10 is a template of six brackets with 3, 2, 3, 2, 2 and 4 alternatives, 288 sentences in all, of which 86 are TRAIN, 101 DEV and 101 EVAL sentences (66, 70 and 80 of them in the 216 view). The kinds of negative are the negated twins (86 / 101 / 101 in TRAIN / DEV / EVAL), the near-miss sentences (187 / 221 / 216, rule-generated: one fixed substitution per bracket) and the broken sentences (835 / 977 / 972). No bag of words is shared between the splits, no word of an EVAL family sentence is missing from the training family sentences, and 99 of the 101 EVAL family sentences differ from the nearest training family sentence in one bracket and 2 in two. The direct-label training file has 10,000 rows: 5,000 family rows, 1,666 negated twins, 1,666 near-miss rows and 1,668 broken rows.

A ceiling for a clause with one group per bracket (same file): a hand-written clause with one group per bracket, plus one group for a fixed word of the template, accepts every family sentence but fires on 43 of the 187 training near-miss sentences (50 of the 216 EVAL ones). Such a clause, if it fires on at most 5% of them, has to drop words from its bracket-5 group, whose Jaccard is then at most 0.667, so it recovers at most 5 of the 6 brackets. E3 stated this as a ceiling set by the data; a clause with more groups is not bound by it (see the correction in the section on E3).

Check 2b of hierarchical_tm, with its scorer `slide10.check2b` unchanged: the best single rule, chosen on the TRAIN sentences, covers at least 90% of the held-back family sentences in the 288 view and in the 216 view, fires on at most 5% of each kind of negative, and recovers at least 4 of the 6 brackets (Jaccard at least 0.8, the fixed words of the template removed). The scorer prints one flag per view, `pass_288` and `pass_216`.

### E2, direct labels, DEV sentences (source: `logs/RUNS_slide10.md`)

Every arm has 20 clauses, K 12, T 8, mode near, r 2, initial state -1 and 20 passes unless named.

| arm | runs | both flags | flag 288 | coverage 288 | coverage 216 | highest firing on a kind of negative | brackets 288, seeds 1 / 2 / 3 / 4 / 5 |
|---|---|---|---|---|---|---|---|
| A0: reinforce all, reject random (the rule of E1) | 5 | 0 | 0 | 0.406 to 0.683 | 0.414 to 0.714 | 0.386 | 1 / 0 / 0 / 0 / 0 |
| A1: reinforce critical, reject weakest | 5 | 2 | 2 | 0.960 to 1.000 | 0.943 to 1.000 | 0.000 | 5 / 2 / 2 / 5 / 3 |
| A5b: A1 with decay 0.1 | 5 | 1 | 2 | 1.000 to 1.000 | 1.000 to 1.000 | 0.000 | 4 / 3 / 3 / 4 / 3 |
| A5b: A1 with decay 0.5 | 5 | 2 | 2 | 1.000 to 1.000 | 1.000 to 1.000 | 0.000 | 3 / 4 / 4 / 3 / 3 |
| A6: A1 with seed-all | 5 | 3 | 5 | 0.941 to 1.000 | 0.914 to 1.000 | 0.000 | 5 / 4 / 5 / 4 / 5 |
| A6d: A6 with decay 0.1 | 5 | 2 | 2 | 1.000 to 1.000 | 1.000 to 1.000 | 0.000 | 3 / 3 / 3 / 4 / 4 |
| A6 with 50 passes | 5 | 3 | 5 | 0.941 to 1.000 | 0.914 to 1.000 | 0.000 | 5 / 4 / 5 / 4 / 5 |
| A6 with K 16 | 5 | 1 | 4 | 0.931 to 1.000 | 0.900 to 1.000 | 0.050 | 4 / 4 / 4 / 4 / 3 |
| A6 with r 4 | 5 | 2 | 4 | 0.693 to 1.000 | 0.686 to 1.000 | 0.048 | 4 / 5 / 4 / 4 / 4 |
| A6 with K 8 | 5 | 2 | 2 | 1.000 to 1.000 | 1.000 to 1.000 | 0.103 | 5 / 5 / 4 / 4 / 4 |

The ten runs of arm A5 (A1 with decay, first code) are void: they ran while a unit test failed (the note INVALID in the log), and their numbers are not used. The go-bar of E2, both flags on at least 3 of 5 DEV seeds, is met by A6 only, at exactly 3 of 5; the 50-pass arm repeats the best rules of A6 and is not a second sample. Bracket 5 is recovered by no text run, and no run recovers all six brackets (publication note of the log).

### E2, pair version, DEV sentences (source: `logs/RUNS_slide10.md`)

Two sentences in, label 1 when both come from the same family: the procedure of the proposal applied to sentences.

| arm | runs | pair accuracy (DEV) | pair AUC (DEV) | coverage 288 of the best rule, per run | highest firing | most brackets |
|---|---|---|---|---|---|---|
| P1: 60 clauses, T 8, mode near, 20 passes | 3 | 0.8263 to 0.8515 | 0.9193 to 0.9271 | 0.000 / 0.000 / 0.000 | 0.000 | 0 |
| P2: as P1 with mode side | 3 | 0.8218 to 0.8293 | 0.9394 to 0.9477 | 0.000 / 0.000 / 0.000 | 0.000 | 0 |
| P3: as P1 with 300 passes (stopped after its first run) | 1 | 0.8448 to 0.8448 | 0.9284 to 0.9284 | 0.000 | 0.000 | 0 |
| P4: 100 clauses, T 2 or 3, 100 passes | 4 | 0.8791 to 0.8942 | 0.9296 to 0.9415 | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | 0 |
| P5: P4 with tied groups, T 2 or 8 | 4 | 0.9008 to 0.9173 | 0.9302 to 0.9708 | 0.337 / 0.168 / 0.693 / 0.000 | 0.369 | 0 |

No pair arm meets the go-bar, and the pair version has no registered run.

### E3, registered, EVAL sentences, direct labels (source: `logs/e3_report.json`)

E3 was fixed at 2026-11-12T17:48:56Z (record `registrations/E3_slide10_text.frozen.json`, plan SHA-256 `f97da6b4b9fd30824e6e1ff6fc168362f99983d05f562a941dad4194b8176c09`). Its job trained the registered seeds 11 to 15 and the further seeds 16 to 35 with the setting of arm A6, and the report was written at 2026-11-12T17:51:57Z after all 72 of its checks had passed, with no problem listed and no EVAL scoring outside the registered runs. The report prints the result in the wording that E3 fixed, with its two companion sentences:

> With direct family labels and a setting chosen on the DEV sentences, on 5 of 5 new seeds the best single clause of the model (picked by the study's scorer on the training sentences) accepts at least 90% of the 101 held-back sentences of the slide-10 template family (and of the 80 among them that are in the 216 view), fires on at most 5% of the negated, near-miss and broken sentences, and recovers at least 4 of the 6 brackets in the 288 reading; 5 of 5 seeds when both flags of the scorer are required.

> The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911).

> The pair version (the procedure of the proposal) met none of the bars on the DEV sentences.

Held-back sentences are combinations of the alternatives of the six brackets that are in neither the TRAIN nor the DEV sentences, while every alternative and every word of them occurs in the training family sentences (E3).

R1 (both flags of the scorer on at least 3 of the 5 registered seeds) is met; the report prints "R1 is met: both flags of the scorer on at least 3 of 5 seeds." The counts are never to be read without each other: both flags on 5 of 5 registered seeds (exact 95% interval of the share 0.478 to 1.000); on 20 further seeds of the same pass, which carry no rule, on 8 of 20 (0.191 to 0.639); pooled, on 13 of 25 (0.313 to 0.722); the 288 reading on 23 of 25 (0.740 to 0.990). All counts with exact 95% intervals in brackets:

| reading | registered seeds 11 to 15 | further seeds 16 to 35 | all 25 seeds |
|---|---|---|---|
| both flags of the scorer (the reading of R1) | 5 of 5 [0.478, 1.000] | 8 of 20 [0.191, 0.639] | 13 of 25 [0.313, 0.722] |
| the 288 reading (parts a and b, brackets in the 288 reading only) | 5 of 5 [0.478, 1.000] | 18 of 20 [0.683, 0.988] | 23 of 25 [0.740, 0.990] |
| flag 288 alone | 5 of 5 [0.478, 1.000] | 18 of 20 [0.683, 0.988] | 23 of 25 [0.740, 0.990] |
| flag 216 alone | 5 of 5 [0.478, 1.000] | 8 of 20 [0.191, 0.639] | 13 of 25 [0.313, 0.722] |
| coverage and firing parts (a) and (b) only | 5 of 5 [0.478, 1.000] | 19 of 20 [0.751, 0.999] | 24 of 25 [0.796, 0.999] |
| best rule template-like (at least 90% of the intact and at most 5% of the deletion texts) | 5 of 5 [0.478, 1.000] | 19 of 20 [0.751, 0.999] | 24 of 25 [0.796, 0.999] |
| whole model template-like (at least one of the 20 clauses fires) | 0 of 5 [0.000, 0.522] | 0 of 20 [0.000, 0.168] | 0 of 25 [0.000, 0.137] |

On 10 of the 12 seeds without both flags, the flag of the 288 view holds and the flag of the 216 view fails on its bracket count (3 of 6 brackets in the 216 reading); the other two fail both flags (seed 21: 3 brackets in the 288 reading, coverage 0.861; seed 25: 3 brackets in the 288 reading, coverage 1.000).

The registered seeds, one row each:

| seed | coverage 288 / 216 | firing negated / near-miss / broken | brackets 288 / 216 | flags 288 / 216 | best rule: deletion texts accepted | best rule: one-word texts accepted | whole model: deletion texts accepted | template-like clauses of 20 | clauses tied at the top TRAIN score | DEV plain accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.907 | 0.285 | 3 | 2 | 0.926 |
| 12 | 0.970 / 0.988 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.879 | 0.262 | 2 | 2 | 0.915 |
| 13 | 0.960 / 0.963 | 0.000 / 0.005 / 0.002 | 5 / 4 | True / True | 0.000 | 0.861 | 0.117 | 4 | 4 | 0.965 |
| 14 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True | 0.000 | 0.884 | 0.215 | 3 | 3 | 0.946 |
| 15 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.907 | 0.304 | 1 | 1 | 0.909 |

The further seeds:

| seed | coverage 288 / 216 | firing negated / near-miss / broken | brackets 288 / 216 | flags 288 / 216 | best rule: deletion texts accepted | best rule: one-word texts accepted | whole model: deletion texts accepted | template-like clauses of 20 | clauses tied at the top TRAIN score | DEV plain accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.970 / 0.963 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.884 | 0.178 | 5 | 2 | 0.956 |
| 17 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.907 | 0.170 | 5 | 5 | 0.958 |
| 18 | 0.921 / 0.900 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.830 | 0.196 | 3 | 3 | 0.952 |
| 19 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 5 | True / True | 0.000 | 0.870 | 0.177 | 4 | 2 | 0.951 |
| 20 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.907 | 0.239 | 4 | 4 | 0.932 |
| 21 | 0.861 / 0.850 | 0.000 / 0.014 / 0.001 | 3 / 2 | False / False | 0.005 | 0.772 | 0.224 | 0 | 1 | 0.942 |
| 22 | 0.970 / 0.963 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.886 | 0.149 | 3 | 3 | 0.961 |
| 23 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True | 0.000 | 0.907 | 0.277 | 4 | 3 | 0.935 |
| 24 | 0.980 / 0.975 | 0.000 / 0.014 / 0.007 | 5 / 4 | True / True | 0.002 | 0.889 | 0.376 | 2 | 1 | 0.891 |
| 25 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 2 | False / False | 0.000 | 0.907 | 0.269 | 3 | 3 | 0.928 |
| 26 | 0.960 / 0.950 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.877 | 0.249 | 3 | 2 | 0.933 |
| 27 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.907 | 0.315 | 3 | 2 | 0.920 |
| 28 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.907 | 0.190 | 3 | 3 | 0.959 |
| 29 | 0.960 / 0.975 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.865 | 0.168 | 4 | 4 | 0.960 |
| 30 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.907 | 0.257 | 1 | 1 | 0.928 |
| 31 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.907 | 0.257 | 2 | 2 | 0.922 |
| 32 | 0.960 / 0.950 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.879 | 0.236 | 3 | 2 | 0.931 |
| 33 | 0.921 / 0.938 | 0.000 / 0.000 / 0.000 | 4 / 5 | True / True | 0.000 | 0.794 | 0.282 | 2 | 1 | 0.908 |
| 34 | 0.970 / 0.963 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False | 0.000 | 0.888 | 0.195 | 2 | 1 | 0.930 |
| 35 | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 5 | True / True | 0.000 | 0.870 | 0.236 | 4 | 3 | 0.938 |

Ranges over the registered seeds (same file): coverage 0.96 to 1.0 in the 288 view and 0.963 to 1.0 in the 216 view, highest firing 0.0 to 0.005, the best rule accepts 0.0 of the deletion texts and 0.861 to 0.907 of the one-word texts, the whole model accepts 0.117 to 0.304 of the deletion texts, 1 to 4 of the 20 clauses are template-like by themselves, and 1 to 4 clauses tie at the top TRAIN score. The texts of the readings (E3): a deletion text is a family sentence with the words of one bracket's alternative deleted and nothing added (606 per split); a one-word text is a family sentence with one word of a multi-word alternative removed (569 on EVAL). The best-rule deletion share is a consistency check, not a second piece of evidence: for a clause of positive words a deletion text holds a subset of the words of a broken sentence, so the share follows from part (b). The one-word share measures the limit that a group is an OR of single words: a multi-word alternative is matched by any one of its words.

Over all 25 seeds (same file) the best rule accepts 0.772 to 0.907 of the one-word texts and 0.000 to 0.005 of the deletion texts, the whole model accepts 0.117 to 0.376 of the deletion texts, 1 to 5 clauses tie at the top TRAIN score, and the Jaccard of bracket 5 is at most 0.667 in the 288 reading, so bracket 5 is recovered on no seed.

E3 is the first use of the EVAL sentences by this study and the seventh completed scoring of them (E3, section on the data); this study does not score them again without a new registration that lists the numbers of E3.

The dry job (`registrations/E3_job.sh dry`: DEV sentences, its own throwaway plan, report `logs/e3_dry_report.json`) passed every check and gave both flags on 3 of 5 seeds of 1 to 5 (4 of 7 with seeds 6 and 7) and the 288 reading on 5 of 5 (7 of 7). By E3 its five models of seeds 1 to 5 are byte for byte the models of the design rows of A6.

#### The tie-break

The scorer takes the first of the clauses that tie at the top TRAIN score; E3 fixed this in advance and the runner asserts it. The number of tied clauses per seed is in the tables above. The independent verification (next section) counted the other choices: if any tied clause counted, both flags would hold on 5 of 5 registered and 12 of 20 further seeds (17 of 25); if every tied clause had to pass, on 3 of 5 and 4 of 20 (7 of 25); with the first tied clause, as registered, on 5 of 5 and 8 of 20 (13 of 25). Other tie choices therefore give 3 to 5 of 5 and 4 to 12 of 20.

#### Independent verification of E3 (2026-11-12, verdict MATCH; its record is not distributed)

- Separate code, using the standard library and numpy and importing nothing from this study, recomputed from the 25 stored models every number of `logs/e3_report.json`: 41 fields per seed for all 25 seeds, the counts and exact intervals of the three sets of seeds (21 intervals), the registered ranges, the four printed sentences and the 25 rows of E3 in `logs/RUNS_slide10.md`. There were 0 mismatches.
- The record was fixed before every run (the first started 1 second after the freeze), the hashes of the record hold for the files on the path of E3 and for the pinned files, each of the 25 seeds ran once, no model was scored on the EVAL sentences outside the 25 registered runs, and all 56 lines of `logs/e3/SHA256SUMS` hold.
- The pinned runner made the 25 models again on the DEV sentences only, and they came out identical (state arrays, include bits, the DEV history of every pass, the best rule and the tied rules).
- Post hoc diagnostics, not registered tests: with a share of 13 of 25, five of five comes up with probability 0.038; Fisher's exact test of 5 of 5 against 8 of 20 gives 0.024 one-sided and 0.039 two-sided.
- Further points: other jobs of this study ran in the folder around and during the registered job (DEV runs of the Wikipedia arm, and a run of the unit tests of the engine at 17:43Z), with no effect on any number of E3; the run folders carry no stored hash, a gap that the identical rerun closes; the console lines of the job were not saved.

#### The check of the frozen text

Draft 1 of E3 was checked from two sides (design and statistics; data, provenance and code path) and found not ready (E3, section on the draft history). Draft 2, the text that was frozen, was checked once more at 17:48:29Z, 27 seconds before the freeze, and not independently.

#### Correction to the text of E3 (written 2026-11-12T18:27:10Z; the registration stays as it was fixed)

E3 says, in its section on the data and in its limits, that the data allow at most 5 of the 6 brackets to be recovered. That holds only for a clause with one group per bracket (`logs/slide10_facts.json`). The independent verification found a hand-written clause of 8 groups, the clause with one group per bracket and one group for a fixed word of the template plus one blocking group of 4 words, that accepts 86 of 86 TRAIN and 101 of 101 DEV family sentences, fires on no TRAIN or DEV negative of any kind, and recovers 6 of 6 brackets in the 288 reading (5 of 6 in the 216 reading); the model of E3 has 12 groups. Bracket 5 is therefore not out of reach by the data: that no seed recovered it is a property of this learner. No number of E3 changes.

#### Files changed after the report

Files of the record that E3 does not use changed after the report: `src/run_wiki.py` and `src/wiki_report.py` were edited and the files of E5 were added. The files on the path of E3 and the pinned files are unchanged, but `src/e3_report.py` checks every file under `src/` and `tests/` against the record, so it can no longer pass that check on this tree, and the frozen report cannot be made again byte for byte from the files as they stand.

### Reference: hierarchical_tm, Check 2b

In hierarchical_tm (its docs/results.md, Check 2b) Check 2b was not met: 0 of 5 seeds in the pair version and 0 of 5 in the direct-label fallback. In the fallback the rule of the family fires on everything and the other class lists words that only the negatives contain (a negative-class word blacklist), and its deletion probe accepted 100% of the deletion texts on all five seeds. That model was shaped to the template and had one registered setting; the model of E3 is a different model whose setting was chosen on the DEV sentences. That verdict stands, and no comparison is made.

## Wikipedia pairs: E4 (design) and the DEV runs before E5

The data are the pair versions of hierarchical_tm, 1,024 words, 40,000 train and 20,000 dev pairs per version (E4): `sa` (same article or not), `np` (next paragraph or not) and `ns_all` (next paragraph with every shared word removed). Every AUC in this section is an AUC on the DEV pairs; the test pairs are read by E5 only.

### E4 design rows (source: `logs/RUNS_wiki.md`)

Every arm has 2,000 clauses of positive polarity, K 1, T 40, mode side, reject weakest, r 2 and 10 passes unless named.

| arm (seed 1; sa unless named) | run | AUC, vote sum | AUC, ratio reading | partner-shuffle AUC |
|---|---|---|---|---|
| first look: untied, reinforce critical, initial state -1 | `20261112T162238Z_e4_sa_first_s1` | 0.7497 | - | 0.4983 |
| untied, initial state -3 | `20261112T162602Z_e4_sa_init3_s1` | 0.7721 | - | 0.4985 |
| untied, reinforce share, initial state -3, weights to 64 (B) | `20261112T163646Z_e4_sa_B_w64_s1` | 0.7774 | 0.8081 (note of 16:51:28Z) | 0.5000 |
| B with tied groups (the base TB) | `20261112T164052Z_e4_sa_B_tied_w64_s1` | 0.8255 | 0.8456 (note of 16:51:28Z) | 0.5001 |
| B with half the clauses voting against | `20261112T164513Z_e4_sa_B_neg50_w64_s1` | 0.5047 | - | 0.5043 |
| TB, reinforce critical | `20261112T165739Z_e4_sa_TB_critical_s1` | 0.8243 | 0.8445 | 0.4971 |
| TB, initial state -1 | `20261112T170257Z_e4_sa_TB_init1_s1` | 0.8220 | 0.8422 | 0.4985 |
| TB, initial state -5 | `20261112T170816Z_e4_sa_TB_init5_s1` | 0.8257 | 0.8458 | 0.4962 |
| TB, T 100 | `20261112T171351Z_e4_sa_TB_T100_s1` | 0.8227 | 0.8515 | 0.4988 |
| TB, K 2 with seed-all | `20261112T172330Z_e4_sa_TB_K2_s1` | 0.8006 | 0.8368 | 0.5008 |
| TB, r 4 | `20261112T172848Z_e4_sa_TB_r4_s1` | 0.8224 | 0.8404 | 0.4950 |
| TB, 4,000 clauses, T 80 | `20261112T173913Z_e4_sa_TB_c4000_s1` | 0.8295 | 0.8517 | 0.4962 |
| B tied, 8,000 clauses, T 100 | `20261112T171101Z_e4_sa_B_tied_w64_c8000_s1` | 0.8304 | - | 0.4950 |
| TB on np | `20261112T164742Z_e4_np_TB_s1` | 0.8567 | 0.8787 (note of 16:51:28Z) | 0.4976 |
| TB on ns_all | `20261112T165216Z_e4_nsall_TB_s1` | 0.7023 | - | 0.4986 |

Tied groups are the one change from B to TB and raise the DEV AUC of the vote sum on `sa` from 0.7774 to 0.8255. The ratio reading was defined after the tied rows had been seen (note of 16:51:28Z), so it is a reading chosen on the DEV pairs. The arm with half of the clauses voting against was dropped.

### Arm TB on seeds 1 to 3 (source: the nine rows `e5dev_*` of `logs/RUNS_wiki.md`; references from E4)

| data | AUC, vote sum, seeds 1 / 2 / 3 (mean, from the note of 18:12:43Z) | AUC, ratio reading, seeds 1 / 2 / 3 (mean, same note) | partner-shuffle AUC | stock headline setting, DEV AUC seeds 1 / 2 / 3 (E4) | word overlap, DEV AUC (E4) |
|---|---|---|---|---|---|
| sa | 0.8255 / 0.8278 / 0.8283 (mean 0.8272) | 0.8456 / 0.8474 / 0.8481 (mean 0.8470) | 0.4995 to 0.5064 | 0.745 / 0.739 / 0.733 | 0.819 |
| np | 0.8567 / 0.8561 / 0.8574 (mean 0.8567) | 0.8787 / 0.8779 / 0.8793 (mean 0.8787) | 0.4976 to 0.4994 | 0.752 / 0.728 / 0.751 | 0.864 |
| ns_all | 0.7023 / 0.7053 / 0.7016 (mean 0.7031) | 0.7323 / 0.7349 / 0.7336 (mean 0.7336) | 0.4949 to 0.4986 | 0.622 / 0.622 / 0.611 | 0.500 by construction |

The note of 18:12:43Z in the same log applies the bars of E4 to these rows: G1 (reaching the stock model) is met in both readings; G2 (the go-bar for a registered test against word overlap: 0.830 on `sa`, 0.875 on `np`) is missed in the vote sum, the score that E4 named, and passed only in the ratio reading, a reading defined after the miss. Seed 1 of each version gives its design row back exactly with the engine now in `src/` (same note).

### Word groups of one design run (source: `logs/design/score_groups_sa_tied_s1_vs_stock.json`)

A group of this rule is the member set of one tied group of one clause; groups of fewer than 2 words are left out, repeated groups are kept. NPMI is computed over 2,847 reference articles that are in no split, and the WordNet score is the share of a group's word pairs that are synonyms or close hypernyms in WordNet 3.0; both are the measures of hierarchical_tm (`src/score_groups.py`).

| groups | groups of 2 or more words | mean words | NPMI | NPMI of random groups | difference | WordNet score | WordNet score of random groups | difference |
|---|---|---|---|---|---|---|---|---|
| this rule: `20261112T164052Z_e4_sa_B_tied_w64_s1` | 1802 | 2.97 | 0.218 | 0.078 | +0.140 [0.134, 0.147] | 0.109 | 0.038 | +0.071 [0.060, 0.083] |
| stock headline setting, sa seeds 1 to 3 (hierarchical_tm) | 198 | 63.21 | 0.120 | 0.088 | +0.032 [0.029, 0.034] | 0.065 | 0.052 | +0.013 [0.011, 0.016] |

Differences are to frequency-matched random groups, with 95% group-bootstrap intervals. Against the groups of the stock model, the groups of this run score +0.098 [0.091, 0.105] on NPMI and +0.044 [0.031, 0.056] on the WordNet score. This is one design run of one seed; groups of about 3 words and groups of about 63 words are not alike, and the scores measure topical coherence and WordNet relatedness, not synonym sets.

### The tied model against the baselines on the DEV pairs of sa (source: `logs/design/wiki_report_dev_sa_tied_s1_design.json`)

The vote sum of `20261112T164052Z_e4_sa_B_tied_w64_s1` (seed 1, engine `53cc7b4b...`) has a DEV AUC of 0.8255 [0.8181, 0.8327] over 20,000 pairs of 2,846 articles. Paired differences with the article bootstrap of the final test of hierarchical_tm:

| against | its DEV AUC | paired difference in AUC [95% interval] |
|---|---|---|
| word overlap (Jaccard) | 0.8191 | +0.0064 [+0.0004, +0.0121] |
| PPMI word-vector cosine | 0.8538 | -0.0284 [-0.0347, -0.0223] |
| logistic regression, the baseline best on dev in hierarchical_tm | 0.8713 | -0.0459 [-0.0526, -0.0393] |
| the stock model, mean of 5 seeds | 0.7368 | +0.0887 [+0.0820, +0.0952] |

This is a design comparison on DEV, made with the reading of the vote sum and before the ratio reading existed; it carries no rule.

## E5: the registered test on the Wikipedia test pairs

Source: `logs/e5_report.json` (all 199 checks passed). Fixed on 2026-11-12T21:05:33Z (`registrations/E5_wikipedia_test.md`). On the Wikipedia pairs the three rules of E5 are met, as predicted from DEV: on the test pairs of `sa`, `np` and `ns_all` the OR-group rule (arm TB, chosen on DEV) is above the stock model in its registered setting by +0.091, +0.101 and +0.098 AUC of the vote sum, each with both intervals above 0 (lower ends at least +0.080). These are the only rules of E5.

| test pairs (10,000 per version) | OR-group rule, arm TB, vote sum (mean AUC of 5 seeds) | stock model in its registered setting (mean AUC of its 5 seeds) | difference, paired article bootstrap (95%) | with the seeds resampled | rule |
|---|---|---|---|---|---|
| `sa` | 0.8255 | 0.7348 | +0.0907 [+0.0815, +0.0993] | [+0.0803, +0.1009] | R1-sa met, as predicted from DEV |
| `np` | 0.8519 | 0.7504 | +0.1015 [+0.0934, +0.1096] | [+0.0895, +0.1145] | R1-np met, as predicted from DEV |
| `ns_all` | 0.7114 | 0.6136 | +0.0978 [+0.0850, +0.1098] | [+0.0842, +0.1107] | R1-ns_all met, as predicted from DEV |

What this comparison is. The stock model is the setting that hierarchical_tm registered (its `s` and its number of passes were chosen by its rules on the DEV pairs of one seed). Its second setting (2,000 rules) was about 0.03 (`sa`) and 0.05 (`np`) higher on DEV (the reference table of E4) and was never scored on the test pairs, so on DEV the gap to it is about +0.06 and +0.07. Arm TB uses tied groups, which were not in the plan of E4. It is not a comparison at matched size or cost (the stock model: 50 rules of 64 options, 100 passes, another package, on a GPU; arm TB: 2,000 clauses, 10 passes of 17.2 to 21.0 s on one CPU thread), and "above the stock model" is about these two models as they were set. A met R1 was the expected outcome and is not news: the differences on DEV were +0.090 / +0.108 / +0.081 (seeds 1 to 3) and on the test pairs they are +0.091 / +0.101 / +0.098 (seeds 11 to 15). Each of the 5 registered seeds is above each of the 5 stock seeds in AUC of the vote sum; the smallest gap is +0.081 (`sa`), +0.094 (`np`) and +0.092 (`ns_all`) (arithmetic on the per-seed AUCs of the report; no rule).

The 16 sentences of the report, word for word (the wording that E5 fixed; in them "5 new seeds" means seeds not used in the design, see the correction below):

1. Same-article pairs (sa), 10,000 held-back test pairs: the mean AUC of 5 new seeds of the OR-group rule in the vote sum (the score E4 named) is 0.826, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is 0.735; the difference is +0.091 [+0.082, +0.099] (paired article bootstrap, 95%) and [+0.080, +0.101] with the seeds resampled, so R1-sa (above the stock model) is met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is 0.846; the ratio reading minus word overlap (AUC 0.817) is +0.030 [+0.023, +0.036] [+0.022, +0.037], and the vote sum minus word overlap is +0.009 [+0.002, +0.016] [+0.001, +0.016].
2. Next-paragraph pairs (np), 10,000 held-back test pairs: the mean AUC of 5 new seeds of the OR-group rule in the vote sum (the score E4 named) is 0.852, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is 0.750; the difference is +0.101 [+0.093, +0.110] (paired article bootstrap, 95%) and [+0.090, +0.114] with the seeds resampled, so R1-np (above the stock model) is met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is 0.873; the ratio reading minus word overlap (AUC 0.857) is +0.017 [+0.010, +0.023] [+0.010, +0.023], and the vote sum minus word overlap is -0.005 [-0.011, +0.002] [-0.012, +0.002].
3. Next-paragraph pairs with every shared word removed (ns_all), 10,000 held-back test pairs: the mean AUC of 5 new seeds of the OR-group rule in the vote sum (the score E4 named) is 0.711, and the mean AUC of the 5 seeds of the study's stock model (its registered setting) on the same pairs is 0.614; the difference is +0.098 [+0.085, +0.110] (paired article bootstrap, 95%) and [+0.084, +0.111] with the seeds resampled, so R1-ns_all (above the stock model) is met; predicted from DEV: met. In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is 0.742; the ratio reading minus the stock model is +0.128 [+0.116, +0.140] [+0.115, +0.141].
4. The ratio reading was defined on the DEV pairs at 16:51:28Z, after the tied rows of seed 1 were below the bar G2 of E4 in the vote sum that E4 named (sa 0.8255 against 0.830, np 0.8567 against 0.875). G2 is missed in the vote sum (means of seeds 1 to 3: 0.8272 and 0.8567) and passed only in the ratio reading (0.8470 and 0.8787); by E4, the comparison with word overlap carries no rule.
5. Readings, sa (test), the vote sum, mean AUC 0.826: minus word overlap (Jaccard), AUC 0.817: +0.009 [+0.002, +0.016] [+0.001, +0.016], interval above 0; minus tf-idf cosine, AUC 0.821: +0.004 [-0.003, +0.011] [-0.003, +0.011], interval includes 0; minus PPMI cosine, AUC 0.852: -0.026 [-0.034, -0.018] [-0.034, -0.018], interval below 0; minus the logistic regression on both paragraphs and their shared words (the baseline that was best on DEV in the study), AUC 0.869: -0.044 [-0.052, -0.036] [-0.052, -0.035], interval below 0; minus the stock model, AUC 0.735: +0.091 [+0.082, +0.099] [+0.080, +0.101], interval above 0. With the articles of paragraphs A and B resampled (seeds fixed): word overlap (Jaccard) [-0.001, +0.018], tf-idf cosine [-0.005, +0.013], PPMI cosine [-0.035, -0.016], the logistic regression on both paragraphs and their shared words [-0.054, -0.033], the stock model [+0.079, +0.104] (no rule).
6. Readings, sa (test), the ratio reading, mean AUC 0.846: minus word overlap (Jaccard), AUC 0.817: +0.030 [+0.023, +0.036] [+0.022, +0.037], interval above 0; minus tf-idf cosine, AUC 0.821: +0.025 [+0.019, +0.031] [+0.019, +0.032], interval above 0; minus PPMI cosine, AUC 0.852: -0.005 [-0.013, +0.002] [-0.013, +0.003], interval includes 0; minus the logistic regression on both paragraphs and their shared words (the baseline that was best on DEV in the study), AUC 0.869: -0.023 [-0.031, -0.015] [-0.031, -0.015], interval below 0; minus the stock model, AUC 0.735: +0.112 [+0.102, +0.121] [+0.101, +0.122], interval above 0. With the articles of paragraphs A and B resampled (seeds fixed): word overlap (Jaccard) [+0.021, +0.038], tf-idf cosine [+0.017, +0.033], PPMI cosine [-0.015, +0.005], the logistic regression on both paragraphs and their shared words [-0.033, -0.013], the stock model [+0.100, +0.124] (no rule).
7. Readings, np (test), the vote sum, mean AUC 0.852: minus word overlap (Jaccard), AUC 0.857: -0.005 [-0.011, +0.002] [-0.012, +0.002], interval includes 0; minus tf-idf cosine, AUC 0.862: -0.010 [-0.016, -0.004] [-0.017, -0.004], interval below 0; minus PPMI cosine, AUC 0.886: -0.034 [-0.040, -0.027] [-0.041, -0.026], interval below 0; minus the logistic regression on both paragraphs and their shared words (the baseline that was best on DEV in the study), AUC 0.903: -0.051 [-0.059, -0.044] [-0.059, -0.044], interval below 0; minus the stock model, AUC 0.750: +0.101 [+0.093, +0.110] [+0.090, +0.114], interval above 0. With the articles of paragraphs A and B resampled (seeds fixed): word overlap (Jaccard) [-0.013, +0.004], tf-idf cosine [-0.018, -0.002], PPMI cosine [-0.042, -0.025], the logistic regression on both paragraphs and their shared words [-0.060, -0.043], the stock model [+0.089, +0.112] (no rule).
8. Readings, np (test), the ratio reading, mean AUC 0.873: minus word overlap (Jaccard), AUC 0.857: +0.017 [+0.010, +0.023] [+0.010, +0.023], interval above 0; minus tf-idf cosine, AUC 0.862: +0.011 [+0.005, +0.017] [+0.005, +0.017], interval above 0; minus PPMI cosine, AUC 0.886: -0.012 [-0.019, -0.006] [-0.020, -0.005], interval below 0; minus the logistic regression on both paragraphs and their shared words (the baseline that was best on DEV in the study), AUC 0.903: -0.030 [-0.037, -0.023] [-0.037, -0.023], interval below 0; minus the stock model, AUC 0.750: +0.123 [+0.114, +0.131] [+0.111, +0.136], interval above 0. With the articles of paragraphs A and B resampled (seeds fixed): word overlap (Jaccard) [+0.009, +0.024], tf-idf cosine [+0.004, +0.018], PPMI cosine [-0.021, -0.004], the logistic regression on both paragraphs and their shared words [-0.038, -0.022], the stock model [+0.111, +0.134] (no rule).
9. Readings, ns_all (test), the vote sum, mean AUC 0.711: minus PPMI cosine (the baseline that was best on DEV in the study), AUC 0.811: -0.100 [-0.110, -0.090] [-0.110, -0.090], interval below 0; minus the stock model, AUC 0.614: +0.098 [+0.085, +0.110] [+0.084, +0.111], interval above 0. With the articles of paragraphs A and B resampled (seeds fixed): PPMI cosine [-0.113, -0.088], the stock model [+0.081, +0.113] (no rule).
10. Readings, ns_all (test), the ratio reading, mean AUC 0.742: minus PPMI cosine (the baseline that was best on DEV in the study), AUC 0.811: -0.069 [-0.079, -0.060] [-0.079, -0.060], interval below 0; minus the stock model, AUC 0.614: +0.128 [+0.116, +0.140] [+0.115, +0.141], interval above 0. With the articles of paragraphs A and B resampled (seeds fixed): PPMI cosine [-0.082, -0.058], the stock model [+0.112, +0.143] (no rule).
11. Without the 201 test pairs of sa that touch one of the 36 articles that are whole copies of a training article (9,799 pairs left), the vote sum minus the stock model is +0.091 [+0.082, +0.100] [+0.080, +0.101] and minus word overlap +0.009 [+0.001, +0.016] [+0.001, +0.016] (no rule).
12. Without the 1,234 test pairs of sa that touch one of the 242 articles with a paragraph equal to a training paragraph (8,766 pairs left), the vote sum minus the stock model is +0.091 [+0.081, +0.101] [+0.080, +0.102] and minus word overlap +0.006 [-0.003, +0.013] [-0.002, +0.014] (no rule).
13. Without the 205 test pairs of np that touch one of the 36 articles that are whole copies of a training article (9,795 pairs left), the vote sum minus the stock model is +0.101 [+0.093, +0.110] [+0.089, +0.114] and minus word overlap -0.004 [-0.010, +0.003] [-0.011, +0.003] (no rule).
14. Without the 1,226 test pairs of np that touch one of the 242 articles with a paragraph equal to a training paragraph (8,774 pairs left), the vote sum minus the stock model is +0.103 [+0.094, +0.112] [+0.091, +0.116] and minus word overlap -0.004 [-0.011, +0.003] [-0.011, +0.003] (no rule).
15. Without the 183 test pairs of ns_all that touch one of the 36 articles that are whole copies of a training article (9,817 pairs left), the vote sum minus the stock model is +0.097 [+0.085, +0.110] [+0.084, +0.110] (no rule).
16. Without the 1,261 test pairs of ns_all that touch one of the 242 articles with a paragraph equal to a training paragraph (8,739 pairs left), the vote sum minus the stock model is +0.101 [+0.089, +0.115] [+0.087, +0.115] (no rule).

Per seed (`logs/e5_report.json`, field `rows`): AUC on the 10,000 test pairs, scored only after the last pass, in the vote sum and in the ratio reading (defined on DEV; no rule), and the partner-shuffle control of the vote sum.

| data | seed | run folder (not distributed) | AUC, vote sum | AUC, ratio reading | partner-shuffle AUC | members | seconds per pass |
|---|---|---|---|---|---|---|---|
| `sa` | 11 | `20261112T210903Z_e5_sa_s11` | 0.8266 | 0.8474 | 0.5079 | 11,080 | 17.95 |
| `sa` | 12 | `20261112T211237Z_e5_sa_s12` | 0.8246 | 0.8455 | 0.5009 | 11,170 | 18.16 |
| `sa` | 13 | `20261112T211613Z_e5_sa_s13` | 0.8241 | 0.8454 | 0.5023 | 11,158 | 18.29 |
| `sa` | 14 | `20261112T211954Z_e5_sa_s14` | 0.8263 | 0.8470 | 0.5018 | 10,994 | 18.99 |
| `sa` | 15 | `20261112T212338Z_e5_sa_s15` | 0.8261 | 0.8469 | 0.4960 | 11,000 | 19.1 |
| `np` | 11 | `20261112T210935Z_e5_np_s11` | 0.8525 | 0.8737 | 0.4991 | 10,542 | 18.04 |
| `np` | 12 | `20261112T211307Z_e5_np_s12` | 0.8505 | 0.8715 | 0.4997 | 10,628 | 17.89 |
| `np` | 13 | `20261112T211631Z_e5_np_s13` | 0.8522 | 0.8730 | 0.4991 | 10,596 | 17.25 |
| `np` | 14 | `20261112T212016Z_e5_np_s14` | 0.8529 | 0.8746 | 0.5015 | 10,516 | 19.18 |
| `np` | 15 | `20261112T212353Z_e5_np_s15` | 0.8513 | 0.8726 | 0.4900 | 10,638 | 18.52 |
| `ns_all` | 11 | `20261112T211019Z_e5_ns_all_s11` | 0.7100 | 0.7403 | 0.5107 | 16,580 | 19.62 |
| `ns_all` | 12 | `20261112T211406Z_e5_ns_all_s12` | 0.7120 | 0.7423 | 0.4971 | 16,636 | 19.57 |
| `ns_all` | 13 | `20261112T211743Z_e5_ns_all_s13` | 0.7140 | 0.7450 | 0.4976 | 16,600 | 18.68 |
| `ns_all` | 14 | `20261112T212129Z_e5_ns_all_s14` | 0.7099 | 0.7403 | 0.5125 | 16,810 | 19.67 |
| `ns_all` | 15 | `20261112T212532Z_e5_ns_all_s15` | 0.7110 | 0.7402 | 0.4984 | 16,512 | 20.99 |

Stock seeds on the same pairs (the score files of the final test of hierarchical_tm): `sa` 0.7427 / 0.7366 / 0.7363 / 0.7315 / 0.7270; `np` 0.7552 / 0.7303 / 0.7564 / 0.7558 / 0.7543; `ns_all` 0.6108 / 0.6134 / 0.6090 / 0.6165 / 0.6183.

Word groups of the five registered `sa` models (field `groups`; not independently recomputed): 8,972 groups with at least two words (of 10,000), 2.97 words on average; NPMI over reference articles that are in no split 0.219 against 0.078 for random groups of words of the same frequencies. The stock model's 316 counted groups hold 63.5 words on average (0.120 against 0.089); the two sets are not matched in size.

Correction to the text of E5 (written 2026-11-12T22:56:45Z; the registration itself stays as it was fixed). Its sentence "Seeds 11 to 15 were never run" is wrong. The same 15 models had been trained once before the freeze (2026-11-12T20:20:40Z to 20:43:59Z) in a scratch rehearsal of the registered job and scored there on 10,000 DEV pairs per data version (vote sum minus the stock model on those DEV pairs: +0.087 [+0.078, +0.096], +0.106 [+0.097, +0.115] and +0.082 [+0.069, +0.094]). These numbers had been seen before the freeze, and the list "What I knew" of the registration lacks them. No test pair was read in the rehearsal; the registered runs gave the 15 models back bit for bit; seeds, code and setting were fixed before the first rehearsal number. "5 new seeds" in the sentences of the report therefore means seeds that were not used in the design (seeds 1 to 3). The rehearsal's record is not distributed.

How the text was checked. The first draft of E5 was checked from two sides and found not ready (1 blocker, 10 majors, 19 minors); the comparison with word overlap was taken out of the rules after that. The text that was frozen, draft 2, was checked only by its author, 44 seconds before the freeze, and had no independent check. None of these checks is a review by another person or by the supervisor. The job ran three one-thread streams at once, as the frozen text declares.

Verification of E5 (2026-11-12, verdict MATCH; its record is not distributed). Separate code gave 837 of the 1,098 numeric fields of the report back to the last digit, with no mismatch; the record was fixed at 21:05:33Z, 12 s before the first run started; every pinned file matches the record; no model of this study was scored on a test pair outside the 15 registered runs, as far as run folders and score files show. Not recomputed: the word-group scores. The verification read the test pairs again; it reproduced the registered numbers and scored no new model.

Uses of the test pairs. E5 is their second registered use and the first by this study: hierarchical_tm scored them in its one final test and read its saved outputs again on 2026-10-25. The verification of 2026-11-12 read them once more. Any further scoring needs a new registration that lists the numbers of E5.

## Synthetic sequences: E6 (exploratory; source: `logs/RUNS_seq.md`)

The task of the package's example `SequenceDemo.py` (E6): a sequence of 6 positions holds a permutation of 6 symbols, each symbol written with one of 3 interchangeable signs (108 input bits), and the label is the parity of the permutation; 10,000 training and 10,000 test sequences per data seed. Settings: (a) uniform orders and signs; (b) sign skew; (c) sign skew reversed at test; (d) order skew reversed at test. One group per position; the clauses vote for label 1. Bar: test accuracy of at least 0.95 on setting (a) on 3 of 3 data seeds.

| run | data seed, sign skew, order skew, drift | clauses, T, mode, min true, passes | test accuracy | label-1 covered / label-0 fired | clause precision | memoriser / logistic regression |
|---|---|---|---|---|---|---|
| `20261112T170454Z_e6_a_s1` | 1, 0, 0, none | 1000, 1, atleast, 2, 20 | 0.5156 | 0.3623 / 0.3331 | 0.5283 | 0.5133 / 0.4992 |
| `20261112T170553Z_e6_b_s1` | 1, 1, 0, none | 1000, 1, atleast, 2, 20 | 0.5194 | 0.3416 / 0.3050 | 0.5260 | 0.5351 / 0.5014 |
| `20261112T170651Z_e6_c_s1` | 1, 1, 0, sign drift | 1000, 1, atleast, 2, 20 | 0.5035 | 0.4376 / 0.4314 | 0.5087 | 0.5062 / 0.5011 |
| `20261112T170754Z_e6_d_s1` | 1, 0, 1, order drift | 1000, 1, atleast, 2, 20 | 0.4923 | 0.5201 / 0.5396 | 0.5202 | 0.4682 / 0.5172 |
| `20261112T170901Z_e6_a_near_s1` | 1, 0, 0, none | 1000, 1, near, 2, 20 | 0.5097 | 0.0131 / 0.0000 | 1.0000 | 0.5133 / 0.4992 |
| `20261112T170952Z_e6_a_m3_s1` | 1, 0, 0, none | 1000, 1, atleast, 3, 20 | 0.5074 | 0.4074 / 0.3939 | 0.5005 | 0.5133 / 0.4992 |
| `20261112T171028Z_e6_a_m4_s1` | 1, 0, 0, none | 1000, 1, atleast, 4, 20 | 0.7971 | 0.5916 / 0.0000 | 1.0000 | 0.5133 / 0.4992 |
| `20261112T171104Z_e6_a_m5_s1` | 1, 0, 0, none | 1000, 1, atleast, 5, 20 | 0.5097 | 0.0131 / 0.0000 | 1.0000 | 0.5133 / 0.4992 |
| `20261112T171302Z_e6_a_m4_s2` | 2, 0, 0, none | 1000, 1, atleast, 4, 20 | 0.7865 | 0.5721 / 0.0000 | 1.0000 | 0.5090 / 0.4969 |
| `20261112T171348Z_e6_a_m4_s3` | 3, 0, 0, none | 1000, 1, atleast, 4, 20 | 0.7983 | 0.5899 / 0.0000 | 1.0000 | 0.5164 / 0.5034 |
| `20261112T171840Z_e6_a_m4_c3000_p60_s1` | 1, 0, 0, none | 3000, 1, atleast, 4, 60 | 0.6223 | 0.2397 / 0.0000 | 1.0000 | 0.5133 / 0.4992 |
| `20261112T171928Z_e6_b_m4_s1` | 1, 1, 0, none | 1000, 1, atleast, 4, 20 | 0.8322 | 0.6622 / 0.0000 | 1.0000 | 0.5351 / 0.5014 |
| `20261112T172015Z_e6_c_m4_s1` | 1, 1, 0, sign drift | 1000, 1, atleast, 4, 20 | 0.7365 | 0.4696 / 0.0000 | 1.0000 | 0.5062 / 0.5011 |
| `20261112T172105Z_e6_d_m4_s1` | 1, 0, 1, order drift | 1000, 1, atleast, 4, 20 | 0.4805 | 0.0281 / 0.0000 | 1.0000 | 0.4682 / 0.5172 |
| `20261112T172559Z_e6_a_m4_c3000_p60_s2` | 2, 0, 0, none | 3000, 1, atleast, 4, 60 | 0.6243 | 0.2469 / 0.0000 | 1.0000 | 0.5090 / 0.4969 |
| `20261112T173050Z_e6_a_m4_c3000_p60_s3` | 3, 0, 0, none | 3000, 1, atleast, 4, 60 | 0.6243 | 0.2361 / 0.0000 | 1.0000 | 0.5164 / 0.5034 |
| `20261112T173543Z_e6_c_m4_c3000_p60_s1` | 1, 1, 0, sign drift | 3000, 1, atleast, 4, 60 | 0.5781 | 0.1508 / 0.0000 | 1.0000 | 0.5062 / 0.5011 |

No arm meets the bar. With min true 4 on setting (a) the clauses fire on no test sequence of label 0 and cover 0.5721 to 0.5916 of those of label 1 (test accuracy 0.7971 / 0.7865 / 0.7983); min true 4 was chosen from the structure of the task (note of 17:08:25Z). Under order drift, setting (d), the accuracy is at chance, and 3,000 clauses with 60 passes give a lower accuracy than 1,000 clauses with 20. The memoriser and the logistic regression lie between 0.4682 and 0.5351 in every row.

## Tests of the code

`logs/e3/test_orgroup_tm.out` (the unit tests of the engine and the scores, `ALL 23 TESTS PASSED`) and `logs/e3/test_e3_units.out` (the guards of E3, `ALL 11 TESTS PASSED`) were written by the registered job just before the freeze. The pipeline test `tests/test_e3_pipeline.py` passed its 37 cases in a run that started at 17:36:10Z; its output lies in `logs/tests/`, which is not distributed, and E3 quotes its SHA-256 (`47e06efa...`).

## Files copied from the working folder

| file | SHA-256 when this page was written | what it is |
|---|---|---|
| `logs/e0_diag_allocation.out` | `36cd84e6bd60007d2a1b7a5c09efd32e9b46c4e331fc65b835b3e610e8e4a507` | byte copy; cited by E0b |
| `logs/e0b_diag_allocation.out` | `b387a29ae8fddf4edc34be208bfb58737a7a0b9e729e40739929ae3cd9af1e72` | byte copy |
| `logs/design/wiki_report_dev_sa_tied_s1_design.json` | `d80616560734c4f328d9c9f25b45ffe58e6b8dfe4e45032383d0bb9e26d089f4` | byte copy of the DEV design report of 16:45Z |
| `verification/recompute_e1.out` | `8d6b1ab22b21712a1e596998c3af47cc1f7a702c788a4177ff020a7d0bac2dd0` | the output of 16:30Z with a publication note appended (the SHA-256 of the 13 lines of the output is in the note) |
| `logs/RUNS.md` | `04fec4829cc17bf067c33caf3ed6decb4a8742feefbd3d01703c282ba46882d0` | rows of E0 to E1 copied unchanged, with a publication note |
| `logs/RUNS_seq.md` | `9f47f0355b5f7ad9f92252dac78d95e5116254cbdcb5d68afbe337a509ac16e9` | rows and notes of E6 copied unchanged, with a publication note |
| `registrations/E6_sequence_demo.md` | `1b1bd645659f098b28b138454827205082e850550c30d990d0803813e294886a` | the fixed plan of E6 with a publication note |
