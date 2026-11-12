# Runs of this study (every run, newest last; times UTC from the clock of the run)

Rows are never edited or removed; a note stands below the rows it concerns. The rows above the publication note further down were made on 2026-11-12 in a working folder outside this repository, before the code was moved here (plans `registrations/E0_planted_feasibility.md`, `registrations/E0b_one_seed_per_example.md` and `registrations/E1_planted_confirmation.md`); their run folders are not distributed. The columns give the plain accuracy (vote sum at least 1), the tuned accuracy and the AUC of the vote sum on the dev pairs, group recovery by the `pcp_world.recovery` of hierarchical_tm, purity and coverage of all groups, and, at L3, the plain accuracy on the unseen word pairs.

| run folder | rung, seed | mode, clauses, K, T, init, r, passes (+ options) | dev plain / tuned / AUC | recovery | purity / coverage | unseen plain |
|---|---|---|---|---|---|---|
| 20261112T155338Z_e0_L2_near_s1 | L2, 1 | near, 40, 1, 4, -1, 2, 5 | 0.8150 / 0.8150 / 0.8150 | 0.625 | 1.0 / 0.6522 |  |
| 20261112T155339Z_e0_L2_near_s2 | L2, 2 | near, 40, 1, 4, -1, 2, 5 | 0.8047 / 0.8047 / 0.8047 | 0.625 | 1.0 / 0.7127 |  |
| 20261112T155341Z_e0_L2_near_s3 | L2, 3 | near, 40, 1, 4, -1, 2, 5 | 0.7520 / 0.7520 / 0.7520 | 0.500 | 1.0 / 0.6802 |  |
| 20261112T155342Z_e0_L2_side_s1 | L2, 1 | side, 40, 1, 4, -1, 2, 5 | 0.8150 / 0.8150 / 0.8150 | 0.625 | 1.0 / 0.6522 |  |
| 20261112T155344Z_e0_L2_side_s2 | L2, 2 | side, 40, 1, 4, -1, 2, 5 | 0.8047 / 0.8047 / 0.8047 | 0.625 | 1.0 / 0.7127 |  |
| 20261112T155345Z_e0_L2_side_s3 | L2, 3 | side, 40, 1, 4, -1, 2, 5 | 0.7520 / 0.7520 / 0.7520 | 0.500 | 1.0 / 0.6802 |  |
| 20261112T155346Z_e0_L2_all_s1 | L2, 1 | all, 40, 1, 4, -1, 2, 5 | 0.6455 / 0.6455 / 0.6503 | 0.000 | 0.4656 / 0.33 |  |
| 20261112T155348Z_e0_L2_all_s2 | L2, 2 | all, 40, 1, 4, -1, 2, 5 | 0.6815 / 0.6815 / 0.6884 | 0.000 | 0.4848 / 0.3157 |  |
| 20261112T155349Z_e0_L2_all_s3 | L2, 3 | all, 40, 1, 4, -1, 2, 5 | 0.7000 / 0.7000 / 0.7045 | 0.000 | 0.5039 / 0.3253 |  |
| 20261112T155351Z_e0_L3_near_s1 | L3, 1 | near, 40, 1, 4, -1, 2, 5 | 0.7512 / 0.7512 / 0.7512 | 0.500 | 1.0 / 0.5375 | 0.7455 |
| 20261112T155353Z_e0_L3_near_s2 | L3, 2 | near, 40, 1, 4, -1, 2, 5 | 0.6817 / 0.6817 / 0.6817 | 0.125 | 1.0 / 0.4984 | 0.6845 |
| 20261112T155354Z_e0_L3_near_s3 | L3, 3 | near, 40, 1, 4, -1, 2, 5 | 0.7465 / 0.7465 / 0.7465 | 0.250 | 1.0 / 0.4938 | 0.7522 |
| 20261112T155356Z_e0_L3_side_s1 | L3, 1 | side, 40, 1, 4, -1, 2, 5 | 0.7512 / 0.7512 / 0.7512 | 0.500 | 1.0 / 0.5375 | 0.7455 |
| 20261112T155358Z_e0_L3_side_s2 | L3, 2 | side, 40, 1, 4, -1, 2, 5 | 0.6817 / 0.6817 / 0.6817 | 0.125 | 1.0 / 0.4984 | 0.6845 |
| 20261112T155359Z_e0_L3_side_s3 | L3, 3 | side, 40, 1, 4, -1, 2, 5 | 0.7465 / 0.7465 / 0.7465 | 0.250 | 1.0 / 0.4938 | 0.7522 |
| 20261112T155401Z_e0_L3_all_s1 | L3, 1 | all, 40, 1, 4, -1, 2, 5 | 0.6245 / 0.6245 / 0.6428 | 0.000 | 0.3717 / 0.1812 | 0.5825 |
| 20261112T155403Z_e0_L3_all_s2 | L3, 2 | all, 40, 1, 4, -1, 2, 5 | 0.6492 / 0.6492 / 0.6622 | 0.000 | 0.4022 / 0.1906 | 0.5787 |
| 20261112T155404Z_e0_L3_all_s3 | L3, 3 | all, 40, 1, 4, -1, 2, 5 | 0.6302 / 0.6302 / 0.6423 | 0.000 | 0.3703 / 0.175 | 0.5730 |
| 20261112T155450Z_e0_L2_near_p100_s1 | L2, 1 | near, 40, 1, 4, -1, 2, 100 | 0.8150 / 0.8150 / 0.8150 | 0.625 | 1.0 / 0.6522 |  |
| 20261112T155454Z_e0_L2_near_p100_s2 | L2, 2 | near, 40, 1, 4, -1, 2, 100 | 0.8047 / 0.8047 / 0.8047 | 0.625 | 1.0 / 0.7127 |  |
| 20261112T155458Z_e0_L2_near_p100_s3 | L2, 3 | near, 40, 1, 4, -1, 2, 100 | 0.7520 / 0.7520 / 0.7520 | 0.500 | 1.0 / 0.6802 |  |
| 20261112T155503Z_e0_L3_near_p100_s1 | L3, 1 | near, 40, 1, 4, -1, 2, 100 | 0.7512 / 0.7512 / 0.7512 | 0.500 | 1.0 / 0.543 | 0.7455 |
| 20261112T155508Z_e0_L3_near_p100_s2 | L3, 2 | near, 40, 1, 4, -1, 2, 100 | 0.6817 / 0.6817 / 0.6817 | 0.125 | 1.0 / 0.4984 | 0.6845 |
| 20261112T155513Z_e0_L3_near_p100_s3 | L3, 3 | near, 40, 1, 4, -1, 2, 100 | 0.7465 / 0.7465 / 0.7465 | 0.250 | 1.0 / 0.4953 | 0.7522 |
| 20261112T155619Z_e0b_L2_near_ms1_p5_s1 | L2, 1 | near, 40, 1, 4, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.875 |  |
| 20261112T155621Z_e0b_L2_near_ms1_p5_s2 | L2, 2 | near, 40, 1, 4, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.8877 |  |
| 20261112T155622Z_e0b_L2_near_ms1_p5_s3 | L2, 3 | near, 40, 1, 4, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.85 |  |
| 20261112T155627Z_e0b_L2_near_ms1_p100_s1 | L2, 1 | near, 40, 1, 4, -1, 2, 100 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.875 |  |
| 20261112T155631Z_e0b_L2_near_ms1_p100_s2 | L2, 2 | near, 40, 1, 4, -1, 2, 100 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.8877 |  |
| 20261112T155635Z_e0b_L2_near_ms1_p100_s3 | L2, 3 | near, 40, 1, 4, -1, 2, 100 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.85 |  |
| 20261112T155637Z_e0b_L3_near_ms1_p5_s1 | L3, 1 | near, 40, 1, 4, -1, 2, 5 | 0.9998 / 0.9998 / 0.9998 | 1.000 | 1.0 / 0.7391 | 0.9998 |
| 20261112T155638Z_e0b_L3_near_ms1_p5_s2 | L3, 2 | near, 40, 1, 4, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.7398 | 1.0000 |
| 20261112T155640Z_e0b_L3_near_ms1_p5_s3 | L3, 3 | near, 40, 1, 4, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 0.875 | 1.0 / 0.7375 | 1.0000 |
| 20261112T155645Z_e0b_L3_near_ms1_p100_s1 | L3, 1 | near, 40, 1, 4, -1, 2, 100 | 0.9998 / 0.9998 / 0.9998 | 1.000 | 1.0 / 0.7477 | 0.9998 |
| 20261112T155649Z_e0b_L3_near_ms1_p100_s2 | L3, 2 | near, 40, 1, 4, -1, 2, 100 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.7477 | 1.0000 |
| 20261112T155654Z_e0b_L3_near_ms1_p100_s3 | L3, 3 | near, 40, 1, 4, -1, 2, 100 | 1.0000 / 1.0000 / 1.0000 | 0.875 | 1.0 / 0.743 | 1.0000 |
| 20261112T155733Z_design_L2_near_ms1_T8_p5_s1 | L2, 1 | near, 40, 1, 8, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9797 |  |
| 20261112T155734Z_design_L2_near_ms1_T8_p5_s2 | L2, 2 | near, 40, 1, 8, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9873 |  |
| 20261112T155735Z_design_L2_near_ms1_T8_p5_s3 | L2, 3 | near, 40, 1, 8, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9719 |  |
| 20261112T155737Z_design_L3_near_ms1_T8_p5_s1 | L3, 1 | near, 40, 1, 8, -1, 2, 5 | 0.9998 / 0.9998 / 0.9998 | 1.000 | 1.0 / 0.9102 | 0.9998 |
| 20261112T155739Z_design_L3_near_ms1_T8_p5_s2 | L3, 2 | near, 40, 1, 8, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9242 | 0.9990 |
| 20261112T155740Z_design_L3_near_ms1_T8_p5_s3 | L3, 3 | near, 40, 1, 8, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.8586 | 1.0000 |
| 20261112T155742Z_design_L2_near_ms1_T16_p5_s1 | L2, 1 | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155743Z_design_L2_near_ms1_T16_p5_s2 | L2, 2 | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155745Z_design_L2_near_ms1_T16_p5_s3 | L2, 3 | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155746Z_design_L3_near_ms1_T16_p5_s1 | L3, 1 | near, 40, 1, 16, -1, 2, 5 | 0.9998 / 0.9998 / 0.9998 | 1.000 | 1.0 / 0.9648 | 0.9998 |
| 20261112T155748Z_design_L3_near_ms1_T16_p5_s2 | L3, 2 | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9609 | 0.9990 |
| 20261112T155750Z_design_L3_near_ms1_T16_p5_s3 | L3, 3 | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9555 | 1.0000 |
| 20261112T155828Z_e1_L2_s11 | L2, 11 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155829Z_e1_L2_s12 | L2, 12 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155831Z_e1_L2_s13 | L2, 13 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155832Z_e1_L2_s14 | L2, 14 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155834Z_e1_L2_s15 | L2, 15 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T155836Z_e1_L3_s11 | L3, 11 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9586 | 0.9992 |
| 20261112T155839Z_e1_L3_s12 | L3, 12 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9594 | 1.0000 |
| 20261112T155842Z_e1_L3_s13 | L3, 13 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9617 | 1.0000 |
| 20261112T155844Z_e1_L3_s14 | L3, 14 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9641 | 1.0000 |
| 20261112T155847Z_e1_L3_s15 | L3, 15 fresh | near, 40, 1, 16, -1, 2, 5 | 0.9998 / 0.9998 / 0.9998 | 1.000 | 1.0 / 0.9695 | 0.9990 |

Publication note (2026-11-12T18:10:59Z): the rows above are the runs of E0, E0b, the design arms that followed E0b, and E1, copied from the working folder; the log there had no notes. Only its title line was changed (it named the working folder), and the paragraph under the title was added. Added now:

- Which rows belong to which plan. The 24 rows from 15:53:38Z to 15:55:13Z are E0: the first look (18 runs: 5 passes, the gating modes near, side and all) and then mode near with 100 passes (6 runs). The 12 rows from 15:56:19Z to 15:56:54Z are E0b. The 12 rows named `design_*` from 15:57:33Z to 15:57:50Z are the design arms T 8 and T 16 on the files of E0b; no fixed plan lists them, and E1 reports them in its section on what was known at its freeze, where T 16 is chosen for E1 on their numbers. The 10 rows named `e1_*` from 15:58:28Z to 15:58:47Z are E1, on worlds of the new seeds 11 to 15 (`fresh` in the column rung, seed). Each plan was fixed before its first row: E0 at 15:53:26Z, E0b at 15:56:18Z and E1 at 15:58:27Z (`registrations/INDEX.md`).
- The rows from E0b on do not show the one change of E0b, at most one seed event per example: the row formatter leaves out options whose value is 1. The names of the E0b runs and of the design arms carry `ms1`, and the configuration of each of these 34 runs, the E1 runs included, holds `max_seed` 1; the configurations of the 24 runs of E0 have no such entry (the option did not exist yet). `verification/recompute_e1.py` checks this value, with the rest of the setting, in each E1 run before it recomputes.
- Code. The configurations name the engine file `src/orgroup_tm.py` with SHA-256 `75f686c8...` for the 24 runs of E0 (the file pinned by the record of E0) and `6b6e075f...` for the other 34 (the file pinned by the records of E0b and E1). The engine in `src/` now is a later revision (`b94db364...`, the file pinned by E3), and the runner `src/run_pcp.py` was changed after E1 as well (the record of E1 pins `5adb6626...`; the file in `src/` has `8a6b1f29...`). The E1 setting has not been run with the engine now in `src/`. One E1 model (L3, seed 11) was made again in this folder at 16:54Z with the revision `32062302...` of the engine, as a check of the moved code, and came out equal element by element; that record lies in `logs/tests/`, which is not distributed. All ten E1 models were trained again by the separate plain-Python implementation of `verification/recompute_e1.py`, which gave every stored state back element by element and every reported number with it (`verification/recompute_e1.out`).
- `logs/e0_diag_allocation.out` (SHA-256 `36cd84e6bd60007d2a1b7a5c09efd32e9b46c4e331fc65b835b3e610e8e4a507`, the file that E0b cites) and `logs/e0b_diag_allocation.out` (SHA-256 `b387a29ae8fddf4edc34be208bfb58737a7a0b9e729e40739929ae3cd9af1e72`) are byte copies of the two outputs of `src/diag_allocation.py` in the working folder: the clauses per hidden group of the 12 runs of mode near of E0 and of the 6 runs of E0b with 100 passes.

Rows below this note are written by `src/run_pcp.py` in this folder.
| 20261112T182813Z_e1check_L2_s11 | L2, 11 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T182814Z_e1check_L2_s12 | L2, 12 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T182816Z_e1check_L2_s13 | L2, 13 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T182817Z_e1check_L2_s14 | L2, 14 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T182819Z_e1check_L2_s15 | L2, 15 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 1.0 |  |
| 20261112T182822Z_e1check_L3_s11 | L3, 11 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9586 | 0.9992 |
| 20261112T182824Z_e1check_L3_s12 | L3, 12 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9594 | 1.0000 |
| 20261112T182827Z_e1check_L3_s13 | L3, 13 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9617 | 1.0000 |
| 20261112T182830Z_e1check_L3_s14 | L3, 14 fresh | near, 40, 1, 16, -1, 2, 5 | 1.0000 / 1.0000 / 1.0000 | 1.000 | 1.0 / 0.9641 | 1.0000 |
| 20261112T182832Z_e1check_L3_s15 | L3, 15 fresh | near, 40, 1, 16, -1, 2, 5 | 0.9998 / 0.9998 / 0.9998 | 1.000 | 1.0 / 0.9695 | 0.9990 |

Note (2026-11-12T18:28:40Z): regression check of E1 with the engine file that is now in `src/` (sha256 b94db364700f41125dc9c1a12577e91a597c6d5476924330ca7d33b2e728999e): the ten runs `e1check_*` above repeat the E1 command of `registrations/E1_planted_confirmation.md` on the same fresh worlds and seeds. Every number of their rows equals the E1 row of the same rung and seed (accuracies, recovery, purity and coverage, unseen-pair accuracy). They add no new evidence; they show that E1 holds for the published code.
