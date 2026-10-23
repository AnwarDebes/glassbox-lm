# R4b registration: the TM router against logistic regression, with more clauses

Companion to R4 (`R4_router_vs_lr.md`, frozen 2026-10-15T22:21:31Z). Written before any confirmatory run of R4 or R4b. Frozen by `scripts/freeze.py`; amendments in a dated file.

## Why

R4 chooses its TM setting among candidates of at most 8,000 clauses (a cost cap set at freezing). An exploratory run outside that list, `e5_c16k` (16,000 clauses, T 16,000, CLINC150 MiniLM, seed 1), reached val 95.63. The 4,000-clause default reached 94.87, R4's best candidate `e1_thermo8_e80` 95.13, and logistic regression 96.10 on the raw embedding (95.77 on the same bits). Under the CPU load of this machine, its wall time was no larger than that of the 4,000-clause runs. R4b asks the R4 question again with more clauses. R4 stays as frozen and is run and reported as frozen.

## Design

The same cells, LR arm, seeds, splits, statistic and scoring as R4. These parts differ:

- **TM-single setting:** chosen by a rule fixed now. Among the exploratory CLINC150 MiniLM runs `e5_c16k`, `e14_c32k` and `e14_c16k_e80` (seed 1, val), take the one with the highest val accuracy. Ties within 0.10 points go to the fewer clauses × epochs. T equals the number of clauses, as in all runs; the other settings are the defaults: s 8, thermo:8, batch 128, hardest negatives.
- **TM-composite:** as in R4, two disjoint composites of seeds 1 to 5 and 6 to 10.
- **No TM-readout arm in R4b:** R4 covers the read-out.
- **Runs:** `scripts/confirm_router.py --registration R4b`. LR is recomputed in the R4b folder; it is deterministic, so it gives the same model as R4.

## Pass rules (test)

- **R4b.1 (main):** TM-composite minus LR is at least -0.5 points in at least 5 of the 6 cells.
- **R4b.2:** TM-single minus LR, reported per cell with its interval. "Gap closed" is said if it is at least -0.5 in at least 5 of the 6 cells.
- **R4b.3:** an arm whose interval minus LR lies entirely above 0 in a cell is reported as ahead in that cell.
- **R4b.4 (calibration and out of scope):** the rules R1.1 and R1.2 of R4, applied to the R4b TM-single.
