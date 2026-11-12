# E0b: amendment to E0 after its miss: at most one clause is seeded per example (post hoc revision, design data)

Publication note: the text below is the plan as it was fixed on 2026-11-12T15:56:18Z (SHA-256 of that file: `39ed24ad3b000364aaaa22a4eeb433f7c57782c07b1fe83848e9a582f440a4ad`), with folder and file names changed to those of this repository. Rules, bars, settings and numbers are unchanged.

Written after the E0 runs, before any run with the changed rule. Exploratory, design data only.

## What E0 showed (read from `logs/RUNS.md` and `logs/e0_diag_allocation.out`, sha256 36cd84e6...)

- `near` and `side` (identical at K = 1): every group of every seed is pure (purity 1.0; stock rule 0.47 to 0.68 at L2 and about 0.5 at L3), but plain dev accuracy is 0.752 to 0.815 at L2 and 0.682 to 0.751 at L3, and group recovery 0.5 to 0.625 and 0.125 to 0.5. 100 passes give the same numbers as 5. `all` gives mixed groups (purity 0.37 to 0.50) and is dropped.
- The E0 bar (accuracy at least 0.90 and recovery at least 0.80 on 2 of 3 seeds) is NOT met by any variant.
- Diagnostic on the saved states: 3 to 5 of the 8 hidden groups have no clause at all (for example 5, 20, 1, 13, 0, 1, 0, 0 clauses per hidden group at L2 seed 1), no clause is mixed, no clause is blank. Every blank clause that passed the gate was seeded by the same early examples, so the 40 clauses were used up by the first few hidden groups.

## Departure from E0's stop rule, stated openly

E0 said: if no gating variant meets the L2 bar, the rule is a dead end and the line stops. The miss is traced to how blank clauses are handed out, not to the group feedback the test was about (purity 1.0, no mixed clause). I therefore allow ONE revision, this one, and keep E0's stop rule for it: if E0b misses the L2 bar, the line stops.

## The one change

At most one SEED event per example (`max_seed = 1`): the first clause that passes the gate and whose false groups are all empty is seeded; the other such clauses wait for later examples. Everything else is as in E0: 40 clauses, positive polarity, K = 1, T = 4, initial state -1, seed state 0, r = 2, all probabilities 1.0, mode `near`.

## Runs, fixed now

L2 and L3, seeds 1 to 3, mode `near`, `--max-seed 1`, 5 passes and 100 passes (12 runs). The bar is E0's, unchanged, read at 100 passes or fewer; both pass counts are reported.

## What a miss means

- L2 bar missed: dead end, the line stops (no further revision).
- L2 met, L3 missed: reported as "L2 only"; no further revision of the rule inside E0b.
- Both met: E1 (new world seeds, frozen first) confirms; only then the slide-10 check and the Wikipedia pairs.

## Label

Every E0b number is post hoc relative to E0: the rule was changed after E0's results on these same files were seen.
