# Note e72e (2026-10-19): e70's checks and guards on the tuned ABCD agent (R23's five tm_mem_c4_s16 models)

Decided at 20:55 UTC. The guard line of the headline should hold for the agent in the final
comparison, R23's C4 arm at s 16. This is e72d's method unchanged, on the R23 models. No rule. Dev and proof work
only: no test split is read. Frozen with `scripts/freeze.py` before the first property run. The encoding pre-check may
run before the freeze, as in e70 and e72d.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## What is unchanged

Everything but the models:
- the 50 checks;
- the exact MILP encoding and the HiGHS options (scipy 1.17.1, HiGHS 1.12.0, tolerances 1e-6 / 1e-7, seed 0, 600 s);
- the level-B domain rows, built from the models' bit names. The bits are the same as R21's view B on the C4 table:
  only s differs, a training parameter;
- the guards, with M recomputed per model from its own weights.

The script is `scripts/verify_policy_c4_e72e.py`, a thin wrapper. It points the unedited `scripts/verify_policy_c4.py`
at `results/confirm/R23abcd/tm_mem_c4_s16` and writes to `results/explore/verify_e72e.jsonl`, tag e72e.

## Steps (one process, one thread, next to e71 seed 3)

1. **Pre-check** (`--check encoding`). It must give, for all 5 models:
   - votes equal on 500 dev rows;
   - 50 of 50 winner rows right;
   - every train, dev and closed-loop dev state inside the domain.
2. **The raw models** (`--check properties`): 50 checks x 5 seeds.
3. **The guarded models** (`--check properties --guard`): 50 x 5.
4. **The guards' cost on dev** (`--check guardcost`): open and closed loop.

## Expected (from e70 and e72d; a guess, not a rule)

- raw: counterexamples for all or nearly all checks;
- guarded: 50 of 50 hold for all 5 models;
- cost: about a third of a point.

## Wording

If step 3 holds for all 5 models, the tuned ABCD agent "is made to follow its 50 guideline-derived order rules by veto
clauses". It is still never "the guidelines are verified", and never "proved for all possible texts".

## Pre-check result (step 1; before this note's freeze; 2026-10-19 20:59:04-21:03:37 UTC; one thread)

- Output: logs/e72e_encoding.out (1dde8c582520fa62); row in results/explore/verify_e72e.jsonl.
- The domain: 5,927 rows (4,594 thermometer rows over 1,024 chains, 1,264 word rows), the same as e72d's. The bits are
  the same view.
- In the domain: every train state (29,190 of 29,190), every dev state (3,684 of 3,684), and every closed-loop dev state
  for each of the 5 models.
- For each model: the votes are equal on 500 of 500 dev rows, and the winner rows are right on 50 of 50 (own feasible,
  runner-up infeasible), at most 0.25 s each.
- **All passed.** Steps 2 to 4 follow.
