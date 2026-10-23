# Note e72d (2026-10-19): e70's checks and guards on the agent with the C4-word table

A dated note of its own, decided at 05:0x UTC. It covers the one Line B claim that may carry over
to the agent with no neural network at any stage: e70's veto clauses. No rule. Dev and proof work only: no test split
is read. It is frozen with `scripts/freeze.py` before the first property run. The encoding pre-check may run before
the freeze, as in e70.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## The question

R21 trains five ABCD models of view B on the C4-word table (`results/confirm/R21abcd/tm_mem_c4`). The question has
two parts:
- Do e70's 50 rules have counterexamples for these raw models, over the same superset of real inputs?
- With the 50 veto clauses (M recomputed per model), do all 50 checks hold, and what do the guards cost on dev?

## What is e70's and unchanged

- The 50 checks (`scripts/lineb_counts.py`, `properties`): P_first, 25 checks; P_prec, 25 pairs.
- The exact MILP encoding, the domain rows and the HiGHS options (scipy 1.17.1, HiGHS 1.12.0, tolerances 1e-6 and
  1e-7, random seed 0, 2 threads, 600 s per check). They are imported unedited from `scripts/verify_policy.py`.
- The guards: one per check, with weight -M. M is B's positive weights plus the absolute weights of
  pull-up-account plus 1, from each model's own weights.
- The script is `scripts/verify_policy_c4.py`.
- **What differs is only the models and their bits.** The bits are view B on the C4 table, built as in the runs:
  dl.count_text.patch, so potion's table cannot load, and the empty "none" cache, so no encoder vector is loaded.
- The domain rows come from these models' bit names: the thermometer chains of the C4 concepts and the 1-2-gram
  rows.

## Steps (one process, 2 threads; after R21's ABCD TM jobs, when cpu_load is at most 4.5 cores)

1. **Pre-check** (`--check encoding`, all 5 seeds). As e70's, it must give:
   - the encoding's votes equal the CPU decision's on 500 dev rows;
   - 50 of 50 winner rows right;
   - every train, dev and closed-loop dev state inside the domain.
   - If it fails, nothing else runs and the failure is reported.
2. **The raw models** (`--check properties`, 50 checks x 5 seeds): proved, counterexample (replayed through the real
   decision code), or unknown at 600 s.
3. **The guarded models** (`--check properties --guard`, 50 x 5). They should hold by construction, and the runs
   check the guards' encoding.
4. **The guards' cost on dev** (`--check guardcost`): open loop and closed loop, with the changed decisions split
   by right and wrong.

Rows go to `results/explore/verify_e72d.jsonl`, tag e72d. A check already in that file is never run twice.

## Expected (from e70; a guess, not a rule)

- The raw models: counterexamples for most or all of the 50 checks. A free text space lets the solver build a text
  that votes for any button, and the C4 bits are as free as potion's.
- The guarded models: 50 of 50 hold for all 5 seeds.
- Cost: about a third of a point open loop, as e70.

## Wording

- If step 3 holds for all 5 models, the R21 headline may add "made to follow its 50 guideline-derived order rules by
  veto clauses".
- Never write "the guidelines are verified": the rules come from guidelines.json and kb.json through a hand-written
  name mapping.
- Never write "proved for all possible texts": the proofs are over the level-B superset of the bits.
- e71's word-edit proofs are not claimed for this agent.

## Pre-check result (step 1; before this note's freeze; 2026-10-19, 11:15:49-11:18:56 UTC; one thread)

- Command: `scripts/verify_policy_c4.py --check encoding --threads 1`. Output `logs/e72d_encoding.out` (36efd474959820c8); row in `results/explore/verify_e72d.jsonl`.
- The domain: 5,927 rows (4,594 thermometer rows over 1,024 concept chains; 1,264 word rows). e70's potion domain had 6,560 rows, with 5,227 thermometer rows, because the C4 table has fewer distinct thresholds.
- Every train state (29,190 of 29,190) and every dev state (3,684 of 3,684) is in the domain. So is every closed-loop dev state, for each of the 5 models.
- For each of the 5 models:
  - the votes equal the CPU decision's on 500 of 500 dev rows;
  - the winner rows give own feasible 50 of 50 and runner-up infeasible 50 of 50, at most 0.20 s each.
- **All passed.** Steps 2 to 4 follow.
