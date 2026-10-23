# e82's R5 note (2026-10-21): e70's 50 guard checks on the refined-table agent's five ABCD models (CPU; dev and proof work)

Plan logs/plan_e82_refined_table_agent.md (frozen 2026-10-21T01:31:09Z), section "R5, the guards". This note is frozen
with `scripts/freeze.py` before any property run. No test split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Models and code

- **The models:** the refined-table agent on ABCD at e82-R3's s* (s 16: the 4k pick, with no 16k step for ABCD), view B on
  c4words_a3B_v1, 4,000 clauses, seeds 1-5.
  - Seeds 1-2 are Stage A's rows; seeds 3-5 are Stage C's.
  - The files are results/explore/abcd_models/e82_abcd_Bt_c4000_s16_seed<k>.npz.
  - They are the models a registered ABCD run at s 16 would reproduce, and the checker would require seed 1's dev
    accuracy to match.
- **The wrapper:** `scripts/verify_policy_a3B.py` (d5785d13), around scripts/verify_policy_c4.py (unedited).
  - It registers c4words_a3B_v1 through newtable_run.register and sets vpc.TABLE to it, so the bits are built on the
    a3B table, never on the C4 table.
  - vpc.SRC is results/explore/r5_a3B_s16/: the 5 models copied byte for byte by `--prepare`, and bit_names.json from
    the models' own name arrays.
  - vpc.OUT is results/explore/verify_e82r5.jsonl.
- **Everything else is e72d's / e70's:**
  - the 50 checks (lineb_counts.properties);
  - the exact MILP encoding with HiGHS 1.12.0 and its options;
  - the level-B domain rows from the bit names (xcust and xrecent keep their names, so verify_policy's thermometer
    chains apply);
  - the guards, with M recomputed per model.

## Steps (one process, 2 threads at most, nice 19, the cpu_load gate; CPU only, so it may overlap step 3c of the text-understanding line)

1. `verify_policy_a3B.py --s 16 --prepare`.
2. `--check encoding --threads 1 --tag e82r5`, the pre-check:
   - the encoding's votes equal the CPU decision's on 500 dev rows;
   - the winner rows on 50 fixed rows;
   - every train and dev state lies in the domain.
   It must pass before any property run.
3. `--check properties --threads 1 --tag e82r5` (raw), then `--check properties --guard --threads 1 --tag e82r5`.
4. `--check guardcost --tag e82r5`: the guards' cost on dev, open loop and closed loop.

## The pass bar (the plan's)

- **250 of 250 guarded proofs.** That is a precondition of any registered test using this agent, and of the agent using
  the table at all.
- Raw counterexamples are expected (e70, e72d, e72e: 250 of 250 each). They are reported with their nearest dev row's
  distance.
- The guard cost is reported with no rule, beside e72e's (-0.46 to -0.60 open loop, -0.16 to -0.35 closed loop).
