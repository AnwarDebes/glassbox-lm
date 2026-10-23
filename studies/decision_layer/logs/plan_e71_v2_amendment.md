# Plan e71, v2 amendment (2026-10-19): exhaustive search for k = 1 and 2

This amends `logs/plan_e71_local.md` (frozen 2026-10-19T02:10:00Z, sha256 76fac7b0b6940c5c). Everything not named here stays as frozen: the question, the edit model, the 1,626 candidates, the 200 states, the rule groups, the 24 h budget counted from 02:10 UTC, and the wording rules. It is frozen (`scripts/freeze.py`) before any v2 run.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Why (no result was seen that could steer it)

- **The first run.** It was stopped at 02:23:32 UTC after 6 of 579 MILPs (tag e71; rows kept; the frozen script is archived as `logs/archive/verify_local_frozen_4aaf9a57.py`). All 6 hit the 120 s limit, most with no dual bound, because the root was not finished at 154k to 424k rows and 2.5M to 3.7M nonzeros. A 120 s cap gives no information at that size.
- **The reduced encoding** (exact, below). It is 14 to 30 times smaller (14k to 27k rows), but all 6 instances still hit 120 s with no dual bound (tag e71v2_timing). One instance found a 358-word insertion that breaks the rule on replay, with bound 0.
- **Probe, not a result** (tag e71v2_budget_probe): feasibility with at most k edits. k = 1 came out infeasible (certified) on all 4 probe instances in 15 to 54 s; k = 2, 3 and 5 were unknown at 120 s.
- So minimising the edit count with a big-M winner disjunction is too hard for HiGHS here. For small k, exhaustive search is exact, cheaper and stronger: it needs no superset, and every edit set it finds is real text through the real decision code.

## v2 measures

1. **k = 1 and k = 2: exhaustive search** (`scripts/verify_enum.py`, sha256 94e9b1e223a37ed0 at this freeze). For each state, every edit set is tried:
   - delete one or two of u's words, with u keeping at least one word;
   - append one or two candidate words, as an ordered pair, since the bigram order matters, with a repeated word allowed;
   - or one deletion and one insertion. With a single-word u, the deleted word is replaced by the inserted one.
   - That is about 2.65M edit sets per state for k = 2.
   - The bits are rebuilt from the words, as view.transform builds them: word-table codes OR-ed, and unigrams and bigrams from the edited token sequence, with the junction to the word after u.
   - The scores:
     - Deletions are scored with cpu_tm_fast's clause firing and the integer votes (float32 sums of integers below 2^24 are exact).
     - Insertions are scored through the TM partially evaluated on their base row (`Reduced`). Insertions only switch bits on, so a clause with a negated literal on an ON bit can never fire, and a positive literal on an ON bit is true. This is exact.
   - k = 2 runs only for states where some group is certified at k = 1, since a breaking 1-edit set already refutes k = 2 for that group.
   - **Outcomes per (state, group):**
     - "certified at k = 2" (no edit set of size 1 or 2 breaks the rule);
     - "certified at k = 1; a 2-word edit breaks the rule";
     - "a 1-word edit breaks the rule";
     - "own choice breaks the rule".
   - Up to 5 breaking examples per group are replayed through the real pipeline (strings, view.transform, the CPU decision).
2. **k = 3 and k = 5: the decision-version MILP**, for the groups certified at k = 2 only, best effort (`verify_enum.py --check milp35`).
   - It asks whether some edit set of size at most k makes a button of the group win, using `verify_local.solve_group` with `budget=k` and the reduced exact encoding, with its bigram superset.
   - Same 120 s cap. Infeasible means certified; a found set is replayed. If k = 3 is not certified, k = 5 is not run.
3. **The reduced exact encoding of the MILP** (`scripts/verify_local.py`, sha256 2d695a4ca950fdbe at this freeze; it was 4aaf9a57 at e71's freeze): the same integer solutions as the frozen one, smaller.
   - an edit-count variable E, with one defining row; the bigram no-edit rows use E; the objective is E;
   - insertion rows aggregated per word and field;
   - bits that no clause reads are dropped: 6,039 of 7,251 text bits are read;
   - candidates that cannot change a read bit are dropped, with a filler count for such insertions;
   - an optional edit budget.

## Pre-checks (all passed, before this freeze)

- **The enumerator against view.transform** on edited texts rebuilt from strings (30 states, tag e71v2_enum_precheck):
  - with no edit, 30 of 30 exact;
  - every single insertion and deletion, 48,963 of 48,963 rows exact;
  - 1,500 random double edits, all exact;
  - its argmax equals cpu_tm_fast's on 48,780 of 48,780.
- **The Reduced scoring against full rows** (30 states, tag e71v2_enum_precheck2): every single insertion, 48,780 of 48,780 equal; every pair for 5 random first words per state, 243,900 of 243,900 equal.
- **The reduced MILP:**
  - contains the real outcome of random real edit sets on 200 of 200 states, with fixed bits matching 200 of 200 (tag e71v2_contains_precheck);
  - with HiGHS and the edits fixed, own choice feasible on 30 of 30 and runner-up infeasible on 28 of 30 (the bigram superset); at most 0.63 s; median 17,200 rows (tag e71v2_milp_precheck).

## Time per state (k = 2)

- Full scoring: 549.6 s for one state (tag e71v2_enum_timing; that run was stopped after one state).
- With the Reduced insertions: 106 to 172 s per state, 2.65M edit sets each (3 states, tag e71v2_enum_timing_reduced).
- So about 7 to 10 h for the states that need k = 2, in one process.

## Order and CPU

- **Order:**
  1. TM seed 1: k = 1 and 2, all 200 states (tag e71v2);
  2. TM seed 1: k = 3 and 5 for the groups certified at k = 2 (tag e71v2_k35);
  3. TM seeds 2 to 5: k = 1 and 2, as the budget allows;
  4. e71b (MLP-128) after its own dated note, as frozen.
- **CPU:** one process, threads 1 to 2, next to e70's MLP-64 run. The load is checked before each start (`scripts/cpu_load.py`). Above about 5.5 cores, e71 drops to 1 thread until the concurrent F1 job ends (about 04:50 UTC).

## Headline (as frozen, with the method named)

"For X% of 200 sampled real dev states, no deletion of words from the customer's latest message, or insertion of up to k of 1,626 common customer words at its end, makes the agent choose a button the rules forbid in that state."
- k = 1 and 2 are marked "by exhaustive search"; k = 3 and 5 "by MILP, where resolved".
- The states whose u is outside the last 3 turns are reported separately.
