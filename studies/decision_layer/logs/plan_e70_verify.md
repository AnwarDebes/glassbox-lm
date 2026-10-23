# Plan e70 (Line B): prove, for every input, that the agent's policy follows ABCD's written guidelines

Dev and proof work only: the proofs are about the policy as a function and read no test split. The accuracy cost of any guard is measured on dev here. A registered test of that cost would be a separate registration. This plan is frozen (`scripts/freeze.py`) before the first solver run. Changes after the freeze go into a dated note, and every run is reported.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Why, and what is new

The goal names a policy that "can be revised and verified directly". Revision was explored (e40). Verification was not. Novelty check: the novelty report in the project notes (not published; 2026-10-18, web search).
- **Not new, to be cited:**
  - TM verification by SAT: Przybysz et al. 2023 (a vanilla TM, no weights, no argmax).
  - Verify, then repair with clauses: VERITAS (github.com/AnwarDebes/veritas, 2026, a binary +-1 TM guard).
  - Winning-class encodings over Boolean units: Kresse et al. 2025; Kantchelian et al. 2016; Ignatiev et al. 2022.
  - MILP verification of ReLU networks: Tjeng et al. 2019.
- **No earlier work was found** (search scope in the report) that:
  - verifies a coalesced TM's weighted multi-class argmax for all inputs;
  - formally verifies a learned next-action dialogue policy against properties from written guidelines;
  - compares TM and ReLU-MLP verification on the same task, bits and solver.
- An MLP can be verified and guarded too. So the comparison is measured, never claimed.

## The policies

- **TM:** R19abcd `tm_mem_static`, seeds 1 to 5. These are the registered models: view B (mem, words, xcust, xrecent), 4,000 clauses, 7,317 bits, no neural network at run time. They are used as trained, with no retraining. Kept clauses and literals per seed: 3,952 / 17,076; 3,937 / 16,950; 3,952 / 16,938; 3,931 / 16,914; 3,942 / 17,635.
- **MLP twins** (new, dev only): `dl.mlp.fit_mlp` on view B's train bits, best dev epoch, seed 1, hidden 64, 128 and 512 (512 is the study's standard MLP head). Their dev accuracy is reported next to the TM's.

## Properties (the rule, fixed now; `scripts/lineb_counts.py` derives the list)

- **Sources:**
  - the 55 subflows of `data/untrusted/abcd/guidelines.json`, mapped onto the 30 buttons by `scripts/abcd_inject.guideline_steps` (the study's mapping since e40);
  - the 55 sequences of `kb.json`.
  - A property is kept only if it holds in both. The two sources disagree on 4 subflows (`manage_pay_bill`, `refund_status`, `search_results`, `slow_speed`). Three precedences hold in one source only and are dropped: make-purchase before instructions, try-again before instructions, try-again before log-out-in.
- **P_first** (1 property, 25 checks): with an empty history (all 66 memory bits 0), the chosen button starts a subflow in one of the sources: log-out-in, notify-team, pull-up-account, search-faq or try-again. Each check takes one of the 25 other buttons B and asks whether some input makes B beat all 5 allowed buttons. The argmax lies outside the allowed set exactly when some B passes this test.
- **P_prec(A, B)** (25 properties, one check each): if A comes before the first B in every subflow that contains B (in both sources), the agent chooses B only if done(A). The pairs:
  - pull-up-account before ask-the-oracle, make-password, membership, offer-refund, promo-code, record-reason, send-link, shipping-status, subscription-status, update-account, update-order, validate-purchase, verify-identity (13);
  - enter-details before make-password;
  - verify-identity before subscription-status;
  - log-out-in before instructions;
  - search-faq before search-boots, search-jacket, search-jeans, search-membership, search-policy, search-pricing, search-shirt, search-timing, select-faq (9).
- So there are 50 checks per model. The properties are hand-translated, by the mapping above, from the written guidelines. They are not "the guidelines verified".

## What was known at the freeze (all dev, `results/explore/lineb_counts.json`, except the test pre-check)

- **The human agents break these rules.**
  - The first action is outside the allowed set in 53 of 1,004 dev conversations and 372 of 8,034 train conversations.
  - A precedence rule is broken in 113 of 3,684 dev action turns (938 of 29,190 train).
  - 70 dev conversations have at least one violation.
- **The TM breaks them less, but every seed breaks some on dev.**
  - First action: 5 to 8 of 1,004 conversations.
  - Precedence, open loop: 46 to 57 turns; closed loop: 24 to 35.
  - Conversations with any violation: 31 to 41 open loop, 19 to 26 closed loop.
  - So "never" will fail on real dev inputs for most properties' broad form, and the solver must find those cases or others.
  - The search-faq rules for the 8 topics are rarely broken (0 to 2 turns per seed).
- **A pre-check on R19's and R9b's stored test predictions** (post hoc, descriptive, no new test read):
  - the humans break "these buttons need a prior pull-up-account" at the first action in 4.2% of test conversations;
  - the R19 TM breaks it in 0.0 to 0.2% (5 seeds), the R9b TM in 0.0 to 0.3% (10 seeds).

## Input domain (Level B: only constraints that every real input satisfies)

- **Memory:**
  - last = b implies done(b);
  - at most one "last";
  - at least one "last" if and only if "at least 1 action";
  - the count chain (at least k+1 implies at least k);
  - some done(b) if and only if at least 1 action;
  - the number of done bits is at most the count while "at least 6" is off.
- **Thermometers:** for each field (customer's words so far; last 3 turns) and concept, level j+1 implies level j (the levels sorted by threshold, as named in bit_names).
- **Words:** a 1-2-gram bit implies the bits of its two words when they are in the lexicon (the same text).
- **Everything else is free.** This over-approximates the real inputs:
  - a proof holds for every real text;
  - a counterexample may match no real text.
- Level C (the exact token space, with xrecent and words tied through the token codes) is not encoded. Proofs are stated as "for all bit vectors that satisfy the constraints above, a superset of all real inputs".

## Encoding (`scripts/verify_policy.py`, new)

- **Clauses:**
  - only kept clauses (`src/dl/cpu_tm_fast.pack`: a clause with no included literal is dropped, as at inference);
  - partial evaluation of the fixed bits: a clause with a false fixed literal is dropped, true fixed literals are removed, and a clause with no free literal left adds a constant;
  - c_j in [0, 1] continuous, with c_j <= l for each literal and c_j >= sum(l) - (n_j - 1).
- **Votes:** s_k = sum_j w_jk c_j + const_k (integers).
- **Winner:** B wins iff s_B - s_k >= 1 for k < B and s_B - s_k >= 0 for k > B (ties go to the lowest index, as numpy and torch argmax, and `cpu_tm_fast.decide`).
- **MLP:**
  - h = ReLU(W1 x + b1) with exact bounds L, U from the 0/1 inputs;
  - a binary z_i per hidden unit, with h_i >= a_i, h_i <= a_i - L_i (1 - z_i), h_i <= U_i z_i;
  - logits W2 h + b2, the same winner rows with a margin of 1e-6 instead of 1;
  - its proofs carry a floating-point caveat (HiGHS tolerances).
- **Solver:** `scipy.optimize.milp` (HiGHS) in `.venv-dl`, with a zero objective (feasibility). Nothing is installed.

## Checks before any property run

1. **Votes:** on 500 dev rows, with every bit fixed, the encoding's s_k equal `cpu_tm_fast.class_sums` exactly, for every seed; for the MLPs, they agree with torch's logits within 1e-4.
2. **Winner:** for 50 dev rows with every bit fixed, the MILP for the model's own choice is feasible, and for one other class it is infeasible.
3. **Constraints:** every train and dev row (logged memory) and every closed-loop dev state satisfies the Level-B rows.
- If any check fails, the encoding is fixed before going on, and the failure is reported.

## Budget and order (fixed now)

- **Time limit:** 600 s per MILP; a total wall-clock budget of 24 h for all runs.
- **Jobs:** at most 2 solver processes at a time (OMP and MKL threads 2), inside the study's cap of 3 jobs.
- **Order:**
  1. TM seed 1, all 50 checks;
  2. MLP-128, all 50;
  3. the guarded TM seed 1 (below), all 50;
  4. TM seeds 2 to 5;
  5. MLP-64 and MLP-512;
  6. the guarded TM seeds 2 to 5.
- Whatever is not run within the budget is reported as not run.
- **Outcome per check:**
  - **proved** (infeasible);
  - **counterexample** (feasible), kept only after replay through `cpu_tm_fast.decide` (or torch for the MLP) confirms the argmax and the memory condition;
  - **unknown** (time limit).
  - For each, the solve time, the HiGHS status and the node count.

## Counterexamples

- **Decoded:** the memory state, and the words and concepts that are on.
- **Realizability:**
  - every real train or dev row on which the model breaks the property is a realizable counterexample. Their counts are reported per property;
  - for a solver counterexample, the nearest real dev row with the same memory bits (Hamming distance over the text bits) is reported;
  - a solver counterexample is called realizable only if it equals a real row's bits.

## Guards (the repair) and re-proof

- **The guards:** one veto clause per check (50 in all; `verify_policy.guards_for`). Its literal is the check's guard condition (not done(A), or not "at least 1 action" for P_first), with weight -M on B.
  - M = (sum of B's positive weights) + (sum of the absolute weights of pull-up-account) + 1.
  - pull-up-account is never a vetoed button, so whenever a guard fires, B's votes stay below pull-up-account's, for every input.
- **What the guards do, exactly:** the guarded policy picks the highest-voted button among those the rules allow in the current state.
  - It differs from the TM exactly where the TM's choice breaks a rule: lowering a class that is not the winner never changes the winner.
  - A guard on a rule the TM already obeys changes no decision.
  - Guarding every check at once gives the same decisions as adding guards only for the failed checks and repeating until no check fails, in one round.
- **Re-verify** all 50 checks for the guarded model. They hold by construction; the MILP run confirms the encoding of the guards.
- **Measure on dev** (`--check guardcost`), open and closed loop:
  - accuracy with and without guards;
  - the number of changed decisions;
  - how many of them were right before (the human broke the rule too) and how many are right after.
- Code check before the freeze: on 300 random bit vectors (half with an empty history) for seed 1, the clause-by-clause votes with the guards equal the fast path, and no vetoed button ever won.

## What would count as the result

- For TM seed 1: the number of the 50 checks proved, refuted with a counterexample, or unknown; the same after the guards.
- **The headline, if it holds:** "after N guard clauses, all 50 properties are proved for every input of the constrained domain, at a dev cost of X points". No bar is set for X; it is reported.
- The TM and MLP solve times are reported side by side, with their outcomes. No claim of "only a logic-based agent".

## Notes from an independent check (2026-10-19 00:50 UTC), added before the freeze

- **CPU:** until the concurrent F1 job ends (about 04:30 to 05:00 UTC), 1 solver process with 2 threads; then 2 processes with 2 threads each. Both phases are logged.
- **Solver record:** every result row records scipy 1.17.1, HiGHS 1.12.0, the time limit, threads and the MIP tolerances (mip_feasibility_tolerance 1e-6, primal_feasibility_tolerance 1e-7, HiGHS 1.12's defaults, set explicitly; random_seed 0). "Proved" is a statement within those tolerances. For the TM all coefficients are integers.
- **Reporting:** all 50 checks per model, the proved ones too, with times. For solver counterexamples, the distribution of the Hamming distances to the nearest real dev row, not only the count of equal rows.
- **Headline wording**, by outcome: "for these 50 guideline properties, over a superset of all real inputs, the TM policy is proved to comply" or "was repaired with N veto clauses at a dev cost of X". It always names the properties and the superset.

## Pre-checks (run before the freeze, 00:51 to 00:55 UTC; `logs/e70_precheck.out`, row tag e70_precheck in `results/explore/verify_e70.jsonl`)

- The domain has 6,560 rows: 5,227 thermometer rows over 1,024 concept chains, 1,264 word rows, the rest memory.
- It holds for every real state: 29,190 of 29,190 train rows, 3,684 of 3,684 dev rows, and 3,684 of 3,684 closed-loop dev states for each of the 5 seeds.
- **Votes:** the encoding's votes equal `cpu_tm_fast.class_sums` on 500 of 500 dev rows, for each of the 5 seeds.
- **Winner:** with every bit fixed, the MILP for the model's own choice is feasible and the one for its runner-up is infeasible, on 50 of 50 dev rows for each seed (at most 0.16 s each).
- A smoke run before it (20 rows, 2 winner rows, seed 1; tag e70_smoke) gave the same.

## The MLP part

The MLP twins and their encoding are written after this freeze, in new files (a training script and `scripts/verify_mlp.py`). A dated note records them and their own pre-checks before the first MLP property run: logits within 1e-4 of torch's on 500 dev rows, and the winner rows on 50 fixed dev rows. `scripts/verify_policy.py` is not changed for them.

## Wording (from the novelty report)

- Never write "first verification of Tsetlin machines", "first verified-and-repaired TM", "neural policies cannot be verified or repaired", "the guidelines are verified", or "proved for all possible texts". Proofs are over the stated superset.
- Cite Przybysz et al. 2023 and VERITAS next to any statement about TM verification.
