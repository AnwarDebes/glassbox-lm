# Plan e70: dated note, 2026-10-19 (after the freeze at 00:58:12Z)

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

## 1. The MLP part: a deviation from "freeze everything first"

The MLP twins and their encoding were written after the plan's freeze, in new files, as the plan's section "The MLP part" said. `scripts/verify_policy.py` (c10c7722) is unchanged. `scripts/diff_frozen.py logs/plan_e70_verify.frozen.json` lists only NEW files: `scripts/train_mlp_twins.py`, `scripts/verify_mlp.py` and `scripts/verify_local.py` (the last is e71's, not e70's).

- `scripts/train_mlp_twins.py` (sha256 328831c8783c9918): `dl.mlp.fit_mlp` on view B's train bits (hash-checked against R19's), best dev epoch, seed 1. Trained 01:13 UTC; dev accuracy:
  - h64: 0.7902 (`mlp_h64_seed1.npz`, be85cb42fa0041d5);
  - h128: 0.8124 (5dcc210393887e83);
  - h512: 0.8198 (860ec9b3f3df864a).
  - TM seed 1: 0.8325.
- `scripts/verify_mlp.py` (sha256 8e4852c3eb79bd1b): the same 50 checks, domain, solver, options and time limit as `verify_policy.py`.
  - a_i = W1_i . x + b1_i, with exact 0/1 bounds; big-M ReLU for the unstable units;
  - the winner rows use a margin of 1e-6;
  - its proofs carry a floating-point caveat.
- **Its pre-check passed** (01:21 to 01:24 UTC, `logs/e70_mlp_precheck.out`, tag e70_mlp_precheck), before any MLP property run:

  | Size | Max logit difference from torch (500 dev rows) | Argmax equal | Own choice feasible | Runner-up infeasible | Slowest winner check |
  |---|---|---|---|---|---|
  | h64 | 6.7e-6 | 500 / 500 | 50 / 50 | 50 / 50 | 0.48 s |
  | h128 | 4.4e-6 | 500 / 500 | 50 / 50 | 50 / 50 | 0.80 s |
  | h512 | 5.4e-6 | 500 / 500 | 50 / 50 | 50 / 50 | 3.55 s |

## 2. The order

The plan's order was:
1. TM seed 1;
2. MLP-128;
3. guarded TM seed 1;
4. TM seeds 2 to 5;
5. MLP-64 and MLP-512;
6. guarded TM seeds 2 to 5.

The MLP code was not ready after step 1, so steps 3, 4 and 6 and the guard cost ran first (01:01 to 01:18 UTC). The MLP runs follow now: MLP-128, then MLP-64, then MLP-512. Nothing else changed. The TM runs took about 7 minutes of the 24-hour budget.

## 3. CPU phase

Phase 1, until the concurrent F1 job ends (about 04:30 to 05:00 UTC): 1 solver process with 2 threads. All e70 runs kept to it; section 4 records one deviation by e71's pre-checks.

## 4. Deviation: two CPU processes in phase 1 (logged 01:50 UTC)

From 01:29 to 01:49 UTC, e71's pre-checks ran next to e70's MLP solver process, against the rule of one solver process until the concurrent F1 job ends.
- The pre-checks were `verify_local.py --check model` and `--check contains`, which build MILPs and evaluate their rows but never call HiGHS.
- Each ran on one thread, so it was still a second CPU process.
- It was noticed at 01:4x: CPU at 4.24 cores, not throttled, no harm.
- No further e71 process runs until phase 2.

## 5. The first MLP-128 check (01:28 to 01:39 UTC)

P_first, verify-identity hit the 600 s time limit: outcome unknown, no counterexample found. The same check took the TM 0.57 s.
- It is reported as measured.
- The MLP was verified with the same generic MILP encoding and solver (scipy HiGHS, big-M ReLU), not with a dedicated neural-network verifier. Reluplex/Marabou, alpha-beta-CROWN and VeriNet exist and were not tried; one of them might close part of the gap.
- The wording is: "with the same generic MILP encoding and solver, the TM's checks resolved in about 1 s each, while MLP-128's timed out at 600 s".
- Never "MLPs cannot be verified", and never "TMs are N times faster to verify".
- The frozen order and budget stand. Whatever the 24 h do not reach (likely MLP-64 and MLP-512) is reported as not run.

## 6. CPU phase 2 from 02:04 UTC

The phase-1 rule was lifted at 02:04 UTC. W1's GPU job had ended; only F1 (1 core) and e70's MLP solver were running, at 2.21 cores with 0 of 60 periods throttled.
- From then: two solver processes, 2 threads each, while the total stays under about 5 cores and is not throttled.
- The load is measured before each start (`scripts/cpu_load.py`, a read of the cgroup's cpu.stat over 10 s); above 5 cores, back to one solver process.
- A measurement at 02:05:20 UTC: 3.43 cores, 0 of 100 periods throttled.

## 7. MLP-512 not run (budget), decided at 03:03 UTC before it ran

- MLP-128 finishes as frozen; then MLP-64 runs. MLP-512 is dropped and reported as "not run (budget)".
- Why: e71 v2 needs the second CPU slot.
- Why MLP-64 is kept rather than MLP-512: the smallest network is the one most likely to be verifiable in time, so keeping it is fair to the MLP. Dropping the smaller one would look like avoiding an unfavourable result.
- How: at 03:04:05 UTC the shell loop that would have started h64 and then h512 (PID 2410672) was stopped. The h128 run (PID 2410674) went on unchanged. A watcher starts h64 alone when h128 ends.
- What had been seen at the decision: MLP-128's checks so far were all "unknown" at the 600 s limit.

