# speed: faster Tsetlin machine code with byte-identical results

Speed work on the decision_layer agent's code under one rule: a change is kept only if a fixed set of benchmark runs
gives the same results before and after (the same states, scores and models, compared byte for byte). Nothing here
changes a registered run; a new plan may adopt a change by naming its file hash.

## FastCoTM (`phase2/tm/fast_cotm.py`)

A subclass of `dl.tm.CoTM` (`../src/dl/tm.py`) that computes the same feedback with fewer element-wise passes: counts
from the plain bits, probabilities cached per shape, the same random draws in the same order. Results:

| Benchmark | Before | With FastCoTM | Results |
|---|---|---|---|
| B1 (4,000 clauses, 2 chunks) | 140.9 s | 117.1 s (-16.9%) | identical |
| B2 (4,000 clauses, unchunked) | 102.8 s | 95.8 s (-6.8%) | identical |
| B3 (STAR, 16,000 clauses, 5 chunks) | 1,202.8 s | 1,002.5 s (-16.7%) | identical |
| B4 (MultiWOZ, 16,000 clauses, 5 chunks) | 1,403.5 s | 1,195.5 s (-14.8%) | identical |

Accepted on all four shapes (2026-10-22), as the file with SHA-256 `c905a89f438ead98407dae098e0bf7128eda2b6e9ce75b4c69441b4b3115e36f`. The published copy differs from it only in two comments and in finding `src/` from its own place (SHA-256 `a72eb8b849c927464287d92b1d445ceca97c9e0b5aee24aa8d1a71e7e3a307b8`); the hashes that the patch files, the checker and the tests pin were updated to the published copies, and so were the pins of `tmcode/test_identity.py` and `tmcode/prof_tm.py` in `review/bench/identity_cpu.py` (as run `9cb6f4e8bc3119c8886e2d6eb57bb2648e04e40f8e0fece6fdce81e90c2b98e3` and `e2774df7be58425e7c79fe36423f8d12d1d47bdfdfb2fcc9ab0068b1223ec6f1`, published `53b7187208621aa236793ba37f728933c2f2beab178371d984b6e298b17023c8` and `71f1a483b16cf8bba1c93d9591d9e8463a51df6d6c950203ec006f9423f32a41`; they differ in docstring wording and paths only). Two other candidates were rejected: pacing the host thread freed no CPU,
and its combination with FastCoTM failed the same CPU bound. The CPU identity test (`tmcode/test_identity.py`, run
on the prototype `tmcode/fast_tm.py`) passes on the prototype and fails on a planted change of one probability by a
factor of 1 + 2^-20, so it can see a difference that small. No registered run uses FastCoTM.

Files: `phase2/tm/` (FastCoTM, the configs, the patch files that load it before an unedited entry script, the job-log
checker and their tests), `tmcode/` (profiles, the prototype and its identity tests), `review/bench/` (the benchmark
jobs, their private run roots and the fingerprint comparison).

## A C clause evaluator for the agent (`cbench/`)

`tmclause.c` evaluates the trained clauses of the registered seed-1 agents (R22 STAR, R23 ABCD, R24 MultiWOZ) and
`bench_c_clause.py` checks that it gives the same class sums as the numpy path on every dev row before it times them
(one CPU thread, batch 1; rough, on a shared machine; `bench_*.json`):

| Dataset | Clauses | Decision now (numpy) | With the C evaluator | Speed-up |
|---|---|---|---|---|
| star | 15,997 | 2.59 ms | 0.96 ms | 2.70x |
| abcd | 3,995 | 0.44 ms | 0.30 ms | 1.48x |
| multiwoz | 16,000 | 2.93 ms | 1.07 ms | 2.73x |

The self-test (`selftest.out`) compares both paths on five synthetic cases: all pass, with no row differing.
