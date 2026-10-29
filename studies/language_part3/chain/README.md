# Part 3 chain: one CPU job at a time across A-Struct and C-core

Written 2026-10-21 (UTC) to fix issue 2 (major) of the code review of A/ and C/ (2026-10-21; issues
numbered 1-11 in the review's order): neither
arm's runner kept an A job and a C job apart. A's runner locked `A/tmp/cpu_job.lock`; C's runner locked
`C/tmp/c_job.lock` with `flock -n`. So two jobs started together could both pass the CPU gate before either loaded
the CPU. Exit codes also differed (C: 3 = its lock is busy; A: 3 = a registered STOP), and C's stages exit 0 on a gate
STOP. This folder holds the fix. C's code is unchanged (frozen in R1 [C-core]). A's runner now refuses a real job
that does not run under this lock (A amendment 1, LOG.md).

## The rule

Start **every real Part 3 CPU job** of A or C through `part3_cpu_job.sh`, one command per job, in the plan's order
(the Part 3 plan, section 6):

```
chain/part3_cpu_job.sh A tm|dl <script.py> [args...]    # A-Struct: A/README.md, job order 1-20
chain/part3_cpu_job.sh C <stage> [args...]              # C-core: C/README.md (step0, prep, twin, learners, gates)
```

- The wrapper takes an exclusive `flock` on `chain/part3_cpu.lock`, waits while another Part 3 job holds it, and
  holds it until the arm's runner ends. Each runner checks the CPU gate (`cpu_load.py 10`: at most 4.0 cores, 0
  throttled periods, every 2 minutes) only while it holds the lock. So the gate is never read with another Part 3 job
  about to start.
- **Do not** wrap the wrapper or an arm's runner in another `flock` on `chain/part3_cpu.lock`, and do not set
  `PART3_CPU_LOCK` to it: the inner open would wait for the outer one forever. The wrapper refuses both (and nested
  calls).
- Self-tests and dry runs on tiny data (under 2 minutes) need no lock: run them directly (`A/tests/run_tests.sh`,
  `C/scripts/run_stage.sh selftest_tm|selftest_dl`, the C stages with `--dry`). A's full self-test run takes about 4
  minutes, so start it only when the CPU gate is open.
- A real A job started without the wrapper is refused before it writes anything: `A/run_job.sh` checks that
  `PART3_CHAIN_LOCK` names this lock and that the lock is held. Every A script also refuses real outputs without
  `A_RUNNER=1` and nice 19. C's runner has no such check, because its code is frozen. **Never start a real C stage with
  `C/scripts/run_stage.sh` directly.**

## Exit codes (the same meaning for both arms)

| Code | Meaning | What the chain does |
|---|---|---|
| 0 | done | next job |
| 3 | a registered STOP. A: the script's own exit 3 (K1, K8, K9, K3 for both arms, K3c). C: the real `gates` stage ran and `C/data/step6_decision.json` "decision" starts with "STOP" | stop that arm, run nothing after it, report |
| 4 | the imported folders changed during the job | stop and report (read the listing diff) |
| 5 | A's `k3c_gate.py` asks for the shuffled pilot | run the printed command, then the gate again |
| 75 | not run: C's own lock was busy (for example a C self-test) | wait a minute, run the same command again |
| 2 | usage error or refused start (nothing written) | fix the command |
| other | the job failed | read its log in `chain/logs/` and report |

- C's runner also exits 4 when only the ".." lines of a listing changed. Those lines describe a parent folder, which
  other jobs change by writing next to an imported folder (for example a new table in `decision_layer/data/tables`).
  The wrapper reads the runner's diff file. When every changed line is a ".." line, it logs a `note` and returns 0
  (review issue 6). A's `listing.py` already treats ".." lines as notes.
- C's gate decision comes only from a `step6_decision.json` written during this job, read from its "decision" field.
  C's stop reasons, in order: shuffled labels, A/A, K0, K0b. Read a C A/A or shuffled stop next to the diagnostics
  printed beside it (`twin_AA - twin` interval, `class_unigram_diagnostic`; review issue 10).

## Records

- `part3_cpu_jobs.log`: `wait`, `start`, `note` and `end` lines, with times from `date -u` in the writing command.
  The `end` line holds the runner's exit code, the wrapper's code and the reason.
- `logs/<UTC>_<arm>_<name>.log`: the runner's whole output. The wrapper prints its last 40 lines at the end.
- The arms keep their own records too: `A/out/jobs.log` (A's gate checks and job times, which give K7's gate-open
  hours) and `C/runs/` (C's gate log, job log and listings, which give C-K4's hours). Both clocks count only what
  their runners saw, so they are lower bounds on the gate-open hours.

## Order and time figures (review issue 11; for scheduling, not limits)

Plan, section 6 order: C step 0 (exploratory), then A S0-S2 (README jobs 1-15), then A's K3c pilots for every arm
with `go_to_K3c` true (jobs 16-20), then C step 6 (prep, twin, learners, gates). After a STOP, the plan's order says
what runs next. A K8 STOP stops both A arms before any learner runs.
- A's K3c pilots: 0.55-0.60 s per batch of 128 on the CPU at 2 threads (the review's measurement; A's frozen
  self-test log has 0.5659 s and 2.58 h projected), so about 2.5-2.7 h per pilot. Up to 3 pilots per arm (s 15,
  s 40, the shuffled one) makes up to 6 pilots and about 16 h, against the plan's 1.2-3 h for step 5.
- C's learners stage: the review estimates LightGBM's worst case at about 8 h (about 9 s per round x 400 rounds x 8
  trials), plus LR. C's README estimate is 2-6 h.
- By the review's sums the totals stay under A's K5 (34 h of CPU wall time), but they use up much of the 72
  gate-open hours of K7 and C-K4.

## Self-tests

```
cd studies/language_part3 && \
  studies/tm_language_model/.venv-tm/bin/python -I -B chain/tests/test_part3_cpu_job.py
```

The tests use their own lock and log in `chain/tmp/selftest/` (`PART3_CHAIN_TEST_DIR`). They cover: two jobs started
together never overlap; the refusals (usage, the test-only arm, the deadlock guard, nested calls); A's exit codes pass
through, with `PART3_CHAIN_LOCK` and `A_RUNNER=test` set in A's own test mode; C's mapping on a stand-in runner
(".."-only listing change gives 0 and a note, other changes 4, lock busy 75, the gates decision 3 or 0, a stale
decision file 1, `--dry` reads none); and C's real runner on its `selftest_dl` (0, and 75 with C's lock held).
About 1 minute; no CPU gate needed.
