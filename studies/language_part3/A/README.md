# A-Struct v2: code for S0-S2 and the K3c CPU pilot

The Part 3 plan (not distributed), section 3, with every review amendment (section 11). The registered rules are in `REGISTRATION_R1.md`;
every file the code depends on is pinned in `R1_A-Struct.frozen.json` (written by `freeze.py`; the same hashes in
sha256sum format in `MANIFEST_R1.sha256`). The R1 line in `../LOG.md` holds the first frozen record's SHA-256, and
its amendment line holds the current record's. Nothing here had run on the real data at either freeze.

**Amendment 1 (2026-10-21, before any real run; REGISTRATION_R1.md section 17).** It fixes the code review's two
major issues (1, 2) and five of its nine minor ones (3, 4, 5, 9, 11; numbered in the review's order):
- K8 is now the equal-size A/A: rebuilt(seed 2) minus rebuilt(seed 3).
- Every real job runs under one Part 3 chain lock, shared with C.
- K3c also needs the pilot to be better than B.
- Real pilots and K3c decisions need the registered settings.
- Real outputs need the runner (nice 19, `A_RUNNER=1`).
- The leak self-check also covers the fold floors' S-keyed lookups.
The frozen files of the first freeze (05:13:40Z) are kept unchanged in `history/`.

## Run rules (all enforced in code)

- Every real job goes through the Part 3 chain wrapper:
  `chain/part3_cpu_job.sh A tm|dl <script.py> [args]` (from `studies/language_part3/`; see `../chain/README.md`). It
  calls `run_job.sh tm|dl <script.py> [args]`.
  - **One Part 3 CPU job at a time, across A and C.** The wrapper holds `chain/part3_cpu.lock` for the whole job.
    `run_job.sh` refuses a real job (exit 2, nothing written) unless `PART3_CHAIN_LOCK` names that lock and the lock
    is held (`lock_held.sh`). A's own lock `$PART3_CPU_LOCK` (default `A/tmp/cpu_job.lock`) must never be the chain
    lock, which would make the runner wait for the wrapper forever; the runner and the wrapper both refuse it. Never
    wrap either in another `flock` on the chain lock.
  - The CPU gate: decision_layer `scripts/cpu_load.py 10` must show at most 4.0 cores in use and 0 throttled
    periods, checked again every 2 minutes while closed. Every check goes to `A/out/jobs.log` (the gate-open hours of
    K7, read by `k7_clock.py`).
  - `env.sh`: 2 threads (OMP, MKL, OPENBLAS, NUMBA, RAYON), no GPU (`CUDA_VISIBLE_DEVICES=` empty), and
    NUMBA_CACHE_DIR, TMPDIR, JOBLIB_TEMP_FOLDER, MPLCONFIGDIR and XDG_CACHE_HOME in `A/tmp/`; torch at 2 threads
    (`torch.set_num_threads`).
  - `nice -n 19 python -I -B`; every script asserts these (`astruct/paths.py` `check_env`). A real output also needs
    nice 19, `A_RUNNER=1` (set only by the runner's real mode; its self-test mode sets `test`) and the held chain lock
    (`paths.check_runner`, called by `paths.OutDir`).
  - The imported folders are listed before and after (`listing.py`); a change gives exit code 4: stop and report.
- A real run (outputs under `A/out/`) first checks the frozen record: every enforced file must have its frozen
  SHA-256, and LOG.md must hold the record's SHA-256 (`paths.verify_frozen`). Otherwise it stops before writing.
- KenLM `lmplz` runs with `-S 4G -T A/tmp/kenlm_sort`; `build_binary` (probing) and `query` write no temporary files.
- Outputs go to `A/out/<step>/`, written as `<step>.partial` and renamed at the end; an existing step folder is never
  overwritten (a failed job leaves only `.partial`, which the next attempt removes). A crashed job may be rerun
  unchanged (the code is deterministic); any code change needs a dated amendment line and `freeze.py --amend`.
- Environments: `tm` = tm_language_model `.venv-tm` (numpy, numba, scipy, sklearn); `dl` = decision_layer `.venv-dl`
  (torch on the CPU, sklearn).

## Job order (CPU; plan, section 6, steps 3-5)

Every command below is run from `studies/language_part3/` as `chain/part3_cpu_job.sh A <command>`. Exit codes through
the wrapper: 0 done; 2 refused or a usage error (nothing written); 3 a registered STOP; 4 the listings differ; 5 (job
18 only) run the printed shuffled pilot first; anything else, the job failed.

| # | Command | Exit codes | Decides |
|---|---|---|---|
| 1 | `tm s0_data.py` | 0 | data; the frozen bit list |
| 2 | `bash A/log_bitlist.sh` (not a CPU job: no wrapper; one LOG.md line, time from `date -u`) | 0; 2 = S0 not done | bit list logged before the first tune-cal score |
| 3 | `tm s1_floors.py` | 0; 3 = K1 STOP | floor, K1 (and K2 for the floors) |
| 4 | `tm s2a_rows.py` | 0 | rows; leak and no-future tests |
| 5 | `tm s2b_twin.py` | 0; 3 = K9 STOP | B |
| 6 | `tm s2c_aa.py` | 0; 3 = K8 STOP | K8, the equal-size A/A (rebuilt(2) minus rebuilt(3)); B minus each rebuild reported with no rule; before any learner number exists |
| 7-14 | `tm s2b_arm.py --arm {full,nf} --family {tree,lr1,lr2,hgb}` (8 jobs, any order) | 0; 3 = K9 STOP | learners |
| 15 | `tm s2d_gates.py` | 0; 3 = both arms stop | K3 (i)/(ii), K8, K9 per arm |

A K1, K8 or K9 STOP: the job writes its outputs, then exits 3; the chain stops A there, runs nothing after it, and
reports. (S2c refuses to start once a learner job has written its outputs, and S2b-arm refuses to start before S2c
is done, so K8 is always decided before any learner's tune-hold number exists.) Only an arm X with
`go_to_K3c[X]` true in `A/out/s2d_gates/gates.json` goes on (a real pilot refuses to start otherwise):

| # | Command | Exit codes |
|---|---|---|
| 16 | `dl k3c_pilot.py --arm X --s 15` | 0 |
| 17 | `dl k3c_pilot.py --arm X --s 40` | 0 |
| 18 | `tm k3c_gate.py --arm X` | 3 = K3c STOP (written); 5 = run the printed shuffled pilot, then this again |
| 19 | `dl k3c_pilot.py --arm X --s <better s> --shuffled` | 0 |
| 20 | `tm k3c_gate.py --arm X` | 0 = the arm may go to the GPU stage (plan step 7); 3 = stop |

- A real pilot refuses `--epochs` / `--clauses` overrides (dry run only), an s outside {15, 40}, a folder other than
  `A/out/k3c_<arm>_s<s>[_shuffled]`, and (for the shuffled pilot) an s other than the better one or an arm that K3c
  stopped.
- A real K3c decision reads only `A/out/` and its own S2d `gates.json`, and refuses pilots whose settings are not the
  registered ones.
- K3c passes only when the better pilot is better than B **and** gains at least a third of the boosted trees' gain
  over B (amendment 1).
- `python -I -B k7_clock.py` prints the K7 clock (gate-open hours, days since S0) at any time; exit 3 = limit reached.

Time figures for scheduling (code review of 2026-10-21, issue 11). The frozen self-test log measured 0.5659 s
per batch of 128 on synthetic rows of the real width (364 bits, 4,000 clauses, T 8,000, 2 threads), projecting
2.58 h per pilot (about 210,000 rows, 10 epochs). The review measured 0.55-0.60 s per batch, so 2.5-2.7 h per pilot.
Up to 3 pilots per arm and 6 in all is about 16 h, against the plan's 1.2-3 h for step 5. K7's clock counts only job
wall time, a lower bound on the gate-open hours.

## Files

- `astruct/`: data (S0), positions, chain (KN and absolute-discount chains), floors (F1-F6), rates, rows (bits),
  store, build (cross-fitting and the leak check), twins, learners, readout, gates, tmrun (CoTM loop), paths (run
  rules, the runner and chain-lock check, the frozen record).
- Scripts: `s0_data.py`, `s1_floors.py`, `s2a_rows.py`, `s2b_twin.py`, `s2b_arm.py`, `s2c_aa.py`, `s2d_gates.py`,
  `k3c_pilot.py`, `k3c_gate.py`; `run_job.sh`, `lock_held.sh`, `gate_check.sh`, `env.sh`, `listing.py`,
  `log_bitlist.sh`, `k7_clock.py`, `freeze.py`.
- Self-tests: `tests/run_tests.sh` (pytest in `tm`, plain asserts in `dl`, the listing check), including the dry run
  `tests/test_e2e.py` of every entry point on the synthetic corpus of `tests/synth.py`; logs in `tests/results/`. The
  whole run takes about 4 minutes at 2 threads: start it only when the CPU gate is open.
- `review/`: the pre-freeze checklist review of R1.
- `history/`: the first freeze's record, manifest, registration and README (2026-10-21T05:13:40Z), byte for byte.
  Their SHA-256 values are in the R1 line of LOG.md.
