# C-core v2: code and run order (Part 3, the Part 3 plan, section 4)

Registration: `R1_C-core.md`, frozen in `R1_C-core.frozen.json`, line R1 [C-core] in `../LOG.md`.
Everything below writes only under this folder. No GPU. No git.

## Run order (one Part 3 job at a time, each through the chain wrapper)

Since 2026-10-21 (code review of A/ and C/, issue 2), every real C stage starts from `studies/language_part3/` as
`chain/part3_cpu_job.sh C <stage>`. The wrapper holds `chain/part3_cpu.lock`, shared with A-Struct, for the whole
stage, and calls `scripts/run_stage.sh <stage>`, which is unchanged. **Never start a real stage with
`scripts/run_stage.sh` directly**, because C's runner only locks C's own jobs (`tmp/c_job.lock`). The code is frozen,
so this rule lives in the chain (`../chain/README.md`) and not in C's code; this README is not a tracked file of the
frozen record.

| Plan step (section 6) | Command (from `studies/language_part3/`) | Env | Time (estimate) | Decides |
|---|---|---|---|---|
| 2 | `chain/part3_cpu_job.sh C step0` | .venv-tm | a few minutes | nothing (exploratory) |
| 6 | `chain/part3_cpu_job.sh C prep` | .venv-tm | 2-5 min | (checks only) |
| 6 | `chain/part3_cpu_job.sh C twin` | .venv-tm | 5-15 min | C-K0, A/A, shuffled twin |
| 6 | `chain/part3_cpu_job.sh C learners` | .venv-dl | 2-6 h (LR about 1-1.5 h, LightGBM about 1-4 h; the code review's worst case for LightGBM is about 8 h: 9 s per round x 400 rounds x 8 trials) | LR/LightGBM picks, shuffled LR/LightGBM |
| 6 | `chain/part3_cpu_job.sh C gates` | .venv-tm | 1-2 min | **C-K0, C-K0b, A/A, shuffled: `data/step6_decision.json`** |

Exit codes through the wrapper (the same meaning as for A):
- 0: done, go on;
- 3: the real gates stage wrote a "decision" that starts with "STOP". C's own stages exit 0 on a gate STOP, so the
  wrapper reads `data/step6_decision.json`, and only a file written during that job;
- 4: the imported folders changed. A change confined to ".." (parent folder) lines is logged as a note and gives 0
  (review issue 6);
- 75: not run, because C's own lock was busy (C's runner exits 3 then); run it again later;
- 2: usage error; anything else: the stage failed.

Read an A/A or shuffled-label stop next to the `twin_AA - twin` interval and the class-unigram diagnostic printed
beside it (review issue 10).

- Each real stage needs the frozen record and refuses to start if a tracked file changed since the freeze,
  if a finished real run of the same stage exists (no second look at valid), or if an earlier stage's
  output differs from what its run wrote.
- The runner waits for the CPU gate (`cpu_load.py 10`: at most 4.0 cores and 0 throttled periods, checked every
  2 minutes; `runs/gate_log.tsv`), runs `python -I -B` at nice 19 with 2 threads and every cache and temp folder
  in `tmp/`, lists the imported folders before and after (`runs/listings/`) and exits 4 on any change. Job starts
  and ends: `runs/jobs_log.tsv` (the gate-open hours of C-K4).
- One Part 3 CPU job at a time across A and C: the chain wrapper's lock (above); the runner itself only locks C's own
  jobs (`tmp/c_job.lock`).
- A crashed stage may be rerun unchanged (the code is deterministic). Any code change needs a dated amendment line.
- Step 9 (the TM on the GPU, C0, the epoch-1 control) has code since 2026-10-23 (section "Step 9" below). Its dated
  amendment (every file with its SHA-256, and the commands) was written but never frozen, so step 9 has not run. Until `scripts/freeze.py --amend` has run and the amendment line with the
  new record's SHA-256 is in `../LOG.md`, every C stage, dry or real, refuses to start (the frozen record no longer
  matches `ccore/hashing.py` and `scripts/freeze.py`). Step 10 (tree twin, untrained banks, line R2, the test read)
  still needs code; its grids and bars are fixed in R1.
- Code review of 2026-10-21: issue 7 (an `--amend` freeze like A's: a new record kept with its predecessor in
  `history/`, the record's hash required in LOG.md, real stages refusing tracked files that are not in the record) is
  in the step-9 amendment; issue 6 (".." listing lines are a note) holds in the new GPU runner, not in
  `scripts/run_stage.sh`; issues 8 (negative control of the future-change test) and 10 (an equal-size A/A rebuild,
  reported with no rule) are still open. None of these changes a registered number.

## Step 9 (GPU; code under the step-9 amendment, not frozen yet)

Every GPU stage starts as `C/scripts/run_gpu_stage.sh <stage>` (from anywhere), in this order, each only after the
previous one decided PASS. The runner refuses to start (exit 2) without the go file `C/.gpu_go` (no code
here creates it), refuses (exit 75, try later) while `nvidia-smi --query-compute-apps` lists any GPU compute process
or C's lock `tmp/c_job.lock` is held, and exits 2 when the process list cannot be read. It runs python -I -B of
.venv-dl (torch 2.6.0+cu124) at nice 19 with 2 threads, CUDA_VISIBLE_DEVICES=0 and every cache in `tmp/`, lists the
imported folders before and after (exit 4 on a change; ".."-only changes are a note), and logs to `runs/jobs_log.tsv`;
each training run and each stage's wall time also go to `runs/gpu_time.tsv` (C-K3). No CPU gate inside: the GPU
queue checks `cpu_load.py` before it starts a job.

| Stage | What | GPU minutes (estimate) | Decides |
|---|---|---|---|
| `tm_smoke` | smoke TM (10% of the train blocks, subset seed 1, 2 epochs, T 8,000, s 10, seed 1); shuffled-label TM at smoke scale; one timed CPU epoch (655 batches, 2.7-2.9 min of the slot, estimate); C-K3 projection | 3.1-3.9 | C-K2, C-shuffled (TM), C-K3 |
| `tm_grid` | T {8,000, 16,000} x s {3, 10, 30, 90}, seed 1, 20 epochs each; valid scored once per trial after all 8 are trained; the pick; the pick's epoch-1 checkpoint | 22.7-35.9 | pick, C-K1, C-K3 |
| `tm_seeds` | the pick at seeds 2-10, with their epoch-1 checkpoints | 25.7-40.7 | C-K3 |
| `tm_c0` | C0: the pick on 1% of the train positions (subset seed 1), seeds 1-10; then the 10-seed collection (`data/tm/step9_valid.npz`) | 3.3-10.7 | C-K3 (measured total) |

GPU minutes are ESTIMATES (`tmp/step9_build/gpu_estimate.py` on G1's measured 4,000-clause runs and this
build's CPU probes): 54.8-91.2 in all, against the plan's 50-115 and C-K3's line of 230;
the epoch-1 control has no training of its own (0.3-0.5 min of scoring inside `tm_grid` and `tm_seeds`). Measured times go to `runs/gpu_time.tsv`.

Exit codes: 0 PASS; 3 a registered STOP (C-K2, C-shuffled, C-K1) or a C-K3 HOLD (read `data/tm/<name>_decision.json`;
the next stage then refuses); 4 the imported folders changed; 2 refused or usage; 75 busy; anything else: the stage
failed or refused (its run folder's `log.txt`). A queue that retries a non-zero exit once gets exit 1 on the retry (a
finished real run exists, or the previous stage did not decide PASS); nothing runs twice. `--dry` runs the same
stages on `data_dry/` (pseudo splits of train blocks; the registered machine with 2 epochs, seeds 1-3, C0 seeds 1-2;
nothing enforced; a GPU dry run of all four: 1.7-3.9 min, estimate).

## Tests

- `scripts/run_stage.sh selftest_tm` and `scripts/run_stage.sh selftest_dl` (under 2 minutes, no CPU gate).
- Step 9: `bash tests/run_step9_tests.sh` (refuses unless `cpu_load.py` shows at most 4.0 cores): the unit tests
  `tests/selftest_step9.py`, the end-to-end smoke `tests/selftest_step9_e2e.py` (synthetic data in test roots under
  `tmp/step9_build/`, the CoTM on the CPU, the GPU hidden, a fake go file and a fake GPU process list, every stage
  through `scripts/run_gpu_stage.sh`), and `selftest_tm.py` / `selftest_dl.py` on the changed code, with the
  environment of `run_stage.sh` but writing nothing under `runs/`. About 2.5 minutes; outputs in
  `tests/out/step9_*_<UTC>.txt`.
- Dry runs: the same five stages with `--dry` (pseudo splits from whole train blocks only; synthetic T0 bits in Step
  0; at most 120 s each, no CPU gate). Outputs go to `data_dry/` and `runs/*_dry`.

## Files

- `ccore/`: paths and run rules (`paths.py`, `env.py`), hashing and the frozen-record check (`hashing.py`), run
  records (`runlog.py`), PTB splits (`data.py`), floor F (`floor.py`), the 60 bits (`layout.py`), trees E and B
  (`codes.py`), counted chains, T-E and PairKN (`chains.py`), the 8 twin trials (`twin.py`), members and systems
  (`systems.py`), LR and LightGBM (`learners.py`), gate and bar arithmetic (`gates.py`), the seed-inclusive bootstrap
  (`boot.py`), Step 0 (`step0.py`).
- `ccore/tmcore.py` (numpy side of step 9: settings, read-out, picks, C-K2, C-K3, subsets, systems, seed means, the
  TM run record) and `ccore/tmtorch.py` (our CoTM loaded from `dl/tm.py` after its full SHA-256 check, the training
  loop, class sums, the packed CPU check through `dl/cpu_tm_fast.py`).
- `scripts/`: the five CPU stages, their runner, `freeze.py` (`--amend` since step 9); the four GPU stages
  `tm_smoke.py`, `tm_grid.py`, `tm_seeds.py`, `tm_c0.py` and their runner `run_gpu_stage.sh`.
- `tests/`: the self-test files (`selftest_tm.py`, `selftest_dl.py`, `selftest_step9.py`, `selftest_step9_e2e.py`,
  the synthetic-data helper `step9_synth.py`, the runner `run_step9_tests.sh`); `tests/out/`: their saved outputs.
- `history/`: earlier versions kept byte for byte (`README.v1.md`, `README.v2.md`, and the pre-amendment
  `scripts_freeze.py.v1` and `ccore_hashing.py.v1`, equal to the hashes in the first frozen record).
