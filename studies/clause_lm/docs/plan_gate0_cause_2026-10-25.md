# Dev plan: why Gate 0 of M5 step 1 failed with the numba backend (validation split only)

- **Status:** frozen 2026-10-25T12:59:23Z with `scripts/freeze.py` (record `plan_gate0_cause_2026-10-25.frozen.json`),
  before any counting run. Written on 2026-10-25 from 12:53:10Z to 12:57:30Z, after a pre-freeze review (judged by an AI
  model, not a person; READY AFTER FIXES). This published copy leaves out the draft history and private paths and is
  edited in wording only; the registered text's SHA-256 is in `docs/hash-table.md`.
- **Kind:** a diagnosis on the CLINC150 validation split. It is not a registration and makes no claim about the
  registered question of M5 step 1, whose verdict (NEGATIVE, [results page](results.md)) it does not touch.
- **Test split:** not read. The driver loads only `Xtrain`, `ytrain`, `Xval` and `yval` of the feature file and
  records the keys it read; the file is hashed whole, test arrays included, but no test array is loaded. The test
  split of CLINC150 stays as it is: 13 runs of this study have scored it, each once (the numba Gate 0 run of 2026-10-02
  and the 12 runs of 2026-10-14).
- **Statistics:** each rule stands alone; no multiplicity correction is applied and no superiority claim is made. There
  is no interval: the signatures are categorical, one per seed, and every seed is reported. The checker prints one
  table of every run and one outcome line; there is no combined line beyond the outcome table.

## Question

Gate 0 with the numba backend gave validation accuracy 0.5493 (test 0.5407) and a model whose 4,000 clauses hold only
negated literals (median 1,021 literals), while the same settings with the torch backend on a GPU gave validation
accuracy 0.8187 and clauses with a median of 1 literal. The reading of 2026-10-02: for the sampled negative class with
a clause weight <= 0, the numba kernel gives Type I feedback only to clauses that fired (`coalesced.py` lines 173-188,
an `if cj == 1:` with no `else`), while the torch kernel also erodes the clauses that did not fire (`torch_backend.py`
lines 130-131, `g1nf`, and 154, `cnt_dec`). Does that one difference make the degenerate model?

## Setup

- **Driver:** `src/gate0_cause.py`. It imports Clause-Driven-LLM read-only from `work/Clause-Driven-LLM/src` and refuses
  to run unless the files have the pinned SHA-256. Provenance of the pins: `coalesced.py` equals the `coalesced_sha256`
  that the configs of all 15 M5 training runs record (2026-10-02 and 2026-10-14). `torch_backend.py` equals the
  reference copy (`4828a9b5...`), but no run ever hashed it: the setup run `20261002T210555Z_m5_setup_torchcpu` recorded
  only `torch_backend_identical_to_reference: true`, twelve days before the torch runs, so arm A0 is the empirical check
  of the torch path. The whole `cdllm` tree (22 `.py` files, `binarize.py`, `seeding.py`, `utils/io.py` and the package
  `__init__.py` files included) must also match one manifest digest pinned in the driver and in the checker
  (`5c648da62b8b760ac4a0edfd3adeb11e555993c3f73a14f78b65d560ee094dab`; `src/gate0_cause.py --manifest` prints it).
- **Variants:** built from the source text of one upstream function with one exact replacement or insertion, compiled
  in a copy of the module's namespace and put in place of the module attribute only while `fit()` runs. No upstream
  file is edited.
  - `torch` (arm A0): the unmodified torch backend.
  - `torch_noneg` (arm A): torch with `g1nf = (sel_pos & (Wt >= 0)) & (~co)`, the negative-class part removed.
  - `numba_erode` (arm B): numba with an `else:` after the negative-class Type I block that erodes every literal
    with probability 1/s, a copy of the target-class block of `coalesced.py` lines 141-145.
  - A counting wrapper stands in the module attribute for every variant (around the original function for `torch`),
    and the run records how often it was called (`variant_calls` in `metrics.json`).
- **Settings:** the locked settings of M5 step 1: lexical features with 2 bins, 4,000 clauses, T 4000, s 8, 256
  states, 40 epochs, batch 128, hardest negative. The calls are those of the patched `scripts/train_controller.py`:
  `seed_all(seed)`, `Booleanizer(n_bins=2)` fitted on train, `fit(..., X_val, y_val, eval_every=2)`.
- **Measures:** `val_acc`, the validation accuracy of the best-validation snapshot (the quantity that the Gate 0 runs
  recorded as `val_accuracy`), and the literal statistics of that snapshot: clauses with at least one positive literal,
  median literals per clause. Selection caveat: `val_acc` is the best of 21 evaluations (epochs 0, 2, ..., 38 and 39) on
  the same split, so it is optimistic, and the signature is read at that snapshot, which can be any evaluated epoch.
  The same holds for the recorded numbers that the bars come from.
- **Also recorded per run:** the upstream hashes and the manifest, the variant's diff and its SHA-256, the package
  list of the environment and its digest, the versions of numpy, numba, scipy, scikit-learn and torch, the thread
  settings and the nice value.
- **Checker:** `src/gate0_cause_report.py` reads the freeze record of this plan and counts a run folder only when its
  stamp is after the freeze, its copies of `gate0_cause.py` and `runlog.py` have the pinned SHA-256 (the freeze record
  must pin the same), its config holds the pinned upstream hashes, manifest digest, feature-file hash and package
  listing of the arm's environment, only the four arrays were read, the variant diff has the expected SHA-256, the
  device is `cuda` for the torch arms and `cpu` for numba, variant and seed match the folder name, and the variant's
  function was called (replaced for A and B, not replaced for A0). It refuses a second counting run of one variant and
  seed, prints the reason for every folder it does not count, applies the rules and prints one outcome line. The
  pinned values are literals in the checker; `tests/test_gate0_cause.py` checks them against the files and
  environments, and `tests/test_gate0_cause_report.py` checks every refusal and every outcome row.
- **Freeze:** `.venv-cdllm/bin/python scripts/freeze.py docs/plan_gate0_cause_2026-10-25.md` writes
  `docs/plan_gate0_cause_2026-10-25.frozen.json` with the UTC time, the SHA-256 of this text, every pin of the hash
  block below (each hashed again; the tool refuses when one differs) and the study files of `src/`, `scripts/` and
  `tests/`, and never overwrites a record. `--check` verifies the pins and writes nothing.
- **Tests that passed before this revision was written:** all 15 tests of the study, run from 12:50:19Z to
  12:51:47Z (the times in the outputs `final2_<test>.out` in the records of the pass, each ending with PASS and
  exit 0); among them `tests/test_gate0_cause.py`, `tests/test_gate0_cause_report.py`, `tests/test_launch_gpu.py`
  and `tests/test_freeze.py`, whose outputs are pinned below.

## What runs (from `studies/clause_lm`)

Before any counting run, no GPU:

```
.venv-cdllm/bin/python scripts/freeze.py --check docs/plan_gate0_cause_2026-10-25.md
.venv-cdllm/bin/python scripts/freeze.py docs/plan_gate0_cause_2026-10-25.md
CLM_DRY_RUN=1 CLM_JOB_PYTHON=$PWD/.venv-cdllm-gpu/bin/python src/launch_gpu.sh 15 src/gate0_cause.py --variant torch --seed 42
```

The check must pass; if a pinned file changed (for example `../tm_language_model/env/env.sh`, which another agent of
the pass edited today), this revision cannot be frozen and a new revision re-pins it. The dry line prints the chain;
its output must have the pinned SHA-256 (`dry_run_gate0_A0.out` below).

GPU, one job at a time, each started when the GPU is free (the 15 minutes of the timeout include the wait for the GPU
lock). `src/launch_gpu.sh` runs the whole chain at nice 19 with the OpenMP, OpenBLAS, MKL, numexpr, numba and rayon
pools at 2 threads (timeout, memory guard of 12 GiB, `gpu_run.sh exclusive`):

```
CLM_JOB_PYTHON=$PWD/.venv-cdllm-gpu/bin/python src/launch_gpu.sh 15 src/gate0_cause.py --variant torch_noneg --seed 7 --epochs 2
CLM_JOB_PYTHON=$PWD/.venv-cdllm-gpu/bin/python src/launch_gpu.sh 15 src/gate0_cause.py --variant torch --seed 42
CLM_JOB_PYTHON=$PWD/.venv-cdllm-gpu/bin/python src/launch_gpu.sh 15 src/gate0_cause.py --variant torch_noneg --seed 42
CLM_JOB_PYTHON=$PWD/.venv-cdllm-gpu/bin/python src/launch_gpu.sh 15 src/gate0_cause.py --variant torch_noneg --seed 123
CLM_JOB_PYTHON=$PWD/.venv-cdllm-gpu/bin/python src/launch_gpu.sh 15 src/gate0_cause.py --variant torch_noneg --seed 456
```

The first line is a smoke run of 2 epochs with seed 7, which no counting run uses, so it previews no counting run; its
folder does not have the locked settings and the checker ignores it. Its numbers are reported with the results as
known before the counting runs. If it shows a bug of the driver, no counting run starts: the bug is fixed and tested,
and a dated note in a new file, frozen with the same tool, lists the change and the smoke's numbers before any
counting run. Each full run is expected to take about a minute: the 12 torch runs of 2026-10-14 took 47.0 to 76.3 s
of process wall time, test scoring included.

CPU, optional (arm B), started only when the CPU gate shows at most 4.5 cores, first a smoke run with tiny settings
(not counted), then the counting run under a timeout and the memory guard:

```
cd ../decision_layer && .venv-dl/bin/python -I scripts/cpu_load.py 10 && cd ../clause_lm
nice -n 19 env CUDA_VISIBLE_DEVICES= NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 .venv-cdllm/bin/python src/gate0_cause.py --variant numba_erode --seed 7 --device cpu --epochs 1 --clauses 200 --T 200
nice -n 19 env CUDA_VISIBLE_DEVICES= NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout --kill-after=60 180m .venv-cdllm/bin/python -I -B ../tm_language_model/scripts/mem_guard.py --scope group --limit-gib 12 -- .venv-cdllm/bin/python src/gate0_cause.py --variant numba_erode --seed 42 --device cpu
```

The unmodified numba run of 2026-10-02 took 3,419.9 s at 2 threads; the added branch adds erosion work, so 60 to 120
minutes is expected (UNVERIFIED estimate). Arm B has one seed because of this CPU cost, and its reading holds for seed
42 only.

After the runs:

```
CUDA_VISIBLE_DEVICES= .venv-cdllm/bin/python src/gate0_cause_report.py
```

It prints one line per counting run, every problem, the three verdicts, whether A0 reproduces 0.8186666666666667
exactly, and the OUTCOME line, and writes them into a run folder `results/<UTC>_gate0_cause_report/` (`report.md`,
`metrics.json`). That folder is the report of this check; before the runs, `--no-log` shows the checker's state
without writing a folder. Then an independent recompute, outside the
driver's code, of each counting run's `val_acc` from its `tm.npz` (a clause fires when all its included literals are 1
and it includes at least one; the class is the argmax of the summed weights of the firing clauses; Booleanizer with 2
bins fitted on `Xtrain`, applied to `Xval`) and of its literal statistics, compared with `metrics.json`; then a
check for leakage, selection and integrity.

**Rerun rule.** A run that ends without `metrics.json` (a timeout, which includes the wait for the GPU lock, or a
crash) may be rerun once with the same command. The checker lists the unfinished folder; a second unfinished run of
the same variant and seed leaves that key MISSING. A run refused by a check (a wrong file, device or environment) is
listed and does not block a later run with the right setup; the first run that passes every check counts.

## Rules, fixed before the first run

Signatures of a run (snapshot statistics; 4,000 clauses):

- **degenerate:** `val_acc` <= 0.70, at most 400 clauses with a positive literal, median literals >= 100;
- **healthy:** `val_acc` >= 0.78, at least 2,000 clauses with a positive literal, median literals <= 10;
- anything else is neither.

The two Gate 0 models lie far on either side: numba 0.5493, 0 clauses with a positive literal, median 1,021; torch
(seed 42) 0.8187, 2,954, median 1. The torch runs at s 8 span 0.8130 to 0.8187 in validation accuracy over seeds 42,
123 and 456.

- **Driver check (A0, seed 42):** MET when `val_acc` is within 0.01 of the recorded 0.8186666666666667 and at least
  2,000 clauses have a positive literal; NOT MET otherwise; MISSING without a counting run.
- **Arm A (seeds 42, 123, 456):** CONFIRMED when all three runs are degenerate (removing the negative-class erosion
  from the torch kernel reproduces the numba model); REFUTED when all three are healthy; MIXED otherwise; MISSING when a
  seed has no counting run.
- **Arm B (seed 42, optional):** CONFIRMED when the run is healthy; REFUTED when it is degenerate; MIXED otherwise; NOT
  RUN when no folder with the locked settings exists; MISSING when one exists (also one from before the freeze) and
  none counts.

**Outcome, the first row that applies.** The checker prints it as one line, `OUTCOME <row>: <text>`:

| Row | Condition | Statement |
|---|---|---|
| 1 | driver check NOT MET or MISSING | every number is reported; no cause statement |
| 2 | else arm A MIXED or MISSING, or arm B MIXED or MISSING | every number is reported; no cause statement |
| 3 | else arm A or arm B REFUTED | the missing branch alone does not explain the failed Gate 0 |
| 4 | else arm A and arm B CONFIRMED | the missing branch accounts for the degenerate model in both kernels (validation split; seeds 42, 123, 456 for torch; seed 42 for numba) |
| 5 | else (arm A CONFIRMED, arm B NOT RUN) | removing the branch from torch reproduces the numba model (validation split; seeds 42, 123, 456); the numba side stays a reading of the code |

The rows cover every case: after rows 1 to 3 the driver check is MET, arm A is CONFIRMED and arm B is CONFIRMED or NOT
RUN.

**Readings with no rule, printed beside the rules:** the structure of every counting run (clauses with a positive
literal, median literals) beside its signature, so that a run that is structurally healthy but below 0.78 in accuracy
(as the torch models at s 25 were, see below) is visible as such; and whether A0 reproduces 0.8186666666666667 exactly.

**Expected, written before any run:** the driver check MET, by construction (A0 repeats the recorded torch path for
seed 42 through the same calls; a GPU run need not be bit-identical, hence the band); arm A degenerate on all three
seeds, with validation accuracy near 0.55; arm B healthy, near 0.82.

## What each outcome leads to

- **Rows 1 and 2:** every number is reported on the results page, without a cause statement; this plan runs nothing
  further.
- **Row 3:** the results page says that the branch alone does not explain the failure. The other known differences
  remain candidates: online against batch-accumulated updates (the histories below differ from the first epoch on),
  batch-accumulated Type II feedback, and weight 0 counted as Type I by numba and as no feedback by torch. This plan
  runs nothing further.
- **Row 4:** the diagnosis section of the results page states the row's sentence with these runs. The upstream bug (a
  missing `else` in `coalesced.py`, and docstrings that call the two backends equivalent) can be reported to the
  upstream repository; that is the author's decision, and no file of the upstream copies changes here.
- **Row 5:** the results page states the row's sentence; the numba side stays a reading of the code.
- In every case the registered verdict of M5 step 1 and its test numbers stay as they are.

## Not in scope

The test split; any change to the Clause-Driven-LLM copies; values of s other than 8; label orders other than sorted;
the registered question of M5 step 1; an unmodified numba run through the driver (B0, about an hour of CPU, not
planned: arm A0 checks the driver's data path).

## What was known at writing

- The four recorded runs (`metrics.json` and `train_metrics.json` of `20261002T195255Z_m5_gate0_s8_seed42_sorted`,
  `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu`, `20261014T221627Z_m5_s8_seed123_sorted_torchgpu` and
  `20261014T221728Z_m5_s8_seed456_sorted_torchgpu`):

  | Run | Validation accuracy (best snapshot) | Best epoch | Test accuracy | Median literals |
  |---|---|---|---|---|
  | numba, seed 42 | 0.5493333333333333 | 20 | 0.5406666666666666 | 1,021 |
  | torch, seed 42 | 0.8186666666666667 | 39 | 0.8293333333333334 | 1 |
  | torch, seed 123 | 0.813 | 34 | 0.8064444444444444 | 1 |
  | torch, seed 456 | 0.816 | 39 | 0.8244444444444444 | 1 |

- Their validation histories (21 evaluations each): numba 0.354 at epoch 0, 0.456 at 2, between 0.50 and 0.55 from
  epoch 6 on, best 0.5493 at epoch 20, 0.5413 at epoch 39; torch seed 42 0.0163 at epoch 0, 0.489 at epoch 10, rising
  to its best, 0.8187, at epoch 39, the last evaluation; seed 123 0.022 at epoch 0, best 0.813 at epoch 34; seed 456
  0.0133 at epoch 0, best 0.816 at epoch 39. The two kernels thus differ from the first epoch on, beyond the branch
  under test; the signatures concern the end state.
- Two Gate 0 folders hold no score: the retry `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` (stopped, return
  code -15 after 667.3 s, before any evaluation) and the first attempt `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu`
  (exit 127 at its start, no `python3` for the memory guard).
- The structure of all 13 saved models of M5 step 1, described on 2026-10-25 with `src/gate0_cause.py --describe`
  (`describe_saved_models.out`, pinned below; no split read):

  | Model | Clauses with a positive literal (of 4,000) | Median literals | Validation accuracy (recorded) |
  |---|---|---|---|
  | numba, s 8, seed 42 | 0 | 1,021 | 0.5493 |
  | torch, s 8, seeds 42, 123, 456 | 2,954, 2,939, 2,911 | 1, 1, 1 | 0.8187, 0.8130, 0.8160 |
  | torch, s 25, seeds 42, 123, 456 | 2,885, 3,046, 2,974 | 2, 2, 2 | 0.7777, 0.7740, 0.7777 |
  | torch, s 50, seeds 42, 123, 456 | 2,918, 2,963, 2,872 | 3, 3, 3 | 0.7337, 0.7417, 0.7430 |
  | torch, s 100, seeds 42, 123, 456 | 3,216, 3,244, 3,253 | 4, 5, 5 | 0.7233, 0.7170, 0.7207 |

  Every torch model is structurally healthy, also those below 0.78 in accuracy; only the numba model is degenerate.
- The code reading of 2026-10-02 and of the inventory and the review of 2026-10-25.
- The toy runs inside `tests/test_gate0_cause.py` (40 clauses, 4 epochs, made-up data) say nothing about CLINC150.
- The review's dry run of the job line (12:31:08Z) printed the chain only.
- No run of any variant on CLINC150 exists, and no number of this check was seen before writing.

## Code since the last freeze

The study's only earlier frozen text is the registration of M5 step 1 with its amendment of 2026-10-02 21:11 UTC (no
freeze record file; its times are those written in it). Since then, on the path of this check:

- `src/launch_gpu.sh` was rewritten on 2026-10-25 (SHA-256 `7c0f8384...` before, pinned below after; the diff is at
  the end of this section).
- `src/gate0_cause.py`, `src/gate0_cause_report.py` and `scripts/freeze.py` are new.
- `src/runlog.py` is unchanged: byte identical to its copies in the run folders of 2026-10-14.
- In tm_language_model, `scripts/gpu_run.sh`, `scripts/lib_locks.sh` and `scripts/mem_guard.py` are unchanged since
  commit 61f590a (2026-10-11), before the runs of 2026-10-14. `env/env.sh` was changed on 2026-10-25 by another agent
  of this pass: it now puts `.venv-tm/bin` of tm_language_model first on PATH when that environment exists. The runs of
  2026-10-14 did not source `env/env.sh`.
- The upstream copy: `coalesced.py` has the recorded hash; the whole `cdllm` tree is pinned by the manifest.

How the chain differs from the one of the 13 registered runs (2026-10-14: `research/roadmap/run_gate0.sh` of the
project folder starting `src/m5_run.py --backend torch_gpu`):

| | Runs of 2026-10-14 | This plan |
|---|---|---|
| Entry | `src/m5_run.py` (in `.venv-cdllm`) starts `scripts/train_controller.py` of the work copy (second patched state) | `src/launch_gpu.sh` starts `src/gate0_cause.py` |
| Interpreter of the training | `.venv-cdllm-gpu` | `.venv-cdllm-gpu` |
| Timeout | 15 min | 15 min |
| nice | 19, for the training process only | 19, for the whole chain |
| Thread pools | NUMBA, OMP, MKL, OPENBLAS at 2 | the same, and NUMEXPR and RAYON at 2 |
| Memory guard | `mem_guard.py` started by its first line, under a `python3` link placed first on PATH | `mem_guard.py` under the job's interpreter with `-I -B` |
| `env/env.sh` | not sourced | sourced: CUDA paths, thread caps, `PYTHONHASHSEED=0` unless set, `.venv-tm/bin` first on PATH |
| `CUDA_VISIBLE_DEVICES` | `0` | inherited from the caller (the machine has one GPU) |
| Working folder | the work copy of Clause-Driven-LLM | `../tm_language_model` |
| Test split | scored once per run | not loaded |

The diff of `src/launch_gpu.sh` (`diff -u` of the file before and after 2026-10-25; it is also kept in the records of
the pass as `launch_gpu.sh.diff`, pinned below):

```diff
--- a/src/launch_gpu.sh
+++ b/src/launch_gpu.sh
@@ -1,13 +1,53 @@
 #!/usr/bin/env bash
 # Run one GPU job for clause_lm (study rule 3: thesis first).
 # Waits for the shared GPU lock, then holds it EXCLUSIVELY, so it never overlaps a thesis job:
-#   timeout -> mem_guard (12 GiB) -> gpu_run.sh exclusive -> .venv-htm python
+#   nice 19 -> timeout -> mem_guard (12 GiB) -> gpu_run.sh exclusive -> job python
 # usage: launch_gpu.sh MINUTES SCRIPT.py [args...]
+#   MINUTES    whole minutes before timeout stops the job (60 s later it is killed); the wait for the GPU lock
+#              counts, so start a job when the GPU is free
+#   SCRIPT.py  an existing file; a relative path is read from the caller's working directory
+#
+# Changed 2026-10-25 (finish pass). The first version ran `.venv-htm/bin/python` inside ../tm_language_model,
+# which has no such environment, and started scripts/mem_guard.py by its `#!/usr/bin/env python3` line, which
+# fails with exit 127 when no python3 is on PATH. Both interpreters are now explicit:
+#   CLM_JOB_PYTHON    interpreter of the job. Default: ../decision_layer/.venv-htm/bin/python, which holds
+#                     PyHierarchicalTsetlinMachineCUDA 0.3.2 at commit 5b4a223, the commit src/pair_tm.py names.
+#                     It belongs to the decision_layer study and is used read-only: nothing is installed into it.
+#   CLM_GUARD_PYTHON  interpreter that runs scripts/mem_guard.py (standard library only, Python >= 3.11).
+#                     Default: CLM_JOB_PYTHON. It runs with -I -B (isolated, no bytecode written).
+#   CLM_GPU_RUN       the GPU lock wrapper. Default: scripts/gpu_run.sh of tm_language_model. Only the tests
+#                     replace it, with a stub that takes no lock.
+#   CLM_DRY_RUN=1     print the command, one argument per line, and exit 0 without starting it.
+# Every job runs at nice 19 with the thread pools of OpenMP, OpenBLAS, MKL, numexpr, numba and rayon at 2 (set after
+# env/env.sh, which caps them at 6 only): 2 threads and nice 19 (added 2026-10-25 after review).
+# Exit 2: usage error, or an interpreter, script or wrapper that does not exist.
 set -euo pipefail
+die() { echo "launch_gpu.sh: $*" >&2; exit 2; }
+[ "$#" -ge 2 ] || die "usage: launch_gpu.sh MINUTES SCRIPT.py [args...]"
 MIN="$1"; shift
+case "$MIN" in
+  '' | *[!0-9]*) die "MINUTES must be a whole number of minutes, got '$MIN'" ;;
+esac
+[ "$((10#$MIN))" -ge 1 ] || die "MINUTES must be at least 1, got '$MIN'"
+MIN="$((10#$MIN))"
+[ -f "$1" ] || die "job script not found: $1"
+SCRIPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
 STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
-cd "$STUDY_DIR/../tm_language_model"
+TMLM="$(cd "$STUDY_DIR/../tm_language_model" && pwd)"
+JOB_PY="${CLM_JOB_PYTHON:-$STUDY_DIR/../decision_layer/.venv-htm/bin/python}"
+GUARD_PY="${CLM_GUARD_PYTHON:-$JOB_PY}"
+GPU_RUN="${CLM_GPU_RUN:-$TMLM/scripts/gpu_run.sh}"
+[ -x "$JOB_PY" ] || die "job interpreter not found or not executable: $JOB_PY (set CLM_JOB_PYTHON)"
+[ -x "$GUARD_PY" ] || die "guard interpreter not found or not executable: $GUARD_PY (set CLM_GUARD_PYTHON)"
+[ -f "$TMLM/scripts/mem_guard.py" ] || die "memory guard not found: $TMLM/scripts/mem_guard.py"
+[ -x "$GPU_RUN" ] || die "GPU lock wrapper not found or not executable: $GPU_RUN"
+cd "$TMLM"
 source env/env.sh
-export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
-exec timeout --kill-after=60 "${MIN}m" scripts/mem_guard.py --scope group --limit-gib 12 -- \
-  scripts/gpu_run.sh exclusive -- .venv-htm/bin/python -u "$@"
+export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2
+cmd=(nice -n 19 timeout --kill-after=60 "${MIN}m" "$GUARD_PY" -I -B "$TMLM/scripts/mem_guard.py" --scope group --limit-gib 12 --
+     "$GPU_RUN" exclusive -- "$JOB_PY" -u "$SCRIPT" "$@")
+if [ "${CLM_DRY_RUN:-0}" = 1 ]; then
+  printf '%s\n' "${cmd[@]}"
+  exit 0
+fi
+exec "${cmd[@]}"
```

## Hashes (sha256sum, read 2026-10-25T12:54:58Z; paths from `studies/clause_lm`: first the repository files, then the local work copy, reference copy and data (`work/`, `references/`, not distributed), then, by absolute path, test outputs and other records of the finish pass, outside the repository)

```
5b01c9e9f8ada227c60319575e98a6f0dab6440eedde24a93d6d087c26a6ebd3  src/gate0_cause.py
d27adfe35b977d47be45dcad1bd4bc944c5f247ea66d2d60b980a9096c01d5fc  src/gate0_cause_report.py
864a5be754fbaf13c8c99ecfe38519891cc4f79372a3d420f6e57a46db132f74  src/launch_gpu.sh
d0b23bf35b801d8af2ca17c1a9623919506517053cdecd45c7aabd6e276a46be  src/runlog.py
12b3d17e7bc835d9339455c58f409a4ad8e94519b341f14bd245e21e860cba94  scripts/freeze.py
2fed9212a5a0ba96032309ac2e033cff59c382ff488f4d8205fca2ffc508b1fc  tests/test_gate0_cause.py
656f89e87aff89d8698532337af8738e312508127a52d4f7979d05825ceaff68  tests/test_gate0_cause_report.py
abe1b4af04997b93d31cdba8ede27249915beb16371ebcdd378e348f2ebc16b6  tests/test_launch_gpu.py
7d03a0a0bd0844449038c17a2f51a630e211b3e2f175e5f55ab74a21a58218e9  tests/test_freeze.py
1829b3f679079f4c9a0816d4536ebd04df9d2a3ae076f988d3de5b5a0f5d6afd  tests/toyutil.py
2bca37b6d8c0277be2084b258e11b0de88e103119c320350893bd2f4275d9ab9  ../tm_language_model/env/env.sh
1e24a8c8bb9441ae90bd077669ff2e7e620ded244a491307efcb6719f35aaeac  ../tm_language_model/scripts/gpu_run.sh
35b6e0cb158220c3aea09858bb4f09a18abd7f15b52da5bc35091918a3e4c6bc  ../tm_language_model/scripts/lib_locks.sh
33e7c2cd34143fc0d647584bb5dc7d1ea2d743ba6fb4eb98facd43f391dbd00c  ../tm_language_model/scripts/mem_guard.py
42b06c01906145fa363ec4c2877c4f20111147d945f335071521162580d64674  work/Clause-Driven-LLM/src/cdllm/control/controller.py
f01c3bd54880ee588303e7009ba7ded7f1cd4c4f84778c826b21475e84a99a37  work/Clause-Driven-LLM/src/cdllm/control/__init__.py
4683c552899bceeb7a45d02604bc20f18803d90e2da7f314582e20b65369f680  work/Clause-Driven-LLM/src/cdllm/control/policy.py
b851309171cb4d6d2615a84b90437a9c16a7f5d8e8812f62dfa11579b2a07e5e  work/Clause-Driven-LLM/src/cdllm/data/__init__.py
c538a9c90a8b2d57fd653257ca128e2044d93c1e140955f76288a74d9bcf1b90  work/Clause-Driven-LLM/src/cdllm/eval/__init__.py
8ffc69878e69eb293359ad1a3c37cd1db71c4a968ab39cc10d1255187de0f481  work/Clause-Driven-LLM/src/cdllm/eval/logger.py
8aa228f8e81ed87763e062b8b7c444fb8a432b8cca1e1179b98db8fbd1b672a6  work/Clause-Driven-LLM/src/cdllm/eval/metrics.py
523ad029be4a3f2a0ea2b7b8cd5fe21f0dd5226165dc4c443bdc379e615f9933  work/Clause-Driven-LLM/src/cdllm/features/binarize.py
d8350545c55af3d4007bd9205eaaa3313579bc6cc147fb4ead6ae7d0cb99d77c  work/Clause-Driven-LLM/src/cdllm/features/featurizer.py
1ae4af25a5505633be7e8c458a8d2d204ae6427089ac2fa2b75d8bc7e7fdcdb4  work/Clause-Driven-LLM/src/cdllm/features/__init__.py
368b36308f84b9da3e5451ca6098f97dae72bbc7f079cf761b2d22eb3ac68b34  work/Clause-Driven-LLM/src/cdllm/__init__.py
4ce557cc486c795bbeccd865f9bda740891db907c05565a59dbc3636f45b29ee  work/Clause-Driven-LLM/src/cdllm/llm/__init__.py
b34c802f46dd6e4dfe59301a163326ca1ce64477092745ab0beb763cb2429bee  work/Clause-Driven-LLM/src/cdllm/llm/ollama_client.py
bfffb131874cbc0b6edff7e1152d25ae612df69a24c66cf66b26ecd46ca3ef2d  work/Clause-Driven-LLM/src/cdllm/tm/coalesced.py
50f657c800de2ffa0e931ac47750fd7b58339744a24da4cfabeda1bcb2bf96a5  work/Clause-Driven-LLM/src/cdllm/tm/__init__.py
4828a9b515908793dbd5655d7eb3eed8b5ad306ab82cc2e96917db1669c69215  work/Clause-Driven-LLM/src/cdllm/tm/torch_backend.py
1062c43126cfb7d8082065fca4749df33058a721159a599019da7b9fe6bf1b75  work/Clause-Driven-LLM/src/cdllm/utils/__init__.py
63c26559e18f9d4739188a270b99eaedea183ca26ec0a9b7adbd01da68256ab3  work/Clause-Driven-LLM/src/cdllm/utils/io.py
927b50e5f3e4b4d6a9dbfcbd72275d7d0db8d87820a09dcab68a575d607ea264  work/Clause-Driven-LLM/src/cdllm/utils/seeding.py
fa70a7186675b44d17de6191dea5485b9dd5296861b5fe32a5820608e3b8f474  work/Clause-Driven-LLM/src/cdllm/verify/certificate.py
53d76eb4f8d9b26e765e01da8feff652ce0ea4d7840c9abd6e2b335b6655d8e7  work/Clause-Driven-LLM/src/cdllm/verify/__init__.py
85f82a230e19e5a73a691b12bc3dacbea4cebcd3e4d38cf73aa1df5991191071  work/Clause-Driven-LLM/src/cdllm/verify/sat_encoder.py
bfffb131874cbc0b6edff7e1152d25ae612df69a24c66cf66b26ecd46ca3ef2d  references/code/Clause-Driven-LLM/src/cdllm/tm/coalesced.py
4828a9b515908793dbd5655d7eb3eed8b5ad306ab82cc2e96917db1669c69215  references/code/Clause-Driven-LLM/src/cdllm/tm/torch_backend.py
15a987add2b590d8498768045d465167c808d8cbf8afd483330e89330444d0fc  work/m5/emb_sorted/clinc150__lexical.npz
f1183ee36a2a8568564ca4ce60d3dbd48f99fb71dadfc2d63a53027be495191e  work/m5/runs/sorted/s8_seed42/clinc150__lexical__seed42/tm.npz
5e2478a4c0a3a984773901d8c0ee371bf4fa74d1b103f20dbccaf9235608d75e  work/m5/runs/sorted_torchgpu/s8_seed42/clinc150__lexical__seed42/tm.npz
```

Not files, and pinned as literals: the `cdllm` manifest digest, in `src/gate0_cause.py` and
`src/gate0_cause_report.py`, `5c648da62b8b760ac4a0edfd3adeb11e555993c3f73a14f78b65d560ee094dab`; in
`src/gate0_cause_report.py` only, the package listing digests of `.venv-cdllm-gpu`,
`d82154d3cfb0116431768da93895624bacf6c259a9cca0f3a97b62547cb377f6`, and of `.venv-cdllm`,
`5ef3616698497037843caaf26f2c49a1884ada42e7419f379dbe4076185c82cd` (each printed by `<venv>/bin/python
src/gate0_cause.py --venv-listing`); and the SHA-256 of the variant diffs: `torch_noneg`
`18c6b91222e2cb4d1c4c174a3f6872407aedf38c607cad00597aafeb923d702c`, `numba_erode`
`7ac04c1ceeaa1f7b5e44626a52cfc93de23f32c2b32063ce4d2bd40c154fb652`, and for the unmodified variants the hash of an
empty text, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. `tests/test_gate0_cause.py` checks
each of them against the work copy and the two environments. The feature file's SHA-256 equals the one that the setup
run `20261002T195152Z_m5_setup_sorted` recorded.

