# launch: the chains of the overnight runs

This folder holds the shell scripts that started the long runs of the study between 25 Sep and 1 Oct 2026: the exploratory H-AO diagnostics, the confirmatory runs of the self-supervised family (R1, R2 and R4), and the reported R4 controls. A chain runs the stages of one experiment one after the other. Each stage is a logged run of its own. Most stages are started through the function `step` of `lib.sh`, which wraps the command in `scripts/mem_guard.py` and `scripts/run_logged.py` and, for a GPU stage, in `scripts/gpu_run.sh shared`. `chain_r1_confirm.sh` uses a function of its own with the same wrapping (`cstep`), and the R2 control stage of `chain_r2r4_confirm.sh` takes the exclusive GPU lock. The scripts are kept as the record of how the runs were started and in which order. What the runs showed is on the results page, [docs/results.md](../docs/results.md); the experiment scripts that the chains call are described in [experiments/README.md](../experiments/README.md).

The folder lies outside the five folders (`src`, `experiments`, `scripts`, `tests`, `env`) whose files the run logger hashes into `source_tree_hash`. The chains write logs and marker files next to themselves while runs are in progress, and inside a hashed folder those files would change the hash from one run to the next.

## The scripts

There are fourteen files. `lib.sh` is sourced by twelve of the others; `launch_hao_diag3.sh` is the one script that does not source it. In the chains with lanes, a lane is the argument that selects which part of the chain one invocation runs, and every lane is started as a process of its own.

| Script | What it starts | What it waits for |
|---|---|---|
| `lib.sh` | Not a chain. It changes to the study folder, sources `env/env.sh`, and defines `step` (one logged stage), `headroom` (the wait for free memory before a stage) and `latest` (the newest run folder of a given run name under `results/`). | nothing |
| `chain_hao_diag.sh [SEEDS]` | The exploratory H-AO hierarchy diagnostics (docs/13 A8 item 3; `experiments/ssl_r2_hao_diagnostics.py`) for world seeds 100 and 101: the selftest on the later commit of the hierarchical package, the demos for both package commits, per seed the valiant and controls stages, then the oracle, seen_unseen and pospair stages for both commits, the threshold stage for seed 100, and the report. | nothing; it stops at once when `.venv-htm` is missing |
| `chain_hao_diag2.sh [SEEDS]` | The same diagnostics in another order: valiant and controls first, the demos of the later commit last. Its report includes the demo run that the first chain started. | at most 6 h for that demo run, `results/20260925T091022Z_ssl_r2_hao_diag_demos_c85_exploratory`, to end |
| `chain_hao_diag3.sh [SEEDS]` | A rerun of the package stages of the diagnostics (selftest, oracle, seen_unseen, pospair, threshold, the demos of the later commit) and a new report, which also takes the clean valiant and controls runs of both seeds. | without a cap, until every lane of `chain_r1_confirm.sh` has printed its `DONE` line or is gone; then at most 6 h for the demo run named above |
| `launch_hao_diag3.sh` | Takes one CPU slot and executes `chain_hao_diag3.sh`. | the same condition on the lanes of `chain_r1_confirm.sh`, checked before any CPU slot is taken |
| `chain_r1_confirm.sh LANE` | The confirmatory R1 runs (docs/13 A7 family H17 with A8 and A9; `experiments/ssl_r1_confirm.py`). Lanes `seed1`, `seed2`, `seed3`: per arm the train, census, features and probe stages, the full-pool stages for two arms, and the confirm stage with its counted test access, then the qcon, qint and qpred stages. Lane `common`: the seed-independent controls, the random banks, and their confirm stages. Lane `report`: the report over every clean run. Each lane runs the testhash stage first unless a clean one exists. | every lane, without a cap: the exploratory control-study and configuration-arm chains, which were not retained (see below), unless the file `r1_confirm_release` exists; the seed lanes: at most 24 h for a clean common run before qcon; `report`: at most 72 h for the done marker of each of the four other lanes |
| `retry_confirm_common.sh` | The common confirm stage of the R1 chain again, with a memory limit of 80 GiB where the chain sets 24 GiB, then the R1 report. Its two input run folders are named in the script. | nothing |
| `rerun_c85_demos.sh` | The c85 demos stage of the diagnostics again, then the diagnostics report with the run list of `results/20260930T041535Z_ssl_r2_hao_diag_report_exploratory`, in which the new demo run replaces the one of 2026-09-25. | the line `RETRY DONE` in the log of `retry_confirm_common.sh`, or that script being gone |
| `chain_r2r4_confirm.sh LANE` | The confirmatory R2 runs (H17.5, `experiments/ssl_r2_hao.py`) and R4 runs (H17.10, `experiments/ssl_r4_images.py`). Lane `gpu`: the R4 GPU stage for seeds 1 to 3, the R2 GPU stage, then the R2 control under the exclusive GPU lock. Lanes `r4seed1` to `r4seed3`: the two probe stages of one seed. Lane `r2cpu`: the R2 CPU stage. Lane `report`: both reports. | `gpu`: the end of `rerun_c85_demos.sh` (its line `C85 RERUN DONE` or `c85 demos FAILED`, or the script being gone); `r4seedN`: a clean GPU run of its seed, or the done marker of `gpu`; `report`: the done markers of the other five lanes |
| `chain_r4controls.sh LANE` | The reported R4 learning-credit controls (docs/12 F29; `experiments/ssl_r4_controls.py`). Lanes `seed1` to `seed3`: the replay, twin, randbank and diag stages on the shared GPU lock, then the probe. Lane `report`: the controls report. | the done marker of lane `gpu` of `chain_r2r4_confirm.sh`; `report` also the markers of its own seed lanes and of the lanes `r4seedN` |
| `fix_r4controls.sh LANE` | First repair of the controls chain: the randbank stage again with a memory limit of 40 GiB, one seed at a time, then a new diag run, the probe and the report. Clean replay and twin runs are reused. | `report`: the done markers of its three seed lanes; the randbank stages take turns on a lock file |
| `fix2_r4controls.sh LANE` | Second repair: the probe stage with a memory limit of 32 GiB, then the report. | each seed lane: the done marker of the same lane of `fix_r4controls.sh`; `report`: the markers of its three seed lanes |
| `fix3_r4controls.sh LANE` | Third repair: the probe split by representation. Lanes `main1` to `main3`: the ladder and twin representations of one seed. Lane `rand`: the random-bank representation of seeds 1 to 3, one at a time with a memory limit of 78 GiB, then the report over every run. | `rand`: the done markers of the three main lanes |
| `cfg_arms_report.sh` | The report stage of the exploratory configuration-arm study (`experiments/ssl_r1_config_arms.py`) over the clean probe and qpred runs of seeds 100 and 101. | nothing |

A stage whose latest run of the same name exited 0 is not run again by `chain_r1_confirm.sh`, `chain_r2r4_confirm.sh`, `chain_r4controls.sh`, `fix2_r4controls.sh` (for its probe stage) and `fix3_r4controls.sh`, so a relaunch continues where the earlier launch stopped.

## The order of the launches

The waits link the scripts as follows: `chain_r1_confirm.sh` after the exploratory chains; `chain_hao_diag3.sh` after every lane of `chain_r1_confirm.sh`; `rerun_c85_demos.sh` after `retry_confirm_common.sh`; lane `gpu` of `chain_r2r4_confirm.sh` after `rerun_c85_demos.sh`; `chain_r4controls.sh` after that lane; `fix2_r4controls.sh` after `fix_r4controls.sh`.

The launches from 2026-09-28 on are known from a process list that was kept next to the chains (`pids.txt`, not distributed). It has 33 entries, each with a name, a process id and a UTC time:

| Launch (UTC) | Script | Invocations |
|---|---|---|
| 2026-09-28 20:33:57 | `chain_r1_confirm.sh` | common, seed1, seed2, seed3, report |
| 2026-09-28 20:37:42 | `launch_hao_diag3.sh` | one |
| 2026-09-30 16:31:00 | `retry_confirm_common.sh` | one |
| 2026-09-30 16:32:08 | `rerun_c85_demos.sh` | one |
| 2026-09-30 16:43:22 | `chain_r2r4_confirm.sh` | gpu, r4seed1, r4seed2, r4seed3, r2cpu, report |
| 2026-09-30 17:29:51 | `chain_r4controls.sh` | seed1, seed2, seed3, report |
| 2026-09-30 20:35:29 | `fix_r4controls.sh` | seed1, seed2, seed3, report |
| 2026-09-30 20:40:09 | `fix2_r4controls.sh` | seed1, seed2, seed3, report |
| 2026-09-30 20:53:57 | listed as `fix2b_r4controls` | seed1, seed2, report |
| 2026-10-01 01:36:07 | `fix3_r4controls.sh` | main1, main2, main3, rand |

The first two rows are relaunches after the host reboot of 2026-09-27; the list says so. For the entry `fix2b_r4controls` no script of that name exists. The entry carries the note "after the lazy-probe fix", and the run records of that minute are probe runs of seeds 1 and 2 (`20260930T205358Z_ssl_r4c_probe_seed1` and `20260930T205358Z_ssl_r4c_probe_seed2`, both flagged as killed in the [run index](../../../docs/run-index.md)), the stage that `fix2_r4controls.sh` runs. The published files do not say which script that launch used. docs/12 F29 records the decision of 2026-09-30 on how the R2 and R4 runs are read. It is timed 16:45 UTC, about two minutes after the launch of the R2 and R4 chains at 16:43:22 UTC, and it states that the R2 CPU stage had started at 16:43 UTC and that no confirmatory R2 or R4 run had finished.

Launches before 2026-09-28 are not in the process list. They are known from the header comments of the scripts and from run records:

- `chain_hao_diag.sh` and `chain_hao_diag2.sh` have no log in the chain folder and no entry in the process list, so their launches are known only from run records. The run folders of 2026-09-25 whose names start with `ssl_r2_hao_diag_` are that record. The first two, `20260925T090939Z_ssl_r2_hao_diag_selftest_head` and `20260925T091022Z_ssl_r2_hao_diag_demos_c85_exploratory`, follow the stage order of `chain_hao_diag.sh`. A second selftest, `20260925T112628Z_ssl_r2_hao_diag_selftest_head`, is followed by the valiant and controls runs of seed 100, which is the stage order of `chain_hao_diag2.sh`. A run folder does not record which script started it, so this assignment rests on the order of the stages.
- The confirmatory R1 run folders, whose names start with `ssl_r1cf_`, begin on 2026-09-26 at 20:44 UTC with `20260926T204447Z_ssl_r1cf_testhash_confirmatory`. They were started from the original of `chain_r1_confirm.sh`, after its wait for the exploratory chains had been released by hand on 2026-09-26.
- The host rebooted on 2026-09-27 at about 02:10 UTC, as the header of `chain_r1_confirm.sh` states. Four run records of that night are still recorded as running and are flagged as stale in the run index: `20260927T004416Z_ssl_r1cf_probe_rand_seed2_confirmatory`, `20260927T015033Z_ssl_r1cf_probe_A3T_seed1_confirmatory`, `20260927T020716Z_ssl_r1cf_probe_A3T_seed2_confirmatory` and `20260927T020905Z_ssl_r1cf_features_A3T_seed3_confirmatory`. Run folders with the same four run names start on 2026-09-28 at 20:36 UTC, after the relaunch.
- `cfg_arms_report.sh` has no log and no entry either. The run index has one run of its stage, `20260930T203828Z_ssl_r1ca_report_exploratory`, on 2026-09-30.

## Recovered copies

Four of the fourteen files are recovered copies. Their originals lay in a temporary folder that was lost in the host reboot of 2026-09-27. The header of each file says what was rebuilt:

- `lib.sh`: recovered on 2026-09-28 from the saved record. The text is identical to the version of 2026-09-25 08:12 UTC except for the study folder, which is now derived from the location of the file.
- `chain_r1_confirm.sh`: recovered on 2026-09-28, rebuilt from the saved record (the text of 2026-09-25 12:52 UTC and its three later changes of 12:59 and 14:02 UTC). Two changes: the state of the lanes (markers, locks, the list of skipped steps) lives in the folder of the script and no longer in the temporary folder, and a stage whose latest run of the same name exited 0 is not run again. The header states that this concerns scheduling only, that every command, input and lock is unchanged, and that a counted test access that already happened is never repeated.
- `chain_hao_diag3.sh`: recovered on 2026-09-28 from the saved record by replaying the text of 2026-09-25 07:53 UTC and its later changes (07:54, 08:33, 08:41, 11:26, 18:29:42 and 18:29:51 UTC). One change: the chain folder is now the folder of the script.
- `launch_hao_diag3.sh`: recovered on 2026-09-28 from the saved record (original of 2026-09-25 18:30 UTC). Only the chain folder changed.

The other ten files carry no recovery note. Six of them give their date in the header, 2026-09-30 or 2026-10-01 (`cfg_arms_report.sh`, `retry_confirm_common.sh`, `rerun_c85_demos.sh` and the three repair scripts). `chain_r2r4_confirm.sh` refers to the gate reading of 2026-09-30 and `chain_r4controls.sh` to docs/12 F29. `chain_hao_diag.sh` and `chain_hao_diag2.sh` give no date; the launch command quoted in the header of `chain_hao_diag2.sh` and of `chain_hao_diag3.sh` is the one of `chain_hao_diag.sh`.

## Chains that were not retained, and the release by hand

Three scripts that the published files name were not retained, and neither were their logs: `chain_controls.sh`, the chain of the exploratory control study (named in `experiments/ssl_r1_controls.py`); `chain_config_arms.sh`, the chain of the exploratory configuration arms (named in `experiments/ssl_r1_config_arms.py` and in `cfg_arms_report.sh`); and `stop_cfg100_after_last.sh` (named in a comment of `chain_r1_confirm.sh`).

`chain_r1_confirm.sh` decides from the logs of these scripts whether the exploratory chains have finished. It reads `chain_controls_100.out`, `chain_controls_101.out`, `chain_config_arms.out`, `chain_config_arms_101.out` and `stop_cfg100.out`, and it checks that no process of `experiments/ssl_r1_controls.py` or `experiments/ssl_r1_config_arms.py` is running. Without the logs the condition is never met, and this wait has no cap. It ends only when the file `r1_confirm_release` exists in the chain folder. In the recorded runs the wait was released by hand in this way on 2026-09-26. The chain then writes the release, with the conditions that were still pending, into `r1_confirm_skipped.txt`, the list of skipped and failed steps that the report stage reads.

## Files the chains write next to themselves

None of these files is tracked; the ignore rules of the repository name each pattern for this folder.

- Logs `*.out`, one per invocation. No script writes them: they are the redirections of the launch commands that the headers quote.
- Done markers `*.done`, for example `r1_confirm_seed1.done`, `r2r4_gpu.done` and `r4c_seed1.done`. Other lanes and chains wait for them. A marker whose text starts with `STOP` records a lane that stopped.
- Lock files: `r1_confirm_testaccess.lock`, `r1_confirm_testhash.lock` and `r4c_randbank.lock`, and, while an exclusive request waits, a file `<lock>.want.<pid>`. Every stage that makes a counted test access (confirm and qcon) holds `r1_confirm_testaccess.lock` exclusively; the census, qpred and qint stages hold it shared. The lock is taken before the GPU lock.
- The lists of skipped and failed steps, `r1_confirm_skipped.txt` and `r2r4_skipped.txt`.
- The release file `r1_confirm_release` and the process list `pids.txt`, which no script writes.

The chains ran from a folder of their own next to the study folder, which the scripts and one config still call the night folder. In this tree every script derives its folder from its own location (the variable `LAUNCH_DIR`, which several scripts still copy into a variable named `NIGHT`). The old folder name survives in one place outside this folder: two recorded values in `experiments/cvtm/configs/ssl_r1_confirm.json`, under the keys `locks` and `test_access_lock`, still name `night/r1_confirm_testaccess.lock`. In this tree that file is `launch/r1_confirm_testaccess.lock`.

## Lines and names that other chains read

Several chains decide when to start by reading the log of another chain or by looking for its process. The following strings and names are therefore part of the interface between the scripts, and a change to one of them breaks a wait:

- `CHAIN r1_confirm lane <lane> DONE`, printed by `chain_r1_confirm.sh` and read by `chain_hao_diag3.sh` and `launch_hao_diag3.sh`;
- `RETRY DONE`, printed by `retry_confirm_common.sh` and read by `rerun_c85_demos.sh`;
- `C85 RERUN DONE` and `c85 demos FAILED`, printed by `rerun_c85_demos.sh` and read by `chain_r2r4_confirm.sh`;
- `CHAIN controls DONE`, any line that starts with `CHAIN` and contains `DONE`, and `CHAIN STOP`, which `chain_r1_confirm.sh` expects in the logs of the chains that were not retained;
- the word `STOP` at the start of a done marker, written and read by the three repair scripts;
- the file names of the scripts, which `pgrep -f` matches (`chain_r1_confirm.sh <lane>`, `rerun_c85_demos.sh`, `retry_confirm_common.sh`), and the names of the logs and markers listed above.

The word lane stays in these scripts for the same reason: it is a shell variable and part of the echo lines, and the value of the variable is part of the marker names. All three diagnostics chains end with the line `CHAIN hao_diag DONE`, the second and the third included.

## What the chains assume about the machine

The chains encode the workstation of the recorded runs. They call the interpreters `.venv-tm/bin/python` and `.venv-htm/bin/python` by relative path, read the memory counters of the control group directly, ask `nvidia-smi` for free GPU memory before a GPU stage of the R1 chain, assume one GPU and the six CPU slots of `scripts/cpu_slot.sh`, and resolve their inputs as the newest run folder of a given run name under `results/`. Four of them name run folders of the recorded runs by run id (`chain_hao_diag2.sh`, `chain_hao_diag3.sh`, `rerun_c85_demos.sh`, `retry_confirm_common.sh`). Repeating a launch elsewhere needs those run folders and the same layout of environments; the study [README](../README.md) says what a clone contains.
