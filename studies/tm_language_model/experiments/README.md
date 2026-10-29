# experiments

This folder holds the experiment scripts of the study and their configs. The scripts lie flat in one folder: 62 Python files. Most of them state in their docstrings what they measure, which rows of the pre-registration they serve, and the commands that start them; `p0_common.py` and `engine_common.py` are helper modules that are not started by themselves. Nothing in this folder is a result. A run writes its own folder under `results/`, and the outcome of every group is reported on the results page, [docs/results.md](../docs/results.md).

## Scripts by group

The ids in the table are experiment ids of docs/05, hypothesis ids and amendments of docs/13, and kill criteria of docs/08.

| Group | Files | What the scripts do |
|---|---|---|
| `p0_*` | 7 | Phase 0 measurements on the stock GraphTsetlinMachine and on tmu (milestone M1): throughput against the number of outputs (E0, `p0_scaling_harness.py`), the step-time profile (T0.2, `p0_profile.py`), compilation with 16000 outputs (T0.3, `p0_compile_16000.py`), the masked-token sanity run E1 and its diagnosis (`p0_masked_depth2.py`, `p0_masked_diag.py`), and the tmu autoencoder reproduction (T0.5, `p0_tmu_ae_repro.py`). `p0_common.py` holds shared helpers. |
| `m2_*` | 2 | Milestone M2. `m2_build_data.py` builds the stores, tokenizers, cluster codes, frozen evaluation sets, paragraph pairs and the contamination report, and registers their hashes. `m2_baselines.py` runs the in-house baselines, one logged run per baseline family. |
| `m3_*`, `m4_sanitizer.py`, `engine_common.py` | 5 | The engine gates of milestones M3 and M4: parity with tmu in dense-equivalent mode (KB1), planted rules, the tiny LM (KB2, H1.4), and the CUDA engine under compute-sanitizer. `engine_common.py` holds the shared training loops. |
| s-series | 3 | `s3_batch_staleness.py` (KB3), `s4_deviation_ladder.py` (the price of each departure from the published coalesced Tsetlin machine), `s5_shard_equivalence.py` (H3.5). The experiments s1 and s2 exist as planned configs only. |
| e-series, CP, CM and H2 | 6 | `cp_causal_perplexity.py` (the causal anchor run and protocol CP, H1.1), `e3_wikitext_masked.py` (E3, H1.2), `e11_receipts_replay.py` (E11: receipts and replay, H2.2, H2.4, H6.4), `h2_4_recheck.py` (added 2026-10-25: the re-check of H2.4 on the saved E11 receipts with the bound checker; draft rule in `docs/drafts/`, not run), `cm_cost_model.py` (CM), `h2_layer2_greedy.py` (H2, row d). |
| `ckm_posthoc_probe*.py`, `tm_levers_*.py` | 6 | Exploratory probes of count-derived read-outs on saved clause banks, on the validation split only. Their runs precede amendment A3 and are disclosed in it. |
| `cvtm_*` | 18 | The count-vote experiments of amendments A3 to A5: the cache build and core fit (`cvtm_lm.py`), the one test read of H8.3 and H8.4 (`cvtm_test_read.py` and its independent recompute `cvtm_test_read_verify.py`, added 2026-10-25; draft plan in `docs/drafts/`, not run on the test split), the H8.1 read-out (`cvtm_t1_readout.py`), the rows H8.2, H8.7, H8.8 and H8.9 (`cvtm_a3_rows.py`, `cvtm_a3_rows_yield.py`), the masked variant H8.11 (`cvtm_masked.py`), energy and the transformer comparators (`cvtm_energy.py`, `cvtm_transformer_tuned.py`, `cvtm_gx_settle.py`, `cvtm_gx_idle_probe.py`), the vote equilibrium H9.6 (`cvtm_vote_equilibrium.py`), the copy selector H9.1 and H9.2 (`cvtm_copy_selector.py`), the exploratory pilots P1 and P2 of amendment A5 (`cvtm_pilot_p1.py`, `cvtm_pilot_p2.py`), and three early exploratory estimates (`cvtm_early_*`). |
| `ssl_*` | 15 | The self-supervised family of amendment A7 with the additions of A8 and A9. Early exploratory tests: `ssl_transfer_early.py`, `ssl_concepts_early.py`. R1: the pretext tournament (`ssl_r1_pretext.py`), its control study and configuration arms (`ssl_r1_controls.py`, `ssl_r1_config_arms.py`), the equivalence and speed-up checks (`ssl_r1_lr_equivalence.py`, `ssl_r1_speedup_verify.py`, `ssl_r1_speedup_timing.py`) and the confirmatory driver (`ssl_r1_confirm.py`). R2: the H-AO mechanism test (`ssl_r2_hao.py`, H17.5) with exploratory tuning, pilot and diagnostics (`ssl_r2_hao_tune.py`, `ssl_r2_hao_head.py`, `ssl_r2_hao_diagnostics.py`). R4: the image domain (`ssl_r4_images.py`, H17.10) and its reported controls (`ssl_r4_controls.py`). |

The results of every group are reported on the results page under these ids.

## Facts of the layout

The scripts import each other by bare module name, for example `import ssl_r1_controls as C` in `ssl_r1_confirm.py`. The folder therefore has to be on the import path of a run. Starting a script as `.venv-tm/bin/python experiments/<script>.py` puts it there, some scripts insert it themselves, and the tests that import a script add it to the path. The scripts find the study folder from their own location (`ROOT = Path(__file__).resolve().parents[1]`), and one library module, `src/tmlm/cvtm/copy_selector.py`, imports an experiment script in turn. Script names and the depth of the folder are fixed for these reasons: a script moved into a subfolder finds neither the study folder nor its siblings.

`experiments` is one of the five folders (`src`, `experiments`, `scripts`, `tests`, `env`) whose files the run logger hashes into `source_tree_hash` when a run starts. Every file in it enters that hash, this page included.

## Configs

### experiments/configs

`experiments/configs` holds 33 planned run definitions, one JSON file per planned run, written before the runs. `tests/test_configs.py` requires nine keys in each of them:

| Key | Content |
|---|---|
| `experiment` | the experiment id: a heading of docs/05, one of the M3 gates (`tiny_lm`, `parity_tmu`, `planted_rules`), one of the M2 items (`M2_data`, `M2_baselines`), or `T0.2` |
| `docs` | the document that defines the experiment |
| `milestone` | the milestone of docs/04 |
| `lane` | the lane of the schedule, explained below |
| `status` | the string `planned` |
| `prereg` | the hypothesis ids that the run serves; each must be a row of docs/13 |
| `seeds` | seeds from the fixed lists of docs/13 section 3 |
| `script` | the path of the script, starting with `experiments/` |
| `params` | the parameters of the run |

An `engine` block is optional and must build a valid `EngineConfig`. The test requires `status` to be `planned` in every file, so the value does not change when a run exists; whether an experiment ran is read from the [run index](../../../docs/run-index.md), not from its config.

The same test requires a config for every experiment heading of docs/05 except E0. Fifteen of the 33 configs therefore name scripts that were never written: `E2.json` (`e2_binomial_speedup.py`), `E3_trackA.json` (`e3_track_a.py`), `E4.json` (`e4_full_training.py`), `E5.json` (`e5_transfer.py`), `E6.json` (`e6_coherence.py`), `E7.json` (`e7_negation.py`), `E8.json` (`e8_blimp.py`), `E9.json` (`e9_concept_dictionary.py`), `E10.json` (`e10_energy.py`), `H1.json` (`h1_hierarchical_pairs.py`), `H3.json` (`h3_multiview_fusion.py`), `TB.json` (`tb_transformer_equal_cost.py`), `TK.json` (`tk_tokenization_ablation.py`), `s1.json` (`s1_scaling_clauses.py`) and `s2.json` (`s2_scaling_context.py`). The scripts of the other 18 configs are in this folder; `h2_4_recheck.json` (2026-10-25) is the newest.

`experiments/configs/E1_cells` holds 14 configs, one per cell of the E1 grid, all for `p0_masked_depth2.py`. The file pattern of the test does not reach this subfolder.

`m4_cuda_checks.json`, next to the scripts, is the run definition of the M4 engine checks. It is not an experiment of docs/05 and so lies outside `experiments/configs`. `m4_parity_tests.json` (2026-10-25) is the run definition of a logged run of the engine parity tests (H2.3 and the training half of H2.2), which has not run.

### experiments/cvtm/configs

`experiments/cvtm/configs` holds 67 configs of the experiments registered from amendment A3 on (`cvtm_a3_H8_3_H8_4_test.json`, for the test read of H8.3 and H8.4, added 2026-10-25 and not run): the count-vote and self-supervised runs, and `htm_env_install.json` for the build of `.venv-htm`. They are kept as they were used. A run copies its config into its run folder, and the run record holds the SHA-256 of the config file as it stood when the run started. The keys of these files vary with the experiment; many repeat the launch command of the run as a string, and the seven files named `cvtm_a3_H8_*.json` pin the hashes of source files.

`experiments/cvtm/configs/README.md` records which of these as-run configs were edited for publication, what was changed in them, and what follows for the config hashes in the run records. The [hash table](../../../docs/hash-table.md) at the repository root lists the pin values as registered and as published.

### What lane means

In the planned configs `lane` is an integer that names one of the three lanes of the schedule in [docs/04-implementation-plan.md](../docs/04-implementation-plan.md), section "Lane schedule": lane 1 is the engine and the critical path, lane 2 holds data, baselines and Track A, and lane 3 holds documents, tooling and evaluations. A lane is a slot for jobs that run in parallel with the jobs of the other lanes. The value says where a run was placed in the schedule and nothing about the experiment itself.

Where a config under `experiments/cvtm/configs` has the key, it holds a short name of the parallel job that ran the config, for example `L0 FREEZE AND CACHE` for the cache build. The docstrings of the scripts use the same names (A1, B1, GX). The launch chains use the word in a third way, for the argument that selects which part of a chain one invocation runs.

## Starting a run

A run is started through the locks and the run logger (protocol rules 5 and 13 of the [research protocol](../docs/research-protocol.md)), from the study folder:

```bash
source env/env.sh

# GPU work: the shared lock for training and correctness runs, the exclusive lock for timing, energy and profiling
scripts/gpu_run.sh shared -- scripts/mem_guard.py -- \
    scripts/run_logged.py --name <run name> --config experiments/configs/<id>.json -- \
    .venv-tm/bin/python experiments/<script>.py

# CPU work: N of the 6 CPU slots, here all of them
scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name <run name> --config experiments/configs/<id>.json -- \
    .venv-tm/bin/python experiments/<script>.py
```

`scripts/gpu_run.sh` holds the lock on the one GPU (shared jobs may overlap, an exclusive job runs alone), `scripts/cpu_slot.sh N` takes N of the six CPU slots and caps every thread pool at N, and `scripts/mem_guard.py` stops a job that passes its memory limit. Since 2026-09-24 a job that needs both resources takes its CPU slot first and the GPU lock second (`scripts/cpu_slot.sh N -- scripts/gpu_run.sh MODE -- ...`), and jobs that share the machine use `scripts/mem_guard.py --scope group --limit-gib <budget>` (docs/12 F25).

`scripts/run_logged.py` creates `results/<UTC timestamp>_<run name>/` and writes `config.json`, `env.json`, `run.json`, `metrics.json`, `log.txt` and `MANIFEST.json` into it. The timestamp and the run name together form the run id that the documents cite. The command sees `TMLM_RUN_DIR` and `TMLM_RUN_NAME` and can report its metrics through `tmlm.common.runlog.write_metrics`. The commands of each script, with their locks and memory limits, are in the docstring of the script. docs/13 section 3 reserves seeds 100 and above for exploratory runs, and many run names of the later experiments end in `_exploratory`, `_confirmatory` or `_smoke`.

The helper scripts start with `#!/usr/bin/env python3`, so a script started by name runs under the `python3` that comes first on `PATH`. Since 2026-10-25 `source env/env.sh` puts `.venv-tm/bin` there, as the commands above assume; the study [README](../README.md) says more on this and on what a clone can and cannot run. The overnight runs of the `ssl_*` group were started by the chains in `launch/`, described in [launch/README.md](../launch/README.md).
