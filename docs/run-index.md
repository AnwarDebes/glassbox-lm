# Run index

One row per run folder of the studies that keep run folders, 1,076 in all: 643 of `tm_language_model`,
319 of `hierarchical_tm`, 23 of `clause_lm` and 91 of `tally`. `decision_layer` has no run folders and
no run ids: it keeps its results in files (JSON files per seed and per report, NumPy
archives of predictions and models, and some JSON-lines logs) in `results/confirm/`, one folder per
registered test or dataset, and in `results/explore/` for the dev experiments, and its results page cites those files and its run
logs in `logs/`, so it has no rows here. The run folders, these result files and the run logs are not
distributed with the repository; this table records which runs exist, so that a run id cited on a
results page can be placed.

The table was generated on 2026-10-10 from the run records only, that is from the files `run.json`
and `config.json` of each run folder, read in one pass. The 14 rows of `clause_lm` dated 2026-10-14
and the 91 rows of `tally` were added on 2026-10-23 in the same way. The date is taken from the run
id, which is a UTC timestamp. The size is the size of the run folder. The studies use different record
formats, and a cell is left empty where the record does not hold the value: the 23 folders of
`clause_lm` have no `run.json`, so their status, exit status and wall time are empty, and the run
records of `tally` hold a status and a wall time but no exit status. The table lists folder names. One
folder of `tally`, `20261015T072640Z_t0_report_valid_smoke`, holds a record whose run id is
`20261015T072640Z_t0_report_valid`: the suffix `_smoke` of the folder name is not in the record.

Flags mark runs whose numbers are not used as results anywhere in this repository:

- never cite: two runs that the log of design changes of `tm_language_model` marks as not cited
  (rows F9 and F18 of `studies/tm_language_model/docs/12-design-changes.md`).
- stale (recorded as running): the record still says "running" and has no end time; the run did
  not finish in a recorded way.
- killed: the record holds a terminating signal or the exit status 143.
- smoke: a plumbing test, named so in its run id.
- biased exploratory: the `feas_slide11` probe runs of 2026-10-01, whose pair data had a sampling
  bias (see `docs/known-limitations.md`).

A run with exit status 1 and no flag ended with an error of its own.

| run id | study | experiment | date (UTC) | status | exit status | wall time | size | flag |
|---|---|---|---|---|---|---|---|---|
| `20260919T173332Z_envcheck` | tm_language_model | envcheck | 2026-09-19 | finished | 0 | 4 s | 34.5 kB |  |
| `20260919T174511Z_p0_scaling` | tm_language_model | p0_scaling_harness | 2026-09-19 | finished | 143 | 56 s | 18.3 kB | killed |
| `20260919T202216Z_p0_scaling` | tm_language_model | p0_scaling_harness | 2026-09-19 | finished | 0 | 13.1 min | 270.5 kB |  |
| `20260919T202646Z_p0_tmu_ae_run_seed1` | tm_language_model | p0_tmu_ae_run_seed1 | 2026-09-19 | finished | 0 | 7.4 min | 26.3 kB |  |
| `20260919T202646Z_p0_tmu_ae_run_seed2` | tm_language_model | p0_tmu_ae_run_seed2 | 2026-09-19 | finished | 0 | 7.3 min | 26.3 kB |  |
| `20260919T202801Z_p0_scoop_check_repo_facts` | tm_language_model | p0_scoop_check_repo_facts | 2026-09-19 | finished | 0 | 0 s | 5.5 kB |  |
| `20260919T203409Z_p0_tmu_ae_determinism` | tm_language_model | p0_tmu_ae_determinism | 2026-09-19 | finished | 0 | 40 s | 7.3 kB |  |
| `20260919T203409Z_p0_tmu_ae_run_seed3` | tm_language_model | p0_tmu_ae_run_seed3 | 2026-09-19 | running |  |  | 11.9 kB | stale (recorded as running) |
| `20260919T203741Z_p0_compile_16000` | tm_language_model | p0_compile_16000 | 2026-09-19 | finished | 143 | 30 s | 20.9 kB | killed |
| `20260922T111529Z_envcheck` | tm_language_model | envcheck | 2026-09-22 | finished | 0 | 6 s | 35.3 kB | never cite |
| `20260922T125436Z_prior_art_check` | tm_language_model | prior_art_check | 2026-09-22 | finished | 0 | 0 s | 81.3 kB |  |
| `20260922T145432Z_envcheck_clean_copy` | tm_language_model | envcheck | 2026-09-22 | finished | 0 | 7 s | 37.9 kB |  |
| `20260922T145836Z_tmu_pin_change` | tm_language_model | tmu_pin_change | 2026-09-22 | finished | 0 | 12 s | 6.2 kB |  |
| `20260922T152940Z_planted_rules` | tm_language_model | planted_rules | 2026-09-22 | finished | 0 | 33 s | 70.3 kB |  |
| `20260922T153126Z_planted_rules` | tm_language_model | planted_rules | 2026-09-22 | finished | 0 | 19 s | 70.3 kB |  |
| `20260922T155323Z_M2_data` | tm_language_model | M2_data | 2026-09-22 | finished | 143 | 1.8 min | 10.4 kB | killed |
| `20260922T155509Z_parity_tmu` | tm_language_model | parity_tmu | 2026-09-22 | finished | 0 | 3.0 min | 9.7 kB |  |
| `20260922T160257Z_M2_data` | tm_language_model | M2_data | 2026-09-22 | finished | 0 | 51.7 min | 76.5 kB |  |
| `20260922T165439Z_planted_rules` | tm_language_model | planted_rules | 2026-09-22 | finished | 0 | 1.4 min | 97.8 kB |  |
| `20260922T165603Z_M2_data_ptb` | tm_language_model | M2_data | 2026-09-22 | finished | 0 | 12 s | 27.8 kB |  |
| `20260922T170141Z_tiny_lm` | tm_language_model | tiny_lm | 2026-09-22 | finished | 0 | 37.4 min | 9.8 kB |  |
| `20260922T172902Z_M2_baselines_transformer` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 1.01 h | 85.7 MB |  |
| `20260922T173905Z_M2_baselines_ngram_v8000` | tm_language_model | M2_baselines | 2026-09-22 | finished | 1 | 8.4 min | 15.0 kB |  |
| `20260922T174727Z_tiny_lm_diag_noneg` | tm_language_model | tiny_lm | 2026-09-22 | finished | 0 | 20.9 min | 8.6 kB |  |
| `20260922T180823Z_M2_baselines_ngram_v8000` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 16.9 min | 27.0 kB |  |
| `20260922T182517Z_tiny_lm` | tm_language_model | tiny_lm | 2026-09-22 | finished | 0 | 25.5 min | 8.9 kB |  |
| `20260922T182941Z_M2_baselines_transformer_mlm_smoke` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 2.2 min | 21.1 MB | smoke |
| `20260922T183153Z_M2_baselines_bert` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 23 s | 14.7 kB |  |
| `20260922T185048Z_M2_baselines_ngram_v2000` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 12.2 min | 26.6 kB |  |
| `20260922T190301Z_tiny_lm` | tm_language_model | tiny_lm | 2026-09-22 | finished | 0 | 23.8 min | 9.1 kB |  |
| `20260922T192647Z_M2_baselines_count_probe_v8000` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 4.0 min | 13.9 kB |  |
| `20260922T193047Z_tiny_lm` | tm_language_model | tiny_lm | 2026-09-22 | finished | 0 | 1.51 h | 16.9 kB |  |
| `20260922T210123Z_M2_baselines_count_probe_v2000` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 2.1 min | 13.9 kB |  |
| `20260922T210329Z_M2_baselines_cbow_knn_v8000` | tm_language_model | M2_baselines | 2026-09-22 | finished | 0 | 59.4 min | 8.2 MB |  |
| `20260922T220251Z_s4` | tm_language_model | s4 | 2026-09-22 | finished | 143 | 1.23 h | 6.0 kB | never cite, killed |
| `20260922T224238Z_E1_pilot_T1000_s30_d2` | tm_language_model | E1 | 2026-09-22 | finished | 0 | 2.1 min | 17.9 kB |  |
| `20260922T224456Z_E1_pilot_T1000_s30_d2` | tm_language_model | E1 | 2026-09-22 | finished | 0 | 36.3 min | 15.9 kB |  |
| `20260922T224456Z_E1_pilot_T250_s30_d2` | tm_language_model | E1 | 2026-09-22 | finished | 0 | 36.3 min | 16.2 kB |  |
| `20260922T224456Z_E1_pilot_T4000_s30_d2` | tm_language_model | E1 | 2026-09-22 | finished | 0 | 36.3 min | 16.3 kB |  |
| `20260922T225226Z_E1_comparators` | tm_language_model | E1 | 2026-09-22 | finished | 0 | 2.3 min | 10.0 kB |  |
| `20260922T232114Z_m4_cuda_throughput` | tm_language_model | CM | 2026-09-22 | finished | 0 | 31 s | 47.5 kB |  |
| `20260922T232146Z_m4_cuda_throughput_cpu_vs_cuda` | tm_language_model | CM | 2026-09-22 | finished | 0 | 1.1 min | 64.4 kB |  |
| `20260922T232251Z_E11` | tm_language_model | E11 | 2026-09-22 | finished | 0 | 2.65 h | 1.3 GB |  |
| `20260922T232251Z_p0_profile_T02` | tm_language_model | p0_profile | 2026-09-22 | finished | 0 | 6.1 min | 120.5 kB |  |
| `20260922T232858Z_E1_d2_s10_e1` | tm_language_model | E1 | 2026-09-22 | finished | 0 | 57.4 min | 38.3 MB |  |
| `20260922T232858Z_m4_cuda_sanitizer` | tm_language_model | M4_cuda_checks | 2026-09-22 | finished | 1 | 1.3 min | 497.1 kB |  |
| `20260922T233121Z_m4_cuda_sanitizer` | tm_language_model | M4_cuda_checks | 2026-09-22 | finished | 1 | 1.3 min | 505.5 kB |  |
| `20260922T233341Z_m4_cuda_sanitizer` | tm_language_model | M4_cuda_checks | 2026-09-22 | finished | 1 | 1.2 min | 508.4 kB |  |
| `20260922T233629Z_m4_cuda_sanitizer` | tm_language_model | M4_cuda_checks | 2026-09-22 | finished | 1 | 1.3 min | 684.3 kB |  |
| `20260923T002621Z_m4_cuda_b1_eager_throughput` | tm_language_model | s4 | 2026-09-23 | finished | 0 | 19 s | 17.8 kB |  |
| `20260923T002641Z_E1_d2_s30_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 52.0 min | 38.3 MB |  |
| `20260923T011847Z_E1_d2_s100_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 51.9 min | 38.3 MB |  |
| `20260923T020158Z_s3` | tm_language_model | s3 | 2026-09-23 | finished | 0 | 10.91 h | 19.6 kB |  |
| `20260923T021042Z_E1_d2_s300_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 57.5 min | 38.3 MB |  |
| `20260923T030812Z_E1_d2_s1000_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 1.78 h | 38.3 MB |  |
| `20260923T034350Z_E1_diag_v64_d2_left1` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 21.8 min | 8.6 kB |  |
| `20260923T034944Z_s5_gpu` | tm_language_model | s5 | 2026-09-23 | finished | 0 | 3.2 min | 12.3 kB |  |
| `20260923T035425Z_s5_gpu` | tm_language_model | s5 | 2026-09-23 | finished | 0 | 2.9 min | 14.3 kB |  |
| `20260923T040526Z_CP_anchor_probe` | tm_language_model | CP | 2026-09-23 | finished | 0 | 18.1 min | 16.2 kB |  |
| `20260923T040541Z_E1_diag_v64_d1_left1` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 12.8 min | 7.3 kB |  |
| `20260923T041829Z_E1_diag_v2000_d2_left1` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 1.32 h | 7.3 kB |  |
| `20260923T042350Z_CP_anchor` | tm_language_model | CP | 2026-09-23 | finished | 0 | 16.19 h | 1.3 GB |  |
| `20260923T045457Z_E1_d2_s3000_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 6.60 h | 38.3 MB |  |
| `20260923T053858Z_E1_diag_v64_d2_s1_mil4` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 1.41 h | 7.9 kB |  |
| `20260923T070339Z_E1_diag_v64_d3_s1_mil4` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 1.75 h | 7.4 kB |  |
| `20260923T084853Z_E1_diag_v64_d2_s30_mil4` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 1.34 h | 7.4 kB |  |
| `20260923T100903Z_E1_diag_v64_d2_s1_mil4_span1` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 1.18 h | 7.5 kB |  |
| `20260923T112105Z_E1_diag_stock_sequence_demo` | tm_language_model | E1_diag | 2026-09-23 | finished | 143 | 8 s | 5.5 kB | killed |
| `20260923T112132Z_E1_diag_stock_sequence_demo` | tm_language_model | E1_diag | 2026-09-23 | finished | 0 | 9.39 h | 8.2 kB |  |
| `20260923T113055Z_E1_d1_s30_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 4.05 h | 36.3 MB |  |
| `20260923T125642Z_H2_row_d` | tm_language_model | H2 | 2026-09-23 | finished | 0 | 2.77 h | 80.2 MB |  |
| `20260923T153350Z_E1_d3_s30_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 5.91 h | 40.3 MB |  |
| `20260923T154311Z_s4` | tm_language_model | s4 | 2026-09-23 | finished | 143 | 13.83 h | 6.0 kB | killed |
| `20260923T203943Z_E3_v8k_slice_exploratory` | tm_language_model | E3 | 2026-09-23 | finished | 0 | 13.1 min | 59.9 kB |  |
| `20260923T212836Z_E1_d1_s1000_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 41.6 min | 36.3 MB |  |
| `20260923T221011Z_E1_d3_s1000_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 1.22 h | 40.3 MB |  |
| `20260923T232326Z_E1_d2_s30_mil4_e1` | tm_language_model | E1 | 2026-09-23 | finished | 0 | 1.17 h | 38.3 MB |  |
| `20260924T001024Z_ckm_probe_masked_exploratory` | tm_language_model | ckm_posthoc_probe | 2026-09-24 | finished | 0 | 33.5 min | 8.2 MB |  |
| `20260924T003352Z_E1_d2_s30_mil32_e1` | tm_language_model | E1 | 2026-09-24 | finished | 0 | 1.21 h | 38.3 MB |  |
| `20260924T012544Z_ckm_probe_causal_exploratory` | tm_language_model | ckm_posthoc_probe_causal | 2026-09-24 | finished | 0 | 31.4 min | 61.6 MB |  |
| `20260924T014637Z_E1_d2_s1000_mil4_e1` | tm_language_model | E1 | 2026-09-24 | finished | 0 | 1.33 h | 38.3 MB |  |
| `20260924T021631Z_tm_levers_feasibility_causal_exploratory` | tm_language_model | tm_levers_feasibility_causal | 2026-09-24 | finished | 0 | 10.5 min | 79.3 kB |  |
| `20260924T022628Z_ckm_probe_causal_controls_exploratory` | tm_language_model | ckm_posthoc_probe_causal_controls | 2026-09-24 | finished | 0 | 1.04 h | 261.3 MB |  |
| `20260924T022949Z_tm_levers_trigger_control_causal_exploratory` | tm_language_model | tm_levers_trigger_control_causal | 2026-09-24 | finished | 0 | 2.5 min | 28.4 kB |  |
| `20260924T030617Z_E1_d2_s1000_mil32_e1` | tm_language_model | E1 | 2026-09-24 | finished | 0 | 1.29 h | 38.3 MB |  |
| `20260924T032913Z_ckm_probe_causal_controls_20M_exploratory` | tm_language_model | ckm_posthoc_probe_causal_controls | 2026-09-24 | finished | 0 | 27.0 min | 28.9 MB |  |
| `20260924T035633Z_ckm_probe_causal_scale_compare_exploratory` | tm_language_model | ckm_posthoc_probe_causal_scale_compare | 2026-09-24 | finished | 0 | 4 s | 225.3 kB |  |
| `20260924T053302Z_s5_oracle` | tm_language_model | s5 | 2026-09-24 | finished | 0 | 1.24 h | 8.2 kB |  |
| `20260924T054225Z_cvtm_early_fulldata_exploratory` | tm_language_model | cvtm_early_fulldata | 2026-09-24 | finished | 0 | 22.0 min | 8.6 MB |  |
| `20260924T054640Z_cvtm_H8_1_masked_seed2` | tm_language_model | cvtm_H8_1_masked | 2026-09-24 | finished | 0 | 1.8 min | 1.2 MB |  |
| `20260924T054807Z_cvtm_doc_clauses_mixture_smoke` | tm_language_model | doc_clauses_mixture_smoke | 2026-09-24 | finished | 0 | 30 s | 19.1 kB | smoke |
| `20260924T054826Z_cvtm_H8_1_masked_seed3` | tm_language_model | cvtm_H8_1_masked | 2026-09-24 | finished | 0 | 1.6 min | 1.2 MB |  |
| `20260924T055002Z_cvtm_H8_1_masked_seed4` | tm_language_model | cvtm_H8_1_masked | 2026-09-24 | finished | 0 | 1.6 min | 1.2 MB |  |
| `20260924T055112Z_cvtm_doc_clauses_mixture_smoke` | tm_language_model | doc_clauses_mixture_smoke | 2026-09-24 | finished | 0 | 30 s | 19.1 kB | smoke |
| `20260924T055140Z_cvtm_H8_1_masked_seed5` | tm_language_model | cvtm_H8_1_masked | 2026-09-24 | finished | 0 | 1.6 min | 1.2 MB |  |
| `20260924T055525Z_cvtm_early_logical_attention_pilot_exploratory` | tm_language_model | cvtm_early_logical_attention (pilot: TM T and s range) | 2026-09-24 | finished | 0 | 16.0 min | 36.4 kB |  |
| `20260924T060459Z_cvtm_H8_1_causal_cp20M` | tm_language_model | cvtm_H8_1_causal | 2026-09-24 | finished | 143 | 2.6 min | 22.4 kB | killed |
| `20260924T060608Z_cvtm_early_fulldata_mixcheck_exploratory` | tm_language_model | cvtm_early_fulldata_mixcheck | 2026-09-24 | finished | 0 | 22.3 min | 6.5 MB |  |
| `20260924T061202Z_cvtm_early_logical_attention_exploratory` | tm_language_model | cvtm_early_logical_attention | 2026-09-24 | finished | 143 | 8.3 min | 31.1 kB | killed |
| `20260924T061320Z_cvtm_H8_1_causal_cp20M` | tm_language_model | cvtm_H8_1_causal | 2026-09-24 | finished | 143 | 2.3 min | 22.4 kB | killed |
| `20260924T062132Z_cvtm_H8_1_causal_cp20M` | tm_language_model | cvtm_H8_1_causal | 2026-09-24 | finished | 143 | 3.5 min | 22.5 kB | killed |
| `20260924T062408Z_cvtm_early_logical_attention_exploratory` | tm_language_model | cvtm_early_logical_attention | 2026-09-24 | finished | 0 | 1.17 h | 12.4 MB |  |
| `20260924T062704Z_cvtm_H8_1_causal_cp20M` | tm_language_model | cvtm_H8_1_causal | 2026-09-24 | finished | 0 | 17.9 min | 3.9 MB |  |
| `20260924T082109Z_cvtm_cache_build` | tm_language_model | cvtm_cache_build | 2026-09-24 | finished | 1 | 24.0 min | 24.6 GB |  |
| `20260924T085728Z_cvtm_cache_build` | tm_language_model | cvtm_cache_build | 2026-09-24 | finished | 0 | 47.8 min | 31.5 GB |  |
| `20260924T102203Z_cvtm_a3_H8_2` | tm_language_model | cvtm_a3_H8_2 | 2026-09-24 | finished | 0 | 2.6 min | 19.2 kB |  |
| `20260924T102454Z_cvtm_a3_H8_7` | tm_language_model | cvtm_a3_H8_7 | 2026-09-24 | finished | 0 | 10.6 min | 50.5 kB |  |
| `20260924T104118Z_cvtm_H8_5_fit_energy` | tm_language_model | cvtm_H8_5_energy | 2026-09-24 | finished | 143 | 23.2 min | 33.6 kB | killed |
| `20260924T105732Z_cvtm_pilot_p1_blimp_hashes` | tm_language_model | cvtm_pilot_p1_blimp_hashes | 2026-09-24 | finished | 1 | 1.1 min | 40.5 kB |  |
| `20260924T105854Z_cvtm_pilot_p1_blimp_hashes` | tm_language_model | cvtm_pilot_p1_blimp_hashes | 2026-09-24 | finished | 0 | 3 s | 55.2 kB |  |
| `20260924T110143Z_cvtm_copy_selector_pairs` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 0 | 20.3 min | 22.6 GB |  |
| `20260924T110429Z_cvtm_H8_11_masked` | tm_language_model | cvtm_H8_11_masked | 2026-09-24 | finished | 0 | 20.2 min | 1.3 MB |  |
| `20260924T110429Z_cvtm_a3_H8_8b` | tm_language_model | cvtm_a3_H8_8b | 2026-09-24 | finished | 0 | 8.0 min | 25.7 kB |  |
| `20260924T112441Z_cvtm_H8_5_fit_energy` | tm_language_model | cvtm_H8_5_energy | 2026-09-24 | finished | 143 | 1.1 min | 19.4 kB | killed |
| `20260924T112548Z_cvtm_a3_H8_8a` | tm_language_model | cvtm_a3_H8_8a | 2026-09-24 | finished | 143 | 7.5 min | 40.2 kB | killed |
| `20260924T113130Z_cvtm_pilot_p2_exploratory` | tm_language_model | cvtm_pilot_p2 | 2026-09-24 | finished | 143 | 6 s | 15.1 kB | killed |
| `20260924T150704Z_cvtm_a3_H8_8a` | tm_language_model | cvtm_a3_H8_8a | 2026-09-24 | finished | 0 | 6.2 min | 52.8 kB |  |
| `20260924T151202Z_ssl_transfer_data_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 1 | 5 s | 12.1 kB |  |
| `20260924T151234Z_ssl_transfer_data_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 3 s | 11.2 kB |  |
| `20260924T151253Z_ssl_transfer_features_smoke` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 17 s | 98.3 MB | smoke |
| `20260924T151319Z_cvtm_H8_5_fit_energy` | tm_language_model | cvtm_H8_5_energy | 2026-09-24 | finished | 0 | 14.6 min | 51.4 kB |  |
| `20260924T151344Z_cvtm_pilot_p1_arm1_seed100_exploratory` | tm_language_model | cvtm_pilot_p1_arm1 | 2026-09-24 | finished | 143 | 1.51 h | 9.9 MB | killed |
| `20260924T152753Z_cvtm_copy_selector_seed3` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 143 | 1.15 h | 26.9 kB | killed |
| `20260924T152758Z_cvtm_a3_H8_8c` | tm_language_model | cvtm_a3_H8_8c | 2026-09-24 | finished | 143 | 1.2 min | 16.1 kB | killed |
| `20260924T152912Z_ssl_transfer_features_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 15.7 min | 9.9 GB |  |
| `20260924T152913Z_cvtm_copy_selector_seed2` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 143 | 1.13 h | 26.9 kB | killed |
| `20260924T154533Z_cvtm_copy_selector_seed1` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 143 | 51.6 min | 26.7 kB | killed |
| `20260924T154753Z_ssl_transfer_classify_baselines_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 143 | 49.6 min | 403.8 kB | killed |
| `20260924T163710Z_cvtm_a3_H8_8c` | tm_language_model | cvtm_a3_H8_8c | 2026-09-24 | finished | 143 | 6.5 min | 20.4 kB | killed |
| `20260924T163715Z_ssl_transfer_classify_banks_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 143 | 10 s | 27.2 kB | killed |
| `20260924T164342Z_cvtm_GX_idle_probe_exploratory` | tm_language_model | cvtm_GX_idle_probe | 2026-09-24 | finished | 143 | 12 s | 7.4 kB | killed |
| `20260924T164418Z_cvtm_pilot_p2_exploratory` | tm_language_model | cvtm_pilot_p2 | 2026-09-24 | finished | 143 | 3 s | 11.0 kB | killed |
| `20260924T200408Z_cvtm_copy_selector_seed1` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 143 | 1.89 h | 57.2 kB | killed |
| `20260924T200411Z_cvtm_copy_selector_seed2` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 143 | 1.90 h | 56.9 kB | killed |
| `20260924T200414Z_cvtm_copy_selector_seed3` | tm_language_model | cvtm_copy_selector | 2026-09-24 | finished | 143 | 1.96 h | 56.8 kB | killed |
| `20260924T200417Z_ssl_transfer_classify_baselines_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 1 s | 29.0 kB |  |
| `20260924T200420Z_ssl_transfer_classify_banks_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 1 s | 29.0 kB |  |
| `20260924T200541Z_ssl_transfer_classify_baselines_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 2.89 h | 1.2 MB |  |
| `20260924T200544Z_ssl_transfer_classify_banks_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 2.85 h | 935.2 kB |  |
| `20260924T201235Z_ssl_concepts_bank_trained_exploratory` | tm_language_model | ssl_concepts_early | 2026-09-24 | finished | 0 | 17.4 min | 34.5 MB |  |
| `20260924T202957Z_ssl_concepts_bank_young_exploratory` | tm_language_model | ssl_concepts_early | 2026-09-24 | finished | 0 | 15.7 min | 34.5 MB |  |
| `20260924T203523Z_ssl_r4_data_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 37 s | 14.6 kB | smoke |
| `20260924T203617Z_ssl_r2_hao_cpu_smoke` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 20 s | 25.6 kB | smoke |
| `20260924T204539Z_ssl_concepts_baselines_exploratory` | tm_language_model | ssl_concepts_early | 2026-09-24 | finished | 0 | 2.5 min | 50.5 MB |  |
| `20260924T204808Z_ssl_concepts_evaluate_exploratory` | tm_language_model | ssl_concepts_early | 2026-09-24 | finished | 0 | 58 s | 351.9 kB |  |
| `20260924T204906Z_ssl_r1_pretext_smoke` | tm_language_model | ssl_r1_pretext_smoke | 2026-09-24 | finished | 0 | 7.4 min | 256.8 MB | smoke |
| `20260924T205646Z_ssl_r2_hao_gpu_smoke` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 26 s | 35.7 kB | smoke |
| `20260924T205712Z_ssl_r2_hao_control_smoke` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 12 s | 39.2 kB | smoke |
| `20260924T205724Z_ssl_r4_gpu_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 3 s | 36.8 MB | smoke |
| `20260924T205727Z_ssl_r4_probe_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 2.4 min | 23.7 kB | smoke |
| `20260924T205951Z_ssl_r2_hao_report_smoke` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 0 s | 18.1 kB | smoke |
| `20260924T205951Z_ssl_r4_probe_tm_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 1 | 0 s | 10.1 kB | smoke |
| `20260924T205951Z_ssl_r4_report_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 1 | 0 s | 11.2 kB | smoke |
| `20260924T205957Z_ssl_r4_gpu_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 3 s | 25.1 MB | smoke |
| `20260924T210001Z_ssl_r4_probe_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 2.4 min | 23.7 kB | smoke |
| `20260924T210222Z_ssl_r4_probe_tm_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 14 s | 14.9 kB | smoke |
| `20260924T210236Z_ssl_r4_report_smoke` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 0 s | 20.5 kB | smoke |
| `20260924T210629Z_ssl_r1_data` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 1 s | 13.8 kB |  |
| `20260924T210631Z_ssl_r1_train_A1_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 4.4 min | 70.9 MB |  |
| `20260924T211053Z_ssl_r1_features_A1_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 2.8 min | 2.6 GB |  |
| `20260924T211347Z_ssl_r1_train_A0_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 8.0 min | 74.7 MB |  |
| `20260924T212145Z_ssl_r1_features_A0_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 3.1 min | 2.6 GB |  |
| `20260924T212501Z_ssl_r1_train_A2_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 3.5 min | 65.2 MB |  |
| `20260924T212903Z_ssl_r1_features_A2_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 2.5 min | 2.4 GB |  |
| `20260924T213137Z_ssl_r1_train_A6_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 4.4 min | 70.8 MB |  |
| `20260924T215716Z_ssl_r1_train_A3_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 4.5 min | 65.3 MB |  |
| `20260924T215754Z_ssl_r2_hao_gpu_exploratory` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 15.9 min | 45.5 kB |  |
| `20260924T220147Z_ssl_r4_data` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 44 s | 13.9 kB |  |
| `20260924T220231Z_ssl_r4_gpu_seed100_exploratory` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 2.6 min | 1.8 GB |  |
| `20260924T220514Z_ssl_r4_probe_seed100_exploratory` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 1.74 h | 45.3 kB |  |
| `20260924T221350Z_ssl_r2_hao_control_exploratory` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 1.03 h | 42.1 kB |  |
| `20260924T223250Z_ssl_r1_features_A3_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 4.3 min | 2.5 GB |  |
| `20260924T223304Z_ssl_r1_features_A6_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 4.5 min | 2.7 GB |  |
| `20260924T223713Z_ssl_r1_train_A4_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 10.1 min | 73.9 MB |  |
| `20260924T223741Z_ssl_r1_baselines` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 2.4 min | 9.6 MB |  |
| `20260924T224005Z_ssl_r1_probe_baselines` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 45.8 min | 138.7 kB |  |
| `20260924T224717Z_ssl_r1_features_A4_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 2.4 min | 1.4 GB |  |
| `20260924T224944Z_ssl_r1_train_A5_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 12.1 min | 72.0 MB |  |
| `20260924T230149Z_ssl_r1_features_A5_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 5.8 min | 2.9 GB |  |
| `20260924T230251Z_ssl_transfer_report_exploratory` | tm_language_model | ssl_transfer_early | 2026-09-24 | finished | 0 | 15 s | 196.2 kB |  |
| `20260924T230745Z_ssl_r1_probe_A3_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 2.65 h | 241.0 kB |  |
| `20260924T231550Z_ssl_r2_hao_cpu_exploratory` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 41.7 min | 25.3 kB |  |
| `20260924T232556Z_ssl_r1_probe_A1_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-24 | finished | 0 | 2.71 h | 241.0 kB |  |
| `20260924T234925Z_ssl_r4_probe_tm_seed100_exploratory` | tm_language_model | ssl_r4_images | 2026-09-24 | finished | 0 | 46.8 min | 22.2 kB |  |
| `20260924T235732Z_ssl_r2_hao_report_exploratory` | tm_language_model | ssl_r2_hao | 2026-09-24 | finished | 0 | 0 s | 18.3 kB |  |
| `20260925T000043Z_cvtm_copy_selector_seed1` | tm_language_model | cvtm_copy_selector | 2026-09-25 | finished | 0 | 6.09 h | 13.1 MB |  |
| `20260925T000046Z_cvtm_copy_selector_seed2` | tm_language_model | cvtm_copy_selector | 2026-09-25 | finished | 0 | 6.33 h | 13.1 MB |  |
| `20260925T000049Z_cvtm_copy_selector_seed3` | tm_language_model | cvtm_copy_selector | 2026-09-25 | finished | 0 | 6.44 h | 13.1 MB |  |
| `20260925T003613Z_ssl_r2_hao_tune_r1_hier_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 14.2 min | 164.4 kB |  |
| `20260925T003613Z_ssl_r4_report_seed100_exploratory` | tm_language_model | ssl_r4_images | 2026-09-25 | finished | 0 | 0 s | 25.7 kB |  |
| `20260925T005025Z_ssl_r2_hao_tune_r1_flat_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 17.4 min | 156.1 kB |  |
| `20260925T010750Z_ssl_r2_hao_tune_r2_hier_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 16.7 min | 170.5 kB |  |
| `20260925T012431Z_ssl_r2_hao_tune_r2_flat_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 17.3 min | 157.9 kB |  |
| `20260925T014149Z_ssl_r2_hao_tune_r3_hier_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 20.6 min | 197.2 kB |  |
| `20260925T014642Z_ssl_r1_probe_A4_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-25 | finished | 0 | 2.32 h | 241.0 kB |  |
| `20260925T020228Z_ssl_r2_hao_tune_r3_flat_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 22.1 min | 186.0 kB |  |
| `20260925T020849Z_ssl_r1_probe_A0_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-25 | finished | 0 | 2.72 h | 241.1 kB |  |
| `20260925T022436Z_ssl_r2_hao_tune_r4_hier_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 25.9 min | 76.1 kB |  |
| `20260925T025028Z_ssl_r2_hao_tune_r4_flat_exploratory` | tm_language_model | ssl_r2_hao_tune | 2026-09-25 | finished | 0 | 26.8 min | 70.1 kB |  |
| `20260925T032750Z_htm_env_install` | tm_language_model | htm_env_install | 2026-09-25 | finished | 0 | 17 s | 16.1 kB |  |
| `20260925T033205Z_ssl_r2_hao_head_controls_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 1 | 18.5 min | 63.0 kB |  |
| `20260925T035032Z_ssl_r2_hao_head_r1_hier_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 0 | 16.7 min | 220.8 kB |  |
| `20260925T040549Z_ssl_r1_probe_A5_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-25 | finished | 0 | 2.90 h | 241.2 kB |  |
| `20260925T040714Z_ssl_r2_hao_head_r1_flat_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 0 | 17.7 min | 194.0 kB |  |
| `20260925T042459Z_ssl_r2_hao_head_r2_hier_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 0 | 25.5 min | 248.2 kB |  |
| `20260925T045028Z_ssl_r2_hao_head_r2_flat_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 0 | 27.7 min | 229.4 kB |  |
| `20260925T045714Z_ssl_r1_probe_A2_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-25 | finished | 0 | 2.67 h | 241.0 kB |  |
| `20260925T051810Z_ssl_r2_hao_head_r3_hier_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 1 | 12.4 min | 400.3 kB |  |
| `20260925T053032Z_ssl_r2_hao_head_r3_flat_exploratory` | tm_language_model | ssl_r2_hao_head | 2026-09-25 | finished | 1 | 13.1 min | 54.0 kB |  |
| `20260925T055129Z_ssl_r1_lr_equivalence_A3` | tm_language_model | ssl_r1_lr_equivalence | 2026-09-25 | finished | 1 | 41.6 min | 176.7 kB |  |
| `20260925T063437Z_cvtm_copy_selector_H9_verdicts` | tm_language_model | cvtm_copy_selector | 2026-09-25 | finished | 0 | 1 s | 41.7 kB |  |
| `20260925T072256Z_ssl_r1c_train_A3_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 5.7 min | 98.8 MB |  |
| `20260925T072258Z_ssl_r1c_train_A3_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 5.6 min | 99.0 MB |  |
| `20260925T072836Z_ssl_r1c_census_A3_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 44.3 kB |  |
| `20260925T072837Z_ssl_r1c_census_A3_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 44.3 kB |  |
| `20260925T081210Z_ssl_r1_probe_A6_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-25 | finished | 0 | 2.66 h | 241.0 kB |  |
| `20260925T081210Z_ssl_r1c_features_A3_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 6.0 min | 1.1 GB |  |
| `20260925T081211Z_ssl_r1c_features_A3_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 6.0 min | 1.1 GB |  |
| `20260925T081814Z_ssl_r1c_qpred_A3_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 10.3 min | 1.1 MB |  |
| `20260925T081814Z_ssl_r1c_qpred_A3_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 10.9 min | 1.1 MB |  |
| `20260925T082829Z_ssl_r1c_train_A0_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 14.0 min | 109.2 MB |  |
| `20260925T082910Z_ssl_r1c_train_A0_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 14.8 min | 109.4 MB |  |
| `20260925T084228Z_ssl_r1c_census_A0_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 44.3 kB |  |
| `20260925T084232Z_ssl_r1c_features_A0_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 8.2 min | 1.8 GB |  |
| `20260925T084356Z_ssl_r1c_census_A0_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 4 s | 44.3 kB |  |
| `20260925T084400Z_ssl_r1c_features_A0_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 9.0 min | 1.8 GB |  |
| `20260925T085050Z_ssl_r1c_qpred_A0_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 19.2 min | 1.1 MB |  |
| `20260925T085302Z_ssl_r1c_qpred_A0_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 21.5 min | 1.1 MB |  |
| `20260925T090939Z_ssl_r2_hao_diag_selftest_head` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 43 s | 172.2 kB |  |
| `20260925T090941Z_ssl_r1ca_train_A3s8_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 39.6 min | 108.2 MB |  |
| `20260925T090959Z_ssl_r1c_train_A3T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 53.2 min | 63.9 MB |  |
| `20260925T091022Z_ssl_r2_hao_diag_demos_c85_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 4.53 h | 48.7 kB |  |
| `20260925T091431Z_ssl_r1c_train_A3T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 55.5 min | 63.8 MB |  |
| `20260925T094918Z_ssl_r1ca_census_A3s8_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 39 s | 99.0 kB |  |
| `20260925T094957Z_ssl_r1ca_features_A3s8_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 36.1 min | 2.3 GB |  |
| `20260925T100310Z_ssl_r1c_census_A3T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 42.8 kB |  |
| `20260925T100313Z_ssl_r1c_features_A3T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 1.82 h | 20.2 GB |  |
| `20260925T101000Z_ssl_r1c_census_A3T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 42.8 kB |  |
| `20260925T101003Z_ssl_r1c_features_A3T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 1.81 h | 20.2 GB |  |
| `20260925T102611Z_ssl_r1ca_qpred_A3s8_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 1.73 h | 1.1 MB |  |
| `20260925T105202Z_ssl_r1_report_seed100_exploratory` | tm_language_model | ssl_r1_pretext | 2026-09-25 | finished | 0 | 1 s | 371.1 kB |  |
| `20260925T112628Z_ssl_r2_hao_diag_selftest_head` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 16.8 min | 172.1 kB |  |
| `20260925T114316Z_ssl_r2_hao_diag_valiant_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 19 s | 52.9 kB |  |
| `20260925T114335Z_ssl_r2_hao_diag_controls_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 13.4 min | 46.5 kB |  |
| `20260925T115356Z_ssl_r1c_qpred_A3T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 7.59 h | 1.1 MB |  |
| `20260925T115702Z_ssl_r2_hao_diag_valiant_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 21 s | 52.5 kB |  |
| `20260925T115723Z_ssl_r2_hao_diag_controls_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 0 | 13.7 min | 45.1 kB |  |
| `20260925T115935Z_ssl_r1c_qpred_A3T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 8.04 h | 1.1 MB |  |
| `20260925T120944Z_ssl_r1ca_train_A3s32_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 27.8 min | 105.0 MB |  |
| `20260925T121103Z_ssl_r2_hao_diag_oracle_c85_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-25 | finished | 143 | 6.30 h | 1.6 MB | killed |
| `20260925T123731Z_ssl_r1ca_census_A3s32_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 24 s | 98.9 kB |  |
| `20260925T123755Z_ssl_r1ca_features_A3s32_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 7.1 min | 1.6 GB |  |
| `20260925T124505Z_ssl_r1ca_qpred_A3s32_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 20.5 min | 1.1 MB |  |
| `20260925T124726Z_ssl_r1ca_train_A3s8_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 44.3 min | 108.5 MB |  |
| `20260925T130534Z_ssl_r1ca_train_A3pos_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 57.0 min | 97.5 MB |  |
| `20260925T133145Z_ssl_r1ca_census_A3s8_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 41 s | 99.0 kB |  |
| `20260925T133225Z_ssl_r1ca_features_A3s8_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 50.4 min | 2.3 GB |  |
| `20260925T140233Z_ssl_r1ca_census_A3pos_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 33 s | 98.0 kB |  |
| `20260925T140306Z_ssl_r1ca_features_A3pos_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 48.7 min | 4.2 GB |  |
| `20260925T142254Z_ssl_r1ca_qpred_A3s8_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 3.12 h | 1.1 MB |  |
| `20260925T144847Z_ssl_r1_speedup_verify_features_A3_seed100` | tm_language_model | ssl_r1_speedup_verify | 2026-09-25 | finished | 0 | 9.4 min | 1.3 GB |  |
| `20260925T145203Z_ssl_r1ca_qpred_A3pos_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 1.87 h | 1.1 MB |  |
| `20260925T145814Z_ssl_r1_speedup_verify_qpred_A3_seed100` | tm_language_model | ssl_r1_speedup_verify | 2026-09-25 | finished | 0 | 9.7 min | 1.7 MB |  |
| `20260925T150758Z_ssl_r1_speedup_verify_crosscheck_A3_seed100` | tm_language_model | ssl_r1_speedup_verify | 2026-09-25 | finished | 0 | 30.1 min | 575.0 kB |  |
| `20260925T153833Z_ssl_r1_speedup_timing_A3_seed100` | tm_language_model | ssl_r1_speedup_timing | 2026-09-25 | finished | 143 | 24.7 min | 13.2 kB | killed |
| `20260925T161334Z_ssl_r1_speedup_verify_features_A3_seed100` | tm_language_model | ssl_r1_speedup_verify | 2026-09-25 | finished | 0 | 7.4 min | 1.3 GB |  |
| `20260925T162101Z_ssl_r1_speedup_verify_qpred_A3_seed100` | tm_language_model | ssl_r1_speedup_verify | 2026-09-25 | finished | 0 | 9.9 min | 1.7 MB |  |
| `20260925T163055Z_ssl_r1_speedup_verify_crosscheck_A3_seed100` | tm_language_model | ssl_r1_speedup_verify | 2026-09-25 | finished | 0 | 26.8 min | 575.0 kB |  |
| `20260925T164359Z_ssl_r1ca_census_A3_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 19 s | 98.7 kB |  |
| `20260925T164419Z_ssl_r1ca_census_A3T_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 1.4 min | 96.4 kB |  |
| `20260925T164542Z_ssl_r1ca_probe_A3s8_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 21.7 min | 435.6 kB |  |
| `20260925T165742Z_ssl_r1_speedup_timing_A3_seed100` | tm_language_model | ssl_r1_speedup_timing | 2026-09-25 | finished | 0 | 26.7 min | 16.8 kB |  |
| `20260925T170723Z_ssl_r1ca_probe_A3s32_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 18.0 min | 436.4 kB |  |
| `20260925T172525Z_ssl_r1ca_probe_A3pos_seed100_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 50.4 min | 436.3 kB |  |
| `20260925T173022Z_ssl_r1ca_train_A3s32_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 37.4 min | 105.2 MB |  |
| `20260925T180747Z_ssl_r1ca_census_A3s32_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 23 s | 98.9 kB |  |
| `20260925T180810Z_ssl_r1ca_features_A3s32_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 17.0 min | 1.6 GB |  |
| `20260925T182513Z_ssl_r1ca_qpred_A3s32_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 21.7 min | 1.1 MB |  |
| `20260925T184658Z_ssl_r1ca_train_A3pos_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 13.4 min | 97.6 MB |  |
| `20260925T190021Z_ssl_r1ca_census_A3pos_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 34 s | 97.9 kB |  |
| `20260925T190056Z_ssl_r1ca_features_A3pos_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 9.5 min | 4.2 GB |  |
| `20260925T191034Z_ssl_r1ca_qpred_A3pos_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 36.7 min | 1.1 MB |  |
| `20260925T192910Z_ssl_r1c_train_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 1.54 h | 64.7 MB |  |
| `20260925T194717Z_ssl_r1ca_census_A3_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 19 s | 98.7 kB |  |
| `20260925T194736Z_ssl_r1ca_census_A3T_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 1.3 min | 96.4 kB |  |
| `20260925T194853Z_ssl_r1ca_probe_A3s8_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 20.2 min | 435.5 kB |  |
| `20260925T200146Z_ssl_r1c_train_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 2.24 h | 64.9 MB |  |
| `20260925T200905Z_ssl_r1ca_probe_A3s32_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 17.1 min | 436.4 kB |  |
| `20260925T202611Z_ssl_r1ca_probe_A3pos_seed101_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-25 | finished | 0 | 50.8 min | 436.9 kB |  |
| `20260925T210142Z_ssl_r1c_census_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 42.9 kB |  |
| `20260925T210145Z_ssl_r1c_features_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 1.36 h | 22.7 GB |  |
| `20260925T211721Z_ssl_r1c_probe_A3_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 14.1 min | 432.2 kB |  |
| `20260925T213130Z_ssl_r1c_probe_A0_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 18.2 min | 432.8 kB |  |
| `20260925T214941Z_ssl_r1c_probe_A3T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 143 | 4.8 min | 24.1 kB | killed |
| `20260925T215431Z_ssl_r1c_probe_A3_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 14.2 min | 432.5 kB |  |
| `20260925T220844Z_ssl_r1c_probe_A0_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 18.4 min | 432.3 kB |  |
| `20260925T221721Z_ssl_r1c_census_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 3 s | 42.9 kB |  |
| `20260925T221725Z_ssl_r1c_features_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 0 | 59.8 min | 22.7 GB |  |
| `20260925T222522Z_ssl_r1c_qpred_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 143 | 1.63 h | 568.7 kB | killed |
| `20260925T222705Z_ssl_r1c_probe_A3T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 143 | 4.4 min | 24.1 kB | killed |
| `20260925T231824Z_ssl_r1c_qpred_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-25 | finished | 143 | 1.28 h | 568.5 kB | killed |
| `20260926T000313Z_ssl_r1c_probe_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 143 | 2.2 min | 21.2 kB | killed |
| `20260926T003503Z_ssl_r1c_probe_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 143 | 2.1 min | 21.2 kB | killed |
| `20260926T003809Z_ssl_r1c_report_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 0 s | 49.1 kB |  |
| `20260926T003930Z_ssl_r1c_probe_A3T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 3.69 h | 432.9 kB |  |
| `20260926T003930Z_ssl_r1c_probe_A3T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 3.58 h | 432.7 kB |  |
| `20260926T041408Z_ssl_r1c_qpred_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 8.37 h | 1.1 MB |  |
| `20260926T042044Z_ssl_r1c_qpred_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 8.49 h | 1.1 MB |  |
| `20260926T123619Z_ssl_r1c_probe_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 143 | 6.7 min | 35.6 kB | killed |
| `20260926T124956Z_ssl_r1c_probe_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 143 | 6.6 min | 35.5 kB | killed |
| `20260926T125731Z_ssl_r1c_report_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 0 s | 53.0 kB |  |
| `20260926T125917Z_ssl_r1c_probe_A0T_seed100_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 3.85 h | 433.2 kB |  |
| `20260926T165005Z_ssl_r1c_probe_A0T_seed101_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 3.84 h | 433.2 kB |  |
| `20260926T204018Z_ssl_r1c_report_exploratory` | tm_language_model | ssl_r1_controls | 2026-09-26 | finished | 0 | 1 s | 56.2 kB |  |
| `20260926T204447Z_ssl_r1cf_testhash_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 3 s | 58.0 kB |  |
| `20260926T204450Z_ssl_r1cf_common_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 4.0 min | 2.2 GB |  |
| `20260926T204450Z_ssl_r1cf_testhash_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 1 s | 55.6 kB |  |
| `20260926T204452Z_ssl_r1cf_testhash_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 1 s | 55.6 kB |  |
| `20260926T204452Z_ssl_r1cf_train_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 23 s | 61.1 kB |  |
| `20260926T204453Z_ssl_r1cf_testhash_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 1 s | 55.6 kB |  |
| `20260926T204453Z_ssl_r1cf_train_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 22 s | 61.1 kB |  |
| `20260926T204455Z_ssl_r1cf_train_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 23 s | 61.1 kB |  |
| `20260926T204515Z_ssl_r1cf_train_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 23 s | 61.6 kB |  |
| `20260926T204515Z_ssl_r1cf_train_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 23 s | 61.6 kB |  |
| `20260926T204518Z_ssl_r1cf_train_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 22 s | 61.6 kB |  |
| `20260926T204538Z_ssl_r1cf_train_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 14 s | 61.3 kB |  |
| `20260926T204538Z_ssl_r1cf_train_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 13 s | 61.3 kB |  |
| `20260926T204540Z_ssl_r1cf_train_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 12 s | 61.3 kB |  |
| `20260926T204551Z_ssl_r1cf_train_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 12 s | 61.8 kB |  |
| `20260926T204552Z_ssl_r1cf_train_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 12 s | 61.8 kB |  |
| `20260926T204553Z_ssl_r1cf_train_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 13 s | 61.8 kB |  |
| `20260926T204603Z_ssl_r1cf_train_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 11 s | 61.1 kB |  |
| `20260926T204605Z_ssl_r1cf_train_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 12 s | 61.1 kB |  |
| `20260926T204605Z_ssl_r1cf_train_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 12 s | 61.1 kB |  |
| `20260926T204615Z_ssl_r1cf_train_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 13 s | 61.2 kB |  |
| `20260926T204617Z_ssl_r1cf_train_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 13 s | 61.2 kB |  |
| `20260926T204617Z_ssl_r1cf_train_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 1 | 13 s | 61.2 kB |  |
| `20260926T204857Z_ssl_r1cf_probe_common_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 2.17 h | 11.0 GB |  |
| `20260926T205141Z_ssl_r1cf_testhash_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1 s | 64.9 kB |  |
| `20260926T205142Z_ssl_r1cf_train_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 7.4 min | 98.9 MB |  |
| `20260926T205142Z_ssl_r1cf_train_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 7.0 min | 98.9 MB |  |
| `20260926T205142Z_ssl_r1cf_train_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 7.4 min | 99.1 MB |  |
| `20260926T205845Z_ssl_r1cf_census_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 19 s | 130.0 kB |  |
| `20260926T205903Z_ssl_r1cf_features_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 6.4 min | 3.3 GB |  |
| `20260926T205908Z_ssl_r1cf_census_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 19 s | 130.0 kB |  |
| `20260926T205909Z_ssl_r1cf_census_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 19 s | 130.0 kB |  |
| `20260926T205927Z_ssl_r1cf_features_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 7.0 min | 3.3 GB |  |
| `20260926T205928Z_ssl_r1cf_features_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 7.0 min | 3.3 GB |  |
| `20260926T210534Z_ssl_r1cf_probe_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.42 h | 8.1 GB |  |
| `20260926T210636Z_ssl_r1cf_probe_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.45 h | 8.1 GB |  |
| `20260926T210637Z_ssl_r1cf_probe_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.48 h | 8.1 GB |  |
| `20260926T223126Z_ssl_r1cf_fullfeat_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.9 min | 1.7 GB |  |
| `20260926T223325Z_ssl_r1cf_probefull_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 2.29 h | 5.2 MB |  |
| `20260926T223351Z_ssl_r1cf_fullfeat_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.8 min | 1.7 GB |  |
| `20260926T223546Z_ssl_r1cf_probefull_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 2.17 h | 5.1 MB |  |
| `20260926T223601Z_ssl_r1cf_fullfeat_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.9 min | 1.7 GB |  |
| `20260926T223759Z_ssl_r1cf_probefull_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 2.51 h | 5.1 MB |  |
| `20260926T225937Z_ssl_r1cf_randbank_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 2.2 min | 2.2 GB |  |
| `20260926T230154Z_ssl_r1cf_probe_rand_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-26 | finished | 0 | 1.67 h | 73.7 MB |  |
| `20260927T004156Z_ssl_r1cf_randbank_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 2.2 min | 2.2 GB |  |
| `20260927T004416Z_ssl_r1cf_probe_rand_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | running |  |  | 39.7 MB | stale (recorded as running) |
| `20260927T004606Z_ssl_r1cf_confirm_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 2.4 min | 2.4 MB |  |
| `20260927T004827Z_ssl_r1cf_train_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 34.3 min | 63.8 MB |  |
| `20260927T005056Z_ssl_r1cf_confirm_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 3.3 min | 2.4 MB |  |
| `20260927T005415Z_ssl_r1cf_train_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 50.4 min | 63.9 MB |  |
| `20260927T010853Z_ssl_r1cf_confirm_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 4.2 min | 2.4 MB |  |
| `20260927T011306Z_ssl_r1cf_train_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 54.6 min | 63.8 MB |  |
| `20260927T012246Z_ssl_r1cf_census_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 1.3 min | 127.8 kB |  |
| `20260927T012405Z_ssl_r1cf_features_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 26.1 min | 7.9 GB |  |
| `20260927T014437Z_ssl_r1cf_census_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 1.4 min | 127.7 kB |  |
| `20260927T014601Z_ssl_r1cf_features_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 20.9 min | 8.0 GB |  |
| `20260927T015033Z_ssl_r1cf_probe_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | running |  |  | 71.3 kB | stale (recorded as running) |
| `20260927T020716Z_ssl_r1cf_probe_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | running |  |  | 63.4 kB | stale (recorded as running) |
| `20260927T020744Z_ssl_r1cf_census_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | finished | 0 | 1.3 min | 127.7 kB |  |
| `20260927T020905Z_ssl_r1cf_features_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-27 | running |  |  | 5.1 GB | stale (recorded as running) |
| `20260928T203357Z_ssl_r1cf_confirm_rand_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 2.2 min | 1.6 MB |  |
| `20260928T203609Z_ssl_r1cf_features_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 8.6 min | 8.0 GB |  |
| `20260928T203609Z_ssl_r1cf_probe_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 3.81 h | 137.0 MB |  |
| `20260928T203609Z_ssl_r1cf_probe_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 3.88 h | 137.1 MB |  |
| `20260928T203609Z_ssl_r1cf_probe_rand_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 1.66 h | 73.8 MB |  |
| `20260928T204505Z_ssl_r1cf_probe_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 3.83 h | 137.2 MB |  |
| `20260928T221547Z_ssl_r1cf_confirm_rand_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 2.1 min | 1.6 MB |  |
| `20260928T221753Z_ssl_r1cf_randbank_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 2.2 min | 2.2 GB |  |
| `20260928T222008Z_ssl_r1cf_probe_rand_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-28 | finished | 0 | 1.71 h | 73.9 MB |  |
| `20260929T000252Z_ssl_r1cf_confirm_rand_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.2 min | 1.6 MB |  |
| `20260929T000504Z_ssl_r1cf_fullfeat_common_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.4 min | 11.5 GB |  |
| `20260929T000757Z_ssl_r1cf_probefull_common_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 13.99 h | 8.6 MB |  |
| `20260929T002433Z_ssl_r1cf_confirm_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.1 min | 1.6 MB |  |
| `20260929T002838Z_ssl_r1cf_train_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.7 min | 88.3 MB |  |
| `20260929T002903Z_ssl_r1cf_confirm_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.2 min | 1.6 MB |  |
| `20260929T003318Z_ssl_r1cf_census_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 19 s | 130.2 kB |  |
| `20260929T003318Z_ssl_r1cf_train_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 5.2 min | 88.2 MB |  |
| `20260929T003337Z_ssl_r1cf_features_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.9 min | 2.8 GB |  |
| `20260929T003453Z_ssl_r1cf_confirm_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.3 min | 1.6 MB |  |
| `20260929T003737Z_ssl_r1cf_probe_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.37 h | 8.2 GB |  |
| `20260929T003908Z_ssl_r1cf_census_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 18 s | 130.2 kB |  |
| `20260929T003908Z_ssl_r1cf_train_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 5.0 min | 88.0 MB |  |
| `20260929T003926Z_ssl_r1cf_features_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.8 min | 2.8 GB |  |
| `20260929T004321Z_ssl_r1cf_probe_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.36 h | 8.2 GB |  |
| `20260929T004410Z_ssl_r1cf_census_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 18 s | 130.2 kB |  |
| `20260929T004429Z_ssl_r1cf_features_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.2 min | 2.8 GB |  |
| `20260929T004745Z_ssl_r1cf_probe_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.37 h | 8.2 GB |  |
| `20260929T015953Z_ssl_r1cf_fullfeat_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.7 min | 1.4 GB |  |
| `20260929T020142Z_ssl_r1cf_probefull_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.74 h | 6.7 MB |  |
| `20260929T020535Z_ssl_r1cf_fullfeat_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.7 min | 1.3 GB |  |
| `20260929T020723Z_ssl_r1cf_probefull_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.61 h | 6.7 MB |  |
| `20260929T021025Z_ssl_r1cf_fullfeat_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.7 min | 1.3 GB |  |
| `20260929T021210Z_ssl_r1cf_probefull_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.61 h | 6.7 MB |  |
| `20260929T034412Z_ssl_r1cf_confirm_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.1 min | 2.4 MB |  |
| `20260929T034619Z_ssl_r1cf_confirm_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.6 min | 2.4 MB |  |
| `20260929T034619Z_ssl_r1cf_train_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 19.8 min | 55.7 MB |  |
| `20260929T034858Z_ssl_r1cf_confirm_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.2 min | 2.4 MB |  |
| `20260929T034858Z_ssl_r1cf_train_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 26.2 min | 55.8 MB |  |
| `20260929T035212Z_ssl_r1cf_train_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 26.0 min | 55.6 MB |  |
| `20260929T040608Z_ssl_r1cf_census_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 33 s | 127.7 kB |  |
| `20260929T040641Z_ssl_r1cf_features_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 6.1 min | 2.7 GB |  |
| `20260929T041255Z_ssl_r1cf_probe_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.31 h | 108.9 MB |  |
| `20260929T041508Z_ssl_r1cf_census_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 32 s | 127.6 kB |  |
| `20260929T041541Z_ssl_r1cf_features_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.9 min | 2.7 GB |  |
| `20260929T041815Z_ssl_r1cf_census_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 32 s | 127.6 kB |  |
| `20260929T041847Z_ssl_r1cf_features_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.8 min | 2.7 GB |  |
| `20260929T041945Z_ssl_r1cf_probe_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.32 h | 108.9 MB |  |
| `20260929T042143Z_ssl_r1cf_probe_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.31 h | 108.6 MB |  |
| `20260929T053135Z_ssl_r1cf_confirm_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.5 min | 1.6 MB |  |
| `20260929T053308Z_ssl_r1cf_train_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 8.0 min | 109.4 MB |  |
| `20260929T053849Z_ssl_r1cf_confirm_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.6 min | 1.6 MB |  |
| `20260929T054028Z_ssl_r1cf_confirm_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.6 min | 1.6 MB |  |
| `20260929T054028Z_ssl_r1cf_train_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 17.2 min | 109.3 MB |  |
| `20260929T054205Z_ssl_r1cf_census_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 24 s | 130.4 kB |  |
| `20260929T054205Z_ssl_r1cf_train_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 18.0 min | 109.3 MB |  |
| `20260929T054229Z_ssl_r1cf_features_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 7.7 min | 5.2 GB |  |
| `20260929T055025Z_ssl_r1cf_probe_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.48 h | 8.2 GB |  |
| `20260929T055743Z_ssl_r1cf_census_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 24 s | 130.3 kB |  |
| `20260929T055807Z_ssl_r1cf_features_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 7.0 min | 5.3 GB |  |
| `20260929T060008Z_ssl_r1cf_census_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 24 s | 130.5 kB |  |
| `20260929T060032Z_ssl_r1cf_features_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 6.4 min | 5.3 GB |  |
| `20260929T060520Z_ssl_r1cf_probe_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.51 h | 8.2 GB |  |
| `20260929T060711Z_ssl_r1cf_probe_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.52 h | 8.2 GB |  |
| `20260929T071950Z_ssl_r1cf_confirm_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.5 min | 2.4 MB |  |
| `20260929T072219Z_ssl_r1cf_train_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.3 min | 106.3 MB |  |
| `20260929T072639Z_ssl_r1cf_census_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 23 s | 130.5 kB |  |
| `20260929T072702Z_ssl_r1cf_features_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.5 min | 4.3 GB |  |
| `20260929T073143Z_ssl_r1cf_probe_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.43 h | 8.2 GB |  |
| `20260929T073609Z_ssl_r1cf_confirm_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.5 min | 2.4 MB |  |
| `20260929T073840Z_ssl_r1cf_train_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 5.1 min | 106.4 MB |  |
| `20260929T073854Z_ssl_r1cf_confirm_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.8 min | 2.4 MB |  |
| `20260929T074140Z_ssl_r1cf_train_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 7.8 min | 106.3 MB |  |
| `20260929T074346Z_ssl_r1cf_census_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 21 s | 130.4 kB |  |
| `20260929T074407Z_ssl_r1cf_features_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 5.7 min | 4.3 GB |  |
| `20260929T074932Z_ssl_r1cf_census_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 24 s | 130.4 kB |  |
| `20260929T074956Z_ssl_r1cf_features_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.5 min | 4.3 GB |  |
| `20260929T075000Z_ssl_r1cf_probe_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.45 h | 8.2 GB |  |
| `20260929T075440Z_ssl_r1cf_probe_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.51 h | 8.2 GB |  |
| `20260929T085759Z_ssl_r1cf_confirm_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.3 min | 2.4 MB |  |
| `20260929T090020Z_ssl_r1cf_qcon_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.6 min | 68.4 kB |  |
| `20260929T090157Z_ssl_r1cf_qint_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.7 min | 552.3 kB |  |
| `20260929T090538Z_ssl_r1cf_qcon_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.7 min | 68.4 kB |  |
| `20260929T090718Z_ssl_r1cf_qint_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.5 min | 560.4 kB |  |
| `20260929T091049Z_ssl_r1cf_train_A6_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.3 min | 106.4 MB |  |
| `20260929T091510Z_ssl_r1cf_census_A6_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 23 s | 130.4 kB |  |
| `20260929T091533Z_ssl_r1cf_features_A6_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.6 min | 4.6 GB |  |
| `20260929T091736Z_ssl_r1cf_confirm_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.5 min | 2.4 MB |  |
| `20260929T092009Z_ssl_r1cf_qcon_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.6 min | 68.4 kB |  |
| `20260929T092018Z_ssl_r1cf_probe_A6_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.51 h | 8.2 GB |  |
| `20260929T092143Z_ssl_r1cf_qint_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.6 min | 550.7 kB |  |
| `20260929T092519Z_ssl_r1cf_qcon_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.5 min | 68.4 kB |  |
| `20260929T092649Z_ssl_r1cf_confirm_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.2 min | 2.4 MB |  |
| `20260929T092901Z_ssl_r1cf_qint_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.5 min | 559.7 kB |  |
| `20260929T093228Z_ssl_r1cf_qcon_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.7 min | 68.4 kB |  |
| `20260929T093228Z_ssl_r1cf_train_A6_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.6 min | 106.5 MB |  |
| `20260929T093408Z_ssl_r1cf_qint_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.8 min | 546.7 kB |  |
| `20260929T093706Z_ssl_r1cf_census_A6_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 22 s | 130.4 kB |  |
| `20260929T093729Z_ssl_r1cf_features_A6_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.8 min | 4.6 GB |  |
| `20260929T093754Z_ssl_r1cf_qcon_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.8 min | 68.4 kB |  |
| `20260929T093944Z_ssl_r1cf_qint_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.6 min | 559.0 kB |  |
| `20260929T094231Z_ssl_r1cf_probe_A6_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.52 h | 8.2 GB |  |
| `20260929T094318Z_ssl_r1cf_train_A6_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.3 min | 106.4 MB |  |
| `20260929T094738Z_ssl_r1cf_census_A6_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 21 s | 130.4 kB |  |
| `20260929T094800Z_ssl_r1cf_features_A6_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.5 min | 4.6 GB |  |
| `20260929T095241Z_ssl_r1cf_probe_A6_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.46 h | 8.2 GB |  |
| `20260929T105107Z_ssl_r1cf_confirm_A6_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.2 min | 2.4 MB |  |
| `20260929T105319Z_ssl_r1cf_train_A3R_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.9 min | 99.0 MB |  |
| `20260929T105715Z_ssl_r1cf_census_A3R_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 30 s | 130.0 kB |  |
| `20260929T105746Z_ssl_r1cf_features_A3R_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.4 min | 3.4 GB |  |
| `20260929T110118Z_ssl_r1cf_probe_A3R_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.42 h | 8.1 GB |  |
| `20260929T111406Z_ssl_r1cf_confirm_A6_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.2 min | 2.4 MB |  |
| `20260929T111616Z_ssl_r1cf_train_A3R_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.0 min | 98.8 MB |  |
| `20260929T112017Z_ssl_r1cf_census_A3R_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 19 s | 130.0 kB |  |
| `20260929T112036Z_ssl_r1cf_features_A3R_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.5 min | 3.4 GB |  |
| `20260929T112053Z_ssl_r1cf_confirm_A6_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.4 min | 2.4 MB |  |
| `20260929T112318Z_ssl_r1cf_train_A3R_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.1 min | 99.1 MB |  |
| `20260929T112412Z_ssl_r1cf_probe_A3R_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.42 h | 8.1 GB |  |
| `20260929T112725Z_ssl_r1cf_census_A3R_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 19 s | 129.9 kB |  |
| `20260929T112745Z_ssl_r1cf_features_A3R_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.4 min | 3.4 GB |  |
| `20260929T113119Z_ssl_r1cf_probe_A3R_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 1.43 h | 8.1 GB |  |
| `20260929T122700Z_ssl_r1cf_confirm_A3R_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.1 min | 2.4 MB |  |
| `20260929T122906Z_ssl_r1cf_qpred_A3_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 13.9 min | 6.5 MB |  |
| `20260929T124304Z_ssl_r1cf_qpred_A3T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.15 h | 3.2 MB |  |
| `20260929T145215Z_ssl_r1cf_confirm_A3R_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.2 min | 2.4 MB |  |
| `20260929T145430Z_ssl_r1cf_confirm_A3R_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 2.2 min | 2.4 MB |  |
| `20260929T145640Z_ssl_r1cf_confirm_common_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 143 | 2.9 min | 49.4 kB | killed |
| `20260929T145934Z_ssl_r1cf_qpred_A2_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 15.9 min | 6.5 MB |  |
| `20260929T145934Z_ssl_r1cf_qpred_A3_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 17.1 min | 6.5 MB |  |
| `20260929T145934Z_ssl_r1cf_qpred_A3_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 18.2 min | 6.5 MB |  |
| `20260929T151526Z_ssl_r1cf_qpred_A2T_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 39.2 min | 3.2 MB |  |
| `20260929T151643Z_ssl_r1cf_qpred_A3T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 4.15 h | 3.2 MB |  |
| `20260929T151745Z_ssl_r1cf_qpred_A3T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 3.99 h | 3.2 MB |  |
| `20260929T155436Z_ssl_r1cf_qpred_A0_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 53.3 min | 6.5 MB |  |
| `20260929T164756Z_ssl_r1cf_qpred_A1_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 45.0 min | 6.5 MB |  |
| `20260929T173254Z_ssl_r1cf_qpred_A6_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 43.2 min | 6.5 MB |  |
| `20260929T181609Z_ssl_r1cf_qpred_A3R_seed2_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 15.2 min | 6.5 MB |  |
| `20260929T191654Z_ssl_r1cf_qpred_A2_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 12.0 min | 6.5 MB |  |
| `20260929T192544Z_ssl_r1cf_qpred_A2_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 14.9 min | 6.5 MB |  |
| `20260929T192852Z_ssl_r1cf_qpred_A2T_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 28.3 min | 3.2 MB |  |
| `20260929T194040Z_ssl_r1cf_qpred_A2T_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 26.2 min | 3.2 MB |  |
| `20260929T195712Z_ssl_r1cf_qpred_A0_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 31.1 min | 6.5 MB |  |
| `20260929T200651Z_ssl_r1cf_qpred_A0_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 33.5 min | 6.5 MB |  |
| `20260929T202815Z_ssl_r1cf_qpred_A1_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 26.4 min | 6.5 MB |  |
| `20260929T204024Z_ssl_r1cf_qpred_A1_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 26.5 min | 6.5 MB |  |
| `20260929T205442Z_ssl_r1cf_qpred_A6_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 27.3 min | 6.5 MB |  |
| `20260929T210652Z_ssl_r1cf_qpred_A6_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 24.5 min | 6.5 MB |  |
| `20260929T212159Z_ssl_r1cf_qpred_A3R_seed3_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 16.5 min | 6.5 MB |  |
| `20260929T213121Z_ssl_r1cf_qpred_A3R_seed1_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 15.5 min | 6.5 MB |  |
| `20260929T214742Z_ssl_r1cf_report_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-29 | finished | 0 | 13 s | 666.1 kB |  |
| `20260929T215246Z_ssl_r2_hao_diag_selftest_head` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 34 s | 173.6 kB |  |
| `20260929T215320Z_ssl_r2_hao_diag_oracle_c85_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 29.5 min | 3.0 MB |  |
| `20260929T222249Z_ssl_r2_hao_diag_oracle_head_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 30.0 min | 3.0 MB |  |
| `20260929T225249Z_ssl_r2_hao_diag_seen_unseen_c85_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 10.3 min | 458.6 kB |  |
| `20260929T230306Z_ssl_r2_hao_diag_pospair_c85_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 1.8 min | 71.0 kB |  |
| `20260929T230454Z_ssl_r2_hao_diag_seen_unseen_head_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 8.3 min | 426.6 kB |  |
| `20260929T231314Z_ssl_r2_hao_diag_pospair_head_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 2.3 min | 73.7 kB |  |
| `20260929T231535Z_ssl_r2_hao_diag_threshold_head_seed100_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-29 | finished | 0 | 2.01 h | 762.8 kB |  |
| `20260930T011555Z_ssl_r2_hao_diag_oracle_c85_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 29.3 min | 2.8 MB |  |
| `20260930T014512Z_ssl_r2_hao_diag_oracle_head_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 30.5 min | 2.8 MB |  |
| `20260930T021540Z_ssl_r2_hao_diag_seen_unseen_c85_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 10.1 min | 339.0 kB |  |
| `20260930T022546Z_ssl_r2_hao_diag_pospair_c85_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 1.8 min | 69.5 kB |  |
| `20260930T022737Z_ssl_r2_hao_diag_seen_unseen_head_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 8.2 min | 298.0 kB |  |
| `20260930T023551Z_ssl_r2_hao_diag_pospair_head_seed101_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 2.4 min | 72.1 kB |  |
| `20260930T023812Z_ssl_r2_hao_diag_demos_head_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 1.62 h | 241.0 kB |  |
| `20260930T041535Z_ssl_r2_hao_diag_report_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 1 s | 106.5 kB |  |
| `20260930T163059Z_ssl_r1cf_confirm_common_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-30 | finished | 0 | 5.3 min | 2.2 MB |  |
| `20260930T163619Z_ssl_r1cf_report_confirmatory` | tm_language_model | ssl_r1_confirm | 2026-09-30 | finished | 0 | 14 s | 705.6 kB |  |
| `20260930T163807Z_ssl_r2_hao_diag_demos_c85_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 1.02 h | 268.0 kB |  |
| `20260930T164322Z_ssl_r2_hao_cpu_confirmatory` | tm_language_model | ssl_r2_hao | 2026-09-30 | finished | 0 | 3.82 h | 65.7 kB |  |
| `20260930T165149Z_ssl_r4_data_smoke` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 32 s | 14.0 kB | smoke |
| `20260930T165231Z_ssl_r4_gpu_smoke` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 3 s | 21.1 MB | smoke |
| `20260930T165444Z_ssl_r4_probe_smoke` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 2.9 min | 24.3 kB | smoke |
| `20260930T170123Z_ssl_r4c_replay_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 9 s | 23.0 MB | smoke |
| `20260930T170148Z_ssl_r4c_replay_perturbed_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 3 s | 23.0 MB | smoke |
| `20260930T170200Z_ssl_r4c_twin_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 3 s | 37.6 MB | smoke |
| `20260930T170204Z_ssl_r4c_randbank_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 4 s | 37.8 MB | smoke |
| `20260930T170223Z_ssl_r4c_diag_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 1 s | 17.0 kB | smoke |
| `20260930T170252Z_ssl_r4c_diag_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 1 s | 59.8 kB | smoke |
| `20260930T170302Z_ssl_r4c_probe_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 10.6 min | 35.2 kB | smoke |
| `20260930T171512Z_ssl_r4c_report_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 1 s | 32.4 kB | smoke |
| `20260930T171525Z_ssl_r4c_report_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 0 s | 103.1 kB | smoke |
| `20260930T171702Z_ssl_r4c_randbank_chunk40_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 3 s | 37.8 MB | smoke |
| `20260930T171732Z_ssl_r4c_replay_lane_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 3 s | 23.0 MB | smoke |
| `20260930T171735Z_ssl_r4c_twin_lane_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 3 s | 37.6 MB | smoke |
| `20260930T171739Z_ssl_r4c_randbank_lane_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 3 s | 37.8 MB | smoke |
| `20260930T171742Z_ssl_r4c_diag_lane_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 1 s | 60.2 kB | smoke |
| `20260930T171744Z_ssl_r4c_probe_lane_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 10.5 min | 35.7 kB | smoke |
| `20260930T171926Z_ssl_r4c_diag_final_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 5 s | 60.2 kB | smoke |
| `20260930T173910Z_ssl_r2_hao_diag_report_exploratory` | tm_language_model | ssl_r2_hao_diagnostics | 2026-09-30 | finished | 0 | 0 s | 106.5 kB |  |
| `20260930T173922Z_ssl_r4_gpu_seed1_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 2.0 min | 1.8 GB |  |
| `20260930T174129Z_ssl_r4_gpu_seed2_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 2.1 min | 1.8 GB |  |
| `20260930T174322Z_ssl_r4_probe_seed1_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 1.74 h | 45.2 kB |  |
| `20260930T174338Z_ssl_r4_gpu_seed3_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 2.1 min | 1.8 GB |  |
| `20260930T174528Z_ssl_r4_probe_seed2_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 1.70 h | 45.3 kB |  |
| `20260930T174548Z_ssl_r2_hao_gpu_confirmatory` | tm_language_model | ssl_r2_hao | 2026-09-30 | finished | 0 | 1.04 h | 124.8 kB |  |
| `20260930T174725Z_ssl_r4_probe_seed3_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 1.76 h | 45.2 kB |  |
| `20260930T184800Z_ssl_r2_hao_control_confirmatory` | tm_language_model | ssl_r2_hao | 2026-09-30 | finished | 0 | 34.3 min | 42.3 kB |  |
| `20260930T192451Z_ssl_r4c_replay_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 2.3 min | 1.2 GB |  |
| `20260930T192451Z_ssl_r4c_replay_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 2.3 min | 1.2 GB |  |
| `20260930T192713Z_ssl_r4c_twin_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 4.0 min | 1.7 GB |  |
| `20260930T192713Z_ssl_r4c_twin_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 4.1 min | 1.7 GB |  |
| `20260930T192728Z_ssl_r4_probe_tm_seed2_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 46.2 min | 22.2 kB |  |
| `20260930T192800Z_ssl_r4_probe_tm_seed1_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 45.3 min | 22.2 kB |  |
| `20260930T193121Z_ssl_r4c_randbank_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 2.2 min | 5.0 GB | killed |
| `20260930T193121Z_ssl_r4c_randbank_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 2.2 min | 5.1 GB | killed |
| `20260930T193248Z_ssl_r4_probe_tm_seed3_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 44.6 min | 22.2 kB |  |
| `20260930T193337Z_ssl_r4c_diag_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 43 s | 76.2 kB |  |
| `20260930T193337Z_ssl_r4c_diag_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 43 s | 76.1 kB |  |
| `20260930T193420Z_ssl_r4c_probe_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 0 s | 21.8 kB |  |
| `20260930T193420Z_ssl_r4c_probe_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 0 s | 21.8 kB |  |
| `20260930T193421Z_ssl_r4c_replay_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 1.3 min | 1.2 GB |  |
| `20260930T193540Z_ssl_r4c_twin_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 2.0 min | 1.7 GB |  |
| `20260930T193746Z_ssl_r4c_randbank_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 1.4 min | 5.1 GB | killed |
| `20260930T193925Z_ssl_r4c_diag_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 14 s | 76.2 kB |  |
| `20260930T193939Z_ssl_r4c_probe_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 0 s | 21.8 kB |  |
| `20260930T201951Z_ssl_r4c_report` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 1 | 0 s | 222.7 kB |  |
| `20260930T203322Z_ssl_r2_hao_report_confirmatory` | tm_language_model | ssl_r2_hao | 2026-09-30 | finished | 0 | 1 s | 19.6 kB |  |
| `20260930T203322Z_ssl_r4_report_confirmatory` | tm_language_model | ssl_r4_images | 2026-09-30 | finished | 0 | 0 s | 45.9 kB |  |
| `20260930T203529Z_ssl_r4c_randbank_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 2.3 min | 5.0 GB |  |
| `20260930T203757Z_ssl_r4c_diag_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 16 s | 76.2 kB |  |
| `20260930T203757Z_ssl_r4c_randbank_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 2.3 min | 5.1 GB |  |
| `20260930T203813Z_ssl_r4c_probe_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 44 s | 18.8 kB | killed |
| `20260930T203828Z_ssl_r1ca_report_exploratory` | tm_language_model | ssl_r1_config_arms | 2026-09-30 | finished | 0 | 1 s | 176.3 kB |  |
| `20260930T204009Z_ssl_r4c_probe_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 54 s | 18.8 kB | killed |
| `20260930T204025Z_ssl_r4c_diag_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 16 s | 76.2 kB |  |
| `20260930T204025Z_ssl_r4c_randbank_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 2.3 min | 5.1 GB |  |
| `20260930T204041Z_ssl_r4c_probe_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 45 s | 18.8 kB | killed |
| `20260930T204209Z_ssl_r4c_probe_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 52 s | 18.8 kB | killed |
| `20260930T204238Z_ssl_r4c_probe_lazycheck_smoke` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 10.9 min | 35.6 kB | smoke |
| `20260930T204256Z_ssl_r4c_diag_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 0 | 11 s | 76.1 kB |  |
| `20260930T204307Z_ssl_r4c_probe_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 2.40 h | 57.6 kB | killed |
| `20260930T205358Z_ssl_r4c_probe_seed1` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 2.43 h | 57.5 kB | killed |
| `20260930T205358Z_ssl_r4c_probe_seed2` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 2.41 h | 57.6 kB | killed |
| `20260930T230810Z_ssl_r4c_probe_seed3` | tm_language_model | ssl_r4_controls | 2026-09-30 | finished | 143 | 2.44 h | 57.6 kB | killed |
| `20261001T013607Z_ssl_r4c_probe_seed1` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 2.37 h | 58.1 kB |  |
| `20261001T013607Z_ssl_r4c_probe_seed2` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 2.46 h | 58.1 kB |  |
| `20261001T013607Z_ssl_r4c_probe_seed3` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 2.49 h | 58.1 kB |  |
| `20261001T040607Z_ssl_r4c_probe_randbank_seed1` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 3.95 h | 29.5 kB |  |
| `20261001T064508Z_feas_slide11_baselines_exploratory` | tm_language_model | feas_slide11_baselines_exploratory | 2026-10-01 | finished | 0 | 1.1 min | 19.9 kB | biased exploratory |
| `20261001T064733Z_feas_slide11_smoke_hier_exploratory` | tm_language_model | feas_slide11_smoke_hier_exploratory | 2026-10-01 | finished | 0 | 9 s | 28.4 kB | smoke, biased exploratory |
| `20261001T064759Z_feas_slide11_h1_F1024_C200_bin_exploratory` | tm_language_model | feas_slide11_h1_F1024_C200_bin_exploratory | 2026-10-01 | finished | 0 | 1.4 min | 31.0 kB | biased exploratory |
| `20261001T064923Z_feas_slide11_f1_F1024_C200_exploratory` | tm_language_model | feas_slide11_f1_F1024_C200_exploratory | 2026-10-01 | finished | 0 | 1.1 min | 27.9 kB | biased exploratory |
| `20261001T065117Z_feas_slide11_sw_h_s15_exploratory` | tm_language_model | feas_slide11_sw_h_s15_exploratory | 2026-10-01 | finished | 0 | 55 s | 29.7 kB | biased exploratory |
| `20261001T065212Z_feas_slide11_sw_f_s15_exploratory` | tm_language_model | feas_slide11_sw_f_s15_exploratory | 2026-10-01 | finished | 0 | 38 s | 26.5 kB | biased exploratory |
| `20261001T065250Z_feas_slide11_sw_h_s40_exploratory` | tm_language_model | feas_slide11_sw_h_s40_exploratory | 2026-10-01 | finished | 0 | 53 s | 29.5 kB | biased exploratory |
| `20261001T065343Z_feas_slide11_sw_f_s40_exploratory` | tm_language_model | feas_slide11_sw_f_s40_exploratory | 2026-10-01 | finished | 0 | 36 s | 26.6 kB | biased exploratory |
| `20261001T065419Z_feas_slide11_sw_h_s100_exploratory` | tm_language_model | feas_slide11_sw_h_s100_exploratory | 2026-10-01 | finished | 0 | 49 s | 30.1 kB | biased exploratory |
| `20261001T065508Z_feas_slide11_sw_f_s100_exploratory` | tm_language_model | feas_slide11_sw_f_s100_exploratory | 2026-10-01 | finished | 0 | 32 s | 26.6 kB | biased exploratory |
| `20261001T065603Z_feas_slide11_h2_F1024_C2000_bin_s40_exploratory` | tm_language_model | feas_slide11_h2_F1024_C2000_bin_s40_exploratory | 2026-10-01 | finished | 0 | 9.3 min | 33.3 kB | biased exploratory |
| `20261001T070552Z_feas_slide11_ctl_p1word_flat_exploratory` | tm_language_model | feas_slide11_ctl_p1word_flat_exploratory | 2026-10-01 | finished | 0 | 11.7 min | 27.9 kB | biased exploratory |
| `20261001T071645Z_feas_slide11_posthoc_exploratory` | tm_language_model | feas_slide11_posthoc_exploratory | 2026-10-01 | finished | 143 | 10.2 min | 21.1 kB | killed, biased exploratory |
| `20261001T071735Z_feas_slide11_ctl_ov2_flat_exploratory` | tm_language_model | feas_slide11_ctl_ov2_flat_exploratory | 2026-10-01 | finished | 0 | 10.1 min | 28.0 kB | biased exploratory |
| `20261001T072714Z_feas_slide11_posthoc_exploratory` | tm_language_model | feas_slide11_posthoc_exploratory | 2026-10-01 | finished | 0 | 31 s | 23.7 kB | biased exploratory |
| `20261001T072741Z_feas_slide11_ctl_ov2_hier_exploratory` | tm_language_model | feas_slide11_ctl_ov2_hier_exploratory | 2026-10-01 | finished | 0 | 37 s | 31.1 kB | biased exploratory |
| `20261001T072821Z_feas_slide11_q_h_off_C1000_exploratory` | tm_language_model | feas_slide11_q_h_off_C1000_exploratory | 2026-10-01 | finished | 0 | 1.1 min | 34.1 kB | biased exploratory |
| `20261001T072930Z_feas_slide11_q_h_norm_C1000_exploratory` | tm_language_model | feas_slide11_q_h_norm_C1000_exploratory | 2026-10-01 | finished | 0 | 1.0 min | 33.7 kB | biased exploratory |
| `20261001T073033Z_feas_slide11_q_h_allon_C1000_exploratory` | tm_language_model | feas_slide11_q_h_allon_C1000_exploratory | 2026-10-01 | finished | 0 | 1.2 min | 34.4 kB | biased exploratory |
| `20261001T073143Z_feas_slide11_q_h_bin_C1000_A4K4_exploratory` | tm_language_model | feas_slide11_q_h_bin_C1000_A4K4_exploratory | 2026-10-01 | finished | 0 | 1.1 min | 34.4 kB | biased exploratory |
| `20261001T073249Z_feas_slide11_q_h_bin_F3072_C1000_exploratory` | tm_language_model | feas_slide11_q_h_bin_F3072_C1000_exploratory | 2026-10-01 | finished | 0 | 10.9 min | 34.2 kB | biased exploratory |
| `20261001T074342Z_feas_slide11_q_f_C4000_s100_exploratory` | tm_language_model | feas_slide11_q_f_C4000_s100_exploratory | 2026-10-01 | finished | 0 | 10.2 min | 30.1 kB | biased exploratory |
| `20261001T075353Z_feas_slide11_q_h_slide11_multiclass_exploratory` | tm_language_model | feas_slide11_q_h_slide11_multiclass_exploratory | 2026-10-01 | finished | 0 | 2.1 min | 28.1 kB | biased exploratory |
| `20261001T075600Z_feas_slide11_q_inter_hier_C200_exploratory` | tm_language_model | feas_slide11_q_inter_hier_C200_exploratory | 2026-10-01 | finished | 0 | 38 s | 31.2 kB | biased exploratory |
| `20261001T075639Z_feas_slide11_q_inter_flat_C200_exploratory` | tm_language_model | feas_slide11_q_inter_flat_C200_exploratory | 2026-10-01 | finished | 0 | 32 s | 30.0 kB | biased exploratory |
| `20261001T075712Z_feas_slide11_q_h_off_C2000_long_exploratory` | tm_language_model | feas_slide11_q_h_off_C2000_long_exploratory | 2026-10-01 | finished | 0 | 4.0 min | 37.0 kB | biased exploratory |
| `20261001T080322Z_ssl_r4c_probe_randbank_seed2` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 4.76 h | 29.4 kB |  |
| `20261001T080935Z_feas_slide11_posthoc_exploratory` | tm_language_model | feas_slide11_posthoc_exploratory | 2026-10-01 | finished | 0 | 25 s | 21.4 kB | biased exploratory |
| `20261001T091717Z_demo_wrapper_smoke_exploratory` | hierarchical_tm | demo_wrapper_smoke_exploratory | 2026-10-01 | finished | 0 | 13 s | 31.3 kB | smoke |
| `20261001T091807Z_e1_demo_seed1` | hierarchical_tm | e1_demo_seed1 | 2026-10-01 | finished | 0 | 4.7 min | 437.4 kB |  |
| `20261001T092112Z_data_build_exploratory` | hierarchical_tm | data_build_exploratory | 2026-10-01 | finished | 0 | 52 s | 29.2 kB |  |
| `20261001T092250Z_e1_demo_seed2` | hierarchical_tm | e1_demo_seed2 | 2026-10-01 | finished | 0 | 5.0 min | 446.6 kB |  |
| `20261001T092321Z_data_checks_exploratory` | hierarchical_tm | data_checks_exploratory | 2026-10-01 | finished | 0 | 33 s | 39.1 kB |  |
| `20261001T092344Z_driver_smoke_exploratory` | hierarchical_tm | driver_smoke_exploratory | 2026-10-01 | finished | 0 | 4.3 min | 49.4 kB | smoke |
| `20261001T092751Z_e1_demo_seed3` | hierarchical_tm | e1_demo_seed3 | 2026-10-01 | finished | 0 | 4.8 min | 457.3 kB |  |
| `20261001T092825Z_e1_p1word_hier` | hierarchical_tm | e1_p1word_hier | 2026-10-01 | finished | 0 | 5.3 min | 105.7 kB |  |
| `20261001T092928Z_data_build_v2_exploratory` | hierarchical_tm | data_build_v2_exploratory | 2026-10-01 | finished | 0 | 1.5 min | 31.7 kB |  |
| `20261001T093113Z_data_checks_v2_exploratory` | hierarchical_tm | data_checks_v2_exploratory | 2026-10-01 | finished | 0 | 39 s | 42.2 kB |  |
| `20261001T093345Z_e1_verify_binary` | hierarchical_tm | e1_verify_binary | 2026-10-01 | finished | 0 | 21 s | 107.7 kB |  |
| `20261001T093407Z_e1_verify_default` | hierarchical_tm | e1_verify_default | 2026-10-01 | finished | 0 | 19 s | 107.1 kB |  |
| `20261001T093426Z_timing_C2000_A8K2_exploratory` | hierarchical_tm | timing_C2000_A8K2_exploratory | 2026-10-01 | finished | 0 | 1.8 min | 432.7 kB |  |
| `20261001T093613Z_timing_C50_A64K2_exploratory` | hierarchical_tm | timing_C50_A64K2_exploratory | 2026-10-01 | finished | 0 | 55 s | 266.6 kB |  |
| `20261001T093726Z_data_build_v3_exploratory` | hierarchical_tm | data_build_v3_exploratory | 2026-10-01 | finished | 0 | 58 s | 31.2 kB |  |
| `20261001T093814Z_driver_flat_shuffle_exploratory` | hierarchical_tm | driver_flat_shuffle_exploratory | 2026-10-01 | finished | 0 | 8 s | 40.9 kB |  |
| `20261001T093822Z_driver_switches_exploratory` | hierarchical_tm | driver_switches_exploratory | 2026-10-01 | finished | 0 | 8 s | 51.3 kB |  |
| `20261001T093837Z_data_checks_v3_exploratory` | hierarchical_tm | data_checks_v3_exploratory | 2026-10-01 | finished | 0 | 41 s | 42.1 kB |  |
| `20261001T094049Z_data_build_ns_all_exploratory` | hierarchical_tm | data_build_ns_all_exploratory | 2026-10-01 | finished | 0 | 3 s | 13.1 kB |  |
| `20261001T094115Z_data_build_v3_repro_exploratory` | hierarchical_tm | data_build_v3_repro_exploratory | 2026-10-01 | finished | 0 | 1.0 min | 29.0 kB |  |
| `20261001T094221Z_data_checks_ns_all_exploratory` | hierarchical_tm | data_checks_ns_all_exploratory | 2026-10-01 | finished | 0 | 13 s | 31.7 kB |  |
| `20261001T094249Z_data_checks_final_exploratory` | hierarchical_tm | data_checks_final_exploratory | 2026-10-01 | finished | 0 | 40 s | 42.5 kB |  |
| `20261001T094344Z_data_build_ns_all_v2_exploratory` | hierarchical_tm | data_build_ns_all_v2_exploratory | 2026-10-01 | finished | 0 | 3 s | 15.0 kB |  |
| `20261001T094347Z_data_checks_ns_all_v2_exploratory` | hierarchical_tm | data_checks_ns_all_v2_exploratory | 2026-10-01 | finished | 0 | 12 s | 31.7 kB |  |
| `20261001T100004Z_pcp_world_checks` | hierarchical_tm | pcp_world_checks | 2026-10-01 | finished | 0 | 29 s | 13.5 kB |  |
| `20261001T140704Z_ssl_r4c_probe_randbank_seed3` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 6.29 h | 29.4 kB |  |
| `20261001T175601Z_pcp_smoke_exploratory` | hierarchical_tm | pcp_smoke_exploratory | 2026-10-01 | finished | 0 | 43 s | 62.6 kB | smoke |
| `20261001T175704Z_e4_sa_best_C2000_s40_p50` | hierarchical_tm | e4_sa_best_C2000_s40_p50 | 2026-10-01 | finished | 0 | 31.6 min | 448.8 kB |  |
| `20261001T182857Z_e2_L1_hier_s1` | hierarchical_tm | e2_L1_hier_s1 | 2026-10-01 | finished | 0 | 5.8 min | 125.8 kB |  |
| `20261001T183447Z_e2_L1_flat_s1` | hierarchical_tm | e2_L1_flat_s1 | 2026-10-01 | finished | 0 | 3.0 min | 110.3 kB |  |
| `20261001T183749Z_e2_L1_hier_s2` | hierarchical_tm | e2_L1_hier_s2 | 2026-10-01 | finished | 0 | 3.7 min | 125.9 kB |  |
| `20261001T184133Z_e2_L1_flat_s2` | hierarchical_tm | e2_L1_flat_s2 | 2026-10-01 | finished | 0 | 2.9 min | 108.9 kB |  |
| `20261001T184428Z_e2_L1_hier_s3` | hierarchical_tm | e2_L1_hier_s3 | 2026-10-01 | finished | 0 | 3.8 min | 125.8 kB |  |
| `20261001T184814Z_e2_L1_flat_s3` | hierarchical_tm | e2_L1_flat_s3 | 2026-10-01 | finished | 0 | 3.8 min | 108.9 kB |  |
| `20261001T185201Z_e2_L2_hier_s1` | hierarchical_tm | e2_L2_hier_s1 | 2026-10-01 | finished | 0 | 3.9 min | 134.4 kB |  |
| `20261001T185557Z_e2_L2_flat_s1` | hierarchical_tm | e2_L2_flat_s1 | 2026-10-01 | finished | 0 | 3.2 min | 110.4 kB |  |
| `20261001T185907Z_e2_L2_hier_s2` | hierarchical_tm | e2_L2_hier_s2 | 2026-10-01 | finished | 0 | 3.9 min | 132.9 kB |  |
| `20261001T190302Z_e2_L2_flat_s2` | hierarchical_tm | e2_L2_flat_s2 | 2026-10-01 | finished | 0 | 4.0 min | 109.2 kB |  |
| `20261001T190659Z_e2_L2_hier_s3` | hierarchical_tm | e2_L2_hier_s3 | 2026-10-01 | finished | 0 | 3.9 min | 133.2 kB |  |
| `20261001T191056Z_e2_L2_flat_s3` | hierarchical_tm | e2_L2_flat_s3 | 2026-10-01 | finished | 0 | 3.9 min | 109.3 kB |  |
| `20261001T191451Z_e2_L3_hier_s1` | hierarchical_tm | e2_L3_hier_s1 | 2026-10-01 | finished | 0 | 8.7 min | 213.6 kB |  |
| `20261001T192336Z_e2_L3_flat_s1` | hierarchical_tm | e2_L3_flat_s1 | 2026-10-01 | finished | 0 | 7.5 min | 182.1 kB |  |
| `20261001T193109Z_e2_L3_hier_unseen_s1` | hierarchical_tm | e2_L3_hier_unseen_s1 | 2026-10-01 | finished | 0 | 8.6 min | 212.4 kB |  |
| `20261001T193947Z_e2_L3_hier_s2` | hierarchical_tm | e2_L3_hier_s2 | 2026-10-01 | finished | 0 | 8.9 min | 212.2 kB |  |
| `20261001T194841Z_e2_L3_flat_s2` | hierarchical_tm | e2_L3_flat_s2 | 2026-10-01 | finished | 0 | 6.9 min | 180.7 kB |  |
| `20261001T195536Z_e2_L3_hier_unseen_s2` | hierarchical_tm | e2_L3_hier_unseen_s2 | 2026-10-01 | finished | 0 | 8.8 min | 212.6 kB |  |
| `20261001T200422Z_e2_L3_hier_s3` | hierarchical_tm | e2_L3_hier_s3 | 2026-10-01 | finished | 0 | 8.7 min | 213.0 kB |  |
| `20261001T201307Z_e2_L3_flat_s3` | hierarchical_tm | e2_L3_flat_s3 | 2026-10-01 | finished | 0 | 7.3 min | 180.8 kB |  |
| `20261001T202026Z_e2_L3_hier_unseen_s3` | hierarchical_tm | e2_L3_hier_unseen_s3 | 2026-10-01 | finished | 0 | 9.5 min | 213.1 kB |  |
| `20261001T202442Z_ssl_r4c_report` | tm_language_model | ssl_r4_controls | 2026-10-01 | finished | 0 | 1 s | 248.1 kB |  |
| `20261001T202954Z_e4_sa_granmo_C50_A64_s21` | hierarchical_tm | e4_sa_granmo_C50_A64_s21 | 2026-10-01 | finished | 0 | 29.0 min | 337.6 kB |  |
| `20261001T205856Z_e4_sa_granmo_C50_A64_s40` | hierarchical_tm | e4_sa_granmo_C50_A64_s40 | 2026-10-01 | finished | 0 | 31.0 min | 328.5 kB |  |
| `20261001T212959Z_e4_sa_granmo_C50_A64_s100` | hierarchical_tm | e4_sa_granmo_C50_A64_s100 | 2026-10-01 | finished | 0 | 29.6 min | 345.5 kB |  |
| `20261001T215937Z_e2_L3_hier_s1_p1000` | hierarchical_tm | e2_L3_hier_s1_p1000 | 2026-10-01 | finished | 0 | 43.9 min | 786.1 kB |  |
| `20261001T224330Z_e2_L3_hier_s2_p1000` | hierarchical_tm | e2_L3_hier_s2_p1000 | 2026-10-01 | finished | 0 | 43.4 min | 786.2 kB |  |
| `20261001T232651Z_e2_L3_hier_s3_p1000` | hierarchical_tm | e2_L3_hier_s3_p1000 | 2026-10-01 | finished | 0 | 43.6 min | 786.1 kB |  |
| `20261001T234330Z_e2_recovery_posthoc_exploratory` | hierarchical_tm | e2_recovery_posthoc_exploratory | 2026-10-01 | finished | 0 | 0 s | 16.9 kB |  |
| `20261002T001026Z_e2_L3_hier_s1_s100_p300` | hierarchical_tm | e2_L3_hier_s1_s100_p300 | 2026-10-02 | finished | 0 | 14.0 min | 287.8 kB |  |
| `20261002T002428Z_e2_L3_hier_s1_s200_p300` | hierarchical_tm | e2_L3_hier_s1_s200_p300 | 2026-10-02 | finished | 0 | 13.9 min | 286.6 kB |  |
| `20261002T003819Z_e2_L3_hier_s1_s400_p300` | hierarchical_tm | e2_L3_hier_s1_s400_p300 | 2026-10-02 | finished | 0 | 13.7 min | 286.8 kB |  |
| `20261002T003832Z_e2_recovery_L3_stest_exploratory` | hierarchical_tm | e2_recovery_L3_stest_exploratory | 2026-10-02 | finished | 0 | 0 s | 6.7 kB |  |
| `20261002T005202Z_e2_L2_hier_s1_s100` | hierarchical_tm | e2_L2_hier_s1_s100 | 2026-10-02 | finished | 0 | 4.3 min | 135.2 kB |  |
| `20261002T005432Z_e2_purity_L3_equal_footing_exploratory` | hierarchical_tm | e2_purity_L3_equal_footing_exploratory | 2026-10-02 | finished | 0 | 0 s | 8.6 kB |  |
| `20261002T005621Z_e2_L2_hier_s2_s100` | hierarchical_tm | e2_L2_hier_s2_s100 | 2026-10-02 | finished | 0 | 3.9 min | 134.1 kB |  |
| `20261002T010014Z_e2_L2_hier_s3_s100` | hierarchical_tm | e2_L2_hier_s3_s100 | 2026-10-02 | finished | 0 | 5.3 min | 134.2 kB |  |
| `20261002T010534Z_e2_L3_hier_s1_s800_p300` | hierarchical_tm | e2_L3_hier_s1_s800_p300 | 2026-10-02 | finished | 0 | 13.7 min | 283.5 kB |  |
| `20261002T010551Z_e2_L2_s_compare_posthoc_exploratory` | hierarchical_tm | e2_L2_s_compare_posthoc_exploratory | 2026-10-02 | finished | 126 | 0 s | 9.8 kB |  |
| `20261002T010559Z_e2_L2_s_compare_posthoc_exploratory` | hierarchical_tm | e2_L2_s_compare_posthoc_exploratory | 2026-10-02 | finished | 0 | 0 s | 14.4 kB |  |
| `20261002T011938Z_e2_L3_hier_s2_s800_p300` | hierarchical_tm | e2_L3_hier_s2_s800_p300 | 2026-10-02 | finished | 0 | 13.3 min | 283.1 kB |  |
| `20261002T012000Z_e2_purity_L3_s800_exploratory` | hierarchical_tm | e2_purity_L3_s800_exploratory | 2026-10-02 | finished | 0 | 0 s | 8.6 kB |  |
| `20261002T013256Z_e2_L3_hier_s3_s800_p300` | hierarchical_tm | e2_L3_hier_s3_s800_p300 | 2026-10-02 | finished | 0 | 13.3 min | 282.5 kB |  |
| `20261002T014618Z_e2_L3_hier_unseen_s1_s800_p300` | hierarchical_tm | e2_L3_hier_unseen_s1_s800_p300 | 2026-10-02 | finished | 0 | 13.6 min | 282.7 kB |  |
| `20261002T015954Z_e2_L3_hier_unseen_s2_s800_p300` | hierarchical_tm | e2_L3_hier_unseen_s2_s800_p300 | 2026-10-02 | finished | 0 | 13.6 min | 283.6 kB |  |
| `20261002T021328Z_e2_L3_hier_unseen_s3_s800_p300` | hierarchical_tm | e2_L3_hier_unseen_s3_s800_p300 | 2026-10-02 | finished | 0 | 16.7 min | 282.9 kB |  |
| `20261002T023032Z_share_best_n1_s1_exploratory` | hierarchical_tm | share_best_n1_s1_exploratory | 2026-10-02 | finished | 0 | 2.8 min | 428.4 kB |  |
| `20261002T023042Z_e2_L3_s800_posthoc_exploratory` | hierarchical_tm | e2_L3_s800_posthoc_exploratory | 2026-10-02 | finished | 0 | 0 s | 12.9 kB |  |
| `20261002T023323Z_share_best_n1b_s1_exploratory` | hierarchical_tm | share_best_n1b_s1_exploratory | 2026-10-02 | finished | 0 | 2.7 min | 428.4 kB |  |
| `20261002T023603Z_share_best_n2_s1_exploratory` | hierarchical_tm | share_best_n2_s1_exploratory | 2026-10-02 | finished | 0 | 7.0 min | 428.4 kB |  |
| `20261002T023603Z_share_best_n2_s2_exploratory` | hierarchical_tm | share_best_n2_s2_exploratory | 2026-10-02 | finished | 0 | 7.0 min | 430.5 kB |  |
| `20261002T024304Z_share_best_n3_s1_exploratory` | hierarchical_tm | share_best_n3_s1_exploratory | 2026-10-02 | finished | 0 | 10.5 min | 428.4 kB |  |
| `20261002T024304Z_share_best_n3_s2_exploratory` | hierarchical_tm | share_best_n3_s2_exploratory | 2026-10-02 | finished | 0 | 10.4 min | 430.5 kB |  |
| `20261002T024304Z_share_best_n3_s3_exploratory` | hierarchical_tm | share_best_n3_s3_exploratory | 2026-10-02 | finished | 0 | 10.5 min | 427.9 kB |  |
| `20261002T025336Z_share_granmo_n1_s1_exploratory` | hierarchical_tm | share_granmo_n1_s1_exploratory | 2026-10-02 | finished | 0 | 1.3 min | 281.4 kB |  |
| `20261002T025453Z_share_granmo_n1b_s1_exploratory` | hierarchical_tm | share_granmo_n1b_s1_exploratory | 2026-10-02 | finished | 0 | 1.3 min | 281.4 kB |  |
| `20261002T025611Z_share_granmo_n2_s1_exploratory` | hierarchical_tm | share_granmo_n2_s1_exploratory | 2026-10-02 | finished | 0 | 4.5 min | 281.4 kB |  |
| `20261002T025611Z_share_granmo_n2_s2_exploratory` | hierarchical_tm | share_granmo_n2_s2_exploratory | 2026-10-02 | finished | 0 | 4.5 min | 282.2 kB |  |
| `20261002T030044Z_share_granmo_n3_s1_exploratory` | hierarchical_tm | share_granmo_n3_s1_exploratory | 2026-10-02 | finished | 0 | 6.5 min | 281.4 kB |  |
| `20261002T030044Z_share_granmo_n3_s2_exploratory` | hierarchical_tm | share_granmo_n3_s2_exploratory | 2026-10-02 | finished | 0 | 6.7 min | 282.2 kB |  |
| `20261002T030044Z_share_granmo_n3_s3_exploratory` | hierarchical_tm | share_granmo_n3_s3_exploratory | 2026-10-02 | finished | 0 | 6.7 min | 282.6 kB |  |
| `20261002T030724Z_evalsched_best_every1_s1_exploratory` | hierarchical_tm | evalsched_best_every1_s1_exploratory | 2026-10-02 | finished | 0 | 6.5 min | 131.5 MB |  |
| `20261002T031354Z_evalsched_best_every5_s1_exploratory` | hierarchical_tm | evalsched_best_every5_s1_exploratory | 2026-10-02 | finished | 0 | 6.0 min | 422.0 kB |  |
| `20261002T031952Z_saveload_check_exploratory` | hierarchical_tm | saveload_check_exploratory | 2026-10-02 | finished | 65 | 16 s | 385.0 kB |  |
| `20261002T032009Z_share_best_mps2_s1_exploratory` | hierarchical_tm | share_best_mps2_s1_exploratory | 2026-10-02 | finished | 0 | 3.8 min | 428.5 kB |  |
| `20261002T032009Z_share_best_mps2_s2_exploratory` | hierarchical_tm | share_best_mps2_s2_exploratory | 2026-10-02 | finished | 0 | 3.8 min | 430.6 kB |  |
| `20261002T032359Z_share_granmo_mps2_s1_exploratory` | hierarchical_tm | share_granmo_mps2_s1_exploratory | 2026-10-02 | finished | 0 | 1.4 min | 281.5 kB |  |
| `20261002T032359Z_share_granmo_mps2_s2_exploratory` | hierarchical_tm | share_granmo_mps2_s2_exploratory | 2026-10-02 | finished | 0 | 1.4 min | 282.2 kB |  |
| `20261002T032557Z_saveload_check_v2_exploratory` | hierarchical_tm | saveload_check_v2_exploratory | 2026-10-02 | finished | 0 | 16 s | 386.6 kB |  |
| `20261002T032820Z_e4_screen_sa_granmo_s200` | hierarchical_tm | e4_screen_sa_granmo_s200 | 2026-10-02 | finished | 0 | 26.3 min | 26.5 MB |  |
| `20261002T032840Z_e4_screen_sa_granmo_s400` | hierarchical_tm | e4_screen_sa_granmo_s400 | 2026-10-02 | finished | 0 | 25.1 min | 26.6 MB |  |
| `20261002T034314Z_e5a_data_build` | hierarchical_tm | e5a_data_build | 2026-10-02 | finished | 0 | 9 s | 13.8 kB |  |
| `20261002T035105Z_e5a_data_build_v2` | hierarchical_tm | e5a_data_build_v2 | 2026-10-02 | finished | 0 | 9 s | 16.1 kB |  |
| `20261002T035135Z_e5a_cpu_tests_exploratory` | hierarchical_tm | e5a_cpu_tests_exploratory | 2026-10-02 | finished | 0 | 3 s | 6.7 kB |  |
| `20261002T035139Z_e5a_lr_baseline` | hierarchical_tm | e5a_lr_baseline | 2026-10-02 | finished | 0 | 5 s | 9.9 kB |  |
| `20261002T035442Z_e4_headline_s_rule` | hierarchical_tm | e4_headline_s_rule | 2026-10-02 | finished | 0 | 0 s | 8.7 kB |  |
| `20261002T035511Z_e4_curve_sa_granmo_s100_p200` | hierarchical_tm | e4_curve_sa_granmo_s100_p200 | 2026-10-02 | finished | 0 | 1.11 h | 105.3 MB |  |
| `20261002T035531Z_check_rescore_screen_s200_exploratory` | hierarchical_tm | check_rescore_screen_s200_exploratory | 2026-10-02 | finished | 0 | 27 s | 360.4 kB |  |
| `20261002T035606Z_e2_L2_hier_s1_p1000` | hierarchical_tm | e2_L2_hier_s1_p1000 | 2026-10-02 | finished | 0 | 34.2 min | 598.3 kB |  |
| `20261002T040149Z_e5a_cpu_tests_v2_exploratory` | hierarchical_tm | e5a_cpu_tests_v2_exploratory | 2026-10-02 | finished | 0 | 22 s | 9.9 kB |  |
| `20261002T040150Z_slide10_fixture_build_exploratory` | hierarchical_tm | slide10_fixture_build_exploratory | 2026-10-02 | finished | 0 | 3 s | 8.2 kB |  |
| `20261002T040212Z_slide10_tests_exploratory` | hierarchical_tm | slide10_tests_exploratory | 2026-10-02 | finished | 0 | 4 s | 8.3 kB |  |
| `20261002T040216Z_slide10_lr_null_exploratory` | hierarchical_tm | slide10_lr_null_exploratory | 2026-10-02 | finished | 0 | 1.6 min | 7.7 kB |  |
| `20261002T040551Z_slide10_fixture_build_v2_exploratory` | hierarchical_tm | slide10_fixture_build_v2_exploratory | 2026-10-02 | finished | 0 | 3 s | 8.2 kB |  |
| `20261002T040722Z_slide10_tests_v2_exploratory` | hierarchical_tm | slide10_tests_v2_exploratory | 2026-10-02 | finished | 0 | 4 s | 8.4 kB |  |
| `20261002T043029Z_e5a_smoke_hier_exploratory` | hierarchical_tm | e5a_smoke_hier_exploratory | 2026-10-02 | finished | 0 | 19 s | 26.8 MB | smoke |
| `20261002T043039Z_e2_L2_p1000_recovery_posthoc_exploratory` | hierarchical_tm | e2_L2_p1000_recovery_posthoc_exploratory | 2026-10-02 | finished | 0 | 0 s | 6.0 kB |  |
| `20261002T043040Z_e4_curve_repro_check_posthoc` | hierarchical_tm | e4_curve_repro_check_posthoc | 2026-10-02 | finished | 0 | 0 s | 5.7 kB |  |
| `20261002T043105Z_e5a_smoke_flat_exploratory` | hierarchical_tm | e5a_smoke_flat_exploratory | 2026-10-02 | finished | 0 | 23 s | 28.7 MB | smoke |
| `20261002T043139Z_e2_L2_hier_s2_p1000` | hierarchical_tm | e2_L2_hier_s2_p1000 | 2026-10-02 | finished | 0 | 30.7 min | 598.2 kB |  |
| `20261002T043139Z_e2_L2_p1000_purity_posthoc_exploratory` | hierarchical_tm | e2_L2_p1000_purity_posthoc_exploratory | 2026-10-02 | finished | 0 | 0 s | 5.7 kB |  |
| `20261002T043427Z_check4_scorer_tests_exploratory` | hierarchical_tm | check4_scorer_tests_exploratory | 2026-10-02 | finished | 0 | 5.6 min | 22.4 kB |  |
| `20261002T044010Z_check4_scorer_tests_v2_exploratory` | hierarchical_tm | check4_scorer_tests_v2_exploratory | 2026-10-02 | finished | 0 | 5.9 min | 24.1 kB |  |
| `20261002T044613Z_check4_scorer_tests_v3_exploratory` | hierarchical_tm | check4_scorer_tests_v3_exploratory | 2026-10-02 | finished | 0 | 7.3 min | 26.0 kB |  |
| `20261002T045620Z_check4_scorer_tests_v4_exploratory` | hierarchical_tm | check4_scorer_tests_v4_exploratory | 2026-10-02 | finished | 0 | 7.4 min | 24.5 kB |  |
| `20261002T050200Z_e2_L2_hier_s3_p1000` | hierarchical_tm | e2_L2_hier_s3_p1000 | 2026-10-02 | finished | 0 | 30.0 min | 598.4 kB |  |
| `20261002T050204Z_e4_pass_count_rule` | hierarchical_tm | e4_pass_count_rule | 2026-10-02 | finished | 0 | 0 s | 7.4 kB |  |
| `20261002T050235Z_e4_head_sa_s1_scoreP100` | hierarchical_tm | e4_head_sa_s1_scoreP100 | 2026-10-02 | finished | 0 | 25 s | 389.5 kB |  |
| `20261002T050238Z_e2_L2_p1000_s2_recovery_purity_posthoc_exploratory` | hierarchical_tm | e2_L2_p1000_s2_recovery_purity_posthoc_exploratory | 2026-10-02 | finished | 0 | 0 s | 7.0 kB |  |
| `20261002T050310Z_e4_head_sa_s2` | hierarchical_tm | e4_head_sa_s2 | 2026-10-02 | finished | 0 | 26.1 min | 26.5 MB |  |
| `20261002T052930Z_e4_head_sa_s2_score` | hierarchical_tm | e4_head_sa_s2_score | 2026-10-02 | finished | 0 | 24 s | 356.1 kB |  |
| `20261002T053006Z_e5a_screen_hier_s21` | hierarchical_tm | e5a_screen_hier_s21 | 2026-10-02 | finished | 0 | 18.8 min | 26.7 MB |  |
| `20261002T053211Z_e4_head_sa_s3` | hierarchical_tm | e4_head_sa_s3 | 2026-10-02 | finished | 0 | 26.6 min | 26.5 MB |  |
| `20261002T053216Z_e2_L2_p1000_s3_recovery_purity_posthoc_exploratory` | hierarchical_tm | e2_L2_p1000_s3_recovery_purity_posthoc_exploratory | 2026-10-02 | finished | 0 | 0 s | 7.1 kB |  |
| `20261002T053510Z_e2_crossfit_acc_posthoc` | hierarchical_tm | e2_crossfit_acc_posthoc | 2026-10-02 | finished | 0 | 4 s | 21.9 kB |  |
| `20261002T053555Z_e2_crossfit_split_robustness_posthoc_exploratory` | hierarchical_tm | e2_crossfit_split_robustness_posthoc_exploratory | 2026-10-02 | finished | 0 | 3 s | 9.0 kB |  |
| `20261002T053708Z_e2_crossfit_split_avg_all_posthoc_exploratory` | hierarchical_tm | e2_crossfit_split_avg_all_posthoc_exploratory | 2026-10-02 | finished | 0 | 7 s | 14.9 kB |  |
| `20261002T053822Z_e4_rules_recheck_after_float_tolerance` | hierarchical_tm | e4_rules_recheck_after_float_tolerance | 2026-10-02 | finished | 0 | 0 s | 9.4 kB |  |
| `20261002T054901Z_e5a_screen_flat_s21` | hierarchical_tm | e5a_screen_flat_s21 | 2026-10-02 | finished | 0 | 18.3 min | 28.7 MB |  |
| `20261002T055851Z_e4_head_np_s1` | hierarchical_tm | e4_head_np_s1 | 2026-10-02 | finished | 0 | 26.5 min | 26.5 MB |  |
| `20261002T060727Z_e4_head_sa_s3_score` | hierarchical_tm | e4_head_sa_s3_score | 2026-10-02 | finished | 0 | 26 s | 356.5 kB |  |
| `20261002T060802Z_e5a_screen_hier_s60` | hierarchical_tm | e5a_screen_hier_s60 | 2026-10-02 | finished | 0 | 18.0 min | 26.8 MB |  |
| `20261002T062537Z_e4_head_np_s2` | hierarchical_tm | e4_head_np_s2 | 2026-10-02 | finished | 0 | 26.6 min | 26.5 MB |  |
| `20261002T062612Z_e4_head_np_s1_score` | hierarchical_tm | e4_head_np_s1_score | 2026-10-02 | finished | 0 | 25 s | 355.8 kB |  |
| `20261002T062647Z_e5a_screen_flat_s60` | hierarchical_tm | e5a_screen_flat_s60 | 2026-10-02 | finished | 0 | 16.9 min | 28.8 MB |  |
| `20261002T064353Z_e4_head_np_s3` | hierarchical_tm | e4_head_np_s3 | 2026-10-02 | finished | 0 | 26.4 min | 26.5 MB |  |
| `20261002T065213Z_e5a_screen_hier_s230` | hierarchical_tm | e5a_screen_hier_s230 | 2026-10-02 | finished | 0 | 16.2 min | 27.0 MB |  |
| `20261002T070833Z_e4_head_np_s2_score` | hierarchical_tm | e4_head_np_s2_score | 2026-10-02 | finished | 0 | 24 s | 354.3 kB |  |
| `20261002T070908Z_e5a_screen_flat_s230` | hierarchical_tm | e5a_screen_flat_s230 | 2026-10-02 | finished | 0 | 13.7 min | 29.0 MB |  |
| `20261002T071028Z_e4_head_np_s3_score` | hierarchical_tm | e4_head_np_s3_score | 2026-10-02 | finished | 0 | 26 s | 356.9 kB |  |
| `20261002T071104Z_e5a_hier_s60_seed2` | hierarchical_tm | e5a_hier_s60_seed2 | 2026-10-02 | finished | 0 | 17.8 min | 26.8 MB |  |
| `20261002T072254Z_e5a_hier_s60_seed3` | hierarchical_tm | e5a_hier_s60_seed3 | 2026-10-02 | finished | 0 | 17.5 min | 26.8 MB |  |
| `20261002T072322Z_e5a_s_choice_rule` | hierarchical_tm | e5a_s_choice_rule | 2026-10-02 | finished | 0 | 0 s | 6.8 kB |  |
| `20261002T072859Z_e5a_flat_s60_seed2` | hierarchical_tm | e5a_flat_s60_seed2 | 2026-10-02 | finished | 0 | 16.5 min | 28.8 MB |  |
| `20261002T074034Z_e5a_flat_s60_seed3` | hierarchical_tm | e5a_flat_s60_seed3 | 2026-10-02 | finished | 0 | 16.4 min | 28.8 MB |  |
| `20261002T074043Z_e5a_npmi_check2c` | hierarchical_tm | e5a_npmi_check2c | 2026-10-02 | finished | 0 | 28 s | 12.7 kB |  |
| `20261002T074540Z_e4s_nsall_s100_s1` | hierarchical_tm | e4s_nsall_s100_s1 | 2026-10-02 | finished | 0 | 27.6 min | 26.5 MB |  |
| `20261002T075700Z_e4s_nsall_s100_shuf_s1` | hierarchical_tm | e4s_nsall_s100_shuf_s1 | 2026-10-02 | finished | 0 | 27.8 min | 26.5 MB |  |
| `20261002T075715Z_e5a_check2c_accuracy` | hierarchical_tm | e5a_check2c_accuracy | 2026-10-02 | finished | 0 | 0 s | 9.2 kB |  |
| `20261002T081320Z_e4s_nsall_s100_s2` | hierarchical_tm | e4s_nsall_s100_s2 | 2026-10-02 | finished | 0 | 27.6 min | 26.5 MB |  |
| `20261002T082456Z_e4s_nsall_s100_s1_score` | hierarchical_tm | e4s_nsall_s100_s1_score | 2026-10-02 | finished | 0 | 23 s | 343.1 kB |  |
| `20261002T082531Z_e4s_nsall_s100_shuf_s2` | hierarchical_tm | e4s_nsall_s100_shuf_s2 | 2026-10-02 | finished | 0 | 29.5 min | 26.5 MB |  |
| `20261002T084106Z_e4s_nsall_s100_s2_score` | hierarchical_tm | e4s_nsall_s100_s2_score | 2026-10-02 | finished | 0 | 24 s | 342.5 kB |  |
| `20261002T084141Z_e4s_nsall_s100_s3` | hierarchical_tm | e4s_nsall_s100_s3 | 2026-10-02 | finished | 0 | 27.9 min | 26.5 MB |  |
| `20261002T085502Z_e4s_nsall_s100_shuf_s3` | hierarchical_tm | e4s_nsall_s100_shuf_s3 | 2026-10-02 | finished | 0 | 27.5 min | 26.5 MB |  |
| `20261002T090952Z_e4s_nsall_s100_s3_score` | hierarchical_tm | e4s_nsall_s100_s3_score | 2026-10-02 | finished | 0 | 24 s | 344.8 kB |  |
| `20261002T091027Z_e2_L3_flat_unseen_s1_s800_p300_c1280` | hierarchical_tm | e2_L3_flat_unseen_s1_s800_p300_c1280 | 2026-10-02 | finished | 0 | 8.6 min | 1.4 MB |  |
| `20261002T091917Z_e2_L3_flat_unseen_s2_s800_p300_c1280` | hierarchical_tm | e2_L3_flat_unseen_s2_s800_p300_c1280 | 2026-10-02 | finished | 0 | 7.7 min | 1.4 MB |  |
| `20261002T092238Z_e2_L3_flat_unseen_s3_s800_p300_c1280` | hierarchical_tm | e2_L3_flat_unseen_s3_s800_p300_c1280 | 2026-10-02 | finished | 0 | 8.1 min | 1.4 MB |  |
| `20261002T092713Z_e4_head_sa_shuf_s1` | hierarchical_tm | e4_head_sa_shuf_s1 | 2026-10-02 | finished | 0 | 29.2 min | 26.5 MB |  |
| `20261002T093048Z_e4_head_sa_shuf_s2` | hierarchical_tm | e4_head_sa_shuf_s2 | 2026-10-02 | finished | 0 | 26.8 min | 26.5 MB |  |
| `20261002T093158Z_check4b_synonym_groups_s100` | hierarchical_tm | check4b_synonym_groups_s100 | 2026-10-02 | finished | 0 | 1.5 min | 537.5 kB |  |
| `20261002T095639Z_e4_head_sa_shuf_s3` | hierarchical_tm | e4_head_sa_shuf_s3 | 2026-10-02 | finished | 0 | 27.4 min | 26.5 MB |  |
| `20261002T095744Z_e4_flat_sa_s1` | hierarchical_tm | e4_flat_sa_s1 | 2026-10-02 | finished | 0 | 24.9 min | 26.4 MB |  |
| `20261002T102249Z_e4_flat_sa_s2` | hierarchical_tm | e4_flat_sa_s2 | 2026-10-02 | finished | 0 | 27.0 min | 26.4 MB |  |
| `20261002T102409Z_e4_flat_sa_s3` | hierarchical_tm | e4_flat_sa_s3 | 2026-10-02 | finished | 0 | 26.9 min | 26.4 MB |  |
| `20261002T105000Z_e4_flat_np_s1` | hierarchical_tm | e4_flat_np_s1 | 2026-10-02 | finished | 0 | 27.0 min | 26.4 MB |  |
| `20261002T105120Z_e4_flat_np_s2` | hierarchical_tm | e4_flat_np_s2 | 2026-10-02 | finished | 0 | 26.9 min | 26.4 MB |  |
| `20261002T111711Z_e4_flat_np_s3` | hierarchical_tm | e4_flat_np_s3 | 2026-10-02 | finished | 0 | 27.2 min | 26.4 MB |  |
| `20261002T111816Z_e4s_nsall_s400_s1` | hierarchical_tm | e4s_nsall_s400_s1 | 2026-10-02 | finished | 0 | 26.5 min | 26.5 MB |  |
| `20261002T114436Z_e4_flat_sa_s1_score` | hierarchical_tm | e4_flat_sa_s1_score | 2026-10-02 | finished | 0 | 23 s | 233.4 kB |  |
| `20261002T114457Z_e4_flat_sa_s2_score` | hierarchical_tm | e4_flat_sa_s2_score | 2026-10-02 | finished | 0 | 20 s | 233.0 kB |  |
| `20261002T114517Z_e4_flat_sa_s3_score` | hierarchical_tm | e4_flat_sa_s3_score | 2026-10-02 | finished | 0 | 20 s | 233.6 kB |  |
| `20261002T114537Z_e4_flat_np_s1_score` | hierarchical_tm | e4_flat_np_s1_score | 2026-10-02 | finished | 0 | 21 s | 231.8 kB |  |
| `20261002T114557Z_e4_flat_np_s2_score` | hierarchical_tm | e4_flat_np_s2_score | 2026-10-02 | finished | 0 | 22 s | 232.4 kB |  |
| `20261002T114617Z_e4_flat_np_s3_score` | hierarchical_tm | e4_flat_np_s3_score | 2026-10-02 | finished | 0 | 20 s | 232.1 kB |  |
| `20261002T114637Z_e4s_nsall_s400_s1_score` | hierarchical_tm | e4s_nsall_s400_s1_score | 2026-10-02 | finished | 0 | 21 s | 340.4 kB |  |
| `20261002T114657Z_e4s_nsall_s400_shuf_s1` | hierarchical_tm | e4s_nsall_s400_shuf_s1 | 2026-10-02 | finished | 0 | 25.7 min | 26.5 MB |  |
| `20261002T114717Z_e4s_nsall_s400_s2` | hierarchical_tm | e4s_nsall_s400_s2 | 2026-10-02 | finished | 0 | 25.5 min | 26.5 MB |  |
| `20261002T114726Z_check4a_word_groups` | hierarchical_tm | check4a_word_groups | 2026-10-02 | finished | 0 | 2.1 min | 670.2 kB |  |
| `20261002T121253Z_e4s_nsall_s400_shuf_s2` | hierarchical_tm | e4s_nsall_s400_shuf_s2 | 2026-10-02 | finished | 0 | 25.4 min | 26.5 MB |  |
| `20261002T121313Z_e4s_nsall_s400_s2_score` | hierarchical_tm | e4s_nsall_s400_s2_score | 2026-10-02 | finished | 0 | 23 s | 340.1 kB |  |
| `20261002T121348Z_e4s_nsall_s400_s3` | hierarchical_tm | e4s_nsall_s400_s3 | 2026-10-02 | finished | 0 | 25.7 min | 26.5 MB |  |
| `20261002T123824Z_e4s_nsall_s400_shuf_s3` | hierarchical_tm | e4s_nsall_s400_shuf_s3 | 2026-10-02 | finished | 0 | 27.2 min | 26.5 MB |  |
| `20261002T123929Z_e4_second_C2000_s40_sa_s1` | hierarchical_tm | e4_second_C2000_s40_sa_s1 | 2026-10-02 | finished | 0 | 30.6 min | 131.5 MB |  |
| `20261002T130550Z_e4s_nsall_s400_s3_score` | hierarchical_tm | e4s_nsall_s400_s3_score | 2026-10-02 | finished | 0 | 24 s | 341.5 kB |  |
| `20261002T130624Z_check4b_synonym_groups_s400` | hierarchical_tm | check4b_synonym_groups_s400 | 2026-10-02 | finished | 0 | 2.4 min | 881.4 kB |  |
| `20261002T130625Z_e4_second_C2000_s40_sa_s2` | hierarchical_tm | e4_second_C2000_s40_sa_s2 | 2026-10-02 | finished | 0 | 37.4 min | 131.5 MB |  |
| `20261002T131015Z_e4_second_C2000_s40_sa_s3` | hierarchical_tm | e4_second_C2000_s40_sa_s3 | 2026-10-02 | finished | 0 | 37.5 min | 131.5 MB |  |
| `20261002T131023Z_e4_second_repro_check_posthoc` | hierarchical_tm | e4_second_repro_check_posthoc | 2026-10-02 | finished | 0 | 0 s | 5.6 kB |  |
| `20261002T133519Z_final_test_report_devcheck_exploratory` | hierarchical_tm | final_test_report_devcheck_exploratory | 2026-10-02 | finished | 0 | 2.1 min | 185.6 kB |  |
| `20261002T134017Z_final_test_report_devcheck_exploratory` | hierarchical_tm | final_test_report_devcheck_exploratory | 2026-10-02 | finished | 0 | 2.2 min | 188.7 kB |  |
| `20261002T134351Z_e4_second_C2000_s40_np_s1` | hierarchical_tm | e4_second_C2000_s40_np_s1 | 2026-10-02 | finished | 0 | 37.0 min | 131.5 MB |  |
| `20261002T134414Z_final_test_report_codepath_test_exploratory` | hierarchical_tm | final_test_report_codepath_test_exploratory | 2026-10-02 | finished | 0 | 30 s | 250.0 kB |  |
| `20261002T134702Z_final_test_report_devcheck_exploratory` | hierarchical_tm | final_test_report_devcheck_exploratory | 2026-10-02 | finished | 65 | 2.1 min | 192.1 kB |  |
| `20261002T134756Z_e4_second_C2000_s40_np_s2` | hierarchical_tm | e4_second_C2000_s40_np_s2 | 2026-10-02 | finished | 0 | 36.7 min | 131.5 MB |  |
| `20261002T134938Z_final_test_report_devcheck_exploratory` | hierarchical_tm | final_test_report_devcheck_exploratory | 2026-10-02 | finished | 0 | 2.1 min | 193.5 kB |  |
| `20261002T135204Z_final_test_report_codepath_test_exploratory` | hierarchical_tm | final_test_report_codepath_test_exploratory | 2026-10-02 | finished | 0 | 29 s | 252.6 kB |  |
| `20261002T142102Z_e4_second_C2000_s40_np_s3` | hierarchical_tm | e4_second_C2000_s40_np_s3 | 2026-10-02 | finished | 0 | 30.8 min | 131.5 MB |  |
| `20261002T142452Z_e4_head_sa_s4` | hierarchical_tm | e4_head_sa_s4 | 2026-10-02 | finished | 0 | 29.6 min | 26.5 MB |  |
| `20261002T145158Z_e4_head_sa_s5` | hierarchical_tm | e4_head_sa_s5 | 2026-10-02 | finished | 0 | 27.7 min | 26.5 MB |  |
| `20261002T145433Z_e4_second_sa_s1_score` | hierarchical_tm | e4_second_sa_s1_score | 2026-10-02 | finished | 0 | 37 s | 555.2 kB |  |
| `20261002T145523Z_e4_second_sa_s2_score` | hierarchical_tm | e4_second_sa_s2_score | 2026-10-02 | finished | 0 | 37 s | 561.1 kB |  |
| `20261002T145613Z_e4_second_sa_s3_score` | hierarchical_tm | e4_second_sa_s3_score | 2026-10-02 | finished | 0 | 38 s | 561.7 kB |  |
| `20261002T145703Z_e4_second_np_s1_score` | hierarchical_tm | e4_second_np_s1_score | 2026-10-02 | finished | 0 | 37 s | 560.7 kB |  |
| `20261002T145753Z_e4_second_np_s2_score` | hierarchical_tm | e4_second_np_s2_score | 2026-10-02 | finished | 0 | 37 s | 562.0 kB |  |
| `20261002T145843Z_e4_second_np_s3_score` | hierarchical_tm | e4_second_np_s3_score | 2026-10-02 | finished | 0 | 37 s | 551.0 kB |  |
| `20261002T145933Z_e4_head_sa_s4_score` | hierarchical_tm | e4_head_sa_s4_score | 2026-10-02 | finished | 0 | 24 s | 357.6 kB |  |
| `20261002T150009Z_e4_head_np_s4` | hierarchical_tm | e4_head_np_s4 | 2026-10-02 | finished | 0 | 27.1 min | 26.5 MB |  |
| `20261002T151944Z_e4_head_np_s5` | hierarchical_tm | e4_head_np_s5 | 2026-10-02 | finished | 0 | 27.6 min | 26.5 MB |  |
| `20261002T152719Z_e4_head_sa_s5_score` | hierarchical_tm | e4_head_sa_s5_score | 2026-10-02 | finished | 0 | 24 s | 355.9 kB |  |
| `20261002T152755Z_e4_head_np_s4_score` | hierarchical_tm | e4_head_np_s4_score | 2026-10-02 | finished | 0 | 24 s | 355.7 kB |  |
| `20261002T152830Z_e4s_nsall_s100_s4` | hierarchical_tm | e4s_nsall_s100_s4 | 2026-10-02 | finished | 0 | 27.6 min | 26.5 MB |  |
| `20261002T154736Z_e4s_nsall_s100_shuf_s4` | hierarchical_tm | e4s_nsall_s100_shuf_s4 | 2026-10-02 | finished | 0 | 27.9 min | 26.5 MB |  |
| `20261002T155611Z_e4_head_np_s5_score` | hierarchical_tm | e4_head_np_s5_score | 2026-10-02 | finished | 0 | 26 s | 355.2 kB |  |
| `20261002T155646Z_e4s_nsall_s100_s4_score` | hierarchical_tm | e4s_nsall_s100_s4_score | 2026-10-02 | finished | 0 | 25 s | 342.7 kB |  |
| `20261002T155721Z_e4s_nsall_s100_s5` | hierarchical_tm | e4s_nsall_s100_s5 | 2026-10-02 | finished | 0 | 27.7 min | 26.5 MB |  |
| `20261002T161542Z_e4s_nsall_s100_shuf_s5` | hierarchical_tm | e4s_nsall_s100_shuf_s5 | 2026-10-02 | finished | 0 | 28.1 min | 26.5 MB |  |
| `20261002T162502Z_e4_switch_binary_inference_sa_s1` | hierarchical_tm | e4_switch_binary_inference_sa_s1 | 2026-10-02 | finished | 0 | 27.8 min | 26.5 MB |  |
| `20261002T164352Z_e4s_nsall_s100_s5_score` | hierarchical_tm | e4s_nsall_s100_s5_score | 2026-10-02 | finished | 0 | 24 s | 343.3 kB |  |
| `20261002T164428Z_e4_switch_and_group_norm_sa_s1` | hierarchical_tm | e4_switch_and_group_norm_sa_s1 | 2026-10-02 | finished | 0 | 28.5 min | 26.5 MB |  |
| `20261002T164442Z_final_test_report_devcheck_5seeds_exploratory` | hierarchical_tm | final_test_report_devcheck_5seeds_exploratory | 2026-10-02 | finished | 0 | 1.9 min | 126.6 kB |  |
| `20261002T165303Z_e4_switch_constant_update_p_sa_s1` | hierarchical_tm | e4_switch_constant_update_p_sa_s1 | 2026-10-02 | finished | 0 | 28.8 min | 26.5 MB |  |
| `20261002T170451Z_ngram_baselines` | clause_lm | ngram_baselines | 2026-10-02 |  |  |  | 9.5 kB |  |
| `20261002T171309Z_e4_switch_no_clipping_sa_s1` | hierarchical_tm | e4_switch_no_clipping_sa_s1 | 2026-10-02 | finished | 0 | 25.7 min | 26.5 MB |  |
| `20261002T171456Z_pairs_data_K1000` | clause_lm | pairs_data_K1000 | 2026-10-02 |  |  |  | 6.0 kB |  |
| `20261002T171904Z_pair_baselines_K1000` | clause_lm | pair_baselines_K1000 | 2026-10-02 |  |  |  | 8.4 kB |  |
| `20261002T174323Z_clause_lm_kmeans200` | clause_lm | clause_lm_kmeans200 | 2026-10-02 |  |  |  | 10.6 kB |  |
| `20261002T192818Z_e4_head_sa_s1_testscore` | hierarchical_tm | e4_head_sa_s1_testscore | 2026-10-02 | finished | 0 | 16 s | 94.7 kB |  |
| `20261002T192838Z_e4_head_sa_s2_testscore` | hierarchical_tm | e4_head_sa_s2_testscore | 2026-10-02 | finished | 0 | 14 s | 94.2 kB |  |
| `20261002T192858Z_e4_head_sa_s3_testscore` | hierarchical_tm | e4_head_sa_s3_testscore | 2026-10-02 | finished | 0 | 13 s | 94.7 kB |  |
| `20261002T192918Z_e4_head_sa_s4_testscore` | hierarchical_tm | e4_head_sa_s4_testscore | 2026-10-02 | finished | 0 | 14 s | 94.4 kB |  |
| `20261002T192938Z_e4_head_sa_s5_testscore` | hierarchical_tm | e4_head_sa_s5_testscore | 2026-10-02 | finished | 0 | 13 s | 94.2 kB |  |
| `20261002T192959Z_e4_head_np_s1_testscore` | hierarchical_tm | e4_head_np_s1_testscore | 2026-10-02 | finished | 0 | 13 s | 94.3 kB |  |
| `20261002T193019Z_e4_head_np_s2_testscore` | hierarchical_tm | e4_head_np_s2_testscore | 2026-10-02 | finished | 0 | 13 s | 93.7 kB |  |
| `20261002T193039Z_e4_head_np_s3_testscore` | hierarchical_tm | e4_head_np_s3_testscore | 2026-10-02 | finished | 0 | 13 s | 94.6 kB |  |
| `20261002T193059Z_e4_head_np_s4_testscore` | hierarchical_tm | e4_head_np_s4_testscore | 2026-10-02 | finished | 0 | 13 s | 94.3 kB |  |
| `20261002T193119Z_e4_head_np_s5_testscore` | hierarchical_tm | e4_head_np_s5_testscore | 2026-10-02 | finished | 0 | 13 s | 94.5 kB |  |
| `20261002T193140Z_e4s_nsall_s100_s1_testscore` | hierarchical_tm | e4s_nsall_s100_s1_testscore | 2026-10-02 | finished | 0 | 13 s | 92.8 kB |  |
| `20261002T193200Z_e4s_nsall_s100_s2_testscore` | hierarchical_tm | e4s_nsall_s100_s2_testscore | 2026-10-02 | finished | 0 | 13 s | 92.9 kB |  |
| `20261002T193220Z_e4s_nsall_s100_s3_testscore` | hierarchical_tm | e4s_nsall_s100_s3_testscore | 2026-10-02 | finished | 0 | 13 s | 93.2 kB |  |
| `20261002T193240Z_e4s_nsall_s100_s4_testscore` | hierarchical_tm | e4s_nsall_s100_s4_testscore | 2026-10-02 | finished | 0 | 13 s | 93.0 kB |  |
| `20261002T193301Z_e4s_nsall_s100_s5_testscore` | hierarchical_tm | e4s_nsall_s100_s5_testscore | 2026-10-02 | finished | 0 | 14 s | 93.0 kB |  |
| `20261002T193321Z_e4s_nsall_s100_shuf_s1_testscore` | hierarchical_tm | e4s_nsall_s100_shuf_s1_testscore | 2026-10-02 | finished | 0 | 13 s | 90.7 kB |  |
| `20261002T193341Z_e4s_nsall_s100_shuf_s2_testscore` | hierarchical_tm | e4s_nsall_s100_shuf_s2_testscore | 2026-10-02 | finished | 0 | 13 s | 91.9 kB |  |
| `20261002T193401Z_e4s_nsall_s100_shuf_s3_testscore` | hierarchical_tm | e4s_nsall_s100_shuf_s3_testscore | 2026-10-02 | finished | 0 | 13 s | 91.1 kB |  |
| `20261002T193422Z_e4s_nsall_s100_shuf_s4_testscore` | hierarchical_tm | e4s_nsall_s100_shuf_s4_testscore | 2026-10-02 | finished | 0 | 13 s | 91.1 kB |  |
| `20261002T193442Z_e4s_nsall_s100_shuf_s5_testscore` | hierarchical_tm | e4s_nsall_s100_shuf_s5_testscore | 2026-10-02 | finished | 0 | 14 s | 92.0 kB |  |
| `20261002T193502Z_e4s_nsall_s200_s1` | hierarchical_tm | e4s_nsall_s200_s1 | 2026-10-02 | finished | 0 | 27.3 min | 26.5 MB |  |
| `20261002T193513Z_final_test_main_table` | hierarchical_tm | final_test_main_table | 2026-10-02 | finished | 0 | 1.2 min | 127.5 kB |  |
| `20261002T193522Z_e4s_nsall_s200_shuf_s1` | hierarchical_tm | e4s_nsall_s200_shuf_s1 | 2026-10-02 | finished | 0 | 27.3 min | 26.5 MB |  |
| `20261002T194614Z_slide6_data_build` | hierarchical_tm | slide6_data_build | 2026-10-02 | finished | 0 | 3 s | 9.9 kB |  |
| `20261002T194632Z_slide6_tests_exploratory` | hierarchical_tm | slide6_tests_exploratory | 2026-10-02 | finished | 0 | 6 s | 10.2 kB |  |
| `20261002T194816Z_slide6_tests_v2_exploratory` | hierarchical_tm | slide6_tests_v2_exploratory | 2026-10-02 | finished | 0 | 6 s | 8.3 kB |  |
| `20261002T195152Z_m5_setup_sorted` | clause_lm | m5_setup_sorted | 2026-10-02 |  |  |  | 23.6 kB |  |
| `20261002T195255Z_m5_gate0_s8_seed42_sorted` | clause_lm | m5_gate0_s8_seed42_sorted | 2026-10-02 |  |  |  | 29.4 kB |  |
| `20261002T200228Z_slide6_screen_s25` | hierarchical_tm | slide6_screen_s25 | 2026-10-02 | finished | 0 | 10.9 min | 1.3 MB |  |
| `20261002T200248Z_slide6_screen_s50` | hierarchical_tm | slide6_screen_s50 | 2026-10-02 | finished | 0 | 11.2 min | 1.3 MB |  |
| `20261002T201024Z_e3_slide10_data_build` | hierarchical_tm | e3_slide10_data_build | 2026-10-02 | finished | 0 | 3 s | 7.2 kB |  |
| `20261002T201324Z_slide6_screen_s100` | hierarchical_tm | slide6_screen_s100 | 2026-10-02 | finished | 0 | 10.9 min | 1.3 MB |  |
| `20261002T201414Z_e3_slide10_hier_C50_A4_K6_s12_seed1` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s12_seed1 | 2026-10-02 | finished | 0 | 3.4 min | 1.2 MB |  |
| `20261002T201749Z_e3_slide10_hier_C50_A4_K6_s40_seed1` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s40_seed1 | 2026-10-02 | finished | 0 | 3.2 min | 1.2 MB |  |
| `20261002T202109Z_e3_slide10_hier_C50_A4_K6_s150_seed1` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s150_seed1 | 2026-10-02 | finished | 0 | 3.3 min | 1.2 MB |  |
| `20261002T202430Z_e4s_nsall_s200_s2` | hierarchical_tm | e4s_nsall_s200_s2 | 2026-10-02 | finished | 0 | 26.3 min | 26.5 MB |  |
| `20261002T202450Z_e4s_nsall_s200_shuf_s2` | hierarchical_tm | e4s_nsall_s200_shuf_s2 | 2026-10-02 | finished | 0 | 27.0 min | 26.5 MB |  |
| `20261002T202450Z_slide6_s_choice` | hierarchical_tm | slide6_s_choice | 2026-10-02 | finished | 0 | 4 s | 15.5 kB |  |
| `20261002T202633Z_e3_screen_check2b_dev` | hierarchical_tm | e3_screen_check2b_dev | 2026-10-02 | finished | 0 | 3 s | 12.2 kB |  |
| `20261002T202821Z_e5a_cpu_tests_v3_after_classnames_exploratory` | hierarchical_tm | e5a_cpu_tests_v3_after_classnames_exploratory | 2026-10-02 | finished | 0 | 22 s | 9.3 kB |  |
| `20261002T205056Z_slide6_s25_seed2` | hierarchical_tm | slide6_s25_seed2 | 2026-10-02 | finished | 0 | 11.1 min | 1.3 MB |  |
| `20261002T205115Z_m5_setup_first_appearance` | clause_lm | m5_setup_first_appearance | 2026-10-02 |  |  |  | 23.7 kB |  |
| `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` | clause_lm | m5_gate0_s8_seed42_first_appearance | 2026-10-02 |  |  |  | 23.8 kB |  |
| `20261002T205201Z_slide6_s25_seed3` | hierarchical_tm | slide6_s25_seed3 | 2026-10-02 | finished | 0 | 11.1 min | 1.3 MB |  |
| `20261002T210206Z_e3_slide10_hier_C50_A4_K6_s150_seed2` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s150_seed2 | 2026-10-02 | finished | 0 | 11.2 min | 1.2 MB |  |
| `20261002T210312Z_e3_slide10_hier_C50_A4_K6_s150_seed3` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s150_seed3 | 2026-10-02 | finished | 0 | 11.3 min | 1.2 MB |  |
| `20261002T210327Z_slide6_eval_claim` | hierarchical_tm | slide6_eval_claim | 2026-10-02 | finished | 0 | 3 s | 19.5 kB |  |
| `20261002T210555Z_m5_setup_torchcpu` | clause_lm | m5_setup_torchcpu | 2026-10-02 |  |  |  | 14.1 kB |  |
| `20261002T211332Z_e3_slide10_hier_C50_A4_K6_s150_seed4` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s150_seed4 | 2026-10-02 | finished | 0 | 11.2 min | 1.2 MB |  |
| `20261002T211437Z_e3_slide10_hier_C50_A4_K6_s150_seed5` | hierarchical_tm | e3_slide10_hier_C50_A4_K6_s150_seed5 | 2026-10-02 | finished | 0 | 10.9 min | 1.2 MB |  |
| `20261002T212458Z_e3_slide10_flat_C1200_s150_seed1` | hierarchical_tm | e3_slide10_flat_C1200_s150_seed1 | 2026-10-02 | finished | 0 | 9.0 min | 1.2 MB |  |
| `20261002T212548Z_e3_slide10_flat_C1200_s150_seed2` | hierarchical_tm | e3_slide10_flat_C1200_s150_seed2 | 2026-10-02 | finished | 0 | 8.9 min | 1.2 MB |  |
| `20261002T213043Z_e3_check2b_eval_hier_s150` | hierarchical_tm | e3_check2b_eval_hier_s150 | 2026-10-02 | finished | 0 | 1 s | 24.3 kB |  |
| `20261002T213109Z_e3_pair_overlap_diagnostic_posthoc_exploratory` | hierarchical_tm | e3_pair_overlap_diagnostic_posthoc_exploratory | 2026-10-02 | finished | 0 | 1 s | 12.5 kB |  |
| `20261002T213409Z_e3_slide10_flat_C1200_s150_seed3` | hierarchical_tm | e3_slide10_flat_C1200_s150_seed3 | 2026-10-02 | finished | 0 | 9.0 min | 1.2 MB |  |
| `20261002T213459Z_e3_slide10_hier_C50_A8_K8_s150_seed1` | hierarchical_tm | e3_slide10_hier_C50_A8_K8_s150_seed1 | 2026-10-02 | finished | 0 | 11.1 min | 3.1 MB |  |
| `20261002T214319Z_e3_slide10_hier_C50_A8_K8_s150_seed2` | hierarchical_tm | e3_slide10_hier_C50_A8_K8_s150_seed2 | 2026-10-02 | finished | 0 | 11.2 min | 3.1 MB |  |
| `20261002T214610Z_e3_slide10_hier_C50_A8_K8_s150_seed3` | hierarchical_tm | e3_slide10_hier_C50_A8_K8_s150_seed3 | 2026-10-02 | finished | 0 | 11.1 min | 3.1 MB |  |
| `20261002T215430Z_e3fb_slide10_text_C2_A64_K6_s5_seed1` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s5_seed1 | 2026-10-02 | finished | 0 | 30.0 min | 689.1 kB |  |
| `20261002T215720Z_e3fb_slide10_text_C2_A64_K6_s21_seed1` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s21_seed1 | 2026-10-02 | finished | 0 | 29.3 min | 709.7 kB |  |
| `20261002T215757Z_e3_check2b_eval_flat_C1200_s150` | hierarchical_tm | e3_check2b_eval_flat_C1200_s150 | 2026-10-02 | finished | 0 | 2 s | 16.1 kB |  |
| `20261002T215759Z_e3_check2b_eval_hier_A8K8_s150` | hierarchical_tm | e3_check2b_eval_hier_A8K8_s150 | 2026-10-02 | finished | 0 | 1 s | 20.3 kB |  |
| `20261002T222441Z_e3fb_slide10_text_C2_A64_K6_s100_seed1` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s100_seed1 | 2026-10-02 | finished | 0 | 29.8 min | 736.2 kB |  |
| `20261002T222646Z_e4s_nsall_s200_s3` | hierarchical_tm | e4s_nsall_s200_s3 | 2026-10-02 | finished | 0 | 25.5 min | 26.5 MB |  |
| `20261002T225223Z_e4s_nsall_s200_shuf_s3` | hierarchical_tm | e4s_nsall_s200_shuf_s3 | 2026-10-02 | finished | 0 | 25.6 min | 26.5 MB |  |
| `20261002T225427Z_e3diag_R1_hier_C50_A64_K2_T3200_s100_seed1_exploratory` | hierarchical_tm | e3diag_R1_hier_C50_A64_K2_T3200_s100_seed1_exploratory | 2026-10-02 | finished | 0 | 12.2 min | 6.6 MB |  |
| `20261002T225455Z_e3fb_s_choice` | hierarchical_tm | e3fb_s_choice | 2026-10-02 | finished | 0 | 1 s | 24.1 kB |  |
| `20261002T230648Z_e4s_nsall_s200_s1_score` | hierarchical_tm | e4s_nsall_s200_s1_score | 2026-10-02 | finished | 0 | 24 s | 341.9 kB |  |
| `20261002T230723Z_e4s_nsall_s200_s2_score` | hierarchical_tm | e4s_nsall_s200_s2_score | 2026-10-02 | finished | 0 | 24 s | 340.9 kB |  |
| `20261002T230801Z_e4s_nsall_s200_s3_score` | hierarchical_tm | e4s_nsall_s200_s3_score | 2026-10-02 | finished | 0 | 24 s | 341.6 kB |  |
| `20261002T230834Z_e3fb_slide10_text_C2_A64_K6_s21_seed2` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s21_seed2 | 2026-10-02 | finished | 0 | 29.5 min | 702.9 kB |  |
| `20261002T231809Z_e3fb_slide10_text_C2_A64_K6_s21_seed3` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s21_seed3 | 2026-10-02 | finished | 0 | 29.3 min | 702.1 kB |  |
| `20261002T231829Z_check4b_synonym_groups_s200` | hierarchical_tm | check4b_synonym_groups_s200 | 2026-10-02 | finished | 0 | 2.1 min | 730.1 kB |  |
| `20261002T233815Z_e3fb_slide10_text_C2_A64_K6_s21_seed4` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s21_seed4 | 2026-10-02 | finished | 0 | 29.9 min | 706.8 kB |  |
| `20261002T234736Z_e3fb_slide10_text_C2_A64_K6_s21_seed5` | hierarchical_tm | e3fb_slide10_text_C2_A64_K6_s21_seed5 | 2026-10-02 | finished | 0 | 29.1 min | 703.6 kB |  |
| `20261003T000813Z_e3diag_R1_hier_C50_A64_K2_T3200_s100_seed2_exploratory` | hierarchical_tm | e3diag_R1_hier_C50_A64_K2_T3200_s100_seed2_exploratory | 2026-10-03 | finished | 0 | 12.1 min | 6.6 MB |  |
| `20261003T001647Z_e3diag_R1_hier_C50_A64_K2_T3200_s100_seed3_exploratory` | hierarchical_tm | e3diag_R1_hier_C50_A64_K2_T3200_s100_seed3_exploratory | 2026-10-03 | finished | 0 | 11.5 min | 6.6 MB |  |
| `20261003T001719Z_e3fb_check2b_eval_s21` | hierarchical_tm | e3fb_check2b_eval_s21 | 2026-10-03 | finished | 0 | 1 s | 102.6 kB |  |
| `20261003T001721Z_e3fb_deletion_probe_posthoc_exploratory` | hierarchical_tm | e3fb_deletion_probe_posthoc_exploratory | 2026-10-03 | finished | 0 | 2 s | 20.2 kB |  |
| `20261003T001723Z_e3fb_gpu_time` | hierarchical_tm | e3fb_gpu_time | 2026-10-03 | finished | 0 | 0 s | 8.1 kB |  |
| `20261003T002022Z_e3diag_R2_hier_C50_A4_K6_s25_seed1_exploratory` | hierarchical_tm | e3diag_R2_hier_C50_A4_K6_s25_seed1_exploratory | 2026-10-03 | finished | 0 | 11.2 min | 1.2 MB |  |
| `20261003T002827Z_e3diag_R2_hier_C50_A4_K6_s25_seed2_exploratory` | hierarchical_tm | e3diag_R2_hier_C50_A4_K6_s25_seed2_exploratory | 2026-10-03 | finished | 0 | 11.2 min | 1.2 MB |  |
| `20261003T003133Z_e3diag_R2_hier_C50_A4_K6_s25_seed3_exploratory` | hierarchical_tm | e3diag_R2_hier_C50_A4_K6_s25_seed3_exploratory | 2026-10-03 | finished | 0 | 11.2 min | 1.2 MB |  |
| `20261003T003953Z_s6diag_R1_hier_C50_A64_K2_T3200_s100_seed1_exploratory` | hierarchical_tm | s6diag_R1_hier_C50_A64_K2_T3200_s100_seed1_exploratory | 2026-10-03 | finished | 0 | 11.5 min | 6.6 MB |  |
| `20261003T004244Z_s6diag_R1_hier_C50_A64_K2_T3200_s100_seed2_exploratory` | hierarchical_tm | s6diag_R1_hier_C50_A64_K2_T3200_s100_seed2_exploratory | 2026-10-03 | finished | 0 | 11.5 min | 6.5 MB |  |
| `20261003T004307Z_e3diag_R1_report_posthoc_exploratory` | hierarchical_tm | e3diag_R1_report_posthoc_exploratory | 2026-10-03 | finished | 0 | 2 s | 64.2 kB |  |
| `20261003T004309Z_e3diag_R2_report_posthoc_exploratory` | hierarchical_tm | e3diag_R2_report_posthoc_exploratory | 2026-10-03 | finished | 0 | 1 s | 17.3 kB |  |
| `20261003T005134Z_s6diag_R1_hier_C50_A64_K2_T3200_s100_seed3_exploratory` | hierarchical_tm | s6diag_R1_hier_C50_A64_K2_T3200_s100_seed3_exploratory | 2026-10-03 | finished | 0 | 12.9 min | 6.6 MB |  |
| `20261003T005424Z_e3fb_slide10_text_flat_C768_s21_seed1` | hierarchical_tm | e3fb_slide10_text_flat_C768_s21_seed1 | 2026-10-03 | finished | 0 | 30.3 min | 978.0 kB |  |
| `20261003T010445Z_e3fb_slide10_text_flat_C768_s21_seed2` | hierarchical_tm | e3fb_slide10_text_flat_C768_s21_seed2 | 2026-10-03 | finished | 0 | 30.6 min | 961.4 kB |  |
| `20261003T010506Z_s6diag_R1_report_posthoc_exploratory` | hierarchical_tm | s6diag_R1_report_posthoc_exploratory | 2026-10-03 | finished | 0 | 2 s | 53.4 kB |  |
| `20261003T012450Z_e3fb_slide10_text_flat_C768_s21_seed3` | hierarchical_tm | e3fb_slide10_text_flat_C768_s21_seed3 | 2026-10-03 | finished | 0 | 29.1 min | 961.5 kB |  |
| `20261003T015436Z_e3fb_flat_C768_check2b_eval_and_probe_exploratory` | hierarchical_tm | e3fb_flat_C768_check2b_eval_and_probe_exploratory | 2026-10-03 | finished | 126 | 3 s | 16.0 kB |  |
| `20261003T015449Z_e3fb_flat_C768_check2b_eval_and_probe_v2_exploratory` | hierarchical_tm | e3fb_flat_C768_check2b_eval_and_probe_v2_exploratory | 2026-10-03 | finished | 0 | 4 s | 33.1 kB |  |
| `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu` | clause_lm | m5_gate0_s8_seed42_sorted_torchgpu | 2026-10-14 |  |  |  | 27.1 kB |  |
| `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu` | clause_lm | m5_gate0_s8_seed42_sorted_torchgpu | 2026-10-14 |  |  |  | 33.3 kB |  |
| `20261014T221627Z_m5_s8_seed123_sorted_torchgpu` | clause_lm | m5_s8_seed123_sorted_torchgpu | 2026-10-14 |  |  |  | 32.7 kB |  |
| `20261014T221728Z_m5_s8_seed456_sorted_torchgpu` | clause_lm | m5_s8_seed456_sorted_torchgpu | 2026-10-14 |  |  |  | 32.7 kB |  |
| `20261014T221854Z_m5_s25_seed42_sorted_torchgpu` | clause_lm | m5_s25_seed42_sorted_torchgpu | 2026-10-14 |  |  |  | 33.0 kB |  |
| `20261014T221959Z_m5_s25_seed123_sorted_torchgpu` | clause_lm | m5_s25_seed123_sorted_torchgpu | 2026-10-14 |  |  |  | 32.8 kB |  |
| `20261014T222054Z_m5_s25_seed456_sorted_torchgpu` | clause_lm | m5_s25_seed456_sorted_torchgpu | 2026-10-14 |  |  |  | 32.9 kB |  |
| `20261014T222148Z_m5_s50_seed42_sorted_torchgpu` | clause_lm | m5_s50_seed42_sorted_torchgpu | 2026-10-14 |  |  |  | 33.0 kB |  |
| `20261014T222240Z_m5_s50_seed123_sorted_torchgpu` | clause_lm | m5_s50_seed123_sorted_torchgpu | 2026-10-14 |  |  |  | 32.9 kB |  |
| `20261014T222338Z_m5_s50_seed456_sorted_torchgpu` | clause_lm | m5_s50_seed456_sorted_torchgpu | 2026-10-14 |  |  |  | 32.7 kB |  |
| `20261014T222435Z_m5_s100_seed42_sorted_torchgpu` | clause_lm | m5_s100_seed42_sorted_torchgpu | 2026-10-14 |  |  |  | 33.0 kB |  |
| `20261014T222536Z_m5_s100_seed123_sorted_torchgpu` | clause_lm | m5_s100_seed123_sorted_torchgpu | 2026-10-14 |  |  |  | 33.0 kB |  |
| `20261014T222635Z_m5_s100_seed456_sorted_torchgpu` | clause_lm | m5_s100_seed456_sorted_torchgpu | 2026-10-14 |  |  |  | 32.9 kB |  |
| `20261014T222730Z_m5_summary_sorted_torchgpu` | clause_lm | m5_summary_sorted_torchgpu | 2026-10-14 |  |  |  | 15.8 kB |  |
| `20261015T071015Z_b6timing_fmnist_thresh10_c100` | tally | b6timing_fmnist_thresh10_c100 | 2026-10-15 | finished |  | 1.3 min | 3.4 kB |  |
| `20261015T071028Z_a1_baseline_check` | tally | a1_baseline_check | 2026-10-15 | finished |  | 16 s | 50.3 kB |  |
| `20261015T071135Z_b6timing_fmnist_thermo3_c100` | tally | b6timing_fmnist_thermo3_c100 | 2026-10-15 | finished |  | 1.5 min | 3.6 kB |  |
| `20261015T071151Z_a1_baseline_check` | tally | a1_baseline_check | 2026-10-15 | finished |  | 17 s | 50.6 kB |  |
| `20261015T071247Z_t0_smoke_coal_c200_s10_seed1` | tally | t0_smoke_coal_c200_s10_seed1 | 2026-10-15 | finished |  | 5.3 min | 9.2 MB | smoke |
| `20261015T071309Z_b6timing_fmnist_thermo4_c100` | tally | b6timing_fmnist_thermo4_c100 | 2026-10-15 | finished |  | 1.7 min | 4.1 kB |  |
| `20261015T071457Z_b6timing_fmnist_hog_c100` | tally | b6timing_fmnist_hog_c100 | 2026-10-15 | finished |  | 25 s | 4.3 kB |  |
| `20261015T071523Z_b6timing_agnews_word1_c100` | tally | b6timing_agnews_word1_c100 | 2026-10-15 | finished |  | 2.4 min | 4.4 kB |  |
| `20261015T071713Z_b1_smoke_calib` | tally | b1_smoke_calib | 2026-10-15 | finished |  | 30 s | 8.0 kB | smoke |
| `20261015T071752Z_b6timing_agnews_word2_c100` | tally | b6timing_agnews_word2_c100 | 2026-10-15 | finished |  | 3.6 min | 4.7 kB |  |
| `20261015T072026Z_t0_tune_coal_c2000_s10_seed1` | tally | t0_tune_coal_c2000_s10_seed1 | 2026-10-15 | finished |  | 10.4 h | 82.6 MB |  |
| `20261015T072026Z_t0_tune_coal_c2000_s20_seed1` | tally | t0_tune_coal_c2000_s20_seed1 | 2026-10-15 | finished |  | 11.0 h | 82.6 MB |  |
| `20261015T072026Z_t0_tune_coal_c2000_s5_seed1` | tally | t0_tune_coal_c2000_s5_seed1 | 2026-10-15 | finished |  | 7.9 h | 82.6 MB |  |
| `20261015T072127Z_b6timing_agnews_char4_c100` | tally | b6timing_agnews_char4_c100 | 2026-10-15 | finished |  | 1.4 min | 4.7 kB |  |
| `20261015T072146Z_t0_smoke_flat_c20_s10_seed1` | tally | t0_smoke_flat_c20_s10_seed1 | 2026-10-15 | finished |  | 1.5 min | 801.7 MB | smoke |
| `20261015T072257Z_b6timing_agnews_concat_c300` | tally | b6timing_agnews_concat_c300 | 2026-10-15 | finished |  | 15.0 min | 4.8 kB |  |
| `20261015T072321Z_b1_smoke_calib` | tally | b1_smoke_calib | 2026-10-15 | finished |  | 1.9 min | 12.2 kB | smoke |
| `20261015T072425Z_t0_logreg_t0_smoke_coal_c200_s10_seed1` | tally | t0_logreg_t0_smoke_coal_c200_s10_seed1 | 2026-10-15 | crashed |  | 21 s | 5.7 kB | smoke |
| `20261015T072457Z_b1_smoke_dual_M8_s0` | tally | b1_smoke_dual_M8_s0 | 2026-10-15 | finished |  | 2 s | 18.1 kB | smoke |
| `20261015T072500Z_t0_logreg_t0_logreg_t0_smoke_coal_c200_s10_seed1` | tally | t0_logreg_t0_logreg_t0_smoke_coal_c200_s10_seed1 | 2026-10-15 | crashed |  | 1 s | 5.4 kB | smoke |
| `20261015T072511Z_t0_logreg_t0_smoke_coal_c200_s10_seed1` | tally | t0_logreg_t0_smoke_coal_c200_s10_seed1 | 2026-10-15 | crashed |  | 17 s | 5.8 kB | smoke |
| `20261015T072513Z_a1_baseline_check` | tally | a1_baseline_check | 2026-10-15 | finished |  | 18 s | 57.1 kB |  |
| `20261015T072526Z_b1_smoke_greedy_M8_s0` | tally | b1_smoke_greedy_M8_s0 | 2026-10-15 | finished |  | 6 s | 19.7 kB | smoke |
| `20261015T072534Z_b1_smoke_random_M8_s0` | tally | b1_smoke_random_M8_s0 | 2026-10-15 | finished |  | 1 s | 19.6 kB | smoke |
| `20261015T072535Z_t0_tune_flat_c100_s10_seed1` | tally | t0_tune_flat_c100_s10_seed1 | 2026-10-15 | finished |  | 4.1 h | 3.8 GB |  |
| `20261015T072536Z_b1_smoke_oracle_M8_s0` | tally | b1_smoke_oracle_M8_s0 | 2026-10-15 | finished |  | 6 s | 22.1 kB | smoke |
| `20261015T072549Z_t0_logreg_of_smoke_coal_c200_s10_seed1` | tally | t0_logreg_of_smoke_coal_c200_s10_seed1 | 2026-10-15 | finished |  | 25 s | 49.6 kB | smoke |
| `20261015T072600Z_b1_smoke_flat_tune_M8_s10000` | tally | b1_smoke_flat_tune_M8_s10000 | 2026-10-15 | finished |  | 21 s | 5.7 kB | smoke |
| `20261015T072630Z_b1_smoke_flat_M8_s0` | tally | b1_smoke_flat_M8_s0 | 2026-10-15 | running |  |  | 4.5 kB | stale (recorded as running), smoke |
| `20261015T072640Z_t0_report_valid_smoke` | tally | t0_report_valid_smoke | 2026-10-15 | finished |  | 0 s | 5.5 kB | smoke |
| `20261015T072811Z_b1_smoke_ab_M8_s0` | tally | b1_smoke_ab_M8_s0 | 2026-10-15 | finished |  | 4 s | 18.2 kB | smoke |
| `20261015T072816Z_b1_smoke_shared_M8_s0` | tally | b1_smoke_shared_M8_s0 | 2026-10-15 | finished |  | 1 s | 17.8 kB | smoke |
| `20261015T072818Z_b1_smoke_selfish_M8_s0` | tally | b1_smoke_selfish_M8_s0 | 2026-10-15 | finished |  | 1 s | 17.8 kB | smoke |
| `20261015T072820Z_b1_smoke_marginal_M8_s0` | tally | b1_smoke_marginal_M8_s0 | 2026-10-15 | finished |  | 1 s | 17.9 kB | smoke |
| `20261015T073241Z_b1_smoke_flat_M8_s0` | tally | b1_smoke_flat_M8_s0 | 2026-10-15 | finished |  | 28 s | 14.2 kB | smoke |
| `20261015T073554Z_b1_smoke_dual_M8_s0` | tally | b1_smoke_dual_M8_s0 | 2026-10-15 | finished |  | 3 s | 18.9 kB | smoke |
| `20261015T073805Z_b6timing_fmnist_thresh10_c400` | tally | b6timing_fmnist_thresh10_c400 | 2026-10-15 | finished |  | 3.4 min | 5.7 kB |  |
| `20261015T074132Z_b1_smoke_calib2` | tally | b1_smoke_calib2 | 2026-10-15 | finished |  | 2.8 min | 15.8 kB | smoke |
| `20261015T074420Z_a2_smoke` | tally | a2_smoke | 2026-10-15 | finished |  | 1.1 min | 155.4 kB | smoke |
| `20261015T093706Z_c3prep_lstm_medium_seed1_valid` | tally | c3prep_lstm_medium_seed1_valid | 2026-10-15 | finished |  | 20.3 min | 24.1 kB |  |
| `20261015T093937Z_c3prep_lstm_medium_seed2_valid` | tally | c3prep_lstm_medium_seed2_valid | 2026-10-15 | finished |  | 23.5 min | 24.1 kB |  |
| `20261015T095728Z_c3prep_lstm_large_seed1_valid` | tally | c3prep_lstm_large_seed1_valid | 2026-10-15 | finished |  | 1.3 h | 31.7 kB |  |
| `20261015T100311Z_c3prep_lstm_large_seed2_valid` | tally | c3prep_lstm_large_seed2_valid | 2026-10-15 | finished |  | 1.3 h | 31.7 kB |  |
| `20261015T111901Z_c3prep_lstm_medium_seed3_valid` | tally | c3prep_lstm_medium_seed3_valid | 2026-10-15 | finished |  | 10.3 min | 24.1 kB |  |
| `20261015T112933Z_c3prep_lstm_large_seed3_valid` | tally | c3prep_lstm_large_seed3_valid | 2026-10-15 | finished |  | 32.9 min | 31.7 kB |  |
| `20261015T113259Z_t0_tune_flat_c100_s5_seed1` | tally | t0_tune_flat_c100_s5_seed1 | 2026-10-15 | finished |  | 6.6 h | 3.8 GB |  |
| `20261015T120236Z_c3prep_lstm_medium_seed4_valid` | tally | c3prep_lstm_medium_seed4_valid | 2026-10-15 | finished |  | 10.3 min | 24.1 kB |  |
| `20261015T121301Z_c3prep_lstm_large_seed4_valid` | tally | c3prep_lstm_large_seed4_valid | 2026-10-15 | finished |  | 32.6 min | 31.7 kB |  |
| `20261015T124555Z_c3prep_lstm_medium_seed5_valid` | tally | c3prep_lstm_medium_seed5_valid | 2026-10-15 | finished |  | 10.4 min | 24.1 kB |  |
| `20261015T125621Z_c3prep_lstm_large_seed5_valid` | tally | c3prep_lstm_large_seed5_valid | 2026-10-15 | finished |  | 32.8 min | 31.7 kB |  |
| `20261015T162840Z_c3prep_lstm_large_seed6_valid` | tally | c3prep_lstm_large_seed6_valid | 2026-10-15 | finished |  | 1.3 h | 31.7 kB |  |
| `20261015T162840Z_c3prep_lstm_large_seed9_valid` | tally | c3prep_lstm_large_seed9_valid | 2026-10-15 | finished |  | 1.3 h | 31.7 kB |  |
| `20261015T174735Z_c3prep_lstm_medium_seed9_valid` | tally | c3prep_lstm_medium_seed9_valid | 2026-10-15 | finished |  | 23.2 min | 24.1 kB |  |
| `20261015T174824Z_c3prep_lstm_medium_seed6_valid` | tally | c3prep_lstm_medium_seed6_valid | 2026-10-15 | finished |  | 23.1 min | 24.1 kB |  |
| `20261015T181024Z_t0_tune_flat_c100_s20_seed1` | tally | t0_tune_flat_c100_s20_seed1 | 2026-10-15 | finished |  | 3.0 h | 3.8 GB |  |
| `20261015T181049Z_c3prep_lstm_large_seed10_valid` | tally | c3prep_lstm_large_seed10_valid | 2026-10-15 | finished |  | 1.3 h | 31.7 kB |  |
| `20261015T181134Z_c3prep_lstm_large_seed7_valid` | tally | c3prep_lstm_large_seed7_valid | 2026-10-15 | finished |  | 1.3 h | 31.7 kB |  |
| `20261015T182430Z_t0_final_coal_c2000_s20_seed1` | tally | t0_final_coal_c2000_s20_seed1 | 2026-10-15 | finished |  | 23.0 h | 183.5 MB |  |
| `20261015T182432Z_t0_final_coal_c2000_s20_seed2` | tally | t0_final_coal_c2000_s20_seed2 | 2026-10-15 | finished |  | 22.5 h | 184.6 MB |  |
| `20261015T182434Z_t0_final_coal_c2000_s20_seed3` | tally | t0_final_coal_c2000_s20_seed3 | 2026-10-15 | finished |  | 22.6 h | 184.7 MB |  |
| `20261015T182436Z_t0_final_coal_c2000_s20_seed4` | tally | t0_final_coal_c2000_s20_seed4 | 2026-10-15 | finished |  | 22.6 h | 185.1 MB |  |
| `20261015T192938Z_c3prep_lstm_medium_seed10_valid` | tally | c3prep_lstm_medium_seed10_valid | 2026-10-15 | finished |  | 23.2 min | 24.1 kB |  |
| `20261015T193004Z_c3prep_lstm_medium_seed7_valid` | tally | c3prep_lstm_medium_seed7_valid | 2026-10-15 | finished |  | 23.1 min | 24.1 kB |  |
| `20261015T195312Z_c3prep_lstm_large_seed8_valid` | tally | c3prep_lstm_large_seed8_valid | 2026-10-15 | finished |  | 32.6 min | 31.7 kB |  |
| `20261015T202550Z_c3prep_lstm_medium_seed8_valid` | tally | c3prep_lstm_medium_seed8_valid | 2026-10-15 | finished |  | 10.3 min | 24.1 kB |  |
| `20261015T211037Z_t0_final_coal_c2000_s20_seed5` | tally | t0_final_coal_c2000_s20_seed5 | 2026-10-15 | finished |  | 22.1 h | 184.7 MB |  |
| `20261016T165201Z_t0_final_coal_c2000_s20_seed10` | tally | t0_final_coal_c2000_s20_seed10 | 2026-10-16 | finished |  | 15.8 h | 181.6 MB |  |
| `20261016T165204Z_t0_logreg_of_final_coal_c2000_s20_seed2` | tally | t0_logreg_of_final_coal_c2000_s20_seed2 | 2026-10-16 | finished |  | 15.9 min | 1.2 MB |  |
| `20261016T165943Z_t0_final_flat_c100_s5_seed2` | tally | t0_final_flat_c100_s5_seed2 | 2026-10-16 | finished |  | 1.7 h | 3.8 GB |  |
| `20261016T170310Z_t0_final_coal_c2000_s20_seed6` | tally | t0_final_coal_c2000_s20_seed6 | 2026-10-16 | finished |  | 16.3 h | 184.1 MB |  |
| `20261016T170800Z_t0_logreg_of_final_coal_c2000_s20_seed4` | tally | t0_logreg_of_final_coal_c2000_s20_seed4 | 2026-10-16 | finished |  | 15.5 min | 1.2 MB |  |
| `20261016T172331Z_t0_logreg_of_final_coal_c2000_s20_seed3` | tally | t0_logreg_of_final_coal_c2000_s20_seed3 | 2026-10-16 | finished |  | 16.0 min | 1.2 MB |  |
| `20261016T172503Z_t0_final_flat_c100_s5_seed4` | tally | t0_final_flat_c100_s5_seed4 | 2026-10-16 | finished |  | 1.8 h | 3.8 GB |  |
| `20261016T173933Z_t0_logreg_of_final_coal_c2000_s20_seed1` | tally | t0_logreg_of_final_coal_c2000_s20_seed1 | 2026-10-16 | finished |  | 15.6 min | 1.2 MB |  |
| `20261016T184257Z_t0_final_flat_c100_s5_seed5` | tally | t0_final_flat_c100_s5_seed5 | 2026-10-16 | finished |  | 1.7 h | 3.8 GB |  |
| `20261016T191339Z_t0_final_coal_c2000_s20_seed7` | tally | t0_final_coal_c2000_s20_seed7 | 2026-10-16 | finished |  | 16.4 h | 183.5 MB |  |
| `20261016T191524Z_t0_final_flat_c100_s5_seed9` | tally | t0_final_flat_c100_s5_seed9 | 2026-10-16 | finished |  | 1.7 h | 3.8 GB |  |
| `20261016T191529Z_t0_logreg_of_final_coal_c2000_s20_seed5` | tally | t0_logreg_of_final_coal_c2000_s20_seed5 | 2026-10-16 | finished |  | 16.0 min | 1.2 MB |  |
| `20261016T202538Z_t0_final_flat_c100_s5_seed3` | tally | t0_final_flat_c100_s5_seed3 | 2026-10-16 | finished |  | 2.5 h | 3.8 GB |  |
| `20261016T205928Z_t0_final_coal_c2000_s20_seed9` | tally | t0_final_coal_c2000_s20_seed9 | 2026-10-16 | finished |  | 16.0 h | 183.9 MB |  |
| `20261016T225440Z_t0_final_flat_c100_s5_seed8` | tally | t0_final_flat_c100_s5_seed8 | 2026-10-16 | finished |  | 4.3 h | 3.8 GB |  |
| `20261017T031001Z_t0_final_flat_c100_s5_seed1` | tally | t0_final_flat_c100_s5_seed1 | 2026-10-17 | finished |  | 2.3 h | 3.8 GB |  |
| `20261017T052825Z_t0_final_flat_c100_s5_seed10` | tally | t0_final_flat_c100_s5_seed10 | 2026-10-17 | finished |  | 2.8 h | 3.8 GB |  |
| `20261017T081705Z_t0_final_flat_c100_s5_seed6` | tally | t0_final_flat_c100_s5_seed6 | 2026-10-17 | finished |  | 3.1 h | 3.8 GB |  |
| `20261017T083834Z_t0_final_flat_c100_s5_seed7` | tally | t0_final_flat_c100_s5_seed7 | 2026-10-17 | finished |  | 2.8 h | 3.8 GB |  |
| `20261017T091936Z_t0_final_coal_c2000_s20_seed8` | tally | t0_final_coal_c2000_s20_seed8 | 2026-10-17 | finished |  | 10.6 h | 186.6 MB |  |
| `20261017T095156Z_t0_logreg_of_final_coal_c2000_s20_seed6` | tally | t0_logreg_of_final_coal_c2000_s20_seed6 | 2026-10-17 | finished |  | 16.1 min | 1.2 MB |  |
| `20261017T100805Z_t0_logreg_of_final_coal_c2000_s20_seed10` | tally | t0_logreg_of_final_coal_c2000_s20_seed10 | 2026-10-17 | finished |  | 15.4 min | 1.2 MB |  |
| `20261017T113732Z_t0_logreg_of_final_coal_c2000_s20_seed7` | tally | t0_logreg_of_final_coal_c2000_s20_seed7 | 2026-10-17 | finished |  | 16.2 min | 1.2 MB |  |
| `20261017T131723Z_t0_logreg_of_final_coal_c2000_s20_seed9` | tally | t0_logreg_of_final_coal_c2000_s20_seed9 | 2026-10-17 | finished |  | 15.3 min | 1.2 MB |  |
| `20261017T204300Z_t0_logreg_of_final_coal_c2000_s20_seed8` | tally | t0_logreg_of_final_coal_c2000_s20_seed8 | 2026-10-17 | finished |  | 15.9 min | 1.2 MB |  |
