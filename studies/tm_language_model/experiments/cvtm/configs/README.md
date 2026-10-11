# Configs of the count-vote and self-supervised experiments

The JSON files in this folder are the configs of the experiment scripts in `experiments/`. They are
kept as they were used: a run stores the content of its config in its run folder (`config.json`), and the run
record holds the SHA-256 of that content in canonical JSON form (`config_sha256`) as it stood when
the run started.

The text of 23 of these files was edited for publication on 2026-10-10. Only string values were
changed: descriptive text (the values of keys such as `status`, `docs`, `why` and the note keys),
and in `ssl_r4_controls.json` an absolute folder prefix at the start of two command strings. No key
was added, renamed or removed, and no parameter, seed or threshold was changed. The 23 files are:

`H8_11_masked.json`, `H8_1_causal.json`, `H8_1_masked.json`, `cvtm_a3_H8_2.json`,
`cvtm_a3_H8_7.json`, `cvtm_a3_H8_8a.json`, `cvtm_a3_H8_8a_rerun.json`, `cvtm_a3_H8_8b.json`,
`cvtm_a3_H8_8c_rerun.json`, `cvtm_a3_H8_9_yield.json`, `cvtm_copy_selector_H9_verdicts.json`,
`cvtm_early_logical_attention.json`, `cvtm_pilot_p1_arm1_seed100.json`, `ssl_concepts_early.json`,
`ssl_r1_confirm.json`, `ssl_r1_controls.json`, `ssl_r1_pretext.json`, `ssl_r1_pretext_smoke.json`,
`ssl_r1_speedup_verify.json`, `ssl_r2_hao_diagnostics.json`, `ssl_r4_controls.json`,
`ssl_transfer_early_classify_r2.json` and `ssl_transfer_early_report.json`.

Seven of them, the files named `cvtm_a3_H8_*.json`, hold a block `cvtm_sources_sha256` that pins the
SHA-256 of nine source files under `src/tmlm/cvtm/`. Three of those source files (`__init__.py`,
`doc_clauses.py` and `mixture.py`) were reworded in a docstring or a comment for publication, so
their pin values in the seven configs were replaced by the hashes of the published files. The
[hash table](../../../../../docs/hash-table.md) at the repository root gives each of these values as
registered and as published. The other pins of these configs (`core_file_sha256`,
`cache_meta_sha256`) refer to files inside run folders and are unchanged.

A consequence of the edits: the 23 published files no longer have the config hashes that the run
records of their runs hold. The two scripts that compare a config with a hash stored in a run folder
(`experiments/ssl_r4_controls.py` and `experiments/cvtm_transformer_tuned.py`) read
`ssl_r4_images.json`, `H8_5_energy.json` and `H9_7_tuned_pilot.json`, which were not edited, so
these checks are not affected.
