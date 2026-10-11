# Pre-registration: M5 step 1. Does the s rule close part of Clause-Driven-LLM's lexical gap?

- **Written:** 2026-10-02 at 19:45 UTC, **before any run**.
- **Publication note:** the wording of this copy was edited for publication on 2026-10-10. The registered question, settings, gates, grid, rules, amendment and times are unchanged. The copy includes the amendment of 21:11 UTC, which was written after the working notes on the result (21:09 UTC); a version without the amendment was not retained.
- **Question:** the Clause-Driven-LLM router on CLINC150 scores 82.78% with lexical (word-presence) features at s = 8, against 94.2% with sentence embeddings. Under the s rule (s > 1/p), s = 8 admits only words present in at least about 12.5% of a class's Type I examples. Does a higher s, which admits rarer paraphrase words, close part of that gap?

## Setup (the reference snapshot references/code/Clause-Driven-LLM is never modified)
- **Work copy:** `work/Clause-Driven-LLM/`.
  - Rebuilt module `src/cdllm/data/` with `load_clinc150(path)`, `load_banking77(...)` and `load_dataset(name, data_dir)`, to the interface in its tests/test_data.py and scripts. It was missing upstream because of `.gitignore` `data/`.
  - Data: `data_full.json` from the URL in its scripts/download_data.py (clinc/oos-eval). sha256 is recorded.
- **Environment:** `.venv-cdllm` (uv), with the package versions recorded. CPU only, numba backend, `nice -n 19`, 2 threads (NUMBA/OMP/MKL). Runs are sequential and logged with src/runlog.py.
- **Locked settings,** from the run_sweep.py lexical ablation of Clause-Driven-LLM: featurizer lexical (presence of 1 to 2-grams, 2,000 features, min_df 2), n-bins 2, 4,000 clauses, T 4000, 256 states, 40 epochs, batch 128, hardest negative sampling.

## Gate 0: the loader is valid (otherwise stop and report; no s-test)
1. tests/test_data.py passes for CLINC150: 150 classes, 15,000 train, 4,500 test, 1,000 OOS test.
2. The reproduction run (s 8, seed 42) gives test accuracy **0.82778 ± 0.005**. The recorded value is 0.8277777777777777 (results/tables/headline.json), and an exact match is expected because the kernel is deterministic.

## The s-test
- **Grid:** s ∈ {8, 25, 50, 100} × seeds {42, 123, 456}, so 12 runs. The s = 8 / seed 42 run is the Gate-0 run, not repeated.
- **Validation metric:** accuracy on the in-scope **val** split (3,000). It is added by a minimal, recorded patch that does not change training.
- **Choice:** s* = the s with the highest mean val accuracy over the 3 seeds. Ties go to the lower s.
- **Test split:** each of the 12 runs is scored once. The test curve over s is reported as descriptive; **the claim uses s* only**.
- **Success:** mean test accuracy at s* ≥ **0.848** (82.8% + 2.0 points), with every seed above 0.828.
- **Negative result:** s* = 8, or a gain below 2 points. Then the s rule does not explain the lexical gap here, and that is reported as such.
- **Also reported:** OOS AUROC; clause statistics (median and maximum literals); wall time. Timing is not compared with the machine of the recorded run.

## Not in scope
GPU; any change to the Clause-Driven-LLM repo.

## Amendment at 21:11 UTC, after Gate 0 and BEFORE any grid run (the grid has not run)

- **Gate 0 FAILED with `--backend numba`.**
  - Test accuracy 0.5407 against 0.82778; OOS AUROC 0.521 against 0.810.
  - Run results/20261002T195255Z_m5_gate0_s8_seed42_sorted; 57 min on 2 CPU threads.
- **The cause is the backend, not the loader.** Checked against the source in the Clause-Driven-LLM repo:
  - paper/report.tex:303 says "The GPU backend (used for the results here)";
  - run_sweep.py uses BACKEND="auto", which is torch on CUDA;
  - the numba kernel (coalesced.py, negative-class branch with weight ≤ 0) gives Type I feedback ONLY to clauses that fired (`if cj == 1`), with no erosion of non-firing clauses. torch_backend.py does erode them (`g1nf`, "Type I, not fired").
  - Consistent with this, the numba model of this study is degenerate: median 1,021 literals per clause, all negated. The recorded run had median 1.
- **Amended Gate 0:** the same locked settings with `--backend torch` on **GPU**. Pass only if test accuracy is within 0.82778 ± 0.005.
  - torch on CPU was measured at about 246 min per run, which is not workable.
- **GPU rule:** the runs start only after the hierarchical_tm thesis queue is EMPTY and has no unstarted jobs. They take the tm_language_model GPU lock (exclusive) and are capped at 15 min per run. The thesis always comes first.
- **Unchanged:** the grid, the choice of s* on val, the test scored once per run, and the success and negative rules.
