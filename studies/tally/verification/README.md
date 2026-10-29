# Independent checks of T0

This folder holds the independent checks of the T0 verdict ([registration](../registrations/T0.md); results in [docs/results.md](../docs/results.md#verdict-computed-on-2026-10-25)), the outputs they recorded, and their test. None of the scripts imports the study's report code: `recompute_t0.py` was written from the registration text, and the other two read only the run records and the clause outputs that the final runs stored. They read run folders under `results/`, which are not distributed, so they run only where the runs are present. Every command is run from the study folder with the interpreter of `.venv-tm` of tm_language_model (numpy 2.4.6), with `CUDA_VISIBLE_DEVICES=` empty.

| File | What it is |
|---|---|
| `recompute_t0.py` | independent recompute of the verdict: it finds the 30 final runs, checks them (one folder per seed, finished, the registered settings, started after the amendment that fixed them, the source hashes of the amendment records, each logistic-regression run on the bank of its own seed, the same dev-test blocks in every run), recomputes every arm and the four paired differences, and bootstraps the 21 dev-test blocks with its own generator (PCG64, seed 20261025, 10,000 resamples), with a two-level interval (seeds, then blocks) beside it; with `--report` it compares its numbers with the report's `metrics.json` |
| `recompute_t0.json` | its recorded output (`--report results/20261025T104315Z_t0_report_devtest`): MATCH |
| `test_recompute_t0.py`, `test_recompute_t0.out` | its test on 30 made-up run folders (8 checks: the three readings, a missing seed, a crashed run, misaligned blocks, a logistic-regression run on another seed's bank, MATCH and MISMATCH against a report) and the recorded output |
| `explore_alpha_valid.py` | post hoc and exploratory, valid only, no rule: refits the counted read-out of the trained coalesced banks with alpha below the registered grid (down to 1e-5), from the stored clause outputs; at alpha 0.01 it reproduces each run's own counted valid bits |
| `explore_alpha_valid_seed1.json`, `explore_alpha_valid_seeds23.json` | its recorded outputs for seeds 1 to 3 |
| `unigram_place.py`, `unigram_place.out` | an add-one unigram counted on train, on calib and valid, beside the arms' means over seeds 1 to 10 on the same splits (placement only; dev-test not scored) |

Commands:

```bash
OMP_NUM_THREADS=1 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/test_recompute_t0.py
OMP_NUM_THREADS=1 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/recompute_t0.py . \
    --out <file> --report results/20261025T104315Z_t0_report_devtest
OMP_NUM_THREADS=2 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/explore_alpha_valid.py --seeds 1 2 3 --out <file>
OMP_NUM_THREADS=1 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/unigram_place.py
```

`recompute_t0.py` and `unigram_place.py` were rerun from this folder on 2026-10-25 and reproduced their recorded outputs byte for byte; `explore_alpha_valid.py` was rerun for seed 1 with the same result.
