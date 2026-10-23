# R16 registration: Granmo's hierarchical TM (AND-OR clauses) on the ABCD workflow

The goal's plan names a second way to rules over sequences: "or use a hierarchical TM". R16 tests the hierarchical Tsetlin machine of `cair/PyHierarchicalTsetlinMachineCUDA` (commit 5b4a223, the one the thesis study pins), in which a clause is a tree of AND and OR nodes, against the flat machine of the same package, on R9b's readable bits of the ABCD next-action task.

Frozen by `scripts/freeze.py` before any R16 test run. Eighth use of the ABCD test split. The package runs in this study's own environment `.venv-htm` (built 2026-10-17 04:42 UTC with the thesis study's recipe and versions; `env_htm_freeze.txt`); the thesis study's files are untouched.

## Input and machines (`scripts/abcd_htm.py`)

- Bits: R9b's short view (memory 66, subflow centres 335, next-button centres of the last 3 utterances 180, words of the last 3 utterances 1,000; `scripts/abcd_export_bits.py`), each of the 4 blocks zero-padded to F = 1,000 features. Negated literals on, 8 state bits, boost 1. One binary machine per button (the package's `MultiClassTsetlinMachine`: one-vs-rest with balanced negatives); `fit` on batches of 5,000 examples; the model after the last epoch is scored (no snapshot).
- **Hierarchical:** `((AND_GROUP, 1000), (OR_ALTERNATIVES, 8), (AND_GROUP, 4))`, the shape of the package's Hex demo: per block an OR of 8 options, each option a conjunction of that block's literals; the 4 blocks AND-ed. 400 clauses and T 1,000 per button, s 25, 30 epochs.
- **Flat:** `((AND_GROUP, 1000), (AND_GROUP, 4))`, an ordinary conjunctive clause, at its better dev setting: 200 clauses and T 200 per button, s 4, 30 epochs.

## Exploration behind it (dev, seed 1, `results/explore/abcd_htm.jsonl`)

| Run | Machine | Dev accuracy (epochs) |
|---|---|---|
| e35_hier_s8 | AND-OR with 2 OR-groups of 4 options per block, 200 clauses, s 8 | 0.145, 0.142, 0.296, 0.147, 0.165 (5) |
| e38_hex_a8_e40 | **hierarchical (Hex shape), A 8, 400 clauses, T 1,000, s 25** | 0.606 after 1, 0.795 after 6, 0.805 mean of 11-20, **0.8157 after 40** (best 0.8238 after 30) |
| e38_hex_a16_e40 | the same with A 16 | 0.576 to 0.631 (6) |
| e38_flat_e40 | flat with the hierarchical run's settings (s 25, T 1,000, 400 clauses) | 0.248 to 0.458 (6) |
| e38_flat_s4_e40 | **flat, 200 clauses, T 200, s 4** | 0.7158 mean of 11-20, **0.7153 after 40** (best 0.7218) |
| e39_cotm_curve | this study's coalesced flat TM on the same bits (R9b's setting) | 0.798 after 4, 0.828 after 20, 0.8398 after 40 |

Inside the package the hierarchy is about 10 points ahead of the flat machine; this study's coalesced TM is about 2.4 points ahead of both, and 37 times faster (116 s against 4,297 s for 40 epochs).

## Arms (test)

Hierarchical and flat as above, seeds 1, 2, 3 each, 30 epochs.

## Statistic

Per test action turn the mean correctness over seeds; differences with a bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`).

## Pass rule (test)

- **R16.1 (the AND-OR hierarchy helps):** hierarchical minus flat is at least +3.0 points, with the interval above 0. R16 is met if R16.1 holds.

## Reported with no rule

Hierarchical minus this study's coalesced TM (R9b's TM-mem-short on the same bits; on dev it is about 2.4 points below); run times; examples of hierarchical clauses.
