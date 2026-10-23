# R16b registration: the hierarchical TM against a flat machine of matched size

Internal audit 7's main point about R16: the hierarchical machine (400 clauses per button, each clause 32 leaves) had 2 times the clauses and 16 times the automata of the flat machine it beat (200 clauses, 4 leaves each), so R16 did not show that the AND-OR structure, rather than size, made the 9.4 points. R16b compares R16's hierarchical runs with a flat machine of the same package that has as many automata: 3,200 clauses per button (3,200 x 4 leaves = 400 x 32).

**What was known at the freeze:** R16's test results (hierarchical 0.8174, flat 0.7234). Frozen by `scripts/freeze.py` before any R16b test run. Tenth use of the ABCD test split.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Exploration behind it (dev, seed 1, 30 epochs unless noted; `results/explore/abcd_htm.jsonl`)

| Run | Machine | Dev accuracy |
|---|---|---|
| e38_flat_s4_e40 | flat, 200 clauses, T 200, s 4 (R16's flat arm) | 0.716 (mean of epochs 31 to 40); 0.712 after 10 |
| e44_flat_c400_s4 | flat, 400 clauses, T 400, s 4 | 0.7234 after 10, 0.7326 after 30 |
| e44_flat_c800_s4 | flat, 800 clauses, T 800, s 4 | 0.7277 after 10, 0.7419 after 30 |
| **e44_flat_c3200_s4** | **flat, 3,200 clauses, T 3,200, s 4: the hierarchical machine's number of automata** | 0.7332 after 10, **0.7427 after 30** |
| R16's hierarchical runs (dev after epoch 30, seeds 1 to 3) | 400 clauses, Hex shape | 0.8238, 0.8187, 0.8217 |

On dev, more clauses lift the flat machine by about 2.7 points (200 to 800) and no further at 3,200; the hierarchical machine stays about 8 points ahead.

## Arms (test)

- **Hierarchical:** R16's 3 runs, unchanged.
- **Flat, matched automata:** `scripts/abcd_htm.py --model flat --s 4 --T 3200 --clauses 3200 --epochs 30 --eval-every 30`, seeds 1, 2, 3; the model after epoch 30 is scored.

Statistic as R16 (bootstrap over test conversations, 10,000 resamples, seed 0, 95%).

## Pass rule (test)

- **R16b.1 (the hierarchy, not the size):** hierarchical minus flat with matched automata is at least +3.0 points, with the interval above 0. R16b is met if R16b.1 holds.

## Reported with no rule

The flat machine with matched automata against R16's 200-clause flat machine (what size buys the flat machine); run times.
