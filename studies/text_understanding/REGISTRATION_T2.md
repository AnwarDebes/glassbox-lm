# Registration T2: confirmatory test-split run for existsT bits

Written 2026-10-17T10:29:41Z, before any post hoc TM row (results_posthoc.jsonl) was seen; the six waiting jobs were still
running at that time. Status: registered, not run. No model in studies/text_understanding/ has been evaluated on a test
split. The test split is used once, only if the gate below passes.

## What is tested
Post hoc LR rows (posthoc_lr.jsonl, val) say that thermometer bits on the "some word is close to concept k"
similarities (existsT: still a table lookup plus a bitwise OR at run time) add information to the static mean
bits and alone come close to them. The mechanism is not new (Yadav, Jiao, Granmo, Goodwin, BlackboxNLP 2021).
The question is only whether this cheap variant helps a Tsetlin machine.

## Gate (fixed now)
Run the test only if the post hoc val rows (seed 1: P_*_mexT_s1 and P_*_exT_s1) show at least one of:
- mean+existsT at least mean + 0.3 points on 2 of 3 datasets, where mean is the average of the registered
  X_*_mean_s1 and _s2 rows: CLINC150 0.9220, Banking77 0.9073, HWU64 0.8862;
- existsT alone at least mean - 1.0 on 2 of 3 datasets.
If neither holds: no test run, "gate not passed" goes into the study log, and this line stops.

## Runs (only if the gate passes)
confirm_tm.py on CLINC150, Banking77, HWU64; views mean, mean+existsT, existsT, minilm; seeds 1, 2, 3: 36 runs.
TM settings unchanged from test 2: 4,000 clauses, T 4000, s 8, 40 epochs, batch 128, hardest negative sampling,
negation, init half, eval every 2. The epoch is chosen on val; test accuracy is read once, at that epoch. Every
run is reported. A run is repeated only after a crash, and the crash is logged.

## Bars (test accuracy, mean of seeds 1 to 3, same run for all views)
- H1, mean+existsT "works": at least +0.5 points over mean on 2 of 3 datasets and no dataset below -0.5.
  "Strong": closes at least half of the gap mean -> minilm on 2 of 3 datasets.
- H2, existsT alone "works": at least mean - 1.0 on 2 of 3 datasets. "Strong": at least mean on 2 of 3.
- No bar: the minilm row (reference), share of bits on, val rows.

## Frozen code (sha256, first 16 hex digits)
confirm_tm.py 3c8917b53ebe2927; exists_tm.py 7da0aee9f4eef796; exists2_tm.py f69af2d0c548834d (defines existsT);
decision-layer package (read only, not mine): dl/tm.py f649043f5801605b, dl/tm_init.py 557ff69706529738,
dl/boolean.py b5f5917f390da8a6, dl/data.py 0c4f8d22576e817c. Checked again before launch; any change is logged.

## Known limits (written now)
- One static table (potion-base-32M, revision 1e5a03f8), one K (512), one TM size.
- 40 epochs, although the exists views were still improving at epochs 38 to 40 in test 2; kept so that every
  view gets the same budget.
- Concepts are refit for this run by the same procedure (spherical k-means, seed 0, distinct training tokens);
  test texts are tokenized into the cache but do not enter any fit (thresholds are fit on train).
- Val accuracy is optimistic (best epoch on val), the same for all views.
- Speed is not measured again; speed_exists.json is rough (shared machine).
