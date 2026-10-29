# REGISTRATION P1: the Model2Vec authors' low-data protocol, with the Tsetlin machine (written 2026-10-17T23:03:29Z)

Why: a separate search found a published baseline for the same static table under a fixed protocol, read here
at the source (MinishLab/model2vec, results/README.md, "Training Results"): potion-base-32M + scikit-learn
LogisticRegressionCV, "1000 examples from the train set, and the full test set", weighted F1, "No parameters were
tuned on any validation set": banking77 79.95, clinc_oos 68.11, massive_intent 73.83. No TM result under this
protocol is known. P1 asks: with this little data, does a TM on existsT bits (lookup + OR) match that recipe?

## Test reuse (declared)

The test splits used here were read before: Banking77 and CLINC150 (in-scope part) by T2 (and by the
decision-layer study), MASSIVE en-US by T3 part 1, T3 part 2 and T3b (and by decision-layer phase 2). The 1,000
CLINC out-of-scope test queries were not scored in this research line (the decision-layer study's use of them is
not checked here). Nothing in P1 is chosen on these test sets: every setting below is fixed before any P1 run.

## Protocol (fixed)

- Data. banking77: the official train split (10,003, file order), test 3,080, 77 classes. clinc_oos: the "plus"
  configuration (clinc/oos-eval data_oos_plus.json, the source of HF clinc_oos "plus"; sha256 bfcca9ae51562354...,
  in data_untrusted/clinc_oos_plus/; its in-scope train / val / test and oos val / test equal our data_full.json, only
  oos_train differs, 250 against 100): pool 15,000 in-scope + 250 oos, test 4,500 + 1,000 oos, 151 classes with oos
  as a class. massive_intent: MASSIVE 1.1 en-US train 11,514, test 2,974, 60 classes. The authors did not state the
  CLINC configuration, the sampling or the seed: "plus" and uniform sampling are this registration's reading.
- Draws: 5 (d = 1..5). 1,000 training examples drawn uniformly without replacement by numpy default_rng(d) from the
  pool. Every row of a draw uses the same 1,000 examples.
- Features, everything fitted on the 1,000 drawn texts only. Tokens: the potion-base-32M tokenizer (revision
  1e5a03f8), add_special_tokens=False. mean: the normalized average of the token vectors (= StaticModel.encode with
  the model's normalize=true; model2vec also drops [UNK] tokens, which this mean keeps; expected negligible).
  existsT: 512 concepts by spherical k-means (seed 0, 25 iterations) on the distinct tokens of the 1,000 texts; per
  concept the max cosine over the sentence's tokens; 8-level thermometer (7 quantile thresholds per concept over the
  1,000). mean bits: the same thermometer on the mean.
- Rows. lr_mean: scikit-learn 1.9.1 LogisticRegressionCV with default settings (10 Cs, 5-fold stratified CV, lbfgs,
  max_iter 100, L2) on the mean, i.e. the published model2vec + logreg recipe re-run here. lr_existsT: the same on
  the existsT bits. tm_existsT, tm_mean+existsT, tm_mean: the decision-layer CoTM (dl/tm.py, read only; 4,000
  clauses, T 4000, s 8, batch 128, hardest negative, negation, init half, seed d). Its epoch count is chosen on an
  inner hold-out (a random 20% of the 1,000, default_rng(1000 + d); at most 120 epochs, checked every 4 epochs,
  first best); then a FRESH machine with the same seed is trained on ALL 1,000 for that many epochs and its final
  state is scored. So the TM, like LR-CV, ends up trained on all 1,000 examples.
- Metric: weighted F1 (sklearn f1_score, average "weighted") on the full test set, in points; accuracy reported too.
- The TM settings come from the decision-layer study, where they were chosen on full-data val splits of CLINC150,
  Banking77 and HWU64; nothing is tuned for P1. LR-CV tunes its C by 5-fold CV inside the 1,000.

## Bars (mean over the 5 draws; "paired" = mean of the per-draw differences, in points)

- R0, reproduction check (not a hypothesis): lr_mean within 2.0 points of the published value (79.95 / 68.11 /
  73.83). Where it is outside, the published number is not used for that dataset in P1b-strong, and the gap is
  reported.
- P1a (the bits help a linear model with little data): lr_existsT - lr_mean > 0 on at least 2 of 3 datasets.
- P1b (headline: a TM on lookup + OR bits against the authors' recipe): tm_existsT - lr_mean > 0 on at least 2 of 3
  datasets. Strong: > 0 on 3 of 3, and the tm_existsT mean >= the published value on every dataset that passed R0.
- P1c (the best TM view of T2 and T3 part 1): tm_mean+existsT - lr_mean > 0 on at least 2 of 3 datasets.
- Reported, no bar: tm_existsT - lr_existsT (TM against LR on the same bits), tm_mean - lr_mean, per-draw values,
  sd over draws, chosen epochs, accuracy. With 5 draws there is no significance test; the report gives how many draws agree
  in sign.

## Run

p1_queue.sh: one job (at most 2 jobs of this line at once), datasets massive_intent, banking77, clinc_oos, draws 1-5, one
process per dataset and draw under gpu_run.sh shared, CPU threads 1; log runs_P1.log; rows results_p1.jsonl
(m2v_protocol.py); verdict by verdict_p1.py. Before it: a smoke run (--smoke: 200 drawn examples, 4 epochs, scored
on 500 undrawn training-pool examples; the test split is not read).
