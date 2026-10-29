# Results of the clause_lm study

The study has 29 run folders: nine of 2026-10-02, fourteen of 2026-10-14 and six of 2026-10-25 (UTC). None of them is flagged in the [run index](../../../docs/run-index.md), which holds no status for the run folders of this study because they have no `run.json`. Two of the 29 give no result: the retry `20261002T205134Z_m5_gate0_s8_seed42_first_appearance`, which was stopped with a terminating signal, and the first attempt at the amended Gate 0, `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu`, which ended at its start with exit status 127. Every measurement on this page is taken from a file of the run that is named next to it. Thresholds, the grid and the times of the registration are quoted from that document; the time of the working notes is the one they carry, and the approximate times in the protocol note on one grid run come from project notes. The run folders are not distributed, so the file names serve readers who hold them. Perplexities are rounded to two decimals, accuracies and AUC values to four; values that a registered rule compares are given in full. Every result here comes from a single run, except the means over three seeds of the s-test grid and the verdicts of the Gate 0 cause check, which their report runs compute. The values measured in the finish pass of 2026-10-25 come from test outputs and diagnostics that left no run folder; each is named next to its value, with its SHA-256 where a number depends on it, and kept with the records of that pass outside the repository.

The question of the study, its milestones M1 to M5 and its dependencies are described in the [README](../README.md).

## Status of every milestone and registered item

Twelve ids carry a status: the milestones of the study design, with M5 counted as its two steps, and six items of the registration of M5 step 1, [prereg_M5_step1.md](prereg_M5_step1.md). The statuses are met, not met, not judged and not run; for the s-test grid and the choice of s*, which are steps of the procedure, met means that the step was carried out as registered. Of the twelve ids, four are met, three are not met and five are not run. The study design also states, for M1, M2 and M3, what follows if the bar is missed; none of these rules has been applied, because none of the three tests has been run. The finish pass of 2026-10-25 changed no status; [its section](#status-after-the-finish-pass-of-2026-10-25) gives what each open milestone would cost and its chance. Its Gate 0 cause check, a frozen dev plan on the validation split, ended without a cause statement ([Gate 0 cause check of 2026-10-25](#gate-0-cause-check-of-2026-10-25)).

Milestones of the study design:

| Id | Content | Status | Qualifier | Run folders |
|---|---|---|---|---|
| M1 | Pair model on adjacent word pairs of Penn Treebank; it has to beat the statistical class baseline on unseen pairs, on 3 seeds | not run | preparation only: the pair data and the statistical baselines ran, the pair model was written and not run | `20261002T171456Z_pairs_data_K1000`, `20261002T171904Z_pair_baselines_K1000` |
| M2 | Clause-count language model with the OR-groups of M1; it has to beat the k-means class baseline, with the gain on unseen contexts | not run | preparation only: the evaluator ran once with k-means word groups and no Tsetlin Machine | `20261002T170451Z_ngram_baselines`, `20261002T174323Z_clause_lm_kmeans200` |
| M3 | Full clause language model | not run | no code, no run | none |
| M4 | Scale and efficiency | not run | no code, no run | none |
| M5 step 1 | Does a higher s close part of the gap between the lexical and the embedding router of Clause-Driven-LLM on CLINC150 (registered) | not met | negative result by the registered rules: s* = 8 and the success rule not met; it rests on the amended Gate 0, which is post hoc, and on 3 seeds per value of s; one grid run did not have the GPU to itself for about 40 s | the five `m5` run folders of 2026-10-02 and the fourteen of 2026-10-14 listed below |
| M5 step 2 | OR-group word concepts as router features | not run | no code, no run | none |

Registered items of M5 step 1:

| Id | Registered content | Status | Qualifier | Run folders |
|---|---|---|---|---|
| Gate 0 | The loader is valid. Condition 1: the upstream loader test passes for CLINC150 (150 classes; 15,000 train; 4,500 test; 1,000 out-of-scope test). Condition 2: the reproduction run (s 8, seed 42) gives a test accuracy within 0.005 of 0.82778 | not met | one completed run, numba CPU backend, seed 42; cause not confirmed; a retry was stopped without a score; the amendment of 21:11 UTC replaced the backend | `20261002T195152Z_m5_setup_sorted`, `20261002T195255Z_m5_gate0_s8_seed42_sorted`; retry without a score: `20261002T205115Z_m5_setup_first_appearance`, `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` |
| Amended Gate 0 | Amendment of 21:11 UTC: the same locked settings with the torch backend on a GPU; pass only within 0.005 of 0.82778 | met | post hoc: the amendment was written after the Gate 0 result was known; one run on 2026-10-14, seed 42, test accuracy 0.82933, inside the band but not the exact match that the registration expected; a first attempt crashed at its start without a score | `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu`; crashed attempt: `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu` |
| s-test grid | s in {8, 25, 50, 100} x seeds {42, 123, 456}, 12 runs including the Gate 0 run | met | all 12 runs completed with the torch backend on a GPU, each within the limit of 15 minutes; run s 8, seed 456 did not have the GPU to itself for about 40 s (protocol note below) | the amended Gate 0 run and the eleven grid run folders of 2026-10-14 listed below |
| Choice of s* | The s with the highest mean validation accuracy over the 3 seeds; ties go to the lower s | met | s* = 8, with no tie | `20261014T222730Z_m5_summary_sorted_torchgpu` |
| Success rule | Mean test accuracy at s* >= 0.848, with every seed above 0.828 | not met | mean test accuracy at s* 0.8201; seeds 123 and 456 below 0.828 | `20261014T222730Z_m5_summary_sorted_torchgpu` |
| Negative-result rule | s* = 8, or a gain below 2 points | met | s* = 8; the gain over 0.828 is -0.79 points | `20261014T222730Z_m5_summary_sorted_torchgpu` |

Gate 0 is one id with two numbered conditions. Condition 1 is recorded as passed in the setup run, condition 2 failed in the one completed run, and the gate needs both. The amended Gate 0 keeps condition 1 and moves condition 2 to the torch backend on a GPU. Its runs made no new setup run: they record the same SHA-256 of the CLINC150 file and of the lexical feature file as the setup run `20261002T195152Z_m5_setup_sorted` (`data_hashes.json`), so condition 1 rests on that setup run. The remaining lines of the registration (the validation metric, the rule that each run is scored once on the test split, the quantities reported besides accuracy, the scope line, and the GPU rule of the amendment with its limit of 15 minutes per run) describe the procedure and carry no status.

The registration was written on 2026-10-02 at 19:45 UTC, before the first run folder of M5, `20261002T195152Z_m5_setup_sorted` (19:51:52 UTC). Its amendment was written at 21:11 UTC, after the working notes on the result, which date from 21:09 UTC. The runs under the amendment date from 2026-10-14: the amended Gate 0 and the grid ran from 22:14:39 to 22:27:29 UTC, and the report run that applies the registered rules followed at 22:27:30 UTC.

## Run folders

| Run id | What it is | What came out | File that holds it |
|---|---|---|---|
| `20261002T170451Z_ngram_baselines` | `src/ngram_baselines.py`: unigram, bigram and trigram baselines on Penn Treebank validation | perplexity 687.03, 201.48 and 161.72 | `metrics.json` |
| `20261002T171456Z_pairs_data_K1000` | `src/pairs_data.py`: builds the pair data | 1,158,614 training pairs and 93,766 validation pairs | `metrics.json` |
| `20261002T171904Z_pair_baselines_K1000` | `src/pair_baselines.py`: statistical baselines on the pair task | AUC on unseen pairs from 0.5000 (word bigram count) to 0.7239 (class bigram, k = 200) | `metrics.json` |
| `20261002T174323Z_clause_lm_kmeans200` | `src/clause_lm.py` with k-means word groups; no Tsetlin Machine | perplexity 161.72 for the trigram model alone and 157.24 with clause votes | `metrics.json` |
| `20261002T195152Z_m5_setup_sorted` | `src/m5_setup.py`: setup record of M5 step 1, sorted label order | loader test: 1 passed, 1 skipped; lexical features built; training script in its first patched state | `metrics.json`, `pytest_test_data.txt`, `precompute_stdout.txt`, `vocab_cutoff.json`, `pip_freeze.txt`, `train_controller.patch` |
| `20261002T195255Z_m5_gate0_s8_seed42_sorted` | `src/m5_run.py`: Gate 0 with the numba CPU backend, s 8, seed 42 | test accuracy 0.5407 against a required 0.8278 within 0.005: Gate 0 not met | `metrics.json`, `train_metrics.json` |
| `20261002T205115Z_m5_setup_first_appearance` | `src/m5_setup.py`: setup record for a retry with the labels in order of first appearance | features rebuilt with the other label order; loader test again 1 passed, 1 skipped | `metrics.json`, `pytest_test_data.txt` |
| `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` | `src/m5_run.py`: retry of Gate 0 with the other label order | stopped at 21:02:41 UTC before any evaluation; no score | `metrics.json` |
| `20261002T210555Z_m5_setup_torchcpu` | `src/m5_setup_torch.py`: setup and timing probe for the torch backend on the CPU | estimated 246.0 minutes per run, above the limit of 45 minutes; training script in its second patched state | `metrics.json`, `timing_probe.txt`, `pip_freeze.txt`, `train_controller.patch` |
| `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu` | `src/m5_run.py`: first attempt at the amended Gate 0, torch backend on the GPU, s 8, seed 42 | ended at its start with exit status 127, because no `python3` was found to start the memory guard; no training, no score | `metrics.json`, `stderr.txt` |
| `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu` | `src/m5_run.py`: amended Gate 0, torch backend on the GPU, s 8, seed 42, started as a rerun after the crash | test accuracy 0.8293 against a required 0.8278 within 0.005: amended Gate 0 met | `metrics.json`, `train_metrics.json` |
| `20261014T221627Z_m5_s8_seed123_sorted_torchgpu` | `src/m5_run.py`: grid run, s 8, seed 123 | validation accuracy 0.8130, test accuracy 0.8064 | `metrics.json` |
| `20261014T221728Z_m5_s8_seed456_sorted_torchgpu` | `src/m5_run.py`: grid run, s 8, seed 456 | validation accuracy 0.8160, test accuracy 0.8244; did not have the GPU to itself for about 40 s ([protocol note](#amended-gate-0-and-the-grid)) | `metrics.json` |
| `20261014T221854Z_m5_s25_seed42_sorted_torchgpu` | `src/m5_run.py`: grid run, s 25, seed 42 | validation accuracy 0.7777, test accuracy 0.7824 | `metrics.json` |
| `20261014T221959Z_m5_s25_seed123_sorted_torchgpu` | `src/m5_run.py`: grid run, s 25, seed 123 | validation accuracy 0.7740, test accuracy 0.7864 | `metrics.json` |
| `20261014T222054Z_m5_s25_seed456_sorted_torchgpu` | `src/m5_run.py`: grid run, s 25, seed 456 | validation accuracy 0.7777, test accuracy 0.7864 | `metrics.json` |
| `20261014T222148Z_m5_s50_seed42_sorted_torchgpu` | `src/m5_run.py`: grid run, s 50, seed 42 | validation accuracy 0.7337, test accuracy 0.7649 | `metrics.json` |
| `20261014T222240Z_m5_s50_seed123_sorted_torchgpu` | `src/m5_run.py`: grid run, s 50, seed 123 | validation accuracy 0.7417, test accuracy 0.7438 | `metrics.json` |
| `20261014T222338Z_m5_s50_seed456_sorted_torchgpu` | `src/m5_run.py`: grid run, s 50, seed 456 | validation accuracy 0.7430, test accuracy 0.7493 | `metrics.json` |
| `20261014T222435Z_m5_s100_seed42_sorted_torchgpu` | `src/m5_run.py`: grid run, s 100, seed 42 | validation accuracy 0.7233, test accuracy 0.7293 | `metrics.json` |
| `20261014T222536Z_m5_s100_seed123_sorted_torchgpu` | `src/m5_run.py`: grid run, s 100, seed 123 | validation accuracy 0.7170, test accuracy 0.7311 | `metrics.json` |
| `20261014T222635Z_m5_s100_seed456_sorted_torchgpu` | `src/m5_run.py`: grid run, s 100, seed 456 | validation accuracy 0.7207, test accuracy 0.7218 | `metrics.json` |
| `20261014T222730Z_m5_summary_sorted_torchgpu` | `src/m5_report.py`: aggregation of the 12 runs and the registered rules | s* = 8; success rule not met, negative-result rule met, verdict NEGATIVE; the crashed attempt is listed as crashed and not counted | `metrics.json`, `tables.md` |
| `20261025T130238Z_gate0_cause_torch_noneg_s8_seed7` | `src/gate0_cause.py`: smoke run of the Gate 0 cause check, torch without the negative-class erosion, 2 epochs, seed 7; not counted | validation accuracy 0.0943 | `metrics.json` |
| `20261025T130431Z_gate0_cause_torch_s8_seed42` | `src/gate0_cause.py`: arm A0, the torch kernel unchanged, locked Gate 0 settings, seed 42 (validation only) | validation accuracy 0.8186666666666667, the recorded value; model byte-identical to that of 2026-10-14 | `metrics.json`, `tm.npz` |
| `20261025T130626Z_gate0_cause_torch_noneg_s8_seed42` | `src/gate0_cause.py`: arm A, torch without the negative-class erosion, seed 42 | validation accuracy 0.7753; 3,207 clauses with a positive literal, median 1 literal | `metrics.json`, `tm.npz` |
| `20261025T130751Z_gate0_cause_torch_noneg_s8_seed123` | `src/gate0_cause.py`: arm A, seed 123 | validation accuracy 0.7717; 3,225 clauses with a positive literal, median 1 literal | `metrics.json`, `tm.npz` |
| `20261025T130918Z_gate0_cause_torch_noneg_s8_seed456` | `src/gate0_cause.py`: arm A, seed 456 | validation accuracy 0.7747; 3,223 clauses with a positive literal, median 1 literal | `metrics.json`, `tm.npz` |
| `20261025T131008Z_gate0_cause_report` | `src/gate0_cause_report.py`: the rules of the Gate 0 cause plan | driver check met, arm A mixed, arm B not run: outcome row 2, no cause statement ([section](#gate-0-cause-check-of-2026-10-25)) | `report.md`, `metrics.json` |

No document of the study cited the run folder `20261002T205115Z_m5_setup_first_appearance` before this page.

## Penn Treebank: baselines and the evaluator

### N-gram baselines

Run `20261002T170451Z_ngram_baselines` (`src/ngram_baselines.py`). The data are the word-level Penn Treebank text with 10,000 word types, 929,589 training tokens and 73,760 validation tokens, the end-of-sentence token included (`config.json`). The metrics are computed over every token of the validation stream. Top-1 and top-5 accuracy are the shares of tokens whose true next word is the most probable word, or among the five most probable words, over the full vocabulary (`metrics.json`).

| Model | Perplexity | Top-1 accuracy | Top-5 accuracy |
|---|---|---|---|
| Unigram | 687.03 | 0.0559 | 0.2089 |
| Bigram, interpolated Kneser-Ney | 201.48 | 0.1837 | 0.3788 |
| Trigram, interpolated Kneser-Ney | 161.72 | 0.2249 | 0.4174 |

### Pair data and pair baselines

Run `20261002T171456Z_pairs_data_K1000` (`src/pairs_data.py`) built the pair data from the 1000 most frequent training words: 1,158,614 training pairs and 93,766 validation pairs, positives and sampled negatives together. Of the validation positives a share of 0.0681 does not occur as a pair in the training text, and of the validation negatives a share of 0.5067. In counts these are 3,193 unseen positives and 23,757 unseen negatives (`metrics.json`). The README describes how the pairs are made.

Run `20261002T171904Z_pair_baselines_K1000` (`src/pair_baselines.py`) scored five statistical models on the validation pairs. All are fitted on the training text only. The settings are a context vocabulary of the 2000 most frequent words, 100 SVD dimensions, k-means with 50, 100 and 200 classes, and 20 neighbours for similarity smoothing (`config.json`). The values are AUC (`metrics.json`).

| Model | All pairs | Seen pairs | Unseen pairs |
|---|---|---|---|
| Word bigram count | 0.8164 | 0.7111 | 0.5000 |
| Class bigram, k = 50 | 0.7904 | 0.7396 | 0.7024 |
| Class bigram, k = 100 | 0.8158 | 0.7641 | 0.7075 |
| Class bigram, k = 200 | 0.8262 | 0.7712 | 0.7239 |
| Similarity smoothing, 20 neighbours | 0.7696 | 0.6830 | 0.6497 |

The count model is at 0.5000 on unseen pairs by construction, because every unseen pair has the same count. The highest value on unseen pairs, 0.7239 for the class bigram model with k = 200, is the bar that the design of the study sets for the pair model of M1. The pair model has not been run, so nothing can be said about it.

This bar cannot be repeated exactly. `ppmi_svd` of `src/pair_baselines.py` called `scipy.sparse.linalg.svds` without a start vector, and scipy then draws one from operating-system entropy, so the vectors and the k-means classes differ from call to call; since 2026-10-25 the start vector is seeded (section [Changes of 2026-10-25](#changes-of-2026-10-25)). With the seeds 0 to 4 the class bigram model with k = 200 gives an unseen AUC from 0.7223 to 0.7231, and k = 50 and k = 100 give 0.7012 to 0.7025 and 0.7074 to 0.7078; the count model and similarity smoothing do not change (0.5000 and 0.6497). These values come from a diagnostic run of the script on 2026-10-25 whose run folders were not kept; its output is `diag_pair_baselines_svd_seeds.out` (SHA-256 `b354252e6150cdd7eda5910170805a5622948f8dd61412f4440d4e5a9e34a17b`) in the records of that pass, outside the repository. The recorded 0.7239 lies above all five. A bar for a registered M1 test would be computed again with the seeded code and equally tuned.

### Clause evaluator with k-means word groups

Run `20261002T174323Z_clause_lm_kmeans200` (`src/clause_lm.py`, groups `kmeans:200`). No Tsetlin Machine is involved in this run. The clauses are word groups from k-means with 200 classes on vectors of the right contexts of the 1000 most frequent words; 157 groups with at least two words remain (`config.json`). Each group votes with a discounted count distribution over the next word, and the vote is mixed into the Kneser-Ney trigram model with a weight lambda. Lambda is chosen on one half of the validation sentences and scored on the other half, in both directions; the chosen values are 0.15 and 0.2. A clause fires for a share of 0.7851 of the validation tokens, and the trigram context was seen in training for a share of 0.7952 (`metrics.json`).

| Contexts | Tokens | Model | Perplexity | Top-1 accuracy | Top-5 accuracy |
|---|---|---|---|---|---|
| All | 73,760 | Trigram alone | 161.72 | 0.2249 | 0.4174 |
| All | 73,760 | Trigram with clause votes | 157.24 | 0.2253 | 0.4171 |
| Trigram context unseen | 15,108 | Trigram alone | 228.56 | 0.1761 | 0.3593 |
| Trigram context unseen | 15,108 | Trigram with clause votes | 223.77 | 0.1756 | 0.3586 |
| Trigram context seen | 58,652 | Trigram alone | 147.93 | 0.2374 | 0.4324 |
| Trigram context seen | 58,652 | Trigram with clause votes | 143.58 | 0.2381 | 0.4322 |

The trigram model alone gives the same perplexity as in the baseline run. The clause votes lower the perplexity in all three rows. Top-1 accuracy rises slightly overall and on seen contexts and falls slightly on unseen contexts; top-5 accuracy falls slightly in all three rows. This run is the bar that the design sets for M2: clause votes from Tsetlin Machine groups would have to do better than these k-means groups. No such run exists.

The k-means groups of this run come from the unseeded SVD described above, so the run is one draw. On 2026-10-25, `tests/test_clause_lm_ptb_valid.py` (output `final_test_clause_lm_ptb_valid.out`, SHA-256 `9698532b975a25c5d1f9686518e0ab53169b1dfb499c5b44e9eb9d91135e0589`) reproduced the trigram rows of this table exactly (perplexity 161.72056889455956, 228.56260831996894 and 147.9331110816283, and the same accuracies). Four unseeded calls of the k-means step on that day gave 158, 156, 155 and 155 groups (`diag_kmeans_repeat.out`, SHA-256 `63405f9947d2480c0a812def3339519f58cef8025a0b3e30256bf497612a7c51`). With the seeded SVD, seeds 0 to 4, the evaluation keeps 155 to 158 groups and gives perplexities with clause votes from 157.2348 to 157.2949 overall, 223.7657 to 223.8149 on unseen and 143.5736 to 143.6345 on seen trigram contexts; seed 4 gives the recorded 157 groups and, to four decimals, the recorded perplexities, and seed 0, the default of the code, gives 155 groups and 157.2348 (`diag_clause_lm_svd_seeds.out`, SHA-256 `136957fafd33b2afce00bd164ad961dc951c112768c996269d85fff16f0514c0`, a diagnostic whose run folders were not kept; both files are in the records of the pass, outside the repository). The bar thus moves by up to 0.06 in perplexity with the start vector alone.

## M5 step 1: reproduction of the Clause-Driven-LLM router on CLINC150

### Setup

Run `20261002T195152Z_m5_setup_sorted` (`src/m5_setup.py`) is the setup record. It holds the following.

- The CLINC150 file has 2,495,390 bytes and the SHA-256 `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`; it was downloaded on 2026-10-02 at 19:49:03 UTC (`metrics.json`, `config.json`).
- The output of the upstream loader test ends with 1 passed, 1 skipped. The skipped case is Banking77, which was not downloaded (`pytest_test_data.txt`).
- The upstream feature script reports 150 classes, 15,000 training, 3,000 validation and 4,500 test queries, and 100, 100 and 1,000 out-of-scope queries for training, validation and test (`precompute_stdout.txt`).
- The lexical feature file has 2,000 presence features per query and the SHA-256 `15a987add2b590d8498768045d465167c808d8cbf8afd483330e89330444d0fc` (`metrics.json`).
- A diagnostic of the feature vocabulary: 11,860 terms occur in at least 2 training queries; the document frequency at rank 2,000 is 13; 1,971 terms lie above that value and 120 terms are tied at it, of which 29 are selected (`vocab_cutoff.json`).
- The run environment has no visible GPU, 2 threads for numba, OpenMP, MKL and OpenBLAS, and nice level 19 (`config.json`). The package versions are in the README.

### Gate 0 as registered

Run `20261002T195255Z_m5_gate0_s8_seed42_sorted` (`src/m5_run.py`) is the completed Gate 0 run with the numba backend, and until 2026-10-14 it was the only completed training run of the study. Its settings are the locked settings of the registration: lexical features with 2 bins, 4,000 clauses, T 4000, 256 states, 40 epochs, batch size 128, hardest negative sampling, s 8.0, seed 42, and the numba backend on the CPU (`config.json`). The labels are in sorted order.

The registered rule is a test accuracy within 0.005 of 0.82778. The registration gives the value recorded in the upstream repository as 0.8277777777777777 and expected an exact match.

| Quantity | Value | Key in `metrics.json` |
|---|---|---|
| Test accuracy | 0.5406666666666666 | `test_accuracy` |
| Target and tolerance of the gate | 0.8277777777777777 and 0.005 | `gate0_target`, `gate0_tol` |
| Absolute difference from the target | 0.2871111111111111 | `gate0_abs_diff` |
| Gate passed | false | `gate0_pass` |
| Validation accuracy, best snapshot (epoch 20) | 0.5493333333333333 | `val_accuracy`, `best_epoch` |
| Test macro-F1 | 0.5259 | `test_macro_f1` |
| Out-of-scope AUROC | 0.5209 | `oos_auroc` |
| Out-of-scope recall at 95% precision | 0.0 | `oos_recall_at_95prec` |
| Selective accuracy at 80% coverage | 0.6611 | `selective_acc_at_80cov` |
| Area under the risk-coverage curve | 0.1827 | `aurc` |
| Clauses, and clauses that are not empty | 4,000 and 4,000 | `clause_stats` |
| Literals per clause: median, mean, maximum | 1,021, 1,021.9 and 1,411, out of 4,000 literals | `clause_stats`, `n_literals` |
| Training time | 3,398.5 s | `train_seconds` |
| Wall time of the process | 3,419.9 s, from 19:52:55 to 20:49:55 UTC | `process_wall_s`, `start_utc`, `end_utc` |

Gate 0 is not met. The registered consequence is to stop and report, with no s-test. The amendment of 21:11 UTC then moved the gate to the torch backend on a GPU ([amended Gate 0](#amended-gate-0-and-the-grid)).

The cause is not confirmed. The working notes of 2026-10-02, summarised [below](#diagnosis-notes-of-2026-10-02), attribute the failure to the training backend and not to the rebuilt loader: by their reading of the upstream code, the recorded upstream value came from the torch backend on a GPU, and the numba kernel treats clauses that did not fire differently from the torch kernel. The notes mark that reading as not yet confirmed by a torch run. The amended Gate 0 run of 2026-10-14, with the torch backend on a GPU, met the gate; that is consistent with the reading, and it does not test which part of the two kernels makes the difference. The run records support one narrower statement: in both Gate 0 run folders of 2026-10-02 the numba kernel file of the work copy is recorded as identical to the reference copy (`config.json`, key `coalesced_identical_to_reference`), so the kernel that ran was the unmodified kernel of the downloaded copy.

Four limits go with these numbers wherever they are used. They come from one run with one seed. The run used the numba CPU backend. The published `src/m5_run.py` is a later revision than the one that ran; the run folder holds the earlier revision. The commit of the upstream copy is recorded in project notes, not checked against upstream.

### Retry with the labels in order of first appearance

The registration does not fix the order of the class labels. After the Gate 0 result, the features were built again with the labels in order of first appearance in the training split, and Gate 0 was started a second time. Both steps are post hoc: the registered consequence of a failed Gate 0 is to stop and report, and neither the registration nor its amendment of 21:11 UTC provides for a retry.

Run `20261002T205115Z_m5_setup_first_appearance` is the setup record of that retry. Its `metrics.json` shows the same CLINC150 file, the same loader and the same state of the patched training script as the first setup run, feature arrays of the same shapes, the same vocabulary diagnostic, intent names that are not in sorted order, and another SHA-256 of the feature file, `0016bf15b7e079908668ba7baee7f83159b7927d8c76f2fefd21c03bfaa55fc1`.

Run `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` started at 20:51:34 UTC and was stopped at 21:02:41 UTC before any evaluation, so it has no score. Its `metrics.json` holds the return code -15, a process wall time of 667.3 s and no accuracy; its captured standard output is empty. The key `CRASH` in that file is set because `src/m5_run.py` sets it for every exit that is not zero. A note file in the same folder explains the stop.

### Timing probe for the torch backend on the CPU

Run `20261002T210555Z_m5_setup_torchcpu` (`src/m5_setup_torch.py`) records a diagnosis step that was decided at 21:02 UTC, after the Gate 0 result, and is therefore post hoc (the run folder dates from 21:05:55 UTC, before the amendment of 21:11 UTC was written): repeat Gate 0 with the torch backend of Clause-Driven-LLM on the CPU, with a limit of 45 minutes per run (`config.json`). torch 2.14.1+cpu was added to the environment, and the training script was brought to its second patched state. The torch kernel file of the work copy is recorded as identical to the reference copy (`metrics.json`, key `torch_backend_identical_to_reference`).

A timing probe on training rows only, with no validation or test data, took 6.62 s for 2 minibatches and 37.87 s for 12 minibatches, from which the script derives 3.12 s per minibatch, and 0.44 s for one pass over 3,000 rows (`timing_probe.txt`, `metrics.json`). Extrapolated to a run of 40 epochs this gives an estimate of 246.0 minutes. The estimate is an extrapolation from 12 minibatches and not a measured run time. It is above the limit, so Gate 0 was not run with the torch backend on the CPU. The setup run itself took 49.7 s.

### Amended Gate 0 and the grid

The amendment of 21:11 UTC keeps the locked settings and changes the backend of Gate 0 to torch on a GPU, with the same rule. It was written after the Gate 0 result was known, so the amended gate is post hoc. A GPU environment was prepared for it (freeze C in the README), and `src/m5_gpu_all.sh` is the launcher for it. According to the project notes the launcher was started on 2026-10-02 and again on 2026-10-03; no start left a run folder or a recorded outcome, and the records give no cause.

The amended Gate 0 and the grid ran on 2026-10-14. For these runs the two environments of the study were rebuilt from the recorded freezes A and C, and the package lists of the rebuilt environments are identical to the recorded ones (project notes); each training run records torch 2.6.0+cu124 (`config.json`, key `torch_version`). The runs were started by a launcher that runs the same steps as `src/m5_gpu_all.sh` (Gate 0, then, only if it passes, the grid through `src/m5_grid.sh` and the report through `src/m5_report.py`) without its wait on the job queue of hierarchical_tm. That study had finished on 2026-10-03, but the wait reads its local queue state files, which were not in the state that the wait expects, so the wait could not end. The launcher is `research/roadmap/run_gate0.sh` of the project folder, outside this repository (SHA-256 `32b61c95b32f9a7177f50e11cd0ca72dfaf435a7baf1feb0d743b102af7fcaa8`, last modified at 22:15:14 UTC, after the first attempt); it puts a link `python3` to a Python 3.11 interpreter first on the search path and starts Gate 0 with `--after-crash`. Its output is `research/roadmap/gate0.out` (SHA-256 `4d8c9cbf9118bb82460c7324a25352167ac2bd1e755e19b4dff89fb1bc43f89b`), and that of the first attempt `research/roadmap/gate0.attempt1_crash.out` (`cc359333e8e59b54c8405ba58c214f7bf3f288a9b87473bfc68feb18ae04e915`). `src/m5_gpu_all.sh` was retired on 2026-10-25. Every training run held the GPU lock in exclusive mode and had a limit of 15 minutes (`config.json`, key `cmd`). The copies of `src/m5_report.py` and `src/runlog.py` in these run folders are byte identical to the published files; the copies of `src/m5_run.py` have the code of the published file, whose docstring was rewritten on 2026-10-25 ([changes](#changes-of-2026-10-25)).

**First attempt.** Run `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu` ended at 22:14:39 UTC, the second it started, with exit status 127 and a process wall time of 0.0 s (`metrics.json`). Its `stderr.txt` holds one line, in which `/usr/bin/env` reports that it found no `python3`. The memory guard `studies/tm_language_model/scripts/mem_guard.py`, which `src/m5_run.py` puts in front of the training command, begins with `#!/usr/bin/env python3`, and the rebuilt machine had no `python3` on its search path. No training took place, and the run has no score. With a `python3` placed first on the search path, Gate 0 was started again with the option `--after-crash`, which `src/m5_run.py` provides for a rerun after a crash (`config.json`, key `after_crash`). The report run lists the first attempt as crashed and does not count it.

**Amended Gate 0.** Run `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu` (`src/m5_run.py`, backend `torch_gpu`) has the locked settings of the registration, s 8.0, seed 42 and the labels in sorted order (`config.json`).

| Quantity | Value | Key in `metrics.json` |
|---|---|---|
| Test accuracy | 0.8293333333333334 | `test_accuracy` |
| Target and tolerance of the gate | 0.8277777777777777 and 0.005 | `gate0_target`, `gate0_tol` |
| Absolute difference from the target | 0.0015555555555556433 | `gate0_abs_diff` |
| Gate passed | true | `gate0_pass` |
| Exact match with the target | false | `gate0_exact_match` |
| Validation accuracy, best snapshot (epoch 39) | 0.8186666666666667 | `val_accuracy`, `best_epoch` |
| Test macro-F1 | 0.8304 | `test_macro_f1` |
| Out-of-scope AUROC | 0.8000 | `oos_auroc` |
| Out-of-scope recall at 95% precision | 0.0 | `oos_recall_at_95prec` |
| Selective accuracy at 80% coverage | 0.9283 | `selective_acc_at_80cov` |
| Area under the risk-coverage curve | 0.0389 | `aurc` |
| Clauses, and clauses that are not empty | 4,000 and 3,908 | `clause_stats` |
| Literals per clause: median, mean, maximum | 1, 147.8 and 834, out of 4,000 literals | `clause_stats`, `n_literals` |
| Training time | 35.3 s | `train_seconds` |
| Wall time of the process | 57.1 s, from 22:15:27 to 22:16:24 UTC | `process_wall_s`, `start_utc`, `end_utc` |

The amended Gate 0 is met: the test accuracy lies 0.0016 above the target, inside the tolerance of 0.005. It is not the exact match that the registration expected. The gate is post hoc, and it rests on one run with one seed.

Against the numba run `20261002T195255Z_m5_gate0_s8_seed42_sorted`, the records show the same CLINC150 file and the same feature file (`data_hashes.json`), the same loader (`config.json`, key `data_module_sha256`), the same locked settings and the same seed. The two runs differ in the backend, in the state of the patched training script (the second state, which adds the `--device` argument) and in the environment. The test accuracy is 0.8293 against 0.5407, and the median clause has 1 literal against 1,021. This is consistent with the backend explanation of the [working notes](#diagnosis-notes-of-2026-10-02); the runs do not show which part of the two kernels makes the difference.

**The grid.** The eleven grid runs ran one after another from 22:16:27 to 22:27:29 UTC, each with return code 0 and a process wall time between 47.0 and 76.3 s (`metrics.json`). With the amended Gate 0 run they form the registered grid of 12 runs. Validation accuracy is on the in-scope validation split and test accuracy on the test split; each run folder holds one test score. The literals are counted per clause of the final model (`clause_stats`). All values are from `metrics.json` of the run named in the row.

| s | Seed | Validation accuracy | Test accuracy | OOS AUROC | Literals per clause: median and maximum | Wall time of the process (s) | Run id |
|---|---|---|---|---|---|---|---|
| 8 | 42 | 0.8187 | 0.8293 | 0.8000 | 1 and 834 | 57.1 | `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu` (the amended Gate 0 run) |
| 8 | 123 | 0.8130 | 0.8064 | 0.7823 | 1 and 878 | 56.7 | `20261014T221627Z_m5_s8_seed123_sorted_torchgpu` |
| 8 | 456 | 0.8160 | 0.8244 | 0.8038 | 1 and 842 | 76.3 | `20261014T221728Z_m5_s8_seed456_sorted_torchgpu` (protocol note below) |
| 25 | 42 | 0.7777 | 0.7824 | 0.7560 | 2 and 1,258 | 58.8 | `20261014T221854Z_m5_s25_seed42_sorted_torchgpu` |
| 25 | 123 | 0.7740 | 0.7864 | 0.7670 | 2 and 1,249 | 50.4 | `20261014T221959Z_m5_s25_seed123_sorted_torchgpu` |
| 25 | 456 | 0.7777 | 0.7864 | 0.7657 | 2 and 1,281 | 47.9 | `20261014T222054Z_m5_s25_seed456_sorted_torchgpu` |
| 50 | 42 | 0.7337 | 0.7649 | 0.7661 | 3 and 1,525 | 47.7 | `20261014T222148Z_m5_s50_seed42_sorted_torchgpu` |
| 50 | 123 | 0.7417 | 0.7438 | 0.7660 | 3 and 1,490 | 47.4 | `20261014T222240Z_m5_s50_seed123_sorted_torchgpu` |
| 50 | 456 | 0.7430 | 0.7493 | 0.7671 | 3 and 1,505 | 48.2 | `20261014T222338Z_m5_s50_seed456_sorted_torchgpu` |
| 100 | 42 | 0.7233 | 0.7293 | 0.7810 | 4 and 1,846 | 52.2 | `20261014T222435Z_m5_s100_seed42_sorted_torchgpu` |
| 100 | 123 | 0.7170 | 0.7311 | 0.7761 | 5 and 1,824 | 47.0 | `20261014T222536Z_m5_s100_seed123_sorted_torchgpu` |
| 100 | 456 | 0.7207 | 0.7218 | 0.7704 | 5 and 1,863 | 54.1 | `20261014T222635Z_m5_s100_seed456_sorted_torchgpu` |

Protocol note on run `20261014T221728Z_m5_s8_seed456_sorted_torchgpu`: from about 22:17:30 to 22:18:10 UTC a short GPU test unrelated to this study ran without the GPU lock, while this run, from 22:17:28 to 22:18:45 UTC, held the exclusive lock. The run therefore did not have the GPU to itself, as the amendment requires. Its wall time, 76.3 s, the longest of the grid, may be inflated. Its accuracy is computed by the same kernels as in the other runs. The times of the overlap are approximate and come from project notes.

At s = 8 the seed 123 run scores 0.8064 on the test split, below the band of Gate 0 (0.82278 to 0.83278). Gate 0 was registered on seed 42 only, so the gate does not depend on this run; the three seeds at the gate setting span 0.8064 to 0.8293.

Run `20261014T222730Z_m5_summary_sorted_torchgpu` (`src/m5_report.py`) aggregates the 12 runs by s and applies the registered rules. Its means over the three seeds, rounded (`metrics.json`, key `per_s`; also `tables.md`):

| s | Mean validation accuracy | Mean test accuracy | Test accuracy by seed (42, 123, 456) | Mean OOS AUROC | Mean of the median and of the maximum literals per clause |
|---|---|---|---|---|---|
| 8 | 0.8159 | 0.8201 | 0.8293, 0.8064, 0.8244 | 0.7954 | 1 and 851.3 |
| 25 | 0.7764 | 0.7851 | 0.7824, 0.7864, 0.7864 | 0.7629 | 2 and 1,262.7 |
| 50 | 0.7394 | 0.7527 | 0.7649, 0.7438, 0.7493 | 0.7664 | 3 and 1,506.7 |
| 100 | 0.7203 | 0.7274 | 0.7293, 0.7311, 0.7218 | 0.7758 | 4.67 and 1,844.3 |

The values that the registered rules compare, in full:

| Quantity | Value | Key in `metrics.json` |
|---|---|---|
| Completed runs counted | 12 | `n_completed_runs` |
| Runs listed as problems | the crashed first attempt `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu` | `problems` |
| s* | 8.0 | `s_star` |
| Values of s tied at the highest mean validation accuracy | 8 only | `ties_at_best_val` |
| Mean validation accuracy at s* | 0.8158888888888889 | `s_star_mean_val` |
| Mean test accuracy at s* | 0.8200740740740741 | `s_star_mean_test` |
| Test accuracy at s* by seed (42, 123, 456) | 0.8293333333333334, 0.8064444444444444 and 0.8244444444444444 | `s_star_test_by_seed` |
| Gain of the mean test accuracy at s* over 0.828, in points | -0.7925925925925892 | `gain_vs_0.828_points` |
| Success rule met | false | `success_rule_met` |
| Negative-result rule met | true | `negative_rule_met` |
| Verdict | NEGATIVE | `verdict` |

s* is 8, because the mean validation accuracy is highest at s = 8 and lower at each larger s. The success rule needs a mean test accuracy at s* of at least 0.848 with every seed above 0.828; the mean is 0.8201, and two of the three seeds, 123 and 456, are below 0.828. The negative-result rule is met by its first condition, s* = 8, and the gain is also below 2 points. The verdict by the registered rules is negative. In the words of the registration, the s rule does not explain the lexical gap here: on CLINC150 with the locked lexical settings, 3 seeds per value of s, and a Gate 0 that is post hoc.

The test curve over s is descriptive, as registered; the claim uses s* only. The mean test accuracy falls from 0.8201 at s = 8 to 0.7851, 0.7527 and 0.7274 at s = 25, 50 and 100, while the median and the maximum number of literals per clause grow with s. The mean OOS AUROC is highest at s = 8 and does not change monotonically with s. Timing is not compared with the machine of the recorded upstream run, as the registration states.

### Recorded states of the patched training script

The run records hold the SHA-256 of the patched `scripts/train_controller.py` of the work copy. Two values occur.

| State | Change against the upstream file | SHA-256 of the patched file | Run records that hold it | Published patch |
|---|---|---|---|---|
| First | 2 lines added: validation accuracy of the final model | `d6214e313601fcede90e7b04c1dee444412c25141d85272620e03968b6d77e80` | `metrics.json` of `20261002T195152Z_m5_setup_sorted` and of `20261002T205115Z_m5_setup_first_appearance`; `config.json` of `20261002T195255Z_m5_gate0_s8_seed42_sorted` and of `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` | `src/clause_driven_llm/train_controller.state1.patch` |
| Second | 4 lines added and 1 removed in total: the first state and a `--device` argument | `e999bfb14af47181fd910927c74ed09f5c6b43cff880a6f38300b9b38a0068b2` | `metrics.json` of `20261002T210555Z_m5_setup_torchcpu`; `config.json` of the 13 training run folders of 2026-10-14, from `20261014T221439Z_m5_gate0_s8_seed42_sorted_torchgpu` to `20261014T222635Z_m5_s100_seed456_sorted_torchgpu` | `src/clause_driven_llm/train_controller.patch` |

The completed Gate 0 run of 2026-10-02 and the stopped retry both used the first state. The amended Gate 0 run and the grid of 2026-10-14 used the second state, and so did the crashed first attempt. The SHA-256 of the added loader, `b851309171cb4d6d2615a84b90437a9c16a7f5d8e8812f62dfa11579b2a07e5e`, is held by the same four run records as the first state and by the `config.json` of the 13 training run folders of 2026-10-14 (key `data_module_sha256`), and the published file `src/clause_driven_llm/cdllm_data__init__.py` is that loader. The [hash table](../../../docs/hash-table.md) lists the pinned files of the repository.

## The pair model of M1

`src/pair_tm.py` was written and not run. A smoke test of it was queued on 2026-10-02 and cancelled before it ran; its output file `results/smoke.out` (11 bytes, written at 17:17 UTC on that day) holds the one word `Terminated`. No run folder exists for the pair model, and it has no result. On 2026-10-25 its evaluation was changed before any run (section [Changes of 2026-10-25](#changes-of-2026-10-25)) and tested on the CPU against a stand-in for the GPU package; it still has no run on Penn Treebank.

## Limits of these records

The four Penn Treebank runs record no environment, so their interpreter and package versions are not known from the run files. No run folder of the study records a commit of this repository; each holds a copy of the script that ran and of `src/runlog.py`. The keys of `data_hashes.json` in the nine run folders of 2026-10-02 name the data files by the folder layout at the time of the runs, which differs from the layout of this repository; the run folders of 2026-10-14 use keys that begin with `clause_lm/`. Apart from the s-test grid, which has three seeds for each value of s, no result on this page was repeated with a second seed.

## Diagnosis notes of 2026-10-02

The working notes on M5 step 1 were written on 2026-10-02 at 21:09 UTC and are labelled there as reported, not re-checked. Their findings are kept here because they say what a rerun has to test. They rest on files outside the run folders (the downloaded copy of Clause-Driven-LLM and the saved model of the Gate 0 run). The figures of the saved models were checked again on 2026-10-25 (the bullet on the saved model below); the other figures of this section were not checked again.

- **Values recorded upstream.** `results/tables/headline.json` of Clause-Driven-LLM gives, for the run that the gate targets: test accuracy 0.8277777777777777, test macro-F1 0.8300, out-of-scope AUROC 0.8104, selective accuracy at 80% coverage 0.9225, area under the risk-coverage curve 0.0410, 3,919 non-empty clauses, literals per clause with median 1, mean 151.4 and maximum 787, and a training time of 48.9 s.
- **Which backend produced them.** The paper sources and scripts of Clause-Driven-LLM name the GPU backend for its results (`paper/app_hparams.tex`, `paper/report.tex` line 303, `scripts/run_all_seeds.sh`, and `BACKEND = "auto"` in `scripts/run_sweep.py`, which picks torch when CUDA is available). The registration expected an exact match because the kernel is deterministic; that expectation does not hold across backends.
- **The two backends differ.** By a reading of the code, the torch backend (`torch_backend.py`, lines 131 and 154) applies Type I erosion, with probability 1/s per literal, to clauses that did not fire, for the sampled negative class as well whenever the clause weight for it is negative. The numba kernel (`coalesced.py`, lines 158 to 188) gives a non-firing clause no feedback for the negative class. The docstring of `coalesced.py` says that both backends train the same automaton state and clause weights.
- **The saved Gate 0 model.** All 4,000 clauses of the model contain only negated literals; no clause has a positive literal, and the median clause has 1,021 negated literals. Checked again on 2026-10-25 from the file (`work/m5/runs/sorted/s8_seed42/clinc150__lexical__seed42/tm.npz`, SHA-256 `f1183ee36a2a8568564ca4ce60d3dbd48f99fb71dadfc2d63a53027be495191e`; `tests/test_gate0_cause.py`, part D, output `final_test_gate0_cause.out`, SHA-256 `7c6081343eef765b5ab26aed9ea1ceede9cbaa93a0f7166416944e55e7b61296`): 0 of 4,000 clauses have a positive literal, every clause has negated literals, the median is 1,021 literals (mean 1,021.93, maximum 1,411), and 69.17% of the weight entries are negative and 2.79% zero. The saved model of the amended Gate 0 run (`work/m5/runs/sorted_torchgpu/s8_seed42/clinc150__lexical__seed42/tm.npz`, `5e2478a4c0a3a984773901d8c0ee371bf4fa74d1b103f20dbccaf9235608d75e`) differs on every count: 2,954 of 4,000 clauses have a positive literal, 1,165 have negated literals, 3,908 are not empty, the median is 1 literal (mean 147.85, maximum 834), and 57.45% of the weight entries are negative and 0.74% zero. The inventory of the same day found the same counts with an independent reader.
- **Vocabulary ties.** The vocabulary of 2,000 features has ties at the cut-off: 120 terms share the document frequency 13, and 29 of them are selected by an unstable sort, so the exact vocabulary can differ between numpy versions and processors (`vocab_cutoff.json` of run `20261002T195152Z_m5_setup_sorted`).
- **The loader.** The added loader takes an optional label order (argument `label_order`, environment variable `CDLLM_LABEL_ORDER`, default "sorted"). Its Banking77 validation split is a reconstruction that was not verified and not used.
- **Measurements outside the run folders.** Two timing measurements left no run folder: a probe of the numba kernel on 6 minibatches of training data at about 20:26 UTC, while Gate 0 trained, and a micro-benchmark of the torch kernel on synthetic tensors at about 21:07 UTC. Neither read validation or test data.

Taken together the notes read the failed gate as a backend mismatch: with lexical features the numba path and the torch path of Clause-Driven-LLM would train different models. The test that the amendment registers, one Gate 0 run with the torch backend on a GPU, ran on 2026-10-14 and met the gate ([amended Gate 0](#amended-gate-0-and-the-grid)): test accuracy 0.8293, and clauses with a median of 1 literal. That result is consistent with the reading of the notes. It does not confirm the particular difference between the kernels that the notes describe. The check of 2026-10-25, which removed that difference from the torch kernel, ended without a cause statement ([Gate 0 cause check of 2026-10-25](#gate-0-cause-check-of-2026-10-25)).

A dev plan written on 2026-10-25 isolates it on the validation split, [plan_gate0_cause_2026-10-25.md](plan_gate0_cause_2026-10-25.md). Its driver, `src/gate0_cause.py`, trains the locked Gate 0 settings with the torch kernel unchanged (A0), the torch kernel without the erosion of non-firing clauses for the negative class (A, three seeds), and, optionally on the CPU, the numba kernel with that erosion added (B). Its checker, `src/gate0_cause_report.py`, counts a run only after the plan's freeze (`scripts/freeze.py`) and only when the driver, the upstream copy, the feature file and the environment have the pinned SHA-256, and prints one line of an ordered outcome table fixed in the plan. The plan was frozen and run on the same day; the [next section](#gate-0-cause-check-of-2026-10-25) gives the result. Both docstrings that call the two backends equivalent (`coalesced.py` lines 281-282 and `torch_backend.py` lines 4-16) are wrong by the code reading, whatever the check found; the upstream repository is outside this study.

## Gate 0 cause check of 2026-10-25

The dev plan [plan_gate0_cause_2026-10-25.md](plan_gate0_cause_2026-10-25.md) was frozen on 2026-10-25 at 12:59:23 UTC with `scripts/freeze.py` (record `plan_gate0_cause_2026-10-25.frozen.json`, SHA-256 `a6934386fb599bbc26092e902eaf3d480c6a9104ba684fb0f202c2055082be0c`; it holds the plan's SHA-256 `df091761ba70255933f2e97577cb5d3c6941cfa52c59ee4fba29d940f5a4be41` and 48 pinned files), after a review and three revisions (judged by an AI model, not a person). The rules, the signatures and the ordered outcome table were fixed before the first run. The runs used the GPU of the machine one at a time, from 13:02:06 to 13:09:58 UTC (the queue log of the finish pass, outside the repository). Only the validation split was used: every run folder records that the driver read `Xtrain`, `ytrain`, `Xval` and `yval` of the feature file and no other array. The kernel variants existed only in memory while a model trained; the upstream copies did not change.

| Arm | Training kernel | Seed | Run folder | Validation accuracy | Clauses with a positive literal (of 4,000) | Median literals per clause | Signature |
|---|---|---|---|---|---|---|---|
| Smoke, not counted (2 epochs) | torch without the negative-class erosion | 7 | `20261025T130238Z_gate0_cause_torch_noneg_s8_seed7` | 0.0943 | 3,598 | 15 | not counted |
| A0, the driver check | torch, unchanged | 42 | `20261025T130431Z_gate0_cause_torch_s8_seed42` | 0.8186666666666667 | 2,954 | 1 | healthy |
| A | torch without the negative-class erosion | 42 | `20261025T130626Z_gate0_cause_torch_noneg_s8_seed42` | 0.7753 | 3,207 | 1 | neither |
| A | torch without the negative-class erosion | 123 | `20261025T130751Z_gate0_cause_torch_noneg_s8_seed123` | 0.7717 | 3,225 | 1 | neither |
| A | torch without the negative-class erosion | 456 | `20261025T130918Z_gate0_cause_torch_noneg_s8_seed456` | 0.7747 | 3,223 | 1 | neither |
| B, optional | numba with the erosion added | 42 | none | not run | | | |

The signatures are the plan's. Degenerate: a validation accuracy of at most 0.70, at most 400 clauses with a positive literal and a median of at least 100 literals. Healthy: at least 0.78, at least 2,000 clauses with a positive literal and a median of at most 10 literals. Anything else is neither. The report run `20261025T131008Z_gate0_cause_report` applies the rules:

- Driver check: **met**. A0 gives 0.8186666666666667, exactly the recorded value of 2026-10-14, and its saved model is byte-identical to the model of the amended Gate 0 run (SHA-256 `5e2478a4c0a3a984773901d8c0ee371bf4fa74d1b103f20dbccaf9235608d75e`).
- Arm A: **mixed**. Each run is neither: its structure is like that of the torch models (3,207 to 3,225 clauses with a positive literal, a median of 1 literal), and its validation accuracy is below the 0.78 that healthy needs. None is degenerate like the numba model of 2026-10-02 (0.5493, no clause with a positive literal, a median of 1,021 literals).
- Arm B: **not run**. It was optional; with arm A mixed the outcome row is 2 whatever arm B shows, and for rows 1 and 2 the plan runs nothing further.
- **Outcome, row 2: every number is reported; there is no cause statement.**

The expectation written in the plan before any run, arm A degenerate near 0.55 on all three seeds, did not come true. Beside the rules, with no rule: arm A minus the unchanged torch kernel at the same seed (the runs of 2026-10-14, 0.8187, 0.8130 and 0.8160) is -4.33, -4.13 and -4.13 points. The diagnosis stays open: this check neither confirms nor excludes the missing negative-class branch, or any other difference between the two kernels, as the cause of the degenerate numba model. The registered verdict of M5 step 1 and its test numbers stay as they are.

Checks, with the records of the finish pass outside the repository (judged by an AI model, not a person):

- An independent recompute, written from the plan text without the driver, the checker or the upstream package (`recompute_gate0.py`, SHA-256 `6c9de7e317aa3c53a4666fccf98fdbe382a6cdfe5ddf95814ccc4090d52adf31`; its test passed 36 of 36 cases), gives the same validation accuracy and the same 13 literal statistics for the four counting runs. It reproduces the validation accuracies of both saved Gate 0 models (numba 0.5493333333333333, torch 0.8186666666666667) and gives the same verdicts and the same row (output `recompute_gate0.json`, SHA-256 `45e39ae8090deea0477b6e231f1f4791c2d98e706251112b954db578b8d962fe`).
- A direct check of leakage, selection and integrity found nothing to correct (`critic_self.md`, SHA-256 `09c21b5fb427c516adbc06dfa2081515a5a1b08df51aed3d0ff620d17b1fab82`):
  - no test array was read;
  - the freeze came before the first run;
  - every run folder holds copies of the driver and of `src/runlog.py` with the pinned SHA-256;
  - each arm A run records the variant's one-line change, exactly as the plan states it.

## Status after the finish pass of 2026-10-25

The pass repaired and tested code, and wrote, froze and ran one dev plan (the Gate 0 cause check: five GPU runs on the validation split, one of them a smoke run); it changed no status of the twelve ids. Costs are estimates (UNVERIFIED) and the chances are judgements from the project's earlier evidence, as the inventory of the pass gave them.

| Id | State after the pass | What it would need | Compute (estimate) | Chance of a positive, reliable result |
|---|---|---|---|---|
| M1 | not run; code tested on the CPU against a stand-in for the GPU package; the launch chain starts a job in `../decision_layer/.venv-htm` (tested with a stub in place of the GPU lock and a dummy job; the package was not imported on a GPU) | a registration with a held-out split (PTB test pairs through a `final_test.py` that does not exist yet, or a dev-test cut from the training text), baselines tuned to the same degree and scored on the same pairs (the class bigram model beyond k = 200 and with several SVD seeds, a model of the word marginals alone, an additive model), a GPU smoke run that measures the time per pair | about 3 GPU-hours for 3 seeds of one setting, about 8 to 13 GPU-hours with smoke, tuning and the two ablations (extrapolated from timing runs of hierarchical_tm on another task) | low: the hierarchical Tsetlin Machine was below Jaccard, logistic regression and PPMI-SVD on the earlier word-pair tests of the project; clearing the single number 0.7239 on validation is more likely (medium) than beating equally tuned baselines on held-out pairs |
| M2 | not run; blocked on the groups of M1; the evaluator now filters both kinds of groups alike | the groups of an M1 run, the k-means bar over several SVD seeds, a registration with a test path or a claim limited to validation | under a CPU-minute per evaluation (25 to 35 s on 2026-10-25) | low; a stronger count floor for the same idea exists in the tally study (A2) |
| M3 | not run; no code; skipped | superseded, according to the inventory of the pass (not checked here), by the C-core line of the project's research folder outside this repository (registered on 2026-10-21) and by the tally study, which test a clause class model inside a strong count model on Penn Treebank | GPU-days | low |
| M4 | not run; no code; skipped | needs M3; WikiText-2 is not on the machine | days | not judged |
| M5 step 2 | not run; no code | design and code; overlaps the word-table routers of decision_layer | unknown | low |
| Gate 0 cause | run on 2026-10-25: outcome row 2, no cause statement (driver check met, arm A mixed, arm B not run by the plan's rule) | a new frozen plan if the cause is still wanted, for example arm B or the other differences between the kernels one at a time | about 1 to 2 CPU-hours per numba run (UNVERIFIED estimate from the run of 2026-10-02) | not judged |

## Changes of 2026-10-25

Dated note of the finish pass. Every change was tested before this note was written; the test outputs and their SHA-256 values are in the records of the pass (`test_out/` of the clause_lm folder of that pass), outside the repository, as the run folders are. No registered rule, setting or result changed. The SHA-256 values before the change are those of the published files of commit 1bb3935.

| File | Change | SHA-256 before | SHA-256 after | Test (fails before the change, passes after) |
|---|---|---|---|---|
| `src/launch_gpu.sh` | explicit interpreters (`CLM_JOB_PYTHON`, default `../decision_layer/.venv-htm/bin/python`; `CLM_GUARD_PYTHON` for the memory guard, run with `-I -B`); `CLM_GPU_RUN` and `CLM_DRY_RUN` for tests; argument checks; the job path is read from the caller's folder; after the review, nice 19 for the whole chain and the six thread pools at 2 | `7c0f8384dfcee3b23a694e6d51655dc5b39354d7195c3ed4d1a8fb047d913ee3` | `864a5be754fbaf13c8c99ecfe38519891cc4f79372a3d420f6e57a46db132f74` | `tests/test_launch_gpu.py` (before: exit 127, no `python3`) |
| `src/m5_gpu_all.sh` | retired: exits 3 at once; its header names the launcher of 2026-10-14 | `b47dd45a3b67f60f4e8a7adbc48499f04d4a2553361fe05b9b1a956eab6a97ea` | `ac425a915258343b6e99f83231f805530104fb17e887851185a7073c4898d4d4` | `tests/test_m5_gpu_all_retired.py` |
| `src/pair_tm.py` | scores every validation pair by default (before: a subset of 40,000 of the 93,766); reports the pass chosen on the tune half (`best_on_tune`) apart from `last_pass`; saves the model, groups and include bits of the chosen pass and the last model; reads include bits from the saved state; lists only the words of single-word options in the groups and counts empty and dead options | `48b5a18c170847873e31c585a59dcb0246acfd434ce09b31e81d345264c395a6` | `283a45c7a4db5e830cf0be1c056e98bba4502a66439d43b89be7dfa04e4a2fb5` | `tests/test_pair_tm.py` (before: 40,000 of 50,000 pairs scored, and no run end without pycuda) |
| `src/clause_lm.py` | one group filter (`keep_groups`, at least 2 words) for k-means and Tsetlin Machine groups; `svd_seed` and `min_group_words` in the config | `e0257c42576ece474f13552e89459354efb101720732f7991225a22ce7ae6139` | `78b8f25f884b2ea652b527ed1b179dd9c3d5c53bfdbccd66465ec34ec67cef73` | `tests/test_clause_lm.py` (before: a 1-word group admitted) |
| `src/pair_baselines.py` | seeded start vector of the SVD (`ppmi_svd(..., seed=0)`); `svd_seed` in the config | `b5d867a49e30a996b011b37d4d56f092d5113729bd5ed74f8eac7286a1b6dc58` | `70a9a45a3fd5062607ac334ec962b8ed8073552e33dbe773dc5527bf04f00d57` | `tests/test_ppmi_svd.py` (before: two calls differ by up to 0.79) |
| `src/m5_setup.py` | the download time is read from the data file (`setup_config`, `file_time_utc`) instead of a fixed string | `972beb78f9fe6eef399f8a75ecd93c2433e14a6de1c92e628737a78a3ba59c28` | `26b0ca835f790753f3d2cd276ae706fdbed9950ffb7735878d69c85a688d3275` | `tests/test_m5_setup_time.py` |
| `src/m5_run.py` | docstring only: the stale first paragraph rewritten, a note on the memory guard added; the code is unchanged | `23d53448afcb76c675c4dcf5600b3508edf7e6f0fe6a9a9fe4109de78d73f6b3` | `35b7098e22015d8793b35864de5a6b999061882e95b7c6e4977fd2eaf682b494` | `tests/test_m5_files_unchanged.py` (code equal to the 13 copies of 2026-10-14) |
| `src/gate0_cause.py`, `src/gate0_cause_report.py` | new: driver and checker of the Gate 0 cause plan (frozen at 12:59:23 UTC and run the same day); after the review the driver pins the whole cdllm tree by one manifest and records its package list and the call count of the variant's function, and the checker reads the freeze record, compares literal SHA-256 values, refuses a run for any failed check and prints one line of the ordered outcome table | none | `5b01c9e9f8ada227c60319575e98a6f0dab6440eedde24a93d6d087c26a6ebd3` and `d27adfe35b977d47be45dcad1bd4bc944c5f247ea66d2d60b980a9096c01d5fc` | `tests/test_gate0_cause.py`, `tests/test_gate0_cause_report.py` |
| `docs/plan_gate0_cause_2026-10-25.md`, `docs/plan_gate0_cause_2026-10-25.frozen.json` | new: the dev plan of the Gate 0 cause check and its freeze record, written by `scripts/freeze.py` at 12:59:23 UTC | none | `df091761ba70255933f2e97577cb5d3c6941cfa52c59ee4fba29d940f5a4be41` and `a6934386fb599bbc26092e902eaf3d480c6a9104ba684fb0f202c2055082be0c` | `scripts/freeze.py --check` before the freeze; the checker reads the record |
| `scripts/freeze.py` | new: the study's freeze tool (a record beside the plan with the time, the plan's SHA-256, the pinned files after checking them, and the study files; `--check` writes nothing) | none | `12b3d17e7bc835d9339455c58f409a4ad8e94519b341f14bd245e21e860cba94` | `tests/test_freeze.py` |
| `tests/` | new: the tests of the study | none | as read at 12:56:13Z in the records of the pass | the tests themselves |

The run folders of 2026-10-02 hold copies of the earlier `src/clause_lm.py` and `src/pair_baselines.py`, byte identical to the files before the change; their recorded numbers come from that code. `src/m5_report.py` and `src/runlog.py` are unchanged. No file of `src/clause_driven_llm/` and no file of the Clause-Driven-LLM copies changed. The SHA-256 values after the change were read on 2026-10-25 at 12:56:13 UTC, after the last test run; a later edit changes them.

The tests ran on 2026-10-25 and passed; the last run of all of them went from 12:50:19 to 12:51:47 UTC, and each output is a file `test_out/final2_<test>.out` in the records of the pass, with its SHA-256 in the summary of the pass:

| Test | Interpreter | Result |
|---|---|---|
| `tests/test_kn.py`, `tests/test_ptb.py`, `tests/test_runlog.py`, `tests/test_pairs_data.py`, `tests/test_clause_lm.py`, `tests/test_clause_lm_ptb_valid.py`, `tests/test_ppmi_svd.py`, `tests/test_pair_tm.py`, `tests/test_launch_gpu.py`, `tests/test_m5_gpu_all_retired.py`, `tests/test_m5_setup_time.py`, `tests/test_m5_files_unchanged.py`, `tests/test_gate0_cause_report.py`, `tests/test_freeze.py` | `.venv-cdllm` | PASS |
| `tests/test_gate0_cause.py` | `.venv-cdllm-gpu`, no GPU visible | PASS |
