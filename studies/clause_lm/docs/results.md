# Results of the clause_lm study

The study has nine run folders, all of 2026-10-02 (UTC). None of them is flagged in the [run index](../../../docs/run-index.md), which holds no status for the run folders of this study because they have no `run.json`. One of the nine, the retry `20261002T205134Z_m5_gate0_s8_seed42_first_appearance`, was stopped with a terminating signal and gives no result. Every measurement on this page is taken from a file of the run that is named next to it. Thresholds, the grid and the times of the registration are quoted from that document; the time of the working notes is the one they carry. The run folders are not distributed, so the file names serve readers who hold them. Perplexities are rounded to two decimals, accuracies and AUC values to four; values that a registered rule compares are given in full. Every result here comes from a single run.

The question of the study, its milestones M1 to M5 and its dependencies are described in the [README](../README.md).

## Status of every milestone and registered item

Twelve ids carry a status: the milestones of the study design, with M5 counted as its two steps, and six items of the registration of M5 step 1, [prereg_M5_step1.md](prereg_M5_step1.md). The statuses are met, not met, not judged and not run. The study design also states, for M1, M2 and M3, what follows if the bar is missed; none of these rules has been applied, because none of the three tests has been run.

Milestones of the study design:

| Id | Content | Status | Qualifier | Run folders |
|---|---|---|---|---|
| M1 | Pair model on adjacent word pairs of Penn Treebank; it has to beat the statistical class baseline on unseen pairs, on 3 seeds | not run | preparation only: the pair data and the statistical baselines ran, the pair model was written and not run | `20261002T171456Z_pairs_data_K1000`, `20261002T171904Z_pair_baselines_K1000` |
| M2 | Clause-count language model with the OR-groups of M1; it has to beat the k-means class baseline, with the gain on unseen contexts | not run | preparation only: the evaluator ran once with k-means word groups and no Tsetlin Machine | `20261002T170451Z_ngram_baselines`, `20261002T174323Z_clause_lm_kmeans200` |
| M3 | Full clause language model | not run | no code, no run | none |
| M4 | Scale and efficiency | not run | no code, no run | none |
| M5 step 1 | Does a higher s close part of the gap between the lexical and the embedding router of Clause-Driven-LLM on CLINC150 (registered) | not judged | stopped at Gate 0, which was not met; the s-test was not run, so the registered question has no verdict | the five `m5` run folders listed below |
| M5 step 2 | OR-group word concepts as router features | not run | no code, no run | none |

Registered items of M5 step 1:

| Id | Registered content | Status | Qualifier | Run folders |
|---|---|---|---|---|
| Gate 0 | The loader is valid. Condition 1: the upstream loader test passes for CLINC150 (150 classes; 15,000 train; 4,500 test; 1,000 out-of-scope test). Condition 2: the reproduction run (s 8, seed 42) gives a test accuracy within 0.005 of 0.82778 | not met | one completed run, numba CPU backend, seed 42; cause not confirmed; a retry was stopped without a score | `20261002T195152Z_m5_setup_sorted`, `20261002T195255Z_m5_gate0_s8_seed42_sorted`; retry without a score: `20261002T205115Z_m5_setup_first_appearance`, `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` |
| Amended Gate 0 | Amendment of 21:11 UTC: the same locked settings with the torch backend on a GPU; pass only within 0.005 of 0.82778 | not run | post hoc: the amendment was written after the Gate 0 result was known; the GPU rerun did not take place | none |
| s-test grid | s in {8, 25, 50, 100} x seeds {42, 123, 456}, 12 runs including the Gate 0 run | not run | Gate 0 not met; the registration then stops before the s-test | none |
| Choice of s* | The s with the highest mean validation accuracy over the 3 seeds; ties go to the lower s | not run | no grid | none |
| Success rule | Mean test accuracy at s* >= 0.848, with every seed above 0.828 | not run | no grid | none |
| Negative-result rule | s* = 8, or a gain below 2 points | not run | no grid | none |

Gate 0 is one id with two numbered conditions. Condition 1 is recorded as passed in the setup run, condition 2 failed in the one completed run, and the gate needs both. The remaining lines of the registration (the validation metric, the rule that each run is scored once on the test split, the quantities reported besides accuracy, the scope line, and the GPU rule of the amendment with its limit of 15 minutes per run) describe the procedure and carry no status.

The registration was written on 2026-10-02 at 19:45 UTC, before the first run folder of M5, `20261002T195152Z_m5_setup_sorted` (19:51:52 UTC). Its amendment was written at 21:11 UTC, after the working notes on the result, which date from 21:09 UTC.

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

Run `20261002T195255Z_m5_gate0_s8_seed42_sorted` (`src/m5_run.py`) is the one completed training run of the study. Its settings are the locked settings of the registration: lexical features with 2 bins, 4,000 clauses, T 4000, 256 states, 40 epochs, batch size 128, hardest negative sampling, s 8.0, seed 42, and the numba backend on the CPU (`config.json`). The labels are in sorted order.

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

Gate 0 is not met. The registered consequence is to stop and report, with no s-test.

The cause is not confirmed. The working notes of 2026-10-02, summarised [below](#diagnosis-notes-of-2026-10-02), attribute the failure to the training backend and not to the rebuilt loader: by their reading of the upstream code, the recorded upstream value came from the torch backend on a GPU, and the numba kernel treats clauses that did not fire differently from the torch kernel. That reading is not tested by any run of this study, and the notes themselves mark it as not yet confirmed by a torch run. The run records support one narrower statement: in both Gate 0 run folders the numba kernel file of the work copy is recorded as identical to the reference copy (`config.json`, key `coalesced_identical_to_reference`), so the kernel that ran was the unmodified kernel of the downloaded copy.

Four limits go with these numbers wherever they are used. They come from one run with one seed. The run used the numba CPU backend. The published `src/m5_run.py` is a later revision than the one that ran; the run folder holds the earlier revision. The commit of the upstream copy is recorded in project notes, not checked against upstream.

### Retry with the labels in order of first appearance

The registration does not fix the order of the class labels. After the Gate 0 result, the features were built again with the labels in order of first appearance in the training split, and Gate 0 was started a second time. Both steps are post hoc: the registered consequence of a failed Gate 0 is to stop and report, and neither the registration nor its amendment of 21:11 UTC provides for a retry.

Run `20261002T205115Z_m5_setup_first_appearance` is the setup record of that retry. Its `metrics.json` shows the same CLINC150 file, the same loader and the same state of the patched training script as the first setup run, feature arrays of the same shapes, the same vocabulary diagnostic, intent names that are not in sorted order, and another SHA-256 of the feature file, `0016bf15b7e079908668ba7baee7f83159b7927d8c76f2fefd21c03bfaa55fc1`.

Run `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` started at 20:51:34 UTC and was stopped at 21:02:41 UTC before any evaluation, so it has no score. Its `metrics.json` holds the return code -15, a process wall time of 667.3 s and no accuracy; its captured standard output is empty. The key `CRASH` in that file is set because `src/m5_run.py` sets it for every exit that is not zero. A note file in the same folder explains the stop.

### Timing probe for the torch backend on the CPU

Run `20261002T210555Z_m5_setup_torchcpu` (`src/m5_setup_torch.py`) records a diagnosis step that was decided at 21:02 UTC, after the Gate 0 result, and is therefore post hoc (the run folder dates from 21:05:55 UTC, before the amendment of 21:11 UTC was written): repeat Gate 0 with the torch backend of Clause-Driven-LLM on the CPU, with a limit of 45 minutes per run (`config.json`). torch 2.14.1+cpu was added to the environment, and the training script was brought to its second patched state. The torch kernel file of the work copy is recorded as identical to the reference copy (`metrics.json`, key `torch_backend_identical_to_reference`).

A timing probe on training rows only, with no validation or test data, took 6.62 s for 2 minibatches and 37.87 s for 12 minibatches, from which the script derives 3.12 s per minibatch, and 0.44 s for one pass over 3,000 rows (`timing_probe.txt`, `metrics.json`). Extrapolated to a run of 40 epochs this gives an estimate of 246.0 minutes. The estimate is an extrapolation from 12 minibatches and not a measured run time. It is above the limit, so Gate 0 was not run with the torch backend on the CPU. The setup run itself took 49.7 s.

### Amended Gate 0 and the grid

The amendment of 21:11 UTC keeps the locked settings and changes the backend of Gate 0 to torch on a GPU, with the same rule. A GPU environment was prepared for it (freeze C in the README), and `src/m5_gpu_all.sh` is the launcher for it. According to the project notes the launcher was started on 2026-10-02 and again on 2026-10-03; no start left a run folder or a recorded outcome. The rerun did not take place. No run folder exists for it; `src/m5_run.py` would give such a folder the suffix `_torchgpu`. The records give no cause.

The grid of the s-test, 11 runs besides the Gate 0 run, was not run under either version of Gate 0. `src/m5_grid.sh` and `src/m5_report.py`, which run the grid and apply the registered rules, were written on 2026-10-02 and have not been run. There is no table over s and seeds, no s* and no verdict on the registered question.

### Recorded states of the patched training script

The run records hold the SHA-256 of the patched `scripts/train_controller.py` of the work copy. Two values occur.

| State | Change against the upstream file | SHA-256 of the patched file | Run records that hold it | Published patch |
|---|---|---|---|---|
| First | 2 lines added: validation accuracy of the final model | `d6214e313601fcede90e7b04c1dee444412c25141d85272620e03968b6d77e80` | `metrics.json` of `20261002T195152Z_m5_setup_sorted` and of `20261002T205115Z_m5_setup_first_appearance`; `config.json` of `20261002T195255Z_m5_gate0_s8_seed42_sorted` and of `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` | `src/clause_driven_llm/train_controller.state1.patch` |
| Second | 4 lines added and 1 removed in total: the first state and a `--device` argument | `e999bfb14af47181fd910927c74ed09f5c6b43cff880a6f38300b9b38a0068b2` | `metrics.json` of `20261002T210555Z_m5_setup_torchcpu` | `src/clause_driven_llm/train_controller.patch` |

The completed Gate 0 run and the stopped retry both used the first state. No training run used the second state. The SHA-256 of the added loader, `b851309171cb4d6d2615a84b90437a9c16a7f5d8e8812f62dfa11579b2a07e5e`, is held by the same four run records as the first state, and the published file `src/clause_driven_llm/cdllm_data__init__.py` is that loader. The [hash table](../../../docs/hash-table.md) lists the pinned files of the repository.

## The pair model of M1

`src/pair_tm.py` was written and not run. A smoke test of it was queued on 2026-10-02 and cancelled before it ran. No run folder exists for the pair model, and it has no result.

## Limits of these records

The four Penn Treebank runs record no environment, so their interpreter and package versions are not known from the run files. No run folder of the study records a commit of this repository; each holds a copy of the script that ran and of `src/runlog.py`. The keys of `data_hashes.json` in the nine run folders name the data files by the folder layout at the time of the runs, which differs from the layout of this repository. No result on this page was repeated with a second seed.

## Diagnosis notes of 2026-10-02

The working notes on M5 step 1 were written on 2026-10-02 at 21:09 UTC and are labelled there as reported, not re-checked. Their findings are kept here because they say what a rerun has to test. They rest on files outside the run folders (the downloaded copy of Clause-Driven-LLM and the saved model of the Gate 0 run), so no figure in this section was checked again for this page.

- **Values recorded upstream.** `results/tables/headline.json` of Clause-Driven-LLM gives, for the run that the gate targets: test accuracy 0.8277777777777777, test macro-F1 0.8300, out-of-scope AUROC 0.8104, selective accuracy at 80% coverage 0.9225, area under the risk-coverage curve 0.0410, 3,919 non-empty clauses, literals per clause with median 1, mean 151.4 and maximum 787, and a training time of 48.9 s.
- **Which backend produced them.** The paper sources and scripts of Clause-Driven-LLM name the GPU backend for its results (`paper/app_hparams.tex`, `paper/report.tex` line 303, `scripts/run_all_seeds.sh`, and `BACKEND = "auto"` in `scripts/run_sweep.py`, which picks torch when CUDA is available). The registration expected an exact match because the kernel is deterministic; that expectation does not hold across backends.
- **The two backends differ.** By a reading of the code, the torch backend (`torch_backend.py`, lines 131 and 154) applies Type I erosion, with probability 1/s per literal, to clauses that did not fire, for the sampled negative class as well whenever the clause weight for it is negative. The numba kernel (`coalesced.py`, lines 158 to 188) gives a non-firing clause no feedback for the negative class. The docstring of `coalesced.py` says that both backends train the same automaton state and clause weights.
- **The saved Gate 0 model.** All 4,000 clauses of the model contain only negated literals; no clause has a positive literal, and the median clause has 1,021 negated literals.
- **Vocabulary ties.** The vocabulary of 2,000 features has ties at the cut-off: 120 terms share the document frequency 13, and 29 of them are selected by an unstable sort, so the exact vocabulary can differ between numpy versions and processors (`vocab_cutoff.json` of run `20261002T195152Z_m5_setup_sorted`).
- **The loader.** The added loader takes an optional label order (argument `label_order`, environment variable `CDLLM_LABEL_ORDER`, default "sorted"). Its Banking77 validation split is a reconstruction that was not verified and not used.
- **Measurements outside the run folders.** Two timing measurements left no run folder: a probe of the numba kernel on 6 minibatches of training data at about 20:26 UTC, while Gate 0 trained, and a micro-benchmark of the torch kernel on synthetic tensors at about 21:07 UTC. Neither read validation or test data.

Taken together the notes read the failed gate as a backend mismatch: with lexical features the numba path and the torch path of Clause-Driven-LLM would train different models. This is not confirmed. The test is one Gate 0 run with the torch backend on a GPU, which the amendment registers and which has not been run.
