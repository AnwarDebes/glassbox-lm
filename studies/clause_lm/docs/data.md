# Data of the clause_lm study

No dataset, derived data file or model file of the study is distributed. The code reads data from two places: the
raw Penn Treebank store of the tm_language_model study, and the local folders `data/` and `work/` of this study, which
the repository ignores. For each dataset this page says what it is, which part of the study uses it, where the code
expects it, which source and version a record gives, which SHA-256 a record holds, and what is known about its
licence. It was written on 2026-10-25; the repository's [data page](../../../docs/data.md) covers the same files in
less detail.

Paths are relative to the study folder. "Not recorded" means that no file of the repository and no run record gives
the value. A licence is named only where a file on disk states one; otherwise the entry reads "licence to be
confirmed". The run folders are not distributed, so the hashes that only they hold are printed here. Whether a rebuild
from a clone gives the recorded hashes has not been tested.

## Datasets and derived files

| Data | Used by | Expected at | Source and version recorded | SHA-256 and where it is recorded | Licence |
|---|---|---|---|---|---|
| Penn Treebank, word level: training and validation files | M1 and M2 groundwork: `src/ngram_baselines.py`, `src/pairs_data.py`, `src/pair_baselines.py`, `src/clause_lm.py`, all through `src/ptb.py` | `../tm_language_model/data/ptb/raw/train/text.utf8` and `../tm_language_model/data/ptb/raw/validation/text.utf8` | the raw store of tm_language_model; the [data page](../../../docs/data.md) gives its source, the Hugging Face dataset `ptb-text-only/ptb_text_only` at the ref `refs/convert/parquet`, commit not recorded | training file `5145926136ee9aef6f359b267ac09cc8a920879cd71725de17c490dd111d2998`, validation file `fadf6277290823f881b7bf39b84e88536c87a859cd629dcfd1d2dfdef06097cf`, in `data_hashes.json` of `20261002T170451Z_ngram_baselines`, `20261002T171456Z_pairs_data_K1000` and `20261002T174323Z_clause_lm_kmeans200` | licence to be confirmed |
| Penn Treebank, test file | nothing: `src/ptb.py` refuses to load it | `../tm_language_model/data/ptb/raw/test/text.utf8` | as above | not recorded by this study: no run folder hashes it | licence to be confirmed |
| Pair data (derived) | M1: `src/pair_baselines.py`, `src/pair_tm.py`; M2: `src/clause_lm.py` reads its `top` | `data/ptb_pairs_K1000.npz` | written by `src/pairs_data.py` in run `20261002T171456Z_pairs_data_K1000` (file time 2026-10-02 17:14:57 UTC) | `be3836fc4b1a7a8f8578a0557779256e1018560875703da331bead90d207bdc4`, in `data_hashes.json` of `20261002T171904Z_pair_baselines_K1000` | derived from Penn Treebank |
| Vocabulary of the pair data (derived) | `src/pair_tm.py` (the words of the groups) | `data/vocab_K1000.json` | written by the same run | `0a1c1342368ad7c05a30cec3f6e67481c982120205a1c75da866f4ccf12de924`, computed on 2026-10-25; no run record holds it | derived from Penn Treebank |
| CLINC150, `data_full.json` | M5 step 1: `src/m5_setup.py` (features), `src/m5_run.py` (hash only) | `work/Clause-Driven-LLM/data/clinc150_data_full.json` | `https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json`, branch `master`, commit not recorded; downloaded with `curl -fsSL` on 2026-10-02, at 19:49:03 UTC by the file's modification time, the time that the setup runs record | `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`, 2,495,390 bytes, in `data_hashes.json` and `metrics.json` of `20261002T195152Z_m5_setup_sorted` and in `data_hashes.json` of every M5 training run | licence to be confirmed |
| Lexical features of CLINC150, labels in sorted order (derived) | every M5 training run; `src/gate0_cause.py` | `work/m5/emb_sorted/clinc150__lexical.npz` | the feature script `scripts/precompute_embeddings.py` of the Clause-Driven-LLM work copy, run by `src/m5_setup.py` | `15a987add2b590d8498768045d465167c808d8cbf8afd483330e89330444d0fc`, in `metrics.json` of `20261002T195152Z_m5_setup_sorted` and `data_hashes.json` of the M5 training runs | derived from CLINC150 |
| Lexical features, labels in order of first appearance (derived) | the stopped retry `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` only | `work/m5/emb_first_appearance/clinc150__lexical.npz` | as above, with `CDLLM_LABEL_ORDER=first_appearance` | `0016bf15b7e079908668ba7baee7f83159b7927d8c76f2fefd21c03bfaa55fc1`, in `metrics.json` of `20261002T205115Z_m5_setup_first_appearance` | derived from CLINC150 |

## Penn Treebank and its splits

The text is the word-level Penn Treebank with the tokens `<unk>` and `N`, one sentence per line, in the form prepared
by tm_language_model. `src/ptb.py` appends `<eos>` to every sentence, reads each split as one token stream and builds
the vocabulary as the sorted set of the training tokens: 10,000 word types, 929,589 training tokens and 73,760
validation tokens (`config.json` of `20261002T170451Z_ngram_baselines`). Every number of the study on Penn Treebank is
on the validation split; the test split is never opened (`src/ptb.py` exits before it reads a file; `tests/test_ptb.py`
checks this). No script `final_test.py`, which the loader's message names as the only reader of the test split,
exists.

The pair data keeps the 1,000 most frequent training words (ties go to the lower word id). A positive pair is two
adjacent words of a stream, both among them; for each positive there is one negative with the same first word and a
second word drawn from the unigram distribution over the 1,000 words raised to the power 0.75, with seed 0. The
validation file stores the positives first and the negatives after them, negative `n + i` with the first word of
positive `i`. A validation pair is "seen" when it occurs anywhere in the training stream. Run
`20261002T171456Z_pairs_data_K1000` built 1,158,614 training pairs and 93,766 validation pairs, with 3,193 unseen
positives and 23,757 unseen negatives (`metrics.json`). `tests/test_pairs_data.py` checks every one of these properties
against an independent computation on toy data.

## CLINC150 and its features

`src/m5_setup.py` runs the feature script of the Clause-Driven-LLM work copy through the loader rebuilt for this study
(`src/clause_driven_llm/cdllm_data__init__.py`). The feature file holds 150 intents, 15,000 training, 3,000 validation
and 4,500 test queries, and 100, 100 and 1,000 out-of-scope queries for training, validation and test, each as 2,000
presence features of 1- and 2-grams (`precompute_stdout.txt` and `metrics.json` of `20261002T195152Z_m5_setup_sorted`).
The registered runs of M5 step 1 read the test arrays: 13 runs scored the test split once each (the numba Gate 0 run
of 2026-10-02 and the 12 runs of 2026-10-14). `src/gate0_cause.py` reads only `Xtrain`, `ytrain`, `Xval` and `yval`
of the same file and records the keys it read (`tests/test_gate0_cause.py` checks this).

## Model files

The training runs of M5 step 1 write the trained router (`tm.npz`, the automaton states and clause weights), the
booleniser, the test predictions and the out-of-scope margins into `work/m5/runs/<label order>[_torchgpu]/s<S>_seed<SEED>/`.
Two of them serve as references of the Gate 0 cause plan; their SHA-256 values were computed on 2026-10-25:

| Model | File | SHA-256 |
|---|---|---|
| numba Gate 0 run `20261002T195255Z_m5_gate0_s8_seed42_sorted` | `work/m5/runs/sorted/s8_seed42/clinc150__lexical__seed42/tm.npz` | `f1183ee36a2a8568564ca4ce60d3dbd48f99fb71dadfc2d63a53027be495191e` |
| torch amended Gate 0 run `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu` | `work/m5/runs/sorted_torchgpu/s8_seed42/clinc150__lexical__seed42/tm.npz` | `5e2478a4c0a3a984773901d8c0ee371bf4fa74d1b103f20dbccaf9235608d75e` |

The pair model of M1 has never run, so no model file of it exists.

## Known problems of the data

- Penn Treebank is pinned to a ref name and not to a commit, and its hashes are in run records only.
- CLINC150 is pinned by a branch name and the hash of the downloaded file, not by a commit.
- The 2,000 features of the lexical file have ties at the cut-off: 120 terms share the document frequency 13 and 29
  of them are selected by an unstable sort, so the vocabulary can differ between numpy versions and processors
  (`vocab_cutoff.json` of `20261002T195152Z_m5_setup_sorted`).
- The statistical word classes of M1 and M2 are not reproducible from the runs of 2026-10-02: `ppmi_svd` of
  `src/pair_baselines.py` called `scipy.sparse.linalg.svds` without a start vector, and scipy then draws one from
  operating-system entropy. Four calls of the k-means groups on 2026-10-25 gave 158, 156, 155 and 155 groups. Since
  2026-10-25 the start vector is seeded (seed 0), so a run repeats; the bars of 2026-10-02 (unseen AUC 0.7239, perplexity
  157.24) are one unseeded draw each. The results page gives the values with the seeded SVD.
- `src/pairs_data.py` draws the negatives of the training subset and of the validation pairs from one random stream,
  so a call with `n_train_pos` above 0 gets other validation negatives. The recorded pair data used `n_train_pos` = 0.
