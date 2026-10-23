# clause_lm: preparation for a logic-based language model

This study prepares a word-level language model whose knowledge is a set of readable clauses. The model it aims at has Tsetlin Machine clauses that are AND-rules over the preceding word positions, where each position holds an OR-group of words, and it lets smoothed counts vote, so that the output is a probability distribution that can be scored with perplexity. The study started on 2026-10-02 and is the third study of the programme. What exists so far is groundwork: n-gram baselines on Penn Treebank, an evaluator that adds clause votes to a trigram model, data and statistical baselines for a word-pair task, a pair model that was written and not run, and the first step of a bridge to query routing: a reproduction of the Clause-Driven-LLM router on CLINC150 and a registered test of the specificity s on that router, whose verdict is negative.

All results are on the results page, [docs/results.md](docs/results.md).

## Question and milestones

The question is whether clauses with learned OR-groups of words add predictive value over count statistics, first on word pairs and then inside a language model, and whether the same kind of clause helps a readable router. The design of the study splits this into five milestones.

| Id | Content |
|---|---|
| M1 | Concept test: a hierarchical Tsetlin Machine on adjacent word pairs of Penn Treebank, restricted to the 1000 most frequent words (`src/pair_tm.py`). It is compared with statistical word classes on pairs that never occur in the training text. Planned comparisons: one global specificity s against s chosen per frequency bin, and hierarchical against flat with an equal number of leaves. |
| M2 | Clause-count language model, version 0: the OR-groups of M1 as context clauses with count votes, interpolated with a Kneser-Ney trigram model (`src/clause_lm.py`). |
| M3 | Full clause language model: AND-rules over the four preceding positions with OR-group literals and count votes. |
| M4 | Scale and efficiency: WikiText-2 and WikiText-103, with measured throughput and measured energy per token. |
| M5 | Bridge to Clause-Driven-LLM. Step 1 asks whether a higher s closes part of the gap between its lexical router and its embedding router on CLINC150. Step 2 uses OR-group word concepts as router features. |

Only M5 step 1 has a registration, [docs/prereg_M5_step1.md](docs/prereg_M5_step1.md). The docstrings of `src/pair_tm.py` and `src/pairs_data.py` carry the label M1b. No record of the study defines that label; both files belong to milestone M1.

## Status

Running, as preparation. The runs of the study date from 2026-10-02 and 2026-10-14. The n-gram baselines, the pair data, the statistical pair baselines and the clause evaluator each ran once; the evaluator ran with k-means word groups, and no Tsetlin Machine took part in it. In M5 step 1, Gate 0 of the registration, the reproduction of the lexical router, was not met in its one completed run with the numba CPU backend, and the cause is not confirmed. A post hoc retry with a different label order was stopped without a score. The registration was then amended, after the Gate 0 result was known, to repeat Gate 0 with the torch backend on a GPU. That amended Gate 0 ran on 2026-10-14 and was met, and the registered grid of M5 step 1 followed; by the registered rules its verdict is negative. The pair model of M1 was written and not run. M3, M4 and M5 step 2 have no code in the tree and no runs. No Tsetlin Machine has been trained on Penn Treebank in this study.

## What was found

The results so far are bars for later work, a reproduction that met its gate only under a post hoc amendment, and one negative registered result on the router. None of them is a result about a Tsetlin Machine language model.

On the validation split of Penn Treebank, the interpolated Kneser-Ney trigram model is the strongest of the three n-gram baselines. On the pair task, a word-bigram count model carries no information about pairs that are absent from the training text, which holds by construction, while statistical word classes and similarity smoothing do carry some. The best of them is the bar that the pair model of M1 has to beat. Adding the count votes of k-means word groups to the trigram model lowers validation perplexity by a small amount and changes top-1 and top-5 accuracy only marginally; this is the bar for M2.

The reproduction of the lexical Clause-Driven-LLM router first used the numba CPU backend, the backend fixed in the registration. It reached a test accuracy far below the value recorded in the upstream repository, with clauses that carry very many literals. The working notes of the same day, summarised on the [results page](docs/results.md#diagnosis-notes-of-2026-10-02), attribute this to a difference between the numba backend and the torch GPU backend of Clause-Driven-LLM, on the basis of reading both training kernels. A timing probe put the torch backend on the CPU beyond the time limit set for the diagnosis. The amended Gate 0, the same settings with the torch backend on a GPU, was met on 2026-10-14 in run `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu`, with clauses of a median of 1 literal. This is consistent with the backend explanation and does not isolate the difference between the kernels; the amended gate is post hoc.

The registered question about s has a negative answer by the registered rules. Over s in {8, 25, 50, 100} and seeds 42, 123 and 456, the mean validation accuracy is highest at s = 8, so s* = 8; the mean test accuracy at s* is 0.8201, below the 0.848 that the success rule needs, and the mean test accuracy falls as s grows (report run `20261014T222730Z_m5_summary_sorted_torchgpu`). One grid run did not have the GPU to itself for about 40 s; the results page gives the note next to that run.

## What the study needs from other studies

The study is not self-contained. It reaches data, scripts and queue files of the two earlier studies by relative path and expects them as sibling folders under `studies/`.

The Penn Treebank text comes from the tm_language_model study: `src/ptb.py` reads `../tm_language_model/data/ptb/raw/<split>/text.utf8`, and every Penn Treebank script of this study loads its data through that module.

GPU jobs start through scripts of the tm_language_model study. `src/launch_gpu.sh` changes into `../tm_language_model`, sources its `env/env.sh`, and starts the job through its `scripts/mem_guard.py` and `scripts/gpu_run.sh exclusive`, with the interpreter of its `.venv-htm` environment (lock file `../tm_language_model/env/htm.lock`). `src/m5_run.py` puts the same `scripts/mem_guard.py` and `scripts/gpu_run.sh exclusive` in front of its training command when it is called with `--backend torch_gpu`, with the interpreter of this study's `.venv-cdllm-gpu` environment and without sourcing `env/env.sh`, and `src/m5_gpu_all.sh` takes the GPU lock through the same `gpu_run.sh`. `src/pair_tm.py` imports the package PyHierarchicalTsetlinMachineCUDA, which that environment provides.

`src/m5_gpu_all.sh` waits on queue files of the hierarchical_tm study. It reads `../hierarchical_tm/queue/jobs.txt`, `started.txt` and `queue.log`, which are local queue state and are not distributed, checks that no process with the name of one of that study's drivers or queue runners is alive, and requires the last line of `queue.log` to contain `QUEUE EMPTY` or `MPS daemon stopped`. The runs of 2026-10-14 were started by a launcher that runs the same steps without this wait, because the hierarchical_tm study had finished and the wait could not end ([results page](docs/results.md#amended-gate-0-and-the-grid)).

These launchers describe one workstation with a single GPU that the studies share through a lock. On another machine they document how the jobs were ordered; they are not a portable tool.

## Layout

```
clause_lm/
  README.md
  docs/
    prereg_M5_step1.md      registration of M5 step 1, with its amendment
    results.md              results page: statuses, run folders, numbers
  src/
    ptb.py                  Penn Treebank loader (reads the tm_language_model data)
    runlog.py               run logging (study rule 4)
    ngram_baselines.py      unigram, bigram and trigram Kneser-Ney baselines
    kn.py                   Kneser-Ney trigram model with a per-context evaluator
    clause_lm.py            evaluator: clause votes on top of the trigram model (M2)
    pairs_data.py           builds the pair data (M1)
    pair_baselines.py       statistical baselines for the pair task (M1)
    pair_tm.py              pair model, a hierarchical Tsetlin Machine (M1); not run
    m5_setup.py             M5 step 1: setup record
    m5_run.py               M5 step 1: one training run inside a run folder
    m5_setup_torch.py       M5 step 1: setup and timing probe, torch backend on the CPU
    m5_grid.sh              M5 step 1: the grid after Gate 0
    m5_report.py            M5 step 1: aggregation and the registered rules
    m5_gpu_all.sh           M5 step 1: launcher for the amended Gate 0 and the grid on a GPU
    launch_gpu.sh           wrapper for one GPU job (study rule 3)
    pdftext.py              PDF text extractor, used to check quotations against their sources
    clause_driven_llm/
      cdllm_data__init__.py           loader added to the upstream package
      train_controller.state1.patch   change to the upstream training script, first recorded state
      train_controller.patch          change to the upstream training script, second recorded state
```

The code also reads and writes four local folders inside the study folder: `results/` (run folders), `data/` (the pair data), `work/` (the work copy of Clause-Driven-LLM and the outputs of M5) and `references/` (the reference copy of Clause-Driven-LLM). The repository ignores all four, and none is distributed.

`src/m5_run.py` as published is a later revision than the one that produced the two Gate 0 run folders of 2026-10-02: the backends `torch` and `torch_gpu` were added after those runs, as the dates in its docstring show, and the first paragraph of the docstring still describes the earlier state. The copies of the script inside those two run folders are the earlier revision. The copies inside the run folders of 2026-10-14 are byte identical to the published `src/m5_run.py`, and the copy of `src/m5_report.py` in the report run of that day to the published file. No other file in `src/` imports `src/pdftext.py`.

## Upstream dependency: Clause-Driven-LLM

M5 step 1 runs code of Clause-Driven-LLM, an earlier repository of the author (https://github.com/AnwarDebes/Clause-Driven-LLM), version 0.1.0 (the string `cdllm-tm-v0.1.0` opens the model version id in `metrics.json` of the Gate 0 run; the freeze files give no version for it), a router that trains a coalesced Tsetlin Machine on features of a query. Its licence is MIT according to the citation file of the downloaded copy. The repository is not copied into this tree. The copy used by the runs was downloaded on 2026-10-02 from the main branch as a source archive without version history, so the copy itself carries no commit id. The commit `5efe0d7` (dated 2026-06-14) was noted as the latest commit of the repository on that day. Status of this commit id: recorded in project notes, not checked against upstream.

The code of this study expects two local copies: an unmodified reference copy at `references/code/Clause-Driven-LLM` and a work copy at `work/Clause-Driven-LLM`. The work copy differs from upstream in two places, and both differences are published here.

The first is an added file. The module `src/cdllm/data` is absent from the upstream repository; the docstring of the loader gives the reason, an ignore rule for `data/` that also matched the code folder. The module was rebuilt on 2026-10-02 to the interface that the upstream scripts and the upstream loader test use. The rebuilt file is `src/clause_driven_llm/cdllm_data__init__.py`. It is a rebuild and not the original module, and its Banking77 part is an unverified reconstruction that M5 step 1 does not use.

The second is a change to the upstream training script `scripts/train_controller.py`, made in two steps and published as two patches against the upstream file. `src/clause_driven_llm/train_controller.state1.patch` adds two lines that compute the validation accuracy of the final model; it reproduces the first recorded state of the script. `src/clause_driven_llm/train_controller.patch` holds the same two lines and a `--device` argument that is passed on to the training call; it reproduces the second recorded state. Each patch applies to the upstream `scripts/train_controller.py` on its own. They are not applied one after the other. The patch headers use the `a/` and `b/` prefixes.

A reader who wants the code path of the runs therefore needs two steps: place the loader at `src/cdllm/data/__init__.py` of an upstream checkout under `work/Clause-Driven-LLM`, and apply one of the two patches to that checkout. `work/` is not distributed. [docs/results.md](docs/results.md) says which run records hold the SHA-256 of which state, and [../../docs/hash-table.md](../../docs/hash-table.md) lists the pinned files of the repository. The loader and the comment text inside the patches carry an earlier working name of the study. They are kept byte for byte, because the run records hold the SHA-256 of the loader and of the patched script.

## Environment

Package versions are known only for the environments of M5 step 1, from the freeze files that were stored with the runs. The run folders of the four Penn Treebank runs hold no environment record, so the interpreter and the package versions of those runs are not known from the run files.

M5 step 1 used a CPU environment named `.venv-cdllm` with Python 3.11.16. Its freeze was stored three times, and a fourth freeze belongs to the GPU environment of the runs of 2026-10-14:

- A: `results/20261002T195152Z_m5_setup_sorted/pip_freeze.txt`, stored by the setup run that preceded the completed Gate 0 run. The file of the same name in `results/20261002T205115Z_m5_setup_first_appearance` holds the same list.
- B: `results/20261002T210555Z_m5_setup_torchcpu/pip_freeze.txt`, the same environment after torch was added for the timing probe.
- C: `results/m5_gpu_env_freeze.txt`, the environment `.venv-cdllm-gpu` that was prepared on 2026-10-02 for the GPU rerun of Gate 0. The file belongs to no run folder and records no Python version.

For the amended Gate 0 and the grid of 2026-10-14, `.venv-cdllm` was rebuilt from freeze A and `.venv-cdllm-gpu` from freeze C; the package lists of the rebuilt environments are identical to the recorded ones. The training ran in `.venv-cdllm-gpu`, and each of these run folders records torch 2.6.0+cu124 (`config.json`).

Each freeze also lists Clause-Driven-LLM itself, as an editable install of the work copy.

| Package | A | B | C |
|---|---|---|---|
| cloudpickle | 3.1.2 | 3.1.2 | 3.1.2 |
| filelock | not listed | 3.32.3 | 3.32.3 |
| fsspec | not listed | 2026.7.0 | 2026.7.0 |
| iniconfig | 2.3.0 | 2.3.0 | not listed |
| jinja2 | not listed | 3.1.6 | 3.1.6 |
| joblib | 1.6.0 | 1.6.0 | 1.6.0 |
| llvmlite | 0.50.0 | 0.50.0 | 0.50.0 |
| markupsafe | not listed | 3.0.3 | 3.0.3 |
| mpmath | not listed | 1.3.0 | 1.3.0 |
| narwhals | 2.26.0 | 2.26.0 | 2.26.0 |
| networkx | not listed | 3.6.1 | 3.6.1 |
| numba | 0.68.0 | 0.68.0 | 0.68.0 |
| numpy | 2.4.6 | 2.4.6 | 2.4.6 |
| packaging | 26.3 | 26.3 | not listed |
| pluggy | 1.6.0 | 1.6.0 | not listed |
| pygments | 2.21.0 | 2.21.0 | not listed |
| pytest | 9.1.1 | 9.1.1 | not listed |
| pyyaml | not listed | not listed | 6.0.3 |
| scikit-learn | 1.9.1 | 1.9.1 | 1.9.1 |
| scipy | 1.17.1 | 1.17.1 | 1.17.1 |
| setuptools | not listed | 78.1.0 | not listed |
| sympy | not listed | 1.14.0 | 1.13.1 |
| threadpoolctl | 3.7.0 | 3.7.0 | 3.7.0 |
| torch | not listed | 2.14.1+cpu | 2.6.0+cu124 |
| tqdm | 4.70.1 | 4.70.1 | 4.70.1 |
| triton | not listed | not listed | 3.2.0 |
| typing-extensions | not listed | 4.16.0 | 4.16.0 |

Freeze C lists in addition the CUDA runtime packages nvidia-cublas-cu12 12.4.5.8, nvidia-cuda-cupti-cu12 12.4.127, nvidia-cuda-nvrtc-cu12 12.4.127, nvidia-cuda-runtime-cu12 12.4.127, nvidia-cudnn-cu12 9.1.0.70, nvidia-cufft-cu12 11.2.1.3, nvidia-curand-cu12 10.3.5.147, nvidia-cusolver-cu12 11.6.1.9, nvidia-cusparse-cu12 12.3.1.170, nvidia-cusparselt-cu12 0.6.2, nvidia-nccl-cu12 2.21.5, nvidia-nvjitlink-cu12 12.4.127 and nvidia-nvtx-cu12 12.4.127.

`src/launch_gpu.sh` and `src/pair_tm.py` belong to a different environment, `.venv-htm` of the tm_language_model study, whose versions are in `../tm_language_model/env/htm.lock`.

## Study rules

The code cites three rules of the study by number. The numbers are those of the study's original rule list. The other numbers of that list are cited by no published file and are not listed.

| Rule | Statement | Cited in |
|---|---|---|
| 3 | CPU first, and the hierarchical_tm study first. CPU jobs cap the numba and BLAS thread pools at 2 threads. A GPU job starts only when the job queue of the hierarchical_tm study is idle, and never while experiment E3 or the final test of that study runs. | `src/launch_gpu.sh` |
| 4 | Every number comes from a logged run. Each run has a folder `results/<UTC time>_<name>/` with its configuration, its metrics, a copy of the code that ran, and the SHA-256 of the data files used. | `src/runlog.py` |
| 5 | Development uses the validation split of Penn Treebank. The test split is used once, at the end, and the access is logged. | `src/ptb.py` |

For rule 3, `src/launch_gpu.sh` waits for the GPU lock that the studies share and then holds it exclusively. Code comments and the registration write "thesis first" for this rule; "the thesis" and "the thesis queue" there mean the hierarchical_tm study and its job queue.

For rule 4, `src/runlog.py` writes `config.json`, `metrics.json`, `log.txt` and `data_hashes.json` into the run folder and copies the calling script and `runlog.py` itself next to them.

For rule 5, `src/ptb.py` refuses to load the test split. Its error message calls the rule "README rule 5" and names a script `final_test.py` as the only reader of the test split. No such script exists in the study, and no run folder records the test text among its data files.

## Data

No dataset is distributed with the repository. [../../docs/data.md](../../docs/data.md) describes how each one is obtained.

The study uses the word-level Penn Treebank text with the tokens `<unk>` and `N`, one sentence per line, in the form prepared by the tm_language_model study. `src/ptb.py` reads `data/ptb/raw/{train,validation,test}/text.utf8` of that study, appends `<eos>` to every sentence and treats each split as one token stream; the vocabulary is built on the training split. Run `20261002T170451Z_ngram_baselines` records the SHA-256 of the training text as `5145926136ee9aef6f359b267ac09cc8a920879cd71725de17c490dd111d2998` and of the validation text as `fadf6277290823f881b7bf39b84e88536c87a859cd629dcfd1d2dfdef06097cf` (`data_hashes.json`).

`src/pairs_data.py` builds the pair data of milestone M1 from that text. It writes `data/ptb_pairs_K1000.npz` and the vocabulary file `data/vocab_K1000.json`. Only the K = 1000 most frequent words of the training split are kept. A positive pair is two adjacent words of the text that are both among these K words. For each positive pair there is one negative pair with the same first word and a second word drawn from the unigram distribution over the K words, raised to the power 0.75; the random seed is 0. Every validation pair carries a flag that says whether the pair occurs anywhere in the training text (seen) or not (unseen). Run `20261002T171456Z_pairs_data_K1000` built 1,158,614 training pairs and 93,766 validation pairs, positives and negatives together; 3,193 positive and 23,757 negative validation pairs are unseen (`config.json` and `metrics.json` of that run). The pair file holds the index arrays and labels of both splits, the seen flags, the ids of the K words and the noise distribution. Run `20261002T171904Z_pair_baselines_K1000` records the SHA-256 of the pair file as `be3836fc4b1a7a8f8578a0557779256e1018560875703da331bead90d207bdc4`. Neither the pair file nor the vocabulary file is distributed; `src/pairs_data.py` writes both from the Penn Treebank text.

The scripts of M5 step 1 read three more local folders: the reference copy of Clause-Driven-LLM under `references/code/Clause-Driven-LLM`, the work copy under `work/Clause-Driven-LLM`, which also holds the CLINC150 file as `data/clinc150_data_full.json`, and the derived features and training outputs under `work/m5`. None of them is distributed. The CLINC150 file comes from the address that `src/m5_setup.py` names, https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json; run `20261002T195152Z_m5_setup_sorted` records its SHA-256 as `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.

## What a clone contains

A clone holds the code, the registration and the results page. It does not hold the 23 run folders, the Penn Treebank text, the pair data, the CLINC150 file, the copies of Clause-Driven-LLM, the virtual environments or the queue files of the hierarchical_tm study. No number on the results page can therefore be checked from a clone alone; the run ids can be compared with [../../docs/run-index.md](../../docs/run-index.md).

Recomputing the Penn Treebank numbers needs the Penn Treebank text at the place `src/ptb.py` expects. Recomputing the Gate 0 run needs the two copies of Clause-Driven-LLM prepared as described above, the CLINC150 file, and environment A. The amended Gate 0 and the grid need in addition environment C, a GPU, and a `python3` first on the search path, which the memory guard of tm_language_model starts under; without it the first attempt of 2026-10-14 ended at its start. Whether such a rerun gives the recorded numbers has not been tested. Two known reasons for a difference are that the published `src/m5_run.py` is a later revision than the one that ran, and that the feature vocabulary has ties at its cut-off, which the setup script notes can resolve differently across numpy versions and CPUs.

The study has no tests.
