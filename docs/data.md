# Data

No dataset, no derived data file and no model file is distributed with this repository. The folders `data/`, `results/` and `work/` of the studies are local and ignored by git, and so are the download caches and the models saved inside the run folders. What the repository holds is the code that downloads or builds each dataset, the records that pin the hashes of the files, and two data cards. For each dataset this page says what it is, where it comes from, which version a file on disk records, which script obtains it, where the code expects it, which record pins its hashes, and what is known about its licence.

Three conventions hold on this page. Scripts, records and documents are cited from the repository root, while data locations are cited relative to the folder of the study that owns them, because that is how the code builds them. "Not recorded" means that no file of the repository and no run record gives the value. A licence is named only where a file on disk states one; otherwise the entry reads "licence to be confirmed".

The hash records pin files that are not in the repository. A file that has been downloaded or rebuilt can be compared with the SHA-256 in its record. Where a hash exists only in a run folder, which is not distributed either, its value is printed on this page. Whether a rebuild from a clone gives the recorded hashes has not been tested.

## studies/tm_language_model

The study keeps its data under `data/` in the study folder, or under the directory that the environment variable `TMLM_DATA_DIR` names when it is set (`studies/tm_language_model/src/tmlm/data/corpus_store.py`). Files there are only created and never overwritten: a builder that finds an existing file reads it again and checks its recorded hash.

The fifteen hash records are the files `studies/tm_language_model/docs/notes/preregistration-data-hashes*.json`. The [note next to them](../studies/tm_language_model/docs/notes/README.md) says which of them were reworded for publication. They belong to section 4 of the [pre-registration](../studies/tm_language_model/docs/13-preregistration.md) and to its amendments A4, A6, A7, A9 (item 10) and A10. In the table below, "main" stands for `preregistration-data-hashes.json` and a suffix such as `-A6-men` for `preregistration-data-hashes-A6-men.json`.

| Dataset | Source and version recorded | Expected at | Record | Licence |
|---|---|---|---|---|
| WikiText-103, raw text | Hugging Face dataset `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`; revision not recorded | `data/hf_cache/` (download cache) and `data/wikitext103/raw/<split>/` | main | licence to be confirmed |
| BPE tokenizers (derived) | trained on the WikiText-103 train split, vocabulary sizes 2000, 8000, 16000 and 32000 | `data/tokenizers/bpe_<V>/tokenizer.json` | main | derived from WikiText-103 |
| Token stores and cluster codes (derived) | WikiText-103 encoded with these tokenizers; cluster codes from the train split | `data/wikitext103/bpe<V>/<split>/` | main | derived from WikiText-103 |
| Frozen evaluation mask sets (derived) | validation and test split, a subword and a whole-word variant | `data/wikitext103/bpe<V>/evalsets/<split>_<variant>.npz` | main | derived from WikiText-103 |
| Paragraph pairs (derived) | pairs of paragraphs inside one split, BPE 8000 | `data/wikitext103/pairs_bpe8000/<split>.npz` | main | derived from WikiText-103 |
| Contamination report (derived) | overlap of 8-grams and 13-grams of the validation and test splits with the train split, BPE 8000 | `data/wikitext103/contamination_bpe8000/<split>_<n>gram.npz` | main | derived from WikiText-103 |
| Penn Treebank, word level | Hugging Face dataset `ptb-text-only/ptb_text_only` at the ref `refs/convert/parquet`; the commit behind the ref is not recorded | `data/ptb/raw/<split>/` and `data/ptb/bpe<V>/<split>/` | none in the repository, see below | licence to be confirmed |
| BLiMP | `nyu-mll/blimp`, revision `877fba0801ffb7cbd8c39c1ff314a46f053f6036` | the default Hugging Face cache, outside the study folder | `-A4-blimp` | cc-by-4.0, as the record gives it |
| AG News | `fancyzhx/ag_news`, revision `eb185aade064a813bc0b7f42de02595523103ca4`, files `data/train-00000-of-00001.parquet` and `data/test-00000-of-00001.parquet` | `data/hf_cache/hub/datasets--fancyzhx--ag_news/snapshots/<revision>/` | `-A6-agnews`, `-A7-agnews`, `-A7-agnews-test` | licence to be confirmed: the records disagree, see the known problems |
| SST-2 | `nyu-mll/glue`, revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`, files `sst2/train-00000-of-00001.parquet` and `sst2/validation-00000-of-00001.parquet` | `data/hf_cache/hub/datasets--nyu-mll--glue/snapshots/<revision>/` | `-A6-sst2`, `-A7-sst2` | licence to be confirmed: the records give "unknown" for the SST-2 card and note that the GLUE card refers to the original licences |
| TREC | `CogComp/trec`, revision `65752bf53af25bc935a0dce92fb5b6c930728450`, files `default/train/0000.parquet` and `default/test/0000.parquet` | `data/hf_cache/hub/datasets--CogComp--trec/snapshots/<revision>/` | `-A6-trec`, `-A7-trec`, `-A7-trec-test` | licence to be confirmed: the records give "unknown (dataset card)" |
| MEN | archive `MEN.tar.gz` of `http://clic.cimec.unitn.it/~elia.bruni/MEN`; no version beyond the hash of the archive | `data/concept_eval/men/` | `-A6-men` | Creative Commons Attribution 2.0, according to the record (`licence.txt` inside the archive) |
| SimLex-999 | `https://fh295.github.io/SimLex-999.zip`; no version beyond the hash of the archive | `data/concept_eval/simlex999/` | `-A6-simlex999` | licence to be confirmed: the record says that neither the dataset page nor the archive states one |
| WordSim-353 | `https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip`; no version beyond the hash of the archive | `data/concept_eval/wordsim353/` | `-A6-ws353` | Creative Commons Attribution 4.0 International, according to the record (stated on the dataset page) |
| WordNet 3.0 | the nltk_data package `https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip` | `data/concept_eval/nltk_data/corpora/wordnet.zip` | `-A6-wordnet` | WordNet 3.0 licence, according to the record (text inside the archive) |
| Fashion-MNIST, train split | `zalando-datasets/fashion_mnist`, revision `531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2`, file `fashion_mnist/train-00000-of-00001.parquet` | token store `data/fashion_mnist_p4/bpe2000/train/`; the download location is not recorded | `-A7-fashion_mnist` | MIT, according to the record (dataset card) |

### WikiText-103 and what is built from it

`studies/tm_language_model/experiments/m2_build_data.py` builds everything of the first six rows in one logged run, with the settings of `studies/tm_language_model/experiments/configs/M2_data.json`; its docstring gives the command. The script loads the dataset with the `datasets` library from the cache `data/hf_cache`. It sets the Hugging Face offline flags `HF_DATASETS_OFFLINE` and `HF_HUB_OFFLINE` unless the environment already sets them, so the dataset has to be in that cache beforehand, or both flags have to be set to 0 for the run. No revision is passed to the library and none was recorded; the raw text is pinned by its content hashes only.

The raw store holds, per split, the exact UTF-8 text that is tokenized (`text.utf8`), the byte offsets of the articles (`doc_bytes.npy`) and the article titles (`titles.json`). A token store holds seven arrays per split and vocabulary size (`tok`, `doc_start`, `word_id`, `kind_map`, `stop`, `sym_freq` and `targets`, each as `.npy`). The cluster codes are rows 1 to 3 of `kind_map.npy`: a three-level, 16-way exchange clustering on the bigram contexts of the train split (`studies/tm_language_model/src/tmlm/data/clusters.py`). The tokenizers are trained on the train split only; the cluster codes, the stop list and the symbol frequencies are computed from the train split and copied into every split folder. The evaluation mask sets are built from fixed seeds for the validation and the test split (`studies/tm_language_model/src/tmlm/data/evalsets.py`), the paragraph pairs inside one split at a time (`studies/tm_language_model/src/tmlm/data/pairs.py`), and the contamination files per evaluation split (`studies/tm_language_model/src/tmlm/data/contamination.py`). The [evaluation protocol](../studies/tm_language_model/docs/06-evaluation-protocol.md) says how these files are used.

The last step of the build writes the main hash record once and never rewrites it. The record in the repository was written on 2026-09-22 by run `20260922T160257Z_M2_data` and holds 120 hashes (`metrics.json` of that run, `prereg_hashes_n_artifacts`); its keys are paths relative to the data folder. Every loader of a test split appends a line to `data/test_access_log.jsonl`, the counter of test-set accesses, which is local like the rest.

### Penn Treebank

The same script builds the Penn Treebank store in its last step, in a child process with the offline flags switched off. The text is word level with `<unk>`, one sentence per row. It is tokenized with the WikiText-103 tokenizers and reuses the WikiText-103 cluster codes; no tokenizer is trained on it.

The main hash record holds no Penn Treebank hash, because the script writes the record before this step. In run `20260922T160257Z_M2_data` the step then failed with a connection error under offline mode (`ptb_error.json`; `ptb_done` is 0 in `metrics.json`). The store was built by the separate run `20260922T165603Z_M2_data_ptb` (`ptb_done` is 1), with 42,068 training, 3,370 validation and 3,761 test rows (`metrics.json`: `ptb_raw_train_rows`, `ptb_raw_validation_rows`, `ptb_raw_test_rows`). The hashes of the store are in `dataset_hashes.json` of that run and in no file of the repository. The three hashes of the raw text are:

| File under `data/ptb/raw/` | SHA-256 in `dataset_hashes.json` of `20260922T165603Z_M2_data_ptb` |
|---|---|
| `train/text.utf8` | `5145926136ee9aef6f359b267ac09cc8a920879cd71725de17c490dd111d2998` |
| `validation/text.utf8` | `fadf6277290823f881b7bf39b84e88536c87a859cd629dcfd1d2dfdef06097cf` |
| `test/text.utf8` | `c2f8c16a611595d31da5acdb7a61d50e7d5e95f6b9c05fceda5f5ddc4383e791` |

### BLiMP

`studies/tm_language_model/experiments/cvtm_pilot_p1.py`, in its mode `blimp-hashes`, downloads BLiMP at the pinned revision into the Hugging Face cache, compares the SHA-256 of every file with the id that the host publishes, and writes the record once. The record lists 67 files with 67,000 pairs (fields `n_files` and `n_pairs`; the check `check.blimp.67_paradigms_of_1000_pairs` in `metrics.json` of run `20260924T105854Z_cvtm_pilot_p1_blimp_hashes` holds). It also fixes the split of hypothesis H9.4, one half of each paradigm for development and the other half reported once, by its rule and by the hashes of both pair lists.

### AG News, SST-2 and TREC

Three scripts handle the text classification tasks of the self-supervised family. `studies/tm_language_model/experiments/ssl_transfer_early.py`, stage `data`, downloads the pinned train files of the three tasks and the SST-2 validation file into `data/hf_cache/hub` and writes the three `-A6-` records. `studies/tm_language_model/experiments/ssl_r1_pretext.py`, stage `data`, writes the three `-A7-` records, which cover the train files and the carve of `studies/tm_language_model/src/tmlm/ssl/downstream.py` into a development set, a label pool and, for SST-2, a confirmatory set. That module opens the registered train file of a task and refuses every other file. `studies/tm_language_model/experiments/ssl_r1_confirm.py`, stage `testhash`, writes the two `-test` records from the file metadata that the host publishes, without downloading content; the test files of AG News and TREC are fetched only at the counted access of the confirmatory evaluation (`studies/tm_language_model/src/tmlm/ssl/confirm.py`).

### Fashion-MNIST

`studies/tm_language_model/experiments/ssl_r4_images.py`, stage `data` with the option `--write-prereg-hash`, downloads the train file only, checks its SHA-256 against the id that the host publishes, turns the images into token documents (`studies/tm_language_model/src/tmlm/ssl/patch_tokens.py`) and writes the record once. The store holds 60,000 images, of which 10,000 form a development set carved by a hash rule (`metrics.json` of run `20260924T220147Z_ssl_r4_data`: `store.n_images`, `store.n_dev`). The test split of the dataset is never downloaded.

### Word similarity sets and WordNet

MEN, SimLex-999, WordSim-353 and WordNet 3.0 are used to evaluate word representations (`studies/tm_language_model/experiments/ssl_concepts_early.py` and the confirmatory evaluation). Each record names the address of the archive, the unpacked files with their sizes and hashes, the splits and the licence. To rebuild a folder, the archive is fetched from that address and unpacked to the paths in the record; for MEN the record notes that repeat downloads on 2026-09-24 returned error pages. For MEN only the development file is read by the exploratory evaluation; the test pairs are read through the counted path of the confirmatory evaluation (amendment A9, item 6).

### Synthetic data

The synthetic corpora of the engine tests and of the first gates, and the artificial world of the hierarchy test of amendment A7, are generated by code from fixed seeds (`studies/tm_language_model/src/tmlm/data/synthetic.py`, `studies/tm_language_model/src/tmlm/ssl/hao_world.py`). They need no download and have no hash record.

### Datasets that the documents name without a hash record

Amendment A4 registers three further sources, CLUTRR `gen_train23_test2to10`, a seeded synthetic kinship generator and PG-19, and says that their hashes go into one new record named `preregistration-data-hashes-A4.json`. No record of that name exists; the only A4 record is the BLiMP one. The amendment gives the licences as CC BY-NC 4.0 for the CLUTRR generator, with "unknown" on the dataset card, and Apache-2.0 for PG-19. Amendment A6 names `wikimedia/wikipedia` 20231101.en and FineWeb-edu for its data ladder (family H12), again without a hash record. Section V5 of the [verified code facts](../studies/tm_language_model/docs/02-verified-code-facts.md) lists the dataset ids of the initial design (among them R8, R52, IMDB, CondaQA, NaN-NLI and LAMBADA); none of them appears in a hash record. The [results page](../studies/tm_language_model/docs/results.md) gives the status of the hypotheses that would have used these sources.

## studies/hierarchical_tm

The study tests the procedure of the supervisor's project proposal (unpublished). It keeps its data under `data/` in its own folder and builds most of it from data of the tm_language_model study, which it finds as the sibling folder `../tm_language_model`. The hash record of the study is [frozen_settings_hashes.json](../studies/hierarchical_tm/frozen_settings_hashes.json): its key `data_sha256` pins 17 data files, all of them outside the repository. Two data cards are published, for the [Wikipedia pairs](../studies/hierarchical_tm/docs/data_card_wikipedia_pairs.md) and for [AG News](../studies/hierarchical_tm/docs/data_card_agnews.md).

| Data | Source | Built by | Expected at | Pinned files |
|---|---|---|---|---|
| Wikipedia paragraph pairs, versions `sa` (same article), `np` (next paragraph) and `ns` (next paragraph, shared words removed, swap design) | train split of WikiText-103 raw, read from `../tm_language_model/data/wikitext103/raw/train` | `studies/hierarchical_tm/src/build_pairs.py` | `data/sa/`, `data/np/` and `data/ns/`, each with `train.npz`, `dev.npz` and `test.npz`; `data/vocab_F1024.json`, `data/vocab_F3072.json`, `data/splits.json`, `data/paragraphs_F3072.npz`, `data/build_stats.json` | the three files of `sa` and of `np`, `vocab_F1024.json`, `splits.json`, `paragraphs_F3072.npz`; no file of `ns` |
| Synonym-test pairs `ns_all` (next paragraph, every shared word removed, all pairs) | the same paragraphs | `studies/hierarchical_tm/src/build_ns_all.py` | `data/ns_all/` with `train.npz`, `dev.npz` and `test.npz` | all three files |
| AG News as bags of words | the AG News train file at revision `eb185aade064a813bc0b7f42de02595523103ca4`, read from the Hugging Face cache of the tm_language_model study | `studies/hierarchical_tm/src/build_agnews.py` | `data/agnews/` with `train.npz`, `dev.npz`, `test.npz`, `npmi_ref.npz`, `vocab.json` and `build_stats.json` | the four `.npz` files and `vocab.json` |
| Artificial tests L1 to L3 (planted concept pairs) | generated from seeds | `studies/hierarchical_tm/src/pcp_world.py` and `studies/hierarchical_tm/src/export_pcp.py` | `data/pcp/` | none |
| Sentence family of slide 10 of the project proposal | a templates file that holds sentences of the project proposal; not distributed | `studies/hierarchical_tm/src/slide10.py`, command `build`, from `slide10_templates.txt` in the study folder | `data/slide10/` | none |
| Sentence family of slide 6 of the project proposal | sentences of the project proposal; not distributed | a generator that is not distributed | `data/slide6/` | none |
| WordNet 3.0, for the synonym score of Check 4b | the copy of the tm_language_model study | not built here | `../tm_language_model/data/concept_eval/nltk_data` | pinned by the record `-A6-wordnet` of that study |

The Wikipedia pairs are built from the article store of the tm_language_model study: 28,472 articles with 703,005 paragraphs (`metrics.json` of run `20261001T093726Z_data_build_v3_exploratory`: `n_articles`, `n_paragraphs`). A paragraph is a non-heading line with at least 30 lowercase words, kept as the set of its words after a stop-word list is removed. The articles are split by a fixed seed into train, dev, final test and a reference set for NPMI whose articles enter no pair. The builders aim at 40,000 training, 20,000 dev and 10,000 final-test pairs per version, balanced between the two classes (constant `SIZES` in both builders). The data card describes the three versions `sa`, `np` and `ns` and the checks they passed; it does not cover `ns_all`, which the docstring of its builder describes. `studies/hierarchical_tm/src/data_checks.py` and `studies/hierarchical_tm/src/check_ns_all.py` run the checks, and `studies/hierarchical_tm/src/pairdata.py` loads the pair files.

The AG News data uses the train file of the dataset only; the test file of the dataset is never opened. One permutation of its rows with a fixed seed gives 60,000 training, 10,000 dev and 10,000 final-test texts and 40,000 texts that serve as the NPMI reference (`metrics.json` of run `20261002T035105Z_e5a_data_build_v2`, key `splits`). A text is the set of its words within a vocabulary of 1,024 words (constant `F` in the build script).

The data of the artificial tests is generated: `pcp_world.py` defines the three rungs and writes its self-checks and the hidden groups to `data/pcp/`, and `export_pcp.py` writes the pair files that the training driver reads. Nothing has to be downloaded for it.

The two sentence families are derived from the project proposal. The files that carry sentences of it are held back, and this page gives none of their content. Without `slide10_templates.txt` the build of the slide-10 data stops with a message, and the generator of the slide-6 data is not in the repository, so neither family can be rebuilt from a clone. The study README lists the paths that the code expects.

The eight scripts in `studies/hierarchical_tm/src/feas_slide11/` belong to a feasibility probe of 2026-10-01 that came before the study. Its script `prep_data.py` wrote pair files named `pairs_F<F>.npz` into its own folder; these files are not the pair data of the study.

The pair data is derived from WikiText-103 and the news data from AG News. For both sources the licence is to be confirmed (see the table of the tm_language_model study).

## studies/clause_lm

The study has no hash record in the repository. Its run logger writes the SHA-256 of the data files of a run into `data_hashes.json` in the run folder (`studies/clause_lm/src/runlog.py`), and the run folders are not distributed, so the values are printed here. The registration of M5 step 1 is [docs/prereg_M5_step1.md](../studies/clause_lm/docs/prereg_M5_step1.md), and the [study README](../studies/clause_lm/README.md) has a longer section on the data.

| Data | Source and version recorded | Obtained by | Expected at | Hash and where it is recorded | Licence |
|---|---|---|---|---|---|
| Penn Treebank text | the raw store of the tm_language_model study, see above | built there | `../tm_language_model/data/ptb/raw/<split>/text.utf8` | train and validation text as in the table above; `data_hashes.json` of `20261002T170451Z_ngram_baselines` holds the same two values | licence to be confirmed |
| Pair data (derived) | adjacent word pairs of the Penn Treebank text among the K = 1000 most frequent training words | `studies/clause_lm/src/pairs_data.py` | `data/ptb_pairs_K1000.npz` and `data/vocab_K1000.json` | pair file `be3836fc4b1a7a8f8578a0557779256e1018560875703da331bead90d207bdc4`, in `data_hashes.json` of `20261002T171904Z_pair_baselines_K1000` | derived from Penn Treebank |
| CLINC150 | `https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json`, branch `master`; commit not recorded | downloaded on 2026-10-02; `studies/clause_lm/src/m5_setup.py` names the address and the path and records the hash of the file it finds there | `work/Clause-Driven-LLM/data/clinc150_data_full.json` | `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`, 2,495,390 bytes, in `data_hashes.json` and `metrics.json` of `20261002T195152Z_m5_setup_sorted` | licence to be confirmed |
| Lexical features of CLINC150 (derived) | computed from the CLINC150 file by the feature script of Clause-Driven-LLM | `studies/clause_lm/src/m5_setup.py` | `work/m5/emb_<label order>/clinc150__lexical.npz` | `npz_sha256` in `metrics.json` of the setup runs | derived from CLINC150 |

`studies/clause_lm/src/ptb.py` reads the file `text.utf8` of a split, appends `<eos>` to every sentence and treats the split as one token stream; its loader refuses the test split. `studies/clause_lm/src/pairs_data.py` keeps the pairs whose two words are both among the K most frequent words, adds one negative pair per positive pair with a second word drawn from a smoothed unigram distribution, and marks every validation pair as seen or unseen in the training text. Run `20261002T171456Z_pairs_data_K1000` built 1,158,614 training pairs and 93,766 validation pairs, positives and negatives together (`metrics.json`: `n_train_pairs`, `n_valid_pairs`).

The CLINC150 file sits inside the work copy of Clause-Driven-LLM, which is not distributed; the page of [upstream repositories](upstream-repositories.md) says how that copy relates to the public repository. Banking77, a second dataset that Clause-Driven-LLM supports, was not downloaded (`pytest_test_data.txt` of run `20261002T195152Z_m5_setup_sorted` reports its test as skipped for that reason).

## Model files

Trained models are run outputs. The clause-bank checkpoints and baseline checkpoints of the tm_language_model study and the saved models of the hierarchical_tm study sit in the run folders, and the training outputs of M5 step 1 of the clause_lm study sit under `work/m5`. None of them is distributed. The [run index](run-index.md) lists every run folder with its size.

## Known problems of the data

The following points are stated as the records of the studies state them.

For the tm_language_model study:

- WikiText-103 has no recorded revision; the raw text is pinned by content hashes only.
- The article store has 28,472 training articles (`metrics.json` of run `20260922T160257Z_M2_data`, `raw_train_articles`), where the dataset page gives 28,475 according to section V5 of the verified code facts. The docstrings of `studies/tm_language_model/src/tmlm/data/corpus_store.py` give the reason: three headings are broken over two rows in the raw file, and the cutting rule merges these three articles into their predecessors.
- The validation and test splits share 8-grams and 13-grams with the train split (run `20260922T160257Z_M2_data` counts them in `metrics.json`, keys `contamination_8000_*`). The contamination files record the overlap per document, and the evaluation protocol requires every test number on the full test set and on the subset without the overlapping documents.
- Penn Treebank is pinned to a ref name and not to a commit, its first build failed, and its hashes are in no file of the repository (see above).
- The record of MEN marks its provenance as unverified: the local copy came from a download whose log was not found, and repeat downloads on 2026-09-24 returned error pages, so the archive could not be compared with a source.
- The validation split of SST-2 was read by exploratory runs of 2026-09-24. Amendment A7 therefore bars it from every confirmatory number and uses a hash-carved 5 percent of the SST-2 train split as the confirmatory set.
- The host publishes no SHA-256 for the TREC files, which it stores in git. Amendment A10 makes the published git blob id and the file size the reference for the TREC test file.
- The records disagree on the licence of AG News: the `-A6-` record gives "unknown (dataset card)", the `-A7-` records and amendment A7 give "non-commercial research".
- BLiMP sits in the default Hugging Face cache and not under `data/`, so a copy of the data folder does not contain it.
- Amendments A4 and A6 register data sources for which no hash record exists (see above).

For the hierarchical_tm study:

- The 27 probe runs of 2026-10-01 whose ids contain `feas_slide11` used pair data with a sampling bias. They are flagged as biased exploratory in the run index and are not used; the pair data of the study was built anew by `studies/hierarchical_tm/src/build_pairs.py`.
- The pair data in use is the third build. The first two builds did not pass Check 1b of [gates.md](../studies/hierarchical_tm/gates.md) for every version; the data card and the [results page](../studies/hierarchical_tm/docs/results.md) give the history with the runs.
- The check script prints FAIL for `ns_all`, the main synonym data, on its count of skewed words. The addendum of gates.md of 1 Oct 2026 (about 20:10 CEST) accepts the count, makes `ns_all` the main synonym data and adds the safeguard that the verdict of Check 4b leaves out every word group that contains one of the skewed words. The results page gives the numbers.
- The version `ns` is not a random sample of the next-paragraph pairs: pairs that share many words rarely find a swap partner (data card). It is a side variant.
- The splits are disjoint by article id. An internal audit of 2026-10-03 nevertheless found duplicate articles across the splits of the Wikipedia pair data; its count is 2.1% of the test paragraphs (from an internal audit of 2026-10-03, not recomputed).
- The sentence files of the slide-6 and slide-10 tests are not distributed, so these two tests cannot be rerun from a clone. The results page carries the disclosures that the records attach to results on this data.

For the clause_lm study:

- No run records the Penn Treebank test text among its data files, and the loader refuses the test split.
- CLINC150 is pinned by a branch name and by the hash of the downloaded file, not by a commit.
- The data module of Clause-Driven-LLM that loads CLINC150 is absent from the public repository and was rebuilt for this study; the rebuilt file is published, and the study README describes it.
- The feature vocabulary of the lexical features has ties at its cut-off, which the setup script notes can resolve differently across numpy versions and CPUs (study README).

The page of [known limitations](known-limitations.md) collects the limits that concern the programme as a whole, and the [hash table](hash-table.md) lists the pinned files that are in the repository.
