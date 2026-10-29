# Data of the decision_layer study

No dataset, derived data file or model file of this study is distributed. The folders `data/` and `results/` of the study are local and ignored by git, and the pretrained models sit in the Hugging Face cache outside the repository. The repository holds the code that obtains or builds each input, the registrations that record its hashes, and this card; the logs that this card names as hash records (`logs/*.log`, `logs/*.out` and `logs/multiwoz22_git_blobs_expected.txt`) stay local, like `data/` and `results/`. For each dataset, pretrained model and word table the card says what it is used for, where it comes from, which version a file on disk records, which script obtains or builds it, where the code expects it, which record pins it, and what is known about its licence. The [data page of the repository](../../../docs/data.md) covers the other studies.

Three conventions hold on this page. Paths are relative to the study folder `studies/decision_layer/`, for scripts, records and data alike. "Not recorded" means that no file of the study gives the value: no script, module, registration, log or run record, and no small note or manifest inside the local `data/` folder. A licence is named only where such a file states one; otherwise the entry reads "licence to be confirmed". Registered tests are named by their ids (R2 to R24); "e" numbers are exploratory runs on validation or dev splits. The [results page](results.md) gives every result, and the [study README](../README.md) gives the questions.

Most third-party downloads are kept under `data/untrusted/`; the three intent sets of phase 1 are in `data/raw/`. The code parses them as JSON or parquet, the RouterBench pickle goes through a restricted unpickler, and the copy of the ABCD paper's code under `data/untrusted/abcd/` was read and never run.

## Datasets

| Dataset | Used by | Source and version recorded | Obtained by | Expected at | Hash record | Licence |
|---|---|---|---|---|---|---|
| CLINC150 | R2, R2b, R2e, R4 (with R1), R4b, R5, R5c, R6, R7, R7b, R7c, R8 | `clinc/oos-eval`, file `data_full.json` (`src/dl/data.py`); address, commit and download date not recorded | not recorded | `data/raw/clinc150_data_full.json`; the 10 domains of R5 in `data/raw/clinc150_domains.json`, from `domains.json` of `clinc/oos-eval` | `logs/embed.log` | licence to be confirmed |
| Banking77 | R2c, R2d, R4, R5, R5b, R5d, R7, R7b, R7c | `legacy-datasets/banking77`, parquet files of the official split (`src/dl/data.py`); revision not recorded | not recorded | `data/raw/banking77_train.parquet`, `banking77_test.parquet` and `banking77_label_names.json` | `logs/embed.log` (the two parquet files) | licence to be confirmed |
| HWU64 | R2c, R2d, R4, R5, R5b, R7, R7b, R7c | `FastFit/hwu_64`, parquet files of the DialoGLUE split (`src/dl/data.py`); revision not recorded | not recorded | `data/raw/hwu64_train.parquet`, `hwu64_val.parquet` and `hwu64_test.parquet` | `logs/embed.log` | licence to be confirmed |
| MASSIVE 1.1, en-US and nb-NO | R11 (en-US), R12 (en-US and nb-NO) | the archive `amazon-massive-dataset-1.1.tar.gz`; its address is not recorded | not recorded | `data/untrusted/massive/en-US.jsonl` and `nb-NO.jsonl`, beside the archive and its `LICENSE` | `logs/massive_embed.out` (the two `.jsonl` files) | CC BY 4.0: the `LICENSE` beside the files is the Creative Commons Attribution 4.0 International licence; `src/dl/data.py` and the R11 and R12 registrations say the same |
| ABCD | R9, R9b, R10, R10c, R13 to R17, R16b, R19, R20, R21, R23; dev explorations | `github.com/asappresearch/abcd`, file `data/abcd_v1.1.json.gz`, with `ontology.json`, `guidelines.json` and `kb.json`; downloaded 2026-10-16 (`src/dl/abcd.py`, R9); commit not recorded | the download step is not recorded; `scripts/abcd_prepare.py` builds the examples | `data/untrusted/abcd/` | `logs/abcd_prepare.out` (`abcd_v1.1.json.gz`, `ontology.json`, `guidelines.json`); none for `kb.json` | MIT, as the R9 registration states it |
| STAR | R18, R19, R20, R21, R22; dev explorations (e73 among them) | `github.com/RasaHQ/STAR` (`scripts/star_prepare.py`, R18), as the archive `star_master.tar.gz`, unpacked to `STAR-master/`; commit not recorded | the download step is not recorded; `scripts/star_prepare.py` builds the examples | `data/untrusted/star/` | the hash of the archive, printed below | MIT: `LICENSE.txt` inside the archive; `scripts/star_prepare.py` and R18 say the same |
| MultiWOZ 2.2 | dev explorations from e77 on; R24 (test, first use) | `github.com/budzianowski/multiwoz` at commit `fe0c8e65cfcd8462bd33c86e35f21addc84ca82b`, folder `data/MultiWOZ_2.2`, 24 data files, downloaded on 2026-10-19 between 21:16:47 and 21:17:06 UTC (`SOURCE.txt` in the data folder) | one download per file from `raw.githubusercontent.com` at that commit (`SOURCE.txt`); `scripts/multiwoz_prepare.py` builds the examples | `data/untrusted/multiwoz22/` | `MANIFEST.sha256` in the data folder, whose own SHA-256 is in the R24 registration; `logs/multiwoz22_git_blobs_expected.txt` (git blob id and size of each file) | MIT, as `SOURCE.txt` quotes the `LICENSE` of the repository at that commit |
| RouterBench, zero-shot file | one exploration (e11), then set aside | `withmartian/routerbench`, file `routerbench_0shot.pkl` (`scripts/convert_routerbench.py`); the Hugging Face download metadata beside the file records commit `784021482c3f320c6619ed4b3bb3b41a21424fcb` | the download step is not recorded; `scripts/convert_routerbench.py` converts the file | `data/untrusted/routerbench/routerbench_0shot.pkl`, converted to `data/routerbench/routerbench_0shot.parquet` | none in the repository; the download metadata holds the hub's etag of the file | licence to be confirmed |

### CLINC150, Banking77 and HWU64

`src/dl/data.py` loads the three intent sets from `data/raw/`. Labels are indices into the sorted list of intent names; for Banking77 they keep the official order, and the names come from the Hugging Face metadata of the parquet file, extracted to `banking77_label_names.json`. CLINC150 keeps the file's own splits: 15,000 / 3,000 / 4,500 in-scope queries over 150 intents, and 100 / 100 / 1,000 out-of-scope queries. Banking77 (77 intents) has an official train and test split of 10,003 and 3,080 queries; the validation split is a stratified 10 percent of each class of the official train split (permutation seed 0), which leaves 9,000 / 1,003 / 3,080. HWU64 (64 intents) keeps the DialoGLUE split of 8,954 / 1,076 / 1,076. The bandit tests also use two coarser action sets: the 10 CLINC150 domains of `clinc150_domains.json` and the 18 HWU64 scenarios, the first word of each label (`scripts/bandit_run.py`).

No script of the repository downloads these files. Their SHA-256 values are in the provenance lines that `scripts/embed.py` printed to `logs/embed.log`, once per encoder, and every encoder cache stores them as well. The CLINC150 file has the same SHA-256 as the copy of the clause_lm study, whose source address is on the data page of the repository. `clinc150_domains.json` and `banking77_label_names.json` have no recorded hash.

### MASSIVE 1.1

`src/dl/data.py` reads `en-US.jsonl` and `nb-NO.jsonl` (60 intents) with the file's own train / dev / test partitions: 11,514 / 2,033 / 2,974 utterances in each locale. The nb-NO file is a translation of the same utterances. `scripts/embed.py` embedded both locales with bge-large-en-v1.5, bge-m3 and multilingual-e5-large, without labels; its output, `logs/massive_embed.out`, pins the two `.jsonl` files by their SHA-256. The archive itself has no recorded hash. MASSIVE en-US is related to HWU64 (see the known problems).

### ABCD

`src/dl/abcd.py` reads `abcd_v1.1.json.gz` (10,042 conversations; the published splits 8,034 / 1,004 / 1,004) as JSON and makes one example per action turn: 29,190 / 3,684 / 3,608. The label is the button that the human agent pressed next, one of 30 buttons in the order of `ontology.json`. `scripts/abcd_prepare.py` writes the examples to `data/abcd/examples_<split>.json.gz`, with the SHA-256 of `abcd_v1.1.json.gz`, `ontology.json` and `guidelines.json` inside (key `sources`), and builds the frozen encoder caches. For every action turn it also counts the examples that the paper's Action State Tracking code would make, with the `bert-base-uncased` and `roberta-base` tokenizers, so that the paper's B-Slot accuracy can be recomputed as a weighted accuracy. The written guidelines of the 55 subflows (`guidelines.json`) are read by the scripts that set the learned layer against them (`scripts/abcd_workflow.py`) or add them to a trained TM as clauses (`scripts/abcd_inject.py`); `scripts/lineb_counts.py` reads them and the action sequences of `kb.json` for Line B. The folder also holds `utterances.json`, which no script reads, and the paper's code files in `code/`.

### STAR

`scripts/star_prepare.py` reads the dialogues in `STAR-master/dialogues/` as JSON and makes one example per labelled wizard action, either a suggestion that the wizard picked or an API query: 187 labels. The splits are drawn by a hash of the DialogueID, 80 / 10 / 10 percent, fixed before any model: 54,913 / 6,908 / 6,988 examples in 4,811 / 604 / 611 dialogues (`logs/star_prepare.out`). The reply study e73 also reads the reply templates of each task (`STAR-master/tasks/<task>/responses.json`, `scripts/star_replies.py`). The script writes the SHA-256 of the archive into every example file (key `sources`) and into the provenance of `data/emb/star__bge-large.npz`. The value below is the one that a data check of 2026-10-17 read from the example files and recomputed from the archive, with equal results:

| File | SHA-256 |
|---|---|
| `data/untrusted/star/star_master.tar.gz` | `cb884680f89aba6905eb24769470e2c7cda1e6b05bd535c901d7f7c3739970c8` |

### MultiWOZ 2.2

`SOURCE.txt` in the data folder records the download: the 24 data files of `data/MultiWOZ_2.2` (README, schema, dialogue acts, and 17 train, 2 dev and 2 test dialogue files) at the commit above; the conversion script and the requirements file of that folder were not taken. The git blob SHA-1 and the size of every file equal the values that the GitHub tree API gives at that commit (the expected list is `logs/multiwoz22_git_blobs_expected.txt`), and `MANIFEST.sha256` holds the SHA-256 of every file. The R24 registration pins that manifest by its own SHA-256, `87d8b2cd220b8f2cd2f9527f2a223b658151a5cb3f3bd810bbcd4c50209e58cc`, and lists the hashes of the files its runs read.

`scripts/multiwoz_prepare.py` makes one example per system turn that has a dialogue-act annotation. The label is the sorted set of the turn's act names; a combination with fewer than 20 training occurrences becomes OTHER, a rule fixed before any count. This gives 128 labels and 56,581 / 7,359 / 7,355 examples in 8,437 / 1,000 / 1,000 dialogues (`logs/e77_prepare.out`, `logs/R24_prep.out`). The test examples are written only with the option `--with-test`, which was used after the R24 freeze; the test texts were embedded then (`scripts/multiwoz_embed.py --add-test`), and the train and dev arrays of the cache were checked byte for byte against the dev-stage file, which is kept as `data/emb/multiwoz__bge-large.devstage.npz`.

### RouterBench

RouterBench served one exploration: per prompt, routing to a cheaper LLM or to a stronger one. It was then set aside (study README). The file is a pickle; `scripts/convert_routerbench.py` converts it to parquet with an unpickler that resolves only the pandas and numpy functions that rebuild a data frame and raises on anything else. `scripts/rb_run.py` reads the parquet file (36,497 prompts, a random 60 / 10 / 30 split with seed 0) and caches MiniLM embeddings of the first 256 tokens of each prompt in `data/routerbench/emb_minilm.npy`. The download metadata (`.cache/huggingface/download/routerbench_0shot.pkl.metadata` beside the file) records the commit in the table, the hub's etag of the file, `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`, and the time 2026-10-15 22:13:26 UTC.

## Pretrained encoders and tokenizers

The models are loaded by hub id through `sentence-transformers`, and the two tokenizers through `transformers`; both libraries keep them in the Hugging Face cache, outside the study folder. Many run commands set `HF_HUB_OFFLINE=1` (for example the command in the docstring of `scripts/static_prepare.py`), so a model has to be in that cache before such a run. Only potion-base-32M is pinned: `src/dl/static_text.py` checks the SHA-256 of its `model.safetensors` and `tokenizer.json` in the local cache at the pinned revision and loads the model at that revision. `scripts/embed.py` passes no revision for any other encoder. The caches written by `scripts/embed.py`, `scripts/abcd_prepare.py`, `scripts/star_prepare.py` and `scripts/multiwoz_embed.py` hold a provenance record with the hub id; all embeddings are L2-normalised.

| Model | Used by | Hub id | Revision recorded | Loaded by | Licence |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | R4 (with R1), R4b, R5, R5b, R5c, R5d, R6, R8; phase-1 explorations; the RouterBench exploration | `sentence-transformers/all-MiniLM-L6-v2` | not recorded | `scripts/embed.py` (name `minilm`), `scripts/r6_run.py`, `scripts/r8_run.py`, `scripts/rb_run.py` | licence to be confirmed |
| all-mpnet-base-v2 | exploratory baselines on the validation splits (`scripts/baselines.py`) only | `sentence-transformers/all-mpnet-base-v2` | not recorded | `scripts/embed.py` (`mpnet`) | licence to be confirmed |
| bge-base-en-v1.5 | R4 (with R1); exploratory baselines; the encoder choice of R7 (e19); the teacher of potion-base-32M and of the teacher-refined table | `BAAI/bge-base-en-v1.5` | not recorded for the encoder caches; `a5beb1e3` for the teacher of the refined table (R24, `logs/plan_e82_refined_table_agent.md`) | `scripts/embed.py` (`bge-base`) | MIT, as R24 and `logs/plan_e82_refined_table_agent.md` state it for the teacher |
| bge-large-en-v1.5 | R7, R7b, R7c, R11; R9, R9b, R10, R10c, R13 to R17, R16b (ABCD); R18 (STAR); the bge-large agent of R19 to R24; dev explorations | `BAAI/bge-large-en-v1.5` | not pinned; the R24 registration records `d4aa6901d3a41ba39fb536a557fa166f842b0e09` as the only local snapshot before its freeze | `scripts/embed.py`, `scripts/abcd_prepare.py`, `scripts/star_prepare.py`, `scripts/multiwoz_embed.py`, `scripts/static_speed.py` | licence to be confirmed |
| e5-large-v2 | exploratory baselines only | `intfloat/e5-large-v2`, text prefix `query: ` | not recorded | `scripts/embed.py` (`e5-large`) | licence to be confirmed |
| mxbai-embed-large-v1 | exploratory baselines; the encoder choice of R7 (e19) | `mixedbread-ai/mxbai-embed-large-v1` | not recorded | `scripts/embed.py` (`mxbai-large`) | licence to be confirmed |
| Qwen3-Embedding-0.6B | exploratory baselines; e37 (ABCD dev, in place of bge-large-en-v1.5) | `Qwen/Qwen3-Embedding-0.6B` | not recorded | `scripts/embed.py` and `scripts/abcd_prepare.py` (`qwen3-0.6b`) | licence to be confirmed |
| bge-m3 | R12 (chosen on the nb-NO dev split); MASSIVE caches | `BAAI/bge-m3` | not recorded | `scripts/embed.py` (`bge-m3`) | licence to be confirmed |
| multilingual-e5-large | the other encoder tried for R12 on dev; MASSIVE caches | `intfloat/multilingual-e5-large`, text prefix `query: ` | not recorded | `scripts/embed.py` (`me5-large`) | licence to be confirmed |
| potion-base-32M, a Model2Vec table of 63,091 tokens x 512 | R19, R20, the potion agent of R21; phase-3 dev explorations | `minishlab/potion-base-32M` | `1e5a03f8eeb2c98b928fbbd846f22f816360919f`, with the SHA-256 of `model.safetensors` and `tokenizer.json` checked at load | `src/dl/static_text.py`, `scripts/static_prepare.py` | licence to be confirmed |
| bert-base-uncased and roberta-base, tokenizers only | the AST count behind the B-Slot accuracy reported for R9 and R19 | `bert-base-uncased`, `roberta-base` | not recorded | `scripts/abcd_prepare.py` | licence to be confirmed |

Beside the two file hashes, the R19 registration gives the SHA-256 of the potion table's array of floats (`5e33b584aad09dc4c80d8c8ece6dc98768af7a3f4fe3a0eabcf4895bba3e0487`), and it states that the table was distilled once, offline, from bge-base-en-v1.5 with Model2Vec.

### What the local download cache shows (dated note of 2026-10-25)

The Hugging Face cache of the machine is not a file of the study, so the tables above keep "not recorded" and "licence to be confirmed" by the conventions of this page. This note gives what that cache showed when it was read on 2026-10-25. For the encoders, the snapshot is the one that `refs/main` of the cache names; except for potion-base-32M, no file shows that a run loaded that snapshot, because `scripts/embed.py` passes no revision.

| Model or dataset | Snapshot in the cache | Licence in the card of that snapshot |
|---|---|---|
| all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | apache-2.0 |
| all-mpnet-base-v2 | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | apache-2.0 |
| bge-base-en-v1.5 | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | mit |
| bge-large-en-v1.5 | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` | mit |
| e5-large-v2 | `f169b11e22de13617baa190a028a32f3493550b6` | mit |
| mxbai-embed-large-v1 | `b33106f585b9ce46904ad7443a3b52b7a63e231c` | apache-2.0 |
| Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | apache-2.0 |
| bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181`; a second snapshot, `9a0624b896d81da7492a910ffa53731274b6cf3d`, holds no card | mit |
| multilingual-e5-large | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | mit |
| potion-base-32M | `1e5a03f8eeb2c98b928fbbd846f22f816360919f`, the pinned revision | mit |
| bert-base-uncased and roberta-base, tokenizers | `86b5e0934494bd15c9632b12f734a8a67f723594` and `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | no card in the cache |
| Banking77 | `f54121560de48f2852f90be299010d1d6dc612ec` of `legacy-datasets/banking77`; its two parquet files have the SHA-256 of `data/raw/banking77_train.parquet` and `data/raw/banking77_test.parquet` | no card in the cache |
| HWU64 | `9fb5bff1e37c4c5a6d46c1ff6f286b6d3f362222` of `FastFit/hwu_64`; its three parquet files have the SHA-256 of `data/raw/hwu64_train.parquet`, `hwu64_val.parquet` and `hwu64_test.parquet` | no card in the cache |

CLINC150, MASSIVE, ABCD, STAR and RouterBench have no copy in the cache, so the cache adds nothing for them.

## Word tables from C4

Two static word tables were built outside the study and copied into `data/tables/`. Each folder holds `table.npz` (the words and their rows), `words.txt` and `vectors.npy` (the same words and rows as plain files), the reference tokenizer `tokenize_ref.py`, a README and `MANIFEST.sha256`. Their build code and records belong to the text-understanding line of the project and are in the project notes (not published). The study reads no C4 text itself.

| Table | Used by | What it is | Built from | Expected at | Pinned by | Licence |
|---|---|---|---|---|---|---|
| `c4words_S1000_v1`, the C4 counting table | e72 and later dev explorations; R21, R22, R23 and the C4 agent of R24 | 63,091 words x 500 float16 rows of unit length, from counting and SVD (PPMI-SVD); no neural network at any stage | C4 en, `allenai/c4` at `1588ec45`, shards 00000 to 00007, deduplicated against every evaluation split, the dev and test turns of ABCD and STAR included; the word list is the 63,091 most frequent words of 1B tokens of that text | `data/tables/c4words_S1000_v1/` | `table.npz` `e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a`, in `src/dl/count_text.py`, in R21 and in R24 (R22 gives its first 8 characters; R23 reads the table through the same module) | licence to be confirmed |
| `c4words_a3B_v1`, the teacher-refined table | e81 to e85 (dev); the refined agent of R24; R25, which was never frozen | the same 63,091 words, with rows refitted by least squares so that the mean row of a C4 sentence matches the sentence embedding of bge-base-en-v1.5 ([CLS], L2-normalised); 768 dimensions, unit rows; network-free at run time, teacher at training time | 1,999,932 C4 sentences, 39,995 of them held out; lambda* 0.01; mean cosine to the teacher on the held-out sentences 0.8116 (the linear map alone: 0.6319) | `data/tables/c4words_a3B_v1/` | `table.npz` `4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855`, in R24 and in `logs/plan_e82_refined_table_agent.md` | licence to be confirmed |

The C4 counting table comes from the W1 study of the text-understanding line; it is the table of that study's registered test W1-T. It was copied into the study on 2026-10-19 at 04:50 UTC, and `sha256sum -c MANIFEST.sha256` passed for all 5 files after the copy (`data/tables/README.md`). The teacher-refined table comes from step 3a of the same line. Its word list and its tokenizer are byte-identical to those of the counting table (`words.txt` `49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29`, `tokenize_ref.py` `2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803`). Its README and manifest were replaced on 2026-10-21 by versions that add a caveat on the teacher's training data; the first versions are kept beside them, and `table.npz` did not change. `logs/plan_e82_refined_table_agent.md` lists the SHA-256 of every file of that folder.

`src/dl/count_text.py` ports the reference tokenizer: lower case; "n't" and the endings 's, 're, 've, 'll, 'd and 'm split off; the tokens are the matches of `n't|'[a-z]+|[a-z0-9]+|[^\sa-z0-9]`; a token that is not in the word list has no vector and is dropped. The module refuses a `table.npz` whose SHA-256 differs from the registered value or from the manifest, and a `tokenize_ref.py` that differs from the reference, and it reads the table with numpy only (`allow_pickle=False`). The counting table is registered in that module; `scripts/newtable_run.py` registers another folder of `data/tables/` for one process after the same checks, and the runs on the refined table go through it. While a table is in use, `scripts/table_run.py` makes the potion table impossible to load and requires `--encoder none`, so that no encoder vector is read. The C4 mean vectors of e79 are averages of rows of the counting table (see the derived inputs).

## Derived inputs built by the study

All of them are local: they sit in `data/` or `results/`, which git ignores.

| Derived input | Built by | Location | Notes |
|---|---|---|---|
| Next-action examples of ABCD | `scripts/abcd_prepare.py` with `src/dl/abcd.py` | `data/abcd/examples_<split>.json.gz` | text fields `cust`, `recent` and `full`, the earlier buttons, the label and the AST counts; the SHA-256 of three source files inside |
| Next-action examples of STAR | `scripts/star_prepare.py` | `data/star/examples_<split>.json.gz` | the same format; `full` is empty |
| Next-action examples of MultiWOZ 2.2 | `scripts/multiwoz_prepare.py` | `data/multiwoz/examples_<split>.json.gz`; label reports `data/multiwoz/labels.json` (train and dev) and `labels_test.json` | the same format; `full` is empty; the script refuses to overwrite |
| Encoder caches | `scripts/embed.py` (intent sets, MASSIVE), `scripts/abcd_prepare.py`, `scripts/star_prepare.py`, `scripts/multiwoz_embed.py` | `data/emb/<dataset>__<encoder>.npz` | float32, per split, and per text field for ABCD, STAR and MultiWOZ; provenance record inside |
| Static-table caches | `scripts/static_prepare.py` | `data/emb/abcd__potion-32m.npz`, `data/emb/star__potion-32m.npz` | normalised mean of the potion token vectors; the provenance holds the revision, the file and table hashes and the hashes of the example files; hashes of both caches in R19 |
| C4 mean caches (e79) | `scripts/c4mean_prepare.py` | `data/emb/<dataset>__c4mean.npz` for STAR, ABCD and MultiWOZ | train and dev only; the test arrays are empty |
| Empty caches | `scripts/none_prepare.py` | `data/emb/<dataset>__none.npz` for ABCD, STAR and MultiWOZ | nine empty arrays, so that a run with `--encoder none` loads no vector; SHA-256 in R21 and R24 |
| Prefix caches of CLINC150 | `scripts/r6_run.py`, `scripts/r8_run.py` | `data/emb/clinc150__minilm__half.npz`, `data/emb/clinc150__minilm__prefix<k>of4.npz` | MiniLM embeddings of the first half of the words of each query (R6) and of the first 25, 50 and 75 percent of its words (R8) |
| Bit caches of the readable views | `src/dl/bitcache.py` | `data/<dataset>/bits_<parts>[__<encoder>]_<split>.npz` | train and dev only, for dev exploration |
| Bit export for the hierarchical TM package | `scripts/abcd_export_bits.py` | `data/abcd/bits_short_<split>.npz` | R9b's short view, read by `scripts/abcd_htm.py` in `.venv-htm` (R16, R16b) |
| RouterBench in parquet; MiniLM embeddings | `scripts/convert_routerbench.py`, `scripts/rb_run.py` | `data/routerbench/routerbench_0shot.parquet`, `data/routerbench/emb_minilm.npy` | see RouterBench above |
| Router caches | `scripts/r6_run.py`, `scripts/r8_run.py` | `results/<out>/r6_router_seed<k>.pt`, `results/r8_router_seed1.pt` | the R8 cache is pinned by its SHA-256 in the R8 registration |

## Model files

Trained models and predictions are run outputs: `results/explore/` holds those of the exploratory runs and `results/confirm/<registration>/` those of the registered runs. None of them is distributed.

## Known problems of the data

The following points are stated as the records of the study state them.

- Several sources are pinned by content hashes only. No script of the repository downloads CLINC150, Banking77, HWU64, MASSIVE, ABCD, STAR or RouterBench. No record gives the revision of Banking77 and HWU64, the address of CLINC150 and MASSIVE, or the commit of ABCD and STAR; only MultiWOZ 2.2 has a written download record (`SOURCE.txt`). The file times of the local copies are of 2026-10-15 (CLINC150, Banking77, HWU64, RouterBench), 2026-10-16 (MASSIVE, ABCD) and 2026-10-17 (STAR).
- Only potion-base-32M is loaded at a fixed revision. For the other models no record gives the revision used, except bge-large-en-v1.5 (the local snapshot recorded in R24) and bge-base-en-v1.5 as the teacher of the refined table.
- `clinc150_domains.json`, `banking77_label_names.json`, `kb.json` of ABCD and the MASSIVE archive have no recorded hash.
- MASSIVE en-US is not independent of HWU64: both come from the same source collection, with different labels (60 against 64 intents). Counted with lower case and normalised white space, 968 of the 2,970 distinct MASSIVE test texts are in HWU64 train and 1,214 in some HWU64 split; 705 of the 2,031 MASSIVE validation texts are in HWU64 train; 470 of the 1,076 HWU64 test texts are in MASSIVE train. No model trained on HWU64 was scored on MASSIVE, but the router settings were developed and tested on HWU64, so MASSIVE is a related dataset, not a fresh one (`registrations/phase2_amendment_2026-10-17.md`).
- Banking77 has no official validation split; the study carves one from the official train split, and STAR's splits are the study's own (see above).
- In MultiWOZ 2.2, system turns without a dialogue-act annotation are left out: 195, 15 and 17 turns in train, dev and test. OTHER covers 3.0 percent of the train, 3.4 percent of the dev and 3.0 percent of the test examples (`logs/e77_prepare.out`, `logs/R24_prep.out`).
- The C4 counting table was deduplicated against ABCD and STAR but not against MultiWOZ. A check on all 2,243,398 C4 documents behind the table found 365 of 14,748 MultiWOZ dev utterances (2.47%) and 385 of 14,744 test utterances (2.61%) that match a C4 sentence verbatim or by a shared 8-gram; the matches are stock service phrases. The overlap is declared, and the table was not rebuilt (`logs/plan_e77_multiwoz_dev.md`, R24).
- The MultiWOZ test texts were read twice before R24, text only and without labels: by that overlap check, and by the fit filter of the refined table, which read 29,492 dev and test utterances and removed 68 training and 5 held-out fit sentences. Both reads took the text of every turn, the system turns included, whose words express the labels (R24).
- The training data of bge-base-en-v1.5, the teacher of potion-base-32M and of the refined table, cannot be checked, nor can that of bge-large-en-v1.5. It may include text of the intent sets, ABCD, STAR or MultiWOZ. That could inflate the refined table's gain over the counting table, and on MultiWOZ also its results against bge-large-en-v1.5, which is of the same family; it does not inflate a comparison with potion-base-32M, which has the same teacher (R24, `logs/plan_e82_refined_table_agent.md`).

The [results page](results.md) carries the caveats that the registrations attach to results on these data.
