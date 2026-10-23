# R7 registration: a fully readable TM decision layer on top of an LLM

The second half of the goal: a TM as a **readable** decision layer on top of an LLM. In R4 every literal was a threshold on an embedding dimension (`emb[217] >= 0.31`), which no person can read. In R7 every literal has a name a person can read:

- `contains "pin"`: presence of a word or word pair, the 2,000 most frequent 1- and 2-grams of the train texts;
- `sim_w(q, <intent>) >= t`: the LLM encoder finds the query close to the centre of that intent's training queries. This is cosine in the embedding space whitened by the within-class covariance (shrinkage 0.001), at 4 levels per intent. The levels are quantiles of the intent's own training similarities (5, 25, 50 and 75%).
- `one of the r nearest prototypes is <intent>`, for r = 1 and 3.

Frozen by `scripts/freeze.py` before any R7 run and before any exploratory run on Banking77 or HWU64 with this view. Amendments go in a dated file.

## Choice made in exploration (val, CLINC150 only, seed 1, `results/explore/tm.jsonl`)

| Run | Encoder | Readable TM val accuracy |
|---|---|---|
| e19_rdw_minilm | MiniLM | 0.9463 |
| e19_rdw_bge | bge-base | 0.9603 |
| e19_rdw_bgelarge | bge-large | 0.9700 |
| e19_rdw_mxbai | mxbai-large | 0.9690 |

**Encoder: BAAI/bge-large-en-v1.5.** View `proto:k=1:whiten=0.001+lex:2000`. TM: 4,000 clauses, T 4,000, s 8, 40 epochs, best-val snapshot. These are the settings of `e19_rdw_bgelarge`. On CLINC150 val that run was equal to logistic regression on the same bits (0.9703), and 1.13 points below logistic regression on the raw embedding (0.9813).

## Arms

- **TM-readable:** 10 seeds per dataset.
- **TM-readable-composite:** two disjoint 5-machine composites (seeds 1 to 5, 6 to 10), as in R4.
- **LR-readable:** logistic regression on the same readable bits, C from the val grid. A readable linear model: one weight per named feature.
- **LR-raw:** logistic regression on the raw bge-large embedding, C from the val grid. Opaque.

Datasets: CLINC150, Banking77, HWU64 (splits of `src/dl/data.py`). Test split scored once per run. Statistic as in R4: per-query mean correctness over seeds, paired bootstrap over test queries, 10,000 resamples, seed 0, 95% interval.

## Pass rules (test)

- **R7.1 (the TM learns as well as a linear model from readable features):** TM-readable minus LR-readable is at least -0.5 points in at least 2 of the 3 datasets.
- **R7.2 (the cost of readability is at most one point):** TM-readable-composite minus LR-raw is at least -1.0 point in at least 2 of the 3 datasets.
- **R7 is met** if R7.1 and R7.2 both hold.

Reported with no rule:
- TM-readable (single) minus LR-raw;
- out-of-scope AUROC on CLINC150 (top class sum against LR-raw max softmax);
- calibration (ECE after one temperature);
- clause statistics;
- the median number of firing clauses behind a decision;
- example rules and decision receipts.
