# R12 registration: trained in English, used in Norwegian (MASSIVE en-US to nb-NO)

Can the readable decision layer be built once, in English, and serve a language it never saw? The bits are named by intents ("the query is close to the centre of `alarm_set`"), not by words, so they do not depend on the language if the encoder maps languages to one space. R12 asks whether a Tsetlin machine trained only on English queries routes their Norwegian translations, and how much it loses on the way.

Frozen by `scripts/freeze.py` before any R12 test run. Amendments go in a dated file.

## Data

MASSIVE 1.1 (Amazon, CC BY 4.0), 60 intents. Training, the view, every snapshot and every C: en-US train and dev only. Scored: the nb-NO test partition (2,974 utterances, never used in this study; it is the Norwegian translation of the en-US test partition, which R11 scores), and the en-US test partition as the reference. `src/dl/data.py` names `massive_en`, `massive_nb`.

## Choice made in exploration (dev, seed 1, `results/explore/xling.jsonl`)

| Encoder | TM en dev | TM nb dev | LR same bits nb | LR raw nb | nearest-centre rule nb |
|---|---|---|---|---|---|
| BAAI/bge-m3 | 0.8726 | **0.8165** | 0.8101 | 0.8264 | 0.8151 |
| intfloat/multilingual-e5-large | 0.8697 | 0.7595 | 0.7565 | 0.8175 | 0.7678 |

**Encoder: BAAI/bge-m3** (dense output, 1,024 dimensions, normalised). The choice used the Norwegian dev partition (no Norwegian data is used for training or for any setting of the models below).

## Arms (`scripts/xling_run.py`, one run per seed group)

- **TM-xl:** view `proto:k=1:whiten=0.001` fitted on en-US train (360 bits: whitened similarity to the 60 intent centres at 4 levels, and "intent among the 1 / 3 nearest centres"); 4,000 clauses, T 4,000, s 8, 40 epochs, best snapshot on en-US dev, start state half; seeds 1 to 10.
- **TM-xl composite:** two disjoint 5-machine composites (seeds 1 to 5, 6 to 10; sum of class sums).
- **LR-xl-bits:** logistic regression on the same bits, C from en-US dev.
- **LR-xl-raw:** logistic regression on the raw bge-m3 embedding, C from en-US dev.
- **Rule:** route to the intent whose whitened English centre is nearest.

Statistic: per nb-NO test utterance the mean correctness over seeds; paired bootstrap over test utterances, 10,000 resamples, seed 0, 95% interval (`scripts/report_xling.py`).

## Pass rules (nb-NO test)

- **R12.1 (the TM transfers as well as a linear model on the same bits):** TM-xl minus LR-xl-bits is at least -0.5 points.
- **R12.2 (the cost of readability across languages is at most one point):** TM-xl composite minus LR-xl-raw is at least -1.0 point.
- **R12.3 (it keeps most of its English accuracy):** TM-xl accuracy on nb-NO test is at least 0.90 times its accuracy on en-US test (mean over seeds).
- **R12 is met** if all three hold.

## Reported with no rule

TM-xl single minus LR-xl-raw; the rule's accuracy and the TM's agreement with it; the same numbers on en-US test; example Norwegian decisions with the clauses behind them.
