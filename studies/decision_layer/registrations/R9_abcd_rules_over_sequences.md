# R9 registration: rules over sequences on a real agent workflow (ABCD next action)

The goal's plan: with earlier states or actions added as literals, the agent can learn rules over sequences and not only react to the moment, and every decision can be explained by concrete clauses, for example 'if A and not B, choose action 3'. R6 and R8 tested memory literals in tasks that did not need them (+0.001, -0.012). R9 tests them where the next decision depends on what the agent has already done: customer-service agents who follow written guidelines.

Frozen by `scripts/freeze.py` before any R9 test run. Amendments go in a dated file.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Data and task

- ABCD (Chen et al., NAACL 2021; github.com/asappresearch/abcd, MIT licence; downloaded 2026-10-16 to `data/untrusted/abcd`). 10,042 human-to-human customer-service conversations; agents follow written guidelines for 55 subflows and press 30 action buttons. Published splits: train 8,034 / dev 1,004 / test 1,004 conversations. No model in this study has read the test split.
- One decision per action turn: which button the agent presses next (train 29,190 / dev 3,684 / test 3,608 action turns). This is the button part ("B-Slot") of the paper's Action State Tracking task. The decision layer sees what the agent had seen at that point: the delexed utterances so far and its own earlier button presses (`src/dl/abcd.py`).
- Encoder: frozen bge-large (`BAAI/bge-large-en-v1.5`, max 512 tokens; 0.7% of the dev conversations-so-far are longer), three text fields per decision: the customer's words so far, the last 3 utterances, the whole conversation so far (`scripts/abcd_prepare.py`).

## The readable view (`ABCDView`, parts `mem,intent,step,words,mem2,fstep,fwords`; 2,791 bits on train)

Every bit has a name a person can read:
- memory (`mem`, `mem2`): `done(b)`, `last action = b`, `action before last = b`, `at least k actions so far` (k = 1 to 6);
- intent (`intent`): `sim(customer so far, s) >= t`, the whitened similarity of the customer's words to the centre of subflow s's training examples (shrinkage 0.001; levels at 5/25/50/75% of s's own similarities), and `s is among the r nearest subflow centres` (r = 1, 3); 57 subflows (the scenario's subflow without the FAQ variant number);
- next step (`step`, `fstep`): the same with one centre per next button, on the last 3 utterances and on the whole conversation so far;
- words (`words`, `fwords`): presence of the 1,000 most frequent 1- and 2-grams of the last 3 utterances, and of the whole conversation so far.

The subflow and next-button centres are fitted on the training labels only.

## Exploration behind it (dev, seed 1, `results/explore/abcd.jsonl`)

| Run | Arm | Bits | Dev accuracy | AST-weighted |
|---|---|---|---|---|
| e25_full | TM, view without mem2/fstep/fwords | 1,581 | 0.8398 | 0.8498 |
| e25_nomem | the same TM without memory bits | 1,515 | 0.6414 | 0.6914 |
| e25_full | LR on the same 1,581 bits | 1,581 | 0.8360 | 0.8534 |
| e25_raw | MLP on the same 1,581 bits | 1,581 | 0.8431 | 0.8609 |
| e25_raw | LR on the raw embeddings + memory | 3,138 | 0.8480 | 0.8622 |
| e25_raw | MLP on the raw embeddings + memory | 3,138 | 0.8561 | 0.8734 |
| e25_full | one rule: nearest next-button centre of the last 3 utterances | | 0.6080 | 0.6512 |
| e25_full | lookup table: most frequent next button given (nearest subflow, last action) | | 0.7801 | 0.7950 |
| e27_ep80 | TM, e25's view, 80 epochs | 1,581 | 0.8412 | 0.8530 |
| **e27_more** | **TM, the view above (40 epochs, best epoch 38)** | 2,791 | **0.8504** | 0.8646 |
| e27_more | LR on the same 2,791 bits | 2,791 | 0.8401 | 0.8622 |
| e27_more | MLP on the same 2,791 bits | 2,791 | 0.8493 | 0.8691 |

Chosen: the e27_more view and TM setting. Memory bits lift the TM by about 20 points on dev (e25). On the chosen view the TM is 1.03 points above LR on the same bits and 0.11 above the MLP on the same bits (one dev seed; weighted by the AST counts, +0.24 and -0.45), and 0.57 below the MLP on the raw embeddings (e25_raw, whose memory input lacked the action before last).

## Arms (test)

- **TM-mem:** the view above; 4,000 clauses, T 4,000, s 8, 40 epochs, batch 128, best-dev snapshot, start state half; seeds 1 to 10.
- **TM-nomem:** the same view without the memory parts (`intent,step,words,fstep,fwords`), same setting; seeds 1 to 10.
- **TM-mem composite:** two disjoint 5-machine composites of TM-mem (seeds 1 to 5, 6 to 10; sum of class sums), as in R4 and R7.
- **LR-mem, LR-nomem:** logistic regression on the same bits as TM-mem and TM-nomem; C from the dev grid (0.1 to 100); deterministic, run once.
- **MLP-mem:** one hidden layer (512, ReLU, dropout 0.2, AdamW 1e-3, weight decay 1e-4, batch 256, best dev epoch of at most 60, patience 10; `src/dl/mlp.py`) on TM-mem's bits; seeds 1 to 10.
- **MLP-raw:** the same network on the raw input: the three bge-large embeddings plus the memory bits (done, last, before last, count; 3,168 dimensions). The neural agent head that a person cannot read. Seeds 1 to 10.
- **LR-raw:** logistic regression on the same raw input, C from the dev grid.
- **Rule** and **table:** the two baselines of the exploration table, run once.

Script `scripts/abcd_confirm.py` (job file `logs/jobs_R9.txt`); report `scripts/report_abcd.py`.

## Statistic

Per test action turn, each arm's correctness averaged over its seeds. A difference A - B is the mean over the 3,608 test action turns; its 95% interval comes from a bootstrap over the 1,004 test conversations (all action turns of one conversation are resampled together), 10,000 resamples, seed 0.

## Pass rules (test)

- **R9.1 (memory gives rules over sequences):** TM-mem minus TM-nomem is at least +5.0 points, with the interval above 0.
- **R9.2 (as good as a linear model on the same named inputs):** TM-mem minus LR-mem is at least -0.5 points.
- **R9.3 (as good as a neural head on the same named inputs):** TM-mem minus MLP-mem is at least -1.0 point.
- **R9 is met** if R9.1, R9.2 and R9.3 all hold.
- **R9.4, its own verdict (as accurate as the opaque neural agent head):** TM-mem composite minus MLP-raw is at least -1.0 point.
- **R9.5, its own verdict (better than a linear model on the same named inputs):** TM-mem minus LR-mem has its interval above 0. This is the first registered test in the study in which the TM is asked to beat a linear model; the reason to expect it is that the next step depends on conjunctions (this subflow AND that step done AND the next not done), which a linear model can only add up. One dev seed supports it (+1.03 points); the weighted dev numbers do not (+0.24).

## Reported with no rule

- AST-weighted accuracy: every action turn weighted by the number of examples the paper's AST code makes from it (`ast_count_bert`, reproduced from their `utils/process.py`), which is the paper's B-Slot accuracy. Next to it the published fine-tuned numbers: BERT 89.9, AlBERT 90.9, RoBERTa 93.6 (paper Table 3; the paper does not state the split). Not like for like: those models are fine-tuned end to end on the dialogue text; here the encoder is frozen and only the decision layer learns.
- Accuracy by the position of the action in the conversation (1st to 5th, 6th and later).
- LR-nomem minus LR-mem (the memory effect for a linear model), TM-nomem minus LR-nomem.
- Workflow restatement (`scripts/abcd_workflow.py`): the written guideline transitions a -> b (86 pairs over the 55 subflows) against the pairs ranked by the TM's clauses that contain "last action = a" (summed positive weight for b), by LR's weight of "last action = a" for b, and by the training logs themselves; precision at K = 86.
- Clause statistics and example decisions with the clauses behind them.
