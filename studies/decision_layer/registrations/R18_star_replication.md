# R18 registration: do memory literals pay on a second workflow dataset (STAR)?

R9b showed on ABCD that memory literals ("done X", "last action = Y") lift the TM's next-action accuracy by 18.65 points. R18 asks whether that holds on a different dataset with a test split no test has read: STAR (Mosig, Mehri, Kober 2020; github.com/RasaHQ/STAR, MIT), wizard-of-oz dialogues over 24 tasks with written flowcharts, in which the wizard picks one of 187 labelled actions (including API queries).

Frozen by `scripts/freeze.py` before any R18 test run. First use of STAR's test split.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Data (`scripts/star_prepare.py`)

One example per labelled wizard action (train 54,913 / dev 6,908 / test 6,988 examples; 4,811 / 604 / 611 dialogues; splits by a hash of the DialogueID, fixed before any model). Inputs as R9b's short view (`ABCDView` parts `mem,intent,step,words`): memory bits over the wizard's earlier labelled actions; intent centres of the customer's words (fitted on the scenario's task combination, 36 values, train only); next-action centres and words of the last 3 utterances (customer and wizard text). No whole-dialogue field.

## Exploration behind it (dev, seed 1, `results/explore/abcd.jsonl`, e45 and e46)

| Run | Arm | Dev accuracy |
|---|---|---|
| e45_star_mem | TM with memory, 4,000 clauses | 0.6131 |
| e45_star_nomem | TM without memory, 4,000 clauses | 0.4862 |
| e46_star_mem_c12000 | TM with memory, 12,000 clauses | 0.6297 |
| **e46_star_mem_c16000** | **TM with memory, 16,000 clauses** | **0.6394** (best epoch 18) |
| **e46_star_nomem_c16000** | **TM without memory, 16,000 clauses** | **0.4919** (best epoch 18) |
| e45_star_mem | LR on the memory view's bits | 0.6657 |
| e45_star_nomem | LR on the no-memory view's bits | 0.5229 |
| e45_star_mem | MLP on the memory view's bits | 0.6726 |

On STAR the TM is 2.6 to 3.3 points below LR and the MLP on the same bits even with 16,000 clauses (on ABCD they were level); the memory gain is about 13 to 15 points for every learner.

## Arms (test)

- **TM-mem:** 16,000 clauses, T 16,000, s 8, 40 epochs, best-dev snapshot, start state half; seeds 1 to 5.
- **TM-nomem:** the same without the memory parts (`intent,step,words`); seeds 1 to 5.
- **LR-mem, LR-nomem:** logistic regression on the same bits (C from the dev grid). **MLP-mem:** R9's MLP on the memory view's bits, seeds 1 to 5. (Reported, no rule.)

Script `scripts/abcd_confirm.py --dataset star` (commands in `logs/jobs_R18.txt`). Statistic as R9b: per test action the mean correctness over seeds; bootstrap over the 611 test dialogues, 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`).

## Pass rule (test)

- **R18.1 (memory literals pay on STAR too):** TM-mem minus TM-nomem is at least +5.0 points, with the interval above 0. R18 is met if R18.1 holds.

## Reported with no rule

TM-mem minus LR-mem and minus MLP-mem (on dev the TM is behind); LR's memory gain; accuracy by position.
