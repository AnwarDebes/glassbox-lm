# R13 registration: the closed loop (the memory holds the model's own presses)

R9b and R10 built the memory bits from the human agent's logged presses (teacher forcing). In a real agent loop the memory holds the agent's own choices, and a wrong press can mislead the next decisions. R13 asks whether R9b's memory gain survives that.

Frozen by `scripts/freeze.py` before any R13 test run. Amendments go in a dated file. This is the fifth use of the ABCD test split (R9, R9b, R10, R10c before it). No model is trained for R13: it scores R9b's registered models (10 TM seeds, LR) in a new condition.

## Condition (`scripts/abcd_closed_loop.py`)

In every test conversation the action turns are decided in order. At turn k the memory bits (`done(b)`, `last action = b`, `at least j actions so far`) are built from the model's own choices at turns 0 to k-1; the text bits (customer's words so far, last 3 utterances) stay as logged, because the customer's words cannot be replayed differently. Accuracy is still against the human agent's button.

## Exploration behind it (dev, `results/explore/abcd_closed_loop.jsonl`)

| Run | Model | Open loop (logged memory) | Closed loop (own memory) |
|---|---|---|---|
| e31 | R9b TM-mem-short seeds 1, 2, 3 | 0.8398, 0.8388, 0.8393 | 0.8149, 0.8116, 0.8113 |
| e31 | R9b LR-mem-short | 0.8360 | 0.8097 |
| e25_nomem | TM without memory bits (no loop needed) | 0.6414 | 0.6414 |
| e32_noise | TM retrained with noisy memory copies (p 0.15), seed 1 | 0.8358 | 0.8113 |
| e32_dagger | TM retrained with one round of cross-fitted data aggregation, seed 1 | 0.8303 | 0.8067 |

The two remedies for compounding errors do not help on dev (the logged next button stays the label even when the model's own memory is wrong, so they teach the model to trust its memory less); they are not registered.

## Arms (test)

- **TM-mem-short, closed loop:** R9b's 10 TM-mem-short models (`results/confirm/R9b/tm_mem_short/model_seed<k>.npz`), seeds 1 to 10.
- **TM-mem-short, open loop:** R9b's registered predictions (unchanged).
- **TM-nomem-short:** R9b's registered predictions (it has no memory bits, so open and closed loop are the same).
- **LR-mem-short, closed loop:** R9b's LR weights (reported, no rule).

Statistic as R9b: per test action turn the mean correctness over seeds; differences with a bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`).

## Pass rules (test)

- **R13.1 (memory still pays in the closed loop):** TM-mem-short closed loop minus TM-nomem-short is at least +10.0 points, with the interval above 0.
- **R13.2 (its own mistakes cost at most 4 points):** TM-mem-short closed loop minus TM-mem-short open loop is at least -4.0 points.
- **R13 is met** if both hold.

## Reported with no rule

LR-mem-short in closed loop; TM minus LR in closed loop; closed-loop accuracy by position in the conversation.
