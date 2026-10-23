# R2c registration: the literal budget, with a start state that obeys it, on two new datasets

Part 1 of `FAILURE_MAP.md`. R2 (CLINC150) met its accuracy rule (+5.86 points [+5.28, +6.45]) and was non-inferior to logistic regression on the same bits. It missed its length rule by one clause: seed 7 kept one clause of 33 literals, a leftover of the start state, in which about half of all literals are included. The budget only stopped new inclusions.

R2c repairs that and repeats the test on data that no lexical run of this study has scored:

- **Start state "trim"** (`src/dl/tm_init.py`, written after R2): the same start as before, then each clause keeps a random subset of at most L of its included literals. With the exact budget no clause can then hold more than L literals at any time.
- **Datasets:** Banking77 and HWU64 (splits of `src/dl/data.py`), word input `lex:2000`: binary presence of the 2,000 most frequent word 1- and 2-grams of the train texts. No run of this study has scored a lexical router on their test splits. R4, R4b, R5, R5b and R7 scored their test splits with embedding inputs.

Frozen by `scripts/freeze.py` before any R2c test run. Amendments go in a dated file.

## Arms (as in R2)

- **A, default:** coalesced TM (`src/dl/tm.py`), 4,000 clauses, T 4,000, s 8, 256 states, 40 epochs, batch 128, hardest-negative sampling, no budget, the half start. Best-val snapshot, val scored every 2 epochs.
- **B, the fix:** A with the exact literal budget L = 32 (R2's L*, not chosen again) and the "trim" start (`--init trim`).
- **C, linear control:** multinomial logistic regression on the same bits, C from the val grid of `scripts/confirm_router.py`.
- TM seeds 1 to 10 (TM seed 100·k); C is fitted once.
- Script: `scripts/confirm_router.py --registration R2c --encoder minilm --view lex:2000 ...` (the encoder argument is unused by `lex:2000`). The test split is scored once per run.

## Exploration behind it (val, seed 1, `results/explore/tm.jsonl`)

| Run | Dataset | Start | Budget | Val accuracy | Clauses with no plain literal |
|---|---|---|---|---|---|
| e2_lex_default | CLINC150 | half | none | 0.8123 | |
| e2b_lex_ml32 | CLINC150 | half | 32 | 0.8717 | |
| e21_lex_trim32 | CLINC150 | trim | 32 | 0.8747 | 97.5% |
| e21_lex_empty32 | CLINC150 | empty | 32 | 0.8743 | 97.9% |
| e21_b77_lex_default | Banking77 | half | none | 0.8036 | |
| e21_b77_lex_trim32 | Banking77 | trim | 32 | 0.8495 | |
| e21_b77_lex_empty32 | Banking77 | empty | 32 | 0.8455 | |
| e21_hwu_lex_default | HWU64 | half | none | 0.7742 | |
| e21_hwu_lex_trim32 | HWU64 | trim | 32 | 0.8243 | |

LR on the same bits (val): CLINC150 0.8763, Banking77 0.8624, HWU64 0.8587. The start state was chosen on CLINC150 between trim and empty (trim 0.8747, empty 0.8743; a tie would have gone to trim). Every budget run above has a maximum clause length of exactly 32.

## Pass rules (test)

- **R2c.1 accuracy:** in both datasets, B minus A is at least +3.0 points and its interval lies above 0.
- **R2c.2 clause length:** in B, no clause of any of the 10 seeds holds more than 32 literals.
- **R2c is met** if R2c.1 and R2c.2 both hold.
- **R2c.3, reported separately:** B minus C in each dataset; non-inferior if at least -0.5 points. On val, B is 1.3 (Banking77) and 3.4 (HWU64) points below C, so non-inferiority is not expected here.

Statistic as in R2: per-query mean correctness over the 10 seeds, paired difference over test queries, bootstrap over test queries (10,000 resamples, seed 0, 95% percentile interval). Also reported: the share of clauses with a plain literal and the clause length distribution. Rules applied by `scripts/rules_check.py R2c`.
