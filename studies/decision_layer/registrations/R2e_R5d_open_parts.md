# R2e and R5d registration: the two parts whose own task never got a passing fix

Written after every other registered test had finished. Two registered failures were never re-tested on their own task with the fix that works elsewhere:

- **Part 1, CLINC150 (R2):** R2 met its accuracy rule (+5.86 points) and missed its length rule by one clause of 33 literals, a tie bug in the budget selection (see `R2d_R7c_exact_budget_reruns.md`). R2d re-ran the fixed budget on Banking77 and HWU64, not on CLINC150.
- **Part 5, reward-only routing with 77 actions (R5b, Banking77):** R5b was 3.16 points behind LinUCB after 15 passes. R5c passed with 150 actions after 30 passes on another task (CLINC150 intents), but Banking77 was not re-run.

Frozen by `scripts/freeze.py` before any R2e or R5d run. Amendments go in a dated file.

## R2e: R2 again, with the fixed budget (CLINC150, `lex:2000`)

- **Arm B:** R2's arm B (exact budget 32) with the start state that obeys it and the exact selection (`--max-literals 32 --init trim --budget-select rank`), seeds 1 to 10 (TM seed 100·k).
- **Arms A and C:** R2's files (`results/confirm/R2/A_default`, `C_lr_bits`), copied: the same code path and the same test split.
- **Exploration behind it (val, seed 1):** `e21_lex_trim32` 0.8747 (half start, R2's selection run `e2b_lex_ml32`: 0.8717; default `e2_lex_default` 0.8123).
- **Test split:** CLINC150, its second use by a lexical router (after R2).
- **Pass rules (R2's, unchanged):** R2.1: B minus A is at least +3.0 points with the interval above 0. R2.2: the median and the maximum literals per clause are at most 32 in every B seed. **R2e is met** if both hold. R2.3 (B against C) reported separately.

## R5d: the TM bandit on Banking77 with 30 passes

- **Task, machine and stream:** those of R5b (Banking77 intents, 77 actions; MiniLM `thermo:8`; rounds of 32, reward 1 or 0 for the chosen action only; TM bandit with Boltzmann exploration tau 0.5, replay 4, 4,000 clauses), with **30 passes** instead of 15. LinUCB alpha 1.0 and sup-TM (labels, 30 epochs) with the same 30.
- **Exploration behind it (val, seed 1):**

| Run | Learner | Passes | Final val accuracy |
|---|---|---|---|
| e17_b77_boltz05_p15 | TM bandit | 15 | 0.9043 |
| e18_b77_boltz05_p30 | TM bandit | 30 | 0.9192 (0.912 to 0.923 over passes 20 to 30) |
| e16_b77_linucb | LinUCB | 5 | 0.9212 |
| e18_b77_linucb_p15 | LinUCB | 15 | 0.9133 |
| e20_b77_linucb_p30 | LinUCB | 30 | 0.9063 |

  On val the TM bandit still rises after 15 passes and LinUCB falls. In R5b (15 passes) the gap moved by about 2 points against the TM from val (-0.9) to test (-3.16), so R5d may still fail.
- **Seeds:** 1 to 5. Tag `R5d`, results in `results/confirm/R5d/bandit.jsonl`.
- **Test split:** Banking77's third use by a bandit (after R5 and R5b).
- **Pass rules (R5b's, unchanged):** R5d.1: the mean final accuracy of the TM bandit is at least LinUCB's minus 1.0 point. R5d.2: it is at least 0.97 times sup-TM's. **R5d is met** if both hold.
- Reported with no rule: cumulative online accuracy, the bootstrap interval over seeds of TM minus LinUCB, and the mean test accuracy after every pass.
