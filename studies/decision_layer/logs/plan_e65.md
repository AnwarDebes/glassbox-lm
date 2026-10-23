# Dev rules for R20's dev work (e65, e66), frozen before e65 starts

**R20's question (main).** Can the agent decide from the customer's words and its own earlier presses alone, in a closed loop, with no text that the human agent (ABCD) or the wizard (STAR) wrote? And what does that cost? STAR is the main evidence; ABCD (its test split would be on its 12th use) is supporting.

## e65: the customer-only view

One view, fixed here, so no view is selected on dev:
- **View C:** `--parts mem,cwords,xcust,xcrecent --encoder potion-32m` (src/dl/abcd.py, the R20 additions).
  - mem: the memory bits (in the closed loop, the agent's own presses);
  - cwords: the 1,000 most frequent 1-2-grams of the customer's last 3 utterances (`customer_recent`);
  - xcust: existsT bits of the customer's words so far, as in R19;
  - xcrecent: existsT bits of the customer's last 3 utterances.
- C is view B (R19) with every human-agent or wizard text removed. B's `words` and `xrecent` read the last 3 turns of both speakers; C reads the customer's last 3 utterances instead.
- **cwords is in C by this rule**, fixed before any e65 run.
- tests/test_customer_view.py checks that C reads no human-agent text: replacing every row's `recent` and `full` text leaves its bits unchanged.
- **The twin without memory:** `--parts cwords,xcust,xcrecent`.
- **Settings as R19:**
  - STAR 16,000 clauses, ABCD 4,000;
  - T = clauses, s 8, 40 epochs, start state half, best-dev snapshot;
  - seed 1, arms tm and lr_bits, the TM model saved (`--save`).
- **Then** the C model in closed loop on dev (`scripts/closed_loop_explore.py`, CPU decision `src/dl/cpu_tm_fast.py`), on the same job line as its training.
- Jobs: `logs/jobs_e65.txt`.

## e66: arm 1 on STAR dev

R19's and R18's saved STAR models in closed loop on dev (`scripts/abcd_closed_loop.py --dataset star --split dev`):
- R19star tm_mem_static, seeds 1 to 5, and lr_mem_static;
- R18 tm_mem, seeds 1 to 5, and lr_mem.

Jobs: `logs/jobs_e66.txt`.

## What e65 and e66 decide

- **S4 (arm 2 goes to test on a dataset).** On that dataset, C's closed-loop dev accuracy (seed 1) must be at least 5.0 points above its twin's dev accuracy without memory. Otherwise arm 2's dev result goes into the failure map, and that test split is not spent on arm 2.
- **R20's pass rules and margins** are written after e65 and e66, with every e65 and e66 number in the registration.
  - Planned kinds of rule, with values to be set then:
    - memory still pays in the closed loop;
    - the cost of arm 2 against R19's view-B agent;
    - on STAR (arm 1), what the agent's own mistakes cost.

## What they do not decide

- The question, the arms, view C and the settings (all above). Nothing is tuned on dev.
- R19's models, views and verdicts are not changed.
- ABCD's arm 1 is not registered again: R13 covers ABCD's closed loop, and e64 is reported as dev.
