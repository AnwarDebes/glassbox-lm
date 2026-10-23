# R5b registration: does better exploration fix the many-action failure of the TM bandit?

Follow-up to R5, whose verdict is "not met". With 18 and 77 actions the TM bandit was 7.0 and 42.8 points behind LinUCB. Written after R5's test results and after exploratory val runs of the fix. The test split of HWU64 and Banking77 has been read only by R5's runs, not by any run of this fix. Frozen by `scripts/freeze.py`.

## What the exploration showed (val, seed 1, `results/explore/bandit.jsonl`, tags e16 and e17)

Banking77 intents, 77 actions:

| Run | Exploration | Passes | Final val accuracy |
|---|---|---|---|
| e16_b77_top20 | top-20 epsilon-greedy | 5 | 0.605 |
| e16_b77_boltz025 | Boltzmann, tau 0.25 | 5 | 0.668 |
| e16_b77_boltz05 | Boltzmann, tau 0.5 | 5 | 0.725 |
| e17_b77_boltz1 | Boltzmann, tau 1.0 | 5 | 0.679 |
| e17_b77_boltz05_p15 | Boltzmann, tau 0.5 | 15 | 0.904 (still 0.897 to 0.904 over passes 12 to 15) |
| e16_b77_linucb | LinUCB, alpha 1.0 | 5 | 0.921 |

HWU64 scenarios: Boltzmann tau 0.5, 5 passes, 0.935.

Boltzmann exploration samples the action from softmax((S - mean S) / (std S · tau)) over the TM's class sums S. It is `--boltzmann` in `scripts/bandit_run.py`, added after R5 had finished.

## Design

- **Tasks:** HWU64 scenarios (18 actions) and Banking77 intents (77 actions), the two tasks that failed R5. Same splits and stream protocol as R5 (rounds of 32, reward 1 or 0 for the chosen action only), with **15 passes**.
- **TM bandit:** the R5 machine (thermo:8, 4,000 clauses, T 4,000, s 8, replay 4), with Boltzmann exploration, tau 0.5, chosen on Banking77 val as above. A 0.05 share of uniform random actions remains (the epsilon of the code's default; `--eps 0.05`).
- **LinUCB:** alpha 1.0 as in R5, also 15 passes.
- **sup-TM:** the same TM trained on labels for 15 epochs, no snapshot.
- **Seeds:** 1 to 5, for budget reasons: R4b holds the GPU. Test split, scored once per run. Tag `R5b`, results in `results/confirm/R5b/bandit.jsonl`.

## Pass rules (test)

- **R5b.1:** in both tasks, the mean final accuracy of the TM bandit is at least LinUCB's minus 1.0 point.
- **R5b.2:** in both tasks, the TM bandit reaches at least 0.97 times sup-TM.
- **R5b is met** if both rules hold.

Reported with no rule:
- cumulative online accuracy over the 15 passes;
- the bootstrap interval over seeds of TM minus LinUCB;
- the accuracy after every pass.
