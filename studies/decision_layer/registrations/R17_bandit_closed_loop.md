# R17 registration: the reward-learned TM agent in the closed loop

R13 scored the supervised TM with its own presses as memory. R17 does the same for the agent that learned from reward alone (R10's TM bandit): after training, the action turns of every test conversation are decided in order, and the memory bits come from the agent's own earlier presses (`scripts/abcd_closed_loop.py --view free`). As in R13, the text bits (customer's words, last 3 utterances) stay as logged.

R10 saved no models, and the TM bandit's runs are not bit-reproducible on the shared GPU (a dev rerun of seed 1 gave 0.8246 against 0.8350), so R17 trains 5 new runs with R10's setting and saves them (`scripts/abcd_bandit.py --save-model`). Their open-loop test accuracy is a replication of R10.

**What was known at the freeze:** R10's test results (TM bandit 0.8300, without memory 0.6013) and R13's (supervised closed loop 0.8070). Frozen by `scripts/freeze.py` before any R17 test run. Ninth use of the ABCD test split.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Exploration behind it (dev, seed 1, `results/explore/abcd_closed_loop.jsonl`, e43)

R10's setting retrained with the model saved: open loop 0.8246, closed loop 0.7983 (-2.6 points). R10's TM bandit without memory on dev (e28): 0.5934.

## Arms (test)

- **TM bandit, closed loop:** R10's setting (4,000 clauses, s 8, Boltzmann 0.5, 15 passes, rounds of 32, 1,706 label-free bits, no replay), seeds 1 to 5, trained anew; each model scored in closed loop on test.
- **TM bandit, open loop:** the same 5 models with the logged memory.
- **TM bandit without memory:** R10's registered 5 runs (no memory bits, so no loop).

Statistic as R13: per test action turn the mean correctness over seeds; differences with a bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0, 95% interval.

## Pass rules (test)

- **R17.1 (memory still pays in the closed loop):** closed loop minus TM bandit without memory is at least +10.0 points, with the interval above 0.
- **R17.2 (its own mistakes cost at most 4 points):** closed loop minus open loop is at least -4.0 points.
- **R17 is met** if both hold.

## Reported with no rule

The open-loop accuracy of the new runs against R10's registered runs (replication); closed-loop accuracy by position in the conversation; the reward-learned agent in closed loop against the supervised one (R13).
