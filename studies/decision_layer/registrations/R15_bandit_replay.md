# R15 registration: does replay make the TM bandit learn as fast as a neural agent?

R10, R10c and R14 show the same weakness: from reward alone the TM bandit ends as accurate as the others, but it earns less reward while it learns (online 0.718 against 0.801 for the neural bandit and 0.76 to 0.87 for LinUCB). R15 tests one remedy that uses no new information: replay. After each round, the TM gets k extra updates on minibatches drawn from everything it has logged so far (context, its own press, the reward), as R5's bandit did. No label is used.

**What was known at the freeze:** R10's TM test results, R10c's LinUCB results and R14's neural-bandit results. No replay run on test.

Frozen by `scripts/freeze.py` before any R15 test run. Seventh use of the ABCD test split.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Exploration behind it (dev, seed 1, 15 passes, R10's bits and stream, `results/explore/abcd_bandit.jsonl`)

| Run | Learner | Dev accuracy at the end | Online accuracy | Dev accuracy after pass 9 |
|---|---|---|---|---|
| e28_tm | TM bandit, no replay (R10) | 0.8350 | 0.721 | 0.781 |
| e34_tm_replay1 | replay 1 | 0.8409 | 0.747 | 0.817 |
| e34_tm_replay2 | replay 2 | 0.8377 | 0.768 | 0.835 |
| **e34_tm_replay4** | **replay 4** | **0.8409** | **0.780** | 0.839 |
| e34_tm_b025 | no replay, Boltzmann 0.25 (less exploration) | 0.7804 | 0.685 | |
| e33_nn_b05_lr1e-3 | neural bandit (R14's setting) | 0.8415 | 0.798 | |

Chosen: replay 4 (the most reward while learning; end accuracy equal to replay 1).

## Arms (test)

- **TM bandit + replay 4:** R10's setting otherwise (4,000 clauses, s 8, Boltzmann 0.5, 15 passes, rounds of 32, 1,706 bits), seeds 1 to 5 (`scripts/abcd_bandit.py --learner tm --replay 4`).
- **TM bandit, no replay:** R10's 5 runs. **Neural bandit:** R14's 5 runs.

## Statistics

- Online accuracy (the share of reward 1 over the whole training stream, exploration included): one number per run; a difference of means with a bootstrap over seeds (each arm's 5 seeds resampled, 10,000 resamples, seed 0, 95% interval).
- Final test accuracy: as R10 (per test action turn the mean over seeds; bootstrap over test conversations).

## Pass rules

- **R15.1 (it learns faster):** online accuracy of TM + replay 4 minus R10's TM bandit is at least +3.0 points, with the interval above 0.
- **R15.2 (it stays as accurate as the neural agent at the end):** final test accuracy of TM + replay 4 minus the neural bandit (R14) is at least -1.0 point.
- **R15 is met** if R15.1 and R15.2 both hold.
- **R15.3, its own verdict (the online gap to the neural agent is small):** online accuracy of TM + replay 4 minus the neural bandit is at least -3.0 points.

## Reported with no rule

Final test accuracy against R10's TM bandit; per-pass test curves; clause statistics; run time (replay 4 makes 5 times the TM updates).
