# R14 registration: the TM bandit against a neural bandit on the ABCD workflow

The Logic-Based-Agent claim is "as accurate as an NN agent". R9.4 tested it with labels (met: the TM composite is level with an opaque MLP). R14 tests it in the agent loop: the same 0/1 reward per press, the same 1,706 bits and the same stream as R10, with a neural network as the learner.

**What was known at the freeze:** R10's TM bandit test results (mean 0.8300, online 0.718) and R10c's LinUCB results. No neural-bandit result on test.

Frozen by `scripts/freeze.py` before any R14 test run. Sixth use of the ABCD test split (R9, R9b, R10, R10c, R13 before it).

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## The neural bandit (`NeuralBandit`, `src/dl/bandit.py`; `scripts/abcd_bandit.py --learner neural`)

One hidden layer of 512 (ReLU) from the bits to one logit per button, read as the chance that the button gets reward 1; after each round of 32, one AdamW step (step size 1e-3, weight decay 1e-4) on the binary cross-entropy of the chosen buttons' rewards, the same one update per round as the TM bandit. Exploration: Boltzmann on the standardised logits, temperature 0.5, as the TM bandit.

## Exploration behind it (dev, seed 1, 15 passes, `results/explore/abcd_bandit.jsonl`)

| Run | Learner | Dev accuracy at the end | Best pass | Online |
|---|---|---|---|---|
| e28_tm | TM bandit (R10's setting) | 0.8350 | 0.8350 | 0.721 |
| **e33_nn_b05_lr1e-3** | **neural, Boltzmann 0.5, step 1e-3** | **0.8415** | 0.8529 | 0.798 |
| e33_nn_b025_lr1e-3 | neural, Boltzmann 0.25 | 0.8382 | 0.8382 | 0.849 |
| e33_nn_eps05_lr1e-3 | neural, epsilon-greedy 0.05 | 0.8358 | 0.8374 | 0.817 |
| e33_nn_b05_lr3e-3 | neural, step 3e-3 | 0.8328 | 0.8328 | 0.798 |
| e33_sup_mlp_free | the same network trained on labels (best dev epoch) | 0.8407 | | |

Chosen for the neural arm: its best dev setting (Boltzmann 0.5, step 1e-3). On dev it ends 0.65 points above the TM bandit and earns far more reward while learning.

## Arms (test)

- **Neural bandit:** the chosen setting, seeds 1 to 5.
- **TM bandit:** R10's 5 runs, unchanged.

Statistic as R10: per test action turn the mean correctness over seeds, paired; bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0 (`scripts/report_abcd_bandit.py`).

## Pass rule (test)

- **R14.1 (as accurate as a neural agent, from reward alone):** TM bandit minus neural bandit is at least -1.0 point. R14 is met if R14.1 holds.

## Reported with no rule

Online accuracy of both (the neural bandit is expected to be well ahead); the neural bandit's best pass on test (chosen on test); AST-weighted accuracies.
