# R10 registration: the workflow learned from reward alone (ABCD, the agent loop)

The goal's agent loop: observe, act, get a reward, learn, with the reward turned into Type I and Type II feedback. R5 to R5d learned one-step routing this way. R10 asks whether a Tsetlin machine learns a multi-step workflow this way, with memory literals as the state of the conversation: at every action turn of an ABCD conversation the agent presses one of 30 buttons and is told only whether it was the button the human agent pressed (reward 1) or not (0). This is the per-step reward ("reward shaping") the goal names for long horizons; R10 does not test reward given only at the end of a conversation.

Frozen by `scripts/freeze.py` before any R10 test run. Amendments go in a dated file.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Inputs: no label of any kind (`ABCDFreeView`, `src/dl/abcd.py`; 1,706 bits)

R9's view uses subflow and next-button centres, which are built from labels, so it is not allowed here. R10's bits are:
- memory: `done(b)`, `last action = b`, `at least k actions so far` (66 bits);
- words: presence of the 1,000 most frequent 1- and 2-grams of the last 3 utterances;
- unsupervised clusters: spherical k-means (k = 64, no labels) of the bge-large embeddings of the last 3 utterances and of the customer's words; each cluster is named by its member text nearest to its centre; bits = similarity to each centre at the members' 25/50/75% levels, and "cluster j is among the 1 / 3 nearest".

## Protocol (`scripts/abcd_bandit.py`, as R5c/R5d)

The stream is 15 random permutations of the 29,190 training action turns, in rounds of 32. The learner picks one button per turn and sees only that button's reward; it updates after each round. After the stream, its greedy choices are scored on the 3,608 test action turns (once per run). Online accuracy is the share of reward 1 over the whole stream, exploration included.

## Exploration behind it (dev, seed 1, 15 passes, `results/explore/abcd_bandit.jsonl`)

| Run | Learner | Dev accuracy at the end | Online |
|---|---|---|---|
| e28_tm | TM bandit (4,000 clauses, s 8, Boltzmann exploration tau 0.5 on the class sums) | 0.8350 | 0.721 |
| e28_tm_nomem | the same without memory bits | 0.5934 | 0.540 |
| e28_sup_tm | the same TM trained on the labels (15 epochs, no snapshot) | 0.8412 | |
| e28_linucb (original class, alpha 0.5) and alpha 0.1 / 1.0 | LinUCB on the same bits | stopped unfinished after 1 h 49 min to 2 h 34 min (GPU shared with other users' jobs) | |
| e28_linucbfast_a0.5 | LinUCB, Woodbury class, alpha 0.5 | still running at the freeze | |

## Arms (test; seeds 1 to 5)

- **TM bandit:** as e28_tm.
- **TM bandit, no memory:** as e28_tm_nomem.
- **Supervised TM:** as e28_sup_tm (the full-information reference).

## Statistic

Final test accuracy per learner, mean over seeds. Differences: per test action turn the mean correctness over seeds, paired; 95% interval from a bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0 (`scripts/report_abcd_bandit.py`).

## Pass rules (test)

- **R10.1 (reward alone reaches the supervised machine):** TM bandit accuracy is at least 0.97 times the supervised TM's.
- **R10.2 (memory in the agent loop):** TM bandit minus TM bandit without memory is at least +5.0 points, interval above 0.
- **R10 is met** if R10.1 and R10.2 both hold.
- The comparison with LinUCB is not part of R10: no LinUCB run had finished on dev when R10 was frozen. It will get its own registration (R10c) with the dev number in it, before any LinUCB test run.

## Reported with no rule

Online accuracy of each learner; AST-weighted accuracy; per-pass curves; the strongest clauses of the TM bandit for a few buttons.
