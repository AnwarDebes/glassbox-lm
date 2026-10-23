# R5 registration: routing learned from reward alone (contextual bandit)

Part 5 of `FAILURE_MAP.md` (learning signal), and the one-step agent loop of the project idea. The TM is the policy: one class per action, the vote decides, epsilon-greedy exploration. The reward is turned into Type I and Type II feedback. Frozen by `scripts/freeze.py`; amendments in a dated file.

## Question

The agent picks a route for each query and sees only whether that route was right (reward 1 or 0). The correct label is never shown. Does a TM policy learn routing from this feedback as well as the standard bandit learners LinUCB and epsilon-greedy logistic? How close does it come to the same TM trained on labels?

## Tasks

The embedding is all-MiniLM-L6-v2. Splits are those of `src/dl/data.py`.

- CLINC150 domains: 10 actions, from clinc/oos-eval `domains.json`.
- HWU64 scenarios: 18 actions, the first word of the intent label.
- Banking77 intents: 77 actions.

## Protocol (`src/dl/bandit.py`, `scripts/bandit_run.py`)

- The stream is 5 random permutations of the training queries, in rounds of 32.
- In each round the learner acts on every query and sees only the chosen action's reward, then updates.
- Replay, where a learner's setting uses it, means extra updates on rounds drawn uniformly from its own log of (query, action, reward). No label is ever shown.
- After the stream, the greedy policy is scored on test.

## Learners

- **TM bandit:** thermometer bits (thermo:8, quantiles of the training embeddings, no labels), 4,000 clauses, T 4,000, s 8.
  - Reward 1: target-class feedback for the chosen action, and negative-class feedback for the highest-scoring other action. The reward fixes the label, so this is the supervised update.
  - Reward 0: negative-class feedback for the chosen action only.
- **LinUCB:** disjoint arms, ridge with lambda 1, on the raw embedding.
- **Epsilon-greedy logistic:** one logistic reward model per action on the raw embedding, Adam.
- **References with labels:**
  - sup-TM: the same TM trained on labels for 5 epochs, with no snapshot;
  - sup-LR: logistic regression, C chosen on val.

Settings are chosen by a rule fixed now. For each learner family, take the exploratory setting with the highest final val accuracy on CLINC150 domains among the runs tagged `e3` and `e6` (seed 1, `results/explore/bandit.jsonl`). That setting is used unchanged on all three tasks.

The families and their candidates:
- TM: epsilon, top-k, epsilon decay, replay.
- Epsilon-greedy logistic: epsilon, top-k, lr, replay.
- LinUCB: alpha.

## Runs

- Seeds 1 to 10 for every learner and every task. sup-LR is deterministic.
- `scripts/bandit_run.py --eval test --out confirm/R5`.

## Statistic

Per task and learner, the mean over seeds of the final test accuracy. For differences between two learners, the 95% interval is a percentile bootstrap over the 10 seeds, 10,000 resamples, seed 0.

## Pass rules (test)

- **R5.1:** TM bandit minus the better of LinUCB and epsilon-greedy logistic is at least -1.0 point, in at least 2 of the 3 tasks.
- **R5.2:** TM bandit is at least 0.97 times sup-TM, in at least 2 of the 3 tasks. This is the cost of learning from reward instead of labels.
- **R5 is met** if R5.1 and R5.2 both hold.

Reported with no rule: cumulative online accuracy (exploration included), the accuracy curve after every pass, and sup-LR.
