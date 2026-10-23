# R6 registration: clarify-or-route, a Tsetlin Machine policy learned with TD from delayed reward

Part 5 of `FAILURE_MAP.md`. This is the agent loop of the project idea: observe, act, get a reward, learn, with a regression-style TM trained toward r + gamma · max Q(s', a'). Frozen by `scripts/freeze.py`; amendments in a dated file.

## Question

Can a TM learn, from reward alone, when an LLM agent should route a query, when it should ask a clarifying question first, and when it should escalate to a human? The value of asking appears only one step later, so the task needs credit assignment back in time, the open problem of TM reinforcement learning (R1 literature report).

## Environment (`src/dl/mdp.py`, CLINC150)

- **Episode:** one query.
- **Step 0:** the agent sees the first half of the query (the first ceil(n/2) words) through a fixed router. The router is a supervised coalesced TM on the full query: thermo:8, 4,000 clauses, 40 epochs, cross-fitted in 5 folds on train, so training episodes see held-out router scores. Val and test episodes use the mean of the 5 fold routers.
- **Actions at step 0:**
  - ROUTE: commit to the router's top intent. +1 if it is right, -1 if it is wrong or the query is out of scope.
  - ASK: -0.1 now. The full query is shown at step 1.
  - ESCALATE: 0.
- **At step 1:** only ROUTE and ESCALATE are allowed.
- **Episode splits:**
  - train: 15,000 in-scope + 100 oos_train queries, with out-of-scope drawn at a 0.15 share;
  - val: 3,000 + 100;
  - test: 4,500 + 1,000. Every test episode is played once, greedily.
- **Observation bits of the decision layer:**
  - a step flag;
  - thermometers (16 levels, quantiles of the training episodes) of the router's top class sum and margin on the current view;
  - at step 1, the same thermometers of the step-0 view. These are the memory literals.

## Learners

- **TM-Q:** coalesced TM with one class per action; Q(s, a) = class sum / T, clipped to [-1.5, 1.5]. TD target r + gamma · max over allowed a' of Q(s', a'), with gamma = 1 (two steps). Each clause is selected with probability min(1, kappa·|e|), where e is the TD error. The feedback raises the action's sum when e > 0 (target-class feedback) and lowers it when e < 0 (negative-class feedback). It is trained in one of two ways:
  - (a) **online TD:** an epsilon-greedy behaviour policy with 256 new episodes per update, a replay buffer (the last 200,000 transitions) and a target copy synced every 100 updates;
  - (b) **fitted-Q iteration:** 60,000 training episodes with uniformly random allowed actions. Then k iterations: targets from the current machine, followed by epochs of regression feedback on those fixed targets.
- **TM-myopic:** the chosen TM-Q variant with gamma = 0. Only the immediate reward is used, so ASK always looks like -0.1.
- **TM-Q-nomem:** the chosen TM-Q variant without the memory literals (ablation).
- **Lin-Q:** linear Q on the same information as continuous features, with the same two training modes (Adam).
- **Rule:** the best 3-threshold rule on the router margin: route at step 0 if the margin is at least a, escalate if it is below c, otherwise ask; after asking, route if the margin is at least b. The grid runs over val quantiles and is chosen by val mean return.
- **Fixed policies:** always route at step 0; always ask, then route.

The settings are chosen by a rule fixed now. Among the exploratory runs tagged `e12_*` (seed 1, val, `results/explore/r6.jsonl`):
- TM-Q: the run with the highest val mean return among the TM-Q runs (online TD or fitted-Q, any kappa or updates).
- Lin-Q: the same among the Lin-Q runs.
- TM-myopic and TM-Q-nomem use TM-Q's chosen setting.

## Runs

- Seeds 1 to 10. The router seed and the learner seed are both k.
- `scripts/r6_run.py --eval test --out confirm/R6`.
- Every run is reported.

## Statistic

The mean return per test episode of each learner and seed. Differences between two learners are paired by seed. The 95% interval is a percentile bootstrap over the 10 seeds, 10,000 resamples, seed 0.

## Pass rules (test)

- **R6.1 (credit assignment back in time):** TM-Q minus TM-myopic is at least 0.02, and its interval excludes 0.
- **R6.2 (quality):** TM-Q minus Rule is at least -0.01. TM-Q is non-inferior to the best threshold rule tuned on val.
- **R6 is met** if R6.1 and R6.2 both hold.
- **R6.3 (memory):** TM-Q minus TM-Q-nomem, reported with its interval, no threshold.
- **R6.4 (linear):** TM-Q minus Lin-Q, reported with its interval. "Non-inferior" if at least -0.01.

Reported with no rule: action shares, return on in-scope and on out-of-scope episodes, router accuracy on both views, and the learned policy as clauses.
