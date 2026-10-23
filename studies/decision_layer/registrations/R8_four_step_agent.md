# R8 registration: a TM policy over four steps (clarify-or-route over growing prefixes, CLINC150)

Part 5 of `FAILURE_MAP.md` and the agent loop of the goal: the TM as the value function (Q) of an agent that observes, acts, gets a reward and learns from it. R6 had two steps. In R8 the value of asking at the first step shows up to three steps later, the "credit assignment over long horizons" that the goal names as the weakest part of a TM agent and leaves to TD learning.

Frozen by `scripts/freeze.py` before any R8 test run. Amendments go in a dated file.

## Task (`src/dl/mdp_h.py`, `scripts/r8_run.py`)

- One episode per query. CLINC150 test: 4,500 in-scope and 1,000 out-of-scope episodes. The query is revealed in 4 prefixes: the first 25, 50, 75 and 100% of its words (rounded up, at least one word).
- At step t the agent sees a fixed router's class sums on prefix t. The router is a supervised TM (thermometer code of MiniLM, 4,000 clauses, 40 epochs), 5-fold cross-fitted on train. It was trained once (router seed 1) and is shared by every agent seed. Cache `results/r8_router_seed1.pt`, SHA-256 ced1aa263fe16fbb0a0843df3724e94faa6bb38bbe3b0cc7bb4fae345a74e6e7. Its val accuracy on the four prefixes: 0.209, 0.511, 0.780, 0.952.
- Actions: ROUTE (+1 if the router's top intent is right, -1 if wrong or out of scope; ends), ASK (-0.1; the next prefix is revealed; not allowed at the last step), ESCALATE (0; ends).
- Observation bits, each with a name: step thermometer; 16-level thermometers of the router's top class sum and margin on the current prefix. Memory literals (only in the memory arm): the same thermometers on the previous prefix, "same top intent as before", "same top intent on the last 3 prefixes".

## Exploration behind it (val, seed 1, `results/explore/r8.jsonl`; mean return per episode, val holds 3.2% out of scope)

| Run | Learner | Val return | At a 15% out-of-scope share | Asks per episode |
|---|---|---|---|---|
| e22b_rule | margin thresholds per step, tuned on val itself | 0.6151 | 0.4786 | 2.34 |
| e22_linq | linear Q, fitted-Q iteration, memory features | 0.6007 | 0.4766 | 2.52 |
| e22_linq_nomem | the same without memory features | 0.6005 | 0.4765 | 2.43 |
| e22_tmq_nomem | TM-Q, no memory literals | 0.5932 | 0.4671 | 2.34 |
| e22_tmq | TM-Q with memory literals | 0.5723 | 0.4489 | 2.29 |
| e22_tmq_i8e5 | TM-Q with memory, 8 x 5 epochs | 0.5572 | 0.4396 | 2.80 |
| e22_ask_route | always ask to the last prefix, then route | 0.5419 | 0.3178 | 3.00 |
| e22_tmmyopic | TM, gamma 0 | 0.0268 | 0.0235 | 0.00 |

So on val the TM policy learns to ask and when to stop asking, but it does not beat the tuned rule (which is scored on the data it was tuned on) or the linear Q-learner, and the memory literals do not help it (the linear learner neither). The rules below are set accordingly.

## Arms (agent seeds 1 to 5; the rules and the fixed policy are deterministic and run once)

- **TM-Q (the registered TM policy): no memory literals,** the better TM variant on val. Coalesced TM, 2,000 clauses, T 1,000, s 4, one class per action, Q = class sum / T. Fitted-Q iteration toward r + max_a' Q(s', a'), 6 iterations of 10 epochs, batch 256, regression feedback with kappa 0.003, on 60,000 sampled queries with a uniformly random allowed action at every step (`--no-memory`).
- **TM-Q-memory:** the same with the memory literals (the memory ablation).
- **TM-myopic:** TM-Q with gamma 0 (`--no-memory`).
- **Lin-Q:** linear Q on the continuous twin of the observation with memory features, fitted-Q iteration as above, Adam lr 0.003.
- **Rule and Rule2:** thresholds per step on the router margin (Rule2: and on the top class sum), tuned on val by coordinate ascent. "The rule" is the better of the two on val at the 15% out-of-scope weighting.
- **Always ask, then route.**
- Out-of-scope episodes make up 15% of the learners' training queries and are weighted to 15% in the rules' val objective.

## Statistic

Mean return per test episode. Each learner's per-episode return is averaged over its seeds; differences are paired over the 5,500 test episodes, bootstrap over episodes (10,000 resamples, seed 0, 95% percentile interval). Script `scripts/report_r8.py`.

## Pass rules (test)

- **R8.1 (credit assignment over four steps):** TM-Q minus TM-myopic is at least +0.10, with the interval above 0.
- **R8.2 (the TM learns when to stop asking):** TM-Q minus "always ask, then route" has its interval above 0.
- **R8 is met** if R8.1 and R8.2 both hold.

Reported with no rule: TM-Q minus the rule, minus Lin-Q, and TM-Q-memory minus TM-Q, overall, in scope and out of scope; asks per episode; at which step episodes end; decision receipts (`scripts/demo_agent_h.py`). On val the TM is below the rule and Lin-Q and the memory literals do not help; no claim is registered for those comparisons.
