# Results of the decision_layer study

Every number on this page comes from a registered run on a test split, read once per run, unless it is marked "exploratory (val)" or "post hoc". Registrations and amendments are in `../registrations/`; run files are in `../results/confirm/<registration>/`. Intervals are 95% percentile bootstrap intervals: over test queries for accuracy differences (R2, R4, R7 and their follow-ups), over seeds for R5, R5b, R5c, R5d and R6, over test episodes for R8.

## R2 and R2b: the sparse-input fix on the word router (CLINC150, `lex:2000`)

Registration `R2_sparse_input_fix.md` (frozen 2026-10-15T22:18:42Z), `R2b_sparse_input_fix_tmu_budget.md` (22:24:38Z), amendment `R2_amendment_2026-10-15.md`. 10 seeds per TM arm.

| Arm | Test accuracy (mean of 10 seeds) | Out-of-scope AUROC | ECE after temperature | Clause length (median / max) | Clauses with a plain literal |
|---|---|---|---|---|---|
| A: default TM | 0.8197 | 0.839 | 0.043 | 1 / 820 | 75.5% |
| B: exact literal budget 32 | 0.8783 | 0.908 | 0.018 | 32 / 33 | 2.3% |
| B': tmu-style budget 32 | 0.5645 | 0.729 | 0.040 | 1-2 / 1,939 | 99.5% |
| C: logistic regression on the same bits | 0.8809 | 0.902 | 0.016 | n/a | n/a |

- **R2.1 (accuracy):** B minus A = +5.86 points [+5.28, +6.45]. **Met.**
- **R2.2 (clause length):** **not met.** Seed 7 has one clause of 33 literals among 40,000 clauses over 10 seeds; the rule allows at most 32. The first write-up blamed the start state. That was most likely wrong: R2c and R7b, whose start state can never exceed the budget, show a tie of two random keys in the budget's selection that keeps one literal too many (see R2c below), and 16 of this clause's 33 literals were excluded at the start (independent internal audit 3; project notes, not published).
- **R2 as registered: not met.**
- **R2.3:** B minus C = -0.26 [-0.72, +0.20]. B is non-inferior to logistic regression on the same bits. It does not beat it.
- **R2b:** not met.
  - R2b.1: B' minus A = -25.51 points [-26.51, -24.53]. Not met.
  - R2b.2: median literals 1 to 2, 95th percentile at most 40 (2 x L* = 64). Met.
  - R2b.3: B' minus C = -31.64 [-32.81, -30.46]. Not non-inferior.
- **Reading:** the budget repairs the accuracy loss of sparse input and bounds clause length. It does not give keyword rules: 97.7% of the bounded clauses are negation-only. The default router is about half single-word keyword rules (e.g. `contains "pin"` -> pin_change) and a quarter giant blacklists (up to 820 negated words).

## R6: clarify-or-route, a TM policy learned with TD from delayed reward (CLINC150)

Registration `R6_clarify_or_route_td.md` (frozen 2026-10-15T22:58:03Z); amendment `R6_amendment_2026-10-16.md` (parts 1 to 4; part 3 holds the disclosures of an internal audit, part 4 the sensitivity run). Test episodes: 4,500 in scope + 1,000 out of scope. Mean return per episode over 10 seeds.

| Learner | Mean return | Asks at step 0 | Routes at step 0 | Escalates at step 0 | Return in scope | Return out of scope |
|---|---|---|---|---|---|---|
| TM-Q (fitted-Q iteration, gamma 1) | **0.597** | 0.905 | 0.095 | 0.000 | 0.788 | -0.260 |
| TM-Q without memory literals | 0.597 | 0.891 | 0.109 | 0.000 | 0.785 | -0.252 |
| Lin-Q (linear, same information, online TD) | 0.596 | 0.896 | 0.104 | 0.000 | 0.796 | -0.303 |
| Rule (3 thresholds on the router margin, tuned on val) | 0.556 | 0.900 | 0.100 | 0.000 | 0.802 | -0.553 |
| Always ask, then route | 0.446 | 1.000 | 0 | 0 | 0.790 | -1.100 |
| TM-myopic (gamma 0) | 0.198 | 0.006 | 0.342 | 0.652 | 0.266 | -0.107 |
| Always route at once | -0.236 | 0 | 1.000 | 0 | -0.066 | -1.000 |

- **R6.1 (credit assignment back in time):** TM-Q minus TM-myopic = +0.399 [+0.392, +0.405]. **Met.** This contrast is nearly built in. The myopic TM sees ASK only as its immediate -0.1 and cannot value it, so it almost never asks. The TD-trained TM asks on 90% of half queries.
- **R6.2 (against the tuned rule):** TM-Q minus rule = +0.042 [+0.035, +0.049]. **Met.** The whole gain comes from out-of-scope episodes (+0.29 [+0.26, +0.33]).
  - In scope the rule is better: -0.014 [-0.020, -0.010].
  - Reweighted to val's out-of-scope share (3.2%), TM-Q minus rule is -0.0045 [-0.0093, -0.0001] (computed after the results, not registered).
  - In scope, TM-Q returns what "always ask, then route" returns (-0.002 [-0.007, +0.002]).
- **R6 as registered: met.**
- **R6.3 (memory literals):** +0.0006 [-0.0019, +0.0034]. No effect on this two-step task: at step 1 the full query's own confidence carries the decision.
- **R6.4 (linear Q; same episodes and reward, trained by online TD, while TM-Q uses fitted-Q iteration):** TM-Q minus Lin-Q = +0.0011 [-0.0035, +0.0049]. Non-inferior.
- **How to read R6.2.** The out-of-scope share is 3.2% in val, 18.2% in test, and 15% in the training episodes of the TD learners.
  - The rule was tuned on val and sees only the router's margin. On test it routes too many out-of-scope queries (return -0.553).
  - TM-Q and Lin-Q also see the router's top class sum, the better out-of-scope signal (R1), and escalate those queries after asking (-0.26 and -0.30).
  - So the gain over the rule comes from reward-driven training on a closer distribution and from that extra signal, not from the TM as such: the linear Q-learner gets the same return.
  - What the TM adds is a learned policy made of clauses (R6's policy rules of seed 9: project notes, not published), so every decision can be printed with the clauses that carried it (a test demo, seed 9: project notes, not published). At step 0 the memory literals are all 0, so in a step-0 receipt they hold trivially and say nothing.
- **What the policy learned, in words:** ask first on 90% of half queries; never escalate at step 0; after asking, route when the full query's confidence is high, otherwise escalate. It learned when to ask and when to escalate after asking, not a finer timing of immediate routing.
- **Sensitivity of the selection (post hoc, not part of the verdict).** The chosen TM-Q setting came from a candidate queued after the freeze (amendment part 3). The best candidate queued before the freeze (online TD, kappa 0.005, 3,000 updates) was therefore also run on the same 10 seeds and test episodes. It returns 0.598, against 0.597 for the chosen setting; minus rule +0.042 [+0.036, +0.049]. The conclusion does not depend on that choice.
- **Step size matters (exploration, val).** The regression feedback is summed over a minibatch, so the probability scale kappa has to be about 1 / (batch x firing clauses / T). With kappa 1 the TM never learned to ask (val return 0.070); with kappa 0.003 to 0.005 it did (0.763 to 0.764).

## R5: routing learned from reward alone (contextual bandit)

Registration `R5_bandit_routing.md` (frozen 2026-10-15T22:30:00Z); amendment `R5_amendment_2026-10-15.md`. The stream is 5 passes over the training queries in rounds of 32; only the chosen route's reward (1 or 0) is seen. Final greedy accuracy on test, mean of 10 seeds; cumulative online accuracy (exploration included) in brackets.

| Task (actions) | TM bandit | LinUCB | Epsilon-greedy logistic | sup-TM (labels, 5 epochs) | sup-LR (labels) |
|---|---|---|---|---|---|
| CLINC150 domains (10) | **0.9580** (0.858) | 0.9492 (0.949) | 0.9503 (0.857) | 0.9543 | 0.9602 |
| HWU64 scenarios (18) | 0.8559 (0.717) | **0.9255** (0.911) | 0.8691 (0.723) | 0.9205 | 0.9340 |
| Banking77 intents (77) | 0.4947 (0.375) | **0.9230** (0.761) | 0.2417 (0.165) | 0.8978 | 0.9269 |

- **R5.1 (TM within 1 point of the better baseline):**
  - CLINC150 domains: TM minus the better baseline (epsilon-greedy logistic) = +0.77 [+0.54, +0.99] points; minus LinUCB +0.88. Met, and the TM is ahead on final accuracy. Its online accuracy is lower than LinUCB's (0.858 against 0.948), so it pays more regret on the way.
  - HWU64: -6.96 [-9.06, -5.11]. Not met.
  - Banking77: -42.8. Not met.
  - With 1 of 3 tasks, R5.1 is not met.
- **R5.2 (TM at least 0.97 x sup-TM):**
  - CLINC150: 1.004. Met.
  - HWU64: 0.930. Not met.
  - Banking77: 0.551. Not met.
  - With 1 of 3 tasks, R5.2 is not met.
- **R5 as registered: not met.**
- **Reading:**
  - With 10 actions, the TM policy ends more accurate than LinUCB and epsilon-greedy logistic, and as accurate as its supervised twin. With replay it makes 5 times the updates of that twin, and online it makes more errors than LinUCB.
  - With 18 and 77 actions it falls far behind LinUCB.
  - One reading, partly tested afterwards: a 0 reward gives negative feedback for one action only, and top-3 epsilon-greedy exploration rarely finds the right action among 77. Exploratory runs on val (seed 1, not registered) back the exploration part: Boltzmann exploration on the class sums lifts Banking77 to 0.725 in 5 passes (registered setting about 0.50), against LinUCB 0.921 on val.
  - LinUCB's per-action ridge model and its optimism explore far better there. Its cumulative online accuracy is also much higher on every task, so its regret is lower.
  - A TM bandit for many actions needs an uncertainty-driven exploration. Thompson sampling over bootstrapped clause banks (Seraj, Sharma and Granmo, NeurIPS 2022) is the published candidate. This is untested here.

## R4 (with R1): the TM router against logistic regression on frozen LLM embeddings

Registration `R4_router_vs_lr.md` (frozen 2026-10-15T22:21:31Z); amendment `R4_amendment_2026-10-15.md`. TM-single = thermo:8, 4,000 clauses, T 4,000, s 8, 80 epochs, 10 seeds per cell. TM-composite = two disjoint 5-machine composites (seeds 1-5, 6-10). TM-readout = signed logistic vote weights on the clause outputs plus the input bits of each seed's machine (variant chosen per cell on val; "clauses+bits" in every cell). Test accuracy in %, difference to LR with its 95% interval in brackets.

| Dataset / encoder | LR | TM-single | TM-composite | TM-readout | ECE after temperature (TM / LR) |
|---|---|---|---|---|---|
| clinc150 / minilm | 95.64 | 94.29 (-1.36 [-1.75, -0.97]) | 94.80 (-0.84 [-1.27, -0.44]) | 95.24 (-0.40 [-0.71, -0.09]) | 0.016 / 0.009 |
| banking77 / minilm | 92.69 | 92.13 (-0.57 [-1.08, -0.05]) | 92.56 (-0.13 [-0.67, +0.42]) | 92.70 (+0.00 [-0.49, +0.50]) | 0.021 / 0.011 |
| hwu64 / minilm | 91.73 | 89.32 (-2.41 [-3.43, -1.41]) | 90.71 (-1.02 [-2.04, +0.00]) | 90.60 (-1.12 [-2.02, -0.29]) | 0.025 / 0.021 |
| clinc150 / bge-base | 97.02 | 96.13 (-0.90 [-1.19, -0.60]) | 96.44 (-0.58 [-0.90, -0.26]) | 96.85 (-0.17 [-0.38, +0.04]) | 0.013 / 0.006 |
| banking77 / bge-base | 93.60 | 93.05 (-0.56 [-1.03, -0.07]) | 93.83 (+0.23 [-0.28, +0.73]) | 94.06 (+0.45 [+0.02, +0.90]) | 0.022 / 0.010 |
| hwu64 / bge-base | 91.64 | 90.58 (-1.06 [-1.95, -0.15]) | 91.59 (-0.05 [-1.02, +0.93]) | 91.57 (-0.07 [-0.86, +0.72]) | 0.023 / 0.022 |

- **R4.1 (main):** TM-composite is at least -0.5 points from LR in 3 of 6 cells (Banking77 MiniLM, Banking77 bge-base, HWU64 bge-base); the rule needs 5. **Not met.**
- **R4.2:** TM-single trails LR in every cell, by 0.56 to 2.41 points. The gap is not closed.
- **R4.3:** TM-readout is ahead of LR in one cell, Banking77 / bge-base: +0.45 [+0.02, +0.90].
  - Its interval contains 0 in three cells, and it is behind in two (CLINC150 MiniLM -0.40, HWU64 MiniLM -1.12). No composite is ahead.
  - The "ahead" is fragile. It rests on choosing the read-out variant per cell (summed val accuracy). Chosen per seed it is +0.32 [-0.11, +0.75]. Both variants' test accuracies were printed in the job log before the per-cell choice was written down in the amendment. The choice itself was in `rules_check.py` from its first version.
  - The chosen variant ("clauses+bits") weights every input bit as well. The read-out on the clause outputs alone trails LR in all six cells, by 0.30 to 1.63 points.
- **R4 as registered: not met.**
- **R1.1 (calibration):** after one temperature fitted on val, the TM's expected calibration error is 0.013 to 0.025 in every cell (bar 0.05). **Met.** Before the temperature (raw softmax of the normalised class sums) it is 0.215 to 0.336 over the 60 runs.

Out-of-scope detection on CLINC150 (AUROC, 4,500 in-scope against 1,000 oos_test):

| Encoder | LR max-softmax | TM top class sum | TM margin | TM-composite top sum | TM-readout max-softmax | TM above LR in all 10 seeds |
|---|---|---|---|---|---|---|
| clinc150 / minilm | 0.9625 | 0.9623 | 0.9409 | 0.9667 | 0.9606 | no |
| clinc150 / bge-base | 0.9694 | 0.9728 | 0.9577 | 0.9764 | 0.9699 | yes |

- **R1.2:** the TM's top class sum is non-inferior to LR's maximum softmax with both encoders. With bge-base it is above LR in all 10 seeds. **Met.** The margin, which Clause-Driven-LLM used, is clearly worse (0.941 / 0.958).
- **Reading.**
  - With one 4,000-clause machine, the clause router is 0.6 to 2.4 points less accurate than logistic regression on the same embedding.
  - Ensembles, and fitted logistic weights over the TM's clauses plus the input bits, narrow this to between -1.1 and +0.5 points.
  - As a decision layer it is calibrated after one temperature, and it is at least as good at flagging out-of-scope queries.

## R5b: better exploration for the TM bandit (follow-up to R5)

Registration `R5b_bandit_exploration_fix.md` (frozen 2026-10-16T06:03:20Z, after R5's results and after exploratory val runs of the fix); amendment `R5b_amendment_2026-10-16.md`. The amendment corrects the description: exploration is pure Boltzmann sampling over the class sums, tau 0.5. 15 passes, 5 seeds, test split. Final accuracy, with cumulative online accuracy in brackets.

| Task (actions) | TM bandit, Boltzmann | LinUCB | sup-TM (15 epochs) | TM in R5 (top-3 epsilon-greedy, 5 passes) |
|---|---|---|---|---|
| HWU64 scenarios (18) | **0.9314** (0.906) | 0.9145 (0.946) | 0.9327 | 0.8559 |
| Banking77 intents (77) | 0.8930 (0.725) | **0.9245** (0.898) | 0.9161 | 0.4947 |

- **R5b.1 (TM within 1 point of LinUCB in both tasks):**
  - HWU64: +1.69 [+1.23, +2.03] points. Met; the TM is ahead.
  - Banking77: -3.16 [-3.52, -2.83]. Not met.
  - So R5b.1 is not met.
- **R5b.2 (TM at least 0.97 times sup-TM in both tasks):** 0.999 and 0.975. **Met.**
- **R5b as registered: not met** (R5b.1 fails on Banking77).
- **Reading:**
  - Exploration was most of R5's failure. With Boltzmann sampling over the TM's own class sums, the gap to LinUCB moves from -7.0 to +1.7 points with 18 actions, and from -42.8 to -3.2 points with 77.
  - Online, LinUCB still makes fewer errors on the way: 0.946 against 0.906, and 0.898 against 0.725.

## R4b: the router with 16,000 clauses

Registration `R4b_router_capacity.md` (frozen 2026-10-15T22:44:02Z); amendment `R4b_amendment_2026-10-15.md` (16,000 clauses, 40 epochs; clause-chunked update). Several jobs were lost to CUDA out-of-memory and re-run with the same commands (`scripts/r4b_chains.sh`; the run log is in the project notes, not published). Every cell holds seeds 1 to 10. Test accuracy in %, difference to LR with its 95% interval in brackets.

| Dataset / encoder | LR | R4 TM-single (4,000 clauses) | R4b TM-single (16,000) | R4b TM-composite | ECE (R4b TM) |
|---|---|---|---|---|---|
| clinc150 / minilm | 95.64 | 94.29 | 94.69 (-0.96 [-1.34, -0.58]) | 94.79 (-0.86 [-1.28, -0.44]) | 0.017 |
| banking77 / minilm | 92.69 | 92.13 | 92.51 (-0.19 [-0.68, +0.32]) | 92.66 (-0.03 [-0.57, +0.52]) | 0.023 |
| hwu64 / minilm | 91.73 | 89.32 | 90.27 (-1.46 [-2.57, -0.38]) | 90.47 (-1.25 [-2.37, -0.19]) | 0.027 |
| clinc150 / bge-base | 97.02 | 96.13 | 96.39 (-0.64 [-0.94, -0.34]) | 96.42 (-0.60 [-0.93, -0.28]) | 0.013 |
| banking77 / bge-base | 93.60 | 93.05 | 93.54 (-0.07 [-0.53, +0.40]) | 93.62 (+0.02 [-0.49, +0.52]) | 0.023 |
| hwu64 / bge-base | 91.64 | 90.58 | 91.27 (-0.36 [-1.29, +0.57]) | 91.17 (-0.46 [-1.53, +0.60]) | 0.022 |

- **R4b.1 (main):** TM-composite is at least -0.5 points from LR in 3 of 6 cells; the rule needs 5. **Not met.**
- **R4b.2:** TM-single is at least -0.5 points in 3 of 6 cells (Banking77 MiniLM -0.19, Banking77 bge-base -0.07, HWU64 bge-base -0.36). The gap is not closed.
- **R4b.3:** no arm is ahead of LR in any cell.
- **R4b as registered: not met.**
- **R4b.4 (the R1 rules on the 16,000-clause machine): met.**
  - Calibration error is 0.013 to 0.027 after one temperature.
  - Out-of-scope detection:
    - clinc150 / minilm: TM top sum 0.9661 against LR 0.9625, above LR in all 10 seeds: yes; composites 0.9674.
    - clinc150 / bge-base: TM top sum 0.9745 against LR 0.9694, above LR in all 10 seeds: yes; composites 0.9756.
- **Reading:**
  - Four times the clauses, at half the epochs, lift the single machine in every cell, by 0.26 to 0.95 points.
  - They also make it better than logistic regression at flagging out-of-scope queries with both encoders, in every seed.
  - The composites do not gain further: 5 machines of 16,000 clauses are no better than 5 of 4,000.
  - In accuracy, the TM router stays at or slightly below logistic regression.

## R7: a readable TM decision layer on top of an LLM (bge-large; CLINC150, Banking77, HWU64)

Registration `R7_readable_decision_layer.md` (frozen 2026-10-16T08:47:13Z); amendment `R7_amendment_2026-10-16.md` (source changes after the freeze, none used by the R7 runs; which files were opened early). Every input bit has a name a person can read:

- `contains "pin"`: a word or word pair, the 2,000 most frequent 1- and 2-grams of the train texts;
- `near(pin_change) >= p25`: the query's similarity to the centre of pin_change's training queries (bge-large embedding, whitened by the within-class covariance) is at least the 25th percentile of those queries' own similarities (levels p5, p25, p50, p75);
- `nearest = pin_change` and `pin_change in top 3`: the nearest centre(s).

10 seeds per dataset (4,000 clauses, 40 epochs). Test accuracy:

| Dataset | TM-readable (mean of 10) | TM-readable composite (2 x 5) | LR on the same readable bits | LR on the raw embedding (opaque) | One rule, "route to the nearest centre" (post hoc) |
|---|---|---|---|---|---|
| CLINC150 | 0.9638 | 0.9649 | 0.9660 | 0.9747 | 0.9633 |
| Banking77 | 0.9309 | 0.9317 | 0.9321 | 0.9416 | 0.9308 |
| HWU64 | 0.9153 | 0.9150 | 0.9182 | 0.9238 | 0.9164 |

- **R7.1 (TM-readable minus LR-readable >= -0.5 in 2 of 3):** -0.22 [-0.40, -0.03], -0.12 [-0.43, +0.19], -0.29 [-0.85, +0.24]. **Met in 3 of 3.**
- **R7.2 (composite minus LR-raw >= -1.0 in 2 of 3):** -0.98 [-1.34, -0.61], -0.99 [-1.69, -0.29], -0.88 [-2.14, +0.37]. **Met in 3 of 3**, two of them just inside the margin.
- **R7 as registered: met.** Readable inputs cost about one point against the opaque linear model, and on them the TM comes within 0.3 points of a linear model (on CLINC150 it is below it: -0.22 [-0.40, -0.03]).
- TM-readable (single) minus LR-raw: -1.08 [-1.45, -0.73], -1.06 [-1.76, -0.37], -0.85 [-2.04, +0.33]. One readable TM costs more than a point in 2 of the 3 datasets; the composites stay within it.
- Reported with no rule:
  - Calibration error after one temperature: 0.023, 0.035, 0.026.
  - Out-of-scope AUROC (CLINC150, top class sum): 0.916 (composites 0.926 and 0.929), against 0.974 for LR-raw and 0.951 for LR-readable. The readable view is weak here, unlike the thermometer view of R1.
  - Clauses (seed means): 77 to 92% of the non-empty clauses hold a plain literal; 73 to 80% hold at most 8 literals; the longest holds 503 to 755 literals, almost all of them exclusions.
  - About 125 to 180 clauses fire behind a decision (median), 94 to 122 of them with a positive vote for the winner.
- Readability of the decisions (`results/confirm/R7/readability.json`, `scripts/r7_readability.py`; the parts marked post hoc there were computed after the results and are not registered):
  - The decision is exactly the weighted vote of the firing clauses (median 125 to 180 of them). Recomputing that vote from the saved clauses gives the model's prediction for 100% of test queries (post hoc check of the saved models).
  - The printed receipts show the 3 strongest firing clauses, which carry only 5 to 21% of the winner's positive vote. Post hoc: those 3 alone give the same decision in 99.3%, 99.5% and 99.7% of test queries; the strongest single clause in 88.6%, 96.5% and 98.3%.
  - Post hoc: the smallest sufficient explanation holds at most 5 literals in 80%, 93% and 96% of decisions for the TM. For logistic regression on the same bits (its few strongest features, with the bias) it is 100%, 99.9% and 99.8%. The TM's explanations are not smaller than the linear model's; its long exclusion clauses make some of them longer.
  - Post hoc: dropping the clauses with more than 4 literals costs 9.1, 2.9 and 0.3 points. The long exclusion clauses carry part of the accuracy.
- **What the TM learned (post hoc).** It agrees with the one-rule router "route to the nearest whitened centre" on 99.6%, 99.7% and 99.2% of test queries. Where the two disagree (20, 11 and 9 queries per seed), the TM is right 49%, 62% and 25% of the time, the centre rule 37%, 29% and 40%. So the readable layer's accuracy comes from the LLM encoder and the whitened centres; the TM restates the centre rule as clauses and adds little to it. The centre rule's own out-of-scope score (maximum whitened similarity) reaches 0.966 on CLINC150, above the readable TM's 0.916.
- Example receipts (CLINC150 test, seed 1; all receipts in `logs/R7_readability.txt`):
  - "give me a 10 minute timer" -> timer: `+364 IF nearest = timer`; `+252 IF contains "timer"`; `+241 IF contains "timer"`.
  - "make 1234 a new pin for my savings account" -> pin_change: `+335 IF nearest = pin_change`; `+309 IF pin_change in top 3`.
  - "that's true" -> yes: the three strongest clauses are exclusion lists of several hundred conditions each ("near(apr) < p25 AND near(balance) < p25 AND ... AND NONE OF [nearest = apr, ...]"). Such receipts are not readable.

## R2c: the literal budget with a start state that obeys it (Banking77, HWU64 word routers)

Registration `R2c_budget_trim_start.md` (frozen 2026-10-16T09:42:34Z); amendment `R2c_R7b_amendment_2026-10-16.md`. Input `lex:2000`; 10 seeds per TM arm. No lexical run had scored these test splits before.

| Dataset | A: default TM | B: budget 32, trim start | C: LR on the same bits | B minus A | B minus C |
|---|---|---|---|---|---|
| Banking77 | 0.8135 | 0.8607 | 0.8750 | **+4.71 [+4.00, +5.44]** | -1.43 [-2.05, -0.80] |
| HWU64 | 0.7465 | 0.7874 | 0.8392 | **+4.09 [+2.79, +5.41]** | -5.19 [-6.50, -3.93] |

- **R2c.1 (B minus A >= +3.0, interval above 0, both datasets): met.**
- **R2c.2 (no clause over 32 literals): not met.** Banking77 seed 6 holds two clauses of 33 literals.
- **R2c as registered: not met.**
- **Cause, found afterwards:** the exact budget keeps the crossing literals whose random key is at least the k-th largest. Float32 keys from `torch.rand` on the GPU tie (20 million draws gave 14.2 million distinct values), and a tie at the k-th place keeps k + 1. Every overshoot in R2, R2c and R7b is exactly one literal. `src/dl/tm.py` now has an exact selection (`budget_select="rank"`, test `tests/test_budget_exact.py`); the re-run R2d (registered 11:33:29Z) repeats arm B with it.
- Reported with no rule: in B the median clause holds 32 literals and 98.8 to 99.8% of the clauses hold exactly 32; 95.3% (Banking77) and 99.3% (HWU64) are negation-only (A: 21% and 30%). Calibration error after one temperature falls on Banking77 (0.040 to 0.015) and rises on HWU64 (0.069 to 0.121; seeds 6, 8 and 10 reach 0.14, 0.19 and 0.26). The A and B folders also hold a logistic regression on the raw MiniLM embedding, a by-product of `confirm_router.py` that no rule uses.
- **Reading:** on three datasets the budget lifts the word router by 4 to 6 points (CLINC150 +5.86 in R2). It matches LR on the same bits only on CLINC150; on Banking77 and HWU64 it stays 1.4 and 5.2 points below.

## R7b: the readable layer with every clause bounded to 16 literals

Registration `R7b_readable_budget.md` (frozen 2026-10-16T09:45:25Z); amendment `R2c_R7b_amendment_2026-10-16.md`. R7's machine plus the budget (16) and the trim start; LR arms are R7's files.

| Dataset | TM-readable-16 (10 seeds) | Composite | LR-readable | LR-raw | R7's TM-readable |
|---|---|---|---|---|---|
| CLINC150 | 0.9660 | 0.9667 | 0.9660 | 0.9747 | 0.9638 |
| Banking77 | 0.9325 | 0.9328 | 0.9321 | 0.9416 | 0.9309 |
| HWU64 | 0.9169 | 0.9173 | 0.9182 | 0.9238 | 0.9153 |

- **R7b.1 (TM-readable-16 minus LR-readable >= -0.5 in 2 of 3):** -0.00 [-0.16, +0.15], +0.03 [-0.19, +0.25], -0.13 [-0.61, +0.33]. Met in 3 of 3.
- **R7b.2 (composite minus LR-raw >= -1.0 in 2 of 3):** -0.80 [-1.14, -0.44], -0.88 [-1.57, -0.19], -0.65 [-1.86, +0.51]. Met in 3 of 3.
- **R7b.3 (no clause over 16 literals): not met,** by three clauses of 17 literals (CLINC150 seed 2, HWU64 seeds 6 and 10), the tie described under R2c.
- **R7b.4 (CLINC150 out-of-scope AUROC >= 0.936): met,** 0.950 (composites 0.957 and 0.957), against 0.916 for R7.
- **R7b as registered: not met** (R7b.3). The re-run R7c with the exact selection is registered (11:33:29Z).
- R7b runs on the same test splits as R7 (their second use), designed after R7's test results; L was chosen on val.
- Compared with R7 (post hoc, not registered): single TM +0.21 [+0.09, +0.34], +0.15 [-0.03, +0.33], +0.16 [-0.23, +0.55] points (only CLINC150 clear of zero); level with LR on the same bits; calibration error after one temperature 0.015, 0.016 and 0.023 (R7: 0.023, 0.035, 0.026); out-of-scope AUROC +0.034.
- **The cost:** the budget makes every clause short, but most clauses become exclusion lists. Clauses with no plain literal: 90.8%, 74.1% and 73.9% (R7: 22.5%, 13.0%, 7.7%).
- **Readability report** (registered with no rule; `results/confirm/R7b/readability.json`, receipts in `logs/R7b_readability.txt`; parts marked post hoc there are not registered):
  - A decision is now the vote of most of the machine: 3,276, 2,674 and 2,896 of the 4,000 clauses fire (median), against 125 to 180 in R7.
  - The 3 strongest firing clauses carry about 1% of the winner's positive vote. Post hoc, they alone give the same decision in 92.1%, 97.7% and 98.1% of test queries (R7: 99.3 to 99.7%); the strongest single clause in 67.8%, 82.5% and 85.2%.
  - Post hoc: a sufficient explanation needs at most 5 literals in 0.1%, 2.8% and 3.2% of decisions (R7: 80 to 96%; logistic regression on the same bits: 99.8 to 100%).
  - A typical receipt: "please text john" -> text: `+75 IF near(make_call) < p25 AND near(order_checks) < p25 AND near(weather) < p50 AND NONE OF [nearest = exchange_rate, nearest = restaurant_suggestion, ..., contains "hotel", contains "money", ...]`. Each clause is short and readable; the decision is reached by elimination, spread over thousands of them.
  - Post hoc: it agrees with the one-rule nearest-centre router on 99.2%, 99.0% and 98.5% of test queries; where they disagree, the TM is right 55%, 46% and 31% of the time, the centre rule 21%, 28% and 28%.
- **Reading:** the budget, the fix of Part 1, buys bounded clauses, better out-of-scope detection and better calibration. It does not buy readable decisions: R7's receipts ("IF nearest = timer", "IF contains "timer"") explain a decision with one or two short positive rules; R7b's spread it over thousands of exclusion rules.

## R2d: R2c's budget arm again, with the exact budget selection

Registration `R2d_R7c_exact_budget_reruns.md` (frozen 2026-10-16T11:33:29Z); amendment `R2d_R7c_amendment_2026-10-16.md`. Arm B of R2c (budget 32, trim start) with `--budget-select rank`; arms A and C are R2c's files. Second use of these test splits by a lexical router.

| Dataset | A: default TM | B: budget 32, exact selection | C: LR on the same bits | B minus A | B minus C | Longest clause in B |
|---|---|---|---|---|---|---|
| Banking77 | 0.8135 | 0.8616 | 0.8750 | **+4.81 [+4.08, +5.53]** | -1.34 [-1.95, -0.72] | 32 |
| HWU64 | 0.7465 | 0.7861 | 0.8392 | **+3.96 [+2.69, +5.27]** | -5.32 [-6.62, -4.05] | 32 |

- **R2c.1: met in both datasets. R2c.2: met** (no clause of the 20 models holds more than 32 literals; all ran with `budget_select="rank"`). **R2d: met.**
- What R2d adds: it checks the length fix, not the accuracy gain again. Only 4 of the 20 runs met a tie (Banking77 seeds 2, 6, 8; HWU64 seed 1); the other 16 models are bit-identical to R2c's. Banking77 seed 6, the seed with the 33-literal clauses, is one of the 4, which supports the tie explanation.
- Calibration error after one temperature, as in R2c: Banking77 falls (0.040 to 0.016), HWU64 rises (0.069 to 0.120; 5 of 10 seeds above 0.10).
- With R2 (CLINC150, +5.86, its accuracy rule met; R2 itself was not re-run), the literal budget lifts the word router by 4 to 6 points on all three datasets. Its limits are unchanged: 95% and 99% of the bounded clauses are negation-only, so the blacklists of Part 1 become short blacklists, and the TM stays below LR on the same bits on Banking77 and HWU64.

## R5c: routing learned from reward alone with 150 actions (CLINC150 intents)

Registration `R5c_bandit_150_actions.md` (frozen 2026-10-16T11:07:45Z); amendment `R8_R5c_amendment_2026-10-16.md`. MiniLM `thermo:8`; the stream of R5 (rounds of 32, reward 1 or 0 for the chosen route only), 30 passes; 5 seeds; test split. The TM bandit is the R5b machine: Boltzmann exploration (tau 0.5) over its class sums, Type I/II feedback from the reward, replay 4.

| Learner | Final test accuracy (mean of 5) | Cumulative online accuracy |
|---|---|---|
| TM bandit (reward only) | 0.9365 | 0.781 |
| sup-TM (the same TM trained on labels, 30 epochs) | 0.9386 | n/a |
| LinUCB (alpha 1.0) | 0.9463 | 0.936 |

- **R5c.1 (TM bandit >= 0.97 x sup-TM): met,** 0.998. **R5c: met.**
- **R5c.2 (separate verdict; TM bandit >= LinUCB - 1.0 point): met, just:** -0.98 [-1.20, -0.80]; the margin is 5 test answers over 5 seeds, and 2 of the 5 seeds are more than a point behind. For R5 and R5b "met" required this comparison; for R5c it was moved out of "met" after the val run had shown -1.03 (stated in the registration).
- Mean test accuracy after passes 1, 5, 10, 15, 20, 25 and 30 (registered as reported; all 30 in `results/confirm/R5c/curve.json`): TM bandit 0.191, 0.504, 0.760, 0.880, 0.922, 0.934, 0.9365 (still rising); LinUCB 0.389, 0.936, 0.949, 0.949, 0.948, 0.947, 0.946 (best after pass 13, 0.950). The shared 30-pass budget was registered; picking each learner's best pass would be a choice on test data.
- With 150 actions, a TM that only ever learns "this route was right / wrong" ends 0.21 points below the same TM trained on the labels. R5 (top-3 exploration) had failed with 18 and 77 actions; R5c used Boltzmann exploration and 30 passes on a different task (CLINC150 intents), so it does not isolate which change matters, and Banking77 (77 actions, -3.16 in R5b) was not re-tested.
- The cost is on the way: online the TM bandit is right 78% of the time over the 30 passes, LinUCB 94%.

## R8: a TM policy over four steps (clarify-or-route over growing prefixes, CLINC150)

Registration `R8_four_step_agent.md` (frozen 2026-10-16T11:07:16Z); amendment `R8_R5c_amendment_2026-10-16.md`. The query is revealed in 4 prefixes (25, 50, 75, 100% of its words); at each step the agent routes (+1 right, -1 wrong or out of scope), asks for more (-0.1) or escalates (0). The router (a supervised TM, shared by all agent seeds) is right on 21%, 51%, 78% and 95% of in-scope val queries at the four prefixes. Test: 4,500 in-scope + 1,000 out-of-scope episodes; 5 agent seeds; intervals are bootstrap over episodes. Report: `results/confirm/R8/report_R8.json` (`scripts/report_r8.py`).

| Learner | Mean return | In scope | Out of scope | Asks per episode | Out-of-scope escalated |
|---|---|---|---|---|---|
| Rule2 (a margin and a top-sum threshold per step plus an escalation threshold, tuned on val; "the rule": val 0.4865 against 0.4786 for Rule at the 15% weighting) | 0.433 | 0.627 | -0.440 | 2.44 | 86% |
| Lin-Q (linear Q, fitted-Q iteration, memory features) | 0.431 | 0.626 | -0.443 | 2.57 | 85% |
| Rule (margin thresholds only) | 0.417 | 0.635 | -0.562 | 2.45 | 73% |
| **TM-Q (registered TM policy, no memory literals; the variant chosen on one val seed)** | **0.411** | 0.596 | -0.422 | 2.52 | 86% |
| TM-Q with memory literals | 0.399 | 0.586 | -0.443 | 2.55 | 84% |
| Always ask to the last prefix, then route | 0.247 | 0.590 | -1.300 | 3.00 | 0% |
| TM-myopic (gamma 0) | 0.041 | 0.065 | -0.065 | 0.15 | 95% |

- **R8.1 (TM-Q minus TM-myopic >= +0.10, interval above 0): met,** +0.369 [+0.356, +0.383]. Like R6.1 this is nearly built in: a gamma-0 TM cannot value ASK, and even "always ask, then route", which learns nothing, beats it by +0.205 [+0.184, +0.227].
- **R8.2 (TM-Q minus always-ask-then-route, interval above 0): met,** +0.164 [+0.152, +0.176]. It rests on the out-of-scope episodes (+0.88). In scope the TM returns what always asking returns: +0.006 [-0.003, +0.014]. The rule and Lin-Q do gain there (+0.036 and +0.035, internal audit 4).
  - In scope two effects cancel: the TM stops early in 43% of the episodes and routes those well (89 to 96% right), but it escalates 7.8% of in-scope queries (rule 6.2%, Lin-Q 5.3%).
- **R8 as registered: met.** What it shows: trained only on the delayed reward, with TD targets the TM keeps asking through the prefixes (2.44 asks per in-scope episode; the myopic TM escalates 80% of in-scope queries at once) and escalates 86% of the out-of-scope queries at the end instead of routing them. It does not show a finer stopping rule than always asking.
- Reported with no rule, as the exploration predicted:
  - TM-Q minus the rule: -0.022 [-0.028, -0.017] (in scope -0.031, out of scope +0.018).
  - TM-Q minus Lin-Q: -0.021 [-0.025, -0.016]. Lin-Q equals the rule (-0.001 [-0.006, +0.004]).
  - Memory literals: TM-Q with them minus without -0.012 [-0.016, -0.008]. On the one val seed that chose the arm they also cost (-0.018); in the registered runs' own val scores (5 seeds) the two variants are equal (0.4509 and 0.4499).
- Example decisions (test, seed 1; `logs/R8_demo_agent_seed1_test.txt`):
  - "give me" -> ASK (`IF margin_now < 1006 AND step < 3 AND top_now < 15`); "give me my" -> ASK; "give me my credit score" -> ROUTE to credit_score (`IF margin_now >= 2771 AND step >= 1`), return +0.8.
  - "what are the rules for chess" (out of scope): asks three times, then ESCALATE (`IF margin_now < 691 AND top_now < -1481`) instead of routing to roll_dice; return -0.3 instead of -1.3.
- After R8, four one-setting changes were tried on val, one val seed each (exploration e23; 15% weighting): 6 x 20 epochs 0.429, kappa 0.005 0.407 and 8 bins 0.435 scored below every registered seed's val return (0.442 to 0.467); 4,000 clauses (0.452) equalled the registered arm's 5-seed val mean (0.451). Lin-Q's val seeds run 0.470 to 0.478. None closed the gap.
- **Reading:** with TD targets the TM learns a value function over four steps from delayed reward, and its decisions print as rules over named bits. Each decision is the vote of 348 to 574 firing clauses, and the receipt prints only the strongest one (no sufficiency check was run for R8). It is 0.02 below a linear Q-learner and the tuned rule, and the memory literals cost 0.012. A possible reason, not tested: the current prefix already holds what the decision needs.

## R7c: R7b's readable layer again, with the exact budget selection

Registration `R2d_R7c_exact_budget_reruns.md` (frozen 2026-10-16T11:33:29Z); amendment `R2d_R7c_amendment_2026-10-16.md`. R7b's TM arm (budget 16, trim start) with `--budget-select rank`; LR arms are R7's files. Third use of these test splits with this view. Six of the 15 jobs crashed with CUDA out-of-memory while about 19 jobs shared the GPU; the same 6 commands were run again (`logs/jobs_R7c_missing.txt`). The job log, which prints each finished run's test accuracy, had been read to find the crashes; the relaunch repeats the registered commands, so no choice depended on it. 14 of the 30 models are bit-identical to R7b's (their runs met no tie).

| Dataset | TM-readable-16 (10 seeds) | Composite | LR-readable | LR-raw | Longest clause |
|---|---|---|---|---|---|
| CLINC150 | 0.9659 | 0.9662 | 0.9660 | 0.9747 | 16 |
| Banking77 | 0.9322 | 0.9333 | 0.9321 | 0.9416 | 16 |
| HWU64 | 0.9181 | 0.9191 | 0.9182 | 0.9238 | 16 |

- **R7b.1:** -0.01 [-0.17, +0.15], +0.01 [-0.23, +0.24], -0.01 [-0.53, +0.52]. Met in 3 of 3.
- **R7b.2:** -0.84 [-1.20, -0.50], -0.83 [-1.53, -0.15], -0.46 [-1.67, +0.74]. Met in 3 of 3.
- **R7b.3:** no clause over 16 literals in any of the 30 models (all with `budget_select="rank"`). Met.
- **R7b.4 (separate verdict):** CLINC150 out-of-scope AUROC 0.950 (composites 0.957, 0.956). Met.
- **R7c: met.** Calibration error after one temperature 0.015, 0.015, 0.025.
- The readability report was not run again; R7c differs from R7b only where two random keys tied, so R7b's report describes it: short clauses, mostly exclusion lists, a decision spread over thousands of them.

## R2e: R2 again with the fixed budget (CLINC150 word router)

Registration `R2e_R5d_open_parts.md` (frozen 2026-10-16T15:05:29Z). Arm B: budget 32, the trim start and the exact selection; arms A and C are R2's files. Second use of the CLINC150 test split by a lexical router.

| Arm | Test accuracy (mean of 10 seeds) |
|---|---|
| A: default TM (R2) | 0.8197 |
| B: budget 32, trim start, exact selection | 0.8779 |
| C: LR on the same bits (R2) | 0.8809 |

- **R2.1 (B minus A >= +3.0, interval above 0): met,** +5.82 [+5.24, +6.41].
- **R2.2 (median and maximum literals at most 32 in every seed): met;** the longest clause holds 32 literals.
- **R2e: met.** B minus C -0.30 [-0.76, +0.17]: non-inferior to LR on the same bits (R2.3), not ahead.
- With the exact selection R2.2 holds by construction; R2e's new information is that the trim start keeps R2's gain (+5.82 against +5.86, same A arm, same test split).
- Reported with no rule: calibration error after one temperature 0.043 (A) to 0.018 (B); out-of-scope AUROC 0.839 to 0.910 (LR on the same bits 0.902). 97.9% of the bounded clauses are negation-only (A: 24.5%), as in R2.
- With R2d, the fixed budget now meets its registered rules on all three datasets: +5.82 (CLINC150), +4.81 (Banking77), +3.96 (HWU64) points.

## R5d: the TM bandit on Banking77 with 30 passes (77 actions)

Registration `R2e_R5d_open_parts.md` (frozen 2026-10-16T15:05:29Z). R5b's task, machine and stream with 30 passes for every learner; 5 seeds; third use of the Banking77 test split by a bandit.

| Learner | Final test accuracy (mean of 5) | Cumulative online accuracy |
|---|---|---|
| TM bandit (reward only, Boltzmann exploration) | 0.9132 | 0.845 |
| sup-TM (labels, 30 epochs) | 0.9167 | n/a |
| LinUCB (alpha 1.0) | 0.9204 | 0.940 |

- **R5d.1 (TM bandit >= LinUCB - 1.0 point): met,** -0.71 [-0.96, -0.33] (bootstrap over seeds); the whole interval is inside the margin. Per seed +0.03, -1.01, -0.75, -0.94, -0.91: one of five is just over a point behind. The margin is 44 test answers over 5 seeds.
- **R5d.2 (TM bandit >= 0.97 x sup-TM): met,** 0.996.
- **R5d: met.** R5b and R5d differ only in passes: R5d's first 15 passes replay R5b's runs exactly (the same per-pass test accuracies in all 5 TM seeds). With 30 passes for both, the gap shrinks from -3.16 to -0.71: the TM gains 2.0 points and LinUCB loses 0.4. Against LinUCB after 15 passes the TM would be 1.14 behind. On val, replay 8 with 15 passes (0.916) did almost as well as 30 passes (0.919), so more passes and more updates are not separated.
- Mean test accuracy after passes 1, 5, 10, 15, 20, 25, 30 (all 30 in `results/confirm/R5d/curve.json`): TM bandit 0.299, 0.677, 0.846, 0.893, 0.907, 0.911, 0.913; LinUCB 0.593, 0.922, 0.925, 0.925, 0.924, 0.922, 0.920 (best after pass 9, 0.926). LinUCB is ahead at every pass, and online the TM makes 2.6 times its errors (15.5% against 6.0%).

## Phase 2 (2026-10-16 evening): new data and the parts of the plan never tested

Every test below uses data that no phase-1 test had read: MASSIVE 1.1 (en-US and its nb-NO translation; a related dataset (it shares about a third of its test texts with HWU64 train, with different labels); about 40% of its test texts appear in some HWU64 split, found 2026-10-17 in the text-understanding line and counted again for this study), ABCD (customer-service conversations with action logs) and STAR (wizard-of-oz task dialogues; its test split used by R18, and again by R19 in phase 3). Within phase 2 the ABCD test split is used by ten registered tests, in this order: R9, R9b, R10, R10c, R13, R14, R15, R16, R17, R16b (the bandit runs also score it after every pass, for their learning curves; the registered number is the last pass). The en-US test split is used by R11 and, as R12's reference, by R12 (R12's nb-NO test is its translation). Intervals: over test queries (R11, R12); over test conversations (R9, R9b, R10, R10c, R13 to R17, R16b; all action turns of one conversation are resampled together); over test dialogues (R18); over seeds for R15's online accuracy.

## R11: the readable layer on a related dataset (MASSIVE en-US; about a third of its test texts are in HWU64 train, with different labels)

Registration `R11_massive_replication.md` (frozen 2026-10-16T21:45:14Z, before any model was trained or scored on MASSIVE). The registration says that the embedding of the MASSIVE texts was the only step done before the freeze; by file time it was still running then: of its six cache files in `data/emb/` only the first (`massive_en__bge-large.npz`, 21:45:07Z) is older than the freeze, and the other five were written from 21:45:49Z to 21:49:19Z. The embedding reads no labels. R7's and R7c's settings unchanged, bge-large, 10 seeds per TM arm. Two jobs of the bounded arm (seeds 5-6, 9-10) crashed with CUDA out-of-memory (the GPU is shared with other users' jobs) and were re-run with the same commands; three of that arm's seed accuracies had been read in the job log before the re-run.

| Arm | Test accuracy | ECE after one temperature | Longest clause |
|---|---|---|---|
| TM-readable (R7's setting) | 0.8728 (sd 0.0016) | 0.060 (max 0.080) | 874 |
| TM-readable composites (5 + 5) | 0.8719, 0.8736 | | |
| TM-readable-16 (R7c's setting) | 0.8758 (sd 0.0027) | 0.040 | 16 |
| TM-readable-16 composites | 0.8783, 0.8769 | | |
| LR on the same readable bits | 0.8793 | 0.019 | |
| LR on the raw bge-large embedding | 0.8820 | 0.019 | |
| One rule: nearest whitened intent centre (declared in advance) | 0.8722 | | |

- **R11.1 (R7 replicates): not met.** TM-readable minus LR-readable -0.65 [-1.07, -0.22] (the rule asks at least -0.5); its composite minus LR-raw -0.92 [-1.80, -0.03] (at least -1.0: holds).
- **R11.2 (R7c replicates): met.** TM-readable-16 minus LR-readable -0.35 [-0.65, -0.06]; its composite minus LR-raw -0.44 [-1.31, +0.45]; no clause over 16 literals in any of the 10 models.
- **R11.3 (calibration, TM-readable): not met,** 0.060 against the bar 0.05. The bounded arm has 0.040, but the rule names R7's arm.
- **R11 as a whole: not met** (it needs R11.1 and R11.2).
- **Reading.** On a related dataset (it shares about a third of its test texts with HWU64 train, with different labels) the bounded layer (every clause at most 16 literals) passes R7c's rules again, with less room: 0.35 behind LR on the same inputs ([-0.65, -0.06]; on R7c's datasets -0.01 to +0.01) and 0.44 behind the opaque model ([-1.31, +0.45]). The rules judge point estimates, and neither interval lies inside its margin; the same holds for R11.1's composite part (-0.92 [-1.80, -0.03]). The unbounded setting is 0.15 points short of its first rule.
- Reported with no rule: a single TM minus LR-raw is -0.91 [-1.78, -0.04] (R7's setting) and -0.62 [-1.46, +0.24] (R7c's).
- **Readability (declared in advance; `results/confirm/R11/readability.json`, R7's arm):** the strongest firing clause alone gives the decision in 99.1% of test queries, the 3 strongest in 99.8%; the recomputed vote equals the model's decision in 100%. The TM agrees with the one-rule router "nearest whitened centre" on 97.9% of queries (the rule: 0.8722, the TM: 0.8728). As on the first three datasets, the accuracy comes from the encoder and the centres; the TM states that rule as clauses.

## R12: trained in English, used in Norwegian (MASSIVE en-US to nb-NO)

Registration `R12_cross_lingual.md` (frozen 2026-10-16T22:52:06Z). Encoder BAAI/bge-m3, chosen on the Norwegian dev split; the view, the TM, every snapshot and every C use English data only. View `proto:k=1:whiten=0.001` (360 bits: whitened similarity to the 60 English intent centres, nearest and top-3 bits), 10 seeds, everything fitted on en-US train and dev. The nb-NO test split is the translation of the en-US test split that R11 scores.

| Arm | nb-NO test accuracy | en-US test accuracy |
|---|---|---|
| TM-xl (mean of 10 seeds; sd 0.0009) | **0.8216** | 0.8660 |
| TM-xl composites (5 + 5) | 0.8225, 0.8215 | 0.8658, 0.8662 |
| LR on the same bits | 0.8144 | 0.8668 |
| LR on the raw bge-m3 embedding | 0.8399 | 0.8826 |
| One rule: nearest English intent centre | 0.8204 | 0.8642 |

- **R12.1 (the TM transfers as well as a linear model on the same bits): met,** TM-xl minus LR-xl-bits +0.72 [+0.30, +1.16]. The interval is above 0: in Norwegian the TM is ahead of logistic regression on the same named inputs (the rule asked only non-inferiority; in English the two are level, -0.08 [-0.41, +0.24]).
- **R12.2 (cost of readability across languages at most one point): not met,** composite minus LR-xl-raw -1.80 [-2.91, -0.67]. In English the same gap is -1.66 [-2.54, -0.74], so most of the cost comes with the readable view on bge-m3 rather than with the language (post hoc; the two gaps were not tested against each other). With bge-large in English (R11) the composite gap is 0.4 to 0.9. A single TM minus LR-xl-raw on nb-NO: -1.83 [-2.93, -0.70] (reported, no rule).
- **R12.3 (keeps most of its English accuracy): met,** 0.8216 / 0.8660 = 0.949 (the opaque LR keeps 0.952).
- **R12 as a whole: not met** (R12.2).
- **Reading.** A readable layer built once from English examples routes Norwegian requests at 82.2% with no Norwegian training data, ahead of a linear model on the same bits (R12.1 is a non-inferiority rule; its interval also lies above 0; one encoder, one view, one language pair), level with the nearest-centre rule (the TM agrees with it on 97.1% of the Norwegian queries), and 1.8 points below an opaque model on the same encoder.

## R9: the decision layer on a real agent workflow (ABCD next action)

Registration `R9_abcd_rules_over_sequences.md` (frozen 2026-10-16T22:51:09Z); amendment `phase2_amendment_2026-10-17.md`. ABCD test split: 1,004 conversations, 3,608 action turns (which of 30 buttons the human agent pressed next). Frozen bge-large; readable view of 2,791 bits (memory, subflow centres, next-button centres and words of the last 3 utterances and of the whole conversation so far). 10 seeds per TM and MLP arm. Intervals: bootstrap over test conversations.

| Arm | Test accuracy | AST-weighted (the paper's B-Slot) |
|---|---|---|
| TM-mem (readable bits) | **0.8481** (sd 0.0026) | 0.8581 |
| TM-mem composites (5 + 5) | **0.8535** | 0.8618 |
| TM-nomem (no memory bits; the conversation text still lists the earlier actions) | 0.8373 | 0.8491 |
| LR on TM-mem's bits | 0.8445 | 0.8580 |
| LR on TM-nomem's bits | 0.8287 | 0.8444 |
| MLP on TM-mem's bits | 0.8450 (sd 0.0021) | 0.8572 |
| MLP on the raw embeddings plus memory (opaque) | 0.8525 (sd 0.0017) | 0.8646 |
| LR on the same raw input | 0.8473 | 0.8634 |
| Lookup table: most frequent next button given (nearest subflow, last action) | 0.7766 | 0.7913 |
| One rule: nearest next-button centre of the last 3 utterances | 0.6233 | 0.6629 |
| Published, fine-tuned end to end on the dialogue text (paper Table 3) | | BERT 0.899, AlBERT 0.909, RoBERTa 0.936 |

- **R9.1 (memory bits, >= +5 points): not met,** +1.09 [+0.72, +1.47]. The control was not free of history: its whole-conversation parts read the text "action: <button>" of every earlier press (a design error, found when the first seeds came in; amendment). R9b repeats the test with a clean control.
- **R9.2 (TM-mem minus LR on the same bits >= -0.5): met,** +0.36 [-0.29, +1.02].
- **R9.3 (TM-mem minus MLP on the same bits >= -1.0): met,** +0.31 [-0.23, +0.83].
- **R9 as a whole: not met** (R9.1).
- **R9.4 (own verdict; TM composite minus the opaque MLP on raw embeddings >= -1.0): met,** +0.10 [-0.62, +0.82]. The readable composite (a sum of 5 machines, 20,000 clauses) is level with one neural decision head on the raw encoder output; a single TM is 0.44 [-1.09, +0.20] behind it.
- **Fairness note (internal audit 6):** all four ABCD LR-on-bits arms chose C = 0.1, the smallest value of the registered grid, so LR on bits may be slightly under-tuned; that would favour the TM in R9.2, R9.5 and R9b's no-rule TM-vs-LR numbers. LR on the raw input chose C = 3.0.
  - **Checked afterwards (e41, post hoc):** on an extended grid, LR on R9b's short view still peaks at C = 0.1 on dev (0.836; 0.03: 0.831), so R9b's LR was tuned. On R9's full view it peaks lower, at C = 0.03 (dev 0.8474 against 0.8401). Refitted with C = 0.03 and scored on test, it gets 0.8434 (C = 0.1: 0.8445): the dev gain does not carry over, and TM-mem minus this LR is +0.47 [-0.06, +1.01]. R9.2's conclusion does not depend on the grid edge.
- Reported with no rule: LR-mem minus LR-nomem +1.58 [+1.08, +2.10]; TM-nomem minus LR-nomem +0.85 [+0.20, +1.52].
- **R9.5 (own verdict; the TM beats LR on the same bits, interval above 0): not met.** +0.36 [-0.29, +1.02]: ahead, but the interval holds 0.
- **Against the published fine-tuned models** (no rule; not like for like): AST-weighted 0.858 (single) and 0.862 (composite) against BERT 0.899 and RoBERTa 0.936. A frozen encoder with a readable decision layer is 4 to 8 points below models that fine-tune the whole encoder on this dialogue text.
- **By position of the action** (TM-mem): 1st 0.920, 2nd 0.881, 3rd 0.840, 4th 0.797, 5th 0.730, 6th and later 0.452. Every learner falls the same way.
- **Clauses (seed 1):** median 1 literal, longest 84, and 92% of the non-empty clauses have a plain literal, so these are not Part 1's negation-only blacklists. The strongest clauses still carry exclusion lists: the 200 with the largest weight (ranked by their largest class weight) hold 15.0 literals on average, 13.6 of them negated, and clauses longer than 32 literals hold about 17% of the absolute vote weight.
- **Workflow restatement** (precision at K = 86 written guideline transitions a -> b; `results/confirm/R9/workflow_precision.json`): the training logs themselves 0.86, LR's weights on "last action = a" 0.69, the TM's "last action = a" clauses 0.65 (0.57 to 0.73 over seeds). On this measure the linear model restates the written transitions at least as well.

## R9b: memory bits against a control with no record of the earlier actions

Registration `R9b_memory_clean_control.md` (frozen 2026-10-16T23:27:00Z, after R9's first seeds; it says so). Second use of the ABCD test split. Views: `mem,intent,step,words` against `intent,step,words`; no part reads an action line. 10 seeds each.

| Arm | Test accuracy | AST-weighted |
|---|---|---|
| TM-mem-short | **0.8401** (sd 0.0020) | 0.8490 |
| TM-nomem-short | 0.6537 (sd 0.0035) | 0.6958 |
| LR-mem-short | 0.8390 | 0.8538 |
| LR-nomem-short | 0.6749 | 0.7148 |

- **R9b.1 (memory bits, >= +5 points, interval above 0): met,** **+18.65 [+17.20, +20.10] points** (AST-weighted +15.31).
- **R9b: met.** Without memory bits a decision layer that sees only the customer's words and the last 3 utterances gets 65% of the next actions right; with the buttons pressed so far as named literals it gets 84%.
- Reported with no rule: memory adds +16.41 [+14.95, +17.87] to LR. With memory the TM is level with LR (+0.12 [-0.58, +0.84]); without memory it is 2.12 points behind it (-2.12 [-2.87, -1.39]). The TM gains 2.2 points more from the memory bits than the linear model does (post hoc), because without them it is 2.1 points behind LR; with them the two are level.
- Reported with no rule: R9's TM-nomem (whose whole-conversation text lists the earlier presses) minus TM-nomem-short: +18.36 [+16.93, +19.81]. The history carried as words gives back 18.4 of the 18.65 points that the memory bits add.
- Workflow restatement on this view (`results/confirm/R9b/workflow_precision.json`): the TM's "last action = a" clauses 0.69 (0.65 to 0.71 over seeds), LR 0.78, the logs 0.86. A possible reason, not tested: the TM states many of its steps through done(...) literals, which this measure does not count. Read it as "the linear weights list the written transitions more directly".
- **Conjunctions of history and context (post hoc, 10 seeds):** 39% of the absolute vote weight (35 to 40%) sits in clauses that hold both a memory literal and a context literal (customer, last 3 utterances), 8% in clauses with a plain literal of each kind ("last action = X AND near(step Y) ..."), 8% in memory-only clauses and 53% in context-only clauses. So the machine does combine the two; logistic regression can only add them.
- The memory effect is spread over the whole conversation: 1st action 0.912 against 0.775 (the bits also say "no action yet"), 2nd 0.870 against 0.617, 3rd 0.840 against 0.605.
- **What the clauses say** (R9b's ABCD receipts for three test conversations, seed 1; project notes, not published): "[validate-purchase] +352: IF last action = pull-up-account AND NOT (nearest next-button of the last 3 turns = verify-identity) AND NOT the last 3 turns contain "account_id" AND NOT they contain "cart"" (4 literals); "[pull-up-account] +341: IF pull-up-account in the top 3 next buttons of the last 3 turns AND NOT done(pull-up-account) AND NOT the last 3 turns contain "okay agent"" (3 literals); "[enter-details] +490: IF done(pull-up-account) AND NOT done(record-reason) AND NOT done(instructions) AND NOT last action = enter-details AND 71 more negated literals" (75 literals: 1 plain, 74 negated). Some strong clauses are short "if A and not B" rules of the plan's kind; others are one plain literal plus 30 to 95 exclusions. On this view the 200 strongest clauses of seed 1 (ranked by their largest class weight) hold 27.5 literals on average (25.9 negated; longest 113).

## R10: the workflow learned from reward alone (ABCD, the agent loop)

Registration `R10_abcd_reward_only.md` (frozen 2026-10-17T01:07:02Z). At every action turn the learner presses one of 30 buttons and is told only whether it was the human agent's button (reward 1 or 0), Type I/II feedback as in R5 to R5d, Boltzmann exploration on the class sums, 15 passes over the 29,190 training action turns in rounds of 32. Inputs: memory bits, words of the last 3 utterances, and unsupervised k-means clusters of the last 3 utterances and of the customer's words (1,706 bits). No bit is built from the label of the decision being learned or from subflow labels. The memory bits are the buttons pressed earlier in the logged conversation, i.e. the human agent's presses (the same history the paper's AST models get): R10 replays logged conversations; a closed loop in which the TM's own mistakes enter its memory was not tested. 5 seeds; final greedy accuracy on the 3,608 test action turns.

| Learner | Test accuracy (mean of 5) | Online accuracy over the stream |
|---|---|---|
| TM bandit (reward only) | **0.8300** | 0.718 |
| TM bandit without memory bits | 0.6013 | 0.538 |
| Supervised TM on the same bits (labels, 15 epochs) | 0.8404 | n/a |

- **R10.1 (reward alone reaches the supervised machine, ratio >= 0.97): met,** 0.988 (difference -1.04 [-1.49, -0.59] points).
- **R10.2 (memory bits in the agent loop, >= +5 points, interval above 0): met,** **+22.87 [+21.37, +24.39] points.**
- **R10: met.** From a 0/1 reward per press, with no label for the decision and no named intent, the TM learns the agents' multi-step workflow to 0.988 of what labels give it, and the memory literals (the logged earlier presses) are worth 23 points of it.
- Learning curve (mean test accuracy after passes 1, 5, 10, 15): 0.542, 0.701, 0.790, 0.830, still rising at the end. The test split is scored after every pass for this curve (as in R5 to R5d); the registered number is the last pass in every run, with no selection.
- LinUCB is not part of R10 (no LinUCB run had finished on dev when R10 was frozen); it is R10c, below.
- The bandit's clauses on these bits are long exclusion lists (seed 1: median 106 literals, 71% with no plain literal), the Part 1 pattern; its decisions are exact votes of named clauses but not short rules. R9b's supervised view gives clauses with a plain literal, though its strongest clauses are long too.

## R10c: the TM bandit against LinUCB on the ABCD workflow

Registration `R10c_abcd_linucb.md` (frozen 2026-10-17T01:40:29Z; it states that R10's TM results were known then). LinUCB (`LinUCBFast`, the same estimator as `LinUCB` with Woodbury updates; checked on a small problem, not at this scale) on the same 1,706 bits and the same 15-pass stream, alpha 0.1 / 0.5 / 1.0, 3 seeds each; the TM bandit is R10's 5 runs, unchanged. Two LinUCB runs (alpha 0.1 and 1.0, seed 3) crashed at start with CUDA out-of-memory and ran again with the same commands when slots freed (amendment). Dev, before the freeze: LinUCB alpha 0.5 0.7725, TM bandit 0.8350.

| Learner | Test accuracy | Best pass on test (chosen on test) | Online accuracy over the stream |
|---|---|---|---|
| TM bandit (R10) | **0.8300** | 0.8300 (last pass is the best) | 0.718 |
| LinUCB alpha 1.0 | 0.8000 | 0.820 | 0.759 |
| LinUCB alpha 0.5 | 0.7749 | 0.805 | 0.813 |
| LinUCB alpha 0.1 | 0.7309 | 0.780 | 0.865 |

- **R10c.1 (TM bandit minus LinUCB at the dev setting, alpha 0.5, >= +2.0 with the interval above 0): met,** **+5.51 [+4.66, +6.39] points.**
- **R10c.2 (TM bandit minus LinUCB at its best alpha on test, interval above 0): met,** **+3.00 [+2.18, +3.81]** against alpha 1.0.
- **R10c: met.** It is the first registered test of the study in which the TM is asked to beat a linear learner and does (R9.5 was not met). AST-weighted: TM 0.8427; LinUCB 0.7552 / 0.7950 / 0.8164 (alpha 0.1 / 0.5 / 1.0).
- **Why, not tested.** One possible reason: the task needs conjunctions of history and context, and LinUCB scores each button with one linear function of the bits; but with labels, LR on R9b's view is level with the TM (+0.12), and no LR was fitted on R10's bits. The curves suggest another: LinUCB's test accuracy peaks after 3 to 5 passes (0.780, 0.805, 0.820) and then falls, while its accuracy on the training stream climbs to 0.90 to 0.91 in pass 15 (TM 0.825), as if 15 replays over-fit it at ridge 1 (not varied). The TM is still rising at pass 15.
- **What it does not show.** LinUCB earns more reward while learning (online 0.76 to 0.87 against 0.718), and in test accuracy it is ahead after each of the first 7, 9 and 11 passes (alpha 0.1, 0.5, 1.0), so the verdict holds for the 15 registered passes. Its accuracy rises with alpha over the registered range (0.73, 0.77, 0.80). **Larger alphas, tried afterwards on dev (e30, exploration, not registered, one seed per alpha):** alpha 2.0 ends at 0.816 and alpha 4.0 at 0.830, close to the TM bandit's 0.835 on dev, but LinUCB's online accuracy falls to 0.650 and 0.460 (TM 0.721). So R10c's superiority holds for the registered range of alphas. On dev, larger alphas trade reward while learning for end accuracy, which is still rising at 4.0; LinUCB's ridge and the number of passes were not varied (post hoc). Even LinUCB's best single pass on test (an upper bound chosen on test) is a point below the TM's last pass. One task, one input view, 15 passes.

## R13: the closed loop (the memory holds the model's own presses)

Registration `R13_closed_loop.md` (frozen 2026-10-17T04:09:25Z). R9b's registered models, no training: in every test conversation the action turns are decided in order, and the memory bits at turn k come from the model's own presses at turns 0 to k-1 (the text bits stay as logged, including the human agent's own utterances: 93% of the later action turns have one among the last 3 utterances, e.g. "i have pulled up your account" (internal audit 7). Only the memory bits are the model's own, so a fully closed loop probably costs more than the 3.3 points measured here). Fifth use of the ABCD test split. The open-loop accuracies the closed-loop script recomputes equal R9b's registered ones seed by seed.

| Arm | Test accuracy | AST-weighted |
|---|---|---|
| TM-mem-short, closed loop (10 seeds) | **0.8070** (sd 0.0030) | 0.8201 |
| TM-mem-short, open loop (R9b) | 0.8401 | 0.8490 |
| TM-nomem-short (R9b; no memory, so no loop) | 0.6537 | 0.6958 |
| LR-mem-short, closed loop | 0.8071 | 0.8281 |
| LR-mem-short, open loop (R9b) | 0.8390 | 0.8538 |

- **R13.1 (memory still pays in the closed loop, >= +10 points, interval above 0): met,** **+15.34 [+13.80, +16.91] points.**
- **R13.2 (its own mistakes cost at most 4 points): met,** closed minus open -3.31 [-3.95, -2.71].
- **R13: met.** With its own presses as memory the TM still gets 80.7% of the next actions right, against 65.4% without memory. Its mistakes cost 3.3 points; logistic regression loses the same (-3.19), and the two are level in closed loop (-0.01 [-0.86, +0.83]).
- **Where it hurts:** the loss grows along the conversation (closed against open loop: 1st action 0.912 / 0.912, 2nd 0.840 / 0.870, 3rd 0.787 / 0.840, 4th 0.720 / 0.787, 5th 0.650 / 0.699, 6th and later 0.415 / 0.461). From the 6th action on (126 turns in 61 conversations) the closed-loop TM is no better than the TM without memory (0.415 against 0.437; -2.2 [-6.6, +3.0], post hoc): late in long conversations the memory gain disappears.
- **Two remedies tried on dev (exploration, not registered; `results/explore/abcd_closed_loop.jsonl`, e32):** noisy memory copies in training (closed loop 0.8113 against 0.8149 for teacher forcing) and one round of cross-fitted data aggregation (0.8067). Neither helped in its one dev run. A possible reason, not tested: when the model's memory is wrong the logged next button stays the label, so they teach the model to trust its memory less.

## R14: the TM bandit against a neural bandit (ABCD, the "NN agent" comparison)

Registration `R14_abcd_neural_bandit.md` (frozen 2026-10-17T04:13:10Z; R10's TM results were known then). The neural bandit: one hidden layer of 512 from the same 1,706 bits to one logit per button (the chance of reward 1), one AdamW step per round of 32 on the chosen buttons' binary cross-entropy, Boltzmann exploration 0.5; its best of four settings on dev. Same stream as R10, 5 seeds. Sixth use of the ABCD test split.

| Learner | Test accuracy | Best pass (chosen on test) | Online accuracy | AST-weighted |
|---|---|---|---|---|
| TM bandit (R10) | 0.8300 | 0.8300 | 0.718 | 0.8427 |
| Neural bandit | 0.8345 | 0.838 | 0.801 | 0.8476 |

- **R14.1 (TM bandit minus neural bandit >= -1.0 point): met,** -0.45 [-1.04, +0.16]. The rule judges the point estimate; the interval reaches just past the margin.
- **R14: met.** From a 0/1 reward alone the TM agent ends within the registered margin of a small neural bandit on the same inputs, judged on the point estimate (-0.45; the interval [-1.04, +0.16] reaches just past the -1.0 margin; [-1.33, +0.44] with the training seeds also resampled, post hoc). The network is minimal: one hidden layer, one AdamW step per round, the best of four one-seed dev settings. It learns more slowly: the neural bandit earns 8 points more reward over the stream (0.801 against 0.718), and on dev it was ahead at the end too (0.8415 against 0.8350).

## R15: replay closes most of the TM bandit's slow-learning gap

Registration `R15_bandit_replay.md` (frozen 2026-10-17T04:30:39Z; R10, R10c and R14's results were known then). R10's TM bandit with replay: after each round, 4 extra TM updates on minibatches of everything it has logged (context, its own press, the reward), as R5's bandit did; no label is used. 5 seeds on R10's stream. Seventh use of the ABCD test split.

| Learner | Final test accuracy | Online accuracy over the stream | Run time |
|---|---|---|---|
| TM bandit + replay 4 | **0.8397** | **0.782** | 2,324 s |
| TM bandit, no replay (R10) | 0.8300 | 0.718 | 1,000 s |
| Neural bandit (R14) | 0.8345 | 0.801 | 77 s |

- **R15.1 (it learns faster; online +3.0 points with the interval above 0): met,** **+6.35 [+5.87, +6.81]** (bootstrap over seeds).
- **R15.2 (as accurate as the neural agent at the end, >= -1.0): met,** **+0.52 [+0.03, +1.02]**: with replay the TM bandit ends half a point ahead of the neural bandit. The rule asked only non-inferiority; the interval holds the trained models fixed, and with the 5 training seeds also resampled it is [-0.26, +1.28] (post hoc), so "ahead" is not established.
- **R15.3 (own verdict; online gap to the neural agent at least -3.0): met,** -1.90 [-2.34, -1.49] (without replay -8.3).
- **R15: met.** Replay closes 77% of the TM's online gap to the neural agent: it still earns 1.9 points less reward while learning (online 78.2% against 80.1%), and ends level with it (+0.52 under a non-inferiority rule). Against no replay it also ends higher (+0.97 [+0.55, +1.39]).
- **The comparison is uneven:** the TM gets 5 updates per round (replay 4), the neural bandit 1 and no replay; replay was never tried for the network. R15 shows that replay closes most of the TM's gap, not that the TM learns as fast as a network given the same extra updates. The replay runs are not bit-reproducible (two seed-1 runs differ by about 0.3 points of online accuracy; internal audit 7).
- Reported: mean test accuracy after passes 1, 5, 10 and 15: 0.606, 0.784, 0.834, 0.840 (R10 without replay: 0.542, 0.701, 0.790, 0.830). The clauses are like R10's: median 104 to 105 literals, 68 to 69% without a plain literal.
- The price is compute: 5 times the TM updates, 2.3 times the run time; the neural bandit runs in 77 s, 30 times faster.

## Explorations after the registered tests (ABCD dev only, one seed, not registered)

- **Closed-loop remedies** (e32): see R13. Noisy memory copies and one round of cross-fitted data aggregation both lower closed-loop accuracy (0.8113, 0.8067 against 0.8149 for teacher forcing).
- **Shorter clauses after training** (e36, `scripts/abcd_prune.py`, R9b's seed-1 model): dropping exclusions that block at most m training examples. m = 0 removes almost nothing (the 200 strongest clauses: 27.5 -> 27.4 literals; training votes identical); m = 100: 23.2 literals, dev 0.8385 (from 0.8398); m = 500: 9.3 literals, 0.8241. The exclusion lists carry accuracy; there is no free shortening.
- **A stronger encoder** (e37): Qwen3-Embedding-0.6B (32k-token context) instead of bge-large: TM on R9b's short view 0.8385 (bge-large 0.8398), on R9's full view 0.8496 (0.8504). The frozen encoder is not the bottleneck at this level; the 4- to 8-point gap to the fine-tuned models stays.
- **The written guidelines added to the trained TM as clauses** (e40, `scripts/abcd_inject.py`, R9b's models seeds 1 to 3): for each subflow s and each written step a -> b, one clause "IF nearest subflow = s AND last action = a AND NOT done(b) THEN +w for b" (and the first steps), 224 clauses; at least one fires on 87.8% of dev action turns. Each added clause can be read and checked on its own: whenever its literals hold, b gets exactly w more votes, for every input. Dev accuracy change, seeds 1 / 2 / 3: w 100: +0.03 / 0.00 / +0.19 points; w 800: +0.09 / +0.40 / +0.46 (from 0.8398 / 0.8388 / 0.8393). The machine had already learned most of the written workflow; adding it by hand does not hurt and helps a little. Not registered.
- **Granmo's hierarchical TM** (e35, e38; `PyHierarchicalTsetlinMachineCUDA` at commit 5b4a223 in this study's own `.venv-htm`; `scripts/abcd_htm.py`; R9b's short view, its 4 blocks zero-padded to 1,000 bits; one machine per button). With 2 OR-groups of 4 options per block (s 8) it did not learn (0.14 to 0.30 dev accuracy over 5 epochs; cause not tested: in the package an empty option outputs 1, but that holds for the Hex shape too). In the shape of the package's Hex demo (one OR of 8 options per block, 400 clauses and T 1,000 per button, s 25) it learns: 0.795 after 6 epochs, 0.8157 after 40. The package's flat machine reaches 0.7153 at its better setting (s 4); with the hierarchical run's settings 0.25 to 0.46. Inside the package the hierarchical configuration is about 10 points ahead (with 2 times the clauses and 16 times the automata); this study's coalesced flat TM on the same bits is 2.4 points ahead of the hierarchical one (0.8398) and about 13 to 17 times faster in training (116 s for 40 epochs with five package jobs on the GPU, against 1,999 s of training for the hierarchical machine, which also spent 2,298 s scoring dev after every epoch). Registered as R16 and R16b (below).

## R16: Granmo's hierarchical TM (AND-OR clauses) on the ABCD workflow

Registration `R16_hierarchical_tm.md` (frozen 2026-10-17T07:13:31Z); amendment (the runs were restarted once without per-epoch dev scoring). `PyHierarchicalTsetlinMachineCUDA` at commit 5b4a223, in this study's `.venv-htm`; R9b's short view, its 4 blocks zero-padded to 1,000 bits; one machine per button; 30 epochs, the last scored; 3 seeds each. Eighth use of the ABCD test split.

| Machine | Test accuracy (mean of 3) | AST-weighted | Run time |
|---|---|---|---|
| Hierarchical: per block 8 options (conjunctions of that block's literals), counted, and the 4 block counts multiplied, so a clause votes 0 to 4,096; 400 clauses, T 1,000, s 25 per button | **0.8174** (sd 0.0026) | 0.8350 | about 8,550 s |
| The package's flat machine: 200 clauses, T 200, s 4 per button | 0.7234 (sd 0.0047) | 0.7250 | about 8,550 s |
| This study's coalesced flat TM on the same bits (R9b, 10 seeds) | 0.8401 | 0.8490 | dev (e39): 116 s for 40 epochs with five package jobs on the GPU; the hierarchical dev run: 1,999 s of training for 40 epochs (one to four other jobs) plus 2,298 s of dev scoring |

- **R16.1 (hierarchical minus flat, >= +3.0 points with the interval above 0): met,** **+9.40 [+8.35, +10.41] points.**
- **R16: met.** Inside Granmo's package, the hierarchical machine as configured (400 clauses per button; each clause per block a count over 8 conjunctions, the blocks multiplied; 16 times the automata of the flat machine; T 1,000, s 25) scores 9.4 points above the package's flat machine at the better of the flat settings tried (200 clauses, T 200, s 4). Size and settings differ as well as the structure, so the gain is not the AND-OR structure's alone; a flat machine of matched size was not tried here (tried afterwards: R16b), and the 30 epochs are the hierarchical run's best dev epoch.
- Reported with no rule: the hierarchical machine is 2.28 [-2.92, -1.68] points below this study's coalesced flat TM on the same bits, and the package's flat machine 11.7 below it. The two flat machines differ in implementation (one-vs-rest machines with balanced negatives against one coalesced clause bank with hardest-negative feedback) and in size and settings (200 clauses per button against 4,000 shared; T, s; last epoch against R9b's best-dev snapshot); which of these causes the 11.7 points was not tested. The hierarchical configuration closes 80% of that gap. A coalesced hierarchical machine was not tried.
- The first hierarchy tried (2 OR-groups of 4 options per block, s 8) did not learn (e35); the shape and the settings matter.
- Declared but not possible: examples of hierarchical clauses (no model was saved).
- The size question: a first try on dev (e42) was stopped when an epoch took 28 minutes on the shared machine; when the GPU was free again it was answered by e44 and R16b (below).

## R17: the reward-learned TM agent in the closed loop

Registration `R17_bandit_closed_loop.md` (frozen 2026-10-17T11:56:26Z; R10's and R13's results were known then). R10 saved no models and its runs are not reliably bit-reproducible on the shared GPU (seed 1 reproduced R10's seed 1 exactly, seeds 2 to 5 did not), so 5 new runs with R10's setting were trained and saved (launch commands `logs/jobs_R17.txt`). After training, every test conversation is decided in order with the memory bits built from the agent's own earlier presses; the text bits stay as logged (as in R13). Ninth use of the ABCD test split.

| Arm | Test accuracy | AST-weighted |
|---|---|---|
| TM bandit, closed loop (5 new seeds) | **0.7910** (sd 0.0073) | 0.8060 |
| The same models, logged memory (open loop) | 0.8279 (R10's registered runs: 0.8300) | 0.8406 |
| R10's TM bandit without memory bits | 0.6013 | 0.6639 |

- **R17.1 (memory still pays in the closed loop, >= +10 points with the interval above 0): met,** **+18.96 [+17.30, +20.61] points.**
- **R17.2 (its own mistakes cost at most 4 points): met on the point estimate,** -3.70 [-4.44, -2.99]; the interval reaches past the margin.
- **R17: met.** The agent that learned the workflow from a 0/1 reward (in training its memory bits were the human agent's logged presses, as in R10) keeps most of its memory gain when its memory holds its own presses at test: 79.1% against 60.1% without memory. Its own mistakes cost 3.7 points (the supervised TM in R13: 3.3), and in closed loop it is 1.60 [-2.46, -0.76] points below the supervised TM (80.7%).
- Reported with no rule: the new runs' open-loop accuracy minus R10's registered runs -0.21 [-0.49, +0.08] (replication). Dev before the freeze (e43, seed 1): open 0.8246, closed 0.7983.
- By position (closed against open loop): 1st action 0.910 / 0.910, 2nd 0.805 / 0.839, 3rd 0.763 / 0.821, 4th 0.719 / 0.793, 5th 0.649 / 0.702, 6th and later 0.419 / 0.470.
- As in R13, only the memory bits are the agent's own; the human agent's logged utterances stay in the text, so a fully closed loop probably costs more.

## R16b: the hierarchical TM against a flat machine of matched size

Registration `R16b_flat_matched_size.md` (frozen 2026-10-17T12:38:55Z; R16's results were known then). Internal audit 7's main point about R16: size, settings and the vote differ along with the structure. R16b removes the size difference: the flat machine gets the same number of automata, 3,200 clauses per button (3,200 x 4 leaves = 400 x 32), with s 4 and T 3,200 (the only setting tried at this size; at s 25 a 400-clause flat machine with T 1,000 reached 0.25 to 0.46 in e38), 30 epochs, 3 seeds. R16's hierarchical runs unchanged. Tenth use of the ABCD test split.

| Machine | Test accuracy (mean of 3) | AST-weighted |
|---|---|---|
| Hierarchical (R16) | **0.8174** | 0.8350 |
| Flat, 3,200 clauses per button (matched automata) | 0.7471 (sd 0.0014) | 0.7505 |
| Flat, 200 clauses per button (R16) | 0.7234 | 0.7250 |

- **R16b.1 (hierarchical minus flat of matched size >= +3.0 with the interval above 0): met,** **+7.02 [+6.01, +8.04] points.**
- **R16b: met.** At the one setting tried for the large flat machine (T equal to the clauses, s 4, 30 epochs; on dev it was still rising at epoch 30: 0.7332, 0.7375, 0.7427 after 10, 20, 30), 16 times the automata lift it by +2.37 [+1.75, +3.00], about a quarter of R16's 9.4 points, and the hierarchical machine stays 7 points ahead. What still differs: T and s (1,000 / 25 against 3,200 / 4), and the vote, which in the package is a count of true options per block multiplied over the blocks (0 to 4,096) rather than 0 or 1. Both machines are still below this study's coalesced flat TM (0.8401). Run times: 2,229 to 2,235 s per flat run, the three runs at once (about 71 s of training per epoch); not comparable with R16's about 8,550 s, run six at once.
- Dev sizes tried before the freeze (e44): 400 clauses 0.7326, 800 clauses 0.7419, 3,200 clauses 0.7427.

## R18: memory literals on a second workflow dataset (STAR)

Registration `R18_star_replication.md` (frozen 2026-10-17T14:26:43Z). STAR (Mosig, Mehri, Kober 2020; github.com/RasaHQ/STAR, MIT): wizard-of-oz dialogues over 24 tasks with written flowcharts; one decision per labelled wizard action (187 labels, API queries included); splits by a hash of the dialogue id, fixed before any model (`scripts/star_prepare.py`). R9b's short view; 16,000 clauses; 5 seeds per TM and MLP arm. First use of STAR's test split (6,988 actions in 611 dialogues): new dialogues of the same 24 tasks and scenarios, from the same pool of workers (no new task). As in R9b the memory holds the wizard's logged actions (open loop); the closed loop was not tested on STAR. Commands in `logs/jobs_R18.txt`.

| Arm | Test accuracy |
|---|---|
| TM-mem | **0.6265** (sd 0.0047) |
| TM-nomem | 0.4722 (sd 0.0027) |
| LR-mem | 0.6591 |
| LR-nomem | 0.5190 |
| MLP-mem | 0.6580 (sd 0.0050) |

- **R18.1 (memory literals pay on STAR too, >= +5.0 points with the interval above 0): met,** **+15.43 [+14.38, +16.51] points.**
- **R18: met.** The memory result replicates on a test split that had never been read: with the wizard's earlier actions as named literals the TM gets 62.7% of the next actions right; without the memory bits, 47.2%.
- **What the control still sees (internal audit 8, post hoc):** the control without memory bits still reads the wizard's last message. In STAR that is the template text of the last action in 60% of later decisions, and such a text names its action. Where it does, memory adds +8.8 points; after an API query, which leaves no text, it adds +30.0. So +15.43 understates what the whole history is worth, and it is not directly comparable with ABCD's +18.65, where the actions are button presses, not utterances.
- **What does not replicate (reported, no rule):** on STAR the TM is behind the linear and neural heads on the same bits, -3.26 [-4.01, -2.52] against LR and -3.15 [-3.78, -2.50] against the MLP (on ABCD it was level). With the training seeds also resampled: -3.26 [-4.13, -2.39] and -3.15 [-4.03, -2.23] (post hoc, internal audit 8). LR chose C = 0.1, the edge of its grid, in both arms (as on ABCD); post hoc on dev, C = 0.03 is 0.1 point better and C = 0.01 2.0 points worse, so the edge cost LR little. 16,000 clauses lifted the TM from 0.613 to 0.639 on dev (seed 1, best at epoch 18; the registered runs' best dev epochs were 18 to 30; 3,368 to 4,343 s of training each, five at once). A possible reason, not tested by R18: 187 actions share one clause bank (86 clauses per action, against 133 on ABCD). Where the gap sits (post hoc, dev, `scripts/star_gap_diagnose.py`): in the less common actions (21 to 400 train examples: 7 to 9 points behind LR), not in the common ones (level). LR's memory gain is +14.01 [+12.85, +15.17]; without memory the TM is 4.68 points behind LR.
- By position (TM-mem / TM-nomem / LR-mem): 1st 0.783 / 0.739 / 0.791; 2nd 0.675 / 0.520 / 0.725; 3rd 0.650 / 0.469 / 0.670; 6th and later 0.595 / 0.434 / 0.631.

## Phase 3 explorations (2026-10-17 evening; dev only, one seed, not registered)

**Why the TM trails LR on STAR (after R18).**
- **Diagnosis** (d1, `scripts/star_gap_diagnose.py`, R18's saved models on dev):
  - The 5 seeds score 0.639 to 0.640 alone and 0.654 as a summed vote; LR scores 0.666.
  - Ties at the top are not the cause (0.01%).
  - **The gap sits in the less common actions:**

    | Train examples per action | TM | LR |
    |---|---|---|
    | 21 to 100 | 24.6% | 33.7% |
    | 101 to 400 (80 actions, 31% of dev) | 48.0% | 55.0% |
    | 401 to 1,500 | 78.1% | 77.5% |
    | over 1,500 | 67.6% | 69.3% |

  - On 9.8% of dev inputs no action gets a positive vote.
- **Per-class vote corrections after training** (d2, `scripts/star_vote_norm.py`; tau picked on one half of dev and scored on the other): at most +0.3 points. The weakness is in the clauses, not in the scale of the votes.
- **Literal budget** (e47, 4,000 clauses, trim start): budgets 16, 32 and 64 all reach 0.637 to 0.639, against 0.613 without one. That is the level of the 16,000-clause machine without a budget. T 1,000 instead of 4,000 lowers the score (0.606).
- **Class-balanced draws** (e48, `src/dl/tm_balance.py`, rho 0.5): 0.610. The rare actions gain (21 to 100: 39.3%; 101 to 400: 52.8%) and the common ones lose by as much.

**One score per conversation, closed loop** (`scripts/abcd_episodic.py`; ABCD; R10's label-free view; 4,000 clauses; Boltzmann 0.5; rounds of 9 conversations).
- The agent decides every action turn in order, with its own presses as memory. It sees only one number at the end: how many presses were right.
- The reference with a reward after every press reaches 0.809 dev accuracy after 30 passes (e49_step).
- **Credit rules with R10's update** (a 1 makes the press the target and pushes the strongest other action down):
  - uniform credit (each press drawn as 1 with probability score / presses) stops at 0.39 from pass 4;
  - the posterior given the score stops at 0.21;
  - learning only from conversations that are all right or all wrong stops at 0.30.
  - In R10's update, a wrong press drawn as 1 becomes the target. So noisy credit teaches the wrong actions.
- **The same credit with a regression-style update** (`--update reg`): the pressed action's vote, read as a reward in [0, 1], moves toward the credit, and no other action is touched.
  - uniform credit: 0.652 after 30 passes, still rising;
  - **leave-others-out credit** (`loo`: the score minus the learner's own predicted reward of the conversation's other presses): **0.733 after 30 passes**, which is 0.91 of the per-press reference, still rising;
  - the posterior given the score still gets stuck (0.25);
  - presses fixed by all-right or all-wrong conversations taken as labels, the rest by `loo` (`hybrid`): ahead of `loo` only over passes 1 to 4 (0.597 at pass 10, against 0.588).
  - With a reward after every press, the regression update learns faster early on (passes 1 to 4: 0.577 to 0.676, against 0.513 to 0.609 with R10's update) but ends 2.1 points lower (0.788 against 0.809 after 30 passes; its best was 0.800 at pass 25).
- **A small neural reward model with the same one-score credit** (e62: `--learner neural --credit loo --soft`, the R14 network, one AdamW step per round) learns far faster: 0.584 dev after one pass, 0.757 after 5, **0.803 after 30** (closed loop; open loop 0.829). That is about level with the TM's per-press reference (0.809). The TM with the same credit reaches 0.733.
  - **From a delayed score, the TM learns the workflow (0.91 of its per-press accuracy), but much more slowly than a neural learner.** This is a failure-map line, not a win.
  - The per-press neural run (e56_neural_step) was not run; it was paused for R19.

## R19: the agent with no neural network at run time (STAR and ABCD)

Registration `R19_no_network_agent.md` (frozen 2026-10-18T09:40:15Z). Its dev rules are in `logs/plan_e60_static.md` (frozen 2026-10-17T21:09:33Z). The plan was fixed before any run. Every gate was checked independently, and both reports were recomputed byte for byte.

**The question.** R18's agent (STAR) and R9b's (ABCD) read their text through bge-large-en-v1.5. Can the agent decide with no neural network at run time, and what does that cost?

**What runs per decision.** No neural network at run time: a rule-based tokenizer, lookups in a fixed 63,091 x 512 table, then a bitwise OR of precomputed per-token bit codes, then Boolean clauses and an integer vote. The table was distilled once, offline, from a transformer (bge-base-en-v1.5 via Model2Vec); the reference agent runs bge-large-en-v1.5 (335M parameters) per decision. Speeds are rough.
- **The table:** potion-base-32M, revision 1e5a03f8 (file hashes checked at load).
- **The text bits:** "some word of the customer's words so far (or of the last 3 turns) is close to concept k at level j".
  - Each field has 512 concepts: k-means on the table rows of the training tokens.
  - Each concept is named by its 4 nearest tokens, for example "concept 0 (apt|apts|hut|tar)".
  - Each concept has up to 7 cosine levels: thresholds from a thermometer with 8 bins over the training max-cosines, duplicate thresholds dropped.
  - At run time no table vector is read: each token's bit code (one row per table token, computed once from the table when the view is fitted) is looked up and ORed.
  - This is the same procedure as the existsT bits of the text-understanding line (project notes, not published), but the concepts are drawn on the CPU, so they are not that line's.
- The memory and word bits are as before.

**Dev path** (seed 1, not registered; rules S1 to S3 frozen before the first dev run):

| View | STAR TM, 4,000 clauses | STAR LR | STAR TM, 16,000 clauses | ABCD TM, 4,000 clauses | ABCD LR |
|---|---|---|---|---|---|
| A: mem, intent, step, words, on the table's mean vectors (2,715 / 1,581 bits) | 0.6126 | 0.6655 | 0.6296 | 0.8382 | 0.8407 |
| **B: mem, words, xcust, xrecent, lookup + OR (7,938 / 7,317 bits)** | 0.6074 | 0.6409 | **0.6343** | **0.8325** | 0.8111 |
| A+B: mem, intent, step, xrecent, words (5,899 / 4,614 bits) | 0.6042 | 0.6433 | 0.6378 | 0.8341 | 0.8097 |
| B without mem (e61; 7,558 / 7,251 bits) | | 0.5109 | 0.5010 | 0.6200 | 0.6547 |
| bge-large, with mem (e45, e46, e25_full) | 0.6131 | 0.6657 | 0.6394 | 0.8398 | 0.8360 |
| bge-large, without mem (e45, e46, e25_nomem) | 0.4862 | 0.5229 | 0.4919 | 0.6414 | 0.6667 |

- **S1** (STAR, 4,000 clauses): the best view was A (0.6126), and none was more than 3.0 below it, so all three went on to 16,000 clauses.
- **S2** (STAR, 16,000 clauses): take the first of B, A, A+B within 0.5 points of the best. The best was A+B (0.6378); B was 0.35 below, so **B** was chosen.
  - B's dev curve peaks at epoch 22 and drifts down to 0.6144 at epoch 40.
  - Epoch-to-epoch swings are 1 to 2 points, so the 0.35-point gap is noise.
- **S3** (the dev cost is small enough to spend a test split): STAR B 0.6343 >= 0.6094 and ABCD B 0.8325 >= 0.8098, so both datasets went to test.
  - B's dev cost against bge-large: STAR -0.51 points, ABCD -0.73.
- On ABCD dev, memory adds +21.25 points to view B (bge-large: +19.84). On STAR it adds +13.3 (bge-large: +14.75).

**Test setup.**
- Settings as in R18 on STAR (16,000 clauses) and as in R9b on ABCD (4,000), with 5 seeds per TM arm.
- Open loop: the logged actions are the memory, as in R18 and R9b.
- Second use of STAR's test split, eleventh of ABCD's. A post-hoc LR refit (no rule, below) later read STAR's test split a third time.
- Commands are in `logs/jobs_R19.txt`.
- `scripts/r19_check.py` checked all 22 runs against the freeze (`logs/r19_check.out`):
  - code, settings and bits are as frozen;
  - every file was written after the freeze;
  - the seed-1 dev accuracies equal the dev rows above exactly.

| Arm | STAR | ABCD |
|---|---|---|
| TM-mem, word table (no network at run time) | **0.6271** (sd 0.0061) | **0.8301** (sd 0.0028) |
| TM-mem, bge-large (R18; R9b, 10 seeds) | 0.6265 (sd 0.0047) | 0.8401 (sd 0.0020) |
| TM-nomem, word table | 0.4766 (sd 0.0024) | 0.6222 (sd 0.0044) |
| TM-nomem, bge-large | 0.4722 (sd 0.0027) | 0.6537 (sd 0.0035) |
| LR-mem, word table | 0.6392 | 0.8063 |
| LR-mem, bge-large | 0.6591 | 0.8390 |

- **R19.1 (no network, little cost): met on both datasets, in the strong form.** The rule: at most 2.0 points below the bge-large agent on the point estimate. It is "strong" if the interval's lower end is also above -2.0.
  - STAR: **+0.06 [-0.63, +0.74]** points.
  - ABCD: **-1.00 [-1.62, -0.40]**.
- **R19.2 (the memory still pays): met on both.** The rule: at least +5.0 points, with the interval above 0.
  - STAR: **+15.05 [+14.07, +16.05]**.
  - ABCD: **+20.79 [+19.32, +22.29]**.
  - The reference pairs reproduce: R18's +15.43 [+14.38, +16.51] and R9b's +18.65 [+17.20, +20.10].
- **With the training seeds also resampled** (internal audit 9, post hoc): R19.1 STAR +0.06 [-0.94, +1.04], ABCD -1.00 [-1.72, -0.28]; R19.2 +15.05 [+13.89, +16.25] and +20.79 [+19.22, +22.39]. The strong form still holds. "Level" on STAR means within about 1 point.
- **R19: met.** With no neural network at run time, the agent is level with the bge-large agent on STAR (+0.06; the interval spans 0). On ABCD it is 1.0 point below, inside the registered 2-point margin (the interval lies wholly below 0). The memory literals pay as before.

**Reported with no rule:**
- **Without memory** (post hoc, `report_R19star_posthoc.json` and `report_R19abcd_posthoc.json`): STAR +0.44 [-0.23, +1.10], ABCD **-3.14 [-3.95, -2.31]**.
  - So the "almost free" result holds for the agent with memory.
  - On ABCD this is probably because the table reads the customer's task less well. A dev probe in the plan, on the table's mean vectors with nearest centres, showed -9.5 points; view B's bits were not probed.
  - With memory the loss is -1.00. At the first action, before any memory exists, it is -0.4 (0.908 against 0.912).
- **LR on the same bits:**
  - The swap costs LR -1.99 [-3.00, -0.96] on STAR and -3.27 [-4.34, -2.24] on ABCD.
  - TM minus LR on the table's bits: STAR -1.22 [-2.12, -0.31] (with bge-large: -3.26); ABCD +2.38 [+1.56, +3.22] (with bge-large: +0.12 [-0.58, +0.84]).
  - All four LR runs chose C = 0.1, the lowest value of the grid. For the two bge-large runs, lower values were checked on dev afterwards and gained little or nothing (R18: C = 0.03 is 0.1 point better and C = 0.01 2.0 worse; R9b: the peak is at 0.1).
  - **The word-table LR, checked after the runs** (e67, dev). The rules were fixed in the amendment before the low-C numbers:

    | C | 0.1 | 0.03 | 0.01 | 0.003 | 0.001 |
    |---|---|---|---|---|---|
    | STAR dev | 0.6409 | 0.6501 | **0.6555** | 0.6445 | 0.6209 |
    | ABCD dev | 0.8111 | **0.8122** | 0.8086 | 0.7910 | 0.7503 |

  - **STAR: refit at C = 0.01.** Post hoc, no rule, LR baseline refit at the dev-best C, after the registered runs; one test read, STAR test's 3rd use.
    - The tuned LR scores 0.6493 on test (0.6392 at C = 0.1).
    - **On the table's bits the TM is 2.22 [-3.02, -1.42] points behind a tuned LR on STAR.**
    - The swap costs the tuned LR -0.99 [-1.99, +0.06], not -1.99. bge-large's LR was not refit.
  - **ABCD: not refit.** Its dev-best C (0.03) gains only 0.11 points, and the rule asked for 0.5. ABCD's dev curve peaks at C 0.03, 0.11 above C 0.1, so its LR was close to tuned. The TM-LR difference (+2.38) and the swap cost (-3.27) carry no rule.
- **AST B-Slot accuracy (ABCD), the paper's own measure:**
  - In the paper's released code, B-Slot is the choice among the 30 buttons:
    - `action_to_id` (utils/process.py) drops the slot position from the label;
    - `ast_report` (utils/evaluate.py) scores the argmax over the buttons on every AST example.
  - The paper's Sec. 6.3 speaks of "231 known options", but 231 is its count of slots, and its code scores 30 buttons. The registration called this measure not comparable; that is corrected in the amendment.
  - `scripts/abcd_prepare.py` ports the paper's example filter (`collect_examples`, `value_to_id`). On the test split it keeps 4,562 examples with BERT's tokenizer and 4,560 with RoBERTa's, from the 3,608 action turns; 187 turns give none.
  - The word-table agent scores **84.78** (BERT's filter) / 84.82 (RoBERTa's). The bge-large agent scores 84.90 / 84.93; the difference is -0.11 [-0.76, +0.51].
  - Published (the paper's Table 3, which does not name the split):
    - a pipeline BERT, trained on this subtask alone: 86.7;
    - BERT 89.9, AlBERT 90.9 and RoBERTa 93.6, each trained jointly on all AST subtasks.
    - All four fine-tune a transformer on the full dialogue context, including the earlier actions.
  - **The word-table agent is 1.9 points below the pipeline BERT, 5.1 below BERT, 6.1 below AlBERT and 8.8 below RoBERTa.** It reads the customer's words, the last 3 turns and the earlier actions through a fixed table and frozen bits. No parity claim.
- **By position** (TM-mem, word table / bge-large):

  | Action | STAR | ABCD |
  |---|---|---|
  | 1st | 0.798 / 0.783 | 0.908 / 0.912 |
  | 2nd | 0.695 / 0.675 | 0.858 / 0.870 |
  | 3rd | 0.663 / 0.650 | 0.825 / 0.840 |
  | 4th | 0.615 / 0.610 | 0.784 / 0.787 |
  | 5th | 0.624 / 0.625 | 0.685 / 0.699 |
  | 6th and later | 0.588 / 0.595 | 0.425 / 0.461 |

- **Clause length (descriptive only):**
  - STAR: median 5 to 8 literals per clause across seeds, against 27 to 32 with bge-large. The 95th percentile is 30 to 34 against 117 to 123.
  - ABCD: the median is 3 against 1, and the 95th percentile 12 to 14 against 60 to 64.
- **Training cost:**
  - The view has 3 to 5 times the bits: STAR 7,938 against 2,715; ABCD 7,317 against 1,581.
  - Training took about 2 times as long: STAR 7,285 to 9,526 s against 3,368 to 4,343 s; ABCD 643 to 875 s against 454 to 481 s.
  - These times come from a shared GPU under other loads.
- **Speed of one decision** (rough: shared machine; `scripts/static_speed.py`, 1,000 STAR dev rows, one CPU thread, batch 1; `results/explore/static_speed_R19.json`):
  - The bge-large agent: median 806 ms per decision (p90 1,649 ms). Encoding the two texts takes 779 ms of that.
  - The word-table agent: median 29.5 ms (p90 29.9 ms), about 27 times faster. Both agents were run on one CPU thread; the bge-large agent would be faster on a GPU (not measured), which the word-table agent does not need.
  - Its text step (lookup + OR) takes 0.49 ms, about 1,600 times less than the bge-large encoding. That speed-up is Model2Vec's, not this study's.
  - The TM decision itself (16,000 clauses, numpy, `src/dl/cpu_tm.py`) takes 28.7 ms, 97% of the word-table agent's time, so it is now the bottleneck. On bge-large's 2,715 bits it takes 20.5 ms; view B has 7,938.
  - Checks before timing: the CPU decision gives both dev accuracies exactly (0.6394036 and 0.6343370). View B's run-time bits are a lookup and an OR, with no floating-point step, and are identical when computed again on the CPU (0 bits differ). bge-large's view differs in 44 bits in 42 of 6,908 rows, at float edges.

**Limits:**
- The table was distilled from a transformer, once and offline. Only the run time is free of neural networks.
- The speed-up of the text step is Model2Vec's, not this study's.
- B was chosen among three views on STAR dev. From dev to test (5-seed means) it drops 0.74 points, from 0.6345 to 0.6271. The bge-large agent drops more, from 0.6395 to 0.6265 (-1.30), so the selection does not show as an extra drop.
- Open loop on both datasets.
- No claim of state of the art or of parity with transformers. On the ABCD paper's B-Slot the agent is 1.9 to 8.8 points behind the paper's fine-tuned models (above).

## R20: the agent in a closed loop (STAR), and the agent that reads only the customer (ABCD)

Registration `R20_customer_only_closed_loop.md` (frozen 2026-10-18T23:23:30Z). Its dev rules are in `logs/plan_e65.md` (frozen 2026-10-18T20:23:31Z). The kinds of rule and the headline were fixed before e65 landed; the values were set after the dev runs e64 to e67, all of whose numbers are in the registration. Every gate was checked a second time, independently, and both reports were recomputed byte for byte.

**Outcome first: the question named as the headline was stopped at its dev gate.** R20.4 was not taken to STAR test. Arm 2 failed its dev gate S4 (+4.50 against +5.0, seed 1), so by the frozen plan it is a dev-only failure-map line. What went to test:
- arm 1 on STAR (R20.1 to R20.3, reported one by one);
- arm 2 on ABCD, as supporting evidence only.
- There is no combined "R20 met" line.

**The questions.** R19's agent decides with no neural network at run time, but open loop: its memory bits are the logged presses, and it reads text that the human agent (ABCD) or the wizard (STAR) wrote.
- **Arm 1 (STAR, R19's and R18's registered models):** in a closed loop, with its own presses as memory, does the memory still pay? What do its own mistakes cost? How does the word-table agent compare with R18's bge-large agent in the same loop?
- **Arm 2 (the customer-only agent):** can the agent decide from the customer's words and its own earlier presses alone, with no text that the human agent or the wizard wrote?

**The closed loop** (`scripts/abcd_closed_loop.py`, as in R13 and R17). In every conversation the action turns are decided in order. At turn k the memory bits come from the model's own choices at turns 0 to k-1. The text stays as logged, and accuracy is against the logged button.

**View C (arm 2): `mem, cwords, xcust, xcrecent`.** It is view B with every human-agent and wizard text removed:
- cwords: the 1,000 most frequent 1-2-grams of the customer's last 3 utterances;
- xcust: the word-table bits of the customer's words so far, as in R19;
- xcrecent: the same bits for the customer's last 3 utterances.
- `tests/test_customer_view.py` checks that replacing every row's text from both speakers leaves view C's bits unchanged.
- Limit, fixed before the runs: the claim is "no text written by the human agent", not "no information about the human agent's actions". The customer's logged replies answer what the human agent really did and said.

**Dev path** (not registered; seed 1 unless said):

| Dev run | STAR | ABCD |
|---|---|---|
| e66 / e64: R19's view-B TM (seeds 1 to 5), open / closed loop | 0.6345 / 0.5588 (-7.57) | 0.8318 / 0.8041 (-2.77) |
| e66: R18's bge-large TM (seeds 1 to 5), open / closed loop | 0.6395 / 0.5454 (-9.40) | |
| e65: view C, TM open / closed loop | 0.5779 / 0.4767 (-10.12) | 0.7872 / 0.7424 (-4.48) |
| e65: view C's twin without memory, TM | 0.4317 | 0.4984 |
| e65: view C, LR open (C 0.1) | 0.6019 | 0.7600 |
| R19's view B (seed 1): open / closed loop / no memory | 0.6343 / 0.5530 / 0.5010 | 0.8325 / 0.8046 / 0.6200 |

- **S4** (frozen: arm 2 goes to a dataset's test only if view C's closed-loop dev accuracy is at least 5.0 points above its twin's):
  - ABCD: +24.40, passed.
  - STAR: **+4.50, not passed.**
- **Plain reading of STAR's failure:** without the wizard's text, a wrong press poisons the memory. The memory is worth +14.6 points open loop (0.5779 against 0.4317) and +4.5 in the closed loop, and the agent's own mistakes cost 10.1 points.
- S4 was applied as written, with the best-dev snapshot. The twins' dev curves are noisy (STAR's 0.362 to 0.432 between snapshots, ABCD's 0.387 to 0.498), so the twin's best-dev number is optimistic; at the twin's typical level (about 0.42) STAR would give about +5.7. No extra seeds or epochs were run to recheck it.
- Dropping the human agent's or wizard's text costs, against view B on dev: ABCD 4.53 points open loop, 6.22 closed loop, 12.16 without memory; STAR 5.65, 7.63 and 6.93.
- LR's C grid: view B's LR sat at the registered grid's lowest C (e67). So arm 2's LR searched 0.001 to 100 (`--grid-c`, an additive change to `scripts/abcd_confirm.py`; with the flag unset, two R19 dry runs reproduce their stored outputs exactly, `logs/R20_default_path_check.out`). On ABCD it picked C 0.03.

**Test setup.**
- STAR (4th use of its test split): arm 1 only. No training: R19's word-table models (5 TM seeds, LR) and R18's bge-large models (5 TM seeds, LR) scored in the closed loop.
- ABCD (12th use): view C at R19's settings (4,000 clauses, 5 seeds with memory, 5 without, LR), each seed's closed loop right after training; R19's view-B models also scored in the closed loop (no rule).
- Commands: `logs/jobs_R20.txt` (16 jobs; all ran once, none crashed).
- `scripts/r20_check.py`, written and frozen before the runs, checked all 34 result files before any report was read (`logs/r20_check.out`: 285 checks, all passed). It covered code, settings and bits as frozen; every file written after the freeze; the seed-1 dev accuracies equal to the dev rows exactly; and every closed-loop seed's open-loop accuracy equal to its model's registered test accuracy (0 rows apart in all 23).

**Arm 1, STAR (test):**

| Arm (STAR, 5 seeds; LR one fit) | Open loop | Closed loop |
|---|---|---|
| TM-mem, word table (R19) | 0.6271 | **0.5519** (sd 0.0052) |
| TM-mem, bge-large (R18) | 0.6265 | 0.5323 (sd 0.0032) |
| TM-nomem, word table (R19; no memory bits, so open = closed) | 0.4766 | 0.4766 |
| LR-mem, word table (R19, C 0.1) | 0.6392 | 0.5894 |
| LR-mem, bge-large (R18) | 0.6591 | 0.5767 |

- **R20.1 (memory still pays in the closed loop): met.** The rule: the interval above 0. **+7.53 [+6.51, +8.58]** points.
- **R20.2 (its own mistakes cost at most 4 points; R13.2's exact form): not met, as expected.** **-7.51 [-8.34, -6.72]** points. Dev gave -7.57. The bar was kept at R13's value. A failure-map line.
- **R20.3 (the word table in the closed loop): met, strong.** The rule: at least -2.0 points against the bge-large agent in the same loop, "strong" if the interval's lower end is too. **+1.96 [+1.07, +2.85].**
  - In the closed loop the word-table agent is not worse than the bge-large agent. The interval lies above 0, but superiority was not a registered rule.
  - Descriptive: in the closed loop the word-table agent's own mistakes cost less, -7.51 against R18's -9.42 [-10.37, -8.49] (two no-rule pairs; their difference has no interval). Open loop the two agents were level (R19.1: +0.06).
- Reported with no rule:
  - TM minus LR in the closed loop: -3.75 [-4.93, -2.60]. R19's LR sat at C 0.1, under-tuned on STAR (e67), so a tuned LR would probably be further ahead.
  - LR's own mistakes cost -4.98 [-5.89, -4.07].
  - Closed loop by position (word-table TM): 0.798, 0.677, 0.615, 0.542, 0.526, 6th and later 0.493 (open loop: 0.798, 0.695, 0.663, 0.615, 0.624, 0.588).

**Arm 2, ABCD (test; supporting):**

| Arm (ABCD, 5 seeds; LR one fit) | Open loop | Closed loop | Closed loop, AST-weighted (B-Slot) |
|---|---|---|---|
| TM-mem, customer only (view C) | 0.7809 | **0.7263** (sd 0.0041) | 0.7600 |
| TM-nomem, customer only | 0.4869 | 0.4869 | 0.5699 |
| LR-mem, customer only (C 0.03) | 0.7586 | 0.7209 | 0.7534 |
| TM-mem, view B (R19; reads the human agent's text) | 0.8301 | 0.7961 | 0.8176 |
| TM-mem, bge-large with the human agent's text (R13's closed-loop arm, 10 seeds) | 0.8401 | 0.8070 | 0.8201 |

- **R20.4 (memory pays for the customer-only agent in the closed loop): met.** **+23.94 [+22.20, +25.67]** points.
- Reported with no rule:
  - Dropping the human agent's text costs -6.98 [-8.13, -5.84] in the closed loop, -4.92 [-5.81, -4.08] open loop and -13.54 [-14.65, -12.44] without memory.
  - What its own mistakes cost: the customer-only TM -5.46 [-6.26, -4.67]; its LR -3.77 [-4.66, -2.88]; view B's TM -3.40 [-4.02, -2.80].
  - Memory with view B in the closed loop: +17.38 [+15.84, +18.93].
  - TM minus LR (customer only): +0.54 [-0.77, +1.82] in the closed loop, +2.23 [+1.24, +3.21] open loop.
  - The full swap, no neural network and no human-agent text, against R13's bge-large agent with that text, both closed loop: -8.07 [-9.40, -6.79].
  - Closed loop by position (customer-only TM): 0.857, 0.756, 0.695, 0.628, 0.562, 6th and later 0.286 (without memory 0.248). As in R13, the memory helps least late in the conversation.

**Limits:**
- Offline closed loop: the text, including the customer's replies, is replayed from the log, so a customer who reacts to a wrong press is not simulated. A wrong press is also scored against the logged next button, which assumes the human's history, so the closed loop penalises an agent that stays consistent with its own presses.
- The customer's logged replies answer the human agent's real actions (the limit above).
- Arm 2 has no STAR test result (its dev gate failed).
- STAR test's 4th use and ABCD's 12th.
- No claim of state of the art or of parity with transformers.

## Line B (e70): proving that the agent follows its written guidelines (dev; frozen plan, not registered)

Plan `logs/plan_e70_verify.md` (frozen 2026-10-19T00:58:12Z, sha256 271eabf8177cb433, with every script hash); dated note `logs/plan_e70_verify_note_2026-10-19.md`. Dev and proof work: the proofs are about the policy as a function and read no test split. The novelty check is in the project notes (not published). TM verification by SAT exists (Przybysz et al. 2023: a vanilla TM, no weights, no argmax), and so does verify-then-repair with clauses (VERITAS, github.com/AnwarDebes/veritas, 2026). No earlier work was found that verifies a coalesced TM's weighted argmax for all inputs, or a learned next-action dialogue policy against properties from written guidelines.

**The question.** The goal names a policy that "can be revised and verified directly". Can a solver prove, for every input, that R19's ABCD agent follows rules taken from ABCD's written guidelines? Where it cannot, does it find the exceptions, and can the logic be patched so that the proof goes through?

**The rules** (`scripts/lineb_counts.py`): derived mechanically from guidelines.json and kb.json, kept where both agree, through the study's name mapping onto the 30 buttons (`abcd_inject.guideline_steps`, e40). That mapping was written by hand: a 5-entry rename table and one rule for the two FAQ flows (`scripts/abcd_workflow.py`, RENAME and FAQ_FLOWS).
- P_first: with an empty history the first button is log-out-in, notify-team, pull-up-account, search-faq or try-again (25 checks, one per other button);
- P_prec(A, B): B only after A, 25 pairs (pull-up-account before 13 buttons; enter-details before make-password; verify-identity before subscription-status; log-out-in before instructions; search-faq before the 8 topic searches and select-faq).
- The human agents break them on dev: the first action in 53 of 1,004 conversations, a precedence rule in 113 of 3,684 action turns.

**The method.** An exact MILP encoding of the TM (`scripts/verify_policy.py`; scipy 1.17.1, HiGHS 1.12.0, tolerances 1e-6 / 1e-7):
- each kept clause is c_j in [0, 1] with the tight AND rows;
- the integer votes;
- "B wins" with the lowest-index tie rule;
- the domain: every bit vector that satisfies the constraints that real inputs satisfy (memory consistency, thermometer chains, a 1-2-gram implies its words). That is a superset of all real inputs.
- Pre-checks before any property run: the encoding's votes equal the CPU decision's on 500 of 500 dev rows (5 seeds); the winner rows are right on 50 of 50 fixed rows per seed; every train, dev and closed-loop dev state lies in the domain.

**Results** (TM, R19's 5 registered ABCD models; 50 checks each, 600 s per MILP):
- **Over the superset, no rule holds for the raw TM: 250 of 250 checks give a counterexample.** Each takes 0.5 to 6 s (1 node), and every one replays through the real decision code.
  - The counterexamples look nothing like real conversations. Their nearest dev row is 1,458 to 4,596 text bits away (median about 1,504), and none equals a real row. A median of 32 lexicon words and word pairs is on (12 to 63).
  - This is not a bug. A free text space lets the solver build a text that votes for any button. The superset is too loose for a proof about the raw policy.
- **The repair:** 50 veto clauses (one per rule). Each carries the rule's guard literal (not done(A), or no action yet) and a negative weight large enough that the vetoed button can never beat pull-up-account, which no rule vetoes.
  - The guarded policy picks the highest-voted button among those the rules allow in the current state.
  - It differs from the TM exactly where the TM breaks a rule: lowering a class that is not the winner never changes the winner.
  - All 50 checks then hold for all 5 seeds. That holds by construction, and the 250 MILP runs (about 0.1 s each) check the guards' encoding.
  - Of the 50 clauses, 15 to 20 change a decision on dev (17, 15, 19, 15, 20 for seeds 1 to 5).
- **The guards' cost on dev:**
  - open loop, -0.27 to -0.43 points (49 to 59 decisions changed of 3,684);
  - closed loop, -0.08 to -0.27 (30 to 49 changed).
  - Descriptive split: open loop, every decision the guards turn from right to wrong (20 to 23 per seed) is a turn where the logged human agent itself broke a rule. This holds by construction, since it is the same state and button. The guards also fix 7 to 10 TM errors per seed.
  - In the closed loop only 5 to 8 of the 12 to 18 right-to-wrong turns are human rule breaks. The rest follow the agent's own earlier mistakes: its history lacks the prerequisite, so the guard blocks a button the human could press.
- **The neural comparison** (MLP twins on the same bits, `scripts/verify_mlp.py`, written after the freeze as a recorded deviation; dev accuracy 0.790 / 0.812 / 0.820 for 64 / 128 / 512 hidden units, against the TM's 0.833; their encoding pre-check passed):
  - **The results** are on the same 50 checks, with the same generic MILP encoding and solver, at 600 s per check; one process, 2 threads.
    - **The TM (5 seeds):** 250 of 250 resolved, all counterexamples, with a median of 1.3 s (332 s in all).
    - **MLP-64:** 41 of 50 resolved, all counterexamples that replay through the network. The median is 151 s (76 to 494 s), and 9 of 50 are unknown at 600 s. That is 3.6 h in all.
    - **MLP-128:** 1 of 50 resolved (a counterexample in 25 s), and 49 of 50 are unknown at 600 s. That is 8.2 h in all.
    - **MLP-512** was not run (budget, decided at 03:03 UTC before it ran). The smallest network is the one most likely to be verifiable, so leaving out the largest is fair to the MLP.
  - The MLP was not given a dedicated neural-network verifier (Reluplex/Marabou, alpha-beta-CROWN, VeriNet), which might close part of the gap. Never write "MLPs cannot be verified" or "TMs are N times faster to verify": these are one generic encoding, one solver and one time limit.

**Reading.** For these 50 guideline rules, over a superset of all real inputs, the raw TM policy is not proved to comply: every rule has a counterexample, though none of them looks like a real conversation. With 50 veto clauses (one per rule), of which 15 to 20 change a decision on dev, all 50 hold by construction. The cost on dev is a third of a point open loop, and there the loss falls only where the agent disagrees with humans who broke the guidelines. The local question, how many word edits around a real conversation it takes to break a rule, is e71's (plan `logs/plan_e71_local.md`).

**Newest work** (searched 2026-10-19 21:59 UTC to 2026-10-20 00:28 UTC). This tops up the novelty check of
2026-10-18 (project notes, not published). The report and every query are in
the project notes (not published). Agent-C, Truth Table Net and SMiLE
were also checked on their own pages.
- **The verdict stands.** No earlier work was found that verifies a coalesced TM's weighted argmax for all inputs, or
  proves guideline properties of a learned next-action dialogue policy before deployment.
- **Run-time enforcement already reaches full conformance for LLM agents.**
  - Agent-C (Kamath et al., arXiv 2512.23738, Dec 2025; an ICLR 2026 workshop poster) enforces temporal rules during
    generation on retail and airline tasks.
  - Conformance rose from 77.4% to 100% and from 83.7% to 100% for the paper's two commercial LLMs, and utility rose too.
  - So "100% compliance" is not this line's point. Its point is a proof about the learned decision function, made before
    it runs, over a superset of all inputs. The guards make it hold by construction.
- **Run-time guards on tau2-bench are a busy 2026 field,** and every one works while the agent runs:
  - PolicyGuard (arXiv 2606.29225);
  - PolicyGuide (arXiv 2608.19861; mean Pass^4 from 0.42 to 0.62 with GPT-5.4);
  - STAGE (arXiv 2608.22538);
  - ToolGuard (EMNLP 2025 industry track).
  - Rabinovich et al. 2026 (arXiv 2603.29665) find that in 8-17% of airline runs that end in the right state, the
    agent skipped a required policy check.
- **Not new, so not claimed:**
  - "Logic models are easier to verify": Truth Table Net (Benamira et al., IJCAI 2024, older than the first novelty check but
    missed by it) reports robustness verification in 10 ms against 100 s for DNN solvers.
  - "Properties by construction": SMiLE (Francobaldi, Lombardi, Lodi, AAAI 2026) enforces global relational properties
    in neural networks with full satisfaction guarantees.
  - This study's part is the same-inputs, same-solver measurement and the dialogue-policy setting.
- **The neural verification baseline:** VNN-COMP 2026 was won again by alpha-beta-CROWN (search report). The MLP
  comparison above used a generic MILP (HiGHS), not alpha-beta-CROWN. That limit stands as written.

## e71: word edits around real conversations (dev; frozen plan with a v2 amendment; R19's ABCD agent)

**Plans and models:**
- Plan `logs/plan_e71_local.md`, frozen 2026-10-19T02:10:00Z.
- v2 amendment `logs/plan_e71_v2_amendment.md`, frozen 03:30:28Z: exhaustive search for k = 1 and 2, because the
  MILP for the minimal edit count timed out at 120 s.
- Dated notes: `logs/plan_e71_v2_note_2026-10-19.md` (the crash fix below) and `logs/plan_e71_v2_note2_2026-10-19.md`
  (frozen 10:19:18Z: the k = 3/5 sample, then seeds 2 and 3).
- Dev and proof work only: no test split is read.
- The models are R19's ABCD agent (`results/confirm/R19abcd/tm_mem_static`, potion's word table, no neural network at
  run time), seeds 1 to 3.

**The question.** e70's superset is too loose to prove anything about the raw policy. Locally, around real
conversations, how many word edits of the customer's latest message does it take to make the agent press a button the
rules forbid in that state?

**The method:**
- **States:** 200 sampled dev states (rng 0), each with its rule groups: 579 (state, group) pairs.
- **Edits of the customer's latest utterance u:**
  - delete one or two of u's words, with u keeping a word;
  - append one or two of 1,626 candidate words (words in at least 5 train customer utterances);
  - or one of each.
  - That is about 2.65M edit sets per state at k = 2.
- **Exhaustive and exact:**
  - deletions are scored through cpu_tm_fast's clause firing;
  - insertions go through the TM partially evaluated on the base row;
  - both were checked exact in the pre-checks.
- **Replays:** up to 5 breaking edit sets per group go through the real pipeline (strings, view.transform, the CPU
  decision).
- **k = 3 and 5:** a decision-version MILP on a fixed random sample of 50 pairs (seed 1 only; numpy default_rng(0);
  120 s each), best effort.

**Results** (`logs/report_e71v2_seed{1,2,3}.out`):

| Seed | States certified at k = 1 | States certified at k = 2 | Pairs certified at k = 2 (of 579) | States where the agent's own choice breaks a rule | Replays reproduced (k = 1, k = 2) |
|---|---|---|---|---|---|
| 1 | 186 (93.0%) | 151 (75.5%) | 512 | 4 | 34/34, 240/240 |
| 2 | 185 (92.5%) | 152 (76.0%) | 519 | 3 | 31/31, 185/185 |
| 3 | 188 (94.0%) | 155 (77.5%) | 525 | 2 | 30/30, 163/163 |

- **Range over seeds 1 to 3:** 92.5-94.0% at k = 1 and 75.5-77.5% at k = 2. Seeds 4 and 5 were not run (budget).
- **The weak spot is the pull-up-account precedence group.** Per seed, of its 34 pairs:
  - 3 to 5 break with one word;
  - 11 more break with two;
  - the agent's own choice breaks the rule in 2.
  - For comparison, search-faq's 111 pairs have 4 to 6 one-word breaks, and P_first's 58 pairs have 0 to 3.
- **k = 3 and 5** (seed 1, 50 sampled pairs certified at k = 2; `logs/e71_k35_sample.json`):
  - at k = 3, 4 were certified by MILP and 46 were unknown at 120 s;
  - at k = 5, those 4 were unknown.
  - Beyond two edits the solver mostly cannot decide, so no number is claimed there.
- **Off-window states:** 3 of the 200 sampled states have their latest customer utterance outside the last 3 turns.
  They are reported apart, and all 3 are certified at k = 2 for every seed.
- **Deviation:**
  - The run crashed at 07:31:54 UTC on the first off-window state (KeyError 'ja').
  - The fix changed `s["ja"]` to `s.get("ja")` in two lines (verify_enum.py 94e9b1e2 to 99ba6582), with a dated note.
    The off-window pre-check was exact.
  - The run resumed at 07:35:42, and every later state ran under 99ba6582.
  - An earlier note said "68 off-window". That was the count over all 3,684 dev states; the 200 sampled hold 3.

**Wording** (the plan's): "For 92.5-94.0% of 200 sampled real dev states, no deletion of words from the customer's
latest message, and no insertion of one of 1,626 common customer words at its end, makes the agent with no neural
network at run time choose a button the rules forbid in that state. For 75.5-77.5%, no edit of up to 2 words does.
This is by exhaustive search, seeds 1 to 3."
- Never write "proved for all possible texts" or "any rewording".
- Never write "no neural network at any stage" for this agent (potion).

**Reading.** The raw policy is locally robust to one-word edits in nearly every sampled state, and to two-word edits in
about three out of four. The breaks concentrate on one rule group, and every break replays through the real code.
e70's veto clauses make all 50 rules hold for every input, so the guarded agent has no such breaks. e76 repeats e71 for
the agent with no neural network at any stage (R23's tuned ABCD model). That needs one change of method: the C4
tokenizer's contraction rule makes e71's OR decomposition inexact (`logs/plan_e76_wordedit_c4.md`).

**Newest work** (searched 2026-10-20 at about 00:30 UTC: two web searches, with AdaptDel's page opened):
- Certified robustness of text classifiers to edits is an active line, built on randomized smoothing:
  - CERT-ED (Findings of EMNLP 2024, arXiv 2408.00728) certifies under edit distance;
  - AdaptDel (Huang, Marchant, Ohrimenko, Rubinstein, NeurIPS 2025, arXiv 2511.09316) uses adaptable deletion
    rates;
  - CluCERT (AAAI 2026, arXiv 2512.08967) certifies LLMs under word substitutions;
  - Text-CRS (2023, older) covers substitution, reordering, insertion and deletion.
- These give probabilistic certificates for a smoothed classifier, over every edit within a radius. e71's are a
  different kind: exact, by exhaustive search, for the unsmoothed policy itself, for its own rule-breaking property.
  They cover a fixed list of 1,626 inserted words, deletions, and at most 2 edits.
- So no claim is made against those methods. They answer a wider question with a weaker kind of guarantee.

## e76: e71's word-edit search on the agent with no neural network at any stage (dev; frozen plan; R23's tuned ABCD model)

**Plan and model:**
- Plan `logs/plan_e76_wordedit_c4.md` (frozen 2026-10-19T21:14:22Z, after its pre-checks).
- Dated note `logs/plan_e76_note_2026-10-20.md`: the budget was extended so that all 200 states finish. The reason was
  the run's speed, 167 s per state against the planned 90-120 s. The outcomes of the first 6 states had been seen in
  passing and played no part.
- The model is R23's ABCD agent on the C4 table at s 16 (`results/confirm/R23abcd/tm_mem_c4_s16`), seed 1.
- e71's states, rule groups, 1,626 candidate words, edit sets (k = 1 and 2), scoring and replays are reused unchanged.
- The run went from 00:07:33 to 07:43:18 UTC on 2026-10-20 and exited 0. It is `results/explore/verify_e76.jsonl`,
  reported in `logs/report_e76_seed1.out`.

**The one change of method (the contraction finding):**
- e71 rebuilds the word-table bits after deletions as the text's codes OR the kept words' codes.
- That is exact for potion's WordPiece. It is not exact for the C4 tokenizer, whose contraction rule spans e71's word
  boundaries: "don't" is the C4 tokens "do" and "n't".
- Measured on all 400 dev states with a contraction: e71's decomposition with C4 tokens is wrong on 400 of 400
  zero-edit rows and on 3,993 of 4,361 single deletions.
- So the port rebuilds xcust and xrecent from the real edited strings. Every pre-check was then exact (33 states,
  53,854 single edits, 34,654 deletion pairs on the contraction states, 243,900 reduced-scoring pairs).

**Results** (recounted independently from the jsonl with `../verification/recompute_e71.py`; the counts are equal):

| Agent, model | States certified at k = 1 | at k = 2 | Pairs certified at k = 2 (of 579) | States where its own choice breaks a rule | Replays reproduced |
|---|---|---|---|---|---|
| no neural network at any stage (C4 table, s 16), seed 1 | 183 (91.5%) | 153 (76.5%) | 518 | 6 | 33/33 (k = 1), 186/186 (k = 2) |
| potion agent (e71, s 8), seeds 1 to 3 | 185-188 (92.5-94.0%) | 151-155 (75.5-77.5%) | 512-525 | 2-4 | all |

- The two agents are close. That is one C4 seed at s 16 against three potion seeds at s 8, so it is not a comparison
  of the tables as such.
- **The weak spot is the pull-up-account precedence group again.** Of its 34 pairs, 8 break with one word, 9 more
  with two, and the agent's own choice breaks the rule in 4.
- **Elsewhere:**
  - search-faq: 1 one-word break in 111 pairs;
  - P_first: 2 in 58;
  - enter-details, log-out-in and verify-identity: none at k = 1.
- The 3 off-window states are certified at k = 2.

**Wording** (the plan's): "For 91.5% of 200 sampled real dev states, no deletion of words from the customer's latest
message, and no insertion of one of 1,626 common customer words at its end, makes the agent with no neural network at
any stage choose a button the rules forbid in that state. For 76.5%, no edit of up to 2 words does. This is by
exhaustive search, one model."
- Never write "proved for all possible texts" or "any rewording".
- With e70's veto clauses (e72e: 250 of 250 proved on R23's models), every rule holds for every input.

## e72: the agent on a word table made with no neural network at any stage (dev; frozen plan; test to be registered as R21)

Plan `logs/plan_e72_count_table.md` (frozen 2026-10-19T05:05:14Z, sha256 1f800fad11299f26); the gate recomputed from the rows in `logs/e72_gate.out`. Dev only: no test split was read.

**Outcome: both datasets pass the dev gate.** With the C4 word table in place of potion's, the agent loses 0.08 points on ABCD dev and 0.26 on STAR dev. The gate allowed 2.0.

**The question.** R19's agent has no neural network at run time, but its word table (potion-base-32M) was distilled from a transformer. The C4 word table (registered as W1-T in the W1 study) was made with no neural network at any stage. Does the agent keep its accuracy on that table?
- How the table was made: C4 text, a C4 word list (63,091 words), counting + SVD (500 dimensions). The corpus was deduped against ABCD and STAR dev/test turns.
- The agent: k-means + thresholds on the train texts (existsT), then a Tsetlin machine.

**What changed.** Only the table and the tokenizer inside xcust and xrecent:
- the tokenizer is the table's own word rule, and a word not in the list is dropped. That is 0.46% to 0.79% of the tokens on ABCD (names, typos, brands) and 0.70% to 0.87% on STAR (place names, long numbers, quoted words);
- everything else is R19's: the existsT recipe (K 512, CPU k-means seeded 0), the memory and 1-2-gram bits, the TM settings, seed 1.

**How it was run without touching frozen code.** All code is in new files: `src/dl/count_text.py` and `scripts/table_run.py`. The run scripts are unedited (their hashes sit in the e70 and e71 freezes).
- In such a run potion's table cannot load, and no encoder vector is loaded (`--encoder none`).
- Pre-checks (`tests/test_count_text.py`, `logs/e72_precheck.out`), all passed:
  - given potion's own table, the new code reproduces R19's bits exactly;
  - the tokenizer port equals the table's reference tokenizer on 284,085 texts;
  - the lookup + OR equals the max-cosine reference on every train and dev text;
  - sentence_transformers, model2vec and transformers are not imported.

**Results** (dev, seed 1; `results/explore/abcd.jsonl`):

| Dataset | C4 table | potion (R19's seed-1 model) | Difference | Gate (potion - 2.0) | bge-large dev |
|---|---|---|---|---|---|
| ABCD (4,000 clauses) | 0.8317 | 0.8325 | -0.08 | passed (0.8125) | 0.8398 (-0.81) |
| STAR (16,000 clauses) | 0.6317 | 0.6343 | -0.26 | passed (0.6143) | 0.6394 (-0.77) |

**No rule:**
- **Without memory** (ABCD twin, `words,xcust,xrecent`): 0.6107 against potion's 0.6200, -0.92. The table costs about a point when the text carries everything; memory hides it. On the C4 table the memory adds +22.1 points on dev.
- **Fewer bits:** 6,684 on ABCD (potion 7,317) and 7,227 on STAR (7,938), because fewer thresholds stay distinct.
- **STAR's first action:** 0.7599 against potion's 0.7748, -1.49. Later actions: 0.6194 against 0.6209.
- **Training on the free GPU:** 158 s (ABCD) and 1,404 s (STAR). Potion's rows took 770 s and 7,943 s, but on a shared GPU, so the times are not comparable.
- STAR's dev curve swings 1 to 2 points between epochs, for both tables, so the -0.26 is within one model's noise.

**Reading.** On these two workflows, the transformer-distilled table can be replaced by one counted from raw web text with almost no loss on dev. Two things can explain it, and both apply: the memory and 1-2-gram bits carry most of the decision, and the table's cost shows mostly without memory (-0.92). The test (R21) will say whether this holds with 5 seeds on the test splits.

**Wording.** If R21 is met, the line is "no neural network at any stage: C4 text, a C4 word list, counting + SVD, k-means + thresholds, then a Tsetlin machine". It is never used for the potion results (R19, R20, e70, e71).

## R21: the agent with no neural network at any stage (STAR and ABCD; test)

Registration `registrations/R21_c4_table_agent.md` (frozen 2026-10-19T05:48:24Z, sha256 bc4a2d1e3d751726); dev rules `logs/plan_e72_count_table.md` (frozen 05:05:14Z). Fifth use of STAR's test split, thirteenth of ABCD's. The gate was verified independently: reruns of the frozen report commands are byte-identical, and an independent bootstrap (`../verification/recompute_r21.py`) agrees within 0.03.

**Outcome: met on both datasets, R21.1 in the strong form on both.** An agent with no neural network at any stage (C4 text, a C4 word list, counting + SVD, k-means + thresholds, a Tsetlin machine) is within 0.10 points (STAR) and 0.29 points (ABCD) of the potion agent on test. On ABCD, with 50 veto clauses, it follows its 50 guideline-derived order rules for every input of the superset (e72d, below). STAR has no written guidelines here.

| | STAR | ABCD |
|---|---|---|
| C4 agent, tm_mem_c4 (5 seeds) | 62.6% | 82.7% |
| potion agent, R19's tm_mem_static (5 seeds) | 62.7% | 83.0% |
| **R21.1** C4 minus potion (at least -2.0) | **-0.10 [-0.65, +0.45]**, met, strong | **-0.29 [-0.74, +0.17]**, met, strong |
| **R21.2** memory (at least +5.0) | **+14.73 [+13.73, +15.74]**, met | **+20.63 [+19.17, +22.11]**, met |

**Beside the headline (no rule):**
- **Against the bge-large agent** (bge-large-en-v1.5, 335M parameters, run per decision): STAR -0.04 [-0.70, +0.60], ABCD -1.29 [-1.94, -0.66].
- **The TM against LR on the same bits** (LR's C from the wide dev grid):
  - STAR: LR is 2.79 points ahead (-2.79 [-3.61, -1.98]; LR 65.4%);
  - ABCD: the TM is 1.76 points ahead (+1.76 [+0.90, +2.64]; LR 81.0%).
- **Without memory**, the C4 table against potion's: STAR +0.22 [-0.27, +0.71], ABCD -0.13 [-0.69, +0.42].
- **LR on C4 bits against R19's LR on potion bits:** +1.47 [+0.48, +2.45] (STAR) and +0.33 [-0.74, +1.40] (ABCD). The grid differs, because R19's LR was C 0.1, under-tuned on STAR.
- **ABCD's AST-weighted accuracy** (30 buttons; not the paper's 231-way B-Slot): 84.4% against potion's 84.8%.

**The 50 guideline-derived order rules** (e72d, dev; note `logs/plan_e72d_guards_c4.md`, frozen 11:19:16Z). This is e70's method on the five ABCD tm_mem_c4 models, with the encoding pre-check passed for all 5.
- **Raw:** 250 of 250 checks give a counterexample. Each replays through the real decision code, at most 2.1 s each. As with potion, the counterexamples are far from real conversations: the nearest dev row is 1,422 to 3,070 text bits away (median 1,538).
- **With the 50 veto clauses** (M recomputed per model): 250 of 250 proved, about 0.07 s each. So the guarded agent follows all 50 rules for every input of the level-B superset of real inputs.
- **Cost on dev:**
  - open loop, -0.30 to -0.49 points. Every right-to-wrong turn is one where the logged human pressed a vetoed button.
  - closed loop, -0.14 to -0.27 points.

**Speed** (rough: shared machine, one CPU thread, batch 1, 1,000 dev rows per dataset; `results/explore/decision_speed_c4.json`):
- the bits take 0.58 ms (STAR) and 0.30 ms (ABCD);
- the exact CPU decision (cpu_tm_fast) takes 2.00 ms and 0.19 ms;
- so one decision takes 2.6 ms and 0.5 ms.
- The numpy decision of R19's speed test takes 28.9 ms and 1.7 ms on the same rows. R19's own measurements are not carried over, because the tokenizer differs.

**One decision, as it reads.** A demo file of the C4 agent, seed 1, on dev (project notes, not published) holds 4 ABCD dev dialogues drawn with default_rng(0), with 10 of 10 turns right and no guard changing a choice. In conversation 5677, at its first action, the customer has just said "sanya afzal" after the agent asked for the full name.
- **The choice:** pull-up-account (votes 7,316). The runner-up is log-out-in (-3,856), so the margin is 11,172.
- **The strongest firing clause (+471):** it combines the memory literals NOT done(verify-identity) and NOT done(pull-up-account) with concept literals of the last 3 turns.
- **Not every literal carries meaning:** most of these concept literals sit at low levels (0 to 3) that nearly every text passes. They include "close to concept 91 (smith|williams|jones|johnson) at level 0".
- **The guards:** with an empty history the 50 veto clauses fire, and the guarded choice is the same.

**How it was checked** (`scripts/r21_check.py`):
- Run 1 (11:14:29 UTC) failed only on its own list of allowed changes. That list lacked a later e71 script that is off R21's path.
- The list was extended (11:14:39), and run 2 passed all 375 checks (11:14:41).
- The reports came after, at 11:14:55 and 11:14:57.

**Wording.**
- **The line:** "A dialogue agent with no neural network at any stage (C4 text, a C4 word list, counting + SVD, k-means + thresholds, a Tsetlin machine), within 0.10 (STAR) and 0.29 (ABCD) points of the potion agent on test; on ABCD, made to follow its 50 guideline-derived order rules by veto clauses."
- **Beside it:**
  - level with bge-large on STAR (-0.04) and 1.29 below it on ABCD;
  - on the same bits, LR is 2.79 ahead on STAR, and the TM is 1.76 ahead on ABCD.
- **Never write:**
  - "first network-free TM text pipeline" (Kadhim et al. 2025);
  - "state of the art";
  - "parity with published TMs";
  - "the guidelines are verified";
  - "proved for all possible texts".
- e71's word-edit proofs are about the potion agent.

**Limits.**
- Open loop only; R20's closed loop was not repeated for this agent.
- The dialogues are replayed from the logs.
- The rules come from guidelines.json and kb.json through a hand-written name mapping.
- View B reads the last 3 turns, including the human agent's own words, and these often announce the next action. In the
  demo's conversation 3230 the agent says "try logging out of your account and then back in" before log-out-in.
  - R20's customer-only view cost about 4.5 points open loop on ABCD dev (e65: 0.7872 against 0.8325).
  - The comparisons with potion and bge-large use the same view, so they are fair. The absolute accuracies depend on it.

## e73: the STAR agent produces its replies from the task's templates (dev; frozen note, no rule)

Note `logs/plan_e73_replies_star.md` (frozen 2026-10-19T07:49:24Z, sha256 4f7582f359def175). Dev only: no test split was read. A measurement with no pass rule.

**Outcome: on STAR dev, the agent's reply equals the human wizard's word for word in 45.5% of the template turns (40.0% in the closed loop). Filling the logged template the same way gives 79.6%, which is the ceiling.**

**What it is.** STAR's wizards mostly reply by picking one of the task's reply templates, and the study's 187 STAR labels are those picks plus the API queries.
- The agent is R21's STAR models (view B on the C4-word table, so no neural network at any stage; the CPU decision). It picks a label.
- For a template label, its reply is that template's text, with the slots filled from the latest database result of the template's task. The slot-to-field mapping was derived from train beforehand.
- The reply is compared, after strip(), with what the wizard really sent.
- It does not compose sentences: it "produces its replies from the task's templates".

**The mapping, fixed before any dev score.**
- The rule first given ("the field whose value appears most often in the text") cannot tell slots apart: every slot of a template got the same field, and it filled 0.406 of the slotted train picks exactly. That rule was wrong and was replaced.
- The positional rule takes each slot's own place in the logged text and fills 0.749 (`results/explore/star_reply_mapping_positional.json`, 4dd5e768). Both files are kept.
- 11 of 36 slotted templates stay below 0.9 train coverage. They were listed before scoring: trip_directions' step lists, doctor_schedule's "o'clock" mismatch, and unmapped or boolean slots.

**Results** (dev; 6,908 labelled wizard actions; 4,676 template picks, 1,447 of them slotted; mean over R21's 5 seeds, range in brackets; `results/explore/star_replies.jsonl`):

| | Open loop | Closed loop | Ceiling (the logged label) |
|---|---|---|---|
| Exact replies, share of template picks | **45.5%** (44.3-46.7) | **40.0%** (39.1-41.0) | 79.6% |
| plain templates / slotted templates | 43.3% / 50.3% | 41.7% / 36.1% | 81.6% / 74.9% |
| Exact replies, share of all wizard replies (690 free-text replies, 12.9%, included) | 39.6% | 34.8% | 69.3% |
| Agent over ceiling | 0.572 | 0.502 | 1 |
| Label accuracy: template picks / API queries | 56.6% / 77.4% | 50.4% / 67.2% | |

**Reading.**
- When the agent picks the right template, its reply nearly always equals the one the wizard sent. The exact-match share over the ceiling (0.57) is about the label accuracy on picks (0.57). So the loss is in choosing the template, not in filling it.
- 12.9% of the wizard's replies are free text, which a template agent cannot match.
- Of the template turns, 20.4% cannot be matched even with the right label, because the wizard edited the suggestion or a slot cannot be filled.

**Limits.**
- Dev only, one dataset, replayed dialogues. A wrong reply is not shown to a customer, so the customer's next words stay as logged.
- The agent reads the wizard's earlier words (view B), as in R21.
- "Produces its replies from the task's templates", never "generates language".

## e74, e74c and e74d: the TM's s on STAR dev (frozen plans; dev only; test registered as R22)

Plans `logs/plan_e74_s_sweep.md` (frozen 2026-10-19T11:46:21Z) and `logs/plan_e74cd.md` (frozen 15:20:48Z). Dev only: no test split was read. The N8 study had shown that s moves TM accuracy a lot on sparse bits, in either direction; every agent run up to R21 used s 8.

**Outcome: s 16 is the best of the values tried for both agents.** The C4 agent at s 16 gains on dev over s 8:
- STAR (16,000 clauses): +2.28 points, 0.6545 against 0.6317;
- ABCD (4,000 clauses): +1.00, 0.8417 against 0.8317.

On STAR dev it is 0.13 points from a tuned LR on the same bits, where s 8 was 2.41 points behind. Tuned the same way, it is 0.38 points above the bge-large agent on STAR dev.

**The screens** (STAR dev, 4,000 clauses, seed 1):

| s | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|
| C4 agent (view B) | 0.5138 | 0.5450 | 0.5662 | 0.5860 | 0.6025 | 0.6306 | **0.6349** | 0.6310 |
| bge-large agent (R18's view) | | | | | 0.6131 | 0.6372 | **0.6401** | 0.6375 |

**At 16,000 clauses** (STAR dev):

| | Dev |
|---|---|
| C4 agent, s 8 | 0.6317 |
| C4 agent, s 12 | 0.6519 |
| C4 agent, s 16 | 0.6545 |
| bge-large agent, s 8 (e46) | 0.6394 |
| bge-large agent, s 16 | 0.6507 |
| LR on the C4 bits (wide grid) | 0.6558 |

**The picks** followed the frozen rules: at 4k, a new s needs +0.3 points over the reference. For the C4 agent, the 16k STAR and 4k ABCD checks then decided, and s* = 16 passed both.

**Reading.**
- Both agents peak at s 16, and the low values (2 to 4) cost 4 to 9 points.
- A larger s pushes clauses to include more literals, so they are more specific. Here that helped both agents; the cause was not tested further. Training is also shorter at high s.
- The earlier +1.25 against bge-large on dev compared a tuned agent with an untuned one. Under equal tuning the gap is +0.38.

**Limits.**
- s was chosen on STAR dev, among 8 values for the C4 agent and 4 for bge-large, by the same rule; dev numbers are optimistic.
- Seed 1 only.
- R22 (frozen 16:31:24Z; STAR's 6th test use) tests three things:
  - the gain over s 8 (R22.1);
  - the TM against LR (R22.2);
  - the C4 agent against the bge-large agent under equal tuning (R22.3).

## R22: the TM's s, tuned on dev, for the agent with no neural network at any stage (STAR test)

Registration `registrations/R22_tuned_s_star.md` (frozen 2026-10-19T16:31:24Z, sha256 de7794c5407d3b29); dev rules `logs/plan_e74_s_sweep.md` and `logs/plan_e74cd.md`. Sixth use of STAR's test split; ABCD's test split was not read. The gate checker (`scripts/r22_check.py`) passed all 122 checks on its first run.

**Outcome: all three rules met, R22.2 and R22.3 in the strong form.** It was verified by an independent recompute: a rerun of the report is byte-identical, and an independent bootstrap (`../verification/recompute_pairs.py`) agrees within 0.02.

**Headline:** "With the TM's s tuned on STAR dev for both agents (s 16), the agent with no neural network at any stage scores 65.3% on STAR test, against 63.7% for the equally tuned bge-large agent: +1.64 [+0.94, +2.35]. The rule was non-inferiority (at least -2.0); the interval lies above 0, but superiority was not registered. It is level with a tuned logistic regression on the same bits (-0.09 [-0.90, +0.73]), and tuning added +2.70."

**Beside it:**
- this is STAR test's 6th use;
- s was chosen on STAR dev among 8 values for C4 and 4 for bge-large, and both peaked at 16;
- the tuned dev gap was +0.38; on test it is +1.64;
- ABCD at s 16 is dev only, and only for the C4 agent (+1.00);
- saying "better than" would need a fresh registered superiority test.

| Arm (STAR test, 5 seeds unless noted) | Accuracy |
|---|---|
| C4 agent, s 16 (`tm_mem_c4_s16`) | **65.3%** (sd 0.37) |
| C4 agent, s 8 (R21) | 62.6% |
| LR on the same bits (R21, wide grid; 1 seed) | 65.4% |
| bge-large agent, s 16 (`tm_mem_bge_s16`) | 63.7% (sd 0.45) |
| bge-large agent, s 8 (R18) | 62.7% |

| Rule | Difference [95% interval] | Verdict |
|---|---|---|
| **R22.1** C4 s 16 minus C4 s 8 (at least +1.0, interval above 0) | **+2.70 [+2.17, +3.24]** | met |
| **R22.2** C4 s 16 minus LR (at least -1.0) | **-0.09 [-0.90, +0.73]** | met, strong |
| **R22.3** C4 s 16 minus bge-large s 16, equal tuning (at least -2.0) | **+1.64 [+0.94, +2.35]** | met, strong |

**No rule:**
- bge-large's own gain from the tuned s: +1.02 [+0.56, +1.49].
- Unequal tuning (C4 s 16 minus bge-large s 8): +2.66 [+1.97, +3.35].

**Reading.**
- On STAR, the TM's gap to LR came from s, not from the bits. At s 16 the TM reads the same bits as well as a tuned LR does.
- With both agents tuned the same way, the agent whose word table was counted from C4 text is not worse than the agent that runs bge-large-en-v1.5 (335M parameters) per decision.
- On dev the tuned gap was +0.38; on test it is +1.64.
- s was chosen on STAR dev among 8 values for the C4 agent and 4 for bge-large, by the same rule. The test confirms the dev choice.

**Wording.**
- "Tuning the TM's s on dev (s 16) raises the agent with no neural network at any stage by 2.70 points on STAR test."
- "Level with a tuned logistic regression on the same bits."
- "Not worse than the bge-large agent tuned the same way."
- Never "beats the transformer" as a registered claim: R22.3 is a non-inferiority rule. Also never "state of the art" or "parity with published TMs".

**Limits.**
- One dataset (STAR). ABCD's dev gain from s 16 (+1.00) is not tested.
- Open loop, and view B reads the wizard's earlier words in the last 3 turns, as in R21.

## e75 and R23: equal tuning on ABCD (dev, then test)

Plan `logs/plan_e75_abcd_s.md` (frozen 2026-10-19T19:25:09Z). Registration `registrations/R23_tuned_s_abcd.md` (frozen 20:32:10Z, sha256 0d54d187441c7eb3). This is the 14th use of ABCD's test split, and it is worth it:
- R22's equal-tuning comparison rested on STAR alone;
- ABCD jobs cost minutes;
- every earlier ABCD use was pre-registered;
- the picks use ABCD dev only;
- the untuned gap (-1.29, R21) was close to the -2.0 margin, so this was a real test.

Every reuse weakens trust in a split a little. The gate checker passed all 133 checks on its first run.

**Outcome: R23.1 met, in the strong form, but on ABCD the tuned bge-large agent is about 1 point ahead.** The C4 agent minus the bge-large agent, each at its own ABCD dev pick, is **-0.92 [-1.61, -0.26]**. That is within the 2-point margin, but the interval lies below 0.

**e75, the picks** (ABCD dev, 4,000 clauses, seed 1; the same rule as e74):

| s | 8 | 12 | 16 | 24 | Pick |
|---|---|---|---|---|---|
| C4 agent | 0.8317 | 0.8398 | **0.8417** | 0.8341 | s 16 |
| bge-large agent (R9b's view) | 0.8398 | 0.8445 | 0.8447 | **0.8472** | s 24 |

- The C4 rows at s 8, 12 and 16 came from earlier checks with identical settings, and they were known when e75 was planned.
- The bge-large pick is the grid's edge, and its curve was still rising. A larger s could only make R23 harder for the C4 agent.

**R23 (ABCD test, 5 seeds each):**

| Arm | Accuracy | AST-weighted |
|---|---|---|
| C4 agent, s 16 | 83.6% (sd 0.32) | 85.3% |
| bge-large agent, s 24 | 84.6% (sd 0.24) | 85.6% |
| C4 agent, s 8 (R21) | 82.7% | 84.4% |
| bge-large agent, s 8 (R9b, 10 seeds) | 84.0% | 84.9% |

- **R23.1** C4 minus bge-large (at least -2.0): **-0.92 [-1.61, -0.26]**, met, strong.
- **No rule:**
  - the C4 agent's tuning gain is +0.92 [+0.52, +1.33];
  - bge-large's is +0.55 [+0.17, +0.93];
  - AST-weighted, the pair is -0.29 [-1.03, +0.43].

**Equal tuning, both datasets.** The agent with no neural network at any stage against the bge-large agent, each tuned the same way:
- STAR: **+1.64 [+0.94, +2.35]** (R22.3);
- ABCD: **-0.92 [-1.61, -0.26]** (R23.1).

Both are within the registered 2-point margin. On STAR the interval lies above 0, and on ABCD it lies below 0.

**Wording:**
- **The line** (fixed at 20:55 UTC): "Tuned the same way, the agent with no neural network at any stage is within the registered 2-point margin of the bge-large agent on both datasets: +1.64 on STAR, -0.92 on ABCD."
- **Beside it:**
  - the intervals: STAR +1.64 [+0.94, +2.35], above 0; ABCD -0.92 [-1.61, -0.26], below 0;
  - this was ABCD's 14th test use, and STAR's 6th;
  - bge-large's ABCD pick, s 24, is at the edge of the grid.
- **The independent check:** a rerun of the report is byte-identical (3c6c0c1d), and an independent bootstrap (`../verification/recompute_pairs.py`) gives -0.92 [-1.60, -0.25].
- **Never write** "better than the transformer", or "worse than" as a registered claim. Both rules are non-inferiority rules.

**The guards on the tuned ABCD agent** (e72d's method on R23's five tm_mem_c4_s16 models; note `logs/plan_e72e_guards_c4_s16.md`, frozen 21:03:46Z; dev):
- **The pre-check passed:** 500 of 500 votes, 50 of 50 winner rows, and every state inside the domain.
- **Raw:** 250 of 250 checks give a counterexample. They are far from real conversations: the nearest dev row is 1,396 to 3,907 text bits away.
- **With the 50 veto clauses:** 250 of 250 proved.
- **Cost on dev:** -0.46 to -0.60 points open loop and -0.16 to -0.35 closed loop.
- **Result:** on ABCD the tuned agent of the final comparison is also made to follow its 50 guideline-derived order rules by veto clauses.

## Newest work on the agent datasets (STAR, ABCD, MultiWOZ): the baselines today

Searched 2026-10-19 21:59 UTC to 2026-10-20 00:30 UTC, with a pause from 22:14 to 00:22. The reports
and every query are in the project notes, not published (star_abcd_newest.md, multiwoz_newest.md and their
logs). Spot-checked on their own pages: AnyTOD's venue and its state-of-the-art claim, and TOD-Flow's 48.2 F1. **None
of these numbers is directly comparable to this study's**, and none is claimed as beaten.

- **STAR** (the split is this study's own, 80/10/10 by a hash; 187 labels; a smaller input):
  - Best full-data result found: AnyTOD with T5-XXL on STARv2, EMNLP 2023 (older), 75.7% accuracy / 75.4 F1, from
    the paper's tables per the search report.
  - Newest: zero-shot LLM prompting. GPT-4 with self-explanation prompting gives 70.27 weighted F1 on STARv2
    (LREC-COLING 2024). CoDial (ACL 2026) gives 60.2% accuracy, strict zero-shot.
  - This study's tuned agents: 65.3% (C4) and 63.7% (bge-large) on its STAR test.
- **ABCD** (B-Slot: 30-way button accuracy, as in this page's tables):
  - Best found: RoBERTa, B-Slot 93.6 (NAACL 2021, older). AnyTOD T5-XXL reaches joint action accuracy 91.9 (2023,
    older).
  - Newest LLM numbers: DFlow (2024) gives B-Slot 87.62 for a fine-tuned OpenLlama-7B and 73.25 for GPT-4 3-shot,
    both on a 500-item test sample.
  - Later work (2025-2026) is about workflow and guideline compliance, not accuracy.
  - This study: 84.8 B-Slot (R19), with a shorter input window. It is below the fine-tuned 2021-2023 models (89.1-93.6).
- **MultiWOZ 2.2** (for R24; this study's task is the next system act SET as one of 128 classes, which no paper reports):
  - The official leaderboard (budzianowski/multiwoz README, last changed 2025-01-14) still has DiactTOD (2023) on top:
    end-to-end Combined 104.4, and 110.3 in the policy table with ground-truth states.
  - Newer Combined claims: MORE 107.35 (KDD 2026; evaluator not named), DIMF 106.3 (2025 preprint), SUIT 105.02
    (Findings of EMNLP 2024, standardized evaluator).
  - Act prediction: the newest next-act number found is TOD-Flow (Dec 2023, older): average F1 48.2 over 14 domains,
    with GPT-turbo plus its graph (checked).
  - LLM agents: CoALM (ACL 2025), GPT-4o Success 75.5 zero-shot on MultiWOZ 2.4.
  - Interpretable: CoDial (ACL 2026, rule code), Combined 79.5 on 2.2. No Tsetlin machine on MultiWOZ was found.
  - Caveat (Nekvinda and Dušek 2021, per the search report): scores differ across evaluators, and 174 of the 1,000 test goals also appear in
    train or dev.
- **Interpretable or logic models on these datasets:** no Tsetlin machine, decision tree or rule learner was found on
  STAR, ABCD or MultiWOZ. The interpretable work that exists is symbolic code or graphs built by or around LLMs
  (CoDial, AnyTOD, TOD-Flow). The closest Tsetlin machine prior art is text classification (Findings of ACL 2026, arXiv
  2604.12223).

## e77: a third dataset, MultiWOZ 2.2 (dev only; frozen plan; R24 waits at its gate)

Plan `logs/plan_e77_multiwoz_dev.md` (frozen 2026-10-19T21:43:42Z, 6410ca23; scope `logs/plan_e77_third_dataset_scope.md`);
dated note `logs/plan_e77_note_2026-10-20.md` (the R24 gate, fixed at 18:51 UTC before any 16k number, and
its outcome). MultiWOZ 2.2 (MIT): one example per system turn; the label is the turn's sorted domain-act set (128
labels; OTHER for combinations under 20 train turns); majority label 7.16% on dev. Test split: unread.

| MultiWOZ dev, seed 1 | 4k s 8 | 4k s 12 | 4k s 16 | 4k s 24 | 16k s 8 | 16k s 24 | s* |
|---|---|---|---|---|---|---|---|
| the agent with no neural network at any stage (C4 view B) | 0.3085 | 0.3089 | 0.3098 | 0.3029 | **0.3038** | - | 8 |
| bge-large agent (R18's view) | 0.3325 | 0.3375 | 0.3375 | 0.3400 | **0.3359** | 0.3312 | 8 (16k fell back) |

- **The tuned 16k dev gap (C4 minus bge-large, both at s* 8): -3.21 points.** The gate was -1.0, so R24 waits:
  MultiWOZ's test split stays unread, and no "better than" or "not worse" claim is made for MultiWOZ.
- **The reader, not only the bits:** LR on view B's bits (wide grid, dev) is 0.3586 at C 0.001 (the grid's edge;
  checked by e81x: C 0.001 is LR's best over C 1e-4 to 100), 5.5 points above the C4 TM at 16k and 2.3 above the bge-large TM. More clauses did not help the C4 TM (16k 0.3038
  against 4k 0.3085).
- **Where the C4 agent loses** (`scripts/e77_breakdown.py`, both s* models, bits and CPU accuracy checked equal to the
  rows; `results/explore/e77_breakdown_mwoz_tuned_c4_vs_bge_dev.json`):
  - first action of a dialogue -4.00 (0.366 against 0.406); later actions -3.08;
  - "domain right" (the predicted set has the gold set's domains) -2.41 (0.593 against 0.617); "act right" (the same
    acts, any domain) -2.62 (0.341 against 0.367);
  - both right on 19.3% of turns, only C4 on 11.1%, only bge-large on 14.3%;
  - by domain: Attraction -9.4, Taxi -7.0, Restaurant -6.7, Train -6.5; general +3.0; Booking +3.1.
- **TM against LR on the same C4 bits** (`scripts/e77_lr_vs_tm.py`, `results/explore/e77_lr_vs_tm_mwoz.json`):
  - both right 21.9%, only LR 13.9%, only the TM 8.5%;
  - the gap is in the frequent labels, not the rare ones. With 1,000 or more train examples (20 labels, 66% of dev),
    the TM has 0.410 and LR 0.473. With 200-999 examples, 0.129 against 0.180. Under 200 examples (70 labels), both
    are near 0;
  - LR gains most on general-bye+general-welcome, Train-Inform+Train-OfferBook, Booking-Book+general-reqmore and
    Attraction-Inform.
  - So the reader loses on combining several cues for common act sets. It does not lose on rare labels, unlike
    STAR's d1 diagnosis.
- **A large part of the exact-set target is a choice between near-equal variants**, which the two agents resolve
  differently: "general-bye" against "general-bye+general-welcome" (C4 0.945 / 0.008, bge-large 0.622 / 0.342; the top
  C4 confusion, 350 turns), and "X" against "X+general-reqmore" (Booking-Book +25.6 for C4, Booking-Book+general-reqmore
  -39.9). The add-on acts (welcome, reqmore) are partly the human wizard's style. This is a limit of the frozen label
  rule (the exact set), not a reason to change it after the fact; "domain right" and "act right" are reported beside it.
- **Next:** e78 (2d-a counts) and then e79 (view A+B on the C4 table) on dev, each with a bar
  fixed before its runs; R24 comes back only if a later stack passes the same -1.0 gate on fresh dev runs.

## e78: the text stack's 2d-a in the agent, concept counts beside xcust (dev; frozen plan; not taken)

Plan `logs/plan_e78_2d_counts.md` (frozen 2026-10-20T18:55:37Z, d80a9c1b), under the R4 protocol. The spec comes from the plan of the
text-understanding line (project notes, not published; f8f552fd). The code was reviewed independently (judged by an AI model, not a person).
- **Why:** view B's OR bits on the customer's words so far fill up with length. The longest 10% of dev texts have
  0.815 (STAR) and 0.767 (ABCD) of their bits on (`results/explore/c4_saturation.json`).
- **The block:** for each of xcust's 512 concepts, the bits "at least 2 / 4 / 8 of the customer's tokens so far reach
  the concept's level-3 threshold" (1,536 bits, after every view B bit; `src/dl/count_bits.py`).
  - Self-tested: view B's bits are unchanged byte for byte, and n >= 1 equals xcust's counted bit for every concept
    and row.
- **Bar** (fixed 2026-10-19 22:06 UTC, mean of seeds 1-2): STAR at least +0.5, ABCD no worse than -0.5.

| Dev | View B, seeds 1 / 2 (mean) | View B + counts, seeds 1 / 2 (mean) | Difference | Bar |
|---|---|---|---|---|
| STAR, 16k, s 16 | 0.6545 / 0.6579 (0.6562) | 0.6455 / 0.6513 (0.6484) | **-0.78** | at least +0.5: fails |
| ABCD, 4k, s 16 | 0.8417 / 0.8423 (0.8420) | 0.8390 / 0.8393 (0.8392) | -0.29 | no worse than -0.5: holds |

- **Not taken into the agent** (`logs/e78_check.out`, 30 checks ok).
- **MultiWOZ dev** (16k, s 8, seed 1; reported with no rule): view B 0.3038, view B + counts 0.3125, +0.87. That is one
  seed. The R24 gap of -3.21 would still be about -2.3, so this does not reopen R24.
- On STAR the loss is in later actions: 0.6329 / 0.6386 against 0.6424 / 0.6472. The first actions are about level.
- So counting how often the customer's words hit a concept does not help the reader. It costs about 0.8 points where
  the customer text is long.
- **Limit** (noted at 20:51 UTC): it was tested at view B's tuned s 16, as the stacking rule says. A
  view-specific s was not tried, and N8/N9 showed that a block can look worse at another view's s. So 2d-a is checked
  again only at the next milestone's re-tuning, labelled as such.
- **Next** under the same rule: 2d-b (view B + xcrecent, the customer's last 3 utterances).

## e79: view A+B on the C4 table, prototype parts from C4 mean vectors (dev; frozen plan; not taken, narrowly)

Plan `logs/plan_e79_c4_prototypes.md` (frozen 2026-10-20T19:00:04Z, a040045d), with two
additions made before the freeze; dated note `logs/plan_e79_note_2026-10-20.md`. The "as written" branch ran, because e78 did not take 2d-a.
- **Why:** on MultiWOZ dev, LR on view B's bits beats the C4 TM by 5.5 points (LR at the edge of its grid, C 0.001;
  checked by e81x: C 0.001 is LR's best over C 1e-4 to 100).
  The bge-large view has two
  dense-similarity parts that view B lacks.
- **The candidate:** view B plus `intent` and `step` built from C4 mean vectors (`scripts/c4mean_prepare.py`; the
  self-test passed 60 of 60). No neural network at any stage.
- **Bar** (fixed before any run, mean of seeds 1-2): MultiWOZ at least +1.0, STAR and ABCD no worse than -0.5.

| Dev | View B, seeds 1 / 2 (mean) | View A+B, seeds 1 / 2 (mean) | Difference | Bar |
|---|---|---|---|---|
| MultiWOZ, 16k, s 8 | 0.3038 / 0.3079 (0.3059) | 0.3169 / 0.3139 (0.3154) | **+0.95** | at least +1.0: fails |
| STAR, 16k, s 16 | 0.6545 / 0.6579 (0.6562) | 0.6561 / 0.6591 (0.6576) | +0.14 | no worse than -0.5: holds |
| ABCD, 4k, s 16 | 0.8417 / 0.8423 (0.8420) | 0.8458 / 0.8426 (0.8442) | +0.22 | no worse than -0.5: holds |

- **Not taken** (`logs/e79_check.out`, 43 checks ok). The view helps on all three datasets but misses the fixed
  MultiWOZ bar by 0.05. That is reported as it is, and the bar is not moved after the fact.
- The MultiWOZ gap to the tuned bge-large agent (0.3359) would still be about -2.1, well below R24's -1.0 gate.

## e82: the agent on the teacher-refined table (the text stack's 3a): R4, the s re-tune, the go-bars (dev only; frozen plan; no registered test by its rule)

Plan `logs/plan_e82_refined_table_agent.md` (rev. 4, frozen 2026-10-21T01:31:09Z, 1dfea762); dated notes
`logs/plan_e82_note_2026-10-21.md` (the queue runner's stop and restart) and `logs/plan_e82_note2_2026-10-21.md` (the
verification's reporting fixes). No test split was read.
- **The table** (the step-3a export of the text-understanding line, c4words_a3B_v1): the C4 word list's rows refitted by least squares so that a C4
  sentence's mean row matches a TEACHER's sentence embedding, bge-base-en-v1.5@a5beb1e3. "Network-free at run time,
  teacher at training time"; never "no neural network at any stage". The teacher is bge-large's smaller sibling and
  potion's teacher, so this is a replication of potion's recipe on the C4 word list, not a new method. Its training data
  is unpublished and could inflate the gap over the counting table. The fit filter read STAR/ABCD dev+test and
  MultiWOZ dev+test texts (removal only), so MultiWOZ's test texts have been read twice, text only.

| Dev (view B, the agent's tuned s, mean of seeds 1-2) | C4 counting table | Refined table | Difference (best-dev, the rule's statistic) | Final epoch | Mean of the last 10 evaluations |
|---|---|---|---|---|---|
| STAR, 16k, s 16 | 0.6562 | 0.6576 | +0.14 | +0.25 | +0.27 |
| ABCD, 4k, s 16 | 0.8420 | 0.8476 | +0.56 | +0.42 | +0.66 |
| MultiWOZ, 16k, s 8 | 0.3059 | 0.3108 | +0.49 | -0.02 | +0.01 |

- **R4: taken.** No dataset lost more than 0.5 points (`logs/e82_check.out`, 39 checks ok). Verified:
  the independent recompute matched; freeze and hash checks and the three-part review (leakage, selection and registration, integrity; judged by an AI model, not a person) found no blocker
  (the e82 R4 verdict; project notes, not published). The differences are inside seed noise; MultiWOZ's +0.49 is a
  best-epoch effect.
- **Where it changes decisions** (`scripts/e82_breakdown.py`; bits and accuracy checked equal to every row):
  - STAR: first action +0.50, later +0.11;
  - ABCD: first -0.05, later +0.78 (try-again +5.9, membership +5.0; log-out-in -3.7);
  - MultiWOZ: first -2.50, later +0.96; domain right -1.65, act right +0.95; per-label swings of 20 to 27 points between
    near-equal variants. So on MultiWOZ the table changes which variant the agent picks; it does not read the domain
    better.
- **e82-R3** (the e74d / e75 / e77 rules, copied): the picks are the agent's tuned s on all three datasets (STAR s 16:
  0.6582 at 16k against s 8's 0.6404; ABCD s 16; MultiWOZ s 8).
- **The go-bars** (fixed before any e82 number, on the review's power argument): STAR +0.14 against +1.0, ABCD +0.56
  against +0.7. **Neither passes.** The R24 gate for the refined agent (MultiWOZ, seed 1 minus bge-large) is -2.36
  against -1.0: **it fails.** By the frozen precedence there is **no registered test**: the STAR, ABCD and MultiWOZ
  test splits stay unread (`logs/e82_gobars.out`).
- **LR on the refined bits** (MultiWOZ dev, wide grid): 0.3643 at C 0.001, against 0.3586 on the C4 bits. The TM reads
  both about 5 points worse than LR.
  - On both bit sets, LR sat at the edge of its grid (C 0.001), so it might have been under-tuned and the gap
    understated.
  - Checked and resolved by the declared dev-only check e81x (logs/plan_e81_note2_2026-10-21.md; the e81 section):
    C 0.001 is LR's best over C 1e-4 to 100, so the gap stands.
- **R5, passed** (the 50 guard proofs on the five refined ABCD models at s 16; note `logs/plan_e82_r5_guards.md`, frozen
  04:32:36Z; `results/explore/verify_e82r5.jsonl`):
  - the encoding pre-check matches the CPU decisions on 500 of 500 dev rows per model;
  - raw: 250 of 250 checks give a counterexample (as e70, e72d, e72e); all 250 replay; the nearest dev row is 1,313
    to 2,565 text bits away (median 1,433), and none equals a real row;
  - with the 50 veto clauses: 250 of 250 proved;
  - the guards' cost on dev, with no rule:
    - open loop -0.52 to -0.71, a little higher than e72e's -0.46 to -0.60 on the C4 agent;
    - closed loop -0.05 to -0.38 (e72e: -0.16 to -0.35).

**Reading.** A table refined by a teacher reached potion on CLINC150 in the intent tests of the text-understanding line, but inside the agent it
moves dev accuracy by less than seed noise: no loss anywhere, a small and stable +0.4 to +0.7 on ABCD, and on MultiWOZ
only a shift between near-equal label variants. The agent's limit on these datasets is not the word table: the reader
(the TM against LR on the same bits, about -5 on MultiWOZ; LR's C 0.001 checked by e81x and kept as its best) and the
missing information (system results, P9) weigh more.

## e81 and e81x: the TM's vote threshold T, and LR below its grid's edge (dev only; frozen plan and two dated notes; no rule met)

Plan `logs/plan_e81_T_screen.md` (frozen 2026-10-21T05:47:50Z, c58c502a, record 0dc8cfc1). Note 1
`logs/plan_e81_note_2026-10-21.md` (05:49:36Z) pins the driver. Note 2 `logs/plan_e81_note2_2026-10-21.md` (06:04:24Z)
declares e81x post hoc, before any e81 number.
- **How it ran:** 10:26:25-12:50:18Z, alone on the GPU. 35 jobs, all exit 0 on try 1. No test split was read.
  - The run went through `logs/e81_chain.sh` (039c01c2). It runs the pinned driver, then note 2's e81x queue and
    report, with no choice logic.
  - Its test `tests/test_e81_chain.py` PASSED (output 60eb933b) and was checked independently at 06:11Z. No
    frozen note pins the chain.
- **File times:** the plan's mtime and its three records' mtimes (06:19-06:36Z) are later than their freezes.
  - Their birth times equal the freeze times, and every sha256 equals its record.
  - The same hashes were logged, append-only, at the freeze moments (the saved record and the status
    lines of 05:47-06:11Z, both in the project notes, not published).
  - Other files from about 04:30 to 06:40Z show the same delayed mtimes. So this is read as a filesystem artefact, not
    an edit.
- **A correction to the plan:** its line "In every run so far T equals the number of clauses" is wrong. e47
  (2026-10-17; STAR dev, bge-large view, 4k, s 8) ran T 1,000: 0.6060 against 0.6131 at T = clauses.
- **Known at the freeze, not used:** the STAR and ABCD test results of these agents (STAR: R18, R19, R21, R22; ABCD:
  R19, R21, R23). No e81 rule, bar or number uses a test value.
- **The tables' caveat:** the refined (a3B) table's fit sentences were filtered against the STAR, ABCD and MultiWOZ dev
  and test texts, removal only (`logs/plan_e82_note2_2026-10-21.md`). The C4 table's corpus was deduplicated against
  the eval splits. Every e81 arm uses the same table and bits, so this cannot move a comparison.
- **Question:** is T, the TM's vote threshold, part of the reader's gap?
- **The arms:** T in {clauses, clauses/2, clauses/4} at 4,000 clauses, seed 1, at each agent's tuned s:
  - the refined agent (a3B table, view B);
  - the bge-large agent (R18's view);
  - the C4 agent (no rule).
- **The screen:** seed 1 at 4k against the s-tuned reference row at T = clauses; dev accuracy at the best epoch; a pick
  needs +0.3, and ties go to the larger T (`logs/e81_pick_screen.out`).

| Agent | Dataset | T = clauses | clauses/2 | clauses/4 | Pick |
|---|---|---|---|---|---|
| refined | STAR | 0.6375 | 0.6380 | 0.6348 | clauses |
| refined | ABCD | 0.8461 | 0.8415 | 0.8341 | clauses |
| refined | MultiWOZ | 0.3123 | 0.2976 | 0.2950 | clauses |
| bge-large | STAR | 0.6401 | 0.6439 | 0.6506 | clauses/4 (+1.04) |
| bge-large | ABCD | 0.8472 | 0.8480 | 0.8453 | clauses |
| bge-large | MultiWOZ | 0.3325 | 0.3181 | 0.3230 | clauses |
| C4 | STAR | 0.6349 | 0.6314 | 0.6303 | clauses |
| C4 | ABCD | 0.8417 | 0.8396 | 0.8301 | clauses |
| C4 | MultiWOZ | 0.3085 | 0.2999 | 0.2992 | clauses |

  - **Caveat:** 7 of the 9 reference rows won their s screen on the same dev split and seed, which tilts the screen
    toward T = clauses. The tilt looks small: refined ABCD at the same setting gives 0.8461 on seed 1, and seeds 2-5
    average 0.8465.
- **The one pick did not hold.** bge-large on STAR at 16k, T 4,000 against T 16,000, mean of seeds 1-2
  (`logs/e81_pick_final.out`):
  - +0.14 at the best epoch;
  - -0.54 at the final epoch;
  - -0.33 on the last-10 mean;
  - the seed-2 pair alone (both rows new): +0.39.
- **B1** ("T helps this agent's reader": +0.5 on the 2-seed mean, per dataset): **not met** for the refined or the
  bge-large agent on any dataset. The C4 agent (no rule) also kept T = clauses on all three.
- **B2** (the refined TM on MultiWOZ within half its gap to LR, at least -2.677): **not met.** TM minus LR is -5.35, as
  before. The last-10 mean reads -8.03.
- **e81x** (declared post hoc, no rule; `logs/e81x_report.out`), LR on MultiWOZ dev at C 1e-3 / 3e-4 / 1e-4:
  - refined bits: 0.3643 / 0.3528 / 0.3375;
  - C4 bits: 0.3586 / 0.3513 / 0.3313.
  - C 1e-3 reproduces both old rows exactly, so C 0.001 is LR's best over C 1e-4 to 100. LR was not under-tuned, and
    the TM - LR gaps on MultiWOZ stand: refined -5.35, C4 -5.27.
  - The label-set diagnostic's side reading, that more regularisation helps on these bits, does not hold for LR.
- **TM minus LR beside it** (no rule): at T = clauses, except bge-large STAR at its pick (T 4,000 at 16k).

| Agent | STAR | ABCD | MultiWOZ |
|---|---|---|---|
| refined | -0.83 | +3.27 | -5.35 |
| bge-large | -1.51 | +1.11 (1 seed) | -3.00 (1 seed) |
| C4 | +0.04 | +3.01 | -5.27 |

- **Verification:** the independent recompute matched (the e81 verdict; project notes, not published).
  - An independent recompute, written from the plan's text, agrees with every printed number.
  - The three e81 tests, rerun on the frozen code, give byte-identical outputs.
  - The three-part review (leakage, selection and registration, integrity; judged by an AI model, not a person) found no blocker. Its one
    major, the scope of the reading, is applied below.

**Reading.**
- At each agent's s tuned at T = clauses, T = clauses/2 and clauses/4 did not improve the reader on STAR, ABCD or
  MultiWOZ dev (4k screen, seed 1). The one pick that moved (bge-large STAR) gave +0.14 at 16k on 2 seeds.
- Lower T was below T = clauses in 14 of the 18 screen cells. On MultiWOZ it was 0.86 to 1.73 points below for all
  three agents, so these T values do not close the MultiWOZ gap (B2 needed +2.68).
- T's effect depends on s. On bge-large STAR at 4k, T 1,000 gave -0.71 at s 8 (e47) and +1.04 at s 16 (e81), each a
  single run.
- Not run:
  - a joint (T, s) search;
  - other T ratios;
  - T below the clauses at full size anywhere but bge-large STAR.
- The vanilla-TM advice (Tarasyuk et al. 2026: T above 0.333 C is impractical) was not borne out for this coalesced TM
  at these settings.
- MultiWOZ's TM - LR gap (-5.35 refined, -5.27 C4) is neither T at these values nor LR's tuning.
- What the earlier rows show (descriptive, no rule; project notes, not published):
  - on MultiWOZ the TM's dev curve is flat after epoch 4, while STAR's and ABCD's keep rising;
  - its clauses stay at about 2 literals for every s from 8 to 24.
  - The next knobs to test act on the learning dynamics: the negative-sampling rule, clause dropout and the literal
    budget (FAILURE_MAP Part 20).

## e83: the TM's learning dynamics on MultiWOZ: random negative classes (dev only; frozen plan and a dated note; C1 and C2 met, C3 not met)

Plan `logs/plan_e83_learning_dynamics.md` (frozen 2026-10-21T14:36:33Z, 5ddfe717, record e16f0f52). Note
`logs/plan_e83_note_2026-10-21.md` (frozen 15:09:42Z, b78b0018) drops the L8 arm before any screen row was read.
- **Why L8 was dropped:** tm.py's chunked update raises NotImplementedError for an exact literal budget, and every
  MultiWOZ view-B run at 4,000 clauses uses that update. L8 was not run.
- **Data:** no test split was read.
- **The arms:** the refined agent (a3B table, view B) on MultiWOZ dev. Every arm went through `scripts/knob_run.py`,
  which saves its dev predictions.
  - REF: the "hardest" negative class;
  - nrand: a uniform random other class;
  - nmix50: hardest with probability 0.5;
  - drop50: clause_drop 0.5, redrawn per minibatch.
- **The knob-off identity passed** in the GPU smoke (`logs/e83_smoke_check.out`). A row through the wrapper with no
  knob equals the plain row in every field but the clock fields.
  - It was shown at 200 clauses on ABCD, which uses the unchunked update. C1 and C3 compare wrapper rows with plain e82
    rows that used the chunked update (a note from the integrity part of the review).
- **The screen** (4k, T 4,000, s 8, seed 2; best-dev / last-10 mean against REF; `logs/e83_pick_screen_noL8.out`):
  - nrand +3.89 / +3.93;
  - drop50 +2.02 / +2.01;
  - nmix50 +0.96 / -0.40.
  - The pick is nrand.
  - The four-arm screen check and the later stages ran by hand, as the note declares. The screen check's time is that
    of its output file (15:10:03Z).
- **C1, "the knob lifts the MultiWOZ reader": met.** nrand at 16k, mean of seeds 1-2, against e82's 16k rows: +4.23 at
  best-dev and +5.98 on the last-10 mean (final epoch +4.85).
- **C2, "the knob closes half the gap": met.** TM minus LR on the same bits is -1.13. It was -5.35, and the bar was
  -2.68. The last-10 mean minus LR is -2.05 (no rule).
- **C3, "no harm on STAR and ABCD": not met.** At best-dev, on the 2-seed means, STAR is -11.91 and ABCD -4.00. So the
  claim is MultiWOZ-only (`logs/e83_pick_final.out`).
- **P1, the near-duplicate prediction: not met.** On the 6 pair labels, the pooled accuracy moved +0.80 under nrand and
  -1.54 under nmix50; P1 needed +2.0.
  - The lift comes from elsewhere. Per label (screen, 4k), nrand raises:
    - Hotel-Inform 6.1 to 29.7;
    - Booking-Request 17.1 to 47.9;
    - general-reqmore 5.7 to 33.9.
  - It lowers:
    - Train-Request 84.6 to 69.7;
    - Booking-Book+general-reqmore 87.4 to 68.7.
  - The predictions spread from frequent labels to rarer ones.
- **Verification:** matched (the e83 verdict; project notes, not published). Two independent recomputes match every
  printed number. The three-part review (leakage, selection and registration, integrity; judged by an AI model, not a person) found no blocker.
  - **Caveats from the review:**
    - **The 4k screen is noisy.** At the same setting, seed 1 gave 0.3123 and REF's seed 2 0.2969. So drop50's +2.02 and
      nmix50's +0.96 at best-dev sit inside the 4k noise. nrand's pick held on fresh 16k runs (C1).
    - **The plan's dates.** The time "14:4x UTC" that the plan gives for the review it cites is that message's own label,
      not a clock read. The message was received before the 14:36:33Z freeze.
    - **Multiplicity:** 3 arms with one pick, then fresh 16k runs (C1); P1 judged on 2 arms.

**Reading.**
- The negative-class rule accounts for most of the MultiWOZ gap. With a random other class, the TM reads MultiWOZ's
  bits 1.13 points below LR instead of 5.35 (dev, 16k, 2 seeds).
- The same rule costs 11.9 points on STAR and 4.0 on ABCD. So it is not a general setting of the reader: it acts like
  a per-dataset setting, as s does.
- Not run:
  - a choice of rule per dataset on dev;
  - focused negative sampling as published, or other mix values;
  - dropout at 16k;
  - the budget in the chunked update.

## e84: random negatives for every MultiWOZ agent, and the R24 gate re-check (dev only; frozen plan; both gates met, steady)

Plan `logs/plan_e84_negatives_fair.md` (frozen 2026-10-22T10:37:12Z, 50bf9d34, record bcdf2937). It ran 12:16-13:07Z:
4 jobs, all exit 0 on try 1. No test split was read.
- **The start:** a waiter on the end line of R3c (a run of the text-understanding line) ran the plan's two commands. The plan had said "run by hand". The waiter
  (e84_wait_start.sh, 9fe4f7f4; project notes, not published) was written after the freeze, and its dry test passed.
- **The rows** (MultiWOZ dev, random negatives, 16k, T 16,000, s 8, mean of seeds 1-2; best-dev / last-10 mean;
  `logs/e84_check.out`):

| Agent | Best-dev | Last-10 mean | Random minus hardest (best-dev) | TM minus LR on its bits |
|---|---|---|---|---|
| refined (a3B; network-free at run time, teacher at training time: bge-base-en-v1.5) | 0.3530 | 0.3438 | +4.23 (2 seeds) | -1.13 |
| bge-large (R18's view) | 0.3486 | 0.3360 | +1.26 (1 seed) | -1.74 |
| C4 (no neural network at any stage) | 0.3500 | 0.3431 | +4.41 (2 seeds) | -0.86 |

- **G-ref (the R24 gate, refined minus bge-large, at least -1.0): met, steady.** +0.45 at best-dev, +0.77 on the
  last-10 mean.
  - The refined side (the e83b rows, 0.3530) was known when e84 was frozen, so only the bge-large side was fresh.
    G-C4, with both sides fresh, is the cleaner gate (a note from the selection part of the review).
- **G-C4 (the R24 gate, C4 minus bge-large, at least -1.0): met, steady.** +0.14 and +0.71.
- So an R24 draft may follow. It needs its own registration, frozen before MultiWOZ's test split is read.
- **Verification:** an independent recompute matches both gates
  (the e83 verification's recompute of e84; project notes, not published). The three-part review of e83 and e84
  (judged by an AI model, not a person) ran later the same day; its caveats follow.

- **Caveats from the review** (the e83 verdict, which covers e83 and e84; project notes, not published):
  - **The rule was tuned on one agent only.** Random negatives were chosen on MultiWOZ dev for the refined agent and
    applied unchanged to the other two.
  - **No agent's s was re-tuned under the rule.** bge-large's 4k screen under "hardest" preferred s 24.
  - **The test-text path differs between the arms.** The a3B fit filter read MultiWOZ dev and test texts (removal
    only), the C4 table was not deduplicated against MultiWOZ, and bge-large's training data is unknown.

**Reading.**
- With the same negative-class rule for every agent, the two network-free agents are level with the bge-large agent
  on MultiWOZ dev. The gate is a -1.0 margin on 2-seed means, with no interval, so no "better than" follows.
- Before, with "hardest" negatives, they were 2.4 to 3.2 points behind (e77b, e82).
- **Caveat 1:** the rule was picked on this dev split (e83, refined agent, seed 2).
- **Caveat 2:** it costs STAR and ABCD (e83), so it is a MultiWOZ setting.
- MultiWOZ's test split, through a frozen R24, is the real check.


## e84b: seeds 3-4 for the R24 gate (dev only; frozen note; both gates hold)

Note `logs/plan_e84_note_2026-10-22.md` (frozen 2026-10-22T18:03:34Z). It ended 19:36:09Z, exit 0. No test split was
read. The rows: random negatives, MultiWOZ dev, 16k, T 16,000, s 8, mean of seeds 1-4 (`logs/e84b_report.out`):

| Agent | Best-dev | Final epoch | Last-10 mean | TM minus LR on its bits |
|---|---|---|---|---|
| refined (network-free at run time, teacher at training time: bge-base-en-v1.5) | 0.3534 | 0.3447 | 0.3448 | -1.09 |
| bge-large (R18's view) | 0.3460 | 0.3348 | 0.3373 | -1.99 |
| C4 (no neural network at any stage) | 0.3484 | 0.3407 | 0.3417 | -1.02 |

- **The robustness rule (at least -1.0 at best-dev): both gates hold.** G-ref +0.73, G-C4 +0.23. Last-10 (no rule):
  +0.75 and +0.44.

## e85: s for each MultiWOZ agent under random negatives, by e77's rule (dev only; frozen plan; s 8 for all)

Plan `logs/plan_e85_s_under_random.md` (frozen 2026-10-22T18:22:19Z). It ran 19:44:56Z to 20:35:26Z, exit 0. It
started before R24's freeze (20:16:52Z); no e85 number was read before the freeze.
- **The 4k screen** (seed 1, T 4,000; `logs/e85_pick_screen.out`), s 8 / 12 / 16 / 24:
  - refined 0.3371 / 0.3393 / 0.3336 / 0.3309;
  - bge-large 0.3337 / 0.3332 / 0.3322 / 0.3306;
  - C4 0.3355 / 0.3361 / 0.3374 / 0.3265.
- **No s beat s 8 by +0.3,** so s* is 8 for all three agents (`logs/e85_pick_final.out`). No 16k follow-up ran.
- An independent recompute from the rows gives the same picks (the R24 verification's s* recompute; project notes, not published).

## R24: on MultiWOZ 2.2 test, the network-free agents against the bge-large agent, all with random negatives chosen on MultiWOZ dev (registered; 1st use; verified)

Registration `registrations/R24_multiwoz_random_negatives.md` (frozen 2026-10-22T20:16:52Z, 4d675dde, record
78f16551). The chain (`logs/R24_chain.log`) ran 2026-10-22T20:35:39Z to 2026-10-23T00:36:31Z, in this order:
- the gate at s* (G-s*: C4 +0.15, refined +0.12; both hold);
- the network-free path test (passed);
- 18 test jobs, all exit 0 on try 1;
- the checker (`logs/r24_check.out`: all checks passed);
- the registered report (`logs/R24_report.out`) and the two-level intervals (`logs/R24_twolevel.out`).

| Agent | R24.1: non-inferiority at -1.0 | R24.1 strong | R24.2: superiority | Registered interval | Two-level (no rule) |
|---|---|---|---|---|---|
| C4 (no neural network at any stage; the headline) | met | met | not met | +0.53 [-0.18, +1.23] | [-0.32, +1.40] |
| refined (network-free at run time, teacher at training time: bge-base-en-v1.5) | met | met | not met | +0.05 [-0.65, +0.72] | [-0.83, +0.95] |

- **Accuracy** (mean of 5 seeds; 7,355 test turns in 1,000 dialogues):
  - TM: C4 0.3437, refined 0.3389, bge-large 0.3384;
  - LR on the same bits: 0.3592, 0.3638, 0.3573.
- **The registered reading:** "On MultiWOZ 2.2 test (its 1st use), the agent with no neural network at any stage
  (the C4 counting table) is not worse than the bge-large agent by more than 1.0 point: +0.53 [-0.18, +1.23]
  (non-inferiority registered), with all three agents using random negatives, chosen on MultiWOZ dev."
  - The refined agent's line, network-free at run time, teacher at training time (bge-base-en-v1.5): +0.05 [-0.65,
    +0.72].
- **No "better than".** R24.2 is not met for either agent.
  - The earlier headline, superiority in e77's plan and in R24's rev 0, is not met.
  - MultiWOZ test was the study's only fresh split for a registered "better than". It is now spent without one.
- **Beside every line:**
  - **The rule was chosen on MultiWOZ dev** (e83), on the refined agent. On dev it lifts the network-free agents far
    more than bge-large: +4.26 refined and +4.25 C4, against +1.01 (e84b). Under "hardest" negatives the dev gap was
    -3.21 (e77b).
  - **It costs other datasets.** On dev it costs STAR -11.9 and ABCD -4.0 (measured on the refined agent, e83). So it
    is a MultiWOZ setting.
  - **Selection is optimistic.** The rule, the gate and s (e85) were all chosen on MultiWOZ dev, and the dev numbers
    are best-epoch numbers. Every dev try at the gate: e77b -3.21, e78 -2.34, e79 -1.90, e82 (refined) -2.36, then
    e84 +0.14 with random negatives.
  - **The view (B) was chosen on STAR and ABCD dev.** bge-large keeps R18's view. Only s was tuned equally (e85).
  - **mem reads the true earlier acts** of the dialogue. This is the offline setting, the same for all agents.
  - **Text overlap.** The C4 table was not deduplicated against MultiWOZ: it overlaps 2.47% of dev and 2.61% of test
    utterances, about equally.
  - **The refined table's fit filter read dev and test texts** (removal only: 68 and 5 sentences dropped).
  - **The teachers' training data cannot be checked.** Contamination in bge-large would work against the C4 headline.
- **No rule:**
  - TM below LR on the same bits for all three agents: C4 -1.55 [-2.32, -0.79], refined -2.49 [-3.33, -1.64],
    bge-large -1.89 [-2.69, -1.10].
  - LR's C is at the grid's edge (0.001) for the C4 and refined bits. e81x found it LR's best over 1e-4 to 100 on dev.
  - The -2.0 reading: both agents are above it.
- **Verification:** matched (the R24 verdict; project notes, not published).
  - The frozen report reruns byte-identical.
  - Two independent recomputes (`../verification/recompute_pairs.py`, `../verification/recompute_r24.py`) agree.
  - s* was recomputed from the rows.
  - The three-part review (leakage, selection and registration, integrity; judged by an AI model, not a person) found no blocker. The four wording points of its selection part are in this
    section.
- **Corrections to the frozen text:**
  - its gate-try list leaves out e78 and e79;
  - rev 0's hash b19d8c4a is of its 2026-10-21T03:51Z edit;
  - draft 3cb1dff2 (19:39Z) is missing from its draft history.

## e86 and e87: stopped before their analyses (dev only; frozen plans)

- **e86** (`logs/plan_e86_clause_drop.md`, frozen 2026-10-23T00:58:18Z): clause dropout 0.5 for the C4 agent on
  MultiWOZ, STAR and ABCD dev, and random negatives with dropout as one negative-sampling setting for all three. The
  dry runs passed, and the queue of 14 dev jobs started at 01:14:51Z. Eight jobs ended with exit 0: two MultiWOZ runs
  (random negatives with dropout, seeds 1 and 2) and six ABCD runs (the reference, dropout, and random negatives with
  dropout, seeds 1 and 2 each). The chain was stopped at 02:26:38Z during the ninth job, the first STAR run
  (`logs/e86_queue.log`, `logs/e86_end.log`; run logs, local). The plan's check and report were not run, so e86 has no
  result, and no number from its runs is used on this page or elsewhere.
- **e87** (`logs/plan_e87_receipts.md`, frozen 2026-10-23T01:55:38Z): how short the receipts behind the C4 agent's dev
  decisions are, and what short rules cost, against LR and L1-LR on the same bits. Its two chains started at 01:55:53Z,
  waited for the e86 queue to end, and were stopped at 02:26:38Z before any run (`logs/e87_a0_end.log`,
  `logs/e87_a_end.log`; run logs, local).
