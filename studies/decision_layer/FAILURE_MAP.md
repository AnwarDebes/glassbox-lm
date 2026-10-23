# Failure map: where Tsetlin Machines fail on language, part by part

Status: written 2026-10-15 22:30 UTC to 2026-10-16 16:40 UTC; phase 2 (fresh data: ABCD, STAR; and MASSIVE, a related dataset (it shares about a third of its test texts with HWU64 train, with different labels)) 2026-10-16 21:31 UTC to 2026-10-17; phase 3 (the agent with no neural network at run time; its word table was distilled once, offline, from a transformer) 2026-10-17 20:47 UTC to 2026-10-19. Phase 1's registered tests (R2 to R8, R2c, R2d, R2e, R5c, R5d, R7b, R7c) have finished; phase 2's are R9, R9b, R10, R10c, R11, R12, R13, R14, R15, R16, R16b, R17, R18 (section "Phase 2" below); phase 3's are R19, R20, R21, R22 and R23 (section "Phase 3"), and phase 4's is R24 (section "Phase 4", 2026-10-19 evening to 2026-10-23, on MultiWOZ 2.2). Internal audits 1 to 9 are in the project notes (not published). Every registered test below gets its pass rule written and frozen (hash and UTC time in `registrations/`) BEFORE its confirmatory runs. Exploratory numbers are on validation splits only and are labelled as such.

Evidence for every failure is in the project notes (not published): 19 failures F1 to F19 and 11 positive results P1 to P11, each with run ids. This page groups them by cause and attaches one test per cause.

## The verdict, before the parts

- **As the language model itself, the TM loses to counting baselines of matched strength wherever it was tried.**
  - KN-5 beats it on WikiText-103 (F1, 1.62 times the bits) and on Penn Treebank (F18, perplexity about 533 against 153).
  - Learned clauses add 0.0002 bits over a count stack (F5).
  - Its only wins are against weaker counts, for example a left bigram (P4).
  - This direction is closed here. No test below tries to rescue it.
- **As a decision layer on top of an LLM encoder, the TM works: readable, calibrated, and able to learn an agent's decisions from reward. In routing accuracy with labels it never beats a linear model on the same information in the language it was trained in; trained on English only, it is ahead of LR on the same bits in Norwegian (+0.72 [+0.30, +1.16], R12.1, read from a non-inferiority rule). As a bandit with 10 and 18 actions it beat LinUCB (R5, R5b), and on the ABCD workflow it ends 3.0 to 5.5 points above LinUCB after the 15 registered passes (R10c, alpha 0.1 to 1.0; LinUCB is ahead for the first 7 to 11 passes and earns more reward while learning, and on dev with alpha 4.0 it ends 0.5 points below the TM), and at out-of-scope detection on thermometer bits it beat logistic regression in every seed (R1 with bge-base; R4b, 16,000 clauses, with both encoders).**
  - **Met (9 registered tests): R1, R6, R7, R8, R5c, R5d, R2d, R2e, R7c.**
    - R7, the readable layer: inputs a person can read (words, and the LLM's whitened similarity to each intent's centre). Within 0.3 points of a linear model on the same inputs; the 5-machine composite is within one point of the opaque linear model (CLINC150 96.5% against 97.5%), a single machine 0.85 to 1.08 points below it. Every decision is the weighted vote of named clauses.
    - R1: calibrated after one temperature (error 0.013 to 0.025); flags out-of-scope queries as well as logistic regression, with 16,000 clauses better in every seed.
    - R6 and R8, the agent loop with TD learning: from delayed reward alone the TM policy learns to ask while the query is unclear and to escalate out-of-scope queries after asking, over two and over four steps. Its gain over the same TM trained on the immediate reward (+0.40, +0.37) is nearly built in; over always-ask-then-route (R8, +0.16) it comes from the out-of-scope queries, while in scope it equals always asking.
    - R5c and R5d, reward to Type I/II feedback: with 150 and with 77 actions the TM bandit reaches 0.998 and 0.996 of the same TM trained on labels in final accuracy. Against LinUCB it is -0.98 [-1.20, -0.80] (just inside the one-point margin) and -0.71 [-0.96, -0.33]. Online it makes 2.6 to 3.4 times LinUCB's errors.
    - R2e and R2d, the fix of Part 1 with an exact length bound: the literal budget lifts the word router by 5.8, 4.8 and 4.0 points (CLINC150, Banking77, HWU64), with no clause over 32 literals; on CLINC150 it is level with LR on the same bits, on Banking77 and HWU64 it trails it (-1.34, -5.32). The bounded clauses stay 95 to 99% negation-only: short blacklists. R2d mainly checks the bound: 16 of its 20 models are identical to R2c's.
    - R7c, the readable layer with every clause at most 16 literals: level with LR on the same inputs (-0.01 to +0.01), composite within one point of the opaque model (-0.46 to -0.84), out-of-scope AUROC 0.950.
  - **Part by part.** Registered fixes are met for Part 4 (R1), for Part 5 over time (R6, R8) and from reward alone (R5c, R5d: final accuracy; online the TM makes 2.6 to 3.4 times LinUCB's errors), and for Part 1's accuracy and length rules (R2e, R2d; the clauses stay 95 to 99% negation-only). Not fixed: Part 3 (R4, R4b not met; a single TM stays 0.1 to 2.4 points below LR on the raw embedding; on readable inputs the TM equals LR on those inputs, and both stay 0.6 to 0.9 points below LR on the raw embedding), Part 5 through layers (avoided: no stacking) and Part 6 (measured, not tested). Part 2's fix, the LLM encoder, is measured with no rule of its own; R7 and R7c test the readable layer built on it. Part 1 took 5 registered tests (2 met) and the bandit 4 (2 met), on reused test splits.
  - **Not met (8): R2, R2b, R2c, R4, R4b, R5, R5b, R7b.** R2, R2c and R7b failed only their clause-length rules, by clauses one literal over the budget; the cause was a tie bug in the budget code, fixed; R2c and R7b were re-run as R2d and R7c, and R2 as R2e (which also changed the start state), all met. R2b (tmu's budget form with this machine's start state, -25.5 points) is a failed alternative; tmu's own all-excluded start was not tested with it. R4 and R4b: one TM trails logistic regression on embedding thresholds (R4: by 0.6 to 2.4 points; R4b, 16,000 clauses: within 0.5 points in 3 of 6 cells). R5 and R5b: reward-only routing with 18 and 77 actions (R5; the 18-action task then passed inside R5b, +1.69 [+1.23, +2.03]) and 77 actions after 15 passes (R5b, -3.16 points). With Boltzmann exploration and 30 passes both the 77-action task (R5d) and a 150-action task (R5c) are met.
  - **Phase 2, on data no phase-1 test had read (ABCD, STAR; MASSIVE is a related dataset (it shares about a third of its test texts with HWU64 train, with different labels); details in the Phase 2 section):** met R9b (memory literals on a real agent workflow, +18.65 points), R10 (the workflow from a 0/1 reward alone, 0.988 of the supervised TM; memory +22.87 points), R10c (after the 15 registered passes the TM bandit is 3.0 to 5.5 points above LinUCB with alpha 0.1 to 1.0; LinUCB earns more reward while learning), and the rules R9.2 to R9.4 (on ABCD, TM level with LR and MLP on the same named inputs, its composite level with an opaque MLP), R11.2 (the bounded readable layer passes R7c's rules again on MASSIVE, with less room), R12.1 and R12.3 (trained on English, 82.2% on Norwegian, ahead of LR on the same bits under a non-inferiority rule), R13 (closed loop: with its own presses as memory, 80.7%, still +15.3 points over no memory), R14 (from reward alone within the margin of a small neural bandit on the point estimate, -0.45 [-1.04, +0.16]; it learns more slowly), R15 (replay closes 77% of the online gap to the neural bandit, which stays at -1.9 [-2.34, -1.49] points; at the end +0.52 under a non-inferiority rule), R16 (Granmo's hierarchical TM, as configured with 2 times the clauses and 16 times the automata, is 9.4 points above the flat machine of the same package; 2.3 below this study's coalesced TM), R17 (the reward-learned agent with its own presses as memory, the text still logged: 79.1%, +19.0 points over no memory; its own mistakes cost 3.7 points, interval [-4.44, -2.99], past the 4-point margin), R16b (the hierarchical TM stays 7.0 points ahead of a flat machine with as many automata; T, s and the graded vote still differ), R18 (the memory result replicates on STAR, +15.4 points, though there the TM is 3.3 points behind LR). Not met: R9 as a whole (R9.1, a design error in its control), R9.5 (TM beats LR), R11 as a whole (R11.1, R11.3), R12 as a whole (R12.2).
  - **Phase 3 (STAR and ABCD, 2026-10-18): met R19, the agent with no neural network at run time** (a rule-based tokenizer, lookups in a fixed 63,091 x 512 table, then a bitwise OR of precomputed per-token bit codes, then Boolean clauses and an integer vote; the table was distilled once, offline, from a transformer, bge-base-en-v1.5 via Model2Vec). It is level with the bge-large agent on STAR (+0.06 [-0.63, +0.74] points) and 1.0 point below it on ABCD, inside the registered 2-point margin (-1.00 [-1.62, -0.40]). Memory still adds +15.05 and +20.79 points. Without memory the table costs 3.14 points on ABCD (post hoc). One decision takes 29.5 ms on one CPU thread, against 806 ms (medians, rough: shared machine; the text step's speed-up is Model2Vec's). Both agents were run on one CPU thread; the bge-large agent would be faster on a GPU (not measured), which the word-table agent does not need.
  - **Phase 3, the closed loop (R20, 2026-10-19).** The headline question, whether memory pays for an agent that reads only the customer (no text the wizard wrote), was stopped at its dev gate on STAR: memory added +4.50 in the closed loop against the +5.0 gate, so STAR's test split was not spent on it (a dev-only line in Part 13). On STAR test, R19's word-table agent in the closed loop: memory still pays (+7.53 [+6.51, +8.58], R20.1 met); its own mistakes cost 7.51 points, more than R13's bar of 4 (R20.2 not met, as expected); and it is not worse than the bge-large agent in the same loop (+1.96 [+1.07, +2.85], R20.3 met, strong; superiority was not a registered rule). On ABCD, as supporting evidence, the customer-only agent in the closed loop: memory +23.94 [+22.20, +25.67] (R20.4 met), at 72.6%, 7.0 points below the agent that also reads the human agent's text.
  - **Phase 3, no neural network at any stage (R21, 2026-10-19): met on both datasets, in the strong form.** The agent's word table is counted from C4 text (the C4 word table: a C4 word list, counting + SVD), then k-means + thresholds, then a Tsetlin machine.
    - Against the potion agent: STAR -0.10 [-0.65, +0.45] points, ABCD -0.29 [-0.74, +0.17].
    - Against the bge-large agent (no rule): -0.04 and -1.29.
    - With e70's 50 veto clauses all 50 guideline-derived order rules hold for its 5 ABCD models (e72d, dev).
    - Rough speed on one CPU thread: 2.6 ms and 0.5 ms per decision.
  - **R22 (2026-10-19, STAR test): all three rules met.** With the TM's s tuned on dev (s 16), the agent with no neural network at any stage gains +2.70 points.
    - It is level with a tuned LR on the same bits (-0.09 [-0.90, +0.73]).
    - It is not worse than the bge-large agent tuned the same way (+1.64 [+0.94, +2.35]).
  - **R23 (2026-10-19, ABCD test): met, strong.** Under equal tuning on ABCD the C4 agent is -0.92 [-1.61, -0.26] against the bge-large agent: within the 2-point margin, with bge-large about 1 point ahead.
  - **What the TM does not add (post hoc and unregistered comparisons):**
    - On one-step routing the LLM does the work: the readable TM agrees with the one-rule router "route to the nearest centre" on 99.2 to 99.7% of queries, and a linear model on the same inputs explains its decisions at least as compactly.
    - In the agent loop a linear Q-learner on the same information is as good (R6) or 0.02 better (R8). The tuned threshold rule loses in R6 (-0.04, all on out-of-scope episodes) and is 0.02 better in R8, where it was tuned with the same out-of-scope weighting as the learners. Memory literals did not help in R6 and R8, whose tasks did not need them (+0.001, -0.012). On ABCD, where the next step depends on the history, they add +18.65 (R9b) and +22.87 points (R10).
    - The budget bounds clause length but turns most clauses into exclusion lists, and then a decision is spread over thousands of them (R7b).

## The parts

Each part is one cause behind several failures. The order is the order of work.

### Part 1. Sparse word inputs turn clauses into blacklists

- **Failures:** F12, F14, F15, F19. The same shape showed up today on the prototype input (below).
- **Mechanism (measured):** with one-hot or bag-of-words input, almost every negated literal is true. Feedback that includes "true" literals therefore piles up "NOT word" literals. The clause becomes "none of these 1,000 to 2,000 words", which fires for almost everything. T0 today: 1,720 of 2,000 clauses had no positive literal, about 1,900 literals each. Router (numba): all 4,000 clauses negation-only, median 1,021 literals.
- **Measured today (exploratory, CLINC150 validation, word input `lex:2000`):**
  - Default TM: 81.23. Median clause length 1 literal, max 781.
  - Logistic regression on the same bits: 87.63.
  - Negation off: 65.57. The clauses can no longer exclude, so it does not fix anything.
  - A literal budget lifts accuracy: exact budget 8, 16, 32 gives 82.13, 85.00, 87.17. But 98% of the clauses stay negation-only.
  - The budget form of the first exploratory run, a clause that is full gains nothing that step, gives 88.17 and 87.70 (two seeds) and 87.53 at budget 8. 39 to 46% of its clauses have a plain literal.
  - tmu's budget form gives 56 to 59 combined with this machine's start state (half of the literals included).
- **What a default word router really holds** (R2 arm A, seed 1, test 0.8162):
  - 51.7% of its clauses are single-word rules that a person can read, for example `contains "pin"` -> pin_change (weight 858) and `contains "luggage"` -> lost_luggage (772).
  - 25.5% of its clauses hold more than 100 literals. The longest has 799 literals, all of the form "NOT word", and votes for user_name.
  - So the router is half keyword rules and a quarter blacklists.
- **What should fix it:** a dense input in which each bit is on for a large share of the queries. The thermometer code of an LLM embedding is that kind of input. On it, the same machine gives median 6 literals per clause, max 22, and 3,520 of 4,000 clauses with a positive literal (CLINC150, MiniLM, today).
- **Result of R2 (registered, test, 10 seeds; `results/confirm/R2/rules_R2.json`):**
  - Default TM: 81.97. With a literal budget of 32: 87.83. The difference is **+5.86 points [+5.28, +6.45]**, so rule R2.1 is met.
  - Logistic regression on the same bits: 88.09. Budget minus LR is -0.26 [-0.72, +0.20], so the budget TM is non-inferior. It does not beat LR.
  - Out-of-scope AUROC improves from 0.839 to 0.908 (LR on the same bits: 0.902). Calibration error after one temperature drops from 0.043 to 0.018.
  - **R2 as registered is not met:** rule R2.2 (no clause over 32 literals) fails because of a single clause. Seed 7 has one clause of 33 literals among 40,000. The first reading, a leftover of the start state, was most likely wrong; see R2c.
  - The fix has a cost: clauses are short but 97.7% are negation-only, against 24.5% in the default. The budget buys accuracy and bounded length, not keyword rules.
- **Result of R2c (registered, test, Banking77 and HWU64, 10 seeds; budget 32 with a start state that obeys it):** accuracy rule met in both, **+4.71 [+4.00, +5.44]** and **+4.09 [+2.79, +5.41]** points. Length rule not met again: two clauses of 33 literals in one Banking77 seed. **R2c not met.**
  - The cause is a bug in the budget's selection, found then: a tie of two float32 random keys at the k-th place keeps k + 1 literals. Every overshoot (R2, R2c, R7b) is exactly one literal. Fixed (`budget_select="rank"`, `tests/test_budget_exact.py`); re-run R2d registered.
  - The budget lifts the word router by 4 to 6 points on all three datasets, but matches LR on the same bits only on CLINC150 (Banking77 -1.4, HWU64 -5.2).
- **Result of R2e (registered, test; R2 itself again, CLINC150, with the trim start and the exact selection; second use of this test split by a lexical router): met.** B minus A **+5.82 [+5.24, +6.41]** points; no clause over 32 literals; non-inferior to LR on the same bits (-0.30 [-0.76, +0.17]); 97.9% of the clauses negation-only.
- **Result of R2d (registered, test; R2c's arm B again with the exact selection): met.** Banking77 **+4.81 [+4.08, +5.53]**, HWU64 **+3.96 [+2.69, +5.27]** points; no clause over 32 literals in any of the 20 models. With R2e, Part 1's fix, the literal budget, now meets its registered accuracy and length rules on all three datasets.
- **Result of R2b (tmu's budget form):** 56.45, which is -25.5 points. Not met. This machine starts with half of the literals included, and with that start the tmu rule collapses. tmu itself starts with every literal excluded (`tmu/clause_bank/clause_bank.py`, `initialize_clauses`). That combination was not tested here.

### Part 2. Words carry no meaning of their own

- **Failures:** F7, F10, F11, F13, F15.
- **Mechanism:** the lexical router reaches 82.8 against 94.2 for the embedding router (upstream CLINC150 test). Raising s does not close that gap, it makes it worse (F15). A TM over word bits must learn every paraphrase from examples. An LLM encoder already maps paraphrases close together.
- **Fix:** let the LLM encoder supply the semantics, and let the TM decide on top of it.
- **Test R3** (the encoder ladder) is part of R4.

### Part 3. The remaining gap to a linear model: little from the bits, most from the TM's capacity

- **Failures:** P11, plus the 1.4-point gap of Clause-Driven-LLM.
- **Measured today (exploratory; CLINC150, MiniLM, validation):**
  - Logistic regression on the raw embedding: 96.10. On the 2,688 thermometer bits: 95.77. So turning the embedding into bits costs about 0.3 points.
  - The TM on those same bits: 94.6 to 94.9 at 4,000 clauses. That is a learning gap of about 1 point on identical input.
  - On sparse prototype bits, TM and logistic regression are equal: 93.3 against 93.0.
- **What moves it:**
  - Clauses. 4,000: 94.87; 8,000: 95.00; 16,000: 95.63 (T equal to the clauses).
  - A convex read-out: 95.40 with logistic weights on the clause outputs plus all 2,688 input bits, 95.13 on the clause outputs alone. Logistic regression on the bits alone gives 95.77. So the read-out is a weighted vote of TM clauses and single bits, not the TM's own vote.
  - What does not help: clause drop (94.2 to 94.5), a literal budget (94.7 to 94.8), s 4 or 16 (94.4, 93.5).
- **Tests:** R4 (as frozen, at most 8,000 clauses; chosen: 4,000 clauses, 80 epochs) and R4b (chosen: 16,000 clauses, 40 epochs), registered below.
- **Result of R4 (registered, test, 6 cells x 10 seeds):** **not met.**
  - TM-single trails logistic regression in every cell, by 0.56 to 2.41 points: CLINC150 MiniLM 94.29 against 95.64; bge-base 96.13 against 97.02.
  - Ensembles of 5 machines are within 0.5 points of LR in 3 of 6 cells; the rule needs 5.
  - A fitted read-out is within 0.5 points of LR in 5 of 6 cells: signed logistic weights on the TM's clause outputs plus every input bit. It is ahead in one, Banking77 bge-base, +0.45 [+0.02, +0.90].
    - That "ahead" is fragile. It rests on choosing the variant per cell. Chosen per seed it is +0.32 [-0.11, +0.75].
    - The read-out on the clause outputs alone trails LR in all 6 cells, by 0.30 to 1.63 points.
  - So the gap is real at 4,000 clauses. It shrinks with ensembles and with fitted weights that also see the raw bits, but it does not close.
- **Exploration after all registered tests (e24, val, seed 1, not registered):** is the gap about axis alignment? A TM's clauses AND thresholds on single coordinates, so rotating the embedding first (shrinkage-LDA axes, or within-class whitening) could matter to it while LR is rotation-invariant. It does not help: CLINC150 MiniLM 0.9447 / 0.9487 against 0.9457 to 0.9487 for the plain thermometer; HWU64 MiniLM 0.9015 / 0.8978 against 0.9024. On the same LDA bits the TM is ahead of LR on HWU64 (0.9015 against 0.8894) and Banking77 (0.9073 against 0.9053) and 0.6 behind on CLINC150 (0.9447 against 0.951). So the TM learns as well as a linear model from the same bits (also R2e, R7c), with enough clauses (16,000: 0.9563 against 0.9577 on CLINC150 val); what stays is the 0.3 to 0.9 points lost when a continuous embedding becomes bits, which every learner on those bits pays.
- **Result of R4b (16,000 clauses, registered, test): not met.**
  - The single machine improves in every cell, by 0.26 to 0.95 points. It comes within 0.5 points of LR in 3 of 6 cells: Banking77 MiniLM -0.19, Banking77 bge-base -0.07, HWU64 bge-base -0.36.
  - The composites gain nothing more (3 of 6 within 0.5; the rule needs 5).
  - Its out-of-scope detection beats LR in every seed with both encoders (0.966 against 0.962; 0.975 against 0.969).

### Part 4. The vote is not a probability

- **Failures:** F4, F16. Expected calibration error 0.122 against a bar of 0.05. Out-of-scope recall at 95% precision is 0.0.
- **Mechanism:** class sums are clipped integer margins, not log-probabilities.
- **Fix:** one temperature on the class sums, fitted on validation. For out-of-scope detection, use the top class sum, not the margin. Today on validation: AUROC 0.971 with the top sum, against 0.952 with the margin and 0.966 for logistic regression's max softmax.
- **Test R1:** registered below.
- **Result of R1 (registered, test):** **met.**
  - After one temperature, the expected calibration error is 0.013 to 0.025 in all 6 cells. Before it, 0.215 to 0.336.
  - Out-of-scope AUROC with the top class sum: 0.9623 against LR 0.9625 (MiniLM), and 0.9728 against 0.9694 (bge-base, above LR in all 10 seeds).
  - 5-machine composites reach 0.967 and 0.976. The margin, which Clause-Driven-LLM used, gives 0.941 and 0.958.
  - Upstream, with the margin, the gap to LR was -2.2 points (94.0 against 96.2). With the top class sum, the TM ties or beats LR.

### The readable decision layer (R7): readable inputs, rules a person can read

- **Question:** can the decision layer be readable end to end? In R4 every literal was a threshold on an embedding dimension (`emb[217] >= 0.31`). In R7 every literal has a name: `contains "pin"`, `near(pin_change) >= p25` (whitened similarity to the centre of that intent's training queries, at the intent's own 5/25/50/75% levels), `nearest = pin_change`, `pin_change in top 3`. Encoder bge-large.
- **Result of R7 (registered, test, 3 datasets x 10 seeds): met.**
  - TM-readable minus LR on the same readable bits: -0.22, -0.12, -0.29 points (CLINC150, Banking77, HWU64). The TM comes within 0.3 points of a linear model on readable inputs.
  - The cost of readability: composite minus LR on the opaque embedding -0.98, -0.99, -0.88 (within the registered one point, two of them just). CLINC150: 96.5% against 97.5%.
  - The decision is exactly the weighted vote of the firing clauses (recomputed from the saved clauses for 100% of test queries). The printed receipt shows the 3 strongest, which carry 5 to 21% of the vote; post hoc, those 3 alone give the same decision in 99.3 to 99.7% of test queries.
- **What it does not show (post hoc, not registered):**
  - The one-rule router "route to the nearest whitened centre" is as accurate (0.9633, 0.9308, 0.9164), and the TM agrees with it on 99.2 to 99.7% of test queries. The accuracy comes from the encoder and the whitened centres; the TM restates that rule as clauses.
  - Logistic regression on the same bits explains its decisions at least as compactly (at most 5 features in 99.8 to 100% of decisions, the TM 80 to 96%).
  - Out-of-scope detection is weak on this view: 0.916 against 0.974 (LR-raw) and 0.966 (the centre rule's own similarity score).
  - 19 to 26% of the clauses are still long exclusion lists (over 32 literals; Part 1), and dropping them costs 9.6, 3.1 and 0.3 points.
- **Result of R7b (registered, test; R7 plus the Part 1 fix, every clause at most 16 literals):** accuracy and cost rules met in 3 of 3 (TM-readable-16 minus LR-readable -0.00, +0.03, -0.13; composite minus opaque LR -0.80, -0.88, -0.65), out-of-scope AUROC 0.950 against R7's 0.916 (rule met). The length rule fails by three clauses of 17 literals, the same tie as in R2c, so **R7b as registered is not met**; re-run R7c with the exact selection: **met** (all clauses at most 16 literals; level with LR-readable; composite -0.46 to -0.84 against opaque LR; out-of-scope AUROC 0.950). The price of the bound: 74 to 91% of R7b's clauses hold no plain literal (R7: 8 to 23%), and a decision becomes the vote of about 2,700 to 3,300 short exclusion clauses; the 3 strongest carry about 1% of it. Short clauses do not make a readable decision. R7, with its long tail, explains most decisions with one or two short positive rules.

### Part 5. No credit assignment through layers or through time

- **Failures:** F8, F9, F12. A layer that reads another layer's clause bits has no target of its own.
- **Fix in this study:** no stacking. For decisions over time (the agent loop), a TD target gives every step its own target.
- **Exploration on R6 (validation, seed 1; clarify-or-route, CLINC150):**
  - The router is right on 51% of half queries and 95% of full queries.
  - Always routing at once returns -0.010. Always asking first returns 0.742. The best threshold rule, tuned on val itself, returns 0.775.
  - The TM policy trained toward the TD target learns to ask. With the right step size it returns 0.744 to 0.764 (online TD with kappa 0.005: 0.763; fitted-Q iteration 0.748 to 0.764). The chosen run, fitted-Q iteration with 4 x 10 epochs, gives 0.764.
  - With a step 200 times too large (kappa 1) it never asked and returned 0.070. The regression feedback is summed over the batch, so kappa has to be about 1/(batch x firing clauses / T).
  - Its decisions print as rules over the router's confidence (`scripts/demo_agent.py`), for example:
    - `[ASK] +38: IF margin_now < 1201 AND top_now < -790 ...`
    - `[ESCALATE] +6: IF step1 AND top_now < 2483`
- **Tests:** R5 (one step, reward only) and R6 (several steps, delayed reward).
- **Result of R6 (registered, test, 10 seeds):**
  - The TD-trained TM policy returns 0.597 against 0.198 for the same TM trained on the immediate reward alone. That is **+0.399 [+0.392, +0.405]**: credit assignment back in time works. R6.1 is met.
  - That first contrast is nearly built in: a myopic learner can never value ASK.
  - It also beats the best threshold rule tuned on val, 0.556, by **+0.042 [+0.035, +0.049]**. R6.2 is met, so **R6 is met.** But that whole gain comes from out-of-scope episodes, where it is +0.29.
    - In scope the rule is better: -0.014 [-0.020, -0.010].
    - Reweighted to val's out-of-scope share of 3.2%, TM-Q minus rule is -0.0045 [-0.0093, -0.0001] (computed afterwards, not registered).
    - Two things differ from the rule. The TD learners trained on a 15% out-of-scope share, close to test's 18% (val has 3%). And they see the router's top class sum as well as its margin, while the rule sees only the margin.
  - It equals a linear Q-learner on the same episodes and reward (online TD rather than fitted-Q): +0.001 [-0.004, +0.005].
  - The memory literals change nothing on two steps (+0.001).
  - In scope, TM-Q returns what "always ask, then route" returns (-0.002). Its added value is escalating the out-of-scope queries after asking.
  - Every decision prints with the clause behind it (an example output for seed 9 on test is in the project notes, not published).
- **Result of R8 (registered, test, 5 seeds; four steps: the query revealed in 25/50/75/100% prefixes): met.**
  - TM-Q minus TM-myopic +0.369 [+0.356, +0.383]. Nearly built in, as in R6: even always-ask-then-route, which learns nothing, beats the myopic TM by +0.205. What the TD targets do: the TM keeps asking through the prefixes and escalates at the end.
  - TM-Q minus always-ask-then-route +0.164 [+0.152, +0.176], all from escalating out-of-scope queries (86% of them) instead of routing them. In scope it equals always asking (+0.006 [-0.003, +0.014]); the rule and Lin-Q gain +0.036 there.
  - Not claimed, and not true: the TM is 0.022 below the tuned rule and 0.021 below linear Q; memory literals cost 0.012.
- **Result of R5 (registered, test, 10 seeds):** **not met.**
  - With 10 actions (CLINC150 domains), the TM policy learned from reward alone has the best final accuracy: 0.958 against epsilon-greedy logistic 0.950 (+0.77 [+0.54, +0.99] points) and LinUCB 0.949 (+0.88).
    - Its online accuracy is far lower than LinUCB's (0.858 against 0.948), so it pays more regret on the way.
    - It equals its supervised twin (1.004 of it), but with replay it made 5 times the updates of that twin.
  - With 18 actions it is 7.0 points behind LinUCB, and with 77 actions 42.8 points behind.
  - The reading, partly tested: from a 0 reward the TM gets negative feedback for one action only, and top-3 epsilon-greedy exploration rarely finds the right one of 77.
    - Exploratory runs (val, seed 1, not registered) support the exploration part.
      - Banking77: Boltzmann exploration on the class sums (tau 0.5) reaches 0.725 in 5 passes, against about 0.50 for the registered top-3 setting (on test). LinUCB on val: 0.921.
      - HWU64 scenarios: 0.935 with Boltzmann exploration, against 0.856 for the registered setting on test.
    - The 77-action gap to LinUCB is still about 20 points after 5 passes, and the TM is still improving then. With 15 passes the same setting reaches 0.904 (LinUCB 0.921 after 5).
    - **Result of R5b** (registered 06:03:20Z, test, 5 seeds): **not met**, but most of the failure is repaired. Boltzmann exploration, 15 passes:
      - HWU64: 0.931 against LinUCB 0.915, **+1.69 [+1.23, +2.03]**. The TM is now ahead.
      - Banking77: 0.893 against 0.925, -3.16 [-3.52, -2.83], against -42.8 in R5.
      - Both reach at least 0.97 of the supervised TM. LinUCB still makes fewer errors online.
    - **Result of R5d** (registered 15:05:29Z, test, 5 seeds; R5b's Banking77 task, 77 actions, with 30 passes): **met.** TM bandit 0.9132, LinUCB 0.9204: -0.71 [-0.96, -0.33] points; 0.996 of the same TM trained on labels. Third use of this test split by a bandit. R5d's first 15 passes replay R5b's runs exactly; with 30 passes for both, the TM gains 2.0 points and LinUCB loses 0.4. Against LinUCB after 15 passes it would be 1.14 behind; LinUCB is ahead at every pass.
    - **Result of R5c** (registered 11:07:45Z, test, 5 seeds; 150 actions, CLINC150 intents, 30 passes): **met.** The TM bandit reaches 0.9365, 0.998 of the same TM trained on labels (0.9386), and comes just within one point of LinUCB (-0.98 [-1.20, -0.80]; 0.9463; 2 of 5 seeds more than a point behind). Online it is right 78% of the time, LinUCB 94%. LinUCB peaks after pass 13; the TM is still rising at pass 30.

### Part 6. Speed and scale

- **Failure:** F17.
- **Measured today:** with a small action space the GPU machine is fast. One CLINC150 training run (4,000 clauses, 40 epochs) takes 34 s on the V100. The vocabulary-scale cost of F17 does not arise for a router.

## Phase 2 (2026-10-16 21:31 UTC to 2026-10-17): new data and the parts of the plan never tested

Up to here every registered router test ran on CLINC150, Banking77 and HWU64, each test split used three times or more, and two parts of the plan had never been tested where they matter: memory literals ("rules over sequences") and a decision that combines intent with the agent's history. Phase 2 uses data that no phase-1 test had read (within phase 2 the ABCD test split is used by ten registered tests in this order: R9, R9b, R10, R10c, R13, R14, R15, R16, R17, R16b, and the bandit runs also score it after every pass for their curves; the en-US test split is used by R11 and as R12's reference): ABCD (10,042 customer-service conversations in which agents follow written guidelines and press 30 action buttons; Chen et al., NAACL 2021), MASSIVE 1.1 (en-US and its Norwegian translation) and STAR (6,652 wizard-of-oz dialogues over 24 tasks with written flowcharts, 187 wizard actions; Mosig, Mehri and Kober 2020; its test split used by R18, and again by R19 in phase 3). Results in `docs/results.md`, sections R9 to R18.

### Part 7. Rules over sequences: memory literals where the task needs them (R9, R9b, R18)

- **Mechanism.** In ABCD the next button depends on what the agent has already done in the conversation (pull up the account, then verify the identity or validate the purchase, depending on the subflow). R6 and R8 had nothing for memory to do; here it carries the task.
- **R9b (registered, test): met.** The TM with the history as named literals (`done(b)`, `last action = b`: the buttons the human agent has pressed so far in the logged conversation) predicts the next button in **84.0%** of test action turns; the same TM without them **65.4%**: **+18.65 [+17.20, +20.10] points.** Some strong clauses are the plan's "if A and not B, choose action 3": "[validate-purchase] IF last action = pull-up-account AND NOT (verify-identity is the nearest next step) AND NOT ..." (4 literals); others are one plain literal plus a long exclusion list, e.g. "[enter-details] IF done(pull-up-account) AND NOT done(record-reason) AND ..." (75 literals, 74 negated). The 200 strongest clauses (ranked by their largest class weight) hold 27.5 literals on average (seed 1).
- **R18 (registered, test, a second dataset): met.** On STAR (wizard-of-oz dialogues over 24 tasks, 187 actions; a test split never read), memory literals lift the TM from 47.2% to **62.7%: +15.43 [+14.38, +16.51] points** (open loop: the logged actions as memory). The memory result replicates. The control without memory still reads the wizard's last message, which names the last action in 60% of later decisions, so the gain understates the history's value (internal audit 8). The TM's parity with linear and neural heads does not replicate: on STAR it is 3.3 points behind LR and 3.1 behind an MLP on the same bits; post hoc on dev the gap sits in the less common actions.
- **R9 (registered, test): not met as a whole,** because R9.1, its memory rule, was badly built: its control read the earlier presses as words in the conversation text (+1.09 points only). A design error, found when the first seeds came in; R9b repeats the test with a clean control.
- **The decision layer against linear and neural heads (R9, same task, 2,791 readable bits):** TM 84.8%, LR on the same bits 84.5% (R9.2 met, +0.36 [-0.29, +1.02]), MLP on the same bits 84.5% (R9.3 met, +0.31), **TM composite (5 machines) 85.4% against the opaque MLP on the raw embeddings 85.2% (R9.4 met, +0.10 [-0.62, +0.82]).** LR on bits chose the smallest C of its grid in all four ABCD arms, so it may be slightly under-tuned (internal audit 6). The registered test that the TM beats LR (R9.5) is not met: ahead, interval through 0. Fine-tuned BERT and RoBERTa (published) reach 89.9% and 93.6% on the weighted metric; the frozen-encoder layer 85.8 to 86.2%.

### Part 8. The workflow from reward alone (R10, R10c, R14, R15)

- **R10 (registered, test): met.** With only a 0/1 reward per button press (Type I/II feedback, Boltzmann exploration) and inputs built without the label of the decision being learned (memory bits = the human agent's logged earlier presses; words; unsupervised clusters), the TM bandit predicts the next action in **83.0%** of test action turns, **0.988 of the same TM trained on labels** (84.0%). Without the memory bits: 60.1%, so memory is worth **+22.87 [+21.37, +24.39] points** inside the agent loop.
- **R10c (registered, test): met.** On the same bits and stream, after the 15 registered passes, **the TM bandit ends above LinUCB**: +5.51 [+4.66, +6.39] points at the dev setting (alpha 0.5) and +3.00 [+2.18, +3.81] against the best of the three registered alphas, chosen on test (alpha 1.0, 80.0%). It is the first registered test in which the TM is asked to beat a linear learner and does (R9.5 was not met). What it does not show: LinUCB is ahead in test accuracy after each of the first 7 to 11 passes, peaks after 3 to 5 passes and then falls while its accuracy on the training stream climbs to 0.90 (post hoc reading: 15 replays over-fit it at ridge 1, which was not varied); it earns more reward while learning (online 76 to 87% against 72%); and its end accuracy was still rising with alpha at the top of the range tried: on dev, after the test (exploration, one seed per alpha), alpha 4.0 ends at 83.0% against the TM's 83.5%, with 46% online. Even LinUCB's best single pass on test (82.0%, chosen on test) is a point below the TM's last pass.
- **R14 (registered, test): met.** Against a small neural bandit on the same bits and stream (one hidden layer, one AdamW step per round, the best of four one-seed dev settings), the TM bandit ends **-0.45 [-1.04, +0.16] points** behind: within the registered margin of -1.0 on the point estimate, though the interval reaches just past it. The neural agent learns faster (online 80% against 72%).
- **R15 (registered, test): met.** The TM's slow learning has a partial remedy that needs no new information: replay (4 extra updates per round on its own logged presses and rewards). Online accuracy +6.35 [+5.87, +6.81] points (78.2%), the gap to the neural agent down from 8.3 to 1.9 points (it stays below it), and at the end +0.52 [+0.03, +1.02] against the neural bandit (83.97% against 83.45%; R15.2 is a non-inferiority rule; with the 5 training seeds also resampled the interval is [-0.26, +1.28], post hoc). The neural bandit had no replay (one update per round): R15 shows that replay closes most of the TM's gap, not that the TM learns as fast as a network given the same extra updates. The price: 2.3 times the run time; the network is 30 times faster in wall-clock time.
- Limits: it replays logged conversations, so its memory holds the human agent's earlier presses, not its own (Part 10 tests the closed loop: R13 for the supervised TM, R17 for the bandit); online it is right 72% of the time while it learns (78% with replay, R15); the clauses on these bits are long exclusion lists (median 106 literals).

### Part 9. A related dataset and another language (R11, R12)

- **R11 (MASSIVE en-US, frozen before any model touched MASSIVE): R11.2 met, R11.1 and R11.3 not met.** The bounded readable layer (R7c's setting, clauses of at most 16 literals) passes R7c's rules again, with less room: 0.35 behind LR on the same named inputs ([-0.65, -0.06]; on R7c's datasets it was level), its composite 0.44 behind the opaque model ([-1.31, +0.45]). The rules judge point estimates; neither interval lies inside its margin. R7's unbounded setting is 0.65 behind LR on the same inputs (rule: 0.5) and its calibration error after one temperature is 0.060 (rule: 0.05).
- **R12 (trained on English only, scored on Norwegian): R12.1 and R12.3 met, R12.2 not met.** The layer routes Norwegian requests at **82.2%** (English 86.6%: it keeps 0.949), **ahead of LR on the same bits (+0.72 [+0.30, +1.16]; R12.1 is a non-inferiority rule, its interval also lies above 0)**, but 1.8 points below an opaque LR on the same multilingual encoder (rule: 1.0). In English the same gap is 1.66 points, so most of the cost comes with the readable view on bge-m3 rather than with the language (post hoc).

### Part 10. The closed loop: the memory holds the agent's own presses (R13, R17)

- **R13 (registered, test; R9b's models, no training): met.** When the memory bits come from the TM's own earlier presses instead of the logged ones, it still gets **80.7%** of the next actions right, **+15.34 [+13.80, +16.91] points** over the TM without memory; its own mistakes cost 3.31 points (rule: at most 4). Logistic regression loses the same and is level with it in closed loop.
- Limits: the loss grows along the conversation; from the 6th action on (126 turns in 61 conversations) the closed-loop TM is no better than the TM without memory (0.415 against 0.437; -2.2 [-6.6, +3.0], post hoc). Two standard remedies (noisy memory in training, one round of data aggregation) did not help in one dev run each. The text is replayed from the log, including the human agent's own utterances (93% of later action turns have one among the last 3 utterances, e.g. "i have pulled up your account"); only the memory bits are the model's own, so a fully closed loop probably costs more than 3.3 points. A customer who reacts to a wrong press was not simulated.

- **R17 (registered, test): met.** The same for the agent that learned from reward (5 new runs of R10's setting, models saved; in training its memory bits were the human agent's logged presses, as in R10): with its own presses as memory **79.1%**, **+18.96 [+17.30, +20.61] points** over R10's agent without memory; its own mistakes cost 3.70 points (rule: at most 4, on the point estimate; interval [-4.44, -2.99]). In closed loop it is 1.60 [-2.46, -0.76] points below the supervised TM. As in R13, only the memory bits are the agent's own; the text, including the human agent's utterances, is replayed from the log, so a fully closed loop probably costs more than 3.7 points.

### Part 11. Rules through layers: Granmo's hierarchical TM (R16, R16b)

- **Mechanism.** In Granmo's hierarchical TM a clause is a tree: here, per block of the readable view (memory, intent, next step, words), 8 options, each a conjunction of that block's literals, and the blocks combined. One clause can then express "(last action = A OR last action = B) AND (subflow X OR subflow Y) AND ...", which a flat clause needs several clauses for. In the package the OR is computed as the number of options that hold and the AND as a product, so a clause votes with the product of the per-block counts (0 to 4,096), not with 0 or 1.
- **R16 (registered, test): met.** Against the flat machine of the same package, the hierarchical one gets **81.7% against 72.3%: +9.40 [+8.35, +10.41] points.** Inside one implementation the hierarchical configuration beats the flat one by 9.4 points. It also has 2 times the clauses, 16 times the automata, other T and s and graded votes, so the share due to the AND-OR structure is not measured.
- **R16b (registered, test): met.** A flat machine of the same package with as many automata (3,200 clauses per button) gets 74.7%: the hierarchical one stays **+7.02 [+6.01, +8.04] points** ahead. At the one flat setting tried (T equal to the clauses, s 4, 30 epochs; on dev still rising at epoch 30), size explains about a quarter of R16's 9.4 points; T, s and the graded vote still differ.
- Limits: it is 2.3 points below this study's coalesced flat TM on the same bits (84.0%), and much slower (dev: 1,999 s of training for 40 epochs, about 13 to 17 times the coalesced TM's 116 s, plus 2,298 s of dev scoring); the package's flat machine is 11.7 points below the coalesced one, so implementation and settings matter more than the hierarchy here (post hoc). A coalesced hierarchical machine was not tried. The first hierarchy shape tried did not learn.

## Phase 3 (2026-10-17 evening to 2026-10-19): the agent with no neural network at run time

The question: can the whole agent run with no neural network at run time, once the word table has been checked? R19 is the second use of STAR's test split (after R18) and the eleventh use of ABCD's. A post-hoc LR refit (no rule) read STAR's test split a third time, and R20 a fourth time (ABCD's twelfth). Results are in `docs/results.md`, sections R19 and R20.

### Part 12. A transformer per decision: the text encoder (R19)

- **Failure.** Up to R18 every decision of the ABCD and STAR agent (R9 to R18) ran bge-large-en-v1.5 (335M parameters) over the conversation text. Only the decision itself was logic based.
- **Fix tested.** No neural network at run time: a rule-based tokenizer, lookups in a fixed 63,091 x 512 table, then a bitwise OR of precomputed per-token bit codes, then Boolean clauses and an integer vote. The table was distilled once, offline, from a transformer (bge-base-en-v1.5 via Model2Vec); the reference agent runs bge-large-en-v1.5 (335M parameters) per decision. Speeds are rough.
  - The text bits read "some word of the customer's words so far (or of the last 3 turns) is close to concept k at level j". There are 512 concepts per field, each named by its 4 nearest tokens.
  - The view was chosen among three on STAR dev. All dev numbers are in `docs/results.md`.
- **R19 (registered, test, STAR and ABCD): met, in the strong form on both.**
  - STAR: level with the bge-large agent, **+0.06 [-0.63, +0.74]** points (62.7% against 62.7%).
  - ABCD: **1.0 point below it, inside the registered 2-point margin: -1.00 [-1.62, -0.40]** (83.0% against 84.0%).
  - The memory literals still pay: +15.05 [+14.07, +16.05] and +20.79 [+19.32, +22.29].
- **No rule:**
  - Without memory, the table costs 3.14 points on ABCD (-3.14 [-3.95, -2.31], post hoc; on STAR +0.44). Probably the table reads ABCD's customer task less well: a dev probe on its mean vectors showed -9.5 points, and view B's bits were not probed. Memory hides most of it (-1.00 with memory).
  - On the table's bits the TM is 2.22 [-3.02, -1.42] points behind a tuned LR on STAR. This is post hoc: LR refit at its dev-best C (0.01), one test read. With bge-large the TM is 3.26 behind LR. On ABCD the TM is 2.38 ahead of LR. ABCD's dev curve peaks at C 0.03, 0.11 above C 0.1, so its LR was close to tuned; that difference carries no rule.
  - On the ABCD paper's own B-Slot measure (30 buttons, scored on its AST test examples) the word-table agent scores 84.8%. That is 1.9 points below the paper's pipeline BERT (86.7) and 5.1 to 8.8 below its fine-tuned BERT, AlBERT and RoBERTa (89.9 to 93.6).
  - The clauses are shorter: on STAR the median is 5 to 8 literals against 27 to 32 (descriptive only).
  - One decision on one CPU thread (rough: shared machine): median 29.5 ms, against 806 ms for the bge-large agent, about 27 times faster. Both agents were run on one CPU thread; the bge-large agent would be faster on a GPU (not measured), which the word-table agent does not need. The text step's speed-up (779 ms to 0.49 ms) is Model2Vec's. The TM decision (28.7 ms in numpy) is now the bottleneck.
- **Limits.**
  - The table was distilled from a transformer, once and offline. Only the run time is free of neural networks.
  - The speed-up of the text step is Model2Vec's.
  - Open loop on both datasets.
  - Training takes about 2 times as long, with 3 to 5 times the bits.

### Part 13. The closed loop, and an agent that reads only the customer (R20)

- **Failure 1 (a known one, now on STAR): the agent's own mistakes compound.** In the closed loop the memory bits hold the agent's own presses, so one wrong press misleads the next decisions.
- **R20 (registered, test on STAR; R19's and R18's models, no training):**
  - **R20.1, memory still pays in the closed loop: met.** **+7.53 [+6.51, +8.58]** points (55.2% against 47.7% without memory).
  - **R20.2, its own mistakes cost at most 4 points (R13.2's bar): not met, as expected.** **-7.51 [-8.34, -6.72]** points (55.2% closed loop against 62.7% open loop; dev gave -7.57; the bar was kept). On ABCD the same agent pays -3.40 [-4.02, -2.80] (no rule): So the compounding is about twice as costly on STAR (187 actions against ABCD's 30; the cause is not tested).
  - **R20.3, the word table against bge-large in the same loop: met, strong.** **+1.96 [+1.07, +2.85]**. In the closed loop the word-table agent is not worse than the bge-large agent; the interval lies above 0, but superiority was not a registered rule. Descriptive: its own mistakes cost less in the closed loop (-7.51 against R18's -9.42 [-10.37, -8.49]); open loop the two were level (R19.1 +0.06).
- **Failure 2 (new, dev only): without the wizard's text, a wrong press poisons the memory (STAR).** R20.4 was not taken to STAR test. Arm 2 failed its dev gate S4 (+4.50 against +5.0, seed 1), so by the frozen plan it is a dev-only failure-map line.
  - The agent that reads only the customer (view C: the customer's words and its own presses, no wizard text): the memory is worth +14.6 points open loop and +4.5 in the closed loop, and its own mistakes cost 10.1 points (dev: 0.5779 open, 0.4767 closed, 0.4317 without memory).
  - The twin's best-dev number is optimistic (its dev curve swings from 0.362 to 0.432), and S4 was applied as written.
- **R20.4 (registered, test, ABCD, supporting evidence only): met.** The customer-only agent in the closed loop, with no neural network at run time and no text from the human agent: **72.6%**, memory **+23.94 [+22.20, +25.67]** points over its twin without memory (48.7%).
  - No rule: dropping the human agent's text: -6.98 [-8.13, -5.84] points in the closed loop (72.6% against 79.6% for R19's agent); its own mistakes cost 5.46; level with LR on the same bits in the closed loop (+0.54 [-0.77, +1.82]); 8.07 below R13's bge-large agent that reads the human agent's text (80.7%).
  - Limit: the claim is "no text written by the human agent", not "no information about the human agent's actions". The customer's logged replies answer the human agent's real actions.
- **Limits.** Offline closed loop: the text (including the customer's replies) is replayed from the log, and every press is scored against the logged next button, which assumes the human's history. A customer who reacts to a wrong press is not simulated.

### Part 14. "A policy that can be verified directly": proofs against the written guidelines (e70, dev; frozen plan, not registered)

- **Failure (dev):** over a superset of all real inputs (only the constraints that real inputs satisfy; the text bits otherwise free), none of 50 rules taken from ABCD's written guidelines holds for the raw TM. All 250 checks (5 seeds) give a counterexample, in 0.5 to 6 s each. The counterexamples are far from any real conversation: their nearest dev row is about 1,500 text bits away. The superset is too loose for a proof about the raw policy, because a free text can be built to vote for any button.
- **Fix tested (dev):** 50 veto clauses, one per rule, of which 15 to 20 change a decision on dev. All 50 rules then hold by construction, and the solver confirms it. Cost on dev: -0.27 to -0.43 points open loop, every right-to-wrong turn being a turn where the human agent broke the rule; -0.08 to -0.27 closed loop.
- **Measured, no claim:** with the same generic MILP encoding and solver (600 s per check), the TM's 250 checks all resolved, with a median of 1.3 s. MLPs on the same bits resolved 41 of 50 checks (64 hidden units, median 151 s) and 1 of 50 (128 hidden units); the 512-unit twin was not run (budget). No dedicated neural-network verifier was tried.
- **Next:** local proofs around real conversations, in word edits (e71; plan `logs/plan_e71_local.md`).
- Prior art: TM verification by SAT (Przybysz et al. 2023); verify and repair with clauses (VERITAS, 2026). This is not a first TM verification.

### Part 15. The word table was distilled from a transformer (e72 dev, R21 test, e72d dev)

- **Failure.** R19's agent has no neural network at run time, but its word table (potion-base-32M) was distilled once, offline, from a transformer. So a neural network was still used to build the agent.
- **Fix tested (dev):** the C4 word table (the W1 study's registered W1-T table), made with no neural network at any stage: C4 text, a C4 word list, counting + SVD. The agent is otherwise R19's: k-means + thresholds on the train texts, then a Tsetlin machine. A word not in the table is dropped (0.5 to 0.9% of the tokens).
- **e72 (dev, seed 1, R19's settings; gate: within 2.0 points of potion's row): passed on both datasets.**
  - ABCD: 0.8317 against 0.8325, -0.08.
  - STAR: 0.6317 against 0.6343, -0.26.
  - Without memory (ABCD, no rule) the table costs 0.92 points (0.6107 against 0.6200).
- **How it was checked:** all code is in new files, and the frozen run scripts are unedited. In such a run potion's table cannot load and no encoder vector is loaded. Given potion's own table, the new code reproduces R19's bits exactly (`tests/test_count_text.py`).
- **R21 (registered, test; STAR's 5th use of its test split, ABCD's 13th): met on both datasets, R21.1 in the strong form on both.**
  - C4 agent minus the potion agent: STAR **-0.10 [-0.65, +0.45]** (62.6% against 62.7%), ABCD **-0.29 [-0.74, +0.17]** (82.7% against 83.0%).
  - The memory literals still add +14.73 and +20.63 points.
  - No rule:
    - against the bge-large agent: -0.04 (STAR) and -1.29 [-1.94, -0.66] (ABCD);
    - on the same bits LR is 2.79 points ahead on STAR, and the TM is 1.76 ahead on ABCD.
  - e72d (dev): with e70's 50 veto clauses, all 50 order rules hold for all 5 ABCD models over the superset of real inputs (250 of 250 proved; raw, 250 of 250 counterexamples). Cost: -0.30 to -0.49 points open loop.
  - Rough speed on one CPU thread: 2.6 ms (STAR) and 0.5 ms (ABCD) per decision.
  - Limit: view B reads the last 3 turns, including the human agent's own words, which often announce the next action. The comparisons with potion and bge-large use the same view, so they are fair, but the absolute accuracies depend on it. R20's customer-only view cost about 4.5 points open loop on ABCD dev.
- **Wording (R21 met):** "no neural network at any stage: C4 text, a C4 word list, counting + SVD, k-means + thresholds, then a Tsetlin machine". It is never used for the potion results.

### Part 16. The agent writes none of its replies (e73, dev; frozen note, no rule)

- **Failure.** Up to R21 the agent picks a button or a label and says nothing itself.
- **Step tested (STAR dev, R21's C4 models):** the agent produces its replies from the task's templates. It picks the template (its label) and fills the slots from the latest database result, with a slot mapping derived from train beforehand.
- **Result (dev, 5 seeds):**
  - its reply equals the wizard's word for word in 45.5% of the template turns open loop (44.3 to 46.7) and 40.0% closed loop;
  - filling the logged template gives 79.6% (the ceiling), so the agent reaches 0.57 of the ceiling, about its label accuracy on those turns;
  - 12.9% of the wizard's replies are free text and cannot be matched.
- **Wording:** "produces its replies from the task's templates", never "generates language". It does not compose sentences.

### Part 17. The TM's s was never tuned (e74, e74c, e74d; dev; test next as R22)

- **Failure.** Every agent run up to R21 used s 8, and on STAR the TM stayed 2.79 points behind LR on the same bits (R21, test).
- **Fix tested (dev, frozen plans):** a screen of s on STAR dev for the C4 agent (2 to 24) and, for fairness, for the bge-large agent (8 to 24).
- **Result:** both agents peak at s 16.
  - The C4 agent at s 16 gains +2.28 points on STAR dev (0.6545 against 0.6317) and +1.00 on ABCD dev.
  - It is 0.13 points from a tuned LR on the same bits on STAR dev.
  - Under equal tuning it is +0.38 against the bge-large agent.
- **R22 (registered, test; STAR's 6th use): all three rules met, two in the strong form.**
  - The C4 agent at s 16 against s 8: **+2.70 [+2.17, +3.24]** (65.3% against 62.6%).
  - Against a tuned LR on the same bits: **-0.09 [-0.90, +0.73]**, so the TM is level with LR (R22.2, strong).
  - Against the bge-large agent tuned the same way: **+1.64 [+0.94, +2.35]** (R22.3, strong; not worse; superiority was not a registered rule).
  - Part 3's STAR gap (the TM behind LR) came from s.
- **R23 (registered, ABCD test, 14th use): met, strong.** Each agent at its own ABCD dev pick (C4 s 16, bge-large s 24), C4 minus bge-large: **-0.92 [-1.61, -0.26]**. This is within the 2-point margin, but on ABCD the tuned bge-large agent is about 1 point ahead.
  - Equal tuning on both datasets: STAR +1.64, ABCD -0.92.
  - e72e (dev): with the 50 veto clauses, all 50 order rules hold for R23's five tuned ABCD models (250 of 250 proved). The cost is -0.46 to -0.60 points open loop.

## Phase 4 (2026-10-19 evening to 2026-10-23): a third dataset, MultiWOZ 2.2

### Part 18. A third dataset, MultiWOZ 2.2: the agent with no neural network at any stage falls behind (e77, dev only; R24 ran later, see the update below)

- **Failure (dev, one seed, no rule).** On MultiWOZ dev, with s tuned for both agents by the same rule, the C4 agent is
  **3.21 points below** the bge-large agent at 16k (0.3038 against 0.3359; majority 7.16%).
  - The gate for R24 (at least -1.0, fixed before any 16k number) failed. MultiWOZ's test split stays
    unread, and nothing is claimed on test.
  - LR on the same C4 bits reaches 0.3586 on dev, 5.5 points above the C4 TM (LR at the edge of its grid, C 0.001;
    checked by e81x: C 0.001 is LR's best over C 1e-4 to 100). So the reader, not only the text bits, loses here (P5
    of the text-stack map in the project notes, not published).
- **Where:**
  - the first action -4.0;
  - the domain of the next act -2.4;
  - the act -2.6;
  - Attraction, Taxi, Restaurant and Train turns are 6.5 to 9.4 points behind.
  - Part of the exact-set target is a choice between near-equal variants ("general-bye" against
    "general-bye+general-welcome"; "X" against "X+general-reqmore") that the two agents resolve differently.
- **Fixes tried (dev, bars fixed before their runs; both not taken):**
  - e78, concept counts beside xcust (2d-a): STAR -0.78, ABCD -0.29; MultiWOZ +0.87 (one seed, no rule).
  - e79, intent and step prototype parts from C4 mean vectors (view A+B on the C4 table): MultiWOZ +0.95 against a bar
    of +1.0, so it fails narrowly; STAR +0.14 and ABCD +0.22.
  - R24 returns only if a later stack passes the same -1.0 gate on fresh dev runs.
- **Update 2026-10-23: on test, with the negative-class rule, the network-free agents are within the registered margin (R24, registered, MultiWOZ test's 1st use).** With
  random negatives for all three agents (Part 21) and each agent's s by e77's rule (e85: s 8 for all):
  - R24.1 (non-inferiority at -1.0, the headline): the C4 agent minus the bge-large agent +0.53 [-0.18, +1.23], met,
    strong; the refined agent +0.05 [-0.65, +0.72], met, strong. R24.2 (superiority) is not met for either.
  - The gate was met on the 5th stack tried at its setting (e77b -3.21, e78 -2.34, e79 -1.90, e82 -2.36, then e84 with
    random negatives). The rule was chosen on MultiWOZ dev, on the refined agent, and it costs STAR and ABCD on dev.
  - The reader's gap remains: the TM is below LR on the same bits for all three agents (Part 21).

### Part 19. A better word table does not move the agent (e82, dev only; no registered test by its rule)

- **What was tried.** The teacher-refined table (step 3a of the text-understanding line: the C4 word list refitted to bge-base-en-v1.5's sentence
  embeddings; network-free at run time, teacher at training time), in the agent's view B, at the agent's tuned s.
- **Result (dev, mean of seeds 1-2, best-dev):** STAR +0.14, ABCD +0.56, MultiWOZ +0.49 over the C4 counting table. No
  loss (R4 met), but at the final epoch MultiWOZ is -0.02, and every difference is inside seed noise.
- **The go-bars for a registered test failed** (STAR +1.0, ABCD +0.7), and so did the MultiWOZ gate (-2.36 against
  -1.0). No test split was read.
- **Why it matters:** the table that matches potion on intents does not change the agent's next-action decisions much.
  The agent's limit lies in the reader (TM about 5 points below LR on MultiWOZ's bits; LR's C 0.001 checked by e81x
  and kept as its best) and in information the view lacks (system results, P9), not in word meanings.

### Part 20. The reader's gap on MultiWOZ is neither these T values nor LR's tuning (e81, e81x; dev only; no rule met)

- **What was tried.** The TM's vote threshold T at clauses/2 and clauses/4, at each agent's s tuned at T = clauses.
  Until then T equalled the clauses in every agent run except one (e47).
  - Three agents were run on STAR, ABCD and MultiWOZ dev.
  - The plan was frozen before any number.
  - A pick got a 16k follow-up with 2 seeds.
- **Result.**
  - The screen kept T = clauses in 8 of 9 cells. The one pick (bge-large, STAR, +1.04 at 4k) fell to +0.14 at 16k.
  - "T helps the reader" (+0.5) is met for neither agent with a rule.
  - The refined agent's TM - LR on MultiWOZ stays at -5.35, against a bar of -2.68.
  - T's effect depends on s. On the same cell, T 1,000 gave -0.71 at s 8 (e47) and +1.04 at s 16 (e81). A joint
    (T, s) search was not run.
- **LR's grid edge, checked and resolved (e81x):** C 0.001 is LR's best on MultiWOZ dev over C 1e-4 to 100, for both
  bit sets. The gap is real.
- **What the rows show** (descriptive, no rule; the curves are in the project notes, not published):
  - on MultiWOZ the TM's dev curve is flat after epoch 4, while STAR's and ABCD's keep rising;
  - its clauses stay at about 2 literals whatever s is (s 8 to 24);
  - on the count-table bits, 465 to 1,027 of 4,000 clauses hold no positive literal.
  - Ruled out on dev: capacity (4k is level with 16k), T at these values, s (8 to 24), and LR's tuning.
- **Next (not run):** knobs that act on the learning dynamics:
  - the negative-sampling rule. It is "hardest" in every run, while near-duplicate labels are the main losses, such as
    "general-bye" against "general-bye+general-welcome";
  - clause dropout;
  - the literal budget.

### Part 21. The negative-class rule: random negatives lift the MultiWOZ reader but cost STAR and ABCD (e83, dev only)

- **What was tried.** Three arms on MultiWOZ dev, at 4k with seed 2, under a plan frozen before any number:
  - the negative class drawn uniformly at random, instead of the "hardest" competitor used in every run so far;
  - a 50/50 mix of the two;
  - clause dropout 0.5.
  - The literal-budget arm could not run: the exact budget is unsupported in tm.py's chunked update. A note frozen
    before any row was read dropped it.
- **Result** (random negatives; 16k, 2 seeds):
  - C1 met: +4.23 at best-dev, +5.98 on the last-10 mean.
  - C2 met: TM minus LR on the same bits -1.13, from -5.35.
  - C3 not met: STAR -11.91, ABCD -4.00.
  - The near-duplicate prediction P1 was not met (+0.80): the lift spreads over rarer labels.
- **Why it matters:** a large part of the MultiWOZ failure was the training rule's choice of negative class, not the
  TM's clauses or the bits. But the rule that fixes MultiWOZ breaks STAR and ABCD, so it has to be chosen per dataset
  on dev, like s.
- **Verification:** the independent recompute matched; the three-part review of e83 and e84 (leakage, selection and registration, integrity; judged by an AI model, not a person) ran later on 2026-10-22 and found no blocker.
- **e84 (dev, frozen plan): the same rule for every agent.** With random negatives on MultiWOZ (16k, 2 seeds):
  - refined 0.3530, bge-large 0.3486, C4 0.3500;
  - both R24 gates are met, steady: refined minus bge-large +0.45, C4 minus bge-large +0.14 (last-10: +0.77, +0.71).
  - Part 18's failure (the network-free agents 2.4 to 3.2 points behind) came from the negative-class rule.
  - An R24 draft may follow; MultiWOZ's test split is still unread.
- **e84b (dev, frozen note): seeds 3-4.** Mean of seeds 1-4 at best-dev: refined 0.3534, bge-large 0.3460, C4 0.3484;
  both gates hold (+0.73, +0.23). e85 (dev, frozen plan): no s beats s 8 by +0.3 at 4k under random negatives, so s 8
  for all three agents.
- **R24 (test, registered, 1st use; 2026-10-23):** the network-free agents are level with the bge-large agent
  (non-inferiority at -1.0 met, strong, for both; superiority not met; Part 18).
- **Still a failure on test (no rule): the TM is below LR on the same bits for all three agents.** C4 -1.55 [-2.32,
  -0.79], refined -2.49 [-3.33, -1.64], bge-large -1.89 [-2.69, -1.10] (LR's C at the grid's edge for the C4 and
  refined bits; e81x found C 0.001 LR's best on dev). Random negatives narrowed the dev gap from about -5 to about -1,
  but on test it is 1.6 to 2.5 points.
  - Next (e86, dev, frozen 2026-10-23T00:58:18Z): clause dropout 0.5 on top, for the C4 agent on all three
    datasets, and random negatives with dropout as one setting for all three. It was stopped after 8 of its 14 dev
    runs, before its analysis (docs/results.md, e86 and e87).

## Registered tests

Written before the confirmatory runs; see `registrations/` for the frozen versions.

| Test | Question | Frozen (UTC) | Amendments |
|---|---|---|---|
| R2 | sparse-input fix on the word router, exact literal budget | 2026-10-15T22:18:42Z | `R2_amendment_2026-10-15.md` |
| R2b | the same with tmu's budget form | 2026-10-15T22:24:38Z | the same file |
| R4 (with R1) | router against logistic regression, 3 datasets x 2 encoders, 10 seeds; calibration and out of scope | 2026-10-15T22:21:31Z | `R4_amendment_2026-10-15.md` |
| R4b | the same with 16,000 clauses | 2026-10-15T22:44:02Z | `R4b_amendment_2026-10-15.md` |
| R5 | routing learned from reward alone (bandit) against LinUCB and epsilon-greedy logistic | 2026-10-15T22:30:00Z | `R5_amendment_2026-10-15.md` |
| R5b | the same with Boltzmann exploration, 15 passes (follow-up, registered after R5's results) | 2026-10-16T06:03:20Z | `R5b_amendment_2026-10-16.md` |
| R6 | clarify-or-route with delayed reward (TD, memory literals) | 2026-10-15T22:58:03Z | `R6_amendment_2026-10-16.md` |
| R7 | a readable decision layer: named inputs (words, whitened similarity to intent centres), bge-large, 3 datasets x 10 seeds | 2026-10-16T08:47:13Z | `R7_amendment_2026-10-16.md` |
| R2c | the literal budget with a start state that obeys it, on Banking77 and HWU64 | 2026-10-16T09:42:34Z | `R2c_R7b_amendment_2026-10-16.md` |
| R7b | R7 with every clause bounded to 16 literals | 2026-10-16T09:45:25Z | `R2c_R7b_amendment_2026-10-16.md` |
| R8 | a TM policy over four steps (clarify-or-route over growing prefixes) | 2026-10-16T11:07:16Z | `R8_R5c_amendment_2026-10-16.md` |
| R5c | the TM bandit with 150 actions (CLINC150 intents) | 2026-10-16T11:07:45Z | `R8_R5c_amendment_2026-10-16.md` |
| R2d, R7c | R2c's and R7b's budget arms again, with the exact budget selection | 2026-10-16T11:33:29Z | `R2d_R7c_amendment_2026-10-16.md` |
| R2e, R5d | R2 again with the fixed budget (CLINC150); R5b's Banking77 bandit with 30 passes | 2026-10-16T15:05:29Z | |
| R11 | R7 and R7c unchanged on MASSIVE en-US (a related dataset (it shares about a third of its test texts with HWU64 train, with different labels)) | 2026-10-16T21:45:14Z | `phase2_amendment_2026-10-17.md` |
| R9 | ABCD next action: memory bits, TM against LR, MLP and the opaque MLP | 2026-10-16T22:51:09Z | the same file |
| R12 | trained on English, scored on Norwegian (MASSIVE nb-NO) | 2026-10-16T22:52:06Z | the same file |
| R9b | ABCD memory bits against a control with no record of the earlier actions (R9.1's design error) | 2026-10-16T23:27:00Z | the same file |
| R10 | ABCD next action from reward alone, with and without memory | 2026-10-17T01:07:02Z | the same file |
| R10c | the TM bandit against LinUCB on the same bits | 2026-10-17T01:40:29Z | the same file |
| R13 | R9b's models in closed loop (own presses as memory) | 2026-10-17T04:09:25Z | the same file |
| R14 | the TM bandit against a neural bandit on the same bits | 2026-10-17T04:13:10Z | the same file |
| R15 | replay for the TM bandit, against R10's TM and R14's neural bandit | 2026-10-17T04:30:39Z | the same file |
| R16 | Granmo's hierarchical TM against the flat machine of the same package | 2026-10-17T07:13:31Z | the same file |
| R17 | the reward-learned TM agent in closed loop | 2026-10-17T11:56:26Z | the same file |
| R16b | the hierarchical TM against a flat machine of matched size | 2026-10-17T12:38:55Z | the same file |
| R18 | memory literals on a second workflow dataset (STAR) | 2026-10-17T14:26:43Z | `R18_star_replication.md` |
| R19 | the agent with no neural network at run time (a fixed word table distilled offline from bge-base, lookup + OR), STAR and ABCD | 2026-10-18T09:40:15Z | `R19_amendment_2026-10-18.md` |
| R20 | the closed loop on STAR (R19's and R18's models) and the agent that reads only the customer (ABCD; its STAR arm stopped at its dev gate) | 2026-10-18T23:23:30Z | `R20_amendment_2026-10-19.md` |
| R21 | the agent with no neural network at any stage (the C4 word table: counting + SVD over C4 text), STAR and ABCD | 2026-10-19T05:48:24Z | `R21_c4_table_agent.md` |
| R22 | the TM's s tuned on dev (s 16) for the C4 agent and, equally, for the bge-large agent, STAR | 2026-10-19T16:31:24Z | `R22_tuned_s_star.md` |
| R23 | the same equal tuning on ABCD (each agent's s picked on ABCD dev: C4 16, bge-large 24) | 2026-10-19T20:32:10Z | `R23_tuned_s_abcd.md` |
| R24 | MultiWOZ 2.2 test (1st use): the network-free agents (C4, refined) against the bge-large agent, all with random negatives chosen on MultiWOZ dev, each agent's s by e77's rule (8 for all) | 2026-10-22T20:16:52Z | none (`R24_multiwoz_random_negatives.md`) |

## Results

All verdicts are on the test splits, from the run files named in `docs/results.md`.

| Part | Failure | Fix tested | Registered verdict | The number that carries it |
|---|---|---|---|---|
| 1 | Sparse words give negation blacklists | literal budget (R2) | not met, by one clause of 33 > 32 literals | +5.86 points [+5.28, +6.45], equal to LR on the same bits; clauses then 98% negation-only |
| 1 | the same | tmu's budget form (R2b) | not met | -25.5 points |
| 1 | the same | budget with a start state that obeys it, Banking77 and HWU64 (R2c) | not met, by two clauses of 33 literals (a tie bug in the budget) | +4.71 [+4.00, +5.44] and +4.09 [+2.79, +5.41] points |
| 1 | the same, CLINC150 | R2 again with the trim start and the exact selection (R2e) | **met** | +5.82 [+5.24, +6.41] points; longest clause 32; LR on the same bits -0.30 (non-inferior) |
| 1 | the same | the same with the exact selection (R2d) | **met** | +4.81 [+4.08, +5.53] and +3.96 [+2.69, +5.27] points; longest clause 32; 16 of 20 models identical to R2c's; clauses still 95-99% negation-only |
| 2 | Words carry no meaning | LLM encoder input | (measured, no rule) | word router 82.0 against embedding router 94.3 (MiniLM) / 96.1 (bge-base), CLINC150 |
| 3 | Clause learning trails a linear model | ensembles, fitted read-out, more clauses (R4, R4b) | R4 not met; R4b not met | R4: single TM -0.6 to -2.4 points; composite within 0.5 in 3 of 6 cells; read-out (clauses plus input bits) within 0.5 in 5 of 6, ahead in 1 (+0.45, fragile). R4b: 16,000 clauses lift the single TM by 0.3 to 1.0 points, within 0.5 in 3 of 6 cells |
| 4 | The vote is not a probability | temperature; top class sum for out of scope (R1) | **met** | ECE 0.013 to 0.027; OOS AUROC 0.962 / 0.973 against LR 0.962 / 0.969 (4,000 clauses); 0.966 / 0.975, above LR in every seed (16,000 clauses) |
| 5 | No credit assignment over time, four steps | TD target (fitted-Q iteration) (R8) | **met** | +0.369 over the myopic TM (nearly built in); +0.164 over always-ask, all from out-of-scope queries (in scope +0.006); 0.02 below the tuned rule and linear Q (no claim); memory literals -0.012 |
| 5 | No credit assignment over time | TD target (fitted-Q iteration) (R6) | **met** | +0.40 over the myopic TM (nearly built in); +0.04 over the tuned rule, all from out-of-scope episodes (in scope -0.014); equal to linear Q |
| 5 | Reward-only learning of the route | TM bandit (R5) | not met | best final accuracy with 10 actions (+0.77 over epsilon-greedy logistic, +0.88 over LinUCB; more regret online); -7.0 and -42.8 points with 18 and 77 actions |
| 5 | the same | Boltzmann exploration, 15 passes (R5b) | not met (one task) | 18 actions: +1.69 over LinUCB (now ahead); 77 actions: -3.16 (was -42.8) |
| 5 | the same, 77 actions (Banking77) | Boltzmann exploration, 30 passes (R5d) | **met** | -0.71 [-0.96, -0.33] against LinUCB; 0.996 of the supervised TM |
| 5 | the same, 150 actions | Boltzmann exploration, 30 passes, a new task (R5c) | **met** | 0.998 of the supervised TM; -0.98 [-1.20, -0.80] against LinUCB (just inside the margin) |
| 2, 3 | Opaque inputs (thresholds on embedding dimensions) | readable inputs: words and whitened similarity to intent centres (R7) | **met** | TM-readable minus LR-readable -0.12 to -0.29; composite minus opaque LR -0.88 to -0.99; the vote is exactly the sum of the firing clauses. Post hoc: the 3 strongest clauses alone give the decision in 99.3 to 99.7% of queries; equal to the one-rule "nearest centre" router (agrees on 99.2 to 99.7% of queries) |
| 1, 2, 3 | the same | R7b with the exact selection (R7c) | **met** | level with LR-readable (-0.01 to +0.01); composite minus opaque LR -0.46 to -0.84; longest clause 16; out-of-scope AUROC 0.950 |
| 1, 2, 3 | The readable layer's long exclusion clauses and weak out-of-scope detection | R7 plus the literal budget 16 (R7b) | not met, by three clauses of 17 literals (the same tie bug); out-of-scope rule met | TM-readable-16 equal to LR-readable (-0.13 to +0.03); composite minus opaque LR -0.65 to -0.88; out-of-scope AUROC 0.950 (R7 0.916) |
| 6 | Slow at vocabulary scale | small action spaces on the GPU | (measured) | 34 s per 40-epoch CLINC150 router training on one V100 |
| 7 | Memory literals had nothing to do (R6, R8) | a workflow task, ABCD, with a control that has no record of the earlier actions (R9b) | **met** | +18.65 [+17.20, +20.10] points (84.0% against 65.4%) |
| 7 | memory literals on a second dataset | STAR, 187 wizard actions, untouched test split, open loop; the control still reads the wizard's last message, which names the last action in 60% of later decisions (R18) | **met** | +15.43 [+14.38, +16.51] points (62.7% vs 47.2%); the TM is 3.3 points behind LR there |
| 7 | the same, first version | R9.1 (control read the history as words) | not met | +1.09 [+0.72, +1.47]; R9 not met as a whole |
| 2, 3 | Is the readable layer as good as linear and neural heads on a real agent task? | R9.2, R9.3, R9.4, R9.5 | R9.2, R9.3, R9.4 **met**; R9.5 not met | TM 84.8% vs LR 84.5% and MLP 84.5% on the same bits; TM composite 85.4% vs opaque MLP 85.2% (+0.10); TM > LR not shown (+0.36 [-0.29, +1.02]). All on ABCD; on STAR (R18, no rule) the TM is 3.3 and 3.1 points behind LR and the MLP |
| 8 | The workflow from reward alone | TM bandit with memory literals; no input built from the decision's label (R10) | **met** | 83.0% from reward alone = 0.988 of the supervised TM; memory +22.87 [+21.37, +24.39] points |
| 8 | the same, against a linear bandit | TM bandit vs LinUCB on the same bits, 3 alphas (R10c) | **met** | +5.51 [+4.66, +6.39] vs alpha 0.5; +3.00 [+2.18, +3.81] vs the best of the three registered alphas (1.0), after 15 passes; LinUCB is ahead for the first 7 to 11 passes and earns more reward online |
| 8 | the same, against a neural bandit | TM bandit vs a one-hidden-layer network learning from the same reward (R14) | **met** | -0.45 [-1.04, +0.16] (bar -1.0, judged on the point estimate); the network earns more reward online (80% vs 72%) |
| 8 | the TM bandit learns slowly (online 72% vs 80% for the network) | replay: 4 extra updates per round on its own log (R15) | **met** | online +6.35 points (78.2%, still 1.9 below the network, which had no replay); final 83.97% vs 83.45% (+0.52 [+0.03, +1.02], non-inferiority rule); 2.3 times the run time |
| 10 | Memory from the agent's own presses (closed loop) | R9b's models scored in closed loop (R13) | **met** | 80.7% closed loop vs 65.4% without memory (+15.34 [+13.80, +16.91]); own mistakes cost 3.31 points |
| 10 | the same for the agent learned from reward (closed loop) | 5 new runs of R10's TM bandit scored in closed loop (R17) | **met** | 79.1% vs 60.1% without memory (+18.96 [+17.30, +20.61]); own mistakes cost 3.70 points (at most 4 on the point estimate; interval [-4.44, -2.99]; memory bits only) |
| 11 | Rules through layers ("or use a hierarchical TM") | Granmo's AND-OR hierarchical TM against the package's flat machine (R16) | **met** | 81.7% vs 72.3% (+9.40 [+8.35, +10.41]), with 2 times the clauses and 16 times the automata; 2.3 points below this study's coalesced flat TM |
| 11 | the same, against a flat machine of matched size | flat machine with as many automata, 3,200 clauses per button (R16b) | **met** | 81.7% vs 74.7% (+7.02 [+6.01, +8.04]); size buys the flat machine +2.37 at one setting; T, s and the graded vote still differ |
| related data | Do R7 and R7c hold on another dataset? | R11 on MASSIVE en-US (a related dataset (it shares about a third of its test texts with HWU64 train, with different labels)) | R11.2 **met**; R11.1, R11.3 not met | bounded layer -0.35 vs LR-readable, composite -0.44 vs LR-raw; unbounded -0.65 (bar -0.5); ECE 0.060 (bar 0.05) |
| language | Trained in English, used in Norwegian | R12 on MASSIVE nb-NO | R12.1, R12.3 **met**; R12.2 not met | 82.2% in Norwegian, +0.72 [+0.30, +1.16] over LR on the same bits (non-inferiority rule), keeps 0.949 of its English accuracy; -1.80 vs opaque LR (bar -1.0) |
| 12 | A transformer per decision (bge-large, 335M parameters) | no neural network at run time: a fixed word table (distilled offline from bge-base) and lookup + OR bits (R19) | **met** (strong form, both datasets) | against the bge-large agent: STAR +0.06 [-0.63, +0.74] (level), ABCD -1.00 [-1.62, -0.40] (1.0 below, inside the 2-point margin); memory +15.05 and +20.79; without memory ABCD -3.14 (post hoc); on the ABCD paper's B-Slot 84.8% against its fine-tuned models' 86.7 to 93.6; one decision 29.5 ms against 806 ms, both on one CPU thread (rough: shared machine; the bge-large agent would be faster on a GPU, not measured, which the word-table agent does not need; the text step's speed-up is Model2Vec's) |
| 13 | Own mistakes compound in the closed loop (STAR, 187 actions) | R19's word-table models with their own presses as memory (R20.1) | **met** | memory still pays: +7.53 [+6.51, +8.58] (55.2% vs 47.7%) |
| 13 | the same | R13's bar: own mistakes cost at most 4 points (R20.2) | not met (expected from dev) | -7.51 [-8.34, -6.72] (55.2% closed vs 62.7% open); on ABCD -3.40 (no rule) |
| 13 | the same, against bge-large | the word table against R18's bge-large agent, both closed loop (R20.3) | **met** (strong form) | +1.96 [+1.07, +2.85]: not worse; the interval lies above 0, but superiority was not a registered rule |
| 13 | Reading the human agent's text | the agent that reads only the customer, closed loop, STAR (R20.4, dev gate S4) | not taken to test: dev gate failed | dev: memory +4.50 in the closed loop (gate +5.0); own mistakes -10.12; dev-only line |
| 13 | the same, ABCD (supporting) | the customer-only agent, closed loop (R20.4) | **met** | 72.6% vs 48.7% without memory: +23.94 [+22.20, +25.67]; -6.98 [-8.13, -5.84] against the agent that reads the human agent's text |
| 17 | the same, ABCD | each agent's s tuned on ABCD dev (R23) | **met** (strong) | C4 s 16 minus bge-large s 24: -0.92 [-1.61, -0.26] (within the margin; the bge-large agent about 1 point ahead) |
| 17 | The TM's s was never tuned; the TM trailed LR on STAR | s tuned on dev (s 16) for both agents (R22) | **met** (R22.2 and R22.3 strong) | C4 s 16 vs s 8 +2.70 [+2.17, +3.24]; vs LR on the same bits -0.09 [-0.90, +0.73]; vs bge-large tuned the same way +1.64 [+0.94, +2.35] |
| 15 | The word table was distilled from a transformer | the agent on a table made by counting + SVD over C4 text, with no neural network at any stage (R21) | **met** (strong form, both datasets) | against the potion agent: STAR -0.10 [-0.65, +0.45], ABCD -0.29 [-0.74, +0.17]; memory +14.73 / +20.63; against bge-large -0.04 / -1.29; TM vs LR on the same bits -2.79 / +1.76 |
| 18 | The network-free agents fall behind on a third dataset (MultiWOZ) | random negatives for all three agents, chosen on MultiWOZ dev; each agent's s by e77's rule (R24.1, the headline: the C4 agent, no neural network at any stage) | **met** (strong) | C4 minus bge-large +0.53 [-0.18, +1.23] (two-level [-0.32, +1.40]); superiority (R24.2) not met, so no "better than"; the rule costs STAR -11.9 and ABCD -4.0 on dev (refined agent, e83) |
| 18 | the same, the refined agent (network-free at run time, teacher at training time: bge-base-en-v1.5) | R24.1, secondary | **met** (strong) | +0.05 [-0.65, +0.72] (two-level [-0.83, +0.95]); R24.2 not met |
| 21 | The TM is below LR on the same bits (MultiWOZ) | random negatives (R24, no rule) | still a failure | TM minus LR on test: C4 -1.55 [-2.32, -0.79], refined -2.49 [-3.33, -1.64], bge-large -1.89 [-2.69, -1.10] |
