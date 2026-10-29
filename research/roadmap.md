# Open work

What was registered, prepared or designed and has not been completed, with the items worth completing first. Statuses of the studies repeat the [results overview](../docs/results-overview.md); the design items come from the [design page](design.md). The finish pass of 25 and 26 Oct 2026 ran the registered and frozen work that could run on the machine in two days; what it left, and why, is listed here with each item.

## Order

1. **Week 0 of the design.** Read the three unread papers on teams of Tsetlin Machines and the 2026 symposium papers; measure the free CPU. The reference n-gram toolkit, KenLM, is installed outside the repository and used by A1 in tally (its binaries are recorded by the fourth baseline check, `20261025T113821Z_a1_baseline_check`).
2. **Line T of the design.** T0 (counted votes against learned votes with the CAIR library), B1 (the synthetic team game) and B6 (team margin on composites). T0's verdict was drawn on 2026-10-25 in [tally](../studies/tally/README.md): not met, the learned votes beat the counted read-out. B1 and B6 have code and smoke or timing runs and no registration; their grids need about 680 CPU-hours together (B1 about 400, B6 about 280; estimates of the finish pass, UNVERIFIED) and were not run in the finish pass. Before a registration, B1's calibration must show that the shared reward can reach its target (60 of 60 runs were censored), and B6 must declare that its AG News "dev-test" is the official test file, read 28 times by tm_language_model's A7 runs, or cut a fresh split from the training file. B6's launcher must also count a failed CPU-gate reading as closed, as B1's does since 26 Oct 2026 (see the [tally README](../studies/tally/README.md)).
3. **Step A1 of the count floor**, then A2 to A4. A1 has its baseline check and A/A noise runs, and A2's registered look on dev-test met its success line on 2026-10-25 (the exchange-class code, 7.59%). A3, the full floor against Kneser-Ney plus a cache, has no code; after A2's look its contrast on dev-test is already informed, so A3 may carry a dev-test rule only for its increment over KN5 + cache + exchange, or must use the standard test file (A2's registration). A4 has no code.
4. **Line B**, the slots, after the count floor stands.
5. **Line C of the design, C3.** The LSTM reference of tally was rerun on 2026-10-25 and 26 with the loss and the clips of Zaremba et al. 2014: the medium model reaches 90.99 to 92.12 on 10 of 10 seeds (paper 86.2), and the large model 86.22 to 87.05 on 6 of 10 seeds (paper 82.2); the other 4 large seeds collapsed in their first epoch ([results page](../studies/tally/docs/results.md#c3)). Open: the cause of the gap (training on 90% of the standard training text is one candidate, untested), a rule for collapsed runs, and the C3 comparison itself, which has no registration.

## Open in the studies

### decision_layer (running)

| Item | State | What it needs |
|---|---|---|
| R25, an agent on the teacher-refined table | draft, never frozen; its dev go-bars failed (e82) | a stack that passes the go-bars on dev |
| A registered superiority test of the agents with no neural network | R24.2 not met on MultiWOZ, the study's only fresh test split; rev 0 of R24 was never frozen | a fresh test split |
| R20.4 on STAR, the customer-only agent | not run: stopped at its dev gate | a view that passes the gate on dev |
| e86, clause dropout (dev) | closed on 2026-10-25 as a dead end: dropout is kept on no dataset (FAILURE_MAP.md) | nothing |
| e87, short receipts (dev) | Part B's bar (the exact budget 3, as its dated note of 2026-10-25 amends the cells) is not met on any dataset: short receipts cost the agent its accuracy (FAILURE_MAP.md); Part A, the L1-LR receipts, and the kill rule were not run: about 330 CPU-hours, extrapolated | Part A as frozen, in a later pass; its method stays frozen, and a change of it is a new rule |
| e80, view B plus the recent-customer block (dev) | run on 2026-10-25 under its dated note: not taken (STAR -0.40 against +0.5; ABCD -0.11 holds) at view B's s 16 | a view-specific s, which needs its own plan; e80-R3 and R5 are not reached |
| Other dev directions named on the results page | not run: a negative rule chosen per dataset and focused negative sampling (e83), other T ratios (e81), the closed loop of R20 for the C4 agent | a dev plan and a freeze each; a registered test on STAR or ABCD would be their 7th or 15th test use |

### clause_lm (running as preparation)

M5 step 1 is complete: the amended Gate 0 and the s-test grid ran on 2026-10-14, and by the registered rules the verdict is negative ([results page](../studies/clause_lm/docs/results.md#amended-gate-0-and-the-grid)). The Gate 0 cause check of 2026-10-25 ended without a cause statement ([results page](../studies/clause_lm/docs/results.md#gate-0-cause-check-of-2026-10-25)).

| Item | State | What it needs |
|---|---|---|
| M1, pair model | written (`src/pair_tm.py`), tested on the CPU, not run | a registration with a held-out split and equally tuned baselines; about 3 GPU-hours for 3 seeds, 8 to 13 with tuning (estimate); low chance against the earlier word-pair results |
| M2, clause-count language model with the OR-groups of M1 | evaluator ran once with k-means groups and no Tsetlin Machine | M1 first; a stronger count floor for the idea exists in tally (A2) |
| M3, M4, M5 step 2 | no code | M2 first; by the reading of the finish pass's inventory (not checked), M3 is superseded by the C-core line of the research notes and by tally |
| The cause of the failed Gate 0 | removing the negative-class erosion from torch did not reproduce the numba model (arm A mixed: healthy structure, accuracy below the plan's bar); arm B and the other kernel differences not run | a new frozen plan: arm B (the numba kernel with the branch added; 60 to 120 CPU-minutes by the plan's estimate, UNVERIFIED), or the other differences one at a time |

### hierarchical_tm (finished)

- The Wikipedia pair data has duplicate articles across its splits ([known limitations](../docs/known-limitations.md)). The splits are to be deduplicated by content, then split by article, before any new test on that data.
- No learned rule captured either of the two sentence families of the proposal. Why, and whether another training signal changes it, is open.
- The single final test is used up. A new test on the same question needs new held-out data.
- The reproduction check of the freeze rule (HTM-01) ran on 25 and 26 Oct 2026 and re-checked 124 of the 683 page numbers, all exactly ([outcome](../studies/hierarchical_tm/docs/reproduction_after_the_study.md)). Open: the 559 numbers it did not re-check, among them 68 of training runs outside its sample, which retrains of those runs could re-check. L2 at `s` 100 with A = 8 (HTM-03) ran on 25 Oct 2026, and its prediction fails ([results](../studies/hierarchical_tm/docs/results.md)).

### tm_language_model (archived)

- 57 of the 91 registered hypotheses were not run, among them the families on the kinship task (H10), the second corpus (H11), data scale (H12) and efficiency (H15). KB7 did not stop the study: its registered consequence is a reporting rule for H1.1 (a first measurement with no superiority claim), and the families H8 to H17 were registered after it triggered. The rows that did not run were deferred for lack of CPU slots on 2026-09-24, the R3 rows were cut back by gate G-R1, and the study was archived on 2026-10-01 with nothing cut (docs/12 F4 and F23 of the study). They stay as a record. The finish pass of 2026-10-25 ran the prepared ones: H8.3 and H8.4 on the test split (both met), H8.8 part (c), H8.9 and H9.6 (not met), H3.5 at 200 steps, the logged H2.3 run and the H2.4 re-check (all hold).
- The main accuracy hypothesis H1.1 ran on seed 1 only; seeds 2 and 3 were not run.
- H8.10, the one row of A3 that could credit Tsetlin learning on top of the count-vote core, was not run: it needs new banks.
- H4.2 and H4.3 stay as they are: H4.3 needs a new layer-2 learner, and H4.2 needs H4.1 and a paragraph-pair driver that does not exist. The tuned configs of H17.5 (`experiments/cvtm/configs/ssl_r2_hao_tuned.json` and `ssl_r2_hao_head_tuned.json`) are exploratory and post hoc on a registered "not met", and were not run.
- The count-vote finding of family H8 is the part taken forward: it was repeated with the CAIR library as T0 of the design, where the learned votes beat the counted read-out.

## Open in the research

- QTMA: one parked test (B8), one signed counter per feature against the two standard automata.
- CDE: the two open steps after the lookup, a code that changes with the context and categories on the output side.
- The full text of the design, with the output formula and a worked example, is kept in the author's notes and is summarised on the design page.
- The research lines of the agent harness and of text understanding are published since 2026-10-29 as the studies [agent_guards](../studies/agent_guards/README.md) and [text_understanding](../studies/text_understanding/README.md), with [agent_vs_llm](../studies/agent_vs_llm/README.md) and [language_part3](../studies/language_part3/README.md): the registered E4 run of the agent harness ran on 2026-10-26, and its E2 run stopped at its second stage under its own failure rule (a model request with no output cap timed out twice), so E2 needs a dated revision to go on; the registered MTOP test of the text-understanding line ran on 2026-10-25; a drafter for the local LLM (B-helper of the research notes) has a design and no code.
