# Open work

What was registered, prepared or designed and has not been completed, with the items worth completing first. Statuses of the studies repeat the [results overview](../docs/results-overview.md); the design items come from the [design page](design.md).

## Order

1. **Week 0 of the design.** Read the three unread papers on teams of Tsetlin Machines and the 2026 symposium papers; measure the free CPU. The reference n-gram toolkit, KenLM, is installed outside the repository and used by A1 in tally (its version is not recorded).
2. **Line T of the design.** T0 (counted votes against learned votes with the CAIR library), B1 (the synthetic team game) and B6 (team margin on composites). These are the tests in which a positive Tsetlin result is most likely. In the study [tally](../studies/tally/README.md), T0 was registered on 2026-10-15 and its 30 final runs (both machines and the logistic-regression arm, seeds 1 to 10) have finished; its verdict, `studies/tally/src/tally/t0/report.py` over those runs, has not been drawn. B1 and B6 have code and smoke or timing runs there and no registration.
3. **Step A1 of the count floor**, then A2 to A4. A1 and A2 have code in tally, with a baseline check and a smoke run; A3 and A4 have no code.
4. **Line B**, the slots, after the count floor stands.

## Open in the studies

### decision_layer (running)

| Item | State | What it needs |
|---|---|---|
| R25, an agent on the teacher-refined table | draft, never frozen; its dev go-bars failed (e82) | a stack that passes the go-bars on dev |
| A registered superiority test of the agents with no neural network | R24.2 not met on MultiWOZ, the study's only fresh test split; rev 0 of R24 was never frozen | a fresh test split |
| R20.4 on STAR, the customer-only agent | not run: stopped at its dev gate | a view that passes the gate on dev |
| e86 and e87, dev plans frozen on 2026-10-23 | e86 ran 8 of its 14 dev jobs and was stopped; e87 has no run; neither has a result | their runs and their analyses |

### clause_lm (running as preparation)

M5 step 1 is complete: the amended Gate 0 and the s-test grid ran on 2026-10-14, and by the registered rules the verdict is negative ([results page](../studies/clause_lm/docs/results.md#amended-gate-0-and-the-grid)).

| Item | State | What it needs |
|---|---|---|
| M1, pair model | written (`src/pair_tm.py`), not run | 3 seeds against the statistical class baseline on unseen pairs |
| M2, clause-count language model with the OR-groups of M1 | evaluator ran once with k-means groups and no Tsetlin Machine | M1 first |
| M3, M4, M5 step 2 | no code | M2 first |

### hierarchical_tm (finished)

- The Wikipedia pair data has duplicate articles across its splits ([known limitations](../docs/known-limitations.md)). The splits are to be rebuilt by article before any new test on that data.
- No learned rule captured either of the two sentence families of the proposal. Why, and whether another training signal changes it, is open.
- The single final test is used up. A new test on the same question needs new held-out data.

### tm_language_model (archived)

- 61 of the 91 registered hypotheses were not run, among them the families on the kinship task (H10), the second corpus (H11), data scale (H12) and efficiency (H15). The study was stopped by its registered kill criterion KB7, so these stay as a record and are not planned as they stand.
- The main accuracy hypothesis H1.1 ran on seed 1 only; seeds 2 and 3 were not run.
- The count-vote finding of family H8 is the part taken forward: it is repeated with the CAIR library as T0 of the design.

## Open in the research

- QTMA: one parked test (B8), one signed counter per feature against the two standard automata.
- CDE: the two open steps after the lookup, a code that changes with the context and categories on the output side.
- The full text of the design, with the output formula and a worked example, is kept in the author's notes and is summarised on the design page.
