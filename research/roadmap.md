# Open work

What was registered, prepared or designed and has not been completed, with the items worth completing first. Statuses of the studies repeat the [results overview](../docs/results-overview.md); the design items come from the [design page](design.md).

## Order

1. **Week 0 of the design.** Read the three unread papers on teams of Tsetlin Machines and the 2026 symposium papers; measure the free CPU; install a reference n-gram toolkit.
2. **The amended Gate 0 of clause_lm.** One run of the router reproduction with the torch backend on a GPU. It is registered, it has not been run, and it is the test of the backend explanation on the [results page](../studies/clause_lm/docs/results.md#diagnosis-notes-of-2026-10-02). The upstream record gives 48.9 s of training for such a run.
3. **Line T of the design.** T0 (counted votes against learned votes with the CAIR library), B1 (the synthetic team game) and B6 (team margin on composites). These are the tests in which a positive Tsetlin result is most likely.
4. **Step A1 of the count floor**, then A2 to A4.
5. **Line B**, the slots, after the count floor stands.

## Open in the studies

### clause_lm (running as preparation)

| Item | State | What it needs |
|---|---|---|
| Amended Gate 0 | registered, not run | a GPU; the launcher `studies/clause_lm/src/m5_gpu_all.sh` |
| s-test grid of M5 step 1 (12 runs with the Gate 0 run) | not run | Gate 0 met first; `src/m5_grid.sh` and `src/m5_report.py` are written and were never run |
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
