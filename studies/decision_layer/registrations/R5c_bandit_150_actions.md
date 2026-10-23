# R5c registration: does the exploration fix carry to 150 actions? (TM bandit, CLINC150 intents)

Follow-up to R5 and R5b. R5 (top-3 epsilon-greedy) failed with 77 actions (-42.8 points against LinUCB). R5b (Boltzmann exploration, 15 passes) repaired most of it: with 18 actions the TM bandit passed LinUCB (+1.69), with 77 it was 3.16 points behind, and on both tasks it reached at least 0.97 of the same TM trained on labels. R5c asks whether learning the route from reward alone still works with 150 actions, the full CLINC150 intent set. That is the goal's "learning from reward": the reward becomes Type I and Type II feedback for the chosen action.

Written after the exploratory val runs below. No bandit over the 150 CLINC150 intents has scored their test split; the CLINC150 test queries themselves were scored by routers (R2, R4, R4b, R7, R7b), by R5 (10 domains as actions) and by R6/R8 (episodes). Frozen by `scripts/freeze.py` before any R5c run.

## Exploration behind it (val, seed 1, `results/explore/bandit.jsonl`)

| Run | Learner | Passes | Final val accuracy | Cumulative online accuracy |
|---|---|---|---|---|
| e20_c150_boltz05_p30 | TM bandit, Boltzmann tau 0.5, replay 4 | 30 | 0.9387 (0.917 after 19 passes, 0.936 after 25) | 0.777 |
| e20_c150_linucb_p30 | LinUCB, alpha 1.0 | 30 | 0.9490 (0.948 from pass 7 on) | 0.937 |

For reference, the supervised TM on CLINC150 val (thermo:8, 4,000 clauses) is about 0.946 to 0.949 (R4 exploration).

## Design

- **Task:** CLINC150 intents, 150 actions; MiniLM, `thermo:8`; the stream protocol of R5 (rounds of 32, reward 1 or 0 for the chosen action only), **30 passes**.
- **TM bandit:** the R5b machine (4,000 clauses, T 4,000, s 8, replay 4, Boltzmann tau 0.5; `--eps 0.05` has no effect under Boltzmann, see the R5b amendment).
- **LinUCB:** alpha 1.0, 30 passes.
- **sup-TM:** the same TM trained on labels for 30 epochs, no snapshot.
- **Seeds:** 1 to 5. Test split, scored once per run. Tag `R5c`, results in `results/confirm/R5c/bandit.jsonl`.

## Pass rules (test)

- **R5c.1 (the cost of learning from reward instead of labels, with 150 actions):** the mean final accuracy of the TM bandit is at least 0.97 times sup-TM's.
- **R5c is met** if R5c.1 holds.
- **R5c.2, a separate verdict:** the TM bandit's mean final accuracy is at least LinUCB's minus 1.0 point (the margin of R5 and R5b). On val it was 1.03 points behind, so this one is expected to be close. It is kept out of "met" because of that val result; this split was decided after seeing it.

Reported with no rule: cumulative online accuracy (the TM pays far more exploration errors than LinUCB on the way), the bootstrap interval over seeds of TM minus LinUCB, the accuracy after every pass.
