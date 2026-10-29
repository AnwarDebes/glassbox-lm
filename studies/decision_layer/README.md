# decision_layer: the Tsetlin Machine as a readable decision layer on top of an LLM

Question: where do Tsetlin Machines fail on language (part by part, `FAILURE_MAP.md`)? What can work instead, with an LLM encoder doing the language and a Tsetlin Machine making the decision?

- **Router:** intent routing on frozen LLM embeddings (CLINC150, Banking77, HWU64), against logistic regression. This continues Clause-Driven-LLM.
- **Readable layer:** the same router with inputs a person can read (words, and the LLM's whitened similarity to each intent's centre), so every rule and every decision can be printed.
- **Agent loop:** the Tsetlin Machine as the policy or value function that learns from reward. Phase 1 has three tests, phase 2 a real workflow (ABCD):
  - a contextual bandit (routing learned from reward alone; 10 to 150 actions);
  - clarify-or-route, two steps with delayed reward (TD learning, memory literals);
  - clarify-or-route over four growing prefixes of the query (longer credit assignment).
- **LLM routing:** choosing per prompt between a cheap LLM and GPT-4 (RouterBench). Explored only and set aside. On MiniLM prompt embeddings, one logistic-regression router (one encoder, val) routes barely better than random: area 0.722 against 0.712 (exploration e11, val; its run log is local). A TM router was not tried on it.

Status: running. Started 2026-10-15; phase 2 started 2026-10-16 21:31 UTC; the last registered test so far, R24, ended on 2026-10-23.

- Registered tests (`registrations/`): all have finished; phase 1 has 17 verdicts in all (R1 is part of R4's registration). Nine are met: R1 (inside R4), R6, R7 (readable layer), R8 (four-step agent), R5c and R5d (bandits with 150 and 77 actions), R2e and R2d (the literal budget on all three datasets) and R7c (the readable layer with every clause at most 16 literals). R2, R2c and R7b failed only their clause-length rules, by clauses one literal over the budget: a tie bug in the budget code, fixed (`tests/test_budget_exact.py`) and re-run as R2e, R2d and R7c. R2c.3, reported separately (the budget router against logistic regression on the same bits, non-inferior if at least -0.5 points), is not met in R2c and in R2d on both datasets (-1.43 and -5.19; -1.34 and -5.32); this was read on 2026-10-25 from the rule outputs written with the runs, and no verdict had been recorded before. R2b (another budget form), R4 and R4b (the TM against logistic regression on embedding thresholds), R5 (top-3 exploration) and R5b (15 passes on Banking77) are not met; R5c and R5d, with Boltzmann exploration and 30 passes, are.
- **Phase 2 (2026-10-16 evening to 2026-10-17):** fresh data, ABCD (customer-service conversations: which of 30 action buttons an agent presses next) and STAR (wizard-of-oz task dialogues, 187 actions), and MASSIVE 1.1 (en-US and its Norwegian translation), a related dataset (it shares about a third of its test texts with HWU64 train, with different labels).
  - Met:
    - R9b: memory literals ("done X", "last action = Y"): 84.0% against 65.4% without them (+18.65 points).
    - R18: the memory result replicates on a second dataset, STAR (187 wizard actions, untouched test split, open loop): +15.4 points (62.7% against 47.2%). The control without memory still reads the wizard's last message, which names the last action in 60% of later decisions, so the gain understates the history's value. There the TM is 3.3 points behind LR and 3.1 behind an MLP on the same bits.
    - R13: closed loop for the memory bits, with its own presses as memory: 80.7%, still +15.3 points over no memory; its own mistakes cost 3.3 points (the text, including the human agent's utterances, is still replayed from the log, so a fully closed loop probably costs more).
    - R10: the workflow from a 0/1 reward alone (replaying logged conversations): 83.0%, 0.988 of the supervised TM; memory +22.87 points.
    - R10c: the TM bandit ends above LinUCB on the same bits after the 15 registered passes (alpha 0.1 to 1.0): +5.51 points at the dev setting, +3.00 against the best of the three alphas. LinUCB is ahead for the first 7 to 11 passes and earns more reward while learning; with alpha 4.0, tried afterwards on one dev seed, it ends 0.5 points below the TM, and its end accuracy still rises with alpha.
    - R14: within the registered margin of a small neural bandit on the same inputs on the point estimate (-0.45, interval [-1.04, +0.16]); the network learns faster.
    - R15: replay (4 extra updates per round on its own log) closes most of the slow-learning gap: online 78.2% against 71.8% (the neural bandit, without replay, 80.1%), and at the end +0.52 [+0.03, +1.02] against the neural bandit under a non-inferiority rule.
    - R16: Granmo's hierarchical TM (AND-OR clauses, as configured with 2 times the clauses and 16 times the automata of the flat machine) is 9.4 points above the flat machine of the same package on the ABCD workflow (81.7% against 72.3%), but 2.3 points below this study's coalesced flat TM.
    - R16b: against a flat machine of the same package with as many automata (3,200 clauses per button), the hierarchical TM stays 7.0 points ahead (81.7% against 74.7%); at the one flat setting tried, size explains about a quarter of R16's gap. T, s and the vote (a count from 0 to 4,096 against 0 or 1) still differ, so the remaining 7 points are not the AND-OR structure's alone.
    - R17: the reward-learned agent in closed loop for the memory bits (its own presses as memory; the text, including the human agent's utterances, is still replayed from the log): 79.1%, +19.0 points over no memory; its own mistakes cost 3.7 points (rule: at most 4, met on the point estimate; interval [-4.44, -2.99]).
    - Rules met inside tests not met as a whole: R9.2 to R9.4 (on ABCD the TM is level with LR and an MLP on the same named inputs; on STAR, R18, it is 3.3 and 3.1 points behind them; its 5-machine composite is level with an opaque MLP on the raw embeddings), R11.2 (the bounded readable layer passes R7c's rules again on MASSIVE, with less room), R12.1 and R12.3 (trained on English only, it routes Norwegian at 82.2%, ahead of LR on the same bits under a non-inferiority rule, keeping 0.949 of its English accuracy).
  - Not met: R9.1 (a design error: the control read the history as words; R9b repeats it cleanly), R9.5 (the TM beats LR: +0.36, interval through 0), R11.1 (R7's unbounded setting, -0.65 against the -0.5 bar), R11.3 (calibration 0.060 against 0.05), R12.2 (readability costs 1.8 points with the multilingual encoder, bar 1.0).
  - Explored on dev, not registered: two remedies for the closed loop (noisy memory, data aggregation: no help in one run each); shortening clauses after training (the exclusions carry accuracy); Qwen3-Embedding-0.6B instead of bge-large (no gain); hierarchical-TM shapes and settings before R16; flat sizes before R16b (e44); the bandit in closed loop before R17 (e43); the written guidelines added to the trained TM as 224 clauses (0.00 to +0.46 points on dev); STAR clause counts before R18 (e45, e46). See docs/results.md, "Explorations".
- **Phase 3 (2026-10-17 evening to 2026-10-19): the agent with no neural network at run time.**
  - Met: R19. No neural network at run time: a rule-based tokenizer, lookups in a fixed 63,091 x 512 table, then a bitwise OR of precomputed per-token bit codes, then Boolean clauses and an integer vote. The table was distilled once, offline, from a transformer (bge-base-en-v1.5 via Model2Vec); the reference agent runs bge-large-en-v1.5 (335M parameters) per decision.
    - STAR: level with the bge-large agent (62.7% against 62.7%, +0.06 [-0.63, +0.74] points).
    - ABCD: 1.0 point below it, inside the registered 2-point margin (83.0% against 84.0%, -1.00 [-1.62, -0.40]).
    - The memory literals still add +15.0 and +20.8 points. Without memory the table costs 3.1 points on ABCD (post hoc).
    - On the ABCD paper's own B-Slot measure it scores 84.8%, 1.9 to 8.8 points below the paper's fine-tuned BERT-family models.
    - One decision takes about 29.5 ms on one CPU thread, against about 806 ms for the bge-large agent (medians; rough: shared machine). Both agents were run on one CPU thread; the bge-large agent would be faster on a GPU (not measured), which the word-table agent does not need. The text step's speed-up is Model2Vec's, and the TM decision is now the bottleneck.
  - R20 (2026-10-19): the closed loop, with the agent's own presses as memory.
    - The headline question, an agent that reads only the customer on STAR, stopped at its dev gate: without the wizard's text a wrong press poisons the memory, and memory added +4.50 in the closed loop against the +5.0 gate. A dev-only failure-map line; STAR's test split was not spent on it.
    - STAR test, R19's word-table agent in the closed loop: memory still pays (+7.53 [+6.51, +8.58], met); its own mistakes cost 7.51 points, above R13's bar of 4 (not met, as expected); not worse than the bge-large agent in the same loop (+1.96 [+1.07, +2.85], met, strong; superiority was not a registered rule).
    - ABCD test, supporting: the customer-only agent (no neural network at run time, no text written by the human agent) reaches 72.6% in the closed loop, memory +23.94 [+22.20, +25.67] (met); 7.0 points below the agent that also reads the human agent's text.
  - Line B (2026-10-19, dev, frozen plan e70): proofs against ABCD's written guidelines.
    - Over a superset of all real inputs, none of 50 guideline rules holds for the raw TM (250 of 250 checks give a counterexample, each far from any real conversation).
    - With 50 veto clauses (one per rule; 15 to 20 change a dev decision), all 50 hold by construction, at a dev cost of 0.27 to 0.43 points open loop, all of it on turns where the human agent broke the rule.
    - With the same MILP encoding and solver (600 s per check), the TM's checks all resolved (median 1.3 s); MLP twins on the same bits resolved 41 of 50 (64 units, median 151 s) and 1 of 50 (128 units). No dedicated neural verifier was tried.
    - Local proofs in word edits (e71, R19's potion agent, TM seed 1, by exhaustive search): for 93.0% of 200 sampled real dev states, no deletion of a word from the customer's latest message or insertion of one of 1,626 common customer words at its end makes the agent choose a button the rules forbid in that state.
      - With up to two such edits, it holds for 75.5%.
      - Over seeds 1 to 3: 92.5 to 94.0% with one edit and 75.5 to 77.5% with up to two; seeds 4 and 5 were not run.
  - e72 (2026-10-19, dev, frozen plan): the agent on a word table made with no neural network at any stage (the C4 word table: counting + SVD over C4 text).
    - Dev gate passed on both datasets: ABCD -0.08 points against the potion agent (0.8317 against 0.8325), STAR -0.26 (0.6317 against 0.6343). Without memory the table costs 0.92 points on ABCD.
  - R21 (2026-10-19, registered, test; STAR's 5th use, ABCD's 13th): met on both datasets, in the strong form.
    - The agent with no neural network at any stage is within 0.10 (STAR) and 0.29 (ABCD) points of the potion agent: -0.10 [-0.65, +0.45] and -0.29 [-0.74, +0.17].
    - Memory adds +14.73 and +20.63 points.
    - No rule: against bge-large -0.04 and -1.29; on the same bits LR is 2.79 ahead on STAR and the TM 1.76 ahead on ABCD.
    - With 50 veto clauses all 50 guideline-derived order rules hold for its 5 ABCD models (e72d, dev).
    - Rough speed on one CPU thread: 2.6 ms and 0.5 ms per decision.
  - e73 (2026-10-19, dev, frozen note, no rule): the STAR agent produces its replies from the task's templates (template + database slots).
    - Its reply equals the wizard's word for word in 45.5% of template turns (40.0% closed loop), against a ceiling of 79.6% with the logged label.
    - 12.9% of the wizard's replies are free text.
  - e74 / e74c / e74d (2026-10-19, dev, frozen plans): the TM's s.
    - Both agents peak at s 16. The C4 agent gains +2.28 on STAR dev and +1.00 on ABCD dev over s 8.
    - It comes within 0.13 of a tuned LR on STAR dev, and is +0.38 against the equally tuned bge-large agent.
  - R22 (2026-10-19, registered, STAR test, 6th use): all three rules met, two in the strong form.
    - s 16 against s 8: +2.70 [+2.17, +3.24].
    - Against a tuned LR on the same bits: -0.09 [-0.90, +0.73], level.
    - Against the equally tuned bge-large agent: +1.64 [+0.94, +2.35] (not worse; superiority not a registered rule).
  - R23 (2026-10-19, registered, ABCD test, 14th use): met, strong.
    - Under equal tuning on ABCD (C4 s 16, bge-large s 24), C4 minus bge-large is -0.92 [-1.61, -0.26]: within the 2-point margin, with bge-large about 1 point ahead.
    - Across the two datasets, the agent with no neural network at any stage is within the margin of the equally tuned bge-large agent: +1.64 (STAR), -0.92 (ABCD).
  - Explored on dev, not registered:
    - Why the TM trails LR on STAR: the gap is in the less common actions. A literal budget reaches the 16,000-clause level at 4,000 clauses.
    - One score per conversation instead of a reward per press: the TM reaches 0.91 of its per-press accuracy, but learns much more slowly than a small neural learner (0.733 against 0.803 after 30 passes).
- **Phase 4 (2026-10-19 evening to 2026-10-23): a third dataset, MultiWOZ 2.2.**
  - On dev, with the training rule's hardest negative classes, the agent with no neural network at any stage fell behind the bge-large agent (e77b: -3.21 against the -1.0 gate). Random negative classes, chosen on MultiWOZ dev (e83, e84, e85), lift the network-free agents far more than bge-large, but cost STAR -11.9 and ABCD -4.0 points on dev, so the rule is a MultiWOZ setting.
  - R24 (registered, MultiWOZ test's 1st use): R24.1 met, strong, for both network-free agents against the bge-large agent: C4 +0.53 [-0.18, +1.23], refined table +0.05 [-0.65, +0.72]. R24.2, superiority, is not met for either, so the study's only fresh split for a registered "better than" is spent without one. On the same bits LR is ahead of the TM for all three agents (C4 -1.55, refined -2.49, bge-large -1.89; no rule).
  - Two registration drafts were never frozen: R25 (the agent on the teacher-refined table), whose dev go-bars failed, and rev 0 of R24, a superiority test on MultiWOZ written on 2026-10-19, which the frozen R24 superseded. The dev plan e86, frozen on 2026-10-23 and finished on 2026-10-25 under a dated note, is a dead end at its settings: clause dropout 0.5 is kept on no dataset, and random negatives with dropout fail as one setting for all three (docs/results.md, e86). The dev plan e87, frozen on 2026-10-23, ran in part on 2026-10-25 under a dated note: its Part B bar (the C4 agent with clauses of at most 3 literals, within 2.0 points of LR on dev, with a receipt of at most 3 such clauses for at least 90% of dev decisions) is not met on any of the three datasets, so short receipts cost the agent its accuracy there; Part A and the kill rule against L1-LR have not run (docs/results.md, e87). The dev plan e80, frozen on 2026-10-20 (view B plus the customer's last 3 utterances), ran on 2026-10-25 under a dated note and is not taken: STAR -0.40 points against a bar of +0.5, at view B's s 16 (docs/results.md, e80).
- The failure map is in `FAILURE_MAP.md`.
- Results are in `docs/results.md`.

## Layout

```
FAILURE_MAP.md        the failure map, part by part, with the registered test of each part
registrations/        registrations (frozen: *.frozen.json with UTC time and SHA-256 of text and code)
src/dl/               data, boolean views, GPU coalesced TM, linear baselines, bandits, MDP, calibration; the word table's bits (static_text.py) and the TM decision on the CPU (cpu_tm.py; cpu_tm_fast.py, exact and about 10 times faster)
scripts/              embed, baselines, exploration (val only), confirmatory runs (test once), reports
tests/                test_tm.py (fp16 products give the same training as fp32), test_budget_exact.py (the exact literal budget), test_tm_chunkbudget.py (the exact budget in the chunked update), check_chunked_update.py (the chunked update against the unchunked one), test_linucb_fast.py, test_episodic_credit.py, test_static_text.py and test_cpu_tm.py (the word table's bits; the CPU decision), test_cpu_tm_fast.py (the fast CPU decision), test_customer_view.py (view C reads no human-agent text), test_count_text.py (the C4 table's bits), and the tests of the dev checkers, pickers, reports and drivers (test_e80_check.py, test_e81_chain.py to test_e87_readability.py, test_e87b_run.py, test_e87b_check.py, test_knob_run.py, test_newtable_path.py, test_r24_check.py, test_r24_network_free.py, test_star_replies.py)
data/                 raw datasets and cached embeddings (not committed)
results/explore/      exploratory runs, validation splits only
results/confirm/      registered runs (test split)
logs/                 dev plans and dated notes (each frozen one with its freeze record), driver scripts; run logs, job lists and archive/ (local)
ops/                  the launch chains of e80 and e84 to e87 and of R24's dry run, with their tests
verification/         independent recomputes (R21, R24, e71; e81, e82 R4, e83 and e84, e86) and diagnostics
speed/                FastCoTM (exact speed-ups of the TM with byte-identical results) and a C clause evaluator
```

In the repository, `logs/` holds the dev plans and dated notes, the freeze record (`*.frozen.json`) of each one that was frozen, the driver scripts and one sample file that `docs/results.md` cites (`e71_k35_sample.json`), the earlier versions of four scripts that the plans cite by hash (`logs/pre_R19/`, `logs/pre_R20/`, `logs/pre_e77/`), and the run logs and job lists that the results pages cite by name; the raw run logs and job lists stay local, like the run folders in `results/`, and so do the earlier drafts and frozen copies that some plans cite under `logs/archive/`. `verification/` holds independent recomputations of registered results (R21 to R24) and of the e71 and e76 tallies on dev: the scripts and their recorded outputs, and the commands of the R20 report; `verification/README.md` lists each file and what it recomputes. `docs/data.md` is the data card (datasets, pretrained models and word tables).

## Environment

`uv venv --python 3.11 .venv-dl`, then torch 2.6.0 (cu124), sentence-transformers, transformers, datasets, scikit-learn, scipy, pandas, pyarrow, gymnasium, lightgbm.

GPU jobs run through `../tm_language_model/scripts/gpu_run.sh shared`.

No lock file of `.venv-dl` was kept: `env_dl_freeze.txt` lists its packages as found in the environment on 2026-10-23. `.venv-htm`, the environment of the hierarchical Tsetlin Machine package for R16, is listed in `env_htm_freeze.txt`.

## Published copies

The registrations and the dev plans and notes in `logs/` were edited for publication on 2026-10-23, in wording and paths only; each edited copy carries a publication note, and the freeze records keep the SHA-256 of the texts as registered (the [hash table](../../docs/hash-table.md) gives both values). The check scripts that compare a registration or a plan with the SHA-256 in its freeze record (`scripts/r19_check.py` to `scripts/r24_check.py` and `logs/R24_chain.sh`) therefore report a difference on these copies. Comments, docstrings, one log message and absolute path constants (now derived from the location of each file) were edited as well, and the pin `SAMPLE_SHA` of `scripts/e71_k35_sample.py` now holds the SHA-256 of the published `logs/e71_k35_sample.json`, whose description was reworded; the code is otherwise unchanged, and the hash table lists every edited file.

The tests were run on the published files on 2026-10-23, on the CPU with two threads: 23 of the 28 test files passed. `tests/test_tm.py`, `tests/test_tm_chunkbudget.py` and `tests/check_chunked_update.py` need the GPU and were not run. `tests/test_count_text.py` and `tests/test_static_text.py` stopped at a check that compared the code with an earlier state of itself. The first found `scripts/abcd_confirm.py` and `scripts/abcd_explore.py` changed since the freeze records of e70 and e71 that it read (one line each, for e77 on 2026-10-19; `logs/plan_e77_multiwoz_dev.md`), and `src/dl/abcd.py` and `src/dl/static_text.py` reworded for publication. The second expected the list of view parts in `src/dl/abcd.py` to end where it ended before view C was added (R20). Both checks failed before the edits for publication as well. Both were fixed on 2026-10-25: the first now compares the four files with the last freeze record that pins them (e87's) and their published values, and still checks the e70 and e71 records; the second pins the whole list of parts. Both files now pass on the CPU (`logs/test_count_text_2026-10-25.out`, `logs/test_static_text_2026-10-25.out`). `tests/test_static_text.py` also holds `test_default_view_pristine_gpu`, a GPU check that it runs only with `DL_TEST_GPU=1`; it was not run. Five tests were added on 2026-10-25: `tests/test_e87b_run.py` and `tests/test_e87b_check.py` for e87 Part B, and `tests/test_e80_check.py`, `tests/test_e80_dry_check.py` and `tests/test_e80_report.py` for the scripts of e80. `tests/check_chunked_update.py`, which printed statistics and could not fail, gained bars on 2026-10-25, taken from its comment and its recorded output of 2026-10-16. In the final test run of the finish pass all 33 test files pass: 30 on the CPU on 2026-10-25, `tests/test_r24_network_free.py` in its CPU mode `--selftest` (its GPU mode is a step of R24's chain and was not run), and the three that need the GPU on 2026-10-26.

Where the records of the study say that a gate, a text, a download or code was checked or reviewed independently, report the findings of a review or cite an internal audit, those checks, reviews and audits were judged by an AI model, not a person. The recomputations in `verification/` are scripts, and their recorded outputs are kept next to them.

## Rules kept

- Exploration reads validation splits only. A test split is read only by a registered run, once per run (the bandit runs score it after every pass for their learning curves; the registered number is the last pass).
- Every registration is frozen before its confirmatory runs. A change after freezing goes into a dated amendment file, which lists the source files that changed (`scripts/diff_frozen.py`).
- Every run that was started is reported, including the ones that lost.
