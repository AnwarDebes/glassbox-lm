# Reproduction check after the study (HTM-01): outcome

The freeze rule of gates.md was applied after the study ended, under the plan frozen on 25 Oct 2026 at 16:31:59 UTC (HTM-01 in [registrations after the study](registrations_after_the_study.md)). Its headline, in the plan's fixed form: **the freeze rule was applied to 124 of 683 page numbers (123 of class B, 1 of class D-sampled); this is not a general claim that the study's numbers reproduce.**

- **Page numbers:** all 124 re-checked page numbers reproduced, all 124 exactly. No number moves to the list of claims that are not made.
- **Retrains:** 14 of the 16 sampled retrains reproduced exactly. The other two did not reproduce under rule R1. They are the seed-1 news models of Check 2c (`s` 60, hierarchical and flat), and they differ on one field only, the per-text time `ms_per_text`. Every other number of the two runs equals the logged one exactly, and so do their include bits and data hashes. By rule R2, Check 2c is marked "not reproduced in the sample" on the [results page](results.md#check-2c-news-ag-news-word-groups).
- **Not re-checked:** the other 559 page numbers. 485 are mapped to runs that no GPU step re-ran, and 74 are not mapped to any run.

## How it ran

- The GPU steps ran in two stages of the finish pass's queue, one job at a time, each under `scripts/gpu_run.sh exclusive`:
  - stage r1, the 41 dev re-scores of step (i): 25 Oct 2026, 21:04:01 to 21:21:57 UTC;
  - stage r2r3, the 16 retrains of step (ii): from 21:35:55 UTC to 26 Oct 2026, 02:15:59 UTC. The smoke job `e1_verify_default` ran first, then tier 1 and tier 2.
- Every job was complete at its first attempt, and none reached its time cap.
- The comparison (`compare_after_gpu.py`, version 3) covered 57 jobs and 9,234 numbers: 9,204 exact, 9,205 within 0.02, none missing.
  - Step (i): 41 jobs, all exact (1,290 of 1,290 numbers). The rules files and include bits of the 36 re-scores are equal. The 5 shuffled-label re-scores equal the training log at pass 100.
  - Step (ii): 16 jobs. 14 are exact. The other two are the news retrains, with 261 of 276 numbers exact each.
  - Across the 16 retrains, the include bits and data hashes are equal. The VERIFY lines of the two Check 1a runs are equal. `src/compare_runs.py` finds identical training in both modes on the 14 runs of the pair driver; it does not read the news models' text driver.
- Section 3 of the plan sends a retrain that is not exact back to the jobs that read its folder:
  - `e5a_npmi` is the only step-(iii) job that reads the hierarchical news model. It ran again on the CPU with the new folder in place of the logged one (`run_iii.py --subst`), on 26 Oct 2026 from 02:19:50 to 02:20:17 UTC: 70 of 70 numbers exact.
  - No job reads the flat news model.
  - No step-(i) line scores either news model, so the conditional GPU request R2c added no job.
- Environment (rule R6): every new run names the plan's interpreter, decision_layer's `.venv-htm`, and holds code copies equal to the study's `src/` files. The pre-flight found the same GPU, driver and nvcc as the logged runs used. Each job had 2 threads.

## Outcome table

| class | page numbers | re-checked | exact | within 0.02 | not reproduced | not re-checked, with the reason |
|---|---|---|---|---|---|---|
| A | 83 | 0 | | | | 83: final test; only the step (iv) reading below |
| B | 125 | 123 | 123 | 0 | 0 | 2: their only candidates are per-pass history entries of metrics.json, which step (i) does not compare (0.55 on line 348, 0.711 on line 406) |
| C | 244 | 0 | | | | 244: CPU scorers; only the step (iii) reading below |
| D-sampled | 1 | 1 | 1 | 0 | 0 | 0 |
| D-not-sampled | 68 | 0 | | | | 68: training runs outside the sample (R5) |
| E | 94 | 0 | | | | 94: slide-6 and slide-10 tests; held-back inputs, evaluation splits scored once (R5) |
| G | 19 | 0 | | | | 19: data checks built with the HTM-B2 bug; their data were rebuilt and are gone (R5) |
| not mapped | 74 | 0 | | | | 74: no leaf of that value on the runs of their line (list below) |
| **all, each number once** | **683** | **124** | **124** | **0** | **0** | **559** |

- A number counts in the row of each class among its runs, so the class rows add up to more than the total.
- Classes A', H and X carry no mapped page number.
- The D-sampled number is line 171's dev AUC 0.745, from the retrain `20261001T212959Z_e4_sa_granmo_C50_A64_s100`.

**Rows A and C: readings under the unreviewed draft 1, run before the freeze; their outcomes were known when the plan was frozen; no rule.**

- A, step (iv), 25 Oct 2026, 11:52:31 to 11:55:11 UTC: all 529 numbers of the final report equal the logged ones exactly.
- C, step (iii), round 2: 26 jobs and 4,337 numbers, of which 4,298 exact and 4,330 within 0.02.
  - The 7 outside 0.02 are in excluded leaf types: three shares of Check 1b moved by the HTM-B2 fix, and four wall times.
  - 19 are missing by design: they need the test split.
  - 4 strings changed: the printed note on the gate method of `src/wiki_groups.py`.

## Every compared leaf outside 0.02

There are 29, all `ms_per_text`, the time per text that the text driver logs at each evaluated pass and in its FINAL line. None carries a page number. The first differing pass is pass 5 in both runs, the first pass that is evaluated.

| logged run | new run | outside 0.02 | logged (ms) | new (ms) | largest difference |
|---|---|---|---|---|---|
| `20261002T060802Z_e5a_screen_hier_s60` | `20261026T014616Z_htm01_retrain_e5a_screen_hier_s60_posthoc` | 14: every evaluated pass but 49, and FINAL | 0.3417 to 0.3474 | 0.2940 to 0.2984 | 0.0524 (pass 5) |
| `20261002T062647Z_e5a_screen_flat_s60` | `20261026T020150Z_htm01_retrain_e5a_screen_flat_s60_posthoc` | 15: all 14 evaluated passes and FINAL | 0.3205 to 0.3241 | 0.2645 to 0.2690 | 0.0584 (pass 45) |

- The hierarchical model's pass 49 moved from 0.3016 to 0.2951, within 0.02. It is the only number of all 57 jobs that is within 0.02 but not exact.
- Possible cause, the environment (R6): the logged runs ran under `scripts/gpu_run.sh shared`, and the retrains under `exclusive`. The retrains took 934 s and 843 s against 1,082 s and 1,015 s logged.
- A reading with no rule: the plan's list of time fields (`fit_s`, `ms_per_pair`, `eval_s`, `wall_s`, `runtime_s`, `wall_time_s`, `poll_s`) names the pair driver's `ms_per_pair` but not the text driver's `ms_per_text`, so the comparison counted it. With `ms_per_text` set aside, both news retrains are exact in all of their other 261 numbers. This reading changes no outcome under the rules.

## The 74 tokens that are not mapped, by line

Line numbers are those of the current page. The map was built on version e9ca0667 of the page, on which lines from 224 on stood 2 lower.

19: 0.95, 0.98; 34: 0.50; 47: 0.500, 0.499, 0.492, 0.506, 0.510, 0.509, 0.50; 53: 0.90, 0.80, 0.85; 69: 0.9035, 0.8965, 0.90, 0.305, 0.900, 0.8985, 0.875, 0.875, 0.935, 0.575; 71: 21.1; 77: 0.859, 0.819, 0.841; 83: 0.8; 138: 0.70; 150: 0.788; 151: 0.820; 161: 0.65, 0.60, 0.58, 0.2, 0.8, 0.55; 188: 0.842; 235: +0.033, -0.009; 237: +0.005, -0.002, 0.83; 241: 0.5, 0.50; 251: +0.014, +0.013, +0.003, +0.015, +0.014, +0.001, 0.29, +0.032, 0.493; 255: 58.6; 305: 0.500; 307: 0.811; 311: 0.02; 333: 0.70; 343: 0.95; 345: 0.80; 361: 1.3; 366: 0.866; 374: 0.678, 0.680, 0.701, 0.704; 380: 0.918, 0.896, 0.900; 382: 0.510; 393: 0.70; 408: 0.02; 416: 0.575.

## Predictions of the plan (section 7)

- Step (i), exact: held, 41 of 41 jobs.
- Step (ii), exact:
  - held for 14 of 16 retrains;
  - the two news retrains failed R1 on their time field alone, and their training matched in every other compared value.
- Time caps: no run came near its cap.

## Checks and limits

- **Independent recompute: MATCH.** It was written from the plan's definitions, not from the comparator's code. It reads the logs, include bits, data hashes and rules files of all 57 jobs, and gives the comparator's per-job counts on 57 of 57 jobs. Its first try left the 9 numbers of the VERIFY line out of the totals of the two Check 1a jobs; that try is kept in the finish-pass records.
- **Outcome table.** It was built by a script that applies the join keys of section 9; the script's test passes 7 of 7.
- The checks were run and read by an AI model, not a person.
- **Timing and scope.** The check ran 21 and 22 days after its due date, 4 Oct 2026, and after the study ended, which is a deviation. It re-checks 124 of 683 page numbers, on the same inputs and the same GPU type. It is not a general reproduction.
- **Multiplicity.** No multiplicity correction is applied; each number stands alone.
