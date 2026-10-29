# Results of the tally study

The study has 125 run folders: 66 of 2026-10-15, 15 of 2026-10-16, 10 of 2026-10-17, 15 of 2026-10-25 and 19 of 2026-10-26 (UTC, by run id). The first run started at 07:10:15 UTC on 2026-10-15 (`run.json` of `20261015T071015Z_b6timing_fmnist_thresh10_c100`), and the last ended at 09:07:03 UTC on 2026-10-26 (`run.json` of `20261026T083523Z_c3prep_lstm_large_seed10_valid_sum_steps_clip10`). The records give 121 runs as finished, 3 as crashed and 1 as running without an end time (`run.json`, key `state`). The [run index](../../../docs/run-index.md) lists every folder. The run folders and the run logs are not distributed, so the file names serve readers who hold them.

Every number on this page is read from a file of the run that is named next to it. The times of the freezes are read from the freeze records in `registrations/`, and values quoted from the registration or its amendments are marked as such. The dev-test numbers on this page are those of the registered report of T0, computed once on 2026-10-25, and of the two Kneser-Ney models that the registration reports beside it ([Verdict](#verdict-computed-on-2026-10-25)), and those of the registered look of A2 on the same day ([A2](#a2-the-registered-look-on-dev-test-2026-10-25)). Values that the amendments compare are given to the six decimals that the amendments print.

The question of the study, its lines of work and its dependencies are described in the [README](../README.md).

## Status of the registered ids

Four ids carry a status: T0 and its two amendments, which fix the settings of the two machines by the tuning rule of T0, and A2. The statuses are met, not met, not judged and not run; for an amendment, which is a step of the procedure, met means that the step was carried out as registered. Of the four ids, three are met and one is not met.

| Id | Registered content | Status | Qualifier | Run folders |
|---|---|---|---|---|
| T0 ([registration](../registrations/T0.md)) | Penn Treebank next-word prediction with tmu at commit 5605ff0. Claim-carrying comparison: the counted read-out on the clauses of the coalesced machine (arm 2) against its learned votes (arm 1), in bits per word on dev-test, with each arm averaged per position over seeds 1 to 10 and a 95% interval from 1,000 resamples of the 21 dev-test blocks. Further verdict line: arm 2 against the untrained bank (arm 3). Reported with intervals and no verdict: arm 2 against logistic regression on the clause outputs (arm 4), and the counted against the learned read-out of the flat machine. | not met: learned beats counted | computed once on 2026-10-25 by the frozen `src/tally/t0/report.py`: arm 2 minus arm 1 is +0.251 bits per word (+0.222 to +0.280), so the learned votes beat the counted read-out, on every seed; the line "Tsetlin learning helps the counted read-out" is not met (arm 2 minus arm 3 +0.113, +0.104 to +0.120); the per-seed dev-test means had been logged by the final runs and read before the report, and none was used for a choice; independent recompute MATCH | the 30 runs of the [final runs](#final-runs) and the report run `20261025T104315Z_t0_report_devtest` ([Verdict](#verdict-computed-on-2026-10-25)) |
| Amendment of the coalesced settings ([text](../registrations/T0_amendment_coalesced_settings.md)), frozen at 18:24:29 UTC on 2026-10-15 | s = 20 and 9 epochs, chosen by the tuning rule of T0 | met | frozen after the three coalesced tuning runs had ended and before the first coalesced final run; the chosen s is the largest value of the registered grid {5, 10, 20}, and its best epoch, 9, is one below the cap of 10 epochs | the three coalesced runs of the [tuning runs](#tuning-runs) |
| Amendment of the flat settings ([text](../registrations/T0_amendment_flat_settings.md)), frozen at 21:14:27 UTC on 2026-10-15 | s = 5 and 1 epoch, chosen by the tuning rule of T0 | met | frozen after the three flat tuning runs had ended and before the first flat final run; the chosen s is the smallest value of the grid, and all three flat tuning runs had their best learned score after the first epoch; the flat tuning run with s = 10 started before T0 was frozen, the other two after it | the three flat runs of the [tuning runs](#tuning-runs) |
| A2 ([registration](../registrations/A2.md)), frozen at 13:38:45 UTC on 2026-10-25 | Penn Treebank dev-test, count floor (no Tsetlin part): the class members of the exchange code added to a Kneser-Ney 5-gram with a cache; success when the perplexity is at least 5% lower than without class members, at least 1% lower than with the mean of 10 random codes of the same class sizes, and the block interval against the random mean is below 0; kill under 2% | met | one look, the 2nd use of dev-test; gain 7.59% (95% block interval 6.51% to 8.58%, without train-side noise); exploratory under design rule 12 (one build, one split, one benchmark); the random codes are switched off by EM, so the control is in effect the system without classes | `20261025T133912Z_a2_devtest_final` |

T0 was frozen at 07:27:25 UTC on 2026-10-15 (`T0.frozen.json`). The last coalesced tuning run ended at 18:22:35 UTC, the coalesced amendment was frozen at 18:24:29 UTC, and the first coalesced final run started at 18:24:30 UTC. The last flat tuning run ended at 21:10:33 UTC, the flat amendment was frozen at 21:14:27 UTC, and the first flat final run started at 16:59:43 UTC on 2026-10-16 (`run.json` of the runs in the tables below). By the registration, the verdict is computed once, after all 10 seeds of both machines have finished. Every run that the verdict reads had ended by 20:58:57 UTC on 2026-10-17, and the report ran at 10:43:15 UTC on 2026-10-25.

## Run folders by line

| Line | Run folders | Finished | Crashed | Recorded as running | What they are |
|---|---|---|---|---|---|
| T0 | 46 | 43 | 3 | 0 | 2 smoke runs, 4 starts of the logistic-regression arm for a smoke run, of which 3 crashed, 1 smoke run of the report, 6 tuning runs, 20 final runs of the two machines, 10 logistic-regression runs on the coalesced final runs; on 2026-10-25 the registered report and the two Kneser-Ney runs beside it (valid and dev-test) |
| B1 | 15 | 14 | 0 | 1 | 12 smoke runs at team size M = 8 and 3 smoke runs of the member calibration |
| B6 | 9 | 9 | 0 | 0 | timing runs on the training split |
| A1 | 5 | 5 | 0 | 0 | 4 runs of the baseline check and 1 run of the A/A noise (2026-10-25) |
| A2 | 9 | 9 | 0 | 0 | 5 smoke runs, 3 development runs on valid and the registered look on dev-test (2026-10-25) |
| C3 | 41 | 41 | 0 | 0 | LSTM runs on the validation split, two sizes, seeds 1 to 10: 20 runs of 2026-10-15, and on 2026-10-25 and 26 a smoke run of 6 epochs and 20 reruns with the loss and the clips of the paper |
| All | 125 | 121 | 3 | 1 | |

The states are those of `run.json` in each folder; the line is the one that the run name gives.

## T0

### Tuning runs

All six runs ran on 2026-10-15 with seed 1, an epoch cap of 10 and a patience of 3 epochs (`config.json`), and are recorded as finished. The best score is the lowest bits per word of the learned arm on the validation split over the epochs (`metrics.json`, keys `best_learned_valid_bpw` and `best_epoch`); the epochs trained are the entries of `history` in the same file.

| Machine | s | Run id | Best learned bits per word on valid | Best epoch | Epochs trained | Start and end (UTC) |
|---|---|---|---|---|---|---|
| coalesced | 5 | `20261015T072026Z_t0_tune_coal_c2000_s5_seed1` | 9.243282 | 1 | 4 | 07:20:26 to 15:17:04 |
| coalesced | 10 | `20261015T072026Z_t0_tune_coal_c2000_s10_seed1` | 9.084931 | 4 | 7 | 07:20:26 to 17:44:11 |
| coalesced | 20 | `20261015T072026Z_t0_tune_coal_c2000_s20_seed1` | 9.057047 | 9 | 10 | 07:20:26 to 18:22:35 |
| flat | 5 | `20261015T113259Z_t0_tune_flat_c100_s5_seed1` | 10.634485 | 1 | 4 | 11:32:59 to 18:10:20 |
| flat | 10 | `20261015T072535Z_t0_tune_flat_c100_s10_seed1` | 11.115641 | 1 | 4 | 07:25:35 to 11:32:49 |
| flat | 20 | `20261015T181024Z_t0_tune_flat_c100_s20_seed1` | 11.441973 | 1 | 4 | 18:10:24 to 21:10:33 |

By the tuning rule of the registration the lowest best score of each machine fixes s, and its best epoch the number of epochs: s = 20 with 9 epochs for the coalesced machine and s = 5 with 1 epoch for the flat machine. These are the settings in the two amendments, and the values that the amendments print agree with `metrics.json` of these runs. Five runs stopped after 3 epochs without a better score (`log.txt`); the coalesced run with s = 20 trained all 10 epochs. The tuning records also hold, for every epoch, the scores of the counted read-out on the calibration and the validation split (`history`); the tuning rule does not use them.

The tuning runs record no registration (`config.json`, key `registration` is null). The three coalesced runs and the flat run with s = 10 started before T0 was frozen at 07:27:25 UTC; the flat runs with s = 5 and s = 20 started after it, each when the previous flat run had ended. The registration gives 07:22 UTC as the start of the flat tuning; the record of the first flat tuning run starts at 07:25:35 UTC, and the flat smoke run started at 07:21:46 UTC.

Both amendments name one change of the code after the freeze of T0, a change of log text only: `src/tally/t0/run.py` printed 418 dropped positions per epoch also for the flat machine, whose trainer drops none, and prints 0 for it since. The second line of `log.txt` shows this: 418 in `20261015T072535Z_t0_tune_flat_c100_s10_seed1`, which started before the freeze, and 0 in `20261015T113259Z_t0_tune_flat_c100_s5_seed1` and `20261015T181024Z_t0_tune_flat_c100_s20_seed1`. The source records of the flat run with s = 10 and of the two later flat runs differ in 15 files, among them `src/tally/t0/run.py`; the records of the runs with s = 5 and s = 20 are identical (`source_hashes.json`).

### Final runs

All 30 runs are recorded as finished (`run.json`). The settings are read from `config.json` of each run.

The launchers of T0, `scripts/t0_auto.py` and `scripts/t0_final.sh`, start every final run at nice 10 in one CPU slot (`cpu_slot.sh 1`), all ten seeds of a machine at once (`t0_auto.py`) or all twenty Tsetlin runs at once (`t0_final.sh`), with no check of the CPU load. That is how the recorded runs were started, so the two files stay as they are; they do not follow the machine rule adopted later (nice 19, at most 2 threads per job, a start only when `../decision_layer/scripts/cpu_load.py 10` reads at most 4.5 cores) and are not to be reused as they stand. The launchers of B1 and B6 were changed to that rule on 2026-10-25.

| Seed | Coalesced machine: 2,000 clauses, s 20, 9 epochs | Flat machine: 100 clauses per class, s 5, 1 epoch | Logistic regression on the clause outputs of the coalesced run |
|---|---|---|---|
| 1 | `20261015T182430Z_t0_final_coal_c2000_s20_seed1` | `20261017T031001Z_t0_final_flat_c100_s5_seed1` | `20261016T173933Z_t0_logreg_of_final_coal_c2000_s20_seed1` |
| 2 | `20261015T182432Z_t0_final_coal_c2000_s20_seed2` | `20261016T165943Z_t0_final_flat_c100_s5_seed2` | `20261016T165204Z_t0_logreg_of_final_coal_c2000_s20_seed2` |
| 3 | `20261015T182434Z_t0_final_coal_c2000_s20_seed3` | `20261016T202538Z_t0_final_flat_c100_s5_seed3` | `20261016T172331Z_t0_logreg_of_final_coal_c2000_s20_seed3` |
| 4 | `20261015T182436Z_t0_final_coal_c2000_s20_seed4` | `20261016T172503Z_t0_final_flat_c100_s5_seed4` | `20261016T170800Z_t0_logreg_of_final_coal_c2000_s20_seed4` |
| 5 | `20261015T211037Z_t0_final_coal_c2000_s20_seed5` | `20261016T184257Z_t0_final_flat_c100_s5_seed5` | `20261016T191529Z_t0_logreg_of_final_coal_c2000_s20_seed5` |
| 6 | `20261016T170310Z_t0_final_coal_c2000_s20_seed6` | `20261017T081705Z_t0_final_flat_c100_s5_seed6` | `20261017T095156Z_t0_logreg_of_final_coal_c2000_s20_seed6` |
| 7 | `20261016T191339Z_t0_final_coal_c2000_s20_seed7` | `20261017T083834Z_t0_final_flat_c100_s5_seed7` | `20261017T113732Z_t0_logreg_of_final_coal_c2000_s20_seed7` |
| 8 | `20261017T091936Z_t0_final_coal_c2000_s20_seed8` | `20261016T225440Z_t0_final_flat_c100_s5_seed8` | `20261017T204300Z_t0_logreg_of_final_coal_c2000_s20_seed8` |
| 9 | `20261016T205928Z_t0_final_coal_c2000_s20_seed9` | `20261016T191524Z_t0_final_flat_c100_s5_seed9` | `20261017T131723Z_t0_logreg_of_final_coal_c2000_s20_seed9` |
| 10 | `20261016T165201Z_t0_final_coal_c2000_s20_seed10` | `20261017T052825Z_t0_final_flat_c100_s5_seed10` | `20261017T100805Z_t0_logreg_of_final_coal_c2000_s20_seed10` |

Each final run of a Tsetlin machine records the path of `registrations/T0.md` (`config.json`, key `registration`) and the SHA-256 of that file as it was frozen, `78d408591d89c24fbc3a92f0b20784ea3f355b882f82be4b3ec14cd6ac7dc409` (`source_hashes.json`). Each logistic-regression run names the coalesced final run that it reads (`config.json`, key `source_run`). The 30 runs record the same SHA-256 for every file under `src/`, equal to the values in both amendment records (`source_hashes.json`). So do the 22 other runs that started at or after 09:37:06 UTC on 2026-10-15: the flat tuning runs with s = 5 and s = 20 and the 20 LSTM runs of C3.

The final runs of the Tsetlin machines store their bits per position (`bits.npz`), and the coalesced runs also the clause outputs of the four splits (`fired_<split>.npz`), which the logistic-regression runs read. Each final run also writes the mean bits per word of every arm on every split, dev-test included, into its `metrics.json` and the METRICS line of its `log.txt`, as `src/tally/t0/run.py` did when it was frozen. These per-seed dev-test means were read before the registered report ran (first on 2026-10-17), and a project note of 2026-10-21 quotes both directions of the comparison. None of them was used for a choice: the settings had been fixed by the tuning rule on valid before the first final run, every seed has one folder, and `report.py` and `run.py` are unchanged since the freeze.

### Verdict (computed on 2026-10-25)

The registered statistic was computed once, at 10:43:15 UTC on 2026-10-25, by `src/tally/t0/report.py` over the 30 final runs listed above, in the order of the table (run `20261025T104315Z_t0_report_devtest`, `metrics.json`; its `config.json` lists the 30 folders). `report.py` and `src/tally/runlog.py` have the SHA-256 of the freeze records. Dev-test has 46,411 positions in 21 blocks of 100 sentences.

| Arm | Bits per word on dev-test (mean of seeds 1 to 10) | Perplexity | Per seed, lowest and highest |
|---|---|---|---|
| 1, learned votes of the coalesced machine | 9.1598 | 572.0 | 9.1031 to 9.1866 |
| 2, counted read-out on the same clauses | 9.4110 | 680.8 | 9.3907 to 9.4514 |
| 3, untrained banks with the counted read-out (mean of 10 draws) | 9.2984 | 629.7 | 9.2896 to 9.3072 |
| 4, logistic regression on the clause outputs | 8.2797 | 310.8 | 8.2694 to 8.2892 |
| flat machine, learned votes | 10.7443 | 1,715.4 | 10.7267 to 10.7656 |
| flat machine, counted read-out | 9.8513 | 923.7 | 9.8454 to 9.8545 |

| Comparison (registered line) | Mean difference, bits per word | 95% interval (1,000 block resamples) | Reading |
|---|---|---|---|
| arm 2 minus arm 1 (claim-carrying) | +0.2512 | +0.2220 to +0.2801 | learned beats counted |
| arm 2 minus arm 3 ("Tsetlin learning helps the counted read-out") | +0.1126 | +0.1041 to +0.1205 | not met |
| arm 2 minus arm 4 (reported, no verdict) | +1.1313 | +0.9923 to +1.2819 | no verdict |
| flat counted minus flat learned (reported, no verdict) | -0.8930 | -0.9387 to -0.8440 | no verdict |

- **T0 is not met in its claim-carrying line: on the coalesced machine of tmu, the learned votes beat the counted read-out of the same clauses.** Every seed has the same sign (per-seed differences +0.220 to +0.308). The line "Tsetlin learning helps the counted read-out" is not met: the counted read-out of the trained bank is worse than that of the untrained banks, on every seed (+0.086 to +0.162) and in every block; that direction has no registered line.
- `metrics.json` of the report also writes a reading for the two comparisons registered without a verdict, and the field `tsetlin_learning_helps_counted_readout`, which `report.py` computes from a second bootstrap draw (its interval, +0.1044 to +0.1204, agrees with the table). Neither is a verdict beyond the two registered lines.
- H8.1 of tm_language_model (its own engine, WikiText-103, count read-out 1.8280 against signed votes 2.1749 bits per byte) is therefore not shown for the coalesced machine of tmu on Penn Treebank. T0 differs from H8.1 at once in the engine, the count read-out (H8.1's has a prior term and a weight per clause group), the number of clauses, the output side, the unit, the corpus and the split, and it cannot say which difference matters.
- The flat machine's comparison carries no verdict, and both of its read-outs are worse than an add-one unigram counted on train (valid: counted 9.7937, learned 10.6448, unigram 9.4218 bits per word; calib: 9.8747, 10.7431 and 9.4862; means of seeds 1 to 10, from `metrics.json` of the final runs). On valid, the coalesced machine's learned votes are 0.35 and its counted read-out 0.07 bits per word below the unigram.
- **Beside, not compared**, as registered: interpolated modified Kneser-Ney with counts from train (the code of step A1, which matches KenLM on valid), sentence mode, on the same dev-test: 3-gram 7.2847 bits per word (perplexity 155.9) and 5-gram 7.1760 (144.6) (run `20261025T104406Z_t0_kn_beside_devtest`, `metrics.json`; the section-4 contract passed on 1,000 positions of each). For placement only, not a registered comparison: every Tsetlin arm is 1.98 to 3.57 bits per word above the 5-gram. The script, `scripts/t0_kn_beside.py`, was written on 2026-10-25, after the freeze and before the report, and has no free parameter (the orders as registered, counts from train, sentence mode as in A1). Its smoke on valid (run `20261025T104107Z_t0_kn_beside_valid`) reproduces the A1 baseline check's own KN3 and KN5 valid bits per word exactly (7.34256972151113 and 7.255057512770932).

### Checks of the verdict

- **Independent recompute** (`verification/recompute_t0.py`, own code, generator PCG64 with seed 20261025, 10,000 block resamples; its synthetic test `verification/test_recompute_t0.py` passes 8 checks): every arm's bits per word and every mean difference equal the report's, the interval ends differ by at most 0.0065 bits per word, and the readings agree. It also checks the 30 folders (one per seed, finished, the registered settings, started after the amendments, the same source hashes as the amendment records, each logistic-regression run on its own seed's bank, identical dev-test blocks). The two-level interval (seeds, then blocks; no rule) of arm 2 minus arm 1 is +0.218 to +0.285. Its recorded output is `verification/recompute_t0.json`.
- **Three read-only reviews** (leakage; selection and registration; integrity; judged by an AI model, not a person) found no blocker. Their findings, with what follows from them:
  - The splits are blocks of 100 sentences of the training file, not separate documents, so articles continue across block edges into train; 45 dev-test sentences (2.1%, 388 tokens) also occur verbatim in train, mostly market boilerplate. On a subset without block-edge sentences and without these duplicates (37,789 positions), arm 2 minus arm 1 is +0.2445 (+0.2113 to +0.2795); without the duplicates alone, +0.2489 (+0.2194 to +0.2787). No reading changes.
  - The vocabulary is the fixed 10,000-word list of the Mikolov preprocessing, built from the whole training file, dev-test blocks included; 8 of its words occur only in dev-test (63 positions). The class list and the unigram u are counted on train only, and u is the same in every arm, so it cancels in every paired difference.
  - The run records hold no hash of the training text, because `runlog.py` keys the data hashes by file name and the train and validation texts are both named `text.utf8`; the training text equals the hash in `tm_language_model/data/ptb/raw/train/meta.json`, and a rebuild of the splits from it reproduces the stored dev-test words and blocks exactly.
  - The logistic-regression runs check that the distribution sums to one and that its minimum is above 0 on 1,000 positions per split, but not the third check of the registered contract (that the fast path equals the vector entry).
  - The three coalesced tuning runs started before the freeze and ran an earlier read-out; the final run of seed 1, on the frozen code, reproduces tuning epoch 9 on valid to 1e-6 bits per word, and choosing by the counted arm instead would have picked the same setting.
- **Post hoc, on valid only (no rule; dev-test not read again):** all 120 counted read-outs of the final runs chose alpha 0.01, the smallest value of the registered grid. Refitting the counted read-out of seeds 1 to 3 with alpha down to 1e-5 (chosen on calib: 1e-5) moves arm 2 minus arm 1 on valid from +0.281, +0.265 and +0.280 to +0.261, +0.247 and +0.261 (`verification/explore_alpha_valid.py`, outputs `verification/explore_alpha_valid_*.json`). The grid edge accounts for about 0.02 of the 0.25 bits per word.

### Smoke runs, crashed runs and the smoke run of the report

| Run id | What it is | State and cause | File that holds it |
|---|---|---|---|
| `20261015T071247Z_t0_smoke_coal_c200_s10_seed1` | `src/tally/t0/run.py` in mode `smoke`: coalesced machine with 200 clauses, s 10, 2 epochs | finished | `config.json`, `run.json` |
| `20261015T072146Z_t0_smoke_flat_c20_s10_seed1` | `src/tally/t0/run.py` in mode `smoke`: flat machine with 20 clauses per class, s 10, 2 epochs | finished | `config.json`, `run.json` |
| `20261015T072425Z_t0_logreg_t0_smoke_coal_c200_s10_seed1` | `src/tally/t0/logreg.py` on the coalesced smoke run | crashed: a CUDA error, an illegal memory access | `log.txt`, `run.json` |
| `20261015T072500Z_t0_logreg_t0_logreg_t0_smoke_coal_c200_s10_seed1` | `src/tally/t0/logreg.py` started on the folder of the crashed run above, which holds no clause outputs | crashed: `fired_train.npz` not found | `log.txt`, `run.json` |
| `20261015T072511Z_t0_logreg_t0_smoke_coal_c200_s10_seed1` | `src/tally/t0/logreg.py` on the coalesced smoke run, second start | crashed: the same CUDA error | `log.txt`, `run.json` |
| `20261015T072549Z_t0_logreg_of_smoke_coal_c200_s10_seed1` | `src/tally/t0/logreg.py` on the coalesced smoke run | finished | `run.json` |
| `20261015T072640Z_t0_report_valid_smoke` | `src/tally/t0/report.py` on the validation split of the coalesced smoke run, with no flat run and no logistic-regression run | finished | `config.json` (keys `runs`, `split`, `flat` and `logreg`), `run.json` |

The source records of the last crashed run and of the finished logistic-regression run differ in `src/tally/t0/logreg.py` and `src/tally/b1/team.py` (`source_hashes.json`). The folder `20261015T072640Z_t0_report_valid_smoke` holds a record whose run id is `20261015T072640Z_t0_report_valid` (`run.json`, key `run_id`): the suffix `_smoke` is in the folder name only. The numbers of that run come from a smoke run with 200 clauses and 2 epochs and are not results.

## B1

All 15 runs are labelled smoke: 12 runs of `src/tally/b1/run.py` at team size M = 8 and 3 runs of the member calibration `src/tally/b1/calib.py`, with the labels `smoke_calib` and `smoke_calib2` (`run.json`, key `argv`; `config.json`). No B1 run reads a data file (`data_hashes.json` is empty); the game is generated by the code.

| Run id | Arm | Seed | Cap (`--cap`) | State |
|---|---|---|---|---|
| `20261015T071713Z_b1_smoke_calib` | calibration, label `smoke_calib` | | | finished |
| `20261015T072321Z_b1_smoke_calib` | calibration, label `smoke_calib` | | | finished |
| `20261015T072457Z_b1_smoke_dual_M8_s0` | `dual` | 0 | 30,000 | finished |
| `20261015T072526Z_b1_smoke_greedy_M8_s0` | `greedy` | 0 | 100,000 | finished |
| `20261015T072534Z_b1_smoke_random_M8_s0` | `random` | 0 | 100,000 | finished |
| `20261015T072536Z_b1_smoke_oracle_M8_s0` | `oracle` | 0 | 100,000 | finished |
| `20261015T072600Z_b1_smoke_flat_tune_M8_s10000` | `flat_tune` | 10000 | 40,000 | finished |
| `20261015T072630Z_b1_smoke_flat_M8_s0` | `flat` | 0 | 30,000 | recorded as running, no end time |
| `20261015T072811Z_b1_smoke_ab_M8_s0` | `ab` | 0 | 30,000 | finished |
| `20261015T072816Z_b1_smoke_shared_M8_s0` | `shared` | 0 | 30,000 | finished |
| `20261015T072818Z_b1_smoke_selfish_M8_s0` | `selfish` | 0 | 30,000 | finished |
| `20261015T072820Z_b1_smoke_marginal_M8_s0` | `marginal` | 0 | 30,000 | finished |
| `20261015T073241Z_b1_smoke_flat_M8_s0` | `flat` | 0 | 30,000 | finished |
| `20261015T073554Z_b1_smoke_dual_M8_s0` | `dual` | 0 | 30,000 | finished |
| `20261015T074132Z_b1_smoke_calib2` | calibration, label `smoke_calib2` | | | finished |

Run `20261015T072630Z_b1_smoke_flat_M8_s0` has no end time and no `metrics.json`; its `log.txt` holds two lines, both written at 07:26:30 UTC, and the record gives no cause. The same command line ran again as `20261015T073241Z_b1_smoke_flat_M8_s0`, which finished (`run.json`, key `argv`). The source records of the two runs differ in several files, among them `src/tally/b1/flat.py` (`source_hashes.json`).

### Status of B1 (2026-10-25): no registration, four open problems

The smoke runs are one seed at team size 8 and carry no claim. Every arm except the oracle stayed censored, that is, it did not come within 0.01 bits of the entropy before its cap (`metrics.json`, keys `censored` and `final_excess`): final excess over the entropy 0.1319 bits for `marginal`, 0.4691 `shared`, 0.4247 `selfish`, 0.4053 and 0.4695 for the two `dual` runs, 0.3980 `ab`, 0.0523 `flat`, 0.4085 `greedy` and 0.4765 `random`, and 0.0000 for `oracle`, which is not censored.

Before a registration, four problems have to be settled (also listed in `src/tally/b1/README.md`):

- **The shared reward never reached the target in the calibration.** In `20261015T074132Z_b1_smoke_calib2`, which ran the current `src/tally/b1/calib.py` (`source_hashes.json`), all 60 runs of the shared rule (12 grid points of s and RHO, 5 seeds each) are censored at 1,000,000 samples; the selfish and marginal rules reached the target in all 40 of their runs at the 8 grid points with s at most 3.9 and were censored at s = 6.0 (`metrics.json`, key `grid`). The earlier calibration `20261015T072321Z_b1_smoke_calib`, run with an earlier `calib.py`, has the first five runs of each of its 9 grid points censored (45 of 45); which rule they used is not recorded (UNVERIFIED). A primary ratio shared over marginal would rest on a comparator that does not learn on this game.
- **The team constants were not set by the calibration.** `team.default_params()` uses s = 3.9 and RHO = 0.25; the three calibrations chose s = 2.5 with RHO = 1.0, s = 3.9 with RHO = 1.0, and s = 2.5 with RHO = 0.5 (`metrics.json`, key `best`), and RHO = 0.25 is in none of their grids ({1, 4, 16} and {0.5, 1, 2, 4}, `config.json`).
- **The caps are provisional.** `run.CAPS` is 2,000,000 samples for every team size, set by hand; no timing run fixed it. Since 2026-10-25 `run.CAPS_FROM` (unset) has to name the timing run before a registered run starts.
- **The dual-state success line has no interval condition** (reduction at least 0.25 and never worse), while the primary line has one.

Fixed on 2026-10-25, with tests: `src/tally/b1/report.py` no longer reads censored runs as a dual-state kill (before, dual and marginal runs all at the cap gave a reduction of 0, read as "kill"; now success is blocked when dual runs are censored and kill when marginal runs are; `tests/test_b1_report.py`), and `scripts/b1_launch.sh` follows the machine rule (nice 19, thread pools at 1, the CPU gate before every start, and since 26 Oct a gate reading that fails counts as closed; `tests/test_launchers.py`). A registration also needs a timing-labelled calibration and timing runs at M = 128 and 512, and pins `scripts/b1_launch.sh` as an extra file. The grid of 637 runs (7 sizes x 9 arms x 10 seeds + 7 flat tunings) is estimated at about 320 CPU-hours for the team and search arms and 40 to 100 for the flat arm, extrapolated linearly in team size from the smoke speeds (`us_per_sample` in `metrics.json`: 32.8 for `marginal`, 32.0 `shared`, 44.5 `selfish`, 77.6 and 101.2 `dual`, 117.1 `ab`, 60.6 `greedy`, 10.5 `random`, 61.8 `oracle`, 868.1 `flat` microseconds per sample); UNVERIFIED.

## B6

All 9 runs are timing runs of `src/tally/b6/timing.py`, which reads only the training split, and all are recorded as finished. Each records the SHA-256 of the data manifest of its dataset (`data_hashes.json`, key `manifest.json`). The member or view and the number of clauses per class are read from `config.json`. The runs with 300 and 400 clauses per class have the input and the size of the arm `single` of `src/tally/b6/README.md`, one machine with the clauses of the whole team.

| Run id | Dataset | Member or view | Clauses per class |
|---|---|---|---|
| `20261015T071015Z_b6timing_fmnist_thresh10_c100` | Fashion-MNIST | `thresh10` | 100 |
| `20261015T071135Z_b6timing_fmnist_thermo3_c100` | Fashion-MNIST | `thermo3` | 100 |
| `20261015T071309Z_b6timing_fmnist_thermo4_c100` | Fashion-MNIST | `thermo4` | 100 |
| `20261015T071457Z_b6timing_fmnist_hog_c100` | Fashion-MNIST | `hog` | 100 |
| `20261015T071523Z_b6timing_agnews_word1_c100` | AG News | `word1` | 100 |
| `20261015T071752Z_b6timing_agnews_word2_c100` | AG News | `word2` | 100 |
| `20261015T072127Z_b6timing_agnews_char4_c100` | AG News | `char4` | 100 |
| `20261015T072257Z_b6timing_agnews_concat_c300` | AG News | `concat`, the three views together | 300 |
| `20261015T073805Z_b6timing_fmnist_thresh10_c400` | Fashion-MNIST | `thresh10` | 400 |

### Status of B6 (2026-10-25): no registration, no run of the team arms

The prepared data (`data/b6/`, with manifests) and the nine timing runs exist. The arms ruleT, nocomm, coupled, seqboost, fusion and gate have never run, not even as a smoke run, and no tuning run exists. `src/tally/b6/config.py` leaves open the T multiplier per member (`T_SCALE` is empty), the epochs ("set before the registration is frozen") and the base member of the single machine ("fixed by the tuning run"). The success and kill lines of the design are in `src/tally/b6/report.py` (success: a mean gain of at least 0.5 points with the interval above 0 on one dataset and the other's interval not entirely below 0; kill: under 0.2 points on both). The dev-test sets, the official test files, have never been scored. A registration needs smoke runs of every arm, the tuning on valid, and then the grid of 20 (dataset, seed) pairs, estimated at about 145 CPU-hours for Fashion-MNIST and 86 for AG News, plus 30 to 80 for the tuning, from the timing runs (UNVERIFIED). Since 2026-10-25 the launcher starts its jobs at nice 19 with the thread pools at 1, each after the CPU gate (`tests/test_launchers.py`).

## A1 and A2

A1 has three runs of `scripts/a1_baseline_check.py`, all recorded as finished: `20261015T071028Z_a1_baseline_check`, `20261015T071151Z_a1_baseline_check` and `20261015T072513Z_a1_baseline_check`. Their source records differ (`source_hashes.json`). The third records settings that the first two lack, among them that the cache settings are chosen on the validation split and the mixing weights fitted on the calibration split (`config.json`, keys `cache_select` and `em_fit`). The check scores the validation split and not dev-test; it is a baseline measurement without a registration, and this page gives none of its numbers. A fourth run, `20261025T113821Z_a1_baseline_check`, ran the same script on 2026-10-25 with the KenLM binaries recorded (`config.json`, key `kenlm`): its validation table, KenLM comparison, cache choice, duplicate report and bootstrap examples equal those of the third run, and the KenLM files it rewrote are byte-identical.

**The A/A noise runs of A1** (`scripts/a1_aa_noise.py`, run `20261025T113715Z_a1_aa_noise`, valid only, no rule). The count models are deterministic, so a repeat drops a seeded 5% of the train blocks (19 of 379) and rebuilds the systems of the baseline check; repeat 0 keeps every block and reproduces the third baseline check (differences at most 6e-14 in perplexity). Over repeats 1 to 10 (`metrics.json`, keys `spread` and `summary`):

| System | Mean valid perplexity | Standard deviation | Lowest to highest | Largest relative difference between two repeats | Pairs (of 45) whose 95% block interval excludes 0 |
|---|---|---|---|---|---|
| KN3 | 164.252 | 0.407 | 163.809 to 164.892 | 0.66% | 7 |
| KN5 | 154.769 | 0.447 | 154.227 to 155.499 | 0.82% | 9 |
| KN5 + cache, reset per block | 134.569 | 0.304 | 134.223 to 135.021 | 0.59% | 8 |

The block interval resamples the 34 valid blocks, the evaluation text only. If it also covered the variation from the train side, about 5% of the pairs (2 of 45) would exclude 0; 7 to 9 do. So for a comparison whose arms differ on the train side, that interval is too narrow. The pairs share repeats and are not independent; of the 5 disjoint pairs, 1 excludes 0 for each system.

A2 has one smoke run, `20261015T074420Z_a2_smoke`, on the validation split with small sizes (`config.json`, keys `split` and `smoke`), recorded as finished. It records the SHA-256 of the files it read, among them the word-similarity sets, the nltk data and the OR-group file of hierarchical_tm (`data_hashes.json`); the [data card](data.md) lists them. Two more smoke runs on 2026-10-25 test the current script (`20261025T114615Z_a2_smoke` and, with a named claim-carrying code, `20261025T114634Z_a2_smoke_claim_orgroups`).

**The development run of A2** (`20261025T114718Z_a2_valid_dev`, valid, 280.3 s, finished; a development reading, not a verdict). It built the codes from train (`data/a2_codes/`), chose the cache on valid (half-life 100, `<eos>` not added) and fitted the mixing weights of every arm on calib. Valid perplexity of each arm (`metrics.json`, key `arms`), 73,760 tokens, of which 15,782 have a bigram unseen in train:

| Arm | Valid perplexity | Against `none` | Bits difference against `none`, 95% interval (34 blocks) | On unseen-bigram positions |
|---|---|---|---|---|
| `none` (KN5 + cache) | 132.930 | | | 4,980.4 |
| `exchange` (4 levels, 16 to 1,024 classes) | 122.150 | -8.11% | -0.1220 (-0.1323 to -0.1115) | 3,237.8 |
| `kind_topic` (exchange and topic) | 122.150 | -8.11% | -0.1220 (-0.1323 to -0.1115) | 3,237.8 |
| `kmeans200` | 131.573 | -1.02% | -0.0148 (-0.0176 to -0.0120) | 4,636.4 |
| `predefined` | 131.688 | -0.93% | -0.0135 (-0.0172 to -0.0102) | 4,446.8 |
| `topic` | 132.798 | -0.10% | -0.0014 (-0.0021 to -0.0007) | 4,888.6 |
| `orgroups` (from hierarchical_tm) | 132.930 | -0.00% | -0.0000 (-0.0000 to -0.0000) | 4,980.4 |
| 10 random codes | 132.928 to 132.935 | 0.00% | | |

EM on calib gave the topic members of `kind_topic`, the class member of `orgroups` and the members of the random codes weights near 0 (`lam`), so these arms are the system without them. The script's development reading (`reading`) meets its success line on valid. Two later development runs with later versions of the script and `--claim-arm exchange`, `20261025T123159Z_a2_valid_dev` (the codes loaded, 87.8 s) and `20261025T132230Z_a2_valid_dev` (87.1 s; it records the thread settings and the installed versions in `config.json`), reproduce the cache choice, every mixing weight and every arm's perplexity of the first exactly; for `exchange` against `none` they give -0.6212 bits per token (-0.6516 to -0.5919) on the positions whose bigram is unseen in train, -0.1364 (-0.1436 to -0.1298) without the duplicate tokens, and -0.1221 (-0.1324 to -0.1117) on the 73,747 positions whose word has a train count (`subsets_no_verdict`), and KN5 alone scores 7.2551 bits per word, A1's value. Their `logp.npz` files were checked by `scripts/a2_recompute.py`, which matched every reading and, scoring valid again, every per-token log probability (`verification/a2_dev/`). A draft registration of the one dev-test look, `registrations/A2.md` (reviewed twice and revised, not frozen), names `exchange` as the claim-carrying code; it would be the 2nd use of dev-test, after T0. A2 has no Tsetlin part except the code `orgroups`.


### A2: the registered look on dev-test (2026-10-25)

The registration [registrations/A2.md](../registrations/A2.md) was frozen at 13:38:45 UTC on 2026-10-25 (`A2.frozen.json`; the registration's SHA-256 `e59e499494d9ce1c9a1fd5089620236d7421274929b8d7dc8efbfb62a11a6509`, 64 extra files and the 12 imported `src` files checked against the hashes pasted in the text: 76 of 76 equal). It was reviewed before the freeze in three rounds (judged by AI models, not people). The look ran once, from 13:39:12 to 13:40:40 UTC (`20261025T133912Z_a2_devtest_final`, finished; output `verification/a2_devtest_look.out`, SHA-256 `ec861ece677106e2df7889fd80baf45cd24619025e548fecdfe9f56e266f29fa`). It is the 2nd use of the dev-test split; the 1st was the registered report of T0.

| Arm | Dev-test perplexity | Against `none` | Bits difference against `none`, 95% interval (21 blocks) | On unseen-bigram positions |
|---|---|---|---|---|
| `none` (KN5 + cache) | 124.251 | | | 4,811.0 |
| `exchange` (4 levels, 16 to 1,024 classes), the claim code | 114.821 | -7.59% | -0.1139 (-0.1294 to -0.0971) | 3,119.2 |
| `kind_topic` (exchange and topic) | 114.821 | -7.59% | -0.1139 (-0.1294 to -0.0971) | 3,119.2 |
| `kmeans200` | 123.215 | -0.83% | -0.0121 (-0.0154 to -0.0082) | 4,487.9 |
| `predefined` (reported as noisy) | 123.313 | -0.76% | -0.0109 (-0.0160 to -0.0054) | 4,304.4 |
| `topic` | 124.190 | -0.05% | -0.0007 (-0.0013 to -0.0002) | 4,726.9 |
| `orgroups` (from hierarchical_tm) | 124.251 | -0.00% | -0.0000 (-0.0000 to +0.0000) | 4,811.0 |
| 10 random codes | 124.251 to 124.260 | -0.0001% to +0.0074% | | |

The registered reading (`metrics.json`, key `reading`): gain 0.0759, margin against the mean of the 10 random codes 0.0759, and the 95% interval of bits(exchange) - bits(random mean) from -0.1294 to -0.0971, entirely below 0. **The success line is met** (gain at least 0.05, margin at least 0.01, interval below 0); the kill line (gain under 0.02) is not. KN5 alone gives 7.176026 bits per word, T0's value on this split. As the registration words it:

> On 21 held-out 100-sentence blocks of the Penn Treebank training file (dev-test; V = 10,000, `<eos>` counted), the class members of an exchange-class code (Martin, Liermann and Ney 1998; 16 to 1,024 classes, counts from train), chosen on valid among six candidate codes before the look, lowered the perplexity of a Kneser-Ney 5-gram with a cache by 7.59% (95% interval over dev-test blocks 6.51% to 8.58%; it does not include train-side noise). This confirms the development result (valid 8.11%, calib cross-fit 7.93%), from which success was predicted before the look; exploratory in the sense of design rule 12 (one build, one split, one benchmark). The 10 random codes with the same class sizes were -0.0001% to +0.0074% from the system without classes (no rule; EM weights of their members at most 0.0024).

The gain is of the size of the class-model gain in a 2011 table (125.7 to 115.6, 8.0%, as quoted on the design page; another evaluation split and other implementations, so not a comparison under design rule 8). Dev-test is 21 blocks of 100 sentences of the Penn Treebank training file, not separate documents (2.1% of its sentences occur verbatim in train), and the vocabulary is the fixed Mikolov list built from the whole training file, dev-test included.

Beside the reading, with no rule:

- The interval of the gain that adds the estimated train-side standard deviation (0.0097, scaled from the 5% rebuilds of A1; an estimate) to the block standard error: 5.43% to 9.75% (`gain_interval_train_inclusive_no_rule`).
- On the positions outside duplicate spans (42,587 of 46,411 tokens) `exchange` minus `none` is -0.1333 bits (-0.1441 to -0.1216); on the positions whose word has a train count above 0, -0.1144 (-0.1299 to -0.0972); on the unseen-bigram positions the perplexity falls by 35.16%. The gain is larger away from the duplicates, as the registration expected.
- `orgroups`, the one code taken from a Tsetlin Machine (the OR-groups of a hierarchical_tm run), added nothing: EM gave its class member the weight 0.0. Its code is close to the identity (9,070 of its 9,142 classes are single words), so its member nearly repeats the word 5-gram; this reading is uninformative about codes learned by Tsetlin Machines, by construction. A2 is a result about count models with category codes and says nothing about Tsetlin learning.

Checks (judged by an AI model, not a person): the registered recompute (`scripts/a2_recompute.py --rescore`, output `verification/a2_devtest_recompute.json`, SHA-256 `74b42ae31e4a9ce1b2a5f8db0fdc266e59249ab8556eeba2f5fd411ab72ba6ef`) agrees on all 14 of its checks, rescoring every arm from the saved codes with a largest difference of 0.0 in any log probability; its own bootstrap (10,000 resamples, seed 20261025) gives -0.1283 to -0.0981. The three-lens check (`verification/a2_devtest_lenses.md`, SHA-256 `af4d6ca855b73203bfadf8c2e653935308a9547a07dcf62df54ea353b9eef8e9`) found nothing to correct. After this look, A3's contrast against Kneser-Ney plus cache on dev-test is informed by A2: A3 may carry a dev-test rule only for its increment over KN5 + cache + exchange, or must use the standard test file (registration, "What a met or failed line leads to").

## C3

**Rerun, exploratory, no claim:** with the loss and the clips of the paper, the medium model reaches a best valid perplexity of 90.99 to 92.12 on 10 of 10 seeds (paper 86.2). The large model reaches 86.22 to 87.05 on 6 of 10 seeds (paper 82.2); the other 4 collapsed in their first epoch to about the perplexity of a unigram model (686.60 to 688.61). Details are in the last part of this section.

The 20 runs of 2026-10-15 of `src/tally/neural/lstm.py` are recorded as finished and score the validation split only (`config.json`, key `note`, which reads "exploratory, valid only"). The medium model trains for 39 epochs and the large model for 55 (`config.json`, keys `size` and `epochs`). They are a preparation of the C3 comparison and carry no claim.

**These 20 runs are under-trained** (found on 2026-10-25). They minimised the mean cross-entropy over all 35 x 20 tokens of a chunk with plain SGD at learning rate 1, the learning rate of Zaremba et al. 2014, whose reference code adds the losses of the 35 unrolled steps, each averaged over the 20 rows of the minibatch (`main.lua` of github.com/wojzaremba/lstm at commit 76870253, with `ClassNLLCriterion`, whose `sizeAverage` is true by default). Their gradient was therefore 35 times smaller. They also clipped the gradient norm at 5 for both sizes, where the paper uses 5 for the medium and 10 for the large model (arXiv 1409.2329v5, section 4.1). Their best valid perplexities are 147.81 to 148.54 for the medium and 113.38 to 113.84 for the large model (`metrics.json`, key `best_valid_ppl`), against 86.2 and 82.2 in the paper's Table 1, and the best epoch is the last in 17 of the 20 runs. They are no reference for the published model. `src/tally/neural/lstm.py` now follows the paper by default and keeps the old setting behind `--loss mean_tokens --clip 5` (`tests/test_lstm_loss.py`); the rerun is below.

| Seed | Medium | Large |
|---|---|---|
| 1 | `20261015T093706Z_c3prep_lstm_medium_seed1_valid` | `20261015T095728Z_c3prep_lstm_large_seed1_valid` |
| 2 | `20261015T093937Z_c3prep_lstm_medium_seed2_valid` | `20261015T100311Z_c3prep_lstm_large_seed2_valid` |
| 3 | `20261015T111901Z_c3prep_lstm_medium_seed3_valid` | `20261015T112933Z_c3prep_lstm_large_seed3_valid` |
| 4 | `20261015T120236Z_c3prep_lstm_medium_seed4_valid` | `20261015T121301Z_c3prep_lstm_large_seed4_valid` |
| 5 | `20261015T124555Z_c3prep_lstm_medium_seed5_valid` | `20261015T125621Z_c3prep_lstm_large_seed5_valid` |
| 6 | `20261015T174824Z_c3prep_lstm_medium_seed6_valid` | `20261015T162840Z_c3prep_lstm_large_seed6_valid` |
| 7 | `20261015T193004Z_c3prep_lstm_medium_seed7_valid` | `20261015T181134Z_c3prep_lstm_large_seed7_valid` |
| 8 | `20261015T202550Z_c3prep_lstm_medium_seed8_valid` | `20261015T195312Z_c3prep_lstm_large_seed8_valid` |
| 9 | `20261015T174735Z_c3prep_lstm_medium_seed9_valid` | `20261015T162840Z_c3prep_lstm_large_seed9_valid` |
| 10 | `20261015T192938Z_c3prep_lstm_medium_seed10_valid` | `20261015T181049Z_c3prep_lstm_large_seed10_valid` |

### Rerun with the loss and the clips of the paper (2026-10-25 and 26)

The 20 runs were repeated with the defaults of `src/tally/neural/lstm.py`: the loss summed over the 35 steps (`--loss sum_steps`), the gradient norm clipped at 5 for the medium and at 10 for the large model, and every other setting as before. A smoke run of 6 epochs came first (`20261025T140525Z_c3prep_lstm_medium_seed1_valid_sum_steps_clip5_smoke6ep`, valid perplexity 110.25 after epoch 6). The 20 runs ran one at a time on the GPU: medium seed 1 on 2026-10-25 from 15:02:51 to 15:11:33 UTC, the other 19 on 2026-10-26 from 02:20:08 to 09:07:03 UTC (`run.json`, keys `start_utc` and `end_utc`). A medium run took 512 to 522 s and a large run 1,896 to 1,908 s. All 20 are recorded as finished, ran all their epochs and score the validation split only; like the runs above, they carry no claim.

| Seed | Medium: best valid perplexity (epoch) | Large: best valid perplexity (epoch) | Medium run | Large run |
|---|---|---|---|---|
| 1 | 92.12 (39) | 87.05 (53) | `20261025T150251Z_c3prep_lstm_medium_seed1_valid_sum_steps_clip5` | `20261026T034332Z_c3prep_lstm_large_seed1_valid_sum_steps_clip10` |
| 2 | 91.21 (39) | 86.37 (26) | `20261026T022008Z_c3prep_lstm_medium_seed2_valid_sum_steps_clip5` | `20261026T041555Z_c3prep_lstm_large_seed2_valid_sum_steps_clip10` |
| 3 | 91.79 (39) | 688.61 (40), collapsed | `20261026T022926Z_c3prep_lstm_medium_seed3_valid_sum_steps_clip5` | `20261026T044817Z_c3prep_lstm_large_seed3_valid_sum_steps_clip10` |
| 4 | 91.85 (39) | 687.21 (53), collapsed | `20261026T023841Z_c3prep_lstm_medium_seed4_valid_sum_steps_clip5` | `20261026T052040Z_c3prep_lstm_large_seed4_valid_sum_steps_clip10` |
| 5 | 91.37 (39) | 86.72 (28) | `20261026T024756Z_c3prep_lstm_medium_seed5_valid_sum_steps_clip5` | `20261026T055301Z_c3prep_lstm_large_seed5_valid_sum_steps_clip10` |
| 6 | 91.49 (39) | 86.42 (29) | `20261026T025711Z_c3prep_lstm_medium_seed6_valid_sum_steps_clip5` | `20261026T062525Z_c3prep_lstm_large_seed6_valid_sum_steps_clip10` |
| 7 | 91.16 (39) | 687.68 (52), collapsed | `20261026T030632Z_c3prep_lstm_medium_seed7_valid_sum_steps_clip5` | `20261026T065753Z_c3prep_lstm_large_seed7_valid_sum_steps_clip10` |
| 8 | 92.05 (38) | 86.23 (27) | `20261026T031547Z_c3prep_lstm_medium_seed8_valid_sum_steps_clip5` | `20261026T073023Z_c3prep_lstm_large_seed8_valid_sum_steps_clip10` |
| 9 | 90.99 (39) | 86.22 (30) | `20261026T032501Z_c3prep_lstm_medium_seed9_valid_sum_steps_clip5` | `20261026T080256Z_c3prep_lstm_large_seed9_valid_sum_steps_clip10` |
| 10 | 91.61 (38) | 686.60 (55), collapsed | `20261026T033417Z_c3prep_lstm_medium_seed10_valid_sum_steps_clip5` | `20261026T083523Z_c3prep_lstm_large_seed10_valid_sum_steps_clip10` |

- **Medium: 10 of 10 seeds learned.** Best valid perplexity 90.99 to 92.12, mean 91.56, standard deviation 0.39 over the seeds. The best epoch is the last in 8 runs and the 38th in 2. Paper: 86.2; runs of 2026-10-15: 147.81 to 148.54.
- **Large: 6 of 10 seeds learned** (1, 2, 5, 6, 8 and 9): 86.22 to 87.05. Five of them reach their best between epochs 26 and 30 and end at most 0.42 above it at epoch 55; seed 1 reaches its best at epoch 53. Paper: 82.2; runs of 2026-10-15: 113.38 to 113.84.
- **Large: 4 of 10 seeds collapsed in their first epoch** (3, 4, 7 and 10). Their valid perplexity after epoch 1 is 2,589.9, 698.2, 10,587.8 and 1,503.0, and their best over 55 epochs is 686.60 to 688.61. That is about the perplexity of a unigram model: the add-one unigram distribution of the training split gives 685.88 on valid. In their last epoch no step reached the clip (`clipped_share` 0.0), while in the last epoch of every run that learned every step did (1.0).
- **What differs from the paper.** The runs train on this study's training split, 836,253 tokens, which is 90.0% of the standard training file of 929,589 tokens; the other blocks of that file form the calibration and dev-test splits (see the [data card](data.md)). The paper trains on the standard split. The gap to the paper's values, 5.4 for the medium mean and 4.02 to 4.85 for the large runs that learned, is not explained. The smaller training text is one candidate and has not been tested, and the paper's Table 1 gives one value per size, with no seeds or spread. C3 has no registration, so no rule says how a collapsed run counts.
- **Check** (judged by an AI model, not a person). An independent script, kept with the records of the finish pass outside the repository, read `config.json`, `metrics.json`, `run.json` and `data_hashes.json` of the 20 reruns. Every setting equals section 4.1 of the paper, every run ran all its epochs and is recorded as finished, and the valid perplexity of every epoch equals 2 to the power of its bits per word. The best value and its epoch recompute from the history, and every run records the SHA-256 of the training and the validation file given on the data card. The script found no problem.

## Limits of these records

- Two registered statistics have been computed, both on 2026-10-25: the verdict of T0 and the look of A2. Every other result on this page is exploratory and tests no hypothesis.
- Up to 2026-10-17 the run logger keyed `data_hashes.json` by file name. The Penn Treebank runs pass the training and the validation file, which are both named `text.utf8`, so their records hold one value, the SHA-256 of the validation text. The tuning, final and smoke runs of T0, the first three A1 runs and the C3 runs are affected. Among those run files, the SHA-256 of the training text is recorded only in `metrics.json` and `log.txt` of `20261015T074420Z_a2_smoke`, in the source records of its codes; the [data card](data.md) gives both values. Since 2026-10-25 the keys are paths relative to the study folder, and the runs since that day record both texts.
- The run logger recorded a run killed by a signal as running. Since 2026-10-25 it records SIGTERM and SIGHUP as killed; SIGKILL still leaves a run recorded as running.
- The tuning runs of T0 record no registration, and the logistic-regression runs record no data file; they name their source run instead.
- No run folder records a commit of the repository or holds a copy of the code; the source records hold SHA-256 values only. The wording of `registrations/T0.md` and the comments and docstrings of four files under `src/` (`src/tally/t0/data.py`, `src/tally/floor/kenlm.py`, `src/tally/floor/codes.py` and `src/tally/b6/config.py`) were edited for publication on 2026-10-23, so these five files no longer match the values in the run records and the freeze records; the [README](../README.md#records-and-hashes) gives the details.
- One run, `20261015T072630Z_b1_smoke_flat_M8_s0`, is recorded as running without an end time, and one folder name, `20261015T072640Z_t0_report_valid_smoke`, differs from the run id in its record.
