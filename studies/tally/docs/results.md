# Results of the tally study

The study has 91 run folders: 66 of 2026-10-15, 15 of 2026-10-16 and 10 of 2026-10-17 (UTC, by run id). The first run started at 07:10:15 UTC on 2026-10-15 (`run.json` of `20261015T071015Z_b6timing_fmnist_thresh10_c100`), and the last ended at 20:58:57 UTC on 2026-10-17 (`run.json` of `20261017T204300Z_t0_logreg_of_final_coal_c2000_s20_seed8`). The records give 87 runs as finished, 3 as crashed and 1 as running without an end time (`run.json`, key `state`). The [run index](../../../docs/run-index.md) lists every folder. The run folders and the run logs are not distributed, so the file names serve readers who hold them.

Every number on this page is read from a file of the run that is named next to it. The times of the freezes are read from the freeze records in `registrations/`, and values quoted from the registration or its amendments are marked as such. No number of the dev-test split appears on this page: the final runs of T0 store their dev-test bits, and only the registered report reads them, which has not been run. Values that the amendments compare are given to the six decimals that the amendments print.

The question of the study, its lines of work and its dependencies are described in the [README](../README.md).

## Status of the registered ids

Three ids carry a status: T0 and its two amendments, which fix the settings of the two machines by the tuning rule of T0. The statuses are met, not met, not judged and not run; for an amendment, which is a step of the procedure, met means that the step was carried out as registered. Of the three ids, two are met and one is not judged.

| Id | Registered content | Status | Qualifier | Run folders |
|---|---|---|---|---|
| T0 ([registration](../registrations/T0.md)) | Penn Treebank next-word prediction with tmu at commit 5605ff0. Claim-carrying comparison: the counted read-out on the clauses of the coalesced machine (arm 2) against its learned votes (arm 1), in bits per word on dev-test, with each arm averaged per position over seeds 1 to 10 and a 95% interval from 1,000 resamples of the 21 dev-test blocks. Further verdict line: arm 2 against the untrained bank (arm 3). Reported with intervals and no verdict: arm 2 against logistic regression on the clause outputs (arm 4), and the counted against the learned read-out of the flat machine. | not judged | the 20 final runs of the Tsetlin machines and the 10 logistic-regression runs have finished; `src/tally/t0/report.py` has not been run on them, so no interval and no verdict line exists; the Kneser-Ney 3-gram and 5-gram models that the registration reports beside the arms have no run on dev-test either | the 30 runs of the [final runs](#final-runs) |
| Amendment of the coalesced settings ([text](../registrations/T0_amendment_coalesced_settings.md)), frozen at 18:24:29 UTC on 2026-10-15 | s = 20 and 9 epochs, chosen by the tuning rule of T0 | met | frozen after the three coalesced tuning runs had ended and before the first coalesced final run; the chosen s is the largest value of the registered grid {5, 10, 20}, and its best epoch, 9, is one below the cap of 10 epochs | the three coalesced runs of the [tuning runs](#tuning-runs) |
| Amendment of the flat settings ([text](../registrations/T0_amendment_flat_settings.md)), frozen at 21:14:27 UTC on 2026-10-15 | s = 5 and 1 epoch, chosen by the tuning rule of T0 | met | frozen after the three flat tuning runs had ended and before the first flat final run; the chosen s is the smallest value of the grid, and all three flat tuning runs had their best learned score after the first epoch; the flat tuning run with s = 10 started before T0 was frozen, the other two after it | the three flat runs of the [tuning runs](#tuning-runs) |

T0 was frozen at 07:27:25 UTC on 2026-10-15 (`T0.frozen.json`). The last coalesced tuning run ended at 18:22:35 UTC, the coalesced amendment was frozen at 18:24:29 UTC, and the first coalesced final run started at 18:24:30 UTC. The last flat tuning run ended at 21:10:33 UTC, the flat amendment was frozen at 21:14:27 UTC, and the first flat final run started at 16:59:43 UTC on 2026-10-16 (`run.json` of the runs in the tables below). By the registration, the verdict is computed once, after all 10 seeds of both machines have finished. Every run that the verdict reads had ended by 20:58:57 UTC on 2026-10-17.

## Run folders by line

| Line | Run folders | Finished | Crashed | Recorded as running | What they are |
|---|---|---|---|---|---|
| T0 | 43 | 40 | 3 | 0 | 2 smoke runs, 4 starts of the logistic-regression arm for a smoke run, of which 3 crashed, 1 smoke run of the report, 6 tuning runs, 20 final runs of the two machines, 10 logistic-regression runs on the coalesced final runs |
| B1 | 15 | 14 | 0 | 1 | 12 smoke runs at team size M = 8 and 3 smoke runs of the member calibration |
| B6 | 9 | 9 | 0 | 0 | timing runs on the training split |
| A1 | 3 | 3 | 0 | 0 | runs of the baseline check |
| A2 | 1 | 1 | 0 | 0 | a smoke run on the validation split |
| C3 | 20 | 20 | 0 | 0 | LSTM runs on the validation split, two sizes, seeds 1 to 10 |
| All | 91 | 87 | 3 | 1 | |

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

The final runs of the Tsetlin machines store their bits per position (`bits.npz`), and the coalesced runs also the clause outputs of the four splits (`fired_<split>.npz`), which the logistic-regression runs read. The statistic of the registration is computed from these files by `src/tally/t0/report.py`, which has not been run on them; this page therefore gives no number from the final runs.

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

## A1 and A2

A1 has three runs of `scripts/a1_baseline_check.py`, all recorded as finished: `20261015T071028Z_a1_baseline_check`, `20261015T071151Z_a1_baseline_check` and `20261015T072513Z_a1_baseline_check`. Their source records differ (`source_hashes.json`). The third records settings that the first two lack, among them that the cache settings are chosen on the validation split and the mixing weights fitted on the calibration split (`config.json`, keys `cache_select` and `em_fit`). The check scores the validation split and not dev-test; it is a baseline measurement without a registration, and this page gives none of its numbers.

A2 has one smoke run, `20261015T074420Z_a2_smoke`, on the validation split with small sizes (`config.json`, keys `split` and `smoke`), recorded as finished. It records the SHA-256 of the files it read, among them the word-similarity sets, the nltk data and the OR-group file of hierarchical_tm (`data_hashes.json`); the [data card](data.md) lists them.

## C3

All 20 runs of `src/tally/neural/lstm.py` are recorded as finished and score the validation split only (`config.json`, key `note`, which reads "exploratory, valid only"). The medium model trains for 39 epochs and the large model for 55 (`config.json`, keys `size` and `epochs`). They are a preparation of the C3 comparison and carry no claim; this page gives none of their numbers.

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

## Limits of these records

- No registered statistic has been computed, so no result on this page tests a hypothesis.
- The run logger keys `data_hashes.json` by file name. The Penn Treebank runs pass the training and the validation file, which are both named `text.utf8`, so their records hold one value, the SHA-256 of the validation text. The tuning, final and smoke runs of T0, the A1 runs and the C3 runs are affected. Among the run files, the SHA-256 of the training text is recorded only in `metrics.json` and `log.txt` of `20261015T074420Z_a2_smoke`, in the source records of its codes; the [data card](data.md) gives both values.
- The tuning runs of T0 record no registration, and the logistic-regression runs record no data file; they name their source run instead.
- No run folder records a commit of the repository or holds a copy of the code; the source records hold SHA-256 values only. The wording of `registrations/T0.md` and the comments and docstrings of four files under `src/` (`src/tally/t0/data.py`, `src/tally/floor/kenlm.py`, `src/tally/floor/codes.py` and `src/tally/b6/config.py`) were edited for publication on 2026-10-23, so these five files no longer match the values in the run records and the freeze records; the [README](../README.md#records-and-hashes) gives the details.
- One run, `20261015T072630Z_b1_smoke_flat_M8_s0`, is recorded as running without an end time, and one folder name, `20261015T072640Z_t0_report_valid_smoke`, differs from the run id in its record.
