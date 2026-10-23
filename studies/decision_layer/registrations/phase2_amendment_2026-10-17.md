# Amendment for R9, R9b, R10, R10c, R11, R12, R13, R14, R15, R16, R16b, R17, R18 (written 2026-10-17, during and after the runs)

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

Nothing in a pass rule, an arm or a setting changed after a freeze. This file lists what happened around the runs.

## R11

- Two jobs of the bounded arm (`massive_en_ml16`, seeds 5-6 and 9-10) crashed with CUDA out-of-memory between about 22:20 and 22:40 UTC on 2026-10-16. The GPU is shared with other users' jobs, and about 16 of this study's jobs ran at that time. The same two commands were run again at 22:50:33 UTC (`logs/jobs_R11_missing.txt`); `confirm_router.py` skips seeds that exist. Before the re-run, the test accuracies of seeds 2, 4 and 8 of that arm had been read in the job log, which prints them.
- `scripts/r7_readability.py` was run on R11's R7 arm as declared ("reported with no rule").
- R11's list of source changes since R7 omits `src/dl/boolean.py` (a new `lda`/`white` branch in `make_view`, added 2026-10-16 16:44 for exploration e24) and the new `src/dl/boolean_rot.py`. The branch is a separate `if` that the `proto:...+lex:2000` spec never reaches (internal audit 6).
- R11's text says every MASSIVE embedding existed before the freeze; only `massive_en__bge-large.npz` did (21:45:07); the bge-m3 and multilingual-e5-large files were finished between 21:45:49 and 21:49:19 (internal audit 6). No model was trained or scored on MASSIVE before the freeze.

## R9

- R9.1 does not test what it was meant to test. TM-nomem's whole-conversation parts (`fstep`, `fwords`) read the field `full`, and that text contains every earlier button press as a line "action: <button>" (`src/dl/abcd.py`, `action_examples`). So the arm without memory bits still gets the history, as words. This was seen at about 23:26 UTC, when the first four seeds per TM arm were written. R9.1 is judged as registered. R9b (frozen 23:27:00Z) is the follow-up with a control that has no record of the earlier actions.
- `scripts/report_abcd.py` gained the composite computation (`--composite`) after the freeze: two disjoint 5-machine composites, seeds 1-5 and 6-10, sum of class sums, as the registration defines them.
- `scripts/master_queue.sh` counts this phase's scripts since the R9 launch (a queue control, no effect on any run).

## R10

- The exploration LinUCB runs with the original class (alpha 0.1, 0.5, 1.0) were stopped unfinished at 00:42 UTC after 1 h 49 min to 2 h 34 min: they held queue slots that R9 needed. LinUCBFast (`src/dl/bandit.py`, same estimator with Woodbury updates; `tests/test_linucb_fast.py`) is used for the LinUCB comparison, which is R10c, not R10.
- `scripts/abcd_bandit.py` was edited at 01:42:21Z, after R10's freeze and after R10's last run (01:38:52Z): new options `--max-literals` and `--init` for exploration e29, whose defaults (0, half) give the frozen computation. No R10 run used the edited file. Of R10c's runs only the two relaunched ones (LinUCB path, untouched by the edit) used it; the other seven started before the edit.
- R10's registration says inputs carry "no label of any kind": the memory bits are the human agent's logged earlier presses, i.e. the targets of earlier turns (teacher forcing); no bit is built from the label of the decision being learned. It also says the test split is scored once per run: `run_stream` scores it after every pass for the learning curve; the registered number is the last pass, with no selection (internal audit 6).

## R10c

- Two of the nine LinUCB runs (alpha 0.1 seed 3, alpha 1.0 seed 3) crashed at start with CUDA out-of-memory at 01:41 UTC (GPU at 30.4 of 32 GB with other users' jobs). The same commands (`logs/jobs_R10c_missing.txt`, written 01:42:00Z, before any R10c result) both started at 02:41 UTC (02:41:11Z, 02:41:41Z), when slots freed. One or two of the seven other runs had finished by then; all seven were read at 02:44 (project notes, not published). The log line said "02:41 and 02:44"; the addendum of internal audit 6 found the start times.
- `scripts/report_abcd_bandit.py` gained `--out`, so R10c's report (`report_R10c.json`) does not overwrite R10's.
- R10c's frozen text calls the inputs "label-free bits", as R10's did: the memory bits are the human agent's logged earlier presses (see R10 above).

## R16

- The six R16 runs started at 07:13:52 UTC were stopped at 07:25 after their first epoch and started again at 07:25:29 with `--eval-every 30`: scoring dev after every epoch took as long as the training itself (the package scores one example per kernel call), which would have made the runs last about 4 hours. Scoring uses no random state (`_score` in the package), so skipping it leaves the training and the test predictions unchanged. The only numbers seen before the restart were the dev accuracies after epoch 1 (0.54 to 0.62). No test number had been computed.

- R16's report also reads R9b's TM-mem-short prediction files, copied to `results/confirm/R16/tm_mem_short_R9b` at 09:52 UTC (for the no-rule comparison).

## R15

- `scripts/report_r15.py` (R15's report: online accuracy with a bootstrap over seeds; final accuracy with the cluster bootstrap) was created at 04:30:57 UTC, 18 s after R15's freeze, and computes what R15 registered; internal audit 7 reproduced its output independently.
- The replay runs are not bit-reproducible on the GPU (two seed-1 runs differ by about 0.3 points of online accuracy).

## R16, further notes (internal audit 7)

- Slips in the R16 registration: "0.828 after 20" for the coalesced TM is 0.830; "2.4 points ahead of both" holds against the hierarchical machine only (against the package's flat machine it is about 12).
- R16's 30 epochs are the hierarchical dev run's best epoch; the flat machine got fewer dev settings and none of matched size.
- R13's registration lacks a "what was known at the freeze" line: R9b's test results were known when R13 was frozen.
- No launch-command files exist for R13 to R16 (`logs/jobs_R13.txt` and so on); the commands are in the project notes (not published) and the result records.

## R11 and R12: MASSIVE en-US is not independent of HWU64 (noted 2026-10-17 after 23:50 UTC)

- MASSIVE en-US and HWU64 come from the same source collection, with different labels (60 against 64 intents). Counted with lowercase and whitespace normalisation, no text printed:
  - MASSIVE test: 968 of 2,970 distinct texts are in HWU64 train, and 1,214 are in some HWU64 split;
  - MASSIVE val: 705 of 2,031 are in HWU64 train;
  - HWU64 test: 470 of 1,076 are in MASSIVE train.
  This was found in the text-understanding line, re-counted independently, and counted again, with the same numbers.
- No model trained on HWU64 was scored on MASSIVE, so no model saw a test text. But the router settings were developed and tested on HWU64. So R11's "fresh data" and "data never used" overstate its independence: MASSIVE is a related dataset, not a fresh one. R11's frozen text stays as it was; the summaries now say "a related dataset".
- R12 (en-US to nb-NO within MASSIVE) is unaffected in substance; only the word "fresh" changes. ABCD and STAR are unaffected.

## R18

- Before R18's freeze: `src/dl/abcd.py` `load_examples` gained a `dataset` argument (default abcd), and `scripts/abcd_explore.py`, `scripts/abcd_confirm.py` gained `--dataset` (13:42 to 14:18 UTC). After the freeze `scripts/abcd_bandit.py` gained `--dataset` (14:27; not on R18's path). The STAR split is a hash of the dialogue id, written in `scripts/star_prepare.py` before any model was trained.
- `abcd_bandit.py`'s help text says the option was added at 14:28; the file was written at 14:27:15.
- Before the freeze, a code check: `abcd_confirm.py --dataset star --dry-run` for the TM arm (dev in place of test; `results/explore/dryrun/dry_star_tm`, 14:17:54 to 14:18:11).
- The registration's dev note says the memory gain is about 13 to 15 points "for every learner": no MLP was run without memory (dev or test), so it holds for the TM and LR only.
- `results/confirm/R18/report_R18_partial.json` (16:42:14): the report over the finished TM and MLP arms while both LR jobs still ran. No choice was left open: the commands were fixed at the freeze, and LR takes its C from dev.
- The report command was not recorded at the time. Reconstructed and checked after the audit (2026-10-17 17:47:58 UTC; the output is byte-identical to `report_R18.json`): `scripts/report_abcd.py --registration R18 --arms tm_mem tm_nomem lr_mem lr_nomem mlp_mem --pairs tm_mem:tm_nomem tm_mem:lr_mem tm_mem:mlp_mem lr_mem:lr_nomem tm_nomem:lr_nomem --out report_R18.json`. It is now a comment in `logs/jobs_R18.txt`.
- Internal audit 8 reproduced R18 exactly; its wording points were fixed in `docs/results.md`, `FAILURE_MAP.md` and `README.md` on 2026-10-17 after 17:48 UTC.

## R16b

- R16b.1's label "(the hierarchy, not the size)" means "not the size alone": T, s and the graded vote still differ.
- Slip in the R16b registration: "about 2.7 points (200 to 800)" is 2.6 (0.7158 to 0.7419); 2.7 is 200 to 3,200 (internal audit 7 addendum).
- R16's hierarchical and 200-clause flat prediction files were copied to `results/confirm/R16b/htm_hier_R16` and `htm_flat200_R16` (13:17 UTC) for the report; nothing was recomputed. Commands: `logs/jobs_R16b.txt`.

## R17

- Before R17's freeze, `scripts/abcd_bandit.py` gained `--save-model` and `scripts/abcd_closed_loop.py` gained `--view free` and `--model-file` (by 11:47 UTC; the help texts say 11:50 and mention a "--model-glob"; both are slips: the option is `--model-file`). `scripts/r17_arms.py` was written at 11:56:50 UTC, 24 s after R17's freeze, and run at 12:03; its two arms are new npz files whose `pred` and `y` equal the sources (it adds the test rows' conversation ids, positions and AST weights); nothing was recomputed.
- R17's frozen text calls the inputs "label-free bits", as R10's did: the memory bits in training are the human agent's logged presses (teacher forcing); only at test, in closed loop, do they hold the agent's own presses.
- The report commands for R16b and R17 are in the project notes (not published), not in the jobs files.

## R14

- `scripts/abcd_bandit.py` gained the `neural` learner (and `src/dl/bandit.py` the `NeuralBandit` class) at about 04:10 UTC, before R14's freeze and after R10's and R10c's runs; the TM and LinUCB paths are unchanged. `scripts/report_abcd_bandit.py` learned to parse the `neural_` arm names after R14's runs (04:14), a reporting change only. At 04:15 `abcd_bandit.py` gained `--replay` (default 0 = the registered computation) for exploration e34.

## R13

- The arms `tm_mem_short_open`, `tm_nomem_short` and `lr_mem_short_open` in `results/confirm/R13/` are copies of R9b's registered prediction files (seed*.json, seed*.npz), made at 04:09 UTC so that `scripts/report_abcd.py` reads every arm from one folder. Nothing in them was recomputed.

## R12

- None. R12's runs saved predictions, not models; `scripts/xling_demo.py` (post hoc) retrains seed 1, reproduces the registered nb-NO class sums of seed 1 exactly, and prints example receipts (project notes, not published).

## Internal audit 6 (01:53Z)

Every phase-2 verdict reproduces; none is contradicted. Wording fixes applied on 2026-10-17 between 01:53 and 01:56 UTC (clock read 01:55:47 right after them) in `docs/results.md`, `FAILURE_MAP.md` and `README.md`: clause lengths of the quoted clauses and of the strongest clauses; the two old verdict lines on linear models and memory literals qualified; which phase-2 tests reuse which test split; R11.2 "passes again, with less room" instead of "repeats"; R12.1 marked as a non-inferiority rule wherever "ahead" appears; R12.2's cause labelled post hoc; R9b's difference in gains explained; "no record of the earlier actions" instead of "no history at all"; R10's inputs "without the label of the decision being learned"; the LR C-grid edge (C = 0.1 in all four ABCD LR-on-bits arms) disclosed; the declared no-rule numbers added; three rounding slips corrected.
