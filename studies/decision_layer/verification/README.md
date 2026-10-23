# Independent recomputations

This folder holds independent recomputations of results of the decision_layer study, the outputs they recorded, and the command list that reproduces the R20 reports. The four scripts do not use the study's report scripts, so they check the reported numbers instead of repeating the same code: `recompute_r21.py`, `recompute_r24.py` and `recompute_e71.py` were written from the registration or plan text, and `recompute_pairs.py` applies the bootstrap of `recompute_r21.py` to the pairs and bars given on its command line. `r20_report_cmds.txt` is different: it runs the study's own report script with the registered arguments, and its check is that the report files come out identical.

The scripts read run folders under `results/` (per-seed prediction files and result rows). These folders are not distributed with the repository, so the scripts run only where the runs are present. The scripts print to standard output and write no file; the R20 commands write their report files into the run folders. Every command below is run from the study folder with the study's environment, `.venv-dl/bin/python` (Python 3.11.16 and numpy 2.4.6 at the rerun of 2026-10-23).

| File | What it is |
|---|---|
| `r20_report_cmds.txt` | the registered report commands of R20 (the study's `scripts/report_abcd.py`) |
| `recompute_r21.py` | independent recompute of R21 |
| `r21_recompute_independent.txt`, `r21_selftest_refs.txt` | its recorded outputs (full run; self-test on the reference arms) |
| `recompute_pairs.py` | independent recompute of paired differences with their bars, used for R22 and R23 |
| `r22_recompute_independent.txt`, `r23_recompute_independent.txt` | its recorded outputs for R22 and R23 |
| `recompute_r24.py` | independent recompute of R24 |
| `recompute_r24_out.json` | its recorded output |
| `recompute_e71.py` | independent tally of e71 and e76 |
| `e71v2_tally_final_seed1.txt`, `e71v2_tally_seeds12.txt`, `e76_tally_final_seed1.txt` | its recorded outputs |

Every script in this folder was rerun on 2026-10-23 and reproduced its recorded output. The one line that differs is named in the e71 section.

## R20: the registered report commands (`r20_report_cmds.txt`)

Result: registration `registrations/R20_customer_only_closed_loop.md` (frozen 2026-10-18T23:23:30Z), section "R20: the agent in a closed loop (STAR), and the agent that reads only the customer (ABCD)" of `docs/results.md`.

The file holds two command lines. They run `scripts/report_abcd.py` for `R20star` (8 arms, 6 pairs) and for `R20abcd` (9 arms, 11 pairs). This reproduces the registered report; it is not an independent recompute.

Reads: `results/confirm/R20star/<arm>/seed*.json` (the list of seeds) and `seed*.npz` (predictions, labels, dialogue ids, action index and AST weights) for every arm named, and the same under `results/confirm/R20abcd/`.

Method, as `scripts/report_abcd.py` states it: per test action turn, each arm's correctness is averaged over its seeds. A difference A - B is the mean over action turns of (A - B). Its 95% interval comes from a bootstrap over conversations (the action turns of one conversation are resampled together), 10,000 resamples, seed 0. Each pair is also reported with every action turn weighted by its number of AST examples.

Run: `bash verification/r20_report_cmds.txt`. This writes `report_R20star.json` into `results/confirm/R20star/` and `report_R20abcd.json` into `results/confirm/R20abcd/`, replacing the files there. The script joins `--out` to the run folder, so an absolute path given to `--out` leaves the files in the run folders as they are.

Output: the two report files in the run folders (not distributed). Their four rule pairs, accuracy difference in points with the 95% interval, read from the files: R20.1 `tm_mem_static_closed:tm_nomem_static_R19` +7.53 [+6.51, +8.58]; R20.2 `tm_mem_static_closed:tm_mem_static_R19` -7.51 [-8.34, -6.72]; R20.3 `tm_mem_static_closed:tm_mem_closed` +1.96 [+1.07, +2.85]; R20.4 (ABCD) `tm_mem_cust_closed:tm_nomem_cust` +23.94 [+22.20, +25.67]. A rerun on 2026-10-23, with `--out` pointed outside the run folders, gave both files byte for byte (sha256 b959f5255658843c for `report_R20star.json`, ec7a9cafa20a4930 for `report_R20abcd.json`).

## R21: `recompute_r21.py`

Result: registration `registrations/R21_c4_table_agent.md` (frozen 2026-10-19T05:48:24Z), rules R21.1 and R21.2 on STAR and ABCD; section "R21: the agent with no neural network at any stage (STAR and ABCD; test)" of `docs/results.md`.

Reads: `results/confirm/R21star/<arm>/seed*.json` and `seed*.npz`, and the same under `results/confirm/R21abcd/`, for the new arms `tm_mem_c4`, `tm_nomem_c4` and `lr_mem_c4` and for the reference arms in the same folders: `tm_mem_static_R19`, `tm_nomem_static_R19` and `lr_mem_static_R19`, with `tm_mem_R18`, `tm_nomem_R18` and `lr_mem_R18` on STAR or `tm_mem_R9b`, `tm_nomem_R9b` and `lr_mem_R9b` on ABCD.

Method, as the script states it: per test action turn, each arm's correctness (pred == y) is averaged over its seeds. A difference A - B is the mean over action turns of (A - B). Its 95% interval comes from a bootstrap over test dialogues (all action turns of a dialogue resampled together), 10,000 resamples, drawn with numpy's PCG64 generator at seed 20261019, so the interval is a check of the registered one (seed 0), not a copy. R21.1: `tm_mem_c4 - tm_mem_static_R19` at least -2.0 points on the point estimate, "strong" if the lower end is also at least -2.0. R21.2: `tm_mem_c4 - tm_nomem_c4` at least +5.0 points, with the interval above 0. A dataset is met when both hold; the headline needs both datasets. The script also checks the number of seeds per arm, that labels and dialogue ids are equal across arms, that the settings of the new TM arms equal R19's apart from the seed, the encoder (`none`), the parts, the registration field, and that every new seed file was written after the freeze.

Run: `.venv-dl/bin/python verification/recompute_r21.py` (recorded output `r21_recompute_independent.txt`). With `--only-refs` the script reads only the reference arms and no number of the new R21 arms; this is a self-test of the script (recorded output `r21_selftest_refs.txt`).

Recorded output, the rule lines of both datasets and the last two lines (quoted from `r21_recompute_independent.txt`; the file also gives each arm's accuracy per seed and the pairs without a rule):

```
== R21star ==
  tm_mem_c4 - tm_mem_static_R19              -0.10 [-0.63, +0.44]  (611 dialogues, 6988 action turns)
  tm_mem_c4 - tm_nomem_c4                    +14.73 [+13.75, +15.74]  (611 dialogues, 6988 action turns)
  R21.1 MET (strong); R21.2 MET -> dataset MET
== R21abcd ==
  tm_mem_c4 - tm_mem_static_R19              -0.29 [-0.75, +0.17]  (1004 dialogues, 3608 action turns)
  tm_mem_c4 - tm_nomem_c4                    +20.63 [+19.16, +22.08]  (1004 dialogues, 3608 action turns)
  R21.1 MET (strong); R21.2 MET -> dataset MET
problems: none
HEADLINE (both datasets): MET {'star': np.True_, 'abcd': np.True_}
```

The self-test output gives the memory effect of the reference agents and no problem (quoted from `r21_selftest_refs.txt`):

```
  tm_mem_R18 - tm_nomem_R18                  +15.43 [+14.38, +16.49]  (611 dialogues, 6988 action turns)
  tm_mem_R9b - tm_nomem_R9b                  +18.65 [+17.20, +20.09]  (1004 dialogues, 3608 action turns)
problems: none
```

## R22: `recompute_pairs.py`

Result: registration `registrations/R22_tuned_s_star.md` (frozen 2026-10-19T16:31:24Z), rules R22.1 to R22.3 on STAR test; section "R22: the TM's s, tuned on dev, for the agent with no neural network at any stage (STAR test)" of `docs/results.md`.

Reads: `results/confirm/R22star/<arm>/seed*.npz` for `tm_mem_c4_s16`, `tm_mem_c4_s8_R21`, `lr_mem_c4_R21`, `tm_mem_bge_s16` and `tm_mem_bge_s8_R18`. The first argument names the run folder under `results/confirm/`; an absolute path is used as given.

Method, as the script states it: per test action turn, each arm's correctness (pred == y) is averaged over its seeds; a difference A - B is the mean over action turns; its 95% interval comes from a bootstrap over test dialogues (all turns of a dialogue together), 10,000 resamples with numpy's PCG64 generator at seed 20261019, so it checks the registered interval (seed 0) without copying it. Each pair is given as `A:B[:BAR[:LOWBAR]]`: BAR is the bar for the point estimate in points; LOWBAR, if given, is the bar for the lower end ("strong"), or `pos` for an interval that must lie above 0. The script also checks that labels and dialogue ids are equal across all arms and seeds, and prints each arm's seeds and accuracy.

Run: `.venv-dl/bin/python verification/recompute_pairs.py R22star tm_mem_c4_s16:tm_mem_c4_s8_R21:1.0:pos tm_mem_c4_s16:lr_mem_c4_R21:-1.0:-1.0 tm_mem_c4_s16:tm_mem_bge_s16:-2.0:-2.0 tm_mem_bge_s16:tm_mem_bge_s8_R18 tm_mem_c4_s16:tm_mem_bge_s8_R18` (recorded output `r22_recompute_independent.txt`).

Recorded output, the lines of R22.1, R22.2 and R22.3 in this order, and the last line (quoted; the file also gives each arm's accuracy per seed and two pairs without a rule):

```
tm_mem_c4_s16 - tm_mem_c4_s8_R21             +2.70 [+2.16, +3.23]  (611 dialogues, 6988 turns)  point >= +1.0: MET; interval above 0: yes
tm_mem_c4_s16 - lr_mem_c4_R21                -0.09 [-0.88, +0.73]  (611 dialogues, 6988 turns)  point >= -1.0: MET; lower end >= -1.0 (strong): yes
tm_mem_c4_s16 - tm_mem_bge_s16               +1.64 [+0.93, +2.36]  (611 dialogues, 6988 turns)  point >= -2.0: MET; lower end >= -2.0 (strong): yes
problems: none
```

## R23: `recompute_pairs.py`

Result: registration `registrations/R23_tuned_s_abcd.md` (frozen 2026-10-19T20:32:10Z), rule R23.1 on ABCD test; section "e75 and R23: equal tuning on ABCD (dev, then test)" of `docs/results.md`.

Reads: `results/confirm/R23abcd/<arm>/seed*.npz` for `tm_mem_c4_s16`, `tm_mem_bge_s24`, `tm_mem_c4_s8_R21` and `tm_mem_bge_s8_R9b`. Method and seed as for R22.

Run: `.venv-dl/bin/python verification/recompute_pairs.py R23abcd tm_mem_c4_s16:tm_mem_bge_s24:-2.0:-2.0 tm_mem_c4_s16:tm_mem_c4_s8_R21 tm_mem_bge_s24:tm_mem_bge_s8_R9b tm_mem_c4_s16:tm_mem_bge_s8_R9b` (recorded output `r23_recompute_independent.txt`). These arguments are read from the recorded output, which prints every pair and its bar; with them, a rerun gives the file byte for byte.

Recorded output, the line of R23.1 and the last line (quoted; the file also gives each arm's accuracy per seed and three pairs without a rule):

```
tm_mem_c4_s16 - tm_mem_bge_s24               -0.92 [-1.60, -0.25]  (1004 dialogues, 3608 turns)  point >= -2.0: MET; lower end >= -2.0 (strong): yes
problems: none
```

## R24: `recompute_r24.py`

Result: registration `registrations/R24_multiwoz_random_negatives.md` (frozen 2026-10-22T20:16:52Z), rules R24.1 and R24.2 on MultiWOZ 2.2 test; section "R24: on MultiWOZ 2.2 test, the network-free agents against the bge-large agent, all with random negatives chosen on MultiWOZ dev (registered; 1st use; verified)" of `docs/results.md`.

Reads: `<arm>/seed*.json` and `<arm>/seed*.npz` in the run folder given as the first argument (`results/confirm/R24mwoz`), for `tm_mem_c4_nrand`, `tm_mem_ref_nrand`, `tm_mem_bge_nrand`, `lr_mem_c4`, `lr_mem_ref` and `lr_mem_bge`.

Method, as the script states it: per test turn, each arm's correctness is averaged over its seeds, and a pair difference is the mean over all test turns. The 95% interval comes from a bootstrap over test dialogues (convo_id), 10,000 resamples, numpy `default_rng(0)`, percentiles 2.5 and 97.5; the script notes that its RNG scheme may differ from `scripts/report_abcd.py`, so interval ends can differ slightly, while point estimates must match. A two-level interval (no rule) draws, in each resample, each arm's seeds with replacement and then the dialogues. R24.1: difference at least -1.0 point (strong: lower end at least -1.0). R24.2: lower end above 0. The script also checks that every arm and seed has the same test turns (y, convo_id, k), that each seed's stored test accuracy recomputes, that every seed file was written after the freeze time given as the second argument, and that the settings are equal within each arm.

Run: `.venv-dl/bin/python verification/recompute_r24.py results/confirm/R24mwoz 2026-10-22T20:16:52Z` (prints JSON).

Output: `recompute_r24_out.json` in this folder is the recorded output of this script; a rerun on 2026-10-23 printed it byte for byte. Read from it, in points:

| Pair | Difference | 95% interval | Two-level interval | R24.1 | R24.1 strong | R24.2 |
|---|---|---|---|---|---|---|
| `tm_mem_c4_nrand:tm_mem_bge_nrand` | 0.5275 | [-0.1797, 1.2272] | [-0.3242, 1.3985] | true | true | false |
| `tm_mem_ref_nrand:tm_mem_bge_nrand` | 0.0489 | [-0.6474, 0.7247] | [-0.8575, 0.9463] | true | true | false |
| `tm_mem_c4_nrand:lr_mem_c4` | -1.5527 | [-2.3245, -0.7866] | [-2.4098, -0.6994] | | | |
| `tm_mem_ref_nrand:lr_mem_ref` | -2.4935 | [-3.3306, -1.6357] | [-3.4401, -1.5296] | | | |
| `tm_mem_bge_nrand:lr_mem_bge` | -1.8899 | [-2.689, -1.1023] | [-2.7832, -1.041] | | | |

The output also gives the -2.0 reading as true for both pairs against the bge-large agent, 7355 test turns in 1000 dialogues, and an empty list of problems. The difference and the 95% interval equal the registered report's to four decimals (the same seed and the same draws). The two-level intervals are drawn differently from the registered ones (`twolevel_R24mwoz.json` in the run folder) and differ from them by at most 0.04 points.

## e71 and e76: `recompute_e71.py`

Results: not registered; dev results under frozen plans. e71: plan `logs/plan_e71_local.md` (frozen 2026-10-19T02:10:00Z) with the v2 amendment `logs/plan_e71_v2_amendment.md` (frozen 2026-10-19T03:30:28Z); section "e71: word edits around real conversations (dev; frozen plan with a v2 amendment; R19's ABCD agent)" of `docs/results.md`. e76: plan `logs/plan_e76_wordedit_c4.md` (frozen 2026-10-19T21:14:22Z); section "e76: e71's word-edit search on the agent with no neural network at any stage (dev; frozen plan; R23's tuned ABCD model)".

Reads: `results/explore/verify_e71.jsonl`, or the file in `results/explore/` named as the third argument (`verify_e76.jsonl` for e76); only the rows with the given tag and seed, one row per (state, rule group).

Method, as the script states it: a count, with no bootstrap. Outcomes are counted per (state, group). The headline is per state, so a state is certified at k when every one of its groups is certified at k. States whose latest customer utterance u lies outside the last 3 turns (`in_recent` false) are reported separately. The script also checks one row per (state, group), that every zero-edit row is exact, and that every stored breaking example was replayed through the real pipeline with the same choice and a broken rule.

Run: `.venv-dl/bin/python verification/recompute_e71.py` for e71 seed 1 (the defaults are tag `e71v2`, seed 1 and `verify_e71.jsonl`); `.venv-dl/bin/python verification/recompute_e71.py e71v2 2` for e71 seed 2; `.venv-dl/bin/python verification/recompute_e71.py e76 1 verify_e76.jsonl` for e76.

Recorded outputs:
- `e71v2_tally_final_seed1.txt`, e71 seed 1. It was written by an earlier form of the script that printed no seed in its first line, so that line reads `tag e71v2: ...` where the script now prints `tag e71v2 seed 1: ...`. A rerun differs from the file in that line only.
- `e71v2_tally_seeds12.txt`, e71 seeds 1 and 2: the first five lines of the output for each seed (the lines per rule family are not in the file). The first five lines of a rerun for each seed are identical to it.
- `e76_tally_final_seed1.txt`, e76 seed 1. A rerun is identical to it.

No tally of e71 seed 3 is recorded in this folder. The first and the state-level line of each, quoted (e71 seed 1, e71 seed 2, e76 seed 1):

```
tag e71v2: 579 (state, group) pairs over 200 states; duplicates ignored 0; zero-edit not exact 0; replay problems 0
all                          states 200: certified at k = 1 186 (93.0%), at k = 2 151 (75.5%); own choice breaks a rule 4; some 1-word edit breaks a rule 10
```

```
tag e71v2 seed 2: 579 (state, group) pairs over 200 states; duplicates ignored 0; zero-edit not exact 0; replay problems 0
all                          states 200: certified at k = 1 185 (92.5%), at k = 2 152 (76.0%); own choice breaks a rule 3; some 1-word edit breaks a rule 12
```

```
tag e76 seed 1: 579 (state, group) pairs over 200 states; duplicates ignored 0; zero-edit not exact 0; replay problems 0
all                          states 200: certified at k = 1 183 (91.5%), at k = 2 153 (76.5%); own choice breaks a rule 6; some 1-word edit breaks a rule 11
```
