# R6: amendment, part 1 written 2026-10-16 at 00:45Z, right after the first confirmatory jobs were queued (00:45:18Z) and before any of them could finish (each first builds a 5-fold router)

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

## Source changes since the freeze (2026-10-15T22:58:03Z)

- **`scripts/r6_run.py`, rule learner:** the grid search is vectorised. The grid (val-margin quantiles), the objective (val mean return) and the choice (first maximum in a-b-c order) are unchanged. The loop version took about 40 minutes per seed under the load of this machine.
  - Check on seed 1 (val, exploratory): identical thresholds (route if margin >= 6315.84, after asking route if >= 935.75, never escalate at step 0) and an identical val return of 0.77477, from `e12_rule` (loop) and `vec_check` (vectorised).
- **`scripts/r6_seed.sh`:** runs the learners of one seed in order and skips a learner already recorded for that seed.
- **`src/dl/tm.py`:** the chunked update (only above 50,000,000 clause-literal pairs; R6 machines are 2,000 x about 250 literals) and literal-budget modes (not used). **`src/dl/explain.py`:** printing only.
- No change touches the environment (`src/dl/mdp.py`), the TD or fitted-Q training, or the evaluation.

## Selections

Written in part 2, when every `e12_*` run has finished (two were still running at the time of part 1).

# Part 2, written 2026-10-16 at 01:08Z, before any TM-Q, TM-myopic, TM-Q-nomem or Lin-Q confirmatory run

## Selections (rule of the frozen text: highest val mean return among the e12 runs, seed 1)

**TM-Q.** Candidates (val mean return):

| Run | Val return |
|---|---|
| `e12_fqi_k0003_e10` | 0.7642 |
| `e12_tmq_k0005` | 0.7626 |
| `e12_tmq_k005` | 0.7619 |
| `e12_fqi_k0001` | 0.7603 |
| `e12_tmq_k0005_u6k` | 0.7595 |
| `e12_fqi_k0003` | 0.7566 |
| `e12_fqi_k001` | 0.7559 |
| `e12_tmq_k00003_u6k` | 0.7557 |
| `e12_tmq_k002` | 0.7533 |
| `e12_tmq_k0005_u12k` | 0.7530 |
| `e12_tmq_k0001` | 0.7509 |
| `e12_fqi_k003` | 0.7475 |
| `e12_tmq_k001` | 0.7440 |
| `e12_tmq_k001_u6k` | 0.7368 |
| `e12_tmq_k02` | 0.7017 |
| `e12_tmq_k005_u6k` | 0.6923 |

- **Chosen: fitted-Q iteration, kappa 0.003, 4 iterations of 10 epochs, 60,000 random-action episodes, 2,000 clauses, T 1,000, s 4.**
- TM-myopic and TM-Q-nomem use the same setting.

**Lin-Q.** Candidates: `e12_linq_lr0003_u6k` 0.7698, `e12_linfqi_lr001` 0.7676, `e12_linq_lr001` 0.7633. **Chosen: online TD, lr 0.003, 6,000 updates.**

**References on val (seed 1):** rule 0.7748, always ask then route 0.7419, always route -0.0103.

## Runs

`logs/jobs_R6.txt` (built by `scripts/build_r6_jobs.py`): `scripts/r6_seed.sh` for seeds 1 to 10. The rule runs started at 00:45:18Z (part 1) and are skipped where already recorded.

# Part 3, written 2026-10-16 at 02:40Z: disclosures found by internal audit 1

1. **The candidate set stayed open after the freeze.**
   - The frozen rule picks the best val return "among the exploratory runs tagged `e12_*`" and does not close the set at a time. 7 of the 16 TM-Q candidates were queued after the freeze (22:58:03Z): `jobs_e12d` at 23:00:45Z (`e12_fqi_k0001`, `e12_fqi_k0003`, `e12_tmq_k0001`, `e12_tmq_k00003_u6k`) and `jobs_e12e` at 23:17:20Z (`e12_tmq_k0005_u6k`, `e12_tmq_k0005_u12k`, `e12_fqi_k0003_e10`).
   - The chosen run, `e12_fqi_k0003_e10` (val 0.7642), is one of them.
   - It also changes the number of fitted-Q iterations (4) and epochs (10). The frozen text names only "any kappa or updates"; here the iterations and epochs of fitted-Q are read as its "updates".
   - The best candidate queued before the freeze is `e12_tmq_k0005`: online TD, kappa 0.005, 3,000 updates, val 0.7626.
   - Every selection used val only, and no R6 test result existed when the selection was made: selection at 01:08:45Z, first TM-Q test result after 01:40Z.
   - **Sensitivity run, post hoc and not part of the R6 verdict:** the `e12_tmq_k0005` setting on the same 10 seeds and test episodes, tag `R6sens` (`logs/jobs_R6sens.txt`), reported beside R6.
2. **Model files of TM-Q were overwritten by the ablation.**
   - `scripts/r6_run.py` saved the TM-Q machine as `r6_tm_q_seed<k>_R6.npz` with and without memory literals, so the no-memory run of a seed replaced the TM-Q file. Seeds 5, 6 and 7 lost their TM-Q (memory) machine file. Their statistics in `r6.jsonl` are unaffected.
   - Fixed at 02:37:28Z: the no-memory file is now `r6_tm_q_nomem_seed<k>_R6.npz`.
   - The intact TM-Q files of seeds 8 and 9 were copied to `*.memory_copy.npz` before seed 8's no-memory run (started 02:35:23Z with the old code) could replace them.
3. **Corrections to the code-change list of part 1:**
   - the literal-budget modes were already in `src/dl/tm.py` at the freeze;
   - the R6 machines have 61 bits (122 literals), not about 250;
   - `scripts/select_r5r6.py` and `scripts/build_r6_jobs.py` are further helper scripts.
4. **Test inputs in exploration.**
   - `scripts/r6_run.py` builds the router outputs for every split, including test and oos_test, also in exploratory runs (the cache `results/explore/r6_router_seed1.pt`).
   - No exploratory evaluation read them (eval = val), and no labels of the test split enter the cache.
5. **Caution for reading R6.2.**
   - The out-of-scope share is 3.2% in val episodes, 18.2% in test episodes and 0.15 in the TD learners' training episodes.
   - The rule is tuned on val, so on test it sees far more out-of-scope queries than it was tuned for.
   - TM-Q's gain over the rule therefore includes a better match of its training distribution to test. Lin-Q, trained the same way, shows the same gain.

## Part 4, 2026-10-16 04:32Z: result of the post-hoc sensitivity run (announced in part 3)

The best candidate queued before the freeze (online TD, kappa 0.005, 3,000 updates; tag `R6sens`), run on the same 10 seeds and test episodes, returns **0.5976**, against **0.5971** for the registered TM-Q. Minus rule: +0.042 [+0.036, +0.049] (percentile bootstrap over seeds). The R6 conclusion does not depend on the choice between the two.
