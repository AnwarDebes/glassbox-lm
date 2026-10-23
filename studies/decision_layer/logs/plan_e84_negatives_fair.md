# e84: random negatives for the bge-large and C4 agents on MultiWOZ, and the R24 gate re-check (dev only)

Written 2026-10-22 (10:35Z), before any e84 number. No test split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**The question (fixed now):** with random negative classes for every agent, how do the three MultiWOZ agents compare
on dev, and does a network-free agent now pass the R24 gate?

**Why:** e83 (dev; recompute-verified, the three-part review not run) found that random negatives lift the refined
agent's MultiWOZ reader:
- +4.23 at best-dev and +5.98 on the last-10 mean at 16k on seeds 1-2;
- TM minus LR -1.13, from -5.35.
They cost STAR -11.91 and ABCD -4.00. So random negatives are a MultiWOZ setting, and a fair MultiWOZ comparison needs
the same rule for every agent.

## What was known (dev, MultiWOZ, 16k, s 8; results/explore/abcd.jsonl)

| Agent | Hardest negatives (best-dev; last-10) | Random negatives (best-dev; last-10) | LR on its bits |
|---|---|---|---|
| refined (a3B, view B) | 0.3123, 0.3093 (seeds 1-2); 0.2857, 0.2824 | 0.3498, 0.3563 (e83b); 0.3430, 0.3446 | 0.3643 (C 0.001) |
| bge-large (R18's view) | 0.3359 (seed 1 only); 0.3120 | not run | 0.3659 (C 0.01) |
| C4 (view B, no network at any stage) | 0.3038, 0.3079 (seeds 1-2); 0.2841, 0.2837 | not run | 0.3586 (C 0.001) |

- **The R24 gate so far:**
  - it was "at least -1.0" for the network-free agent minus bge-large, on the tuned 16k dev rows;
  - it failed at -3.21 (e77b, C4) and -2.36 (e82, refined, seed 1);
  - R24 returns only if a later stack passes the same gate on fresh dev runs.
- **What is fresh:** the refined random rows (e83b) were used by C1 and C2, not by any gate. The bge-large and C4
  random rows are new.

## Arms (4 jobs, logs/jobs_e84.txt)

- **Agents:** bge-large and C4, with neg_sampling random. Each runs through scripts/knob_run.py, which saves the dev
  predictions.
- **Settings:** 16,000 clauses, T 16,000, s 8 (each agent's tuned s), seeds 1 and 2; everything else as in e83.
- **Beside them:** the refined agent's e83b rows (e83_multiwoz_a3B_c16000_s8_nrand, seeds 1-2).

## Rules (fixed now), each on the mean of seeds 1-2

- **G-ref, the R24 gate for the refined agent** (network-free at run time, teacher at training time: bge-base-en-v1.5):
  refined minus bge-large is at least -1.0 at best-dev.
- **G-C4, the R24 gate for the C4 agent** (no neural network at any stage): C4 minus bge-large is at least -1.0 at
  best-dev.
- **Steady:** a gate is steady if it also holds on the last-10 mean.
  - A gate met at best-dev only is written "met on best-dev only" and allows no R24 draft.
  - A steady gate allows an R24 draft: a registration of its own, frozen before MultiWOZ's test split is read.
- **Reported beside them, with no rule:** each agent's random minus hardest negatives, and TM minus LR on its own bits.
- **Words:** the rule's words only. No "beats", "better than" or "first".

## Checks and commands (pinned)

- **The checker** is scripts/e84_check.py. Its test, tests/test_e84_check.py, PASSED: both gates met at their edge
  (-1.0), a gate missed, met at best-dev only, and five refusals.
- **The two commands, run by hand:**
  1. `bash scripts/seq_queue.sh logs/jobs_e84.txt logs/e84_queue.log` (background);
  2. then `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python scripts/e84_check.py > logs/e84_check.out`.
- **The end line** "e84 end: <branch> <UTC> exit <code>" goes to logs/e84_end.log.
- **GPU estimate:** 2 bge-large jobs (about 10 min each) and 2 C4 jobs (about 29 min each), so about 80 min. It runs
  after the speed block and the R3 job of the text-understanding line.

## Hashes (sha256sum)

| File | SHA-256 |
|---|---|
| scripts/e84_check.py | df63a7f2068cd532ec029d532690b2823f1528ccbf89e1396564337084423eb2 |
| tests/test_e84_check.py | 573f4d68f04b19d4bef5b7fa19a0a55e705c957e87e3941b9e3cd2d0bfe2d1b1 |
| logs/test_e84_check.out | 6e89ff297f673209c111b32a16e44eaccf7ec02c096adafe2f0c9f258c35120e |
| logs/jobs_e84.txt | 594f3040e09592f78809eae5ef65f759430a85ab5d6fa9a6870d5f1f77262a42 |
| scripts/knob_run.py | 113cb74eca36d1f9c3fbea889a2fe8aad2c1ad91878af9f17d93cffa14d65e4e |
| data/tables/c4words_S1000_v1/table.npz | e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a |
| data/emb/multiwoz__bge-large.npz | 324f6d626675c5a3600125561e12784d8b8df19519893644a74b4f3cc5ebe421 |
| data/multiwoz/examples_train.json.gz | 0e20a31434de10a5248f0b10ab01e7474ceb5e020cd5cbc7048bfe8f5db69e9b |
| data/multiwoz/examples_dev.json.gz | 3414cc8a6ecc64aafa0df070748b9d502a12e7996420b84a12df1b094a89fcd2 |

The dev bits checked by e84_check.py:
- refined 612b8f0b75e32a30b50493eaedc1eec9d6bac6893d48075144a1f48fad2e23a3;
- bge-large b068b64787582fc8c00dd8a446cb415fcb9737d3069e7111f092503bf7bf2797;
- C4 311e8933d2724bf8fc3ee1a5cef412010bd3a580c076217e3a964bcd286d68ba.
