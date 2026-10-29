# Note: CPU cost of one decision of the fine-tuned arm (written 2026-10-27T07:43:47Z)

Cost only, no rule, dev rows. ft_speed.py times the fine-tuned arm's network (bge-base-en-v1.5 with a linear head of
the dataset's label count, the same architecture ft_baseline.py trains; the trained weights are not saved and do not
change the compute) on 300 dev rows per dataset: tokenization plus forward pass, batch 1, one CPU thread
(torch.set_num_threads(1), OMP_NUM_THREADS=1), nice 19, after 20 warm-up rows. Rough timing on a shared machine.

| Dataset | median tokens | median ms | p90 ms | file sha256 |
|---|---|---|---|---|
| star | 145 | 256 | 382 | 4c92ab9003a21a41 |
| abcd | 100 | 195 | 281 | 433f2c5b5672b6b4 |
| multiwoz | 152 | 263 | 374 | 6916badf190ed2a0 |

Beside it (earlier): the readable agent's decision on ABCD takes a median 0.50 ms on one CPU thread
(runs/dry/agent_abcd_s1/speed.json, 4,000 clauses); STAR and MultiWOZ (16,000 clauses) are timed on the registered
seed-1 models. The LLM arms take seconds per row on the V100 (their preds files, field s).

## Added 2026-10-27T09:14:27Z: the agent's registered seed-1 models and the TFIDF arm (one CPU thread, batch 1, dev rows)

| Arm | Dataset | median ms | p90 ms | file sha256 |
|---|---|---|---|---|
| agent (R2x seed 1, 15997 clauses; dev accuracy reproduced) | star | 2.71 | 2.98 | 4c72b8a9eb9eef1a |
| agent (R2x seed 1, 3995 clauses; dev accuracy reproduced) | abcd | 0.50 | 0.55 | 94aa9ab59988857a |
| agent (R2x seed 1, 16000 clauses; dev accuracy reproduced) | multiwoz | 2.98 | 3.29 | 0336dd62e7998adf |
| TFIDF (C 4; dev accuracy reproduced) | star | 0.90 | 1.07 | 87203ab2a31b1d38 |
| TFIDF (C 16; dev accuracy reproduced) | abcd | 0.83 | 0.92 | a2078d0560588a95 |
| TFIDF (C 1; dev accuracy reproduced) | multiwoz | 0.88 | 1.01 | 8a19305136b77c7f |
