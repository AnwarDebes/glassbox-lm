# Note: agent runs with saved dev predictions, and the frozen-encoder classifiers (dev only, descriptive)

Written and frozen before these runs. They give the per-row dev predictions that PLAN_llm_dev.md (Q1 comparison on the
same 500 rows, and Q2) needs, and the strongest frozen-encoder baselines of the study on STAR and ABCD dev.

- `agent_preds.py` (re-uses the study's src/dl unchanged):
  - the C4 network-free agent at its tuned settings, seed 1: STAR 16,000 clauses, s 16; ABCD 4,000 clauses, s 16
    (parts mem, words, xcust, xrecent; table c4words-S1000-v1). Check: its dev accuracy must equal the recorded rows
    (STAR 0.6545, ABCD 0.8417, docs/results.md e74c/e75); a different value is reported, not hidden;
  - the bge-large frozen-encoder classifiers on the raw embeddings plus the memory bits (lr_raw with the study's C grid,
    mlp_raw), seed 1, STAR and ABCD.
- No rule; every number is reported. GPU jobs through run_queue.sh, after the fine-tuning queue.
