# Plan LLM-Q3-dev: stronger LLM agents against the readable agent (STAR and ABCD dev, fresh rows)

Dev only, exploratory. No test file is opened. Written and frozen before any answer of these arms.

## Why

Q1 (PLAN_llm_dev.md) found the readable network-free agent 12.2 points above Qwen3.5-9B on STAR and 2.0 above it on
ABCD (500 rows each, thinking off). A claim of that kind must survive the strongest LLM agents this machine can run:
reasoning on, and a larger, newer model.

## Arms (the Q1 prompt: actions, ABCD guidelines / STAR task specifications, 10 retrieved solved examples)

- L9: Qwen3.5-9B Q8_0, thinking off (the LLM-alone arm of PLAN_hybrid_dev.md on the same rows, reused).
- L9t: Qwen3.5-9B Q8_0, thinking on (reasoning budget unlimited, answer from the final content, max 4,096 tokens).
- L35: Qwen3.6-35B-A3B (Qwen/Qwen3.6-35B-A3B @ 995ad96e, apache-2.0, the newest Qwen on 2026-10-26), converted with
  the pinned llama.cpp b11742 to Q8_0 and requantized to Q4_K_M, thinking off.
- L35t: the same model, thinking on.
- Beside them: the readable C4 agent, the best fine-tuned transformer with saved predictions, TF-IDF LR.

Server: `serve_headline.sh` (pinned build, alone on the GPU, port 18083). Rows: the 500 fresh rows of PLAN_hybrid_dev.md
(seed 2, Q1's rows excluded), the same for every arm. If a thinking arm needs more than 3 hours on one dataset, it runs
on the first 250 of those rows (sorted by row id) and that is reported.

## Reading (fixed now)

- Accuracy per arm; paired differences against the readable agent, 95% bootstrap interval over rows (seed 20261026).
- "The readable agent beats LLM agents" is carried to a registered test for a dataset only if the agent minus EVERY
  LLM arm is at least +3.0 points with the interval above 0 on that dataset. Otherwise that claim is not made for it.
- No rule against the fine-tuned transformer here.

## Limits

- One prompt; thinking may change the answer format (mapped as in Q1).
- 500 rows: about plus or minus 3 to 4 points on paired differences.
