# Plan ABL-dev: does the written policy explain the LLM agents' accuracy? (STAR and ABCD dev, fresh rows)

Dev only, exploratory. No test file is opened. Written and frozen before any answer of these arms.

## Why

On the fresh dev rows (PLAN_llm_q3_dev.md, thinking off), the readable agent beats the LLM agents by 5.8 to 13.8 points
on STAR but by only 1.8 to 4.0 on ABCD. ABCD's prompt holds the company's full written guidelines (95 KB); STAR's holds
24 task specifications that give the wizard's replies and only the usual next action. If the LLM agents' accuracy
comes from the written policy, removing it should cost them much on ABCD and little on STAR.

## Arms (the 500 fresh rows of PLAN_hybrid_dev.md, seed 2; thinking off)

- L9-noG and L35-noG: Qwen3.5-9B and Qwen3.6-35B-A3B with `--no-guidelines` (no ABCD guidelines, no STAR task
  specifications); everything else as in L9 and L35 (action list, 10 retrieved solved examples, the dialogue).
- Compared with L9 and L35 (the same rows, with the written policy), and with the readable agent.

## Reading (fixed now)

- Cost of removing the written policy = accuracy(with) minus accuracy(without), per model and dataset, with a 95%
  bootstrap interval over rows (seed 20261026).
- "The written policy explains the LLM agents' competence on ABCD" is stated only if, for both models, the cost on
  ABCD is at least +5.0 points with the interval above 0, and the cost on ABCD exceeds the cost on STAR by at least
  +3.0 points (difference of the two costs; its interval reported, no rule on it). Otherwise the reading is reported as
  measured and not stated.
