# Results of the agent_vs_llm study

The registered test REG_agent_vs_llm (frozen 2026-10-27T18:54:13Z) compares the readable agent of
decision_layer (no neural network at any stage: a word table counted from C4 text, memory bits of the actions taken, a
Tsetlin machine) with LLM* (Qwen3.6-35B-A3B at 4 bits with 50 retrieved examples, the dev-best of the local LLM
configurations), a fine-tuned bge-base (FT) and TF-IDF logistic regression (no rule), for next-action prediction on
STAR, ABCD and MultiWOZ 2.2. Every number below is read from `results_REG_agent_vs_llm/` (registered) or `runs/` (dev).
An independent recompute of every registered number matched (`results_REG_agent_vs_llm/verification/`).

## Every rule's outcome (registered statistic: seed 1, bootstrap over dialogues, 95%)

| Dataset | Rows | AGENT | LLM* | FT | AGENT - LLM* (95%) | NI (-3) | NI2 (-2) | SUP (+3) | FT - AGENT (95%) | FT | 5-seed mean AGENT - LLM* (two-level 95%) | Verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STAR | 4000 | 0.6490 | 0.6340 | 0.7350 | +1.50 [-0.02, +3.05] | met | met | not met | +8.60 [+7.24, +9.98] | met | +1.28 [-0.26, +2.84] | MATCH |
| ABCD | 3608 | 0.8379 | 0.8379 | 0.8786 | +0.00 [-1.17, +1.14] | met | met | not met | +4.07 [+3.11, +5.06] | met | -0.14 [-1.23, +0.91] | MATCH |
| MultiWOZ | 4000 | 0.3548 | 0.3660 | 0.4268 | -1.12 [-2.78, +0.57] | met (seed 1 only) | not met | not met | +7.20 [+5.51, +8.78] | met | -1.70 [-3.21, -0.14] | MATCH |

## No-rule readings

| Dataset | AGENT - LLM* 90% (inside +-3?) | TFIDF | TFIDF - LLM* (95%) | AGENT - TFIDF (95%) | AGENT - TFIDF two-level | FT - AGENT two-level | LLM* s/row (median) | LLM* invalid |
|---|---|---|---|---|---|---|---|---|
| STAR | [+0.21, +2.79] (yes) | 0.6455 | +1.15 [-0.47, +2.81] | +0.35 [-1.00, +1.72] | [-1.10, +1.38] | [+7.57, +10.14] | 10.6 | 1 |
| ABCD | [-0.98, +0.97] (yes) | 0.8473 | +0.94 [-0.14, +2.00] | -0.94 [-1.92, +0.03] | [-1.95, -0.23] | [+3.36, +5.08] | 11.13 | 1 |
| MultiWOZ | [-2.52, +0.30] (yes) | 0.3540 | -1.20 [-2.69, +0.27] | +0.07 [-1.32, +1.47] | [-1.76, +0.74] | [+6.27, +9.23] | 11.63 | 0 |

## Claim lines

- NI met on all three datasets: the readable agent is not worse than the best of the local LLM configurations we ran (chosen on dev; the 35B model at 4-bit), at a margin of 3 points (NI).
- NI2 (2 points) met on: STAR, ABCD; not on: MultiWOZ.

## Seed-1-only notes

- MultiWOZ NI: met on seed 1, not on the 5-seed mean

## What must travel with these numbers

- MultiWOZ NI is met on seed 1 only, not on the 5-seed mean (two-level lower end -3.21); seed 1 is the best of the 5
  seeds on these rows (+0.58 points over their mean), though it was fixed as the statistic before the run.
- The 3-point NI margin was set after a dev result (disclosed in the registration); NI2 at 2 points is met on STAR
  and ABCD only.
- SUP is met on no dataset: never write that the agent beats the LLM.
- FT is ahead of the agent on all three datasets (+8.60, +4.07, +7.20 points).
- The agent is level with TF-IDF logistic regression on STAR and MultiWOZ; on ABCD TF-IDF is 0.94 points ahead (5-seed
  interval excludes 0). The result is not specific to the Tsetlin machine.
- LLM* is the dev-best of 4-8 local configurations, not the best LLM that exists; the evaluation is offline (the
  logged next action is the label).
- Test uses: STAR 7th, ABCD 15th, MultiWOZ 2nd. The agent's test accuracy was known at the freeze; LLM*'s, FT's and
  TF-IDF's were not.

## Dev runs before the registration (500 fresh dev rows per dataset; no test row)

The LLM configurations tried on dev; LLM* is the highest dev accuracy per the rule in `runs/llm_star_choice.json`
("highest dev accuracy on the 500 fresh rows; ties to the lower mean seconds per row"). Plans: `PLAN_llm_dev.md`, `PLAN_llm_q3_dev.md`, `PLAN_mwoz_dev.md` and the amendments `AMEND_*.md`.

| Dataset | Config | Model | Thinking | Retrieved examples | Dev accuracy (500 rows) | Mean s per row |
|---|---|---|---|---|---|---|
| star | L9 | Qwen3.5-9B-Q8_0 | off | 10 | 0.528 | 0.8 |
| star | L35 | Qwen3.6-35B-A3B-Q4_K_M | off | 10 | 0.608 | 2.41 |
| star | L35t | Qwen3.6-35B-A3B-Q4_K_M | on | 10 | 0.552 | 13.92 |
| star | L35t16 | Qwen3.6-35B-A3B-Q4_K_M | on | 10 | 0.558 | 27.04 |
| star | L35k50 | Qwen3.6-35B-A3B-Q4_K_M | off | 50 | 0.638 | 10.57 |
| star | L35k100 | Qwen3.6-35B-A3B-Q4_K_M | off | 100 | 0.622 | 22.44 |
| star | L9-noG | Qwen3.5-9B-Q8_0 | off | 10 | 0.568 | 0.99 |
| star | L35-noG | Qwen3.6-35B-A3B-Q4_K_M | off | 10 | 0.628 | 2.58 |
| abcd | L9 | Qwen3.5-9B-Q8_0 | off | 10 | 0.818 | 1.05 |
| abcd | L35 | Qwen3.6-35B-A3B-Q4_K_M | off | 10 | 0.840 | 2.66 |
| abcd | L35t16 | Qwen3.6-35B-A3B-Q4_K_M | on | 10 | 0.822 | 10.5 |
| abcd | L35k50 | Qwen3.6-35B-A3B-Q4_K_M | off | 50 | 0.864 | 11.54 |
| abcd | L9-noG | Qwen3.5-9B-Q8_0 | off | 10 | 0.774 | 0.97 |
| abcd | L35-noG | Qwen3.6-35B-A3B-Q4_K_M | off | 10 | 0.820 | 2.48 |
| multiwoz | L9 | Qwen3.5-9B-Q8_0 | off | 10 | 0.300 | 1.06 |
| multiwoz | L35 | Qwen3.6-35B-A3B-Q4_K_M | off | 10 | 0.348 | 2.73 |
| multiwoz | L35t16 | Qwen3.6-35B-A3B-Q4_K_M | on | 10 | 0.356 | 10.48 |
| multiwoz | L35k50 | Qwen3.6-35B-A3B-Q4_K_M | off | 50 | 0.376 | 11.49 |

Fine-tuned transformers on dev (`PLAN_ft_dev.md`):

| Dataset | Model | Seed | Best dev accuracy | Result file |
|---|---|---|---|---|
| abcd | bert-base-uncased | 1 | 0.8716 | `runs/ft_abcd_bert_s1/result.json` |
| abcd | BAAI/bge-base-en-v1.5 | 1 | 0.8727 | `runs/ft_abcd_bgebase_s1/result.json` |
| abcd | roberta-base | 1 | 0.8697 | `runs/ft_abcd_roberta_s1/result.json` |
| multiwoz | BAAI/bge-base-en-v1.5 | 1 | 0.4327 | `runs/ft_mwoz_bgebase_s1/result.json` |
| star | bert-base-uncased | 1 | 0.7380 | `runs/ft_star_bert_s1/result.json` |
| star | BAAI/bge-base-en-v1.5 | 1 | 0.7409 | `runs/ft_star_bgebase_s1/result.json` |
| star | roberta-base | 1 | 0.7399 | `runs/ft_star_roberta_s1/result.json` |
| abcd | BAAI/bge-base-en-v1.5 | 2 | 0.8713 | `runs/ftseed_abcd_bgebase_s2/result.json` |
| multiwoz | BAAI/bge-base-en-v1.5 | 2 | 0.4257 | `runs/ftseed_mwoz_bgebase_s2/result.json` |
| star | BAAI/bge-base-en-v1.5 | 2 | 0.7373 | `runs/ftseed_star_bgebase_s2/result.json` |

TF-IDF logistic regression on dev (`NOTE_tfidf_dev.md`):

| Dataset | C | Dev accuracy | Result file |
|---|---|---|---|
| abcd | 16.0 | 0.8415 | `runs/tfidf_abcd/result.json` |
| multiwoz | 1.0 | 0.3506 | `runs/tfidf_multiwoz/result.json` |
| star | 4.0 | 0.6529 | `runs/tfidf_star/result.json` |

Time per decision on one CPU thread, batch 1, median (`runs/cost/`; rough, on a shared machine):

| Dataset | Agent (TM) | TF-IDF LR | Fine-tuned bge-base |
|---|---|---|---|
| star | 2.71 ms | 0.90 ms | 256 ms |
| abcd | 0.50 ms | 0.83 ms | 195 ms |
| multiwoz | 2.98 ms | 0.88 ms | 263 ms |

Other dev readings, each with its frozen plan and its output: the written-policy ablation (`PLAN_ablation_dev.md`,
`runs/abl_table.json`), the LLM-on-top hybrid (`PLAN_hybrid_dev.md`, `runs/hybrid_*.json`), the cascade
(`NOTE_cascade_dev.md`, `runs/cascade_*.json`), the synthetic Zipf grid (`PLAN_zipf_synth.md`, `runs/zipf_report.json`),
the readability of the agent's decisions (`runs/readability/`) and the energy note (`NOTE_energy.md`).

## Log size on dev (`PLAN_logsize_dev.md`, frozen 2026-10-28T08:04:16Z; no test row)

How many logged dialogues the readable agent needs to reach LLM*. Pools of 64, 256, 1024 and 4096 training dialogues
(nested, one draw) and the whole split; every arm at its whole-split settings, nothing re-tuned; the same 500 dev rows
per dataset; 95% bootstrap over rows. The gate passed (the ABCD whole-split rerun equals the earlier run,
0.8417480998914224). All values, with TF-IDF's differences, the share of rows whose action never occurs in the pool,
the LLM's invalid answers and its median seconds per row: `runs/logsize/table.json` (written by `logsize_table.py`).

| Dataset | Dialogues | AGENT | LLM* | TF-IDF | AGENT - LLM* (95%) | Label |
|---|---|---|---|---|---|---|
| STAR | 64 | 0.202 | 0.516 | 0.316 | -31.4 [-36.4, -26.6] | LLM* ahead |
| STAR | 256 | 0.434 | 0.576 | 0.494 | -14.2 [-19.2, -9.2] | LLM* ahead |
| STAR | 1024 | 0.598 | 0.612 | 0.592 | -1.4 [-6.4, +3.4] | level within the interval |
| STAR | 4096 | 0.670 | 0.616 | 0.680 | +5.4 [+1.0, +10.0] | AGENT ahead |
| STAR | 4811 (whole) | 0.666 | 0.638 | 0.672 | +2.8 [-1.4, +7.0] | level within the interval |
| ABCD | 64 | 0.358 | 0.838 | 0.534 | -48.0 [-53.0, -43.0] | LLM* ahead |
| ABCD | 256 | 0.558 | 0.844 | 0.654 | -28.6 [-33.6, -23.6] | LLM* ahead |
| ABCD | 1024 | 0.746 | 0.848 | 0.802 | -10.2 [-14.4, -6.0] | LLM* ahead |
| ABCD | 4096 | 0.840 | 0.864 | 0.838 | -2.4 [-5.8, +1.0] | level within the interval |
| ABCD | 8034 (whole) | 0.858 | 0.864 | 0.866 | -0.6 [-3.6, +2.6] | level within the interval |
| MultiWOZ | 64 | 0.182 | 0.338 | 0.134 | -15.6 [-20.0, -11.2] | LLM* ahead |
| MultiWOZ | 256 | 0.314 | 0.380 | 0.236 | -6.6 [-11.8, -1.4] | LLM* ahead |
| MultiWOZ | 1024 | 0.334 | 0.362 | 0.304 | -2.8 [-7.8, +2.2] | level within the interval |
| MultiWOZ | 4096 | 0.364 | 0.374 | 0.360 | -1.0 [-5.8, +3.8] | level within the interval |
| MultiWOZ | 8437 (whole) | 0.374 | 0.376 | 0.384 | -0.2 [-4.8, +4.4] | level within the interval |

Readings, in the plan's words (dev only):

- With 64 and 256 logged dialogues the LLM agent is ahead of the readable agent on all three datasets, and on ABCD
  also with 1024.
- The readable agent reaches the LLM agent from 4096 logged dialogues on (point estimates) on STAR. On ABCD and
  MultiWOZ it does not reach it at any size (point estimates); there the interval holds 0 from 4096 dialogues on
  (ABCD) and from 1024 on (MultiWOZ).
- With 4096 logged dialogues the readable agent is ahead of the LLM agent on STAR. This is 1 interval of 15, and at
  the whole split the interval holds 0.

What must travel with these numbers:

- Dev only: 500 rows per dataset, one pool draw, one agent seed, one LLM run; FT was not run. Nothing is re-tuned at
  small sizes: the agent keeps its whole-split clauses, s and epochs, the LLM its 50 retrieved examples, TF-IDF its C.
- The agent's epoch is picked on the whole dev split, which includes the 500 rows; LLM* was chosen on these 500 rows
  at the whole split.
- LLM* also reads the written policy (ABCD's guidelines, STAR's task specifications) and brings its pretraining: on
  ABCD it is at 0.838 with 64 dialogues and at 0.864 with the whole split.
- The 500 rows are easier than the whole dev split: at the whole split the agent scores 0.666, 0.858 and 0.374 on them
  against 0.654, 0.842 and 0.350 on all dev rows (STAR, ABCD, MultiWOZ). The comparisons are paired on the same rows.
- LLM* takes 10.5 to 15.4 s per row (median per size and dataset, one GPU); the agent decides in milliseconds (the
  time table above).
- A separate recompute from the raw predictions, whose files are kept outside the repository, gave every value of the
  table again (interval ends within 0.2 points with another bootstrap seed). The 50 solved examples the LLM saw were rebuilt for every row and size; all came from
  the pool.
