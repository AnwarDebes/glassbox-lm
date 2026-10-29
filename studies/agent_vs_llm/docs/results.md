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

## Still running

- The logsize dev study (`PLAN_logsize_dev.md`, frozen 2026-10-28T08:04:16Z, dev only): how many logged dialogues the
  readable agent needs to reach LLM*. Its TF-IDF arm and 14 agent jobs had ended, and its LLM arm was running (its
  first job started 2026-10-29T09:08:45Z), when this page was written. Its results are not on this page yet.
