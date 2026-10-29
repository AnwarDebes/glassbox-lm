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
