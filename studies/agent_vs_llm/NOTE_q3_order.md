# Note to PLAN_llm_q3_dev.md: order of the arms (written 2026-10-26T23:07:27Z)

- L9t (Qwen3.5-9B, thinking on) started 2026-10-26T22:52:18Z and averaged 20.4 s per row over its first 40 ABCD rows
  (projected 2.8 h per dataset, under the plan's 3 h, so it keeps all 500 rows).
- When the Qwen3.6-35B-A3B Q4_K_M file was ready (CONVERT OK 23:06:35Z; sha256 f92dceb280a94b54...), L9t was stopped
  at 23:07Z after 42 ABCD rows (queue exit 143) so that the much faster L35 arm runs first. L9t then resumes with the
  same command: the client keeps the finished rows and answers the remaining ones in the same order, at temperature 0.
- No number of any Q3 arm was used for this decision; only the timing of L9t's first rows and its 40-row running
  accuracy (0.725) had been seen.

## Addendum (2026-10-26T23:52:44Z)

- L35 (thinking off) ended 23:51:34Z: ABCD 0.840, STAR 0.608 on the fresh rows, where the agent has 0.858 and 0.666.
  So on ABCD the agent minus L35 is +1.8, below the plan's +3.0 bar: the ABCD claim cannot be met whatever the thinking
  arms do (the rule needs the agent at least +3.0 above EVERY arm). On STAR it is +5.8, so STAR's claim is still open.
- The thinking arms therefore run STAR first (llm_q3_job2.sh, the same job with an optional dataset order,
  Q3_DATASETS="star abcd"); their ABCD rows follow as description. The resumed L9t job of 23:51:44Z was stopped at
  23:52:13Z before it answered a new row (still 42 ABCD rows). L35t goes before L9t because it is the stronger model.
- Seen before this decision: the L35 accuracies above and the earlier arms; no L35t answer, no new L9t answer.
- (2026-10-26T23:53:26Z) The thinking arms run STAR only first; their ABCD rows are queued at the end (descriptive,
  ABCD's claim is already closed), so that a MultiWOZ dev check can run before them.
