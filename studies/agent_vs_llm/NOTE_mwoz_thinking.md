# Note to PLAN_mwoz_dev.md: thinking arms on MultiWOZ (written 2026-10-27T00:42:03Z)

Before any MultiWOZ thinking answer. Any claim that the readable agent is not worse than LLM agents must hold against
the strongest LLM configuration on every dataset, so L35t and L9t (PLAN_llm_q3_dev.md settings, thinking on) also run
on MultiWOZ's 500 dev rows (seed 2, the rows of L9 and L35). llm_q3_job3.sh is llm_q3_job2.sh with the Q1 exclusion
applied only where a Q1 file exists (none for MultiWOZ). The plan's 3-hour rule for thinking arms applies here too.
Seen before this note: MultiWOZ L9 0.300 and L35 0.348 against the agent's 0.374 on these rows.
