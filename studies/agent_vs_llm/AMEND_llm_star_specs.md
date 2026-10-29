# Amendment to PLAN_llm_dev.md: STAR's task specifications in the LLM prompt (written 2026-10-26T21:04:13Z)

Before any LLM answer exists (the LLM job had not started). The frozen plan listed as a limit that STAR's prompt has
no task flowcharts while ABCD's has the full guidelines. To make the LLM baseline as strong and as even as possible,
STAR's system prompt now carries all 24 task specifications of STAR-master/tasks (for each task: the wizard's replies
by action name and the graph of the usual next action; 36,066 characters of compact JSON), as ABCD's carries its
guidelines. The task of the current dialogue is not given; the LLM has to infer it, as with ABCD's flows. Nothing else
changes: the same 500 rows, 10 retrieved examples, temperature 0, the same answer mapping and statistics. The previous
script is kept as llm_next_action.v1_38869e84.py; tests: test_llm_next_action.out (9 checks plus the STAR check).
