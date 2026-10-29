# Amendment to PLAN_llm_q3_dev.md: 100 retrieved examples (written 2026-10-27T05:55:14Z)

Before any answer of this arm. L35k50 (50 retrieved examples) reached 0.638 on STAR's 500 fresh dev rows against 0.608
with 10, the best LLM configuration on STAR so far (the agent: 0.666). To learn whether more examples close the rest of
the gap before any test split is spent, L35k100 (the same with 100 retrieved examples) runs on STAR. Rule fixed now: if
L35k100 is at least 1.0 point above L35k50 on STAR, it also runs on ABCD and MultiWOZ; otherwise the example curve is
read as flattening and k = 100 is the largest k tested. L35k100 is an LLM configuration under the plan's rule and a
candidate for LLM* (select_llm_star.py gains it before the freeze).
Note on the running job: the L35k100 line was first added to llm_q3_job3.sh while the L35k50 job was running that
file; because bash reads a script as it runs and the line sat before the dataset loop, the original file was
restored byte for byte within a minute (its sha256 equals the copy llm_q3_job3.v2_*.sh), and L35k100 runs from the
new file llm_q3_job4.sh (llm_q3_job3.sh plus the L35k100 case line).
