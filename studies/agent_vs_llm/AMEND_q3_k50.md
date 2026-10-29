# Amendment to PLAN_llm_q3_dev.md: an LLM arm with 50 retrieved examples (written 2026-10-27T00:57:34Z)

Before any answer of this arm. The LLM arms saw 10 retrieved solved examples; the agent learned from the whole training
split. LLMs often gain from many more examples in the prompt, so arm L35k50 (Qwen3.6-35B-A3B Q4_K_M, thinking off, 50
retrieved solved examples, otherwise the Q1 prompt) runs on the same fresh rows of STAR, ABCD and MultiWOZ. It is an
LLM arm under the plan's rule (the agent must be at least +3.0 above EVERY arm, interval above 0, for a dataset's claim).
Script: llm_q3_job3.sh with the arm L35k50 (k from the arm; the other arms keep k = 10).
