# Plan MWOZ-dev: the readable agent against LLM agents on MultiWOZ 2.2 dev (no written policy)

Dev only, exploratory. No test file is opened. Written and frozen before any LLM answer on MultiWOZ.

## Why

On the fresh dev rows the readable network-free agent beats Qwen3.5-9B and Qwen3.6-35B-A3B (thinking off) on STAR by
+13.8 and +5.8 points, and on ABCD by +4.0 and +1.8 (PLAN_llm_q3_dev.md). ABCD's LLM prompt carries the company's full
written guidelines; STAR's carries task specifications whose graphs give only the usual path. MultiWOZ 2.2 has no written
policy at all: its next system acts live only in the logged dialogues. If the gap to the LLM agents comes from policy
that is not written down, it should be large on MultiWOZ too.

## Arms (500 dev rows drawn with random.Random(2); the same rows for every arm)

- AGENT: the C4 agent with random negatives, 16,000 clauses, s 8, seed 1 (e84's run; its saved dev predictions
  reproduce its recorded dev accuracy 0.35004756 exactly; converted to runs/agent_multiwoz_c4_tm_s1/tm.npz, top 1 only).
- L9 and L35: Qwen3.5-9B Q8_0 and Qwen3.6-35B-A3B Q4_K_M, thinking off, the Q1 prompt with MultiWOZ's 128 labels (no
  guidelines exist), 10 retrieved solved examples, temperature 0 (llm_next_action.py fc7eae6d; STAR/ABCD prompts
  unchanged, checked byte for byte).
- Reported beside them: FT bge-base fine-tuned on MultiWOZ train (PLAN_ft_dev.md settings, seed 1), TF-IDF LR.

## Reading (fixed now)

Agent minus each LLM arm with a 95% bootstrap interval over rows (seed 20261026). The claim "the readable agent beats
the LLM agents" is open for MultiWOZ only if the agent minus every LLM arm is at least +3.0 points with the interval
above 0. The accuracy here is exact match of the whole act set, a strict measure for every arm alike.
