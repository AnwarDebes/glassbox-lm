# Plan LLM-dev: the local 9B LLM as a next-action agent, and what it adds to the network-free agent (dev only)

Dev only, exploratory. No test file is opened. Written and frozen before the first LLM answer.

## Why

PLAN_ft_dev.md showed on STAR dev that fine-tuning a transformer goes well past the ceiling that counted words, frozen
transformer features and TF-IDF all reach (fine-tuned BERT 0.7326 after 3 epochs against about 0.655). The other
neural route a team would take today is an LLM agent: a general model told the actions, the written guidelines and some
solved examples. Two questions follow, and the second can give a headline:

- Q1: how accurate is the local LLM at choosing the next action, against the network-free agent and the fine-tuned
  transformer on the same examples?
- Q2 (CPU only, on saved predictions): does a cascade help, the network-free agent where its clause votes are
  confident and the LLM elsewhere? A cascade that reaches the fine-tuned transformer with no fine-tuning would be a
  strong result; one that does not is reported as it is.

## Q1 design

- LLM: Qwen3.5-9B Q8_0 on llama-server b11742 (`studies/agent_guards/llm/serve_qwen_agent.sh`, thinking off,
  temperature 0, alone on the GPU).
- `llm_next_action.py` (tested with a stub server, `test_llm_next_action.out`): the action list; ABCD's full written
  guidelines (`guidelines.json`); the 10 training examples most similar to the current dialogue (TF-IDF cosine on the
  text the other arms see), each with its gold action; the current dialogue (ABCD: the full transcript so far with the
  actions taken; STAR: the previous actions, the recent exchange and the customer's messages). The answer is mapped to an
  action name; an unusable answer counts as wrong.
- Examples: 500 dev rows per dataset, drawn with seed 1 (`random.Random(1).sample`), the same rows for every arm.
- Arms on the same 500 rows: the LLM; the label of the most similar training example (1-NN, no rule); the majority
  label of the 10 shown examples (no rule); TF-IDF LR (NOTE_tfidf_dev.md, best C); the fine-tuned transformers of
  PLAN_ft_dev.md where their dev predictions exist; the C4 network-free agent where its dev predictions exist.

## Statistic and reading (fixed now)

- Accuracy on the 500 rows, with a 95% interval from a bootstrap over rows (seed 20261026, 10,000 draws) for each
  difference against the LLM.
- Reading only; no claim is made from Q1 alone. If the LLM is at least 5 points below the network-free agent on both
  datasets, Q2 is the next step only if the LLM is right on at least 40% of the rows where the network-free agent is
  wrong (otherwise a cascade cannot gain much and Q2 is skipped).

## Q2 (only after Q1, by a note frozen before it runs)

The cascade's threshold must be chosen on one half of the dev rows and scored on the other half; its exact rule is
written in that note.

## Limits known now

- One open 9B model, one prompt; a larger or tuned LLM could do better. The prompt is fixed before any answer is read.
- STAR's prompt has no task flowcharts; ABCD's has the full guidelines.
- 500 rows per dataset: an interval of roughly plus or minus 4 points on a single accuracy.
