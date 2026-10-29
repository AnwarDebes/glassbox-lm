# agent_vs_llm: the readable agent against local LLM agents and a fine-tuned transformer

Next-action prediction on STAR, ABCD and MultiWOZ 2.2. The readable agent is the one of
[decision_layer](../decision_layer/README.md) with no neural network at any stage (a word table counted from C4 text,
memory bits of the actions taken, a Tsetlin machine). It is compared with prompted local LLMs (Qwen3.5-9B and
Qwen3.6-35B-A3B through llama.cpp, with retrieved training examples), a fine-tuned transformer (bge-base, BERT,
RoBERTa) and TF-IDF logistic regression, on the same rows.

**Status:** the registered test is done and verified (results below); the logsize dev study was still running when
this folder was published.

**Result in one line** (registered, seed 1, 95% bootstrap over dialogues): the agent is not worse than the best local
LLM configuration by more than 3 points on all three datasets (MultiWOZ on seed 1 only), it beats it on none, the
fine-tuned transformer is ahead on all three, and TF-IDF logistic regression is level with the agent. The full table,
the no-rule readings and the caveats are in [docs/results.md](docs/results.md).

## Files

| Path | What |
|---|---|
| `REG_agent_vs_llm.md`, `REG_agent_vs_llm.frozen.json` | the registration and its freeze record (425 pinned files) |
| `PLAN_*.md`, `NOTE_*.md`, `AMEND_*.md` (each with its `.frozen.json`) | the dev plans, notes and amendments, frozen before their runs |
| `agent_preds.py`, `llm_next_action.py`, `ft_baseline.py`, `tfidf_reg.py`, `tfidf_lr.py` | the four arms |
| `rows_rule.py`, `select_llm_star.py`, `make_reg_jobs.py`, `queue_reg_jobs.py`, `freeze_reg.py`, `check_reg.py` | rows, the LLM* choice, the job lines, the freeze and the pin check |
| `reg_report.py`, `reg_report_all.sh`, `reg_table.py`, `verify_reg.py` | the registered report, its table and its check |
| `logsize_*.py`, `logsize_llm_job.sh` | the logsize dev study |
| `*_table.py`, `cascade.py`, `combine_reask.py`, `readability_dev.py`, `zipf_*.py`, `*_speed.py`, `design_effect_dev.py` | dev tables, the cascade, readability, the synthetic grid and the cost measurements |
| `llm_*job*.sh`, `reg_llm_job.sh`, `serve_headline.sh`, `run_queue.sh`, `convert_qwen36_35b.sh` | the LLM jobs, the local server, the queue and the model conversion |
| `test_*.py` | the tests (run each as its docstring says) |
| `results_REG_agent_vs_llm/` | the registered result: registration copy, reports, checks, `results.json`, the independent recompute |
| `runs/` | dev result files (`result.json` and tables); `MODELS.md` names the model files |
| `verification/` | a second independent recompute of the registered numbers, with its test |

## Running

Every script finds the repository from its own place. The agent and the tables run in the decision_layer environment
(`../decision_layer/.venv-dl/bin/python -I <script>`); the registration tools run under Python 3.12. The LLM arms
need llama.cpp b11742 and the GGUF files in `tools/` beside the repository (`MODELS.md`); the fine-tuned arm needs a
GPU. Data, model files, prediction files, the frozen snapshot and run logs are not distributed; the datasets are
public (STAR, ABCD, MultiWOZ 2.2) and their loaders are in decision_layer.
