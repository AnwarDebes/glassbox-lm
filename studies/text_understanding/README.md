# text_understanding: making the Tsetlin machine read text

How far a Tsetlin machine gets on intent and text classification when its input is built from static word tables:
potion-base-32M, GloVe, and word tables counted from C4 text with no neural network at any stage. The main encoder is
existsT: each word is looked up in the table, its similarity to learned concepts is thresholded, and the bits of all
words are joined by an OR. The line then refines the C4 table toward a teacher (step 3a), adds paraphrases written by
a local LLM (step 3c), and tests the stack on held-out test splits (M1) and on a fresh test set, MTOP English (M2).
The C4 tables built here are the tables of decision_layer's agents R21 to R24 and of [agent_vs_llm](../agent_vs_llm/README.md).

**Status:** closed for now. Every test, its verdict and its files are in [docs/results.md](docs/results.md).

## Files

| Path | What |
|---|---|
| `REGISTRATION_*.md` | registered tests T2 to T5, P1, N7 and W1-T |
| `PLAN_text_understanding.md`, `PLAN_tm_word_table.md` | the line's plan and the W1 plan |
| `exists_tm.py`, `static_tm.py`, `tok_tm.py`, `graph_*.py`, `glove_*.py`, `n7_*.py`, `slot_*.py`, `m2v_protocol.py` | tests 1 to 5, N7, the slot filler (T4) and P1 |
| `results*.jsonl`, `verdict_*`, `rules_*.txt` | result rows, verdict scripts and outputs, and the learned rules |
| `w1/` | the C4 corpus builder, the counting table (PPMI-SVD), the TM autoencoder and W1-T |
| `w2/`, `w3/`, `linec/`, `t5/` | the pair signal, the lexicon step, the TM arbiter and the TM autoencoder codes |
| `tu/` | the steps 0b, 3a to 3d, R3, M1 and M2 (`tu/PLAN_*.md`; the plan of 3d was never frozen and is not included), their code, pins, results and tests |
| `fresh/`, `mwoz_leak/`, `novelty/` | the fresh test set (MTOP) and its overlap list, a leakage check, a novelty probe |
| `verify_3c/`, `verify_m1/`, `verify_m2/`, `verification/` | independent recomputes |

## Running

Scripts run from this folder in the decision_layer environment
(`../decision_layer/.venv-dl/bin/python -I <script>`); GPU jobs go through `../tm_language_model/scripts/gpu_run.sh`.
The datasets, the C4 text, the word tables, the feature caches (for example `emb_potion_test2/`, the potion features of
test 2) and the model files are not distributed. The tables can be rebuilt with `w1/` (C4) and `tu/a3_*` (3a); the
datasets are public (CLINC150, Banking77, HWU64, MASSIVE, TREC, MR, SST-2, R52, R8, MTOP).
