# Plan LOGSIZE-dev: how many logged dialogues the readable agent needs to reach the LLM agent

Dev only, exploratory. No test file is opened. Written and frozen before any log-size number exists.

## Why

REG_agent_vs_llm (frozen 2026-10-27T18:54:13Z; its test jobs are running) compares the readable agent with LLM* when
both learn from the whole training split. The two do not get the same things: the agent has only the logs, while the
LLM agent also reads the written policy where one exists (ABCD's guidelines, STAR's task specifications) and brings its
pretraining. A reader will ask: with fewer logged dialogues, which agent is better, and from what log size does the
readable agent reach the LLM agent? This plan measures both at nested log sizes on the same 500 dev rows. It is the
dose-response part of the paper's thesis that the policy lives in the logs.

## Known before this plan (dev, the same 500 rows per dataset, whole training split)

Read from the files on 2026-10-28 (agent: runs/agent_<ds>_c4_tm_s1/tm.npz, MultiWOZ e84's run converted to
runs/agent_multiwoz_c4_tm_s1/tm.npz; LLM*: runs/llm_q3_L35k50/<ds>/preds.jsonl, 0 invalid answers; TF-IDF:
runs/dry/tfidf_<ds>/dev_preds.npz through runs/results_table.md):

| | STAR | ABCD | MultiWOZ |
|---|---|---|---|
| AGENT | 0.666 | 0.858 | 0.374 |
| LLM* (L35k50) | 0.638 | 0.864 | 0.376 |
| TF-IDF LR | 0.672 | 0.866 | 0.384 |

No arm has been run at any smaller log size. The registered test's numbers are not read before its report.

## Log sizes

- Each dataset's training dialogues (STAR 4,811, ABCD 8,034, MultiWOZ 8,437, read from the train files) are sorted by
  convo_id and shuffled once with random.Random(20261028); the first N form the pool of size N, so the pools are
  nested. N is 64, 256, 1024, 4096 and the whole split.
- logsize_subsets.py writes runs/logsize/subsets.json: the order, the train file's sha256, rows per N.

## Arms (the same 500 dev rows; every arm at its whole-split settings; nothing is re-tuned at any N)

- AGENT(N): agent_preds.py, unchanged, run through logsize_agent.py, which keeps only the training rows of the N
  dialogues (so the view's word list, the TM and everything else are fit on them). Settings as in the whole-split
  runs: table c4words-S1000-v1, parts mem,words,xcust,xrecent, seed 1, 40 epochs, best epoch on the whole dev split,
  init half; STAR 16,000 clauses s 16; ABCD 4,000 clauses s 16; MultiWOZ 16,000 clauses s 8 with random negatives
  (e84's settings).
- LLM*(N): llm_next_action.py, unchanged, run through logsize_llm.py, which keeps only the N dialogues' rows as the
  retrieval pool. L35k50's arguments: Qwen3.6-35B-A3B Q4_K_M, thinking off (budget 0), k 50, the written policy where
  it exists, rows drawn with --seed 2 (Q1's rows excluded on STAR and ABCD, as for L35k50), serve_headline.sh (65,536
  context, one slot), temperature 0.
- TFIDF(N), no rule: tfidf_reg.py's setting (ft_baseline.render text; TfidfVectorizer 1-2 grams, min_df 2, sublinear
  tf, lowercase; LogisticRegression saga, max_iter 200, tol 1e-4, random_state 0) at its C (STAR 4, ABCD 16,
  MultiWOZ 1), trained on the N dialogues' rows; CPU (logsize_tfidf.py).
- Whole split: LLM* is runs/llm_q3_L35k50 (not rerun); AGENT is runs/agent_star_c4_tm_s1 and runs/agent_abcd_c4_tm_s1;
  on MultiWOZ, AGENT is rerun on the whole split through logsize_agent.py, because the existing file came from e84's
  code path (its dev accuracy 0.35004756 is written beside the rerun's). TFIDF is computed by logsize_tfidf.py and must
  give the same top-1 predictions as runs/dry/tfidf_<ds>/dev_preds.npz on all dev rows (a check of the code; if it
  does not, the TFIDF column is not read until the cause is found).
- Gate before any AGENT(N) at a smaller N: logsize_agent.py on ABCD's whole split must reproduce
  runs/agent_abcd_c4_tm_s1's dev accuracy 0.8417480998914224 exactly (the wrapper must change nothing when the pool is
  the whole split). If it does not, no AGENT(N) run is read until the cause is found.
- Not run: FT (its 4-epoch protocol gives a few dozen updates at N = 64; a fair small-data schedule would be a new
  choice); a second pool draw; agent seeds other than 1.

## Statistic and readings (fixed now)

- Accuracy on the 500 rows; an invalid LLM answer counts as wrong. Paired differences with a 95% bootstrap interval
  over rows (seed 20261026, 10,000 draws), as in the earlier dev tables: AGENT(N) - LLM*(N); with no rule
  TFIDF(N) - LLM*(N) and AGENT(N) - TFIDF(N).
- Label per dataset and N: "LLM* ahead" if the AGENT - LLM* interval lies wholly below 0; "AGENT ahead" if wholly
  above 0; otherwise "level within the interval".
- N_reach per dataset: the smallest size in (64, 256, 1024, 4096, whole) from which on (it and every larger size)
  AGENT - LLM* >= 0 at the point estimate; "not reached" otherwise. With 500 rows a point estimate moves by a few points
  between draws, so N_reach is shown with the intervals beside it and is not a non-inferiority claim.
- Allowed wording: "on dev, with N logged dialogues the LLM agent is ahead of the readable agent" only where the label
  is "LLM* ahead" (and the reverse only where it is "AGENT ahead"); "on dev the readable agent reaches the LLM agent
  from N logged dialogues on (point estimates)" only for N_reach. Never "non-inferior" (no margin is tested here) and
  never a claim about the test splits.
- Also reported, no rule: the share of the 500 rows whose gold action never occurs in the N-dialogue pool (no model
  trained on that pool can output it); the LLM's invalid answers and seconds per row at each N.

## Order and cost

The GPU queue (jobs_ft.txt) after reg_report_all: the gate (ABCD whole split, about 3 min); AGENT at N = 64 to 4096 for
STAR, ABCD and MultiWOZ, and MultiWOZ's whole split (about 1.5 h); then LLM*(N) for N = 64, 256, 1024, 4096, one queue
line per N (one server session; STAR, ABCD, MultiWOZ; about 4.6 h per line, 18.5 h in all); then logsize_table.py
(CPU). TFIDF runs on the CPU right after the freeze (2 threads, nice 19, CPU gate).

## Limits known now

- 500 rows per dataset: about plus or minus 4 points on one accuracy, 3-4 on a paired difference. One pool draw per N
  (the luck of the draw is not measured), one agent seed, one LLM run.
- Nothing is re-tuned at small N (TM clauses, s and epochs; the LLM's k; TF-IDF's C): settings chosen on the whole
  split may favour or hurt either arm at small N.
- The agent's epoch is picked on the whole dev split, which includes the 500 rows (as in the whole-split runs). LLM*
  was chosen on these 500 rows at the whole split, so its whole-split number is optimistic.
- The LLM reads written policy that the agent never sees; that is the intended comparison (what each gets when
  deployed), not a matched-input one.
