# Plan e76 (2026-10-19): e71 v2's exhaustive word-edit search on the agent with no neural network at any stage

Decided at 20:55 UTC. The goal is word-edit certificates for the tuned C4 agent of the final
comparison (R23's ABCD arm, tm_mem_c4_s16, seed 1), not only for the potion agent (e71). Frozen with
`scripts/freeze.py` after the pre-checks and before the run. Dev and proof work only: no test split is read. No rule;
it is a measurement.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## What is e71 v2's and unchanged

From `logs/plan_e71_v2_amendment.md`, frozen 2026-10-19T03:30:28Z, and its dated notes:
- the 200 sampled dev states (verify_local.sample_states, rng 0) and the rule groups (verify_local.groups_for with the
  50 rules);
- the 1,626 candidate words: words in at least 5 train customer utterances, by e71's word rule. That rule does not
  depend on the table, so it is the same list;
- the edit sets of size 1 and 2:
  - delete one or two of u's words, with u keeping a word;
  - append one or two candidate words in order;
  - one of each;
- the scoring: deletions through cpu_tm_fast's clause firing; insertions through the Reduced model of their base row;
- the outcomes ("certified at k = 2", "certified at k = 1; a 2-word edit breaks the rule", "a 1-word edit breaks the
  rule", "own choice breaks the rule");
- up to 5 replays per group through the real pipeline;
- the report per group and per state, with the off-window states apart.

Only k = 1 and 2 by exhaustive search. No MILP for k = 3 and 5.

## What is new (`scripts/verify_enum_c4.py`, a port; verify_enum.py and verify_local.py are imported unedited)

- **TextModelC4:** e71's text model on the C4 view.
  - dl.count_text.patch is applied, so potion's table cannot load, with the empty "none" cache.
  - The per-word codes and the text codes use the C4 table's tokenizer.
  - The model is results/confirm/R23abcd/tm_mem_c4_s16, seed 1.
- **EnumeratorC4.base, the one change of method (the contraction finding):**
  - e71 builds the word-table bits after deletions as the fixed text's codes OR the kept words' codes. That is exact
    for potion's WordPiece.
  - It is not exact for the C4 tokenizer, whose contraction rule spans e71's word boundaries. "don't" has e71 word
    units "don", "t" and the residue "'", but its C4 tokens are "do" and "n't".
  - So after deletions, xcust and xrecent are rebuilt from the real edited strings (verify_local.edited_row) with the
    C4 tokenizer.
  - Insertions are appended after a space, and a candidate has no apostrophe, so OR-ing their codes stays exact.
  - An empty text can only occur before an insertion (a one-word u deleted, then a word inserted), so it gets no codes
    there.
- **Rows** go to results/explore/verify_e76.jsonl, tag e76.

## Pre-checks (before the freeze; one thread; tags e76_pre, e76_contr, e76_pre2)

1. **`--check pre`:** the enumerator's rows against view.transform on edited texts rebuilt from strings. It covers 30
   random dev states (rng 3, as v2) plus the 3 off-window sampled states:
   - the zero-edit row;
   - every single deletion and insertion;
   - 50 random double edits per state;
   - argmax against cpu_tm_fast on every single insertion.
2. **`--check contr`:** every dev state whose latest customer utterance has a contraction (n't, 's, 're, 've, 'll,
   'd, 'm):
   - every single deletion;
   - every deletion pair;
   - 20 random double edits;
   - all against view.transform.
   This is an added check: "n't" and "'s" next to a deletion.
3. **`--check pre2`:** the Reduced scoring against full scoring, on 30 random dev states (rng 5, as v2): every single
   insertion, and every pair after 5 random first words.

**The run starts only if every count is exact.**

## The run (after e71 seed 3 ends, about 23:35 UTC; one process, 2 threads)

- **Command:** `verify_enum_c4.py --check run --seed 1 --tag e76`.
- **Time:** about 1.5 to 2 min per state, so about 6 to 7 h.
- **Budget:** it ends before 07:00 UTC. A stop for lack of budget reports the finished states with their count.

## Reported

- the same tables as e71 (per group, and per state at k = 1 and 2), beside e71's seeds 1 to 3 of the potion agent;
- every replayed break.

**Wording:** "for X% of 200 sampled real dev states, no deletion of words from the customer's latest message, or
insertion of up to k of 1,626 common customer words at its end, makes the agent with no neural network at any stage
choose a button the rules forbid in that state (by exhaustive search)".
- Never write "proved for all possible texts".
- Never write "any rewording".

## Pre-check results (all exact; 2026-10-19 21:06-21:13 UTC; one thread; results/explore/verify_e76.jsonl)

- **pre** (logs/e76_pre.out 399c2fe7): 33 states (30 random + the 3 off-window: 2485, 2668, 3289).
  - zero-edit rows: 33 of 33;
  - single deletions and insertions: 53,854 of 53,854 exact;
  - argmax: 53,658 of 53,658 equal;
  - random doubles: 1,650 of 1,650 exact.
- **contr** (logs/e76_contr.out 9950b9cc): all 400 dev states whose latest utterance has a contraction.
  - zero-edit rows: 400 of 400;
  - single deletions: 4,361 of 4,361;
  - deletion pairs: 34,654 of 34,654;
  - random doubles: 8,000 of 8,000.
- **pre2** (logs/e76_pre2.out 3ad00a01): 30 states. Reduced against full scoring is equal on 48,780 of 48,780 singles
  and 243,900 of 243,900 pairs.
- **The finding, measured** (tag e76_contr_naive): on the same 400 contraction states, e71's own decomposition with the
  C4 tokens (Enumerator.base, not the port's) is wrong on 400 of 400 zero-edit rows and on 3,993 of 4,361 single
  deletions. The port's base is wrong on 0. Without the fix, e71's method would not be exact for the C4 agent.
