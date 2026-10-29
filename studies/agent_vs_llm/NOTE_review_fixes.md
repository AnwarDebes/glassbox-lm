# Note: dev changes after the review of registration draft v9 (written 2026-10-27T08:31:44Z)

Review: the pre-freeze review of v9 (an independent read-only review,
judged by an AI model, not a person). Written before any answer of the arms below exists. Dev only.

1. **The reasoning arm's token cap.** L35t (PLAN_llm_q3_dev.md: thinking on, at most 4,096 tokens) left 51 of its 500
   STAR answers empty, 50 of them at the cap; they count as wrong. On the 449 rows it answered, L35t and L35 (thinking
   off) are equal: 0.6147 both. So the cap, not reasoning, made L35t's 0.552. New arm **L35t16**: the same with a
   16,384-token cap (--max-tokens 16384).
   - STAR: the 51 rows without a usable answer are asked again with the new cap (llm_q3_job5.sh REASK=1; rows from
     combine_reask.py rows), and combine_reask.py combine keeps L35t's 449 answered rows. Under greedy decoding with the
     same prompt, a generation that stopped before 4,096 tokens stops at the same place with a higher cap (up to the
     numerics of llama.cpp's prompt cache), so this equals a full run with the higher cap up to that caveat.
   - ABCD and MultiWOZ: L35t16 runs on all 500 rows (it replaces L35t at 4,096 there).
   - L35t (4,096) stays a reported configuration on STAR. Both are LLM* candidates where they have 500 rows.
2. **L9t (the 9B with reasoning) is dropped.** On the same first 42 ABCD rows it answered, it had 30 right (5 empty
   answers, all at the 4,096 cap), the 9B without reasoning 34, L35 35 and L35k50 36. With all 5 capped rows right it
   would tie L35 (35) and stay below L35k50, so it cannot plausibly be LLM*; reasoning stays covered by the 35B
   (L35t, L35t16) on all three datasets. The draft's L9t-after-the-freeze clause is removed.
3. **llm_next_action.py** (previous file archived as archive/llm_next_action.v7_4fca5acb.py; new sha256 1018f2820175fbdc8231045e67997299abd032a249b39bbdc5b01d0e3c75896b):
   a failed request is re-sent twice (30 s apart) before it is recorded as an error; on a restart, rows recorded with
   an error are asked again (the old file is kept as preds.jsonl.errors_<time>); a run that still holds errors exits
   with code 4; --only-rows answers a listed subset of the drawn rows. No prompt and no answer changes when no
   request fails (test_llm_next_action.py, 30 checks passed). L35k100 on STAR (AMEND_q3_k100.md pinned 4fca5acb) runs
   this file; the change cannot alter its answers unless a request fails.
4. **The agent's registered script on STAR and MultiWOZ.** agent_preds.py --dry-run-dev runs on both before the freeze
   (queued after L35k50 MultiWOZ); each must reproduce the reference dev accuracy exactly (STAR 0.6544585987261147,
   MultiWOZ 0.35004756080989263), or the registration is not frozen until the difference is explained.
5. **RoBERTa on ABCD** (ft_abcd_roberta_s1, queued since PLAN_ft_dev.md) runs before the freeze, so the FT choice is
   made with it known; FT stays bge-base on every dataset (one model for all three).

Queue after L35k50 MultiWOZ: dry_agent_star, dry_agent_multiwoz, ft_abcd_roberta_s1, L35k100 STAR, L35t16 STAR
re-ask, L35t16 MultiWOZ, L35t16 ABCD; then the agent bge raw runs (not needed for the registration).
