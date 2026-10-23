# P9 proposal (2026-10-19, for after R24; dev plan, NOT frozen; design fixed at 21:52 UTC, to be frozen before its runs): what the system returned, as fact bits for the agent

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged. This draft was never frozen.

From the text-stack map in the project notes (not published; P9, 21:46 UTC). The agent's view has
the customer's words, the last 3 turns and its own earlier actions. It does not have what the system returned to the
human agent. This adds one block of facts (L2, agent side). The block is additive: it removes no bit and changes no
existing part. No text is read by it, so "no neural network at any stage" still holds for the C4 agent.

## Why (STAR dev, the tuned C4 agent e74c_star_Bc4_c16000_s16, 0.6545; scripts/p9_size.py, no rule)

| Class of dev error (first match wins) | Errors | Share of errors | Accuracy points |
|---|---|---|---|
| query or inform: query:&lt;task&gt; against an inform label of the same task | 783 | 32.8% | 11.3 |
| result pair: two result labels of one task (failed / successful, available / unavailable, correct / incorrect ...) | 107 | 4.5% | 1.5 |
| result-like gold, any other prediction | 193 | 8.1% | 2.8 |
| other | 1,304 | 54.6% | 18.9 |

- **The largest pairs:**
  - query:weather -> weather_inform_forecast, 56;
  - query:trivia -> trivia_ask_question, 27;
  - query:doctor_schedule -> doctor_inform_booking_successful, 22;
  - trivia incorrect -> correct, 20;
  - restaurant booking failed -> successful, 19.
- **What these counts are:** an upper bound on what facts could touch, not a forecast. Some "query or inform" errors
  are about the words. For example, weather_ask_day -> query:weather turns on whether the customer gave the day.
- The STAR examples are built from user and wizard texts only. The knowledge base's returns are never in them
  (scripts/star_prepare.py).

## The facts (built from events before the decision, in the same dialogue)

- **STAR** (KnowledgeBase return_item events; 26 distinct result messages, a closed set):
  - a query was made since the customer's latest utterance;
  - the last query's API (one bit per API);
  - the last return's TotalItems: -1 (a transaction), 0, 1 or more than 1;
  - the last return's message (one bit per message, 26), and "no message";
  - the number of queries so far, at least 1, 2 or 3;
  - kept apart, and reported as its own ablation: for trivia, whether the customer's latest utterance contains the
    returned answer (an exact, case-folded string match). It reads the customer's words, so it is not counted as an
    action-log fact.
- **ABCD** (earlier action turns; the values the agent itself entered): the membership level, the shipping status and
  the recorded reason, as one bit per value seen in train. Everything else stays as it is.
- **MultiWOZ** (later, separate): the classic "DB pointer". That is the number of venues matching the belief state per
  domain, bucketed 0, 1, 2-3, more than 3. It needs:
  - the repository's db/ files: a download that needs a recorded decision first, MIT, the same pinned commit;
  - the annotated belief state, so it is an oracle-state fact and is labelled so.
  - MultiWOZ's booking outcomes are not in the data. NoBook stays unpredictable from facts.
- **Leakage check:** every fact is computed from events strictly before the decision's event index. A unit test
  rebuilds the facts with all later events deleted and must get the same bits.

## Measurement (dev only; after R24; no test split)

- **Arms, per dataset:**
  - the tuned C4 agent with and without the facts (STAR: 16k, s 16; ABCD: 4k, s 16; seeds 1 and 2);
  - the tuned bge-large agent with the same facts, so both agents get the same information (reported).
- **Taken into the agent if:**
  - STAR improves by at least +0.5 (the mean of 2 seeds; the primary dataset);
  - ABCD loses no more than 0.5 (the map's R4).
- **Then:**
  - s is re-tuned by the e74 rule (R3);
  - the 50 guards are proved again with the fact bits free in the level-B domain (R5);
  - a later registration would be STAR test's 7th use, declared.
- **Reported:** e77_breakdown.py and p9_size.py again, so the gain is placed (query or inform, result pairs, other).
- **Wording** (fixed at 21:52 UTC): these are "action-log facts" for the agent, not text understanding. They
  help the agent; they do not count toward the text stack. The view is new and sits beside the earlier results, never
  in place of them (R8).
- **Scope** (fixed at 21:52 UTC): strictly what the agent itself knows at that moment, its own API calls and
  their returns. The leakage unit test is the guard for that.

## Cost

- **CPU:** the facts are counting over the dialogue JSON, minutes.
- **GPU:** STAR 16k about 20 min per run and ABCD 4k about 2.5 min per run, so 8 to 10 runs come to about 2 h.
