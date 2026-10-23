# e85: the TM's s for each MultiWOZ agent under random negatives, by e77's rule (dev only)

Written 2026-10-22, before any e85 number. No test split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**The question (fixed now):** with random negatives (chosen on MultiWOZ dev, e83), which s does e77's rule pick for
each MultiWOZ agent? R24 then takes each agent's pick.

**Why:** every agent so far ran at s 8, an s tuned under "hardest" negatives. bge-large's 4k screen under "hardest"
preferred s 24 (e77). Unequal tuning under the new rule would flatter the network-free agents. This was the major
finding of the selection part of the e83/e84 review.

## The rule (e77's, copied; every agent the same)

- **Screen** (e85a, 12 jobs): random negatives, MultiWOZ dev, 4,000 clauses (T 4,000), seed 1, s in {8, 12, 16, 24},
  for the refined, C4 and bge-large agents.
  - The pick is the highest 4k dev score, if it is at least 0.3 points above the agent's own s 8. Otherwise the pick
    is s 8. An exact tie goes to the first of [8, 12, 16, 24].
- **Final** (e85b, only for the picks that moved): the pick at 16k (T 16,000), seed 1. s* is the pick, unless its 16k
  dev is below the agent's own s 8 at 16k, seed 1. Then s* is 8.
  - The 16k s 8 rows are e83b's (refined) and e84's (C4, bge-large), all with random negatives.
- The grid is not extended after the curve is seen.

## What was known

- **Rows under "hardest" negatives:**
  - e77's 4k screens: bge-large s 8 / 12 / 16 / 24 = 0.3325 / 0.3375 / 0.3375 / 0.3400; C4 0.3085 / 0.3089 / 0.3098 /
    0.3029.
  - e82-R3, refined: 0.3123 / 0.3007 / 0.3037 / 0.3068.
- **Rows under random negatives:**
  - only s 8: e83 (4k seed 2, the refined agent), e83b and e84 (16k, seeds 1-2, all three agents);
  - e84's seeds 3-4 are running under the note frozen 18:03:34Z.
- **Noise:** the 4k best-dev seed spread can reach 1.5 points (e83's review). So a 0.3 pick is a weak bar, and the
  16k check is the real one, as e77 had it.

## Checks and commands (pinned)

- **The checker** is scripts/e85_pick.py, reusing e84_check.py's agent table. Its test, tests/test_e85_pick.py, PASSED:
  - the picks at their edges, a tie, the 16k fallback;
  - jobs_e85b.txt and jobs_R24.txt;
  - four refusals.
- **The run** is chained by e85_chain.sh (project notes, not published): a waiter on the R24 dry runs' end, then:
  1. `bash scripts/seq_queue.sh logs/jobs_e85a.txt logs/e85a_queue.log`;
  2. `e85_pick.py --stage screen`, which writes jobs_e85b.txt;
  3. the e85b queue, if it has jobs;
  4. `e85_pick.py --stage final`, which writes logs/jobs_R24.txt with each agent at its s*;
  5. the end line "e85 end: <branch> <UTC> exit <code>" in logs/e85_end.log.
- **GPU estimate:** e85a 4 x about 11.5 min (refined), 4 x about 10 (C4) and 4 x about 3.7 (bge-large), so about
  100 min. e85b takes at most 3 x about 9-28 min. In all, about 1.7-2.8 h.

## Hashes (sha256sum)

| File | SHA-256 |
|---|---|
| scripts/e85_pick.py | 971cbb14b26013148f38a217f587c1064a7091926a6bd9eef55bb8d69a416be1 |
| tests/test_e85_pick.py | ec95c63f5399c44a829bdc603d02354db2d75149cd45e6fbf00f44319e944e25 |
| logs/test_e85_pick.out | d0f9e3751fde078bb5e3352d2ee2b8c44185ee68b3f32e27cb7aba51e110b3b3 |
| logs/jobs_e85a.txt | d87984f67d26f97b63c4501a2fb5a7c90ae57d0a7c1487c2198ff59e080001d5 |
| scripts/e84_check.py | df63a7f2068cd532ec029d532690b2823f1528ccbf89e1396564337084423eb2 |
| scripts/knob_run.py | 113cb74eca36d1f9c3fbea889a2fe8aad2c1ad91878af9f17d93cffa14d65e4e |


The chain is pinned (the dry test above PASSED):

| File | SHA-256 |
|---|---|
| e85_chain.sh (project notes, not published) | 46556c25cda4370f2a6d4e9f7f38758549235b59130dcf3842c6a2262f7ceb7c |
| test_e85_chain.sh (project notes, not published) | 521925e2897677108b924bcf475692441cc52858245f30f0f259ee7684570631 |
