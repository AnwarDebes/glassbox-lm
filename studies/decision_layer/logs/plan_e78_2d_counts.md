# Plan e78 (2026-10-20): the text stack's step 2d-a in the agent (concept counts beside xcust), dev only

Under the R4/R5 protocol (`logs/protocol_text_stack_agent_checks.md`, design fixed 2026-10-19
21:52 UTC). The spec comes from the plan of the text-understanding line (project notes, not published; f8f552fd: count at level 3, or a
concept's highest level where it has none). The code was reviewed independently (judged by an AI model, not a person) at 18:51 UTC on
2026-10-20 and passed; the text-understanding line was paused until 2026-10-22. This plan is frozen with `scripts/freeze.py` before its first
run. No test split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## The block (code as reviewed)

- `src/dl/count_bits.py` (c9434a1b): ABCDViewN with the part "xcustn".
  - For each of xcust's 512 concepts, it counts the customer's tokens so far that reach the concept's level-3
    threshold, using the same tokens as xcust's OR, "|" and punctuation included.
  - The bits are n >= 2, 4 and 8, which is 1,536 bits. They come after every view B bit and remove nothing.
- `scripts/stack_run.py` (bd250b8a): table_run.py's way, plus count_bits.patch; the run scripts are unedited.
- Self-test (`scripts/count_bits_selftest.py` 56f34ab1, `logs/count_bits_selftest.out` fbf888b2) on STAR and ABCD,
  dev and train[:5000]. Everything passed:
  - view B's bits are byte-identical to the tuned dev rows';
  - n >= 1 equals xcust's counted bit for every concept and row;
  - the chains and the 1,536 domain rows hold.
- On-share at level 3, dev, n >= 2/4/8:
  - STAR: 0.24/0.07/0.01, and 0.65/0.34/0.08 in the longest 10%;
  - ABCD: 0.21/0.04/0.004, and 0.51/0.20/0.03 in the longest 10%.

## Arms (dev; the stack's tuned s; R21's other settings)

| Dataset | Clauses, s | Reference: view B (table_run.py) | Candidate: view B + xcustn (stack_run.py) |
|---|---|---|---|
| STAR | 16,000, s 16 | seed 1 = e74c_star_Bc4_c16000_s16 (0.6545); seed 2 new: e78_star_B_c16000_s16_seed2 | seeds 1, 2: e78_star_BN_c16000_s16 |
| ABCD | 4,000, s 16 | seed 1 = e74c_abcd_Bc4_c4000_s16 (0.8417); seed 2 new: e78_abcd_B_c4000_s16_seed2 | seeds 1, 2: e78_abcd_BN_c4000_s16 |
| MultiWOZ (reported, no rule) | 16,000, the C4 agent's s* from e77 | seed 1 = e77b's row at s* | seed 1: e78_mwoz_BN_c16000_s&lt;s*&gt; |

- **Jobs:** `logs/jobs_e78.txt` (938f6746 at this freeze): 6 jobs, ABCD first, then STAR. That is 2
  references and 4 candidates, one at a time through scripts/seq_queue.sh, with its own log logs/e78_queue.log and
  cpu_load at most 4.5. Models are saved (`--save`).
  - The MultiWOZ candidate (seed 1, 16k) is appended by rule once e77b ends, as one more line in the same format, at
    the C4 agent's s* from `scripts/e77_pick.py --stage final`. It is reported with no rule.
  - GPU order (set at 18:55 UTC): after e77b, whichever of the text-understanding line's step 0b and e78 is frozen and ready first runs
    first, one big job at a time.
- **Checks (`scripts/e78_check.py`, 1305d894, written before any e78 number; dry-tested on made-up rows):**
  - one row per tag, after this freeze;
  - the settings as the table;
  - the candidate's n_bits equals view B's plus 1,536, and its static_info has the xcustn block (spec f8f552fd);
  - each reference seed 2 has the same train and dev bits as its seed-1 row.

## The rule (the bar, fixed 2026-10-19 22:06 UTC)

- **Taken into the agent** only if both hold, on the mean of seeds 1 and 2, candidate minus reference:
  - STAR dev at least +0.5;
  - ABCD dev no worse than -0.5.
- MultiWOZ dev is reported with no rule.
- **If taken:** R3, re-tuning s by the e74 rule (s 8, 12, 16, 24 at 4k, then the pick at full size), and R5, the 50
  guards proved again on the new view's ABCD models with the 1,536 domain rows added. Both come before the agent uses
  the block. Each has its own dated note.
- **If not taken:** reported as a measured failure of 2d-a in the agent, with the saturation numbers beside it. 2d-b
  (view B + xcrecent) is next, under the same rule.

## Budget and order

- GPU, alone (the text-understanding line was paused until 2026-10-22). STAR 16k is about 20-25 min a run, so 3 runs; ABCD 4k is about 2.5 min
  a run, so 3 runs; MultiWOZ 16k is about 25 min. About 1.7 h in all.
- Order: e77b and the R24 gate -> this plan frozen -> jobs_e78.txt -> e78_check.py -> the rule.
