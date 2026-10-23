# Plan e80 (2026-10-20): the text stack's step 2d-b in the agent (view B + xcrecent), dev only

Under the R4 protocol (`logs/protocol_text_stack_agent_checks.md`). The spec is from the plan of the text-understanding
line (project notes, not published; f8f552fd, 2d-b), the second candidate after 2d-a. The e78 plan said 2d-b
comes next under the same rule. 2d-a was not taken (`logs/e78_check.out` f7b886cd: STAR -0.78, ABCD -0.29). This plan
is frozen with `scripts/freeze.py` before its first run. GPU slot: after step 3a of the text-understanding line (fixed at
20:51 UTC). No test split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## The block (existing, tested code; no new file on the run path)

- `xcrecent` of ABCDView (added for R20): existsT bits of the customer's last 3 utterances only
  (dl.abcd.customer_recent), on the C4 table through table_run.py.
- It goes on top of view B, so the parts are mem, words, xcust, xrecent and xcrecent. The block comes after view B's
  bits, so view B's bits stay a prefix.
- What it adds: a short, unsaturated, customer-only signal beside the cumulative xcust.
- Self-test (`scripts/e80_selftest.py` bd93a4fa -> `logs/e80_selftest.out` 967f60a4; CPU; dev bits only):
  - B + xcrecent has 10,361 bits on STAR, 9,657 on ABCD and 9,315 on MultiWOZ;
  - the first 7,227 / 6,684 / 6,580 bits equal the e74c / e77b dev rows' bits (3fc00b17 / 56297e10 / 311e8933);
  - the xcrecent block's on-share is 0.53 to 0.57.

## Arms (dev; view B's tuned s; R21's other settings)

| Dataset | Clauses, s | Reference: view B | Candidate: view B + xcrecent |
|---|---|---|---|
| STAR | 16,000, s 16 | e74c_star_Bc4_c16000_s16 (seed 1, 0.6545) and e78_star_B_c16000_s16 (seed 2, 0.6579) | e80_star_BX_c16000_s16, seeds 1 and 2 |
| ABCD | 4,000, s 16 | e74c_abcd_Bc4_c4000_s16 (seed 1, 0.8417) and e78_abcd_B_c4000_s16 (seed 2, 0.8423) | e80_abcd_BX_c4000_s16, seeds 1 and 2 |
| MultiWOZ (reported, no rule) | 16,000, s 8 | e77_mwoz_Bc4_c16000_s8 (seed 1, 0.3038) | e80_mwoz_BX_c16000_s8, seed 1 |

- **Jobs:** `logs/jobs_e80.txt` (df5a6354), 5 jobs, one at a time through scripts/seq_queue.sh, with its own log
  logs/e80_queue.log and cpu_load at most 4.5. Models are saved.
- **Checker:** `scripts/e80_check.py` (43f28b80), adapted from e78_check.py, written before any e80 number. It checks:
  - one row per tag and seed, after this freeze;
  - the settings;
  - the references' seeds share their bits;
  - the candidate's seeds share theirs;
  - the candidate's bits are view B's plus xcrecent's, on the C4 table file.

## The rule (the bar fixed 2026-10-19 22:06 UTC; the same as e78)

- **Taken into the agent** only if both hold, on the mean of seeds 1 and 2, candidate minus reference:
  - STAR dev at least +0.5;
  - ABCD dev no worse than -0.5.
- MultiWOZ dev is reported with no rule.
- **If taken:** R3, re-tuning s by the e74 rule, then R5. xcrecent's thermometer chains are verify_policy's own, but
  its names ("customer's last 3 utterances") are not in its concept regex, so its bits are free, a superset. The 50
  guards are then proved again on the new view's ABCD models.
- **If not taken:** reported as a measured failure of 2d-b in the agent.
- **Limit, as for 2d-a:** it is tested at view B's tuned s 16, and a view-specific s is not tried here.

## Budget

- GPU: ABCD 4k about 3 min a run (2 runs), STAR 16k about 26 min (2 runs), MultiWOZ 16k about 25 min (1 run). About
  1.4 h.
