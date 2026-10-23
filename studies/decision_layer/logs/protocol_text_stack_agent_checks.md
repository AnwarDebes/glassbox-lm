# Standing protocol (2026-10-19): the agent check (R4) and the guards (R5) for changes accepted by the text stack (design fixed at 21:52 UTC; frozen before its first run, after R24)

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged. It was never frozen on its own; its rules R4 and R5 were applied inside the frozen plans e78, e80 and e82.

From the text-stack map in the project notes (not published; 21:46 UTC), rules R4 and R5. They are run when the text
stack accepts a change on intent val (a new table, or a new bit block). Dev only; no test split is read here.

## What this study receives

- A frozen export from the text-understanding line: a table folder with MANIFEST.sha256 and its tokenizer reference,
  or a bit block with its code and a reference output. It is copied into data/tables/ or a new src/dl file and checked
  against its MANIFEST.
- Nothing from the text-understanding line's own folder (not published) is imported live.

## R4: the agent check (dev; the C4 agent, view B or view B plus the new block)

- **Runs:**
  - STAR dev: 16,000 clauses, s 16 (R22's setting). Reference: e74c_star_Bc4_c16000_s16, 0.6545.
  - ABCD dev: 4,000 clauses, s 16 (R23's setting). Reference: e74c_abcd_Bc4_c4000_s16, 0.8417.
- **Seeds:** 1 and 2 for the candidate. The references get their seed 2 the first time (one run each), so both sides
  are means of the same 2 seeds.
- **Rule:**
  - A stack that costs the agent more than 0.5 points on either dataset (the mean of 2 seeds, candidate minus
    reference) is not taken into the agent.
  - It may still be kept for classification, labelled.
  - One seed's TM noise on these dev sets is about 0.3 to 0.5 points, hence the 2 seeds.
- **Re-tuning:** if the stack is taken, s is re-tuned by the e74 rule (s 8, 12, 16, 24 at 4k; the pick at full size)
  before any registered use (R3).
- **Path:** a table goes through a wrapper that registers it in dl.count_text.TABLES in-process, then table_run.py.
  So no frozen file changes. A bit block goes in a new ABCDView subclass in a new file, the count_text.patch way.
- **Reported:**
  - both datasets, both seeds;
  - e77_breakdown.py's first / later and per-label lines against the reference, so a gain or loss is placed;
  - "no neural network at any stage" or "network-free at run time, teacher at training time" (R6).

## R5: the guards (ABCD; e70's 50 rules, e72d/e72e's method)

- **The domain:** a new text block's bits are free in the level-B domain, like xcust and xrecent; it is a superset of
  every real input. Memory bits keep their domain rows. A block of facts (P9) is free too.
- **Before the agent uses a new view:**
  - the 50 rules are proved again, raw and guarded, with M recomputed per model, on the new view's ABCD models: the
    2 dev seeds of R4, and at a registered milestone its 5 test seeds;
  - the guards are ABCD's only, since the guidelines are ABCD's;
  - this uses a wrapper in the style of verify_policy_c4_e72e.py (new SRC and OUT, its note frozen first);
  - the guard cost is measured open and closed loop on dev.
- **Pass:** 250 of 250 guarded proofs. A raw counterexample is expected and reported; it is not a failure.

## Not changed by this protocol

e71 seed 3, e76, e77 and R24 continue as frozen (R8 of the map).

## Dated note (2026-10-19T22:07:06Z): step 2d (saturation) goes through this protocol, on xcust only

At 22:06 UTC the criterion of the text-understanding line was applied, fixed before the measurement: the longest 10%
at a mean on-share of 0.75 or more on the C4 view. By it, STAR xcust (0.815) and ABCD xcust (0.767) are saturated; the
xrecent fields (0.743 and 0.724) are not (results/explore/c4_saturation.json 8a368035).
- **The block:** the text-understanding line defines it. It is additive: count bits per concept, and/or concepts
  present in the latest customer utterance, beside the cumulative OR and never replacing it. ABCDView already has
  `xcrecent` (existsT on the customer's last 3 utterances, R20), the closest existing part.
- **Order:** after N9, R24 and step 0b of the text-understanding line, with a dated plan frozen first.
- **Bar, fixed now:** STAR at least +0.5 and ABCD no worse than -0.5, the mean of seeds 1-2. Then R3 re-tuning and R5
  guards before the agent uses it.
