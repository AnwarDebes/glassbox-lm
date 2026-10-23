# R9b registration: memory literals against a control that has no history at all (ABCD)

**Why this follow-up exists (written before any R9b run, after R9's first test seeds).** R9.1 compares TM-mem with TM-nomem, and TM-nomem was meant to have no access to the agent's earlier actions. It has: two of its parts, `fstep` and `fwords`, read the whole conversation so far, and that text lists every earlier button press as a line "action: <button>" (`src/dl/abcd.py`, field `full`). So R9's TM-nomem gets the history as words. This was seen at about 23:26 UTC, when R9's first four seeds per TM arm were written (TM-mem test 0.844 to 0.850, TM-nomem 0.834 to 0.839) and the gap was about 1 point instead of the 20 points of the exploration. It was a design error: the exploration contrast (e25, +19.8 points on dev) used a view without the whole-conversation parts, and when they were added for R9 the history was not removed from their text. R9.1 stays as registered and will be judged as registered.

R9b tests what R9.1 was meant to test, with the exploration contrast that existed before R9 was frozen. It is the second use of the ABCD test split (R9 is the first), for two new arms.

Frozen by `scripts/freeze.py` before any R9b run. Amendments go in a dated file.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Arms (test; `scripts/abcd_confirm.py`, registration folder `R9b`)

- **TM-mem-short:** view parts `mem,intent,step,words` (1,581 bits: memory; subflow centres of the customer's words; next-button centres of the last 3 utterances; words of the last 3 utterances). None of the non-memory parts reads an action line: the customer's words and the last 3 utterances contain only utterances.
- **TM-nomem-short:** parts `intent,step,words` (1,515 bits): no history in any form.
- Both: 4,000 clauses, T 4,000, s 8, 40 epochs, batch 128, best-dev snapshot, start state half; seeds 1 to 10. These are the settings of e25_full and e25_nomem (dev, seed 1: 0.8398 and 0.6414).
- **LR-mem-short, LR-nomem-short:** logistic regression on the same bits, C from the dev grid (reported, no rule).

Statistic as R9: per test action turn the mean correctness over seeds; bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0, 95% interval.

## Pass rule (test)

- **R9b.1 (memory gives rules over sequences):** TM-mem-short minus TM-nomem-short is at least +5.0 points, with the interval above 0. R9b is met if R9b.1 holds.

## Reported with no rule

- The same difference for logistic regression (LR-mem-short minus LR-nomem-short).
- R9's TM-nomem (the whole-conversation parts, which carry the history as text, plus the short view's parts) minus TM-nomem-short: what the whole-conversation text adds when no memory bit exists.
- Named memory bits on top of the history as text: R9's TM-mem minus R9's TM-nomem (this is R9.1's number).
