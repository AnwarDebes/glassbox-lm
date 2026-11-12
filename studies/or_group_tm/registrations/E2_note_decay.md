# Note to E2 (dated, frozen before the arm it adds): a decay step for weaker present members

Publication note: the text below is the plan as it was fixed on 2026-11-12T16:06:22Z (SHA-256 of that file: `50fdd5746ca1402e7710528ff243d178c5ab22d6135eb6b45d1cc7f0d5d055b6`), with folder and file names changed to those of this repository. Rules, bars, settings and numbers are unchanged.

Seen before this note (dev sentences, `logs/RUNS_slide10.md`): arm A0 (the rule of E1) fails as expected (coverage 0.41 to 0.68, 17% to 39% firing on near-miss and broken sentences). Arm A1 (reinforce critical, reject weakest) covers 0.96 to 1.00 of the unseen family sentences and fires on 0.000 of the negated twins, near-miss and broken sentences on all 5 seeds, but recovers 5 / 2 / 2 / 5 / 3 brackets (288 view), so Check 2b passes on 2 of 5 seeds in both views (seeds 1 and 4). Reading the rules of seeds 2 and 3: the clause is right as a whole, but several groups keep stray words of other brackets that never get hit, because the clause no longer fires on any negative.

New rule element, arm A5: DECAY. When a clause fires as it should, a present member whose state is below that of the strongest present member of its group loses 1 with probability p_decay. Members of equal state (the words of one multi-word alternative, which enter together) do not decay. Arms: A1 + p_decay 0.1 and A1 + p_decay 0.5, seeds 1 to 5, otherwise as A1.

This is a change made after dev results were seen; every number of A5 is design work, and only the registered run E3 on the EVAL sentences with new model seeds can confirm it. The go-bar of E2 is unchanged.
