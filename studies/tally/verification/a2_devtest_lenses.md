# A2, the registered look of 2026-10-25: the three-lens check

Written 2026-10-25T13:44:19Z (clock read in the writing command), judged by an AI model, not a person, by direct checks of the files named below (no further reviewer was started for this check).

Registration `registrations/A2.md` (sha256 e59e4994...), frozen 2026-10-25T13:38:45Z (`registrations/A2.frozen.json`, 72041c26...). The look `results/20261025T133912Z_a2_devtest_final` (13:39:12Z to 13:40:40Z, state finished, exit 0; output `verification/a2_devtest_look.out`, ec861ece...). The recompute `verification/a2_devtest_recompute.json` (74b42ae3...): all_match true.

## Leakage: no finding
- Dev-test was read by this look only (the 2nd use of the split; T0's report was the 1st). The guard refuses a second look; `results/` holds one `*_a2_devtest*` folder.
- Nothing fitted in the look saw dev-test: the codes were loaded from `data/a2_codes/` (built from train before the freeze, read-only since 13:38:52Z), the cache was chosen on valid and the mixing weights fitted on calib; the look stopped otherwise unless the cache grid, the cache choice, every arm's weights and EM steps equalled the development run's, and it did not stop.
- Stated caveats, not leaks: the Mikolov vocabulary is built from the whole training file, dev-test included; 2.1% of dev-test sentences occur verbatim in train (the subset without duplicate spans is reported).

## Selection and registration: no finding
- The claim code (`exchange`) was named in the frozen text and forced by the script; the look started 27 s after the freeze; the lines are the registered ones.
- Every arm is reported, the ones that lost included; `kind_topic` equals `exchange` (topic weights 0.0), as stated before the look.
- The reading is the conjunction of the three registered conditions; the two gain intervals, the subsets and KN5 alone carry no verdict.

## Integrity: no finding
- The recompute (own bootstrap, PCG64 seed 20261025, 10,000 resamples; and a full rescore of every arm from the saved codes) matches on all 14 checks: same arms, perplexities within 1e-12, same gain, margin, success and kill, code exchange and the registered label, logp.npz as recorded, KN5 alone 7.176025782864251 = T0's 7.1760 within 5e-5, the freeze record still matching, every log probability rescored with difference 0.0.
- The run records the pinned script (51ad257f...), 2 threads for each of the five thread variables, NLTK_DATA unset, and the versions of tm.lock; the fifteen code prefixes and the cache line equal the registration's expected log lines.

## Verdict
MATCH. Success line met: gain 0.0759 (registered interval of bits(exchange) - bits(random mean) -0.1294 to -0.0971; the recompute's -0.1283 to -0.0981). No blocker, major or minor.
