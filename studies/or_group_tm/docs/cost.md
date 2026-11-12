# Cost of the OR-group rule: training, inference and model size

Measured on 2026-11-12 on the study's workstation: one CPU thread, `nice -n 19`, median of 5 repeats, on a shared machine (other jobs ran at the same time, so every timing is rough). The measurement scripts and their raw outputs are kept with the working folder and are not distributed; the numbers below were read from them. Stock-package numbers come from the run records of the hierarchical_tm study (`../../hierarchical_tm/docs/results.md` and its run logs) and, for the current package version, from descriptive runs of that version made on the same day (GPU board power sampled once per second).

## What is measured, bounded and not measured

- **Measured:** time per item for inference of saved models on their DEV items (one CPU thread); training seconds per pass (the runs' own clocks); model size (counted from the saved states); GPU board power of the current stock package during its own runs.
- **Bounded, not measured:** every CPU energy figure. The container has no readable CPU energy counter, so a CPU figure is written as the rated power of one CPU package (205 W) times the time, which charges the whole package to one thread. One core's share would be about 24 times lower; that is an estimate only.
- **Not measured:** CPU joules of anything, the power of the stock package version that the hierarchical_tm study used, any low-power hardware.

## Wikipedia paragraph pairs (DEV, 20,000 pairs; arm TB of `registrations/E4_wikipedia_design.md`)

| | OR-group rule, arm TB | stock package, headline setting |
|---|---|---|
| DEV AUC, same article (sa), seeds 1 / 2 / 3 | 0.826 / 0.828 / 0.828 (vote sum); 0.846 / 0.847 / 0.848 (ratio reading, chosen on DEV) | 0.745 / 0.739 / 0.733 |
| DEV AUC, next paragraph (np) | 0.857 / 0.856 / 0.857; ratio 0.879 / 0.878 / 0.879 | 0.752 / 0.728 / 0.751 |
| DEV AUC, shared words removed (ns_all) | 0.702 / 0.705 / 0.702; ratio 0.732 / 0.735 / 0.734 | 0.622 / 0.622 / 0.611 |
| training | 17.9 s per pass over 40,000 pairs on one CPU thread, 10 passes (179 s, sa, median of 3 seeds) | 13.9 s per pass on a whole V100 GPU, 100 passes |
| inference | 285 microseconds per pair, batched (3,510 pairs per second); 311 microseconds one pair at a time; 499 microseconds with the ratio reading | not measured alone (a loose upper bound from the study's scoring time: 164 microseconds per pair on the GPU) |
| model size | 5,517 member words, 22.1 kB as (group, word) pairs (median of the nine models of the three data versions) | 3.28 MB of include bits (31,035 bits set; 124.1 kB as pairs) |
| energy | not measured; CPU bound at most 58 mJ per pair and 3.7 kJ per training pass | not measured |

References on the same DEV pairs (the hierarchical_tm study): word overlap 0.819 (sa) and 0.864 (np); best logistic regression 0.871 and 0.907. The registered test on the test pairs is E5 (`../README.md`); it is not a comparison at matched size or cost. Arm TB is not faster per pair than the stock package's GPU bound; what this page shows is its size and its CPU-only training in minutes.

## Slide-10 sentences, direct labels (same training and DEV files for all rows)

| | OR-group rule (E3 setting) | stock package, version of the study | stock package, current version 0.3.4 |
|---|---|---|---|
| DEV result | plain accuracy 0.933 (median of 25 seeds), AUC 0.9996; Check 2b on the held-back sentences: see `../README.md` | accuracy 1.0 with a word blacklist; Check 2b failed (its slide-10 rule is always true) | accuracy 1.0; Check 2b on the DEV sentences not passed on the two seeds with a literal budget of 1 (coverage 1.000, firing 1.000 on every kind of negative, 0 brackets; descriptive runs, checked only by their own scripts) |
| training | 0.122 s per pass on one CPU thread, 20 passes (2.6 s in all) | 1.48 s per pass on a shared GPU, 1,000 passes | 1.18 to 1.30 s per pass on the GPU, 300 passes |
| inference | 8.8 microseconds per sentence | not measured | not measured |
| model size | 1,384 member words (median), 5.5 kB as pairs | 42.4 kB of include bits | not measured |
| energy | not measured; CPU bound at most 25 J per pass, about 0.5 kJ per run | not measured | measured GPU board power x time: 105 to 120 J per pass, 35 to 38 kJ per run (the idle board draws about 50 W, included) |

The current package's figure is a measurement of board power; the OR-group figure is a bound that charges a whole CPU package to one thread. Even so, one OR-group run on this task (at most about 0.5 kJ) stays below 2% of the measured energy of one run of the current package (35 to 38 kJ), and its best rule met the bars of Check 2b with direct labels on 5 of 5 registered seeds of E3 (13 of 25 with the further seeds).

## Limits

- One machine, shared, one thread; repeats spread by up to about 10%.
- The ratio reading was defined after the tied-group design rows were seen, so its DEV AUC is optimistic; the vote sum is the plain reading.
- The project's other references (the dialogue agent, the fine-tuned transformer, the prompted local LLM) are on another task and are listed in `../../../docs/efficiency.md`; they set the cost scale only.
- Not tried: an index from each word to the groups that hold it, which would let inference touch only the members present in a text.
