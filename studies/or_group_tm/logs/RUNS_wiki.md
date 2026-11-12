# Wikipedia pair runs of this study (every run, newest last)

Rows are never edited or removed; a note stands below the rows it concerns. The rows above the publication note further down were made on 2026-11-12 in a working folder outside this repository, before the code was moved here (design plan `registrations/E4_wikipedia_design.md`); their run folders are not distributed. Column 5 gives the AUC and tuned accuracy of the unit vote sum, then (from 16:51Z on) the AUC of the ratio reading, then the AUC of the two weighted readings.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T162238Z_e4_sa_first_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 2, -1, 10 | dev | 0.7497 / 0.7079 | 0.4983 | 17336, 8.87 |
| 20261112T162602Z_e4_sa_init3_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 2, -3, 10 | dev | 0.7721 / 0.7219 | 0.4985 | 19247, 11.85 |
| 20261112T162803Z_e4_sa_share_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -1, 10 | dev | 0.7635 / 0.7146 | 0.4990 | 23649, 8.82 |
| 20261112T163009Z_e4_sa_qseed25_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 2, -1, 10 | dev | 0.7498 / 0.7059 | 0.4965 | 18765, 9.13 |
| 20261112T163259Z_e4_sa_init3_qseed25_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 2, -3, 10 | dev | 0.7758 / 0.7261 | 0.5009 | 17735, 12.02 |
| 20261112T163624Z_e4_sa_T100_s1 | sa, 1024, 1 | 2000, 1, 100, side, critical, weakest, 2, -1, 10 | dev | 0.7562 / 0.7128 | 0.4947 | 23392, 16.04 |
| 20261112T163646Z_e4_sa_B_w64_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, wmax=64 | dev | 0.7774 / 0.7276 ; 0.7790 / 0.7798 | 0.5000 | 19752, 16.56 |
| 20261112T163905Z_e4_sa_decay01_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 2, -1, 10, p_decay=0.1 | dev | 0.7564 / 0.7146 | 0.4997 | 17682, 12.66 |
| 20261112T164052Z_e4_sa_B_tied_w64_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, wmax=64 | dev | 0.8255 / 0.7728 ; 0.8249 / 0.8237 | 0.5001 | 11106, 17.5 |
| 20261112T164126Z_e4_sa_r4_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 4, -1, 10 | dev | 0.7346 / 0.6935 | 0.4986 | 13535, 11.12 |

Note (2026-11-12T16:42:18Z): rows written before this note do not show options whose value is True (tied, seed_all), because the row formatter compared True with 1.0; the run names and each config.json carry them. The row e4_sa_B_tied_w64_s1 above was run with tied groups. Fixed for later rows.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc (unit votes; then AUC of the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T164513Z_e4_sa_B_neg50_w64_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, neg_frac=0.5, wmax=64 | dev | 0.5047 / 0.5087 ; 0.5051 / 0.5082 | 0.5043 | 703639, 37.8 |

Note (2026-11-12T16:47:07Z): queue arms1 stopped by hand after its seventh run (r 4), to bring the CPU use back under the 4.5-core rule (it read 5.26 with three ungated queues). Its last four arms (K 2 untied, mode near, reinforce all + reject random, 30 passes; all untied) did not run or did not finish and have no rows; the untied K 2 run was killed mid-run.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc (unit votes; then AUC of the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T164742Z_e4_np_TB_s1 | np, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8567 / 0.7994 ; 0.8560 / 0.8547 | 0.4976 | 10550, 18.44 |
| 20261112T164854Z_e4_sa_init5_share_w64_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -5, 10, wmax=64 | dev | 0.7861 / 0.7360 ; 0.7862 / 0.7850 | 0.5025 | 16426, 16.69 |

Note (2026-11-12T16:51:28Z): design reading added after the tied-group rows: RATIO = the number of positive clauses that fire divided by the number whose groups are all true on at least one of the two paragraphs (the overlap of the two paragraphs counted in clauses instead of words, as Jaccard does with words). Reason, from the study's own baselines on dev: the shared-word count has AUC 0.7911 and its normalised form (Jaccard) 0.8191 on sa, so the unit vote sum (a count) was missing the normalisation. Recomputed from saved models, no training (src/rescore_wiki.py, after all tests passed; its output is kept with the working folder): sa tied seed 1 unit 0.8255 -> ratio 0.8456; np tied seed 1 0.8567 -> 0.8787; sa untied (B_w64) 0.7774 -> 0.8081. A first rescore of the same three runs was made while one unit test had an outdated expectation (the set of readings); it was repeated after the fix with identical values. Negative-polarity arm (B_neg50, row above): AUC 0.5047 with 703,639 members: half the clauses voting against does not work with this rule; dropped.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc (unit votes; then the ratio reading; then AUC of the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T165216Z_e4_nsall_TB_s1 | ns_all, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.7023 / 0.6530 ; 0.7030 / 0.7026 | 0.4986 | 16676, 19.54 |
| 20261112T165739Z_e4_sa_TB_critical_s1 | sa, 1024, 1 | 2000, 1, 40, side, critical, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8243 / 0.7719 ; ratio 0.8445 ; 0.8237 / 0.8223 | 0.4971 | 10898, 17.94 |
| 20261112T170257Z_e4_sa_TB_init1_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -1, 10, tied=True, wmax=64 | dev | 0.8220 / 0.7710 ; ratio 0.8422 ; 0.8214 / 0.8201 | 0.4985 | 11414, 17.62 |
| 20261112T170816Z_e4_sa_TB_init5_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -5, 10, tied=True, wmax=64 | dev | 0.8257 / 0.7746 ; ratio 0.8458 ; 0.8253 / 0.8239 | 0.4962 | 10844, 17.79 |
| 20261112T171101Z_e4_sa_B_tied_w64_c8000_s1 | sa, 1024, 1 | 8000, 1, 100, side, share, weakest, 2, -3, 10, wmax=64 | dev | 0.8304 / 0.7782 ; 0.8298 / 0.8281 | 0.4950 | 35900, 124.36 |
| 20261112T171351Z_e4_sa_TB_T100_s1 | sa, 1024, 1 | 2000, 1, 100, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8227 / 0.7695 ; ratio 0.8515 ; 0.8219 / 0.8195 | 0.4988 | 16420, 19.35 |
| 20261112T172330Z_e4_sa_TB_K2_s1 | sa, 1024, 1 | 2000, 2, 40, side, share, weakest, 2, -3, 10, seed_all=True, tied=True, wmax=64 | dev | 0.8006 / 0.7497 ; ratio 0.8368 ; 0.7995 / 0.7985 | 0.5008 | 86294, 33.0 |
| 20261112T172848Z_e4_sa_TB_r4_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 4, -3, 10, tied=True, wmax=64 | dev | 0.8224 / 0.7682 ; ratio 0.8404 ; 0.8216 / 0.8207 | 0.4950 | 9618, 17.29 |
| 20261112T173913Z_e4_sa_TB_c4000_s1 | sa, 1024, 1 | 4000, 1, 80, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8295 / 0.7763 ; ratio 0.8517 ; 0.8288 / 0.8273 | 0.4962 | 22136, 35.46 |

Publication note (2026-11-12T17:43:05Z): the rows above are the design rows of E4, copied from the working folder with their notes. Added now:

- Rows 16:22Z to 16:41Z are untied arms on `sa`; the tied arms (one set of groups used for both paragraphs, `tied=True`) start with `e4_sa_B_tied_w64_s1`, whose row does not show the option (see the note of 16:42:18Z). The tied base TB (2,000 clauses, T 40, mode side, reinforce share, reject weakest, r 2, initial state -3, 10 passes, weights up to 64 as extra readings) was then run on `np` and `ns_all` (seed 1) and varied one setting at a time on `sa` (reinforce critical, initial state -1 and -5, T 100, K 2 with seed-all, r 4, 4,000 clauses with T 80; 8,000 clauses with T 100 ran before as `e4_sa_B_tied_w64_c8000_s1`).
- The ratio reading (note of 16:51:28Z) was defined after the tied rows were seen; it is a reading chosen on the DEV pairs.
- Arm chosen for the next step: TB. It is the only arm that ran on all three data versions; on `sa`, seed 1, T 100 (ratio 0.8515) and 4,000 clauses (0.8517) are slightly higher than TB (0.8456) on one seed each, at 1.1 and 2.0 times the time per pass, and were not run on `np` or `ns_all`.
- Next: TB on `sa`, `np` and `ns_all` with seeds 1 to 3 on the DEV pairs, run in this folder (rows below). Seed 1 repeats the design rows above and serves as a regression check of the moved code (same engine file).

Rows below this note are written by `src/run_wiki.py` in this folder.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc (unit votes; then ratio; then the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T174840Z_e5dev_sa_TB_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8255 / 0.7728 ; ratio 0.8456 ; 0.8249 / 0.8237 | 0.5001 | 11106, 17.37 |
| 20261112T174911Z_e5dev_ns_all_TB_s2 | ns_all, 1024, 2 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.7053 / 0.6528 ; ratio 0.7349 ; 0.7062 / 0.7054 | 0.4949 | 16586, 18.63 |
| 20261112T175359Z_e5dev_np_TB_s1 | np, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8567 / 0.7994 ; ratio 0.8787 ; 0.8560 / 0.8547 | 0.4976 | 10550, 17.54 |
| 20261112T175452Z_e5dev_sa_TB_s3 | sa, 1024, 3 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8283 / 0.7808 ; ratio 0.8481 ; 0.8278 / 0.8266 | 0.4995 | 10964, 18.69 |
| 20261112T180008Z_e5dev_ns_all_TB_s1 | ns_all, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.7023 / 0.6530 ; ratio 0.7323 ; 0.7030 / 0.7026 | 0.4986 | 16676, 22.42 |
| 20261112T180042Z_e5dev_np_TB_s3 | np, 1024, 3 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8574 / 0.7984 ; ratio 0.8793 ; 0.8566 / 0.8550 | 0.4989 | 10448, 20.06 |
| 20261112T180530Z_e5dev_sa_TB_s2 | sa, 1024, 2 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8278 / 0.7776 ; ratio 0.8474 ; 0.8273 / 0.8261 | 0.5064 | 11034, 17.91 |
| 20261112T180559Z_e5dev_ns_all_TB_s3 | ns_all, 1024, 3 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.7016 / 0.6487 ; ratio 0.7336 ; 0.7021 / 0.7012 | 0.4968 | 16830, 18.97 |
| 20261112T181044Z_e5dev_np_TB_s2 | np, 1024, 2 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8561 / 0.7987 ; ratio 0.8779 ; 0.8551 / 0.8538 | 0.4994 | 10626, 17.2 |

Note (2026-11-12T18:12:43Z): the bars of E4 from the DEV rows of seeds 1 to 3 of arm TB (the nine `e5dev_*` rows above; mean of the three seeds).

- Unit reading (the vote sum, the score that E4 named): sa 0.8255 / 0.8278 / 0.8283, mean 0.8272; np 0.8567 / 0.8561 / 0.8574, mean 0.8567; ns_all 0.7023 / 0.7053 / 0.7016, mean 0.7031.
- Ratio reading (defined on the DEV pairs after the tied rows were seen, note of 16:51:28Z): sa 0.8456 / 0.8474 / 0.8481, mean 0.8470; np 0.8787 / 0.8779 / 0.8793, mean 0.8787; ns_all 0.7323 / 0.7349 / 0.7336, mean 0.7336.
- G1 (sa at least 0.759, np at least 0.764): met in both readings.
- G2 (sa at least 0.830, np at least 0.875): not met in the unit reading (sa 0.8272, np 0.8567); met in the ratio reading only (sa 0.8470, np 0.8787, that is 0.0037 above its bar).
- Partner-shuffle AUC of all nine runs within 0.47 to 0.53 (0.4949 to 0.5064); no run is void.
- Seed 1 of each data version equals its design row (`20261112T164052Z_e4_sa_B_tied_w64_s1`, `20261112T164742Z_e4_np_TB_s1`, `20261112T165216Z_e4_nsall_TB_s1`): the state, polarity and weight arrays, the stored scores, the event counts and every final number are equal, and so is the ratio AUC of the rescore. The design rows were made with the engine files 53cc7b4b (sa, np) and 32062302 (ns_all), these rows with b94db364.
- Decision, by "What a miss means" of E4: G1 is met, so E5 is a registered test of "above the stock model" on sa, np and ns_all (the stock test scores exist), with word overlap, PPMI cosine and the logistic regression beside it. G2 is met only in the ratio reading, a reading chosen on the DEV pairs after the tied rows were seen. The text of E5 therefore proposes the ratio reading as its primary reading and a rule "above word overlap" on sa and np in that reading only, says so where the rule is stated, and reports the unit reading beside it with no rule. In the unit reading, the reading that E4 named, G2 is missed and the comparison with word overlap carries no rule.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc (unit votes; then ratio; then the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T190622Z_e5dry_sa_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8255 / 0.7728 ; ratio 0.8456 ; 0.8249 / 0.8237 | 0.5001 | 11106, 17.69 |
| 20261112T190656Z_e5dry_np_s1 | np, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8567 / 0.7994 ; ratio 0.8787 ; 0.8560 / 0.8547 | 0.4976 | 10550, 17.95 |
| 20261112T190738Z_e5dry_ns_all_s1 | ns_all, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.7023 / 0.6530 ; ratio 0.7323 ; 0.7030 / 0.7026 | 0.4986 | 16676, 19.3 |

Note (2026-11-12T20:18:34Z): the dry run of E5 frozen at 19:02:39Z is superseded: its run folders `20261112T190622Z_e5dry_sa_s1`, `20261112T190656Z_e5dry_np_s1` and `20261112T190738Z_e5dry_ns_all_s1` (rows above), its record `registrations/dry/E5_dry.frozen.json`, its report `logs/e5_dry_report.json` and its outputs `logs/e5_dry/` now end in `_SUPERSEDED`. Reason: after two read-only reviews of draft 1 of E5 the code was changed before any freeze of E5: the rules are read in the vote sum and need a second interval that also resamples the seeds; there is no rule against word overlap; a third interval resamples the articles of paragraphs A and B; a reading leaves out the pairs that touch a duplicated article; the report checks the environment of every run and of itself and the run log before the freeze, and it keeps every number when a check fails; the job applies the CPU rule itself and takes no setting from the environment. The dry job is run again below with the changed code.

| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | AUC / tuned acc (unit votes; then ratio; then the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |
|---|---|---|---|---|---|---|
| 20261112T202255Z_e5dry_sa_s1 | sa, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8255 / 0.7728 ; ratio 0.8456 ; 0.8249 / 0.8237 | 0.5001 | 11106, 17.63 |
| 20261112T202407Z_e5dry_np_s1 | np, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.8567 / 0.7994 ; ratio 0.8787 ; 0.8560 / 0.8547 | 0.4976 | 10550, 22.0 |
| 20261112T202532Z_e5dry_ns_all_s1 | ns_all, 1024, 1 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | dev | 0.7023 / 0.6530 ; ratio 0.7323 ; 0.7030 / 0.7026 | 0.4986 | 16676, 27.47 |
| 20261112T210903Z_e5_sa_s11 | sa, 1024, 11 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8266 / 0.7690 ; ratio 0.8474 ; 0.8259 / 0.8247 | 0.5079 | 11080, 17.95 |
| 20261112T210935Z_e5_np_s11 | np, 1024, 11 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8525 / 0.7971 ; ratio 0.8737 ; 0.8517 / 0.8510 | 0.4991 | 10542, 18.04 |
| 20261112T211019Z_e5_ns_all_s11 | ns_all, 1024, 11 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.7100 / 0.6607 ; ratio 0.7403 ; 0.7107 / 0.7103 | 0.5107 | 16580, 19.62 |
| 20261112T211237Z_e5_sa_s12 | sa, 1024, 12 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8246 / 0.7696 ; ratio 0.8455 ; 0.8238 / 0.8221 | 0.5009 | 11170, 18.16 |
| 20261112T211307Z_e5_np_s12 | np, 1024, 12 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8505 / 0.7961 ; ratio 0.8715 ; 0.8500 / 0.8490 | 0.4997 | 10628, 17.89 |
| 20261112T211406Z_e5_ns_all_s12 | ns_all, 1024, 12 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.7120 / 0.6611 ; ratio 0.7423 ; 0.7124 / 0.7110 | 0.4971 | 16636, 19.57 |
| 20261112T211613Z_e5_sa_s13 | sa, 1024, 13 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8241 / 0.7688 ; ratio 0.8454 ; 0.8234 / 0.8219 | 0.5023 | 11158, 18.29 |
| 20261112T211631Z_e5_np_s13 | np, 1024, 13 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8522 / 0.7967 ; ratio 0.8730 ; 0.8516 / 0.8503 | 0.4991 | 10596, 17.25 |
| 20261112T211743Z_e5_ns_all_s13 | ns_all, 1024, 13 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.7140 / 0.6687 ; ratio 0.7450 ; 0.7142 / 0.7121 | 0.4976 | 16600, 18.68 |
| 20261112T211954Z_e5_sa_s14 | sa, 1024, 14 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8263 / 0.7686 ; ratio 0.8470 ; 0.8261 / 0.8248 | 0.5018 | 10994, 18.99 |
| 20261112T212016Z_e5_np_s14 | np, 1024, 14 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8529 / 0.7964 ; ratio 0.8746 ; 0.8517 / 0.8502 | 0.5015 | 10516, 19.18 |
| 20261112T212129Z_e5_ns_all_s14 | ns_all, 1024, 14 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.7099 / 0.6618 ; ratio 0.7403 ; 0.7102 / 0.7093 | 0.5125 | 16810, 19.67 |
| 20261112T212338Z_e5_sa_s15 | sa, 1024, 15 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8261 / 0.7697 ; ratio 0.8469 ; 0.8254 / 0.8238 | 0.4960 | 11000, 19.1 |
| 20261112T212353Z_e5_np_s15 | np, 1024, 15 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.8513 / 0.7939 ; ratio 0.8726 ; 0.8512 / 0.8503 | 0.4900 | 10638, 18.52 |
| 20261112T212532Z_e5_ns_all_s15 | ns_all, 1024, 15 | 2000, 1, 40, side, share, weakest, 2, -3, 10, tied=True, wmax=64 | test | 0.7110 / 0.6609 ; ratio 0.7402 ; 0.7104 / 0.7073 | 0.4984 | 16512, 20.99 |
