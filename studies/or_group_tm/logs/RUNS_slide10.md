# Slide-10 runs (every run, newest last; Check 2b by the slide10.check2b of hierarchical_tm)

Rows are never edited or removed; a note stands below the rows it concerns. The rows above the publication note further down were made on 2026-11-12 in a working folder outside this repository, before the code was moved here; their run folders are not distributed (they hold the words of the sentences of slide 10 of the project proposal, which is not published). `dev` in the column split means the DEV sentences; the EVAL sentences are scored only by the registered runs of `registrations/E3_slide10_text.md`.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T160417Z_e2_text_A0_s1 | text, 1 | 20, 12, 8, near, all, random, 2, -1, 20 | 0.5194 / 0.8230 | dev | 0.406 / 0.414 | 0.040 / 0.172 / 0.179 | 1 / 1 | False / False |
| 20261112T160423Z_e2_text_A1_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20 | 0.9110 / 0.9899 | dev | 0.960 / 0.943 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T160428Z_e2_text_A0_s2 | text, 2 | 20, 12, 8, near, all, random, 2, -1, 20 | 0.5226 / 0.7838 | dev | 0.614 / 0.543 | 0.099 / 0.376 / 0.354 | 0 / 0 | False / False |
| 20261112T160433Z_e2_text_A1_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20 | 0.9245 / 0.9946 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 2 / 1 | False / False |
| 20261112T160438Z_e2_text_A0_s3 | text, 3 | 20, 12, 8, near, all, random, 2, -1, 20 | 0.6101 / 0.7771 | dev | 0.446 / 0.457 | 0.000 / 0.244 / 0.254 | 0 / 0 | False / False |
| 20261112T160443Z_e2_text_A1_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20 | 0.9610 / 0.9999 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 2 / 3 | False / False |
| 20261112T160448Z_e2_text_A0_s4 | text, 4 | 20, 12, 8, near, all, random, 2, -1, 20 | 0.5367 / 0.8866 | dev | 0.426 / 0.471 | 0.069 / 0.271 / 0.238 | 0 / 0 | False / False |
| 20261112T160454Z_e2_text_A1_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20 | 0.9019 / 0.9841 | dev | 0.970 / 0.957 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T160459Z_e2_text_A0_s5 | text, 5 | 20, 12, 8, near, all, random, 2, -1, 20 | 0.6440 / 0.8582 | dev | 0.683 / 0.714 | 0.000 / 0.362 / 0.386 | 0 / 0 | False / False |
| 20261112T160504Z_e2_text_A1_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20 | 0.9481 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 4 | False / True |
| 20261112T160635Z_e2_text_A5_d0.1_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9732 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160640Z_e2_text_A5_d0.1_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9592 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T160645Z_e2_text_A5_d0.1_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9705 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160650Z_e2_text_A5_d0.1_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9902 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160656Z_e2_text_A5_d0.1_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9624 / 0.9991 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160701Z_e2_text_A5_d0.5_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9712 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T160706Z_e2_text_A5_d0.5_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9738 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T160712Z_e2_text_A5_d0.5_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9761 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160717Z_e2_text_A5_d0.5_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9685 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160723Z_e2_text_A5_d0.5_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9531 / 0.9999 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |

INVALID (written 2026-11-12T16:08:42Z): the 10 rows e2_text_A5_d0.1_s1..5 and e2_text_A5_d0.5_s1..5 above were run while the unit test decay_hits_only_weaker_present_members was failing (the decay ran before the count of present members, so a surviving member was reinforced in the same event). They are not used. The arm is rerun below as A5b with the corrected order (18 of 18 unit tests pass, test output of 16:08:15Z, kept with the working folder).

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T160847Z_e2_text_A5b_d0.1_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9526 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T160852Z_e2_text_A5b_d0.1_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9836 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160858Z_e2_text_A5b_d0.1_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9499 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160903Z_e2_text_A5b_d0.1_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9627 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T160908Z_e2_text_A5b_d0.1_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1 | 0.9661 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160914Z_e2_text_A5b_d0.5_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9808 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160919Z_e2_text_A5b_d0.5_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9655 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T160924Z_e2_text_A5b_d0.5_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9672 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T160930Z_e2_text_A5b_d0.5_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9805 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T160935Z_e2_text_A5b_d0.5_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.5 | 0.9446 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |

Note (2026-11-12T16:11:08Z): design arms A6 (seed_all: every empty group is seeded with every present word, so that the words of one multi-word alternative start together) follow; reason: the rules of A5b keep one word per multi-word alternative (dev rows above), which the bracket criterion of Check 2b counts as not recovered. Design work on the dev sentences, as all of E2.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T161113Z_e2_text_A6_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9433 / 0.9997 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161118Z_e2_text_A6d_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1, seed_all | 0.9796 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T161123Z_e2_text_A6_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9653 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161128Z_e2_text_A6d_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1, seed_all | 0.9658 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T161134Z_e2_text_A6_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9607 / 1.0000 | dev | 0.960 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161139Z_e2_text_A6d_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1, seed_all | 0.9667 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 3 | False / False |
| 20261112T161144Z_e2_text_A6_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9234 / 0.9917 | dev | 0.941 / 0.914 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161149Z_e2_text_A6d_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1, seed_all | 0.9783 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T161154Z_e2_text_A6_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9306 / 0.9971 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161159Z_e2_text_A6d_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, decay 0.1, seed_all | 0.9837 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T161225Z_e2_text_A6_p50_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 50, seed_all | 0.9587 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161231Z_e2_text_A6_K16_s1 | text, 1 | 20, 16, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9327 / 0.9999 | dev | 0.960 / 0.943 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161236Z_e2_text_A6_r4_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 4, -1, 20, seed_all | 0.8628 / 0.9648 | dev | 0.693 / 0.686 | 0.000 / 0.000 / 0.048 | 4 / 4 | False / False |
| 20261112T161240Z_e2_text_A6_K8_s1 | text, 1 | 20, 8, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9418 / 0.9951 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161251Z_e2_text_A6_p50_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 50, seed_all | 0.9883 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161257Z_e2_text_A6_K16_s2 | text, 2 | 20, 16, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9415 / 0.9982 | dev | 0.960 / 0.943 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161302Z_e2_text_A6_r4_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 4, -1, 20, seed_all | 0.9181 / 0.9926 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 5 | True / True |
| 20261112T161306Z_e2_text_A6_K8_s2 | text, 2 | 20, 8, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9756 / 0.9972 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161317Z_e2_text_A6_p50_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 50, seed_all | 0.9651 / 1.0000 | dev | 0.960 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161323Z_e2_text_A6_K16_s3 | text, 3 | 20, 16, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9385 / 0.9997 | dev | 1.000 / 1.000 | 0.050 / 0.023 / 0.010 | 4 / 5 | True / True |
| 20261112T161328Z_e2_text_A6_r4_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 4, -1, 20, seed_all | 0.9253 / 0.9984 | dev | 0.960 / 0.943 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161332Z_e2_text_A6_K8_s3 | text, 3 | 20, 8, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9055 / 0.9834 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.103 | 4 / 4 | False / False |
| 20261112T161343Z_e2_text_A6_p50_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 50, seed_all | 0.9178 / 0.9984 | dev | 0.941 / 0.914 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161349Z_e2_text_A6_K16_s4 | text, 4 | 20, 16, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9337 / 0.9982 | dev | 0.931 / 0.900 | 0.000 / 0.000 / 0.017 | 4 / 3 | True / False |
| 20261112T161354Z_e2_text_A6_r4_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 4, -1, 20, seed_all | 0.9006 / 0.9954 | dev | 0.931 / 0.929 | 0.000 / 0.000 / 0.000 | 4 / 5 | True / True |
| 20261112T161358Z_e2_text_A6_K8_s4 | text, 4 | 20, 8, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.8531 / 0.9844 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.103 | 4 / 4 | False / False |
| 20261112T161409Z_e2_text_A6_p50_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 50, seed_all | 0.9122 / 0.9992 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T161415Z_e2_text_A6_K16_s5 | text, 5 | 20, 16, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9343 / 0.9997 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 2 | False / False |
| 20261112T161420Z_e2_text_A6_r4_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 4, -1, 20, seed_all | 0.9411 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T161424Z_e2_text_A6_K8_s5 | text, 5 | 20, 8, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9197 / 0.9840 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.103 | 4 / 4 | False / False |

Note (2026-11-12T16:15:00Z): pair version (two sentences, same family or not), design arms on the dev sentences. P1: the setting of A6 with 60 clauses, mode near; P2: the same with mode side.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T161517Z_e2_pairs_P1_near_s1 | pairs, 1 | 60, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.8421 / 0.9270 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T161540Z_e2_pairs_P2_side_s1 | pairs, 1 | 60, 12, 8, side, critical, weakest, 2, -1, 20, seed_all | 0.8266 / 0.9394 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T161602Z_e2_pairs_P1_near_s2 | pairs, 2 | 60, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.8263 / 0.9271 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T161626Z_e2_pairs_P2_side_s2 | pairs, 2 | 60, 12, 8, side, critical, weakest, 2, -1, 20, seed_all | 0.8218 / 0.9461 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T161647Z_e2_pairs_P1_near_s3 | pairs, 3 | 60, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.8515 / 0.9193 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T161712Z_e2_pairs_P2_side_s3 | pairs, 3 | 60, 12, 8, side, critical, weakest, 2, -1, 20, seed_all | 0.8293 / 0.9477 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |

Note (2026-11-12T16:17:43Z): pair version, design arms P3: as P1 / P2 with 300 passes.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T162158Z_e2_pairs_P3_near_p300_s1 | pairs, 1 | 60, 12, 8, near, critical, weakest, 2, -1, 300, seed_all | 0.8448 / 0.9284 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |

Note (2026-11-12T16:24:43Z): the P3 queue (300 passes, T 8) was stopped by hand after its first run (near, seed 1: no clause for the slide-10 family, row above); its other five runs did not finish and have no rows. Inspection of that run (clauses per family from state.npz): every clause accepts one of the other or near-miss families at a rate of 1.0, none the slide-10 family; with T 8 and about 3 clauses per family the vote gate never closes, so every clause is pushed to cover every family. Design arms P4 / P5 follow: T 2 and T 3, 100 clauses, mode near, 100 passes.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T162702Z_e2_pairs_P4_T2_c100_p100_s1 | pairs, 1 | 100, 12, 2, near, critical, weakest, 2, -1, 100, seed_all | 0.8909 / 0.9296 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T162932Z_e2_pairs_P4_T2_c100_p100_s2 | pairs, 2 | 100, 12, 2, near, critical, weakest, 2, -1, 100, seed_all | 0.8942 / 0.9327 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T163152Z_e2_pairs_P4_T3_c100_p100_s1 | pairs, 1 | 100, 12, 3, near, critical, weakest, 2, -1, 100, seed_all | 0.8839 / 0.9415 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |
| 20261112T163413Z_e2_pairs_P4_T3_c100_p100_s2 | pairs, 2 | 100, 12, 3, near, critical, weakest, 2, -1, 100, seed_all | 0.8791 / 0.9372 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |

Note (2026-11-12T16:43:28Z): P4 rows above (T 2 and T 3, 100 clauses, 100 passes): pair accuracy 0.879 to 0.894, AUC 0.930 to 0.942, still no clause for the slide-10 family (0.000). Design arms P5 follow: tied groups (one set of 12 groups used for both sentences, added to the engine today after the Wikipedia design runs showed its effect), T 2 and T 8, 100 clauses, 100 passes.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T164623Z_e2_pairs_P5_tied_T2_c100_p100_s1 | pairs, 1 | 100, 12, 2, near, critical, weakest, 2, -1, 100, seed_all, tied | 0.9173 / 0.9322 | dev | 0.337 / 0.357 | 0.000 / 0.095 / 0.120 | 0 / 0 | False / False |
| 20261112T164916Z_e2_pairs_P5_tied_T2_c100_p100_s2 | pairs, 2 | 100, 12, 2, near, critical, weakest, 2, -1, 100, seed_all, tied | 0.9116 / 0.9302 | dev | 0.168 / 0.186 | 0.000 / 0.100 / 0.125 | 0 / 0 | False / False |
| 20261112T165131Z_e2_pairs_P5_tied_T8_c100_p100_s1 | pairs, 1 | 100, 12, 8, near, critical, weakest, 2, -1, 100, seed_all, tied | 0.9059 / 0.9708 | dev | 0.693 / 0.714 | 0.000 / 0.317 / 0.369 | 0 / 0 | False / False |
| 20261112T165346Z_e2_pairs_P5_tied_T8_c100_p100_s2 | pairs, 2 | 100, 12, 8, near, critical, weakest, 2, -1, 100, seed_all, tied | 0.9008 / 0.9589 | dev | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0 / 0 | False / False |

Note (2026-11-12T16:54:50Z): slide-10 pair version with tied groups (P5 rows above): pair accuracy 0.90 to 0.92 and AUC 0.93 to 0.97; a clause for the family starts to form (coverage of unseen family sentences 0.34 and 0.17 at T 2 with firing 0.10 to 0.13 on near-miss and broken sentences; 0.69 and 0.00 at T 8 with firing up to 0.37), far from the bars of Check 2b (0.90 coverage, 0.05 firing, 4 brackets). The curves are flat from pass 20 on. The pair version stays not met in design runs; no registered run is made for it.

Publication note (2026-11-12T17:29:49Z): the rows above are the design rows of E2 (`registrations/E2_slide10_design.md`), copied from the working folder with their notes. Three additions to the notes above, written now:

- To the INVALID note of 16:08:42Z: the test output that shows the failure is stamped 16:06:22Z and the first of the ten runs 16:06:35Z, so the ten runs were started although the test had already failed; the launch did not stop on a failed test. The corrected engine of the rows A5b is in no fixed record (the record of the decay note pins the faulty one). The numbers the ten invalid rows showed (both flags of the scorer on 1 of 5 seeds with decay 0.1 and on 3 of 5 with decay 0.5) are not used anywhere.
- A second run beside a failing test: `20261112T165131Z_e2_pairs_P5_tied_T8_c100_p100_s1` started at about 16:49:27Z; a unit test failed at 16:49:36Z (a test of the score readings whose expectation was out of date) and the engine file was written again at 16:50:30Z. The run stored the hash of the engine at its end, so which of the two versions it used cannot be told from its record. The two versions have the same training code (the later one adds a counting function and one reading of the scores), so the row does not depend on it; the row stays, and no claim rests on it. Since then the runner takes the hash of the engine before it imports it and stops if the file changes during the run.
- What of the plan E2 ran and what did not. Ran: A0 and A1 (the first look), the decay arms A5 (invalid), A5b and A6d by the fixed note `registrations/E2_note_decay.md`, and A6 (seed-all) with its variants (50 passes, K 16, K 8, r 4), which was added by the log note of 16:11:08Z and not by a fixed note. Not run although listed in E2: the arms A2, A3 and A4 (the two refinements one at a time) and the initial state -2. Changed against E2 for the pair version: 60 and 100 clauses, T 2, 3 and 8, tied groups, 100 and 300 passes and one to three seeds, where E2 said "the best text arm with mode near and side". The note of 16:24:43Z announces "P4 / P5" for T 2 and T 3; those rows are named P4 only, and P5 names the later arm with tied groups.

Go-bar of E2 (both flags of the scorer on at least 3 of 5 DEV seeds): met by A6 only, at exactly 3 of 5 (A1: 2; A5b: 1 and 2; A6d: 2; K 16: 1; r 4: 2; K 8: 2; 50 passes repeats A6's best rules and is not a second sample). Bracket 5 was recovered by no text run above, and no run recovered all six brackets.

Dry runs in the working folder (no rows above, made without the run log): four runs of arm A6 on the DEV sentences, seeds 1 and 2, at 16:34:40Z, 16:34:46Z, 16:35:06Z and 16:35:12Z. The first pair was made before the throwaway record of that dry run was fixed (16:35:01Z), the second pair after it; the folders of the second pair were renamed for the first version of the checker. That dry run exercised the passing path only and skipped two checks. It is superseded by the dry job of this folder (`registrations/E3_job.sh dry`, rows below), which runs every check and is covered by `tests/test_e3_pipeline.py`.

Rows below this note are written by `src/run_slide10.py` in this folder.

| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |
|---|---|---|---|---|---|---|---|---|
| 20261112T173808Z_e3dry_text_s1 | text, 1 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9433 / 0.9997 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T173815Z_e3dry_text_s2 | text, 2 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9653 / 1.0000 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T173822Z_e3dry_text_s3 | text, 3 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9607 / 1.0000 | dev | 0.960 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T173829Z_e3dry_text_s4 | text, 4 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9234 / 0.9917 | dev | 0.941 / 0.914 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T173836Z_e3dry_text_s5 | text, 5 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9306 / 0.9971 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T173843Z_e3dry_text_s6 | text, 6 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9355 / 0.9999 | dev | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 5 | True / True |
| 20261112T173850Z_e3dry_text_s7 | text, 7 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9027 / 0.9942 | dev | 1.000 / 1.000 | 0.000 / 0.009 / 0.027 | 4 / 3 | True / False |
| 20261112T174901Z_e3_text_s11 | text, 11 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9260 / 0.9987 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T174908Z_e3_text_s12 | text, 12 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9155 / 0.9861 | eval | 0.970 / 0.988 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T174916Z_e3_text_s13 | text, 13 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9652 / 1.0000 | eval | 0.960 / 0.963 | 0.000 / 0.005 / 0.002 | 5 / 4 | True / True |
| 20261112T174923Z_e3_text_s14 | text, 14 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9458 / 0.9996 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 4 | True / True |
| 20261112T174930Z_e3_text_s15 | text, 15 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9089 / 0.9944 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T174937Z_e3_text_s16 | text, 16 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9556 / 0.9999 | eval | 0.970 / 0.963 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T174945Z_e3_text_s17 | text, 17 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9577 / 1.0000 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T174952Z_e3_text_s18 | text, 18 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9516 / 0.9997 | eval | 0.921 / 0.900 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T174959Z_e3_text_s19 | text, 19 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9510 / 1.0000 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 5 | True / True |
| 20261112T175006Z_e3_text_s20 | text, 20 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9319 / 1.0000 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T175013Z_e3_text_s21 | text, 21 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9420 / 0.9991 | eval | 0.861 / 0.850 | 0.000 / 0.014 / 0.001 | 3 / 2 | False / False |
| 20261112T175021Z_e3_text_s22 | text, 22 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9610 / 1.0000 | eval | 0.970 / 0.963 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175028Z_e3_text_s23 | text, 23 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9353 / 1.0000 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 4 | True / True |
| 20261112T175035Z_e3_text_s24 | text, 24 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.8915 / 0.9919 | eval | 0.980 / 0.975 | 0.000 / 0.014 / 0.007 | 5 / 4 | True / True |
| 20261112T175043Z_e3_text_s25 | text, 25 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9277 / 0.9995 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 3 / 2 | False / False |
| 20261112T175050Z_e3_text_s26 | text, 26 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9330 / 1.0000 | eval | 0.960 / 0.950 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175057Z_e3_text_s27 | text, 27 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9196 / 0.9983 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175104Z_e3_text_s28 | text, 28 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9589 / 0.9997 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175111Z_e3_text_s29 | text, 29 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9602 / 1.0000 | eval | 0.960 / 0.975 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175119Z_e3_text_s30 | text, 30 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9277 / 0.9877 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175126Z_e3_text_s31 | text, 31 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9219 / 0.9933 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175133Z_e3_text_s32 | text, 32 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9309 / 0.9991 | eval | 0.960 / 0.950 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175140Z_e3_text_s33 | text, 33 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9080 / 0.9994 | eval | 0.921 / 0.938 | 0.000 / 0.000 / 0.000 | 4 / 5 | True / True |
| 20261112T175147Z_e3_text_s34 | text, 34 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9297 / 0.9818 | eval | 0.970 / 0.963 | 0.000 / 0.000 / 0.000 | 4 / 3 | True / False |
| 20261112T175154Z_e3_text_s35 | text, 35 | 20, 12, 8, near, critical, weakest, 2, -1, 20, seed_all | 0.9379 / 0.9998 | eval | 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 5 / 5 | True / True |
