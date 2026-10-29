# Results of the text_understanding study

How a Tsetlin machine reads text: static word tables (potion-base-32M, GloVe, and tables counted from C4 text with no
neural network at any stage) turned into bits, mostly by existsT (a table lookup per word, thresholds per learned
concept, and an OR over the words), then a CoTM. Every verdict below is the output of the test's own verdict script,
saved beside its result rows; registered tests were frozen before their test split was read. Validation numbers are
on validation splits; "test" means the test split named in the registration.

## Registered tests

| Test | Question | Verdict (verbatim) | Files |
|---|---|---|---|
| T2 | Do existential bits (existsT) add to the mean-vector bits, on CLINC150, Banking77 and HWU64? | "on 3 complete datasets: H1 works True, H1 strong True, H2 works True, H2 strong True"; mean+existsT closes 0.81 / 0.56 / 0.59 of the gap from the mean bits to MiniLM | `REGISTRATION_T2.md`, `verdict_t2.py`, `results_exists.jsonl` |
| T3 | The same on a related set (MASSIVE en, test), and word-plus-neighbour token clauses | MASSIVE test: mean 82.59, mean+existsT 85.21, existsT 84.98, MiniLM bits 84.41 ("R1 works True, R1 strong True, R2 works True, R2 strong True"); token clauses G+S+P+N 85.90 ("H3 works True, H3 strong True") | `REGISTRATION_T3.md`, `REGISTRATION_T3b.md`, `verdict_t3.py`, `results_tok.jsonl` |
| T4 | Can a TM label every word of an utterance with its slot type (MASSIVE)? | "H1 (TM > minilm3): MET (+4.05); strong (every seed > minilm3): MET"; "H2 (TM >= 0.90 x LR twin): MET (ratio 0.951)"; test span F1 70.00 (seeds 69.48, 70.54, 69.97), LR twin 73.57, frozen MiniLM probe 65.95 | `REGISTRATION_T4.md`, `verdict_t4.txt`, `results_slot_*.jsonl`, `rules_slot.txt` |
| T5 | Can a TM autoencoder learn word codes from text alone that match counting (PPMI-SVD)? | "works NOT MET, strong NOT MET" for both variants (R1 and R3a) | `REGISTRATION_T5.md`, `t5/verdict_t5.txt`, `t5/results_t5.jsonl` |
| P1 | Under the published low-data protocol of Model2Vec (1,000 training examples), do existsT bits beat the mean vector with logistic regression? | P1a, P1b and P1c "NOT MET" (0 of 3 sets each); the published LR baseline reproduced within 0.39 | `REGISTRATION_P1.md`, `verdict_p1.txt`, `results_p1.jsonl` |
| N7 | existsT on the TM community's own text benchmarks (TREC, MR, SST-2, R52, R8; test) | bars B and C "NOT MET" (0 of 4); "D_potion: adds on 3 of 5 sets -> MET"; "D_glove: adds on 4 of 5 sets -> MET"; "E: adds on 0 of 5 sets -> NOT MET" | `REGISTRATION_N7.md`, `verdict_n7.txt`, `results_n7_test.jsonl` |
| W1-T | Does the C4 counting table (no neural network at any stage) reach potion's table on N7's test splits (2nd use)? | "word vs potion: within 1.0 on 3 of 5, within 2.0 on 3 of 5 -> REACHES POTION" (above potion on TREC, R52, R8; below on MR -2.03 and SST-2 -4.00) | `REGISTRATION_W1T.md`, `w1/verdict_w1t.txt`, `w1/verdict_w1t_extra.txt`, `w1/results_w1t_test.jsonl` |
| M1 | The stack of steps 3a and 3c on the test splits of CLINC150, Banking77 and HWU64 (reused across the project) | "H1 MET (3 of 3, STRONG); H2 MET (3 of 3); H3a MET (3 of 3, strong on 1); H3b MET (3 of 3, STRONG)"; e.g. CLINC150 H1_tm +1.60 [+1.05, +2.13] | `tu/PLAN_M1.md`, `tu/m1/verdict_m1.txt`, `tu/m1/results_m1_*.jsonl`, `verify_m1/` |
| M2 | M1's H1 and H2 on a fresh test set, MTOP English (first use) | "H1 NOT MET; H2 MET (not strong)"; with the TM a3B - C4 = +0.75 [+0.24, +1.27], with LR +0.43 [-0.23, +1.07]; TM accuracy C4 92.94 / a3B 93.69 / potion 93.97 | `tu/PLAN_M2.md`, `tu/m2/verdict_m2.txt`, `tu/m2/results_m2_*.jsonl`, `verify_m2/` |

H1 and H2 of M1 and M2 are separate claims with no multiplicity correction. M1's test splits had been read many times
across the project before M1, so M1 is not fresh-test evidence; M2 is the fresh test.

## Validation steps of the stack (rules fixed before each run)

| Step | Question | Verdict (verbatim) | Files |
|---|---|---|---|
| 0b | Does tuning the TM's s help either table? | "tuning helps: C4 table on 0 of 3 sets, potion on 0 of 3" | `tu/b0_*`, `tu/results_0b_val.jsonl` |
| 3a | The C4 table refined toward a teacher (bge-base) by a linear fit | "VERDICT 3a: PASS (3 of 3 sets at +0.5 with both probes)"; TM with the refined table 94.49 / 91.56 / 90.27 against potion 94.45 / 91.69 / 91.17 (CLINC150 / Banking77 / HWU64) | `tu/PLAN_3a.md`, `tu/a3/verdict_a3.txt` |
| 3b | A TM student of the teacher's bits | closed at the CPU pilot P0, NO-GO: best TM balanced bit accuracy 52.33 against ridge 63.66 (rule: TM >= linear - 2.0) | `tu/PLAN_3b.md`, `tu/b3/p0/p0_verdict.json`, `tu/b3/p0/p0t_verdict.json` |
| 3c | Paraphrases written by a local LLM (Qwen3.5-9B) as extra training text | "VERDICT 3c: PASS (2 of 3 sets at +0.5 with both probes)" | `tu/PLAN_3c.md`, `tu/c3/verdict_c3.txt`, `verify_3c/` |
| c3x | 3c's follow-ups and one-k-means replication | descriptive only | `tu/c3x/` |
| 3d | Pair counts for the table | code and a pair-count pre-run only; its plan was never frozen and it was not run | `tu/d3_*` |
| W1 | Can a TM learn its own word table from raw C4 text? | closed at step 0, negative: "with this engine, a Tsetlin machine does not learn a word table as good as counting from the same text at 1-10M tokens" | `PLAN_tm_word_table.md`, `w1/` |
| W2 | FAQ question-answer pairs as a training signal for the table | "J - ppw_S1000 >= +0.5 with both probes on 0 of 3 sets -> FAIL" | `w2/verdict_p2.txt` |
| W3 | Counter-fitting the table with a lexicon | "CF beats ppw_S1000 by +0.5 and CFrand by +0.3 with both probes on 0 of 3 sets -> FAIL" | `w3/verdict_w3.txt` |
| N8, N9 | Bag-of-words TM tuned to its ceiling, and meaning bits beside it | exploratory sweeps on carved validation splits | `n8_*`, `n9_*`, `results_n8_*.jsonl`, `results_n9_sweep.jsonl` |
| Line C | A TM as an arbiter inside a count language model | exploratory, validation only; stopped after its gate G1 as a dead end | `linec/` |

## Independent recomputes

`verification/recompute_n7.py`, `verification/recompute_w1t.py` and `verification/recompute_3c.py` recompute the
verdicts of N7, W1-T and 3c from the result rows with their own code; `verify_m1/` and `verify_m2/` hold the
recomputes of M1 and M2.
