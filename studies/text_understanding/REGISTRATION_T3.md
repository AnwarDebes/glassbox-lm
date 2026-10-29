# Registration T3: MASSIVE en-US test split (flat replication of T2, then token clauses)

Written 2026-10-17T15:55:50Z. At this time no row of phase A (results_tok.jsonl) exists and no model of this line has been
evaluated on MASSIVE. MASSIVE en-US (Amazon, CC BY 4.0): 60 intents, train / dev / test 11,514 / 2,033 / 2,974.
"val" below is its dev split. The decision-layer study used MASSIVE for another question; MASSIVE descends from
SLURP, whose text overlaps HWU64 (so HWU64 is not fully independent of it; noted, not corrected).

## Part 1: does T2 replicate on a fresh dataset? (flat views, runs now)
flat3_tm.py on massive_en, views mean, mean+existsT, existsT, minilm; seeds 1, 2, 3: 12 runs. Settings, epoch
choice and the single test read as REGISTRATION_T2.md. Bars (test accuracy, mean of 3 seeds):
- R1, mean+existsT "works": at least mean + 0.5 points. "Strong": closes at least half the gap mean -> minilm.
- R2, existsT alone "works": at least mean - 1.0. "Strong": at least mean.

## Part 2: token clauses (after phase A; the rule is fixed now)
Selection, on the phase A val rows (seed 1) of CLINC150 / Banking77 / HWU64: among the views G+S+P+N, S+P+N, S
and G+S, the one with the largest average val gain over flat existsT (the mean of the confirm-run val rows of
seeds 1 to 3: CLINC150 0.9460, Banking77 0.9146, HWU64 0.9123). Call it P*.
Gate: Part 2 runs only if P* gains at least +0.3 points over flat existsT on 2 of the 3 datasets. Otherwise:
"gate not passed" in the study log, and the token line is reported as not better than flat bits.
Runs if the gate passes: tok_tm.py on massive_en, views P*, W (word-identity tokens, the Graph TM's usual node
symbol) and G (control: the flat existsT input through the token code); seeds 1, 2, 3: 9 runs; same settings.
Bars (test accuracy, mean of 3 seeds; the flat rows are those of Part 1):
- H3, P* "works": at least flat existsT + 0.5. "Strong": at least flat mean+existsT + 0.5 (beats the best flat view).
- H4, meaning symbols beat word symbols in the token reading: P* at least W + 2.0.
- Control, no bar: G should be within 1.0 of flat existsT; if not, the token code is suspect and that is reported.

## Frozen code (sha256, first 16 hex digits)
flat3_tm.py 7ae40209efccbbf3, confirm_tm.py 3c8917b53ebe2927, exists_tm.py 7da0aee9f4eef796, tok_tm.py
7e62d5093dba6905 (AMENDED 16:30 UTC, before any part 2 run: fc70de2e21086ac4, the memory-lean version, bit-identical to the capped version by test_tok_equiv.py; AMENDED again 17:42 UTC: df15c29d561182a2, adds part V only; see the study log), graph_feats.py cd0725094c7e014d, embed_minilm.py 19b31cbaf302c361; decision-layer package (read
only): dl/tm.py f649043f5801605b, dl/tm_init.py 557ff69706529738, dl/boolean.py b5f5917f390da8a6, dl/data.py
0c4f8d22576e817c. Checked again before each launch; any change is logged.

## Known limits (written now)
- One fresh dataset; 3 seeds; the test set is fixed (1 point = about 30 queries).
- Phase A picks among four token views on val of datasets whose test splits are used; that choice is exploratory.
- Same TM size and 40 epochs for every view, although wider inputs may need more.

## AMENDMENT (2026-10-17 17:58:38Z UTC, before any W test read)
Part 2's W runs (M2_W_s1..s3) are dropped. W alone lets a token clause name one word only (it cannot join words,
noted in the study log at 17:04), and the phase A curves show the word views still improving at the 40-epoch cap (best
epoch 38 to 40, +1.7 to +6.7 points from epoch 30 to 40; pointed out in a separate check at
17:55 UTC and confirmed on results_tok.jsonl). H4 of part 2 would therefore be uninformative; it is not tested
here. The meaning-against-word question goes to REGISTRATION_T3b.md, amended to train to convergence. The rest of
part 2 is unchanged: P* = G+S+P+N, seeds 1-3 (M2_GSPN_*, 40 epochs), with the G control M3_G_s1..s3 (40 epochs).
tok_tm.py is now v4 (sha256 f17ef85c8f2c2c6d): it only adds an optional early stop (--patience, default 0 = off); with the
default the training loop is the same as v3.
