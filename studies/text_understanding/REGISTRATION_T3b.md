# Registration T3b: meaning symbols against word symbols, MASSIVE en-US test split

Written 2026-10-17T17:04:28Z, before any of its runs and before the phase A rows of V and V+W exist. Exploratory evidence that
suggested it (the study log, test 3): HWU64 val, token reading, S 0.9080 against W 0.6413; flat TMs of tests 1-2, existsT
against the study's words-only TM, +11 to +14 points on val. The claim tested: for Tsetlin machines that read
words, meaning bits from the static table (graph_feats.py) beat word-identity symbols, the symbols CAIR's Graph TM
uses for words (random hypervectors; its LogicalTransformer README asks how to make them "less random and more
semantic"). Here the word symbols are one-hot (no hashing collisions), which can only favour them.

## Runs
tok_tm.py (sha256 df15c29d561182a2) on massive_en with --test, settings of REGISTRATION_T3.md (4,000 clauses, T 4000,
s 8, 40 epochs, batch 128, hardest negative, negation, init half; epoch chosen on val, its state restored, test
read once). Views, each seeds 1, 2, 3 (12 runs):
- G: the sentence's meaning bits (OR over its tokens = existsT), the flat reading through the token code;
- V: the sentence's bag of training words (OR of the word bits), the flat word reading through the same code;
- G+S: G plus each token's own meaning bits (token clauses);
- V+W: V plus each token's own word bit (the word twin of G+S).

## Bars (test accuracy, mean of 3 seeds)
- H4a, sentence level: G at least V + 2.0 points ("works"); at least V + 5.0 ("strong").
- H4b, token level: G+S at least V+W + 2.0 ("works"); at least V+W + 5.0 ("strong").
- No bar: G against the flat existsT rows of REGISTRATION_T3.md part 1 (84.98), as a check of the token code.

## Frozen code
tok_tm.py df15c29d561182a2, graph_feats.py cd0725094c7e014d; decision-layer package read only: dl/tm.py
f649043f5801605b, dl/tm_init.py 557ff69706529738, dl/boolean.py b5f5917f390da8a6, dl/data.py 0c4f8d22576e817c.

## Known limits
- One dataset, 3 seeds. The MASSIVE test split was already read for the flat views of REGISTRATION_T3.md part 1;
  nothing here was tuned on it.
- Word symbols here cover tokens seen in training only (unseen tokens get none); meaning bits cover every token of
  the static table. That is part of what is compared, not a flaw to correct.

## AMENDMENT (2026-10-17 17:58:38Z UTC, before any of its runs)
1. Every view is trained to convergence: at most 120 epochs, val every 2 epochs, stop after 10 evaluations in a row
   without a new best val, the best-val state restored, the test read once (tok_tm.py v4, sha256 f17ef85c8f2c2c6d,
   --epochs 120 --patience 10). Reason: the phase A word views were still improving at the 40-epoch cap (best epoch
   38 to 40, +1.7 to +6.7 points from epoch 30 to 40), so a 40-epoch comparison would favour the meaning bits for
   learning speed, not only for what they say (a separate check, 17:55 UTC; confirmed).
2. H4b (G+S against V+W) is dropped: token clauses added nothing in phase A, and the shared CPU is throttled, so at
   most two jobs of this line run at once. Runs now: G and V, seeds 1-3 (6 runs, tags M3c_*). H4a bars unchanged.
3. The G control of REGISTRATION_T3.md part 2 runs separately at 40 epochs (M3_G_s1..s3), as registered there.
