# WebGuard veto test: a first guard check for the harness (plan; nothing run yet)

The harness plan's first safety-guard step (PLAN_tm_harness.md sections 4 and 11). Option A was picked on
2026-10-20 (22:29:55Z) and set the conditions below. CPU only. It runs after 3c's
registration (done 23:19:40Z), and only after these hashes are recorded.

**Plainly, in every report:** the test part has 253 rows from 18 sites, so its site-bootstrap interval is about ±5-7
points. This is a first guard check, not a benchmark result. The paper's inputs and splits differ: its test a11y files
are not released. So there is no numeric comparison with the paper.

## Data (research only: CC BY-NC 4.0; never committed)

- monitor_factory_train_a11y.json of WebGuard (Zheng et al., arXiv 2507.14293), commit 4135c4a9, in
  data_untrusted/webguard/4135c4a9/ (FILES.sha256, LICENCE.md). It has 1,982 rows: the page's accessibility tree,
  the target node and the label (SAFE / LOW / HIGH).
- The same 1,982 rows, in the same order, in monitor_factory_train.json: the URL, the tagHead, and the site
  (the website folder in its image path). The labels are equal in every row.
- The official test files are not used. Their only text is the URL, the tag and "CLICK".

## Split (fixed before any feature: webguard/wg_split.py 01e3ffe6d1b001b7 -> wg_split.json a31de065415fb9d7)

- The 115 sites go through a numpy default_rng(20261021) permutation: 80 train, 17 dev, 18 test. Every row follows
  its site. Site lists sha256: 78fb403b084f72c9.

  | part | sites | rows | SAFE | LOW | HIGH |
  |---|---|---|---|---|---|
  | train | 80 | 1,248 | 300 | 569 | 379 |
  | dev | 17 | 481 | 176 | 209 | 96 |
  | test | 18 | 253 | 69 | 98 | 86 |

- Dev makes every choice. The test part is read once, by wg_eval.py, after everything else is frozen.
- The site lists, copied from wg_split.json (site names as the dataset spells them):
  - test (18): Australia gov au, Build, Hiking Porject, Marketwatch, ROCKAUTO, Redbubble, TheGuardian, The_Economist,
    Trustpilot, Usa_gov, aliexpress, cnbc, edx, gandi, health mil, hm, howstuffworks, investing;
  - dev (17): ALJAZEERA, Angieslist, DoubleVerify, FreetaxUSA, Google Ads, Grand_nord_auto, HOUZZ, Newegg, Soundcloud,
    WEATHER, Xiaomi, asana, dropbox, instructables, legalzoom, usgs, washingtonpost;
  - train (80): the other 80 sites, listed in wg_split.json.

## Bits (all fitting on the train part only)

Words come from the C4 counting table ppw_S1000 (no neural network at any stage), with its tokenizer
(wordtab_feats.glove_tokens). A spherical k-means with K 256 runs over the table rows of the train part's word list
(25 iterations, seed 0). Then:

- **Node block** (the target node's own words):
  - existsT over the node's name words: per concept, the max cosine over the words, Thermo(8) fit on the train rows;
  - one-hot role (link, button, checkbox, ...) and one-hot tagHead (values seen 5 or more times in train; the rest
    go to "other");
  - the veto bit: the node's name contains one of the fixed irreversible-action words (below).
- **Parsing facts** (checked on all 1,982 rows before any feature; no labels used):
  - every target node parses: button 1,129, link 432, div 181, input 177, and others;
  - 744 rows have no node index and an empty tree;
  - no row has a page title, because the RootWebArea has no name in this file;
  - 373 rows have no names in the tree;
  - the veto bit is on in 129 rows.
- **Page block** (the context, kept apart from the node):
  - existsT over the URL's words (and the page title, which is always empty here);
  - and, so that a long tree does not saturate the OR bits (P7), count bits over the accessibility tree's names:
    per concept, the number of tree words whose nearest concept it is, as bits >= 1, >= 4, >= 16.
- The TM and LR read the same bits.

## Models (dev chooses; seeds 1-3 at the choice)

- **TM:** dl CoTM on the CPU (2 threads; torch.cuda.synchronize is a no-op in this script, since the GPU is not
  used). 1,000 clauses, T 1,000, 40 epochs, batch 64, hardest negatives, negated literals, val on dev at the best
  epoch. s from e74's set {2, 4, 8, 12, 16, 24} at seed 1 on dev; then seeds 1-3 at that s.
- **LR:** dl.linear multinomial logistic regression on the CPU, C in {1, 10, 100}, best on dev.
- **Speed test first:** one TM fit on 200 train rows for 2 epochs, timed. If the projected total (6 + 3 fits) is
  over 2 hours, the TM waits for a GPU slot after 3c.

## Bar and metrics (fixed now; the test part once)

- **Metrics:** 3-class accuracy and HIGH recall. The TM is the mean over seeds 1-3.
  - "False alarm" is a SAFE row predicted HIGH. LR's HIGH recall is also given at the TM's own false-alarm rate:
    the threshold on P(HIGH) is set on dev to match the TM's dev rate.
- **Interval:** 10,000 bootstrap resamples over the 18 test SITES, 95% percentile intervals.
- **BAR:** the TM is non-inferior to LR on the same bits when the lower end of the site-bootstrap interval of
  (TM accuracy - LR accuracy) is above -5 points. Otherwise it is not shown to be non-inferior. HIGH recall and
  everything else is descriptive.
- **Veto overlay** (e70 style; the property holds by construction, so its COST is the result):
  - **Rule (e70's guard form):** a veto clause on the veto bit, with weight -M on SAFE. M is e70's rule: SAFE's
    positive weights + the absolute weights of HIGH + 1. When the bit is on, SAFE cannot win, and the TM's own votes
    pick LOW or HIGH. For LR, the same rule: with the bit on, the argmax over LOW and HIGH.
  - **Fixed word list**, from the WebGuard schema's HIGH examples (purchases, money matters, contracts, loans, bets,
    creating accounts, public posting): buy, purchase, checkout, pay, payment, order, donate, subscribe, bet, wager,
    loan, register, signup, publish, transfer, and the pairs "sign up", "place order", "create account".
  - **Reported:** accuracy and HIGH recall with and without the overlay, for the TM and for LR, and how many test
    rows it changes.
- **Solver check** (scripts/verify_policy.py's build_and_solve, imported unedited: the exact MILP encoding,
  HiGHS):
  - Can SAFE win with the veto bit on? Asked of the TM alone and of the TM with the guard clause.
  - The domain rows: thermometer order within each concept, one role, one tag, and ordered count bits.
  - Expected: the TM alone gives a counterexample or a proof (reported as found); with the guard it is infeasible
    by construction.
  - Plainly: a proof covers bits -> label, not page -> bits.

## Novelty (from novelty_tm_guards.md, narrowed 2026-10-20)

- ActGov and ROPE govern agent actions without proofs over a learned classifier.
- We found no published study that trains a Tsetlin machine veto on WebGuard, or proves a property of a learned web
  action-risk classifier.
- Say "a first guard check", never "first".

## Files

- wg_split.py (done)
- wg_feats.py (the bits)
- wg_run.py (dev screen, final fits, test predictions saved without scoring)
- wg_eval.py (the test metrics once, bootstrap, overlay, solver check)
- wg_chain.sh

Hashes are recorded before any run.
