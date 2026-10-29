# DATA_CARD: clean Wikipedia paragraph pairs (Thu 1 Oct 2026)

Built by `src/build_pairs.py`, checked by `src/data_checks.py`. Every number below comes from the logged run
`results/20261001T094249Z_data_checks_final_exploratory`. Load with `src/pairdata.py`: `load_xy("sa", "train")` gives the TM input.
`data/ns_all/` was built separately (`src/build_ns_all.py`); this card does not cover it.

## What is in it
- Source: WikiText-103 (raw), its train part: 28,472 Wikipedia articles, 703,005 paragraphs.
- A paragraph is a non-heading line with at least 30 lowercase words. Its words: lowercase letters only, minus the probe's stop-word list and one-letter words, kept as a set (no counts, no order).
- Splits are by whole article (seed 20261001): train 19,930 articles, dev 2,848, final test 2,847, NPMI reference 2,847. No article is in two splits. NPMI-reference articles are in no pair. Article ids: `splits.json`.
- The splits are made on article indices only: `src/build_pairs.py` permutes the indices of the WikiText-103 train articles and cuts the permutation into the four parts. There is no deduplication by content or by title, and WikiText-103 contains repeated articles, so the same text can sit in two splits under two indices. About 2.1% of the final-test paragraphs are word for word identical to training paragraphs, in 235 of the 2,847 final-test articles (from an internal audit of 2026-10-03, not recomputed, because recomputing it reads the test split). The one final test was scored with these duplicates in its test split. (Note added on 2026-10-25.)
- Vocabulary: the 1,024 words that occur in the most train paragraphs (`vocab_F1024.json`; top 3,072 in `vocab_F3072.json`). A paragraph holds 21.8 of the 1,024 words on average.
- Each version has train 40,000, dev 20,000 and final test 10,000 pairs. Half are "yes" (label 1), half "no" (label 0). Which paragraph is A and which is B is random.
- Also here: `paragraphs_F3072.npz` (every paragraph as word ids) and `build_stats.json` (counts).

## How the pairs were built
- **sa (same article).** Yes: pick an article, then 2 different paragraphs of it. No: pick 2 different articles, then 1 paragraph of each.
- **np (next paragraph).** Yes: pick an article, then a paragraph and the one right after it. No: pick 2 different articles; in each, pick a "paragraph and the next one" and keep one of the two at random. By design, both classes then hold first and last paragraphs equally often.
- **ns (next paragraph, shared words removed; the synonym test).**
  - Start from np yes-pairs. Group them by the exact set of words the two paragraphs share, and by the length of B.
  - Inside a group, swap the B paragraphs between pairs from different articles, so each new pair shares exactly the same words. The swapped pairs are the no-pairs.
  - Then every word that is in both paragraphs is removed from both. Word overlap is 0 in every pair (checked with 1,024 and 3,072 words).
  - Each paragraph thus appears as often in yes-pairs as in no-pairs, in the same role (earlier or later paragraph). A no-pair has the same shared-word count and the same two lengths as its yes-pair.
  - Limit: most ns pairs shared 0 or 1 words before removal (train 48% and 46%, dev 65% and 33%). Pairs sharing more words rarely find a swap partner. So ns is not a random sample of np pairs.
  - Use ns only with the 1,024-word vocabulary. The matching holds only there, and the loader refuses other sizes.

## Check 1b (gates.md, "no shortcuts"): all three versions pass
| version | words with \|z\| > 3 | one-paragraph LR [95% CI] | partner-shuffle LR [95% CI] | length-only LR / boosted trees [95% CI] |
|---|---|---|---|---|
| sa | 2 | 0.497 [0.490, 0.504] | 0.502 [0.495, 0.509] | - |
| np | 3 | 0.499 [0.492, 0.506] | 0.4935 [0.4866, 0.5005] | - |
| ns | 0 | 0.499 [0.492, 0.506] | 0.498 [0.492, 0.505] | 0.502 [0.495, 0.509] / 0.500 [0.493, 0.507] |

- z-test: is a word more common in single paragraphs of yes-pairs than of no-pairs (paragraph A of the 40,000 train pairs)? Pure chance gives about 3 words with \|z\| > 3. We count "at most about 3" strictly as 3 or fewer.
- LR = logistic regression, trained on train and tested on the 20,000 dev pairs. CI = 95% confidence interval. Partner-shuffle: B is replaced by the B of a random other pair. All three must stay near 0.50, and they do.
- The length-only values were computed with the `length_feats` of 1 Oct 2026, in which the feature |len A - len B| wrapped around to about 1.8e19 whenever B was the longer paragraph (unsigned counts; fixed on 2026-10-25, see the comment in `src/data_checks.py`). The other four length features were right. A dev re-run of 2026-10-25 with the fixed function changes no pass verdict: every length-only interval still includes 0.50. The table keeps the logged values; at its three decimals the fixed function gives 0.499 [0.492, 0.506] for the boosted trees on ns (the logistic regression is unchanged). (Note added on 2026-10-25.)

How we got here (all logged in `results/`):
- v1 followed the study design exactly. sa passed. np failed narrowly: 5 words, one-paragraph 0.507 [0.5001, 0.5139]. Likely cause: its no-pairs took any paragraph, so by design of that sampler first and last paragraphs were up to twice as common there as in yes-pairs.
- ns v1 failed badly: 690 words, one-paragraph 0.661, length-only 0.679. Removing shared words takes topic words such as "album" out of yes-pairs only, and the matched no-pairs were longer (26.1 vs 18.8 words).
- v2 fixed np as described above and tried a different ns. ns still failed (15 words, one-paragraph 0.516 [0.509, 0.523]).
- v3 (this data) uses the swap. A rebuild with the final code reproduces every file exactly (`*_data_build_v3_repro_exploratory`).

## Baselines (dev pairs, fitted and tuned on train; accuracy / AUC; always answering "no" scores 0.500)
| baseline | sa | np | ns |
|---|---|---|---|
| Jaccard (shared words / all words), tuned cut-off | 0.761 / 0.819 | 0.804 / 0.864 | 0.500 / 0.500 |
| tf-idf cosine (rare shared words count more) | 0.771 / 0.824 | 0.812 / 0.869 | 0.500 / 0.500 |
| shared-word count | 0.731 / 0.791 | 0.764 / 0.828 | 0.500 / 0.500 |
| LR on the shared words (A AND B) | 0.774 / 0.821 | 0.802 / 0.861 | 0.500 / 0.500 |
| LR on [A, B, A AND B] | 0.792 / 0.871 | 0.829 / 0.907 | 0.499 / 0.498 |
| PPMI word vectors, cosine of paragraph means | 0.774 / 0.854 | 0.810 / 0.889 | 0.648 / 0.704 |

- AUC: how well a score ranks yes-pairs above no-pairs (0.5 = guessing, 1.0 = perfect).
- PPMI: 100-number word vectors from how often two words share a train paragraph. A paragraph is the mean of its word vectors.
- On ns, every overlap baseline is at chance by construction. PPMI still reaches 0.648, so different-but-related words carry signal.
