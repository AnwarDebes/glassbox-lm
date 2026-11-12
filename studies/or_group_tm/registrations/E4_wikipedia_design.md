# E4: Wikipedia paragraph pairs with the OR-group rule (design work on the DEV pairs)

Publication note: the text below is the plan as it was fixed on 2026-11-12T16:20:17Z (SHA-256 of that file: `89f51bae1fc39c077a792dbd1e1ff912c3c53866b875d90da0bd70d8fa3b98f8`), with folder and file names changed to those of this repository. Rules, bars, settings and numbers are unchanged.

Written before any Wikipedia run of this study. Exploratory design work; no test pair is read in E4.

## Question

On the Wikipedia pairs of the hierarchical_tm study (same files, same 1,024-word vocabulary), where the stock rule ended below word overlap in the one final test: does the OR-group rule (G1) reach the stock model, and (G2) reach word overlap?

## Data (the study's, read only)

`data/{sa,np,ns_all}/{train,dev}.npz`: 40,000 train and 20,000 dev pairs per version; `sa` = same article or not, `np` = next paragraph or not, `ns_all` = next paragraph with every shared word removed. F = 1,024. The test pairs (10,000 per version, read once by the study) are not opened in E4; `run_wiki.py` refuses them without a frozen record.

## Reference numbers (read from `../hierarchical_tm/docs/results.md`)

| | sa | np | ns_all |
|---|---|---|---|
| stock, headline setting (50 rules), dev AUC, seeds 1 / 2 / 3 | 0.745 / 0.739 / 0.733 | 0.752 / 0.728 / 0.751 | 0.622 / 0.622 / 0.611 |
| stock, second setting (2,000 rules), dev AUC | 0.768 / 0.763 / 0.771 | 0.796 / 0.792 / 0.778 | not run |
| word overlap (Jaccard), dev AUC | 0.819 | 0.864 | 0.500 by construction |
| best logistic regression, dev AUC | 0.871 | 0.907 | - |
| final test: stock / word overlap / PPMI cosine / logistic regression | 0.735 / 0.817 / 0.852 / 0.869 | 0.750 / 0.857 / 0.886 / 0.903 | 0.614 / 0.500 / 0.811 / 0.494 |

## Model and first look, fixed now

2,000 clauses of positive polarity, K = 1 group per paragraph, T = 40, mode `side` (a group learns from a paragraph when the clause's other paragraph matched), reinforce `critical`, reject `weakest`, r = 2, initial state -1, one seed event per example, 10 passes, seed 1, on `sa`. Score: AUC of the vote sum on the dev pairs, as in the study; the partner-shuffle control is reported with every run.

## Design arms that may follow (each one logged in `logs/RUNS_wiki.md`, whatever it shows)

Single changes on the first look: initial state -3; reinforce `share` or `all`; T in {20, 100}; clauses in {500, 8,000}; K = 2 with `seed_all`; decay 0.1; r = 4; 30 passes; mode `near`. The chosen arm then runs on `sa`, `np` and `ns_all` with seeds 1 to 3.

## Bars, fixed now

- G1 (reaches the stock model): mean dev AUC of seeds 1 to 3 at least 0.02 above the stock headline setting on `sa` (0.739 + 0.02 = 0.759) and on `np` (0.744 + 0.02 = 0.764).
- G2 (go-bar for a registered test against word overlap, E5): mean dev AUC of seeds 1 to 3 at least 0.830 on `sa` and 0.875 on `np`, that is word overlap + 0.011 (the half-width of the paired interval of the study's final test), so that a registered "above word overlap" has a fair chance. Below it, the test pairs are not read for that comparison.
- `ns_all` (no shared word): reported against the stock 0.618 and against PPMI cosine (0.811 on test); no bar.
- Every run: partner-shuffle AUC within 0.47 to 0.53, else the run is void.

## What a miss means

- G1 missed: "the rule solves the planted and the template tests and does not carry to Wikipedia pairs"; reported as measured; no registered test.
- G1 met, G2 missed: a registered test of "above the stock model" only (the stock test scores exist), with word overlap, PPMI cosine and logistic regression shown beside it.

## Limits stated before the runs

- Selection on dev is optimistic; only a registered test read counts.
- 40,000 pairs and 1,024 mostly frequent words are a small world; PPMI cosine and the logistic regression are counting and linear baselines that the stock model did not reach.
- The dev and test splits contain duplicate articles across splits (known limitation of the data, see the repository's page of known limitations); it applies to every method alike.
