# The glass box: the count floor, readable

A page that counts ten public-domain books in the browser and predicts the next word with the
CVTM-core of the tm_language_model study ([section 11 of its engine specification](../studies/tm_language_model/docs/11-engine-spec.md)):
Kneser-Ney 5-gram suffix clauses in their exact vote form, the document clauses D0 to D4, the
pointer clause and the bucketed mixture with MAP-EM weights. Every candidate word shows the clauses
that vote for it, the memory of the text typed so far, and the passages of the books where the
voting pattern was counted.

**Open it:** https://anwardebes.github.io/glassbox-lm/

No Tsetlin Machine runs on the page. It is the count floor of [research/design.md](../research/design.md),
the model that a Tsetlin language model has to beat, rebuilt so that it can be read.

## Files

| File | Role |
|---|---|
| `export.py` | downloads the ten books, cuts them into chapters, tokenizes them, fits the CVTM-core with `tmlm.cvtm.model.CVTMCore` (numpy backend) and writes the data files and `tests/golden.json` |
| `corpus.json`, `tokens.u16.gz` | vocabulary, books, chapters and splits; the token ids (uint16, little-endian, gzip) |
| `core.json` | the MAP-EM weights of the mixture (520 buckets x 7 components), the fit and the held-out scores of the Python code |
| `study.json` | the 91 registered ids of tm_language_model with their statuses, read from its results page |
| `model.js` | the estimator in JavaScript: counting (a stable counting sort that numbers the types as `numpy.unique` does), Kneser-Ney, suffix-clause votes, document clauses, pointer, buckets and mixture |
| `app.js`, `app.css`, `index.html` | the page |
| `tests/model.test.mjs` | checks `model.js` against `tests/golden.json` |

```bash
python live/export.py      # about 10 s on a CPU; uses numpy and numba through the study's code
cd live && npm test        # the browser model against the study's code
```

## What the test checks

On the same tokens the JavaScript fit gives the type counts and the discounts of `KNModel`
exactly. On 2,500 sampled held-out positions the vote sums agree with `SuffixClauses` to within
4e-15 in ln P; the document counts, the pointer, k* and the bucket agree exactly; the probability
of the core agrees with `CVTMCore.gold` to within 4e-15, relative. The top 8 of the full
distribution agree with `CVTMCore.dense_pass` at 40 positions, and the held-out bits of the
Kneser-Ney orders 1 to 5 and of the core agree with the Python totals to within 1e-9, relative.
The page repeats the vote-form check on every held-out position when it loads.

## Data and splits

| Book | Author | Year | Gutenberg id | Chapters | Held out | Tokens |
|---|---|---|---|---|---|---|
| A Tale of Two Cities | Charles Dickens | 1859 | 98 | 45 | 3 | 159,859 |
| Pride and Prejudice | Jane Austen | 1813 | 1342 | 58 | 5 | 140,589 |
| Alice's Adventures in Wonderland | Lewis Carroll | 1865 | 11 | 12 | 1 | 31,095 |
| The Adventures of Sherlock Holmes | Arthur Conan Doyle | 1892 | 1661 | 12 | 1 | 120,425 |
| Frankenstein | Mary Shelley | 1818 | 84 | 28 | 1 | 84,497 |
| The Strange Case of Dr Jekyll and Mr Hyde | Robert Louis Stevenson | 1886 | 43 | 10 | 1 | 29,691 |
| The Time Machine | H. G. Wells | 1895 | 35 | 16 | 4 | 36,982 |
| A Christmas Carol | Charles Dickens | 1843 | 46 | 5 | 0 | 33,885 |
| Peter Pan | J. M. Barrie | 1911 | 16 | 17 | 1 | 54,896 |
| The Wonderful Wizard of Oz | L. Frank Baum | 1900 | 55 | 24 | 2 | 44,388 |

A chapter (a story, for Sherlock Holmes) is one document. Held-out chapters: residues 90 to 99 of
`mix64(i XOR 0xC0FFEE) mod 100` over the global chapter index, except the first chapter of each
book (19 chapters, 58,266 tokens). Fit documents of the EM: residues 0 to 9 of the training index
(14 chapters, 45,383 tokens); the study uses residue 0 of 28,472 articles. Tokens: lowercase words
with inner apostrophes, numbers, and the marks . , ; : ! ? (quotation marks, dashes and brackets
are dropped). Vocabulary: every word of the 208 training chapters (19,173) and 64 document-word
slots for words the books do not contain.

## Measured on the held-out chapters

Perplexity per token, by `export.py` (the study's code) and again by the page:

| Model | Perplexity |
|---|---|
| Kneser-Ney, order 1 | 595.78 |
| order 2 | 221.89 |
| order 3 | 196.09 |
| order 4 | 193.59 |
| order 5 | 193.20 |
| CVTM-core (order 5 and the document clauses) | 165.91 |

The vote form and the recursion differ by at most 4.4e-15 in ln P over the 58,266 positions.
These numbers belong to this corpus and tokenizer and are not comparable with the bits per byte
of the study.

The books are in the public domain. `export.py` downloads them from gutenberg.org and removes the
headers, the licence text, the contents pages and the illustration captions. The fonts are under
the SIL Open Font License (`fonts/`).
