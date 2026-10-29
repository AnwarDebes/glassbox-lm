# The glass box: the count floor, readable

A page that counts ten public-domain books in the browser and predicts the next word with the
CVTM-core of the tm_language_model study ([section 11 of its engine specification](../studies/tm_language_model/docs/11-engine-spec.md)):
Kneser-Ney 5-gram suffix clauses in their exact vote form, the document clauses D0 to D4, the
pointer clause and the bucketed mixture with MAP-EM weights. Every candidate word shows the clauses
that vote for it, the memory of the text typed so far, and the passages of the books where the
voting pattern was counted.

**Open it:** https://anwardebes.github.io/glassbox-lm/

No Tsetlin Machine runs on the page. Its model is the CVTM-core, the count model that
tm_language_model measured its Tsetlin Machine language model against, rebuilt so that it can be
read. It is the word part of the count floor of [research/design.md](../research/design.md)
(counted word patterns, a memory of the document and a copy rule), without the word-category codes
and the patterns over categories that the design adds.

## Files

| File | Role |
|---|---|
| `export.py` | reads the ten books (downloads them, or takes them from `--cache`), prints their SHA-256, cuts them into chapters, tokenizes them, fits the CVTM-core with `tmlm.cvtm.model.CVTMCore` (numpy backend) and writes the data files and `tests/golden.json` |
| `corpus.json`, `tokens.u16.gz` | vocabulary, books (with the SHA-256 of each book file), chapters and splits; the token ids (uint16, little-endian, gzip) |
| `core.json` | the MAP-EM weights of the mixture (520 buckets x 7 components), the fit and the held-out scores of the Python code |
| `study.json` | the 91 registered ids of tm_language_model with their statuses and qualifiers, read cell by cell from its results page, with the SHA-256 of the page it was read from |
| `model.js` | the estimator in JavaScript: counting (a stable counting sort that numbers the types as `numpy.unique` does), Kneser-Ney, suffix-clause votes, document clauses, pointer, buckets and mixture; the tokenizer, the document-word slots, sampling and the number formats of the page |
| `app.js`, `app.css`, `index.html` | the page |
| `tests/model.test.mjs` | `model.js` against `tests/golden.json` |
| `tests/data.test.mjs` | `study.json` against the results page of tm_language_model, the counts in `index.html`, and the slots of the corpus |
| `tests/text.test.mjs` | the tokenizer and `Vocab.encode` against the cases that `export.py` writes into `tests/golden.json`; `detok`, sampling and the number formats against outputs written by hand |
| `tests/page.test.mjs` | the page in a headless DOM (jsdom): keyboard and screen-reader wiring, the pause control, the error path, sampling |
| `tests/test_export.py` | the parser, the slots, the tokenizer and the book hashes of `export.py`; the combining-mark table of `model.js` against Python |
| `package.json`, `package-lock.json` | the test command and its one development dependency, jsdom 30.1.3 (MIT licence, from registry.npmjs.org; the lock file pins it and its 35 dependencies, all under permissive licences). It is a test tool and is not part of the published site. |

From the repository root:

```bash
# the data files; the study's environment has numpy and numba; the books are read from (or downloaded into) the folder
studies/tm_language_model/.venv-tm/bin/python live/export.py --cache <folder for the ten books>
studies/tm_language_model/.venv-tm/bin/python live/tests/test_export.py
cd live && npm ci && npm test     # npm ci installs jsdom into live/node_modules for tests/page.test.mjs
```

`export.py` took 24 s on a first run and 7.4 s on later runs on 2026-10-25 (one CPU process, two
threads, books read from `--cache`). Without jsdom, `npm test` skips the page test and runs the rest;
on a CI runner (`CI` set) a missing jsdom is an error.

## What the tests check

On the same tokens the JavaScript fit gives the type counts and the discounts of `KNModel`
exactly. On 2,500 sampled held-out positions, compared one by one, the largest difference of the
vote sums from `SuffixClauses` is 3.55e-15 in ln P; the document counts, the pointer, k* and the bucket
agree exactly; the largest relative difference of the probability of the core from `CVTMCore.gold`
is 3.54e-15 (the test fails above 1e-12). The top 8 of the full distribution agree with
`CVTMCore.dense_pass` at 40 positions. Over all 58,266 held-out positions the vote form differs from
the recursion by at most 3.55e-15 in ln P, and the held-out bits of the Kneser-Ney orders 1 to 5 and of
the core agree with the Python totals to within 1e-9, relative; single probabilities are compared at
the 2,500 sampled positions only. The page repeats the vote-form check on every held-out position
when it loads.

The tokenizer of `model.js` removes after NFKD exactly the code points that Python's
`unicodedata.combining` (Python 3.11, Unicode 14.0) calls combining, as `export.py` does; on the
227 chapters both give the same tokens. Only a character that the browser's Unicode version
decomposes differently from Unicode 14.0 can still give a different token (in Node 26, with
Unicode 17, these are 119 code points that Unicode 14.0 does not have).

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
are dropped, accents and other combining marks removed). Vocabulary: every word of the 208 training
chapters (19,173) and 256 document-word slots for words the books do not contain. A chapter holds
at most 160 such words, so each new word of a chapter has its own id; in typed text the new words
after the 256th share the last slot.

## Measured on the held-out chapters

Perplexity per token, by `export.py` (the study's code) and again by the page:

| Model | Perplexity |
|---|---|
| Kneser-Ney, order 1 | 596.06 |
| order 2 | 221.99 |
| order 3 | 196.17 |
| order 4 | 193.68 |
| order 5 | 193.29 |
| CVTM-core (order 5 and the document clauses) | 170.38 |

The vote form and the recursion differ by at most 3.55e-15 in ln P over the 58,266 positions.
These numbers belong to this corpus and tokenizer and are not comparable with the bits per byte
of the study.

Up to 2026-10-25 the page had 64 slots, so the 65th and later new words of a chapter shared one id
(266 held-out tokens in three chapters). The memory then counted different words as one repeated
word, and the core scored 165.91 (Kneser-Ney order 5: 193.20). With 256 slots it scores 170.38. A run with
160 slots gives 170.35: the larger vocabulary alone moves the core by a few hundredths, and almost
all of the change comes from the words that are no longer merged.

## The books

`export.py` reads each book from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` and keeps the
text between the `*** START OF` and `*** END OF` lines, so the Project Gutenberg header and licence
text are removed, and then removes the contents pages and the illustration captions. The files of
the export of 2026-10-21 were not kept. The files below were downloaded from gutenberg.org on
2026-10-25 (11:20:00 to 11:20:09 UTC). Run on them with the 64 slots of 2026-10-21, `export.py`
reproduces that day's data files byte for byte, except the fit time recorded in `core.json`
(`tokens_sha256` b1d68e1f3a91ad31…); the data files of this folder were written from them. `export.py` prints these hashes and writes them into
`corpus.json`.

| File | Book | SHA-256 |
|---|---|---|
| `pg98.txt` | A Tale of Two Cities | `d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784` |
| `pg1342.txt` | Pride and Prejudice | `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75` |
| `pg11.txt` | Alice's Adventures in Wonderland | `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e` |
| `pg1661.txt` | The Adventures of Sherlock Holmes | `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` |
| `pg84.txt` | Frankenstein | `2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42` |
| `pg43.txt` | The Strange Case of Dr Jekyll and Mr Hyde | `b43448a88391591f9cf82b25553df00faf47a2752a873185fcdd1156eebdb990` |
| `pg35.txt` | The Time Machine | `2892e919000e17c83e1dac51b30f4675db50536b644d7579fe8a89bb399a9bdc` |
| `pg46.txt` | A Christmas Carol | `8c36b93723d454982df045ec0efdf2664e5a48e6b5e2f670e883575f75ad0cc8` |
| `pg16.txt` | Peter Pan | `6b08714281fe38266a756741e4c62915cda7536c2f78cca501f7fd53f3f445ae` |
| `pg55.txt` | The Wonderful Wizard of Oz | `969bffab7740d4d8a0bac332d78bd0152ad4a40a2efc52f06c74a9bb6120be75` |

Project Gutenberg's notice in each file says that the eBook is for the use of anyone in the United
States at no cost and with almost no restrictions, and that readers elsewhere have to check the laws
of their country; the books are in the public domain in the United States. The fonts are under the
SIL Open Font License (`fonts/`).
