# AG News data for E5a (news topics, Check 2c)

Built by `src/build_agnews.py`, logged run `results/20261002T035105Z_e5a_data_build_v2`. The first build (`results/20261002T034314Z_e5a_data_build`) gave identical arrays; v2 only added statistics.

## Source

- AG News: news texts (title plus first sentences), each labelled with one of 4 topics: 0 World, 1 Sports, 2 Business, 3 Sci/Tech.
- Only the dataset's **train** file was read: `fancyzhx/ag_news`, revision `eb185aa`, 120,000 texts (30,000 per topic), sha256 `fc508d6d...1f52ee`. It comes from the Hugging Face cache of the tm_language_model study, read-only.
- The dataset's own test file was never opened.

## Words

- HTML codes are decoded (`#39;` is an apostrophe, `quot;` a quote mark). HTML tags and web addresses are removed.
- A word is a run of letters a-z, lowercased.
- Removed: the standard NLTK English stop-word list (179 words, such as "the", "and", "not") and one-letter tokens.
- Each text becomes the **set** of its words. Only presence counts, not how often a word appears.
- Vocabulary: the 1,024 words that appear in the most train texts (ties: alphabetical). `vocab.json` lists them in that order.

## Splits

One random shuffle of the 120,000 texts (seed 20261002):

| Split | Texts | World / Sports / Business / Sci-Tech | Use |
|---|---|---|---|
| train | 60,000 | 15,104 / 14,962 / 14,966 / 14,968 | training |
| dev | 10,000 | 2,485 / 2,514 / 2,511 / 2,490 | choosing `s` and passes, all reported dev numbers |
| test | 10,000 | (not looked at) | **the one final test, Day 5 only** |
| npmi_ref | 40,000 | 9,998 / 9,910 / 10,048 / 10,044 | only for NPMI word-group scores; in no other split |

- A text has 11.5 vocabulary words on average (median 11; 10% of texts have 6 or fewer).
- 17 train texts and 3 dev texts have no vocabulary word at all.

## How common the words are

"Document frequency" (DF) = the share of texts that contain the word.

- Over all train texts: the median vocabulary word is in 0.75% of texts. The rarest word is in 0.40%, the most common ("said") in 16.3%.
- Within its own most frequent topic, the median word is in 1.7% of that topic's texts (10th to 90th percentile: 0.8% to 4.7%).

**Which words can enter a rule.** A word with frequency p (among the texts of the topic that train a rule) can enter the rule only if `s` is above 1/p (Granmo's TM book, chapter 2). This table counts the words that pass this test in their most frequent topic:

| `s` | share of the 1,024 words | words per topic (World / Sports / Business / Sci-Tech) |
|---|---|---|
| 21.1 (Granmo's demo) | 10% | 42 / 37 / 36 / 30 |
| 60 | 50% | 198 / 173 / 212 / 166 |
| 100 | 80% | 354 / 290 / 365 / 314 |
| 230 | 100% | 628 / 529 / 689 / 649 |
| 400 | 100% | 783 / 652 / 835 / 818 |

`build_stats.json` has the full percentiles: overall, per topic, and the `s` each word needs.

## Things to know

- News-agency names ("reuters", "ap", "afp") and weekday names are among the most common words, and they are uneven across topics. They are real words in the data, so they were kept; they may show up in word groups.
- The most common words per topic:
  - World: said, reuters, iraq, president;
  - Sports: first, game, new, season;
  - Business: said, new, reuters, us, company;
  - Sci/Tech: new, said, microsoft, software.

## Files

- `{train,dev,test,npmi_ref}.npz`:
  - `indices` and `indptr`: the rows of a 0/1 matrix [texts x 1,024] in compressed sparse-row (CSR) form;
  - `y`: the topic, 0-3;
  - `row`: the text's row in the source file.
- `vocab.json`: the 1,024 words.
- `build_stats.json`: sizes and frequencies.
- Loader: `text_driver.load_split(DATA, split)`. It refuses `test` unless asked for the final test.
