# PLAN W1: a Tsetlin machine learns its own word table from raw text

> Status 2026-10-19 (after W2 and W3): Line A is closed on every bet tried. W1's TM autoencoder lost to counting at
> step 0; H-B's question->answer pairs did not improve the counting table; a lexicon (W3) did not either. The
> transformer-free table stays ppw_S1000 (on par with potion on topic benchmarks, W1-T; 2 to 3.5 TM points below on
> intents). See W2_NOTE.md section 6 and SUMMARY_2026-10-19.md.

Status: CLOSED AT STEP 0, NEGATIVE (2026-10-19 ~02:02Z). No val or test run of W1 happened, and no registration was
written. The sections after "Result of step 0" are the plan as drafted before the engine search, kept as the record.

## Result of step 0 (all label-free; S1 / S10 train slices and held-out C4 only)

- **Verdict:** with this engine, a Tsetlin machine does not learn a word table as good as counting from the same
  text at 1-10M tokens. It does not catch up as the data grows in that range. W1 stops before any val run, by the
  rule fixed in the study log (01:1xZ) before the decisive runs.
- **Engine:** the TM autoencoder recipe of Bhattarai et al. on the GPU CoTM (w1/mo_tm_ae.py, dl/tm.py unchanged).
- **The probe:** a label-free score. The true word of a held-out C4 position is ranked by the cosine to the mean of its
  context vectors, against 99 unigram^0.75 negatives, as mean reciprocal rank. Chance is about 0.052.

| Table | Probe at S1 (1M tokens) | Probe at S10 (10M tokens) | Gain | Neighbour check (4 of 10 needed) |
|---|---|---|---|---|
| PPMI-SVD twin (500 dims) | 0.096 | 0.196 | +0.100 | 8 / 10 at S1, 10 / 10 at S10 |
| Tsetlin table (autoencoder) | 0.067 | 0.111 | +0.044 | 1 / 10 at S1, 2 / 10 at S10 |
| SGNS (diagnostic, neural) | 0.089 | - | - | 3 / 10 |
| potion (reference) | 0.151 | 0.172 | - | 10 / 10 |

- **The rule needed two things; both failed.**
  - The TM's gain had to beat counting's by at least +0.010. It was -0.056.
  - At least 4 of the 10 probe words needed a sensible neighbour from a list fixed in advance. The TM got 2 of 10.
  - At 10M tokens the TM table is only about where counting is at 1M.
  - SGNS also loses to counting at S1. So the S1 loss alone is the normal small-data regime; the slope is the real
    negative.
- **Engine search, every variant (S1 unless stated; held-out class-sum MRR, then probe):**
  1. **CBOW family** (one context window per example, random negatives), 4 variants. Held-out MRR was 0.020-0.027 in
     all of them, which is the value of a full tie: almost no clause fires on a single 10-token window. Probes 0.058
     (no budget) and 0.058 (budget 4, batch 256).
     - No literal budget, batch 1,024: Type II feedback on sparse inputs makes clauses thousands of literals long.
     - dl/tm.py's exact budget of 4 (batch 1,024 or 256) or 8 keeps clauses at 3-8 literals, but they still rarely fire.
  2. **Autoencoder family** (the paper's accumulation of 25 contexts per example):
     - u 25, no budget, 30 rounds: held-out 0.059, probe 0.063.
     - u 25, budget 8: 0.062, probe 0.057.
     - u 10, budget 8: 0.069, probe 0.051.
     - u 25, no budget, 1,000 rounds: the held-out score peaked at round 50 and fell back to chance; probe 0.067.
     - The same setting at S10 (650 rounds, 46 min): held-out 0.124, probe 0.111.
- **Speed:**
  - The card has 31.7 GiB.
  - Throughput: about 1.5-2.2 x 10^4 examples/s at S1, 1.6 x 10^4 at S10, and 5.6 x 10^3 at S100 (61k outputs, batch
    1,024). That is 2.75 h per S100 epoch.
  - torch.binomial over the whole clause x literal matrix takes about half of each minibatch.
- **What the probe is:** a distributional probe on C4, not task quality. On it, PPMI at 10M tokens even beats potion
  (0.196 against 0.172). That says nothing about the intent or N7 tasks.
- **Plainly:** counting learns word meaning from raw text faster than the TM autoencoder (the published recipe on the
  GPU CoTM). The gap widens from 1M to 10M tokens.
- **It is still a real result.** As far as the W1 prior-art search found, it is the first measured data-scaling
  comparison of a TM word table against a same-corpus, same-vocabulary count twin. Its scope: on this label-free probe,
  at 1M and 10M tokens. It is not through a downstream TM classifier; that was never run.
- **Runs counted:** 9 engine runs (4 CBOW, 4 autoencoder at S1, 1 autoencoder at S10), plus the twin at S1 and S10, the
  SGNS diagnostic and potion.
- **What it answers in CAIR's open question 2 (section 0b):** at 1-10M tokens of C4, the PMI-style counting
  ("Bayesian pre-processing") gives a clearly better word table than a Tsetlin machine trained on the same text.
  This holds for the engine tried here. It is not a proof for every possible TM design.
- **Also found on the way:**
  - GloVe 42B (counting at 42B web tokens) is as good as potion on N7's five sets, but not on the intent sets
    (2.5-3.6 below potion). The transformer table's edge is task-dependent (the study log, N7-F2).

## 0. In short

- Goal (the line's "wow"): a Tsetlin machine (TM) learns a word table from raw text alone (no labels, no neural
  network at any stage). Used through existsT, the table comes within 1-2 points of the transformer-distilled
  potion table on the intent sets.
- Decisive experiment: a scaling curve. The TM table against a count-only twin (PPMI-SVD) built from the same text
  and the same vocabulary, at 1M, 10M and 100M tokens. Both go through the same existsT harness and TM classifier,
  intent val first.
- Honest forecast:
  - The "wow" is unlikely at 100M tokens or fewer. GloVe (counting at 6B tokens) is 2.7-4.1 points below potion in
    this harness. The published TM embedding (One Billion Word) is about GloVe level on word similarity.
  - Beating counting is open.
  - The curve is informative either way. In the neural literature, predictors lose to counting at 1-10M words, tie
    near 100M and win at 1B (Sahlgren and Lenci 2016). If the TM follows that pattern, a 1B-token point is next.
- Declared negative: test 5 (TM autoencoder, about 300k tokens, one CPU thread) gave word codes no better than
  counting.

## After step 0: the counting pipeline (W1-P, W1-S) and its registered test (W1-T) (added 2026-10-19T04:46:52Z)

The useful side of the negative: the counting twin itself, through existsT and the same TM classifier, gives a Tsetlin
text pipeline with **no neural network at any stage**: C4 text, a C4 word list, counting + SVD, k-means + thresholds,
then a Tsetlin machine.

- **W1-P / W1-S** (exploratory, val and carved val; the study log 02:3xZ and 04:2xZ):
  - **Intent sets:** the curve is flat after 100M tokens. At S1000: 92.22 / 88.76 / 89.47 with potion's token strings,
    and 92.29 / 87.91 / 89.34 with the C4 word list.
    - That is level with or above GloVe 42B, but 1.8-3.6 below potion.
    - Not reached. Nothing is claimed for intents.
  - **N7 sets, C4 word list at S1000:** within 1.0 of potion on TREC, R52 and R8; 2.25-2.74 below on MR and SST-2.
  - **Potion's token list does not help:** C4 words are better on 4 of 5 N7 sets and about equal on intents.
- **W1-T, REGISTERED** (REGISTRATION_W1T.md; the second use of N7's test splits, declared; independently recomputed):
  - Test S2 (mean of 5 seeds), TREC / MR / SST-2 / R52 / R8: C4-word table 85.72 / 68.01 / 73.98 / 86.92 / 94.92;
    potion 84.24 / 70.03 / 77.98 / 85.63 / 94.42.
  - Paired difference against potion: +1.48 / -2.03 / -4.00 / +1.29 / +0.50. **REACHES POTION** (within 1.0 on 3 of 5;
    above it on TREC, R52, R8).
  - It also reaches GloVe 42B (4 of 5), and adds over bow on 3 of 5.
- **Caveats, kept with every number:**
  - this is the second use of these test splits;
  - the comparison is to potion's table in OUR TM setting, which is below the published TM numbers (N7 bar A failed).
    So never write "state of the art" or "parity with published TMs";
  - it is below potion on sentiment;
  - the intent sets were not reached;
  - never write "first network-free TM text pipeline" (Kadhim et al. 2025).
- **Handover:** the frozen table and its reference tokenizer are in w1/export/c4words_S1000_v1/ (README.md = the note
  for the decision-layer agent).

## 0b. Motivation: two open questions from CAIR's own research notes

The README of cair/LogicalTransformer, CAIR's research-notes repository, lists W1's two questions almost word for
word. Quoted from the copy the W1 agent archived (archive/w1_agent/lt/README.md, read as data), under "Questions to
investigate":
- line 12: "How can we make the random hypervectors less random and more semantic for boosted accuracy, e.g., in
  integration with logical embeddings?"
  - existsT already is a semantic hypervector: per-token concept bits, OR-ed.
  - W1 makes its table logical (learned by a TM). That is the "integration with logical embeddings".
- line 17: "Should the transformer part be Bayesian pre-processing (see Bayesian Word Profiler), or should the
  Tsetlin machine perform the transformation?"
  - The Bayesian Word Profiler in the same repository is PMI-style counting.
  - W1's TM table against a same-corpus PPMI-SVD twin answers this question with a measurement.

The same README's "Promising Solution Ingredients" (line 30) has the idea W1's engine is built around: "For each
class, train a classifier per token, predicting the presence of that token (by Ahmed). Can be done self-supervised,
scales well. Incorporates logical language modelling (Bimal). Each classifier is an "embedding" of its token."
- That idea is CAIR's. W1 cites it and does not claim it.
- W1's own part is the measured data-scaling comparison against a counting twin through the same TM classifier.
- A clean answer to these two open questions is useful whichever way it comes out.

## 1. Where we start

Val accuracy, test 2 harness: existsT, 512 concepts, Thermo(8) bits, then the CoTM classifier.

| Row | CLINC150 | Banking77 | HWU64 | Source |
|---|---|---|---|---|
| potion-base-32M existsT (transformer-distilled) | 94.60 | 91.46 | 91.23 | test 2, 3 seeds |
| MiniLM bits (reference) | 94.62 | 92.25 | 90.30 | the study log 19:10:27Z |
| GloVe 6B existsT (transformer-free, 6B tokens) | 91.50 | 87.34 | 88.51 | GloVe arm, 3 seeds |
| words only (bag of tokens) | 81.23 | 80.36 | 77.42 | test 2 |
| T5 TM autoencoder, weight rows (R3a), about 300k tokens | 85.62 | 85.19 | 81.41 | REGISTRATION_T5.md |
| T5 PPMI-SVD twin, same 300k tokens | 86.06 | 84.50 | 84.58 | REGISTRATION_T5.md |
| T5 potion, same 4,564 tokens | 93.12 | 91.03 | 89.78 | REGISTRATION_T5.md |

- The T5 rows used 2 seeds and a 4,564-token vocabulary. The other rows used 3 seeds and each dataset's own
  vocabulary.
- T5 lessons:
  1. 300k tokens is far too little.
  2. The OR over clause bits saturates (R1: 45.2 / 31.1 / 48.0). The weight rows through existsT work.
  3. One CPU thread of tmu is too slow (2.19 h for 4,564 words).

## 2. Prior art and what would be new

Agent logs, all quotes and links: novelty/W1_tm_word_table_prior_art.md and novelty/W1b_count_predict_corpus.md.

**TM word tables**
- Bhattarai et al., Findings of EACL 2024 (arXiv 2301.00709): a TM autoencoder trained on One Billion Word.
  - Tokens used and vocabulary size are not stated. Settings: 600 clauses, T 1200, s 5, 2,000 rounds, 25 sentences
    OR-ed per example.
  - Average Spearman on 6 similarity sets: TM 0.52, word2vec 0.58, GloVe 0.57, FastText 0.54. All four were trained
    on the same corpus.
  - Downstream classifier: a BiLSTM, not a TM. No count model, no data-size curve.
  - It claims "the first logic-based word embedding".
- Kadhim et al. 2025 (arXiv 2501.19018; ISTM 2025), two-phase TM autoencoder: One Billion Word, 40k vocabulary,
  "six months on a DGX H100". Similarity average 0.41, against 0.50 for the old TM autoencoder. Used only for IMDb
  data augmentation.
- Omni TM-AE 2025 (arXiv 2505.16386) and FastOmniTMAE 2026 (arXiv 2605.06982): one small TM per word, and vectors as
  long as the 40k vocabulary.
  - Similarity about level with word2vec, FastText and GloVe trained on the same corpus (0.546 vs 0.545 / 0.550 /
    0.512).
  - On an H100, the CUDA version ran 5.6x slower than one CPU thread.
  - No scaling curve and no count model.
- CAIR code, cair/LogicalTransformer (2023-24; no paper, no recorded results):
  - one yes/no TM per target word;
  - a GPU TM with one shared clause pool that predicts the next word's random 512-bit code;
  - GPU next-word classification;
  - a PMI-style count profile.
  The pieces of this plan exist there as code ideas. Nothing was measured or compared.
- TM language models found:
  - next character (Rahimi Gorji, Granmo, Goodwin 2023);
  - contextual word prediction (Abdelwahab et al., ISTM 2025, abstract only);
  - character-level generation (Tsetlin.jl).
  No word-level TM language model with perplexity was found.
- "Clusters are All You Need" (Gao et al., arXiv 2606.19815): BERT document embeddings, k-means (k = 200), then
  pre-training a non-negated TM. A neural encoder is used; it is not a word table learned from raw text. Must be cited.

**Counting versus prediction**
- Levy, Goldberg and Dagan 2015 (TACL): "much of the performance gains of word embeddings are due to certain system
  design choices and hyperparameter optimizations, rather than the embedding algorithms themselves".
  - Tuned PPMI-SVD matches SGNS on word similarity.
  - Recipe: context smoothing 0.75 always, eigenvalue weight 0 or 0.5 (never 1), no shift for SVD, window 2 best for
    counting, unit-length rows.
- Levy and Goldberg 2014: SGNS factorizes shifted PMI.
- Baroni et al. 2014: prediction wins on 2.8B tokens, but against untuned count models.
- Sahlgren and Lenci 2016 (EMNLP): nested 1M / 10M / 100M / 1B-word prefixes of one corpus, the design used here.
  - Mean of their 5 scores (agent's arithmetic): SGNS 8.1 / 28.6 / 45.7 / 55.1; PPMI 14.3 / 32.2 / 43.2 / 47.6.
  - So the predictors gain the most from data.
- Altszyler et al. 2016: skip-gram beats LSA above about 10M words.
- Lai et al. 2015: "corpus domain is more important than corpus size".

**What would be new** (not found published):
- the first data-scaling comparison (1M / 10M / 100M tokens) of a TM word table against a same-corpus,
  same-vocabulary count-only twin;
- judged through the same downstream TM classifier;
- with a GPU multi-output TM at thousands of shared clauses.

**Do not claim:**
- the first logic based or TM word embedding (Bhattarai et al. 2024);
- the first TM embedding on a big corpus (One Billion Word is done);
- the first GPU TM embedding code (tmu has CUDA kernels; the CAIR demos);
- that TM tables beat count models (untested until W1 runs).

**Must cite:** Bhattarai et al. 2024; Kadhim et al. 2025; Omni and FastOmni; cair/LogicalTransformer; Gao et al. 2026;
Levy, Goldberg and Dagan 2015; Sahlgren and Lenci 2016. For existsT: Yadav et al. 2021, SFBoW / Zhelezniak et al.
2019, Kenter and de Rijke 2015.

## 3. Design principle: only the learner differs

- **Same raw text: C4 en.**
  - Potion's Tokenlearn step distilled bge-base on C4 text. The Model2Vec blog says about 1M C4 texts for the
    original potion models. The model card now names minishlab/tokenlearn-c4-en-bge-base-v1.5. Which exact set
    trained the 32M model is not verified.
- **Same tokens: potion's 63,091-token WordPiece list.**
  - It is 29,528 BERT uncased pieces plus about 33,563 added lowercase words. How those words were chosen is not
    stated.
  - The list is used only as a list of strings. No number from potion's vectors is used anywhere in the TM table or
    the twin.
  - Every table then sees exactly the tokens potion sees, so a comparison changes only the numbers in the table.
- **Same harness and seeds:** existsT, then the test 2 CoTM classifier.
- **So three learners read the same text:** the TM, counting (PPMI-SVD), and distillation from a transformer (potion).
- **What still differs, plainly:** potion's teacher, bge-base, was trained on far more text and on text pairs before
  the C4 step.
  - So W1 is not a same-data contest with potion. It is a same-data contest between the TM and counting.
  - Potion and GloVe-6B are fixed references.
- **Word tokens instead of WordPiece:** not chosen. They would change the harness (as in the GloVe arm) and mix the
  table with the tokenizer.

## 4. Corpus

- **Source:** allenai/c4, en, shards c4-train.00000 and .00001 of 01024, revision 1588ec45. Licence: ODC-BY, "also
  bound by the Common Crawl terms of use".
  - Shard sizes: 319,308,785 and 318,039,285 bytes; sha256 = the LFS ids on Hugging Face. Script: w1/get_c4.sh.
  - Stored in data_untrusted/c4_en/ with a SOURCE.txt, read as data with python -I, never run.
  - C4 is web text from April 2019, before chatbot text. FineWeb is the second choice: it may hold chatbot text from
    after 2022.
- **Order:** all documents of the two shards, permuted once by numpy default_rng(2026).
- **Slices, nested by construction:** S1, S10 and S100 are the shortest prefixes of that order with at least 1M, 10M
  and 100M potion tokens. H holds at least 2M tokens from the end of the order, disjoint from S100. H is the TM's own
  self-supervised check, never an evaluation set.
  - The sha256 of each slice's int32 token array goes into w1/corpus/c4_slices.json.
- **Dedupe before any run** (w1/build_corpus_c4.py), at sentence level:
  - A sentence is dropped when its normalized key equals an evaluation text's key (the test 5 rule).
  - It is also dropped when it shares 8 consecutive words with an evaluation text of 8 or more words.
  - The evaluation texts:
    - CLINC150 and clinc_oos_plus, all val, test, oos_val and oos_test splits;
    - Banking77, HWU64, MASSIVE en-US and nb-NO, val and test;
    - TREC-6, MR, SST-2, R52 and R8, test plus the published train, which holds N7's carved val;
    - ABCD and STAR, dev and test, every turn. The decision_layer agent may later use this table.
  - Counts per split go into the manifest. No text is printed.
- **Results:** (filled after the build) tokens, documents and hashes per slice; drop counts.

## 5. Engine: a multi-output TM on the GPU (w1/mo_tm.py)

- **Machine:**
  - The decision-layer CoTM (dl/tm.py), read only and unchanged.
  - One shared pool of N clauses and one integer weight per (clause, output token).
  - Outputs: the tokens with at least 10 occurrences in the slice. Inputs: the 20,000 most frequent tokens.
  - No negated literals and an empty start, as the TM autoencoder.
- **Task:** self-supervised and CBOW-like. An example is one position of the stream.
  - Input: the set of context tokens within a dynamic window, 1 to 5 tokens on each side, inside the document. The
    target's own type is masked.
  - Positive: the target token.
  - Negative: one token drawn from the unigram^0.75 distribution. A draw equal to the target gets no feedback.
  - Not the CoTM's hardest negative, which would push near-synonyms apart.
- **Subsampling:** frequent tokens are subsampled once, before the windows. Word2vec's rule, t = 1e-4, fixed seed.
- **Training:** one CoTM.feedback(X, pos, None, neg, neg != pos) call per minibatch, i.e. dl/tm.py's own batch rule.
- **Held-out check:** 10,000 positions of H. The true token is ranked against 99 unigram^0.75 negatives by class sum,
  scored by mean reciprocal rank.
  - It stops training: 2 checks without a better MRR, and the best state is kept.
  - In step 1 it also chooses the engine settings.
  - It never sees an evaluation set.
- **Readout R3a (primary; test 5's rule, the readout that worked):** row w = clip(W[:, w], 0, T) * [W >= 2],
  L2-normalized.
- **Readout R5 (reported, no bar), if built before the freeze:** the mean clause output over the token's training
  contexts, L2-normalized. It is the TM's own learned features counted around each word.
- **No job lost:** a checkpoint every 20 minutes and at the end (TA states, weights, step), and --resume. Rows go to
  w1/results_w1_train.jsonl and w1/results_w1_pilot.jsonl.
- **Line C (later, the second "wow"):** the same wrapper with left-context features (token at each offset) and the
  next token as the target.
  - Perplexity from a softmax over class sums, with one temperature fitted on H.
  - Baseline: Kneser-Ney on the same tokens. Not part of W1.

## 6. Count twin, built strong (w1/ppmi_svd.py)

- **Registered twin:**
  - Same slice, subsampled stream, output list, context list and window as the TM. Each pair is weighted
    (W - d + 1) / W, which is the expectation of the TM's dynamic window.
  - PPMI with context smoothing 0.75 and no shift.
  - SVD with 500 dimensions and eigenvalue weight 0.5, unit-length rows.
- **Twin grid at S100** (reported, and used by the strong bar):
  - window 2 and 10 (at 500 dimensions, eigenvalue weight 0.5);
  - 300 and 1,000 dimensions (window 5);
  - eigenvalue weight 0 (window 5, 500 dimensions).
  The TM must beat the best of these for the strong claim.
- **Reference, no bar:** SGNS (word2vec skip-gram, a shallow neural network) on the same slices. It shows what a
  neural predictor gains from the same data, so we can tell a weak TM learner from a weak corpus. No SGNS number
  enters a TM claim.

## 7. Controls ladder

All rows go through the same existsT harness (w1/w1_feats.py, w1/w1_exists_tm.py), on val, with classifier seeds 1-3.

1. words only (existing rows);
2. PPMI-SVD at each slice;
3. TM table at each slice;
4. GloVe 6B (existing rows);
5. potion, restricted to the tokens that the S100 TM table covers (new, coverage-matched);
6. potion, full (existing rows).

Before any table row: an equivalence check. Potion's own 63,091 rows through w1_feats.py must reproduce test 2's
cached features.

## 8. Speed pilot (train slice S1 only; timing and memory, no table)

(filled after the pilot) Configurations:
- N = 2,000, 4,000 and 8,000 clauses at batch 1,024;
- batch 256 at N = 4,000;
- 10k inputs at N = 4,000.

Also one short learning check: 3 epochs on S1, with the held-out MRR against chance (about 0.052 for 100
candidates). Nothing is read from val or test.

## 9. GPU budget (shared card)

(filled after the pilot) Card, free memory, the per-process cap (--mem), and the hours per epoch at S1 / S10 / S100.

## 10. Levers after the curve, cheapest first

Each lever gets its own registration or amendment.

1. **Data size.** If the TM's curve rises faster than the twin's, add 300M and 1B points. C4 has 1,024 shards, and
   Sahlgren and Lenci's predictors win only at 1B.
2. **In-domain human-written text, for both learners.** Ranked:
   - Taskmaster-1/2 (CC BY 4.0, human-written, about 2.5M tokens for TM-1);
   - MultiWOZ 2.2 (MIT);
   - SGD (CC BY-SA 4.0; humans paraphrased machine-made outlines; flagged).
   No LLM-written text. 2b, declared: the intent sets' own unlabeled train texts. Potion did not use them.
3. **More clauses:** N from 4,000 up to 8,000-16,000 if the pilot allows. Potion has 512 learned dimensions; the
   paper's 600 clauses are small.
4. **Readouts:** R5 against R3a.
5. **A second TM layer on the TM's own codes:** the context words enter as their layer-1 existsT bits, not as word
   identities. CAIR's hierarchical TM (AND-OR clauses) is the natural engine. W1's flat CoTM is the baseline it must
   beat.

## 11. Registration draft (REGISTRATION_W1.md, val first; frozen after step 1)

- **Tables per slice (S1, S10, S100):**
  - TM: settings fixed in Amendment 1 after step 1. Table seeds 1 and 2 at S1 and S10. At S100 seed 1; seed 2 is run
    if a bar is met with seed 1, and the bar then uses the mean of both tables.
  - PPMI-SVD twin: registered setting.
  - Twin grid at S100.
- **Harness:** test 2 settings throughout.
  - Features: existsT, K 512, 25 k-means iterations, CUDA generator seed 0, thermometer levels of exists_tm.
  - Classifier: CoTM, 4,000 clauses, T 4000, s 8, 40 epochs, batch 128, hardest negative, negation, init half.
  - Statistic: val accuracy at the best val epoch, the protocol of every reference row. Classifier seeds 1-3.
  - Datasets: CLINC150, Banking77, HWU64.
- **Bars** (mean of the classifier seeds, per dataset, met on at least 2 of 3):

| Bar | Meaning | Rule |
|---|---|---|
| B1 count (the scientific claim) | beats counting at 100M | TM_S100 - PP_S100 >= +0.5. Strong: also >= +0.5 against the best twin of the grid |
| B2 slope (the question) | gains more from data | (TM_S100 - TM_S1) - (PP_S100 - PP_S1) >= +1.0 |
| B3 the "wow" | as good as the transformer table | potion - TM_S100 <= 2.0. Strong: <= 1.0 |
| B4 transformer-free level | counting at 6B, matched at 100M | TM_S100 >= GloVe-6B existsT (91.50 / 87.34 / 88.51) |

- **Reported, no bar:** the S10 points and per-decade slopes; the share of val tokens without a vector; the held-out
  MRR; training time; bits; probe neighbours; readable clauses; the twin grid; SGNS; R5.
- **Then:** if B1 or B3 is met, a test registration follows (test read once, same tables and seeds).
- **Declared before any W1 row:** test 5 is negative (above). Wording: "transformer-free" for GloVe, never "no neural
  network". "No neural network at any stage" is said only for the TM table and the PPMI-SVD twin, never for potion.

## 12. Steps

- **Step 0 (now):**
  - this plan;
  - the C4 download and corpus build, with dedupe and hashes;
  - the harness equivalence check;
  - the speed pilot on S1.
  Then a separate check.
- **Step 1:** engine development on S1, with one confirmation on S10.
  - Uses the held-out check only.
  - Grid: T x s (3 x 3), init (empty / half), batch (from the pilot).
  - Fixed in Amendment 1 before any val row.
- **Step 2:** the registered val runs, at most 2 jobs of this line at once, through gpu_run.sh shared.
- **Step 3:** verdict and the study log. A test registration if a bar is met.

