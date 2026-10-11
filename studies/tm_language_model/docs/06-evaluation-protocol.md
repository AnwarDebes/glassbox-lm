# 06 Evaluation protocol

## Primary metrics
Masked-token top-1 and top-5 on WikiText-103 and PTB, on FROZEN eval mask sets with recorded hashes, with subword and whole-word targets reported separately. Causal perplexity as bits per byte (protocol CP, docs/05). ELBO-perplexity reported only as an upper bound (MDLM/RADD bounds), never as a like-for-like AR perplexity [verified in paper: MDLM, RADD]. Scaling curves: accuracy and cost vs clause count (s1), vs context length (s2), vs batch size (s3), and per deviation (s4).

## Causal perplexity (summary of protocol CP)
- Report bits per byte (primary) AND per-word perplexity (secondary). The per-word figure is a conversion from the same log-likelihood and is labelled "open-vocabulary, converted, not like-for-like with published closed-vocabulary numbers".
- The in-house interpolated Kneser-Ney 5-gram on identical tokens is the ONLY like-for-like comparator. It is validated against NLTK's `nltk.lm.KneserNeyInterpolated` on a toy corpus, with the caveat that NLTK's implementation is single-discount interpolated KN (default discount 0.1, set explicitly in the test) and pure Python, so it serves as a correctness reference on a small slice only; the in-house model implements modified KN with the three standard discounts and reports which variant every number uses [verified in source: NLTK docs, docs/12 F12]. KenLM is not available here (no cmake).
- No perplexity from the literature is used as a comparator. In particular the figure "KN about 150" (flagged in the design decisions) is unsupported and is never quoted (docs/01).

## Baselines (all measured in-house on identical tokenization; the initial design's list is extended)
- unigram;
- left bigram and right bigram;
- interpolated Kneser-Ney up to order 5 (validated against NLTK on a toy corpus);
- bidirectional fill-in-the-blank n-gram: a candidate t at a hidden position scores log P_fwd(t | left context) + log P_bwd(t | right context), the forward KN model times a second KN model fitted on the reversed documents (a documented product-of-experts approximation, argmax only; the exact joint P(left, t, right) under one model is not computed: `src/tmlm/eval/baselines/ngram.py`). Hidden context tokens cut the context at the nearest hidden token ("visible" variant); the "true context" variant uses the true tokens as the M3 bigram gate does;
- in-house CBOW-kNN: the clean stand-in for the initial design's kNN-over-GloVe (pretrained word vectors do not align with BPE targets and are contaminated);
- a small transformer MLM trained from scratch on the same tokens, capped at 24 GPU-h: the contamination-free neural comparator (the initial design's "small BERT");
- BERT-base MLM on whole-word targets, labelled "contaminated reference" (it saw Wikipedia);
- `count_probe`: what plain counting achieves in Track B's literal space (diagnostic, M3);
- the modified KN-5 itself, scored on the frozen masked sets with only the VISIBLE context on each side (docs/06 bidirectional scorer): reported in every masked-token table next to the bigram, because it is far stronger than the pre-registered Gate 1 comparators (validation at V=8000: 0.442 vs left bigram 0.169 [measured: `results/20260922T180823Z_M2_baselines_ngram_v8000/`]); Gate 1 keeps its pre-registered comparators, and the KN-5 column is there so no reader can miss the gap;
- the Track A GraphTM row (one time-boxed run).
For transfer: vanilla TM R8 96.16, TM-GloVe 97.50, Clusters 97.92 (teacher-dependent) [stated], with the from-scratch TM re-measured in-house. For negation: BERT-base. For energy: the small transformer and BERT-base on the same GPU, same positions.

Numbers that do not exist in the literature and must be measured here, never cited: BERT-base MLM top-1 on WikiText-103, KN-5 perplexity on WikiText-103, bidirectional n-gram top-1, BERT joules per token on a V100.

## Equal-cost comparison protocol (TB, pillar P1)
- The neural comparator for the causal likelihood is a from-scratch causal transformer trained on the same BPE tokens, the same document boundaries and scored on the same test bytes, under two budgets measured on the same V100 under the exclusive lock: equal wall-clock to the reported causal TM run, and equal joules (NVML, idle subtracted). Both directions are reported; no threshold (docs/13, H1.3).
- The 24 GPU-h masked transformer stays as the neural masked-LM comparator.

## The proposal's task (H1, H2; E6 tier 1)
- Pairs of paragraphs from WikiText-103, balanced, built inside one split only (no test document in any training pair), frozen and hashed (docs/13 section 4).
- Rows: the hierarchical package with the slide-11 structure; a flat coalesced TM on the same bag-of-words; a logistic regression on the same bag; engine layer 1 bits read by a logistic regression; engine layer 1 bits read by the engine in tabular mode (layer 2); a second greedy layer. 5 seeds per row, paired one-sided Wilcoxon for the pre-registered comparisons (docs/13, H4.1 to H4.3).

## Contamination rules
- `contamination.py` reports the 8-gram and 13-gram overlap between test and train; results are reported on the full test set AND on the de-overlapped subset.
- BERT-base is labelled "saw Wikipedia" wherever it appears. The from-scratch transformer is the clean neural comparator.
- Subword-target and whole-word-target results are never mixed in one column.
- Tokenizer and cluster codes are fitted on the train split only, verified by hash (`test_leakage.py`).
- Validation data for ALL tuning. The test split is touched once per reported configuration, and every test-set access is counted by the access counter built in M2; the count is reported.
- Eval mask sets are frozen and hashed (`test_evalset_frozen.py`).

## Leakage rules
- LEAKAGE RULE: parser-derived features (spaCy negation scope, scope adverbs, dependency edges, discourse connectives) are an ablation computed from MASKED text only. Token-only features come first.
- Metamorphic tests (`test_leakage.py`): replacing the target token leaves the scores bit-identical, polarity features included; in the causal configuration changing any position >= p changes nothing; bag literals exclude p and never cross documents.
- Generation: prompts are PRE-REGISTERED (first K test documents, seeds 0..k); every sample is logged; no human-quality claims.

## Reporting conventions
- Every reported number cites a log path under results/.
- Every number labelled measured/derived/unverified.
- Timings and throughput: repeats plus variance (the host is shared and the CPU quota is 6); timing, energy, and profiling runs hold the exclusive GPU lock.
- Seed variance: the seed lists of docs/13 (3 seeds at small scaling sizes; 5 seeds for the tiny-LM gates and the proposal table, where a paired test is used); single large runs are labelled single-seed. Every run.json carries `prereg_sha256`.
- For BLiMP, report aggregate and the per-phenomenon table (pseudo-log-likelihood scoring); state the human 96.4% reference and that GPT-2 is 8 points below, without claiming a human-parity result [verified in paper].
- For CondaQA, run once; report the consistency metric and cite human 81.58% and best-prior-model 42.18% [verified in paper].
- Negation: report sensitivity AND correctness.
- Diffusion likelihood is an ELBO upper bound; label it as such.
- Confidence: report ECE; target below 5% after calibration (two-scalar fit; split-conformal sets as the ConfTM stand-in).
- Energy: GPU joules measured through NVML with idle power subtracted and reported, batch-1 and max-throughput regimes; CPU joules [derived] unless RAPL is readable.
- Negative results and triggered kill criteria are reported as such, with their log paths.

## Ablations required
Binomial draw on/off (Track A); compositional head vs flat vs label tree; absorbing threshold band (50/75/100); literal sampling rate (Track B insertion probability); depth 1 vs 2 vs 3 (Track A only, Track B has no depth); polarity symbols on/off (for negation, from masked text only); batch size B (s3); the deviation ladder D0..D7 (s4); bag buckets on/off and context length (s2); wide-vote tier on/off; pointer clauses on/off.
