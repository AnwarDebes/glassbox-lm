"""In-house baselines on identical tokens (M2). Normative text: docs/06 'Baselines'.

Modules (numpy only unless stated; the experiment driver is experiments/m2_baselines.py):
- ngram: unigram, left and right bigram argmax; interpolated modified Kneser-Ney up to order 5
  (`KNModel`, three discounts, single-discount mode validated against NLTK in
  tests/test_kn_vs_nltk.py), per-token log2 probabilities for bits per byte (protocol CP),
  causal argmax, and the bidirectional fill-in (`BidirectionalKN`, a documented product of the
  forward and backward models).
- count_probe: what plain counting achieves in the position-bound literal space (docs/08 KB2);
  the experiments import it from here.
- cbow_knn: word2vec CBOW with negative sampling, then k-NN over training context vectors.
- transformer_small (.venv-dl): the from-scratch causal transformer of TB with its size sweep,
  bits per byte, NVML joules, and the masked (BERT-style) variant.
- bert_reference (.venv-dl): BERT-base whole-word accuracy on strings, always labelled
  "contaminated reference (saw Wikipedia)".
- hierarchical_tm (H1 rows): planned, written in M9.
"""
