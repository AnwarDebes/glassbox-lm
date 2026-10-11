"""Corpus, tokenizer, cluster codes, frozen eval sets, paragraph pairs, contamination (M2).

Normative text: docs/05 E3, s1, s2, H1; docs/06; docs/11 section 1.1; docs/13 section 4.

Modules (2026-09-22):
- corpus_store: raw article store and the token store (tok/doc_start/word_id/kind_map/stop/
  sym_freq/targets as .npy, hashed, create-only); loaders of a test split count the access
- bpe: byte-level BPE at 2k/8k/16k/32k, train split only, deterministic; word ids
- clusters: 3-level 16-way exchange clustering on train bigram contexts (kind_map rows 1 to 3)
- evalsets: frozen masked eval sets (subword and whole-word), hashed; the test-set access counter
- pairs: the slide-11 paragraph-pair task (E6 tier 1, H1, H2), inside one split, frozen, hashed
- contamination: 8-gram and 13-gram validation/test vs train overlap, de-overlapped subsets
- synthetic: synthetic corpora for engine tests and the M3 gates
The pipeline that builds everything with a log is experiments/m2_build_data.py.
"""
