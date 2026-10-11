"""Leakage rules of docs/06 for the data pipeline: the tokenizer and the cluster
codes are fitted on the train split only. Built from train, the saved artifacts are unchanged
when validation text is added to what the pipeline sees; and no validation or test token feeds
the bigram counts of the clusters."""

from __future__ import annotations

import numpy as np

from tmlm.common import hashing
from tmlm.data import bpe, clusters, corpus_store
from data_helpers import synthetic_rows


def test_tokenizer_depends_on_train_text_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "d"))
    train, _ = corpus_store.rows_to_articles(synthetic_rows(21, n_articles=8))
    valid, _ = corpus_store.rows_to_articles(synthetic_rows(22, n_articles=3, words_per_para=40))
    tok = bpe.train_bpe(train, 300)
    meta = bpe.save_tokenizer(tok, 300, {"split": "train"})
    # the saved hash equals a fresh train-only training, whatever else the pipeline holds in memory
    again = bpe.train_bpe(list(train), 300)
    assert hashing.sha256_bytes(again.to_str().encode()) == meta["sha256"]["tokenizer.json"]
    # the check is meaningful: training on train + validation gives a different tokenizer
    leaky = bpe.train_bpe(train + valid, 300)
    assert leaky.to_str() != tok.to_str()


def test_cluster_counts_use_train_tokens_only():
    train, _ = corpus_store.rows_to_articles(synthetic_rows(23, n_articles=8))
    valid, _ = corpus_store.rows_to_articles(synthetic_rows(24, n_articles=3))
    tok = bpe.train_bpe(train, 300)
    V = tok.get_vocab_size()
    ids_tr, ds_tr = bpe.encode_docs(tok, train)
    ids_va, ds_va = bpe.encode_docs(tok, valid)
    B = clusters.bigram_counts(ids_tr, ds_tr, V)
    # every counted pair is a train pair inside a train document: total = N_tok - N_doc
    assert int(B.sum()) == ids_tr.shape[0] - (ds_tr.shape[0] - 1)
    # brute force: the count of every pair equals the in-document train bigram count
    d_of = np.searchsorted(ds_tr, np.arange(ids_tr.shape[0] - 1), side="right") - 1
    same = d_of == (np.searchsorted(ds_tr, np.arange(1, ids_tr.shape[0]), side="right") - 1)
    ref = np.zeros((V, V), dtype=np.int64)
    np.add.at(ref, (ids_tr[:-1][same].astype(np.int64), ids_tr[1:][same].astype(np.int64)), 1)
    assert np.array_equal(np.asarray(B.todense()), ref)
    # the same assignment falls out of the same train counts; adding validation pairs changes B
    levels_a, _ = clusters.hierarchical_exchange(B, seed=1, max_iter=4)
    levels_b, _ = clusters.hierarchical_exchange(clusters.bigram_counts(ids_tr, ds_tr, V), seed=1, max_iter=4)
    assert all(np.array_equal(x, y) for x, y in zip(levels_a, levels_b))
    both = np.concatenate([ids_tr, ids_va])
    ds_both = np.concatenate([ds_tr, ds_va[1:] + ids_tr.shape[0]]).astype(np.uint32)
    B_leaky = clusters.bigram_counts(both, ds_both, V)
    assert (B_leaky != B).nnz > 0
    km = clusters.kind_map_from_levels(V, levels_a)
    assert km.shape == (4, V) and km.dtype == np.uint16
    assert np.array_equal(km[0], np.arange(V))
