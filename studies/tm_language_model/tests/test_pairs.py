"""The paragraph-pair task (docs/05 E6 tier 1, H1, H2; docs/13 section 4): no pair
crosses splits, pairs are balanced and distinct, positives are two paragraphs of one article and
negatives are paragraphs of two articles, the build is deterministic."""

from __future__ import annotations

import numpy as np

from tmlm.data import bpe, corpus_store, pairs
from data_helpers import synthetic_rows


def _paragraphs(seed: int, n_articles: int, n_paras: int, tok):
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(seed, n_articles=n_articles, n_paras=n_paras))
    return texts, pairs.paragraphs_of_split(texts, tok)


def test_blocks_drop_headings_and_short_blocks():
    text = "\n = Title = \n\n para one is here \n\n = = Section = = \n\n para two \n line continued \n\n"
    blocks = pairs.blocks_of_article(text)
    assert blocks == [" para one is here \n", " para two \n line continued \n"]
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(31, n_articles=3, n_paras=3))
    tok = bpe.train_bpe(texts, 300)
    paras = pairs.paragraphs_of_split(texts, tok, min_tokens=1)
    assert [len(p) for p in paras] == [3, 3, 3]
    assert all(np.asarray(p).dtype == np.uint16 for a in paras for p in a)
    long_only = pairs.paragraphs_of_split(texts, tok, min_tokens=10_000)
    assert all(len(p) == 0 for p in long_only)


def test_pairs_are_inside_one_split_balanced_distinct_and_deterministic():
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(32, n_articles=10, n_paras=5))
    tok = bpe.train_bpe(texts, 300)
    _, paras = _paragraphs(32, 10, 5, tok)
    built = pairs.build_pairs(paras, 60, "validation")
    again = pairs.build_pairs(paras, 60, "validation")
    n = built["label"].shape[0]
    assert n == 60 and built["n_pos"] == 30 and built["n_neg"] == 30 and built["rule"] == "sampled"
    for key in ("a_doc", "b_doc", "a_para", "b_para", "label", "a_tok", "b_tok", "a_ptr", "b_ptr"):
        assert np.array_equal(built[key], again[key]), key
    other = pairs.build_pairs(paras, 60, "test")
    assert not np.array_equal(other["a_doc"], built["a_doc"])
    # inside the split: every article id is an article of this split
    assert built["a_doc"].max() < len(paras) and built["b_doc"].max() < len(paras)
    pos = built["label"] == 1
    assert np.all(built["a_doc"][pos] == built["b_doc"][pos]) and np.all(built["a_para"][pos] != built["b_para"][pos])
    assert np.all(built["a_doc"][~pos] != built["b_doc"][~pos])
    keys = set()
    for a_d, a_p, b_d, b_p in zip(built["a_doc"], built["a_para"], built["b_doc"], built["b_para"]):
        key = tuple(sorted([(int(a_d), int(a_p)), (int(b_d), int(b_p))]))
        assert key not in keys
        keys.add(key)
    # the token arrays are the paragraphs they name
    for k in range(n):
        a = built["a_tok"][built["a_ptr"][k]:built["a_ptr"][k + 1]]
        assert np.array_equal(a, paras[built["a_doc"][k]][built["a_para"][k]])
        assert a.shape[0] >= pairs.MIN_TOKENS


def test_few_positives_are_enumerated_and_balance_kept():
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(33, n_articles=3, n_paras=2))
    tok = bpe.train_bpe(texts, 300)
    paras = pairs.paragraphs_of_split(texts, tok)
    built = pairs.build_pairs(paras, 1000, "train")
    assert built["n_pos_possible"] == 3 and built["n_pos"] == 3 and built["n_neg"] == 3
    assert built["rule"].startswith("all 3 distinct positive pairs enumerated")


def test_save_and_load_with_access_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "d"))
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(34, n_articles=4, n_paras=3))
    tok = bpe.train_bpe(texts, 300)
    paras = pairs.paragraphs_of_split(texts, tok)
    built = pairs.build_pairs(paras, 8, "test")
    meta = pairs.save_pairs("toy", 300, "test", built)
    loaded = pairs.load_pairs("toy", 300, "test", purpose="unit test")
    assert np.array_equal(loaded["label"], built["label"]) and loaded["meta"]["sha256_combined"] == meta["sha256_combined"]
    from tmlm.data.evalsets import test_access_count
    assert test_access_count() == 1
