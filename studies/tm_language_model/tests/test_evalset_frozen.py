"""Frozen evaluation mask sets (docs/06, docs/13 section 4): the same seed gives
the same arrays and hash, different splits or vocabularies differ, the whole-word variant never
hides part of a word, and a saved set round-trips with its hash checked."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.common import hashing
from tmlm.data import evalsets
from tmlm.data.corpus_store import NO_WORD


def toy_word_ids(n_tok: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wid = np.zeros(n_tok, dtype=np.uint32)
    w = -1
    for i in range(n_tok):
        r = rng.random()
        if r < 0.1:
            wid[i] = NO_WORD  # whitespace-only token
            continue
        if r < 0.6 or w < 0:
            w += 1
        wid[i] = w
    return wid


def test_subword_set_is_frozen_by_seed():
    a = evalsets.build_subword(5000, 8000, "validation")
    b = evalsets.build_subword(5000, 8000, "validation")
    assert hashing.sha256_array(a["hidden"]) == hashing.sha256_array(b["hidden"])
    assert np.array_equal(a["targets"], b["targets"])
    assert np.array_equal(a["targets"], np.flatnonzero(a["hidden"]))
    assert abs(a["hidden"].mean() - 0.15) < 0.03
    c = evalsets.build_subword(5000, 8000, "test")
    d = evalsets.build_subword(5000, 16000, "validation")
    assert not np.array_equal(a["hidden"], c["hidden"]) and not np.array_equal(a["hidden"], d["hidden"])


def test_subword_target_cap_is_a_fixed_subset():
    a = evalsets.build_subword(20000, 2000, "validation", max_targets=100)
    b = evalsets.build_subword(20000, 2000, "validation", max_targets=100)
    assert a["targets"].shape == (100,) and np.array_equal(a["targets"], b["targets"])
    assert a["rule"].startswith("fixed random subset of 100")
    assert np.all(a["hidden"][a["targets"]] == 1) and np.all(np.diff(a["targets"]) > 0)
    full = evalsets.build_subword(20000, 2000, "validation")
    assert full["rule"] == "all hidden positions" and np.array_equal(full["hidden"], a["hidden"])


def test_wholeword_hides_whole_words_only():
    wid = toy_word_ids(6000, 3)
    a = evalsets.build_wholeword(wid, 8000, "test")
    b = evalsets.build_wholeword(wid, 8000, "test")
    assert hashing.sha256_array(a["hidden"]) == hashing.sha256_array(b["hidden"])
    hidden = a["hidden"].astype(bool)
    assert not hidden[wid == NO_WORD].any()
    content = wid != NO_WORD
    for w in np.unique(wid[content]):
        pieces = hidden[wid == w]
        assert pieces.all() or not pieces.any(), f"word {w} is partially hidden"
    hidden_words = set(a["hidden_words"].tolist())
    assert set(np.unique(wid[hidden]).tolist()) == hidden_words
    assert np.array_equal(a["targets"], np.flatnonzero(hidden))
    assert abs(len(hidden_words) / a["n_words"] - 0.15) < 0.03


def test_save_load_round_trip_checks_hashes_and_counts_test_access(tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "d"))
    wid = toy_word_ids(3000, 4)
    built = evalsets.build_wholeword(wid, 2000, "test")
    meta = evalsets.save_evalset("toy", 2000, "test", "wholeword", built)
    assert set(meta["sha256"]) == {"hidden", "targets", "word_id", "hidden_words"}
    with pytest.raises(FileExistsError):
        evalsets.save_evalset("toy", 2000, "test", "wholeword", built)
    with pytest.raises(ValueError):
        evalsets.load_evalset("toy", 2000, "test", "wholeword")
    loaded = evalsets.load_evalset("toy", 2000, "test", "wholeword", purpose="unit test")
    assert evalsets.test_access_count() == 1
    assert np.array_equal(loaded["hidden"], built["hidden"]) and loaded["meta"]["sha256_combined"] == meta["sha256_combined"]
    # a tampered array is refused
    path = evalsets.evalset_path("toy", 2000, "test", "wholeword")
    arrays = dict(np.load(path))
    arrays["hidden"][0] ^= 1
    path.unlink()
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        evalsets.load_evalset("toy", 2000, "test", "wholeword", purpose="unit test")


def test_access_counter_needs_a_purpose(tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "d"))
    with pytest.raises(ValueError):
        evalsets.record_test_access("x", "")
    assert evalsets.record_test_access("x", "first") == 1
    assert evalsets.record_test_access("y", "second") == 2
    assert evalsets.access_log_path() == tmp_path / "d" / "test_access_log.jsonl"
