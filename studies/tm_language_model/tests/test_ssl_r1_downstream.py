"""Tests of src/tmlm/ssl/downstream.py: train splits only, the hard refusal of every test split and of SST-2
validation (docs/13 A7), and the mix64 carve (dev residues 1 and 2 mod 20, SST-2 confirm residue 0)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from tmlm.engine import rng as R
from tmlm.ssl import downstream as D

_OPENED: list[str] = []
_ACTIVE: list[bool] = [False]


def _hook(event, args):
    """Records the path of every Python-level file open while active (pyarrow gets Python file objects)."""
    if _ACTIVE[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_hook)


@pytest.mark.parametrize("task,split", [("sst2", "validation"), ("sst2", "test"), ("agnews", "test"), ("trec", "test"),
                                        ("agnews", "validation"), ("trec", "dev"), ("sst2", "Validation")])
def test_open_split_refuses_everything_but_train(monkeypatch, task, split):
    def boom(*a, **k):
        raise AssertionError("no file may be looked up for a refused split")

    monkeypatch.setattr(D, "train_file", boom)
    with pytest.raises(D.ForbiddenSplit):
        D.open_split(task, split)


@pytest.mark.parametrize("task,rel", [("sst2", "sst2/validation-00000-of-00001.parquet"), ("sst2", "sst2/test-00000-of-00001.parquet"),
                                      ("agnews", "data/test-00000-of-00001.parquet"), ("trec", "default/test/0000.parquet")])
def test_path_helpers_refuse_other_files(task, rel):
    with pytest.raises(D.ForbiddenSplit):
        D.local_path(task, rel)
    with pytest.raises(D.ForbiddenSplit):
        D.check_allowed(task, rel)


def test_download_is_refused_unless_allowed(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.train_file("agnews", root=tmp_path)


def test_carve_residues_and_disjointness():
    n = 5000
    r = D.residues(n)
    assert [int(x) for x in r[:50]] == [R.mix64(i) % 20 for i in range(50)]
    for task in ("sst2", "agnews", "trec"):
        c = D.carve(task, n)
        allidx = np.concatenate([c["pool"], c["dev"], c["confirm"]])
        assert np.array_equal(np.sort(allidx), np.arange(n))
        assert np.all(np.isin(r[c["dev"]], [1, 2]))
        if task == "sst2":
            assert np.all(r[c["confirm"]] == 0) and not np.any(r[c["pool"]] == 0)
            assert abs(c["confirm"].shape[0] / n - 0.05) < 0.01
        else:
            assert c["confirm"].size == 0
        assert abs(c["dev"].shape[0] / n - 0.10) < 0.015


def test_loading_all_tasks_opens_only_train_files(monkeypatch):
    for task in D.TASKS:
        try:
            D.train_file(task)
        except FileNotFoundError:
            pytest.skip(f"{task} train file not in the local cache")

    def no_download(*a, **k):
        raise AssertionError("no download may happen when the train file is present")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    val = D._root(None) / D.HF_CACHE_REL / "datasets--nyu-mll--glue" / "snapshots" / D.TASKS["sst2"]["revision"] / "sst2" / "validation-00000-of-00001.parquet"
    _OPENED.clear()
    _ACTIVE[0] = True
    try:
        data = {t: D.load_task(t) for t in ("agnews", "sst2", "trec")}
    finally:
        _ACTIVE[0] = False
    opened_cache = [p for p in _OPENED if "hf_cache" in p]
    assert opened_cache, "the audit hook saw no read of the cache (the proof would be empty)"
    allowed = {str(D.train_file(t)) for t in D.TASKS}
    assert set(opened_cache) <= allowed, set(opened_cache) - allowed
    assert not any("validation" in p or "/test" in p for p in opened_cache)
    if val.exists():
        assert str(val) not in _OPENED
    s = data["sst2"]
    assert s.n_confirm_excluded > 0
    conf = D.carve("sst2", s.n_train)["confirm"]
    assert not np.intersect1d(conf, s.pool_idx).size and not np.intersect1d(conf, s.dev_idx).size
    for t, d in data.items():
        assert len(d.pool_text) == d.pool_y.shape[0] and len(d.dev_text) == d.dev_y.shape[0]
        assert d.pool_y.shape[0] + d.dev_y.shape[0] + d.n_confirm_excluded == d.n_train
        assert set(np.unique(d.dev_y).tolist()) == set(range(d.n_classes))
    assert data["agnews"].n_train == 120000 and data["trec"].n_train == 5452 and data["sst2"].n_train == 67349


def test_verify_against_the_a6_hash_file_until_a7_exists():
    try:
        v = D.verify_train_file("trec")
    except FileNotFoundError:
        pytest.skip("trec train file not in the local cache")
    assert v["equal"] and v["reference_file"].endswith(("A7-trec.json", "A6-trec.json"))
