"""Every baseline family runs end to end on a tiny SYNTHETIC corpus store (never the real data:
conftest points TMLM_DATA_DIR at a temporary directory) in well under 30 s each.

The transformer family needs torch, which lives in .venv-dl only: those tests are skipped in
.venv-tm (`make test`) and run with `.venv-dl/bin/python -m pytest tests/test_baselines_smoke.py`.
The BERT reference needs the pretrained weights and is smoke-tested on the real validation set
with `--limit` instead (experiments/m2_baselines.py --family bert --limit N).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.common import hashing
from tmlm.data import corpus_store, evalsets
from tmlm.engine.config import n_sym

ROOT = Path(__file__).resolve().parents[1]
V = 64


def _make_store(root: Path, seed: int = 1, n_train_doc: int = 40, n_val_doc: int = 6) -> None:
    """A synthetic WikiText-103 store at V=64 under `root`: train and validation splits, raw text,
    word ids, both frozen validation eval sets."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, V + 1) ** 1.05
    p /= p.sum()
    # kind_map, stop, sym_freq from "train"
    kind_map = np.tile(np.arange(V, dtype=np.uint16), (4, 1))
    stop = np.zeros(V, dtype=np.uint8)
    stop[:4] = 1
    for split, n_doc in (("train", n_train_doc), ("validation", n_val_doc)):
        lengths = rng.integers(30, 120, size=n_doc)
        tok = rng.choice(V, size=int(lengths.sum()), p=p).astype(np.uint16)
        doc_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
        # words: every token with an even id starts a new word; odd ids continue it (no whitespace-only tokens)
        new_word = (tok % 2 == 0)
        new_word[doc_start[:-1]] = True
        word_id = (np.cumsum(new_word) - 1).astype(np.uint32)
        sym_freq = np.zeros(n_sym(V, 0), dtype=np.uint32)
        sym_freq[:V] = np.bincount(tok, minlength=V)
        arrays = {"tok": tok, "doc_start": doc_start, "word_id": word_id, "kind_map": kind_map, "stop": stop,
                  "sym_freq": sym_freq, "targets": np.arange(tok.shape[0], dtype=np.uint32)}
        hashes = corpus_store.write_split_arrays("wikitext103", V, split, arrays, root)
        # raw text: token t is the word piece "t" + letters; whole words are joined without spaces
        pieces = [f"x{int(t)}" for t in tok]
        texts = []
        for d in range(n_doc):
            s = ""
            for i in range(doc_start[d], doc_start[d + 1]):
                s += (" " if new_word[i] and i > doc_start[d] else "") + pieces[i]
            texts.append(s + "\n")
        meta_raw = corpus_store.write_raw_split("wikitext103", split, texts, [f"doc{d}" for d in range(n_doc)], {"synthetic": True}, root)
        corpus_store.write_split_meta("wikitext103", V, split, {"vocab": V, "split": split, "n_tok": int(tok.shape[0]),
                                                                  "n_bytes_utf8": meta_raw["n_bytes"], "sha256": hashes}, root)
        if split == "validation":
            evalsets.save_evalset("wikitext103", V, split, "subword", evalsets.build_subword(tok.shape[0], V, split), root)
            evalsets.save_evalset("wikitext103", V, split, "wholeword", evalsets.build_wholeword(word_id, V, split), root)
    assert hashing.sha256_file(root / "wikitext103" / "raw" / "validation" / "text.utf8")


@pytest.fixture
def store(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "_data"
    monkeypatch.setenv("TMLM_DATA_DIR", str(root))
    _make_store(root)
    return root


def _run_family(family: str, python: str, extra: list[str], env: dict) -> dict:
    cmd = [python, str(ROOT / "experiments" / "m2_baselines.py"), "--family", family, "--vocab", str(V), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    return json.loads(out[out.index("\n{") + 1:])


def _env(tmp_path, store: Path) -> dict:
    import os

    env = dict(os.environ)
    env["TMLM_DATA_DIR"] = str(store)
    env["TMLM_SCRATCH"] = str(tmp_path / "scratch")
    env.pop("TMLM_RUN_DIR", None)
    env["OMP_NUM_THREADS"] = "1"
    return env


def test_ngram_family(store, tmp_path):
    m = _run_family("ngram", sys.executable, ["--order", "3"], _env(tmp_path, store))
    for k in ("kn3.validation.bits_per_byte", "kn1.validation.bits_per_byte", "kn3_both.validation_subword.top1_visible",
              "kn3_left.validation_wholeword.word_acc_visible", "left_bigram.validation_subword.top1_truectx",
              "kn3_single_D0.1.validation.bits_per_byte"):
        assert k in m, k
        assert m[k]["label"] == "measured"
    assert 0 < m["kn3.validation.bits_per_byte"]["value"] < 20
    assert m["kn3.validation.bits_per_byte"]["value"] <= m["kn1.validation.bits_per_byte"]["value"] + 1.0


def test_count_probe_family(store, tmp_path):
    m = _run_family("count_probe", sys.executable, [], _env(tmp_path, store))
    assert 0 <= m["count_probe_masked.validation_subword.top1_visible"]["value"] <= 1
    assert "count_probe_causal.validation.causal_top1" in m


def test_cbow_knn_family(store, tmp_path):
    m = _run_family("cbow_knn", sys.executable, ["--dim", "16", "--batch", "256", "--bank-size", "2000", "--knn-k", "8"],
                    _env(tmp_path, store))
    assert m["cbow.train.steps"]["value"] > 0
    assert 0 <= m["cbow_knn.validation_subword.top1_visible"]["value"] <= 1
    assert 0 <= m["cbow_softmax.validation_wholeword.word_acc_visible"]["value"] <= 1
    assert (tmp_path / "scratch" / "m2_baselines_unlogged" / "cbow_w_in.npy").exists()


def test_cbow_direct_api():
    from tmlm.eval.baselines.cbow_knn import CbowModel, context_index

    rng = np.random.default_rng(0)
    tok = rng.integers(0, 20, size=500).astype(np.uint16)
    ds = np.array([0, 100, 500], dtype=np.uint32)
    idx, mask = context_index(ds, np.array([0, 99, 100, 499]), 2, 500)
    assert mask.tolist() == [[False, False, True, True], [True, True, False, False], [False, False, True, True], [True, True, False, False]]
    m = CbowModel(20, dim=8, batch=64, epochs=2, seed=3)
    st = m.train(tok, ds)
    assert st["steps"] > 0 and np.isfinite(m.w_in).all() and np.isfinite(m.w_out).all()
    bank = m.build_bank(tok, ds, 100, seed=1)
    assert m.predict_knn(tok, ds, np.array([5, 50]), bank, k=5).shape == (2,)
    assert m.predict_softmax(tok, ds, np.array([5, 50])).shape == (2,)


def test_transformer_family(store, tmp_path):
    pytest.importorskip("torch")
    m = _run_family("transformer", sys.executable, ["--layers", "1", "--widths", "16", "--seq-len", "32", "--batch-seqs", "8",
                                                      "--time-cap-s", "3", "--warmup", "2"], _env(tmp_path, store))
    assert m["transformer.best_config"]["value"] == "transformer_L1_W16"
    assert 0 < m["transformer_L1_W16.validation_bits_per_byte"]["value"] < 30
    assert m["transformer_L1_W16.steps"]["value"] > 0


def test_transformer_masked_smoke(store, tmp_path):
    pytest.importorskip("torch")
    m = _run_family("transformer_mlm_smoke", sys.executable, ["--layers", "1", "--widths", "16", "--seq-len", "32", "--batch-seqs", "8",
                                                                "--time-cap-s", "3", "--warmup", "2"], _env(tmp_path, store))
    assert m["transformer_mlm_L1_W16.steps"]["value"] > 0
    assert 0 <= m["transformer_mlm_L1_W16.validation_subword.top1_visible"]["value"] <= 1


def test_transformer_units():
    torch = pytest.importorskip("torch")
    from tmlm.eval.baselines.transformer_small import SmallTransformer, attention_mask, pack_stream

    packed, doc = pack_stream(np.array([1, 2, 3, 4, 5], dtype=np.uint16), np.array([0, 2, 5], dtype=np.uint32), bos=9)
    assert packed.tolist() == [9, 1, 2, 9, 3, 4, 5] and doc.tolist() == [0, 0, 0, 1, 1, 1, 1]
    m = attention_mask(torch.tensor([[0, 0, 1, 1]]), causal=True)[0, 0]
    assert m.tolist() == [[True, False, False, False], [True, True, False, False], [False, False, True, False], [False, False, True, True]]
    model = SmallTransformer(9, layers=1, width=8, heads=2, seq_len=4)
    out = model(torch.tensor([[9, 1, 2, 9]]), torch.tensor([[0, 0, 0, 1]]))
    assert out.shape == (1, 4, 9)
