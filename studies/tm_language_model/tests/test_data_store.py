"""Round trip of tiny synthetic text through the real BPE, store, cluster and pipeline code
(docs/04 M2 ACCEPT: every artifact has a hash, a test-set access counter exists).
Never touches data/ (conftest redirects TMLM_DATA_DIR) and never the real corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.data import bpe, clusters, contamination, corpus_store, evalsets
from tmlm.data.synthetic import stop_of, sym_freq_of
from tmlm.engine.config import EngineConfig
from data_helpers import synthetic_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("TMLM_DATA_DIR", str(root))
    return root


def test_rows_to_articles_keeps_every_byte_and_cuts_at_top_headings():
    rows = synthetic_rows(1, n_articles=3)
    texts, titles = corpus_store.rows_to_articles(rows)
    assert titles == ["Article 1-0", "Article 1-1", "Article 1-2"]
    assert "".join(texts) == "".join(r if r else "\n" for r in rows)
    assert texts[0].startswith("\n = Article 1-0 = \n\n")  # the leading empty row stays with article 0
    assert corpus_store.is_top_heading(" = Title = \n")
    assert not corpus_store.is_top_heading(" = = Section = = \n")
    # an equation fragment that looks like a heading but is not surrounded by empty rows
    rows2 = rows[:5] + [" impedance ω \n", " = 1 rad / s and k = \n", " 1 Ω . text \n", ""] + rows[5:]
    texts2, titles2 = corpus_store.rows_to_articles(rows2)
    assert titles2 == titles and "".join(texts2) == "".join(r if r else "\n" for r in rows2)


def test_raw_store_round_trip_and_hashes(data_root):
    texts, titles = corpus_store.rows_to_articles(synthetic_rows(2))
    meta = corpus_store.write_raw_split("toy", "train", texts, titles, {"n_rows": 0})
    raw = corpus_store.load_raw_split("toy", "train")
    assert raw.articles() == texts and raw.titles == titles
    assert len(raw.text) == sum(len(t.encode("utf-8")) for t in texts) == meta["n_bytes"]
    assert set(meta["sha256"]) == {"text.utf8", "doc_bytes.npy", "titles.json"}
    with pytest.raises(FileExistsError):
        corpus_store.write_raw_split("toy", "train", texts, titles, {"n_rows": 0})


def test_bpe_is_deterministic_and_word_ids_match_offsets(data_root):
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(3))
    tok_a = bpe.train_bpe(texts, 300)
    tok_b = bpe.train_bpe(texts, 300)
    assert tok_a.to_str() == tok_b.to_str()
    assert tok_a.get_vocab_size() == 300
    meta = bpe.save_tokenizer(tok_a, 300, {"split": "train"})
    tok_c = bpe.load_tokenizer(300)
    assert tok_c.to_str() == tok_a.to_str()
    assert bpe.tokenizer_sha256(300) == meta["sha256"]["tokenizer.json"]
    ws_lead, all_ws = bpe.token_whitespace_tables(tok_a)
    ids, doc_start = bpe.encode_docs(tok_a, texts)
    assert ids.dtype == np.uint16 and doc_start.dtype == np.uint32 and doc_start[-1] == ids.shape[0]
    wid = bpe.word_ids(ids, doc_start, ws_lead, all_ws)
    # reference from character offsets, document by document
    base = 0
    for d, text in enumerate(texts):
        enc = tok_a.encode(text)
        ref = bpe.word_ids_from_offsets(text, enc.offsets)
        got = wid[doc_start[d]:doc_start[d + 1]]
        assert list(enc.ids) == ids[doc_start[d]:doc_start[d + 1]].tolist()
        ref_norm = [corpus_store.NO_WORD if r == corpus_store.NO_WORD else r + base for r in ref]
        assert got.tolist() == ref_norm, f"document {d}"
        base += len(set(r for r in ref if r != corpus_store.NO_WORD))
    n_words_text = sum(len(t.split()) for t in texts)
    assert int(wid[wid != corpus_store.NO_WORD].max()) + 1 == n_words_text
    assert tok_a.decode(ids[doc_start[0]:doc_start[1]].tolist()) == texts[0]


def test_store_round_trip_validates_as_corpus_arrays_and_counts_test_access(data_root):
    texts, _ = corpus_store.rows_to_articles(synthetic_rows(4))
    tok = bpe.train_bpe(texts, 300)
    V = tok.get_vocab_size()
    ws_lead, all_ws = bpe.token_whitespace_tables(tok)
    ids, doc_start = bpe.encode_docs(tok, texts)
    wid = bpe.word_ids(ids, doc_start, ws_lead, all_ws)
    B = clusters.bigram_counts(ids, doc_start, V)
    assert B.sum() == ids.shape[0] - (doc_start.shape[0] - 1)
    levels, logs = clusters.hierarchical_exchange(B, seed=1, max_iter=5)
    assert clusters.check_refinement(levels)
    for lg in logs:
        assert all(np.diff([lg.initial_objective] + lg.objective) >= -1e-6)
    km = clusters.kind_map_from_levels(V, levels)
    stop = stop_of(ids, V, 4)
    sym_freq = sym_freq_of(ids, km, V)
    for split in ("train", "test"):
        hashes = corpus_store.write_split_arrays("toy", V, split, {
            "tok": ids, "doc_start": doc_start, "word_id": wid, "kind_map": km, "stop": stop,
            "sym_freq": sym_freq, "targets": np.arange(ids.shape[0], dtype=np.uint32)})
        assert set(hashes) == {f"{n}.npy" for n in corpus_store.STORE_ARRAYS}
        corpus_store.write_split_meta("toy", V, split, {"sha256": hashes})
    cfg = EngineConfig(mode="masked", vocab=V, n_clauses=16, seed=1)
    corpus = corpus_store.load_split("toy", V, "train", check=True)
    corpus.validate(cfg)
    assert np.array_equal(corpus.tok, ids)
    assert corpus.sym_freq[-1] == 0xFFFFFFFF and corpus.sym_freq[-2] == 0xFFFFFFFF
    # the test split needs a purpose and every load is counted
    with pytest.raises(ValueError):
        corpus_store.load_split("toy", V, "test")
    assert evalsets.test_access_count() == 0
    corpus_store.load_split("toy", V, "test", purpose="unit test")
    corpus_store.load_split("toy", V, "test", purpose="unit test again")
    assert evalsets.test_access_count() == 2
    lines = [json.loads(line) for line in evalsets.access_log_path().read_text().splitlines()]
    assert lines[1]["purpose"] == "unit test again" and lines[1]["caller"] == "corpus_store.load_split"
    with pytest.raises(FileExistsError):
        corpus_store.write_split_arrays("toy", V, "train", {"tok": ids})


def test_contamination_counts_exact_overlap():
    rng = np.random.default_rng(5)
    train = rng.integers(0, 50, size=2000).astype(np.uint16)
    train_ds = np.array([0, 700, 2000], dtype=np.uint32)
    ts = contamination.train_ngram_set(train, train_ds, 8)
    assert ts.shape[0] <= 2000 - 7 * 2
    # a split that copies a train slice is fully seen; fresh random text is (almost surely) not
    ev = np.concatenate([train[100:300], rng.integers(0, 50, size=200).astype(np.uint16)])
    ev_ds = np.array([0, 200, 400], dtype=np.uint32)
    res = contamination.overlap(ts, ev, ev_ds, 8)
    assert res["n_ngrams"] == 2 * (200 - 7)
    seen = res["seen"]
    assert seen[:193].all() and seen[193:].sum() <= 2
    assert res["doc_keep"].tolist() == [0, 1]
    assert res["token_covered"][:200].all()
    # an n-gram crossing a document boundary is never formed
    starts = contamination.ngram_starts(np.array([0, 5, 20], dtype=np.uint32), 8)
    assert starts.tolist() == list(range(5, 13))


@pytest.mark.slow
def test_pipeline_script_runs_end_to_end_on_synthetic_rows(tmp_path, data_root):
    rows = {"train": synthetic_rows(11, n_articles=12, n_paras=6),
            "validation": synthetic_rows(12, n_articles=4, n_paras=4),
            "test": synthetic_rows(13, n_articles=4, n_paras=4)}
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows))
    cfg = json.loads((ROOT / "experiments" / "configs" / "M2_data.json").read_text())
    cfg["params"].update({"vocab_sizes": [300], "pairs_vocab": 300, "contamination_vocab": 300,
                          "pair_sizes": {"train": 40, "validation": 10, "test": 10}, "cluster_max_iter": 3})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    prereg = tmp_path / "prereg.json"
    cmd = [sys.executable, str(ROOT / "experiments" / "m2_build_data.py"), "--config", str(cfg_path),
           "--data-root", str(data_root), "--prereg-file", str(prereg), "--skip-ptb",
           "--synthetic-rows", str(rows_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    hashes = json.loads(prereg.read_text())["dataset_hashes"]
    assert "tokenizers/bpe_300/tokenizer.json" in hashes
    assert "wikitext103/bpe300/train/tok.npy" in hashes
    assert "wikitext103/bpe300/evalsets/test_wholeword.npz" in hashes
    assert "wikitext103/pairs_bpe300/train.npz" in hashes
    assert "wikitext103/contamination_bpe300/test_13gram.npz" in hashes
    assert all(len(h) == 64 for h in hashes.values())
    # the store loads as CorpusArrays; a second run is a no-op that must not overwrite anything
    corpus = corpus_store.load_split("wikitext103", 300, "validation", check=True)
    corpus.validate(EngineConfig(mode="masked", vocab=300, n_clauses=16, seed=1))
    before = sorted(p.stat().st_mtime_ns for p in data_root.rglob("*.npy"))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NOT overwritten" in proc.stdout
    assert sorted(p.stat().st_mtime_ns for p in data_root.rglob("*.npy")) == before
    assert evalsets.test_access_count() >= 5
