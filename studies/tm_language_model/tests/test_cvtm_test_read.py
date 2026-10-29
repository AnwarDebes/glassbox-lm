"""experiments/cvtm_test_read.py (the H8.3 and H8.4 test read), on synthetic data only.

Helpers: the rows equal the model's own scoring, the de-overlap flags equal tmlm.data.contamination, the
de-overlapped rows are the kept documents' bits over their bytes, the verdict rules of docs/13 lines 178
and 179, the exact reproduction checks, the AST code hash, every refusal condition by name.

End to end (main with --split test) on a tiny synthetic store under the temporary TMLM_DATA_DIR of
tests/conftest.py, with a synthetic access log, a toy core and a stand-in docs/13: every refusal leaves the
log byte for byte unchanged and exits 3; an accepted read appends exactly one row (this purpose, this
process) and writes both verdicts, the rows and the score arrays; a forced KN-5 mismatch makes both
verdicts invalid; a forced de-overlap mismatch keeps the verdicts valid and labels the de-overlapped rows
unverified. No real data and no real test split are read.

    .venv-tm/bin/python -m pytest -q tests/test_cvtm_test_read.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tmlm.common import hashing, runlog
from tmlm.cvtm import protocol
from tmlm.cvtm.model import CVTMCore
from tmlm.data import contamination, corpus_store, evalsets
from tmlm.eval.baselines.ngram import bits_per_byte

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import cvtm_test_read as tr  # noqa: E402
import cvtm_test_read_verify as vf  # noqa: E402

V = 40
CONFIG = EXPERIMENTS / "cvtm" / "configs" / "cvtm_a3_H8_3_H8_4_test.json"
HEADING = "### A12 (2026-10-26 10:00 UTC): the CVTM-core test read of H8.3 and H8.4\n"


def corpus(seed: int, n_docs: int, lo: int = 30, hi: int = 200):
    """Markov documents with copied spans (document clauses and the pointer fire), as test_cvtm_model."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(rng.dirichlet(np.full(V, 0.3), size=V), axis=1)
    docs = []
    for _ in range(n_docs):
        n = int(rng.integers(lo, hi))
        t = [int(rng.integers(V))]
        for x in rng.random(n - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), V - 1))
        if n > 40 and rng.random() < 0.6:
            a, L = int(rng.integers(0, n // 2 - 12)), int(rng.integers(6, 12))
            b = int(rng.integers(n // 2, n - L))
            t[b:b + L] = t[a:a + L]
        docs.append(t)
    return docs


def stream(docs):
    tok = np.asarray([x for d in docs for x in d], dtype=np.uint16)
    ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
    return tok, ds


def eval_split(docs, split="validation"):
    tok, ds = stream(docs)
    doc_bytes = np.diff(ds) * 3 + 1
    return protocol.EvalSplit(split=split, tok=tok, doc_start=ds, doc_bytes=doc_bytes, n_bytes=int(doc_bytes.sum()),
                              n_words=0, meta={}, raw_meta={})


@pytest.fixture(scope="module")
def world():
    train_docs = corpus(1, 700)
    tok, ds = stream(train_docs)
    core = CVTMCore.fit(tok, ds, vocab=V, backend="numpy", expect_h=False)
    val_docs = corpus(2, 12)
    test_docs = corpus(3, 10) + [train_docs[5], train_docs[9]]      # two documents copied from the training split
    variants = {name: core if comps == "full" else core.variant(comps) for name, comps in tr.ROWS}
    train_set = contamination.train_ngram_set(tok, ds, 13)
    return SimpleNamespace(core=core, variants=variants, train_docs=train_docs, val_docs=val_docs,
                           test_docs=test_docs, train_set=train_set,
                           ev=eval_split(val_docs), ev_test=eval_split(test_docs, "test"))


# ------------------------------------------------------------------ helpers of the driver
def test_rows_equal_the_models_own_scoring(world):
    w = world
    rows = tr.score_rows(w.core, w.variants, w.ev)
    for name, _ in tr.ROWS:
        direct = w.variants[name].score_stream(w.ev.tok, w.ev.doc_start, n_bytes=w.ev.n_bytes)
        assert rows[name]["bits_per_byte"] == direct["bits_per_byte"], name
    lp = w.core.kn.logprob(w.ev.tok, w.ev.doc_start)
    assert rows[tr.KNMODEL]["bits_per_byte"] == bits_per_byte(lp, w.ev.n_bytes) == rows["kn5_knmodel_bits_per_byte"]
    assert np.isclose(rows[tr.KNMODEL]["doc_bits"].sum(), -lp.sum())
    assert abs(rows["kn5"]["bits_per_byte"] - rows[tr.KNMODEL]["bits_per_byte"]) <= 1e-9
    assert rows["cvtm_core"]["bits_per_byte"] < rows["kn5"]["bits_per_byte"]   # copies help on this corpus


def test_deoverlap_flags_and_summaries(world):
    w = world
    rows = tr.score_rows(w.core, w.variants, w.ev_test)
    keep, frac = tr.deoverlap(w.train_set, w.ev_test)
    o = contamination.overlap(w.train_set, w.ev_test.tok, w.ev_test.doc_start, 13)
    assert np.array_equal(keep, o["doc_keep"].astype(bool)) and np.array_equal(frac, o["doc_frac"])
    assert not keep[-1] and not keep[-2] and keep[:10].all()          # the copied documents are dropped
    tokens = np.diff(w.ev_test.doc_start)
    full = tr.summarize("test", rows, tokens, w.ev_test.doc_bytes, None, n_boot=50)
    kept = tr.summarize("test.deoverlapped", rows, tokens, w.ev_test.doc_bytes, keep, n_boot=50)
    assert full["test.cvtm_core.bits_per_byte"]["value"] == rows["cvtm_core"]["bits_per_byte"]
    db = rows["cvtm_core"]["doc_bits"]
    assert kept["test.deoverlapped.cvtm_core.bits_per_byte"]["value"] == db[keep].sum() / w.ev_test.doc_bytes[keep].sum()
    assert kept["test.deoverlapped.documents_tokens_bytes"]["value"][0] == 10
    for a, b in tr.PAIRS:                                              # every paired difference on both sets
        assert f"test.{a}.minus_{b}.bits_per_byte" in full and f"test.deoverlapped.{a}.minus_{b}.bits_per_byte" in kept
    assert full["test.kn5_knmodel.bits_per_byte_ci95"]["label"] == "derived"


def _rows(core: float, cache: float) -> dict:
    return {"cvtm_core": {"bits_per_byte": core}, "kn5_D0": {"bits_per_byte": cache}}


def test_verdict_rules_of_docs13():
    cfg = {"thresholds": {"H8.3_max_test_bits_per_byte": 1.3219}}
    v = tr.verdicts(_rows(1.3219, 1.33), cfg, True)
    assert v["verdict.H8.3"]["value"]["verdict"] == "met"                # "at most": the threshold itself is met
    assert v["verdict.H8.4"]["value"]["verdict"] == "met"
    v = tr.verdicts(_rows(1.32190001, 1.32190001), cfg, True)
    assert v["verdict.H8.3"]["value"]["verdict"] == "not met"
    assert v["verdict.H8.4"]["value"]["verdict"] == "not met"            # "lower": a tie is not met
    v = tr.verdicts(_rows(1.2, 1.3), cfg, False)
    assert v["verdict.H8.3"]["value"]["verdict"].startswith("invalid")


def test_only_the_two_reporting_checks_leave_the_verdicts_valid():
    assert tr.verdict_valid([])
    assert tr.verdict_valid(["test.doc_keep_equals_M2_flags", "test.kn5_vote_form_within_1e-9_of_knmodel"])
    for name in ("reproduces.test_kn5_knmodel", "test.one_access_row_appended", "test.access_row_is_this_run",
                 "test.raw_doc_bytes_sha256"):
        assert not tr.verdict_valid([name])


def test_exact_reproduction_checks(world):
    w = world
    rows = tr.score_rows(w.core, w.variants, w.ev)
    ref = {"validation_cvtm_core": rows["cvtm_core"]["bits_per_byte"], "validation_kn5_D0": rows["kn5_D0"]["bits_per_byte"],
           "validation_kn5_knmodel": rows["kn5_knmodel_bits_per_byte"]}
    assert all(ok for ok, _ in tr.exact_checks(rows, ref, "validation").values())
    res = tr.exact_checks(rows, {**ref, "validation_kn5_D0": ref["validation_kn5_D0"] + 1e-15}, "validation")
    assert not res["reproduces.validation_kn5_D0"][0] and res["reproduces.validation_cvtm_core"][0]


def test_code_hash_ignores_docstrings_and_comments_only(tmp_path):
    a = tmp_path / "a.py"
    a.write_text('"""Module doc."""\n# a comment\ndef f(x):\n    """Doc."""\n    return x + 1  # inline\n')
    b = tmp_path / "b.py"
    b.write_text('"""Another doc."""\ndef f(x):\n    """Changed doc."""\n    return x + 1\n')
    c = tmp_path / "c.py"
    c.write_text('def f(x):\n    return x + 2\n')
    assert tr.code_sha256(a) == tr.code_sha256(b) != tr.code_sha256(c)


def test_the_code_of_the_core_equals_the_freeze():
    """The published cvtm sources differ from the freeze in docstrings and comments only (config pins)."""
    cfg = json.loads(CONFIG.read_text())
    assert all(ok for _, ok, _ in tr.code_checks(cfg))


def test_every_refusal_is_named(monkeypatch):
    cfg = json.loads(CONFIG.read_text())
    t = cfg["test"]
    rows = [{"purpose": "other"}] * t["access_rows_before"]
    ok_args = SimpleNamespace(confirm_test_read=True)
    good = dict(docs13=HEADING, rows=rows, log_sha=t["access_log_sha256_before"], failed=[], run_dir="good")
    monkeypatch.setattr(tr, "run_identity_ok", lambda cfg, run_dir: run_dir == "good")

    def refusals(**over):
        k = {**good, **over}
        return tr.test_read_refusals(cfg, k.get("args", ok_args), k["docs13"], k["rows"], k["log_sha"], k["failed"],
                                     k["run_dir"])

    assert refusals() == []
    assert "checks failed before the read: freeze.core_state_hash" in refusals(failed=["freeze.core_state_hash"])
    assert refusals(args=SimpleNamespace(confirm_test_read=False)) == ["--confirm-test-read not given"]
    assert "not run by scripts/run_logged.py" in refusals(run_dir="bad")[0]
    assert "no heading line" in refusals(docs13="the CVTM-core test read of H8.3 and H8.4 (not a heading)\n")[0]
    assert "rows, the plan declares" in refusals(rows=rows + [{"purpose": "x"}])[0]
    assert "access log sha256" in refusals(log_sha="0" * 64)[0]
    assert "already carry this read's purpose" in refusals(rows=rows[:-1] + [{"purpose": t["purpose"]}])[0]


def test_the_config_holds_the_registered_numbers():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["thresholds"]["H8.3_max_test_bits_per_byte"] == 1.3219
    assert cfg["reference"]["validation_cvtm_core"] == 1.240880320113308
    assert cfg["reference"]["validation_kn5_D0"] == 1.3338053432747112
    assert cfg["reference"]["test_kn5_knmodel"] == 1.3718501779650507
    assert cfg["test"]["access_rows_before"] == 112 and cfg["test"]["require_project_data_root"] is True
    registered = (EXPERIMENTS.parent / "docs" / "13-preregistration.md").read_text().splitlines()
    assert "at most 1.3219 (KN-5 minus 0.05)" in registered[177] and registered[177].startswith("| H8.3 |")
    assert registered[178].startswith("| H8.4 |") and "| lower |" in registered[178]


# ------------------------------------------------------------------ main end to end on a synthetic store
def _write_split(split: str, docs, root: Path) -> None:
    tok, ds = stream(docs)
    n = int(tok.shape[0])
    arrays = {"tok": tok, "doc_start": ds.astype(np.uint32), "word_id": np.zeros(n, np.uint32),
              "kind_map": np.zeros((3, V), np.uint16), "stop": np.zeros(V, np.uint8), "sym_freq": np.zeros(8, np.uint32),
              "targets": np.arange(n, dtype=np.uint32)}
    hashes = corpus_store.write_split_arrays(protocol.CORPUS, 8000, split, arrays, root=root)
    texts = ["x" * (3 * len(d) + 1) for d in docs]
    corpus_store.write_split_meta(protocol.CORPUS, 8000, split,
                                  {"sha256": hashes, "n_bytes_utf8": sum(len(s) for s in texts), "n_words": n}, root=root)
    corpus_store.write_raw_split(protocol.CORPUS, split, texts, [f"t{i}" for i in range(len(docs))],
                                 {"source": "synthetic"}, root=root)


def _write_flags(split: str, ev, train_set, root: Path, wrong: bool = False) -> None:
    keep = contamination.overlap(train_set, ev.tok, ev.doc_start, 13)["doc_keep"].astype(np.uint8)
    if wrong:
        keep = 1 - keep
    path = contamination.contamination_dir(protocol.CORPUS, 8000, root) / f"{split}_13gram.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sha256": {"doc_keep": hashing.sha256_array(keep)}}))


@pytest.fixture
def store(world, tmp_path, monkeypatch):
    """A synthetic data root (the conftest's TMLM_DATA_DIR), a 3-row access log, a run dir and a config."""
    w = world
    root = corpus_store.data_root()
    for split, docs in (("train", w.train_docs), ("validation", w.val_docs), ("test", w.test_docs)):
        _write_split(split, docs, root)
    _write_flags("validation", w.ev, w.train_set, root)
    _write_flags("test", w.ev_test, w.train_set, root)
    log = evalsets.access_log_path()
    log.write_text("".join(json.dumps({"caller": "x", "pid": 1, "purpose": f"earlier {i}", "utc": "2026-09-22T00:00:00Z"},
                                      sort_keys=True) + "\n" for i in range(3)))
    cfg = json.loads(CONFIG.read_text())
    cfg["frozen"]["core_state_hash"] = w.core.state_hash()
    cfg["frozen"]["data_digests"] = protocol.dataset_hashes(("train", "validation", "test"), protocol.CORPUS, 8000)
    rv = tr.score_rows(w.core, w.variants, w.ev)
    rt = tr.score_rows(w.core, w.variants, w.ev_test)
    cfg["reference"] = {"validation_cvtm_core": rv["cvtm_core"]["bits_per_byte"],
                        "validation_kn5_D0": rv["kn5_D0"]["bits_per_byte"],
                        "validation_kn5_knmodel": rv["kn5_knmodel_bits_per_byte"],
                        "test_kn5_knmodel": rt["kn5_knmodel_bits_per_byte"]}
    cfg["test"].update(access_rows_before=3, access_log_sha256_before=hashing.sha256_file(log),
                       require_project_data_root=False)
    prereg = tmp_path / "13-preregistration.md"
    prereg.write_text("# 13\n\n" + HEADING + "text\n")
    monkeypatch.setattr(tr, "PREREG", prereg)
    monkeypatch.setattr(tr, "load_frozen_core", lambda c: w.core)
    monkeypatch.setattr(tr, "freeze_file_checks", lambda c: [("freeze.cache_meta_sha256", True, {}),
                                                             ("freeze.core_file_sha256", True, {})])
    s = SimpleNamespace(cfg=cfg, log=log, prereg=prereg, root=root, tmp=tmp_path, rt=rt, n=0, run=None,
                        run_name=cfg["test"]["run_name"], monkeypatch=monkeypatch)
    return s


def _main(s, cfg=None, extra=("--confirm-test-read",), run_name=None) -> int:
    """One driver run in a fresh run folder (as scripts/run_logged.py gives every run its own)."""
    s.n += 1
    s.run = s.tmp / f"run{s.n}"
    s.run.mkdir()
    (s.run / "run.json").write_text(json.dumps({"name": run_name or s.run_name, "status": "running"}))
    s.monkeypatch.setenv("TMLM_RUN_DIR", str(s.run))
    path = s.tmp / f"cfg{s.n}.json"
    path.write_text(json.dumps(cfg or s.cfg))
    return tr.main(["--config", str(path), "--split", "test", *extra])


def _metrics(s) -> dict:
    data = json.loads((s.run / "metrics.json").read_text())
    runlog.validate_metrics(data)
    return data


def test_an_accepted_read_appends_exactly_one_row_and_writes_everything(store):
    s = store
    before = s.log.read_text().splitlines()
    assert _main(s) == tr.EXIT_OK
    after = s.log.read_text().splitlines()
    assert after[:3] == before and len(after) == 4
    row = json.loads(after[-1])
    assert row["caller"] == "corpus_store.load_split" and row["purpose"] == s.cfg["test"]["purpose"]
    assert row["pid"] == os.getpid()
    met = _metrics(s)
    assert met["checks_failed"]["value"] == []
    assert met["test.access_row"]["value"] == row
    assert met["verdict.H8.3"]["value"]["verdict"] in ("met", "not met")
    assert met["verdict.H8.4"]["value"]["cvtm_core"] == s.rt["cvtm_core"]["bits_per_byte"]
    assert met["test.deoverlapped.documents_tokens_bytes"]["value"][0] == 10
    npz = s.run / "scores_test.npz"
    assert hashing.sha256_file(npz) == met["test.scores_npz_sha256"]["value"]
    with np.load(npz) as z:
        assert np.array_equal(z["cvtm_core.doc_bits"], s.rt["cvtm_core"]["doc_bits"])
        assert z["cvtm_core.log2p"].dtype == np.float64 and z["doc_keep"].sum() == 10
    with np.load(npz) as z:
        n_tok, n_byte = int(z["doc_tokens"].sum()), int(z["doc_bytes"].sum())
    thr = s.cfg["thresholds"]["H8.3_max_test_bits_per_byte"]
    ident = dict(kn5_test=s.rt["kn5_knmodel_bits_per_byte"], test_tokens=n_tok, test_bytes=n_byte)
    report = vf.verify(s.run, "test", thr, **ident)
    assert report["agrees"] and report["npz_sha256_equal"]          # the report step's independent recompute
    assert report["recomputed_verdicts"]["H8.3"] == (met["verdict.H8.3"]["value"]["verdict"] == "met")
    assert report["kn5_identity"]["equal_within_1e-12"] and report["split_size"]["equal"]
    assert set(report["pairs_new_seed_deoverlapped"]) == set(report["pairs_new_seed"])
    assert not vf.verify(s.run, "test", thr, **{**ident, "kn5_test": ident["kn5_test"] + 1e-9})["agrees"]
    assert not vf.verify(s.run, "test", thr, **{**ident, "test_bytes": n_byte + 1})["agrees"]


def test_every_refusal_leaves_the_access_log_unchanged(store, monkeypatch):
    s = store
    sha = hashing.sha256_file(s.log)
    cases = []
    cases.append((dict(extra=()), "--confirm-test-read not given"))
    s.prereg.write_text("# 13 without the amendment\n")
    cases.append((dict(), "no heading line"))
    for kwargs, reason in cases:
        assert _main(s, **kwargs) == tr.EXIT_REFUSED
        assert hashing.sha256_file(s.log) == sha
        assert any(reason in r for r in _metrics(s)["test.refused"]["value"])
    s.prereg.write_text("# 13\n\n" + HEADING)
    wrong = copy.deepcopy(s.cfg)
    wrong["test"]["access_rows_before"] = 4
    assert _main(s, wrong) == tr.EXIT_REFUSED and hashing.sha256_file(s.log) == sha
    wrong = copy.deepcopy(s.cfg)
    wrong["reference"]["validation_kn5_D0"] += 1e-12                 # the validation pass does not reproduce
    assert _main(s, wrong) == tr.EXIT_REFUSED and hashing.sha256_file(s.log) == sha
    assert "reproduces.validation_kn5_D0" in _metrics(s)["test.refused"]["value"][0]
    assert _main(s, run_name="another run") == tr.EXIT_REFUSED and hashing.sha256_file(s.log) == sha
    assert "not run by scripts/run_logged.py" in " ".join(_metrics(s)["test.refused"]["value"])
    wrong = copy.deepcopy(s.cfg)                                       # a log with another sha256 than the pin
    wrong["test"]["access_log_sha256_before"] = "0" * 64
    assert _main(s, wrong) == tr.EXIT_REFUSED and hashing.sha256_file(s.log) == sha
    assert _metrics(s)["test.refused"]["value"] == [f"the access log sha256 is {sha}, the plan pins {'0' * 64}"]
    rows = s.log.read_text().splitlines()                              # an earlier row with this read's purpose
    first = json.loads(rows[0])
    first["purpose"] = s.cfg["test"]["purpose"]
    s.log.write_text("\n".join([json.dumps(first, sort_keys=True)] + rows[1:]) + "\n")
    same = copy.deepcopy(s.cfg)
    same["test"]["access_log_sha256_before"] = hashing.sha256_file(s.log)
    sha2 = hashing.sha256_file(s.log)
    assert _main(s, same) == tr.EXIT_REFUSED and hashing.sha256_file(s.log) == sha2
    assert _metrics(s)["test.refused"]["value"] == ["access-log rows [1] already carry this read's purpose"]


def test_a_kn5_mismatch_after_the_read_makes_both_verdicts_invalid(store):
    s = store
    cfg = copy.deepcopy(s.cfg)
    cfg["reference"]["test_kn5_knmodel"] += 1e-9
    assert _main(s, cfg) == tr.EXIT_CHECK_FAILED
    met = _metrics(s)
    assert met["checks_failed"]["value"] == ["reproduces.test_kn5_knmodel"]
    assert met["verdict.H8.3"]["value"]["verdict"].startswith("invalid")
    assert met["verdict.H8.4"]["value"]["verdict"].startswith("invalid")
    assert len(s.log.read_text().splitlines()) == 4


def test_a_deoverlap_mismatch_keeps_the_verdicts_valid(store):
    s = store
    path = contamination.contamination_dir(protocol.CORPUS, 8000, s.root) / "test_13gram.meta.json"
    path.write_text(json.dumps({"sha256": {"doc_keep": "0" * 64}}))
    assert _main(s) == tr.EXIT_CHECK_FAILED
    met = _metrics(s)
    assert met["checks_failed"]["value"] == ["test.doc_keep_equals_M2_flags"]
    assert met["verdict.H8.3"]["value"]["verdict"] in ("met", "not met")
    assert met["test.deoverlapped.cvtm_core.bits_per_byte"]["label"] == "unverified"
    assert met["test.cvtm_core.bits_per_byte"]["label"] == "measured"


def test_a_crash_after_the_read_keeps_the_rows_and_exits_4(store, monkeypatch):
    s = store

    real = tr.save_scores

    def broken(path, *args, **kwargs):                                  # only the test split's npz fails
        if Path(path).name == "scores_test.npz":
            raise OSError("disk full (simulated)")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(tr, "save_scores", broken)
    assert _main(s) == tr.EXIT_CRASHED_AFTER_READ
    met = _metrics(s)
    assert len(s.log.read_text().splitlines()) == 4                     # the row was written: the split is used
    assert met["test.crashed_after_read"]["value"]["row_appended"] is True
    assert "test.crashed_after_read" in met["checks_failed"]["value"]
    assert met["test.cvtm_core.bits_per_byte"]["value"] == s.rt["cvtm_core"]["bits_per_byte"]   # full split kept
    for key in ("verdict.H8.3", "verdict.H8.4"):
        assert met[key]["value"]["verdict"] == "invalid: crashed after the read"
    assert met[key]["value"]["cvtm_core"] == s.rt["cvtm_core"]["bits_per_byte"]


def test_a_failed_vote_form_check_labels_the_pairs_that_use_it(store, monkeypatch):
    s = store
    exact = tr.exact_checks

    def forced(rows, ref, split):
        out = exact(rows, ref, split)
        if split == "test":
            out["test.kn5_vote_form_within_1e-9_of_knmodel"] = (False, {"forced": True})
        return out
    monkeypatch.setattr(tr, "exact_checks", forced)
    assert _main(s) == tr.EXIT_CHECK_FAILED
    met = _metrics(s)
    assert met["checks_failed"]["value"] == ["test.kn5_vote_form_within_1e-9_of_knmodel"]
    assert met["verdict.H8.3"]["value"]["verdict"] in ("met", "not met")   # reporting only: the verdicts stay valid
    assert met["test.kn5.bits_per_byte"]["label"] == "unverified"
    for pair in ("cvtm_core.minus_kn5", "kn5_D0.minus_kn5"):
        assert met[f"test.{pair}.bits_per_byte"]["label"] == "unverified"
        assert met[f"test.deoverlapped.{pair}.bits_per_byte"]["label"] == "unverified"
    for pair in ("cvtm_core.minus_kn5_D0", "cvtm_core.minus_kn5_knmodel"):
        assert met[f"test.{pair}.bits_per_byte"]["label"] == "derived"


def test_an_empty_kept_set_gives_no_rows(world):
    w = world
    rows = tr.score_rows(w.core, w.variants, w.ev_test)
    doc_tokens = np.diff(np.asarray(w.ev_test.doc_start, dtype=np.int64))
    doc_bytes = np.asarray(w.ev_test.doc_bytes)
    out = tr.summarize("test.deoverlapped", rows, doc_tokens, doc_bytes, np.zeros(doc_bytes.shape[0], dtype=bool))
    assert out["test.deoverlapped.documents_tokens_bytes"]["value"] == [0, 0, 0]
    assert out["test.deoverlapped.empty"]["value"] is True and len(out) == 2
