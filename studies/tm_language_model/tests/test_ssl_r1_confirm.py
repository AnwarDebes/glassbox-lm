"""Tests of the confirmatory R1 (experiments/ssl_r1_confirm.py, src/tmlm/ssl/confirm.py; docs/13 A7, A8, A9).

(a) the counted loader: the A9 item 10 reference must exist before an access, one access recorded BEFORE the set is
    read, the downloaded bytes verified against the reference before parsing, a second access refused without a
    reason, the write-once reference with its race (FileExistsError, a file still being written), the testhash stage
    (metadata only, no access), the SST-2 confirm carve, the MEN test pairs, per-process access lines;
(b) the two-level bootstrap against an exact enumeration and a loop reference, the seed t interval;
(c) the verdict rules on synthetic inputs, estimate rows without a threshold, the A9 readings of the config;
(d) MNB against sklearn, NBSVM against a brute-force computation, the frozen LR and TM probes against R1's probes,
    the memory-lean full-pool matrices bit for bit against the R1 path;
(e) the random conjunction bank: literal-frequency sampling, featurisation against a pure-Python reference;
(f) Q_int (NPMI, top words, intrusion) and Q_con (occurrences, rounds, word codes, dimension-matched columns) against
    brute force;
(g) a tiny end-to-end run of every stage on toy data (CPU engine; the counted loader replaced by a spy that records
    real access lines in a temporary log and asserts exactly one access per frozen configuration and task, none
    before the dev stage and the preflight succeeded, including the full-pool rows and the arm A3R); the report's
    verdicts on synthetic cells. Q_pred is ssl_r1_controls.stage_qpred (imported unchanged and tested there) or, with
    the exact speedups the config enables, qpred_core (tests/test_ssl_r1_qpred_fast.py).
(h) opt-in (TMLM_HF_METADATA_TEST=1): the real host's file metadata of the two test files, printed (no content)."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT / "experiments", ROOT / "src"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_confirm as S  # noqa: E402
import ssl_r1_controls as C  # noqa: E402
import ssl_r1_pretext as R  # noqa: E402
from tmlm.data.synthetic import toy_corpus  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402
from tmlm.ssl import confirm as CF  # noqa: E402
from tmlm.ssl import controls as K  # noqa: E402
from tmlm.ssl import downstream as D  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402
from tmlm.ssl import probes as PR  # noqa: E402

CJ = S.load_config(S.DEFAULT_CONFIG)


@pytest.fixture
def arms(monkeypatch):
    monkeypatch.setattr(C, "CONTROL_ARMS", dict(C.CONTROL_ARMS))
    monkeypatch.setattr(P, "ARMS", dict(P.ARMS))
    S.register(CJ)
    return CJ


# ====================================================================== config
def test_import_registers_nothing_and_register_is_idempotent(monkeypatch):
    assert set(C.CONTROL_ARMS) == {"A3", "A0", "A3T", "A0T"} and "A3R" not in P.ARMS
    monkeypatch.setattr(C, "CONTROL_ARMS", dict(C.CONTROL_ARMS))
    monkeypatch.setattr(P, "ARMS", dict(P.ARMS))
    S.register(CJ)
    S.register(CJ)
    assert C.CONTROL_ARMS["A2T"] == ("A2", True) and C.CONTROL_ARMS["A2"] == ("A2", False) and C.CONTROL_ARMS["A6"] == ("A6", False)
    assert C.CONTROL_ARMS["A3R"] == ("A3R", False)
    a = P.ARMS["A3R"]
    assert a.mode == "masked" and a.f2 and a.random_kind_map and not a.f1 and a.engine == ()
    C.CONTROL_ARMS["A2"] = ("A3", False)
    with pytest.raises(SystemExit):
        S.register(CJ)
    C.CONTROL_ARMS["A2"] = ("A2", False)
    P.ARMS["A3R"] = P.Arm("A3R", "masked", f2=True, description="other")
    with pytest.raises(SystemExit):
        S.register(CJ)


def test_config_is_confirmatory_and_matches_the_registered_rows(arms):
    assert CJ["status"].startswith("CONFIRMATORY")
    S.check_config(CJ, S.Rec(False))
    v = CJ["verdicts"]
    assert (v["H17.1"]["min_points"], v["H17.2"]["min_points"], v["H17.4"]["min_points"], v["H17.11"]["min_points"]) == (1.0, 2.0, 1.0, -2.0)
    assert (v["H17.3"]["min_vs_ppmi"], v["H17.3"]["min_vs_c0y"], v["literal_credit"]["min_points"]) == (0.0, 0.05, 1.0)
    assert CJ["statistics"]["n_boot"] == 10000 and CJ["params"]["seeds"] == [1, 2, 3]
    assert CJ["winner"]["seed100_winner"] == "A3" and CJ["winner"]["seed100_runner_up"] == "A2" and CJ["winner"]["rule"] == "seed100"
    a = CJ["configurations"]["arms"]
    assert {k for k, s in a.items() if s["role"] in ("top1", "top2")} == {"A3", "A2"} and {"A0", "A1", "A6", "A3R"} <= set(a)
    assert a["A3"]["twin_arm"] == "A3T" and a["A2"]["twin_arm"] == "A2T"
    # the twins are the control study's twin overrides on the arm's own data and config
    for tw, base in (("A3T", "A3"), ("A2T", "A2")):
        cfg, ref = C.arm_config(tw, 1, CJ["params"]), P.engine_config(base, 1)
        assert {k for k, x in cfg.to_dict().items() if ref.to_dict()[k] != x} == {"p_ins", "p_neg", "s"}
        assert C.twin_thresholds_ok(cfg, ref)
    assert P.engine_config("A2", 1).neg_budget == 1 and C.arm_config("A2T", 1, CJ["params"]).cand_tries == 4
    # A3R (A9 item 9): A3's engine config, A3's F2 targets, A6's random kind_map
    assert C.arm_config("A3R", 2, CJ["params"]) == P.engine_config("A3", 2)
    c = toy_corpus(vocab=30, n_docs=4, doc_len=50, seed=3)
    r, a3, a6 = P.arm_data("A3R", c, 30), P.arm_data("A3", c, 30), P.arm_data("A6", c, 30)
    assert np.array_equal(r.kind_map, a6.kind_map) and np.array_equal(r.sym_freq, a6.sym_freq) and not np.array_equal(r.kind_map, a3.kind_map)
    assert all(np.array_equal(r.targets(e), a3.targets(e)) for e in range(3))
    # every probe budget is an R1 budget; the arm plan carries the registered R1 probe set
    blk = CJ["probe_plans"]["arm"][0]
    assert blk["lr"] == [8, 32, 128, 512] and blk["tm"]["budgets"] == [32]
    for plan in [v for v in CJ["probe_plans"].values() if isinstance(v, list)]:
        for b in plan:
            for clf in ("lr", "tm", "mnb", "nbsvm"):
                spec = b.get(clf)
                bs = spec["budgets"] if isinstance(spec, dict) else (spec or [])
                assert set(bs) <= set(CJ["params"]["budgets"])
    assert CJ["a8"]["exposure_matched_batch"] == 196 and 196 in CJ["params"]["young_ladder"]["abs_batches"]
    # A9: 28 configurations, 84 + 6 expected accesses
    assert S.a9_problems(CJ) == [] and CJ["test_discipline"]["n_configurations"] == 28
    keys = S.expected_accesses(CJ)
    assert len(keys) == 90 and len(set(keys)) == 90 and "config=A3R_seed3; task=trec" in keys and "config=qcon_A2_seed1; task=men_test" in keys
    assert S.needs_full(CJ, "arm", "A3") and S.needs_full(CJ, "arm", "A2") and S.needs_full(CJ, "common", None)
    assert not S.needs_full(CJ, "arm", "A0") and not S.needs_full(CJ, "arm", "A3T") and not S.needs_full(CJ, "rand", None)


@pytest.mark.parametrize("path,value", [(("winner", "rule"), "dev_seeds_1_3"), (("verdicts", "code"), "dim"),
                                        (("qint", "primary_reference"), "validation"), (("qcon", "men_confirm"), "test"),
                                        (("verdicts", "H17.11", "lr_set"), "epoch4.all"), (("full_budget", "draw"), 1),
                                        (("a8", "rows_without_threshold"), []), (("statistics", "decides"), ["H17.1"]),
                                        (("test_discipline", "non_lfs_reference_amendment"), None)])
def test_check_config_refuses_a_reading_that_disagrees_with_a9(arms, path, value):
    cj = copy.deepcopy(CJ)
    d = cj
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    assert S.a9_problems(cj)
    with pytest.raises(SystemExit):
        S.check_config(cj, S.Rec(False))


# ====================================================================== (a) counted loader
def _fake_project(tmp: Path) -> Path:
    root = tmp / "proj"
    (root / "docs" / "notes").mkdir(parents=True)
    (root / "data").mkdir()
    return root


def _write_parquet(path: Path, cols: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def _fake_meta(path: Path, rel: str):
    """What HfApi.get_paths_info reports for an LFS file (sha256 and size of the content), for a local file."""
    from tmlm.common import hashing

    size = int(path.stat().st_size)
    return {"repo": "x", "repo_type": "dataset", "revision": "r", "call": "fake", "path": rel, "size": size, "blob_id": "b" * 40,
            "lfs": {"sha256": hashing.sha256_file(path), "size": size, "pointer_size": 134}, "xet_hash": None, "type": "RepoFile"}


def test_load_confirm_needs_the_reference_counts_before_reading_and_refuses_a_second_access(tmp_path, monkeypatch):
    monkeypatch.delenv("TMLM_DATA_DIR", raising=False)
    root = _fake_project(tmp_path)
    tp = CF.test_file_path("agnews", root)
    _write_parquet(tp, {"text": [f"doc {i}" for i in range(10)], "label": [i % 4 for i in range(10)]})
    log = root / "data" / "test_access_log.jsonl"
    # no reference yet (the testhash stage has not run): refused BEFORE any access
    with pytest.raises(FileNotFoundError):
        CF.load_confirm("agnews", "A3_seed1", "unit test", root=root, allow_download=False)
    assert not log.exists()
    rel = CF.CONFIRM_SOURCES["agnews"]["file"]
    w = CF.write_test_hash_reference("agnews", root, metadata_fn=lambda task: _fake_meta(tp, rel))
    assert w["written"] and w["metadata"]["lfs"]["sha256"] == w["sha256"]
    assert not log.exists()                               # the reference costs no access
    opened = []
    real_read = CF._read_parquet

    def spy_read(*a, **k):
        opened.append(log.exists() and len(log.read_text().splitlines()))   # lines in the log when the file is read
        return real_read(*a, **k)

    monkeypatch.setattr(CF, "_read_parquet", spy_read)
    cs = CF.load_confirm("agnews", "A3_seed1", "unit test", root=root, allow_download=False)
    assert opened == [1]                                  # the access was counted before the file was read
    assert cs.texts[3] == "doc 3" and cs.y.tolist() == [i % 4 for i in range(10)]
    assert cs.info["access"]["after"] == cs.info["access"]["before"] + 1 == 1
    assert cs.info["equal"] and not cs.info["written"] and cs.info["rows"] == 10 and cs.info["sha256"] == w["sha256"]
    with pytest.raises(CF.ConfirmAccessRefused):
        CF.load_confirm("agnews", "A3_seed1", "again", root=root, allow_download=False)
    assert len(log.read_text().splitlines()) == 1        # a refused access is not read and not counted
    CF.load_confirm("agnews", "A2_seed1", "other configuration", root=root, allow_download=False)
    CF.load_confirm("agnews", "A3_seed1", "again", root=root, allow_download=False, reaccess_reason="GPU error after the access")
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 3 and "REACCESS 2: GPU error after the access" in lines[-1]["purpose"]
    assert CF.prior_accesses("A3_seed1", "agnews", root) == [lines[0], lines[2]]
    assert CF.prior_accesses("A3", "agnews", root) == [] and CF.prior_accesses("A3_seed1", "trec", root) == []
    assert len(CF.own_accesses(root)) == 3 and CF.own_accesses(root, pid=1) == []
    # a changed test file: the access is counted, the bytes are rejected BEFORE they are parsed
    _write_parquet(tp, {"text": ["x"] * 10, "label": [0] * 10})
    opened.clear()
    with pytest.raises(ValueError):
        CF.load_confirm("agnews", "A0_seed1", "changed file", root=root, allow_download=False)
    assert opened == [] and len(log.read_text().splitlines()) == 4


def test_write_once_reference_race_and_partial_file(tmp_path):
    import threading
    import time as _t

    root = _fake_project(tmp_path)
    sha = "a" * 64
    res, errs = [], []
    barrier = threading.Barrier(4)

    def go():
        try:
            barrier.wait()
            res.append(CF.verify_or_write_test_hash("trec", sha, 123, root))
        except Exception as exc:  # pragma: no cover
            errs.append(exc)

    th = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs and sum(r["written"] for r in res) == 1 and all(r["equal"] for r in res)
    with pytest.raises(ValueError):
        CF.verify_or_write_test_hash("trec", "b" * 64, 123, root)          # another sha: refused
    with pytest.raises(ValueError):
        CF.verify_or_write_test_hash("trec", sha, 124, root)               # another size: refused
    # the creator is still writing: FileExistsError, the reader retries until the content is complete
    hp = CF.test_hash_path("agnews", root)
    hp.write_text("")
    rec = CF._test_hash_record("agnews", "c" * 64, 9, None)
    t = threading.Timer(0.3, lambda: hp.write_text(json.dumps(rec)))
    t.start()
    t0 = _t.time()
    r = CF.verify_or_write_test_hash("agnews", "c" * 64, 9, root)
    t.join()
    assert not r["written"] and r["equal"] and _t.time() - t0 >= 0.25
    # a reference naming another revision is refused
    bad = json.loads(hp.read_text())
    bad["source"]["revision"] = "other"
    hp.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        CF.test_hash_reference("agnews", root)


def test_testhash_stage_uses_metadata_only_and_makes_no_access(tmp_path, monkeypatch):
    root = _fake_project(tmp_path)
    monkeypatch.setenv("TMLM_DATA_DIR", str(root / "data"))
    monkeypatch.setattr(CF, "project_root", lambda: root)
    calls = []

    def meta(task):
        calls.append(task)
        return {"path": CF.CONFIRM_SOURCES[task]["file"], "size": 1000 + len(task), "blob_id": "b" * 40,
                "lfs": {"sha256": ("%064x" % (7 + len(task))), "size": 1000 + len(task), "pointer_size": 134}, "xet_hash": None}

    monkeypatch.setattr(CF, "test_file_metadata", meta)
    rec = S.Rec(False)
    S.stage_testhash(argparse.Namespace(), CJ, rec)
    assert rec.failed == [] and sorted(calls) == ["agnews", "trec"]
    for task in ("agnews", "trec"):
        ref = CF.test_hash_reference(task, root)
        assert ref["sha256"] == "%064x" % (7 + len(task)) and ref["bytes"] == 1000 + len(task)
        assert rec.metrics[f"testhash.{task}"]["value"]["metadata"]["lfs"]["sha256"] == ref["sha256"]
    assert not (root / "data" / "test_access_log.jsonl").exists()
    rec2 = S.Rec(False)
    S.stage_testhash(argparse.Namespace(), CJ, rec2)                        # a second run: equal, not rewritten
    assert rec2.failed == [] and not rec2.metrics["testhash.trec"]["value"]["written"]
    # no LFS sha256 published: an error, never a silent fallback
    monkeypatch.setattr(CF, "test_file_metadata", lambda task: {"path": "x", "size": 5, "blob_id": "b", "lfs": None})
    with pytest.raises(ValueError):
        CF.write_test_hash_reference("agnews", tmp_path / "other")
    # the RepoFile object of huggingface_hub is converted field by field
    from huggingface_hub.hf_api import RepoFile
    rf = RepoFile(path="p", size=7, oid="o", lfs={"size": 7, "oid": "d" * 64, "pointerSize": 130})
    assert CF.repo_file_metadata(rf) == {"path": "p", "size": 7, "blob_id": "o", "lfs": {"sha256": "d" * 64, "size": 7, "pointer_size": 130},
                                         "xet_hash": None, "type": "RepoFile"}


def test_non_lfs_test_file_is_refused_unless_an_amendment_allows_its_git_blob_id(tmp_path, monkeypatch):
    import hashlib

    # git blob ids of known contents (git hash-object)
    f = tmp_path / "e"
    f.write_bytes(b"")
    assert CF.git_blob_sha1(f) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    f.write_bytes(b"hello\n")
    assert CF.git_blob_sha1(f, chunk=2) == "ce013625030ba8dba906f756967f9e9ca394464a"
    root = _fake_project(tmp_path)
    monkeypatch.setenv("TMLM_DATA_DIR", str(root / "data"))
    monkeypatch.setattr(CF, "project_root", lambda: root)
    tp = CF.test_file_path("trec", root)
    _write_parquet(tp, {"text": ["q1", "q2"], "coarse_label": [0, 5]})
    blob = CF.git_blob_sha1(tp)
    non_lfs = {"path": CF.CONFIRM_SOURCES["trec"]["file"], "size": int(tp.stat().st_size), "blob_id": blob, "lfs": None, "xet_hash": None}
    lfs_meta = {"path": CF.CONFIRM_SOURCES["agnews"]["file"], "size": 11, "blob_id": "c" * 40,
                "lfs": {"sha256": "f" * 64, "size": 11, "pointer_size": 130}, "xet_hash": None}
    monkeypatch.setattr(CF, "test_file_metadata", lambda task: non_lfs if task == "trec" else lfs_meta)
    cj0 = copy.deepcopy(CJ)                                                  # A9 without A10: non-LFS files refused
    cj0["test_discipline"].update(non_lfs_reference="refuse", non_lfs_reference_amendment=None)
    rec = S.Rec(False)
    S.stage_testhash(argparse.Namespace(), cj0, rec)                         # A9 as written: TREC refused, AG News written
    assert rec.failed == ["testhash.trec.reference_written"] and CF.test_hash_reference("agnews", root)["sha256"] == "f" * 64
    assert rec.metrics["testhash.trec.metadata"]["value"]["blob_id"] == blob and not CF.test_hash_path("trec", root).exists()
    with pytest.raises(CF.ConfirmPreflightFailed):
        CF.preflight_references(["agnews", "trec"], root)
    with pytest.raises(ValueError):                                          # the mode alone is not enough: an amendment is named
        CF.write_test_hash_reference("trec", root, metadata_fn=lambda t: non_lfs, non_lfs="git_blob_sha1")
    cj = copy.deepcopy(CJ)
    cj["test_discipline"].update(non_lfs_reference="git_blob_sha1", non_lfs_reference_amendment="docs/13 A10 (test)")
    prereg = tmp_path / "13-preregistration.md"
    prereg.write_text("## Amendments\n### A9 (2026-09-25 12:25 UTC): readings\n")
    monkeypatch.setattr(S, "PREREG", prereg)
    assert S.a9_problems(cj)                                                 # the named amendment must exist in docs/13
    prereg.write_text("## Amendments\n### A9 (2026-09-25 12:25 UTC): readings\n### A10 (2026-09-25): git blob ids\n")
    assert S.a9_problems(cj) == []
    rec = S.Rec(False)
    S.stage_testhash(argparse.Namespace(), cj, rec)
    assert rec.failed == [], rec.failed
    ref = CF.test_hash_reference("trec", root)
    assert ref["reference"] == "git_blob_sha1" and ref["digest"] == blob and ref["sha256"] is None
    cs = CF.load_confirm("trec", "A3_seed1", "unit test", root=root, allow_download=False)
    assert cs.info["reference"] == "git_blob_sha1" and cs.info["sha256"] == hashlib.sha256(tp.read_bytes()).hexdigest()
    assert cs.texts == ["q1", "q2"] and cs.y.tolist() == [0, 5]
    _write_parquet(tp, {"text": ["q1", "qX"], "coarse_label": [0, 5]})
    with pytest.raises(ValueError):
        CF.load_confirm("trec", "A2_seed1", "changed file", root=root, allow_download=False)


def test_preflight_references_and_men_reference(tmp_path, monkeypatch):
    monkeypatch.delenv("TMLM_DATA_DIR", raising=False)
    root = _fake_project(tmp_path)
    with pytest.raises(CF.ConfirmPreflightFailed):
        CF.preflight_references(["agnews", "sst2", "trec"], root)
    for task in ("agnews", "trec"):
        CF.verify_or_write_test_hash(task, "e" * 64, 10, root)
    with pytest.raises(CF.ConfirmPreflightFailed):                       # SST-2: no A7 hash file yet
        CF.preflight_references(["agnews", "sst2", "trec"], root)
    n = 60
    _write_parquet(D.local_path("sst2", D.TASKS["sst2"]["train_file"], root), {"sentence": [f"s{i}" for i in range(n)], "label": [i % 2 for i in range(n)]})
    D.write_hash_file("sst2", root)
    r = CF.preflight_references(["agnews", "sst2", "trec"], root)
    assert r["agnews"]["sha256"] == "e" * 64 and r["sst2"]["carve_idx_sha256"]
    with pytest.raises(CF.ConfirmPreflightFailed):
        CF.men_test_reference(root)


def test_sst2_confirm_carve_and_men_test_are_counted(tmp_path, monkeypatch):
    monkeypatch.delenv("TMLM_DATA_DIR", raising=False)
    root = _fake_project(tmp_path)
    n = 200
    _write_parquet(D.local_path("sst2", D.TASKS["sst2"]["train_file"], root),
                   {"sentence": [f"s{i}" for i in range(n)], "label": [i % 2 for i in range(n)]})
    D.write_hash_file("sst2", root)
    cs = CF.load_confirm("sst2", "common", "unit test", root=root)
    rows = np.flatnonzero(D.residues(n) == 0)
    assert cs.texts == [f"s{i}" for i in rows] and cs.y.tolist() == [int(i % 2) for i in rows] and len(rows) > 0
    assert cs.info["access"]["after"] == 1
    # MEN test: sha256 checked against the A6 hash file, counted under its own tag
    rel = "data/concept_eval/men/MEN/MEN_dataset_lemma_form.test"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_text("cat-n dog-n 40.0\ncar-n road-n 30.0\n")
    from tmlm.common import hashing
    (root / "docs" / "notes" / "preregistration-data-hashes-A6-men.json").write_text(json.dumps({"files": {rel: {"sha256": hashing.sha256_file(root / rel)}}}))
    assert CF.men_test_reference(root)["file"] == rel
    pairs, info = CF.load_men_test("qcon_A3_seed1", "unit test", root=root)
    assert pairs == [("cat", "dog", 40.0), ("car", "road", 30.0)] and info["access"]["after"] == 2
    with pytest.raises(CF.ConfirmAccessRefused):
        CF.load_men_test("qcon_A3_seed1", "again", root=root)


# ====================================================================== (b) statistics
UNITS = {1: [0.3, 1.7], 2: [-0.4, 2.2], 3: [0.9, 5.1]}


def _exact_distribution(units: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(units)
    arrs = [np.asarray(units[k], float) for k in keys]
    S_ = len(arrs)
    vals, w = [], []
    for pick in itertools.product(range(S_), repeat=S_):
        per_pos = [[np.mean([arrs[s][i] for i in idx]) for idx in itertools.product(range(len(arrs[s])), repeat=len(arrs[s]))] for s in pick]
        for combo in itertools.product(*per_pos):
            vals.append(np.mean(combo))
    v = np.asarray(vals)
    return v, np.full(v.shape[0], 1.0 / v.shape[0])


def test_two_level_bootstrap_matches_the_exact_enumeration():
    v, w = _exact_distribution(UNITS)
    assert v.shape[0] == 27 * 4 ** 3
    b = CF.two_level_bootstrap(UNITS, n_boot=200000, seed=3)
    assert b["mean"] == pytest.approx(np.mean([np.mean(x) for x in UNITS.values()]))
    assert np.average(v, weights=w) == pytest.approx(b["mean"])            # the bootstrap is centred on the estimate
    # the percentile interval of the resampled statistic approaches the exact quantiles
    lo, hi = np.quantile(v, [0.025, 0.975])
    assert abs(b["lo"] - lo) < 0.03 and abs(b["hi"] - hi) < 0.03
    assert b["boot_sd"] == pytest.approx(v.std(), rel=0.01)
    assert b["n_seeds"] == 3 and b["n_units"] == {"1": 2, "2": 2, "3": 2}


def _loop_reference(units: dict, n_boot: int, seed: int, chunk: int) -> np.ndarray:
    keys = sorted(units)
    arrs = [list(map(float, units[k])) for k in keys]
    S_ = len(arrs)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1718, S_] + [len(a) for a in arrs]))
    out = []
    for c0 in range(0, n_boot, chunk):
        m = min(chunk, n_boot - c0)
        pick = rng.integers(0, S_, size=(m, S_))
        draws = {(k, s): rng.integers(0, len(arrs[s]), size=(m, len(arrs[s]))) for k in range(S_) for s in range(S_)}
        for i in range(m):
            vals = []
            for k in range(S_):
                s = int(pick[i, k])
                vals.append(sum(arrs[s][int(j)] for j in draws[(k, s)][i]) / len(arrs[s]))
            out.append(sum(vals) / S_)
    return np.asarray(out)


def test_two_level_bootstrap_equals_a_loop_reference_and_the_t_interval():
    u = {1: [0.1, 0.5, -0.2, 0.7, 0.3], 2: [1.1, 0.9, 1.4, 0.2, 0.8], 3: [0.0, -0.5, 0.4, 0.6, 0.1]}
    ref = _loop_reference(u, 1000, 11, 250)
    b = CF.two_level_bootstrap(u, n_boot=1000, seed=11, chunk=250)
    assert b["lo"] == pytest.approx(float(np.quantile(ref, 0.025)), abs=1e-12)
    assert b["hi"] == pytest.approx(float(np.quantile(ref, 0.975)), abs=1e-12)
    t = CF.seed_t_interval(u)
    means = [np.mean(x) for x in u.values()]
    h = 4.303 * np.std(means, ddof=1) / np.sqrt(3)
    assert t["mean"] == pytest.approx(np.mean(means)) and t["lo"] == pytest.approx(np.mean(means) - h)
    # the unpaired variant: identical groups give a zero-centred interval; a shift moves it
    ab = {s: (np.arange(10.0) + s, np.arange(10.0) + s) for s in (1, 2, 3)}
    z = CF.two_level_bootstrap_unpaired(ab, n_boot=2000, seed=1)
    assert z["mean"] == 0.0 and z["lo"] < 0 < z["hi"]
    sh = CF.two_level_bootstrap_unpaired({s: (a + 20.0, b) for s, (a, b) in ab.items()}, n_boot=2000, seed=1)
    assert sh["mean"] == pytest.approx(20.0) and sh["lo"] > 0 and sh["per_seed"] == {"1": 20.0, "2": 20.0, "3": 20.0}
    assert CF.two_level_bootstrap({}, 10)["mean"] is None


# ====================================================================== (c) verdicts
def _boot(mean, lo, hi, n=3):
    return {"mean": mean, "lo": lo, "hi": hi, "n_seeds": n}


@pytest.mark.parametrize("mean,lo,thr,expect", [(1.2, 0.1, 1.0, CF.MET), (1.0, 0.01, 1.0, CF.MET), (0.99, 0.2, 1.0, CF.NOT_MET),
                                                (1.5, -0.1, 1.0, CF.NOT_MET), (1.5, 0.0, 1.0, CF.NOT_MET), (2.5, 0.3, 2.0, CF.MET)])
def test_verdict_min_ci(mean, lo, thr, expect):
    assert CF.verdict_min_ci("x", _boot(mean, lo, lo + 3), None, thr)["verdict"] == expect


def test_verdicts_incomplete_above_min_mean_h17_3_label_and_winner():
    assert CF.verdict_min_ci("x", _boot(3.0, 1.0, 5.0, n=2), None, 1.0)["verdict"] == CF.INCOMPLETE
    assert CF.verdict_min_ci("x", None, None, 1.0)["verdict"] == CF.INCOMPLETE
    assert CF.verdict_above_ci("x", _boot(0.01, 0.001, 0.02), None)["verdict"] == CF.MET
    assert CF.verdict_above_ci("x", _boot(0.01, -0.001, 0.02), None)["verdict"] == CF.NOT_MET
    assert CF.verdict_min_mean("h", {"1": -1.0, "2": -3.0, "3": -1.9}, -2.0)["verdict"] == CF.MET       # mean -1.967 >= -2
    assert CF.verdict_min_mean("h", {"1": -1.0, "2": -3.5, "3": -1.9}, -2.0)["verdict"] == CF.NOT_MET
    assert CF.verdict_min_mean("h", {"1": -1.0, "2": -3.0, "3": -1.0}, -2.0)["verdict"] == CF.MET
    assert CF.verdict_min_mean("h", {"1": -1.0, "2": None, "3": -1.0}, -2.0)["verdict"] == CF.INCOMPLETE
    w = {"1": 0.30, "2": 0.32, "3": 0.28}
    assert CF.verdict_h17_3(w, {"1": 0.29, "2": 0.30, "3": 0.28}, {"1": 0.20, "2": 0.25, "3": 0.22})["verdict"] == CF.MET
    assert CF.verdict_h17_3(w, {"1": 0.35, "2": 0.30, "3": 0.28}, {"1": 0.20, "2": 0.25, "3": 0.22})["verdict"] == CF.NOT_MET
    assert CF.verdict_h17_3(w, {"1": 0.29, "2": 0.30, "3": 0.28}, {"1": 0.28, "2": 0.25, "3": 0.22})["verdict"] == CF.NOT_MET
    assert CF.verdict_h17_3(w, {"1": 0.29}, {"1": 0.2})["verdict"] == CF.INCOMPLETE
    met = CF.verdict_min_ci("c", _boot(1.5, 0.2, 3.0), None, 1.0)
    miss = CF.verdict_min_ci("c", _boot(0.5, 0.2, 3.0), None, 1.0)
    comp = {"ladder.all": met, "ladder.dim": met, "twin.all": met, "twin.dim": met}
    assert CF.literal_learning_label(comp)["label"] == "literal-learning credit"
    assert CF.literal_learning_label({**comp, "twin.dim": miss})["label"] == "no literal-learning credit"
    assert CF.literal_learning_label({k: v for k, v in comp.items() if k != "ladder.dim"})["label"] == CF.INCOMPLETE
    r = CF.winner_by_rule("seed100", "A3", ["A3", "A2"], {"A3": 50.0, "A2": 51.0}, {"A3": 0.2, "A2": 0.1})
    assert r["winner"] == "A3" and r["dev_seeds_1_3"] == "A2"
    r = CF.winner_by_rule("dev_seeds_1_3", "A3", ["A3", "A2"], {"A3": 50.0, "A2": 50.3}, {"A3": 0.2, "A2": 0.1})
    assert r["winner"] == "A3" and r["decided_by"].startswith("Q_con")
    assert CF.winner_by_rule("dev_seeds_1_3", "A3", ["A3", "A2"], {"A3": None, "A2": 50.0}, {})["winner"] is None


# ====================================================================== (d) probes
def _toy_features(seed=0, n=120, F_=60, k=3):
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    X = rng.poisson(0.4, size=(n, F_)).astype(float)
    for c in range(k):
        X[y == c, c * 5:(c + 1) * 5] += rng.poisson(1.5, size=((y == c).sum(), 5))
    return sp.csr_matrix(X), y


def test_mnb_equals_sklearn():
    from sklearn.naive_bayes import MultinomialNB

    X, y = _toy_features()
    for a in (0.01, 0.5, 1.0):
        ours = CF.MNB(a).fit(X[:80], y[:80])
        sk = MultinomialNB(alpha=a).fit(X[:80], y[:80])
        assert np.allclose(ours.joint_log_likelihood(X[80:]), sk.predict_joint_log_proba(X[80:]), atol=1e-10)
        assert np.array_equal(ours.predict(X[80:]), sk.predict(X[80:]))
    f = CF.fit_mnb(X[:80], y[:80], 5, [0.01, 0.1, 1.0], 4)
    assert f.choice in (0.01, 0.1, 1.0) and set(f.cv["acc"]) == {0.01, 0.1, 1.0}
    assert np.mean(CF.frozen_predict(f, X[80:]) == y[80:]) > 0.6


def test_nbsvm_equals_a_brute_force_computation():
    from sklearn.svm import LinearSVC

    X, y = _toy_features(seed=1, n=90, F_=25)
    Xb = (X > 0).astype(float)
    for k in (2, 3):
        yk = y % k
        m = CF.NBSVM(C=1.0, alpha=1.0, beta=0.25, seed=3).fit(Xb[:60], yk[:60])
        D_ = Xb.toarray()
        targets = [1] if k == 2 else list(range(k))
        dec = []
        for c in targets:
            pos = yk[:60] == c
            p = 1.0 + np.array([sum(D_[i, f] for i in range(60) if pos[i]) for f in range(D_.shape[1])])
            q = 1.0 + np.array([sum(D_[i, f] for i in range(60) if not pos[i]) for f in range(D_.shape[1])])
            r = np.log(p / p.sum()) - np.log(q / q.sum())
            assert np.allclose(r, CF.nb_log_ratio(Xb[:60], pos, 1.0))
            Xr = D_[:60] * r[None, :]
            svm = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", dual=True, tol=1e-4, max_iter=10000, random_state=3).fit(Xr, pos.astype(int))
            w = svm.coef_.ravel()
            w2 = np.array([0.75 * np.mean(np.abs(w)) + 0.25 * wi for wi in w])
            dec.append(np.array([sum(D_[i, f] * r[f] * w2[f] for f in range(D_.shape[1])) + svm.intercept_[0] for i in range(60, 90)]))
        dec = np.stack(dec, axis=1)
        assert np.allclose(m.decision_function(Xb[60:]), dec, atol=1e-6)
        pred = (dec[:, 0] > 0).astype(int) if k == 2 else dec.argmax(axis=1)
        assert np.array_equal(m.predict(Xb[60:]), pred)
    f = CF.fit_nbsvm(Xb[:60], y[:60], 2, [0.1, 1.0], 4, 1.0, 0.25, 10000)
    assert f.kind == "nbsvm" and f.choice in (0.1, 1.0)


def test_frozen_lr_and_tm_reproduce_the_r1_probes():
    X, y = _toy_features(seed=2, n=100, F_=40)
    Xl, yl, Xe, ye = X[:60], y[:60], X[60:], y[60:]
    for dense in (False, True):
        A, B = (Xl.toarray(), Xe.toarray()) if dense else (Xl, Xe)
        ref = PR.lr_probe(A, yl, B, ye, 17, dense=dense, c_grid=[0.1, 1.0, 10.0], folds=4, max_iter=300)
        f = CF.fit_lr(A, yl, 17, dense, [0.1, 1.0, 10.0], 4, 300)
        assert float(np.mean(CF.frozen_predict(f, B) == ye)) == ref.accuracy and f.choice == ref.choice and f.cv == ref.cv
    Bl, Be = (Xl.toarray() > 0).astype(np.uint32), (Xe.toarray() > 0).astype(np.uint32)
    grid = [{"clauses": 20, "T": 10, "s": 2.0}, {"clauses": 20, "T": 10, "s": 5.0}]
    ref = PR.tm_probe(Bl, yl, Be, ye, 5, grid=grid, epochs=2, folds=3)
    f = CF.fit_tm(Bl, yl, 5, grid, 2, 3)
    assert float(np.mean(CF.frozen_predict(f, Be) == ye)) == ref.accuracy and f.choice == ref.choice
    import pickle
    g = pickle.loads(pickle.dumps(f))           # the frozen model survives the pickle of the dev stage
    assert np.array_equal(CF.frozen_predict(g, Be), CF.frozen_predict(f, Be))


# ====================================================================== (e) random conjunction bank
def test_random_bank_draws_literals_with_the_text_frequencies():
    from scipy.stats import chisquare

    c = toy_corpus(vocab=12, n_docs=5, doc_len=40, seed=4)
    single = CF.literal_single_counts(np, c.tok, c.doc_start, c.kind_map, "masked", 12)
    b = CF.random_bank(c.tok, c.doc_start, c.kind_map, "masked", 12, 60000, seed=1, max_lit=1)
    assert np.all(b.n_lit == 1) and np.all(b.col[:, 1:] == -1)
    obs = np.zeros_like(single)
    np.add.at(obs, (b.col[:, 0], b.sym[:, 0]), 1)
    exp = single / single.sum() * b.n
    keep = exp >= 20
    assert keep.sum() > 50 and obs[~keep].sum() < 0.05 * b.n
    stat = chisquare(obs[keep], exp[keep] * obs[keep].sum() / exp[keep].sum())
    assert stat.pvalue > 1e-4
    assert np.all(single[b.col[:, 0], b.sym[:, 0]] > 0)                        # every drawn literal occurs
    b3 = CF.random_bank(c.tok, c.doc_start, c.kind_map, "masked", 12, 3000, seed=2)
    assert set(np.unique(b3.n_lit).tolist()) == {1, 2, 3} and abs(np.mean(b3.n_lit) - 2.0) < 0.1
    off, _ = CF._col_tables("masked")
    for j in range(b3.n):
        cols = b3.col[j, : b3.n_lit[j]]
        assert np.all(cols >= 0) and len(set(off[cols].tolist())) == b3.n_lit[j]   # distinct offsets
        assert np.all(b3.col[j, b3.n_lit[j]:] == -1)
    again = CF.random_bank(c.tok, c.doc_start, c.kind_map, "masked", 12, 3000, seed=2)
    other = CF.random_bank(c.tok, c.doc_start, c.kind_map, "masked", 12, 3000, seed=3)
    assert np.array_equal(again.col, b3.col) and np.array_equal(again.sym, b3.sym) and not np.array_equal(other.sym, b3.sym)


def test_random_bank_features_equal_the_pure_python_reference(tmp_path):
    c = toy_corpus(vocab=15, n_docs=4, doc_len=50, seed=6)
    b = CF.random_bank(c.tok, c.doc_start, c.kind_map, "masked", 15, 300, seed=4)
    b.save(tmp_path / "rb.npz")
    b2 = CF.RandomBank.load(tmp_path / "rb.npz")
    assert np.array_equal(b2.col, b.col) and b2.info == b.info
    single = CF.literal_single_counts(np, c.tok, c.doc_start, c.kind_map, "masked", 15, chunk=37)
    ref_single = np.zeros_like(single)
    for (l1, l2), n in K.gplus_bruteforce(c.tok, c.doc_start, c.kind_map, "masked", 15).items():
        if l2 == 0xFFFFFFFF:
            from tmlm.engine.config import lit_unpack
            seg, sym, _ = lit_unpack(int(l1))
            o = seg - 8 if seg < 8 else seg - 7
            kinds = [k for k in range(4) if (k == 0 and sym < 15) or (k > 0 and 15 + (0 if k == 1 else 16 if k == 2 else 272) <= sym < 15 + (16 if k == 1 else 272 if k == 2 else 4368))]
            if sym == 15 + 4369:        # SYM_EDGE: kind-0 column
                kinds = [0]
            for k in kinds:
                ref_single[K.columns("masked").index((o, k)), sym] += n
    assert np.array_equal(single, ref_single)
    firer = CF.RandomBankFirer(np, b, single, c.kind_map)
    d = toy_corpus(vocab=15, n_docs=6, doc_len=17, seed=8)
    sd = CF.random_bank_doc_counts(np, firer, d.tok, d.doc_start, chunk=20)
    bf = CF.random_bank_bruteforce(b, d.tok, d.doc_start, c.kind_map, range(int(d.tok.shape[0])))
    ds = np.asarray(d.doc_start)
    for doc in range(6):
        exp: dict = {}
        for q in range(int(ds[doc]), int(ds[doc + 1])):
            for j in bf[q]:
                exp[j] = exp.get(j, 0) + 1
        row = sd.rows(np.asarray([doc]))
        assert dict(zip(row.ids.tolist(), row.cnt.tolist())) == exp
    fc = CF.random_bank_fire_counts(np, firer, d.tok, d.doc_start, chunk=11)
    assert fc.tolist() == np.bincount(np.concatenate([np.asarray(x, np.int64) for x in bf] + [np.zeros(0, np.int64)]), minlength=b.n).tolist()


# ====================================================================== (f) Q_int and Q_con
def test_npmi_group_coherence_top_words_and_intrusion():
    tok = np.array([1, 2, 3, 1, 2, 4, 5, 1, 2, 0, 3, 3], np.uint16)
    ds = np.array([0, 9, 12], np.uint32)
    X = CF.cooc_window(tok, ds, 6, window=3)
    ref = np.zeros((6, 6), np.int64)
    for p in range(12):
        d = 0 if p < 9 else 1
        for q in range(12):
            if p != q and abs(p - q) <= 2 and (0 if q < 9 else 1) == d:
                ref[tok[p], tok[q]] += 1
    assert np.array_equal(X, ref)
    npmi = CF.NPMI(X)
    M = X.sum()
    pa = X.sum(axis=1) / M
    v = npmi(np.array([1]), np.array([2]))[0]
    pab = X[1, 2] / M
    assert v == pytest.approx(np.log(pab / (pa[1] * pa[2])) / -np.log(pab))
    assert npmi(np.array([4]), np.array([0]))[0] == -1.0 and npmi(np.array([3]), np.array([3]))[0] == 1.0
    coh = CF.group_coherence(np.array([[1, 2, 3], [1, 2, -1], [4, -1, -1]]), npmi)
    assert coh[0] == pytest.approx(np.mean([npmi(np.array([a]), np.array([b]))[0] for a, b in ((1, 2), (1, 3), (2, 3))]))
    assert coh[1] == pytest.approx(v) and np.isnan(coh[2])
    # top words: word-like filter, min count, PPMI ranking, ties to the lower id
    O = np.array([[10, 0, 5, 5, 1, 0], [0, 8, 8, 0, 0, 2], [1, 1, 1, 1, 1, 1]], np.int64)
    ok = np.array([False, True, True, True, True, True])
    g, q = CF.top_words_ppmi(O, np.array([0, 1, 2]), ok, 2, 2)
    Nc = O.sum(axis=0).astype(float)
    Z = (Nc[Nc > 0] ** 0.75).sum()
    pm = lambda j, c: max(0.0, np.log(O[j, c]) + np.log(Z) - np.log(O[j].sum()) - 0.75 * np.log(Nc[c]))  # noqa: E731
    exp0 = sorted([c for c in range(6) if ok[c] and O[0, c] >= 2], key=lambda c: (-pm(0, c), c))[:2]
    assert g[0].tolist() == exp0 and q.tolist() == [True, True, False] and g[2].tolist() == [-1, -1]
    # intrusion: two coherent groups whose words never meet the other group's
    tok2 = np.array(([1, 2, 3, 4, 5] * 20) + ([6, 7, 8, 9, 10] * 20), np.uint16)
    X2 = CF.cooc_window(tok2, np.array([0, 100, 200], np.uint32), 12, window=5)
    it = CF.intruder_task(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), CF.NPMI(X2), seed=1, n_keep=3)
    assert it["accuracy"] == 1.0 and it["n_groups_scored"] == 2 and it["chance"] == 0.25


def test_occurrences_rounds_and_word_codes_equal_per_occurrence_brute_force(tmp_path):
    c = toy_corpus(vocab=30, n_docs=6, doc_len=60, seed=12)
    tok = np.asarray(c.tok).astype(np.int64)
    wid = np.zeros(tok.shape[0], np.uint32)          # words: tokens 3 and 4 glue to the next token ("two-piece words")
    w = 0
    for i in range(tok.shape[0]):
        if i > 0 and not (tok[i - 1] in (3, 4) and np.searchsorted(np.asarray(c.doc_start), i, side="right") == np.searchsorted(np.asarray(c.doc_start), i - 1, side="right")):
            w += 1
        wid[i] = w
    starts, lens = CF.word_runs(wid)
    assert lens.sum() == tok.shape[0] and np.all(np.diff(starts) > 0)
    targets = {"a": [5], "b": [3, 6], "c": [4, 1], "z": [29, 29, 29]}
    occ_all = CF.find_occurrences(tok, starts, lens, targets)
    for name, ids in targets.items():
        exp = [int(s) for s, L in zip(starts, lens) if L == len(ids) and tok[s:s + L].tolist() == ids]
        assert occ_all[name].tolist() == exp
    assert CF.select_occurrences(np.arange(100), 7).tolist() == sorted(np.arange(100)[np.argsort(P.mix64_np(np.arange(100, dtype=np.uint64)))[:7]].tolist())
    occ = CF.build_occurrences(tok, wid, targets, max_occ=4)
    assert occ.n_total.tolist() == [len(occ_all[k]) for k in targets] and np.all(np.bincount(occ.widx, minlength=4) <= 4)
    ds = np.asarray(c.doc_start, np.int64)
    dop = np.repeat(np.arange(ds.shape[0] - 1), np.diff(ds))
    rounds = CF.occurrence_rounds(occ.start, dop)
    for r in np.unique(rounds):                       # at most one occurrence per document per round
        d = dop[occ.start[rounds == r]]
        assert np.unique(d).shape[0] == d.shape[0]
    cfg = P.engine_config("A0", 1, 30, 64, batch=16, sweep_period=8, n_bag_k=2, dead_sweeps=2)
    eng = CpuEngine(cfg, corpus=c)
    for e in range(2):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    wj, jj, cc = CF.word_codes_api(eng, occ, tok.shape[0], rounds)
    ref: dict = {}
    for i in range(occ.start.shape[0]):               # brute force: each occurrence alone hidden
        hidden = np.zeros(tok.shape[0], np.uint8)
        hidden[occ.start[i]:occ.start[i] + occ.length[i]] = 1
        u = set()
        for q in range(int(occ.start[i]), int(occ.start[i] + occ.length[i])):
            u.update(int(x) for x in eng.fired_clauses(0, q, hidden))
        for j in u:
            ref[(int(occ.widx[i]), j)] = ref.get((int(occ.widx[i]), j), 0) + 1
    assert dict(zip(zip(wj.tolist(), jj.tolist()), cc.tolist())) == ref and len(ref) > 0
    X = CF.ppmi_rows(wj, jj, cc, 4, 64)
    cos = CF.pair_cosines_sparse(X, np.array([0, 1]), np.array([0, 2]))
    assert cos[0] == pytest.approx(1.0) or X[0].nnz == 0
    E = np.random.default_rng(0).normal(size=(30, 5))
    Wv = CF.word_vectors_mean(E, [[5], [3, 6]])
    assert np.allclose(Wv[1], (E[3] + E[6]) / np.linalg.norm(E[3] + E[6]))
    sets = {"s": {"a": np.array([0, 0, 1]), "b": np.array([1, 2, 2]), "gold": np.array([1.0, 2.0, 3.0])}}
    res = CF.q_con_from_similarity(sets, lambda a, b: np.array([0.1, 0.2, 0.3]))
    assert res["s"] == pytest.approx(1.0) and res["q_con"] == pytest.approx(1.0)


def test_estimate_rows_restrict_columns_and_lean_matrices():
    import scipy.sparse as sp

    # an estimate row carries intervals and never MET or NOT MET (A9 item 5)
    e = CF.estimate_row("x", _boot(3.0, 1.0, 5.0), {"lo": 0.5, "hi": 5.5})
    assert e["verdict"] == CF.ESTIMATE and e["complete"] and e["verdict"] not in (CF.MET, CF.NOT_MET)
    assert not CF.estimate_row("x", _boot(3.0, 1.0, 5.0, n=2), None)["complete"]
    # restrict_columns: the COO entries of the kept columns, renumbered in the order of `cols`
    rng = np.random.default_rng(4)
    w, j, c = rng.integers(0, 5, 200), rng.integers(0, 40, 200), rng.integers(1, 9, 200)
    cols = np.array([33, 2, 17, 5, 39])
    wd, jd, cd = CF.restrict_columns(w, j, c, cols)
    exp = [(int(a), list(cols).index(int(b)), int(x)) for a, b, x in zip(w, j, c) if int(b) in set(cols.tolist())]
    assert list(zip(wd.tolist(), jd.tolist(), cd.tolist())) == exp
    assert CF.restrict_columns(w, j, c, np.zeros(0, np.int64))[0].shape == (0,)
    with pytest.raises(ValueError):
        CF.restrict_columns(w, j, c, np.array([1, 1]))
    # rows_csr: bit for bit the R1 path sd.rows(rows).to_csr('log1p', cols), any chunk size, with and without cols
    n_doc, n_cols = 57, 300
    lens = rng.integers(0, 40, n_doc)
    ip = np.zeros(n_doc + 1, np.int64)
    ip[1:] = np.cumsum(lens)
    ids = np.concatenate([np.sort(rng.choice(n_cols, size=k, replace=False)) for k in lens]).astype(np.uint32)
    cnt = rng.integers(1, 30, ip[-1]).astype(np.uint32)
    sd = F.SparseDocs(ip, ids, cnt, n_cols)
    rows = np.array([5, 3, 3, 56, 0, 17, 22, 9, 40, 41, 2], np.int64)
    for cols_ in (None, rng.choice(n_cols, 50, replace=False)):
        B = sd.rows(rows).to_csr("log1p", cols=cols_)
        for chunk in (1, 3, 4096):
            A = S.rows_csr(sd, rows, "log1p", cols_, chunk=chunk)
            assert A.shape == B.shape and np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
            assert np.array_equal(A.data.view(np.uint64), B.data.view(np.uint64))
            assert (A != B).nnz == 0 and isinstance(A, sp.csr_matrix)


# ====================================================================== (g) tiny end-to-end pipeline
V = 40
END = 39
TASKS = ["agnews", "sst2", "trec"]


def _tiny_cj(tmp: Path) -> dict:
    cj = copy.deepcopy(CJ)
    p = cj["params"]
    p.update(vocab=V, n_clauses=64, epochs=2, young_fraction=0.3, draws=[0, 1], budgets=[8], c_grid=[0.1, 1.0],
             lr_max_iter=200, tm_grid=[{"clauses": 20, "T": 10, "s": 3.0}], tm_epochs=1, tm_ppmi_bits=4, tm_budgets=[8],
             dim_matched=16, forward_chunk=64, gplus_chunk_positions=4096, gplus_feature_chunk=64)
    p["ppmi"] = {"window": 3, "dim": 6, "power": 0.5}
    p["young_ladder"] = {**p["young_ladder"], "abs_batches": [1, 2], "percent_of_epoch1": [50]}
    p["diagnostics"] = {**p["diagnostics"], "class_sum_rows": 30, "fired_chunk_rows": 7, "engine": "cpu"}
    for plan in [v for v in cj["probe_plans"].values() if isinstance(v, list)]:
        for blk in plan:
            for clf in ("lr", "mnb", "nbsvm"):
                if clf in blk:
                    blk[clf] = [8]
            if "tm" in blk:
                blk["tm"]["budgets"] = [8]
    cj["doc_set_reference"] = {}
    for spec in cj["configurations"]["arms"].values():      # the tiny runs train 2 epochs: the final bank is epoch2
        spec["features_checkpoints"] = spec["features_checkpoints"].replace("epoch4", "epoch2")
        spec["qpred_checkpoints"] = spec["qpred_checkpoints"].replace("epoch4", "epoch2")
    for blk in cj["probe_plans"]["arm"] + cj["probe_plans"]["twin"]:
        if isinstance(blk["sets"], list):
            blk["sets"] = [x.replace("epoch4", "epoch2") for x in blk["sets"]]
        if "tm" in blk:
            blk["tm"]["sets"] = [x.replace("epoch4", "epoch2") for x in blk["tm"]["sets"]]
    np.save(tmp / "cbow.npy", np.random.default_rng(1).normal(size=(V, 8)).astype(np.float32))
    cj["baselines"]["cbow"] = {"file": str(tmp / "cbow.npy"), "dim": 8}
    cj["baselines"]["r1_reference"] = {"gplus": str(tmp / "none"), "ppmi_svd": str(tmp / "none.npy")}
    for item in S.SPEEDUPS:        # the config's enabled speedups, with stand-in clean verification runs (no results/ access)
        spec = cj["speedups"][item]
        if spec["enabled"]:
            d = tmp / f"verified_{item}"
            d.mkdir(exist_ok=True)
            (d / "run.json").write_text(json.dumps({"exit_status": 0}))
            (d / "metrics.json").write_text(json.dumps({"failed_checks": {"value": []}}))
            spec["verified_by"] = [str(d)]
    cj["random_bank"].update(chunk_positions=64, check_clauses=16)
    cj["qint"].update(top_words=3, min_count=1, window=3)
    cj["statistics"]["n_boot"] = 200
    return cj


def _doc_tokens(task_code: int, part: int, i: int, y: int) -> list[int]:
    """A toy document with label-dependent tokens, a function of (task, dev or pool, index) only, so the dev block is
    the same in the R1 and the full-pool doc sets."""
    rng = np.random.default_rng([task_code, part, i])
    return list(rng.integers(1 + 4 * int(y), 5 + 4 * int(y), size=8)) + list(rng.integers(1, END, size=6)) + [END]


def _toy_docs(full: bool = False):
    """The dev doc set: per task a dev carve and the labelled union of budget 8, draws 0 and 1 (with `full`: the whole
    pool)."""
    rng = np.random.default_rng(0)
    layout, labels, docs = {}, {}, []
    base = 0
    for task in TASKS:
        k = D.TASKS[task]["n_classes"]
        code = D.TASKS[task]["code"]
        pool_y = rng.integers(0, k, size=30 * k)
        dev_y = rng.integers(0, k, size=6 * k)
        union = PR.sample_union(pool_y, [8] + (["full"] if full else []), [0, 1], code)
        docs += [_doc_tokens(code, 0, i, y) for i, y in enumerate(dev_y)] + [_doc_tokens(code, 1, int(i), pool_y[i]) for i in union]
        n = dev_y.shape[0] + union.shape[0]
        layout[task] = {"dev": [base, base + dev_y.shape[0]], "lab": [base + dev_y.shape[0], base + n], "lab_pool_rows": union.tolist()}
        base += n
        labels[task] = {"pool_y": pool_y.tolist(), "dev_y": dev_y.tolist()}
    tok = np.concatenate([np.asarray(x, np.uint16) for x in docs])
    ds = np.zeros(len(docs) + 1, np.uint32)
    ds[1:] = np.cumsum([len(x) for x in docs])
    return F.TokenDocs(tok=tok, doc_start=ds, end_id=END), layout, labels


class World:
    """Stage runner: a fresh run directory per stage and the metrics.json that run_logged.py would write."""

    def __init__(self, tmp: Path, monkeypatch, cj: dict) -> None:
        self.tmp, self.cj, self.n = tmp, cj, 0
        self.cur: Path | None = None
        monkeypatch.setattr(C, "out_dir", lambda stage: self.cur)

    def run(self, name: str, fn, args, cj: dict, *extra) -> tuple[Path, S.Rec]:
        """fn(args, cj, rec, *extra), the signature of every stage function."""
        self.n += 1
        self.cur = self.tmp / f"{self.n:02d}_{name}"
        self.cur.mkdir()
        rec = S.Rec(False)
        fn(args, cj, rec, *extra)
        rec.m("failed_checks", rec.failed)
        from tmlm.common import runlog
        (self.cur / "metrics.json").write_text(json.dumps(runlog.validate_metrics(rec.metrics)))   # what run_logged accepts
        (self.cur / "run.json").write_text(json.dumps({"exit_status": 1 if rec.failed else 0}))
        return self.cur, rec

    def refused(self, fn, args, cj: dict, *extra, exc=SystemExit) -> None:
        """The stage must stop with `exc` (SystemExit: a refusal of the stage) in a scratch directory."""
        self.n += 1
        self.cur = self.tmp / f"{self.n:02d}_refused"
        self.cur.mkdir()
        with pytest.raises(exc):
            fn(args, cj, S.Rec(False), *extra)


def test_every_stage_end_to_end_on_toy_data(arms, tmp_path, monkeypatch):
    import tmlm.data.corpus_store as CS
    import tmlm.data.evalsets as EV

    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(P, "F2_M", 60)
    cj = _tiny_cj(tmp_path)
    full_train = toy_corpus(vocab=V, n_docs=60, doc_len=200, seed=5)
    prefix = P.slice_prefix(full_train, 40 * 200)
    docs, layout, labels = _toy_docs(False)
    docs_full, layout_full, labels_full = _toy_docs(True)
    monkeypatch.setattr(R, "load_prefix", lambda p_, rec: prefix)
    monkeypatch.setattr(R, "doc_set", lambda p_, rec, full: (docs_full, layout_full, labels_full) if full else (docs, layout, labels))
    monkeypatch.setattr(R, "dataset_hashes", lambda d: None)
    val = toy_corpus(vocab=V, n_docs=5, doc_len=80, seed=9)
    targets = np.arange(3, val.tok.shape[0], 5, dtype=np.int64)
    hidden = np.zeros(val.tok.shape[0], np.uint8)
    hidden[targets] = 1
    vdir = tmp_path / "vmeta"
    vdir.mkdir()
    (vdir / "meta.json").write_text(json.dumps({"sha256": {"tok.npy": "x"}}))
    monkeypatch.setattr(CS, "load_split", lambda corpus, vocab, split, **kw: val if split == "validation" else full_train)
    monkeypatch.setattr(CS, "split_dir", lambda corpus, vocab, split, root=None: vdir)
    monkeypatch.setattr(EV, "load_evalset", lambda corpus, vocab, split, variant, **kw: {"targets": targets, "hidden": hidden, "meta": {"sha256_combined": "e"}})
    monkeypatch.setattr(S.DevStore, "kind_map", lambda self: np.asarray(prefix.kind_map))
    monkeypatch.setattr(CF, "preflight_references", lambda tasks, root=None: {t: {"stub": True} for t in tasks})
    monkeypatch.setattr(S, "preflight_host", lambda tasks, refs: {"passed": True, "stub": True})   # no network in tests
    # the counted loader: a spy that records REAL access lines (temporary log) and returns toy texts (space-separated ids)
    calls: list[tuple[str, str]] = []
    rng = np.random.default_rng(3)

    def spy_load(task, cid, detail, root=None, allow_download=True, reaccess_reason=None):
        CF.record_confirm_access(CF.ACCESS_TAG, cid, task, detail, None, reaccess_reason)
        calls.append((cid, task))
        if (cid, task) == ("A3R_seed1", "sst2"):          # another lane accesses while this configuration reads
            CF.record_confirm_access(CF.ACCESS_TAG, "A1_seed1", "trec", "a concurrent lane")
        k = D.TASKS[task]["n_classes"]
        y = rng.integers(0, k, size=5 * k)
        texts = [" ".join(str(t) for t in list(rng.integers(1 + 4 * int(c), 5 + 4 * int(c), size=8)) + list(rng.integers(1, END, size=6))) for c in y]
        return CF.ConfirmSet(task, texts, y, {"access": {"before": 0, "after": 1}, "sha256": "t"})

    def toy_encode(texts):
        parts = [[int(x) for x in t.split()] + [END] for t in texts]
        tok = np.concatenate([np.asarray(x, np.uint16) for x in parts])
        ds = np.zeros(len(parts) + 1, np.uint32)
        ds[1:] = np.cumsum([len(x) for x in parts])
        return F.TokenDocs(tok=tok, doc_start=ds, end_id=END)

    monkeypatch.setattr(CF, "load_confirm", spy_load)
    monkeypatch.setattr(S, "make_encoder", lambda vocab: toy_encode)
    w = World(tmp_path, monkeypatch, cj)
    ns = argparse.Namespace
    cargs = (lambda: ns(engine="cpu", reaccess_reason=None))
    # ---- train, census, features, probe for the winner, its twin and A3R (seed 1)
    runs = {}
    for arm in ("A3", "A3T", "A3R"):
        tr, rec = w.run(f"train_{arm}", S.stage_train, ns(arm=arm, seed=1, engine="cpu"), cj)
        assert rec.failed == [], rec.failed
        assert rec.metrics["confirm.config_id"]["value"] == f"{arm}_seed1" and rec.metrics["confirm.config_id"]["confirmatory"]
        _, crec = w.run(f"census_{arm}", S.CA.stage_census, ns(), cj, tr)
        assert crec.failed == [], crec.failed
        fr, frec = w.run(f"features_{arm}", S.stage_features, ns(checkpoints=None, engine="cpu"), cj, tr)
        assert frec.failed == [], frec.failed
        runs[arm] = (tr, fr)
    names = {q.stem[len("features_"):] for q in runs["A3"][1].glob("features_*.npz")}
    assert {"c0y", "epoch2"} <= names and any(n_.startswith("young_b") for n_ in names)
    assert {q.stem[len("features_"):] for q in runs["A3T"][1].glob("features_*.npz")} == {"epoch2"}
    assert json.loads((runs["A3R"][0] / "metrics.json").read_text())["arm"]["value"]["random_kind_map"] is True
    # ---- confirm BEFORE the dev probe: refused, no access
    fake_probe = tmp_path / "fake_probe"
    fake_probe.mkdir()
    (fake_probe / "run.json").write_text(json.dumps({"exit_status": 1}))
    (fake_probe / "metrics.json").write_text(json.dumps({"failed_checks": {"value": ["x"]}}))
    w.refused(S.stage_confirm, cargs(), cj, fake_probe)
    assert calls == []
    # ---- dev probes (freeze); the full-pool rows of A3 (A9 item 8)
    probes = {}
    for arm in ("A3", "A3T", "A3R"):
        pr, prec = w.run(f"probe_{arm}", S.stage_probe, ns(), cj, runs[arm][1])
        assert prec.failed == [] and prec.metrics["probe.complete"]["value"], prec.failed
        probes[arm] = pr
    rows = [json.loads(x) for x in open(probes["A3"] / "cells.jsonl")]
    assert {r["set"] for r in rows} >= {"epoch2.all", "epoch2.dim", "c0y.all", "young_b1.dim"} and {r["clf"] for r in rows} == {"lr", "tm"}
    assert all(r["split"] == "dev" for r in rows)
    w.refused(S.stage_fullfeat, ns(train_run=str(runs["A3T"][0]), features_run=str(runs["A3T"][1]), common_run=None, engine="cpu"), cj)
    ff, ffrec = w.run("fullfeat_A3", S.stage_fullfeat, ns(train_run=str(runs["A3"][0]), features_run=str(runs["A3"][1]), common_run=None, engine="cpu"), cj)
    assert ffrec.failed == [], ffrec.failed
    assert ffrec.metrics["confirm.doc_set"]["value"] == "full" and {q.stem for q in ff.glob("features_*.npz")} == {"features_epoch2", "features_c0y"}
    pf, pfrec = w.run("probefull_A3", S.stage_probe, ns(), cj, ff)
    assert pfrec.failed == [] and pfrec.metrics["probe.complete"]["value"] and pfrec.metrics["probe.doc_set"]["value"] == "full", pfrec.failed
    frows = [json.loads(x) for x in open(pf / "cells.jsonl")]
    assert {(r["set"], r["budget"], r["draw"]) for r in frows} == {(s_, "full", 0) for s_ in ("epoch2.all", "epoch2.dim", "c0y.all", "c0y.dim")}
    assert {r["n_labelled"] for r in frows if r["task"] == "trec"} == {30 * 6} and calls == []
    # ---- refusals before any access: the full-pool run missing for A3, present for A3T, a changed model file, a dev
    # prediction that the loaded model does not reproduce
    w.refused(S.stage_confirm, cargs(), cj, probes["A3"])
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"], pf)
    mf = S.model_file(probes["A3T"], "trec")
    keep = mf.read_bytes()
    mf.write_bytes(keep + b"x")
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"])
    mf.write_bytes(keep)
    real_predict = CF.frozen_predict
    monkeypatch.setattr(CF, "frozen_predict", lambda f, X: (real_predict(f, X) + 1) % 2)
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"])
    monkeypatch.setattr(CF, "frozen_predict", real_predict)
    monkeypatch.setattr(CF, "preflight_references", lambda tasks, root=None: (_ for _ in ()).throw(CF.ConfirmPreflightFailed("no reference")))
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"])
    monkeypatch.setattr(CF, "preflight_references", lambda tasks, root=None: {t: {"stub": True} for t in tasks})
    # the post-access path is run on dev documents in the preflight: a test-time code that differs from the dev code,
    # or an error of that path, refuses before the access
    real_codes = S.confirm_sparse_codes

    def shifted_codes(ctx, docs_, cj_, args_, rec_, tag="test"):
        sp_ = real_codes(ctx, docs_, cj_, args_, rec_, tag)
        return {k_: F.SparseDocs(v.indptr, ((v.ids.astype(np.int64) + 1) % v.n_cols).astype(np.uint32), v.cnt, v.n_cols) for k_, v in sp_.items()}

    monkeypatch.setattr(S, "confirm_sparse_codes", shifted_codes)
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"])
    monkeypatch.setattr(S, "confirm_sparse_codes", lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("device error")))
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"])
    monkeypatch.setattr(S, "confirm_sparse_codes", real_codes)
    monkeypatch.setattr(S, "preflight_host", lambda tasks, refs: {"passed": False, "trec": {"error": "unreachable"}})
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"])
    monkeypatch.setattr(S, "preflight_host", lambda tasks, refs: {"passed": True, "stub": True})
    assert calls == [] and not EV.access_log_path().exists()
    # ---- confirm: one counted access per task per configuration, checked per (configuration, task)
    conf = {}
    for arm, extra in (("A3", (pf,)), ("A3T", ()), ("A3R", ())):
        cr, crec = w.run(f"confirm_{arm}", S.stage_confirm, cargs(), cj, probes[arm], *extra)
        assert crec.failed == [], crec.failed
        assert calls[-3:] == [(f"{arm}_seed1", t) for t in TASKS]
        assert crec.metrics["check.confirm.one_counted_access_per_configuration_and_task"]["value"] is True
        dry = crec.metrics["confirm.preflight"]["value"]["dry_run"]
        assert dry["passed"] and dry["cells"] > 0 and dry["n_mismatches"] == 0
        conf[arm] = cr
        trows = [json.loads(x) for x in open(cr / "cells.jsonl")]
        n_dev = len(open(probes[arm] / "cells.jsonl").readlines()) + (len(frows) if arm == "A3" else 0)
        assert len(trows) == n_dev and all(r["dev_prediction_reproduced"] for r in trows)
    assert {r["doc_set"] for r in map(json.loads, open(conf["A3"] / "cells.jsonl"))} == {"r1", "full"}
    assert len(calls) == 9 and len(set(calls)) == 9
    # the concurrent access during A3R's read did not disturb its per-configuration count; a repeat is refused by the
    # loader after the preflight passed (no access recorded)
    n_log = len(EV.access_log_path().read_text().splitlines())
    assert n_log == 10
    w.refused(S.stage_confirm, cargs(), cj, probes["A3T"], exc=CF.ConfirmAccessRefused)
    assert len(calls) == 9 and len(EV.access_log_path().read_text().splitlines()) == n_log
    # ---- common (with its full-pool rows) and random bank configurations; S8: the random bank's doc set is checked
    cm, cmrec = w.run("common", S.stage_common, ns(engine="cpu"), cj)
    assert cmrec.failed == [], cmrec.failed
    cj["doc_set_reference"] = {"run": str(cm)}
    rb, rbrec = w.run("randbank", S.stage_randbank, ns(engine="cpu", seed=1), cj)
    assert rbrec.failed == [] and rbrec.metrics["check.docs.equal_to_r1_features_run"]["value"] is True, rbrec.failed
    cj["doc_set_reference"] = {}
    pc, pcrec = w.run("probe_common", S.stage_probe, ns(), cj, cm)
    assert pcrec.failed == [], pcrec.failed
    fc, fcrec = w.run("fullfeat_common", S.stage_fullfeat, ns(train_run=None, features_run=None, common_run=str(cm), engine="cpu"), cj)
    assert fcrec.failed == [], fcrec.failed
    pfc, pfcrec = w.run("probefull_common", S.stage_probe, ns(), cj, fc)
    assert pfcrec.failed == [] and {json.loads(x)["set"] for x in open(pfc / "cells.jsonl")} == {"gplus.all", "gplus.dim", "bow"}
    n0 = len(calls)
    cr, crec = w.run("confirm_common", S.stage_confirm, cargs(), cj, pc, pfc)
    assert crec.failed == [] and calls[n0:] == [("common", t) for t in TASKS], crec.failed
    assert crec.metrics["confirm.preflight"]["value"]["dry_run"]["passed"]          # bow, cluster, ppmi, cbow, G+ (r1 and full)
    probes["common"], conf["common"] = pc, cr
    pr, prec = w.run("probe_rand", S.stage_probe, ns(), cj, rb)
    assert prec.failed == [], prec.failed
    moved = rb / "random_bank.npz.moved"
    (rb / "random_bank.npz").rename(moved)                          # a missing artefact: refused before the access
    w.refused(S.stage_confirm, cargs(), cj, pr)
    moved.rename(rb / "random_bank.npz")
    n0 = len(calls)
    cr, crec = w.run("confirm_rand", S.stage_confirm, cargs(), cj, pr)
    assert crec.failed == [] and calls[n0:] == [("rand_seed1", t) for t in TASKS], crec.failed
    crow = [json.loads(x) for x in open(probes["common"] / "cells.jsonl")]
    assert {r["clf"] for r in crow} == {"lr", "tm", "mnb", "nbsvm"} and {"gplus.dim", "bow", "cluster", "ppmi", "cbow"} <= {r["set"] for r in crow}
    # ---- Q_int on the winner run: both references; no access by this process
    monkeypatch.setattr(S, "vocab_texts", lambda vocab: [" " + "abcdefghijklmnopqrstuvwxyz"[i % 26] * 3 + "abcdefghijklmnop"[i // 26] for i in range(vocab)])
    cj["qint"]["checkpoints"] = ["epoch2", "c0y"]
    qi, qirec = w.run("qint", S.stage_qint, ns(engine="cpu"), cj, runs["A3"][0], runs["A3"][1], probes["A3"])
    assert qirec.failed == [], qirec.failed
    z = np.load(qi / "qint_groups.npz")
    assert {"epoch2.train_remainder.npmi", "c0y.train_remainder.npmi", "epoch2.validation.npmi"} <= set(z.files)
    assert qirec.metrics["qint.reference.train_remainder"]["value"]["tokens"] == int(full_train.tok.shape[0] - prefix.tok.shape[0])
    # ---- Q_con: dev first, then exactly one counted MEN access, then MEN in full; refused when the dev stage failed
    order, men_sizes = [], []

    def fake_eval(label, pairs_by_ds, *a, **k):
        order.append(label)
        men_sizes.append(len(pairs_by_ds["men"]))
        assert k.get("dims") is not None or (len(a) > 9 and a[9] is not None)
        return {n_: {"q_con": 0.1} for n_ in ("epoch2", "epoch2.dim", "c0y", "c0y.dim", "ppmi_svd", "cbow")}

    monkeypatch.setattr(S, "qcon_eval", fake_eval)
    monkeypatch.setattr(CF, "men_test_reference", lambda root=None: {"stub": True})

    def men_counted(cid, detail, root=None, reaccess_reason=None):
        CF.record_confirm_access(CF.MEN_TAG, cid, "men_test", detail, None, reaccess_reason)
        order.append(f"men:{cid}")
        return [("a", "b", 1.0), ("c", "d", 2.0)], {"access": {}}

    monkeypatch.setattr(CF, "load_men_test", men_counted)
    monkeypatch.setattr(CS, "load_word_id", lambda corpus, vocab, split, root=None: np.arange(full_train.tok.shape[0], dtype=np.uint32))
    monkeypatch.setattr(S, "word_tokenizer", lambda vocab: (lambda w_: [1]))
    cj["qcon"]["checkpoints"] = ["epoch2", "c0y"]
    import tmlm.ssl.concepts as KC
    monkeypatch.setattr(KC, "parse_men_lemma", lambda path: [("m", "n", 1.0)] * 5)       # MEN dev (the test pairs: men_counted)
    w.refused(S.stage_qcon, cargs(), cj, runs["A3"][0], fake_probe, cm)
    assert order == []
    qc, qcrec = w.run("qcon", S.stage_qcon, cargs(), cj, runs["A3"][0], probes["A3"], cm)
    assert qcrec.failed == [], qcrec.failed
    assert order == ["dev", "men:qcon_A3_seed1", "confirm"] and men_sizes == [5, 7]      # MEN in full = dev + test pairs
    assert qcrec.metrics["qcon.men_full_pairs"]["value"] == {"dev": 5, "test": 2, "full": 7}
    audit = S.access_audit()
    assert audit["every_pair_once"] and audit["per_config_task"]["config=qcon_A3_seed1; task=men_test"] == 1
    assert audit["total_confirmatory_accesses"] == 15 + 1 + 1       # 5 configurations x 3 tasks, the concurrent lane, MEN


# ====================================================================== report on synthetic cells
def _synthetic_rows(effect: dict, seeds=(1, 2, 3), draws=(0, 1, 2, 3, 4), budget=32) -> list[dict]:
    rng = np.random.default_rng(7)
    rows = []
    for (arm, seed, fs, clf), base in effect.items():
        for split in ("dev", "test"):
            for d in draws:
                for task in TASKS:
                    acc = base + rng.normal(0, 0.002)
                    rows.append({"arm": arm, "seed": seed, "set": fs, "clf": clf, "task": task, "budget": budget, "draw": d, "split": split, "acc": acc})
    return rows


def test_report_computes_every_verdict_mechanically(arms, tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "data"))
    cj = copy.deepcopy(CJ)
    cj["statistics"]["n_boot"] = 400
    eff = {}
    for s in (1, 2, 3):
        for code in ("all", "dim"):
            eff[("A3", s, f"epoch4.{code}", "lr")] = 0.60
            eff[("A3", s, f"c0y.{code}", "lr")] = 0.55
            for r in ("young_b77", "young_b196", "young_b255"):
                eff[("A3", s, f"{r}.{code}", "lr")] = 0.56
            eff[("A3T", s, f"epoch4.{code}", "lr")] = 0.595          # twin within 0.5 points: no literal-learning credit
            eff[("A2", s, f"epoch4.{code}", "lr")] = 0.58
            eff[("A0", s, f"epoch4.{code}", "lr")] = 0.570
            eff[("A1", s, f"epoch4.{code}", "lr")] = 0.575            # A1 - A0 = 0.5 point < 1.0
            eff[("A0", s, f"young_b196.{code}", "lr")] = 0.565
            for a_ in ("A2", "A1", "A6"):                              # every arm of a8.young_any_arm has young banks
                eff[(a_, s, f"c0y.{code}", "lr")] = 0.55
            eff[("A6", s, f"epoch4.{code}", "lr")] = 0.54
            eff[("A3R", s, f"epoch4.{code}", "lr")] = 0.53
        eff[("A3", s, "epoch4.dim", "tm")] = 0.585                      # TM - LR = -1.5 points >= -2.0
    for code in ("all", "dim"):
        eff[("common", 0, f"gplus.{code}", "lr")] = 0.50
    eff[("common", 0, "bow", "lr")] = 0.57                              # winner - BoW = 3 points
    full = {("A3", s, fs, "lr"): v for s in (1, 2, 3) for fs, v in (("epoch4.all", 0.80), ("epoch4.dim", 0.79), ("c0y.all", 0.78), ("c0y.dim", 0.77))}
    full.update({("common", 0, "gplus.all", "lr"): 0.81, ("common", 0, "gplus.dim", "lr"): 0.80, ("common", 0, "bow", "lr"): 0.82})
    runs = tmp_path / "cells"
    runs.mkdir()
    with open(runs / "cells.jsonl", "w") as fh:
        for r in _synthetic_rows(eff) + _synthetic_rows(full, draws=(0,), budget="full"):
            fh.write(json.dumps(r) + "\n")
    qcons, qints = [], []
    for s in (1, 2, 3):
        d = tmp_path / f"qcon{s}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"confirm.config_id": {"value": f"A3_seed{s}"},
                                                    "qcon.summary": {"value": {"dev": {"epoch4": 0.3},
                                                                               "confirm": {"epoch4": 0.30 + 0.01 * s, "epoch4.dim": 0.25, "c0y": 0.20,
                                                                                           "c0y.dim": 0.24, "ppmi_svd": 0.28, "cbow": 0.25}}}}))
        qcons.append(str(d))
        d = tmp_path / f"qint{s}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"confirm.config_id": {"value": f"A3_seed{s}"}}))
        g = np.random.default_rng(s)
        np.savez(d / "qint_groups.npz", **{"epoch4.train_remainder.npmi": g.normal(0.10, 0.05, 300), "c0y.train_remainder.npmi": g.normal(0.05, 0.05, 280),
                                           "epoch4.validation.npmi": g.normal(0.0, 0.05, 300), "c0y.validation.npmi": g.normal(0.05, 0.05, 280)})
        qints.append(str(d))
    skipped = tmp_path / "skipped.txt"
    skipped.write_text("=== lane seed2 start\nSKIP/FAIL lane seed2 T: train A2 seed 2 failed\n")
    out = tmp_path / "report"
    out.mkdir()
    monkeypatch.setattr(C, "out_dir", lambda stage: out)
    rec = S.Rec(False)
    S.stage_report(argparse.Namespace(probe_runs=str(runs), confirm_runs="", qpred_runs="", qcon_runs=",".join(qcons),
                                      qint_runs=",".join(qints), census_runs="", skipped_file=str(skipped)), cj, rec)
    v = json.loads((out / "verdicts.json").read_text())
    assert v["winner"]["winner"] == "A3" and v["winner"]["dev_seeds_1_3"] == "A3"
    assert v["H17.1"]["verdict"] == "MET" and v["H17.1"]["boot"]["mean"] == pytest.approx(5.0, abs=0.2)   # 0.60 - max(0.55, 0.50)
    assert v["H17.1.dim_reported"]["decides"] is False and v["H17.1.dim_reported"]["verdict"] == "MET"
    assert v["H17.2"]["verdict"] == "MET" and v["H17.4"]["verdict"] == "NOT MET" and v["H17.11"]["verdict"] == "MET"
    assert v["H17.2.dim_reported"]["decides"] is False and v["H17.4.dim_reported"]["decides"] is False
    assert v["H17.11.vs_all_reported"]["decides"] is False
    assert v["H17.3"]["verdict"] == "MET" and v["H17.3.dim_reported"]["verdict"] == "NOT MET"      # dim: 0.25 - 0.24 < 0.05
    assert v["H17.12"]["verdict"] == "MET" and v["H17.12"]["name"] == "H17.12 (train_remainder)"
    assert v["H17.12.validation_reported"]["verdict"] == "NOT MET"                                  # the secondary reference differs
    assert v["A8_rows"]["ladder.all"]["verdict"] == "MET" and v["A8_rows"]["twin.all"]["verdict"] == "NOT MET"
    assert v["A8_literal_learning_credit"]["label"] == "no literal-learning credit"
    for k in ("minus_exposure_matched.all", "minus_exposure_matched.dim", "minus_best_young_any_arm.all", "minus_best_young_any_arm.dim"):
        assert v["A8_rows"][k]["verdict"] == CF.ESTIMATE                                             # A9 item 5: no MET / NOT MET
    assert v["A8_rows"]["minus_best_young_any_arm.all"]["inputs"]["best_young_per_seed"]["1"][0] == "A0"
    assert v["A8_rows"]["minus_exposure_matched.all"]["boot"]["mean"] == pytest.approx(4.0, abs=0.2)
    assert v["reported_controls"]["winner_minus_A3R.all"]["verdict"] == CF.ESTIMATE
    assert v["reported_controls"]["winner_minus_A3R.all"]["boot"]["mean"] == pytest.approx(7.0, abs=0.2)
    assert v["reported_controls"]["winner_minus_A6.dim"]["boot"]["mean"] == pytest.approx(6.0, abs=0.2)
    fp = v["full_pool"]
    assert fp["q_text"]["A3.epoch4.all.test"]["1"] == pytest.approx(0.80, abs=0.01) and fp["q_text"]["common.bow.test"] == pytest.approx(0.82, abs=0.01)
    assert fp["differences"]["winner_minus_bow.all"]["verdict"] == CF.ESTIMATE
    assert fp["differences"]["winner_minus_bow.all"]["t"]["mean"] == pytest.approx(-2.0, abs=0.3)
    assert fp["differences"]["winner_minus_max_c0y_gplus.all"]["inputs"]["units"]["1"][0] == pytest.approx(-1.0, abs=0.3)
    sb = v["side_by_side"]
    assert sb["decides"] == "all" and set(sb["rows"]["H17.1"]) == {"all", "dim"} and sb["rows"]["H17.3"]["dim"]["verdict"] == "NOT MET"
    assert v["chain_skips_and_failures"][1].endswith("train A2 seed 2 failed")
    a = rec.metrics["report.test_access_audit"]["value"]
    assert a["total_confirmatory_accesses"] == 0 and a["expected_pairs"] == 90 and len(a["missing_pairs"]) == 90


def test_report_rows_with_a_missing_control_are_incomplete(arms, tmp_path, monkeypatch):
    """A9 item 4 compares against max(C0y, G+): without the G+ cells (the common confirm failed) H17.1 is INCOMPLETE,
    not 'winner minus C0y'; a young rung whose confirmatory cells are missing makes the ladder row (an A8 label
    component) INCOMPLETE; an R1 arm without young banks makes the best-young estimate incomplete."""
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "data"))
    cj = copy.deepcopy(CJ)
    cj["statistics"]["n_boot"] = 200
    eff = {}
    for s in (1, 2, 3):
        for code in ("all", "dim"):
            eff[("A3", s, f"epoch4.{code}", "lr")] = 0.60
            eff[("A3", s, f"c0y.{code}", "lr")] = 0.55
            eff[("A3", s, f"young_b77.{code}", "lr")] = 0.56
            for a_ in ("A2", "A0", "A1"):          # A6 has no young banks
                eff[(a_, s, f"c0y.{code}", "lr")] = 0.55
    rows = [r for r in _synthetic_rows(eff) if not (r["set"].startswith("young_b77") and r["split"] == "test" and r["seed"] == 2)]
    runs = tmp_path / "cells"
    runs.mkdir()
    with open(runs / "cells.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "report"
    out.mkdir()
    monkeypatch.setattr(C, "out_dir", lambda stage: out)
    S.stage_report(argparse.Namespace(probe_runs=str(runs), confirm_runs="", qpred_runs="", qcon_runs="", qint_runs="",
                                      census_runs="", skipped_file=None), cj, S.Rec(False))
    v = json.loads((out / "verdicts.json").read_text())
    assert v["H17.1"]["verdict"] == "INCOMPLETE" and set(v["H17.1"]["inputs"]["stronger_control_per_seed"].values()) == {None}
    assert v["H17.1.dim_reported"]["verdict"] == "INCOMPLETE"
    assert v["A8_rows"]["ladder.all"]["verdict"] == "INCOMPLETE" and v["A8_rows"]["ladder.all"]["inputs"]["max_rung_per_seed"]["2"] is None
    assert v["A8_rows"]["ladder.all"]["inputs"]["max_rung_per_seed"]["1"] == ["A3", 1, "young_b77.all", "lr"]
    assert v["A8_literal_learning_credit"]["label"] == "INCOMPLETE"
    assert not v["A8_rows"]["minus_best_young_any_arm.all"]["complete"]
    assert S.stronger({("x", 1, "a", "lr", "32", "test"): {0: 0.5}}, [("x", 1, "a", "lr"), ("y", 0, "b", "lr")], "test", "32") is None
    assert S.stronger({("x", 1, "a", "lr", "32", "test"): {0: 0.5}}, [("x", 1, "a", "lr"), ("y", 0, "b", "lr")], "test", "32",
                      require_all=False) == ("x", 1, "a", "lr")


def test_preflight_host_checks_the_published_reference_only_for_files_not_cached(tmp_path, monkeypatch):
    paths = {"agnews": tmp_path / "ag.parquet", "trec": tmp_path / "trec.parquet"}
    monkeypatch.setattr(CF, "test_file_path", lambda task, root=None: paths[task])
    calls = []
    meta = {"agnews": {"size": 11, "blob_id": "c" * 40, "lfs": {"sha256": "f" * 64, "size": 11}},
            "trec": {"size": 7, "blob_id": "b" * 40, "lfs": None}}

    def fake_meta(task):
        calls.append(task)
        return meta[task]

    monkeypatch.setattr(CF, "test_file_metadata", fake_meta)
    refs = {"agnews": {"reference": "sha256", "digest": "f" * 64, "bytes": 11},
            "trec": {"reference": "git_blob_sha1", "digest": "b" * 40, "bytes": 7}}
    r = S.preflight_host(["agnews", "sst2", "trec"], refs)
    assert r["passed"] and calls == ["agnews", "trec"] and r["agnews"]["equal"] and r["trec"]["equal"]
    paths["agnews"].write_bytes(b"x")                                    # cached: no metadata call
    calls.clear()
    assert S.preflight_host(["agnews", "trec"], refs)["agnews"] == "cached" and calls == ["trec"]
    bad = dict(refs, trec={"reference": "git_blob_sha1", "digest": "d" * 40, "bytes": 7})
    assert not S.preflight_host(["trec"], bad)["passed"]                 # another published digest
    monkeypatch.setattr(CF, "test_file_metadata", lambda task: (_ for _ in ()).throw(OSError("unreachable")))
    r = S.preflight_host(["trec"], refs)
    assert not r["passed"] and "unreachable" in r["trec"]["error"]


def test_subset_docs():
    docs, _, _ = _toy_docs(False)
    sub = S.subset_docs(docs, np.array([3, 0, 7]))
    assert sub.n_doc == 3 and sub.end_id == docs.end_id
    ds = docs.doc_start.astype(np.int64)
    for i, r in enumerate([3, 0, 7]):
        assert np.array_equal(sub.tok[sub.doc_start[i]:sub.doc_start[i + 1]], docs.tok[ds[r]:ds[r + 1]])
    assert S.subset_docs(docs, np.zeros(0, np.int64)).n_doc == 0


# ====================================================================== CUDA paths (skip without a device; run under scripts/gpu_run.sh shared)
def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1).sum()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    return cp


@pytest.mark.gpu
def test_random_bank_cupy_equals_numpy():
    cp = _cupy_or_skip()
    c = toy_corpus(vocab=20, n_docs=5, doc_len=70, seed=13)
    b = CF.random_bank(c.tok, c.doc_start, c.kind_map, "masked", 20, 500, seed=5)
    s_np = CF.literal_single_counts(np, c.tok, c.doc_start, c.kind_map, "masked", 20, chunk=100)
    s_cp = CF.literal_single_counts(cp, c.tok, c.doc_start, c.kind_map, "masked", 20, chunk=100)
    assert np.array_equal(s_np, s_cp)
    f_np, f_cp = CF.RandomBankFirer(np, b, s_np, c.kind_map), CF.RandomBankFirer(cp, b, s_np, c.kind_map)
    a = CF.random_bank_doc_counts(np, f_np, c.tok, c.doc_start, chunk=64)
    g = CF.random_bank_doc_counts(cp, f_cp, c.tok, c.doc_start, chunk=64)
    assert np.array_equal(a.indptr, g.indptr) and np.array_equal(a.ids, g.ids) and np.array_equal(a.cnt, g.cnt)
    assert np.array_equal(CF.random_bank_fire_counts(np, f_np, c.tok, c.doc_start, 50), CF.random_bank_fire_counts(cp, f_cp, c.tok, c.doc_start, 50))


@pytest.mark.gpu
def test_word_codes_cuda_equal_the_public_api():
    _cupy_or_skip()
    from tmlm.engine.cuda.cuda_engine import CudaEngine

    c = toy_corpus(vocab=30, n_docs=8, doc_len=90, seed=14)
    tok = np.asarray(c.tok).astype(np.int64)
    wid = np.arange(tok.shape[0], dtype=np.uint32)
    wid[1::7] = wid[0::7][: wid[1::7].shape[0]]       # some two-token words
    targets = {"a": [2], "b": [5], "c": [1], "d": [0]}
    occ = CF.build_occurrences(tok, wid, targets, max_occ=6)
    ds = np.asarray(c.doc_start, np.int64)
    dop = np.repeat(np.arange(ds.shape[0] - 1), np.diff(ds))
    rounds = CF.occurrence_rounds(occ.start, dop)
    cfg = P.engine_config("A0", 2, 30, 64, batch=16, sweep_period=8, n_bag_k=2, dead_sweeps=2)
    cpu = CpuEngine(cfg, corpus=c)
    for e in range(2):
        for b in range(cpu.n_batches()):
            cpu.train_batch(e, b)
    gpu = CudaEngine(cfg, corpus=c)
    gpu.load_state(cpu.state())
    a = CF.word_codes_api(cpu, occ, tok.shape[0], rounds)
    g = CF.word_codes_cuda(gpu, occ, tok.shape[0], rounds, max_positions=5)   # blocks split occurrences: deduplicated per round
    assert all(np.array_equal(x, y) for x, y in zip(a, g)) and a[0].shape[0] > 0


def test_access_audit_counts_each_configuration_and_task(tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path))
    CF.record_confirm_access(CF.ACCESS_TAG, "A3_seed1", "agnews", "d")
    CF.record_confirm_access(CF.ACCESS_TAG, "A3_seed1", "trec", "d")
    CF.record_confirm_access(CF.MEN_TAG, "qcon_A3_seed1", "men_test", "d")
    a = S.access_audit()
    assert a["total_confirmatory_accesses"] == 3 and a["every_pair_once"] and a["every_extra_access_has_a_logged_reason"]
    CF.record_confirm_access(CF.ACCESS_TAG, "A3_seed1", "trec", "d", reaccess_reason="engine crash after the access")
    a = S.access_audit()
    assert a["per_config_task"]["config=A3_seed1; task=trec"] == 2 and not a["every_pair_once"]
    assert a["every_extra_access_has_a_logged_reason"] and len(a["reaccesses"]) == 1


def test_access_audit_lists_missing_and_unexpected_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path))
    CF.record_confirm_access(CF.ACCESS_TAG, "A3_seed1", "agnews", "d")
    CF.record_confirm_access(CF.ACCESS_TAG, "B9_seed1", "agnews", "d")
    a = S.access_audit(expected=S.expected_accesses(CJ))
    assert a["expected_pairs"] == 90 and len(a["missing_pairs"]) == 89 and a["unexpected_pairs"] == ["config=B9_seed1; task=agnews"]


# ====================================================================== (h) the real host's metadata (opt-in, no content)
@pytest.mark.skipif(not __import__("os").environ.get("TMLM_HF_METADATA_TEST"), reason="opt-in: TMLM_HF_METADATA_TEST=1 (network, metadata only)")
def test_real_host_metadata_of_both_test_files(tmp_path):
    """docs/13 A9 item 10 against the real host: HfApi.get_paths_info on the two test files at the pinned revisions
    (metadata only; nothing is downloaded; references are written to a temporary root only). The response is printed so
    that the test output records it. Recorded 2026-09-25: AG News is an LFS file (a published sha256); TREC is stored in
    git (no LFS entry, only its git blob id), so under A9 as written its reference is refused."""
    import re

    for task in ("agnews", "trec"):
        m = CF.test_file_metadata(task)
        print("HF_METADATA_RESPONSE", json.dumps(m, sort_keys=True, default=str))
        assert m["path"] == CF.CONFIRM_SOURCES[task]["file"] and m["revision"] == D.TASKS[task]["revision"] and int(m["size"]) > 0
        root = tmp_path / task
        if m["lfs"] is not None:
            assert re.fullmatch(r"[0-9a-f]{64}", m["lfs"]["sha256"]) and int(m["lfs"]["size"]) == int(m["size"])
            r = CF.write_test_hash_reference(task, root, metadata_fn=lambda t, m=m: m)
            assert r["reference"] == "sha256" and r["written"]
        else:
            assert re.fullmatch(r"[0-9a-f]{40}", m["blob_id"])
            with pytest.raises(ValueError):
                CF.write_test_hash_reference(task, root, metadata_fn=lambda t, m=m: m)
            r = CF.write_test_hash_reference(task, root, metadata_fn=lambda t, m=m: m, non_lfs="git_blob_sha1", amendment="test only")
            assert r["reference"] == "git_blob_sha1" and r["digest"] == m["blob_id"]
        print("REFERENCE_KIND", task, "sha256" if m["lfs"] is not None else "git blob id only (no published sha256)")


@pytest.mark.gpu
def test_cuda_context_order_is_fixed_before_any_cupy_work():
    """tmu's import runs pycuda.autoinit; CuPy modules loaded before it fail afterwards (2026-09-25). main() imports tmu
    first; this test runs the confirm-stage order (CuPy, tmu probe applied, CuPy again) in a fresh process."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    code = (
        "import sys; sys.path.insert(0, 'experiments'); import ssl_r1_confirm as S; print(S.cuda_context_first());\n"
        "import numpy as np, cupy as cp\n"
        "src = r'extern \"C\" __global__ void inc(int* a, int n){int i=threadIdx.x; if(i<n) a[i]+=1;}'\n"
        "k = cp.RawModule(code=src, options=('-arch=sm_70',)).get_function('inc'); a = cp.zeros(32, dtype=cp.int32)\n"
        "k((1,), (32,), (a, np.int32(32)))\n"
        "from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier\n"
        "k((1,), (32,), (a, np.int32(32))); cp.cuda.Device().synchronize(); print('sum', int(a.sum()))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-2000:]
    assert "tmu imported before any CuPy work" in out.stdout and "sum 64" in out.stdout


@pytest.mark.gpu
def test_cuda_context_first_leaves_the_cuda_engine_usable():
    """Regression (2026-09-26): after cuda_context_first() the CUDA engine must still launch its kernels."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    import subprocess
    import sys as _sys
    code = (
        "import sys; sys.path.insert(0, 'experiments'); sys.path.insert(0, 'src')\n"
        "import ssl_r1_confirm as CF\n"
        "note = CF.cuda_context_first()\n"
        "from tmlm.data.synthetic import toy_corpus\n"
        "from tmlm.engine.config import EngineConfig\n"
        "from tmlm.engine.cuda.cuda_engine import CudaEngine\n"
        "g = CudaEngine(EngineConfig(mode='masked', vocab=40, n_clauses=64, seed=1, sweep_period=8, n_bag_k=2),"
        " corpus=toy_corpus(vocab=40, n_docs=40, doc_len=300, seed=7))\n"
        "g.train_batch(0, 0); g.synchronize(); print('OK', note)\n"
    )
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
