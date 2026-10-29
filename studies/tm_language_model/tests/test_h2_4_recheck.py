"""experiments/h2_4_recheck.py on a tiny E11-like run folder (toy corpus, two seeds, three decoders): the literal
pins and the manifest, the selection that runs before any checker call (and stops an invalid run before step
3), every receipt through the checker process with failures classified by kind, the tamper reading by kind
and decoder, a checker that raises, validity, and the verdict inputs. No real data is read.

    .venv-tm/bin/python -m pytest -q tests/test_h2_4_recheck.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.certify.receipt import TAMPER_MODES, build_receipts, tamper, write_receipts
from tmlm.data.synthetic import toy_corpus
from tmlm.decode.anyorder import anyorder_generate
from tmlm.decode.conftm import fit_two_scalar
from tmlm.decode.diffusion import diffusion_generate
from tmlm.decode.eliminate import elimination_generate
from tmlm.decode.scorer import TrainedModel
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import h2_4_recheck as rc  # noqa: E402

SEEDS = (1, 2)
TAMPER = 6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(run_dir: Path) -> None:
    files = {p.name: {"sha256": _sha(p), "size": p.stat().st_size} for p in sorted(run_dir.iterdir())
             if p.name != "MANIFEST.json"}
    (run_dir / "MANIFEST.json").write_text(json.dumps({"files": files}))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("e11like")
    corpus = toy_corpus(vocab=40, n_docs=6, doc_len=80, seed=7)
    cfg = EngineConfig(mode="masked", vocab=40, n_clauses=256, seed=1, batch=32, sweep_period=8, n_bag_k=2)
    eng = CpuEngine(cfg, corpus=corpus)
    for e in range(4):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    model = TrainedModel.from_engine(eng, corpus.tok)
    pos = np.arange(0, corpus.tok.shape[0], 3, dtype=np.uint32)
    cal, _ = fit_two_scalar(eng.predict_scores(0, pos), corpus.tok[pos], model.class_bias)
    model = model.with_calibration(cal)
    bank = model.bank()
    bank.save(d / "bank.npz")
    prompts = [corpus.tok[:8], corpus.tok[80:88]]
    gen = {"anyorder": lambda p, s: anyorder_generate(model.bind(20), p, 12, seed=s, order="confidence"),
           "diffusion": lambda p, s: diffusion_generate(model.bind(20), p, 12, seed=s, steps=6, remask_frac=0.1),
           "eliminate": lambda p, s: elimination_generate(model.bind(20), p, 12, seed=s)}
    for seed in SEEDS:
        for dec in rc.E11_DECODERS:
            write_receipts(d / f"receipts_{dec}_seed{seed}.json",
                           {"runs": [build_receipts(gen[dec](p, seed), model, bank) for p in prompts]})
    _manifest(d)
    return d


@pytest.fixture
def e11_like(monkeypatch):
    monkeypatch.setattr(rc, "E11_SEEDS", SEEDS)
    monkeypatch.setattr(rc, "E11_TAMPER_PER_MODE", TAMPER)


def _cfg(run_dir: Path, tmp_path: Path) -> dict:
    """The real config with the toy folder's pins, the toy selection's counts and the toy validity numbers."""
    cfg = json.loads(rc.CONFIG.read_text())
    p = cfg["params"]
    p["input_sha256"] = {name: _sha(run_dir / name) for name in ["MANIFEST.json", "bank.npz", *rc.e11_files()]}
    met, code = rc.run(run_dir, rc.e11_files(), TAMPER, 1.0, tmp_path / "sel", cfg, selection_only=True)
    assert code == rc.EXIT_OK
    p["tamper_counts"] = met["selection.counts"]["value"]
    n = sum(len(d["tokens"]) for name in rc.e11_files() for d in json.loads((run_dir / name).read_text())["runs"])
    p["receipts_checked"] = n
    p["runs_per_file"] = 2
    first = json.loads((run_dir / "receipts_anyorder_seed1.json").read_text())["runs"]
    k = sum(len(d["tokens"]) for d in first)
    p["smoke_row"] = {"file": "receipts_anyorder_seed1.json", "result": f"{k}/{k} pass"}
    return cfg


def _v(metrics, key):
    return metrics[key]["value"]


def test_the_file_order_and_seeds_follow_e11():
    names = rc.e11_files()
    assert len(names) == 15 and names[:3] == ["receipts_anyorder_seed1.json", "receipts_diffusion_seed1.json",
                                              "receipts_eliminate_seed1.json"]
    assert [rc.seed_of(n) for n in names] == [s for s in (1, 2, 3, 4, 5) for _ in range(3)]
    with pytest.raises(ValueError):
        rc.seed_of("receipts_other_seed1.json")
    assert rc.one_sided_upper_95(0, 100) == pytest.approx(0.0295, abs=5e-5)
    assert rc.one_sided_upper_95(1, 100) is None and rc.one_sided_upper_95(0, 0) is None


def test_the_real_config_declares_e11s_counts_and_the_selection_counts():
    p = json.loads(rc.CONFIG.read_text())["params"]
    assert p["tamper_counts"]["context"] == 88 and p["tamper_counts"]["survivors"] == 25
    assert [p["tamper_counts"][m] for m in rc.NEW_MODES] == [100, 99, 100]
    assert set(p["input_sha256"]) == {"MANIFEST.json", "bank.npz", *rc.e11_files()}
    assert p["receipts_checked"] == 11400 and p["runs_per_file"] == 20


def test_full_recheck_of_an_honest_run_meets_every_input(run_dir, e11_like, tmp_path):
    cfg = _cfg(run_dir, tmp_path)
    metrics, code = rc.run(run_dir, rc.e11_files(), TAMPER, 1.0, tmp_path / "out", cfg)
    assert code == rc.EXIT_OK
    assert _v(metrics, "smoke") is False and _v(metrics, "input_mismatches") == []
    assert all(_v(metrics, "validity").values())
    assert _v(metrics, "receipts_pass") == _v(metrics, "receipts_checked") > 0
    for mode in TAMPER_MODES:
        n_fail, n_try = map(int, _v(metrics, f"tamper.{mode}").split()[0].split("/"))
        assert n_fail == n_try, mode
    for mode in rc.NEW_MODES:                         # the forgeries pass items 1 to 5 and fail only the binding
        stats = metrics[f"tamper.{mode}"]["per_decoder"]
        assert all(s["binding_only"] == s["n"] and s["fail_items_1_to_5"] == 0 for s in stats.values()), mode
        assert sum(s["n"] for s in stats.values()) > 0
    for mode in ("claimed", "sums", "state_hash"):    # the old modes fail items 1 to 5, as in E11
        stats = metrics[f"tamper.{mode}"]["per_decoder"]
        assert all(s["fail_items_1_to_5"] == s["n"] for s in stats.values()), mode
    assert _v(metrics, "H2_4_recheck_all_receipts_pass") is True
    assert _v(metrics, "H2_4_recheck_all_tampered_fail") is True
    assert _v(metrics, "H2_4_recheck_forged_draw_qualifier_lifted") is True
    assert (tmp_path / "out" / "report_receipts_eliminate_seed2.json").is_file()   # the full reports stay


def test_a_subset_is_a_smoke_without_verdict_inputs(run_dir, e11_like, tmp_path):
    cfg = _cfg(run_dir, tmp_path)
    metrics, code = rc.run(run_dir, ["receipts_anyorder_seed1.json"], 2, 1.0, tmp_path / "o", cfg)
    assert code == rc.EXIT_OK and _v(metrics, "smoke") is True
    assert "H2_4_recheck_all_receipts_pass" not in metrics


def _copy(run_dir: Path, dest: Path) -> Path:
    dest.mkdir()
    for p in run_dir.iterdir():
        (dest / p.name).write_bytes(p.read_bytes())
    return dest


def test_inputs_that_differ_from_their_pins_stop_the_recheck(run_dir, e11_like, tmp_path):
    cfg = _cfg(run_dir, tmp_path)
    copy_dir = _copy(run_dir, tmp_path / "copy")
    target = copy_dir / "receipts_eliminate_seed2.json"
    target.write_bytes(target.read_bytes().replace(b'"claimed": ', b'"claimed":  ', 1))
    _manifest(copy_dir)                                # a forger who also rewrites the manifest
    metrics, code = rc.run(copy_dir, rc.e11_files(), TAMPER, 1.0, tmp_path / "out", cfg)
    assert code == rc.EXIT_INPUTS
    assert _v(metrics, "input_mismatches") == ["MANIFEST.json: differs from its pin",
                                               "receipts_eliminate_seed2.json: differs from its pin"]
    assert "receipts_checked" not in metrics and "selection.counts" not in metrics


def test_a_selection_that_differs_from_the_declared_counts_stops_before_any_check(run_dir, e11_like, tmp_path):
    cfg = _cfg(run_dir, tmp_path)
    bad = copy.deepcopy(cfg)
    bad["params"]["tamper_counts"]["forged_draw"] += 1
    metrics, code = rc.run(run_dir, rc.e11_files(), TAMPER, 1.0, tmp_path / "out", bad)
    assert code == rc.EXIT_SELECTION
    assert "forged_draw" in _v(metrics, "selection.invalid")
    assert "receipts_checked" not in metrics and not list((tmp_path / "out").glob("report_*"))


def test_a_saved_forged_receipt_fails_on_the_binding_only(run_dir, e11_like, tmp_path):
    copy_dir = _copy(run_dir, tmp_path / "forged")
    name = "receipts_anyorder_seed2.json"
    doc = json.loads((copy_dir / name).read_text())
    bank = rc.Bank.load(copy_dir / "bank.npz")
    forged = next(t for i in range(len(doc["runs"][0]["tokens"]))
                  if (t := tamper(doc["runs"][0], i, "forged_draw", bank, seed=1)) is not None)
    doc["runs"][0] = forged
    write_receipts(copy_dir / name, doc)
    _manifest(copy_dir)
    cfg = _cfg(copy_dir, tmp_path)                    # pins of the forged folder: only the checker can catch it
    metrics, code = rc.run(copy_dir, rc.e11_files(), TAMPER, 1.0, tmp_path / "out", cfg)
    assert code == rc.EXIT_OK
    assert _v(metrics, "receipts_pass") == _v(metrics, "receipts_checked") - 1
    assert metrics[f"receipts.{name}"]["failures_by_kind"] == {"items_1_to_5": 0, "binding_only": 1}
    assert _v(metrics, "H2_4_recheck_all_receipts_pass") is False


def test_a_checker_that_raises_counts_its_file_as_failed(run_dir, e11_like, tmp_path, monkeypatch):
    """An exception on one file counts its receipts as failed (NOT MET); on the smoke's file the run is invalid."""
    cfg = _cfg(run_dir, tmp_path)
    real = rc.CHECKER
    for victim, valid in (("receipts_eliminate_seed2.json", True), ("receipts_anyorder_seed1.json", False)):
        wrapper = tmp_path / f"checker_raising_on_{victim}.py"
        wrapper.write_text(
            "import runpy, sys\n"
            f"if sys.argv[sys.argv.index('--receipts') + 1].endswith({victim!r}):\n"
            "    raise RuntimeError('boom')\n"
            f"sys.argv[0] = {str(real)!r}\n"
            f"runpy.run_path({str(real)!r}, run_name='__main__')\n")
        monkeypatch.setattr(rc, "CHECKER", wrapper)
        monkeypatch.setattr(rc, "load_checker", lambda: _load(real))
        metrics, code = rc.run(run_dir, rc.e11_files(), TAMPER, 1.0, tmp_path / f"out_{victim}", cfg)
        assert code == rc.EXIT_OK
        assert "checker raised" in _v(metrics, f"receipts.{victim}")
        assert _v(metrics, "receipts_checked") == cfg["params"]["receipts_checked"]
        assert all(_v(metrics, "validity").values()) is valid
        if valid:
            assert _v(metrics, "H2_4_recheck_all_receipts_pass") is False
        else:
            assert "H2_4_recheck_all_receipts_pass" not in metrics


def _load(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("checker_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_writes_metrics_into_the_out_dir(run_dir, e11_like, tmp_path):
    cfg = _cfg(run_dir, tmp_path)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    code = rc.main(["--run-dir", str(run_dir), "--config", str(path), "--files", "receipts_diffusion_seed1.json",
                    "--tamper-per-mode", "2", "--out-dir", str(tmp_path / "o")])
    assert code == rc.EXIT_OK
    metrics = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert metrics["smoke"]["value"] is True and metrics["receipts_checked"]["value"] > 0
