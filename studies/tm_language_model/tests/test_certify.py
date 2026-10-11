"""Receipts and the standalone checker (docs/04 M8 ACCEPT: 100% of receipts pass the standalone
checker, tampered receipts fail; docs/13 H2.4). The checker is imported from its file and also
run as a separate process; an AST scan proves it imports nothing from the project."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.certify.receipt import TAMPER_MODES, build_receipts, load_receipts, receipt_text, tamper, write_receipts
from tmlm.data.synthetic import toy_corpus
from tmlm.decode.anyorder import anyorder_generate
from tmlm.decode.conftm import fit_two_scalar
from tmlm.decode.diffusion import diffusion_generate
from tmlm.decode.eliminate import elimination_generate
from tmlm.decode.scorer import TrainedModel
from tmlm.engine.config import EngineConfig
from tmlm.engine.cpu.cpu_engine import CpuEngine

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "src" / "tmlm" / "certify" / "checker_standalone.py"
STDLIB_OK = {"argparse", "hashlib", "json", "sys", "numpy", "z3", "__future__"}


def _load_checker():
    spec = importlib.util.spec_from_file_location("checker_standalone", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cert")
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
    bank.save(tmp / "bank.npz")
    prompt = corpus.tok[:8]
    gens = {"anyorder": anyorder_generate(model.bind(20), prompt, 12, seed=3, order="confidence"),
            "greedy": anyorder_generate(model.bind(20), prompt, 12, seed=3, greedy=True),
            "diffusion": diffusion_generate(model.bind(20), prompt, 12, seed=3, steps=4, remask_frac=0.2),
            "eliminate": elimination_generate(model.bind(20), prompt, 12, seed=3)}
    docs = {k: build_receipts(g, model, bank) for k, g in gens.items()}
    for k, d in docs.items():
        write_receipts(tmp / f"{k}.json", d)
    chk = _load_checker()
    return {"tmp": tmp, "model": model, "bank": bank, "docs": docs, "chk": chk, "loaded_bank": chk.load_bank(str(tmp / "bank.npz"))}


def test_checker_imports_nothing_from_the_project():
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in the standalone checker"
            names.add((node.module or "").split(".")[0])
    assert names <= STDLIB_OK, names
    assert not any(n.startswith("tmlm") for n in names)


@pytest.mark.parametrize("decoder", ["anyorder", "greedy", "diffusion", "eliminate"])
def test_every_untampered_receipt_passes(setup, decoder):
    res = setup["chk"].check_receipts(setup["loaded_bank"], setup["docs"][decoder])
    assert res and all(r["pass"] for r in res), [r["errors"] for r in res if not r["pass"]]
    assert len(res) == len(setup["docs"][decoder]["tokens"])


@pytest.mark.parametrize("decoder", ["anyorder", "greedy", "diffusion", "eliminate"])
@pytest.mark.parametrize("mode", TAMPER_MODES)
def test_every_tampered_receipt_fails(setup, decoder, mode):
    doc, bank, chk, lb = setup["docs"][decoder], setup["bank"], setup["chk"], setup["loaded_bank"]
    n_try = 0
    for i in range(len(doc["tokens"])):
        t = tamper(doc, i, mode, bank, seed=1)
        if t is None:
            continue
        n_try += 1
        res = chk.check_receipts(lb, t)
        assert not res[i]["pass"], (decoder, mode, i)
    if mode == "survivors" and decoder != "eliminate":
        assert n_try == 0
    elif mode == "draw" and decoder == "greedy":
        assert n_try == 0
    else:
        assert n_try > 0, (decoder, mode, "no token to tamper")


def test_tampering_is_covered_by_every_mode_on_at_least_one_decoder(setup):
    covered = set()
    for decoder, doc in setup["docs"].items():
        for mode in TAMPER_MODES:
            if any(tamper(doc, i, mode, setup["bank"]) is not None for i in range(len(doc["tokens"]))):
                covered.add(mode)
    assert covered == set(TAMPER_MODES)


def test_checker_as_a_subprocess_exit_codes(setup):
    tmp = setup["tmp"]
    ok = subprocess.run([sys.executable, str(CHECKER), "--bank", str(tmp / "bank.npz"), "--receipts", str(tmp / "eliminate.json"),
                         "--json", str(tmp / "report.json")], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "receipts pass" in ok.stdout
    bad = tamper(setup["docs"]["eliminate"], 2, "claimed", setup["bank"])
    write_receipts(tmp / "bad.json", bad)
    rc = subprocess.run([sys.executable, str(CHECKER), "--bank", str(tmp / "bank.npz"), "--receipts", str(tmp / "bad.json"), "--quiet"],
                        capture_output=True, text=True)
    assert rc.returncode == 1
    reloaded = load_receipts(tmp / "eliminate.json")
    assert reloaded == setup["docs"]["eliminate"]


def test_optional_z3_check_agrees_on_greedy_receipts(setup):
    pytest.importorskip("z3")
    res = setup["chk"].check_receipts(setup["loaded_bank"], setup["docs"]["greedy"], use_z3=True)
    assert all(r["pass"] for r in res)
    bad = tamper(setup["docs"]["greedy"], 0, "claimed", setup["bank"])
    assert not setup["chk"].check_receipts(setup["loaded_bank"], bad, use_z3=True)[0]["pass"]


def test_receipt_text_is_readable(setup):
    txt = receipt_text(setup["docs"]["greedy"]["tokens"][0])
    assert txt.startswith("position ") and "sums:" in txt
