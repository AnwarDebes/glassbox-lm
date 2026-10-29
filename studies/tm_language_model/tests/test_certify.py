"""Receipts and the standalone checker (docs/04 M8 ACCEPT: 100% of receipts pass the standalone
checker, tampered receipts fail; docs/13 H2.4). The checker is imported from its file and also
run as a separate process; an AST scan proves it imports nothing from the project.

Since 2026-10-25 (finish pass, bug TLM-B2) the checker binds every decision to the generation run;
the tests at the end show that each new forgery mode passes the re-derivation of items 1 to 5 of the
checker (so the checker before the fix accepted it) and fails only on the binding of item 6.

    .venv-tm/bin/python -m pytest -q tests/test_certify.py
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import json

from tmlm.certify.receipt import TAMPER_MODES, build_receipts, load_receipts, receipt_text, tamper, write_receipts
from tmlm.certify.receipt import _token_probs
from tmlm.decode import sampling
from tmlm.decode.sampling import cdf_q32, select
from tmlm.engine.config import Stream
from tmlm.engine.rng import rng32
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
    elif mode in ("draw", "forged_draw", "temperature", "mode_switch") and decoder == "greedy":
        assert n_try == 0  # a greedy token has no draw
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


# ------------------------------------------------------------------ binding to the generation run (TLM-B2)
SAMPLED = ["anyorder", "diffusion", "eliminate"]
FORGERIES = {"forged_draw": "draw: the counters", "temperature": "temperature differs", "mode_switch": "mode "}


def test_checker_constants_equal_the_decoders():
    chk = _load_checker()
    assert chk.STREAM_SAMPLER == int(Stream.SAMPLER)
    assert chk.DRAW_TOKEN == sampling.DRAW_TOKEN


@pytest.mark.parametrize("decoder", SAMPLED)
def test_untampered_receipts_are_bound_to_their_run(setup, decoder):
    doc, chk = setup["docs"][decoder], setup["chk"]
    assert all(chk.binding_errors(doc, i) == [] for i in range(len(doc["tokens"])))
    res = chk.check_receipts(setup["loaded_bank"], doc, expected_seed=3, expected_temperature=1.0)
    assert all(r["pass"] for r in res)


@pytest.mark.parametrize("decoder", SAMPLED)
@pytest.mark.parametrize("mode", sorted(FORGERIES))
def test_forgeries_pass_the_rederivation_and_fail_only_the_binding(setup, decoder, mode):
    """The bug: items 1 to 5 alone (the whole checker before 2026-10-25) accept each forgery."""
    doc, chk, lb = setup["docs"][decoder], setup["chk"], setup["loaded_bank"]
    n_forged = 0
    for i in range(len(doc["tokens"])):
        t = tamper(doc, i, mode, setup["bank"], seed=1)
        if t is None:
            continue
        n_forged += 1
        assert t["tokens"][i]["claimed"] != doc["tokens"][i]["claimed"] or mode == "mode_switch"
        assert chk.doc_errors(lb, t) == []
        assert chk.check_token(lb, t, t["tokens"][i]) == [], (decoder, mode, i)  # accepted by items 1 to 5
        bound = chk.binding_errors(t, i)
        assert bound and any(FORGERIES[mode] in e for e in bound), bound
        assert not chk.check_receipts(lb, t)[i]["pass"]
    assert n_forged > 0


def test_forged_draw_with_the_step_moved_along_is_rejected(setup):
    """A forger who also rewrites the step to match the forged counter a."""
    doc, chk, lb = setup["docs"]["anyorder"], setup["chk"], setup["loaded_bank"]
    i = 2
    e = json.loads(json.dumps(doc))
    entry = e["tokens"][i]
    cdf = cdf_q32(_token_probs(e, entry, setup["bank"], 1.0))
    width = np.diff(np.concatenate([[0], cdf]))
    width[entry["claimed"]] = 0
    target = int(np.argmax(width))
    dr = entry["draw"]
    for k in range(1, 1 << 16):
        u = rng32(dr["seed"], dr["stream"], dr["a"] + k, dr["b"], dr["c"])
        if select(cdf, u) == target:
            break
    else:
        pytest.skip("no counter found")
    entry["draw"] = {**dr, "a": dr["a"] + k, "u32": int(u)}
    entry["step"] = dr["a"] + k
    entry["claimed"] = target
    assert chk.check_token(lb, e, entry) == []
    errors = chk.check_receipts(lb, e)[i]["errors"]
    assert any("is not the token's index" in x for x in errors), errors


def test_diffusion_token_moved_to_another_step_is_rejected(setup):
    doc, chk = setup["docs"]["diffusion"], setup["chk"]
    steps = [tok["step"] for tok in doc["tokens"]]
    i = next(k for k in range(1, len(steps)) if steps[k] == steps[k - 1] + 1)  # the first token of a step
    e = json.loads(json.dumps(doc))
    e["tokens"][i - 1]["step"] += 1  # the last token of the step before, moved forward
    e["tokens"][i - 1]["draw"]["a"] += 1
    assert any("window differs" in x or "does not follow" in x for x in chk.binding_errors(e, i - 1))


def test_expected_seed_binds_a_consistently_reseeded_document(setup):
    """A forger who rewrites the seed of the whole run, every draw and every claim consistently: only the
    seed the verifier expects (the registered one) can catch that."""
    doc, chk, lb = setup["docs"]["anyorder"], setup["chk"], setup["loaded_bank"]
    e = json.loads(json.dumps(doc))
    e["generation"]["seed"] = 4
    for entry in e["tokens"]:
        dr = entry["draw"]
        u = rng32(4, dr["stream"], dr["a"], dr["b"], dr["c"])
        entry["draw"] = {**dr, "seed": 4, "u32": int(u)}
        entry["claimed"] = select(cdf_q32(_token_probs(e, entry, setup["bank"], 1.0)), u)
    assert all(r["pass"] for r in chk.check_receipts(lb, e))  # self-consistent: the header is not bound
    res = chk.check_receipts(lb, e, expected_seed=3)
    assert not any(r["pass"] for r in res)
    assert "is not the expected seed 3" in res[0]["errors"][0]
    assert not any(r["pass"] for r in chk.check_receipts(lb, doc, expected_temperature=0.5))


def test_a_document_without_its_generation_record_fails(setup):
    chk, lb = setup["chk"], setup["loaded_bank"]
    e = json.loads(json.dumps(setup["docs"]["eliminate"]))
    del e["generation"]
    res = chk.check_receipts(lb, e)
    assert res and not any(r["pass"] for r in res)


def test_checker_cli_binds_seed_and_reads_the_runs_layout(setup):
    tmp, docs = setup["tmp"], setup["docs"]
    write_receipts(tmp / "runs.json", {"runs": [docs["anyorder"], docs["diffusion"], docs["eliminate"]]})
    base = [sys.executable, str(CHECKER), "--bank", str(tmp / "bank.npz"), "--receipts", str(tmp / "runs.json")]
    ok = subprocess.run([*base, "--seed", "3", "--temperature", "1.0", "--json", str(tmp / "runs_report.json"),
                         "--quiet"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    report = json.loads((tmp / "runs_report.json").read_text())
    n = sum(len(docs[k]["tokens"]) for k in ("anyorder", "diffusion", "eliminate"))
    assert report["n_runs"] == 3 and report["n_tokens"] == n and report["n_pass"] == n
    assert {r["run"] for r in report["results"]} == {0, 1, 2}
    wrong_seed = subprocess.run([*base, "--seed", "4", "--quiet"], capture_output=True, text=True)
    assert wrong_seed.returncode == 1
    forged = tamper(docs["anyorder"], 0, "forged_draw", setup["bank"], seed=1)
    write_receipts(tmp / "forged.json", forged)
    bad = subprocess.run([sys.executable, str(CHECKER), "--bank", str(tmp / "bank.npz"), "--receipts",
                          str(tmp / "forged.json")], capture_output=True, text=True)
    assert bad.returncode == 1 and "draw: the counters" in bad.stdout
