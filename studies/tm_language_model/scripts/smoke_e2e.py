#!/usr/bin/env python3
"""End-to-end smoke test of the study on tiny data (finish pass of 2026-10-25; completeness critic CR-15).

It runs the path of the study once, end to end, in under a minute on the CPU, and prints one line per step:
  1. env       the interpreter, numpy and the compiled CPU engine load;
  2. train     the CPU engine trains a toy corpus; two runs with seed 1 give the same state hash, which equals
               the golden hash below, and seed 2 gives another;
  3. learn     on a small planted-rules corpus in causal mode the trained engine predicts the held-out rule
               positions better than the most frequent token (top-1, the last 10% of the corpus);
  4. receipts  the any-order decoder generates 12 tokens with receipts from the toy model; the standalone
               checker, run as a separate process bound to the run's seed and temperature, passes every
               receipt (exit 0), and fails a receipt with a changed claim and one with a forged draw (exit 1);
  5. gpu       only with --gpu, under the GPU lock: the CUDA engine trains the toy corpus to the same state hash.
The Track A patched kernels of the Makefile's earlier placeholder are not part of it (that line is archived).
The exit status is 0 only when every step that ran passed; a summary goes to <out>/summary.json.

Run from the study folder after `source env/env.sh` (`make smoke` runs it after lint and the CPU tests):
    CUDA_VISIBLE_DEVICES= scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/smoke_e2e.py [--out DIR]
    scripts/gpu_run.sh shared -- .venv-tm/bin/python scripts/smoke_e2e.py --gpu [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHECKER = ROOT / "src" / "tmlm" / "certify" / "checker_standalone.py"
# State hash of the toy run of step 2 (seed 1, 4 epochs), recorded on 2026-10-25 by the first run of this script;
# a change of the engine that changes training changes it, and then this value is updated with a dated note.
GOLDEN_TOY_STATE_HASH = "ca5a88dce8c003c03aa0c316963f241e9688fe6ca440489af51bafd76bf9ead0"


def toy_engine(seed: int, engine_cls=None):
    from tmlm.data.synthetic import toy_corpus
    from tmlm.engine.config import EngineConfig
    from tmlm.engine.cpu.cpu_engine import CpuEngine
    corpus = toy_corpus(vocab=40, n_docs=6, doc_len=80, seed=7)
    cfg = EngineConfig(mode="masked", vocab=40, n_clauses=256, seed=seed, batch=32, sweep_period=8, n_bag_k=2)
    eng = (engine_cls or CpuEngine)(cfg, corpus=corpus)
    for e in range(4):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    return eng, corpus


def step_env() -> dict:
    import tmlm  # noqa: F401
    from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: F401  (loads the compiled engine)
    return {"ok": True, "python": platform.python_version(), "numpy": np.__version__}


def step_train() -> dict:
    h1 = toy_engine(1)[0].state_hash()
    h2 = toy_engine(1)[0].state_hash()
    h3 = toy_engine(2)[0].state_hash()
    golden_ok = GOLDEN_TOY_STATE_HASH is None or h1 == GOLDEN_TOY_STATE_HASH
    return {"ok": h1 == h2 and h1 != h3 and golden_ok, "seed1": h1, "seed1_again": h2, "seed2": h3,
            "golden": GOLDEN_TOY_STATE_HASH, "golden_checked": GOLDEN_TOY_STATE_HASH is not None}


def rule_positions(tok, doc_start, rule):
    p = np.arange(8, tok.shape[0])
    p = p[(tok[p + rule.o1] == rule.a) & (tok[p + rule.o2] == rule.b)]
    d = np.searchsorted(doc_start, p, side="right") - 1
    return p[p + rule.o1 >= doc_start[d]]


def step_learn() -> dict:
    from tmlm.data.synthetic import planted_rules_corpus
    from tmlm.engine.config import EngineConfig
    from tmlm.engine.cpu.cpu_engine import CpuEngine
    corpus, rules = planted_rules_corpus(vocab=100, n_rules=20, n_tokens=20_000, seed=1, n_docs=10)
    # the engine settings of the planted-rules gate (experiments/configs/planted_rules.json), at a smaller size
    cfg = EngineConfig(mode="causal", vocab=100, n_clauses=2048, seed=1, batch=64, n_bag_k=0, seg_weighting="uniform",
                       neg_budget=2, s=32, dead_sweeps=32, p_ins=0.25, p_neg=0.0625, k_cover=8, v_slots=16,
                       seed_cap=4, rho_seed=0.25, neg_mode="focused")
    eng = CpuEngine(cfg, corpus=corpus)
    epochs = 5
    for e in range(epochs):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    cut = int(corpus.tok.shape[0] * 0.9)
    pos = np.unique(np.concatenate([rule_positions(corpus.tok, corpus.doc_start, r) for r in rules]))
    pos = pos[pos >= cut]
    top1 = float((eng.predict_scores(epochs, pos).argmax(axis=1) == corpus.tok[pos]).mean())
    most_frequent = int(np.bincount(corpus.tok[:cut], minlength=cfg.vocab).argmax())
    unigram = float((corpus.tok[pos] == most_frequent).mean())
    return {"ok": bool(pos.size > 0 and top1 > unigram), "positions": int(pos.size), "top1": top1,
            "most_frequent_token_top1": unigram}


def step_receipts(out: Path) -> dict:
    from tmlm.certify.receipt import build_receipts, tamper, write_receipts
    from tmlm.decode.anyorder import anyorder_generate
    from tmlm.decode.conftm import fit_two_scalar
    from tmlm.decode.scorer import TrainedModel
    eng, corpus = toy_engine(1)
    model = TrainedModel.from_engine(eng, corpus.tok)
    pos = np.arange(0, corpus.tok.shape[0], 3, dtype=np.uint32)
    cal, _ = fit_two_scalar(eng.predict_scores(0, pos), corpus.tok[pos], model.class_bias)
    model = model.with_calibration(cal)
    bank = model.bank()
    bank.save(out / "bank.npz")
    gen = anyorder_generate(model.bind(20), corpus.tok[:8], 12, seed=3, order="confidence")
    doc = build_receipts(gen, model, bank)
    write_receipts(out / "receipts.json", doc)

    def check(name: str, *extra: str) -> int:
        cmd = [sys.executable, str(CHECKER), "--bank", str(out / "bank.npz"), "--receipts", str(out / name), *extra]
        return subprocess.run(cmd, capture_output=True, text=True).returncode

    rc_ok = check("receipts.json", "--seed", "3", "--temperature", "1.0", "--json", str(out / "report.json"), "--quiet")
    report = json.loads((out / "report.json").read_text()) if (out / "report.json").is_file() else {}
    write_receipts(out / "claimed.json", tamper(doc, 2, "claimed", bank))
    rc_claimed = check("claimed.json", "--seed", "3", "--temperature", "1.0", "--quiet")
    forged = tamper(doc, 0, "forged_draw", bank, seed=1)
    rc_forged = None
    if forged is not None:
        write_receipts(out / "forged.json", forged)
        rc_forged = check("forged.json", "--seed", "3", "--temperature", "1.0", "--quiet")
    n_tok, n_pass = report.get("n_tokens"), report.get("n_pass")
    ok = rc_ok == 0 and n_tok == len(doc["tokens"]) and n_pass == n_tok and rc_claimed == 1 and rc_forged in (1, None)
    return {"ok": bool(ok), "receipts": n_tok, "pass": n_pass, "checker_exit": rc_ok, "claimed_exit": rc_claimed,
            "forged_draw_exit": rc_forged}


def step_gpu() -> dict:
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    h_cpu = toy_engine(1)[0].state_hash()
    h_gpu = toy_engine(1, CudaEngine)[0].state_hash()
    return {"ok": h_cpu == h_gpu, "cpu": h_cpu, "gpu": h_gpu}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output folder (default: a new temporary folder)")
    ap.add_argument("--gpu", action="store_true", help="add the GPU parity step (run under scripts/gpu_run.sh)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="tmlm_smoke_"))
    out.mkdir(parents=True, exist_ok=True)
    steps = [("env", step_env), ("train", step_train), ("learn", step_learn), ("receipts", lambda: step_receipts(out))]
    if a.gpu:
        steps.append(("gpu", step_gpu))
    summary: dict = {"out": str(out), "steps": {}}
    for name, fn in steps:
        t0 = time.time()
        try:
            res = fn()
        except Exception as e:  # a crash is a failed step, reported with its message
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        res["seconds"] = round(time.time() - t0, 1)
        summary["steps"][name] = res
        print(f"{'ok  ' if res['ok'] else 'FAIL'} {name}: " + json.dumps({k: v for k, v in res.items() if k != 'ok'}),
              flush=True)
    summary["ok"] = all(r["ok"] for r in summary["steps"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(("PASSED" if summary["ok"] else "FAILED") + f": summary in {out / 'summary.json'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
