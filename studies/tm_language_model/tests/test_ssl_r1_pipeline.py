"""Probe and report stages of experiments/ssl_r1_pretext.py on a synthetic features run (no GPU, no real
data): cells for every set, task, budget and draw; Q_text over the three tasks; the paired comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tmlm.ssl import downstream as D
from tmlm.ssl import features as F
from tmlm.ssl import probes as PR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import ssl_r1_pretext as S  # noqa: E402

V, C = 40, 64
PARAMS = {"smoke": True, "vocab": V, "tasks": ["agnews", "sst2", "trec"], "budgets": [8], "draws": [0, 1], "folds": 4,
          "c_grid": [0.1, 1.0], "lr_max_iter": 200, "tm_budgets": [8], "tm_grid": [{"clauses": 20, "T": 10, "s": 3.0}],
          "tm_epochs": 2, "tm_ppmi_bits": 4, "dim_matched": 16}


def _fake_run(d: Path) -> None:
    rng = np.random.default_rng(0)
    layout, labels, docs = {}, {}, []
    base = 0
    for task in PARAMS["tasks"]:
        k = D.TASKS[task]["n_classes"]
        pool_y = rng.integers(0, k, size=80 * k)
        dev_y = rng.integers(0, k, size=10 * k)
        union = PR.sample_union(pool_y, PARAMS["budgets"], PARAMS["draws"], D.TASKS[task]["code"])
        ys = list(dev_y) + list(pool_y[union])
        for y in ys:  # label-dependent tokens so that the probes can learn; END token id 0 closes every document
            docs.append(list(rng.integers(1 + 3 * int(y), 4 + 3 * int(y), size=6)) + list(rng.integers(1, V, size=4)) + [0])
        layout[task] = {"dev": [base, base + dev_y.shape[0]], "lab": [base + dev_y.shape[0], base + len(ys)],
                        "lab_pool_rows": union.tolist()}
        base += len(ys)
        labels[task] = {"pool_y": pool_y.tolist(), "dev_y": dev_y.tolist()}
    tok = np.concatenate([np.asarray(x, np.uint16) for x in docs])
    ds = np.zeros(len(docs) + 1, np.uint32)
    ds[1:] = np.cumsum([len(x) for x in docs])
    np.savez(d / "tokens.npz", tok=tok, doc_start=ds, end_id=np.int64(0), layout=json.dumps(layout),
             labels=json.dumps(labels), arm="A1", seed=np.int64(100))
    td = F.TokenDocs(tok=tok, doc_start=ds, end_id=0)
    bow = F.bag_of_tokens(td, V)
    for name in ("bank", "c0y", "gplus"):  # clause-like codes: token counts spread over C columns
        ids = (bow.ids.astype(np.int64) * (3 if name == "bank" else 5)) % C
        sd = F.aggregate_doc_counts(np.repeat(np.arange(bow.n_doc), np.diff(bow.indptr)), ids, bow.n_doc, C)
        S.save_sparse(d / f"features_{name}.npz", sd)
        np.save(d / f"dim_{name}.npy", np.arange(16, dtype=np.int64))
    np.save(d / "ppmi_svd.npy", rng.normal(size=(V, 6)).astype(np.float32))


def test_probe_and_report_on_a_synthetic_run(tmp_path, monkeypatch):
    monkeypatch.delenv("TMLM_RUN_DIR", raising=False)
    monkeypatch.setenv("SSL_SCRATCH", str(tmp_path))
    S._SCRATCH.clear()
    run = tmp_path / "run"
    run.mkdir()
    _fake_run(run)
    sets = ",".join(S.ARM_SETS + ("bow", "ppmi"))
    args = argparse.Namespace(sets=sets, classifiers="lr,tm", full=False, smoke=True, probe_runs=None)
    rec = S.Rec(smoke=True)
    out = S.stage_probe(args, PARAMS, rec, run, run)
    rows = [json.loads(x) for x in open(out / "cells.jsonl")]
    n_sets = len(sets.split(","))
    n_tm_sets = 5  # bank.dim, c0y.dim, gplus.dim, bow, ppmi
    assert len(rows) == 3 * 2 * (n_sets + n_tm_sets)
    assert {r["task"] for r in rows} == set(PARAMS["tasks"])
    assert all(r["n_labelled"] == 8 * D.TASKS[r["task"]]["n_classes"] for r in rows)
    lr_bank = [r["acc"] for r in rows if r["set"] == "bow" and r["clf"] == "lr"]
    assert np.mean(lr_bank) > 0.5
    args.probe_runs = str(out)
    rec2 = S.Rec(smoke=True)
    S.stage_report(args, PARAMS, rec2)
    m = rec2.metrics
    assert "q_text.A1.100.bank.all.lr.8" in m and m["q_text.A1.100.bank.all.lr.8"]["value"]["n"] == 2
    assert "compare.A1.100.lr.8.all.bank_minus_max_c0y_gplus" in m
    assert "compare.A1.100.lr.8.all.bank_minus_bow" in m
    assert "compare.A1.100.tm.8.dim.bank_minus_c0y" in m
    assert m["tournament.ranking_lr_32"]["value"] == []  # the ranking is defined at 32 labels per class
    assert (out / "q_text_table.json").exists()
