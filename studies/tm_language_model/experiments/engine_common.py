"""Shared helpers for the engine experiments (M3 gates, s3, s4, TK): corpus loading, training
loops with per-epoch validation, the count-probe and bigram baselines on the same positions
(re-exported from tmlm.eval.baselines), and metric labelling. No experiment logic of its own.

Every experiment script reads its JSON config (experiments/configs), trains the C++ engine
(`CpuEngine`) unless `--oracle` is given, and writes labelled metrics through
`tmlm.common.runlog.write_metrics` when it runs under scripts/run_logged.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402
from tmlm.engine.api import CorpusArrays, EngineBase  # noqa: E402
from tmlm.engine.config import EngineConfig  # noqa: E402


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def make_engine(cfg: EngineConfig, corpus: CorpusArrays, oracle: bool = False, engine: str | None = None) -> EngineBase:
    """engine: "cpu" (default), "cuda" (M4 engine, bit-exact with the CPU engine, needs the GPU lock
    of scripts/gpu_run.sh) or "oracle"; the `oracle` flag is kept for the older call sites."""
    kind = engine or ("oracle" if oracle else "cpu")
    if kind == "oracle":
        from tmlm.engine.oracle.oracle import OracleEngine
        return OracleEngine(cfg, corpus=corpus)
    if kind == "cuda":
        from tmlm.engine.cuda.cuda_engine import CudaEngine
        return CudaEngine(cfg, corpus=corpus)
    if kind != "cpu":
        raise ValueError(f"unknown engine {kind!r}")
    from tmlm.engine.cpu.cpu_engine import CpuEngine
    return CpuEngine(cfg, corpus=corpus)


def top1(engine: EngineBase, epoch: int, positions: np.ndarray, gold: np.ndarray,
         hidden: np.ndarray | None = None, chunk: int = 4096) -> float:
    hits = 0
    for i in range(0, positions.shape[0], chunk):
        p = positions[i:i + chunk]
        hits += int((engine.predict_scores(epoch, p, hidden).argmax(axis=1) == gold[i:i + chunk]).sum())
    return hits / max(1, positions.shape[0])


def train_epochs(engine: EngineBase, epochs: int, on_epoch: Callable[[int, dict], None] | None = None) -> dict:
    """Train `epochs` full epochs; returns totals (examples, wall seconds, summed DIAG counters)."""
    totals: dict[str, float] = {"examples": 0, "wall_s": 0.0}
    for e in range(epochs):
        t0 = time.time()
        diag_sum: dict[str, int] = {}
        for b in range(engine.n_batches()):
            d = engine.train_batch(e, b)
            for k, v in d.items():
                diag_sum[k] = diag_sum.get(k, 0) + v
        wall = time.time() - t0
        totals["examples"] += diag_sum.get("n_examples", 0)
        totals["wall_s"] += wall
        for k, v in diag_sum.items():
            totals[k] = totals.get(k, 0) + v
        if on_epoch is not None:
            on_epoch(e, {"wall_s": wall, **diag_sum})
    return totals


# The count probe and the bigram baseline live in the baselines package since M2; the
# names are re-exported here so the M3 experiments keep working unchanged.
from tmlm.eval.baselines.count_probe import count_probe_tables, count_probe_top1  # noqa: E402, F401
from tmlm.eval.baselines.ngram import left_bigram_top1  # noqa: E402, F401


def write_metrics_or_print(metrics: dict) -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        runlog.write_metrics(metrics)
    else:
        print(json.dumps(metrics, indent=1, default=str))


def measured(value, **extra) -> dict:
    d = {"value": value, "label": "measured"}
    d.update(extra)
    return d


class TinyLmTask:
    """The M3 tiny-LM task exactly as experiments/m3_tiny_lm.py defines it (docs/04 M3, docs/13 H1.4):
    train = the first `n_tokens` of WikiText-103 train at BPE `vocab`, cut at a document boundary;
    evaluation = the FROZEN validation eval set (subword variant), the hidden mask applied in masked
    mode; comparators = left bigram (true left token) and count probe (visible context) on the same
    positions. s3 and s4 reuse it so their last rung / default batch is the tiny LM run itself."""

    def __init__(self, vocab: int = 2000, n_tokens: int = 5_000_000, mode: str = "masked", corpus: str = "wikitext103"):
        from tmlm.data.corpus_store import load_split
        from tmlm.data.evalsets import load_evalset
        from m3_tiny_lm import slice_train
        self.vocab, self.mode = int(vocab), mode
        self.train = slice_train(load_split(corpus, vocab, "train"), int(n_tokens))
        self.val = load_split(corpus, vocab, "validation")
        ev = load_evalset(corpus, vocab, "validation", "subword")
        self.hidden = ev["hidden"].astype(np.uint8) if mode == "masked" else None
        self.targets = ev["targets"].astype(np.uint32)
        self.gold = self.val.tok[self.targets]
        self.evalset_sha256 = ev["meta"].get("sha256")
        self.offsets = (-1, -2, -3) if mode == "causal" else (-3, -2, -1, 1, 2, 3)
        self.count_probe_top1 = count_probe_top1(self.val.tok, self.targets, count_probe_tables(self.train.tok, vocab, self.offsets), self.hidden)
        self.left_bigram_top1 = left_bigram_top1(self.train.tok, vocab, self.targets, self.val.tok)

    def metrics(self) -> dict:
        return {"train_tokens": measured(int(self.train.tok.shape[0])), "val_targets": measured(int(self.targets.shape[0])),
                "evalset_sha256": measured(self.evalset_sha256), "mode": measured(self.mode),
                "left_bigram_top1": measured(self.left_bigram_top1, note="true left token, counts from the train slice"),
                "count_probe_top1": measured(self.count_probe_top1, note=f"offsets {self.offsets}, visible context only")}

    def train_eval(self, cfg: EngineConfig, epochs: int, oracle: bool = False, engine: str | None = None) -> dict:
        """Train `epochs` epochs on the slice, evaluate ONCE after the last epoch on the frozen set
        (the per-epoch curve is the tiny-LM experiment's job; here it would cost more than the training).
        Returns top1, examples/s (training only), n_examples, summed DIAG counters, alive clauses."""
        eng = make_engine(cfg, self.train, oracle=oracle, engine=engine)
        totals = train_epochs(eng, epochs)
        probe = make_engine(cfg, self.val, oracle=oracle, engine=engine)
        probe.load_state(eng.state())
        acc = top1(probe, epochs, self.targets, self.gold, self.hidden)
        n_ex = totals["examples"]
        return {"top1": acc, "examples_per_s": n_ex / max(1e-9, totals["wall_s"]), "n_examples": int(n_ex),
                "alive": int(eng.state()["alive"].sum()), "candidates_per_example": totals.get("n_candidates", 0) / max(1, n_ex),
                "fired_per_example": totals.get("n_fired", 0) / max(1, n_ex), "n_seeded": int(totals.get("n_seeded", 0)),
                "n_recycled": int(totals.get("n_recycled", 0)), "wall_s": totals["wall_s"]}
