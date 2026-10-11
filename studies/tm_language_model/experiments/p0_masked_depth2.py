"""E1 (task T0.4 of the initial design): masked-token sanity of the STOCK CAIR GraphTsetlinMachine.

One graph per masked position (docs/02 "Graph-level, not node-level, classification"): the
graph is a fixed window of WINDOW tokens around the target position, nodes are tokens, ONE
node (the target) plus every other position hidden by the mask carries the symbol MASK, and
the edges are one shared BANDED signed-distance template of span SPAN (docs/03 Track A:
node p reaches p+d over edge type `dist_+d` and p-d over `dist_-d`, d = 1..SPAN). The label is
the vocabulary id of the hidden target token; the stock multi-class model predicts it.

Data: the first `tokens` tokens of WikiText-103 train at BPE `vocab` cut at a document boundary
(the tiny-LM slice of experiments/m3_tiny_lm.py, docs/04 M3). Training targets of an epoch are a
fresh seeded hidden mask at `mask_rate` over the slice (every hidden position is a target; the
order of the examples is a seeded permutation, because the stock `_fit` never shuffles, docs/02
blocker 2). Evaluation after every epoch: masked top-1 and top-5 on the FROZEN validation eval
set (subword variant, docs/06; the hidden mask is applied to the input, so the model sees MASK at
every hidden position of the window, exactly as the comparators of `--comparators` do). The test
split is never touched.

Stock-package facts this file respects (docs/02):
  - grid=(16*13,1,1), block=(128,1,1) always (blocker 4); np.random is seeded BEFORE the
    GraphTsetlinMachine import (blocker 5) and (seed, grid, block) are logged.
  - `MultiClassGraphTsetlinMachine.fit` sets number_of_outputs = max(Y) + 1 and builds a dense
    (N, CLASSES) int32 label matrix, so with shards of the corpus the class count would follow
    the shard. This file fixes `number_of_outputs = vocab` on the model and calls the stock
    `_fit` with the same +T / -T label matrix that `fit` would build at full width (blocker 1,
    "check this in every harness"). The kernels are not touched.
  - The whole `Graphs` object is uploaded at once (blocker 6), so an epoch is trained in shards
    of `shard` graphs, each a `Graphs` object built from arrays (the bypass of blocker 3: X,
    edge, node_type, number_of_graph_node_edges, node_index and edge_index are filled directly
    and the signature is computed by the stock `encode()`); shards share the codebook through
    `Graphs(n, init_with=base)`.

Modes:
    python experiments/p0_masked_depth2.py --config cell.json            # one cell, one process
    python experiments/p0_masked_depth2.py --config cfg.json --comparators
        # unigram, left/right bigram, count probe, bidirectional KN on the same positions
    python experiments/p0_masked_depth2.py --launch cells.json --max-concurrent 3
        # spawn one logged run per cell under the shared GPU lock, N at a time

Every mode runs under scripts/run_logged.py (or opens its own RunLog) so the numbers land in
results/<timestamp>_<name>/ (metrics.json, plus per-epoch predictions and the saved model).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import p0_common as common  # noqa: E402

CORPUS = "wikitext103"
MASK_SYMBOL = "MASK"

DEFAULT_PARAMS: dict[str, Any] = {
    "vocab": 2000,
    "tokens": 5_000_000,
    "window": 24,
    "span": 8,
    "clauses": 4000,
    "T": 1000,
    "s": 30.0,
    "q": 1.0,
    "max_included_literals": None,
    "depth": 2,
    "hypervector_size": 512,
    "hypervector_bits": 2,
    "message_size": 256,
    "message_bits": 2,
    "number_of_state_bits": 8,
    "mask_rate": 0.15,
    "epochs": 3,
    "shard": 100_000,
    "seed": 1,
    "grid": list(common.SAFE_GRID),
    "block": list(common.SAFE_BLOCK),
    "max_train_examples": None,   # pilots only: cap the targets of every epoch
    "eval_subset": None,          # None = every frozen target; else a fixed seeded subset size
    "eval_subset_seed": 1,
    "save_model": True,
    "init_model": None,           # path of a model.pkl saved by an earlier cell run (resume)
    "start_epoch": 0,             # first epoch index to train when resuming (0-based)
    "kn_orders": [2, 5],          # --comparators only
}


# --------------------------------------------------------------------------- windows


class WindowTemplate:
    """The shared banded edge template of a WINDOW-node graph with signed distance types.

    Edges are grouped by source node in position order, which is what the stock `edge_index`
    (cumulative out-degree) expects. Type ids: `dist_+d` -> d-1, `dist_-d` -> span+d-1.
    """

    def __init__(self, window: int, span: int) -> None:
        if window < 2 or span < 1:
            raise ValueError("window >= 2 and span >= 1")
        self.window, self.span = int(window), int(span)
        self.edge_type_names = [f"dist_+{d}" for d in range(1, span + 1)] + \
                               [f"dist_-{d}" for d in range(1, span + 1)]
        edges: list[tuple[int, int]] = []
        degree = np.zeros(window, dtype=np.uint32)
        for p in range(window):
            for d in range(1, span + 1):
                if p + d < window:
                    edges.append((p + d, d - 1))
                    degree[p] += 1
                if p - d >= 0:
                    edges.append((p - d, span + d - 1))
                    degree[p] += 1
        self.degree = degree
        self.edge = np.asarray(edges, dtype=np.uint32).reshape(-1, 2)
        self.edges_per_graph = int(self.edge.shape[0])

    def register(self, graphs: Any) -> None:
        """Give the codebook holder the node type and edge type ids the arrays use."""
        graphs.node_type_id.setdefault("Plain", 0)
        for i, name in enumerate(self.edge_type_names):
            graphs.edge_type_id[name] = i


def symbol_rows(graphs: Any) -> np.ndarray:
    """One X row (uint32 chunks) per symbol: its hypervector bits set in the first half and
    cleared in the negated half, every other negated bit set (what the stock
    `_initialize_node_hypervectors` plus `_add_graph_node_property` produce for a node that
    carries exactly one symbol)."""
    hv = int(graphs.hypervector_size)
    n_sym = int(graphs.hypervectors.shape[0])
    if (2 * hv) % 32:
        raise ValueError("hypervector_size must be a multiple of 16")
    bits = np.zeros((n_sym, 2 * hv), dtype=np.uint8)
    bits[:, hv:] = 1
    rows = np.repeat(np.arange(n_sym), graphs.hypervectors.shape[1])
    ks = graphs.hypervectors.reshape(-1).astype(np.int64)
    bits[rows, ks] = 1
    bits[rows, ks + hv] = 0
    packed = np.packbits(bits, axis=1, bitorder="little")
    out = np.ascontiguousarray(packed).view(np.uint32)
    if out.shape != (n_sym, graphs.number_of_hypervector_chunks):
        raise AssertionError("chunk layout mismatch")
    return np.ascontiguousarray(out)


def make_codebook(graphs_cls: Any, vocab: int, hypervector_size: int, hypervector_bits: int,
                  template: WindowTemplate) -> Any:
    """The `Graphs` object that holds the symbol codebook (drawn from the seeded global
    np.random, as the stock package does), node type and edge type ids. Its own graph slot is
    never used; every real set is built with init_with=codebook."""
    symbols = [f"tok_{i}" for i in range(vocab)] + [MASK_SYMBOL]
    base = graphs_cls(1, symbols=symbols, hypervector_size=hypervector_size,
                      hypervector_bits=hypervector_bits)
    template.register(base)
    return base


def window_starts(positions: np.ndarray, n_tok: int, window: int) -> np.ndarray:
    return np.clip(positions.astype(np.int64) - window // 2, 0, n_tok - window)


def build_window_graphs(graphs_cls: Any, codebook: Any, rows: np.ndarray, template: WindowTemplate,
                        tok: np.ndarray, hidden: np.ndarray, positions: np.ndarray,
                        mask_id: int) -> Any:
    """A stock `Graphs` object with one WINDOW-node graph per position, built from arrays.

    Node j of the graph for position p carries tok[start + j] (start = p - window//2, shifted
    to stay inside the stream), or MASK where hidden[start + j] is set; the target itself is
    always hidden. The edge template is shared by every graph.
    """
    W = template.window
    n = int(positions.shape[0])
    n_tok = int(tok.shape[0])
    if n_tok < W:
        raise ValueError("stream shorter than the window")
    g = graphs_cls(n, init_with=codebook)
    g.number_of_graph_nodes[:] = W
    g.node_index = (np.arange(n, dtype=np.int64) * W).astype(np.uint32)
    g.max_number_of_graph_nodes = W
    g.max_number_of_graph_node_chunks = (W - 1) // 32 + 1
    g.number_of_nodes = n * W
    g.node_type = np.zeros(n * W, dtype=np.uint32)
    g.number_of_graph_node_edges = np.tile(template.degree, n)
    g.graph_node_edge_counter = g.number_of_graph_node_edges.copy()
    g.edge_index = np.zeros(n * W, dtype=np.uint32)
    g.edge_index[1:] = np.cumsum(g.number_of_graph_node_edges[:-1], dtype=np.int64)
    g.edge = np.tile(template.edge, (n, 1))
    starts = window_starts(positions, n_tok, W)
    idx = starts[:, None] + np.arange(W, dtype=np.int64)[None, :]
    sym = tok[idx].astype(np.int64)
    sym[hidden[idx].astype(bool)] = mask_id
    sym[np.arange(n), positions.astype(np.int64) - starts] = mask_id
    g.X = np.ascontiguousarray(rows[sym.ravel()])
    g.encode()  # no named nodes were registered, so only the signature is computed
    return g


def encoded_labels(y: np.ndarray, vocab: int, T: int) -> np.ndarray:
    """The dense +T / -T matrix that MultiClassGraphTsetlinMachine.fit builds, at width vocab."""
    enc = np.full((int(y.shape[0]), vocab), -int(T), dtype=np.int32)
    enc[np.arange(y.shape[0]), y.astype(np.int64)] = int(T)
    return enc


def fit_graphs(tm: Any, graphs: Any, y: np.ndarray, vocab: int) -> None:
    """One pass of the stock `_fit` over `graphs` with the class count fixed at `vocab`."""
    if getattr(tm, "initialized", False) and tm.number_of_outputs != vocab:
        raise AssertionError("number_of_outputs changed after initialization")
    tm.number_of_outputs = int(vocab)
    tm.max_y = None
    tm.min_y = None
    tm._fit(graphs, encoded_labels(y, vocab, tm.T), epochs=1, incremental=True)


def epoch_targets(n_tok: int, mask_rate: float, seed: int, epoch: int,
                  cap: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Hidden mask and the permuted target list of one training epoch (seeded by seed and epoch)."""
    rng = np.random.default_rng([int(seed), 7, int(epoch)])
    hidden = (rng.random(n_tok) < mask_rate).astype(np.uint8)
    targets = np.flatnonzero(hidden).astype(np.int64)
    targets = targets[rng.permutation(targets.shape[0])]
    if cap is not None:
        targets = targets[:int(cap)]
    return hidden, targets


def eval_targets(ev: dict, subset: int | None, subset_seed: int) -> np.ndarray:
    targets = ev["targets"].astype(np.int64)
    if subset is None or subset >= targets.shape[0]:
        return targets
    rng = np.random.default_rng([int(subset_seed), 11])
    pick = np.sort(rng.choice(targets.shape[0], size=int(subset), replace=False))
    return targets[pick]


def included_literals(tm: Any) -> dict[str, Any]:
    """Per layer: how many literals the clauses include (stock `get_hyperliterals`), as a check
    that the message layers (depth > 1) took part in learning at all."""
    out: dict[str, Any] = {}
    for depth in range(int(tm.depth)):
        lit = tm.get_hyperliterals(depth)
        per_clause = lit.sum(axis=1)
        out[f"layer{depth + 1}"] = {
            "literals_per_clause_mean": float(per_clause.mean()),
            "literals_per_clause_max": int(per_clause.max()),
            "clauses_with_no_literal": int((per_clause == 0).sum()),
            "positive_literals_included": int(lit[:, :lit.shape[1] // 2].sum()),
            "negated_literals_included": int(lit[:, lit.shape[1] // 2:].sum()),
        }
    return out


def topk_hits(class_sums: np.ndarray, gold: np.ndarray, k: int) -> np.ndarray:
    """Boolean per row: gold among the k largest class sums (ties: lowest id first, as argmax)."""
    if k == 1:
        return class_sums.argmax(axis=1) == gold
    idx = np.arange(class_sums.shape[0])
    part = np.argpartition(-class_sums, k - 1, axis=1)[:, :k]
    return (part == gold[:, None]).any(axis=1)


# --------------------------------------------------------------------------- one cell


def load_task(params: dict[str, Any]) -> dict[str, Any]:
    from m3_tiny_lm import slice_train
    from tmlm.data.corpus_store import load_split
    from tmlm.data.evalsets import load_evalset

    vocab = int(params["vocab"])
    train = slice_train(load_split(CORPUS, vocab, "train"), int(params["tokens"]))
    val = load_split(CORPUS, vocab, "validation")
    ev = load_evalset(CORPUS, vocab, "validation", "subword")
    targets = eval_targets(ev, params.get("eval_subset"), int(params.get("eval_subset_seed", 1)))
    return {
        "train_tok": np.ascontiguousarray(train.tok), "train_doc_start": np.ascontiguousarray(train.doc_start),
        "val_tok": np.ascontiguousarray(val.tok), "val_doc_start": np.ascontiguousarray(val.doc_start),
        "val_hidden": ev["hidden"].astype(np.uint8), "targets": targets,
        "gold": np.ascontiguousarray(val.tok)[targets].astype(np.int64),
        "evalset_sha256": ev["meta"].get("sha256"), "evalset_combined": ev["meta"].get("sha256_combined"),
        "n_frozen_targets": int(ev["targets"].shape[0]),
    }


def run_cell(params_in: dict[str, Any], run: Any) -> dict[str, Any]:
    params = {**DEFAULT_PARAMS, **params_in}
    unknown = set(params_in) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown params: {sorted(unknown)}")
    vocab, T = int(params["vocab"]), int(params["T"])
    grid, block = tuple(params["grid"]), tuple(params["block"])
    if grid != common.SAFE_GRID:
        raise ValueError(f"grid must be {common.SAFE_GRID} on the V100 (protocol rule 12)")
    seed = int(params["seed"])

    from tmlm.common.seeds import seed_everything

    seed_everything(seed)  # BEFORE importing GraphTsetlinMachine.tm
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
    import pycuda.driver as cuda

    mem_context_mib = common.gpu_used_mib(cuda)
    tracker = common.AllocationTracker(cuda).install()

    task = load_task(params)
    train_tok, val_tok = task["train_tok"], task["val_tok"]
    targets, gold = task["targets"], task["gold"]
    n_tok = int(train_tok.shape[0])
    W, span = int(params["window"]), int(params["span"])
    template = WindowTemplate(W, span)
    codebook = make_codebook(Graphs, vocab, int(params["hypervector_size"]),
                             int(params["hypervector_bits"]), template)
    rows = symbol_rows(codebook)
    mask_id = codebook.symbol_id[MASK_SYMBOL]

    t0 = time.perf_counter()
    eval_graphs = build_window_graphs(Graphs, codebook, rows, template, val_tok, task["val_hidden"],
                                      targets, mask_id)
    eval_build_s = time.perf_counter() - t0
    print(f"[cell] train slice {n_tok} tokens, {targets.shape[0]} eval targets "
          f"(frozen set {task['n_frozen_targets']}), eval graphs built in {eval_build_s:.1f} s, "
          f"{template.edges_per_graph} edges per graph", flush=True)

    tm = MultiClassGraphTsetlinMachine(
        int(params["clauses"]), T, float(params["s"]), q=float(params["q"]),
        max_included_literals=params["max_included_literals"],
        number_of_state_bits=int(params["number_of_state_bits"]),
        depth=int(params["depth"]), message_size=int(params["message_size"]),
        message_bits=int(params["message_bits"]), grid=grid, block=block,
    )

    metrics: dict[str, Any] = {}

    def m(name: str, value: Any, label: str = "measured", **extra: Any) -> None:
        metrics[name] = {"value": value, "label": label, **extra}

    from tmlm.common import hashing

    codebook_sha = hashing.sha256_array(codebook.hypervectors)
    m("codebook_sha256", codebook_sha, note="symbol hypervectors drawn from the seeded np.random")
    start_epoch = int(params["start_epoch"])
    if params["init_model"]:
        # Resume: TA states, message TA states and clause weights come from the file; the
        # graphs codebook is re-drawn from the same seed (checked by codebook_sha256 against
        # the earlier run); the stock curand stream is re-seeded, so a resumed run is not
        # bit-identical to an uninterrupted one (E1 is an accuracy gate, not the replay demo).
        tm.number_of_outputs = vocab
        tm.load(fname=str(params["init_model"]))
        if tm.number_of_outputs != vocab or tm.max_number_of_graph_nodes != W:
            raise AssertionError("loaded model does not match this cell")
        m("resumed_from", str(params["init_model"]), note="stock load(); curand stream restarts")
    elif start_epoch:
        raise ValueError("start_epoch without init_model")

    m("params", {k: v for k, v in params.items()})
    m("seed_grid_block", {"seed": seed, "grid": list(grid), "block": list(block)})
    m("train_tokens", n_tok)
    m("train_docs", int(task["train_doc_start"].shape[0] - 1))
    m("eval_targets", int(targets.shape[0]),
      note="frozen validation subword set" + ("" if params.get("eval_subset") is None else
                                              f", fixed subset seeded with {params['eval_subset_seed']}"))
    m("evalset_sha256", task["evalset_sha256"])
    m("edges_per_graph", template.edges_per_graph)
    m("eval_graphs_build_s", eval_build_s, unit="s")
    m("hypervector_bits", int(params["hypervector_bits"]), note="stock default, as E0 used")

    epochs = int(params["epochs"])
    shard = int(params["shard"])
    cap = params.get("max_train_examples")
    curve: list[float] = []
    total_examples = 0
    for epoch in range(start_epoch, epochs):
        hidden, ep_targets = epoch_targets(n_tok, float(params["mask_rate"]), seed, epoch, cap)
        n_ex = int(ep_targets.shape[0])
        y_all = train_tok[ep_targets].astype(np.int64)
        t_epoch = time.perf_counter()
        build_s = 0.0
        fit_s = 0.0
        for start in range(0, n_ex, shard):
            sl = ep_targets[start:start + shard]
            t0 = time.perf_counter()
            g = build_window_graphs(Graphs, codebook, rows, template, train_tok, hidden, sl, mask_id)
            build_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            fit_graphs(tm, g, y_all[start:start + shard], vocab)
            cuda.Context.synchronize()
            fit_s += time.perf_counter() - t0
            done = min(start + shard, n_ex)
            print(f"[epoch {epoch + 1}] {done}/{n_ex} examples, fit {fit_s:.0f} s "
                  f"({done / max(fit_s, 1e-9):.0f} ex/s), build {build_s:.0f} s", flush=True)
            del g
            gc.collect()
        epoch_wall = time.perf_counter() - t_epoch
        total_examples += n_ex
        t0 = time.perf_counter()
        class_sums = tm.score(eval_graphs)
        eval_s = time.perf_counter() - t0
        top1 = float(topk_hits(class_sums, gold, 1).mean())
        top5 = float(topk_hits(class_sums, gold, 5).mean())
        curve.append(top1)
        pred = class_sums.argmax(axis=1).astype(np.uint16)
        np.save(run.run_dir / f"predictions_epoch{epoch + 1}.npy", pred)
        used_mib = common.gpu_used_mib(cuda)
        p = f"epoch{epoch + 1}."
        m(p + "top1", top1)
        m(p + "top5", top5)
        m(p + "train_examples", n_ex)
        m(p + "train_wall_s", epoch_wall, unit="s", note="graph building plus stock _fit, all shards")
        m(p + "fit_s", fit_s, unit="s", note="stock _fit only")
        m(p + "graph_build_s", build_s, unit="s")
        m(p + "examples_per_s_fit", n_ex / max(fit_s, 1e-9), unit="examples/s")
        m(p + "eval_wall_s", eval_s, unit="s", note="stock score() over every eval graph")
        m(p + "gpu_used_mib_device_wide", used_mib, unit="MiB",
          note="pycuda mem_get_info; includes every process on the GPU")
        m(p + "gpu_alloc_mib_this_process", tracker.total_bytes / 2**20, unit="MiB",
          note="sum of pycuda mem_alloc sizes of this process (frees not subtracted)")
        print(f"[epoch {epoch + 1}] top1 {top1:.4f} top5 {top5:.4f} | fit {fit_s:.0f} s "
              f"({n_ex / max(fit_s, 1e-9):.0f} ex/s), build {build_s:.0f} s, eval {eval_s:.0f} s, "
              f"GPU used {used_mib:.0f} MiB", flush=True)
        del class_sums
        gc.collect()
    m("best_top1", max(curve))
    m("best_epoch", int(np.argmax(curve)) + 1 + start_epoch)
    m("final_top1", curve[-1])
    m("total_train_examples", total_examples)
    m("gpu_used_mib_context_only", mem_context_mib, unit="MiB")
    m("gpu_alloc_total", tracker.snapshot())
    m("included_literals", included_literals(tm), note="per layer after the last epoch")
    if params["save_model"]:
        tm.save(str(run.run_dir / "model.pkl"))
        m("model_file", "model.pkl", note="stock save(): pickle of ta_state, message_ta_state, weights")
    tracker.uninstall()
    return metrics


# --------------------------------------------------------------------------- comparators


def run_comparators(params_in: dict[str, Any], run: Any) -> dict[str, Any]:
    """The pre-registered comparators and the count probe on the SAME positions and with the
    SAME training data (the slice) as the cells. `truectx` uses the true neighbouring tokens
    (the M3 bigram gate convention); `visible` cuts the context at hidden tokens, which is what
    the TM sees."""
    from tmlm.eval.baselines.count_probe import count_probe_tables, count_probe_top1
    from tmlm.eval.baselines.ngram import (BidirectionalKN, bigram_table, left_bigram_top1, right_bigram_top1,
                                           unigram_top1)

    params = {**DEFAULT_PARAMS, **params_in}
    vocab = int(params["vocab"])
    task = load_task(params)
    train_tok, val_tok = task["train_tok"], task["val_tok"]
    targets, gold, hidden = task["targets"], task["gold"], task["val_hidden"]
    metrics: dict[str, Any] = {}

    def m(name: str, value: Any, label: str = "measured", **extra: Any) -> None:
        metrics[name] = {"value": value, "label": label, **extra}
        print(f"[comparator] {name} = {value}", flush=True)

    m("train_tokens", int(train_tok.shape[0]))
    m("eval_targets", int(targets.shape[0]))
    m("evalset_sha256", task["evalset_sha256"])
    big = bigram_table(train_tok, vocab)
    m("unigram_top1", unigram_top1(train_tok, vocab, targets, val_tok))
    m("left_bigram_top1_truectx", left_bigram_top1(train_tok, vocab, targets, val_tok, table=big),
      note="true left token, hidden or not (H5.1 comparator)")
    m("right_bigram_top1_truectx", right_bigram_top1(train_tok, vocab, targets, val_tok, table=big),
      note="true right token, hidden or not")
    offsets = (-3, -2, -1, 1, 2, 3)
    tables = count_probe_tables(train_tok, vocab, offsets)
    m("count_probe_top1_visible", count_probe_top1(val_tok, targets, tables, hidden),
      note=f"offsets {offsets}, visible context only")
    m("count_probe_top1_truectx", count_probe_top1(val_tok, targets, tables, None),
      note=f"offsets {offsets}, true context")
    del tables
    for order in params["kn_orders"]:
        t0 = time.perf_counter()
        bi = BidirectionalKN.fit(train_tok, task["train_doc_start"], int(order), vocab)
        fit_s = time.perf_counter() - t0
        for side in ("both", "left", "right"):
            for variant, hid in (("visible", hidden), ("truectx", None)):
                pred = bi.predict(val_tok, task["val_doc_start"], targets, hid, side=side)
                m(f"kn{order}_{side}.top1_{variant}", float((pred == gold).mean()),
                  note="modified KN, product of forward and backward models (docs/06)")
        m(f"kn{order}.fit_wall_s", fit_s, unit="s")
        del bi
    return metrics


# --------------------------------------------------------------------------- launcher


def launch(cells_path: Path, max_concurrent: int, results_note: str) -> int:
    """One logged run per cell, `max_concurrent` at a time, under the shared GPU lock."""
    spec = json.loads(cells_path.read_text(encoding="utf-8"))
    base = spec.get("base", {})
    cells = spec["cells"]
    cfg_dir = ROOT / "experiments" / "configs" / "E1_cells"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    python = str(ROOT / ".venv-tm" / "bin" / "python")
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[var] = env.get("TMLM_E1_THREADS", "2")
    running: list[tuple[str, subprocess.Popen]] = []
    codes: dict[str, int] = {}
    queue = list(cells)
    while queue or running:
        while queue and len(running) < max_concurrent:
            cell = queue.pop(0)
            name = cell["name"]
            config = {"experiment": "E1", "docs": "docs/05", "milestone": "M1", "lane": 2,
                      "prereg": ["H5.1"], "script": "experiments/p0_masked_depth2.py",
                      "venv": ".venv-tm", "gpu_lock": "shared", "cell": name,
                      "note": results_note,
                      "params": {**DEFAULT_PARAMS, **base, **cell.get("params", {})}}
            cfg_path = cfg_dir / f"{name}.json"
            cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            cmd = [str(ROOT / "scripts" / "gpu_run.sh"), "shared", "--",
                   str(ROOT / "scripts" / "mem_guard.py"), "--",
                   str(ROOT / "scripts" / "run_logged.py"), "--name", f"E1_{name}",
                   "--config", str(cfg_path), "--",
                   python, str(Path(__file__).resolve()), "--config", str(cfg_path)]
            print(f"[launch] {name}: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            running.append((name, proc))
        time.sleep(5.0)
        still: list[tuple[str, subprocess.Popen]] = []
        for name, proc in running:
            code = proc.poll()
            if code is None:
                still.append((name, proc))
            else:
                codes[name] = code
                print(f"[launch] {name} finished with exit code {code} at "
                      f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        running = still
    print(f"[launch] all done: {json.dumps(codes, sort_keys=True)}", flush=True)
    return 0 if all(c == 0 for c in codes.values()) else 1


# --------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="cell config JSON (the 'params' key)")
    parser.add_argument("--comparators", action="store_true")
    parser.add_argument("--launch", default=None, help="cells JSON: {base: {...}, cells: [{name, params}]}")
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--note", default="", help="free text copied into every launched config")
    args = parser.parse_args(argv)
    if args.launch:
        return launch(Path(args.launch), args.max_concurrent, args.note)
    if args.config is None:
        parser.error("--config is required")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    params = config.get("params", config)
    name = "E1_comparators" if args.comparators else f"E1_{config.get('cell', 'cell')}"
    with common.open_run(name, config) as run:
        metrics = run_comparators(params, run) if args.comparators else run_cell(params, run)
        for key, entry in metrics.items():
            run.add_metric(key, entry["value"], entry["label"],
                           **{k: v for k, v in entry.items() if k not in ("value", "label")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
