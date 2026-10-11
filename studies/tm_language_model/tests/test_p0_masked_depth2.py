"""experiments/p0_masked_depth2.py: the array-built window graphs must be byte-identical to
what the stock Graphs API (add_graph_node / add_graph_node_edge / add_graph_node_property)
produces for the same windows. No GPU: GraphTsetlinMachine.graphs imports numba only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import p0_masked_depth2 as e1  # noqa: E402

graphs_mod = pytest.importorskip("GraphTsetlinMachine.graphs")
Graphs = graphs_mod.Graphs


def _stock_graphs(codebook, template, tok, hidden, positions, mask_id):
    W = template.window
    n = positions.shape[0]
    g = Graphs(n, init_with=codebook)
    for gid in range(n):
        g.set_number_of_graph_nodes(gid, W)
    g.prepare_node_configuration()
    for gid in range(n):
        for p in range(W):
            g.add_graph_node(gid, f"n{p}", int(template.degree[p]))
    g.prepare_edge_configuration()
    starts = e1.window_starts(positions, tok.shape[0], W)
    for gid in range(n):
        for p in range(W):
            for d in range(1, template.span + 1):
                if p + d < W:
                    g.add_graph_node_edge(gid, f"n{p}", f"n{p + d}", f"dist_+{d}")
                if p - d >= 0:
                    g.add_graph_node_edge(gid, f"n{p}", f"n{p - d}", f"dist_-{d}")
        for p in range(W):
            t = int(starts[gid]) + p
            hid = hidden[t] or t == int(positions[gid])
            g.add_graph_node_property(gid, f"n{p}", e1.MASK_SYMBOL if hid else f"tok_{int(tok[t])}")
    g.encode()
    return g


def test_array_builder_matches_stock_api():
    np.random.seed(3)
    vocab, W, span = 50, 8, 3
    template = e1.WindowTemplate(W, span)
    codebook = e1.make_codebook(Graphs, vocab, 64, 2, template)
    rows = e1.symbol_rows(codebook)
    mask_id = codebook.symbol_id[e1.MASK_SYMBOL]
    rng = np.random.default_rng(0)
    tok = rng.integers(0, vocab, size=40).astype(np.uint16)
    hidden = (rng.random(40) < 0.3).astype(np.uint8)
    positions = np.array([0, 3, 20, 39], dtype=np.int64)  # both stream ends included
    mine = e1.build_window_graphs(Graphs, codebook, rows, template, tok, hidden, positions, mask_id)
    stock = _stock_graphs(codebook, template, tok, hidden, positions, mask_id)
    assert np.array_equal(mine.X, stock.X)
    assert np.array_equal(mine.edge, stock.edge)
    assert np.array_equal(mine.edge_index, stock.edge_index)
    assert np.array_equal(mine.node_index, stock.node_index)
    assert np.array_equal(mine.number_of_graph_node_edges, stock.number_of_graph_node_edges)
    assert np.array_equal(mine.node_type, stock.node_type)
    assert mine.max_number_of_graph_nodes == stock.max_number_of_graph_nodes
    assert mine.signature == stock.signature


def test_encoded_labels_match_stock_fit_matrix():
    y = np.array([0, 2, 1], dtype=np.int64)
    enc = e1.encoded_labels(y, 3, 100)
    ref = np.empty((3, 3), dtype=np.int32)
    for i in range(3):
        ref[:, i] = np.where(y == i, 100, -100)
    assert np.array_equal(enc, ref)


def test_topk_hits():
    cs = np.array([[5, 1, 3], [0, 0, 9]], dtype=np.int32)
    gold = np.array([2, 2])
    assert e1.topk_hits(cs, gold, 1).tolist() == [False, True]
    assert e1.topk_hits(cs, gold, 2).tolist() == [True, True]


def test_epoch_targets_are_seeded_and_capped():
    h1, t1 = e1.epoch_targets(1000, 0.15, 1, 0, None)
    h2, t2 = e1.epoch_targets(1000, 0.15, 1, 0, None)
    h3, t3 = e1.epoch_targets(1000, 0.15, 1, 1, 10)
    assert np.array_equal(h1, h2) and np.array_equal(t1, t2)
    assert not np.array_equal(h1, h3) and t3.shape[0] == 10
    assert set(t1.tolist()) == set(np.flatnonzero(h1).tolist())
