#!/usr/bin/env python
"""Timing probe for graph_tm.py (test 3 preparation; no accuracy is recorded).

    prof_graph.py --dataset hwu64 --symbols S --edges chain --depth 2 --clauses 4000 --n 300 --emb-dir DIR

Prints the time to build the graphs, to construct the TM, to fit one epoch on the first n training graphs
(the first fit call includes the CUDA kernel compile, so a second epoch is timed too) and to score n val graphs.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_tm as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--symbols", default="S")
ap.add_argument("--edges", default="chain")
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--clauses", type=int, default=4000)
ap.add_argument("--T", type=int, default=4000)
ap.add_argument("--s", type=float, default=8.0)
ap.add_argument("--max-lits", type=int, default=0)
ap.add_argument("--msg-onehot", type=int, default=1)
ap.add_argument("--message-size", type=int, default=512)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--emb-dir", type=Path, required=True)
a = ap.parse_args()
random.seed(1)
np.random.seed(1)
from GraphTsetlinMachine.graphs import Graphs  # noqa: E402
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine  # noqa: E402

t = time.time()
z = np.load(a.emb_dir / f"{a.dataset}__graph_K512.npz")
g_tr, y_tr = G.build(z, "train", a.symbols, a.edges, Graphs)
print(f"build train {time.time() - t:.1f}s nodes {g_tr.number_of_nodes} symbols {g_tr.hypervector_size}", flush=True)
t = time.time()
g_va, y_va = G.build(z, "val", a.symbols, a.edges, Graphs, init_with=g_tr)
print(f"build val {time.time() - t:.1f}s", flush=True)
t = time.time()
sub, ysub = g_tr[list(range(a.n))], y_tr[: a.n]
vsub, yvsub = g_va[list(range(a.n))], y_va[: a.n]
print(f"subsets {time.time() - t:.1f}s", flush=True)
# the class count comes from max(y) in fit: make sure the subset has the last class, as the full set does
ysub = ysub.copy()
ysub[-1] = y_tr.max()
tm = MultiClassGraphTsetlinMachine(a.clauses, a.T, a.s, depth=a.depth, max_included_literals=a.max_lits or None,
                                   one_hot_encoding=bool(a.msg_onehot), message_size=a.message_size, message_bits=2,
                                   number_of_state_bits=8)
for ep in range(2):
    t = time.time()
    tm.fit(sub, ysub, epochs=1, incremental=True)
    dt = time.time() - t
    print(f"fit epoch {ep + 1} on {a.n} graphs {dt:.1f}s = {1000 * dt / a.n:.2f} ms/graph", flush=True)
t = time.time()
acc = (tm.predict(vsub) == yvsub).mean()
dt = time.time() - t
print(f"predict {a.n} val graphs {dt:.1f}s = {1000 * dt / a.n:.2f} ms/graph (acc {acc:.3f}, meaningless)", flush=True)
