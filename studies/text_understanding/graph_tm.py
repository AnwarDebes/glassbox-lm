#!/usr/bin/env python
"""Test 3 (LOG.md): CAIR's Graph Tsetlin Machine reading a sentence as a graph of its tokens.

    graph_tm.py --dataset clinc150 --symbols S|W|SW --edges none|chain|chain+other --depth 1|2 \
                --seed 1 --emb-dir DIR --tag TAG [--clauses 4000 --T 4000 --s 8 --epochs 20] [--test] [--smoke]

One node per token (potion-base-32M tokenizer), in sentence order. Node symbols, one-hot (one bit each):
- S: meaning bits from graph_feats.py, "cos(token, concept k) >= t" (the OR over the nodes is the existsT
     vector of test 2, checked there);
- W: the token itself, for tokens seen in training (CAIR's own text encoding uses word symbols).
Edges: none; chain = Left / Right to the neighbouring tokens; chain+other = also "Other" to every other token.
A clause holds at a node; the graph matches the clause if some node does (existential over the tokens).
Depth 2 adds one round of messages, so a clause can also say what a neighbour (or another token) satisfies.

Run under the shared GPU lock, with the .venv-tm python (GraphTsetlinMachine vendored by the tm_language_model
study, used read only, stock files). The epoch is chosen on val; the state at that epoch is restored (all TA
states, message TA states and clause weights) and val is scored again as a check. --test scores the test
split once, after the restore. Appends one JSON line to results_graph.jsonl (--smoke: 1 epoch, nothing written).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EDGE_TYPES = {"none": [], "chain": ["Left", "Right"], "chain+other": ["Left", "Right", "Other"]}


def node_edges(n: int, edges: str):
    """For a graph of n nodes: per node, the list of (destination, edge type name)."""
    out = []
    for j in range(n):
        e = []
        if edges in ("chain", "chain+other"):
            if j > 0:
                e.append((j - 1, "Left"))
            if j < n - 1:
                e.append((j + 1, "Right"))
        if edges == "chain+other":
            e += [(k, "Other") for k in range(n) if k != j]
        out.append(e)
    return out


def build(z, sp, symbols, edges, Graphs, init_with=None, nodes="token"):
    tok, off = z[f"{sp}__tok"], z[f"{sp}__off"]
    N = len(off) - 1
    if nodes == "sentence":  # one node per sentence holding the OR of its token symbols (= flat existsT bits)
        assert edges == "none"
    n_sym = int(z["n_sym"])
    used = z["used"]
    lut = np.full(int(used.max()) + 1, -1, np.int64)
    lut[used] = np.arange(len(used))
    S = np.unpackbits(z["node_bits"], axis=1)[:, :n_sym].astype(bool) if "S" in symbols else None
    wv = z["train_vocab"]
    wlut = np.full(int(used.max()) + 1, -1, np.int64)
    wlut[wv] = np.arange(len(wv))
    names = []
    if "S" in symbols:
        names += [f"s{j}" for j in range(n_sym)]
    if "W" in symbols:
        names += [f"w{t}" for t in wv]
    if init_with is None:
        g = Graphs(N, symbols=names, one_hot_encoding=True)
        for e in EDGE_TYPES[edges]:  # fixed edge-type ids, shared with the val/test graphs
            g.edge_type_id[e] = len(g.edge_type_id)
    else:
        g = Graphs(N, init_with=init_with)
    lens = np.diff(off)
    nn = np.maximum(lens, 1) if nodes == "token" else np.ones_like(lens)  # an empty sentence: one node, no symbols
    for i in range(N):
        g.set_number_of_graph_nodes(i, int(nn[i]))
    g.prepare_node_configuration()
    per_graph = {}
    for i in range(N):
        n = int(nn[i])
        if n not in per_graph:
            per_graph[n] = node_edges(n, edges)
        for j, e in enumerate(per_graph[n]):
            g.add_graph_node(i, j, len(e))
    g.prepare_edge_configuration()
    tid = g.edge_type_id
    for i in range(N):
        base = int(g.node_index[i])
        for j, e in enumerate(per_graph[int(nn[i])]):
            k0 = int(g.edge_index[base + j])
            for m, (dst, name) in enumerate(e):
                g.edge[k0 + m, 0] = dst
                g.edge[k0 + m, 1] = tid[name]
    g.graph_node_edge_counter[:] = g.number_of_graph_node_edges
    # node symbols: bits 0..hv-1 = symbol present, hv..2hv-1 = symbol absent (the layout of Graphs)
    hv = g.hypervector_size
    assert hv == len(names)
    width = g.X.shape[1] * 32
    for s0 in range(0, N, 2048):
        s1 = min(N, s0 + 2048)
        rows = []
        for i in range(s0, s1):
            t = tok[off[i] : off[i + 1]]
            P = np.zeros((len(t), hv), bool)
            if len(t):
                c = 0
                if S is not None:
                    P[:, :n_sym] = S[lut[t]]
                    c = n_sym
                if "W" in symbols:
                    w = wlut[t]
                    hit = w >= 0
                    P[np.arange(len(t))[hit], c + w[hit]] = True
            if nodes == "sentence":
                P = P.any(0, keepdims=True)
            elif not len(t):
                P = np.zeros((1, hv), bool)
            rows.append(P)
        P = np.concatenate(rows, 0)
        full = np.zeros((P.shape[0], width), bool)
        full[:, :hv] = P
        full[:, hv : 2 * hv] = ~P
        a, b = int(g.node_index[s0]), int(g.node_index[s1 - 1] + nn[s1 - 1])
        g.X[a:b] = np.packbits(full, axis=1, bitorder="little").view(np.uint32)
    g.encode()
    return g, z[f"y_{sp}"].astype(np.uint32)


def snapshot(tm, cuda):
    ta = np.empty(tm.number_of_clauses * tm.number_of_ta_chunks * tm.number_of_state_bits, np.uint32)
    cuda.memcpy_dtoh(ta, tm.ta_state_gpu)
    w = np.empty(tm.number_of_outputs * tm.number_of_clauses, np.int32)
    cuda.memcpy_dtoh(w, tm.clause_weights_gpu)
    msg = []
    for d in range(tm.depth - 1):
        m = np.empty(tm.number_of_clauses * tm.number_of_message_chunks * tm.number_of_state_bits, np.uint32)
        cuda.memcpy_dtoh(m, tm.message_ta_state_gpu[d])
        msg.append(m)
    return ta, w, msg


def restore(tm, snap, cuda):
    cuda.memcpy_htod(tm.ta_state_gpu, snap[0])
    cuda.memcpy_htod(tm.clause_weights_gpu, snap[1])
    for d, m in enumerate(snap[2]):
        cuda.memcpy_htod(tm.message_ta_state_gpu[d], m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--symbols", required=True, choices=("S", "W", "SW"))
    ap.add_argument("--edges", required=True, choices=tuple(EDGE_TYPES))
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--nodes", default="token", choices=("token", "sentence"))
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--max-lits", type=int, default=0, help="0 = no limit")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--test", action="store_true", help="score the test split once, at the best-val epoch")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.nodes == "sentence" and args.edges != "none":
        sys.exit("one node per sentence has no edges")
    if args.edges == "none" and args.depth != 1:
        sys.exit("edges none needs depth 1")
    random.seed(args.seed)  # pycuda's XORWOW generator takes its seed from Python's random at import
    np.random.seed(args.seed)
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
    import pycuda.driver as cuda

    t0 = time.time()
    z = np.load(args.emb_dir / f"{args.dataset}__graph_K{args.K}.npz")
    g_tr, y_tr = build(z, "train", args.symbols, args.edges, Graphs, nodes=args.nodes)
    g_va, y_va = build(z, "val", args.symbols, args.edges, Graphs, init_with=g_tr, nodes=args.nodes)
    build_s = time.time() - t0
    tm = MultiClassGraphTsetlinMachine(args.clauses, args.T, args.s, depth=args.depth,
                                       max_included_literals=args.max_lits or None, one_hot_encoding=True,
                                       number_of_state_bits=8)
    epochs = 1 if args.smoke else args.epochs
    best, snap, best_epoch, curve, ep_s = -1.0, None, None, [], []
    for ep in range(epochs):
        t1 = time.time()
        tm.fit(g_tr, y_tr, epochs=1, incremental=True)
        ep_s.append(round(time.time() - t1, 1))
        if (ep + 1) % args.eval_every == 0 or ep == epochs - 1:
            acc = float((tm.predict(g_va) == y_va).mean())
            curve.append([ep + 1, round(acc, 4)])
            print(f"epoch {ep + 1} val {acc:.4f} ({ep_s[-1]}s/epoch)", flush=True)
            if acc > best:
                best, best_epoch, snap = acc, ep + 1, snapshot(tm, cuda)
    restore(tm, snap, cuda)
    check = float((tm.predict(g_va) == y_va).mean())
    rec = {"tag": args.tag, "dataset": args.dataset, "nodes": args.nodes, "symbols": args.symbols, "edges": args.edges, "depth": args.depth,
           "clauses": args.clauses, "T": args.T, "s": args.s, "max_lits": args.max_lits, "epochs": epochs,
           "seed": args.seed, "n_symbols": int(g_tr.hypervector_size), "val_acc": round(best, 4),
           "val_acc_check": round(check, 4), "best_epoch": best_epoch, "curve": curve,
           "epoch_s_median": float(np.median(ep_s)), "build_s": round(build_s, 1)}
    if args.test:
        g_te, y_te = build(z, "test", args.symbols, args.edges, Graphs, init_with=g_tr, nodes=args.nodes)
        rec["test_acc"] = round(float((tm.predict(g_te) == y_te).mean()), 4)
        rec["n_test"] = int(len(y_te))
    rec["total_s"] = round(time.time() - t0, 1)
    if args.smoke:
        print(json.dumps({"smoke": True, **{k: rec[k] for k in ("n_symbols", "val_acc", "val_acc_check", "epoch_s_median", "build_s", "total_s")}}))
        return
    with open(HERE / "results_graph.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
