"""H2 (docs/05 H2; docs/13 H4.2, H4.3): the engine's self-supervised clauses as layer 1, a TM over
their fired bits as layer 2, trained greedily with layer 1 frozen (the HTM answer of docs/03).

This file holds row (d), the LM row at tiny-LM scale (M3): layer 2 in tabular mode over the
layer-1 fired bits at the target position, predicting the token, vs layer 1 alone, on the frozen
validation targets, 5 seeds, paired one-sided Wilcoxon on the per-seed deltas. Rows (a) to (c)
on the paragraph pairs are M9 and are added here under `--rows a,b,c` when written.

Row d, per seed: (1) train layer 1 = the tiny-LM engine (config `tiny_lm.json`, 4 epochs) on the
5M-token slice and record its top-1 (a replay check against the H1.4 runs); the state is saved into
the run dir as `layer1_seed{S}.npz` for reuse; (2) fired bits: for a fixed seeded subset of
training target positions and for every frozen validation target, the set of layer-1 clauses that
fire (`fired_clauses`, the frozen hidden mask on validation); (3) layer 2 = the engine in tabular
mode (config `H2.json` engine block: F = C1 features, one per layer-1 clause) trained on the
training bits, evaluated on the validation bits; (4) delta = layer-2 top-1 minus layer-1 top-1 on
the same targets; a few layer-2 clauses are printed as `IF clause_17 AND NOT clause_903 THEN token`.

    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name H2_row_d --config experiments/configs/H2.json \\
        --no-gpu-sampler -- .venv-tm/bin/python -u experiments/h2_layer2_greedy.py --rows d
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from engine_common import ROOT, EngineConfig, TinyLmTask, load_json, make_engine, measured, top1, train_epochs, write_metrics_or_print
from tmlm.engine.api import TabularData
from tmlm.engine.config import lit_unpack
from tmlm.engine.rng import rng32


def fired_bits(engine, epoch: int, positions: np.ndarray, hidden: np.ndarray | None, n_features: int) -> np.ndarray:
    x = np.zeros((positions.shape[0], n_features), dtype=np.uint8)
    for i, p in enumerate(positions):
        f = engine.fired_clauses(epoch, int(p), hidden)
        x[i, f] = 1
    return x


def layer2_clause_text(state: dict, cfg: EngineConfig, j: int) -> str | None:
    lits = [(int(l), int(s)) for l, s in zip(state["lit_id"][j], state["lit_state"][j]) if s >= 128 and l != 0xFFFFFFFF]
    if not lits or not state["alive"][j]:
        return None
    parts = []
    for l, _ in lits:
        _, f, neg = lit_unpack(l)
        parts.append(("NOT " if neg else "") + f"clause_{f}")
    votes = [(int(w), int(c)) for c, w in zip(state["vote_cls"][j], state["vote_w"][j]) if c != 0xFFFF]
    if not votes:
        return None
    w, c = max(votes)
    return f"IF {' AND '.join(parts)} THEN token_{c} (vote {w:+d}; {len(votes)} classes)"


def row_d(cfg_json: dict, args, metrics: dict) -> None:
    params = cfg_json["params"].get("row_d", {})
    l1_json = load_json(ROOT / "experiments" / "configs" / params.get("layer1_config", "tiny_lm.json"))
    l1_eng = dict(l1_json["engine"])
    if args.layer1_clauses:
        l1_eng["n_clauses"] = args.layer1_clauses
    epochs1 = args.epochs1 or int(l1_json["params"].get("epochs", 4))
    epochs2 = args.epochs2 or int(params.get("epochs2", 4))
    n_train = args.n_train or int(params.get("n_train_positions", 65536))
    tokens = args.tokens or int(l1_json["params"].get("tokens", 5_000_000))
    task = TinyLmTask(vocab=int(l1_eng["vocab"]), n_tokens=tokens, mode=l1_eng.get("mode", "masked"))
    metrics.update({f"row_d.{k}": v for k, v in task.metrics().items()})
    metrics["row_d.epochs1"], metrics["row_d.epochs2"], metrics["row_d.n_train_positions"] = measured(epochs1), measured(epochs2), measured(n_train)
    l2_base = dict(cfg_json["engine"])
    C1 = int(l1_eng["n_clauses"])
    l2_base["n_features"] = C1
    l2_base["vocab"] = int(l1_eng["vocab"])  # row d predicts the token: the class space is the token vocabulary
    if args.layer2_clauses:
        l2_base["n_clauses"] = args.layer2_clauses
    run_dir = os.environ.get("TMLM_RUN_DIR")
    deltas = []
    for seed in (args.seeds or cfg_json["seeds"]):
        t0 = time.time()
        cfg1 = EngineConfig(**{**l1_eng, "seed": seed})
        eng1 = make_engine(cfg1, task.train)
        train_epochs(eng1, epochs1)
        probe1 = make_engine(cfg1, task.val)
        st1 = eng1.state()
        probe1.load_state(st1)
        acc1 = top1(probe1, epochs1, task.targets, task.gold, task.hidden)
        if run_dir:
            np.savez_compressed(Path(run_dir) / f"layer1_seed{seed}.npz", **st1)
        # fired bits on a fixed, seeded subset of training targets (counter RNG, docs/11 section 3)
        n_tr = int(task.train.targets.shape[0])
        pick = np.array([rng32(seed, 63, i, 0, 0) % n_tr for i in range(n_train)], dtype=np.int64)
        pick = np.unique(task.train.targets[pick])
        x_tr = fired_bits(eng1, epochs1, pick, None, C1)
        y_tr = task.train.tok[pick].astype(np.uint16)
        x_va = fired_bits(probe1, epochs1, task.targets, task.hidden, C1)
        y_va = task.gold.astype(np.uint16)
        t_bits = time.time() - t0
        cfg2 = EngineConfig(**{**l2_base, "seed": seed})
        from tmlm.engine.cpu.cpu_engine import CpuEngine
        eng2 = CpuEngine(cfg2, tabular=TabularData(x=x_tr, y=y_tr))
        t1 = time.time()
        train_epochs(eng2, epochs2)
        wall2 = time.time() - t1
        probe2 = CpuEngine(cfg2, tabular=TabularData(x=x_va, y=y_va))
        st2 = eng2.state()
        probe2.load_state(st2)
        acc2 = top1(probe2, epochs2, np.arange(x_va.shape[0]), y_va, None)
        # layer 1 + layer 2: summed class scores (both integer score vectors on the same targets)
        s1 = np.concatenate([probe1.predict_scores(epochs1, task.targets[i:i + 4096], task.hidden) for i in range(0, task.targets.shape[0], 4096)])
        s2 = np.concatenate([probe2.predict_scores(epochs2, np.arange(i, min(i + 4096, x_va.shape[0])), None) for i in range(0, x_va.shape[0], 4096)])
        acc_sum = float(((s1 + s2).argmax(axis=1) == task.gold).mean())
        delta = acc2 - acc1
        deltas.append(delta)
        texts = [t for t in (layer2_clause_text(st2, cfg2, j) for j in range(cfg2.n_clauses)) if t]
        if run_dir:
            Path(run_dir, f"layer2_clauses_seed{seed}.txt").write_text("\n".join(texts[:200]) + "\n")
        metrics.update({
            f"row_d.seed{seed}.layer1_top1": measured(acc1), f"row_d.seed{seed}.layer2_top1": measured(acc2),
            f"row_d.seed{seed}.layer1_plus_layer2_top1": measured(acc_sum, note="argmax of the summed integer class scores"),
            f"row_d.seed{seed}.delta_points": measured(100.0 * delta), f"row_d.seed{seed}.layer2_alive": measured(int(st2["alive"].sum())),
            f"row_d.seed{seed}.train_bits_mean_ones": measured(float(x_tr.sum(axis=1).mean())),
            f"row_d.seed{seed}.n_train_examples": measured(int(x_tr.shape[0])), f"row_d.seed{seed}.layer2_train_wall_s": measured(wall2),
            f"row_d.seed{seed}.layer1_and_bits_wall_s": measured(t_bits), f"row_d.seed{seed}.layer2_clauses_printed": measured(len(texts)),
        })
        print(f"row d seed {seed}: layer1 {acc1:.4f}  layer2 {acc2:.4f}  sum {acc_sum:.4f}  delta {100 * delta:+.2f} points  "
              f"bits/ex {x_tr.sum(axis=1).mean():.1f}  l2 alive {int(st2['alive'].sum())}  {t_bits:.0f}+{wall2:.0f} s", flush=True)
    if len(deltas) >= 2:
        from scipy.stats import wilcoxon
        d = np.array(deltas)
        try:
            p = float(wilcoxon(d, alternative="greater").pvalue)
        except ValueError as e:  # all-zero deltas
            p = float("nan")
            metrics["row_d.wilcoxon_note"] = measured(str(e))
        metrics["row_d.delta_points_mean"] = measured(100.0 * float(d.mean()))
        metrics["row_d.delta_points_min_max"] = measured([100.0 * float(d.min()), 100.0 * float(d.max())])
        metrics["row_d.wilcoxon_one_sided_p"] = measured(p, note="H4.3: layer 2 > layer 1, paired over seeds")
        metrics["row_d.H4_3_met"] = measured(bool(p < 0.05))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "H2.json"))
    ap.add_argument("--rows", default="d")
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--epochs1", type=int, default=None)
    ap.add_argument("--epochs2", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--layer1-clauses", type=int, default=None, help="smoke only")
    ap.add_argument("--layer2-clauses", type=int, default=None, help="smoke only")
    args = ap.parse_args()
    cfg_json = load_json(args.config)
    metrics = {"rows": measured(args.rows)}
    rows = set(args.rows.split(","))
    if "d" in rows:
        row_d(cfg_json, args, metrics)
    if rows - {"d"}:
        raise SystemExit(f"rows {sorted(rows - {'d'})} are M9 and not written yet")
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
