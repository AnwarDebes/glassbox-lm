"""E1 diagnosis (docs/05 E1 "Failure: stop and diagnose"): can the stock GraphTsetlinMachine
learn a POSITIONAL copy task through the E1 graph pipeline at all?

Synthetic stream of `vocab` uniformly random tokens; the E1 windows (24 nodes, banded
signed-distance edges of span 8, MASK at the target, experiments/p0_masked_depth2.py) with the
label = the token at a fixed offset from the target (`task` "left1": tok[p-1], "right1":
tok[p+1], "left2": tok[p-2]). Nothing but the position-bound neighbour predicts the label, so:
  - depth 1 (clause = OR over the nodes, no position) can only reach chance = 1/vocab;
  - depth 2 and 3 must reach well above chance if the message passing carries the offset.
At `vocab` = 64 this isolates the pipeline from the class count; at `vocab` = 2000 it shows what
the stock multi-class model does with 2000 equiprobable classes on a task that IS learnable.
Held-out evaluation: `n_eval` fresh windows of the same stream. Runs under the shared lock:

    scripts/gpu_run.sh shared -- scripts/run_logged.py --name E1_diag_<tag> --config cfg.json -- \
        python experiments/p0_masked_depth2.py ...   (see --config example in DEFAULT_PARAMS)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as common  # noqa: E402
import p0_masked_depth2 as e1  # noqa: E402

DEFAULT_PARAMS: dict[str, Any] = {
    "vocab": 64, "n_tok": 400_000, "task": "left1",
    "window": 24, "span": 8, "hypervector_size": 512, "hypervector_bits": 2,
    "clauses": 4000, "T": 1000, "s": 30.0, "q": 1.0, "max_included_literals": None,
    "depth": 2, "message_size": 256, "message_bits": 2, "number_of_state_bits": 8,
    "epochs": 3, "examples_per_epoch": 50_000, "n_eval": 5000, "shard": 25_000, "seed": 1,
    "grid": list(common.SAFE_GRID), "block": list(common.SAFE_BLOCK),
}
OFFSETS = {"left1": -1, "right1": 1, "left2": -2}


def run(params_in: dict[str, Any], run: Any) -> dict[str, Any]:
    params = {**DEFAULT_PARAMS, **params_in}
    unknown = set(params_in) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown params: {sorted(unknown)}")
    vocab, W, seed = int(params["vocab"]), int(params["window"]), int(params["seed"])
    grid, block = tuple(params["grid"]), tuple(params["block"])
    if grid != common.SAFE_GRID:
        raise ValueError("grid must be (16*13,1,1)")
    offset = OFFSETS[params["task"]]

    from tmlm.common.seeds import seed_everything

    seed_everything(seed)
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
    import pycuda.driver as cuda

    rng = np.random.default_rng([seed, 99])
    tok = rng.integers(0, vocab, size=int(params["n_tok"])).astype(np.uint16)
    hidden = np.zeros(tok.shape[0], dtype=np.uint8)
    template = e1.WindowTemplate(W, int(params["span"]))
    codebook = e1.make_codebook(Graphs, vocab, int(params["hypervector_size"]),
                                int(params["hypervector_bits"]), template)
    rows = e1.symbol_rows(codebook)
    mask_id = codebook.symbol_id[e1.MASK_SYMBOL]
    lo, hi = W, tok.shape[0] - W
    n_eval = int(params["n_eval"])
    eval_pos = rng.choice(np.arange(hi - 2 * n_eval, hi), size=n_eval, replace=False).astype(np.int64)
    eval_graphs = e1.build_window_graphs(Graphs, codebook, rows, template, tok, hidden, eval_pos, mask_id)
    gold = tok[eval_pos + offset].astype(np.int64)

    tm = MultiClassGraphTsetlinMachine(
        int(params["clauses"]), int(params["T"]), float(params["s"]), q=float(params["q"]),
        max_included_literals=params["max_included_literals"],
        number_of_state_bits=int(params["number_of_state_bits"]), depth=int(params["depth"]),
        message_size=int(params["message_size"]), message_bits=int(params["message_bits"]),
        grid=grid, block=block)
    metrics: dict[str, Any] = {}

    def m(name: str, value: Any, label: str = "measured", **extra: Any) -> None:
        metrics[name] = {"value": value, "label": label, **extra}

    m("params", params)
    m("seed_grid_block", {"seed": seed, "grid": list(grid), "block": list(block)})
    m("chance_top1", 1.0 / vocab, "derived")
    shard = int(params["shard"])
    for epoch in range(int(params["epochs"])):
        erng = np.random.default_rng([seed, 5, epoch])
        pos = erng.choice(np.arange(lo, hi - 2 * n_eval), size=int(params["examples_per_epoch"]),
                          replace=False).astype(np.int64)
        y_all = tok[pos + offset].astype(np.int64)
        t0 = time.perf_counter()
        for start in range(0, pos.shape[0], shard):
            g = e1.build_window_graphs(Graphs, codebook, rows, template, tok, hidden,
                                       pos[start:start + shard], mask_id)
            e1.fit_graphs(tm, g, y_all[start:start + shard], vocab)
            cuda.Context.synchronize()
        fit_s = time.perf_counter() - t0
        cs = tm.score(eval_graphs)
        top1 = float(e1.topk_hits(cs, gold, 1).mean())
        m(f"epoch{epoch + 1}.top1", top1)
        m(f"epoch{epoch + 1}.examples_per_s", pos.shape[0] / fit_s, unit="examples/s")
        print(f"[diag {params['task']} V={vocab} depth={params['depth']}] epoch {epoch + 1}: "
              f"top1 {top1:.4f} (chance {1 / vocab:.4f}), {pos.shape[0] / fit_s:.0f} ex/s", flush=True)
    m("final_top1", top1)
    m("included_literals", e1.included_literals(tm), note="per layer after the last epoch")
    weights = tm.get_weights()
    m("clause_weights", {"abs_mean": float(np.abs(weights).mean()), "abs_max": int(np.abs(weights).max())})
    m("beats_chance_10x", bool(top1 > 10.0 / vocab), "derived")
    return metrics


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    params = config.get("params", config)
    with common.open_run(f"E1_diag_{config.get('tag', 'diag')}", config) as r:
        metrics = run(params, r)
        for key, entry in metrics.items():
            r.add_metric(key, entry["value"], entry["label"],
                         **{k: v for k, v in entry.items() if k not in ("value", "label")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
