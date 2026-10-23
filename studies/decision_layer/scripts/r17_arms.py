#!/usr/bin/env python
"""R17: put the open-loop runs (R17's new TM bandits) and R10's no-memory bandits into the
format of scripts/report_abcd.py (one folder per arm, seed<k>.npz with pred, y, convo_id, k,
AST weights, and seed<k>.json). Copies predictions only; nothing is recomputed."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl.abcd import load_examples  # noqa: E402

te, _ = load_examples("test")
meta = {"convo_id": np.array([r["convo_id"] for r in te]), "k": np.array([r["k"] for r in te]),
        "w_bert": np.array([r["ast_count_bert"] for r in te]), "w_roberta": np.array([r["ast_count_roberta"] for r in te])}
base = ROOT / "results" / "confirm" / "R17"
for name, src, pattern in (("tm_bandit_open", base, "preds_tm_R17_seed{k}.npz"),
                           ("tm_bandit_nomem_R10", ROOT / "results" / "confirm" / "R10", "preds_tm_R10nomem_seed{k}.npz")):
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    for k in range(1, 6):
        z = np.load(src / pattern.format(k=k))
        y = z["y"]
        assert np.array_equal(y, np.array([r["y"] for r in te]))
        np.savez_compressed(out / f"seed{k}.npz", pred=z["pred"], y=y, **meta)
        (out / f"seed{k}.json").write_text(json.dumps({"name": name, "seed": k, "source": str((src / pattern.format(k=k)).relative_to(ROOT)),
                                                        "test_acc": float((z["pred"] == y).mean())}))
    print(name, "written")
