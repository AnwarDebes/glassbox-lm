"""A stub for tu/d3_chain.sh's steps (tests/test_d3_chain.sh only; after the pre-freeze review's stub, review B,
2026-10-23). It records each call and writes the outputs the real step would write, so that the chain's skips and its
python checks run on realistic inputs. It never creates an --out-dir: the chain must. Knobs (env): STUB_CODE_TM
(val_acc of the code-check TM row, default 0.945), STUB_FIT_PASS (true/false), STUB_FIT_EXIT (the pair fit's exit
code, default 0), STUB_FEAT_NOISE (added to the a3B arm's val__exists), STUB_PAIR_BAD (an arm whose meta loses its
pair tokens), STUB_JWC_LOW (1: Jwc's exists below JnoP's). The fake M1 caches and c3 rows are made by the test."""

import json
import os
import sys
from pathlib import Path

import numpy as np

argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(" ".join(argv) + "\n")
script, a = Path(argv[0]).name, argv[1:]


def opt(name, default=None):
    return a[a.index(name) + 1] if name in a else default


d3, m1 = Path(os.environ["D3_DIR"]), Path(os.environ["D3_M1"])
P = 5  # the stub's pair count
if script == "d3_exists_tm.py":
    out, emb = Path(opt("--out-dir")), Path(opt("--emb-dir"))
    arm = "orig" if emb.name == "orig" else emb.name
    val = float(os.environ.get("STUB_CODE_TM", "0.945")) if opt("--tag") == "d3-check" else 0.9
    row = {"tag": opt("--tag"), "study": "3d", "arm": arm, "dataset": opt("--dataset"), "seed": int(opt("--seed")),
           "val_acc": val, "smoke": False, "view": "existsT", "repeat": 1,
           "cfg": {"n_clauses": 4000, "T": 4000, "s": float(opt("--s")), "epochs": 40, "batch_size": 128}}
    with open(out / "results_d3_val.jsonl", "a") as f:  # no mkdir here: the chain must have made the folder
        f.write(json.dumps(row) + "\n")
elif script == "d3_lr.py":
    out, emb = Path(opt("--out-dir")), Path(opt("--emb-dir"))
    arm = "orig" if emb.name == "orig" else emb.name
    val = 0.9503 if opt("--name") == "check" else 0.9
    with open(out / "results_d3_lr.jsonl", "a") as f:
        f.write(json.dumps({"name": opt("--name"), "study": "3d", "arm": arm, "dataset": opt("--dataset"), "lr_val": val,
                            "C": 1.0, "view": "existsT"}) + "\n")
elif script == "d3_fit.py":
    if a == ["--check-a3B"]:
        (d3 / "check_P0.json").write_text(json.dumps({"check": {"pass": os.environ.get("STUB_FIT_PASS", "true") == "true"}}))
    else:
        e = int(os.environ.get("STUB_FIT_EXIT", "0"))
        (d3 / "tables").mkdir(parents=True, exist_ok=True)
        (d3 / "tables" / "d3J_d768.npz").write_bytes(b"stub")
        (d3 / "fit_meta_d3J.json").write_text(json.dumps({"pairs": P, "converged": {"0.01": e == 0}}))
        sys.exit(e)
elif script == "d3_mw.py":
    (d3 / "tables" / "d3MW_d768.npz").write_bytes(b"stub")
    (d3 / "mw_meta.json").write_text("{}")
elif script == "d3_twins.py":
    for n in ("d3MID_d768", "d3BJ_d768"):
        (d3 / "tables" / f"{n}.npz").write_bytes(b"stub")
    (d3 / "twins_meta.json").write_text("{}")
elif script == "d3_feats.py":
    arm, ds = opt("--arm"), opt("--dataset")
    root = Path(opt("--out-root")) / arm
    root.mkdir(parents=True, exist_ok=True)
    z = np.load(m1 / "feats" / "a3B" / f"{ds}__p32tok_K512.npz")
    arr = {k: z[k].copy() for k in ("train__exists", "val__exists", "y_train", "y_val")}
    if arm == "a3B":
        arr["val__exists"] = arr["val__exists"] + float(os.environ.get("STUB_FEAT_NOISE", "0"))
    if arm == "Jwc":
        for sp in ("train", "val"):
            arr[f"{sp}__exists"] = arr[f"{sp}__exists"] + (-1e-3 if os.environ.get("STUB_JWC_LOW") == "1" else 1e-3)
    np.savez(root / f"{ds}__p32tok_K512.npz", **arr)
    np.save(root / f"{ds}__p32tok_K512.centroids.npy", np.ones((4, 3), np.float32))
    on = opt("--pair-tokens") == "on" and arm != os.environ.get("STUB_PAIR_BAD")
    meta = {"arm": arm, "pair_tokens": on, "table_pairs": P if opt("--table").endswith(("d3J_d768.npz", "d3MW_d768.npz",
            "d3MID_d768.npz", "d3BJ_d768.npz")) else 0, "train_vocab_pairs": 3 if on and opt("--kmeans") == "all" else 0}
    (root / f"{ds}__p32tok_K512.meta.json").write_text(json.dumps(meta))
elif script == "d3_verdict.py":
    (d3 / "verdict_d3.json").write_text("{}")
sys.exit(0)
