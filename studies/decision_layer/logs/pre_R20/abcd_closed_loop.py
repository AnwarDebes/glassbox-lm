#!/usr/bin/env python
"""Closed loop: the decision layer's memory bits come from its OWN earlier presses.

In R9, R9b and R10 the memory bits (done(b), last action = b, at least k actions so far) were
built from the human agent's logged presses (teacher forcing). Here, within every conversation,
the action turns are decided in order, and the memory bits at turn k are built from the model's
own choices at turns 0 .. k-1. The text bits (customer words, last 3 utterances) stay as logged:
the customer's words cannot be replayed differently. A wrong press therefore changes what the
model believes it has done, and can mislead the next decisions (compounding errors).

    abcd_closed_loop.py --registration R9b --name tm_mem_short --kind tm --split dev --seeds 1 2 ...

Models are read from results/confirm/<registration>/<name>/ (TM: model_seed<k>.npz; LR:
lr_weights_seed<k>.npz). With --split test and --out <registration>, predictions are written to
results/confirm/<out>/<name>_closed/ in the format of scripts/report_abcd.py; with --split dev
one JSON line goes to results/explore/abcd_closed_loop.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDFreeView, ABCDView, load_examples, memory_from_rows  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def load_tm(path: Path, n_classes: int, n_bits: int) -> CoTM:
    m = np.load(path)
    cfg = TMConfig(**json.loads(str(m["cfg"])))
    tm = CoTM(n_classes, n_bits, cfg)
    tm.ta = torch.as_tensor(m["ta"].astype(np.int32), device="cuda")
    tm.w = torch.as_tensor(m["w"].astype(np.int32), device="cuda")
    return tm


def closed_loop(score, X: np.ndarray, rows: list[dict], buttons: list[str], n_mem: int) -> np.ndarray:
    """Decide every conversation's action turns in order; memory bits from the model's own choices."""
    pred = np.full(len(rows), -1, dtype=np.int64)
    own_prev = [None] * len(rows)
    by_conv: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_conv.setdefault(r["convo_id"], []).append(i)
    for idx in by_conv.values():
        idx.sort(key=lambda i: rows[i]["k"])
    max_k = max(len(v) for v in by_conv.values())
    Xc = X.copy()
    for k in range(max_k):
        turn = [idx[k] for idx in by_conv.values() if len(idx) > k]
        fake = []
        for i in turn:
            conv = by_conv[rows[i]["convo_id"]]
            prev = [buttons[pred[j]] for j in conv[:k]]
            own_prev[i] = prev
            fake.append({"prev": prev})
        Xc[turn, :n_mem] = memory_from_rows(fake, buttons)
        pred[turn] = score(Xc[turn]).argmax(1)
    assert (pred >= 0).all()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True, help="where the trained models are")
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", required=True, choices=["tm", "lr"])
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="", help="registration folder for test predictions")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--view", default="readable", choices=["readable", "free"], help="free: the bandit's label-free view (R10); models from --model-glob")
    ap.add_argument("--model-file", default="", help="free view: model file pattern with {seed}, e.g. results/explore/model_tm_X_seed{seed}.npz")
    args = ap.parse_args()
    if args.split == "test" and not args.out:
        raise SystemExit("test split only in registered runs (--out <registration>)")
    parts = tuple(args.parts.split(","))
    assert parts[0] == "mem", "memory bits must be the first block"
    tr, blist = load_examples("train")
    ev, _ = load_examples(args.split)
    E = np.load(ROOT / "data" / "emb" / f"abcd__{args.encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    eev = {f: E[f"{args.split}__{f}"] for f in ("cust", "recent", "full")}
    src = ROOT / "results" / "confirm" / args.registration / args.name
    if args.view == "readable":
        view = ABCDView(parts=parts, seed=0).fit(tr, etr, blist)
        assert view.names == json.loads((src / "bit_names.json").read_text()), "view differs from the trained models'"
    else:
        view = ABCDFreeView(parts=parts, seed=0).fit(tr, {k: v for k, v in etr.items() if k != "full"}, blist)
    X = view.transform(ev, eev)
    n_mem = 2 * len(blist) + 6
    y = np.array([r["y"] for r in ev])
    C_ = len(blist)
    for seed in args.seeds:
        t0 = time.time()
        if args.kind == "tm":
            mpath = (ROOT / args.model_file.format(seed=seed)) if args.model_file else (src / f"model_seed{seed}.npz")
            if args.view == "free":
                assert [str(n) for n in np.load(mpath)["names"]] == view.names, "view differs from the trained model's"
            tm = load_tm(mpath, C_, X.shape[1])

            def score(Z):
                return tm.class_sums(torch.as_tensor(Z, dtype=torch.uint8, device="cuda")).cpu().numpy()
        else:
            wts = np.load(src / f"lr_weights_seed{seed}.npz")
            W, b = wts["W"], wts["b"]

            def score(Z):
                return Z.astype(np.float32) @ W + b
        p_open = score(X).argmax(1)
        p_closed = closed_loop(score, X, ev, blist, n_mem)
        ks = np.array([r["k"] for r in ev])
        rec = {"tag": args.tag, "registration": args.registration, "name": args.name, "kind": args.kind, "split": args.split,
               "seed": seed, "acc_open": float((p_open == y).mean()), "acc_closed": float((p_closed == y).mean()),
               "closed_by_index": {str(k): float((p_closed == y)[ks == k].mean()) for k in range(5)} | {"5+": float((p_closed == y)[ks >= 5].mean())},
               "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if args.split == "test":
            out = ROOT / "results" / "confirm" / args.out / f"{args.name}_closed"
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"seed{seed}.npz", pred=p_closed, y=y, convo_id=np.array([r["convo_id"] for r in ev]), k=ks,
                                w_bert=np.array([r["ast_count_bert"] for r in ev]), w_roberta=np.array([r["ast_count_roberta"] for r in ev]))
            (out / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
        else:
            with (ROOT / "results" / "explore" / "abcd_closed_loop.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
