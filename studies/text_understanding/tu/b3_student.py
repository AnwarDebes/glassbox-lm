#!/usr/bin/env python
"""Step 3b, the TM student (tu/PLAN_3b.md, "Students"): a multi-output CoTM that learns a teacher's code bits from
concept bits.

    python -I b3_student.py train --train T.npz --held H.npz --out-dir DIR --tag NAME [--clauses 4000 --T 4000 --s 8
        --R 4 --epochs 10 --batch 1024 --seed 1 --device cuda --max-hours 3]
    python -I b3_student.py predict --state DIR/NAME.state.npz --x X.npz --out P.npz [--device cuda]

Machine: dl/tm.py's CoTM (read only, unchanged) with n_classes = the number of code bits J and n_features = the input
bits F: one shared pool of clauses, one integer weight per (clause, output). Rule: R feedback calls per minibatch; in
each call every example draws one output j uniformly (torch generator seeded by --seed) and gets target-class
feedback for j when its target bit j is 1 and negative-class feedback for j when it is 0 (CoTM.feedback with
pos = neg = j and complementary masks): the CoTM rule for J binary outputs on one clause pool. A predicted bit is
S_j > 0 (inference rule: an empty clause is silent).
Data files (npz): "X" = np.packbits(bits, axis=1) of the input bits with "n_features", and "Y" = the packed code bits
with "n_outputs"; a held-out file has the same keys. Bits are unpacked per minibatch, so 2M rows fit in memory.
Metrics on the held-out file after every epoch: the mean over outputs of the balanced bit accuracy (outputs whose
held-out target is constant are left out of the mean and counted), the plain bit accuracy, and the implied cosine
cos(pi x Hamming / J) of the predicted code against the target code (meaningful for sign codes such as SimHash).
Stop rule: after the first epoch the projected training time is checked against --max-hours; over it, the run stops
(exit 3) before the second epoch. Writes DIR/NAME.state.npz (ta int16, w int32, cfg) and DIR/NAME.json (cfg, the
curve, seconds, input hashes); never overwrites either.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.tm import CoTM, TMConfig


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def pack(bits: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(bits, np.uint8), axis=1)


def unpack(packed: torch.Tensor, n: int) -> torch.Tensor:
    """[b, ceil(n/8)] uint8 on any device -> [b, n] uint8 bits, the inverse of np.packbits(axis=1) (big-endian)."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(2) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :n].contiguous()


def load(path: Path) -> dict:
    z = np.load(path)
    return {"X": z["X"], "Y": z["Y"] if "Y" in z.files else None, "F": int(z["n_features"]),
            "J": int(z["n_outputs"]) if "n_outputs" in z.files else None}


def predict_bits(tm: CoTM, Xp: np.ndarray, F: int, bs: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, Xp.shape[0], bs):
        xb = unpack(torch.as_tensor(Xp[i : i + bs], device=tm.dev), F)
        out.append((tm.class_sums(xb, bs=bs) > 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, tm.C), np.uint8)


def metrics(pred: np.ndarray, Y: np.ndarray) -> dict:
    pred, Y = pred.astype(bool), Y.astype(bool)
    pos, neg = Y.sum(0), (~Y).sum(0)
    ok = (pos > 0) & (neg > 0)
    tpr = (pred & Y).sum(0)[ok] / pos[ok]
    tnr = (~pred & ~Y).sum(0)[ok] / neg[ok]
    ham = (pred != Y).mean(1)
    return {"balanced_acc": round(float(((tpr + tnr) / 2).mean()) * 100, 3) if ok.any() else None,
            "bit_acc": round(float((pred == Y).mean()) * 100, 3), "constant_outputs": int((~ok).sum()),
            "implied_cos": round(float(np.cos(np.pi * ham).mean()), 4), "on_share_pred": round(float(pred.mean()), 4),
            "on_share_target": round(float(Y.mean()), 4)}


def train(tm: CoTM, tr: dict, held: dict | None, R: int, epochs: int, batch: int, seed: int, max_hours: float = 0.0,
          log=print) -> list[dict]:
    """The rule of the module docstring. Returns the curve; raises SystemExit(3) on the stop rule."""
    n, F, J = tr["X"].shape[0], tr["F"], tr["J"]
    dev = tm.dev
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    order_rng = np.random.default_rng(seed)
    Xd = torch.as_tensor(tr["X"])
    Yd = torch.as_tensor(tr["Y"])
    curve, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        perm = torch.as_tensor(order_rng.permutation(n))
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = unpack(Xd[idx].to(dev), F)
            yb = unpack(Yd[idx].to(dev), J).bool()
            ar = torch.arange(xb.shape[0], device=dev)
            for _ in range(R):
                j = torch.randint(0, J, (xb.shape[0],), generator=gen, device=dev)
                t = yb[ar, j]
                tm.feedback(xb, j, t, j, ~t)
        el = time.time() - t0
        row = {"epoch": ep, "seconds": round(el, 1)}
        if held is not None:
            row.update(metrics(predict_bits(tm, held["X"], F), unpack(torch.as_tensor(held["Y"]), J).numpy()))
        curve.append(row)
        log(json.dumps(row))
        if ep == 1 and max_hours and el * epochs / 3600 > max_hours:
            log(json.dumps({"STOP": f"projected {el * epochs / 3600:.2f} h > {max_hours} h"}))
            raise SystemExit(3)
    return curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_tr = sub.add_parser("train")
    a_tr.add_argument("--train", type=Path, required=True)
    a_tr.add_argument("--held", type=Path)
    a_tr.add_argument("--out-dir", type=Path, required=True)
    a_tr.add_argument("--tag", required=True)
    a_tr.add_argument("--clauses", type=int, default=4000)
    a_tr.add_argument("--T", type=int, default=4000)
    a_tr.add_argument("--s", type=float, default=8.0)
    a_tr.add_argument("--R", type=int, default=4)
    a_tr.add_argument("--epochs", type=int, default=10)
    a_tr.add_argument("--batch", type=int, default=1024)
    a_tr.add_argument("--seed", type=int, default=1)
    a_tr.add_argument("--device", default="cuda")
    a_tr.add_argument("--threads", type=int, default=2)
    a_tr.add_argument("--max-hours", type=float, default=0.0)
    a_pr = sub.add_parser("predict")
    a_pr.add_argument("--state", type=Path, required=True)
    a_pr.add_argument("--x", type=Path, required=True)
    a_pr.add_argument("--out", type=Path, required=True)
    a_pr.add_argument("--device", default="cuda")
    a_pr.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.cmd == "train":
        state, js = a.out_dir / f"{a.tag}.state.npz", a.out_dir / f"{a.tag}.json"
        for p in (state, js):
            if p.exists():
                print(f"refusing to overwrite {p}", file=sys.stderr)
                return 2
        tr = load(a.train)
        held = load(a.held) if a.held else None
        cfg = TMConfig(n_clauses=a.clauses, T=a.T, s=a.s, epochs=a.epochs, batch_size=a.batch, seed=a.seed,
                       half=a.device.startswith("cuda"))
        tm = CoTM(tr["J"], tr["F"], cfg, device=a.device)
        t0 = time.time()
        curve = train(tm, tr, held, a.R, a.epochs, a.batch, a.seed, a.max_hours)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(state, ta=tm.ta.cpu().numpy().astype(np.int16), w=tm.w.cpu().numpy().astype(np.int32),
                 cfg=json.dumps(dataclasses.asdict(cfg)), n_features=tr["F"], n_outputs=tr["J"])
        rec = {"tag": a.tag, "cfg": dataclasses.asdict(cfg), "R": a.R, "device": a.device, "n_train": int(tr["X"].shape[0]),
               "n_held": int(held["X"].shape[0]) if held else 0, "curve": curve, "seconds": round(time.time() - t0, 1),
               "inputs_sha256": {str(a.train): sha16(a.train), **({str(a.held): sha16(a.held)} if a.held else {})},
               "b3_student.py": sha16(Path(__file__).resolve()), "dl/tm.py": sha16(Path(sys.modules[CoTM.__module__].__file__)),
               "state_sha256": sha16(state), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(js, "x") as f:
            f.write(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "curve"}))
        return 0
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    st = np.load(a.state)
    cfg = TMConfig(**json.loads(str(st["cfg"])))
    cfg.half = a.device.startswith("cuda")
    tm = CoTM(int(st["n_outputs"]), int(st["n_features"]), cfg, device=a.device)
    tm.ta = torch.as_tensor(st["ta"].astype(np.int32), device=tm.dev)
    tm.w = torch.as_tensor(st["w"], device=tm.dev)
    x = load(a.x)
    assert x["F"] == int(st["n_features"]), (x["F"], int(st["n_features"]))
    bits = predict_bits(tm, x["X"], x["F"])
    np.savez(a.out, X=pack(bits), n_features=bits.shape[1], state_sha256=sha16(a.state), x_sha256=sha16(a.x))
    print(json.dumps({"out": str(a.out), "rows": int(bits.shape[0]), "bits": int(bits.shape[1]),
                      "on_share": round(float(bits.mean()), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
