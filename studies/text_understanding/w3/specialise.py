#!/usr/bin/env python
"""W3 step 1 (EXPLORATORY, intent VAL only; rules in LOG.md before any run): the lexicon-specialised C4 tables.

    specialise.py [--device cuda] [--smoke]

Input: the unit rows of ppw_S1000 (w1/tables/ppw_S1000_d500_e0.5.npz; rows without a vector stay zero) and
w3/pairs/oewn_pairs.npz (lexicon_pairs.py). Tables, all non-neural vector updates:
- RF, retrofitting (Faruqui et al. 2015), synonyms only: 10 Jacobi sweeps of q_i = (v_i + sum_{j in N(i)} q_j / deg_i) / 2
  for words with deg_i > 0 (alpha 1, beta 1 / deg_i); the other rows unchanged. (Faruqui's code updates in place,
  word by word; the Jacobi form is declared.)
- CF, counter-fitting (Mrksic et al. 2016): minimise AR + SA + VSP over the rows of the words in a pair, d = 1 - cos:
  AR = sum over antonym pairs of relu(1.0 - d), SA = sum over synonym pairs of relu(d - 0.0),
  VSP = sum over (i, j), i a moved word and j one of its 10 nearest words in ppw_S1000 (no radius: in this 500-d
  counting space few words have neighbours within the paper's rho = 0.2, so the radius would leave the moved words
  unanchored; a declared adaptation), of relu(d(i, j) - d0(i, j)); equal weights; full-batch Adam, lr 0.01, 300
  steps, torch.manual_seed(0). (Our implementation of the published objective; the paper optimised it by SGD.)
- CFrand, the control: CF with the control graphs of lexicon_pairs.py (the same words and degrees, pairs re-drawn).
Rows are written unit length, float16, in ppw_S1000's format (vocab_ids, codes, counts, words), as
w3/tables/{CF,RF,CFrand}_d500_e0.5.npz with w3/tables/specialise.json (mean cosine of the pairs before and after,
the mean change of the moved rows, hashes). Refuses to overwrite. --smoke: 2,000 pairs per set, 20 steps,
w3/tables_smoke/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
PAIRS = HERE / "pairs" / "oewn_pairs.npz"


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def cosines(T: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    if P.shape[0] == 0:
        return torch.zeros(0, device=T.device)
    return torch.nn.functional.cosine_similarity(T[P[:, 0]], T[P[:, 1]], dim=1)


def retrofit(V: torch.Tensor, S: torch.Tensor, sweeps: int = 10) -> torch.Tensor:
    n = V.shape[0]
    src = torch.cat([S[:, 0], S[:, 1]])
    dst = torch.cat([S[:, 1], S[:, 0]])
    deg = torch.bincount(src, minlength=n).to(V.dtype)
    has = deg > 0
    Q = V.clone()
    for _ in range(sweeps):
        acc = torch.zeros_like(V).index_add_(0, src, Q[dst])
        upd = (V + acc / deg.clamp(min=1)[:, None]) / 2
        Q = torch.where(has[:, None], upd, V)
    return Q


def vsp_pairs(V: torch.Tensor, moved: torch.Tensor, k: int = 10, chunk: int = 1024) -> tuple:
    I, J, D = [], [], []
    for s in range(0, moved.shape[0], chunk):
        rows = moved[s : s + chunk]
        C = V[rows] @ V.T
        C[torch.arange(rows.shape[0], device=V.device), rows] = -2.0
        top = C.topk(k, dim=1)
        keep = top.values > -2.0  # every one of the k nearest (the word itself is masked out)
        r = rows[:, None].expand_as(top.indices)
        I.append(r[keep])
        J.append(top.indices[keep])
        D.append(1.0 - top.values[keep])
    return torch.cat(I), torch.cat(J), torch.cat(D)


def counterfit(V: torch.Tensor, S: torch.Tensor, A: torch.Tensor, steps: int = 300, lr: float = 0.01) -> tuple:
    torch.manual_seed(0)
    moved = torch.unique(torch.cat([S.reshape(-1), A.reshape(-1)]))
    I, J, D0 = vsp_pairs(V, moved)
    P = torch.nn.Parameter(V[moved].clone())
    opt = torch.optim.Adam([P], lr=lr)
    hist = []
    for step in range(steps):
        T = V.index_copy(0, moved, P)
        ar = torch.relu(1.0 - (1.0 - cosines(T, A))).sum()
        sa = torch.relu((1.0 - cosines(T, S)) - 0.0).sum()
        vsp = torch.relu((1.0 - torch.nn.functional.cosine_similarity(T[I], T[J], dim=1)) - D0).sum()
        loss = ar + sa + vsp
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, steps - 1) or step % 50 == 0:
            hist.append({"step": step, "AR": float(ar), "SA": float(sa), "VSP": float(vsp)})
    with torch.no_grad():
        T = V.index_copy(0, moved, P.detach())
    return T, {"moved_words": int(moved.shape[0]), "vsp_pairs": int(I.shape[0]), "loss": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--out-dir", type=Path, default=None)  # a test copy elsewhere
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(2)
    out = args.out_dir or HERE / ("tables_smoke" if args.smoke else "tables")
    names = ("CF", "RF", "CFrand")
    if any((out / f"{n}_d500_e0.5.npz").exists() for n in names):
        sys.exit(f"{out} holds a table already; not overwritten")
    z = np.load(TABLE)
    codes = z["codes"].astype(np.float32)
    live = np.abs(codes).sum(1) > 0
    V = torch.as_tensor(codes, device=args.device)
    V = torch.where(torch.as_tensor(live, device=args.device)[:, None], torch.nn.functional.normalize(V, dim=1), V)
    pz = np.load(args.pairs)
    sets = {k: torch.as_tensor(pz[k], device=args.device) for k in ("syn", "ant", "syn_random", "ant_random")}
    steps = 300
    if args.smoke:
        sets = {k: v[:2000] for k, v in sets.items()}
        steps = 20
    info = {"device": args.device, "smoke": args.smoke, "steps": steps,
            "pairs": {k: int(v.shape[0]) for k, v in sets.items()}}
    tabs = {}
    tabs["RF"] = retrofit(V, sets["syn"])
    tabs["CF"], info["CF"] = counterfit(V, sets["syn"], sets["ant"], steps=steps)
    tabs["CFrand"], info["CFrand"] = counterfit(V, sets["syn_random"], sets["ant_random"], steps=steps)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        T = tabs[n]
        T = torch.where(torch.as_tensor(live, device=args.device)[:, None], torch.nn.functional.normalize(T, dim=1), T)
        changed = (T - V).norm(dim=1)
        info.setdefault(n, {}).update({
            "mean_cos_synonyms_before_after": [round(float(cosines(V, sets["syn"]).mean()), 4), round(float(cosines(T, sets["syn"]).mean()), 4)],
            "mean_cos_antonyms_before_after": [round(float(cosines(V, sets["ant"]).mean()), 4), round(float(cosines(T, sets["ant"]).mean()), 4)],
            "rows_changed": int((changed > 1e-6).sum()), "mean_change_of_changed_rows": round(float(changed[changed > 1e-6].mean()), 4)})
        np.savez_compressed(out / f"{n}_d500_e0.5.npz", vocab_ids=z["vocab_ids"], codes=T.cpu().numpy().astype(np.float16),
                            counts=z["counts"], words=z["words"])
        info[n]["table_sha256"] = sha16(out / f"{n}_d500_e0.5.npz")
    info["sha256"] = {"specialise.py": sha16(Path(__file__).resolve()), "ppw_S1000_d500_e0.5.npz": sha16(TABLE),
                      "oewn_pairs.npz": sha16(args.pairs)}
    info["seconds"] = round(time.time() - t0, 1)
    (out / "specialise.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
