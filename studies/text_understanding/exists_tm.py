#!/usr/bin/env python
"""Exploratory run (VAL only): existential concept bits as Tsetlin Machine input.

    exists_tm.py --dataset clinc150 --view mean|exists|mean+exists|mean+max+min --seed 1 --emb-dir DIR

A sentence is read as the set of its tokens. Every token has a fixed vector in the static table of
potion-base-32M (the table whose average gave the rows of 2026-10-16). Dense parts of a view:
- mean:   the normalized average of the token vectors (= the static_tm.py input), thermometer bits.
- max, min: elementwise max / min over the token vectors (max pooling on the table's axes), thermometer bits.
- exists: K concepts = spherical k-means centroids (seed 0) of the distinct tokens of the training texts.
          sim[k] = max over the sentence's tokens of cos(token, concept k). Bits "sim[k] >= t", with t at the
          0/10/25/50/75/90/100th percentiles of the cosines of the concept's own member tokens (the Proto
          rule of the study: every concept on its own scale). A bit then says "some word is at least as
          close to concept k as a typical member", which is the OR over tokens of a per-token bit: at run
          time a table lookup and a bitwise OR, no arithmetic.
Thermometer bits: per-column quantiles fit on train, 8 levels (7 thresholds), as thermo:8.
The TM and logistic regression are imported read-only from the decision-layer study with the settings of
static_tm.py. Test splits are never embedded or read. Appends one JSON line to results_exists.jsonl and the
top rules of a few classes to rules_exists.txt.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(DL / "src"))

from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402
from dl.explain import decision_receipt, top_rules  # noqa: E402
from dl.linear import auroc, fit_logreg, logits  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "oos_train", "oos_val")  # never test, never oos_test
HF_ID, REV = "minishlab/potion-base-32M", "1e5a03f8eeb2c98b928fbbd846f22f816360919f"
LEVELS = (0, 10, 25, 50, 75, 90, 100)
# the classes of the losses listed in errors.txt (2026-10-16), for readable rules
SHOW = {"clinc150": ["translate", "ingredient_substitution", "distance", "transfer"],
        "banking77": ["transfer_into_account", "card_arrival"], "hwu64": ["qa_definition", "weather_query"]}
EXAMPLES = ['how can i say "cancel my order" in french', "are butter and oil interchangable",
            "travel time to bend, oregon", "whats the travel time to redmond right now",
            "take one hundred and fifty bucks from my wells fargo checking account and put it in my wells fargo savings account",
            "what month does my card stop working", "is chocolate good for you"]


def pooled(ids_list, E, En, Cn, dev, bs=1024):
    n, d, K = len(ids_list), E.shape[1], Cn.shape[0]
    out = {k: np.zeros((n, w), np.float32) for k, w in (("mean", d), ("max", d), ("min", d), ("exists", K))}
    for s in range(0, n, bs):
        chunk = ids_list[s : s + bs]
        L = max(1, max(len(x) for x in chunk))
        pad = torch.zeros((len(chunk), L), dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.bool)
        for i, x in enumerate(chunk):
            if x:
                pad[i, : len(x)] = torch.as_tensor(x)
                mask[i, : len(x)] = True
        pad, mask = pad.to(dev), mask.to(dev)
        m3 = mask[..., None]
        V = E[pad]
        mean = torch.nn.functional.normalize((V * m3).sum(1) / mask.sum(1, keepdim=True).clamp(min=1), dim=1)
        mx = V.masked_fill(~m3, -1e9).max(1).values
        mn = V.masked_fill(~m3, 1e9).min(1).values
        ex = (En[pad] @ Cn.T).masked_fill(~m3, -2.0).max(1).values
        empty = ~mask.any(1)
        for t in (mx, mn, ex):
            t[empty] = 0.0
        sl = slice(s, s + len(chunk))
        for k, t in (("mean", mean), ("max", mx), ("min", mn), ("exists", ex)):
            out[k][sl] = t.cpu().numpy()
    return out


def features(dataset: str, K: int, emb_dir: Path, dev: str):
    path, meta_path = emb_dir / f"{dataset}__p32tok_K{K}.npz", emb_dir / f"{dataset}__p32tok_K{K}.meta.json"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}, json.loads(meta_path.read_text())
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(HF_ID, device="cpu", revision=REV)
    tok = st[0].tokenizer
    E = st[0].embedding.weight.detach().float().to(dev)
    En = torch.nn.functional.normalize(E, dim=1)
    ds = load(dataset)
    ids = {sp: [e.ids for e in tok.encode_batch(ds.texts[sp], add_special_tokens=False)] for sp in SPLITS if sp in ds.texts}
    vocab = sorted({t for x in ids["train"] for t in x})
    Z = En[torch.as_tensor(vocab, device=dev)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(Z, K, 25, gen)
    sims = Z @ Cn.T
    a = sims.argmax(1)
    own = sims.gather(1, a[:, None])[:, 0].cpu().numpy()
    a = a.cpu().numpy()
    th = [np.unique(np.percentile(own[a == k], LEVELS)).astype(np.float32).tolist() if (a == k).any() else [2.0]
          for k in range(K)]
    near = sims.T.topk(min(4, len(vocab)), dim=1).indices.cpu().numpy()
    names = ["|".join(tok.id_to_token(vocab[j]) for j in row) for row in near]
    out = {}
    for sp, x in ids.items():
        for k, v in pooled(x, E, En, Cn, dev).items():
            out[f"{sp}__{k}"] = v
    out["y_train"], out["y_val"] = ds.labels["train"], ds.labels["val"]
    ref = np.load(emb_dir / f"{dataset}__potion-32m.npz")
    meta = {"concepts": names, "thresholds": th, "train_vocab": len(vocab), "K": K,
            "members": np.bincount(a, minlength=K).tolist(),
            "mean_vs_static_tm_input_maxabs": float(np.abs(out["train__mean"] - ref["train"]).max())}
    np.savez(path, **out)
    meta_path.write_text(json.dumps(meta))
    return out, meta


class ExistsBits:
    def __init__(self, th, concept_names):
        self.th = [np.asarray(t, np.float32) for t in th]
        self.names = [f"some word ~ {{{concept_names[k]}}} >= {t:.3f}" for k, ts in enumerate(self.th) for t in ts]

    def transform(self, X):
        return np.concatenate([(X[:, k : k + 1] >= t[None, :]) for k, t in enumerate(self.th)], 1).astype(np.uint8)


class View:
    """mean / max / min parts share one thermometer; the exists part uses ExistsBits."""

    def __init__(self, spec: str, meta: dict):
        self.parts = spec.split("+")
        self.dense = [p for p in self.parts if p != "exists"]
        self.ex = ExistsBits(meta["thresholds"], meta["concepts"]) if "exists" in self.parts else None

    def fit(self, F):
        self.names = []
        if self.dense:
            self.th = Thermo(8).fit(np.concatenate([F[f"train__{p}"] for p in self.dense], 1))
            cols = [f"{p}[{d}]" for p in self.dense for d in range(F[f"train__{p}"].shape[1])]
            self.names += [f"{cols[d]} >= {t:.3f}" for d, ts in enumerate(self.th.th) for t in ts]
        if self.ex is not None:
            self.names += self.ex.names
        return self

    def transform(self, F, split):
        out = []
        if self.dense:
            out.append(self.th.transform(np.concatenate([F[f"{split}__{p}"] for p in self.dense], 1)))
        if self.ex is not None:
            out.append(self.ex.transform(F[f"{split}__exists"]))
        return np.concatenate(out, 1)


def best_logreg(Xtr, ytr, Xva, yva, C, grid):
    best = (-1.0, None)
    for Cc in grid:
        W, b = fit_logreg(Xtr.astype(np.float32), ytr, C, Cc)
        acc = float((logits(Xva.astype(np.float32), W, b).argmax(1).cpu().numpy() == yva).mean())
        if acc > best[0]:
            best = (acc, Cc)
    return round(best[0], 4), best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    F, meta = features(args.dataset, args.K, args.emb_dir, "cuda")
    ds = load(args.dataset)
    C = len(ds.intents)
    ytr, yva = F["y_train"], F["y_val"]
    view = View(args.view, meta).fit(F)
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
    cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=args.epochs, batch_size=128,
                   neg_sampling="hardest", negation=True, max_literals=0, budget_mode="exact",
                   budget_select="threshold", max_negated=0, clause_drop=0.0, half=True,
                   seed=args.seed, eval_every=2)
    tm = make_tm(C, Btr.shape[1], cfg, "half")
    tm.fit(Btr, ytr, Bva, yva)
    pv = tm.predict(torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")).cpu().numpy()
    hs = json.loads((HERE / "hardset.json").read_text())[args.dataset]
    rec = {
        "tag": args.tag, "dataset": args.dataset, "view": args.view, "K": args.K, "seed": args.seed,
        "n_bits": int(Btr.shape[1]), "cfg": asdict(cfg),
        "val_acc": round(tm.best_val, 4), "val_acc_check": round(float((pv == yva).mean()), 4),
        "best_epoch": getattr(tm, "best_epoch", None), "train_s": round(tm.train_seconds, 1),
        "lost_fixed": round(float((pv[hs["lost"]] == yva[hs["lost"]]).mean()), 4), "n_lost": len(hs["lost"]),
        "won_kept": round(float((pv[hs["won"]] == yva[hs["won"]]).mean()), 4), "n_won": len(hs["won"]),
        "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats(),
        "mean_check_maxabs": meta["mean_vs_static_tm_input_maxabs"],
    }
    rec["bits_logreg_val"], rec["bits_logreg_C"] = best_logreg(Btr, ytr, Bva, yva, C, [1.0, 10.0, 100.0])
    Dtr = np.concatenate([F[f"train__{p}"] for p in view.parts], 1)
    Dva = np.concatenate([F[f"val__{p}"] for p in view.parts], 1)
    rec["raw_logreg_val"], rec["raw_logreg_C"] = best_logreg(Dtr, ytr, Dva, yva, C, [1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    if "oos_val__mean" in F:
        Xin = torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")
        Xoo = torch.as_tensor(view.transform(F, "oos_val"), dtype=torch.uint8, device="cuda")
        S = torch.cat([tm.class_sums(Xin), tm.class_sums(Xoo)]).cpu().numpy()
        is_oos = np.r_[np.zeros(len(Bva)), np.ones(len(F["oos_val__mean"]))]
        rec["oos_auroc_val_max_sum"] = round(auroc(is_oos, -np.sort(S, 1)[:, -1]), 4)
    vt = ds.texts["val"]
    rec["examples"] = {t: {"true": ds.intents[yva[vt.index(t)]], "tm": ds.intents[pv[vt.index(t)]]} for t in EXAMPLES if t in vt}
    rec["total_s"] = round(time.time() - t0, 1)
    with open(HERE / "results_exists.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    st = tm.state()
    classes = [ds.intents.index(c) for c in SHOW.get(args.dataset, []) if c in ds.intents]
    lines = top_rules(st["ta"], st["w"], view.names, ds.intents, n_states=cfg.n_states, per_class=2, classes=classes)
    for t in EXAMPLES[:1]:
        if t in vt:
            lines.append(f"receipt for {t!r}:\n" + decision_receipt(st["ta"], st["w"], Bva[vt.index(t)], view.names,
                                                                     ds.intents, n_states=cfg.n_states, top=2))
    with open(HERE / "rules_exists.txt", "a") as f:
        f.write(f"=== {args.tag} {args.dataset} {args.view} seed {args.seed} val {rec['val_acc']}\n" + "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve", "stats")}))


if __name__ == "__main__":
    main()
