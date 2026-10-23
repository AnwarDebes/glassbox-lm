#!/usr/bin/env python
"""R7 readability report for the single TM of every seed (member 0, `model_seed<k>.npz`).

Registered as "reported with no rule": clause statistics, the number of clauses that fire behind
a decision, example rules and decision receipts. Computed afterwards and labelled post hoc in the
output: (a) the decision recomputed from the printed clauses equals the model's prediction;
(b) sufficiency: the k strongest firing clauses alone give the same decision; (c) accuracy after
pruning long or negation-only clauses.

    r7_readability.py [--datasets clinc150 banking77 hwu64] [--receipts 6]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import fit_logreg  # noqa: E402

VIEW = "proto:k=1:whiten=0.001+lex:2000"
ENC = "bge-large"
DEV = "cuda"
BASE = ROOT / "results" / "confirm" / "R7"


def short_names(view, intents) -> list[str]:
    """Readable short names: near(<intent>) >= p<level>, nearest = <intent>, <intent> in top 3, contains "w"."""
    proto = view.views[0]
    lev = list(proto.levels)
    out = []
    for j, th in enumerate(proto.th):
        c = intents[int(proto.owner[j])]
        if len(th) == len(lev):  # thresholds = the registered quantiles of the class's own training similarities
            out += [f"near({c}) >= p{q:g}" for q in lev]
        else:  # np.unique merged equal thresholds: name them by value
            out += [f"near({c}) >= t{float(t):.3f}" for t in th]
    for r in proto.ranks:
        for c in range(len(intents)):
            out.append(f"nearest = {intents[c]}" if r == 1 else f"{intents[c]} in top {r}")
    out += list(view.views[1].names)
    assert len(out) == len(view.names)
    return out


def merge_levels(lits: list[str]) -> list[str]:
    """near(x) >= pA and NOT near(x) >= pB -> pA <= near(x) < pB."""
    lo, hi, other = {}, {}, []
    for l in lits:
        m = re.fullmatch(r"(NOT )?near\((.+)\) >= p(\d+)", l)  # value-named levels stay as they are
        if not m:
            other.append(l)
            continue
        neg, f, t = m.group(1), m.group(2), int(m.group(3))
        if neg:
            hi[f] = min(hi.get(f, 999), t)
        else:
            lo[f] = max(lo.get(f, -1), t)
    out = []
    for f in sorted(set(lo) | set(hi)):
        a, b = lo.get(f), hi.get(f)
        if a is not None and b is not None:
            out.append(f"p{a} <= near({f}) < p{b}" if a < b else f"[never: near({f})]")
        elif a is not None:
            out.append(f"near({f}) >= p{a}")
        else:
            out.append(f"near({f}) < p{b}")
    pos = [o for o in out + other if not o.startswith("NOT ")]
    neg = [o for o in out + other if o.startswith("NOT ")]
    return pos + neg


def clause_text(incl_row: np.ndarray, names: list[str], max_pos: int = 8, max_neg: int = 6) -> str:
    """Plain literals first, then at most ``max_neg`` negated ones and how many more there are."""
    f = len(names)
    lits = merge_levels([names[k] if k < f else f"NOT {names[k - f]}" for k in np.flatnonzero(incl_row)])
    pos = [l for l in lits if not l.startswith("NOT ")]
    neg = [l[4:] for l in lits if l.startswith("NOT ")]
    parts = pos[:max_pos] + ([f"(+{len(pos) - max_pos} more)"] if len(pos) > max_pos else [])
    if neg:
        shown = ", ".join(neg[:max_neg]) + (f", ... (+{len(neg) - max_neg} more)" if len(neg) > max_neg else "")
        parts.append(f"NONE OF [{shown}]")
    return " AND ".join(parts) if parts else "(empty)"


def fire(incl: torch.Tensor, B: np.ndarray, bs: int = 2048) -> torch.Tensor:
    """Clause outputs at inference (an empty clause does not fire), [n, N] bool."""
    inc = incl.half()
    has = incl.sum(1) > 0
    out = []
    for i in range(0, B.shape[0], bs):
        X = torch.as_tensor(B[i : i + bs], device=DEV).half()
        L = torch.cat([X, 1 - X], 1)
        out.append((((1 - L) @ inc.T) == 0) & has[None, :])
    return torch.cat(out)


K_MAX = 20


def minimal_explanation(contrib: torch.Tensor, Wsel, cstar: torch.Tensor, sizes: torch.Tensor, bias=None) -> dict:
    """Smallest k such that the k items (clauses or features) with the largest positive vote for the
    decided class, alone (with the bias, for LR), give the same decision; and how many literals those
    k items hold. ``Wsel(idx)`` returns the vote vectors [n, k, C] of the items ``idx`` [n, k]."""
    order = contrib.argsort(1, descending=True)[:, :K_MAX]
    okm = torch.gather(contrib, 1, order) > 0
    cum = torch.cumsum(Wsel(order) * okm[:, :, None], dim=1)
    if bias is not None:
        cum = cum + bias[None, None, :]
    match = cum.argmax(2) == cstar[:, None]
    kmin = torch.where(match.any(1), match.float().argmax(1) + 1, torch.full_like(cstar, K_MAX + 1))
    lits = torch.cumsum(sizes[order] * okm, 1).gather(1, (kmin.clamp(max=K_MAX) - 1).unsqueeze(1)).squeeze(1).float()
    found = kmin <= K_MAX
    return {"median_items": float(kmin.float().median()), "share_1_item": float((kmin == 1).float().mean()),
            "share_le3_items": float((kmin <= 3).float().mean()), "share_found_le20": float(found.float().mean()),
            "median_literals": float(lits[found].median()) if found.any() else None,
            "share_le5_literals": float(((lits <= 5) & found).float().mean()),
            "share_le10_literals": float(((lits <= 10) & found).float().mean())}


def lr_explanations(ds_name: str, Btr, ytr, Bte, n_classes: int, base: Path) -> dict:
    """The same minimal-explanation numbers for logistic regression on the same readable bits
    (refitted with the C of its registered run; its predictions are checked against the run's)."""
    folder = base / f"{ds_name}_lrbits"
    lr = json.loads((folder / "lr.json").read_text())
    W, b = fit_logreg(Btr.astype(np.float32), ytr, n_classes, lr["C"])
    X = torch.as_tensor(Bte, dtype=torch.float32, device=DEV)
    pred = (X @ W + b).argmax(1)
    saved = np.load(folder / "lr_preds.npz")["pred"]
    out = {"refit_equals_registered_prediction": float((pred.cpu().numpy() == saved).mean())}
    contrib = X * W[:, pred].T
    out.update(minimal_explanation(contrib, lambda idx: W[idx], pred, torch.ones(W.shape[0], device=DEV), bias=b))
    return out


def analyse(ds_name: str, n_receipts: int, base: Path = BASE) -> dict:
    folder = base / ds_name
    seeds = sorted(int(p.stem[10:]) for p in folder.glob("model_seed*.npz"))
    if not seeds:
        return {}
    d = np.load(ROOT / "data" / "emb" / f"{ds_name}__{ENC}.npz")
    ds = load(ds_name)
    view = make_view(VIEW, seed=0)
    view.fit(d["train"], d["y_train"], texts=ds.texts["train"], class_names=ds.intents)
    names = short_names(view, ds.intents)
    f = len(names)
    B = {sp: view.transform(d[sp], ds.texts[sp]) for sp in ("val", "test")}
    y = {"val": d["y_val"], "test": d["y_test"]}
    res = {"seeds": seeds, "per_seed": {}}
    # post hoc baseline: the one-rule router "route to the intent whose (whitened) centre is nearest",
    # i.e. the view's own "nearest = <intent>" bit
    proto = view.views[0]
    near = {sp: proto.owner[proto.sims(d[sp]).argmax(1)].cpu().numpy() for sp in ("val", "test")}
    res["nearest_centre_rule_post_hoc"] = {sp: float((near[sp] == y[sp]).mean()) for sp in near}
    if "oos_test" in d.files:
        from dl.linear import auroc
        mi = proto.sims(d["test"]).max(1).values.cpu().numpy()
        mo = proto.sims(d["oos_test"]).max(1).values.cpu().numpy()
        res["nearest_centre_rule_post_hoc"]["oos_auroc_test_max_sim"] = auroc(np.r_[np.zeros(len(mi)), np.ones(len(mo))], -np.r_[mi, mo])
    for k in seeds:
        m = np.load(folder / f"model_seed{k}.npz")
        cfg = json.loads(str(m["cfg"]))
        incl_np = m["ta"].astype(np.int32) > cfg["n_states"]
        incl = torch.as_tensor(incl_np, device=DEV)
        w = torch.as_tensor(m["w"].astype(np.float32), device=DEV)
        nlit = incl_np.sum(1)
        npos = incl_np[:, :f].sum(1)
        nonempty = nlit > 0
        r = {"clauses": int(len(nlit)), "nonempty": int(nonempty.sum()),
             "median_literals_nonempty": float(np.median(nlit[nonempty])), "max_literals": int(nlit.max()),
             "share_le_8_literals": float((nlit[nonempty] <= 8).mean()), "share_le_32_literals": float((nlit[nonempty] <= 32).mean()),
             "share_with_plain_literal": float((npos[nonempty] > 0).mean())}
        kinds = {"near": 0, "nearest/top3": 0, "contains": 0}
        for j in np.flatnonzero(incl_np[:, :f].any(0)):
            nm = names[j]
            key = "near" if nm.startswith("near(") else ("contains" if nm.startswith("contains") else "nearest/top3")
            kinds[key] += int(incl_np[:, j].sum())
        r["plain_literals_by_kind"] = kinds
        co = {sp: fire(incl, B[sp]) for sp in B}
        S = {sp: co[sp].float() @ w for sp in B}
        pred = {sp: S[sp].argmax(1).cpu().numpy() for sp in B}
        saved = np.load(folder / f"preds_seed{k}.npz")["pred"]
        r["receipt_equals_model_prediction"] = float((pred["test"] == saved).mean())  # post hoc (a)
        r["test_acc"] = float((pred["test"] == y["test"]).mean())
        dis = pred["test"] != near["test"]
        r["vs_nearest_centre_post_hoc"] = {"agree": float(1 - dis.mean()), "n_disagree": int(dis.sum()),
                                           "tm_right_when_disagree": float((pred["test"][dis] == y["test"][dis]).mean()) if dis.any() else None,
                                           "centre_right_when_disagree": float((near["test"][dis] == y["test"][dis]).mean()) if dis.any() else None}
        nf = co["test"].sum(1).float()
        r["firing_clauses_median"] = float(nf.median())
        cstar = torch.as_tensor(pred["test"], device=DEV)
        contrib = co["test"].float() * w[:, cstar].T  # [n, N] vote of each firing clause for the winner
        r["firing_with_positive_vote_for_winner_median"] = float((contrib > 0).sum(1).float().median())
        # (b) sufficiency: the k strongest firing clauses for the winner, alone
        suff = {}
        order = contrib.argsort(1, descending=True)
        for kk in (1, 3, 5, 10):
            top = order[:, :kk]
            Sk = w[top].sum(1)  # [n, C]
            ok_mask = torch.gather(contrib, 1, top) > 0
            Sk = (w[top] * ok_mask[:, :, None]).sum(1)
            suff[f"top{kk}"] = float((Sk.argmax(1) == cstar).float().mean())
        r["sufficiency_post_hoc"] = suff
        r["min_explanation_post_hoc"] = minimal_explanation(contrib, lambda idx: w[idx], cstar, torch.as_tensor(nlit, device=DEV).float())
        # (c) pruning, post hoc: accuracy with only short clauses kept
        prune = {}
        for L in (4, 8, 16, 32, 64):
            keep = torch.as_tensor((nlit <= L), device=DEV).float()
            prune[f"le{L}"] = {sp: float(((co[sp].float() * keep) @ w).argmax(1).cpu().numpy().__eq__(y[sp]).mean()) for sp in B}
        keep = torch.as_tensor(npos > 0, device=DEV).float()
        prune["plain_only"] = {sp: float(((co[sp].float() * keep) @ w).argmax(1).cpu().numpy().__eq__(y[sp]).mean()) for sp in B}
        prune["all"] = {sp: float((pred[sp] == y[sp]).mean()) for sp in B}
        r["pruning_post_hoc"] = prune
        res["per_seed"][k] = r
        if k == seeds[0] and n_receipts:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(y["test"]), n_receipts, replace=False)
            lines = []
            for i in pick:
                c = int(cstar[i])
                Si = S["test"][i]
                top2 = Si.topk(2)
                ru = int(top2.indices[1])
                ci = contrib[i]
                pos_total = float(ci.clamp(min=0).sum())
                lines.append(f'QUERY: "{ds.texts["test"][i]}"   (true intent: {ds.intents[int(y["test"][i])]})')
                lines.append(f"  DECISION: {ds.intents[c]}  (vote {float(Si[c]):.0f}; runner-up {ds.intents[ru]} {float(top2.values[1]):.0f};"
                             f" {int(co['test'][i].sum())} of {len(nlit)} clauses fire)")
                for j in order[i, :3].tolist():
                    if float(ci[j]) <= 0:
                        break
                    lines.append(f"    +{float(ci[j]):.0f}  IF {clause_text(incl_np[j], names)}")
                top3 = float(ci[order[i, :3]].clamp(min=0).sum())
                lines.append(f"    (these 3 clauses carry {100 * top3 / max(pos_total, 1):.0f}% of the positive vote for {ds.intents[c]})")
                lines.append("")
            res["receipts_seed"] = k
            res["receipts"] = lines
    ps = res["per_seed"].values()
    agg = lambda key: float(np.mean([p[key] for p in ps]))  # noqa: E731
    res["mean"] = {key: agg(key) for key in ("test_acc", "median_literals_nonempty", "share_le_8_literals", "share_le_32_literals",
                                            "share_with_plain_literal", "firing_clauses_median", "firing_with_positive_vote_for_winner_median",
                                            "receipt_equals_model_prediction")}
    res["mean"]["max_literals_max"] = int(max(p["max_literals"] for p in ps))
    res["mean"]["vs_nearest_centre_post_hoc"] = {kk: float(np.mean([p["vs_nearest_centre_post_hoc"][kk] for p in ps if p["vs_nearest_centre_post_hoc"][kk] is not None]))
                                                 for kk in ("agree", "n_disagree", "tm_right_when_disagree", "centre_right_when_disagree")}
    res["mean"]["nearest_centre_rule_post_hoc"] = res["nearest_centre_rule_post_hoc"]
    res["mean"]["sufficiency_post_hoc"] = {kk: float(np.mean([p["sufficiency_post_hoc"][kk] for p in ps])) for kk in ("top1", "top3", "top5", "top10")}
    res["mean"]["pruning_post_hoc"] = {kk: {sp: float(np.mean([p["pruning_post_hoc"][kk][sp] for p in ps])) for sp in ("val", "test")}
                                       for kk in next(iter(ps))["pruning_post_hoc"]}
    res["mean"]["min_explanation_post_hoc"] = {kk: float(np.mean([p["min_explanation_post_hoc"][kk] for p in ps if p["min_explanation_post_hoc"][kk] is not None]))
                                               for kk in next(iter(ps))["min_explanation_post_hoc"]}
    if (base / f"{ds_name}_lrbits" / "lr.json").exists():
        Btr = view.transform(d["train"], ds.texts["train"])
        res["lr_readable_min_explanation_post_hoc"] = lr_explanations(ds_name, Btr, d["y_train"], B["test"], len(ds.intents), base)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["clinc150", "banking77", "hwu64"])
    ap.add_argument("--receipts", type=int, default=6)
    ap.add_argument("--registration", default="R7", help="results/confirm/<registration> (R7 or R7b)")
    ap.add_argument("--lr-base", default="R7", help="registration folder holding the <ds>_lrbits LR runs")
    args = ap.parse_args()
    base = ROOT / "results" / "confirm" / args.registration
    out = {}
    for ds_name in args.datasets:
        r = analyse(ds_name, args.receipts, base)
        if r and args.lr_base != args.registration and (ROOT / "results" / "confirm" / args.lr_base / f"{ds_name}_lrbits").exists() \
                and "lr_readable_min_explanation_post_hoc" not in r:
            d = np.load(ROOT / "data" / "emb" / f"{ds_name}__{ENC}.npz")
            ds = load(ds_name)
            view = make_view(VIEW, seed=0)
            view.fit(d["train"], d["y_train"], texts=ds.texts["train"], class_names=ds.intents)
            r["lr_readable_min_explanation_post_hoc"] = lr_explanations(
                ds_name, view.transform(d["train"], ds.texts["train"]), d["y_train"], view.transform(d["test"], ds.texts["test"]),
                len(ds.intents), ROOT / "results" / "confirm" / args.lr_base)
        if r:
            out[ds_name] = r
            print(ds_name, json.dumps(r["mean"], indent=1))
            print("LR-readable:", json.dumps(r.get("lr_readable_min_explanation_post_hoc"), indent=1))
            print("\n".join(r.get("receipts", [])))
    (base / "readability.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
