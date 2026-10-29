"""E2 learned guards, leave one suite out (DESIGN E2, "Training and split"). Decision-layer venv, CPU only:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python -I -B scripts/train_loso.py \
        --rows data/rows_gt_v1.2.2.jsonl [more rows files] --qbits data/query_bits --out models/<tag>

For each held-out suite h (4 folds): the D4 TM, the D5 LR twin and the D5 tree twin are fit on the 3 other suites.
- Rows: labelled rows (ALLOW / DENY) of known tools. Rows on which a W1-W2 premise holds are left out of the learned
  part's training: the veto clauses decide them by construction (DESIGN 1.6, "when it does not fire, the learned TM
  decides"). Their counts are logged.
- Bits: the 75 fact bits + the fold's query bits (query_bits.py: fitted on the training suites only).
- Choices (TM: clauses, s, tau; LR: C, threshold; tree: depth, threshold) on an inner leave-one-suite-out over the 3
  training suites (query bits refitted on each inner training pair). Score, pooled over the 3 inner validation suites
  and over ALL their labelled rows with the veto overlay applied: J = DENY recall - false-block rate on ALLOW rows.
  Ties: higher DENY recall, then the smaller model (TM: fewer clauses; LR: smaller C; tree: smaller depth), then grid
  order (choose_key: the first key of the grid; TM: lower s, then lower tau, since every TM has 200 clauses; LR and
  tree: the lower threshold). Frozen before the held-out suite is touched.
- The chosen configuration is refit on all 3 training suites (seed 1) and exported for the pure-Python decision
  (e2guard.models). The DENY class is oversampled to the ALLOW count in every epoch (fixed, not a choice).
Checks written per fold: pure-Python votes == dl.cpu_tm_fast votes == the CoTM's own votes on up to 500 rows.
Nothing here reads a held-out suite's rows except to write its model's bit vectors for the MILP pre-check.
rev2 (2026-10-21, review decision 6): the TM grid is clauses {200} x s {2, 5, 10} x tau {1, 50, 100}: 9 points, the
same as each twin (800 clauses dropped before any Qwen row exists). --standin-grid is kept as a no-op for the old
commands in logs/. Diagnostic with no rule (review decision 4): --label-rule design_literal relabels the dropped
ground-truth injection rows as DENY (DESIGN's literal "injection-task calls -> DENY") and the agent rows dropped by the
rev2 rule as ALLOW (the pre-rev2 agent rule); with --inner-only it writes the inner tables and choices only.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from itertools import pairwise, product
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

E2 = Path(__file__).resolve().parents[1]
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, str(DL / "src"))

from dl import cpu_tm_fast
from dl.tm import CoTM, TMConfig

from e2guard.bits import FACT_BITS, QueryBits, veto_premises, vetoes_fired
from e2guard.models import GuardedTM, LRTwin, TreeTwin

SUITES = ["banking", "slack", "travel", "workspace"]
DENY, ALLOW = 0, 1
TM_GRID = {"clauses": [200], "s": [2.0, 5.0, 10.0], "tau_frac": [0.0, 0.25, 0.5]}  # tau = 1, 50, 100
LR_GRID = {"C": [0.1, 1.0, 10.0], "threshold": [0.5, 0.7, 0.9]}
TREE_GRID = {"depth": [4, 8, 12], "threshold": [0.5, 0.7, 0.9]}
EPOCHS = 30
BATCH = 64


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def literal_label(r: dict) -> str | None:
    """DESIGN's literal label rule (diagnostic only): dropped ground-truth injection rows -> DENY; agent rows dropped by
    the rev2 rule -> ALLOW (the pre-rev2 agent rule); every other row keeps its label."""
    if r.get("label") is not None:
        return r["label"]
    if r.get("source") == "gt_replay" and r.get("part") == "injection":
        return "DENY"
    if r.get("source") in ("d0_trace", "standin_trace") and "dropped" in str(r.get("label_reason", "")):
        return "ALLOW"
    return None


def load_rows(paths: list[str], label_rule: str = "registered") -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if label_rule == "design_literal":
                r["label"] = literal_label(r)
            if r.get("label") in ("ALLOW", "DENY") and r["facts"]["kind"] is not None:
                r["_on"] = set(r["facts_on"])
                r["_veto"] = bool(vetoes_fired(r["_on"]))
                r["_y"] = ALLOW if r["label"] == "ALLOW" else DENY
                rows.append(r)
    return rows


def matrix(rows: list[dict], qb: QueryBits) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = FACT_BITS + qb.names
    col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)), np.uint8)
    for k, r in enumerate(rows):
        for n in r["_on"] | qb.on(r["query_sha256"]):
            X[k, col[n]] = 1
    y = np.array([r["_y"] for r in rows], np.int64)
    return X, y, names


def balanced_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = np.flatnonzero(y == ALLOW), np.flatnonzero(y == DENY)
    if len(neg) == 0 or len(pos) == 0:
        idx = np.arange(len(y))
    else:
        big, small = (pos, neg) if len(pos) >= len(neg) else (neg, pos)
        idx = np.concatenate([big, rng.choice(small, size=len(big), replace=True)])
    return rng.permutation(idx)


def fit_tm(X: np.ndarray, y: np.ndarray, clauses: int, s: float, seed: int) -> CoTM:
    cfg = TMConfig(n_clauses=clauses, T=clauses, s=s, epochs=EPOCHS, batch_size=BATCH, seed=seed, half=False)
    tm = CoTM(2, X.shape[1], cfg, device="cpu")
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.uint8)
    yt = torch.as_tensor(y, dtype=torch.long)
    for _ in range(EPOCHS):  # CoTM.fit's loop without its CUDA synchronize and without epoch selection
        order = torch.as_tensor(balanced_order(y, rng))
        for st in range(0, len(order), BATCH):
            b = order[st : st + BATCH]
            tm._step(Xt[b], yt[b])
    return tm


def tm_margin(tm: CoTM, X: np.ndarray) -> np.ndarray:
    pk = cpu_tm_fast.pack(tm.ta.numpy(), tm.w.numpy())
    S = cpu_tm_fast.class_sums(pk, X)
    return S[:, ALLOW] - S[:, DENY]


def score(y: np.ndarray, allow: np.ndarray, veto: np.ndarray) -> dict:
    allow = allow & ~veto
    deny_rows, allow_rows = y == DENY, y == ALLOW
    rec = float((~allow[deny_rows]).mean()) if deny_rows.any() else float("nan")
    fbr = float((~allow[allow_rows]).mean()) if allow_rows.any() else float("nan")
    return {"recall_deny": rec, "false_block": fbr, "J": rec - fbr, "n_deny": int(deny_rows.sum()),
            "n_allow": int(allow_rows.sum()), "tp": int((~allow[deny_rows]).sum()), "fp": int((~allow[allow_rows]).sum())}


def pooled(parts: list[dict]) -> dict:
    tp = sum(p["tp"] for p in parts)
    fp = sum(p["fp"] for p in parts)
    nd = sum(p["n_deny"] for p in parts)
    na = sum(p["n_allow"] for p in parts)
    rec = tp / nd if nd else float("nan")
    fbr = fp / na if na else float("nan")
    return {"recall_deny": rec, "false_block": fbr, "J": rec - fbr, "n_deny": nd, "n_allow": na}


def choose_key(pooled_tab: dict) -> str:
    """The inner choice (registration section 6): the highest pooled J; ties: higher DENY recall, then the smaller model
    (the first field of the key: clauses, C or depth), then grid order (min() keeps the first of equal keys, and the
    keys are in grid order: TM clauses x s x tau, LR C x threshold, tree depth x threshold)."""
    return min(pooled_tab, key=lambda k: (-round(pooled_tab[k]["J"], 9), -round(pooled_tab[k]["recall_deny"], 9),
                                          float(k.split("|")[0])))


def export_tm(tm: CoTM, names: list[str], tau: int) -> dict:
    pk = cpu_tm_fast.pack(tm.ta.numpy(), tm.w.numpy())
    F = len(names)
    starts = list(pk["starts"]) + [len(pk["lits"])]
    clauses = [pk["lits"][a:b].tolist() for a, b in pairwise(starts)]
    W = pk["w"].astype(int).tolist()
    kind = ["learned"] * len(clauses)
    vid: list = [None] * len(clauses)
    clauses.append([])  # bias: ALLOW wins iff s_allow - s_deny >= tau
    W.append([0, 1 - int(tau)])
    kind.append("bias")
    vid.append(None)
    Wa = np.array(W)
    M = int(np.clip(Wa[:, ALLOW], 0, None).sum() + np.abs(Wa[:, DENY]).sum() + 1)
    col = {n: i for i, n in enumerate(names)}
    for v in veto_premises():
        clauses.append([col[b] for b in v["bits"]])
        W.append([0, -M])
        kind.append("veto")
        vid.append(v["id"])
    return {"type": "tm", "names": names, "F": F, "clauses": clauses, "W": W, "clause_kind": kind, "veto_id": vid,
            "M": M, "tau": int(tau), "classes": ["DENY", "ALLOW"], "n_learned": int(len(starts) - 1)}


def tree_export(t: DecisionTreeClassifier) -> list[dict]:
    tr = t.tree_
    nodes = []
    for i in range(tr.node_count):
        if tr.children_left[i] == -1:
            v = tr.value[i][0]
            cls = list(t.classes_)
            p = float(v[cls.index(ALLOW)] / v.sum()) if ALLOW in cls else 0.0
            nodes.append({"p": p})
        else:
            # bits are 0/1 and sklearn splits at x <= 0.5: left = bit 0, right = bit 1
            nodes.append({"f": int(tr.feature[i]), "l": int(tr.children_left[i]), "r": int(tr.children_right[i])})
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--qbits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", nargs="*", default=SUITES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="code test only: tiny grids and 2 epochs (never for a result)")
    ap.add_argument("--standin-grid", action="store_true",
                    help="no-op since rev2 (the registered TM grid is now the former stand-in grid, clauses {200})")
    ap.add_argument("--label-rule", choices=["registered", "design_literal"], default="registered",
                    help="design_literal: the CPU-only diagnostic with no rule (never the registered models)")
    ap.add_argument("--inner-only", action="store_true", help="inner tables and choices only; no final fit or export")
    args = ap.parse_args()
    global EPOCHS
    if args.label_rule != "registered":
        assert args.inner_only, "--label-rule design_literal is a diagnostic: use it with --inner-only"
    if args.quick:
        TM_GRID.update({"clauses": [50], "s": [5.0], "tau_frac": [0.0, 0.25]})
        LR_GRID.update({"C": [1.0], "threshold": [0.5]})
        TREE_GRID.update({"depth": [4], "threshold": [0.5]})
        EPOCHS = 2
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    rows = load_rows(args.rows, args.label_rule)
    qdir = Path(args.qbits)
    log = {"rows_files": {p: sha256(Path(p)) for p in args.rows}, "script_sha256": sha256(Path(__file__)),
           "grids": {"tm": TM_GRID, "lr": LR_GRID, "tree": TREE_GRID, "epochs": EPOCHS, "batch": BATCH},
           "seed": args.seed, "tag": args.tag, "label_rule": args.label_rule, "inner_only": args.inner_only,
           "folds": {}}
    by_suite = {s: [r for r in rows if r["suite"] == s] for s in SUITES}
    for held in args.folds:
        train_suites = [s for s in SUITES if s != held]
        fl: dict = {"train_suites": train_suites}
        tr_rows = [r for s in train_suites for r in by_suite[s]]
        fl["train_rows"] = {"labelled": len(tr_rows), "vetoed_left_out": sum(r["_veto"] for r in tr_rows),
                            "ALLOW_used": sum((not r["_veto"]) and r["_y"] == ALLOW for r in tr_rows),
                            "DENY_used": sum((not r["_veto"]) and r["_y"] == DENY for r in tr_rows),
                            "sources": sorted({r["source"] for r in tr_rows})}
        # ---- inner folds: the choices ----
        inner: dict = {"tm": {}, "lr": {}, "tree": {}}
        for val in train_suites:
            fit_s = [s for s in train_suites if s != val]
            qb = QueryBits(qdir / f"fold_{held}__inner_{val}.json")
            fit_rows = [r for s in fit_s for r in by_suite[s] if not r["_veto"]]
            val_rows = by_suite[val]
            Xf, yf, _ = matrix(fit_rows, qb)
            Xv, yv, _ = matrix(val_rows, qb)
            vv = np.array([r["_veto"] for r in val_rows], bool)
            for c, s_ in product(TM_GRID["clauses"], TM_GRID["s"]):
                tm = fit_tm(Xf, yf, c, s_, args.seed)
                mg = tm_margin(tm, Xv)
                for tf in TM_GRID["tau_frac"]:
                    tau = max(1, round(tf * c))
                    inner["tm"].setdefault(f"{c}|{s_}|{tf}", []).append(score(yv, mg >= tau, vv))
            for C in LR_GRID["C"]:
                lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xf, yf)
                p = lr.predict_proba(Xv)[:, list(lr.classes_).index(ALLOW)]
                for th in LR_GRID["threshold"]:
                    inner["lr"].setdefault(f"{C}|{th}", []).append(score(yv, p >= th, vv))
            for d in TREE_GRID["depth"]:
                t = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0).fit(Xf, yf)
                p = t.predict_proba(Xv)[:, list(t.classes_).index(ALLOW)] if ALLOW in t.classes_ else np.zeros(len(yv))
                for th in TREE_GRID["threshold"]:
                    inner["tree"].setdefault(f"{d}|{th}", []).append(score(yv, p >= th, vv))
            print(f"[{held}] inner {val} done ({time.time() - t0:.0f}s)", flush=True)
        choice = {}
        for model, table in inner.items():
            pooled_tab = {k: pooled(v) for k, v in table.items()}
            best = choose_key(pooled_tab)
            choice[model] = {"key": best, **pooled_tab[best]}
            fl[f"inner_{model}"] = pooled_tab
        fl["choice"] = choice
        if args.inner_only:
            fl["seconds"] = round(time.time() - t0, 1)
            log["folds"][held] = fl
            print(f"[{held}] inner-only choice {json.dumps({k: v['key'] for k, v in choice.items()})} "
                  f"J {json.dumps({k: round(v['J'], 4) for k, v in choice.items()})}", flush=True)
            (out / "train_log.json").write_text(json.dumps(log, indent=1, default=str))
            continue
        # ---- final fit on the 3 training suites ----
        qb = QueryBits(qdir / f"fold_{held}.json")
        fit_rows = [r for r in tr_rows if not r["_veto"]]
        Xf, yf, names = matrix(fit_rows, qb)
        c, s_, tf = choice["tm"]["key"].split("|")
        c, s_, tf = int(c), float(s_), float(tf)
        tau = max(1, round(tf * c))
        tm = fit_tm(Xf, yf, c, s_, args.seed)
        mdl = export_tm(tm, names, tau)
        mdl.update({"held_out": held, "train_suites": train_suites, "config": {"clauses": c, "s": s_, "T": c,
                    "tau": tau, "epochs": EPOCHS, "seed": args.seed}, "stand_in": args.tag})
        (out / f"d4_tm_{held}.json").write_text(json.dumps(mdl))
        # equality checks of the exported decision
        g = GuardedTM(mdl)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(Xf), size=min(500, len(Xf)), replace=False)
        pk = cpu_tm_fast.pack(tm.ta.numpy(), tm.w.numpy())
        S_fast = cpu_tm_fast.class_sums(pk, Xf[pick])
        S_torch = tm.class_sums(torch.as_tensor(Xf[pick], dtype=torch.uint8)).numpy().astype(np.int64)
        S_py = np.array([g.scores(Xf[i].tolist(), with_veto=False, with_bias=False)[0] for i in pick])
        fl["decision_checks"] = {"rows": len(pick), "py_equals_cpu_tm_fast": int((S_py == S_fast).all(1).sum()),
                                 "cpu_tm_fast_equals_torch": int((S_fast == S_torch).all(1).sum())}
        fl["tm_stats"] = tm.stats()
        C = float(choice["lr"]["key"].split("|")[0])
        th = float(choice["lr"]["key"].split("|")[1])
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xf, yf)
        sign = 1.0 if list(lr.classes_).index(ALLOW) == 1 else -1.0
        (out / f"d5_lr_{held}.json").write_text(json.dumps({"type": "lr", "names": names,
            "coef": (sign * lr.coef_[0]).tolist(), "intercept": float(sign * lr.intercept_[0]), "threshold": th,
            "held_out": held, "config": {"C": C, "threshold": th}, "stand_in": args.tag}))
        d = int(choice["tree"]["key"].split("|")[0])
        th_t = float(choice["tree"]["key"].split("|")[1])
        t = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0).fit(Xf, yf)
        (out / f"d5_tree_{held}.json").write_text(json.dumps({"type": "tree", "names": names, "nodes": tree_export(t),
            "threshold": th_t, "held_out": held, "config": {"depth": d, "threshold": th_t}, "stand_in": args.tag}))
        # the exported twins against sklearn's own probabilities on the same rows
        lr_py = LRTwin.load(out / f"d5_lr_{held}.json")
        tr_py = TreeTwin.load(out / f"d5_tree_{held}.json")
        p_lr = lr.predict_proba(Xf[pick])[:, list(lr.classes_).index(ALLOW)]
        p_tr = t.predict_proba(Xf[pick])[:, list(t.classes_).index(ALLOW)] if ALLOW in t.classes_ else np.zeros(len(pick))
        fl["decision_checks"]["lr_py_equals_sklearn"] = int(sum(abs(lr_py.p_allow(Xf[i].tolist()) - p_lr[k]) < 1e-9
                                                                 for k, i in enumerate(pick)))
        fl["decision_checks"]["tree_py_equals_sklearn"] = int(sum(abs(tr_py.p_allow(Xf[i].tolist()) - p_tr[k]) < 1e-9
                                                                   for k, i in enumerate(pick)))
        fl["seconds"] = round(time.time() - t0, 1)
        log["folds"][held] = fl
        print(f"[{held}] choice {json.dumps({k: v['key'] for k, v in choice.items()})} checks {fl['decision_checks']} "
              f"({fl['seconds']}s)", flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=1, default=str))
    log["minutes"] = round((time.time() - t0) / 60, 2)
    log["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "train_log.json").write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
