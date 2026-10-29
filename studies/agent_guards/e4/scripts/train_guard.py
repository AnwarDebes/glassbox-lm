"""Train A2's learned part (the project's CoTM, CPU) and its twins (LR, depth-limited tree) on write rows; export them.

Runs in the decision-layer venv (torch CPU, scikit-learn), never on a GPU:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python -I -B train_guard.py --rows rows.json \
      --out-dir models/<tag> --tag <tag> [--oof-rows all_writes.json --oof-sources qwen_A1 --oof-out oof.json] [--stand-in]

- CoTM = glassbox-lm/studies/decision_layer/src/dl/tm.py, imported read-only, device="cpu", half=False. Its fit()
  ends with torch.cuda.synchronize(), which raises on a machine with no GPU visible; this process replaces
  torch.cuda.synchronize by a no-op when CUDA is unavailable (the file itself is not touched).
- Rows: {"names", "rows": [{"x", "label" (0 ALLOW / 1 DENY), "task", "split", "source", ...}]}. Train = split "train";
  test = split "test" (scored once, descriptive; stand-in only); rows without a split are left out unless
  --include-unsplit.
- Revision 2 (review issue 8): the rows must be VETO-CLEAN (no veto clause fires on any of them; the script refuses
  otherwise) and are deduplicated by build_rows_from_results.py. The minimum of --min-class rows of EACH class (10) is
  counted on these rows; below it the learned part is not trained (exit 3; E4.3 "not testable").
- Choices (TM s, LR C, tree depth) by grouped K-fold CV over TRAIN tasks (folds by a seeded task permutation, seed
  20261021, 5 folds), on balanced accuracy; ties go to the earlier grid value; the TM uses seed 1 in the CV. Then a
  refit on all train rows (TM: one model per seed 1, 2, 3).
- Out-of-fold reports at each pick (review issue 8): balanced accuracy, DENY recall and allow_kept (the share of
  gold-matching rows kept) from the same folds; the TM's refit fold predictions must equal the CV's own (asserted).
- --oof-rows (review issue 4, the E4.3 go-bar): every row of the given sources (A1's executed train-task writes) is
  predicted by the model of the fold its task belongs to (TM seed 1 at the pick, LR, tree), also after a SIMULATED
  re-confirmation (bits.reconfirm); written to --oof-out for scripts/gobar.py.
- Exactness: the exported TM (numpy, e4guard.models.TMGuard unguarded) must give the CoTM's class sums on every row;
  LR and tree exports must give sklearn's predictions on every row. The script stops if not.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(DL / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from e4guard import bits as B  # noqa: E402
from e4guard.models import TMGuard  # noqa: E402

torch.set_num_threads(2)
if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None   # dl.tm.CoTM.fit calls it unconditionally; CPU has nothing to sync

from dl.tm import CoTM, TMConfig  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_hashes():
    out = {}
    for p in sorted((E4 / "e4guard").glob("*.py")) + [Path(__file__), DL / "src/dl/tm.py"]:
        out[str(p)] = sha256_bytes(p.read_bytes())
    return out


def balanced_acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = [float((p[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return float(np.mean(r)) if r else float("nan")


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"n": int(len(y)), "n_deny": int((y == 1).sum()), "acc": float((p == y).mean()) if len(y) else None,
            "balanced_acc": balanced_acc(y, p),
            "deny_recall": float((p[y == 1] == 1).mean()) if (y == 1).any() else None,
            "allow_kept": float((p[y == 0] == 0).mean()) if (y == 0).any() else None}


def tm_fit(X, y, s, seed, cfg_base):
    cfg = TMConfig(**{**cfg_base, "s": float(s), "seed": int(seed)})
    tm = CoTM(2, X.shape[1], cfg, device="cpu")
    tm.fit(X, y)
    return tm


def tm_export(tm) -> dict:
    ta = tm.ta.cpu().numpy()
    w = tm.w.cpu().numpy()
    incl = ta > tm.cfg.n_states
    clauses, W = [], []
    for j in range(incl.shape[0]):
        lits = np.flatnonzero(incl[j]).tolist()      # literal index over [x, 1 - x]: same layout as dl.tm._literals
        if lits:                                     # an empty clause is silent at inference (CoTM.clause_outputs)
            clauses.append(lits)
            W.append([int(w[j, 0]), int(w[j, 1])])
    return {"kind": "cotm", "n_features": int(tm.f), "classes": ["ALLOW", "DENY"], "clauses": clauses, "W": W}


def tm_predict_torch(tm, X):
    S = tm.class_sums(torch.as_tensor(X, dtype=torch.uint8)).cpu().numpy()
    return S, S.argmax(1)


def folds_by_task(tasks, k, seed):
    uniq = sorted(set(tasks))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[i]: int(j % k) for j, i in enumerate(perm)}
    return np.array([fold_of[t] for t in tasks]), fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stand-in", action="store_true", help="mark every exported model as stand-in (code test only)")
    ap.add_argument("--include-unsplit", action="store_true")
    ap.add_argument("--sources", default=None, help="comma list of row sources to keep (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=20261021)
    ap.add_argument("--s-grid", default="2,3,5,8,12,16,24")
    ap.add_argument("--lr-C", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--tree-depth", default="2,3,4,6,8")
    ap.add_argument("--tm-clauses", type=int, default=200)
    ap.add_argument("--tm-T", type=int, default=100)
    ap.add_argument("--tm-epochs", type=int, default=60)
    ap.add_argument("--tm-batch", type=int, default=16)
    ap.add_argument("--min-class", type=int, default=10, help="refuse to train with fewer train rows of a class")
    ap.add_argument("--oof-rows", default=None, help="rows to predict out of fold (e.g. all executed A1 writes)")
    ap.add_argument("--oof-sources", default="qwen_A1")
    ap.add_argument("--oof-out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    rows, rows_sha = [], {}
    for p in args.rows:
        b = Path(p).read_bytes()
        rows_sha[p] = sha256_bytes(b)
        d = json.loads(b)
        assert d["names"] == B.NAMES, f"bit names differ in {p}"
        rows += d["rows"]
    if args.sources:
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    X = np.array([r["x"] for r in rows], np.uint8)
    y = np.array([r["label"] for r in rows], np.int64)
    vetoed = [i for i in range(len(X)) if B.veto_fires(X[i])]
    if vetoed:
        sys.stderr.write(f"REFUSING: {len(vetoed)} rows fire a veto clause; the learned part is trained only on "
                         f"veto-clean rows (build_rows_from_results.py)\n")
        sys.exit(4)
    split = np.array([r.get("split") or "" for r in rows])
    task = np.array([str(r.get("task")) for r in rows])
    tr = (split == "train") | ((split == "") & args.include_unsplit)
    te = split == "test"
    Xtr, ytr, gtr = X[tr], y[tr], task[tr]
    print(f"rows {len(rows)} (veto-clean): train {int(tr.sum())} (DENY {int(ytr.sum())}), test {int(te.sum())} "
          f"(DENY {int(y[te].sum())}); train tasks {len(set(gtr))}", flush=True)
    if min(int((ytr == 0).sum()), int((ytr == 1).sum())) < args.min_class:
        sys.stderr.write(f"REFUSING: veto-clean train rows ALLOW {int((ytr == 0).sum())}, DENY {int((ytr == 1).sum())}; "
                         f"fewer than {args.min_class} of a class, the learned part is not trained (E4.3 not testable)\n")
        sys.exit(3)
    fold, fold_of = folds_by_task(gtr, args.cv_folds, args.cv_seed)
    cfg_base = {"n_clauses": args.tm_clauses, "T": args.tm_T, "epochs": args.tm_epochs, "batch_size": args.tm_batch,
                "half": False, "negation": True}

    def fold_ok(k):
        a, b = fold != k, fold == k
        return b.sum() > 0 and len(set(ytr[a])) == 2

    def cv(fit_pred, keep_pred=None):
        sc = []
        pred = np.full(len(ytr), -1, np.int64)
        for k in range(args.cv_folds):
            if not fold_ok(k):
                continue
            a, b = fold != k, fold == k
            pb = np.asarray(fit_pred(Xtr[a], ytr[a], Xtr[b]))
            pred[b] = pb
            sc.append(balanced_acc(ytr[b], pb))
        if keep_pred is not None:
            keep_pred.append(pred)
        return float(np.mean(sc)) if sc else float("nan")

    tm_cv_pred = {}
    cv_tm = {}
    for s in map(float, args.s_grid.split(",")):
        keep = []
        cv_tm[s] = cv(lambda A, ya, Bx, s=s: tm_predict_torch(tm_fit(A, ya, s, 1, cfg_base), Bx)[1], keep)
        tm_cv_pred[s] = keep[0]
    cv_lr = {C: cv(lambda A, ya, Bx, C=C: LogisticRegression(C=C, max_iter=5000).fit(A, ya).predict(Bx))
             for C in map(float, args.lr_C.split(","))}
    cv_tree = {dd: cv(lambda A, ya, Bx, dd=dd: DecisionTreeClassifier(max_depth=dd, random_state=0).fit(A, ya).predict(Bx))
               for dd in map(int, args.tree_depth.split(","))}
    best = lambda d: max(d, key=lambda k: (np.nan_to_num(d[k], nan=-1.0), -list(d).index(k)))  # noqa: E731
    s_best, C_best, d_best = best(cv_tm), best(cv_lr), best(cv_tree)
    print("cv TM", cv_tm, "-> s", s_best, "| LR", cv_lr, "-> C", C_best, "| tree", cv_tree, "-> depth", d_best, flush=True)

    # ---------------------------------------------------------------- per-fold models at the picks (out of fold)
    fold_models = {}
    oof_pred = {m: np.full(len(ytr), -1, np.int64) for m in ("tm_seed1", "lr", "tree")}
    for k in range(args.cv_folds):
        if not fold_ok(k):
            continue
        a, b = fold != k, fold == k
        tm = tm_fit(Xtr[a], ytr[a], s_best, 1, cfg_base)
        lr = LogisticRegression(C=C_best, max_iter=5000).fit(Xtr[a], ytr[a])
        tree = DecisionTreeClassifier(max_depth=d_best, random_state=0).fit(Xtr[a], ytr[a])
        fold_models[k] = {"tm_seed1": lambda Z, tm=tm: tm_predict_torch(tm, Z)[1],
                          "lr": lr.predict, "tree": tree.predict}
        for m, fn in fold_models[k].items():
            oof_pred[m][b] = fn(Xtr[b])
    assert (oof_pred["tm_seed1"] == tm_cv_pred[s_best]).all(), "the TM's refit fold predictions differ from the CV's"
    have = oof_pred["lr"] >= 0
    oof_report = {m: metrics(ytr[have], p[have]) for m, p in oof_pred.items()}
    print("out-of-fold at the picks:", json.dumps(oof_report), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_base = {"tag": args.tag, "stand_in": bool(args.stand_in), "rows_files_sha256": rows_sha,
                 "n_rows_train": int(tr.sum()), "n_rows_test": int(te.sum()), "train_deny": int(ytr.sum()),
                 "rows_rule": "veto-clean, deduplicated (build_rows_from_results.py)",
                 "bit_names_sha256": sha256_bytes(json.dumps(B.NAMES).encode()), "code_sha256": code_hashes(),
                 "cv": {"folds": args.cv_folds, "seed": args.cv_seed, "tm": cv_tm, "lr": cv_lr, "tree": cv_tree,
                        "picks": {"tm_s": s_best, "lr_C": C_best, "tree_depth": d_best}},
                 "oof_at_picks": oof_report,
                 "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---------------------------------------------------------------- out-of-fold predictions for the go-bar
    if args.oof_rows:
        d = json.loads(Path(args.oof_rows).read_text())
        assert d["names"] == B.NAMES, "bit names differ in the oof rows"
        srcs = set(args.oof_sources.split(","))
        orows = [r for r in d["rows"] if r["source"] in srcs and (r.get("split") == "train")]
        out_rows, missing, unfit = [], 0, 0
        for r in orows:
            k = fold_of.get(str(r["task"]))
            if k is None:
                missing += 1
                continue
            x = np.array(r["x"], np.uint8)[None]
            xr = B.reconfirm(x[0])[None]
            rec = {key: r[key] for key in ("source", "task", "tool", "label", "veto_clean", "strict_r3a", "args")
                   if key in r}
            rec["fold"] = int(k)
            if k not in fold_models:
                # the fold's training part lacks a class, so no model exists for it: its writes count as NOT denied
                # (conservative for GO; registration revision 2, section 10.4), and the count is reported
                unfit += 1
                rec["no_fold_model"] = True
                for m in ("tm_seed1", "lr", "tree"):
                    rec[f"deny_{m}"] = rec[f"deny_{m}_after_reconfirm"] = 0
            else:
                for m, fn in fold_models[k].items():
                    rec[f"deny_{m}"] = int(fn(x)[0])
                    rec[f"deny_{m}_after_reconfirm"] = int(fn(xr)[0])
            out_rows.append(rec)
        assert missing == 0, f"{missing} oof rows have no fold (their task has no training row)"
        oo = {"meta": {"rows_file": args.oof_rows, "rows_file_sha256": sha256_bytes(Path(args.oof_rows).read_bytes()),
                       "sources": sorted(srcs), "picks": {"tm_s": s_best, "lr_C": C_best, "tree_depth": d_best},
                       "folds": args.cv_folds, "cv_seed": args.cv_seed, "tm_seed": 1, "stand_in": bool(args.stand_in),
                       "rows_in_folds_without_a_model": unfit, "folds_with_a_model": sorted(fold_models)},
              "rows": out_rows}
        Path(args.oof_out or (out_dir / "oof_rows.json")).write_text(json.dumps(oo))
        print(f"oof rows predicted: {len(out_rows)}", flush=True)

    report = {"meta": meta_base, "models": {}}
    for seed in args.seeds:
        tm = tm_fit(Xtr, ytr, s_best, seed, cfg_base)
        d = tm_export(tm)
        S_torch, p_torch = tm_predict_torch(tm, X)
        S_np = TMGuard(d, guarded=False).class_sums(X)
        assert (S_np == S_torch.astype(np.int64)).all(), "exported TM differs from the CoTM"
        d["meta"] = {**meta_base, "model": "cotm", "seed": seed, "cfg": {**cfg_base, "s": s_best},
                     "n_clauses_kept": len(d["clauses"]), "exact_rows_checked": int(len(X))}
        p = out_dir / f"tm_seed{seed}.json"
        p.write_text(json.dumps(d))
        report["models"][p.name] = {"train": metrics(ytr, p_torch[tr]),
                                    "test": metrics(y[te], p_torch[te]) if te.any() else None,
                                    "n_clauses_kept": len(d["clauses"])}
    lr = LogisticRegression(C=C_best, max_iter=5000).fit(Xtr, ytr)
    dl_ = {"kind": "lr", "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    p_np = ((X.astype(float) @ np.array(dl_["coef"]) + dl_["intercept"]) > 0).astype(int)
    assert (p_np == lr.predict(X)).all(), "exported LR differs from sklearn"
    dl_["meta"] = {**meta_base, "model": "lr", "C": C_best}
    (out_dir / "lr.json").write_text(json.dumps(dl_))
    report["models"]["lr.json"] = {"train": metrics(ytr, lr.predict(Xtr)),
                                   "test": metrics(y[te], lr.predict(X[te])) if te.any() else None}
    tree = DecisionTreeClassifier(max_depth=d_best, random_state=0).fit(Xtr, ytr)
    tt = tree.tree_
    dt = {"kind": "tree", "feature": tt.feature.tolist(), "threshold": tt.threshold.tolist(),
          "left": tt.children_left.tolist(), "right": tt.children_right.tolist(),
          "value": [list(map(float, v[0])) for v in tt.value]}
    from e4guard.models import TreeGuard

    tg = TreeGuard(dt)
    p_tree = np.array([int(tg.deny(x)[0]) for x in X])
    assert (p_tree == tree.predict(X)).all(), "exported tree differs from sklearn"
    dt["meta"] = {**meta_base, "model": "tree", "max_depth": d_best}
    (out_dir / "tree.json").write_text(json.dumps(dt))
    report["models"]["tree.json"] = {"train": metrics(ytr, tree.predict(Xtr)),
                                     "test": metrics(y[te], tree.predict(X[te])) if te.any() else None}  # none in a registered run
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report["models"].items()}, indent=1), "\nseconds", report["seconds"])


if __name__ == "__main__":
    main()
