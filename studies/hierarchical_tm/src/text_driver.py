"""E5a driver: one news text -> its topic (AG News, 4 classes) with Granmo's MultiConcept hierarchy.
(Design decision 'Real words (E5)', news topics (E5a); gates.md Check 2c.)

Model: MultiClassTsetlinMachine of PyHierarchicalTsetlinMachineCUDA 5b4a223 (.venv-htm): 4 class machines, positive
literals only (append_negated=False), 8 state bits, boost 0, fixed +1/-1 clause weights (VANILLA_TM, even clauses +1),
default (counting) inference. Input: the 1,024 word bits of one text (data/agnews, build_agnews.py).
  --model hier (his MultiConceptDemo):    ((AND_GROUP,F), (OR_ALTERNATIVES,A), (AND_ALTERNATIVES,K))
  --model flat (his demo's --vanilla arm): ((AND_GROUP,F), (AND_GROUP,1)): one option per clause (A = K = 1)
Leaf j = a + A*k: option a of OR-group k. An option is the AND of its included words. An EMPTY option is true, so an
OR-group with an empty option is always true: printed ALWAYS-TRUE and never counted as a word group.
Changes from his demo (imported/MultiConceptDemo_79726a2.py): F 32 -> 1,024 words; 2 -> 4 classes; clauses 2 -> 50
per class and T 128 -> 3,200 (his T per clause, 64, kept; the 'Granmo regime' of the Wikipedia runs); s screened,
his 21.1 first. Flat: his --vanilla rule (clauses x A x K clauses per class, same T and s). Training: batches of
--batch texts, each with all 4 topics, one fit() per batch, global numpy RNG seeded before every pass.
Printed: CONFIG, 'EPOCH {json}' per evaluated pass ('PASS {json}' otherwise), 'FINAL {json}', the rules per topic.
Written to $TMLM_RUN_DIR (or --out): metrics.json, rules.txt, groups.json, include_bits.npz, dataset_hashes.json,
model_final.pkl (--save-model: the package's save()), a copy of this file. The test split is never loaded here.
"""
import argparse, hashlib, json, os, pickle, shutil, sys, time
import numpy as np
import scipy.sparse as sp

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import htm_driver as hd  # noqa: E402  (class_sums, include_bits, auc, option_text; no GPU import at load time)

DATA = os.path.join(os.path.dirname(SRC), "data", "agnews")
NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def get_args(argv=None):
    ap = argparse.ArgumentParser(description="E5a driver (see the module docstring)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--model", choices=["hier", "flat"], default="hier")
    ap.add_argument("--clauses", type=int, default=50, help="clauses per class machine")
    ap.add_argument("--T", type=float, default=3200)
    ap.add_argument("--s", type=float, default=21.1)
    ap.add_argument("--A", type=int, default=64, help="options per OR-group (hier only)")
    ap.add_argument("--K", type=int, default=2, help="OR-groups per clause (hier only)")
    ap.add_argument("--passes", type=int, default=50)
    ap.add_argument("--batch", type=int, default=5000, help="max texts per fit() call (one fit = one pass over the batch)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=0, help="fixed random subset of train texts (0 = all)")
    ap.add_argument("--n-train-eval", type=int, default=2000, help="train texts used for the train accuracy")
    ap.add_argument("--n-diag", type=int, default=5000, help="dev texts used for clause fire rates and the rule ranking")
    ap.add_argument("--top", type=int, default=10, help="positive clauses printed per topic (the 'top' set of Check 2c)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after the pass that crosses this wall time (0 = off)")
    ap.add_argument("--eval-every", type=int, default=1, help="evaluate dev every N passes; the last 5 passes, the early-stop pass and the final pass are always evaluated")
    ap.add_argument("--early-stop-margin", type=float, default=0, help="screening runs only: stop if dev accuracy < majority share + this at --early-stop-frac of the passes (0 = off)")
    ap.add_argument("--early-stop-frac", type=float, default=0.3)
    ap.add_argument("--save-model", action="store_true", help="pickle the final model (package save()) as model_final.pkl")
    ap.add_argument("--out", default=".", help="output dir when not run under run_logged.py")
    ap.add_argument("--class-names", default=",".join(NAMES), help="comma-separated class names for printing (default: the AG News topics)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- data
def load_split(data, split, allow_test=False):
    """(X csr uint8 [n, F], y int64, row) of one split written by build_agnews.py. The final-test split is refused
    unless allow_test (the one final test, Day 5)."""
    if split == "test" and not allow_test:
        raise SystemExit("the final-test split is used once (gates.md); this driver never loads it")
    z = np.load(os.path.join(data, f"{split}.npz"))
    F = len(json.load(open(os.path.join(data, "vocab.json"))))
    X = sp.csr_matrix((np.ones(len(z["indices"]), np.uint8), z["indices"], z["indptr"]), shape=(len(z["indptr"]) - 1, F))
    return X, z["y"].astype(np.int64), z["row"]


def batches(y, size, rng_perm):
    """Split a permutation into ceil(n/size) batches; every batch must contain all labels (the package sets its number
    of classes from max(y) of each fit() call)."""
    out = np.array_split(rng_perm, -(-len(y) // size))
    labels = np.unique(y)
    for b in out:
        assert len(np.unique(y[b])) == len(labels), "every batch must contain all labels"
    return out


# ---------------------------------------------------------------- model
def build(a):
    import PyHierarchicalTsetlinMachineCUDA.tm as htm
    if a.model == "hier":
        hs = ((htm.AND_GROUP, a.F), (htm.OR_ALTERNATIVES, a.A), (htm.AND_ALTERNATIVES, a.K))
    else:
        hs = ((htm.AND_GROUP, a.F), (htm.AND_GROUP, 1))
    tm = htm.MultiClassTsetlinMachine(a.clauses, a.T, a.s, hierarchy_structure=hs, append_negated=False,
                                      number_of_state_bits=8, boost_true_positive_feedback=0, seed=a.seed)
    return tm, hs


# ---------------------------------------------------------------- metrics (no GPU)
def metrics(cs, y):
    """cs: class sums [C, n] (clipped, as the package scores). Prediction = argmax (ties -> lowest class, as predict())."""
    C = cs.shape[0]
    pred = np.argmax(cs, axis=0)
    share = np.bincount(pred, minlength=C) / len(y)
    rec = np.array([(pred[y == c] == c).mean() for c in range(C)])
    prec = np.array([(y[pred == c] == c).mean() if (pred == c).any() else 0.0 for c in range(C)])
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    aucs = [hd.auc(cs[c], (y == c).astype(np.int64)) for c in range(C)]
    return dict(acc=float((pred == y).mean()), macro_f1=float(f1.mean()), per_class_acc=[round(float(v), 4) for v in rec],
                pred_share=[round(float(v), 4) for v in share], tie_share=float(((cs == cs.max(0)).sum(0) > 1).mean()),
                macro_auc=float(np.mean(aucs)), majority_acc=float(np.bincount(y, minlength=C).max() / len(y)),
                flag_constant=bool(share.max() > 0.95), flag_unused_class=bool(share.min() < 0.05))


def or_sums(inc, X, A, K):
    """int [n, clauses, K]: true options per OR-group. An option is true iff all its included words are present (an
    empty option is always true). inc: bool [clauses, A*K, F]; X: csr [n, F]."""
    n, C, F = X.shape[0], inc.shape[0], inc.shape[2]
    out = np.zeros((n, C, K), np.int32)
    Xs, step = sp.csr_matrix(X, dtype=np.float32), max(1, 16_000_000 // max(1, n * A * K))
    for c0 in range(0, C, step):
        leaves = inc[c0:c0 + step].reshape(-1, F)
        hits = (Xs @ sp.csc_matrix(leaves.T, dtype=np.float32)).toarray()
        out[:, c0:c0 + step] = (hits == leaves.sum(1)).reshape(n, -1, K, A).sum(-1)
    return out


def group_info(inc_clause, A, K, vocab):
    """The K OR-groups of one clause: non-empty options (word lists), empty count, always-true flag, distinct words."""
    out = []
    for k in range(K):
        opts = inc_clause[k * A:(k + 1) * A]
        full = [tuple(vocab[i] for i in np.flatnonzero(o)) for o in opts if o.any()]
        words = sorted({w for o in full for w in o}, key=vocab.index)
        out.append({"options": [list(o) for o in full], "n_empty": int(A - len(full)), "always_true": len(full) < A,
                    "words": words})
    return out


def group_text(g, A):
    """Slide-8 style: distinct options joined by OR (most repeated first), multi-word options in brackets."""
    cnt = {}
    for o in g["options"]:
        cnt[tuple(o)] = cnt.get(tuple(o), 0) + 1
    body = " OR ".join(o[0] if len(o) == 1 else "(" + " & ".join(o) + ")" for o in sorted(cnt, key=lambda o: -cnt[o]))
    if g["n_empty"] == A:
        return "ALWAYS-TRUE (empty)" if A == 1 else f"ALWAYS-TRUE (all {A} options empty)"
    if g["always_true"]:
        return f"ALWAYS-TRUE ({g['n_empty']} of {A} options empty; others: {body})"
    return body + (f"   [{A} options, {len(cnt)} distinct]" if A > 1 else "")


def rules(incs, fires, y, vocab, A, K, top, names=NAMES):
    """Per topic c: its machine's positive clauses ranked by dev fire-rate difference (texts of topic c minus the
    others). Returns (printed lines, groups.json dict with ALL positive clauses in rank order)."""
    lines, js = [], {"A": A, "K": K, "n_diag": int(len(y)), "classes": []}
    for c, (inc, fire) in enumerate(zip(incs, fires)):
        fin, fout = fire[y == c].mean(0), fire[y != c].mean(0)
        pos = np.arange(0, inc.shape[0], 2)
        order = pos[np.argsort(-(fin - fout)[pos], kind="stable")]
        cl = [{"rank": r + 1, "clause": int(j), "fire_topic": round(float(fin[j]), 4), "fire_other": round(float(fout[j]), 4),
               "groups": group_info(inc[j], A, K, vocab)} for r, j in enumerate(order)]
        js["classes"].append({"name": names[c], "clauses": cl})
        lines.append(f"RULES {names[c]}: top {min(top, len(cl))} positive clauses by dev fire-rate difference ({len(y)} dev texts)")
        for d in cl[:top]:
            lines.append(f"#{d['rank']} clause {d['clause']} (+): fires on {names[c]} {d['fire_topic']:.3f}, "
                         f"other topics {d['fire_other']:.3f}, diff {d['fire_topic'] - d['fire_other']:+.3f}")
            lines += [f"   {names[c]}: {group_text(g, A)}" for g in d["groups"]]
    return lines, js


def group_shares(incs, A, K):
    """Empty-option share and always-true OR-group share over the positive clauses of all class machines."""
    e = np.concatenate([~inc[0::2].any(-1).reshape(-1, K, A) for inc in incs])  # [pos clauses, K, A] empty options
    return float(e.mean()), float(e.any(-1).mean())


# ---------------------------------------------------------------- main
def main():
    a = get_args()
    names = a.class_names.split(",")  # printing only; the model does not use them
    if a.model == "flat":
        a.A = a.K = 1
    run_dir = os.environ.get("TMLM_RUN_DIR", a.out)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("text_driver.py", "htm_driver.py"):
        shutil.copy(os.path.join(SRC, f), run_dir)
    t0 = time.time()
    vocab = json.load(open(os.path.join(a.data, "vocab.json")))
    a.F = len(vocab)
    Xtr, ytr, _ = load_split(a.data, "train")
    Xdv, ydv, _ = load_split(a.data, "dev")
    json.dump({f"{s}.npz": hashlib.sha256(open(os.path.join(a.data, f"{s}.npz"), "rb").read()).hexdigest()
               for s in ("train", "dev")}, open(os.path.join(run_dir, "dataset_hashes.json"), "w"))
    if a.n_train:
        keep = np.sort(np.random.default_rng(0).choice(len(ytr), a.n_train, replace=False))
        Xtr, ytr = Xtr[keep], ytr[keep]
    Xtr_d, Xdv_d, ytr_u = Xtr.toarray(), Xdv.toarray(), ytr.astype(np.uint32)
    tm, hs = build(a)
    print("CONFIG", json.dumps({**vars(a), "hierarchy": [[int(t), int(n)] for t, n in hs], "n_train": len(ytr),
                                "n_dev": len(ydv), "train_per_class": np.bincount(ytr).tolist(),
                                "dev_per_class": np.bincount(ydv).tolist(), "class_names": names}), flush=True)
    hist, stopped = [], ""
    check_pass = int(np.ceil(a.early_stop_frac * a.passes)) if a.early_stop_margin else -1
    for p in range(1, a.passes + 1):
        np.random.seed(10_000 * a.seed + p)  # the package balances and shuffles with the global numpy RNG
        t1 = time.time()
        for b in batches(ytr, a.batch, np.random.permutation(len(ytr))):
            tm.fit(Xtr_d[b].astype(np.uint32), ytr_u[b])
        fit_s, t2 = time.time() - t1, time.time()
        out_of_time = bool(a.max_minutes and time.time() - t0 > 60 * a.max_minutes and p < a.passes)
        tm_ = {"pass": p, "fit_s": round(fit_s, 2), "ms_per_text": round(1000 * fit_s / len(ytr), 4)}
        if not (p % a.eval_every == 0 or p > a.passes - 5 or p == check_pass or out_of_time):
            print("PASS", json.dumps({**tm_, "wall_s": round(time.time() - t0, 1)}), flush=True)
            continue
        dv = metrics(hd.class_sums(tm, Xdv_d), ydv)
        tr = metrics(hd.class_sums(tm, Xtr_d[:a.n_train_eval]), ytr[:a.n_train_eval])
        empty, always = group_shares([hd.include_bits(t) for t in tm.tms], a.A, a.K)
        r = {**tm_, "train_acc": tr["acc"], **{f"dev_{k}": v for k, v in dv.items()}, "empty_option_share": empty,
             "always_true_group_share": always, "eval_s": round(time.time() - t2, 1), "wall_s": round(time.time() - t0, 1)}
        hist.append(r)
        print("EPOCH", json.dumps(r), flush=True)
        if p == check_pass and dv["acc"] < dv["majority_acc"] + a.early_stop_margin:
            stopped = f"stopped early by rule at pass {p}: dev accuracy {dv['acc']:.4f} < majority {dv['majority_acc']:.4f} + {a.early_stop_margin}"
            print("STOP:", stopped, flush=True)
            break
        if out_of_time:
            stopped = f"stopped by --max-minutes at pass {p}"
            print("STOP: --max-minutes reached", flush=True)
            break
    incs = [hd.include_bits(t) for t in tm.tms]
    Xdg, ydg = Xdv[:a.n_diag], ydv[:a.n_diag]
    fires = [(or_sums(inc, Xdg, a.A, a.K) > 0).all(axis=2) for inc in incs]
    last5 = [h["dev_acc"] for h in hist if h["pass"] > a.passes - 5]
    nw = np.concatenate([inc[0::2].sum(-1).ravel() for inc in incs])  # words per option, positive clauses
    multi = float((nw > 1).sum() / max(1, (nw > 0).sum()))
    final = {**hist[-1], "stopped": stopped, "last5_mean_dev_acc": float(np.mean(last5)) if len(last5) == 5 and not stopped else None,
             "best_pass_by_dev_acc": hist[int(np.argmax([h["dev_acc"] for h in hist]))]["pass"],
             "rules_per_class": a.clauses, "multiword_option_share": multi, "n_train": len(ytr), "n_dev": len(ydv)}
    print("FINAL", json.dumps(final), flush=True)
    lines, gj = rules(incs, fires, ydg, vocab, a.A, a.K, a.top, names=names)
    print("\n".join(lines), flush=True)
    open(os.path.join(run_dir, "rules.txt"), "w").write("\n".join(lines) + "\n")
    json.dump({**gj, "model": a.model, "s": a.s, "seed": a.seed, "pass": hist[-1]["pass"]}, open(os.path.join(run_dir, "groups.json"), "w"))
    if a.save_model:
        pickle.dump(tm.save(), open(os.path.join(run_dir, "model_final.pkl"), "wb"))
    np.savez_compressed(os.path.join(run_dir, "include_bits.npz"), **{f"class{c}": np.packbits(inc, axis=-1) for c, inc in enumerate(incs)},
                        F=a.F, A=a.A, K=a.K, model=a.model, clauses=a.clauses, T=a.T, s=a.s, vocab=np.array(vocab), class_names=np.array(names))
    m = {k: {"value": v, "label": "measured"} for k, v in final.items()}
    m["history"] = {"value": hist, "label": "measured"}
    json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
