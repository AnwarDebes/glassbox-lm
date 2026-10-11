"""E3 direct-label fallback: the deletion probe (pre-registered on 3 Oct 2026, 00:29 CEST, see docs/registrations.md;
post-hoc diagnostic; no training; it cannot change Check 2b).
Texts: each slide-10 EVAL sentence with one bracket's alternative deleted (nothing added): 6 x 101 = 606 texts.
Scoring: class sums rebuilt in numpy from include_bits.npz (counting inference: a clause's value is the product over its
K OR-groups of the number of true options, an empty option always true; class sum = sum of +/- clause values clipped to
+-T; prediction = argmax, ties to class 0, as the package predicts). The rebuild must reproduce the run's logged final
dev accuracy exactly. Accepted = predicted class 1 (slide10).
Prediction (fixed before any seed existed): a template rule rejects >= 95% of deletion texts; a blacklist accepts > 50%.
Usage: python e3fb_deletion_probe.py [--texts-only] RUN_DIR ..."""
import json, os, shutil, sys
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slide10
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "slide10")


def deletion_texts():
    d, meta = np.load(f"{D}/pairs_slide10.npz"), json.load(open(f"{D}/slide10_meta.json"))
    vocab = json.load(open(f"{D}/vocab_slide10.json")); idx = {w: i for i, w in enumerate(vocab)}
    kinds = np.array([f["kind"] for f in meta["families"]])[d["fam"]]
    rows = np.flatnonzero((kinds == "slide10") & (d["split"] == 2))
    brackets = meta["families"][0]["brackets"]
    bow = lambda toks: np.isin(np.arange(len(vocab)), [idx[w] for w in toks])
    X, info, intact = [], [], []
    for r in rows:
        toks = slide10.words(meta["sentences"][r])
        assert (bow(toks) == d["Xp"][r].astype(bool)).all(), f"row {r}: text does not rebuild its bag of words"
        intact.append(bow(toks))
        for k, alts in enumerate(brackets):
            hits = []
            for alt in alts:
                a = alt.split()
                hits += [(i, len(a)) for i in range(len(toks) - len(a) + 1) if toks[i:i + len(a)] == a]
            assert len(hits) == 1, (r, k, hits)
            i, n = hits[0]
            X.append(bow(toks[:i] + toks[i + n:])); info.append((int(r), k))
    return np.array(X), np.array(intact), info, vocab


def class_sums(inc, X, A, K, T):
    """[n] class sum of one class machine: inc bool [clauses, K*A, F] (even clauses +, odd -)."""
    C, L, F = inc.shape
    Xs = sp.csr_matrix(X, dtype=np.float32)
    hits = (Xs @ sp.csc_matrix(inc.reshape(C * L, F).T, dtype=np.float32)).toarray()
    true_opts = (hits == inc.reshape(C * L, F).sum(1)).reshape(len(X), C, K, A).sum(-1)  # [n, C, K]
    val = np.prod(true_opts.astype(np.float64), axis=2)
    w = 1 - 2 * (np.arange(C) % 2)
    return np.clip(val @ w, -T, T)


def sums(run, X):
    z = np.load(os.path.join(run, "include_bits.npz"))
    F, A, K, T = int(z["F"]), int(z["A"]), int(z["K"]), float(z["T"])
    return np.stack([class_sums(np.unpackbits(z[f"class{c}"], axis=-1, count=F).astype(bool), X, A, K, T) for c in (0, 1)])


def predict(run, X):
    return np.argmax(sums(run, X), axis=0)


def dev_check(run):
    log = open(os.path.join(run, "log.txt")).read().splitlines()
    fin = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("FINAL "))
    z = np.load(f"{D}/text/dev.npz"); F = len(json.load(open(f"{D}/text/vocab.json")))
    X = sp.csr_matrix((np.ones(len(z["indices"]), np.uint8), z["indices"], z["indptr"]), shape=(len(z["indptr"]) - 1, F)).toarray()
    import htm_driver as hd
    cs, y = sums(run, X), z["y"].astype(np.int64)
    acc = float((np.argmax(cs, axis=0) == y).mean())
    mauc = float(np.mean([hd.auc(cs[c], (y == c).astype(np.int64)) for c in (0, 1)]))
    return (acc, mauc), (fin["dev_acc"], fin["dev_macro_auc"]), fin["pass"]


if __name__ == "__main__":
    args = sys.argv[1:]
    Xdel, Xint, info, vocab = deletion_texts()
    print(f"deletion texts: {len(Xdel)} from {len(Xint)} eval sentences")
    if args and args[0] == "--texts-only":
        sys.exit(0)
    out_dir = os.environ.pop("TMLM_RUN_DIR", ".")
    shutil.copy(os.path.abspath(__file__), out_dir)
    kb = np.array([k for _, k in info])
    rows = []
    for run in args:
        (acc, mauc), logged, p = dev_check(run)
        assert (acc, mauc) == logged, f"{run}: rebuilt dev accuracy / macro AUC {(acc, mauc)} != logged {logged}"
        acc_del, acc_int = predict(run, Xdel) == 1, predict(run, Xint) == 1
        r = {"run": os.path.basename(run), "final_pass": p, "dev_acc_and_macro_auc_rebuilt_equal_logged": True, "dev_acc": acc, "dev_macro_auc": mauc,
             "accepts_intact_eval": float(acc_int.mean()), "accepts_deletion_all": float(acc_del.mean()),
             "accepts_deletion_by_bracket": [float(acc_del[kb == k].mean()) for k in range(6)]}
        r["reading"] = ("template-like (rejects >= 95%)" if r["accepts_deletion_all"] <= 0.05 else
                        "blacklist-like (accepts > 50%)" if r["accepts_deletion_all"] > 0.5 else "neither (between)")
        rows.append(r); print("PROBE", json.dumps(r))
    print("\n| run | final pass | accepts intact eval | accepts deletion texts | by bracket 1-6 | reading |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run'][17:]} | {r['final_pass']} | {r['accepts_intact_eval']:.1%} | {r['accepts_deletion_all']:.1%} | "
              + " / ".join(f"{v:.0%}" for v in r["accepts_deletion_by_bracket"]) + f" | {r['reading']} |")
    json.dump({"runs": {"value": rows, "label": "measured"}}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
