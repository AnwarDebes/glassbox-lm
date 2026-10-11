"""Training driver for the hierarchical Tsetlin machine of the supervisor's project proposal (unpublished).

Model: MultiClassTsetlinMachine of PyHierarchicalTsetlinMachineCUDA 5b4a223 (.venv-htm), positive
literals only (append_negated=False), 8 state bits, boost 0. Input X = [paragraph A bits | paragraph B bits].
  --model hier (slide 11): ((AND_GROUP,F), (OR_ALTERNATIVES,A), (AND_ALTERNATIVES,K), (AND_GROUP,2))
  --model flat:            ((AND_GROUP,F), (AND_GROUP,2))
Leaf index j = a + A*(k + K*p): option a of OR-group k on paragraph p (p=0 is A, p=1 is B).
An option (leaf) is the AND of its included words. An EMPTY option is TRUE, in training and at
inference, so an OR-group with an empty option is always true. OR adds its options, AND multiplies.
Class c's machine has fixed clause weights +1 (even clauses) and -1 (odd clauses).
Score = class_sum[1] - class_sum[0]; the prediction is the package's argmax.

Printed: CONFIG, one 'EPOCH {json}' per pass, 'FINAL {json}', the class-1 rules, 'VERIFY {json}'.
Written to $TMLM_RUN_DIR (or --out): metrics.json, rules.txt, include_bits.npz, copies of the code.
include_bits.npz: class1, class0 = np.packbits(include bits [clauses, leaves, F], axis=-1); unpack with
np.unpackbits(x, axis=-1, count=F).astype(bool).
"""
import argparse, hashlib, json, os, pickle, shutil, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

SRC = os.path.dirname(os.path.abspath(__file__))


def get_args():
    ap = argparse.ArgumentParser(description="slide-11 training driver (see the module docstring)")
    ap.add_argument("--data", required=True, help="sa, np or ns (data/V/{train,dev}.npz via pairdata.py), or an old probe pairs_F*.npz")
    ap.add_argument("--F", type=int, default=1024, help="vocabulary size = bits per paragraph")
    ap.add_argument("--model", choices=["hier", "flat"], default="hier")
    ap.add_argument("--clauses", type=int, default=200, help="clauses per class machine")
    ap.add_argument("--T", type=float, default=400)
    ap.add_argument("--s", type=float, default=40.0)
    ap.add_argument("--A", type=int, default=8, help="options per OR-group (hier only)")
    ap.add_argument("--K", type=int, default=2, help="OR-groups per paragraph (hier only)")
    ap.add_argument("--binary-inference", action="store_true")
    ap.add_argument("--and-group-normalization", action="store_true")
    ap.add_argument("--constant-update-p", action="store_true")
    ap.add_argument("--no-clipping", action="store_true")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--batch", type=int, default=5000, help="max pairs per fit() call (one fit = one pass over the batch)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=0, help="fixed random subset of train pairs (0 = all)")
    ap.add_argument("--n-dev", type=int, default=0, help="fixed random subset of dev pairs (0 = all)")
    ap.add_argument("--n-train-eval", type=int, default=2000, help="train pairs used for the train accuracy")
    ap.add_argument("--n-diag", type=int, default=2000, help="dev pairs used for clause fire rates and the rule ranking")
    ap.add_argument("--label-mode", default="pair", help="pair, or p1word=WORD (planted control: Y = paragraph A contains WORD)")
    ap.add_argument("--shuffle-labels", action="store_true", help="shuffle the train labels (the no-signal control)")
    ap.add_argument("--top", type=int, default=30, help="number of class-1 rules to print")
    ap.add_argument("--verify", action="store_true", help="numpy recomputation vs the package on 500 dev pairs")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after the pass that crosses this wall time (0 = off)")
    ap.add_argument("--eval-every", type=int, default=1, help="evaluate dev every N passes; the last 5 passes, the early-stop pass and the final pass are always evaluated (gates.md addendum, 2 Oct)")
    ap.add_argument("--early-stop-auc", type=float, default=0, help="screening runs only: stop if dev AUC is below this at --early-stop-frac of the planned passes (0 = off)")
    ap.add_argument("--early-stop-frac", type=float, default=0.3)
    ap.add_argument("--save-model", action="store_true", help="pickle the final model (package save()) as model_final.pkl in the run dir")
    ap.add_argument("--save-at", default="", help="comma list of passes at which to pickle the model as model_pN.pkl (always evaluated)")
    ap.add_argument("--out", default=".", help="output dir when not run under run_logged.py")
    return ap.parse_args()


# ---------------------------------------------------------------- data
def load(a, split):
    """X uint8 [n, 2F] = [A bits | B bits], y uint32, vocabulary list, data file path. split: train or dev."""
    if a.data.endswith(".npz"):  # old probe format (feas_slide11): paragraph bits Xp + index pairs P_{split}
        d, s = np.load(a.data), {"train": "train", "dev": "held"}[split]
        P, Xp = d[f"P_{s}"], d["Xp"]
        assert Xp.shape[1] == a.F, f"{a.data} has F={Xp.shape[1]}"
        vocab = json.load(open(a.data.replace("pairs_", "vocab_").replace(".npz", ".json")))
        return np.hstack([Xp[P[:, 0]], Xp[P[:, 1]]]), d[f"Y_{s}"].astype(np.uint32), vocab, a.data
    sys.path.insert(0, SRC)
    import pairdata  # the data loader: data/{sa,np,ns}/{train,dev,test}.npz
    p = pairdata.load_pairs(a.data, split, a.F)
    try:
        vocab = pairdata.load_vocab(a.F)
    except FileNotFoundError:
        vocab = pairdata.load_vocab(3072)[:a.F]
    X = np.hstack([p["A"].toarray(), p["B"].toarray()]).astype(np.uint8)
    return X, p["y"].astype(np.uint32), vocab, os.path.join(pairdata.DATA, a.data, f"{split}.npz")


# ---------------------------------------------------------------- model
def build(a):
    import PyHierarchicalTsetlinMachineCUDA.tm as htm
    if a.model == "hier":
        hs = ((htm.AND_GROUP, a.F), (htm.OR_ALTERNATIVES, a.A), (htm.AND_ALTERNATIVES, a.K), (htm.AND_GROUP, 2))
    else:
        hs = ((htm.AND_GROUP, a.F), (htm.AND_GROUP, 2))
    tm = htm.MultiClassTsetlinMachine(
        a.clauses, a.T, a.s, hierarchy_structure=hs, append_negated=False, number_of_state_bits=8,
        boost_true_positive_feedback=0, binary_inference=a.binary_inference,
        and_group_normalization=a.and_group_normalization, constant_update_p=int(a.constant_update_p),
        no_clipping=a.no_clipping, seed=a.seed)
    return tm, hs


def class_sums(tm, X, chunk=5000):
    """Package class sums [2, n] (clipped to +-T unless no_clipping)."""
    return np.hstack([tm.score(X[i:i + chunk].astype(np.uint32)) for i in range(0, len(X), chunk)])


def include_bits(t):
    """bool [clauses, leaves, F]: include bits of one class machine, read from the top bit of the TA states."""
    import pycuda.driver as cuda
    C, L, ch, b = t.number_of_clauses, t.number_of_ta_teams, t.number_of_literal_chunks_per_leaf, t.number_of_state_bits
    raw = np.empty(C * L * ch * b, np.uint32)
    cuda.memcpy_dtoh(raw, t.ta_state_hierarchy_gpu)
    msb = np.ascontiguousarray(raw.reshape(C, L, ch, b)[..., b - 1])
    bits = np.unpackbits(msb.view(np.uint8), axis=-1, bitorder="little")
    return bits[..., :t.number_of_literals_per_leaf].astype(bool)


# ---------------------------------------------------------------- numpy recomputation
def or_sums(inc, X, F, A, K):
    """int [n, clauses, 2, K]: how many options of each OR-group are true. An option is true iff all its
    included words are present in its paragraph (an empty option is always true). Flat model: A = K = 1."""
    n, C = X.shape[0], inc.shape[0]
    out = np.zeros((n, C, 2, K), np.int32)
    step = max(1, 16_000_000 // max(1, n * A * K))
    for p in range(2):
        Xs = sp.csr_matrix(X[:, p * F:(p + 1) * F], dtype=np.float32)
        for c0 in range(0, C, step):
            leaves = inc[c0:c0 + step, p * A * K:(p + 1) * A * K].reshape(-1, F)
            hits = (Xs @ sp.csc_matrix(leaves.T, dtype=np.float32)).toarray()
            out[:, c0:c0 + step, p] = (hits == leaves.sum(1)).reshape(n, -1, K, A).sum(-1)
    return out


def clause_values(sums, K, norm):
    """Clause output under the default (counting) inference: AND levels multiply, or take geometric means
    with and_group_normalization (K groups per paragraph, then the 2 paragraphs)."""
    s = sums.astype(np.float64)
    if norm:
        return np.prod(np.prod(s ** (1.0 / K), axis=3) ** 0.5, axis=2)
    return np.prod(s, axis=(2, 3))


def verify(tm, X, a):
    """Recompute both class machines in numpy from their include bits and compare with the package."""
    w = 1 - 2 * (np.arange(a.clauses) % 2)
    res, ok = {}, True
    for c, t in enumerate(tm.tms):
        inc = include_bits(t)
        L = inc.shape[1]
        fmap = (np.arange(L) // (a.A * a.K))[:, None] * a.F + np.arange(a.F)[None, :]
        sums = or_sums(inc, X, a.F, a.A, a.K)
        fire, val = (sums > 0).all(axis=(2, 3)), clause_values(sums, a.K, a.and_group_normalization)
        hv, _ = t.calc_hierarchy_votes(X.astype(np.uint32))
        pkg = np.array([h[-1][:, 0] for h in hv])  # the package's clause outputs [n, clauses]
        cs = (fire if a.binary_inference else val) @ w
        cs = cs if a.no_clipping else np.clip(cs, -a.T, a.T)
        cs_pkg = t.score(X.astype(np.uint32))
        r = dict(include_bits_match=bool((inc == t.get_literals().astype(bool)).all()),
                 feature_map_match=bool((t.map_ta_id_to_feature_id() == fmap).all()),
                 clause_fire_agreement=float((fire == (pkg > 0)).mean()),
                 clause_value_max_abs_diff=float(np.abs(val - pkg).max()),
                 class_sum_exact_share=float((cs == cs_pkg).mean()),
                 class_sum_max_abs_diff=float(np.abs(cs - cs_pkg).max()))
        ok &= r["include_bits_match"] and r["feature_map_match"] and r["clause_fire_agreement"] == 1.0
        ok &= r["class_sum_max_abs_diff"] == 0 if a.binary_inference else True
        res[f"class{c}"] = r
    res.update(n_pairs=len(X), inference="binary" if a.binary_inference else "default", passed=bool(ok))
    return res


# ---------------------------------------------------------------- metrics and rules
def auc(score, y):
    r, n1 = rankdata(score), int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def tuned_acc(score, y):
    """Accuracy at the best cut-off 'predict 1 iff score >= c', chosen on these same pairs."""
    o = np.argsort(-score, kind="stable")
    s, yy = score[o], y[o].astype(np.int64)
    tp, fp, n0 = np.cumsum(yy), np.cumsum(1 - yy), len(y) - int(yy.sum())
    end = np.r_[s[1:] != s[:-1], True]
    return float(max(((tp[end] + n0 - fp[end]) / len(y)).max(), n0 / len(y)))


def evaluate(tm, X, y):
    cs = class_sums(tm, X)
    score, pred = cs[1] - cs[0], np.argmax(cs, axis=0)
    return dict(acc=float((pred == y).mean()), auc=auc(score, y), tuned_acc=tuned_acc(score, y),
                always_no=float(1 - y.mean()), pred_pos=float(pred.mean()))


def option_text(bits, vocab, m=6):
    w = [vocab[i] for i in np.flatnonzero(bits)]
    return " & ".join(w[:m]) + (f" & ...(+{len(w) - m})" if len(w) > m else "")


def rules(inc, fire, y, vocab, A, K, top):
    """Top positive-polarity clauses of the class-1 machine by dev fire-rate difference (Y=1 minus Y=0).
    One OR-group per line; a group with an empty option is printed as ALWAYS-TRUE."""
    f1, f0 = fire[y == 1].mean(0), fire[y == 0].mean(0)
    pos = np.arange(0, inc.shape[0], 2)
    order = pos[np.argsort(-(f1 - f0)[pos], kind="stable")][:top]
    out = [f"RULES class 1: top {len(order)} positive clauses by dev fire-rate difference ({len(y)} dev pairs)"]
    for rank, c in enumerate(order, 1):
        out.append(f"#{rank} clause {c} (+): fires on dev Y=1 {f1[c]:.3f}, Y=0 {f0[c]:.3f}, diff {f1[c] - f0[c]:+.3f}")
        for p in range(2):
            for k in range(K):
                opts = [option_text(inc[c, o + A * (k + K * p)], vocab) for o in range(A)]
                body, empty = "[" + " | ".join(o for o in opts if o) + "]", opts.count("")
                if empty == A:
                    body = "ALWAYS-TRUE (empty)" if A == 1 else f"ALWAYS-TRUE (all {A} options empty)"
                elif empty:
                    body = f"ALWAYS-TRUE ({empty} of {A} options empty; others {body})"
                out.append(f"   {'AB'[p]}{k + 1}: {body}")
    return out


# ---------------------------------------------------------------- main
def main():
    a = get_args()
    if a.model == "flat":
        a.A = a.K = 1
    run_dir = os.environ.get("TMLM_RUN_DIR", a.out)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("htm_driver.py", "pairdata.py"):
        if os.path.exists(os.path.join(SRC, f)):
            shutil.copy(os.path.join(SRC, f), run_dir)
    t0 = time.time()
    Xtr, ytr, vocab, ftr = load(a, "train")
    Xdv, ydv, _, fdv = load(a, "dev")
    json.dump({f"{k}:{os.path.basename(f)}": hashlib.sha256(open(f, "rb").read()).hexdigest()
               for k, f in (("train", ftr), ("dev", fdv))}, open(os.path.join(run_dir, "dataset_hashes.json"), "w"))
    if a.n_train:
        keep = np.sort(np.random.default_rng(0).choice(len(ytr), a.n_train, replace=False))
        Xtr, ytr = Xtr[keep], ytr[keep]
    keep = np.random.default_rng(1).permutation(len(ydv))[:a.n_dev or None]
    Xdv, ydv = Xdv[keep], ydv[keep]
    if a.label_mode.startswith("p1word="):
        wid = vocab.index(a.label_mode.split("=", 1)[1])
        ytr, ydv = Xtr[:, wid].astype(np.uint32), Xdv[:, wid].astype(np.uint32)
    elif a.label_mode != "pair":
        raise SystemExit(f"unknown --label-mode {a.label_mode}")
    if a.shuffle_labels:
        ytr = np.random.default_rng(a.seed + 99991).permutation(ytr)
    tm, hs = build(a)
    print("CONFIG", json.dumps({**vars(a), "hierarchy": [[int(t), int(n)] for t, n in hs], "n_train": len(ytr),
                                "n_dev": len(ydv), "train_pos": float(ytr.mean()), "dev_pos": float(ydv.mean())}), flush=True)
    Xdg, ydg = Xdv[:a.n_diag], ydv[:a.n_diag]
    n_batches, hist = -(-len(ytr) // a.batch), []
    check_pass = int(np.ceil(a.early_stop_frac * a.passes)) if a.early_stop_auc else -1
    save_at = {int(x) for x in a.save_at.split(",") if x.strip()}
    for p in range(1, a.passes + 1):
        np.random.seed(a.seed + p)  # the package balances and shuffles with the global numpy RNG
        t1 = time.time()
        for b in np.array_split(np.random.permutation(len(ytr)), n_batches):
            assert 0 < ytr[b].sum() < len(b), "every batch must contain both labels"
            tm.fit(Xtr[b].astype(np.uint32), ytr[b])
        fit_s, t2 = time.time() - t1, time.time()
        out_of_time = bool(a.max_minutes and time.time() - t0 > 60 * a.max_minutes)
        if p in save_at:
            pickle.dump(tm.save(), open(os.path.join(run_dir, f"model_p{p}.pkl"), "wb"))
        if not (p % a.eval_every == 0 or p > a.passes - 5 or p == check_pass or out_of_time or p in save_at):
            # no evaluation this pass (training is unaffected: evaluation only reads the model)
            print("PASS", json.dumps({"pass": p, "fit_s": round(fit_s, 2), "ms_per_pair": round(1000 * fit_s / len(ytr), 4),
                                      "wall_s": round(time.time() - t0, 1)}), flush=True)
            continue
        tr, dv = evaluate(tm, Xtr[:a.n_train_eval], ytr[:a.n_train_eval]), evaluate(tm, Xdv, ydv)
        inc1 = include_bits(tm.tms[1])
        fire = (or_sums(inc1, Xdg, a.F, a.A, a.K) > 0).all(axis=(2, 3))
        r = {"pass": p, "train_acc": tr["acc"], "dev_acc": dv["acc"], "dev_auc": dv["auc"],
             "dev_tuned_acc": dv["tuned_acc"], "always_no": dv["always_no"], "pred_pos": dv["pred_pos"],
             "flag_constant": not 0.05 <= dv["pred_pos"] <= 0.95, "empty_leaf_share": float(1 - inc1.any(-1).mean()),
             "fire95_share": float((fire.mean(0) > 0.95).mean()), "fit_s": round(fit_s, 2),
             "ms_per_pair": round(1000 * fit_s / len(ytr), 4), "eval_s": round(time.time() - t2, 1),
             "wall_s": round(time.time() - t0, 1)}
        hist.append(r)
        print("EPOCH", json.dumps(r), flush=True)
        if p == check_pass and dv["auc"] < a.early_stop_auc:
            print(f"STOP: early stop, dev AUC {dv['auc']:.3f} < {a.early_stop_auc} at pass {p} of {a.passes}", flush=True)
            break
        if out_of_time:
            print("STOP: --max-minutes reached", flush=True)
            break
    perm = np.random.default_rng(a.seed + 7).permutation(len(ydv))  # partner-shuffle: keep A, random B
    ps = evaluate(tm, np.hstack([Xdv[:, :a.F], Xdv[perm, a.F:]]), ydv)
    final = {**hist[-1], "partner_shuffle_acc": ps["acc"], "partner_shuffle_auc": ps["auc"],
             "partner_shuffle_pred_pos": ps["pred_pos"], "best_pass_by_dev_auc": hist[int(np.nanargmax([h["dev_auc"] for h in hist]))]["pass"],
             "n_train": len(ytr), "n_dev": len(ydv)}
    print("FINAL", json.dumps(final), flush=True)
    text = "\n".join(rules(inc1, fire, ydg, vocab, a.A, a.K, a.top))
    print(text, flush=True)
    open(os.path.join(run_dir, "rules.txt"), "w").write(text + "\n")
    if a.save_model:
        pickle.dump(tm.save(), open(os.path.join(run_dir, "model_final.pkl"), "wb"))
    np.savez_compressed(os.path.join(run_dir, "include_bits.npz"), class1=np.packbits(inc1, axis=-1),
                        class0=np.packbits(include_bits(tm.tms[0]), axis=-1), F=a.F, A=a.A, K=a.K, model=a.model,
                        clauses=a.clauses, T=a.T, s=a.s, vocab=np.array(vocab))
    metrics = {k: {"value": v, "label": "measured"} for k, v in final.items()}
    metrics["history"] = {"value": hist, "label": "measured"}
    if a.verify:
        ver = verify(tm, Xdv[:500], a)
        print("VERIFY", json.dumps(ver), flush=True)
        metrics["verify"] = {"value": ver, "label": "measured"}
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
