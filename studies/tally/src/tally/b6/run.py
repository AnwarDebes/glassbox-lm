"""One B6 job: one (arm, dataset, seed), or one member of it. Single process, single thread.

    PYTHONPATH=src python -m tally.b6.run --arm ARM --dataset {fmnist,agnews} --seed S
        [--member NAME] [--label b6] [--smoke ROWS] [--epochs E]

Arms (see README.md):
    member    one blind member (--member), trained alone on every train example each epoch
    single    one machine with the clauses of the whole team
    ruleT     rule T: member i learns an example with probability = out-of-fold error of the rest
    nocomm    one member (--member) at the rule T rates of the same seed, shuffled over examples
    coupled   all members vote into one sum from the start
    seqboost  member k learns with probability = out-of-fold error of members 1..k-1
    fusion    TM head over the clause outputs of the frozen blind members
    gate      the neural gate of tmu.composite (MLP 128-64) over the frozen blind members

Every job writes results/<UTC>_<label>_<arm>_<dataset>_s<seed>[_<member>]/ through
tally.runlog.Run. Class sums of the final machines on calib, valid and devtest are saved in
scores.npz; dev-test accuracy is not printed (the report computes the verdict once).
Smoke runs (--smoke ROWS) use small stratified subsets and put a slice of the validation split
in place of dev-test, so the dev-test file is never read by a smoke run.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from tally.b6 import combine, config, data, engine
from tally.runlog import RESULTS, Run


def run_name(label, arm, ds, seed, member=None):
    base = f"{label}_{arm}_{ds}_s{seed}"
    return f"{base}_{member}" if member else base


def find_run(label, arm, ds, seed, member=None):
    """The finished run folder of a job this one depends on (exactly one must exist)."""
    name = run_name(label, arm, ds, seed, member)
    hits = []
    for d in sorted(glob.glob(os.path.join(RESULTS, f"*_{name}"))):
        try:
            with open(os.path.join(d, "run.json")) as f:
                if json.load(f).get("state") == "finished":
                    hits.append(d)
        except OSError:
            pass
    if len(hits) != 1:
        raise RuntimeError(f"need exactly one finished run {name}, found {len(hits)}")
    return hits[0]


class Data:
    """Views and labels of one dataset, restricted to smoke subsets when asked."""

    def __init__(self, ds, smoke=None):
        self.ds, self.smoke = ds, smoke
        self.y, self.rows = {}, {}
        for split in data.SPLITS:
            src = "valid" if (smoke and split == "devtest") else split
            y = data.labels(ds, src)
            if smoke:
                n = smoke if split == "train" else max(smoke // 4, 40)
                rows = data.subset_rows(y, n, seed={"train": 0, "calib": 1, "valid": 2, "devtest": 3}[split])
            else:
                rows = np.arange(len(y))
            self.rows[split], self.y[split] = (src, rows), y[rows]
        self._cache = {}

    def X(self, view, split):
        key = (view, split)
        if key not in self._cache:
            src, rows = self.rows[split]
            X = data.view(self.ds, view, src)
            self._cache[key] = X[rows] if self.smoke else X
        return self._cache[key]

    def files(self, views):
        names = [f"y_{s}.npy" for s in data.SPLITS]
        ext = ".npz" if self.ds == "agnews" else ".npy"
        for v in views:
            for s in data.SPLITS:
                if self.ds == "agnews" and v == "concat":
                    names += [f"{w}_{s}.npz" for w in data.AG_VIEWS]
                else:
                    names.append(f"{v}_{s}{ext}")
        return [p for p in data.files(self.ds, sorted(set(names))) if os.path.exists(p)]


def build(D, spec, seed, clauses=config.CLAUSES, T=None):
    m = engine.Member(spec, config.CLASSES[D.ds], D.X(spec["view"], "train"), seed, clauses, T)
    return m


def encode_all(D, m):
    return {s: m.encode(D.X(m.spec["view"], s)) for s in data.SPLITS}


def final_scores(members, encs, splits=("calib", "valid", "devtest")):
    return {f"{m.name}__{s}": m.class_sums(e[s]) for m, e in zip(members, encs) for s in splits}


def folds_of(y, K, seed):
    rng = np.random.default_rng(10_000 + seed)
    f = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        f[idx] = np.arange(len(idx)) % K
    return f


def valid_acc(m, enc, y):
    return combine.acc(m.class_sums(enc), y)


# ---------------------------------------------------------------- arms

def arm_single_members(run, D, a, specs, seeds, clauses, Ts, sampler=None):
    """Train members independently (blind members, the single machine, the no-communication
    control). sampler(epoch, i, n) returns the row indices member i sees in that epoch."""
    members = [build(D, s, sd, clauses, T) for s, sd, T in zip(specs, seeds, Ts)]
    encs = [encode_all(D, m) for m in members]
    y = D.y["train"]
    n = len(y)
    seen = np.zeros((a.epochs, len(members)), dtype=np.int64)
    for ep in range(a.epochs):
        t = time.time()
        for i, (m, e) in enumerate(zip(members, encs)):
            idx = np.arange(n) if sampler is None else sampler(ep, i, n)
            seen[ep, i] = m.fit_indices(e["train"], y, idx)
        msg = f"epoch {ep + 1} {time.time() - t:.1f}s seen {seen[ep].tolist()}"
        if a.eval_every and (ep + 1) % a.eval_every == 0:
            msg += " valid " + " ".join(f"{m.name}={valid_acc(m, e['valid'], D.y['valid']):.4f}"
                                        for m, e in zip(members, encs))
        run.print(msg)
    return members, encs, seen


def oof_scores(member_shadows, encs_train, folds, K):
    """Out-of-fold class sums on train: rows of fold k are scored by shadow copy k."""
    out = []
    for shadows, enc in zip(member_shadows, encs_train):
        S = np.zeros((enc.shape[0], shadows[0].n_classes), dtype=np.int32)
        for k in range(K):
            rows = np.flatnonzero(folds == k)
            S[rows] = shadows[k].class_sums(enc, rows)
        out.append(S)
    return out


def rest_error(oof, y, skip, kind):
    keep = [j for j in range(len(oof)) if j not in skip]
    if not keep:
        return np.full(len(y), 1.0 - 1.0 / oof[0].shape[1]), None
    S = combine.paper([oof[j] for j in keep], [combine.alpha(oof[j]) for j in keep])
    return combine.team_error(S, y, kind)


def arm_ruleT(run, D, a, specs):
    M, K, y = len(specs), a.folds, D.y["train"]
    n = len(y)
    rng = np.random.default_rng(20_000 + a.seed)
    folds = folds_of(y, K, a.seed)
    main = [build(D, s, config.member_seed(a.seed, i)) for i, s in enumerate(specs)]
    shadows = [[build(D, s, config.member_seed(a.seed, i, k + 1)) for k in range(K)] for i, s in enumerate(specs)]
    encs = [encode_all(D, m) for m in main]
    tr = [e["train"] for e in encs]
    p = np.full((M, n), 1.0 - 1.0 / config.CLASSES[D.ds])        # no information yet
    rates = np.zeros((a.epochs, M, n), dtype=np.float16)
    log = []
    for ep in range(a.epochs):
        t = time.time()
        if ep > 0 and ep % a.refresh == 0:
            oof = oof_scores(shadows, tr, folds, K)
            taus = []
            for i in range(M):
                p[i], tau = rest_error(oof, y, {i}, a.error)
                taus.append(tau)
            full_err, _ = rest_error(oof, y, set(), "01")
            log.append(dict(epoch=ep + 1, oof_team_err01=float(full_err.mean()), taus=taus,
                            rate=p.mean(axis=1).tolist()))
        t_ref = time.time() - t
        rates[ep] = p
        seen = []
        for i in range(M):
            idx = np.flatnonzero(rng.random(n) < p[i])
            seen.append(main[i].fit_indices(tr[i], y, idx))
            for k in range(K):
                idx_k = np.flatnonzero((folds != k) & (rng.random(n) < p[i]))
                shadows[i][k].fit_indices(tr[i], y, idx_k)
        run.print(f"epoch {ep + 1} {time.time() - t:.1f}s (refresh {t_ref:.1f}s) "
                  f"rate {np.round(p.mean(axis=1), 4).tolist()} seen {seen}")
    np.savez_compressed(run.path("rates.npz"), rates=rates)
    with open(run.path("refresh_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return main, encs, dict(mean_rate=float(rates.astype(np.float32).mean()))


def arm_seqboost(run, D, a, specs):
    M, K, y = len(specs), a.folds, D.y["train"]
    n = len(y)
    rng = np.random.default_rng(30_000 + a.seed)
    folds = folds_of(y, K, a.seed)
    main, encs, oof, info = [], [], [], []
    for i, s in enumerate(specs):
        m = build(D, s, config.member_seed(a.seed, i))
        e = encode_all(D, m)
        sh = [build(D, s, config.member_seed(a.seed, i, k + 1)) for k in range(K)] if i < M - 1 else []
        if i == 0:
            p = np.ones(n)
        else:
            p, _ = rest_error(oof, y, set(), a.error)
        for ep in range(a.epochs):
            t = time.time()
            seen = m.fit_indices(e["train"], y, np.flatnonzero(rng.random(n) < p))
            for k, shk in enumerate(sh):
                shk.fit_indices(e["train"], y, np.flatnonzero((folds != k) & (rng.random(n) < p)))
            run.print(f"member {m.name} epoch {ep + 1} {time.time() - t:.1f}s rate {p.mean():.4f} seen {seen}")
        if sh:
            oof.extend(oof_scores([sh], [e["train"]], folds, K))
        info.append(dict(member=m.name, rate=float(p.mean())))
        main.append(m)
        encs.append(e)
    return main, encs, dict(members=info)


def arm_coupled(run, D, a, specs):
    y = D.y["train"]
    rng = np.random.default_rng(40_000 + a.seed)
    main = [build(D, s, config.member_seed(a.seed, i)) for i, s in enumerate(specs)]
    encs = [encode_all(D, m) for m in main]
    for ep in range(a.epochs):
        t = time.time()
        engine.coupled_epoch(main, [e["train"] for e in encs], y, np.arange(len(y)), rng)
        run.print(f"epoch {ep + 1} {time.time() - t:.1f}s")
    return main, encs, {}


def blind_runs(a, specs):
    return {s["name"]: find_run(a.label, "member", a.dataset, a.seed, s["name"]) for s in specs}


def arm_fusion(run, D, a, specs):
    dirs = blind_runs(a, specs)
    bits = {}
    for s in specs:
        m = engine.Member(s, config.CLASSES[D.ds], D.X(s["view"], "train"), seed=0)
        m.tm = engine.load_tm(os.path.join(dirs[s["name"]], f"member_{s['name']}.pkl"))
        for split in data.SPLITS:
            bits.setdefault(split, []).append(m.clause_outputs(m.encode(D.X(s["view"], split))))
    X = {split: np.ascontiguousarray(np.concatenate(v, axis=1)) for split, v in bits.items()}
    h = config.HEAD
    spec = dict(name="head", view="clauses", patch=None, weighted=h["weighted"], t_ratio=h["T"] / h["clauses"],
                s=h["s"], max_lit=h["max_lit"], drop=h["drop"])
    head = engine.Member(spec, config.CLASSES[D.ds], X["train"], config.member_seed(a.seed, 99), h["clauses"])
    enc = {split: head.encode(X[split]) for split in data.SPLITS}
    y = D.y["train"]
    for ep in range(a.epochs):
        t = time.time()
        head.fit_indices(enc["train"], y, np.arange(len(y)))
        run.print(f"epoch {ep + 1} {time.time() - t:.1f}s")
    return [head], [enc], dict(inputs={k: list(v.shape) for k, v in X.items()}, blind_runs=dirs)


def arm_gate(run, D, a, specs):
    from sklearn.neural_network import MLPClassifier
    dirs = blind_runs(a, specs)
    sc = {}
    for s in specs:
        z = np.load(os.path.join(dirs[s["name"]], "scores.npz"))
        for split in ("calib", "valid", "devtest"):
            sc.setdefault(split, []).append(z[f"{s['name']}__{split}"].astype(np.float64))
    view = "raw" if D.ds == "fmnist" else "concat"

    def feats(split):
        X = D.X(view, split)
        X = X.toarray() if hasattr(X, "toarray") else X
        return X.astype(np.float32) / (255.0 if view == "raw" else 1.0)

    yc = D.y["calib"]
    margins = []
    for S in sc["calib"]:
        N = combine.per_sample(S)
        r = np.arange(len(yc))
        other = np.where(np.eye(N.shape[1], dtype=bool)[yc], -np.inf, N).max(axis=1)
        margins.append(N[r, yc] - other)
    target = np.argmax(np.stack(margins, axis=1), axis=1)       # best member on each calib example
    mlp = MLPClassifier(hidden_layer_sizes=config.GATE["hidden"], activation="relu", solver="adam",
                        max_iter=config.GATE["max_iter"], random_state=a.seed)
    mlp.fit(feats("calib"), target)
    out = {}
    for split in ("calib", "valid", "devtest"):
        W = np.zeros((len(D.y[split]), len(specs)))
        W[:, mlp.classes_] = mlp.predict_proba(feats(split))
        out[split] = combine.library(sc[split], W)
    run.print(f"gate target shares {np.bincount(target, minlength=len(specs)).tolist()}")
    return out, dict(blind_runs=dirs, target_share=np.bincount(target, minlength=len(specs)).tolist())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=config.ARMS)
    ap.add_argument("--dataset", required=True, choices=list(config.MEMBERS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--member", default=None)
    ap.add_argument("--label", default="b6")
    ap.add_argument("--smoke", type=int, default=None, help="train rows of a smoke run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=config.FOLDS)
    ap.add_argument("--refresh", type=int, default=config.REFRESH)
    ap.add_argument("--error", default="soft", choices=["soft", "01"])
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--t-scale", type=float, default=None, help="tuning only: T multiplier of --member")
    ap.add_argument("--registration", default=None)
    a = ap.parse_args()
    if a.smoke and not a.label.startswith("b6smoke"):
        raise SystemExit("smoke runs carry a label starting with b6smoke")
    if a.t_scale is not None:
        if a.arm != "member" or not a.label.startswith("b6tune"):
            raise SystemExit("--t-scale is for member runs labelled b6tune*")
        config.T_SCALE[a.member] = a.t_scale
    if a.arm in ("member", "nocomm") and not a.member:
        raise SystemExit(f"--member is required for arm {a.arm}")
    ds = a.dataset
    a.epochs = a.epochs or config.EPOCHS[ds]
    specs = config.MEMBERS[ds]
    M = len(specs)
    D = Data(ds, a.smoke)

    views = sorted({s["view"] for s in specs})
    if a.arm == "single":
        base = config.SINGLE[ds]["base"]
        views = ["concat"] if base == "concat" else [next(s for s in specs if s["name"] == base)["view"]]
    if a.arm == "gate":
        views = ["raw" if ds == "fmnist" else "concat"]
    cfg = dict(vars(a), members=specs, clauses=config.CLAUSES, single=config.SINGLE[ds], head=config.HEAD,
               gate=config.GATE, data_manifest=data.manifest(ds), smoke_devtest_is_valid=bool(a.smoke))
    run = Run(run_name(a.label, a.arm, ds, a.seed, a.member), cfg, D.files(views), a.registration)
    t0 = time.time()
    extra = {}
    if a.arm == "member":
        i = next(j for j, s in enumerate(specs) if s["name"] == a.member)
        members, encs, seen = arm_single_members(run, D, a, [specs[i]], [config.member_seed(a.seed, i)],
                                                 config.CLAUSES, [None])
        extra["seen"] = int(seen.sum())
    elif a.arm == "single":
        base = config.SINGLE[ds]["base"]
        spec = dict(specs[0], view="concat") if base == "concat" else next(s for s in specs if s["name"] == base)
        T_single = config.T_of(spec, M * config.CLAUSES)       # the recipe ratio of the base member
        spec = dict(spec, name="single")
        members, encs, seen = arm_single_members(run, D, a, [spec], [config.member_seed(a.seed, 0)],
                                                 M * config.CLAUSES, [T_single])
        extra["T"] = T_single
    elif a.arm == "nocomm":
        i = next(j for j, s in enumerate(specs) if s["name"] == a.member)
        src = find_run(a.label, "ruleT", ds, a.seed)
        rates = np.load(os.path.join(src, "rates.npz"))["rates"][:, i, :].astype(np.float64)
        if rates.shape[0] < a.epochs:
            raise RuntimeError("rule T run has fewer epochs than asked")
        rng = np.random.default_rng(50_000 + 100 * a.seed + i)

        def sampler(ep, _i, n):
            return np.flatnonzero(rng.random(n) < rng.permutation(rates[ep]))
        members, encs, seen = arm_single_members(run, D, a, [specs[i]], [config.member_seed(a.seed, i)],
                                                 config.CLAUSES, [None], sampler)
        extra.update(seen=int(seen.sum()), ruleT_run=os.path.basename(src))
    elif a.arm == "ruleT":
        members, encs, extra = arm_ruleT(run, D, a, specs)
    elif a.arm == "seqboost":
        members, encs, extra = arm_seqboost(run, D, a, specs)
    elif a.arm == "coupled":
        members, encs, extra = arm_coupled(run, D, a, specs)
    elif a.arm == "fusion":
        members, encs, extra = arm_fusion(run, D, a, specs)
    t_train = time.time() - t0

    if a.arm == "gate":
        out, extra = arm_gate(run, D, a, specs)
        np.savez_compressed(run.path("scores.npz"), **{f"team__{k}": v for k, v in out.items()})
        metrics = dict(valid_acc=combine.acc(out["valid"], D.y["valid"]))
    else:
        scores = final_scores(members, encs)
        np.savez_compressed(run.path("scores.npz"), **scores)
        if a.arm in ("member", "nocomm", "fusion"):
            for m in members:
                m.save(run.path(f"member_{m.name}.pkl"))
        metrics = {f"valid_acc_{m.name}": combine.acc(scores[f"{m.name}__valid"], D.y["valid"]) for m in members}
        if len(members) > 1:
            names = [m.name for m in members]
            al = [combine.alpha(scores[f"{n}__calib"]) for n in names]
            team = combine.paper([scores[f"{n}__valid"] for n in names], al)
            metrics["valid_acc_team_paper"] = combine.acc(team, D.y["valid"])
            if a.arm == "coupled":
                joint = sum(scores[f"{m.name}__valid"] / m.T for m in members)
                metrics["valid_acc_team_joint"] = combine.acc(joint, D.y["valid"])
    metrics.update(extra, train_s=round(t_train, 1), total_s=round(time.time() - t0, 1),
                   epochs=a.epochs, smoke=bool(a.smoke), n_train=int(len(D.y["train"])))
    with open(run.path("labels.json"), "w") as f:
        json.dump({s: D.y[s].tolist() for s in ("calib", "valid", "devtest")}, f)
    run.finish(metrics)


if __name__ == "__main__":
    main()
