"""Train the OR-group Tsetlin machine (orgroup_tm.py) on the slide-10 sentence data of the hierarchical_tm study
(read only) and score it with that study's own Check 2b scorer, slide10.check2b.

  --version text   one sentence, y = 1 for a sentence of the family, 0 for a negated twin, near-miss or broken
                   sentence (data/slide10/text/{train,dev}.npz; the direct-label fallback of Check 2b)
  --version pairs  two sentences, y = 1 iff both come from the same family (data/slide10/pairs_slide10.npz; the
                   procedure of the proposal)
The model's groups are exported in the layout of the study's include_bits.npz (one member word per option; a
group without members becomes one option that holds every word, which no sentence satisfies, because here an empty
group is false). For the text version the export is checked on every TRAIN and DEV sentence: the study's accept()
on the exported bits must equal the model's own firing, clause by clause. check2b then picks the best rule on the
TRAIN sentences (the first one in clause order when several tie) and scores it on --split.

--split eval is refused unless --registered names the record of registrations/E3_slide10_text.md, the plan, the
sources and the pinned files are unchanged since that record, and the arguments are the registered ones
(e3_common.py). The same checks run for the dry run of the pipeline (--registered registrations/E3_dry.frozen.json,
DEV sentences).

Usage (from the study folder):
  .venv/bin/python -I src/run_slide10.py --version text --K 12 --seed 1 --name e2_text_K12_s1
Writes runs/<UTC>_<name>/{config.json, metrics.json, include_bits.npz, check2b_<split>.txt, state.npz} and appends
one row to logs/RUNS_slide10.md (not with --no-log).
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import platform
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
STUDY = os.path.normpath(os.path.join(ROOT, "..", "hierarchical_tm"))
DATA = os.path.join(STUDY, "data", "slide10")
RUNS = os.path.join(ROOT, "runs")
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True  # nothing is written into the study folder
sys.path.insert(1, os.path.join(STUDY, "src"))

import e3_common as e3

ENGINE_SHA_AT_START = e3.sha(os.path.join(SRC, "orgroup_tm.py"))  # taken before the engine is imported

import numba
import slide10 as s10

from metrics import auc, plain_acc, tuned_acc
from orgroup_tm import OrGroupTM, pack_pairs


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", choices=["text", "pairs"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--clauses", type=int, default=20)
    ap.add_argument("--K", type=int, default=12, help="groups per sentence")
    ap.add_argument("--T", type=float, default=8)
    ap.add_argument("--mode", choices=["near", "side", "all"], default="near")
    ap.add_argument("--init-state", type=int, default=-1)
    ap.add_argument("--seed-state", type=int, default=0)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--p-rec", type=float, default=1.0)
    ap.add_argument("--p-rep", type=float, default=1.0)
    ap.add_argument("--q-seed", type=float, default=1.0)
    ap.add_argument("--max-seed", type=int, default=1)
    ap.add_argument("--reject", choices=["random", "all", "weakest"], default="weakest")
    ap.add_argument("--reinforce", choices=["all", "critical", "share"], default="critical")
    ap.add_argument("--neg-frac", type=float, default=0.0)
    ap.add_argument("--p-decay", type=float, default=0.0)
    ap.add_argument("--seed-all", action="store_true")
    ap.add_argument("--tied", action="store_true", help="pairs: one set of groups for both sentences")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--split", choices=["dev", "eval"], default="dev", help="sentences Check 2b is scored on")
    ap.add_argument("--registered", default="", help="the record of the registration this run belongs to")
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default=RUNS)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    if a.split == "eval" and not a.registered:
        ap.error("--split eval needs --registered <the record of registrations/E3_slide10_text.md>")
    if a.registered:
        dry = e3.is_dry_record(a.registered)
        if a.split == "eval" and dry:
            ap.error("the dry record does not open the EVAL sentences")
        rec, problems = e3.preflight(a.registered, dry)
        problems += e3.check_run_args(a, dry)
        if rec and rec.get("sources_sha256", {}).get("src/orgroup_tm.py") != ENGINE_SHA_AT_START:
            problems.append("the engine read at the start is not the engine of the record")
        if problems:
            ap.error("registered run refused before any model was trained or scored: " + "; ".join(problems))
    return a


def load(a):
    """(ids, bounds, y) for train and dev, the number of sides, the vocabulary size and the file hashes."""
    pairs_file = os.path.join(DATA, "pairs_slide10.npz")
    prov = {"pairs_slide10.npz": e3.sha(pairs_file), "slide10_meta.json": e3.sha(os.path.join(DATA, "slide10_meta.json")),
            "vocab_slide10.json": e3.sha(os.path.join(DATA, "vocab_slide10.json"))}
    d = np.load(pairs_file)
    F = d["Xp"].shape[1]
    out = {}
    if a.version == "pairs":
        out["train"] = (*pack_pairs(d["Xp"], d["P_train"]), d["Y_train"].astype(np.int64))
        out["dev"] = (*pack_pairs(d["Xp"], d["P_held"]), d["Y_held"].astype(np.int64))
        sides = 2
    else:
        for sp in ("train", "dev"):
            f = os.path.join(DATA, "text", f"{sp}.npz")
            t = np.load(f)
            prov[f"text/{sp}.npz"] = e3.sha(f)
            n = len(t["y"])
            bounds = np.zeros((n, 2), np.int64)
            bounds[:, 0], bounds[:, 1] = t["indptr"][:-1], t["indptr"][1:]
            out[sp] = (t["indices"].astype(np.int32), bounds, t["y"].astype(np.int64))
        sides = 1
    if a.n_train:
        ids, bounds, y = out["train"]
        out["train"] = (ids, bounds[:a.n_train], y[:a.n_train])
    return out, sides, F, prov, pairs_file


def include_bits_of(tm, F):
    """Include bits in the study's layout, [2 * positive clauses, sides * K * A, F] with the clauses at the even
    rows, and A. A member word is one option; spare options repeat the first member; a group without members gets
    options that hold every word."""
    groups = [gs for gs, p in zip(tm.groups(), tm.pol) if p > 0]
    A = max([1] + [len(w) for gs in groups for w in gs])
    inc = np.zeros((2 * len(groups), tm.G * A, F), bool)
    for c, gs in enumerate(groups):
        for j, words in enumerate(gs):
            if len(words) == 0:
                inc[2 * c, j * A:(j + 1) * A, :] = True  # never satisfied: the group is false
                continue
            for o in range(A):
                inc[2 * c, j * A + o, words[o] if o < len(words) else words[0]] = True
    return inc, A


def export_include_bits(tm, F, run_dir):
    inc, A = include_bits_of(tm, F)
    np.savez_compressed(os.path.join(run_dir, "include_bits.npz"), class1=np.packbits(inc, axis=-1), F=F, A=A, K=tm.K)
    return inc, A


def export_equals_model(tm, inc, A, X):
    """True iff the study's accept() on the exported bits equals the model's own firing on every row of X, clause by
    clause (text version: one sentence per row)."""
    rows = np.arange(len(X))
    ids, bounds = pack_pairs(X, rows[:, None])
    _, fire = tm.votes(ids, bounds, return_fire=True)
    pos = np.flatnonzero(tm.pol > 0)
    acc = s10.accept(inc[0::2], X, A, tm.K, 1)[:, :, 0]
    return bool(np.array_equal(acc, fire[:, pos]))


def tied_best_rules(inc, A, K, sides, pairs_file, split_name):
    """The clauses that share the highest selection score of check2b on the TRAIN sentences, each with its own
    numbers on the scored split (the formulas of check2b). check2b itself takes the first of them."""
    d = np.load(pairs_file)
    with open(os.path.join(os.path.dirname(pairs_file), "slide10_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(os.path.dirname(pairs_file), "vocab_slide10.json")) as f:
        vocab = json.load(f)
    X, fam, split, in216 = d["Xp"], d["fam"], d["split"], d["in216"]
    kind = np.array([fm["kind"] for fm in meta["families"]])[fam]
    acc = s10.accept(inc[0::2], X, A, K, sides)
    cover, fire = acc.all(-1), acc.any(-1)
    e = {"dev": 1, "eval": 2}[split_name]

    def rate(v, sel):
        return v[sel].mean(0) if sel.any() else np.full(v.shape[1:], np.nan)
    score = rate(cover, (kind == "slide10") & (split == 0)) - np.nanmax(
        [rate(fire, (kind == k) & (split == 0)) for k in s10.NEG_KINDS], axis=0)
    top = np.flatnonzero(score == score.max())
    fam0 = meta["families"][0]
    fixed, out = set(fam0["fixed"]), []
    for b in top:
        row = {"rule": int(2 * b),
               "coverage_288": float(rate(cover, (kind == "slide10") & (split == e))[b]),
               "coverage_216": float(rate(cover, (kind == "slide10") & (split == e) & in216)[b]),
               "fires_max": float(np.nanmax([rate(fire, (kind == k) & (split == e))[b] for k in s10.NEG_KINDS]))}
        for view, key in (("288", "brackets"), ("216", "brackets_216")):
            true = [set(" ".join(alts).split()) for alts in fam0[key]]
            jac = s10.bracket_jaccard(inc[0::2][b], vocab, true, fixed, A, K, sides)
            row[f"brackets_{view}"] = int(sum(j >= 0.8 for j, _ in jac))
            row[f"pass_{view}"] = bool(row[f"coverage_{view}"] >= 0.9 and row["fires_max"] <= 0.05 and row[f"brackets_{view}"] >= 4)
        out.append(row)
    return out


def score_set(tm, ids, bounds, y):
    v = tm.votes(ids, bounds)
    return {"plain_acc": plain_acc(v, y), "tuned_acc": tuned_acc(v, y), "auc": auc(v, y)}


def main(argv=None):
    a = get_args(argv)
    start = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    t0 = time.time()
    data, sides, F, prov, pairs_file = load(a)
    tm = OrGroupTM(a.clauses, F, sides=sides, K=a.K, T=a.T, mode=a.mode, init_state=a.init_state,
                   seed_state=a.seed_state, r=a.r, p_rec=a.p_rec, p_rep=a.p_rep, q_seed=a.q_seed, neg_frac=a.neg_frac,
                   max_seed=a.max_seed, reject=a.reject, reinforce=a.reinforce, p_decay=a.p_decay, seed_all=a.seed_all,
                   tied=a.tied, seed=a.seed)
    hist = []
    for p in range(1, a.passes + 1):
        t1 = time.time()
        tm.fit_pass(*data["train"])
        row = {"pass": p, "fit_s": round(time.time() - t1, 3)}
        row.update({f"dev_{k}": v for k, v in score_set(tm, *data["dev"]).items()})
        row["members"] = int(tm.nmem.sum())
        hist.append(row)
        print("PASS", json.dumps(row), flush=True)
    assert tm.nmem_consistent()
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(a.out_root, f"{stamp}_{a.name}")
    os.makedirs(run_dir, exist_ok=False)
    inc, A = export_include_bits(tm, F, run_dir)
    export_checked = None
    if a.version == "text":
        d = np.load(pairs_file)
        export_checked = export_equals_model(tm, inc, A, d["Xp"][d["split"] <= 1])
        assert export_checked, "the exported include bits do not reproduce the model's firing on the TRAIN and DEV sentences"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        chk = s10.check2b(run_dir, data=pairs_file, split_name=a.split)
    with open(os.path.join(run_dir, f"check2b_{a.split}.txt"), "w") as f:
        f.write(buf.getvalue())
    tied = tied_best_rules(inc, A, tm.K, sides, pairs_file, a.split)
    assert tied[0]["rule"] == chk["best_rule"], "check2b did not take the first of the tied rules"
    assert (tied[0]["coverage_288"], tied[0]["brackets_288"], tied[0]["pass_288"], tied[0]["pass_216"]) == (
        chk[f"coverage_{a.split}_288"], chk["brackets_recovered_288"], chk[f"check2b_{a.split}_pass_288"],
        chk[f"check2b_{a.split}_pass_216"]), "the numbers of the first tied rule differ from those of check2b"
    final = dict(hist[-1])
    final.update({f"train_{k}": v for k, v in score_set(tm, *data["train"]).items()})
    final["check2b"] = chk
    final["tied_best_rules"] = tied
    final["export_equals_model_on_train_and_dev"] = export_checked
    final["events"] = tm.event_counts()
    final["options_per_group_exported"] = A
    final["wall_s"] = round(time.time() - t0, 2)
    engine_now = e3.sha(os.path.join(SRC, "orgroup_tm.py"))
    assert engine_now == ENGINE_SHA_AT_START, "src/orgroup_tm.py changed while the run was going on"
    cfg = {**vars(a), "sides": sides, "F": F, "data": prov, "orgroup_tm_sha256": ENGINE_SHA_AT_START,
           "slide10_py_sha256": e3.sha(os.path.join(STUDY, "src", "slide10.py")),
           "htm_driver_py_sha256": e3.sha(os.path.join(STUDY, "src", "htm_driver.py")),
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__},
           "run_start_utc": start, "run_utc": stamp}
    cfg["out_root"] = os.path.relpath(os.path.abspath(a.out_root), ROOT)  # no machine paths in the stored config
    cfg["registered"] = os.path.relpath(os.path.abspath(a.registered), ROOT) if a.registered else ""
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": hist}, f, indent=1)
    np.savez_compressed(os.path.join(run_dir, "state.npz"), state=tm.state, pol=tm.pol)
    sn = a.split
    print("FINAL", json.dumps({k: v for k, v in final.items() if k != "check2b"}), flush=True)
    print("CHECK2B", json.dumps(chk), flush=True)
    if not a.no_log:
        log = os.path.join(ROOT, "logs", "RUNS_slide10.md")
        new = not os.path.exists(log)
        with open(log, "a") as f:
            if new:
                f.write("# Slide-10 runs (every run, newest last; Check 2b by the slide10.check2b of hierarchical_tm)\n\n"
                        "| run folder | version, seed | clauses, K, T, mode, reinforce, reject, r, init, passes | dev plain acc / AUC | "
                        "split | coverage 288 / 216 | fires negated / near-miss / broken | brackets 288 / 216 | pass 288 / 216 |\n"
                        "|---|---|---|---|---|---|---|---|---|\n")
            f.write(f"| {os.path.basename(run_dir)} | {a.version}, {a.seed} | {a.clauses}, {a.K}, {a.T:g}, {a.mode}, {a.reinforce}, "
                    f"{a.reject}, {a.r}, {a.init_state}, {a.passes}{'' if not a.p_decay else f', decay {a.p_decay:g}'}"
                    f"{', seed_all' if a.seed_all else ''}{', tied' if a.tied else ''} | {final['dev_plain_acc']:.4f} / "
                    f"{final['dev_auc']:.4f} | {sn} | "
                    f"{chk[f'coverage_{sn}_288']:.3f} / {chk[f'coverage_{sn}_216']:.3f} | {chk[f'fires_{sn}_negated']:.3f} / "
                    f"{chk[f'fires_{sn}_nearmiss']:.3f} / {chk[f'fires_{sn}_broken']:.3f} | {chk['brackets_recovered_288']} / "
                    f"{chk['brackets_recovered_216']} | {chk[f'check2b_{sn}_pass_288']} / {chk[f'check2b_{sn}_pass_216']} |\n")
    return run_dir, final


if __name__ == "__main__":
    main()
