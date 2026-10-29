"""Bitwise-identity test of the prototype (fast_tm.FastCoTM) against the decision_layer CoTM, on the CPU.

For each case: train the original and the prototype from the same seed on the same data; compare, exactly,
  - the automaton states ta and the weights w after training (np.array_equal),
  - the dev history (every val_acc value, as floats, equal) and best_epoch,
  - the generator state at the end (torch.equal on gen.get_state(): the same random numbers were drawn),
  - the integer class sums of the final machine on the dev rows (original class_sums vs prototype class_sums).
A case that trains without dev data compares the last-epoch state (no best-dev snapshot).
Writes <out>.json and prints PASS / FAIL per case. CPU only (CUDA_VISIBLE_DEVICES=), 2 threads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prof_tm  # noqa: E402  (data loader, config table; patches torch.cuda.synchronize for CPU)
from fast_tm import FastCoTM  # noqa: E402

import dl.tm as T  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def run(cls, C, f, cfg, data, chunk, with_dev=True, **opt):
    T.CHUNK_ELEMS = chunk
    tm = cls(C, f, cfg, device="cpu", **opt) if opt else cls(C, f, cfg, device="cpu")
    Xtr, ytr, Xdv, ydv = data
    t0 = time.perf_counter()
    if with_dev:
        tm.fit(Xtr, ytr, Xdv, ydv)
    else:
        tm.fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    S = CoTM.class_sums(tm, torch.as_tensor(Xdv, dtype=torch.uint8)).numpy()  # original evaluation code
    S2 = tm.class_sums(torch.as_tensor(Xdv, dtype=torch.uint8)).numpy()  # the class's own (prototype's E1 if fast)
    return tm, dt, S, S2


def feedback_direct(cls, C, f, cfg, data, chunk, steps=12, **opt):
    """The general feedback path (pos / neg / masks, as the bandit learner calls it), driven by a fixed numpy stream."""
    T.CHUNK_ELEMS = chunk
    tm = cls(C, f, cfg, device="cpu", **opt) if opt else cls(C, f, cfg, device="cpu")
    Xtr, ytr, _, _ = data
    rng = np.random.default_rng(123)
    for _ in range(steps):
        idx = rng.choice(len(Xtr), cfg.batch_size, replace=False)
        Xb = torch.as_tensor(Xtr[idx], dtype=torch.uint8)
        pos = torch.as_tensor(ytr[idx], dtype=torch.long)
        pm = torch.as_tensor(rng.random(cfg.batch_size) < 0.6)
        neg = torch.as_tensor(np.where(rng.random(cfg.batch_size) < 0.3, -1, rng.integers(0, C, cfg.batch_size)), dtype=torch.long)
        nm = torch.as_tensor(rng.random(cfg.batch_size) < 0.8)
        tm.feedback(Xb, pos, pm, neg, nm)
    return tm


def same(a: CoTM, b: CoTM) -> dict:
    return {
        "ta": bool(np.array_equal(a.ta.numpy(), b.ta.numpy())),
        "w": bool(np.array_equal(a.w.numpy(), b.w.numpy())),
        "gen_state": bool(torch.equal(a.gen.get_state(), b.gen.get_state())),
        "history": [h["val_acc"] for h in a.history] == [h["val_acc"] for h in b.history],
        "best_epoch": getattr(a, "best_epoch", None) == getattr(b, "best_epoch", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--group", default="all", choices=["all", "unchunked", "chunks2", "chunks5", "misc"],
                    help="run one group of cases (each group stays well under 10 minutes on 2 threads)")
    args = ap.parse_args()
    torch.set_num_threads(2)
    Xtr, ytr, Xdv, ydv, C = prof_tm.load("abcd", args.rows, 1024)
    f = Xtr.shape[1]
    data = (Xtr, ytr, Xdv, ydv)
    m = 2 * f
    N = 256
    big = 50_000_000
    cases = []
    allopt = {}
    variants = [("all", allopt), ("xonly", dict(probs=False, elem=False, evalonce=False)),
                ("probs", dict(xonly=False, elem=False, evalonce=False)),
                ("probs_expand", dict(xonly=False, probs=True, probs_expand=True, elem=False, evalonce=False)),
                ("elem", dict(xonly=False, probs=False, evalonce=False)),
                ("all+expand", dict(probs_expand=True))]
    if args.quick:
        variants = variants[:1]
    groups = [("unchunked", big), ("chunks2", (N // 2) * m), ("chunks5", -(-N // 5) * m)]
    for chunk_name, chunk in groups:
        if args.group not in ("all", chunk_name):
            continue
        for s in (16.0, 8.0):
            cfg = TMConfig(n_clauses=N, T=N, s=s, epochs=args.epochs, batch_size=128, seed=1)
            ref, t_ref, S_ref, _ = run(CoTM, C, f, cfg, data, chunk)
            for vname, opt in variants:
                fast, t_fast, S_o, S_f = run(FastCoTM, C, f, cfg, data, chunk, **opt)
                r = same(ref, fast)
                r["class_sums_orig_code"] = bool(np.array_equal(S_ref, S_o))
                r["class_sums_fast_code"] = bool(np.array_equal(S_ref, S_f))
                ok = all(r.values())
                cases.append({"case": f"{chunk_name}/s{s:g}/{vname}", "pass": ok, **r, "t_ref_s": round(t_ref, 3),
                              "t_fast_s": round(t_fast, 3), "history": [h["val_acc"] for h in ref.history]})
                print(f"{'PASS' if ok else 'FAIL'} {chunk_name}/s{s:g}/{vname} {r} ref {t_ref:.2f}s fast {t_fast:.2f}s", flush=True)
    if args.group not in ("all", "misc"):
        return finish(args, cases, N, f, C)
    # no dev data: the last-epoch state (no snapshot)
    cfg = TMConfig(n_clauses=N, T=N, s=16.0, epochs=args.epochs, batch_size=128, seed=7)
    for chunk_name, chunk in (("unchunked", big), ("chunks2", (N // 2) * m)):
        ref, _, _, _ = run(CoTM, C, f, cfg, data, chunk, with_dev=False)
        fast, _, _, _ = run(FastCoTM, C, f, cfg, data, chunk, with_dev=False)
        r = same(ref, fast)
        ok = all(r.values())
        cases.append({"case": f"nodev/{chunk_name}/all", "pass": ok, **r})
        print(f"{'PASS' if ok else 'FAIL'} nodev/{chunk_name}/all {r}", flush=True)
    # negation off, and mix negative sampling (the other branch of _neg_classes)
    for name, cfg in (("negation_off", TMConfig(n_clauses=N, T=N, s=8.0, epochs=args.epochs, batch_size=128, seed=2, negation=False)),
                      ("neg_mix", TMConfig(n_clauses=N, T=N, s=8.0, epochs=args.epochs, batch_size=128, seed=3, neg_sampling="mix:0.5"))):
        for chunk_name, chunk in (("unchunked", big), ("chunks2", (N // 2) * (2 * f if cfg.negation else f))):
            ref, _, S_ref, _ = run(CoTM, C, f, cfg, data, chunk)
            fast, _, S_o, S_f = run(FastCoTM, C, f, cfg, data, chunk)
            r = same(ref, fast)
            r["class_sums_fast_code"] = bool(np.array_equal(S_ref, S_f))
            ok = all(r.values())
            cases.append({"case": f"{name}/{chunk_name}/all", "pass": ok, **r})
            print(f"{'PASS' if ok else 'FAIL'} {name}/{chunk_name}/all {r}", flush=True)
    # small s, more epochs: automata reach the clamp bounds 1 and 2 * n_states (counted in the record)
    cfg = TMConfig(n_clauses=N, T=N, s=2.0, epochs=6, batch_size=128, seed=4)
    for chunk_name, chunk in (("unchunked", big), ("chunks2", (N // 2) * m)):
        ref, _, S_ref, _ = run(CoTM, C, f, cfg, data, chunk)
        fast, _, S_o, S_f = run(FastCoTM, C, f, cfg, data, chunk)
        r = same(ref, fast)
        r["class_sums_fast_code"] = bool(np.array_equal(S_ref, S_f))
        ok = all(r.values())
        ta = ref.ta.numpy()
        cases.append({"case": f"s2_clamp/{chunk_name}/all", "pass": ok, **r, "ta_at_1": int((ta == 1).sum()),
                      "ta_at_max": int((ta == 2 * cfg.n_states).sum()), "ta_elems": int(ta.size)})
        print(f"{'PASS' if ok else 'FAIL'} s2_clamp/{chunk_name}/all {r} ta==1: {(ta == 1).sum()} ta==512: {(ta == 512).sum()} of {ta.size}", flush=True)
    # the general feedback path (masks, given negatives with -1 = drawn)
    cfg = TMConfig(n_clauses=N, T=N, s=8.0, epochs=1, batch_size=128, seed=5)
    for chunk_name, chunk in (("unchunked", big), ("chunks2", (N // 2) * m)):
        ref = feedback_direct(CoTM, C, f, cfg, data, chunk)
        fast = feedback_direct(FastCoTM, C, f, cfg, data, chunk)
        r = same(ref, fast)
        ok = all(r.values())
        cases.append({"case": f"feedback_masks/{chunk_name}/all", "pass": ok, **r})
        print(f"{'PASS' if ok else 'FAIL'} feedback_masks/{chunk_name}/all {r}", flush=True)
    return finish(args, cases, N, f, C)


def finish(args, cases, N, f, C):
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "torch": torch.__version__, "rows": args.rows, "group": args.group,
           "epochs": args.epochs, "n_clauses": N, "n_features": f, "n_classes": C, "all_pass": all(c["pass"] for c in cases),
           "n_cases": len(cases), "cases": cases}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("ALL PASS" if res["all_pass"] else "SOME FAIL", len(cases), "cases")


if __name__ == "__main__":
    main()
