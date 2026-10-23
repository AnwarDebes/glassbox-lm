#!/usr/bin/env python
"""Rough CPU speed of one decision, text to action, one thread: the bge-large agent against the no-network one.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
      .venv-dl/bin/python scripts/static_speed.py --parts <chosen view> --model <saved e60b model> --n 1000

STAR dev rows only (n rows drawn with numpy default_rng(0)), batch 1, after warm-up. Per decision:
- reference (R18's agent): bge-large-en-v1.5 encodes the customer's words and the last 3 turns on the CPU,
  the view's whitened centres turn the vectors into bits, plus memory and word bits, then R18's TM-mem
  model of seed 1 decides (src/dl/cpu_tm.py);
- static mean (view A's text path): the table's average vectors, then the same kind of centres;
- existsT (view B's text path): a lookup of each token's bit code and a bitwise OR;
- the chosen view's full decision with the given model.
Before timing it checks that the bit names match each model and that the CPU decisions reproduce the
models' dev accuracies. Output: median, p90 and mean milliseconds per stage and per decision, and ratios,
in results/explore/static_speed_<tag>.json. Rough: the machine is shared; read the ratios, not the times.
The Proto centres are fitted on the GPU (as in every run) and then moved to the CPU for timing.
Added 2026-10-17 for the no-network agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import cpu_tm  # noqa: E402
from dl.abcd import ABCDView, load_examples, memory_from_rows  # noqa: E402
from dl.static_text import table  # noqa: E402


def to_cpu(view: ABCDView) -> ABCDView:
    for p in (view.intent, view.step, view.fstep):
        if p is not None:
            p.dev = "cpu"
            p.P, p.owner = p.P.cpu(), p.owner.cpu()
            if p.W is not None:
                p.W = p.W.cpu()
    return view


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "mean_ms": float(a.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="the chosen network-free view")
    ap.add_argument("--model", required=True, help="its saved model (abcd_explore.py --save)")
    ap.add_argument("--dev-acc", type=float, required=True, help="the model's dev accuracy from its abcd.jsonl row")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tag", default="R19")
    args = ap.parse_args()
    torch.set_num_threads(1)
    tr, blist = load_examples("train", "star")
    dv, _ = load_examples("dev", "star")
    ydv = np.array([r["y"] for r in dv])
    Eb = np.load(ROOT / "data" / "emb" / "star__bge-large.npz")
    Ep = np.load(ROOT / "data" / "emb" / "star__potion-32m.npz")
    emb = lambda E, s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")}  # noqa: E731
    parts = tuple(args.parts.split(","))
    ref_view = ABCDView(parts=("mem", "intent", "step", "words"), seed=0).fit(tr, emb(Eb, "train"), blist)
    view = ABCDView(parts=parts, seed=0).fit(tr, emb(Ep, "train"), blist)
    X_gpu = {"reference": ref_view.transform(dv, emb(Eb, "dev")), "chosen": view.transform(dv, emb(Ep, "dev"))}
    ref_view, view = to_cpu(ref_view), to_cpu(view)
    ref_dir = ROOT / "results" / "confirm" / "R18" / "tm_mem"
    assert ref_view.names == json.loads((ref_dir / "bit_names.json").read_text())
    zr = np.load(ref_dir / "model_seed1.npz")
    ref_model = cpu_tm.pack(zr["ta"], zr["w"])
    zm = np.load(ROOT / args.model)
    assert [str(n) for n in zm["names"]] == view.names, "the view differs from the model's"
    model = cpu_tm.pack(zm["ta"], zm["w"])
    # the CPU decision on the bits of the registered path (centres on the GPU) must give the registered dev
    # accuracy exactly; bits computed with the centres on the CPU can differ at float edges (reported)
    checks = {}
    for name, v, m, E, want in (("reference", ref_view, ref_model, Eb, 0.6394035900405327), ("chosen", view, model, Ep, args.dev_acc)):
        acc = float((cpu_tm.class_sums(m, X_gpu[name]).argmax(1) == ydv).mean())
        assert abs(acc - want) < 1e-12, (name, acc, want)
        X_cpu = v.transform(dv, emb(E, "dev"))
        acc_cpu = float((cpu_tm.class_sums(m, X_cpu).argmax(1) == ydv).mean())
        checks[name] = {"dev_acc_cpu_decision": acc, "dev_acc_cpu_bits": acc_cpu, "bits_differ": int((X_cpu != X_gpu[name]).sum()),
                        "rows_differ": int((X_cpu != X_gpu[name]).any(1).sum()), "n_rows": len(dv)}
    print("checks", checks, flush=True)

    from sentence_transformers import SentenceTransformer

    bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    tb = table()
    idx = np.random.default_rng(0).choice(len(dv), size=min(args.n, len(dv)), replace=False)
    rows = [dv[i] for i in idx]
    n_mem = 2 * len(blist) + 6
    T = {k: [] for k in ("bge_encode", "bge_proto", "mean_encode", "mean_proto", "existsT", "mem_words", "decision_ref",
                         "decision_chosen", "total_ref", "total_chosen")}
    for warm in (True, False):
        for r in (rows[:20] if warm else rows):
            texts = [r["cust"] or "(nothing yet)", r["recent"] or "(nothing yet)"]
            t0 = time.perf_counter()
            vb = bge.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=2)
            t1 = time.perf_counter()
            bits_ref = np.concatenate([ref_view.intent.transform(vb[:1]), ref_view.step.transform(vb[1:])], 1)
            t2 = time.perf_counter()
            mw = np.concatenate([memory_from_rows([r], blist), ref_view.lex.transform(None, texts=[r["recent"]])], 1)
            t3 = time.perf_counter()
            x_ref = np.concatenate([mw[:, :n_mem], bits_ref, mw[:, n_mem:]], 1)
            cpu_tm.class_sums(ref_model, x_ref).argmax(1)
            t4 = time.perf_counter()
            vm = tb.mean(texts)
            t5 = time.perf_counter()
            if view.intent is not None:
                view.intent.transform(vm[:1])
                view.step.transform(vm[1:])
            t6 = time.perf_counter()
            for x in (view.xcust, view.xrecent):
                if x is not None:
                    x.transform([r["cust"] if x is view.xcust else r["recent"]])
            t7 = time.perf_counter()
            x_ch = view.transform([r], {"cust": vm[:1], "recent": vm[1:], "full": vm[:1]})
            t8 = time.perf_counter()
            cpu_tm.class_sums(model, x_ch).argmax(1)
            t9 = time.perf_counter()
            if warm:
                continue
            ms = lambda a, b: (b - a) * 1000  # noqa: E731
            T["bge_encode"].append(ms(t0, t1)); T["bge_proto"].append(ms(t1, t2)); T["mem_words"].append(ms(t2, t3))
            T["decision_ref"].append(ms(t3, t4)); T["mean_encode"].append(ms(t4, t5)); T["mean_proto"].append(ms(t5, t6))
            T["existsT"].append(ms(t6, t7)); T["decision_chosen"].append(ms(t8, t9))
            T["total_ref"].append(ms(t0, t4))
            # the chosen agent: the mean vectors only if its view reads them, the full view build, its decision
            T["total_chosen"].append((ms(t4, t5) if view.intent is not None else 0.0) + ms(t7, t9))
    out = {"label": "rough: shared machine, one CPU thread, batch 1", "n": len(rows), "parts": parts, "model": args.model,
           "checks": checks, "stages": {k: stats(v) for k, v in T.items() if v},
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    s = out["stages"]
    out["ratio_total_ref_over_chosen_median"] = s["total_ref"]["median_ms"] / s["total_chosen"]["median_ms"]
    has_x = view.xcust is not None or view.xrecent is not None
    out["ratio_text_bge_over_existsT_median"] = s["bge_encode"]["median_ms"] / s["existsT"]["median_ms"] if has_x else None
    out["ratio_text_bge_over_mean_median"] = s["bge_encode"]["median_ms"] / s["mean_encode"]["median_ms"]
    (ROOT / "results" / "explore" / f"static_speed_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
