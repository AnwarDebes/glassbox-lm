"""The decision_layer agents on STAR or ABCD dev, with their dev predictions saved (dev only).

usage (decision_layer's .venv-dl):
  python -I agent_preds.py --dataset star --encoder none --table c4words-S1000-v1 --parts mem,words,xcust,xrecent \
      --arms tm --clauses 16000 --s 16 --seed 1 --out DIR
  python -I agent_preds.py --dataset star --encoder bge-large --arms lr_raw,mlp_raw --seed 1 --out DIR

Re-uses the study's src/dl unchanged, the way scripts/abcd_explore.py and scripts/table_run.py use it: the same view
(ABCDView, lex_k 1000), the same TM (TMConfig, T = clauses, 40 epochs, best dev epoch kept, init 'half'), the same
LR grid and MLP. Arms: tm (on the readable bits), lr_raw and mlp_raw (raw embeddings of full, cust and recent plus the
memory bits: the strongest frozen-encoder classifiers of the study). Writes DIR/<arm>.npz (top-5 label ids and scores
per dev row, gold) and DIR/result.json. The test split is never read.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(ROOT / "src"))
GRID_C = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]  # scripts/abcd_explore.py GRID_C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--encoder", default="none")
    ap.add_argument("--table", default="")
    ap.add_argument("--parts", default="mem,words,xcust,xrecent")
    ap.add_argument("--arms", default="tm")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--neg-sampling", default="hardest", choices=["hardest", "random"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-registration", help="score the TEST split too; needs this registration's .frozen.json")
    ap.add_argument("--dry-run-dev", action="store_true", help="with --test-registration: dev stands in for test")
    ap.add_argument("--save-model", action="store_true", help="also save the TM state (ta, w, cfg, names, buttons)")
    ap.add_argument("--expect-dev-acc", type=float, default=None,
                    help="the reference dev accuracy: if the best-dev snapshot differs, stop (exit 5) before any test row")
    a = ap.parse_args(argv)
    if a.test_registration and not a.dry_run_dev:
        fr = os.path.splitext(a.test_registration)[0] + ".frozen.json"
        if not os.path.exists(fr):
            raise SystemExit(f"refused: no freeze record {fr}; the test split is read only under a frozen registration")
    import numpy as np
    import torch

    meta = None
    if a.table:
        from dl import count_text
        meta = count_text.patch(a.table)
    from dl.abcd import ABCDView, load_examples, memory_from_rows
    from dl.linear import fit_logreg, logits
    from dl.mlp import fit_mlp, mlp_scores
    from dl.tm import TMConfig
    from dl.tm_init import make_tm

    frac = os.environ.get("DL_GPU_FRAC")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))
    tr, blist = load_examples("train", a.dataset)
    dv, _ = load_examples("dev", a.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{a.dataset}__{a.encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    edv = {f: E[f"dev__{f}"] for f in ("cust", "recent", "full")}
    ytr = np.array([r["y"] for r in tr])
    ydv = np.array([r["y"] for r in dv])
    C_ = len(blist)
    arms = a.arms.split(",")
    os.makedirs(a.out, exist_ok=True)
    res = {"dataset": a.dataset, "encoder": a.encoder, "table": meta, "parts": a.parts, "seed": a.seed,
           "n_dev": len(dv), "arms": {}, "argv": sys.argv}

    def save(arm, scores, extra):
        sc = torch.as_tensor(scores).float()
        top = sc.topk(min(5, C_), 1)
        acc = float((sc.argmax(1).cpu().numpy() == ydv).mean())
        np.savez_compressed(os.path.join(a.out, f"{arm}.npz"), top_ids=top.indices.cpu().numpy().astype(np.int16),
                            top_scores=top.values.cpu().numpy().astype(np.float32), gold=ydv.astype(np.int16))
        res["arms"][arm] = {"dev_acc": acc, **extra}
        print(arm, round(acc, 4), json.dumps(extra)[:300], flush=True)

    def load_test():  # called only after the dev check passed
        sp = "dev" if a.dry_run_dev else "test"
        te_, _ = load_examples(sp, a.dataset)
        res["dry_run_dev"] = a.dry_run_dev
        res["test_registration"] = a.test_registration
        res["n_test"] = len(te_)
        return te_, np.array([r["y"] for r in te_]), {f: E[f"{sp}__{f}"] for f in ("cust", "recent", "full")}

    if "tm" in arms:
        t0 = time.time()
        view = ABCDView(parts=tuple(p for p in a.parts.split(",") if p), lex_k=1000, seed=0).fit(tr, etr, blist)
        Xtr, Xdv = view.transform(tr, etr), view.transform(dv, edv)
        cfg = TMConfig(n_clauses=a.clauses, T=a.clauses, s=a.s, epochs=a.epochs, max_literals=0, budget_select="rank",
                       seed=a.seed, neg_sampling=a.neg_sampling)
        tm = make_tm(C_, Xtr.shape[1], cfg, init="half")
        tm.fit(Xtr, ytr, Xdv, ydv)
        sums = tm.class_sums(torch.as_tensor(Xdv, dtype=torch.uint8, device="cuda"))
        dev_acc = float((sums.argmax(1).cpu().numpy() == ydv).mean())
        if a.expect_dev_acc is not None:
            res["dev_check"] = {"expected": a.expect_dev_acc, "got": dev_acc,
                                "equal": abs(dev_acc - a.expect_dev_acc) < 1e-12}
            if not res["dev_check"]["equal"]:
                res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with open(os.path.join(a.out, "result.json"), "w") as h:
                    json.dump(res, h, indent=1, default=str)
                print("dev check FAILED", res["dev_check"], "- no test row was loaded", flush=True)
                sys.exit(5)
        if a.test_registration:  # the best-dev snapshot, as selected on dev; scored once on test
            te, yte, ete = load_test()
            Xte = view.transform(te, ete)
            ts = tm.class_sums(torch.as_tensor(Xte, dtype=torch.uint8, device="cuda")).float()
            tt = ts.topk(min(5, C_), 1)
            np.savez_compressed(os.path.join(a.out, "tm_test.npz"), top_ids=tt.indices.cpu().numpy().astype(np.int16),
                                top_scores=tt.values.cpu().numpy().astype(np.float32), gold=yte.astype(np.int16))
            res["test_acc_tm"] = float((ts.argmax(1).cpu().numpy() == yte).mean())
        if a.save_model:  # the same fields as scripts/abcd_explore.py --save
            st = tm.state()
            np.savez_compressed(os.path.join(a.out, "tm_model.npz"), ta=st["ta"], w=st["w"], cfg=st["cfg"],
                                names=np.array(view.names), buttons=np.array(blist))
        save("tm", sums, {"best_epoch": getattr(tm, "best_epoch", None), "n_bits": int(Xtr.shape[1]),
                          "clauses": a.clauses, "s": a.s, "neg_sampling": a.neg_sampling, "train_s": round(tm.train_seconds, 1),
                          "seconds": round(time.time() - t0, 1)})

    if "lr_raw" in arms or "mlp_raw" in arms:
        def raw(e, rows):
            return np.concatenate([e["full"], e["cust"], e["recent"],
                                   memory_from_rows(rows, blist).astype(np.float32)], 1)

        Rtr, Rdv = raw(etr, tr), raw(edv, dv)
        if "lr_raw" in arms:
            best = None
            for c in GRID_C:
                W, b = fit_logreg(Rtr, ytr, C_, c)
                lg = logits(Rdv, W, b)
                acc = float((lg.argmax(1).cpu().numpy() == ydv).mean())
                if best is None or acc > best[0]:
                    best = (acc, c, lg)
            save("lr_raw", best[2], {"C": best[1], "dim": int(Rtr.shape[1])})
        if "mlp_raw" in arms:
            net, info = fit_mlp(Rtr, ytr, Rdv, ydv, C_, seed=a.seed)
            save("mlp_raw", mlp_scores(net, Rdv), {**info, "dim": int(Rtr.shape[1])})

    # the "no neural network" claim: no encoder library may be loaded in this process (REG_agent_vs_llm, item 15)
    res["nn_modules_loaded"] = sorted(m for m in sys.modules if m.split(".")[0] in
                                      ("sentence_transformers", "transformers", "model2vec"))
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(a.out, "result.json"), "w") as h:
        json.dump(res, h, indent=1, default=str)


if __name__ == "__main__":
    main()
