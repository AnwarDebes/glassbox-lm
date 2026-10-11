"""E3 direct-label fallback: Check 2b on the EVAL split per run (slide10.check2b, unchanged), plus the report items of
the decision of 3 Oct 2026, 00:02 CEST:
  - the best rule's OR-groups in slide-8 style, each next to the true bracket it matches best (Jaccard, fixed words removed);
  - whether ONE AND THE SAME clause both accepts the slide-10 sentences and rejects their negated twins: per OR-group,
    the share of eval slide-10 sentences and of eval negated twins on which that group is true, and the words required
    by every option of a group (so required by the clause) that the negated twins lack.
The best rule is chosen on TRAIN sentences exactly as check2b does. Usage: python e3fb_report.py LABEL RUN_DIR ..."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slide10, htm_driver as hd
out_dir = os.environ.pop("TMLM_RUN_DIR", ".")
shutil.copy(os.path.abspath(__file__), out_dir)
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "slide10")
d, meta = np.load(f"{D}/pairs_slide10.npz"), json.load(open(f"{D}/slide10_meta.json"))
vocab = json.load(open(f"{D}/vocab_slide10.json"))
X, fam, split, in216 = d["Xp"].astype(bool), d["fam"], d["split"], d["in216"]
kind = np.array([f["kind"] for f in meta["families"]])[fam]
s10 = meta["families"][0]; fixed = set(s10["fixed"])
true = [(" | ".join(alts), set(" ".join(alts).split())) for alts in s10["brackets"]]
label, runs = sys.argv[1], sys.argv[2:]
rows, n_pass = [], {"288": 0, "216": 0}
for r in runs:
    log = open(os.path.join(r, "log.txt")).read().splitlines()
    cfg = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("CONFIG "))
    fin = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("FINAL "))
    print(f"\n== {label} {os.path.basename(r)}: s {cfg['s']}, seed {cfg['seed']}, final pass {fin['pass']} of {cfg['passes']}, "
          f"dev accuracy {fin['dev_acc']}")
    c = slide10.check2b(r, None, "eval")
    z = np.load(os.path.join(r, "include_bits.npz"))
    F, A, K = int(z["F"]), int(z["A"]), int(z["K"])
    inc = np.unpackbits(z["class1"], axis=-1, count=F).astype(bool)[c["best_rule"]]  # [K*A, F], one side
    assert inc.shape[0] == K * A, "expected a one-sentence (text_driver) rule"
    pos, neg = (kind == "slide10") & (split == 2), (kind == "negated") & (split == 2)
    groups = []
    for k in range(K):
        o = inc[A * k:A * (k + 1)]
        empty = ~o.any(1)
        common = o.all(0) if not empty.any() else np.zeros(F, bool)
        alts = list(dict.fromkeys(hd.option_text(x & ~common, vocab, m=12) or "-" for x in o if x.any()))
        words = {vocab[i] for i in np.flatnonzero(o.any(0))}
        sat = (X.astype(np.float32) @ o.T.astype(np.float32)) == o.sum(1)  # [n, A]: option k satisfied
        gtrue = sat.any(1)
        req = set(vocab[i] for i in np.flatnonzero(common))
        twins_lack = sorted(w for w in req if not X[neg][:, vocab.index(w)].any())
        jac = [len((words - fixed) & (W - fixed)) / max(1, len((words - fixed) | (W - fixed))) for _, W in true]
        j = int(np.argmax(jac))
        g = {"group": k + 1, "always_true": bool(empty.any()), "required_words": sorted(req), "alternatives": alts,
             "best_bracket": true[j][0], "jaccard": round(jac[j], 3),
             "true_on_slide10_eval": float(gtrue[pos].mean()), "true_on_negated_eval": float(gtrue[neg].mean()),
             "required_words_all_twins_lack": twins_lack}
        groups.append(g)
        style = "ALWAYS-TRUE (an empty option)" if g["always_true"] else \
            ((" & ".join(sorted(req)) + " & ") if req else "") + "[" + " | ".join(alts) + "]"
        print(f"   group {k + 1}: {style}\n      best bracket [{g['best_bracket']}] Jaccard {g['jaccard']}; true on "
              f"{g['true_on_slide10_eval']:.1%} of slide-10 eval sentences, {g['true_on_negated_eval']:.1%} of negated twins"
              + (f"; requires {twins_lack}, which every negated twin lacks" if twins_lack else ""))
    same = c["coverage_eval_288"] >= 0.9 and c["fires_eval_negated"] <= 0.05
    print(f"   ONE CLAUSE: this single clause accepts {c['coverage_eval_288']:.1%} (288) / {c['coverage_eval_216']:.1%} (216) "
          f"of unseen slide-10 sentences and fires on {c['fires_eval_negated']:.1%} of their negated twins -> "
          f"{'one and the same clause does both' if same else 'not both'}")
    rows.append({"run": os.path.basename(r), "s": cfg["s"], "seed": cfg["seed"], "final_pass": fin["pass"],
                 "dev_acc": fin["dev_acc"], "check2b": c, "groups": groups, "one_clause_does_both": bool(same)})
    n_pass["288"] += c["check2b_eval_pass_288"]; n_pass["216"] += c["check2b_eval_pass_216"]
res = {"label": label, "n_runs": len(rows), "pass_288": n_pass["288"], "pass_216": n_pass["216"]}
print("\nGATE", json.dumps(res))
nan = lambda v: None if isinstance(v, float) and v != v else v
for r in rows:
    r["check2b"] = {k: nan(v) for k, v in r["check2b"].items()}
json.dump({"runs": {"value": rows, "label": "measured"}, **{k: {"value": v, "label": "derived"} for k, v in res.items()}},
          open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
