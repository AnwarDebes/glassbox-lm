"""Check 2b (gates.md) on the EVAL split for a set of finished E3 runs; the gate needs >= 3 of 5 seeds.
Calls slide10.check2b per run unchanged (its own per-run metrics.json is suppressed so it is not overwritten), then
writes one metrics.json with every run's output and the pass counts for the 288- and 216-sentence views.
Usage: python e3_gate.py LABEL RUN_DIR [RUN_DIR ...]"""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slide10
out_dir = os.environ.pop("TMLM_RUN_DIR", ".")
shutil.copy(os.path.abspath(__file__), out_dir)
label, runs = sys.argv[1], sys.argv[2:]
rows = []
for r in runs:
    print("==", label, os.path.basename(r))
    rows.append(slide10.check2b(r, None, "eval"))
res = {"label": label, "n_runs": len(rows),
       "pass_288": sum(r["check2b_eval_pass_288"] for r in rows), "pass_216": sum(r["check2b_eval_pass_216"] for r in rows)}
print("GATE", json.dumps(res))
print("\n| run | cov 288 | cov 216 | fires negated | near-miss (rule-generated) | broken | other | brackets 288 / 216 | pass 288 / 216 |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['run'][17:]} | {r['coverage_eval_288']:.3f} | {r['coverage_eval_216']:.3f} | {r['fires_eval_negated']:.3f} | "
          f"{r['fires_eval_nearmiss']:.3f} | {r['fires_eval_broken']:.3f} | {r['fires_eval_other']:.3f} | "
          f"{r['brackets_recovered_288']} / {r['brackets_recovered_216']} | {r['check2b_eval_pass_288']} / {r['check2b_eval_pass_216']} |")
nan = lambda v: None if isinstance(v, float) and v != v else v
json.dump({"runs": {"value": [{k: nan(v) for k, v in r.items()} for r in rows], "label": "measured"},
           **{k: {"value": v, "label": "derived"} for k, v in res.items()}},
          open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
