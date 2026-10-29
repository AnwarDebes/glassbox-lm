"""REG_agent_vs_llm: pick LLM* per dataset from the dev result files only (no test file is read).

LLM* = the configuration with the highest accuracy on the 500 fresh dev rows (seed 2) among all evaluated ones; ties
to the faster one (lower mean seconds per row). Eligible: 500 answered rows, a result.json written at the end of the
run (or by combine_reask.py) with n 500, and no request error; others are listed and not eligible. L9t is not a
candidate (NOTE_review_fixes.md). usage: python -I select_llm_star.py --out runs/llm_star_choice.json
"""
import argparse
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {  # name: (folder pattern, model, thinking, k, policy text, max tokens (None: the script's default))
    "L9": ("runs/llm_hybrid_dev/{ds}_alone", "Qwen3.5-9B-Q8_0", False, 10, True, None),
    "L35": ("runs/llm_q3_L35/{ds}", "Qwen3.6-35B-A3B-Q4_K_M", False, 10, True, None),
    "L35t": ("runs/llm_q3_L35t/{ds}", "Qwen3.6-35B-A3B-Q4_K_M", True, 10, True, 4096),
    "L35t16": ("runs/llm_q3_L35t16/{ds}", "Qwen3.6-35B-A3B-Q4_K_M", True, 10, True, 16384),
    "L35k50": ("runs/llm_q3_L35k50/{ds}", "Qwen3.6-35B-A3B-Q4_K_M", False, 50, True, None),
    "L35k100": ("runs/llm_q3_L35k100/{ds}", "Qwen3.6-35B-A3B-Q4_K_M", False, 100, True, None),
    "L9-noG": ("runs/llm_abl/L9/{ds}", "Qwen3.5-9B-Q8_0", False, 10, False, None),
    "L35-noG": ("runs/llm_abl/L35/{ds}", "Qwen3.6-35B-A3B-Q4_K_M", False, 10, False, None),
}
MWOZ = {"L9": "runs/llm_mwoz_dev/L9", "L35": "runs/llm_mwoz_dev/L35"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"rule": "highest dev accuracy on the 500 fresh rows; ties to the lower mean seconds per row", "datasets": {}}
    for ds in ("star", "abcd", "multiwoz"):
        rows = []
        for name, (pat, model, think, k, pol, mt) in CONFIGS.items():
            folder = MWOZ.get(name, pat) if ds == "multiwoz" and name in MWOZ else pat
            p = os.path.join(H, folder.format(ds=ds), "preds.jsonl")
            if not os.path.exists(p):
                continue
            recs = [json.loads(x) for x in open(p) if x.strip()]
            n = len(recs)
            if n == 0:  # a run that has started but answered nothing yet
                continue
            acc = sum(r["pred"] == r["gold"] for r in recs) / n
            secs = sum(r["s"] for r in recs) / n
            rj = os.path.join(H, folder.format(ds=ds), "result.json")
            res = json.load(open(rj)) if os.path.exists(rj) else {}
            errs = sum(r.get("error") is not None for r in recs)
            ok = n == 500 and res.get("n") == 500 and errs == 0 and res.get("errors", 0) == 0
            rows.append({"config": name, "model": model, "thinking": think, "k": k, "policy_text": pol,
                         "max_tokens": mt, "n": n, "accuracy": acc, "mean_s": round(secs, 2), "errors": errs,
                         "result_json": bool(res), "eligible": ok, "path": folder.format(ds=ds)})
        elig = [r for r in rows if r["eligible"]]
        best = sorted(elig, key=lambda r: (-r["accuracy"], r["mean_s"]))[0] if elig else None
        out["datasets"][ds] = {"configs": rows, "llm_star": best}
    with open(a.out, "w") as h:
        json.dump(out, h, indent=1)
    for ds, v in out["datasets"].items():
        for r in v["configs"]:
            print(f"{ds:8s} {r['config']:8s} n {r['n']:3d} acc {r['accuracy']:.3f} {r['mean_s']:5.1f} s/row"
                  f"{'' if r['eligible'] else '  (not eligible: partial, no result.json or errors)'}")
        print(f"{ds:8s} LLM* = {v['llm_star']['config'] if v['llm_star'] else None}")


if __name__ == "__main__":
    main()
