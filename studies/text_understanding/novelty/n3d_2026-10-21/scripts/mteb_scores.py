# usage: python3.11 -I mteb_scores.py <json files...>
# prints task, model dir, revision, subset, main_score/accuracy, mteb_version for English test subsets
import json, sys, os
for p in sys.argv[1:]:
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception as e:
        print(f"{p}\tERROR {e}")
        continue
    parts = p.split(os.sep)
    model, rev = parts[-3], parts[-2]
    task = d.get("task_name", os.path.basename(p))
    ver = d.get("mteb_version")
    sc = d.get("scores")
    rows = []
    if isinstance(sc, dict):
        for split, lst in sc.items():
            if split != "test":
                continue
            for r in lst:
                sub = r.get("hf_subset")
                if sub in ("default", "en", "eng", None):
                    rows.append((split, sub, r.get("main_score"), r.get("accuracy")))
    else:
        t = d.get("test")
        if isinstance(t, dict):
            if "en" in t:
                t = t["en"]
            rows.append(("test", "legacy", t.get("main_score"), t.get("accuracy")))
    for split, sub, ms, acc in rows:
        print(f"{task}\t{model}\t{rev}\t{split}/{sub}\tmain={ms}\tacc={acc}\tmteb={ver}")
