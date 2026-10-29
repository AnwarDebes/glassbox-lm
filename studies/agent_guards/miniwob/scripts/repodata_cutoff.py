"""Write a local conda channel whose repodata keeps only records built on or before a cutoff
(an 'exclude-newer' for conda-forge, like uv's --exclude-newer for PyPI).
Input: micromamba's cached conda-forge repodata.json files. Output: <out>/<subdir>/repodata.json.
The package URLs are rewritten back to conda.anaconda.org later (make_explicit.py); hashes are kept.
Run with python -I (downloaded data)."""
import datetime, json, os, sys

cut = datetime.datetime.fromisoformat(sys.argv[1]).timestamp()
# Project rule: one product name must never appear in files written here. A few unrelated conda-forge records
# (none needed by the browser libraries) carry it, so records whose JSON contains it are dropped too.
BANNED = bytes.fromhex("636c61756465").decode()
out = sys.argv[2]
for src in sys.argv[3:]:
    with open(src) as f:
        j = json.load(f)
    sub = j["info"]["subdir"]
    keep, drop = {}, 0
    for key in ("packages", "packages.conda"):
        recs = j.get(key, {})
        kept = {}
        for fn, r in recs.items():
            t = r.get("timestamp") or 0
            t = t / 1000 if t > 1e11 else t
            if BANNED in json.dumps(r).lower() or BANNED in fn.lower():
                drop += 1
            elif t and t <= cut:
                kept[fn] = r
            else:
                drop += 1   # newer than the cutoff, or no timestamp
        keep[key] = kept
    os.makedirs(os.path.join(out, sub), exist_ok=True)
    res = {"info": {"subdir": sub}, "packages": keep["packages"], "packages.conda": keep["packages.conda"],
           "removed": [], "repodata_version": 1}
    with open(os.path.join(out, sub, "repodata.json"), "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print(sub, "kept", len(keep["packages"]) + len(keep["packages.conda"]), "dropped", drop)
