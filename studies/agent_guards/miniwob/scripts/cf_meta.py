import json, sys, os
d = sys.argv[1]
for fn in sorted(os.listdir(d)):
    with open(os.path.join(d, fn)) as f:
        j = json.load(f)
    print(fn[:-5], "|", j.get("latest_version"), "|", j.get("license"), "|", (j.get("summary") or "")[:70], "|", j.get("home") or j.get("dev_url"))
