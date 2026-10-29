import json, sys, datetime
j = json.load(open(sys.argv[1]))
link = j["actions"].get("LINK", [])
cut = datetime.datetime(2026, 10, 7, tzinfo=datetime.timezone.utc).timestamp()
newer = 0
tot = 0
for p in sorted(link, key=lambda p: p["name"]):
    ts = p.get("timestamp") or 0
    if ts > 1e11: ts = ts / 1000
    d = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d") if ts else "?"
    flag = "NEWER-THAN-2026-10-07" if ts and ts > cut else ""
    newer += bool(flag)
    tot += p.get("size", 0) or 0
    print(f'{p["name"]:32s} {p["version"]:14s} {p["build_string"]:22s} {d} {p.get("license","?")[:45]:45s} {flag}')
print("n_link", len(link), "n_newer", newer, "total MB", round(tot/1e6, 1))
