"""For packages in a micromamba dry-run solution that were built after a cutoff, find the newest
conda-forge record built on or before the cutoff (same name, same subdir) and print exact pins.
Reads micromamba's own JSON output (search --json) only; run with python -I."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import datetime, json, subprocess, sys

MM = f"{_REPO_ROOT}/studies/agent_guards/miniwob/scripts/mm.sh"
CUT = datetime.datetime(2026, 10, 7, tzinfo=datetime.timezone.utc).timestamp()


def ts(p):
    t = p.get("timestamp") or 0
    return t / 1000 if t > 1e11 else t


def day(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")


sol = json.load(open(sys.argv[1]))
link = sol["actions"].get("LINK", [])
pins, report = [], []
for p in sorted(link, key=lambda p: p["name"]):
    if ts(p) <= CUT:
        continue
    out = subprocess.run([MM, "search", "-c", "conda-forge", "--override-channels", "--platform", "linux-64",
                          "--offline", "--json", p["name"]], capture_output=True, text=True, check=True).stdout
    recs = [r for r in json.loads(out)["result"]["pkgs"] if r["name"] == p["name"] and r["subdir"] == p["subdir"]
            and not any(d.startswith(("__win", "__osx")) for d in r.get("depends", []))]
    old = sorted((r for r in recs if ts(r) <= CUT), key=ts, reverse=True)
    if not old:
        report.append(f"{p['name']}: no record on or before the cutoff")
        continue
    o = old[0]
    pins.append(f"{o['name']}={o['version']}={o['build_string']}")
    report.append(f"{p['name']}: solved {p['version']} {p['build_string']} ({day(ts(p), )}) -> pin {o['version']} {o['build_string']} ({day(ts(o))})")
print("\n".join(report))
print("PINS " + " ".join(pins))
