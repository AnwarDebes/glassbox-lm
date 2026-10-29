"""Summarise the GitHub compare JSON (v0.1.35...main) and the v0.1.35 release notes. Paths as arguments."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
c = json.loads((d / "gh_compare_v0.1.35_main.json").read_text())
print("status:", c.get("status"), "| ahead_by:", c.get("ahead_by"), "| behind_by:", c.get("behind_by"), "| total_commits:", c.get("total_commits"))
for cm in c.get("commits", []):
    msg = cm["commit"]["message"].splitlines()[0][:110]
    print("  ", cm["commit"]["committer"]["date"], cm["sha"][:10], msg)
files = c.get("files", [])
print("files changed:", len(files))
for f in files[:60]:
    print("   ", f["status"], f["filename"], "+%d -%d" % (f["additions"], f["deletions"]))
rels = json.loads((d / "gh_releases.json").read_text())
for r in rels[:2]:
    print("==== release", r["tag_name"], r["published_at"], "target", r.get("target_commitish"))
    print((r.get("body") or "")[:1500])
