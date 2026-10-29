"""Summarise AgentDojo metadata downloaded from PyPI and GitHub (paths passed as arguments; standard library only)."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
pypi = json.loads((d / "pypi_agentdojo.json").read_text())
info = pypi["info"]
print("PyPI name:", info.get("name"), "| latest version:", info.get("version"))
print("PyPI license field:", repr(info.get("license"))[:200])
print("PyPI license_expression:", info.get("license_expression"))
print("PyPI classifiers (licence):", [c for c in info.get("classifiers", []) if "icense" in c])
print("PyPI requires_python:", info.get("requires_python"))
print("PyPI project_urls:", info.get("project_urls"))
rel = pypi["releases"]
rows = []
for v, files in rel.items():
    if files:
        t = min(f["upload_time_iso_8601"] for f in files)
        rows.append((t, v, [f["packagetype"] for f in files], [f.get("yanked") for f in files]))
rows.sort()
print("Number of PyPI versions with files:", len(rows))
print("Last 12 PyPI uploads (time, version, types, yanked):")
for r in rows[-12:]:
    print("  ", r)
latest = info.get("version")
for f in rel.get(latest, []):
    print("latest file:", f["filename"], f["packagetype"], f["upload_time_iso_8601"], "sha256", f["digests"]["sha256"], "size", f["size"])
deps = info.get("requires_dist") or []
print("requires_dist (%d):" % len(deps))
for x in deps:
    print("   ", x)
repo = json.loads((d / "gh_repo.json").read_text())
print("GH repo:", repo.get("full_name"), "| default branch:", repo.get("default_branch"), "| license:", (repo.get("license") or {}).get("spdx_id"), "| pushed_at:", repo.get("pushed_at"), "| archived:", repo.get("archived"), "| stars:", repo.get("stargazers_count"))
lic = json.loads((d / "gh_license.json").read_text())
print("GH license file:", lic.get("path"), lic.get("license", {}).get("spdx_id"), "sha", lic.get("sha"))
import base64
txt = base64.b64decode(lic.get("content", "")).decode("utf-8", "replace")
print("---- LICENSE text (first 400 chars) ----")
print(txt[:400])
print("----")
rels = json.loads((d / "gh_releases.json").read_text())
print("GH releases listed:", len(rels))
for r in rels[:8]:
    print("  ", r.get("tag_name"), "| published", r.get("published_at"), "| prerelease", r.get("prerelease"), "| draft", r.get("draft"), "| name", r.get("name"))
tags = json.loads((d / "gh_tags.json").read_text())
print("GH tags listed:", len(tags), "first 10:", [t["name"] for t in tags[:10]])
