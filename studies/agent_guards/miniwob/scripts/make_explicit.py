"""Turn a micromamba dry-run JSON solution (solved against the cutoff channel) into an @EXPLICIT
lock file whose URLs point at conda.anaconda.org/conda-forge, each with its sha256 from repodata."""
import json, sys

sol = json.load(open(sys.argv[1]))
lines = ["# conda-forge browser libraries for chrome-headless-shell / chromedriver (CfT 153.0.8010.12)",
         "# solved offline against conda-forge repodata filtered to builds <= 2026-10-07T00:00:00Z",
         "# platform: linux-64", "@EXPLICIT"]
for p in sorted(sol["actions"]["LINK"], key=lambda p: p["name"]):
    url = f"https://conda.anaconda.org/conda-forge/{p['subdir']}/{p['fn']}"
    lines.append(f"{url}#{p['sha256']}")
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
print(len(lines) - 4, "packages ->", sys.argv[2])
