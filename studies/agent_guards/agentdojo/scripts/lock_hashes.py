"""Print hashed requirement lines for the named packages from a uv.lock (wheels + sdist hashes). Args: uv.lock name..."""
import sys
import tomllib
from pathlib import Path

lock = tomllib.loads(Path(sys.argv[1]).read_text())
want = set(sys.argv[2:])
for pkg in lock["package"]:
    if pkg["name"] in want:
        hashes = [w["hash"] for w in pkg.get("wheels", [])]
        if "sdist" in pkg:
            hashes.append(pkg["sdist"]["hash"])
        print(f"{pkg['name']}=={pkg['version']} " + " ".join(f"--hash={h}" for h in hashes))
