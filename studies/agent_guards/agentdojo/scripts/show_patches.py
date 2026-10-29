"""Print the patches of selected files from the GitHub compare JSON. Args: compare.json file-substring..."""
import json
import sys
from pathlib import Path

c = json.loads(Path(sys.argv[1]).read_text())
for f in c.get("files", []):
    if any(s in f["filename"] for s in sys.argv[2:]):
        print("=====", f["filename"])
        print(f.get("patch", "(no patch)"))
