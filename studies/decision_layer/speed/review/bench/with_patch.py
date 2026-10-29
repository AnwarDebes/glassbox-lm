#!/usr/bin/env python
"""with_patch.py PATCH.py ENTRY.py [args...]: run a NEW patch file, then the decision_layer unedited entry script as __main__.

The pattern the decision_layer own scripts/table_run.py uses (dl.count_text.patch, then runpy): the patch may only replace names
in already-imported modules for this process (for example dl.tm_init.CoTM = FastCoTM, so make_tm builds the new
class); it never writes a file. ENTRY.py is run with runpy exactly as `python ENTRY.py args` would run it, with the
entry's folder first on sys.path (as for a plain run). Used by bench/run_gpu_job.sh for every non-"orig" run.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    patch, entry, rest = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3:]
    root = entry.resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(entry.resolve().parent))
    runpy.run_path(str(patch), run_name="speed_patch")
    sys.argv = [str(entry)] + rest
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
