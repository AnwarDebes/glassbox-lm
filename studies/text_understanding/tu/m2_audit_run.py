#!/usr/bin/env python
"""M2 dry runs and CPU tests only (tu/PLAN_M2.md "Dry run"): runs one M2 script under an audit. Not used by the
registered chain.

    python -I -B m2_audit_run.py [--cpu] SCRIPT.py ARGS...

- Every member read from a zip whose member names start with "mtop/" (zipfile.ZipFile.read and .open) is recorded; a
  request for MTOP's en/test.txt raises at once ("AUDIT: mtop/en/test.txt was opened"), so a step that passes the
  audit never opened the test file, and no test label column could be read.
- At exit one JSON line goes to $M2_AUDIT_LOG (if set): the script, its arguments, the MTOP members read, and which of
  sentence_transformers, transformers and model2vec were imported (the network-free check: the C4 and a3B steps must
  import none of them; only the potion feature build loads the potion snapshot through sentence_transformers).
- --cpu: torch.cuda.synchronize becomes a no-op (dl.tm.fit calls it unconditionally), for runs on the CPU.
"""

from __future__ import annotations

import atexit
import json
import os
import runpy
import sys
import zipfile

TEST_MEMBER = "mtop/en/test.txt"
NN = ("sentence_transformers", "transformers", "model2vec")


def main() -> None:
    args = sys.argv[1:]
    cpu = bool(args) and args[0] == "--cpu"
    if cpu:
        args = args[1:]
    assert args, "usage: m2_audit_run.py [--cpu] SCRIPT.py ARGS..."
    members: list[str] = []
    orig_open = zipfile.ZipFile.open

    def audited_open(self, name, *a, **k):
        n = name if isinstance(name, str) else name.filename
        if n.startswith("mtop/"):
            members.append(n)
        if n == TEST_MEMBER:
            raise RuntimeError(f"AUDIT: {TEST_MEMBER} was opened")
        return orig_open(self, name, *a, **k)

    zipfile.ZipFile.open = audited_open  # ZipFile.read goes through ZipFile.open

    def report() -> None:
        path = os.environ.get("M2_AUDIT_LOG")
        if path:
            line = {"script": os.path.basename(args[0]), "args": args[1:], "mtop_members": sorted(set(members)),
                    "nn_imported": {m: m in sys.modules for m in NN}}
            with open(path, "a") as f:
                f.write(json.dumps(line) + "\n")

    atexit.register(report)
    if cpu:
        import torch

        torch.cuda.synchronize = lambda *a, **k: None  # no GPU in this run
    sys.argv = list(args)
    runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
