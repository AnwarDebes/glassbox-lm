#!/usr/bin/env python
"""Freeze R1 of A-Struct: the SHA-256 of every file the A-Struct CPU code depends on, written to
A/R1_A-Struct.frozen.json (enforced at the start of every real run, astruct/paths.verify_frozen) and the same hashes
in sha256sum format to A/MANIFEST_R1.sha256 (check with: cd .. && sha256sum -c <file>).

    python -I -B freeze.py                      # before the first real run only
    python -I -B freeze.py --amend "<reason>"   # a later, dated amendment (also needs its own line in LOG.md)

Enforced: A's code, tests and documents (README.md, REGISTRATION_R1.md), the imported study code (tally floor/,
dl tm.py / tm_init.py / cpu_tm_fast.py), the C4 table files and its tokenizer, MultiWOZ's MANIFEST.sha256 (S0 checks
every train file and dialog_acts.json against it), and the KenLM binaries. Recorded, not enforced (so that an edit
for another arm does not block A): the plan and its inputs, the CPU gate script, and (amendment 1) the Part 3 chain
wrapper `chain/part3_cpu_job.sh` with its helper `chain/part3_status.py`, which hold the lock shared with C.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths

ROOT = paths.ROOT


def enforced_files() -> list:
    A = paths.A_DIR
    files = sorted(A.glob("astruct/*.py")) + sorted(A.glob("*.py")) + sorted(A.glob("*.sh"))
    files += sorted(A.glob("tests/*.py")) + sorted(A.glob("tests/*.sh")) + [A / "tests" / "pytest.ini"]
    files += [A / "README.md", A / "REGISTRATION_R1.md"]
    t = paths.TALLY_SRC / "tally"
    files += [t / "__init__.py"] + [t / "floor" / f for f in ("__init__.py", "kn.py", "kenlm.py", "mix.py", "stats.py", "aa.py")]
    files += [paths.DL_SRC / "dl" / f for f in ("tm.py", "tm_init.py", "cpu_tm_fast.py")]
    files += [paths.C4 / f for f in ("words.txt", "vectors.npy", "tokenize_ref.py", "MANIFEST.sha256")]
    files += [paths.MWOZ / "MANIFEST.sha256"]
    files += [paths.KENLM_BIN / f for f in ("lmplz", "build_binary", "query")]
    return files


def recorded_files() -> list:
    d = paths.A_DIR.parent
    return [d / "PLAN_part3.md", d / "proposal_A-structured.md", d / "evidence_2026-10-21.md",
            d / "literature_2026-10-21.md", paths.CPU_LOAD,
            d / "chain" / "part3_cpu_job.sh", d / "chain" / "part3_status.py"]  # the chain wrapper (amendment 1)


def versions() -> dict:
    code = ("import sys, importlib; out = {'python': sys.version.split()[0]}\n"
            "for m in ('numpy', 'scipy', 'sklearn', 'numba', 'torch'):\n"
            "    try:\n        out[m] = importlib.import_module(m).__version__\n"
            "    except ImportError:\n        pass\nprint(__import__('json').dumps(out))")
    out = {}
    venvs = {"tm": ROOT / "glassbox-lm/studies/tm_language_model/.venv-tm/bin/python",
             "dl": ROOT / "glassbox-lm/studies/decision_layer/.venv-dl/bin/python"}
    for name, py in venvs.items():
        r = subprocess.run([str(py), "-I", "-B", "-c", code], capture_output=True, text=True, check=True, cwd=str(paths.TMP))
        out[name] = json.loads(r.stdout)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amend", default=None, help="reason of a dated amendment after real runs started")
    args = ap.parse_args()
    started = sorted(p.name for p in paths.OUT.iterdir()) if paths.OUT.exists() else []
    if started and not args.amend:
        raise SystemExit(f"real runs exist in {paths.OUT} ({started}); a new freeze needs --amend and a LOG.md line")
    missing = [str(p) for p in enforced_files() + recorded_files() if not p.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    prev = None
    if paths.FROZEN.exists():
        prev = {"sha256": paths.sha256_file(paths.FROZEN), **{k: json.loads(paths.FROZEN.read_text()).get(k)
                                                             for k in ("frozen_utc", "amendment")}}
    rec = {"what": "R1 of A-Struct v2 (PLAN_part3.md section 3, review amendments binding; A/REGISTRATION_R1.md)",
           "frozen_utc": paths.utc(),
           "enforced": {str(p.relative_to(ROOT)): paths.sha256_file(p) for p in enforced_files()},
           "recorded": {str(p.relative_to(ROOT)): paths.sha256_file(p) for p in recorded_files()},
           "versions": versions(), "amendment": args.amend, "previous_record": prev,
           "real_outputs_at_freeze": started}
    paths.FROZEN.write_text(json.dumps(rec, indent=1) + "\n")
    lines = [f"{h}  {p}" for p, h in list(rec["enforced"].items()) + list(rec["recorded"].items())]
    (paths.A_DIR / "MANIFEST_R1.sha256").write_text("\n".join(lines) + "\n")
    print(f"frozen {rec['frozen_utc']}: {len(rec['enforced'])} enforced, {len(rec['recorded'])} recorded files")
    print(f"{paths.sha256_file(paths.FROZEN)}  {paths.FROZEN.relative_to(ROOT)}")
    print(f"{paths.sha256_file(paths.A_DIR / 'MANIFEST_R1.sha256')}  {(paths.A_DIR / 'MANIFEST_R1.sha256').relative_to(ROOT)}")
    print(f"{rec['enforced'][str((paths.A_DIR / 'REGISTRATION_R1.md').relative_to(ROOT))]}  "
          f"{(paths.A_DIR / 'REGISTRATION_R1.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
