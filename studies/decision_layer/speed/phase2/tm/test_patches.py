"""CPU self-test of patch_T1T3.py, patch_R1.py and patch_T1T3_R1.py through review/bench/with_patch.py, exactly as
review/bench/run_gpu_job.sh runs a patch (python -B with_patch.py PATCH ENTRY ...), with CUDA hidden.

Per patch (a fresh process each): the entry unit_root/scripts/patch_check_entry.py checks that only dl.tm_init.CoTM
was replaced, that make_tm builds a FastCoTM with the named configuration, and that a short CPU training through make_tm
equals dl.tm.CoTM's (unchunked and 2 chunks). This test also checks the patch's two log lines (import: configuration
and the pinned SHA-256 of fast_cotm.py / configs.py; construction: make_tm's arguments and the switches).
Refusals (MUST FAIL): a patch next to a tampered fast_cotm.py copy stops before anything runs; a second patch in the
same process stops ("already replaced").

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
      HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 <decision_layer>/.venv-dl/bin/python -I -B test_patches.py \
      --out OUTDIR
Writes only into OUTDIR (must not exist) plus a temporary folder in this directory that it removes again, and makes
the link unit_root/src -> ../../../../src (the study's src) if it is missing.
Exit 0 only if every check passes and every refusal happens.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
WITH_PATCH = f"{_REPO_ROOT}/studies/decision_layer/speed/review/bench/with_patch.py"
ENTRY = HERE / "unit_root" / "scripts" / "patch_check_entry.py"
NAMES = ("T1T3", "R1", "T1T3_R1")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "OPENBLAS_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(argv: list[str], log: Path) -> tuple[int, str]:
    p = subprocess.run(["nice", "-n", "19", PY, "-B", WITH_PATCH, *argv], env=ENV, capture_output=True, text=True,
                       timeout=600, check=False)
    text = f"$ with_patch.py {' '.join(argv)}\n--- stdout\n{p.stdout}--- stderr\n{p.stderr}--- exit {p.returncode}\n"
    log.write_text(text)
    return p.returncode, p.stdout + p.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refused: {a.out} exists")
    a.out.mkdir(parents=True)
    utc0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    link = HERE / "unit_root" / "src"  # the repository tracks no links: the published copy makes this one here
    if not link.exists() and not link.is_symlink():
        link.symlink_to("../../../../src")
    if not link.resolve().samefile(f"{_REPO_ROOT}/studies/decision_layer/src"):
        raise SystemExit("unit_root/src must link to the study's src")
    fc_sha = sha(HERE / "fast_cotm.py")
    results = []
    for n in NAMES:
        patch = HERE / f"patch_{n}.py"
        rc, text = run([str(patch), str(ENTRY), "--name", n, "--out", str(a.out / f"{n}.json")], a.out / f"{n}.log")
        body = json.loads((a.out / f"{n}.json").read_text()) if (a.out / f"{n}.json").exists() else {}
        r = {"case": f"patch_{n}", "exit": rc, "entry_all_pass": body.get("all_pass") is True,
             "import_line": f"speed_patch {n}: dl.tm_init.CoTM -> FastCoTM switches" in text and fc_sha in text,
             "construction_line": f"speed_patch {n}: make_tm -> FastCoTM(" in text,
             "patch_sha_in_log": sha(patch) in text}
        r["pass"] = rc == 0 and all(v for k, v in r.items() if k not in ("case", "exit"))
        results.append(r)
        print(("PASS" if r["pass"] else "FAIL"), r, flush=True)
    # MUST-FAIL 1: a tampered fast_cotm.py next to the patch is refused before anything runs
    tmp = Path(tempfile.mkdtemp(prefix="tmp_tamper_", dir=HERE))
    try:
        for f in ("patch_T1T3.py", "configs.py", "fast_cotm.py"):
            shutil.copyfile(HERE / f, tmp / f)
        with (tmp / "fast_cotm.py").open("a") as fh:
            fh.write("\n# tampered\n")
        rc, text = run([str(tmp / "patch_T1T3.py"), str(ENTRY), "--name", "T1T3", "--out", str(a.out / "tamper.json")],
                       a.out / "tamper.log")
        r = {"case": "MUST-FAIL tampered fast_cotm.py", "exit": rc, "refused": rc != 0 and "refused" in text
             and "fast_cotm.py sha256" in text, "entry_never_ran": not (a.out / "tamper.json").exists()}
        r["pass"] = r["refused"] and r["entry_never_ran"]
    finally:
        shutil.rmtree(tmp)
    results.append(r)
    print(("PASS" if r["pass"] else "FAIL"), r, flush=True)
    # MUST-FAIL 2: a second patch in the same process is refused (the entry runs patch_R1.py after patch_T1T3.py)
    rc, text = run([str(HERE / "patch_T1T3.py"), str(ENTRY), "--name", "T1T3", "--out", str(a.out / "double.json"),
                    "--second-patch", str(HERE / "patch_R1.py")], a.out / "double.log")
    body = json.loads((a.out / "double.json").read_text()) if (a.out / "double.json").exists() else {}
    r = {"case": "MUST-FAIL second patch in one process", "exit": rc,
         "refused": body.get("checks", {}).get("second_patch_refused") is True}
    r["pass"] = r["refused"] and rc == 0
    results.append(r)
    print(("PASS" if r["pass"] else "FAIL"), r, flush=True)
    res = {"utc_start": utc0, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "test_sha256": sha(Path(__file__).resolve()), "entry_sha256": sha(ENTRY), "with_patch_sha256": sha(Path(WITH_PATCH)),
           "fast_cotm_sha256": fc_sha, "configs_sha256": sha(HERE / "configs.py"),
           "patch_sha256": {n: sha(HERE / f"patch_{n}.py") for n in NAMES}, "python_used": PY, "cases": results,
           "all_pass": all(x["pass"] for x in results), "tmp_removed": not tmp.exists()}
    res["all_pass"] = res["all_pass"] and res["tmp_removed"]
    (a.out / "test_patches.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'ALL PASS' if res['all_pass'] else 'FAIL'}: {sum(x['pass'] for x in results)} of {len(results)} -> "
          f"{a.out / 'test_patches.json'}", flush=True)
    raise SystemExit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
