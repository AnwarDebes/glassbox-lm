#!/usr/bin/env python
"""check_job_log.py: after one GPU benchmark job, show from the job's own log which code ran (README, GPU block).

    check_job_log.py --runner-out F --root ROOT --job B1 --impl orig|patch_T1T3|patch_R1|patch_T1T3_R1
                     [--expect-rc 0] [--sample-log PREFIX.sample.log] --out J.json

F is review/bench/run_gpu_job.sh's STDOUT only (`> X.runner.out 2> X.runner.err`). run_gpu_job.sh prints one stdout
line, as its last output, and only after its own check that the decision_layer record is untouched:
    <ROOT>/logs/<JOB>_<IMPL>_<YYYYmmddTHHMMSSZ>.log rc=<rc> tag=sb_<IMPL>_<tag>
The job log is taken from that line, never from a glob (B1_patch_T1T3_*.log also matches B1_patch_T1T3_R1_*.log).

Checks (each one a named entry of J.json; PASS needs every one):
  runner_line            the last line of F has the form above
  rc                     rc = --expect-rc (0 for a real GPU run; a CPU dry run stops at its first CUDA call with 1)
  tag                    tag = sb_<IMPL>_<tag of JOB in review/bench/jobs.tsv>
  log_name               the log is ROOT/logs/<JOB>_<IMPL>_<timestamp>.log, exactly (patch_T1T3 never takes a
                         patch_T1T3_R1 log)
  log_exists
  log_exit_line          the log holds run_gpu_job.sh's "<UTC> exit <expected rc>" line
  log_untouched_line     the log ends with run_gpu_job.sh's "the decision_layer record untouched: ... no row or model with tag
                         <tag> there"
  orig only:
    no_patch_line        no line of the log starts with "speed_patch"
  a patch only:
    one_import_line      exactly one "speed_patch <CONFIG>: dl.tm_init.CoTM -> FastCoTM switches {...}; fast_cotm.py
                         sha256 H1; configs.py sha256 H2; <file>.py sha256 H3"
    pin_fast_cotm, pin_configs, pin_patch   H1, H2, H3 = the pins below (the sha256 in README's file table)
    import_patch_file    <file>.py = <IMPL>.py
    import_switches      its switches = configs.py's <CONFIG>
    one_construction_line   exactly one "speed_patch <CONFIG>: make_tm -> FastCoTM(n_classes=..., n_features=...,
                         device=..., cfg=TMConfig(...)) switches {...}"
    construction_after_import
    construction_switches   its switches = configs.py's <CONFIG>
    device_cuda          device='cuda'
    plain_path           max_literals <= 0, max_negated <= 0 and clause_drop <= 0 in the printed TMConfig (FastCoTM's
                         own path; any other setting falls back to CoTM.feedback, so the candidate would not run)
    no_other_patch_line  no other line of the log starts with "speed_patch"
    files_still_pinned   fast_cotm.py, configs.py and <IMPL>.py in --files-dir (default: this folder) hash to the pins
  with --sample-log only:
    sample_keyed_same_log   bg2_sample2.sh's single "line found in <path>" names this same log
    sample_done          the sample log has bg2_sample2.sh's "done:" line
Prints the verdict and the matched lines (they go into the LEDGER row) and writes J.json (an existing file is refused).
Reads only, writes only J.json. Exit 0 PASS, 1 FAIL, 2 usage or refused.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = Path(f"{_REPO_ROOT}/studies/decision_layer/speed/review/bench/jobs.tsv")
PINS = {
    "fast_cotm.py": "a72eb8b849c927464287d92b1d445ceca97c9e0b5aee24aa8d1a71e7e3a307b8",
    "configs.py": "a7205c918060f3812bd29052e2128551dda0597058a6b49ca266da13977ee5a0",
}
PATCH_PINS = {
    "patch_T1T3": "ca30139a2a161560ce06d358dff0defef8489e05d768569d198ee0ea419ffb92",
    "patch_R1": "f6d3e4891b936bef06914fabc96dc24e2fc4d0be6f9cd3f0f4121b76861d642a",
    "patch_T1T3_R1": "f915803b000a59e0a58196f1b00ff79f09f787ffcfc715f967e55460f12c6f10",
}
IMPLS = ("orig", *PATCH_PINS)

RUNNER_RX = re.compile(r"^(?P<log>/\S+\.log) rc=(?P<rc>-?\d+) tag=(?P<tag>\S+)$")
IMPORT_RX = re.compile(r"^speed_patch (?P<name>\w+): dl\.tm_init\.CoTM -> FastCoTM switches (?P<sw>\{[^{}]*\}); "
                       r"fast_cotm\.py sha256 (?P<h1>[0-9a-f]{64}); configs\.py sha256 (?P<h2>[0-9a-f]{64}); "
                       r"(?P<pf>\w+\.py) sha256 (?P<h3>[0-9a-f]{64})$")
CONSTRUCT_RX = re.compile(r"^speed_patch (?P<name>\w+): make_tm -> FastCoTM\(n_classes=(?P<nc>\d+), "
                          r"n_features=(?P<nf>\d+), device=(?P<dev>'[^']*'), cfg=TMConfig\((?P<cfg>[^()]*)\)\) "
                          r"switches (?P<sw>\{[^{}]*\})$")
SCOPE_KEYS = ("n_clauses", "T", "s", "n_states", "epochs", "batch_size", "neg_sampling", "negation", "max_literals",
              "max_negated", "clause_drop", "half", "seed")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def job_tag(job: str) -> tuple[str, str]:
    for line in JOBS.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if f[0] == job:
            return f[1], f[2]
    raise SystemExit(f"check_job_log: unknown job {job} in {JOBS}")


def load_config(files_dir: Path, name: str) -> dict:
    p = files_dir / "configs.py"
    if sha(p) != PINS["configs.py"]:
        raise SystemExit(f"check_job_log: refused: {p} does not hash to the configs.py pin")
    spec = importlib.util.spec_from_file_location("speed_p2tm_configs_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, name))


def parse_cfg(inner: str) -> dict:
    out = {}
    for k, v in re.findall(r"(\w+)=('[^']*'|[^,]+)", inner):
        out[k] = ast.literal_eval(v.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runner-out", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--job", required=True)
    ap.add_argument("--impl", required=True, choices=IMPLS)
    ap.add_argument("--expect-rc", type=int, default=0)
    ap.add_argument("--sample-log", type=Path)
    ap.add_argument("--files-dir", type=Path, default=HERE, help="where fast_cotm.py, configs.py and the patch are "
                    "re-hashed (default: this folder; the self-test points it at a tampered copy)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists():
        print(f"check_job_log: refused: {a.out} exists", file=sys.stderr)
        return 2
    tag, seed = job_tag(a.job)
    btag = f"sb_{a.impl}_{tag}"
    chk: dict[str, bool] = {}
    rep: dict = {"job": a.job, "impl": a.impl, "expect_rc": a.expect_rc, "expected_tag": btag, "seed": seed}

    lines = a.runner_out.read_text(errors="replace").splitlines() if a.runner_out.is_file() else []
    last = lines[-1] if lines else ""
    rep["runner_out"] = str(a.runner_out.resolve())
    rep["runner_last_line"] = last
    m = RUNNER_RX.match(last)
    chk["runner_line"] = m is not None
    log = Path(m["log"]) if m else None
    chk["rc"] = bool(m) and int(m["rc"]) == a.expect_rc
    chk["tag"] = bool(m) and m["tag"] == btag
    name_rx = re.compile(rf"^{re.escape(a.job)}_{re.escape(a.impl)}_\d{{8}}T\d{{6}}Z\.log$")
    chk["log_name"] = bool(m) and os.path.realpath(log.parent) == os.path.realpath(a.root / "logs") and \
        name_rx.match(log.name) is not None
    chk["log_exists"] = bool(m) and log.is_file()
    text = log.read_text(errors="replace").splitlines() if chk["log_exists"] else []
    rep["log"] = str(log) if log else None
    rep["log_sha256"] = sha(log) if chk["log_exists"] else None
    exit_rx = re.compile(rf"^\d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\dZ exit {a.expect_rc}$")
    exit_lines = [x for x in text if exit_rx.match(x)]
    chk["log_exit_line"] = len(exit_lines) == 1
    tail = next((x for x in reversed(text) if x.strip()), "")
    chk["log_untouched_line"] = tail.startswith("the decision_layer record untouched: reference model ") and \
        tail.endswith(f"no row or model with tag {btag} there")
    rep["exit_line"] = exit_lines[0] if exit_lines else None
    rep["untouched_line"] = tail
    rep["view_line"] = next((x for x in text if re.match(r"^view \d+ bits", x)), None)
    patch_lines = [(i, x) for i, x in enumerate(text) if x.startswith("speed_patch")]

    if a.impl == "orig":
        chk["no_patch_line"] = not patch_lines
        rep["patch_lines"] = [x for _, x in patch_lines]
    else:
        cname = a.impl[len("patch_"):]
        want = load_config(a.files_dir, cname)
        rep["expected_switches"] = want
        imps = [(i, IMPORT_RX.match(x)) for i, x in patch_lines if IMPORT_RX.match(x)]
        imps = [(i, mm) for i, mm in imps if mm["name"] == cname]
        cons = [(i, CONSTRUCT_RX.match(x)) for i, x in patch_lines if CONSTRUCT_RX.match(x)]
        cons = [(i, mm) for i, mm in cons if mm["name"] == cname]
        chk["one_import_line"] = len(imps) == 1
        chk["one_construction_line"] = len(cons) == 1
        im = imps[0][1] if len(imps) == 1 else None
        co = cons[0][1] if len(cons) == 1 else None
        chk["pin_fast_cotm"] = bool(im) and im["h1"] == PINS["fast_cotm.py"]
        chk["pin_configs"] = bool(im) and im["h2"] == PINS["configs.py"]
        chk["pin_patch"] = bool(im) and im["h3"] == PATCH_PINS[a.impl]
        chk["import_patch_file"] = bool(im) and im["pf"] == f"{a.impl}.py"
        chk["import_switches"] = bool(im) and ast.literal_eval(im["sw"]) == want
        chk["construction_after_import"] = bool(im and co) and imps[0][0] < cons[0][0]
        chk["construction_switches"] = bool(co) and ast.literal_eval(co["sw"]) == want
        chk["device_cuda"] = bool(co) and co["dev"] == "'cuda'"
        cfg = parse_cfg(co["cfg"]) if co else {}
        chk["plain_path"] = bool(co) and cfg.get("max_literals", 1) <= 0 and cfg.get("max_negated", 1) <= 0 and \
            cfg.get("clause_drop", 1.0) <= 0
        known = {i for i, _ in imps} | {i for i, _ in cons}
        others = [x for i, x in patch_lines if i not in known]
        chk["no_other_patch_line"] = not others
        files = {"fast_cotm.py": PINS["fast_cotm.py"], "configs.py": PINS["configs.py"], f"{a.impl}.py": PATCH_PINS[a.impl]}
        now = {f: (sha(a.files_dir / f) if (a.files_dir / f).is_file() else None) for f in files}
        chk["files_still_pinned"] = all(now[f] == files[f] for f in files)
        rep["import_line"] = text[imps[0][0]] if im else None
        rep["construction_line"] = text[cons[0][0]] if co else None
        rep["other_patch_lines"] = others
        rep["scope"] = {"n_classes": int(co["nc"]) if co else None, "n_features": int(co["nf"]) if co else None,
                        **{k: cfg.get(k) for k in SCOPE_KEYS}}
        rep["pins"] = files
        rep["files_now"] = now

    if a.sample_log is not None:
        st = a.sample_log.read_text(errors="replace").splitlines() if a.sample_log.is_file() else []
        keyed = [mm["p"] for x in st if (mm := re.search(r" line found in (?P<p>\S+) after about ", x))]
        chk["sample_keyed_same_log"] = len(keyed) == 1 and bool(m) and \
            os.path.realpath(keyed[0]) == os.path.realpath(log)
        chk["sample_done"] = any(re.search(r" done: \S+\.sample\.json$", x) for x in st)
        rep["sample_log"] = str(a.sample_log.resolve())
        rep["sample_keyed_log"] = keyed

    failed = [k for k, v in chk.items() if not v]
    verdict = "PASS" if not failed else "FAIL"
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "verdict": verdict, "failed": failed,
           "checks": chk, **rep, "checker_sha256": sha(Path(__file__).resolve())}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"check_job_log: {verdict} {a.job} {a.impl} (expect rc {a.expect_rc}); {len(chk) - len(failed)} of {len(chk)} "
          f"checks hold{'; failed: ' + ', '.join(failed) if failed else ''}")
    print(f"  runner line: {last}")
    for k in ("import_line", "construction_line", "view_line", "exit_line", "untouched_line"):
        if rep.get(k):
            print(f"  {k}: {rep[k]}")
    if a.sample_log is not None:
        print(f"  sample keyed on: {rep['sample_keyed_log']}")
    print(f"  -> {a.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
