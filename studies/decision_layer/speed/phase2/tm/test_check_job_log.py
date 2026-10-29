#!/usr/bin/env python
"""test_check_job_log.py OUTDIR: self-test of check_job_log.py. No GPU, no torch; OUTDIR must not exist and is the only
place written (fixtures, one JSON per case, test_check_job_log.json).

    $E nice -n 19 $P -I -B $TM/test_check_job_log.py $TM/fix1/smokes/test_check_job_log

Part 1, REAL logs of run_gpu_job.sh --dry-cpu (session 1: roots/dry1; the review: review_rerun/root_dry). The runner's
stdout line is copied out of the saved runner outputs into a one-line file, as `> X.runner.out` gives it. A dry run stops
at its first CUDA call (rc 1), so these cases use --expect-rc 1:
  - B1 with patch_T1T3, patch_R1, patch_T1T3_R1 (both roots): must PASS (import line with the 3 pinned hashes,
    construction line, plain path, device 'cuda', the runner's own exit and untouched lines).
  B2's dry logs are not used: B2's dry run stops in its view build, before make_tm, so it has no construction line
  (the planted case no_construction covers that check), and copying its runner line would write the decision_layer encoder name,
  which is part of B2's tag, into this folder.
Part 2, SYNTHETIC GPU-like logs: a real B1 dry log of each configuration with the traceback replaced by training lines and
'exit 0', plus an orig log (the T1T3 one without its speed_patch lines). Each must PASS with --expect-rc 0. Then one planted
defect per case, each must FAIL on the named check (and the planted case must fail nothing it should not need to):
  a wrong fast_cotm.py / configs.py / patch hash in the import line; another patch file name; the T1T3 check pointed at a
  T1T3_R1 log (the glob pitfall); no construction line; construction or import switches off by one switch; two import
  lines; a speed_patch line of another configuration; a non-plain TMConfig; the construction line before the import line;
  device 'cpu'; runner rc 1 where 0 is expected; the runner's last line replaced by a refusal message; an empty runner
  output; another job's tag; the log outside ROOT/logs; no exit line; a FAILURE tail instead of the untouched line; a
  missing log; an orig log with a patch line; a tampered fast_cotm.py copy (files_still_pinned, through --files-dir);
  a sample log keyed on another log (and one keyed on the right log: PASS). Plus: an existing --out is refused (exit 2).
Exit 0 when every case behaves as required.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
CHECK = HERE / "check_job_log.py"
REAL = [  # (saved runner output, root, job, impl, must pass)
    (HERE / "smokes/S6c_dry_B1_patch_T1T3.out", HERE / "roots/dry1", "B1", "patch_T1T3", True),
    (HERE / "smokes/bg2probe_R1.runner.txt", HERE / "roots/dry1", "B1", "patch_R1", True),
    (HERE / "smokes/S6c_dry_B1_patch_T1T3_R1.out", HERE / "roots/dry1", "B1", "patch_T1T3_R1", True),
    (HERE / "review_rerun/Rh_dry_B1_patch_T1T3.out", HERE / "review_rerun/root_dry", "B1", "patch_T1T3", True),
    (HERE / "review_rerun/bg2probe_R1.runner.txt", HERE / "review_rerun/root_dry", "B1", "patch_R1", True),
    (HERE / "review_rerun/Rl_dry_B1_patch_T1T3_R1.out", HERE / "review_rerun/root_dry", "B1", "patch_T1T3_R1", True),
]
B1_TAG = "e74c_abcd_Bc4_c4000_s16"
REF_LINE = ("the decision_layer record untouched: reference model e74c_abcd_Bc4_c4000_s16_seed1.npz "
            "a11db58f5345e6dde6252ff85ca92e39f0c415c34ba86ce3d70fae9627c08f1e; no row or model with tag {btag} there")


def runner_line(saved: Path) -> str:
    import re
    hits = [x for x in saved.read_text().splitlines() if re.match(r"^/\S+\.log rc=-?\d+ tag=\S+$", x)]
    if len(hits) != 1:
        raise SystemExit(f"test: {saved} holds {len(hits)} runner lines")
    return hits[0]


def check(out: Path, name: str, runner_out: Path, root: Path, job: str, impl: str, rc: int = 0,
          extra: tuple = ()) -> tuple[int, dict, str]:
    j = out / "cases" / f"{name}.json"
    p = subprocess.run([PY, "-I", "-B", str(CHECK), "--runner-out", str(runner_out), "--root", str(root), "--job", job,
                        "--impl", impl, "--expect-rc", str(rc), *extra, "--out", str(j)],
                       capture_output=True, text=True, check=False)
    (out / "cases" / f"{name}.txt").write_text(p.stdout + p.stderr)
    return p.returncode, (json.loads(j.read_text()) if j.exists() else {}), p.stdout + p.stderr


def gpu_like(dry_log: Path, impl: str) -> list[str]:
    """A real B1 dry log as a GPU run would end: the traceback goes, training lines and 'exit 0' come."""
    keep = []
    for x in dry_log.read_text().splitlines():
        if x.startswith("Traceback"):
            break
        keep.append(x)
    btag = f"sb_{impl}_{B1_TAG}"
    return keep + ["epoch 2 val_acc 0.81 (synthetic line of test_check_job_log.py)",
                   '{"tag": "' + btag + '", "arm": "tm", "note": "synthetic row line"}',
                   "2026-10-21T09:00:00Z exit 0", REF_LINE.format(btag=btag)]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    (out / "cases").mkdir(parents=True)
    (out / "runner").mkdir()
    results = []

    def record(name, rc, res, want_rc, want_failed=None, note=""):
        failed = res.get("failed")
        ok = rc == want_rc and (want_failed is None or (failed is not None and want_failed in failed))
        if want_rc == 0:
            ok = ok and failed == []
        results.append({"case": name, "pass": ok, "exit": rc, "want_exit": want_rc, "failed_checks": failed,
                        "must_fail_on": want_failed, "note": note})
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit {rc} (want {want_rc}); failed checks {failed}"
              f"{' (must include ' + want_failed + ')' if want_failed else ''}", flush=True)

    # ---- part 1: real dry logs
    for k, (saved, root, job, impl, must_pass) in enumerate(REAL, 1):
        ro = out / "runner" / f"real{k}.runner.out"
        ro.write_text(runner_line(saved) + "\n")
        name = f"real{k}_{root.parent.name}_{root.name}_{job}_{impl}"
        rc, res, _ = check(out, name, ro, root, job, impl, rc=1)
        if must_pass:
            record(name, rc, res, 0, note=f"real dry log via {saved.name}")
        else:
            record(name, rc, res, 1, "one_construction_line", note=f"real dry log via {saved.name}")

    # ---- part 2: synthetic GPU-like logs in a fake root
    froot = out / "root_fake"
    (froot / "logs").mkdir(parents=True)
    dry = {"patch_T1T3": HERE / "roots/dry1/logs/B1_patch_T1T3_20261021T063609Z.log",
           "patch_R1": HERE / "roots/dry1/logs/B1_patch_R1_20261021T063723Z.log",
           "patch_T1T3_R1": HERE / "roots/dry1/logs/B1_patch_T1T3_R1_20261021T063810Z.log"}
    base = {impl: gpu_like(p, impl) for impl, p in dry.items()}
    orig = [x.replace("sb_patch_T1T3_", "sb_orig_").replace("DRY-CPU B1 patch_T1T3", "start B1 orig")
            for x in base["patch_T1T3"] if not x.startswith("speed_patch")]
    base["orig"] = orig
    stamp = {"patch_T1T3": "20261021T090000Z", "patch_R1": "20261021T090500Z", "patch_T1T3_R1": "20261021T091000Z",
             "orig": "20261021T085500Z"}

    def make(case: str, impl: str, lines: list[str], rc: int = 0, job: str = "B1", tag_job_tag: str = B1_TAG,
             last_line: str | None = None, logdir: Path | None = None) -> tuple[Path, Path]:
        d = (logdir or (froot / "logs"))
        d.mkdir(parents=True, exist_ok=True)
        log = d / f"{job}_{impl}_{stamp[impl][:-3]}{len(results) % 100:02d}Z.log"
        log.write_text("\n".join(lines) + "\n")
        ro = out / "runner" / f"{case}.runner.out"
        line = f"{log} rc={rc} tag=sb_{impl}_{tag_job_tag}" if last_line is None else last_line
        ro.write_text(line + "\n" if line else "")
        return log, ro

    for impl in ("patch_T1T3", "patch_R1", "patch_T1T3_R1", "orig"):
        _, ro = make(f"gpu_{impl}", impl, base[impl])
        rc, res, _ = check(out, f"gpu_{impl}", ro, froot, "B1", impl)
        record(f"gpu_{impl}", rc, res, 0, note="synthetic GPU-like log")

    t = base["patch_T1T3"]
    imp_i = next(i for i, x in enumerate(t) if x.startswith("speed_patch T1T3: dl.tm_init"))
    con_i = next(i for i, x in enumerate(t) if x.startswith("speed_patch T1T3: make_tm"))

    def flip(s: str, h: str) -> str:
        i = s.index(h)
        c = "0" if s[i] != "0" else "1"
        return s[:i] + c + s[i + 1:]

    def edited(i: int, new: str) -> list[str]:
        x = list(t)
        x[i] = new
        return x

    fc, cf, pt = ("a72eb8b849c927464287d92b1d445ceca97c9e0b5aee24aa8d1a71e7e3a307b8",
                  "a7205c918060f3812bd29052e2128551dda0597058a6b49ca266da13977ee5a0",
                  "ca30139a2a161560ce06d358dff0defef8489e05d768569d198ee0ea419ffb92")
    planted = [
        ("pin_fast_cotm", edited(imp_i, flip(t[imp_i], fc)), "pin_fast_cotm"),
        ("pin_configs", edited(imp_i, flip(t[imp_i], cf)), "pin_configs"),
        ("pin_patch", edited(imp_i, flip(t[imp_i], pt)), "pin_patch"),
        ("patch_file_name", edited(imp_i, t[imp_i].replace("patch_T1T3.py sha256", "patch_T1T3_R1.py sha256")),
         "import_patch_file"),
        ("no_construction", [x for i, x in enumerate(t) if i != con_i], "one_construction_line"),
        ("construction_switch", edited(con_i, t[con_i].replace("'pace': 0", "'pace': 1")), "construction_switches"),
        ("import_switch", edited(imp_i, t[imp_i].replace("'elem': True", "'elem': False")), "import_switches"),
        ("two_imports", t[:imp_i + 1] + [t[imp_i]] + t[imp_i + 1:], "one_import_line"),
        ("other_config_line", t[:con_i + 1] + ["speed_patch R1: make_tm -> FastCoTM(planted line)"] + t[con_i + 1:],
         "no_other_patch_line"),
        ("not_plain", edited(con_i, t[con_i].replace("max_literals=0", "max_literals=8")), "plain_path"),
        ("construction_first", [t[con_i] if i == imp_i else t[imp_i] if i == con_i else x for i, x in enumerate(t)],
         "construction_after_import"),
        ("device_cpu", edited(con_i, t[con_i].replace("device='cuda'", "device='cpu'")), "device_cuda"),
        ("no_exit_line", [x for x in t if x != "2026-10-21T09:00:00Z exit 0"], "log_exit_line"),
        ("failure_tail", t[:-1] + ["FAILURE: the decision_layer record touched (planted)"], "log_untouched_line"),
    ]
    for case, lines, want in planted:
        _, ro = make(case, "patch_T1T3", lines)
        rc, res, _ = check(out, case, ro, froot, "B1", "patch_T1T3")
        record(case, rc, res, 1, want, note="planted")

    # the glob pitfall: the T1T3 check given the runner line of a T1T3_R1 job
    log_r1, ro_r1 = make("glob_pitfall_src", "patch_T1T3_R1", base["patch_T1T3_R1"])
    rc, res, _ = check(out, "glob_pitfall", ro_r1, froot, "B1", "patch_T1T3")
    record("glob_pitfall", rc, res, 1, "log_name", note="a T1T3_R1 log offered to the T1T3 check")
    # runner rc 1 where 0 is expected
    _, ro = make("rc_mismatch", "patch_T1T3", t, rc=1)
    rc, res, _ = check(out, "rc_mismatch", ro, froot, "B1", "patch_T1T3")
    record("rc_mismatch", rc, res, 1, "rc", note="planted")
    # last stdout line is not the runner line (refusal printed last), and an empty runner output
    _, ro = make("refusal_last", "patch_T1T3", t, last_line="refused: no go marker (planted)")
    rc, res, _ = check(out, "refusal_last", ro, froot, "B1", "patch_T1T3")
    record("refusal_last", rc, res, 1, "runner_line", note="planted")
    _, ro = make("empty_runner", "patch_T1T3", t, last_line="")
    rc, res, _ = check(out, "empty_runner", ro, froot, "B1", "patch_T1T3")
    record("empty_runner", rc, res, 1, "runner_line", note="planted")
    # another job's tag
    _, ro = make("tag_mismatch", "patch_T1T3", t, tag_job_tag="e74c_star_Bc4_c16000_s16")
    rc, res, _ = check(out, "tag_mismatch", ro, froot, "B1", "patch_T1T3")
    record("tag_mismatch", rc, res, 1, "tag", note="planted")
    # the log outside ROOT/logs
    _, ro = make("outside_root", "patch_T1T3", t, logdir=out / "elsewhere" / "logs")
    rc, res, _ = check(out, "outside_root", ro, froot, "B1", "patch_T1T3")
    record("outside_root", rc, res, 1, "log_name", note="planted")
    # a missing log
    ro = out / "runner" / "missing_log.runner.out"
    ro.write_text(f"{froot / 'logs' / 'B1_patch_T1T3_20261021T099999Z.log'} rc=0 tag=sb_patch_T1T3_{B1_TAG}\n")
    rc, res, _ = check(out, "missing_log", ro, froot, "B1", "patch_T1T3")
    record("missing_log", rc, res, 1, "log_exists", note="planted")
    # an orig log with a patch line
    o = base["orig"]
    vi = next(i for i, x in enumerate(o) if x.startswith("view "))
    _, ro = make("orig_with_patch_line", "orig", o[:vi] + [t[imp_i]] + o[vi:])
    rc, res, _ = check(out, "orig_with_patch_line", ro, froot, "B1", "orig")
    record("orig_with_patch_line", rc, res, 1, "no_patch_line", note="planted")
    # a tampered fast_cotm.py copy (files_still_pinned) through --files-dir
    fdir = out / "files_tampered"
    fdir.mkdir()
    for f in ("fast_cotm.py", "configs.py", "patch_T1T3.py"):
        shutil.copyfile(HERE / f, fdir / f)
    with (fdir / "fast_cotm.py").open("a") as fh:
        fh.write("# planted change\n")
    _, ro = make("tampered_copy", "patch_T1T3", t)
    rc, res, _ = check(out, "tampered_copy", ro, froot, "B1", "patch_T1T3", extra=("--files-dir", str(fdir)))
    record("tampered_copy", rc, res, 1, "files_still_pinned", note="planted")
    # sample logs: keyed on the right log (PASS) and on another log (FAIL)
    log_ok, ro_ok = make("sample_src", "patch_T1T3", t)
    good = out / "runner" / "good.sample.log"
    good.write_text(f"2026-10-21T09:00:01Z start: (synthetic)\n2026-10-21T09:00:30Z line found in {log_ok} after about 28 s; "
                    "settling 20 s\n2026-10-21T09:01:50Z done: /x/BG1_B1_T1T3.sample.json\n")
    rc, res, _ = check(out, "sample_same_log", ro_ok, froot, "B1", "patch_T1T3", extra=("--sample-log", str(good)))
    record("sample_same_log", rc, res, 0, note="sample keyed on the runner's own log")
    bad = out / "runner" / "bad.sample.log"
    bad.write_text(f"2026-10-21T09:00:01Z start: (synthetic)\n2026-10-21T09:00:30Z line found in {log_r1} after about 28 s; "
                   "settling 20 s\n2026-10-21T09:01:50Z done: /x/BG1_B1_T1T3.sample.json\n")
    rc, res, _ = check(out, "sample_other_log", ro_ok, froot, "B1", "patch_T1T3", extra=("--sample-log", str(bad)))
    record("sample_other_log", rc, res, 1, "sample_keyed_same_log", note="planted: sampler keyed on a T1T3_R1 log")
    # an existing --out is refused
    j = out / "cases" / "gpu_patch_T1T3.json"
    p = subprocess.run([PY, "-I", "-B", str(CHECK), "--runner-out", str(ro_ok), "--root", str(froot), "--job", "B1",
                        "--impl", "patch_T1T3", "--out", str(j)], capture_output=True, text=True, check=False)
    ok = p.returncode == 2 and "refused" in p.stderr
    results.append({"case": "existing_out_refused", "pass": ok, "exit": p.returncode, "want_exit": 2})
    print(f"{'PASS' if ok else 'FAIL'} existing_out_refused: exit {p.returncode} (want 2)", flush=True)

    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checker": str(CHECK),
           "checker_sha256": __import__("hashlib").sha256(CHECK.read_bytes()).hexdigest(),
           "n_cases": len(results), "n_pass": sum(r["pass"] for r in results),
           "all_pass": all(r["pass"] for r in results), "cases": results}
    (out / "test_check_job_log.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'ALL PASS' if res['all_pass'] else 'SOME FAIL'}: {res['n_pass']} of {res['n_cases']} cases", flush=True)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
