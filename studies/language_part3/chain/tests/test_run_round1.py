"""Self-tests of chain/run_round1.sh and chain/round1_lib.py (plain asserts, no pytest).

    cd studies/language_part3 && \\
      studies/tm_language_model/.venv-tm/bin/python -I -B chain/tests/test_run_round1.py

The chain runs in its test mode (ROUND1_TEST_DIR in chain/tmp/r1test/<case>/): a stand-in wrapper
(tests/fake_round1_wrapper.py) writes the outputs a finished step would leave, so no real job ever starts, and the
CPU gate reads lines from a file. The real hash check of the A and C trees runs in every case (read-only; a per-case
hash cache keeps it short). Covers the job order, every STOP route, the failure stops, restarts, the gate wait, the
C lock retry, the time boxes, the clean environment, the singleton lock and the dry mode.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CH = Path(__file__).resolve().parents[1]
TM = CH.parent
SELF = CH / "run_round1.sh"
PY = f"{_REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python"
BASE = CH / "tmp" / "r1test"
sys.path.insert(0, str(CH))
import round1_lib as L  # noqa: E402

PASSED = []
OPEN = "2026-10-21T08:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods"


def check(name, cond, info=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {str(info)[:3000]}")
    PASSED.append(name)
    print("ok", name, flush=True)


def new_case(name, scenario=None, gate_lines=None):
    T = BASE / name
    if T.exists():
        shutil.rmtree(T)
    T.mkdir(parents=True)
    (T / "fake_wrapper.sh").write_text(f'#!/bin/bash\nexec {PY} -I -B {CH}/tests/fake_round1_wrapper.py "$@"\n')
    (T / "LOG.md").write_text("# test LOG\n")
    (T / "scenario.json").write_text(json.dumps(scenario or {}))
    if gate_lines:
        (T / "gate_lines.txt").write_text("\n".join(gate_lines) + "\n")
    return T


def chain(T, *args, pins=None, cache=True, timeout=600):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROUND1_", "PART3_", "A_TEST", "A_JOBLOG", "A_RUNNER"))}
    env.update(ROUND1_TEST_DIR=str(T), ROUND1_TEST_SLEEP="0")
    if cache:
        env["ROUND1_TEST_HASHCACHE"] = str(T / "hashcache.json")
    if pins:
        env["ROUND1_TEST_PINS"] = str(pins)
    return subprocess.run(["bash", str(SELF), *map(str, args)], env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


def calls(T):
    p = T / "calls.txt"
    return p.read_text().splitlines() if p.exists() else []


def logl(T):
    p = T / "round1.log"
    return p.read_text().splitlines() if p.exists() else []


def has(T, pattern):
    return any(re.search(pattern, ln) for ln in logl(T))


LEARNERS = [f"A tm s2b_arm.py --arm {a} --family {f}" for a in ("full", "nf") for f in ("tree", "lr1", "lr2", "hgb")]
A_S = ["A tm s0_data.py", "A tm s1_floors.py", "A tm s2a_rows.py", "A tm s2b_twin.py", "A tm s2c_aa.py"]
C6 = ["C prep", "C twin", "C learners", "C gates"]


def k3c(arm, s=None):
    out = [f"A dl k3c_pilot.py --arm {arm} --s 15", f"A dl k3c_pilot.py --arm {arm} --s 40", f"A tm k3c_gate.py --arm {arm}"]
    if s is not None:
        out += [f"A dl k3c_pilot.py --arm {arm} --s {s} --shuffled", f"A tm k3c_gate.py --arm {arm}"]
    return out


PILOTS = {"full": {"15": 5.1, "40": 5.2}, "nf": {"15": 5.3, "40": 5.2}}
FULL_SC = {"pilots": PILOTS, "codes": {"A tm k3c_gate.py --arm full": [5, 0], "A tm k3c_gate.py --arm nf": [5, 0]}}
FULL = ["C step0", *A_S, *LEARNERS, "A tm s2d_gates.py", *k3c("full", 15), *k3c("nf", 40), *C6]


# ---- round1_lib units ------------------------------------------------------------------------------------------
def test_lib_units():
    for line, want in (("2026-10-21T07:00:00Z cores 4.00 over 10.0 s; throttled 0 of 100 periods", 0),
                       ("2026-10-21T07:00:00Z cores 4.01 over 10.0 s; throttled 0 of 100 periods", 1),
                       ("2026-10-21T07:00:00Z cores 1.00 over 10.0 s; throttled 1 of 100 periods", 1),
                       ("Traceback (most recent call last)", 1), ("", 1)):
        r = subprocess.run(["bash", str(TM / "A" / "gate_check.sh")], input=line + "\n", text=True, check=False)
        check(f"A/gate_check.sh on {line[21:45] or 'an empty line'!r}: {'open' if want == 0 else 'closed'}", r.returncode == want)
    T = new_case("lib")
    os.environ["ROUND1_TEST_DIR"] = str(T)
    try:
        P = L.paths()
        check("test paths", P["a_out"] == T / "A_out" and P["bitlist_log"] == T / "LOG.md")
        for arm, (a, b, want) in {"full": (5.0, 5.0, 15), "nf": (5.2, 5.1, 40)}.items():
            for s, v in ((15, a), (40, b)):
                d = P["a_out"] / f"k3c_{arm}_s{s}"
                d.mkdir(parents=True)
                (d / "done.json").write_text("{}")
                (d / "results.json").write_text(json.dumps({"tune_cal_bits": v}))
            check(f"better s {arm} (ties to 15)", L.better_s(arm, P) == want)
        check("status absent", L.status("A_s1")[0] == 10)
        (P["a_out"] / "s1.partial").mkdir()
        c, t = L.status("A_s1")
        check("status partial", c == 10 and "partial" in t, t)
        d = P["a_out"] / "s1"
        d.mkdir()
        (d / "done.json").write_text(json.dumps({"K1": True, "floor": "F1n4", "frozen_record_sha256": "0" * 64}))
        check("status: missing files -> inconsistent", L.status("A_s1")[0] == 1)
        for f in ("results.json", "summ_cal.npz", "summ_hold.npz"):
            (d / f).write_text(json.dumps({"floor": "F1n4", "K1": {"rel_ppl": 1e-5, "pass": True},
                                           "tune_cal_bits": {"F1n4": 5.0}, "tune_hold_bits": {"F1n4": 5.1}}))
        c, t = L.status("A_s1")
        check("status: another frozen record -> inconsistent", c == 1 and "frozen record" in t, t)
        pin = L.FILE_PINS["A"][0][1]
        (d / "done.json").write_text(json.dumps({"K1": False, "floor": "F1n4", "frozen_record_sha256": pin}))
        c, t = L.status("A_s1")
        check("status: K1 STOP -> 3", c == 3 and "K1 STOP" in t, t)
        (d / "done.json").write_text(json.dumps({"K1": True, "floor": "F1n4", "frozen_record_sha256": pin}))
        c, t = L.status("A_s1")
        check("status: K1 pass -> 0, no manifest yet", c == 0 and "no chain manifest yet" in t, t)
        check("manifest write", L.main(["manifest", "write", "A_s1"]) == 0 and (P["state"] / "A_s1.sha256").exists())
        c, t = L.status("A_s1")
        check("status with manifest ok", c == 0 and "chain manifest ok" in t, t)
        (d / "summ_cal.npz").write_text("changed")
        check("status: a file changed after the step -> inconsistent", L.status("A_s1")[0] == 1)
        # C: no run, one finished run, a changed output, a second finished run, the gates decision
        check("C status absent", L.status("C_prep")[0] == 10)
        out = P["c_data"] / "prep.npz"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"x")
        r = P["c_runs"] / "20261021T000000Z_prep"
        r.mkdir(parents=True)
        (r / "run.json").write_text(json.dumps({"state": "crashed"}))
        check("C status: a crashed run is not done", L.status("C_prep")[0] == 10)
        (r / "run.json").write_text(json.dumps({"state": "finished"}))
        (r / "outputs.json").write_text(json.dumps({str(out.relative_to(L.ROOT)): L._sha256_file(out)}))
        check("C status done", L.status("C_prep")[0] == 0)
        (P["c_runs"] / "20261021T000000Z_prep_dry").mkdir()
        check("C status ignores dry runs", L.status("C_prep")[0] == 0)
        out.write_bytes(b"y")
        check("C status: output changed -> inconsistent", L.status("C_prep")[0] == 1)
        out.write_bytes(b"x")
        r2 = P["c_runs"] / "20261021T000001Z_prep"
        shutil.copytree(r, r2)
        check("C status: two finished real runs -> inconsistent", L.status("C_prep")[0] == 1)
        dec = P["c_data"] / "step6_decision.json"
        g = P["c_runs"] / "20261021T000002Z_gates"
        g.mkdir()
        (g / "run.json").write_text(json.dumps({"state": "finished"}))
        for text, want in (("STOP at K0b: x", 3), ("PASS: K0 and K0b passed", 0), ("maybe", 1)):
            dec.write_text(json.dumps({"decision": text}))
            (g / "outputs.json").write_text(json.dumps({str(dec.relative_to(L.ROOT)): L._sha256_file(dec)}))
            check(f"C gates status {want}", L.status("C_gates")[0] == want)
        # C-K4: hours from the real stages only; the calendar deadline
        shutil.rmtree(g)
        dec.unlink()
        jl = P["c_runs"] / "jobs_log.tsv"
        jl.write_text("a\tb\tlearners\t--dry\t0\t300000\tno change\na\tb\tselftest_tm\t\t0\t300000\tno change\n"
                      "a\tb\ttwin\t\t0\t3600\tno change\n")
        c, res = L.ck4(L.parse_utc("2026-10-22T00:00:00Z"))
        check("C-K4 open (dry runs and self-tests not counted)", c == 0 and res["real_stage_hours_lower_bound"] == 1.0, res)
        c, res = L.ck4(L.parse_utc("2026-10-31T03:09:23Z"))
        check("C-K4 reached at the deadline", c == 3, res)
        jl.write_text("a\tb\tlearners\t\t0\t259200\tno change\n")
        c, res = L.ck4(L.parse_utc("2026-10-22T00:00:00Z"))
        check("C-K4 reached at 72 hours", c == 3, res)
    finally:
        os.environ.pop("ROUND1_TEST_DIR", None)


def test_hashcheck_real_tree():
    """The real A and C trees, without the test cache: every pin and frozen file matches; the four A scripts whose
    R1-line prefix is the first freeze's show as notes."""
    for arm, step in (("A", "A_s1"), ("C", "C_twin")):
        r = subprocess.run([PY, "-I", "-B", str(CH / "round1_lib.py"), "hashcheck", arm, step], capture_output=True,
                           text=True, check=False, env={"PATH": "/usr/bin:/bin"})
        check(f"hashcheck {arm} real tree", r.returncode == 0 and f"hashcheck {arm} {step}: OK" in r.stdout, r.stdout + r.stderr)
        if arm == "A":
            notes = [ln for ln in r.stdout.splitlines() if ln.startswith("NOTE") and "first freeze" in ln]
            check("hashcheck A: 4 superseded R1-line prefixes noted", len(notes) == 4
                  and all(any(n.startswith(f"NOTE {s}") for n in notes) for s in
                          ("s0_data.py", "s1_floors.py", "s2b_twin.py", "s2b_arm.py")), notes)


def test_hashcheck_synthetic():
    """A small fake tree: a changed file, a pin missing from its line, a wrong pin, a stale prefix with and without
    the first record's support, and a changed recorded-only file."""
    R = BASE / "hashroot"
    if R.exists():
        shutil.rmtree(R)
    A = R / "glassbox-lm" / "studies" / "language_part3" / "A"
    (A / "history").mkdir(parents=True)
    (R / "glassbox-lm" / "studies" / "language_part3" / "chain").mkdir(parents=True)
    files = {"glassbox-lm/studies/language_part3/A/s0_data.py": "print(0)\n", "glassbox-lm/studies/language_part3/A/s1_floors.py": "print(1)\n",
             "glassbox-lm/studies/language_part3/A/run_job.sh": "true\n", "glassbox-lm/studies/language_part3/A/env.sh": "\n",
             "glassbox-lm/studies/language_part3/A/listing.py": "\n#l\n", "glassbox-lm/studies/language_part3/A/gate_check.sh": "\n#g\n",
             "glassbox-lm/studies/language_part3/A/lock_held.sh": "\n#h\n", "glassbox-lm/studies/language_part3/A/k7_clock.py": "\n#k\n",
             "glassbox-lm/studies/language_part3/chain/part3_cpu_job.sh": "w\n", "glassbox-lm/studies/language_part3/chain/part3_status.py": "s\n",
             "glassbox-lm/studies/language_part3/A/notes.md": "n\n"}
    for rel, txt in files.items():
        (R / rel).write_text(txt)
    h = {rel: L._sha256_file(R / rel) for rel in files}
    enforced = {k: v for k, v in h.items() if "/A/" in k and not k.endswith("notes.md")}
    v1 = dict(enforced, **{"glassbox-lm/studies/language_part3/A/s1_floors.py": "ab" * 32})
    (A / "history" / "R1_A-Struct.frozen.v1.json").write_text(json.dumps({"enforced": v1}))
    (A / "R1_A-Struct.frozen.json").write_text(json.dumps({"enforced": enforced,
                                                           "recorded": {"glassbox-lm/studies/language_part3/A/notes.md": h["glassbox-lm/studies/language_part3/A/notes.md"]}}))
    rec_h = L._sha256_file(A / "R1_A-Struct.frozen.json")
    v1_h = L._sha256_file(A / "history" / "R1_A-Struct.frozen.v1.json")
    fp = {"A": [("glassbox-lm/studies/language_part3/A/R1_A-Struct.frozen.json", rec_h, "A1"),
                ("glassbox-lm/studies/language_part3/A/history/R1_A-Struct.frozen.v1.json", v1_h, "A")],
          "C": [], "chain": [("glassbox-lm/studies/language_part3/chain/part3_cpu_job.sh", h["glassbox-lm/studies/language_part3/chain/part3_cpu_job.sh"], "A1")]}
    p0 = h["glassbox-lm/studies/language_part3/A/s0_data.py"][:16]
    cp = {"A": [(f"s0_data {p0}", "glassbox-lm/studies/language_part3/A/s0_data.py", "A1"),
                ("s1_floors abababababababab", "glassbox-lm/studies/language_part3/A/s1_floors.py", "A")], "C": []}

    def write_log(extra_a1=""):
        log = R / "LOG.md"
        log.write_text("# log\n"
                       f"{L.R1_LINE_HEADS['C']} (nothing here)\n"
                       f"{L.R1_LINE_HEADS['A']} record {v1_h}; s1_floors abababababababab\n"
                       f"{L.R1_LINE_HEADS['A1']} record {rec_h}; wrapper {h['glassbox-lm/studies/language_part3/chain/part3_cpu_job.sh']}; "
                       f"s0_data {p0}{extra_a1}\n")
        return log

    log = write_log()
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp, code_prefixes=cp)
    check("synthetic: ok", ok, rep)
    check("synthetic: stale prefix backed by the first record is a note",
          any(x.startswith("NOTE s1_floors.py") for x in rep), rep)
    (R / "glassbox-lm/studies/language_part3/A/notes.md").write_text("changed\n")
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp, code_prefixes=cp)
    check("synthetic: a changed recorded-only file is a note", ok and any("recorded, not enforced" in x for x in rep), rep)
    (R / "glassbox-lm/studies/language_part3/A/s0_data.py").write_text("print('changed')\n")
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp, code_prefixes=cp)
    check("synthetic: a changed enforced file fails", not ok and any("changed since the freeze" in x for x in rep), rep)
    (R / "glassbox-lm/studies/language_part3/A/s0_data.py").write_text(files["glassbox-lm/studies/language_part3/A/s0_data.py"])
    log.write_text(log.read_text().replace(rec_h, "f" * 64))
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp, code_prefixes=cp)
    check("synthetic: a pin missing from its R1 line fails", not ok and any("is not in the R1 line" in x for x in rep), rep)
    log = write_log()
    bad = {**fp, "A": [("glassbox-lm/studies/language_part3/A/R1_A-Struct.frozen.json", "e" * 64, "A1")] + fp["A"][1:]}
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=bad, code_prefixes=cp)
    check("synthetic: a wrong pin fails", not ok, rep)
    cp2 = {"A": [("s0_data 0000000000000000", "glassbox-lm/studies/language_part3/A/s0_data.py", "A1")], "C": []}
    log = write_log(extra_a1="; s0_data 0000000000000000")
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp, code_prefixes=cp2)
    check("synthetic: a prefix of the amendment line that does not match fails", not ok, rep)
    v1_bad = dict(enforced)
    (A / "history" / "R1_A-Struct.frozen.v1.json").write_text(json.dumps({"enforced": v1_bad}))
    fp2 = {**fp, "A": [fp["A"][0], ("glassbox-lm/studies/language_part3/A/history/R1_A-Struct.frozen.v1.json",
                                    L._sha256_file(A / "history" / "R1_A-Struct.frozen.v1.json"), "A")]}
    log = R / "LOG.md"
    log.write_text(log.read_text().replace(v1_h, fp2["A"][1][1]))
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp2, code_prefixes=cp)
    check("synthetic: a stale prefix the first record does not back fails", not ok
          and any("does not start with the R1-line prefix" in x for x in rep), rep)
    log.write_text("# no R1 lines\n")
    ok, rep = L.hashcheck("A", "A_s0", root=R, logmd=log, file_pins=fp, code_prefixes=cp)
    check("synthetic: missing R1 lines fail", not ok and any("exactly one line" in x for x in rep), rep)


# ---- the chain in test mode --------------------------------------------------------------------------------------
def start_end_pairs(T):
    starts = [ln.split()[2].rstrip(":") for ln in logl(T) if ln.split()[1:2] == ["start"]]
    ends = [ln.split()[2].rstrip(":") for ln in logl(T) if ln.split()[1:2] == ["end"]]
    return starts, ends


def test_full_pass_restart_tamper():
    T = new_case("full", FULL_SC | {"codes": {**FULL_SC["codes"], "C prep": [75, 0]}},
                 gate_lines=["2026-10-21T08:00:00Z cores 4.50 over 10.0 s; throttled 0 of 100 periods",
                             "2026-10-21T08:02:00Z cores 3.00 over 10.0 s; throttled 2 of 100 periods"])
    t0 = time.time()
    r = chain(T)
    dt = time.time() - t0
    want = FULL[:FULL.index("C prep")] + ["C prep"] + FULL[FULL.index("C prep"):]
    check("full: exit 0", r.returncode == 0, r.stdout + r.stderr + "\n".join(logl(T)[-30:]))
    check("full: job order (plan, section 6; C prep retried once after exit 75)", calls(T) == want,
          "\n".join(calls(T)))
    lines = logl(T)
    check("full: every log line starts with a UTC time", all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", ln) for ln in lines))
    check("full: the gate waited twice before the first job", sum("gate closed" in ln and "before C_step0" in ln
                                                                  for ln in lines) == 2)
    check("full: a gate reading before every job", sum("gate OPEN" in ln for ln in lines) == len(want))
    check("full: the retry on C's busy lock is logged", has(T, r"retry C_prep in 0 s: C's own lock was busy"))
    starts, ends = start_end_pairs(T)
    check("full: every start has its end", starts == ends and len(starts) == len(want) + 1, (starts, ends))
    check("full: wrapper exit codes logged", has(T, r"end A_k3c_gate_full: wrapper exit 5") and has(T, r"end C_gates: wrapper exit 0"))
    check("full: hash check before every step", sum(re.search(r"hashcheck \S+: OK", ln) is not None for ln in lines)
          == len(set(want)) + 1 + 2)  # each step once, the bit-list step, the second K3c gate of each arm
    check("full: decisions logged", has(T, r"done A_s1: s1: K1 pass") and has(T, r"done A_s2c_aa: s2c_aa: K8")
          and has(T, r"A-full: K3c and K9 \(TM\) pass") and has(T, r"C: K0 and K0b pass"))
    check("full: time boxes logged", has(T, r"timebox A_s0: K7 \{") and has(T, r"timebox C_prep: C-K4 \{"))
    check("full: chain end line", lines[-1].split(" ", 1)[1].startswith("chain end: A: A-full: K3c pass")
          and "C step 6: PASS" in lines[-1], lines[-1])
    bl = json.loads((T / "A_out" / "s0" / "done.json").read_text())["bit_list_sha256"]
    check("full: the bit-list line went to the test LOG.md", bl in (T / "LOG.md").read_text())
    check("full: a chain manifest for every A step", len(list((T / "state").glob("A_*.sha256"))) == 22)
    env = json.loads((T / "env_seen.json").read_text())
    check("full: clean environment (nothing from the calling session)", env["unexpected_variables"] == 0, env)
    check("full: 2 threads, no GPU, nice 19", set(env["threads"].values()) == {"2"} and env["cuda_visible_devices_empty"]
          and env["nice"] == 19, env)
    print(f"   (full run took {dt:.1f} s)")
    # restart: every step is done and verifies -> nothing runs again
    n = len(calls(T))
    r = chain(T)
    check("restart: exit 0, no job runs again", r.returncode == 0 and len(calls(T)) == n, r.stdout + r.stderr)
    check("restart: skips logged", sum(" skip " in ln for ln in logl(T)) >= len(set(want)) - 1)
    # an output changed after its step: the restart stops there
    p = T / "A_out" / "s1" / "results.json"
    p.write_text(p.read_text().replace("5.1", "5.0"))
    r = chain(T)
    check("tamper: exit 1 at A_s1", r.returncode == 1 and has(T, r"FAIL A_s1: outputs exist but do not verify"), logl(T)[-3:])
    check("tamper: nothing ran", len(calls(T)) == n)


def test_crash_then_restart():
    T = new_case("crash", FULL_SC | {"codes": {**FULL_SC["codes"], "A tm s2c_aa.py": [1, 0]}})
    r = chain(T)
    check("crash: exit 1", r.returncode == 1 and has(T, r"FAIL A_s2c_aa: the job failed \(exit 1\)"), logl(T)[-3:])
    check("crash: stopped right after the failing job", calls(T)[-1] == "A tm s2c_aa.py" and len(calls(T)) == 6)
    check("crash: the .partial folder is left", (T / "A_out" / "s2c_aa.partial").exists())
    r = chain(T)
    c = calls(T)
    check("crash restart: exit 0", r.returncode == 0, r.stdout + "\n".join(logl(T)[-5:]))
    check("crash restart: resumes at the failed step", c[6:] == FULL[5:], c)
    check("crash restart: the partial run is noted", has(T, r"todo A_s2c_aa: s2c_aa.partial exists"))


def test_a_stops_then_c():
    cases = {
        "k1": ({"A tm s1_floors.py": [3]}, ["C step0", "A tm s0_data.py", "A tm s1_floors.py", *C6], r"A STOP at A_s1: K1 STOP"),
        "k9twin": ({"A tm s2b_twin.py": [3]}, ["C step0", *A_S[:4], *C6], r"A STOP at A_s2b_twin: K9 STOP"),
        "k8": ({"A tm s2c_aa.py": [3]}, ["C step0", *A_S, *C6], r"A STOP at A_s2c_aa: K8 STOP"),
        "k9lr1": ({"A tm s2b_arm.py --arm full --family lr1": [3]}, ["C step0", *A_S, *LEARNERS[:2], *C6],
                  r"A STOP at A_s2b_full_lr1: K9 STOP for full/lr1"),
        "k3": ({"A tm s2d_gates.py": [3]}, ["C step0", *A_S, *LEARNERS, "A tm s2d_gates.py", *C6],
               r"A STOP at A_s2d_gates: both arms stop"),
    }
    for name, (codes, want, pat) in cases.items():
        T = new_case(name, {"pilots": PILOTS, "codes": codes})
        r = chain(T)
        check(f"{name}: exit 0, A ends, C step 6 follows", r.returncode == 0 and calls(T) == want and has(T, pat),
              r.stdout + "\n".join(calls(T)) + "\n" + "\n".join(logl(T)[-6:]))
        check(f"{name}: the end line names the stop", "chain end: A: STOP at" in logl(T)[-1], logl(T)[-1])


def test_k3c_routes():
    # A-full stops at the S2 gates, A-nf runs its pilots; its gate asks for the shuffled pilot, then stops (K9 TM)
    T = new_case("k3c_partial", {"pilots": PILOTS, "go_to_K3c": {"full": False, "nf": True},
                                 "codes": {"A tm k3c_gate.py --arm nf": [5, 3]}})
    r = chain(T)
    want = ["C step0", *A_S, *LEARNERS, "A tm s2d_gates.py", *k3c("nf", 40), *C6]
    check("k3c partial: order", r.returncode == 0 and calls(T) == want, "\n".join(calls(T)) + "\n".join(logl(T)[-5:]))
    check("k3c partial: A-full has no pilot", has(T, r"A-full: go_to_K3c\[full\] = False: no pilot"))
    check("k3c partial: A-nf STOP after the shuffled pilot", has(T, r"A-nf: K3c STOP"))
    check("k3c partial: end line", re.search(r"A-full: stopped at the S2 gates.*; A-nf: K3c STOP", logl(T)[-1]), logl(T)[-1])
    # A-full's gate stops at once (no shuffled pilot); A-nf passes
    T = new_case("k3c_stop", {"pilots": PILOTS, "codes": {"A tm k3c_gate.py --arm full": [3],
                                                          "A tm k3c_gate.py --arm nf": [5, 0]}})
    r = chain(T)
    want = ["C step0", *A_S, *LEARNERS, "A tm s2d_gates.py", *k3c("full"), *k3c("nf", 40), *C6]
    check("k3c stop: order", r.returncode == 0 and calls(T) == want, "\n".join(calls(T)))
    check("k3c stop: end line", "A-full: K3c STOP: no GPU job; A-nf: K3c pass" in logl(T)[-1], logl(T)[-1])
    # the gate asks for another s than the pilots' better one: stop
    T = new_case("k3c_mismatch", {"pilots": PILOTS, "gate_s": {"full": 40},
                                  "codes": {"A tm k3c_gate.py --arm full": [5, 0]}})
    r = chain(T)
    check("k3c mismatch: exit 1", r.returncode == 1 and has(T, r"FAIL A_k3c_gate_full: the gate asks for s '40'"),
          logl(T)[-3:])
    # the gate asks for the shuffled pilot twice: stop
    T = new_case("k3c_twice", {"pilots": PILOTS, "codes": {"A tm k3c_gate.py --arm full": [5, 5]}})
    r = chain(T)
    check("k3c twice: exit 1", r.returncode == 1 and has(T, r"asks for the shuffled pilot again"), logl(T)[-3:])


def test_failure_stops():
    cases = {
        "exit4": ({"codes": {"C step0": [4]}}, ["C step0"], r"FAIL C_step0: the imported folders changed"),
        "exit2": ({"codes": {"A tm s0_data.py": [2]}}, ["C step0", "A tm s0_data.py"], r"FAIL A_s0: the start was refused"),
        "liar0": ({"nowrite": ["A tm s0_data.py"]}, ["C step0", "A tm s0_data.py"],
                  r"FAIL A_s0: wrapper exit 0, but the outputs do not show"),
        "liar3": ({"codes": {"A tm s0_data.py": [3]}}, ["C step0", "A tm s0_data.py"],
                  r"FAIL A_s0: wrapper exit 3 \(a registered STOP\), but the outputs do not show one"),
        "cstop0": ({"codes": {"C prep": [3]}}, None, r"FAIL C_prep: wrapper exit 3"),
        "busy": ({"codes": {"C step0": [75]}}, ["C step0"] * 3, r"FAIL C_step0: C's own lock stayed busy"),
    }
    for name, (sc, want, pat) in cases.items():
        T = new_case(name, {"pilots": PILOTS, **sc} if name != "cstop0" else
                     {"pilots": PILOTS, "codes": {"A tm s1_floors.py": [3], "C prep": [3]}})
        r = chain(T)
        ok = r.returncode == 1 and has(T, pat) and (want is None or calls(T) == want)
        check(f"{name}: the chain stops with exit 1", ok, r.stdout + "\n".join(calls(T)) + "\n" + "\n".join(logl(T)[-4:]))
        check(f"{name}: no chain end line", not any("chain end" in ln for ln in logl(T)))


def test_hash_mismatch_stops():
    T = new_case("pin", {"pilots": PILOTS})
    pins = T / "pins.json"
    pins.write_text(json.dumps({L.RECORD["A"]: "0" * 64}))
    r = chain(T, pins=pins)
    check("pin: exit 1 before the first A job", r.returncode == 1 and calls(T) == ["C step0"]
          and has(T, r"FAIL A_s0: the hash check against the R1 lines"), "\n".join(logl(T)[-6:]))
    check("pin: the failing check is logged", has(T, r"hashcheck A_s0: +FAIL glassbox-lm/studies/language_part3/A/R1_A-Struct.frozen.json"))


def test_time_boxes():
    T = new_case("k7", {"pilots": PILOTS})
    (T / "A_out").mkdir()
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 11 * 86400))
    (T / "A_out" / "jobs.log").write_text(f"{old} start s0_data \n")
    r = chain(T)
    check("K7: A stops before its first job, C step 6 follows", r.returncode == 0 and calls(T) == ["C step0", *C6]
          and has(T, r"A STOP at A_s0: K7 time box reached"), "\n".join(calls(T)) + "\n".join(logl(T)[-4:]))
    T = new_case("ck4", {"pilots": PILOTS, "codes": {"A tm s1_floors.py": [3]}})
    (T / "C_runs").mkdir()
    (T / "C_runs" / "jobs_log.tsv").write_text("a\tb\tstep0\t\t1\t262800\tno change\n")
    r = chain(T)
    check("C-K4: no C job, A runs", r.returncode == 0 and calls(T) == ["A tm s0_data.py", "A tm s1_floors.py"]
          and has(T, r"C step 6 not run: C-K4 time box reached"), "\n".join(calls(T)) + "\n".join(logl(T)[-4:]))


def test_broken_gate_reading():
    T = new_case("gatejunk", {"pilots": PILOTS}, gate_lines=["Traceback (most recent call last):"] * 5)
    r = chain(T)
    check("gate: 5 broken readings in a row stop the chain", r.returncode == 1 and not calls(T)
          and has(T, r"FAIL C_step0: the CPU gate reading failed 5 times"), logl(T)[-3:])


def test_guards_and_dry():
    T = new_case("single", {"pilots": PILOTS})
    with open(T / "run_round1.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        r = chain(T)
    check("singleton: a second chain is refused", r.returncode == 2 and "refused" in r.stdout and not calls(T), r.stdout)
    r = subprocess.run(["bash", str(SELF), "--bogus"], capture_output=True, text=True, check=False,
                       env={"PATH": "/usr/bin:/bin", "ROUND1_TEST_DIR": str(T)})
    check("usage: exit 2", r.returncode == 2 and "usage" in r.stdout, r.stdout)
    outside = TM / "A" / "tmp" / "r1test_never"
    r = subprocess.run(["bash", str(SELF)], capture_output=True, text=True, check=False,
                       env={"PATH": "/usr/bin:/bin", "ROUND1_TEST_DIR": str(outside)})
    check("test dir outside chain/tmp is refused", r.returncode == 2 and not outside.exists(), r.stdout)
    T = new_case("dry", {"pilots": PILOTS})
    before = sorted(str(p.relative_to(T)) for p in T.rglob("*"))
    r = chain(T, "--dry", cache=False)
    after = sorted(str(p.relative_to(T)) for p in T.rglob("*"))
    order = re.findall(r"^  (C_\w+|A_\w+) ", r.stdout, re.M)
    want = (["C_step0", "A_s0", "A_bitlist", "A_s1", "A_s2a", "A_s2b_twin", "A_s2c_aa"]
            + [f"A_s2b_{a}_{f}" for a in ("full", "nf") for f in ("tree", "lr1", "lr2", "hgb")] + ["A_s2d_gates"]
            + [x for a in ("full", "nf") for x in (f"A_k3c_{a}_s15", f"A_k3c_{a}_s40", f"A_k3c_gate_{a}",
                                                  f"A_k3c_{a}_shuffled", f"A_k3c_gate_{a}")]
            + ["C_prep", "C_twin", "C_learners", "C_gates"])
    check("dry: exit 0 and the plan's order", r.returncode == 0 and order == want, r.stdout[-3000:] + str(order))
    check("dry: writes nothing, runs nothing", before == after and not calls(T), (before, after))


def main():
    t0 = time.time()
    print("round-1 chain self-tests start", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)
    for name in ("run_round1.sh", "round1_lib.py", "tests/fake_round1_wrapper.py", "tests/test_run_round1.py",
                 "part3_cpu_job.sh", "part3_status.py"):
        print(f"{L._sha256_file(CH / name)}  chain/{name}")
    if BASE.exists():
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True)
    for t in (test_lib_units, test_hashcheck_real_tree, test_hashcheck_synthetic, test_full_pass_restart_tamper,
              test_crash_then_restart, test_a_stops_then_c, test_k3c_routes, test_failure_stops,
              test_hash_mismatch_stops, test_time_boxes, test_broken_gate_reading, test_guards_and_dry):
        t()
    print(f"ROUND-1 CHAIN TESTS: {len(PASSED)} passed in {time.time() - t0:.1f} s")
    print("round-1 chain self-tests end", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
