#!/usr/bin/env python
"""Helpers of chain/run_round1.sh, the round-1 Part 3 CPU chain (PLAN_part3.md section 6, steps 2-6).
Run with python -I -B (the tm venv). Reads only, except `manifest write` (chain/round1_state/<step>.sha256).

    round1_lib.py hashcheck A|C [<step>]  the R1 lines' pins, the arm's frozen record and every file in it, the code
                                          prefixes written in the R1 lines, the chain wrapper; exit 0 ok, 1 not
    round1_lib.py status <step>           exit 0 done (and its gate passed), 3 done with a registered STOP,
                                          10 not done (absent, or a crashed run's leftovers), 1 outputs inconsistent
    round1_lib.py go-k3c full|nf          exit 0 when S2d's gates.json has go_to_K3c[arm] true, 3 when false, 1 missing
    round1_lib.py better-s full|nf        prints the better pilot s (lower tune-cal bits; ties: s 15), exit 1 if missing
    round1_lib.py ck4 [--now UTC]         the C-K4 time box; exit 3 when reached
    round1_lib.py manifest write <step>   the chain's SHA-256 list of an A step's output folder (restart check)

Test mode (chain/tests/test_run_round1.py only): ROUND1_TEST_DIR, a folder inside chain/tmp/, moves A/out, C/runs,
C/data, the bit-list LOG and the chain state into it; ROUND1_TEST_PINS (a JSON file) may replace pins.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(f"{_WS_ROOT}")
TM = ROOT / "glassbox-lm" / "studies" / "language_part3"
CH = TM / "chain"

# ---- pins, copied from the R1 lines of LOG.md (every pin is also checked to appear in its line) ----------------
R1_LINE_HEADS = {
    "C": "- 2026-10-21T03:57:47Z: [C-core] R1 REGISTERED",
    "A": "- 2026-10-21T05:18:52Z: [A-Struct] R1 REGISTERED",
    "A1": "- 2026-10-21T07:22:13Z: [A-Struct] R1 [A-Struct] AMENDMENT 1",
}
# (file relative to ROOT, full SHA-256, the R1 line that writes it)
FILE_PINS = {
    "A": [("glassbox-lm/studies/language_part3/A/R1_A-Struct.frozen.json",
           "5d5f154b5d714cc7c7fdc0b66e261544c9638999337fecfe51bd5d7c70e08aaf", "A1"),
          ("glassbox-lm/studies/language_part3/A/MANIFEST_R1.sha256",
           "7c77f4f0f14d4503dc83ce070612c97be8ce5e7677f27b1f6af428dba0f899bc", "A1"),
          ("glassbox-lm/studies/language_part3/A/REGISTRATION_R1.md",
           "cf95e70900caff533f5231131052b1d205fd0aa4108f199b5643867bd3fd262d", "A1"),
          ("glassbox-lm/studies/language_part3/A/README.md",
           "28fc8daaca2df074994fa45b5f19241a1de211160dfdb365c69a9408748d5b1c", "A1"),
          ("glassbox-lm/studies/language_part3/A/history/R1_A-Struct.frozen.v1.json",
           "a2b7f6c8b0be228a70ca2e49cd50923a9e02e576d89775887ed2f5486ce802b0", "A")],
    "C": [("glassbox-lm/studies/language_part3/C/R1_C-core.frozen.json",
           "91c54085c3be4e9ee6e27f1c8206d6a97a280046f563622edc15373ca1453a42", "C"),
          ("glassbox-lm/studies/language_part3/C/R1_C-core.md",
           "6e3517bafd8b9a22f4e013faf48712ee06c50f388a2862df165239a211d35f0e", "C")],
    "chain": [("glassbox-lm/studies/language_part3/chain/part3_cpu_job.sh",
               "77ea3c1ef9e5ddfeae86e6dae356d63ba8812ffc27baa3390e7d4a970b8a43c8", "A1"),
              ("glassbox-lm/studies/language_part3/chain/part3_status.py",
               "4715d020b875e774beeefe9cd0b5a552dbad704aba9e52b62b63972220b493f7", "A1")],
}
RECORD = {"A": "glassbox-lm/studies/language_part3/A/R1_A-Struct.frozen.json", "C": "glassbox-lm/studies/language_part3/C/R1_C-core.frozen.json"}
RECORD_V1_A = "glassbox-lm/studies/language_part3/A/history/R1_A-Struct.frozen.v1.json"
_A, _C = "glassbox-lm/studies/language_part3/A/", "glassbox-lm/studies/language_part3/C/"
# code prefixes as the R1 lines write them: (text in the line, file, line); a later line supersedes an earlier one
CODE_PREFIXES = {
    "A": [("s0_data 27ba5373bb7c8c4e", _A + "s0_data.py", "A"),
          ("s1_floors 9c5eba94b8be0285", _A + "s1_floors.py", "A"),
          ("s2a_rows 85025c6c416e5167", _A + "s2a_rows.py", "A"),
          ("s2b_twin 7cfc379555a0f989", _A + "s2b_twin.py", "A"),
          ("s2b_arm a00be78ae022324b", _A + "s2b_arm.py", "A"),
          ("s2c_aa 1aae492241098d3f", _A + "s2c_aa.py", "A"),
          ("s2d_gates a3ffb6428fb0512e", _A + "s2d_gates.py", "A"),
          ("k3c_pilot e918e8bcaaed66db", _A + "k3c_pilot.py", "A"),
          ("k3c_gate e6ec930f572a395b", _A + "k3c_gate.py", "A"),
          ("run_job.sh 1f81ad92d91dc179", _A + "run_job.sh", "A"),
          ("astruct/paths f83cab239b2837e1", _A + "astruct/paths.py", "A"),
          ("s2c_aa 35b68fbd2c9e90d8", _A + "s2c_aa.py", "A1"),
          ("s2d_gates 9cfaf7f2243b4d99", _A + "s2d_gates.py", "A1"),
          ("k3c_pilot 6705fd5de024c542", _A + "k3c_pilot.py", "A1"),
          ("k3c_gate 81228d53d87b9822", _A + "k3c_gate.py", "A1"),
          ("s2a_rows 465d142aac2c34d6", _A + "s2a_rows.py", "A1"),
          ("run_job.sh ebe20827c3027a36", _A + "run_job.sh", "A1"),
          ("lock_held.sh 08cb1f1ff68a1004", _A + "lock_held.sh", "A1"),
          ("freeze.py f5379a80cf4f3416", _A + "freeze.py", "A1"),
          ("astruct/paths 728e5d122ec6225a", _A + "astruct/paths.py", "A1"),
          ("astruct/gates 0303c10426cd500d", _A + "astruct/gates.py", "A1"),
          ("astruct/build 672a6fd6f1fe0cfa", _A + "astruct/build.py", "A1")],
    "C": [("ccore/chains.py a15c4b1f1de0ba7b", _C + "ccore/chains.py", "C"),
          ("codes.py f80fcfdb51bafaaa", _C + "ccore/codes.py", "C"),
          ("floor.py 2d0e8ae4360f11f5", _C + "ccore/floor.py", "C"),
          ("gates.py 21e792a464cdfb02", _C + "ccore/gates.py", "C"),
          ("systems.py d69d65cf439d7989", _C + "ccore/systems.py", "C"),
          ("twin.py c1471bd74c2415fa", _C + "ccore/twin.py", "C"),
          ("learners.py e722456122eaff3a", _C + "ccore/learners.py", "C"),
          ("step0.py 8b332d898fe8165d", _C + "ccore/step0.py", "C"),
          ("layout.py 488d6b99643503ea", _C + "ccore/layout.py", "C"),
          ("scripts/run_stage.sh e523d5f7e18388fe", _C + "scripts/run_stage.sh", "C")],
}
LINE_ORDER = {"C": 0, "A": 1, "A1": 2}
# C-K4 (R1 [C-core] line): a C-K0/C-K0b decision within 72 gate-open hours of 2026-10-21T03:09:23Z and by this time
C_K4_DEADLINE = "2026-10-31T03:09:23Z"
C_K4_HOURS = 72.0
C_REAL_STAGES = ("step0", "prep", "twin", "learners", "gates")
ARMS, FAMILIES, PILOT_S = ("full", "nf"), ("tree", "lr1", "lr2", "hgb"), (15, 40)


# ---- paths: real, or the test folder ---------------------------------------------------------------------------
def test_dir():
    t = os.environ.get("ROUND1_TEST_DIR")
    if not t:
        return None
    p = Path(t).resolve()
    if not p.is_relative_to((CH / "tmp").resolve()):
        raise SystemExit(f"ROUND1_TEST_DIR must be inside {CH / 'tmp'}, not {p}")
    return p


def paths() -> dict:
    t = test_dir()
    if t is None:
        return {"test": None, "a_out": TM / "A" / "out", "c_runs": TM / "C" / "runs", "c_data": TM / "C" / "data",
                "bitlist_log": TM / "LOG.md", "state": CH / "round1_state"}
    return {"test": t, "a_out": t / "A_out", "c_runs": t / "C_runs", "c_data": t / "C_data",
            "bitlist_log": t / "LOG.md", "state": t / "state"}


def pins() -> tuple:
    """FILE_PINS and CODE_PREFIXES; in test mode ROUND1_TEST_PINS may replace a file pin ({"<rel path>": "<sha>"})."""
    fp = {k: list(v) for k, v in FILE_PINS.items()}
    t = test_dir()
    over = os.environ.get("ROUND1_TEST_PINS")
    if t is not None and over:
        rep = json.loads(Path(over).read_text())
        fp = {k: [(f, rep.get(f, h), ln) for f, h, ln in v] for k, v in fp.items()}
    return fp, CODE_PREFIXES


def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_HCACHE: dict | None = None


def sha256(p) -> str:
    """SHA-256 of a file. Test mode only: ROUND1_TEST_HASHCACHE (a JSON file in the test folder) keeps hashes by path,
    size, mtime and inode, so the scenario tests stay short; real runs always hash every file."""
    cache = os.environ.get("ROUND1_TEST_HASHCACHE")
    t = test_dir() if cache else None
    if t is None or not Path(cache).resolve().is_relative_to(t):
        return _sha256_file(p)
    global _HCACHE
    cp = Path(cache)
    if _HCACHE is None:
        _HCACHE = json.loads(cp.read_text()) if cp.exists() else {}
        import atexit
        atexit.register(lambda: cp.write_text(json.dumps(_HCACHE)))
    st = os.stat(p)
    key = str(Path(p).resolve())
    sig = [st.st_size, st.st_mtime_ns, st.st_ino]
    hit = _HCACHE.get(key)
    if hit and hit[:3] == sig:
        return hit[3]
    h = _sha256_file(p)
    _HCACHE[key] = sig + [h]
    return h


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


# ---- hash check -------------------------------------------------------------------------------------------------
STEP_FILES = {  # the files each kind of step runs (besides the runner and the wrapper)
    "C_step0": [_C + "scripts/step0.py"], "C_prep": [_C + "scripts/prep.py"], "C_twin": [_C + "scripts/twin.py"],
    "C_learners": [_C + "scripts/learners.py"], "C_gates": [_C + "scripts/gates.py"],
    "A_s0": [_A + "s0_data.py"], "A_bitlist": [_A + "log_bitlist.sh"], "A_s1": [_A + "s1_floors.py"],
    "A_s2a": [_A + "s2a_rows.py"], "A_s2b_twin": [_A + "s2b_twin.py"], "A_s2c_aa": [_A + "s2c_aa.py"],
    "A_s2d_gates": [_A + "s2d_gates.py"],
}
for _arm in ARMS:
    for _f in FAMILIES:
        STEP_FILES[f"A_s2b_{_arm}_{_f}"] = [_A + "s2b_arm.py"]
    for _s in ("s15", "s40", "shuffled"):
        STEP_FILES[f"A_k3c_{_arm}_{_s}"] = [_A + "k3c_pilot.py"]
    STEP_FILES[f"A_k3c_gate_{_arm}"] = [_A + "k3c_gate.py"]
RUNNER_FILES = {"A": [_A + "run_job.sh", _A + "env.sh", _A + "listing.py", _A + "gate_check.sh", _A + "lock_held.sh",
                      _A + "k7_clock.py"],
                "C": [_C + "scripts/run_stage.sh"]}


def r1_lines(logmd: Path) -> tuple[dict, list]:
    lines = logmd.read_text().splitlines()
    out, bad = {}, []
    for k, head in R1_LINE_HEADS.items():
        hits = [ln for ln in lines if ln.startswith(head)]
        if len(hits) != 1:
            bad.append(f"LOG.md must hold exactly one line starting {head!r}; found {len(hits)}")
        else:
            out[k] = hits[0]
    return out, bad


def hashcheck(arm: str, step: str | None = None, root: Path = ROOT, logmd: Path | None = None,
              file_pins: dict | None = None, code_prefixes: dict | None = None) -> tuple[bool, list]:
    """The checks before every step of arm A or C. Returns (ok, report lines starting 'ok', 'NOTE' or 'FAIL')."""
    root = Path(root)
    logmd = Path(logmd) if logmd is not None else root / "glassbox-lm" / "studies" / "language_part3" / "LOG.md"
    fp, cp = (file_pins, code_prefixes) if file_pins is not None else pins()
    rep, fails = [], []

    def fail(msg):
        fails.append(msg)
        rep.append("FAIL " + msg)

    lines, bad = r1_lines(logmd)
    for b in bad:
        fail(b)
    # 1. the pinned files: the hash written in the R1 line, and the file now
    for group in (arm, "chain"):
        for rel, h, ln in fp[group]:
            p = root / rel
            if ln in lines and h not in lines[ln]:
                fail(f"{rel}: pin {h[:16]} is not in the R1 line {R1_LINE_HEADS[ln]!r}")
            if not p.exists():
                fail(f"{rel}: missing")
            elif sha256(p) != h:
                fail(f"{rel}: sha256 {sha256(p)[:16]} differs from the pin {h[:16]} (R1 line {ln})")
            else:
                rep.append(f"ok {rel} {h[:16]} (pinned in the R1 line {ln})")
    # 2. the frozen record: every file in it
    rec_rel = RECORD[arm]
    rec = json.loads((root / rec_rel).read_text()) if (root / rec_rel).exists() else None
    if rec is None:
        fail(f"{rec_rel}: missing")
        return False, rep
    key = "enforced" if arm == "A" else "files"
    files = rec[key]
    bad_files = []
    for rel, h in sorted(files.items()):
        p = root / rel
        if not p.exists():
            bad_files.append(f"{rel}: missing")
        elif sha256(p) != h:
            bad_files.append(f"{rel}: changed since the freeze")
    for b in bad_files:
        fail(f"frozen record {rec_rel}: {b}")
    if not bad_files:
        label = "enforced" if arm == "A" else "tracked"
        rep.append(f"ok frozen record {rec_rel}: all {len(files)} {label} files have their frozen SHA-256")
    if arm == "C":
        reg = [h for rel, h, _ in fp["C"] if rel.endswith("R1_C-core.md")][0]
        if rec.get("registration_sha256") != reg:
            fail(f"{rec_rel}: registration_sha256 {str(rec.get('registration_sha256'))[:16]} is not the pin {reg[:16]}")
    for rel, h in sorted(rec.get("recorded" if arm == "A" else "reference_files_not_enforced", {}).items()):
        p = root / rel
        now = sha256(p) if p.exists() else "missing"
        if now != h:
            rep.append(f"NOTE recorded, not enforced: {rel} {h[:16]} at the freeze, now {now[:16]}")
    # 3. the code prefixes written in the R1 lines (the latest line that names a file counts)
    v1 = None
    if arm == "A" and (root / RECORD_V1_A).exists():
        v1 = json.loads((root / RECORD_V1_A).read_text())["enforced"]
    latest = {}
    for text, rel, ln in cp[arm]:
        if ln in lines and text not in lines[ln]:
            fail(f"code prefix {text!r} is not in the R1 line {R1_LINE_HEADS[ln]!r}")
        if rel not in latest or LINE_ORDER[ln] > LINE_ORDER[latest[rel][1]]:
            latest[rel] = (text.split()[-1], ln)
    n_match, superseded = 0, []
    for rel, (pre, ln) in sorted(latest.items()):
        p = root / rel
        now = sha256(p) if p.exists() else ""
        if now.startswith(pre):
            n_match += 1
        elif (arm == "A" and ln == "A" and v1 is not None and v1.get(rel, "").startswith(pre)
              and now and now == files.get(rel)):
            superseded.append(f"{rel.split('/A/')[-1]} {pre} -> {now[:16]}")
        else:
            fail(f"{rel}: sha256 {now[:16]} does not start with the R1-line prefix {pre} (line {ln})")
    rep.append(f"ok code prefixes of the R1 lines: {n_match} of {len(latest)} files match their latest R1-line prefix")
    for s in superseded:
        rep.append(f"NOTE {s}: the R1 line (05:18:52Z) prefix is the first freeze's (v1 record); the AMENDMENT 1 "
                   "line does not list this file; its hash now is the amended record's (pinned in the amendment line)")
    # 4. the files this step runs: each one in the frozen record, with its hash now
    run_files = RUNNER_FILES.get(arm, []) + (STEP_FILES.get(step, []) if step else [])
    for rel in run_files:
        p = root / rel
        if rel not in files:
            fail(f"{rel}: runs in this step but is not in the frozen record {rec_rel}")
        elif p.exists() and sha256(p) == files[rel]:
            if step and rel in STEP_FILES.get(step, []):
                rep.append(f"ok step {step}: {rel} {files[rel][:16]} (frozen record)")
    return not fails, rep


# ---- done checks ------------------------------------------------------------------------------------------------
def _jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def a_step_folder(step: str, P: dict) -> tuple[str | None, list]:
    """(folder name under A/out, required files) of an A step; folder None when it cannot be known yet."""
    if step == "A_s0":
        return "s0", ["done.json", "bit_list.json", "meta.json", "corpus.npz"]
    if step == "A_s1":
        return "s1", ["done.json", "results.json", "summ_cal.npz", "summ_hold.npz"]
    if step == "A_s2a":
        return "s2a", ["done.json", "results.json", "twin_rows.npz", "train_rows.npz", "cal_rows.npz", "hold_rows.npz"]
    if step == "A_s2b_twin":
        return "s2b_twin", ["done.json", "results.json", "bits.npz", "B.pkl"]
    if step == "A_s2c_aa":
        return "s2c_aa", ["done.json", "results.json", "k8.json", "bits.npz"]
    if step == "A_s2d_gates":
        return "s2d_gates", ["done.json", "gates.json", "words.txt"]
    m = re.fullmatch(r"A_s2b_(full|nf)_(tree|lr1|lr2|hgb)", step)
    if m:
        return f"s2b_{m[1]}_{m[2]}", ["done.json", "results.json", "bits.npz", "pick.pkl"]
    m = re.fullmatch(r"A_k3c_(full|nf)_s(15|40)", step)
    if m:
        return f"k3c_{m[1]}_s{m[2]}", ["done.json", "results.json", "state.npz", "bits.npz"]
    m = re.fullmatch(r"A_k3c_(full|nf)_shuffled", step)
    if m:
        s = better_s(m[1], P)
        return (f"k3c_{m[1]}_s{s}_shuffled" if s else None), ["done.json", "results.json", "state.npz", "bits.npz"]
    m = re.fullmatch(r"A_k3c_gate_(full|nf)", step)
    if m:
        return f"k3c_gate_{m[1]}", ["done.json", "k3c.json", "words.txt"]
    raise SystemExit(f"unknown A step {step}")


def better_s(arm: str, P: dict) -> int | None:
    res = {}
    for s in PILOT_S:
        d = P["a_out"] / f"k3c_{arm}_s{s}"
        if not (d / "done.json").exists() or not (d / "results.json").exists():
            return None
        res[s] = _jload(d / "results.json")["tune_cal_bits"]
    return min(PILOT_S, key=lambda s: (res[s], s))


def folder_manifest(d: Path) -> list:
    return [f"{sha256(p)}  {p.relative_to(d)}" for p in sorted(d.rglob("*")) if p.is_file()]


def a_decision(step: str, d: Path, done: dict) -> tuple[int, str]:
    """(0 pass or no gate | 3 registered STOP, text) from an A step's outputs."""
    if step == "A_s0":
        return 0, (f"bit list {done['bit_list_sha256'][:16]} ({done['bits']} bits, {done['count_derived_bits']} "
                   f"count-derived); replies {done['counts']['replies']}")
    if step == "A_s1":
        r = _jload(d / "results.json")
        f = r["floor"]
        ok = bool(done["K1"])
        return (0 if ok else 3), (f"K1 {'pass' if ok else 'STOP'} (rel ppl {r['K1']['rel_ppl']:.2e}, limit 1e-3); floor "
                                  f"{f}: tune-cal {r['tune_cal_bits'][f]:.5f}, tune-hold {r['tune_hold_bits'][f]:.5f} bits/token")
    if step == "A_s2a":
        r = _jload(d / "results.json")
        return 0, f"rows {json.dumps(r['rows'])}; leak test {r['leak_test']['pass']}, no-future {r['nofuture']['pass']}"
    if step == "A_s2b_twin":
        r = _jload(d / "results.json")
        ok = bool(done["K9"])
        return (0 if ok else 3), (f"B = floor + {r['pick']}: tune-hold {r['B']['tune_hold_bits']:.5f} bits/token; K9 "
                                  f"(count twin) {'pass' if ok else 'STOP'} (b {r['K9']['b']}, d_hold {r['K9']['d_hold']:+.5f})")
    if step == "A_s2c_aa":
        k = _jload(d / "k8.json")["K8"]
        ok = bool(done["K8"])
        return (0 if ok else 3), (f"K8 (rebuilt(2) - rebuilt(3)) {'pass' if ok else 'STOP'}: {k['mean']:+.5f} "
                                  f"[{k['lo']:+.5f}, {k['hi']:+.5f}] (needs within +-0.0036 and covering 0)")
    if step.startswith("A_s2b_"):
        r = _jload(d / "results.json")
        ok = bool(done["K9"])
        p = r["pick"]
        return (0 if ok else 3), (f"pick {p}: tune-hold minus B {r['configs'][p]['tune_hold_minus_B']:+.5f}; K9 "
                                  f"{'pass' if ok else 'STOP'} (b {r['K9']['b']}, d_hold {r['K9']['d_hold']:+.5f})")
    if step == "A_s2d_gates":
        g = _jload(d / "gates.json")
        go = done["go_to_K3c"]
        parts = [f"go_to_K3c full={go['full']} nf={go['nf']}", f"K8 {g['K8']['pass']}", f"K9 all {g['K9_all_pass']}"]
        for a in ARMS:
            k = g["arms"][a]["K3"]
            parts.append(f"A-{a} K3 (i) {k['best_trees_lr2']:+.5f} {'pass' if k['i_pass'] else 'stop'}, (ii) LR1 "
                         f"{k['d_lr1']:+.5f} {'pass' if k['ii_pass'] else 'stop'}")
        return (0 if any(go.values()) else 3), "; ".join(parts)
    if step.startswith("A_k3c_gate_"):
        k = _jload(d / "k3c.json")
        ok = bool(done["go_to_GPU"])
        g = k["K3c"]
        k9 = k.get("K9_tm")
        return (0 if ok else 3), (f"better s {k['better_s']}; B - pilot {g['gain_pilot']:+.5f}, B - trees "
                                  f"{g['gain_trees']:+.5f}; K3c {'pass' if g['pass'] else 'STOP'}; K9 (TM) "
                                  f"{'not needed' if k9 is None else ('pass' if k9['pass'] else 'FAIL')}; go_to_GPU {ok}")
    if step.startswith("A_k3c_"):
        r = _jload(d / "results.json")
        return 0, (f"{r['tag']}: b {r['b']}, tune-cal {r['tune_cal_bits']:.5f}, tune-hold minus B "
                   f"{r['tune_hold_minus_B']:+.5f}")
    raise SystemExit(f"unknown A step {step}")


def status_a(step: str, P: dict) -> tuple[int, str]:
    if step == "A_bitlist":
        s0 = P["a_out"] / "s0"
        if not (s0 / "done.json").exists():
            return 10, "S0 not done"
        h = sha256(s0 / "bit_list.json")
        if h != _jload(s0 / "done.json")["bit_list_sha256"]:
            return 1, f"bit_list.json {h[:16]} changed after S0"
        if h in P["bitlist_log"].read_text():
            return 0, f"LOG.md holds the bit-list SHA-256 {h}"
        return 10, f"LOG.md does not hold the bit-list SHA-256 {h} yet"
    name, need = a_step_folder(step, P)
    if name is None:
        return 10, "the pilots are not both done (the better s is not known)"
    d = P["a_out"] / name
    if not d.exists():
        part = d.with_name(name + ".partial")
        return 10, (f"{name}.partial exists (a crashed run; the rerun removes it)" if part.exists() else f"{name}: absent")
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        return 1, f"{name}: files missing {missing}"
    try:
        done = _jload(d / "done.json")
        pin = [h for rel, h, _ in pins()[0]["A"] if rel == RECORD["A"]][0]
        if done.get("frozen_record_sha256") != pin:
            return 1, f"{name}: written under frozen record {str(done.get('frozen_record_sha256'))[:16]}, not {pin[:16]}"
        if step == "A_s0" and sha256(d / "bit_list.json") != done["bit_list_sha256"]:
            return 1, "s0: bit_list.json differs from done.json's SHA-256"
        code, text = a_decision(step, d, done)
    except (KeyError, ValueError, TypeError) as e:
        return 1, f"{name}: outputs unreadable ({type(e).__name__}: {e})"
    man = P["state"] / f"{step}.sha256"
    if man.exists():
        if [ln for ln in man.read_text().splitlines() if ln] != folder_manifest(d):
            return 1, f"{name}: files differ from the chain's manifest {man.name} (written when the step ended)"
        text += "; chain manifest ok"
    else:
        text += "; no chain manifest yet"
    return code, f"{name}: {text}"


def c_finished(stage: str, P: dict) -> list:
    out = []
    for d in sorted(P["c_runs"].glob(f"*_{stage}")):
        try:
            if _jload(d / "run.json")["state"] == "finished":
                out.append(d)
        except (OSError, ValueError, KeyError):
            pass
    return out


def status_c(step: str, P: dict) -> tuple[int, str]:
    stage = step[2:]
    if stage not in C_REAL_STAGES:
        raise SystemExit(f"unknown C step {step}")
    runs = c_finished(stage, P)
    if not runs:
        return 10, f"no finished real {stage} run"
    if len(runs) > 1:
        return 1, f"{len(runs)} finished real {stage} runs (a real stage runs once): {[r.name for r in runs]}"
    d = runs[0]
    try:
        outs = _jload(d / "outputs.json")
    except (OSError, ValueError) as e:
        return 1, f"{d.name}: outputs.json unreadable ({e})"
    for rel, h in outs.items():
        p = ROOT / rel
        if not p.exists() or sha256(p) != h:
            return 1, f"{d.name}: {rel} differs from what the run wrote"
    if stage == "gates":
        dec_p = P["c_data"] / "step6_decision.json"
        rel = str(dec_p.relative_to(ROOT))
        if rel not in outs:
            return 1, f"{d.name}: outputs.json does not list {rel}"
        dec = _jload(dec_p).get("decision", "")
        if dec.startswith("STOP"):
            return 3, f"{d.name}: {dec}"
        if dec.startswith("PASS"):
            return 0, f"{d.name}: {dec}"
        return 1, f"{d.name}: unknown decision {dec!r}"
    return 0, f"{d.name}: finished, {len(outs)} output file(s) verified"


def status(step: str) -> tuple[int, str]:
    P = paths()
    return status_c(step, P) if step.startswith("C_") else status_a(step, P)


def go_k3c(arm: str) -> tuple[int, str]:
    p = paths()["a_out"] / "s2d_gates" / "gates.json"
    if not p.exists():
        return 1, f"{p} missing"
    go = _jload(p)["go_to_K3c"][arm]
    return (0 if go else 3), f"go_to_K3c[{arm}] = {go}"


# ---- C-K4 ----------------------------------------------------------------------------------------
def ck4(now: float | None = None) -> tuple[int, dict]:
    P = paths()
    now = time.time() if now is None else now
    code, dec = status_c("C_gates", P)
    if code in (0, 3):
        return 0, {"C_K4": "the C-K0/C-K0b decision exists; the time box no longer applies"}
    log = P["c_runs"] / "jobs_log.tsv"
    secs = 0.0
    if log.exists():
        for ln in log.read_text().splitlines():
            f = ln.split("\t")
            if len(f) >= 6 and f[2] in C_REAL_STAGES and "--dry" not in f[3]:
                try:
                    secs += float(f[5])
                except ValueError:
                    pass
    hours = secs / 3600
    late = now >= parse_utc(C_K4_DEADLINE)
    res = {"real_stage_hours_lower_bound": round(hours, 3), "limit_hours": C_K4_HOURS, "deadline": C_K4_DEADLINE,
           "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
           "limit_reached": bool(hours >= C_K4_HOURS or late)}
    return (3 if res["limit_reached"] else 0), res


def main(argv: list) -> int:
    if len(argv) >= 2 and argv[0] == "hashcheck" and argv[1] in ("A", "C"):
        ok, rep = hashcheck(argv[1], argv[2] if len(argv) > 2 else None)
        for ln in rep:
            print(ln)
        print(f"hashcheck {argv[1]}{' ' + argv[2] if len(argv) > 2 else ''}: {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if len(argv) == 2 and argv[0] == "status":
        code, text = status(argv[1])
        print(text)
        return code
    if len(argv) == 2 and argv[0] == "go-k3c" and argv[1] in ARMS:
        code, text = go_k3c(argv[1])
        print(text)
        return code
    if len(argv) == 2 and argv[0] == "better-s" and argv[1] in ARMS:
        s = better_s(argv[1], paths())
        if s is None:
            print("the two pilots are not both done")
            return 1
        print(s)
        return 0
    if argv[:1] == ["ck4"]:
        now = parse_utc(argv[2]) if len(argv) == 3 and argv[1] == "--now" else None
        code, res = ck4(now)
        print(json.dumps(res))
        return code
    if len(argv) == 3 and argv[:2] == ["manifest", "write"] and argv[2].startswith("A_") and argv[2] != "A_bitlist":
        P = paths()
        name, _ = a_step_folder(argv[2], P)
        d = P["a_out"] / str(name)
        if name is None or not (d / "done.json").exists():
            print(f"{argv[2]}: no finished output folder")
            return 1
        P["state"].mkdir(parents=True, exist_ok=True)
        lines = folder_manifest(d)
        tmp = P["state"] / f".{argv[2]}.sha256.part"
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, P["state"] / f"{argv[2]}.sha256")
        print(f"{argv[2]}: {len(lines)} files")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
