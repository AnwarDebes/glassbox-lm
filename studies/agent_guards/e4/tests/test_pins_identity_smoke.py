"""Tests of check_pins.py (literal hashes in the registration; review issue 1; revision 3: the freeze record at every
registered stage, the five H6 files, venv: pins of every dist-info; revision 4: the integrity re-run's record and its
evidence, one test for each kind, the stop-effects table of section 9.1, the pins_changed flag), e4_run.server_identity
(the frozen server pins; an in-process stub, the process scan replaced) and its end-of-call check (revision 3), the
kept manifests, and the smoke report's budget, cut order, matcher gate S8 (revision 3), extension rule (revision 4),
tool-call health and the repeat check's upper bound (scripts/smoke_report.py, repeat_check.py). Revision 5 (round-4
review M1 (e)): the failed-attempt count of section 11.6 (check_pins.stage_attempts: only exit 3 and 6 count; the two
instances of rerun) and the deferred end check of a call stopped before its end (e4_run.deferred_end_check).
Revision 6 (round-5 review Major 1 (b), (d)): the full start check comes FIRST (e4_run.start_checks): a server that is
not the pinned one exits 2 without "(identity_changed)" and leaves an earlier call's record untouched; with the pinned
server back (a restart: a new pid) the deferred check writes "restart_same_identity" and the run goes on; a record that
cannot be read is still "identity_changed"; the order inverted in the static check; and the end check at the stop
(e4_run.at_stop_end_check) reads only the stopped call's own record.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_pins_identity_smoke.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import hashlib
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import check_pins as CP  # noqa: E402
import repeat_check as RC  # noqa: E402
import smoke_report as SR  # noqa: E402


def _reg_text(f, h_f, d, h_t, h6_rows=""):
    return (f"## 12. Hashes\n\n### H1 code\n\n| file | sha256 |\n|---|---|\n| `{f}` | `{h_f}` |\n"
            f"| `tree:{d}` | `{h_t}` | 2 files |\n\n### H6 smoke outcome (filled at the freeze)\n\n"
            f"| file | sha256 |\n|---|---|\n{h6_rows}\n## 13. Next\n| `{f}` | `{'0' * 64}` |\n")


def _record(path, reg, campaign):
    path.write_text(json.dumps({"registration": str(reg.resolve()), "registration_sha256": CP.sha256_file(reg),
                                "campaign": campaign}))


def test_check_pins_literals_trees_and_stage_rules(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    d = tmp_path / "tree"
    (d / "sub").mkdir(parents=True)
    (d / "x.py").write_text("x = 1\n")
    (d / "sub" / "y.py").write_text("y = 2\n")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"junk")
    h_f = hashlib.sha256(b"hello\n").hexdigest()
    h_t, n = CP.tree_hash(d)
    assert n == 2                                                      # __pycache__ and .pyc are left out
    reg = tmp_path / "REG.md"
    reg.write_text(_reg_text(f, h_f, d, h_t))
    r = CP.check(reg, "smoke", "e4_test")
    assert r["differences"] == [] and r["tables"]["H1 code"] == {"rows": 2, "match": 2}
    rec = tmp_path / "frozen.json"
    r = CP.check(reg, "a1", "e4_test", frozen=rec)                     # H6 empty AND no freeze record
    assert any("H6" in x for x in r["differences"]) and any(x.startswith("not frozen") for x in r["differences"])
    r = CP.check(reg, "a1", "DRYRUN_x", frozen=rec)                    # dry runs skip H6, never the freeze record
    assert len(r["differences"]) == 1 and r["differences"][0].startswith("not frozen")
    _record(rec, reg, "DRYRUN_x")
    assert CP.check(reg, "a1", "DRYRUN_x", frozen=rec)["differences"] == []
    assert any("not the frozen campaign" in x for x in CP.check(reg, "a1", "DRYRUN_y", frozen=rec)["differences"])
    reg.write_text(reg.read_text() + "\nan edit after the freeze\n")   # the registration changed after the freeze
    assert any("differs from the frozen" in x for x in CP.check(reg, "summary", "DRYRUN_x", frozen=rec)["differences"])
    assert CP.check(reg, "smoke", "DRYRUN_x", frozen=rec)["differences"] == []   # the smoke never reads the record
    assert CP.check(reg, "no_such_stage", "DRYRUN_x").get("malformed") is True
    reg.write_text(_reg_text(f, h_f, d, h_t))
    (d / "sub" / "y.py").write_text("y = 3\n")                         # one changed file anywhere in the tree
    r = CP.check(reg, "smoke", "e4_test")
    assert len(r["differences"]) == 1 and "tree:" in r["differences"][0]
    # a0: a store that exists before the registered a0 started is refused
    camp = "e4_pins_test_tmp"
    cdir = E4 / "runs" / camp
    cdir.mkdir(parents=True, exist_ok=True)
    try:
        (cdir / "replay.sqlite").write_bytes(b"")
        assert any("must start empty" in x for x in CP.check(reg, "a0", camp, frozen=rec)["differences"])
        (cdir / ".a0_started").write_text("")
        assert not any("must start empty" in x for x in CP.check(reg, "a0", camp, frozen=rec)["differences"])
        # the one integrity re-run <campaign>_icr needs runs/<campaign>/integrity_failure.json (section 9.1)
        (d / "sub" / "y.py").write_text("y = 2\n")
        _record(rec, reg, camp)
        fd = CP.frozen_differences(reg, rec, camp + "_icr")
        assert len(fd) == 1 and "integrity_failure.json" in fd[0]
        (cdir / "integrity_failure.json").write_text(json.dumps({"failed_checks": ["IC5"], "cause": "a test",
                                                                  "logged_utc": "2026-10-21T00:00:00Z"}))
        assert any("integrity_failure.json" in x for x in CP.frozen_differences(reg, rec, camp + "_icr"))  # rev 3's form
        assert CP.frozen_differences(reg, rec, camp) == []
    finally:
        for q in cdir.iterdir():
            q.unlink()
        cdir.rmdir()


def _icr_record(cdir, camp, failed, rerun=True, campaign=None):
    (cdir / "integrity_failure.json").write_text(json.dumps({
        "campaign": campaign or camp, "failed": failed, "failed_checks": [f["name"] for f in failed],
        "cause": "a test cause", "logged_utc": "2026-10-22T00:00:00Z", "rerun_required": rerun}))


def test_icr_needs_a_record_whose_evidence_holds_now(tmp_path):
    """Round-3 review M1 (c), one test for each evidence kind: a named check must read False in the campaign's
    summary.json; a named stop (identity_changed, pins_changed) must be a line of its run_e4_gpu.log; a record whose fix
    changes a pinned file (rerun_required false) or that names another campaign refuses the _icr."""
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    reg = tmp_path / "REG.md"
    reg.write_text(_reg_text(f, hashlib.sha256(b"hello\n").hexdigest(), tmp_path / "nodir", "0" * 64))
    camp = "DRYRUN_icrtest"
    cdir = E4 / "runs" / camp
    assert not cdir.exists()
    cdir.mkdir(parents=True)
    rec = tmp_path / "frozen.json"
    _record(rec, reg, camp)
    icr = camp + "_icr"
    try:
        # (1) a failed check
        _icr_record(cdir, camp, [{"kind": "check", "name": "IC5"}])
        assert any("summary.json does not read IC5 False" in x for x in CP.frozen_differences(reg, rec, icr))
        (cdir / "summary.json").write_text(json.dumps({"integrity": {"checks": {"IC5": True}}}))
        assert any("does not read IC5 False" in x for x in CP.frozen_differences(reg, rec, icr))
        (cdir / "summary.json").write_text(json.dumps({"integrity": {"checks": {"IC5": False}}}))
        assert CP.frozen_differences(reg, rec, icr) == []
        _icr_record(cdir, camp, [{"kind": "check", "name": "IC5"}], rerun=False)
        assert any("does not require the re-run" in x for x in CP.frozen_differences(reg, rec, icr))
        _icr_record(cdir, camp, [{"kind": "check", "name": "IC5"}], campaign="DRYRUN_other")
        assert any("names campaign" in x for x in CP.frozen_differences(reg, rec, icr))
        _icr_record(cdir, camp, [{"kind": "check", "name": "IC9"}])
        assert any("unknown integrity check" in x for x in CP.frozen_differences(reg, rec, icr))
        # (2) an identity_changed stop
        line = "2026-10-22T10:00:00Z [a2a3] STOP [identity_changed] run=A2s1_test: e4_run.py failed (exit 2): x"
        _icr_record(cdir, camp, [{"kind": "stop", "name": "identity_changed", "stage": "a2a3", "line": line}])
        assert any("does not hold the stop line" in x for x in CP.frozen_differences(reg, rec, icr))
        (cdir / "run_e4_gpu.log").write_text("2026-10-22T09:00:00Z [a2a3] pins ok (x)\n" + line + "\n")
        assert CP.frozen_differences(reg, rec, icr) == []
        # (3) a pins_changed stop (exit 7)
        line7 = "2026-10-22T11:00:00Z [a2a3] STOP [pins_changed]: check_pins.py found a difference (see x)"
        _icr_record(cdir, camp, [{"kind": "stop", "name": "pins_changed", "stage": "a2a3", "line": line7}])
        assert any("does not hold the stop line" in x for x in CP.frozen_differences(reg, rec, icr))
        with open(cdir / "run_e4_gpu.log", "a") as fh:
            fh.write(line7 + "\n")
        assert CP.frozen_differences(reg, rec, icr) == []
        bad = line7.replace("[a2a3]", "[a1]")                              # the stage must be the named one
        _icr_record(cdir, camp, [{"kind": "stop", "name": "pins_changed", "stage": "a2a3", "line": bad}])
        assert any("does not hold the stop line" in x for x in CP.frozen_differences(reg, rec, icr))
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
        shutil.rmtree(E4 / "runs" / icr, ignore_errors=True)


def test_stop_effects_table_of_section_9_1():
    L = lambda utc, st, msg: f"{utc} [{st}] {msg}"  # noqa: E731
    pins_ok = "pins ok (x)"
    log = "\n".join([
        L("T1", "smoke", "STOP [identity_changed] run=A0_smoke: e4_run.py failed (exit 2): x"),   # a smoke stop: S1
        L("T2", "a0", pins_ok), L("T3", "a0", "start arm A0 split base as c/A0_base"),
        L("T4", "a1", "STOP [restart_same_identity] run=A1_base: e4_run.py failed (exit 2): x"),  # not an integrity stop
        L("T5", "a1", "STOP [refused]: check_pins.py found a difference"),                        # not an integrity stop
    ])
    assert CP.stop_effects(log) == {"events": [], "void_bars": [], "judged_not_available": []}
    e = CP.stop_effects(L("T", "a1", "STOP [identity_changed] run=A1_base: e4_run.py failed (exit 2): x"))
    assert e["void_bars"] == ["E4.2", "E4.3", "E4.5"] and e["events"][0]["run"] == "A1_base"
    e = CP.stop_effects(L("T", "rerun", "STOP [identity_changed] run=A2s3_test_rerun: x failed (exit 2): y"))
    assert e["void_bars"] == ["E4.3", "E4.5"]
    e = CP.stop_effects(L("T", "judge", "STOP [identity_changed] run=judge:A0_base: the judge failed (exit 2): y"))
    assert e["void_bars"] == [] and e["judged_not_available"] == ["A0_base"] and len(e["events"]) == 1
    # pins_changed at stage S: the bars of the stage that ran before S
    def pins_after(prev_lines):
        return CP.stop_effects("\n".join(prev_lines + [L("T9", "a2a3", "STOP [pins_changed]: check_pins.py found x")]))
    e = pins_after([L("T1", "train_guard", pins_ok), L("T2", "train_guard", "rows: ...")])
    assert e["void_bars"] == ["E4.3", "E4.5"] and e["events"][0]["previous_stage"] == "train_guard"
    e = pins_after([L("T1", "a1", pins_ok), L("T2", "a1", "start arm A1 split base as c/A1_base x")])
    assert e["void_bars"] == ["E4.2", "E4.3", "E4.5"] and e["events"][0]["runs_of_previous_stage"] == ["A1_base"]
    e = pins_after([L("T1", "rerun", pins_ok), L("T2", "rerun", "rerun: A1_base has at most 3 ...; no re-run")])
    assert e["void_bars"] == []                                              # a rerun stage that re-ran nothing
    e = pins_after([L("T1", "judge", pins_ok), L("T2", "judge", "judged A1_base (self-judged, unvalidated): {}")])
    assert e["void_bars"] == [] and e["judged_not_available"] == ["A1_base"]
    e = pins_after([L("T1", "smoke", pins_ok), L("T2", "smoke", "start arm A0 split train as S/A0_smoke x")])
    assert e["void_bars"] == [] and e["events"][0]["previous_stage"] == "smoke"   # the freeze check came after it
    assert CP.stop_effects(L("T", "a0", "STOP [pins_changed]: x"))["void_bars"] == []   # nothing ran before it
    assert CP.bars_fed_by_run("judge:A3_test") == () and CP.bars_fed_by_run("something_else") == CP.BARS


def test_check_flags_pins_changed_only_for_a_changed_pin(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    (tmp_path / "emptydir").mkdir()
    reg = tmp_path / "REG.md"
    reg.write_text(_reg_text(f, hashlib.sha256(b"hello\n").hexdigest(), tmp_path / "emptydir",
                             CP.tree_hash(tmp_path / "emptydir")[0]))
    r = CP.check(reg, "a1", "DRYRUN_flagtest", frozen=tmp_path / "missing.json")
    assert r["differences"] and r["pins_changed"] is False                  # not frozen: [refused], not pins_changed
    rec = tmp_path / "frozen.json"
    _record(rec, reg, "DRYRUN_flagtest")
    assert CP.check(reg, "a1", "DRYRUN_flagtest", frozen=rec)["pins_changed"] is False
    f.write_text("changed\n")
    r = CP.check(reg, "a1", "DRYRUN_flagtest", frozen=rec)
    assert r["pins_changed"] is True and any("on disk" in x for x in r["differences"])
    f.write_text("hello\n")
    reg.write_text(reg.read_text() + "\nedited after the freeze\n")
    assert CP.check(reg, "a1", "DRYRUN_flagtest", frozen=rec)["pins_changed"] is True


def test_smoke_extension_rule():
    """Revision 4 (round-3 review m6): the first 10 train tasks; one more in split order while their label candidates
    are fewer than 10; at most 20."""
    assert SR.extension_k([1] * 10) == 10                                     # 10 candidates on 10 tasks
    assert SR.extension_k([0] * 9 + [10]) == 10
    assert SR.extension_k([0] * 10 + [3, 3, 4, 9]) == 13                      # the smallest k > 10 with 10
    assert SR.extension_k([0] * 20) == 20 and SR.extension_k([0] * 25) == 20  # the cap: S8 then reads "not measured"
    assert SR.extension_k([0] * 12) is None                                   # fewer tasks ran than the rule needs
    assert SR.extension_k([1, 1, 1, 1], k0=3, min_c=4, k_max=5) == 4          # the dry run's hooks
    order = SR.train_order()
    assert order[:10] == ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11"] and len(order) == 74
    assert SR.next_smoke_task(order[:10], order) == "13" and SR.next_smoke_task(["0", "1", "2"], order) == "3"
    assert SR.next_smoke_task(order, order) is None


def test_h6_must_name_the_five_smoke_files():
    camp = "e4reg_00000000"
    sd = E4 / "runs" / f"SMOKE_{camp}"
    five = [(str(sd / x), "0" * 64) for x in CP.H6_FILES]
    assert CP.h6_differences([], camp)[0].startswith("H6 (the smoke outcome) is empty")
    assert CP.h6_differences(five, camp) == []
    assert CP.h6_differences(five, camp + "_icr") == []                 # the re-run uses the frozen campaign's smoke
    rel = [(f"runs/SMOKE_{camp}/{x}", "0" * 64) for x in CP.H6_FILES]
    assert CP.h6_differences(rel, camp) == []                            # relative to the E4 folder is the same
    assert any("lacks" in x for x in CP.h6_differences(five[:4], camp))
    assert any("not section 11.4" in x for x in CP.h6_differences(five + [("/tmp/other.json", "0" * 64)], camp))
    assert any("lacks" in x for x in CP.h6_differences(five, "e4reg_11111111"))   # another campaign's smoke


def test_venv_hash_pins_every_dist_info(tmp_path):
    site = tmp_path / "site-packages"
    (site / "pkg_a-1.0.dist-info").mkdir(parents=True)
    (site / "pkg_a-1.0.dist-info" / "RECORD").write_text("pkg_a/__init__.py,sha256=x,1\n")
    (site / "pkg_a").mkdir()
    (site / "pkg_a" / "__init__.py").write_text("")
    (site / "__pycache__").mkdir()
    h0, n0 = CP.venv_hash(site)
    assert n0 == 1 and CP.venv_hash(site) == (h0, 1)                    # stable
    assert CP.spec_hash(f"venv:{site}") == (h0, 1) and CP.spec_hash(f"venv:{tmp_path / 'nope'}")[0] == "MISSING"
    (site / "__pycache__" / "x.pyc").write_bytes(b"cache")                # bytecode caches are ignored
    assert CP.venv_hash(site)[0] == h0
    (site / "pkg_a" / "__init__.py").write_text("x = 1")                  # an in-place edit is NOT seen (dist: is)
    assert CP.venv_hash(site)[0] == h0
    hs = {h0}
    (site / "pkg_b-2.0.dist-info").mkdir()                                # an install
    (site / "pkg_b-2.0.dist-info" / "RECORD").write_text("pkg_b.py,sha256=y,1\n")
    hs.add(CP.venv_hash(site)[0])
    (site / "pkg_b-2.0.dist-info" / "RECORD").write_text("pkg_b.py,sha256=z,1\n")   # an upgrade in place
    hs.add(CP.venv_hash(site)[0])
    (site / "evil.pth").write_text("import os\n")                        # a startup hook
    hs.add(CP.venv_hash(site)[0])
    (site / "evil.pth").write_text("import sys\n")
    hs.add(CP.venv_hash(site)[0])
    (site / "loose_module.py").write_text("")                             # a module with no dist-info
    hs.add(CP.venv_hash(site)[0])
    assert len(hs) == 6 and CP.venv_hash(site)[1] == 2
    lines = CP.venv_lines(site)
    assert ("entry", "loose_module.py", "file") in lines and lines[0][0] in ("dist", "pth", "entry")


def test_venv_rows_of_both_real_venvs_are_stable():
    for site in (f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/lib/python3.12/site-packages",
                 f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/lib/python3.11/site-packages"):
        h1, n1 = CP.venv_hash(Path(site))
        h2, n2 = CP.venv_hash(Path(site))
        assert h1 == h2 and n1 == n2 > 50


def test_dist_hash_reads_the_record():
    site = Path(f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/lib/python3.12/site-packages")
    di = next(site.glob("loguru-*.dist-info"))
    h1, n1 = CP.dist_hash(di)
    h2, n2 = CP.dist_hash(di)
    assert h1 == h2 and n1 > 5 and len(h1) == 64


class _Props(BaseHTTPRequestHandler):
    props = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = {"/props": _Props.props, "/v1/models": {"data": [{"id": "m"}]}, "/health": {"status": "ok"}}.get(self.path)
        b = json.dumps(body).encode()
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def test_server_identity_detects_every_difference(tmp_path, monkeypatch):
    import e4_run

    gg = tmp_path / "m.gguf"
    gg.write_bytes(b"GGUF fake")
    exe = Path(sys.executable).resolve()
    tmpl = "tmpl"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Props)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    argv = ["/bin/llama-server", "--port", str(port)]
    pins = {"port": port, "props": {"total_slots": 1, "n_ctx_per_slot": 65536, "build_info": "b1-4c3c581",
                                    "model_alias": "m", "model_path": str(gg),
                                    "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()},
            "models_ids": ["m"], "gguf_sha256": hashlib.sha256(b"GGUF fake").hexdigest(), "exe_path": str(exe),
            "exe_sha256": e4_run.sha256_file(exe), "build_dir_files_sha256": {}, "argv": argv}
    good = {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 1, "build_info": "b1-4c3c581",
            "model_alias": "m", "model_path": str(gg), "chat_template": tmpl}
    import os

    monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv)])
    base = f"http://127.0.0.1:{port}/v1"
    try:
        _Props.props = good
        rec, diff = e4_run.server_identity(base, pins)
        assert diff == [] and rec["gguf_sha256"] == pins["gguf_sha256"] and rec["props"]["total_slots"] == 1
        for k, v in (("total_slots", 2), ("chat_template", "other"), ("build_info", "b1-0000000"),
                     ("model_path", "/nonexistent.gguf")):
            _Props.props = {**good, k: v}
            assert e4_run.server_identity(base, pins)[1], k
        _Props.props = {**good, "default_generation_settings": {"n_ctx": 32768}}
        assert any("n_ctx_per_slot" in x for x in e4_run.server_identity(base, pins)[1])
        _Props.props = good
        monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv + ["-np", "2"])])
        assert any("argv" in x for x in e4_run.server_identity(base, pins)[1])
        monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [])
        assert any("exactly 1" in x for x in e4_run.server_identity(base, pins)[1])
        gg.write_bytes(b"GGUF other")
        monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv)])
        assert any("GGUF sha256" in x for x in e4_run.server_identity(base, pins)[1])
        # revision 3: the light check when a call ends (no re-hashing) against the pins and the call's own start
        gg.write_bytes(b"GGUF fake")
        start, d0 = e4_run.server_identity(base, pins)
        assert d0 == []
        end, kind = e4_run.identity_end_check(base, pins, start)
        assert kind == "same" and end["differences"] == [] and "exe_sha256" not in end
        import subprocess

        child = subprocess.Popen([str(exe), "-I", "-c", "import time; time.sleep(60)"])   # a restarted server: a new
        try:                                                                              # pid, the same executable
            monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(child.pid, argv)])
            end, kind = e4_run.identity_end_check(base, pins, start)
            assert kind == "restart_same_identity" and any("pid" in x for x in end["differences"]), end
        finally:
            child.kill()
            child.wait(timeout=10)
        monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv)])
        _Props.props = {**good, "total_slots": 2}                          # changed during the call
        end, kind = e4_run.identity_end_check(base, pins, start)
        assert kind == "identity_changed"
        _Props.props = good
        monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv + ["-np", "2"])])
        assert e4_run.identity_end_check(base, pins, start)[1] == "identity_changed"
    finally:
        srv.shutdown()


def test_keep_earlier_manifest(tmp_path):
    import e4_run

    assert e4_run.keep_earlier_manifest(tmp_path) is None
    (tmp_path / "manifest.json").write_text("{\"call\": 1}")
    assert e4_run.keep_earlier_manifest(tmp_path) == "manifest.1.json" and not (tmp_path / "manifest.json").exists()
    (tmp_path / "manifest.json").write_text("{\"call\": 2}")
    assert e4_run.keep_earlier_manifest(tmp_path) == "manifest.2.json"
    assert json.loads((tmp_path / "manifest.1.json").read_text()) == {"call": 1}


def test_budget_formula_and_cut_order():
    b = SR.budget(c=30, t=2.0, f1=0.5, tj=5.0, slot=None)
    m = b["minutes_by_arm"]
    assert abs(m["A0"] - (114 * 30 * 2.0 + 40 * 5.0) / 60) < 1e-9
    assert abs(m["A1"] - (114 * 30 * 0.5 * 2.0 + 40 * 5.0) / 60) < 1e-9
    assert abs(m["A3t"] - (40 * 30 * 0.30 * 2.0 + 11 * 5.0) / 60) < 1e-9
    full = b["minutes_full"]
    assert b["decision"].startswith("pending")
    assert SR.budget(30, 2.0, 0.5, 5.0, slot=full)["cut"] == []
    r = SR.budget(30, 2.0, 0.5, 5.0, slot=full - m["A3t"])
    assert r["cut"] == ["A3t"] and r["fits"]
    r = SR.budget(30, 2.0, 0.5, 5.0, slot=full - m["A3t"] - 1e-6)
    assert r["cut"] == ["A3t", "A2s2", "A2s3"] and r["fits"]
    r = SR.budget(30, 2.0, 0.5, 5.0, slot=full - m["A3t"] - m["A2s2"] - m["A2s3"] - 1e-6)
    assert r["fits"] is False and r["decision"].startswith("STOP") and r["cut"] is None


def test_matcher_gate_s8_revision3():
    """S8 cannot pass unchecked: fewer than 10 candidates = NO-GO; missing labels, labeller or any label = PENDING."""
    cands = [{"id": str(i), "matcher_yes": i < 10} for i in range(12)]
    L = lambda lab, who="judged by an AI model, not a person": {"labeller": who, "labels": lab}  # noqa: E731
    labels = {str(i): (i < 9 or i == 11) for i in range(12)}            # tp 9, fp 1 (9), fn 1 (11)
    r = SR.matcher_scores(cands, L(labels))
    assert (r["tp"], r["fp"], r["fn"]) == (9, 1, 1) and r["gate"] == "pass" and r["pass"] is True
    assert r["labeller"] == "judged by an AI model, not a person"
    assert SR.matcher_scores(cands, L(labels, "a person"))["pass"] is True
    labels8 = {str(i): (i < 8 or i == 11) for i in range(12)}           # precision 8/10 < 0.90
    assert SR.matcher_scores(cands, L(labels8))["gate"] == "FAIL" and SR.matcher_scores(cands, L(labels8))["pass"] is False
    r = SR.matcher_scores(cands[:9], L(labels))                          # 9 candidates: not measured = NO-GO
    assert r["pass"] is False and r["gate"].startswith("not measured") and "NO-GO" in r["gate"]
    assert SR.matcher_scores([], None)["pass"] is False
    assert SR.matcher_scores(cands, None)["pass"] is None                # no label file: PENDING
    assert SR.matcher_scores(cands, labels)["pass"] is None              # revision 2's bare dict: no labeller
    assert SR.matcher_scores(cands, L(labels, "someone else"))["pass"] is None
    part = {k: v for k, v in labels.items() if k != "3"}
    r = SR.matcher_scores(cands, L(part))                                # one candidate unlabelled: PENDING
    assert r["pass"] is None and r["unlabelled"] == ["3"]
    assert SR.matcher_scores(cands, L(labels, SR.DRY_LABELLER))["pass"] is None          # dry labeller: dry runs only
    assert SR.matcher_scores(cands, L(labels, SR.DRY_LABELLER), allow_dry_labeller=True)["pass"] is True
    assert SR.matcher_scores(cands[:3], L(labels), min_candidates=3)["candidates"] == 3
    bad = {str(i): True for i in range(12)}                             # recall 10/12 = 0.83 passes
    assert SR.matcher_scores(cands, L(bad))["gate"] == "pass"
    worse = [{"id": str(i), "matcher_yes": i < 9} for i in range(12)]   # tp 9 of 12 yes: recall 0.75 < 0.80
    assert SR.matcher_scores(worse, L(bad))["gate"] == "FAIL"


def test_tool_health():
    def reply(content=None, calls=(), rc=None):
        m = {"role": "assistant", "content": content,
             "tool_calls": [{"id": "x", "type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]}
        if rc:
            m["reasoning_content"] = rc
        return json.dumps({"choices": [{"message": m}]})

    rows = [("agent", reply(calls=[("get_order_details", '{"order_id": "#W1"}')])),
            ("agent", reply(calls=[("get_order_details", "{bad json")])),
            ("agent", reply(calls=[("no_such_tool", "{}")])),
            ("agent", reply(content="<tool_call>{...}</tool_call>")),
            ("agent", reply(content="Sure.", calls=[("calculate", '{"expression": "1+1"}'), ("calculate", "{}")])),
            ("agent", reply(content="", rc="thinking")),
            ("user", reply(content="hi"))]
    h = SR.tool_health(rows)
    assert h["replies"] == 6 and h["tool_calls"] == 5 and h["parsed_tool_calls"] == 3
    assert h["unparsed_arguments"] == 1 and h["unknown_tool_names"] == 1 and h["tool_calls_written_as_text"] == 1
    assert h["text_and_tool_call"] == 1 and h["multi_call_messages"] == 1 and h["empty_replies"] == 1
    assert h["reasoning_text"] == 1 and abs(h["parse_rate"] - 3 / 6) < 1e-12


def test_repeat_check_upper_bound():
    assert abs(RC.upper95(0, 20) - (1 - 0.05 ** (1 / 20))) < 1e-6     # about 13.9%: why 20 requests were too few
    assert abs(RC.upper95(0, 100) - (1 - 0.05 ** (1 / 100))) < 1e-6
    assert RC.upper95(3, 100) > 0.03 and RC.upper95(100, 100) == 1.0 and RC.upper95(0, 0) is None
    picked = RC.pick({"agent": [(str(i), "q", "r") for i in range(80)], "user": [(f"u{i}", "q", "r") for i in range(60)]},
                     50, 50, 20261021)
    assert sum(1 for p in picked if p[0] == "agent") == 50 and sum(1 for p in picked if p[0] == "user") == 50
    assert [p[0] for p in picked[:10]] != ["agent"] * 10                # shuffled across roles


# ------------------------------------------------------------------ revision 5 (round-4 review M1 (e))
def test_stage_attempts_count_only_failed_attempts():
    L = lambda t, st, msg: f"2026-10-22T{t}Z [{st}] {msg}\n"  # noqa: E731
    log = (L("10:00:00", "a0", "attempt 1 of 3 (failed attempts so far: 0; section 11.6)")
           + L("10:01:00", "a0", "STOP [client] run=A0_base: e4_run.py for c/A0_base failed (exit 1): x")
           + L("10:02:00", "a0", "STOP [slot_end] run=A0_base: the operator stopped stage a0 (SIGTERM): ...")
           + L("10:03:00", "a0", "STOP [server] run=A0_base: e4_run.py for c/A0_base failed (exit 2): x")
           + L("10:04:00", "a0", "STOP [restart_same_identity] run=A0_base: e4_run.py failed (exit 2): x")
           + L("10:05:00", "a0", "STOP [cpu_gate]: cgroup still above 3.5 cores")
           + L("10:06:00", "a0", "STOP [refused]: check_pins.py found a difference")
           + L("10:06:30", "a0", "STOP [identity_changed] run=A0_base: e4_run.py failed (exit 2): x")
           + L("10:07:00", "smoke", "STOP [client]: smoke_report.py failed")          # the smoke has no attempt rule
           + L("10:08:00", "rerun", "STOP [client] run=A0_base_rerun: e4_run.py failed (exit 1): x")
           + L("10:09:00", "rerun", "STOP [client]: a step of stage rerun failed with exit 1 before any tagged stop")
           + L("10:10:00", "rerun", "STOP [client] run=A1_base_rerun: e4_run.py failed (exit 1): x")
           + L("10:11:00", "train_guard", "STOP [train_guard]: train_guard.py failed (exit 1)")
           + L("10:11:30", "train_guard", "STOP [milp]: the MILP check failed for seed 1 (IC2(b))")
           + L("10:12:00", "rerun", "STOP [client] run=A2s1_test_rerun: e4_run.py failed (exit 1): x"))
    sa = CP.stage_attempts(log)
    assert sa["failed"] == {"a0": 1, "rerun": 3, "train_guard": 1, "rerun_test": 1} and sa["final"] == ["rerun"]
    assert CP.failed_attempts(log, "a0") == 1 and CP.failed_attempts(log, "a2a3") == 0
    assert CP.failed_attempts(log, "rerun") == 1          # after train_guard: the second rerun, the test arms
    log2 = log + L("10:13:00", "a0", "STOP [client] run=A0_base: x") + L("10:14:00", "a0", "STOP [client]: y")
    assert CP.stage_attempts(log2)["final"] == ["a0", "rerun"] and CP.failed_attempts(log2, "a0") == 3
    assert CP.ATTEMPTS_MAX == 3 and CP.stage_attempts("") == {"failed": {}, "final": [], "lines": {},
                                                              "after_train_guard": False}


def _stub_and_pins(tmp_path, tmpl="tmpl"):
    """The in-process /props stub of test_server_identity_detects_every_difference, its pins and its good /props."""
    import e4_run

    gg = tmp_path / "m.gguf"
    gg.write_bytes(b"GGUF fake")
    exe = Path(sys.executable).resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Props)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    argv = ["/bin/llama-server", "--port", str(port)]
    pins = {"port": port, "props": {"total_slots": 1, "n_ctx_per_slot": 65536, "build_info": "b1-4c3c581",
                                    "model_alias": "m", "model_path": str(gg),
                                    "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()},
            "models_ids": ["m"], "gguf_sha256": hashlib.sha256(b"GGUF fake").hexdigest(), "exe_path": str(exe),
            "exe_sha256": e4_run.sha256_file(exe), "build_dir_files_sha256": {}, "argv": argv}
    good = {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 1, "build_info": "b1-4c3c581",
            "model_alias": "m", "model_path": str(gg), "chat_template": tmpl}
    return srv, f"http://127.0.0.1:{port}/v1", pins, good, argv


def test_the_full_start_check_comes_before_the_deferred_end_check(tmp_path, monkeypatch, capsys):
    """Round-5 review Major 1 (b), (d): a run whose earlier call has no end record (a kill -9) meets, at its next start,
    a server that is not the pinned one: exit 2 'not the pinned one' WITHOUT '(identity_changed)' (run_e4_gpu.sh tags
    it [server]: no integrity event, not counted), the record byte for byte unchanged. With the pinned server back after
    a restart (a new pid), the deferred check writes 'restart_same_identity' into the record and the run goes on. A
    record that cannot be read is still 'identity_changed'; a record with its end is left alone."""
    import os
    import subprocess

    import pytest
    import e4_run

    srv, base, pins, good, argv = _stub_and_pins(tmp_path)
    monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv)])
    child = None
    try:
        _Props.props = good
        start, d0 = e4_run.server_identity(base, pins)
        assert d0 == []
        rec = tmp_path / "A0_base" / "manifest.json"
        rec.parent.mkdir()
        rec.write_text(json.dumps({"utc": "2026-10-22T10:00:00Z", "server_identity": start}, indent=1))
        before = rec.read_bytes()
        _Props.props = {**good, "chat_template": "a template that is not the pinned one"}   # not the pinned one
        with pytest.raises(SystemExit) as e:
            e4_run.start_checks(base, pins, rec, "run")
        err = capsys.readouterr().err
        assert e.value.code == 2 and "the server is not the pinned one" in err and "(identity_changed)" not in err
        assert rec.read_bytes() == before                                  # the record is untouched
        _Props.props = good                                                # the pinned server back, restarted
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"])
        monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(child.pid, argv)])
        ident = e4_run.start_checks(base, pins, rec, "run")
        assert ident["differences"] == [] and ident["processes"][0]["pid"] == child.pid           # the run goes on
        end = json.loads(rec.read_text())["server_identity_end"]
        assert end["kind"] == "restart_same_identity" and end["deferred"] is True and "at_stop" not in end
        after = rec.read_bytes()
        assert e4_run.start_checks(base, pins, rec, "run")["differences"] == [] and rec.read_bytes() == after
        rec.write_text("{not json")                                        # unreadable: still identity_changed
        with pytest.raises(SystemExit) as e:
            e4_run.start_checks(base, pins, rec, "run")
        assert e.value.code == 2 and "(identity_changed)" in capsys.readouterr().err
    finally:
        if child is not None:
            child.kill()
            child.wait(timeout=10)
        srv.shutdown()


def test_the_end_check_at_the_stop_reads_only_the_stopped_calls_own_record(tmp_path, monkeypatch):
    """Round-5 review Major 1 (a): e4_run.at_stop_end_check writes the light end check into the stopped call's own
    start record ("at_stop": true): only a readable record with its start record, without an end, written at or after
    the client's start; any other record is left untouched (the deferred check at the next start takes it)."""
    import os

    import e4_run

    srv, base, pins, good, argv = _stub_and_pins(tmp_path)
    monkeypatch.setattr(e4_run, "processes_on_port", lambda p: [(os.getpid(), argv)])
    try:
        _Props.props = good
        start, _ = e4_run.server_identity(base, pins)
        rec = tmp_path / "judged.start.json"
        assert e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:00Z") == (None, [])        # no record
        rec.write_text(json.dumps({"utc": "2026-10-22T10:00:05Z", "server_identity": start}))
        before = rec.read_bytes()
        assert e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:06Z") == (None, [])        # an older one
        assert rec.read_bytes() == before
        kind, diffs = e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:05Z")               # its own
        end = json.loads(rec.read_text())["server_identity_end"]
        assert (kind, diffs) == ("same", []) and end["kind"] == "same" and end["at_stop"] is True
        assert "deferred" not in end
        after = rec.read_bytes()
        assert e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:00Z") == (None, [])        # has its end
        assert rec.read_bytes() == after
        rec.write_text(json.dumps({"utc": "2026-10-22T10:00:05Z", "server_identity": start}))
        _Props.props = {**good, "total_slots": 2}                          # the server changed during the call
        kind, diffs = e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:00Z")
        assert kind == "identity_changed" and any("total_slots" in x for x in diffs)
        assert json.loads(rec.read_text())["server_identity_end"]["at_stop"] is True
        rec.write_text("{cut mid-write")                                   # unreadable: left for the deferred check
        assert e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:00Z") == (None, [])
        assert rec.read_text() == "{cut mid-write"
        rec.write_text(json.dumps({"utc": "2026-10-22T10:00:05Z"}))         # no start record: left alone too
        assert e4_run.at_stop_end_check(base, pins, rec, "2026-10-22T10:00:00Z") == (None, [])
    finally:
        srv.shutdown()


def test_deferred_end_check_of_a_call_stopped_before_its_end(tmp_path, monkeypatch):
    import e4_run

    m = tmp_path / "manifest.json"
    assert e4_run.deferred_end_check("b", {}, m) is None                         # no earlier call
    m.write_text(json.dumps({"server_identity": {"props": 1}, "server_identity_end": {"kind": "same"}}))
    assert e4_run.deferred_end_check("b", {}, m) is None                         # it has its end check
    for kind in ("same", "restart_same_identity", "identity_changed"):
        m.write_text(json.dumps({"server_identity": {"props": 1}}))
        seen = []
        monkeypatch.setattr(e4_run, "identity_end_check",
                            lambda b, p, st, k=kind: (seen.append(st), ({"kind": k, "differences": []}, k))[1])
        assert e4_run.deferred_end_check("b", {}, m) == kind and seen == [{"props": 1}]
        end = json.loads(m.read_text())["server_identity_end"]
        assert end["deferred"] is True and end["kind"] == kind                   # kept on record
    m.write_text("{not json")
    assert e4_run.deferred_end_check("b", {}, m) == "identity_changed"           # cannot show the end was pinned
    m.write_text(json.dumps({"utc": "x"}))
    assert e4_run.deferred_end_check("b", {}, m) == "identity_changed"           # no start record
    # revision 6 (round-5 review Major 1 (d)): the order is inverted: the full start check FIRST, then the deferred
    # check (e4_run.start_checks), in e4_run.py and in the judge; neither calls the deferred check on its own
    src = (E4 / "scripts/e4_run.py").read_text()
    f = src[src.index("def start_checks("):src.index("def keep_earlier_manifest(")]
    f = f[f.index('"""', f.index('"""') + 3) + 3:]                              # the code, after the docstring
    i, j = f.index("ident, diff = server_identity(base, pins)"), f.index("if deferred_end_check(base, pins, rec_path)")
    assert i < f.index("if diff:") < f.index('die(2, "the server is not the pinned one: "') < j
    assert "(identity_changed)" in f[j:j + 300] and "(identity_changed)" not in f[:j]
    main = src[src.index("def main():"):]
    assert 'ident = start_checks(args.base, pins, out / "manifest.json", "run")' in main
    assert "deferred_end_check(" not in main and "server_identity(" not in main
    js = (E4 / "scripts/judge_posthoc.py").read_text()
    jm = js[js.index("def main():"):]
    assert 'ident = start_checks(a.base, pins, sp, "judge run")' in jm
    assert "deferred_end_check(" not in jm and "server_identity(" not in jm
