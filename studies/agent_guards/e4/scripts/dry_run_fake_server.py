"""Offline END-TO-END dry run of every stage of scripts/run_e4_gpu.sh (smoke to summary) against a FAKE local server
(scripts/fake_llama_server.py: a scripted HTTP stub in its own process; no model, no GPU). It is not llama-server and
never stands in for a result (review issue 3: run every stage against the stub on 2-3 tasks before the freeze).

Checks, written to --out:
  refusals: no server -> exit 2; a non-local endpoint -> exit 2; /health not ok -> exit 2; a server that differs from
            the pins (one wrong chat-template hash) -> exit 2; A2 with a stand-in model and no --allow-stand-in -> exit 3;
            (revision 3) a registered stage before the freeze -> exit 7; SLOT_MINUTES given to a stage -> exit 1;
            (revision 4) STAGE=smoke once the dry freeze record exists -> exit 7 (round-3 review m3); the integrity
            re-run DRYRUN_<tag>_icr without an integrity record -> exit 7, and integrity_failure.py on the clean
            campaign (no bar void) -> exit 1 (round-3 review M1);
  the smoke's extension rule (revision 4, round-3 review m6): smoke tasks 0 1 2 with SMOKE_MIN_CANDIDATES=4 and
            SMOKE_MAX_TASKS=5, so A0 runs on train tasks one at a time until 4 label candidates exist (S0 checks it);
  the freeze flow (revision 3, registration 11.4-11.5): STAGE=smoke with no slot (its decision must read PENDING);
            dry labels (labeller "dry-run test labels ...") and smoke_report.py again with the slot (decision GO);
            S8 with too few candidates (a lower bound above the count) must read STOP; freeze_e4.py --dry-campaign
            writes the dry record, and a second call must be refused (write-once);
  pipeline: STAGE=a0, a1, rerun, train_guard, a2a3, rerun, judge, summary with CAMPAIGN=DRYRUN_<tag> and FROZEN = the
            dry record, its smoke in SMOKE_DRYRUN_<tag> (its own store), smoke tasks 0 1 2, base tasks 0 1 2 3 4 5 9
            12, test tasks 5 9 12, MIN_CLASS=1 and DRY_FORCE_GO=1 (test hooks, refused for any other campaign);
  the operator's stop (revision 5, round-4 review M1 (a)): STAGE=a0 of the main campaign with the stub slowed down;
            TERM to run_e4_gpu.sh once A0 has finished at least one task: exit 8, a 'STOP [slot_end] run=A0_base' line,
            no failed attempt and no integrity stop counted; revision 6 (round-5 review Major 1 (a)): the stopped
            call's end check AT THE STOP, written into its manifest ("same", at_stop) and logged;
  a resume that meets a server that is not the pinned one (revision 6, round-5 review Major 1 (b)): a0 again, killed
            with SIGKILL (script and client, its process group) once it has finished one more task, so its call's
            manifest keeps no end; then a0 with the stub's /props changed (POST /stub/props: a chat template that is
            not the pinned one): exit 2 tagged [server], that record byte for byte unchanged, no integrity stop and no
            failed attempt; then the stub's /props restored and a0 again ("the next slot", exit 0): the full start
            check passes first, then the killed call's record gets its deferred end check ("same", deferred), and the
            pipeline goes on from there;
  the 'not testable' path (revision 3, re-review item 1): DRYRUN_<tag>_nt holds copies of A0_base, A1_base and the
            store, its own dry record; STAGE=train_guard with MIN_CLASS=1000 (train_guard.py exit 3) must exit 0 and
            write gobar.json "not testable"; a2a3 must refuse with exit 0; judge and summary must run; the summary
            must read E4.3 and E4.5 "not testable", IC2(b) not applicable, and no flag on E4.2;
  outputs:  the smoke report's gates (blind), the rows (veto-clean, deduplicated), the go-bar with its power block,
            the MILP summaries, the summary's table and integrity checks, the judge's count-mismatch errors, the
            server-identity end checks, and the stub's request counts: the replay-only runs (STOCK and the two
            re-runs) must send ZERO requests to the stub.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B dry_run_fake_server.py --reg <registration with H tables> \
       --out results/dry_run_rev6.json [--cpu-gate 3.5]
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
PY = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as r:
        return json.loads(r.read().decode())


def post(url, obj):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=5) as r:
        return json.loads(r.read().decode())


def e4_run(args_list, env):
    p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/e4_run.py")] + args_list, env=env, capture_output=True,
                       text=True, timeout=900)
    return p.returncode, (p.stderr or "")[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True, help="the registration whose H tables check_pins.py compares")
    ap.add_argument("--tag", default="e4rev6")
    ap.add_argument("--cpu-gate", default="3.5", help="CPU_GATE for every stage (a dry-run hook; at most 4.5)")
    ap.add_argument("--cpu-gate-later", default=None, help="the CPU_GATE from --cpu-gate-switch on (at most 4.5)")
    ap.add_argument("--cpu-gate-switch", default=None, help="UTC time, e.g. 2026-10-21T09:05:00Z")
    ap.add_argument("--standin-model", default=str(E4 / "models/standin_rev2/tm_seed1.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    camp, scamp = f"DRYRUN_{a.tag}", f"SMOKE_DRYRUN_{a.tag}"
    ncamp, nscamp = f"DRYRUN_{a.tag}_nt", f"SMOKE_DRYRUN_{a.tag}_nt"
    for c in (camp, scamp, ncamp, nscamp, f"DRYRUN_{a.tag}_refusals", f"DRYRUN_{a.tag}_icr"):
        shutil.rmtree(E4 / "runs" / c, ignore_errors=True)
    work = E4 / ".tmp" / f"dryrun_{a.tag}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    gguf = work / "fake-model.gguf"
    gguf.write_bytes(b"GGUF-fake-model-of-the-E4-dry-run\n" * 64)
    port = free_port()
    argv = [PY, "-I", "-B", str(E4 / "scripts/fake_llama_server.py"), "--port", str(port), "--model-path", str(gguf)]
    from fake_llama_server import TEMPLATE

    pins = {"what": "DRY RUN pins for the fake stub (not the registered server)", "port": port,
            "props": {"total_slots": 1, "n_ctx_per_slot": 65536, "build_info": "b1-4c3c581", "model_alias": "fake-model",
                      "model_path": str(gguf), "chat_template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest()},
            "models_ids": ["fake-model"], "gguf_sha256": sha(gguf), "exe_path": os.path.realpath(PY),
            "exe_sha256": sha(os.path.realpath(PY)), "build_dir_files_sha256": {}, "argv": argv}
    pins_path = work / "pins_dry.json"
    pins_path.write_text(json.dumps(pins, indent=1))
    bad = json.loads(json.dumps(pins))
    bad["props"]["chat_template_sha256"] = "0" * 64
    bad_path = work / "pins_dry_wrong_template.json"
    bad_path.write_text(json.dumps(bad, indent=1))
    env = dict(os.environ)
    res = {"port": port, "pins_sha256": sha(pins_path), "refusals": {}, "stages": {}}
    base = f"http://127.0.0.1:{port}/v1"
    rcamp = f"DRYRUN_{a.tag}_refusals"
    res["refusals"]["no_server"] = e4_run(["--arm", "A1", "--base", "http://127.0.0.1:9/v1", "--campaign", rcamp,
                                           "--expect-server", str(pins_path)], env)
    res["refusals"]["non_local"] = e4_run(["--arm", "A1", "--base", "http://10.0.0.1:8080/v1", "--campaign", rcamp,
                                           "--expect-server", str(pins_path)], env)
    srv = subprocess.Popen(argv, stdout=open(work / "stub.out", "w"), stderr=open(work / "stub.err", "w"), env=env,
                           cwd=str(work))
    try:
        for _ in range(200):
            try:
                get(f"http://127.0.0.1:{port}/health")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        post(f"http://127.0.0.1:{port}/stub/health", {"status": "loading model"})
        res["refusals"]["health_not_ok"] = e4_run(["--arm", "A1", "--base", base, "--campaign", rcamp,
                                                   "--expect-server", str(pins_path)], env)
        post(f"http://127.0.0.1:{port}/stub/health", {"status": "ok"})
        res["refusals"]["server_not_pinned"] = e4_run(["--arm", "A1", "--base", base, "--campaign", rcamp,
                                                       "--expect-server", str(bad_path), "--split", "train",
                                                       "--task-ids", "0"], env)
        if Path(a.standin_model).exists():
            res["refusals"]["standin_model"] = e4_run(["--arm", "A2", "--base", base, "--campaign", rcamp,
                                                       "--expect-server", str(pins_path), "--learned",
                                                       a.standin_model, "--split", "train", "--task-ids", "0"], env)
        res["refusals_ok"] = (all(v[0] == 2 for k, v in res["refusals"].items() if k != "standin_model")
                              and res["refusals"].get("standin_model", [3])[0] == 3)
        frozen = E4 / "runs" / camp / "E4_frozen_dry.json"
        senv = dict(env, CAMPAIGN=camp, PORT=str(port), PINS=str(pins_path), REG=a.reg, SMOKE_TASKS="0 1 2",
                    SMOKE_MIN_TASKS="3", SMOKE_MIN_CANDIDATES="4", SMOKE_MAX_TASKS="5",
                    BASE_TASKS="0 1 2 3 4 5 9 12", TEST_TASKS="5 9 12",
                    MIN_CLASS="1", DRY_FORCE_GO="1", FROZEN=str(frozen), CPU_GATE=a.cpu_gate)
        posts = {}

        def gate_now():
            if a.cpu_gate_later and a.cpu_gate_switch and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) >= \
                    a.cpu_gate_switch:
                return a.cpu_gate_later
            return a.cpu_gate

        def stage(name, envx, key=None):
            before = get(f"http://127.0.0.1:{port}/stub/stats")["posts"]
            t1, g = time.time(), gate_now()
            p = subprocess.run(["bash", str(E4 / "scripts/run_e4_gpu.sh")],
                               env=dict(envx, STAGE=name, CPU_GATE=g),
                               capture_output=True, text=True, timeout=3600)
            after = get(f"http://127.0.0.1:{port}/stub/stats")["posts"]
            key = key or (name if name not in res["stages"] else name + "_2")
            res["stages"][key] = {"rc": p.returncode, "seconds": round(time.time() - t1, 1),
                                  "cpu_gate": g, "stub_posts": after - before,
                                  "tail": (p.stdout or "")[-500:] + (p.stderr or "")[-300:]}
            posts[key] = after - before
            return p.returncode

        # revision 3: a registered stage before the freeze is refused (7); a slot given to a stage is refused (1)
        res["refusals"]["a0_before_freeze"] = [stage("a0", senv, key="refusal_a0_before_freeze")]
        res["refusals"]["slot_minutes_to_a_stage"] = [stage("smoke", dict(senv, SLOT_MINUTES="100"),
                                                             key="refusal_slot_minutes")]
        res["refusals_ok"] = bool(res["refusals_ok"] and res["refusals"]["a0_before_freeze"][0] == 7
                                  and res["refusals"]["slot_minutes_to_a_stage"][0] == 1
                                  and not (E4 / "runs" / camp / ".a0_started").exists())
        # the smoke with no slot: PENDING
        stage("smoke", senv)
        S = E4 / "runs" / scamp
        sr0 = json.loads((S / "smoke_report.json").read_text()) if (S / "smoke_report.json").exists() else {}
        res["smoke_before_freeze"] = {"decision": sr0.get("decision"),
                                      "gates": {k: v.get("pass") for k, v in (sr0.get("gates") or {}).items()},
                                      "S0": (sr0.get("gates") or {}).get("S0_design")}
        n_smoke = int(((sr0.get("gates") or {}).get("S0_design") or {}).get("n_tasks") or 0)
        # the freeze step: labels (dry labeller: the matcher's own verdicts, not a check), the slot, the record
        from smoke_report import DRY_LABELLER

        cands = json.loads((S / "confirmations.json").read_text()) if (S / "confirmations.json").exists() else []
        (S / "confirmations_labels.json").write_text(json.dumps(
            {"labeller": DRY_LABELLER, "labels": {c["id"]: c["matcher_yes"] for c in cands}}, indent=1))
        srx = [PY, "-I", "-B", str(E4 / "scripts/smoke_report.py"), "--campaign", scamp, "--min-tasks", "3"]
        q = subprocess.run(srx + ["--min-candidates", str(len(cands) + 1), "--max-tasks", str(n_smoke),
                                  "--slot-minutes", "100", "--out", str(work / "smoke_report_too_few_candidates.json")],
                           env=env, capture_output=True, text=True, timeout=900)
        tf = json.loads((work / "smoke_report_too_few_candidates.json").read_text()) if q.returncode == 0 else {}
        res["s8_too_few_candidates"] = {"rc": q.returncode, "decision": tf.get("decision"),
                                        "S8": (tf.get("gates") or {}).get("S8_yes_matcher"),
                                        "S0_pass": ((tf.get("gates") or {}).get("S0_design") or {}).get("pass")}
        q = subprocess.run(srx + ["--min-candidates", "4", "--max-tasks", "5", "--slot-minutes", "100",
                                  "--out", str(S / "smoke_report.json")],
                           env=env, capture_output=True, text=True, timeout=900)
        sr1 = json.loads((S / "smoke_report.json").read_text())
        res["smoke_at_freeze"] = {"rc": q.returncode, "decision": sr1.get("decision"),
                                  "S8": sr1["gates"].get("S8_yes_matcher"), "S9": sr1["gates"].get("S9_budget"),
                                  "S1_end_checks": sr1["gates"]["S1_server_identity"].get("end_checks")}
        fz = [PY, "-I", "-B", str(E4 / "scripts/freeze_e4.py"), "--registration", a.reg, "--dry-campaign", camp,
              "--dry-slot", "100", "--cpu-gate", gate_now()]                # the freeze tool's own CPU gate
        f1 = subprocess.run(fz, env=env, capture_output=True, text=True, timeout=1800)   # waits at most 20 checks
        f2 = subprocess.run(fz, env=env, capture_output=True, text=True, timeout=900)
        res["freeze"] = {"first_rc": f1.returncode, "first": (f1.stdout or "")[-700:], "second_rc": f2.returncode,
                         "second_refused": "write-once" in (f2.stdout or ""),
                         "record_sha256": sha(frozen) if frozen.exists() else None,
                         "record_smoke_tasks": (json.loads(frozen.read_text()).get("smoke_tasks")
                                                if frozen.exists() else None)}
        # revision 4 (round-3 review m3): once the freeze record exists, STAGE=smoke is refused (exit 7)
        res["refusals"]["smoke_after_freeze"] = [stage("smoke", senv, key="refusal_smoke_after_freeze")]
        # revision 5 (round-4 review M1 (a)): the operator's stop at a slot's end, on the main campaign's a0
        R0 = E4 / "runs" / camp
        post(f"http://127.0.0.1:{port}/stub/delay", {"seconds": 0.4})
        before = get(f"http://127.0.0.1:{port}/stub/stats")["posts"]
        t1, g = time.time(), gate_now()
        pr = subprocess.Popen(["bash", str(E4 / "scripts/run_e4_gpu.sh")], env=dict(senv, STAGE="a0", CPU_GATE=g),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        n_at_stop, t_stop = None, time.time() + 1800
        while time.time() < t_stop and pr.poll() is None:
            try:
                n_now = len(json.loads((R0 / "A0_base" / "results.json").read_text())["simulations"])
            except (OSError, ValueError, KeyError):
                n_now = 0
            if n_now >= 1:
                n_at_stop = n_now
                pr.send_signal(signal.SIGTERM)
                break
            time.sleep(0.5)
        tail = pr.communicate(timeout=900)[0] or ""
        res["stages"]["a0_slot_end"] = {"rc": pr.returncode, "seconds": round(time.time() - t1, 1), "cpu_gate": g,
                                        "stub_posts": get(f"http://127.0.0.1:{port}/stub/stats")["posts"] - before,
                                        "tail": tail[-500:]}
        A0d = R0 / "A0_base"

        def man(name="manifest.json"):
            try:
                return json.loads((A0d / name).read_text())
            except (OSError, ValueError):
                return {}

        def n_sims():
            try:
                return len(json.loads((A0d / "results.json").read_text())["simulations"])
            except (OSError, ValueError, KeyError):
                return 0

        stop_end = man().get("server_identity_end") or {}                  # revision 6: written AT the stop
        log0 = (R0 / "run_e4_gpu.log").read_text() if (R0 / "run_e4_gpu.log").exists() else ""
        from check_pins import stage_attempts, stop_effects

        slot_lines = [x for x in log0.splitlines() if "[a0] STOP [slot_end] run=A0_base:" in x]
        at_stop_lines = [x for x in log0.splitlines() if "[a0] end check at the stop, run=A0_base:" in x]
        # revision 6 (round-5 review Major 1 (b)): a kill -9 of the script and its client (no trap runs, so the
        # killed call's record keeps no end), then a start that meets a server that is not the pinned one
        n0, t1, g = n_sims(), time.time(), gate_now()
        pk = subprocess.Popen(["bash", str(E4 / "scripts/run_e4_gpu.sh")], env=dict(senv, STAGE="a0", CPU_GATE=g),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        n_at_kill, t_kill = None, time.time() + 1800
        while time.time() < t_kill and pk.poll() is None:
            if ((A0d / "manifest.1.json").exists() and man() and "server_identity_end" not in man()
                    and n_sims() >= n0 + 1):
                n_at_kill = n_sims()
                os.killpg(pk.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        ktail = pk.communicate(timeout=300)[0] or ""
        post(f"http://127.0.0.1:{port}/stub/delay", {"seconds": 0})
        res["stages"]["a0_killed"] = {"rc": pk.returncode, "seconds": round(time.time() - t1, 1), "cpu_gate": g,
                                      "tail": ktail[-500:]}
        killed = man()
        h_killed = sha(A0d / "manifest.json") if (A0d / "manifest.json").exists() else None
        post(f"http://127.0.0.1:{port}/stub/props", {"chat_template": "{# a chat template that is not the pinned one #}"})
        stage("a0", senv, key="a0_not_pinned")
        post(f"http://127.0.0.1:{port}/stub/props", {})                    # the pinned server again
        h_after = sha(A0d / "manifest.json") if (A0d / "manifest.json").exists() else None
        log_np = (R0 / "run_e4_gpu.log").read_text() if (R0 / "run_e4_gpu.log").exists() else ""
        server_lines = [x for x in log_np.splitlines() if "[a0] STOP [server] run=A0_base:" in x]
        for st in ("a0", "a1", "rerun", "train_guard", "a2a3", "rerun", "judge", "summary"):
            stage(st, senv)
        m1 = man("manifest.1.json").get("server_identity_end") or {}
        m2 = man("manifest.2.json").get("server_identity_end") or {}
        log1 = (R0 / "run_e4_gpu.log").read_text() if (R0 / "run_e4_gpu.log").exists() else ""
        sa = stage_attempts(log1)
        res["slot_end"] = {"rc": pr.returncode, "simulations_done_at_the_stop": n_at_stop, "log_lines": slot_lines,
                           "end_check_at_the_stop": {"kind": stop_end.get("kind"), "at_stop": stop_end.get("at_stop")},
                           "end_check_at_the_stop_lines": at_stop_lines,
                           "kept_as_manifest_1": {"kind": m1.get("kind"), "at_stop": m1.get("at_stop"),
                                                  "deferred": m1.get("deferred")},
                           "integrity_stop_events": stop_effects(log1)["events"], "failed_attempts": sa["failed"],
                           "resumed_a0_rc": res["stages"].get("a0", {}).get("rc"),
                           "done_lines": sum(1 for x in log1.splitlines() if " DONE: stage " in x)}
        res["not_pinned_resume"] = {
            "killed_rc": pk.returncode, "simulations_done_at_the_kill": n_at_kill,
            "killed_record_has_no_end": bool(killed) and "server_identity_end" not in killed,
            "not_pinned_rc": res["stages"]["a0_not_pinned"]["rc"], "server_lines": server_lines,
            "record_sha256_before": h_killed, "record_sha256_after": h_after,
            "record_untouched": bool(h_killed) and h_killed == h_after,
            "deferred_end_kept_as_manifest_2": {"kind": m2.get("kind"), "deferred": m2.get("deferred"),
                                                "at_stop": m2.get("at_stop")},
            "integrity_stop_events": stop_effects(log1)["events"], "failed_attempts": sa["failed"],
            "resumed_a0_rc": res["stages"].get("a0", {}).get("rc")}
        res["slot_end"]["ok"] = bool(pr.returncode == 8 and n_at_stop and len(slot_lines) == 1
                                     and stop_end.get("kind") == "same" and stop_end.get("at_stop") is True
                                     and len(at_stop_lines) == 1 and at_stop_lines[0].split("run=A0_base: ", 1)[1]
                                     .startswith("same ")
                                     and m1.get("kind") == "same" and m1.get("at_stop") is True and not m1.get("deferred")
                                     and not res["slot_end"]["integrity_stop_events"] and not sa["failed"]
                                     and res["slot_end"]["resumed_a0_rc"] == 0)
        npr = res["not_pinned_resume"]
        npr["ok"] = bool(pk.returncode == -signal.SIGKILL and n_at_kill and npr["killed_record_has_no_end"]
                         and npr["not_pinned_rc"] == 2 and len(server_lines) == 1
                         and "not the pinned one" in server_lines[0] and "(identity_changed)" not in log1
                         and npr["record_untouched"] and m2.get("kind") == "same" and m2.get("deferred") is True
                         and not npr["integrity_stop_events"] and not sa["failed"] and npr["resumed_a0_rc"] == 0)
        # revision 4 (round-3 review M1): the integrity re-run without a record is refused at its first stage (exit 7);
        # integrity_failure.py refuses a campaign where no bar is void (exit 1, nothing written)
        res["refusals"]["icr_without_record"] = [stage("a0", dict(senv, CAMPAIGN=f"{camp}_icr"),
                                                       key="refusal_icr_without_record")]
        q = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/integrity_failure.py"), "--campaign", camp,
                            "--cause", "dry run: no failure", "--fix-changes", "--frozen", str(frozen)],
                           env=env, capture_output=True, text=True, timeout=900)
        res["integrity_tool_on_a_clean_campaign"] = {"rc": q.returncode, "out": (q.stdout or "")[-600:],
                                                     "record_written": (E4 / "runs" / camp /
                                                                        "integrity_failure.json").exists()}
        res["refusals_ok"] = bool(res["refusals_ok"] and res["refusals"]["smoke_after_freeze"][0] == 7
                                  and res["refusals"]["icr_without_record"][0] == 7 and q.returncode == 1
                                  and "no bar is void" in (q.stdout or "")
                                  and not res["integrity_tool_on_a_clean_campaign"]["record_written"])
        # the 'not testable' path through the script (train_guard.py exit 3)
        R0, N = E4 / "runs" / camp, E4 / "runs" / ncamp
        N.mkdir(parents=True)
        shutil.copytree(E4 / "runs" / scamp, E4 / "runs" / nscamp)       # its dry freeze reads the smoke report
        nfrozen = N / "E4_frozen_dry.json"
        f3 = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/freeze_e4.py"), "--registration", a.reg,
                             "--dry-campaign", ncamp, "--dry-slot", "100", "--cpu-gate", gate_now()], env=env,
                            capture_output=True, text=True, timeout=1800)  # before the store exists, as at a real freeze
        for x in ("A0_base", "A1_base", "A0_base_rerun", "A1_base_rerun"):
            if (R0 / x).exists():
                shutil.copytree(R0 / x, N / x)
        for x in ("replay.sqlite", ".a0_started"):
            if (R0 / x).exists():
                shutil.copy2(R0 / x, N / x)
        nenv = dict(senv, CAMPAIGN=ncamp, FROZEN=str(nfrozen), MIN_CLASS="1000")
        nenv.pop("DRY_FORCE_GO")
        res["not_testable_freeze_rc"] = f3.returncode
        for st in ("train_guard", "a2a3", "judge", "summary"):
            stage(st, nenv, key=f"nt_{st}")
        res["stub_stats"] = get(f"http://127.0.0.1:{port}/stub/stats")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
    R, S, N = E4 / "runs" / camp, E4 / "runs" / scamp, E4 / "runs" / ncamp
    # replay-only runs of the smoke sent nothing to the stub
    rs = {}
    for n in ("STOCK_smoke", "A0_smoke_replay", "A1_smoke_replay"):
        s = json.loads((S / n / "run_summary.json").read_text()) if (S / n / "run_summary.json").exists() else {}
        rs[n] = s.get("store")
    res["smoke_replay_only_store_stats"] = rs
    res["smoke_replay_only_ok"] = all(v and v.get("miss") == 0 and v.get("replay_miss") == 0 and v.get("hit", 0) > 0
                                      for v in rs.values())
    if (S / "smoke_report.json").exists():
        sr = json.loads((S / "smoke_report.json").read_text())
        res["smoke_report"] = {"decision": sr["decision"], "gates": {k: v.get("pass") for k, v in sr["gates"].items()},
                               "budget": {k: sr["budget"].get(k) for k in ("minutes_full", "decision", "cut")},
                               "S0_tasks": sr["gates"]["S0_design"].get("tasks"),
                               "S0_extension": sr["gates"]["S0_design"].get("extension"),
                               "S1_calls_per_arm": sr["gates"]["S1_server_identity"].get("calls_per_arm"),
                               "no_reward_keys": not any(k in json.dumps(sr) for k in ('"db_reward"', '"pass1_db"',
                                                                                         '"pass1_judged"'))}
        res["smoke_extension_ok"] = bool(sr["gates"]["S0_design"].get("pass")
                                         and (sr["gates"]["S0_design"].get("extension") or {}).get("tasks_added", 0) >= 1)
    for f in ("rows_train.json", "gobar.json", "summary.json"):
        res[f"exists_{f}"] = (R / f).exists()
    if (R / "rows_train.json").exists():
        rows = json.loads((R / "rows_train.json").read_text())
        res["rows"] = rows["counts"]
        res["rows_all_veto_clean"] = all(r["veto_clean"] for r in rows["rows"])
    if (R / "gobar.json").exists():
        g = json.loads((R / "gobar.json").read_text())
        res["gobar"] = {"GO": g.get("GO"), "outcome": g.get("outcome"), "forced": g.get("forced"),
                        "tm": (g.get("models") or {}).get("tm_seed1"),
                        "power_at_observed_rates": (g.get("power") or {}).get("at_observed_rates")}
    res["milp"] = {p.name: json.loads(p.read_text())["summary"] for p in sorted(R.glob("milp_tm_seed*.json"))}
    if (R / "summary.json").exists():
        sm = json.loads((R / "summary.json").read_text())
        res["summary_table"] = sm["table"]
        res["summary_json_sha256"] = sha(R / "summary.json")
        keep_whole = ("checks", "by_bar", "scope", "stop_events", "stop_void_bars", "void_bars", "record",
                      "judged_readings_not_available")             # revision 4 (round-3 review m4): kept whole
        res["summary_integrity"] = {k: (v if (not isinstance(v, dict) or k in keep_whole)
                                        else {kk: vv for kk, vv in v.items() if kk in ("ok", "tasks_checked")})
                                    for k, v in sm["integrity"].items()}
        res["summary_readings_E4_2_without_smoke_tasks"] = (sm.get("readings") or {}).get(
            "E4.2_db_without_smoke_tasks", {}).get("n_tasks")
        res["summary_inputs_smoke_tasks"] = (sm.get("inputs") or {}).get("smoke_tasks")
        res["summary_judge_errors"] = {k: v.get("judge_errors") for k, v in sm["arms"].items()}
        res["summary_arms"] = {k: {kk: v[kk] for kk in ("n", "ends", "executed_violations_total", "guard",
                                                        "server_identity")}
                               for k, v in sm["arms"].items()}
        res["summary_anomalies"] = sm.get("anomalies")
    if (N / "summary.json").exists():
        sn = json.loads((N / "summary.json").read_text())
        gn = json.loads((N / "gobar.json").read_text()) if (N / "gobar.json").exists() else {}
        res["not_testable_summary_json_sha256"] = sha(N / "summary.json")
        res["not_testable"] = {"gobar_outcome": gn.get("outcome"), "table": sn["table"],
                               "checks": sn["integrity"]["checks"], "by_bar": sn["integrity"]["by_bar"],
                               "anomalies": sn.get("anomalies"),
                               "ok": bool(gn.get("outcome") == "not testable"
                                          and sn["bars"]["E4.2"].get("computed") and "flag" not in sn["bars"]["E4.2"]
                                          and str(sn["bars"]["E4.3"].get("why", "")).startswith("not testable")
                                          and str(sn["bars"]["E4.5"].get("why", "")).startswith("not testable")
                                          and sn["integrity"]["checks"]["IC2b"] is None
                                          and res["stages"].get("nt_train_guard", {}).get("rc") == 0
                                          and res["stages"].get("nt_a2a3", {}).get("rc") == 0)}
    res["all_stage_rc"] = {k: v["rc"] for k, v in res["stages"].items()}
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("stages",)}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
