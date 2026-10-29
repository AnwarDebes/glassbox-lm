"""E4 run driver: one arm on one task set of tau2-bench v1.0.1 retail, every LLM role on the shared local llama-server.
NOT RUN by the preparer against a real server (it needs the GPU server). It never starts, stops or signals a server.

  e4_run.py --arm A1 --split base --base http://127.0.0.1:18082/v1 --campaign <name> --expect-server pins.json \
            [--learned m.json] [--task-ids ...] [--run-name ...] [--store-mode replay_only] [--blind]

What it does, in order (any failed check exits non-zero BEFORE any simulation starts):
  1. the endpoint must be 127.0.0.1 / localhost; GET <root>/health must answer {"status": "ok"} and GET <base>/models
     must list the served model (llama-server's own endpoints); otherwise exit 2 ("server not healthy");
  2. SERVER IDENTITY (revision 2, review issue 1): GET <root>/props (b11742 field names: total_slots,
     default_generation_settings.n_ctx, build_info, model_alias, model_path, chat_template), the process that serves
     the port (/proc/<pid>/cmdline, read only), its executable and every file of its build directory, and the GGUF at
     model_path are compared with --expect-server (the frozen pins/server_expected.json); any difference -> exit 2.
     The identity record goes into manifest.json. Revision 3 (re-review item 8): when the call ends, the identity is
     checked AGAIN (light: the pid, /props, /v1/models, argv and executable path, no re-hashing) against the pins and
     the call's own start: "same", "restart_same_identity" (only the pid changed) or "identity_changed"; it is written
     into manifest.json as server_identity_end, and anything but "same" exits 2 after the results are written (a
     STOP; registration section 11.5 says what follows). An earlier manifest.json of the same folder (a resumed stage)
     is kept as manifest.<k>.json, so every call's identity stays on record. Revision 6 (registration 4.1 and 11.6;
     round-5 review Major 1; revision 5 ran the deferred check BEFORE the start check): a call stopped before its end
     check (the operator's stop at a slot's end, or a failed client) gets it AT THE STOP from run_e4_gpu.sh
     (scripts/end_check_at_stop.py -> at_stop_end_check), while the server that served it is still up. At the next
     start the FULL start check comes first (start_checks): a server that is not the pinned one exits 2 and leaves an
     earlier call's record untouched; only then does a manifest.json still WITHOUT server_identity_end (a kill -9 or a
     crash of the script itself) get its deferred end check against the pins and that call's own start record
     (deferred_end_check): "same" or "restart_same_identity" goes on, "identity_changed" (also: a record that cannot
     be read or has no start record) exits 2 with "(identity_changed)" (an integrity stop for this run, registration
     9.1);
  3. a learned model (A2/A3/A3t) must exist, its bit names must match, and it must NOT be a stand-in model unless
     --allow-stand-in (code tests only; never for a registered run); exit 3 otherwise;
  4. every outbound HTTP except 127.0.0.1/localhost goes to a dead proxy; no API keys;
  5. the paired replay store (e4guard/replay.py) is installed for all LLM roles (agent, user simulator);
  6. the guarded agent for the arm is registered as "e4_<arm>" (STOCK = tau2's own "llm_agent", for the smoke's
     "A0 equals stock" check) and tau2's batch runner runs the tasks with EvaluationType.ENV: the inline reward is the
     judge-free DB score; the NL judge runs post hoc (scripts/judge_posthoc.py);
  7. a run summary (per task termination, duration, agent model calls, user turns, tau2 attempts, store hits/misses;
     DB rewards unless --blind) is written. --blind (the smoke) leaves every reward out of run_summary.json and stdout.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
LOCAL = {"127.0.0.1", "localhost"}


def die(code, msg):
    sys.stderr.write(f"e4_run.py: REFUSING: {msg}\n")
    sys.exit(code)


def http_json(url, timeout=5):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy
    with opener.open(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read().decode())


def root_of(base: str) -> str:
    u = urllib.parse.urlparse(base)
    return f"{u.scheme}://{u.hostname}:{u.port}"


def server_health(base: str) -> dict:
    u = urllib.parse.urlparse(base)
    if u.hostname not in LOCAL:
        die(2, f"endpoint {base} is not local")
    root = root_of(base)
    try:
        h = http_json(root + "/health")
    except Exception as e:  # noqa: BLE001
        die(2, f"server not healthy: GET {root}/health failed ({type(e).__name__}: {e})")
    if not (isinstance(h, dict) and h.get("status") == "ok"):
        die(2, f"server not healthy: /health answered {h!r}")
    try:
        m = http_json(base.rstrip("/") + "/models")
        ids = [x.get("id") for x in m.get("data", [])]
    except Exception as e:  # noqa: BLE001
        die(2, f"server not healthy: GET {base}/models failed ({type(e).__name__}: {e})")
    if not ids:
        die(2, "server not healthy: /v1/models lists no model")
    return {"health": h, "models": ids}


def local_only_env(base: str) -> None:
    """Every outbound HTTP except 127.0.0.1/localhost goes to a dead proxy; no API keys (all LLM-calling scripts).
    Revision 3: EVERY variable whose name ends in _API_KEY, _API_BASE or _BASE_URL is removed (no provider is named),
    then the local endpoint's placeholder key and base are set."""
    for k in [k for k in os.environ if k.endswith(("_API_KEY", "_API_BASE", "_BASE_URL"))]:
        os.environ.pop(k, None)
    os.environ.update({"OPENAI_API_KEY": "sk-local-no-key", "OPENAI_BASE_URL": base, "OPENAI_API_BASE": base,
                       "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                       "ALL_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                       "https_proxy": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
                       "no_proxy": "127.0.0.1,localhost", "LITELLM_LOCAL_MODEL_COST_MAP": "True"})


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def processes_on_port(port: int) -> list[tuple[int, list[str]]]:
    """(pid, argv) of every process of this machine whose command line has '--port <port>' (read-only /proc scan)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        if any(a == "--port" and i + 1 < len(argv) and argv[i + 1] == str(port) for i, a in enumerate(argv)):
            out.append((int(d.name), argv))
    return out


def server_identity(base: str, pins: dict, hash_model: bool = True, hash_files: bool = True) -> tuple[dict, list[str]]:
    """Compare the live server with the frozen pins. Returns (identity record, list of differences). hash_files=False
    (the light check at a call's end) skips the executable, build-file and GGUF hashes."""
    t0 = time.time()
    hash_model = hash_model and hash_files
    port = urllib.parse.urlparse(base).port
    diff, rec = [], {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "port": port}
    if port != pins["port"]:
        diff.append(f"port {port} != pinned {pins['port']}")
    try:
        p = http_json(root_of(base) + "/props", timeout=10)
    except Exception as e:  # noqa: BLE001
        return rec, [f"GET /props failed ({type(e).__name__}: {e})"]
    tmpl = p.get("chat_template")
    live = {"total_slots": p.get("total_slots"),
            "n_ctx_per_slot": (p.get("default_generation_settings") or {}).get("n_ctx"),
            "build_info": p.get("build_info"), "model_alias": p.get("model_alias"), "model_path": p.get("model_path"),
            "chat_template_sha256": hashlib.sha256(tmpl.encode("utf-8")).hexdigest() if isinstance(tmpl, str) else None}
    rec["props"] = live
    for k, v in pins["props"].items():
        if live.get(k) != v:
            diff.append(f"/props {k} = {live.get(k)!r}, pinned {v!r}")
    try:
        ids = [x.get("id") for x in http_json(base.rstrip("/") + "/models").get("data", [])]
    except Exception as e:  # noqa: BLE001
        ids = []
        diff.append(f"GET /v1/models failed ({type(e).__name__}: {e})")
    rec["models_ids"] = ids
    if ids != pins["models_ids"]:
        diff.append(f"/v1/models ids {ids} != pinned {pins['models_ids']}")
    procs = processes_on_port(port)
    rec["processes"] = [{"pid": pid, "argv": argv} for pid, argv in procs]
    if len(procs) != 1:
        diff.append(f"{len(procs)} processes have '--port {port}' in their command line (need exactly 1)")
    else:
        pid, argv = procs[0]
        if argv != pins["argv"]:
            diff.append(f"server argv differs from the pinned command line: {argv}")
        try:
            exe = os.path.realpath(f"/proc/{pid}/exe")
            rec["exe"] = exe
            if hash_files:
                rec["exe_sha256"] = sha256_file(exe)
        except OSError as e:
            exe = None
            diff.append(f"cannot read /proc/{pid}/exe ({e})")
        if exe is not None:
            if exe != os.path.realpath(pins["exe_path"]):
                diff.append(f"server executable {exe} != pinned {pins['exe_path']}")
            if hash_files and rec.get("exe_sha256") != pins["exe_sha256"]:
                diff.append(f"server executable sha256 {rec.get('exe_sha256')} != pinned {pins['exe_sha256']}")
            bad = []
            for name, h in (sorted(pins.get("build_dir_files_sha256", {}).items()) if hash_files else []):
                f = Path(exe).parent / name
                if not f.exists() or sha256_file(f) != h:
                    bad.append(name)
            rec["build_dir_files_checked"] = len(pins.get("build_dir_files_sha256", {})) if hash_files else 0
            if bad:
                diff.append(f"build files differ from the pinned list: {bad}")
        try:
            env = dict(kv.split("=", 1) for kv in (Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
                                                    .split("\0")) if "=" in kv)
            rec["environ"] = {k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
        except OSError:
            rec["environ"] = None
    mp = live.get("model_path")
    if hash_model and isinstance(mp, str) and Path(mp).exists():
        rec["gguf_sha256"] = sha256_file(mp)
        if rec["gguf_sha256"] != pins["gguf_sha256"]:
            diff.append(f"GGUF sha256 {rec['gguf_sha256']} != pinned {pins['gguf_sha256']}")
    elif hash_model:
        diff.append(f"model_path {mp!r} is not a readable file")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["differences"] = diff
    return rec, diff


def identity_end_check(base: str, pins: dict, start: dict) -> tuple[dict, str]:
    """The light identity check when a call ends (revision 3): against the pins AND the call's own start record.
    kind: "same"; "restart_same_identity" (only the server's pid changed); "identity_changed" (anything else)."""
    rec, diff = server_identity(base, pins, hash_files=False)
    other = list(diff)
    for k in ("props", "models_ids", "exe"):
        if rec.get(k) != start.get(k):
            other.append(f"{k} changed during the call: {start.get(k)!r} -> {rec.get(k)!r}")
    s_argv = [p.get("argv") for p in start.get("processes") or []]
    e_argv = [p.get("argv") for p in rec.get("processes") or []]
    if s_argv != e_argv:
        other.append("the server's command line changed during the call")
    s_pid = [p.get("pid") for p in start.get("processes") or []]
    e_pid = [p.get("pid") for p in rec.get("processes") or []]
    kind = "identity_changed" if other else ("restart_same_identity" if s_pid != e_pid else "same")
    rec.update({"kind": kind, "start_pids": s_pid, "end_pids": e_pid,
                "differences": other + ([f"server pid {s_pid} -> {e_pid}"] if s_pid != e_pid else [])})
    return rec, kind


def deferred_end_check(base: str, pins: dict, rec_path: Path) -> str | None:
    """Revision 5 (registration 4.1, 11.6; round-4 review M1 (a)): a call of this run that ended without its end check
    gets it at the run's next start: the light end check against the pins AND that call's own start record, written
    into its record as server_identity_end with "deferred": true. Returns its kind ("same", "restart_same_identity" or
    "identity_changed"), or None when there is no record or it has its end. A record that cannot be read, or has no
    start record, is "identity_changed": its call's end cannot be shown to be the pinned server. Revision 6 (round-5
    review Major 1 (b)): called only by start_checks, AFTER the full start check has found the pinned server; the
    operator's stop and a failed client get their end check at the stop (at_stop_end_check), so this is left for a
    record whose call was ended by a kill -9 or a crash of the script itself."""
    p = Path(rec_path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return "identity_changed"
    if not isinstance(d, dict) or not isinstance(d.get("server_identity"), dict):
        return None if isinstance(d, dict) and "server_identity_end" in d else "identity_changed"
    if "server_identity_end" in d:
        return None
    end, kind = identity_end_check(base, pins, d["server_identity"])
    d["server_identity_end"] = {**end, "deferred": True}
    p.write_text(json.dumps(d, indent=1, default=str))
    return kind


def at_stop_end_check(base: str, pins: dict, rec_path, since_utc: str) -> tuple[str | None, list[str]]:
    """Revision 6 (registration 4.1, 11.6; round-5 review Major 1 (a)): the end check AT THE STOP of a call that ended
    without its own. run_e4_gpu.sh runs it (scripts/end_check_at_stop.py) right after the operator's stop at a slot's
    end (slot_end, or a client ended by TERM or INT) and after a failed client (exit 3), while the server that served
    the call is still up (the operator stops the client only, never the server). The light end check against the pins
    and the call's own start record, written into the record as server_identity_end with "at_stop": true. Only the
    stopped call's own record is checked: readable, with its start record, without an end, and written at or after
    since_utc (the client's start; "%Y-%m-%dT%H:%M:%SZ" strings compare in time order). Any other record is left
    untouched: one without its end that this call did not write (a kill -9 or a crash of the script) gets the deferred
    check at the run's next start, after the full start check (start_checks). Returns (kind, differences); kind None
    means nothing was checked."""
    p = Path(rec_path)
    if not p.is_file():
        return None, []
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None, []
    if (not isinstance(d, dict) or "server_identity_end" in d or not isinstance(d.get("server_identity"), dict)
            or not isinstance(d.get("utc"), str) or d["utc"] < since_utc):
        return None, []
    end, kind = identity_end_check(base, pins, d["server_identity"])
    d["server_identity_end"] = {**end, "at_stop": True}
    p.write_text(json.dumps(d, indent=1, default=str))
    return kind, list(end.get("differences") or [])


def start_checks(base: str, pins: dict, rec_path, what: str = "run") -> dict:
    """Revision 6 (registration 4.1, 11.6; round-5 review Major 1 (b)): the server checks when a call starts, in this
    order. FIRST the full start check (server_identity): a server that is not the pinned one exits 2 "the server is
    not the pinned one" (run_e4_gpu.sh tags it [server]: not counted, no integrity event) and leaves an earlier call's
    record untouched. THEN, only with the pinned server up, an earlier call's record that still has no end gets its
    deferred end check (deferred_end_check); "identity_changed" (also: a record that cannot be read or has no start
    record) exits 2 with "(identity_changed)", an integrity stop for this run (registration 9.1). What the deferred
    check can see: it compares the PINNED server as it is now with the call's start record, so it sees a restart (a
    new pid) and a record it cannot read, never a change made during the call and undone before the restart.
    Returns the start identity record."""
    ident, diff = server_identity(base, pins)
    if diff:
        die(2, "the server is not the pinned one: " + "; ".join(diff))
    if deferred_end_check(base, pins, rec_path) == "identity_changed":
        die(2, f"the server identity changed during this {what}'s earlier call, which ended without its end check "
               "(identity_changed): its deferred end check, with the pinned server up at this start, differs from its "
               f"start record, or the record cannot be read (see {rec_path}; registration 4.1, 11.6)")
    return ident


def keep_earlier_manifest(out: Path) -> str | None:
    """A resumed stage keeps the earlier call's manifest.json as manifest.<k>.json (k = 1, 2, ...)."""
    m = out / "manifest.json"
    if not m.exists():
        return None
    k = 1
    while (out / f"manifest.{k}.json").exists():
        k += 1
    m.rename(out / f"manifest.{k}.json")
    return f"manifest.{k}.json"


def per_task_guard_counts(log_path: Path) -> dict:
    out = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text().splitlines():
        e = json.loads(line)
        c = out.setdefault(str(e.get("task")), Counter())
        c[e["event"]] += 1
        if e["event"] == "verdict" and not e["allow"]:
            c["denied"] += 1
    return {k: dict(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A0", "A1", "A2", "A3", "A3t", "STOCK"])
    ap.add_argument("--split", default="base", choices=["train", "test", "base"])
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default=None, help="served model id (default: the first id in /v1/models)")
    ap.add_argument("--campaign", required=True, help="runs/<campaign>/ holds the shared replay store and every arm")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--learned", default=None)
    ap.add_argument("--allow-stand-in", action="store_true")
    ap.add_argument("--expect-server", required=True, help="the frozen server pins (pins/server_expected.json)")
    ap.add_argument("--store-mode", default="record", choices=["record", "replay_only"])
    ap.add_argument("--blind", action="store_true", help="leave every reward out of run_summary.json and stdout")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-errors", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=3, help="tau2 re-runs of a simulation after an exception")
    ap.add_argument("--guard-retries", type=int, default=3, help="model calls after denials in one agent turn")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    t0 = time.time()

    info = server_health(args.base)
    model = args.model or info["models"][0]
    if model not in info["models"]:
        die(2, f"model {model!r} not served (served: {info['models']})")
    pins = json.loads(Path(args.expect_server).read_text())
    camp = E4 / "runs" / args.campaign
    name = args.run_name or f"{args.arm}_{args.split}"
    out = camp / name
    ident = start_checks(args.base, pins, out / "manifest.json", "run")   # revision 6: the full start check first
    ident["pins_file"] = args.expect_server
    ident["pins_sha256"] = sha256_file(args.expect_server)
    if args.arm in ("A2", "A3", "A3t"):
        if not args.learned or not Path(args.learned).exists():
            die(3, f"arm {args.arm} needs --learned <exported model json>")
        from e4guard.models import check_arm_model

        try:
            check_arm_model(args.arm, args.learned)
        except ValueError as e:
            die(3, str(e))
        meta = json.loads(Path(args.learned).read_text()).get("meta", {})
        if meta.get("stand_in") and not args.allow_stand_in:
            die(3, f"{args.learned} is a STAND-IN model (code test only); a registered run needs a model trained on "
                   f"this campaign's train-task rows")
        from e4guard import bits as B

        if meta.get("bit_names_sha256") != hashlib.sha256(json.dumps(B.NAMES).encode()).hexdigest():
            die(3, "the learned model was trained on other bit names")
    elif args.learned:
        die(3, f"arm {args.arm} takes no learned model")

    # 4. network: only localhost; no keys
    local_only_env(args.base)

    out.mkdir(parents=True, exist_ok=True)

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator import evaluator_nl_assertions as nl
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.batch import run_tasks
    from tau2.runner.helpers import get_tasks

    from e4guard.agent import make_factory
    from e4guard.replay import ReplayStore, install

    llm = "openai/" + model
    llm_args = {"api_base": args.base, "api_key": "sk-local-no-key", "temperature": 0.0, "max_tokens": args.max_tokens,
                "timeout": 600, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    nl.DEFAULT_LLM_NL_ASSERTIONS = llm                     # only the post-hoc judge uses it (inline scoring is ENV)
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {k: v for k, v in llm_args.items()}
    store = ReplayStore(str(camp / "replay.sqlite"), mode=args.store_mode)
    store.arm = name
    install(store)
    if args.arm == "STOCK":
        agent_name = "llm_agent"                           # tau2's own agent, unregistered by us
    else:
        agent_name = f"e4_{args.arm}"
        registry.register_agent_factory(make_factory(args.arm, learned_path=args.learned,
                                                     log_path=str(out / "guard_log.jsonl"),
                                                     max_retries=args.guard_retries), agent_name)
    cfg = TextRunConfig(domain="retail", agent=agent_name, llm_agent=llm, llm_args_agent=llm_args,
                        user="user_simulator", llm_user=llm, llm_args_user=dict(llm_args), num_trials=1,
                        max_steps=args.max_steps, max_errors=args.max_errors, timeout=args.timeout,
                        max_concurrency=1, seed=args.seed, log_level="WARNING", max_retries=args.max_retries,
                        auto_resume=True, auto_review=False, hallucination_retries=0, task_split_name=args.split)
    tasks = get_tasks("retail", task_split_name=args.split, task_ids=args.task_ids, num_tasks=args.num_tasks)
    code = sorted((E4 / "e4guard").glob("*.py")) + sorted((E4 / "scripts").glob("*.py")) + \
        sorted((E4 / "scripts").glob("*.sh"))
    manifest = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arm": args.arm, "split": args.split,
                "task_ids": [t.id for t in tasks], "server": info, "server_identity": ident, "model": model,
                "llm_args": {**llm_args, "api_key": "<local>"}, "learned": args.learned,
                "learned_sha256": sha256_file(args.learned) if args.learned else None,
                "store": str(camp / "replay.sqlite"), "store_mode": args.store_mode, "seed": args.seed,
                "max_steps": args.max_steps, "max_errors": args.max_errors, "max_retries": args.max_retries,
                "guard_retries": args.guard_retries, "timeout": args.timeout, "blind": args.blind,
                "code_sha256": {str(p.relative_to(E4)): sha256_file(p) for p in code}}
    manifest["earlier_manifest"] = keep_earlier_manifest(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    res = run_tasks(cfg, tasks, save_path=out / "results.json", evaluation_type=EvaluationType.ENV,
                    console_display=False)
    end, kind = identity_end_check(args.base, pins, ident)          # revision 3: the identity when the call ends
    manifest["server_identity_end"] = end
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    guard = per_task_guard_counts(out / "guard_log.jsonl")
    rows = []
    for sim in res.simulations:
        ri = sim.reward_info
        term = str(getattr(sim.termination_reason, "value", sim.termination_reason))
        g = guard.get(str(sim.task_id), {})
        r = {"task": sim.task_id, "termination": term, "duration_s": sim.duration,
             "agent_model_calls": g.get("model_call"), "denied": g.get("denied"), "fallbacks": g.get("fallback"),
             "tau2_attempts": g.get("agent_init"),
             "user_turns": sum(1 for m in sim.messages if getattr(m, "role", "") == "user"
                               and not getattr(m, "tool_calls", None)),
             "messages": len(sim.messages)}
        if not args.blind:
            r["db_reward"] = ri.reward if ri is not None else 0.0
        rows.append(r)
    summary = {"arm": args.arm, "split": args.split, "n": len(rows), "blind": args.blind,
               "terminations": dict(Counter(r["termination"] for r in rows)),
               "store": store.stats, "seconds": round(time.time() - t0, 1), "server_identity_end": kind, "rows": rows}
    if not args.blind:
        summary["pass1_db"] = sum(r["db_reward"] for r in rows) / max(1, len(rows))
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    if kind != "same":
        die(2, f"the server identity changed during the call ({kind}): " + "; ".join(end["differences"]))


if __name__ == "__main__":
    main()
