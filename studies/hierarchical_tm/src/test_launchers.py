"""CPU tests of the launchers of this study (bug HTM-B1, 2026-10-25): interpreter choice, argument handling and the whole
launch chain (timeout -> mem_guard.py -> run_logged.py -> gpu_run.sh -> job), with stand-in jobs.

Nothing here uses the GPU or writes into a study folder. The launchers are copied into a scratch tree
<tmp>/studies/hierarchical_tm (each copy is checked to be byte-identical to its source first), next to a stand-in
<tmp>/studies/tm_language_model whose env/, scripts/, src/ and .venv-tm are links to the real ones, and a stand-in
<tmp>/studies/decision_layer whose .venv-htm is a link to the real one. The drivers of the scratch tree are stand-ins
that only report their interpreter, arguments and environment; they import no GPU package. Fake nvidia-smi and
nvidia-cuda-mps-control commands come first on PATH, CUDA_VISIBLE_DEVICES is empty, and the GPU lock folder
(TMLM_LOCK_DIR), the default run folder of the logger (TMLM_RESULTS_DIR) and every run folder are inside the scratch
tree. At the end the results/ folders of this study and of tm_language_model are checked to hold no run folder of these
tests, and src/ of this study no new entry.

Run from the study folder:
  PYTHONDONTWRITEBYTECODE=1 ../tm_language_model/.venv-tm/bin/python src/test_launchers.py [--source DIR] [--keep]
--source DIR takes the launchers from DIR (same layout as this folder) instead of this study, e.g. the versions before
the fix of 2026-10-25; --tlm-env DIR links env/ of the stand-in tm_language_model to DIR instead of the real env/ (e.g. a
copy of an env.sh that puts no python3 on PATH, to show that the launchers do not need one); --keep keeps the scratch
tree. About 4 to 5 minutes, mostly the fixed waits of the two queue
scripts (20 s after each job start, 60 s per MPS setting). Exit status 0 when every test passes, else 1.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.dirname(STUDY)
TLM = os.path.join(STUDIES, "tm_language_model")
DL_HTM = os.path.join(STUDIES, "decision_layer", ".venv-htm")
LAUNCHERS = ("src/interpreters.sh", "src/launch.sh", "src/feas_slide11/launch.sh", "queue/run_queue.sh",
             "queue/run_queue_mps.sh", "queue/gpu_share_test.sh")
HTM_COMMIT = "5b4a223075c1e621577d508bc7c3c061230e29d5"  # frozen settings, README of this study, env/htm.lock
TLM_ENV = os.path.join(TLM, "env")  # --tlm-env replaces it, e.g. by a copy of an env.sh that puts no python3 on PATH

STANDIN = r'''"""Stand-in job of the launcher tests: reports interpreter, arguments and environment; imports no GPU package."""
import importlib.metadata as md, json, os, sys, time
args = sys.argv[1:]
keys = ("TMLM_RUN_DIR", "TMLM_RUN_NAME", "TMLM_GPU_LOCK_MODE", "CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY", "HTM_PY", "HTM_TOOL_PY", "FAKE_OLD_HTM", "OMP_NUM_THREADS", "PYTHONHASHSEED")
info = {"script": os.path.basename(__file__), "executable": sys.executable, "prefix": sys.prefix,
        "real_prefix": os.path.realpath(sys.prefix), "argv": args, "env": {k: os.environ.get(k) for k in keys}}
try:
    d = md.distribution("PyHierarchicalTsetlinMachineCUDA")
    info["htm_version"] = d.version
    info["htm_commit"] = json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
except md.PackageNotFoundError:
    info["htm_version"] = info["htm_commit"] = None
info["versions"] = {}
for name in ("numpy", "scipy", "scikit-learn", "pycuda", "scikit-image", "networkx"):
    try:
        info["versions"][name] = md.version(name)
    except md.PackageNotFoundError:
        info["versions"][name] = None
if "--sleep" in args:
    time.sleep(float(args[args.index("--sleep") + 1]))
with open(os.path.join(os.environ.get("TMLM_RUN_DIR") or ".", "standin.json"), "w") as f:
    json.dump(info, f)
print("STANDIN " + json.dumps(info), flush=True)
print('FINAL {"dev_acc": 0.5, "dev_auc": 0.5, "partner_shuffle_auc": 0.5}', flush=True)
'''
ANALYSE = 'import sys\nprint("ANALYSIS STAND-IN under", sys.executable)\n'
FAKE_SMI = '#!/bin/sh\necho "fake nvidia-smi of the launcher tests: no GPU query" >&2\nexit 9\n'
FAKE_MPS = '#!/bin/sh\n{ echo "args: $*"; [ "$#" -eq 0 ] && cat; } >> "$FAKE_MPS_LOG" 2>&1\nexit 0\n'

RESULTS = []


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""), flush=True)
    return bool(cond)


def write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def make_tree(root, source, old_htm=False):
    """The scratch tree; returns its hierarchical_tm folder. A launcher missing from source is skipped (and reported)."""
    st = os.path.join(root, "studies")
    h = os.path.join(st, "hierarchical_tm")
    for rel in LAUNCHERS:
        src = os.path.join(source, rel)
        if not os.path.exists(src):
            print(f"note: {rel} does not exist in {source}", flush=True)
            continue
        dst = os.path.join(h, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        if sha256(src) != sha256(dst):
            raise SystemExit(f"copy of {rel} differs from its source")
    for rel in ("src/htm_driver.py", "src/other_driver.py", "src/score_saved.py", "src/feas_slide11/train_tm.py"):
        write(os.path.join(h, rel), STANDIN)
    write(os.path.join(h, "src/feas_slide11/prep_data.py"), "# stand-in of the launcher tests\n")
    write(os.path.join(h, "queue/analyse_share_test.py"), ANALYSE)
    os.makedirs(os.path.join(h, "results"))
    t = os.path.join(st, "tm_language_model")
    os.makedirs(t)
    for name in ("env", "scripts", "src", ".venv-tm"):
        os.symlink(TLM_ENV if name == "env" else os.path.join(TLM, name), os.path.join(t, name))
    if old_htm:  # a stand-in for the absent ../tm_language_model/.venv-htm: marks itself, then runs the real interpreter
        write(os.path.join(t, ".venv-htm/bin/python"),
              f'#!/usr/bin/env bash\nexport FAKE_OLD_HTM=1\nexec "{DL_HTM}/bin/python" "$@"\n', 0o755)
    os.makedirs(os.path.join(st, "decision_layer"))
    os.symlink(DL_HTM, os.path.join(st, "decision_layer", ".venv-htm"))
    write(os.path.join(root, "fakebin/nvidia-smi"), FAKE_SMI, 0o755)
    write(os.path.join(root, "fakebin/nvidia-cuda-mps-control"), FAKE_MPS, 0o755)
    for d in ("locks", "tmp", "tlm_results"):
        os.makedirs(os.path.join(root, d))
    return h


def env_for(root, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HTM_", "TMLM_", "CUDA_MPS_"))}
    env.update(PATH=os.path.join(root, "fakebin") + ":" + os.environ["PATH"], CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", TMLM_LOCK_DIR=os.path.join(root, "locks"),
               TMLM_RESULTS_DIR=os.path.join(root, "tlm_results"), TMPDIR=os.path.join(root, "tmp"),
               FAKE_MPS_LOG=os.path.join(root, "fake_mps.log"))
    for k in ("OMP", "MKL", "OPENBLAS", "NUMEXPR", "NUMBA", "RAYON"):
        env[f"{k}_NUM_THREADS"] = "2"
    env.update(extra)
    return env


def run(cmd, env, cwd, timeout=600):
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def run_dirs(folder, name):
    return sorted(glob.glob(os.path.join(folder, f"*_{name}")))


def load(path):
    with open(path) as f:
        return json.load(f)


def one_run(h, name, folder=None):
    """(run dir, run.json, config.json, env.json, standin.json or None) of the single run NAME."""
    ds = run_dirs(folder or os.path.join(h, "results"), name)
    if len(ds) != 1:
        return None
    d = ds[0]
    s = os.path.join(d, "standin.json")
    return d, load(os.path.join(d, "run.json")), load(os.path.join(d, "config.json")), load(os.path.join(d, "env.json")), \
        load(s) if os.path.exists(s) else None


def test_launch(source, keep):
    root = tempfile.mkdtemp(prefix="htm_launch_")
    h = make_tree(root, source)
    st = os.path.dirname(h)
    dl_py = os.path.join(st, "decision_layer/.venv-htm/bin/python")
    tool_py = os.path.join(st, "tm_language_model/.venv-tm/bin/python")
    L = os.path.join(h, "src/launch.sh")
    odd = ["--data", "sa", "--note", "a b  c", "--glob", "*", "--quote", "it's \"x\"", "--empty", ""]
    rc, out, err, _ = run([L, "t_default_exploratory", "5", os.path.join(h, "src/htm_driver.py")] + odd, env_for(root), root)
    r = one_run(h, "t_default_exploratory")
    check("launch.sh: default chain exits 0", rc == 0, f"rc {rc}; {err[-400:]}")
    if r and r[4]:
        d, rj, cfg, envj, s = r
        check("launch.sh: default job interpreter is ../decision_layer/.venv-htm (old path absent)",
              s["real_prefix"] == os.path.realpath(DL_HTM) and cfg.get("job_python") == dl_py, f"{s['real_prefix']}, {cfg}")
        check("launch.sh: that virtualenv holds PyHierarchicalTsetlinMachineCUDA at commit 5b4a223",
              s["htm_commit"] == HTM_COMMIT, str(s["htm_commit"]))
        check("launch.sh: arguments reach the job unchanged (spaces, glob, quotes, empty)", s["argv"] == odd, str(s["argv"]))
        check("launch.sh: run logger ran under ../tm_language_model/.venv-tm", envj["python_executable"] == tool_py,
              envj["python_executable"])
        check("launch.sh: exclusive GPU lock mode, CUDA_VISIBLE_DEVICES empty, lock folder in the scratch tree",
              s["env"]["TMLM_GPU_LOCK_MODE"] == "exclusive" and s["env"]["CUDA_VISIBLE_DEVICES"] == ""
              and os.path.exists(os.path.join(root, "locks/gpu.lock")), json.dumps(s["env"]))
        check("launch.sh: run record exit status 0 and config names the run", rj["exit_status"] == 0
              and cfg["name"] == "t_default_exploratory" and cfg["argv"][1:] == odd, f"{rj['exit_status']} {cfg}")
    else:
        check("launch.sh: default run folder with a stand-in record exists", False, f"{r}; rc {rc}; {err[-400:]}")

    # the old path wins when it exists
    root2 = tempfile.mkdtemp(prefix="htm_launch_old_")
    h2 = make_tree(root2, source, old_htm=True)
    rc, out, err, _ = run([os.path.join(h2, "src/launch.sh"), "t_old_exploratory", "5", os.path.join(h2, "src/htm_driver.py"), "--x"],
                          env_for(root2), root2)
    r = one_run(h2, "t_old_exploratory")
    ok = rc == 0 and r and r[4] and r[4]["env"]["FAKE_OLD_HTM"] == "1" and \
        r[2].get("job_python") == os.path.join(os.path.dirname(h2), "tm_language_model/.venv-htm/bin/python")
    check("launch.sh: ../tm_language_model/.venv-htm is used when it exists", ok, f"rc {rc}; {err[-300:]}")

    # an explicit relative HTM_PY is made absolute from the caller's folder
    rel = os.path.relpath(dl_py, root)
    rc, out, err, _ = run([L, "t_relpy_exploratory", "5", os.path.join(h, "src/htm_driver.py")], env_for(root, HTM_PY=rel), root)
    r = one_run(h, "t_relpy_exploratory")
    check("launch.sh: a relative HTM_PY is resolved from the caller's folder",
          rc == 0 and r and r[2].get("job_python") == os.path.join(root, rel), f"rc {rc}; {err[-300:]}")

    # missing interpreters and missing arguments stop before anything starts
    for var in ("HTM_PY", "HTM_TOOL_PY"):
        rc, out, err, _ = run([L, f"t_bad_{var.lower()}_exploratory", "5", os.path.join(h, "src/htm_driver.py")],
                              env_for(root, **{var: "/nonexistent/bin/python"}), root)
        check(f"launch.sh: {var} that does not exist gives exit 127, a message and no run folder",
              rc == 127 and f"{var}=/nonexistent/bin/python is not an executable file" in err
              and not run_dirs(os.path.join(h, "results"), f"t_bad_{var.lower()}_exploratory"), f"rc {rc}; {err[-300:]}")
    rc, out, err, _ = run([L, "t_noargs"], env_for(root), root)
    check("launch.sh: fewer than 3 arguments gives exit 2 and the usage line", rc == 2 and "usage: launch.sh" in err,
          f"rc {rc}; {err[-300:]}")

    # the wall-time cap still stops the whole chain
    tag = uuid.uuid4().hex
    rc, out, err, secs = run([L, "t_cap_exploratory", "0.05", os.path.join(h, "src/htm_driver.py"), "--sleep", "120", "--tag", tag],
                             env_for(root), root, timeout=200)
    time.sleep(1)
    left = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
    r = one_run(h, "t_cap_exploratory")
    check("launch.sh: a cap of 0.05 min stops the job (timeout exit 124, run record 143, no process left)",
          rc == 124 and secs < 60 and not left and r and r[1]["exit_status"] == 143,
          f"rc {rc}; {secs:.0f} s; left {left}; record {r and r[1]['exit_status']}")
    check("launch.sh: nothing written to the logger's default run folder", os.listdir(os.path.join(root, "tlm_results")) == [])
    if not keep:
        shutil.rmtree(root)
        shutil.rmtree(root2)


def wrapper(path, real, record):
    """An interpreter that records PYTHONDONTWRITEBYTECODE and starts the real one with -B (so it never writes bytecode)."""
    write(path, f'#!/usr/bin/env bash\necho "PYTHONDONTWRITEBYTECODE=${{PYTHONDONTWRITEBYTECODE:-}}" >> "{record}"\n'
                f'exec "{real}" -B "$@"\n', 0o755)


def test_bytecode(source, keep):
    """PYTHONDONTWRITEBYTECODE=1 is exported when the job interpreter lies in ../decision_layer/ (added 2026-10-25,
    second round), and not otherwise. Both interpreters are recording wrappers that start the real ones with -B."""
    root = tempfile.mkdtemp(prefix="htm_pyc_")
    h = make_tree(root, source)
    st = os.path.dirname(h)
    dl = os.path.join(st, "decision_layer", ".venv-htm")
    os.unlink(dl)  # the link to the real virtualenv is replaced by a recording wrapper
    rec_dl, rec_other = os.path.join(root, "record_dl.txt"), os.path.join(root, "record_other.txt")
    wrapper(os.path.join(dl, "bin", "python"), os.path.join(DL_HTM, "bin", "python"), rec_dl)
    tool = os.path.join(root, "tool_venv", "bin", "python")
    wrapper(tool, os.path.join(TLM, ".venv-tm", "bin", "python"), os.path.join(root, "record_tool.txt"))
    env = env_for(root, HTM_TOOL_PY=tool)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    rc, out, err, _ = run([os.path.join(h, "src/launch.sh"), "t_pyc_dl_exploratory", "5", os.path.join(h, "src/htm_driver.py")], env, root)
    got = open(rec_dl).read().split() if os.path.exists(rec_dl) else []
    check("interpreters.sh: PYTHONDONTWRITEBYTECODE=1 when the job interpreter lies in ../decision_layer/",
          rc == 0 and got and all(g == "PYTHONDONTWRITEBYTECODE=1" for g in got), f"rc {rc}; {got}; {err[-300:]}")
    other = os.path.join(root, "other_venv", "bin", "python")
    wrapper(other, os.path.join(DL_HTM, "bin", "python"), rec_other)
    rc, out, err, _ = run([os.path.join(h, "src/launch.sh"), "t_pyc_other_exploratory", "5", os.path.join(h, "src/htm_driver.py")],
                          dict(env, HTM_PY=other), root)
    got = open(rec_other).read().split() if os.path.exists(rec_other) else []
    check("interpreters.sh: no PYTHONDONTWRITEBYTECODE for a job interpreter elsewhere",
          rc == 0 and got and all(g == "PYTHONDONTWRITEBYTECODE=" for g in got), f"rc {rc}; {got}; {err[-300:]}")
    if not keep:
        shutil.rmtree(root)


def test_run_queue(source, keep):
    root = tempfile.mkdtemp(prefix="htm_queue_")
    h = make_tree(root, source)
    q = os.path.join(h, "queue")
    write(os.path.join(q, "jobs.txt"), "# comment\nq_a_exploratory|5|--data sa --seed 1\n\nq_b_exploratory|5|--data np --seed 2\n")
    rc, out, err, secs = run(["bash", os.path.join(q, "run_queue.sh")], env_for(root), root, timeout=300)
    log = open(os.path.join(q, "queue.log")).read() if os.path.exists(os.path.join(q, "queue.log")) else ""
    done = open(os.path.join(q, "done.txt")).read().split() if os.path.exists(os.path.join(q, "done.txt")) else []
    a, b = one_run(h, "q_a_exploratory"), one_run(h, "q_b_exploratory")
    check("run_queue.sh: both jobs ran through src/launch.sh and ended with exit 0",
          rc == 0 and done == ["q_a_exploratory", "q_b_exploratory"] and log.count("exit=0") == 2
          and '"dev_auc": 0.5' in log and "QUEUE EMPTY" in log, f"rc {rc}; done {done}; log {log[-400:]}; {err[-300:]}")
    check("run_queue.sh: job arguments and interpreter", bool(a and a[4] and b and b[4]) and a[4]["argv"] == ["--data", "sa", "--seed", "1"]
          and b[4]["argv"] == ["--data", "np", "--seed", "2"] and a[4]["real_prefix"] == os.path.realpath(DL_HTM),
          f"{a and a[4]} {b and b[4]}")
    if not keep:
        shutil.rmtree(root)


def test_run_queue_mps(source, keep):
    root = tempfile.mkdtemp(prefix="htm_mps_")
    h = make_tree(root, source)
    q = os.path.join(h, "queue")
    write(os.path.join(q, "jobs.txt"), "m_a_exploratory|7|--data sa --seed 1\nm_b_exploratory|7|@other_driver.py --data np\n")
    write(os.path.join(q, "queue.log"), "")
    rc, out, err, secs = run(["bash", os.path.join(q, "run_queue_mps.sh")], env_for(root), root, timeout=400)
    log = open(os.path.join(q, "queue.log")).read()
    mps = open(os.path.join(root, "fake_mps.log")).read() if os.path.exists(os.path.join(root, "fake_mps.log")) else ""
    a, b = one_run(h, "m_a_exploratory"), one_run(h, "m_b_exploratory")
    check("run_queue_mps.sh: both jobs ended with exit 0, MPS start and stop logged",
          rc == 0 and log.count("exit=0") == 2 and "MPS daemon started" in log and "MPS daemon stopped" in log
          and "QUEUE EMPTY" in log and "args: -d" in mps and "quit" in mps, f"rc {rc}; log {log[-500:]}; mps {mps}; {err[-300:]}")
    ok = bool(a and a[4] and b and b[4])
    check("run_queue_mps.sh: drivers, --max-minutes, shared lock, MPS pipe, interpreters", ok
          and a[4]["script"] == "htm_driver.py" and b[4]["script"] == "other_driver.py"
          and a[4]["argv"] == ["--data", "sa", "--seed", "1", "--max-minutes", "7"]
          and b[4]["argv"] == ["--data", "np", "--max-minutes", "7"]
          and a[4]["env"]["TMLM_GPU_LOCK_MODE"] == "shared" and a[4]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == os.path.join(q, "mps/pipe")
          and a[2].get("job_python") == os.path.join(os.path.dirname(h), "decision_layer/.venv-htm/bin/python")
          and a[3]["python_executable"] == os.path.join(os.path.dirname(h), "tm_language_model/.venv-tm/bin/python"),
          f"{a and a[4]} {b and b[4]} {a and a[2]}")
    # a missing interpreter stops the runner before the MPS daemon starts
    root2 = tempfile.mkdtemp(prefix="htm_mps_bad_")
    h2 = make_tree(root2, source)
    write(os.path.join(h2, "queue/jobs.txt"), "m_c_exploratory|7|--data sa\n")
    rc, out, err, _ = run(["bash", os.path.join(h2, "queue/run_queue_mps.sh")], env_for(root2, HTM_PY="/nonexistent/bin/python"), root2)
    check("run_queue_mps.sh: a missing HTM_PY gives exit 127 before any MPS call",
          rc == 127 and not os.path.exists(os.path.join(root2, "fake_mps.log")), f"rc {rc}; {err[-300:]}")
    if not keep:
        shutil.rmtree(root)
        shutil.rmtree(root2)


def test_gpu_share_test(source, keep):
    root = tempfile.mkdtemp(prefix="htm_share_")
    h = make_tree(root, source)
    rc, out, err, secs = run(["bash", os.path.join(h, "queue/gpu_share_test.sh")], env_for(root), root, timeout=900)
    names = [f"share_{s}_{k}_s{i}_exploratory" for s in ("best", "granmo")
             for k, n in (("n1", 1), ("n1b", 1), ("n2", 2), ("n3", 3)) for i in range(1, n + 1)]
    names += ["evalsched_best_every1_s1_exploratory", "evalsched_best_every5_s1_exploratory", "saveload_check_exploratory"]
    names += [f"share_{s}_mps2_s{i}_exploratory" for s in ("best", "granmo") for i in (1, 2)]
    runs = {n: one_run(h, n) for n in names}
    missing = [n for n, r in runs.items() if not (r and r[4] and r[1]["exit_status"] == 0)]
    summary = open(os.path.join(h, "queue/share_test/summary.txt")).read()
    result = open(os.path.join(h, "queue/share_test/RESULT.md")).read()
    check("gpu_share_test.sh: all 21 runs ended with exit 0", rc == 0 and not missing, f"rc {rc}; missing {missing}; {err[-300:]}")
    ev1, sl, mp = runs["evalsched_best_every1_s1_exploratory"], runs["saveload_check_exploratory"], runs["share_best_mps2_s1_exploratory"]
    ok = not missing and sl[4]["argv"] == [ev1[0], "model_final.pkl", "dev"] and sl[4]["script"] == "score_saved.py" \
        and mp[4]["argv"][-4:] == ["--max-minutes", "25", "--seed", "1"] and mp[4]["env"]["TMLM_GPU_LOCK_MODE"] == "shared" \
        and mp[4]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == os.path.join(h, "queue/mps/pipe") \
        and runs["share_granmo_n1_s1_exploratory"][4]["argv"] == "--data sa --clauses 50 --A 64 --K 2 --T 3200 --s 100 --passes 4 --seed 1".split()
    check("gpu_share_test.sh: re-score step, MPS arm and driver arguments", ok, f"{sl and sl[4]} {mp and mp[4]}")
    tool = os.path.join(os.path.dirname(h), "tm_language_model/.venv-tm/bin/python")
    check("gpu_share_test.sh: MPS start/stop recorded and the analysis step ran under HTM_TOOL_PY",
          "MPS daemon started" in summary and "MPS daemon stopped" in summary and f"ANALYSIS STAND-IN under {tool}" in result,
          f"{summary[-300:]} | {result}")
    if not keep:
        shutil.rmtree(root)


def test_feas_launch(source, keep):
    root = tempfile.mkdtemp(prefix="htm_feas_")
    h = make_tree(root, source)
    rc, out, err, _ = run([os.path.join(h, "src/feas_slide11/launch.sh"), "v1", "--epochs", "1", "--arg", "a b"], env_for(root), root)
    d = os.path.join(h, "src/feas_slide11")
    rds = run_dirs(os.path.join(root, "tlm_results"), "feas_slide11_v1_exploratory")
    ok = rc == 0 and len(rds) == 1 and os.path.exists(os.path.join(rds[0], "standin.json"))
    if ok:
        s, cfg = load(os.path.join(rds[0], "standin.json")), load(os.path.join(d, "v1.json"))
        copies = sorted(f for f in os.listdir(rds[0]) if f in ("train_tm.py", "prep_data.py", "launch.sh", "interpreters.sh"))
        ok = s["argv"] == ["--tag", "v1", "--epochs", "1", "--arg", "a b"] and s["env"]["TMLM_GPU_LOCK_MODE"] == "shared" \
            and cfg["args"] == ["--epochs", "1", "--arg", "a b"] and cfg.get("job_python", "").endswith("decision_layer/.venv-htm/bin/python") \
            and copies == ["interpreters.sh", "launch.sh", "prep_data.py", "train_tm.py"]
    check("feas_slide11/launch.sh: job runs, arguments pass, config and script copies written", ok, f"rc {rc}; {rds}; {err[-300:]}")
    if not keep:
        shutil.rmtree(root)


RUN_NAMES = ("t_default_exploratory", "t_old_exploratory", "t_relpy_exploratory", "t_bad_htm_py_exploratory",
             "t_pyc_dl_exploratory", "t_pyc_other_exploratory",
             "t_bad_htm_tool_py_exploratory", "t_cap_exploratory", "q_a_exploratory", "q_b_exploratory", "m_a_exploratory",
             "m_b_exploratory", "m_c_exploratory", "evalsched_best_every1_s1_exploratory",
             "evalsched_best_every5_s1_exploratory", "saveload_check_exploratory", "feas_slide11_v1_exploratory")


def strays(since):
    """Entries that the tests could have written into real folders: run folders of this test's run names (or of the
    sharing test) newer than `since` in both results/ folders, and any new entry in this study's src/."""
    out = []
    for d in (os.path.join(STUDY, "results"), os.path.join(TLM, "results")):
        for e in os.listdir(d):
            p = os.path.join(d, e)
            if (e.endswith(RUN_NAMES) or e.startswith("share_") or "_share_" in e) and os.path.getmtime(p) >= since:
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=STUDY, help="folder to take the launchers from (default: this study)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch trees")
    ap.add_argument("--tlm-env", help="folder holding the env.sh to source (default: ../tm_language_model/env)")
    ap.add_argument("--only", nargs="*", default=[], help="run only these tests (names without test_)")
    x = ap.parse_args()
    global TLM_ENV
    if x.tlm_env:
        TLM_ENV = os.path.abspath(x.tlm_env)
    print(f"env.sh sourced by the launchers: {os.path.join(TLM_ENV, 'env.sh')} "
          f"(sha256 {sha256(os.path.join(TLM_ENV, 'env.sh'))})", flush=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("note: CUDA_VISIBLE_DEVICES was not empty in the caller; every launcher here gets it empty", flush=True)
    print(f"source of the launchers: {os.path.abspath(x.source)}", flush=True)
    for rel in LAUNCHERS:
        p = os.path.join(x.source, rel)
        print(f"  {sha256(p) if os.path.exists(p) else 'absent':64}  {rel}", flush=True)
    src_before, t0 = sorted(os.listdir(os.path.join(STUDY, "src"))), time.time()
    tests = [test_launch, test_bytecode, test_run_queue, test_run_queue_mps, test_gpu_share_test, test_feas_launch]
    for t in tests:
        if x.only and t.__name__[5:] not in x.only:
            continue
        print(f"-- {t.__name__}", flush=True)
        try:
            t(os.path.abspath(x.source), x.keep)
        except Exception as e:  # a crash of a test is a failure of that test
            check(f"{t.__name__} ran to its end", False, f"{type(e).__name__}: {e}")
    s = strays(t0 - 1)
    check("no run folder of these tests in the results/ folders of both studies, and no new entry in src/",
          not s and sorted(os.listdir(os.path.join(STUDY, "src"))) == src_before, str(s))
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed, {time.time() - t0:.0f} s")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
