"""A stand-in for chain/part3_cpu_job.sh in the tests of chain/run_round1.sh (test mode only; never a real job).

    fake_round1_wrapper.py <the wrapper's arguments>      (called through <test dir>/fake_wrapper.sh)

It writes into $ROUND1_TEST_DIR only: the outputs a finished A or C step would leave (A_out/<step>/ with done.json,
C_runs/<stamp>_<stage>/ with run.json and outputs.json, C_data/), as told by <test dir>/scenario.json:
    {"codes": {"<arguments joined by spaces>": [exit code of call 1, call 2, ...]},   default 0; the last one repeats
     "nowrite": ["<arguments>", ...],              exit with the code but write nothing (a lying job)
     "go_to_K3c": {"full": true, "nf": true},      what S2d writes when it exits 0
     "pilots": {"full": {"15": 5.1, "40": 5.2}},   tune-cal bits of the pilots (the better s follows from them)
     "gate_s": {"full": 40},                       the s the K3c gate prints on exit 5 (default: the better s)
     "c_decision_stop": "STOP at K0b: ..."}        the decision C's gates write on exit 3
Every call is appended to <test dir>/calls.txt; the first call's view of its environment goes to env_seen.json (only
counts and the expected values, never variable names or values from outside the expected set).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

CH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CH))
import round1_lib as L  # noqa: E402

ALLOWED_ENV = {"HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "PATH", "PWD", "SHLVL", "_", "OLDPWD",
               "ROUND1_ENV_CLEAN", "ROUND1_TEST_DIR", "ROUND1_TEST_PINS", "ROUND1_TEST_SLEEP", "ROUND1_TEST_HASHCACHE",
               "CUDA_VISIBLE_DEVICES", "CUDA_CACHE_DISABLE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_CACHE_DIR",
               "TMPDIR", "JOBLIB_TEMP_FOLDER", "MPLCONFIGDIR", "XDG_CACHE_HOME", "LC_CTYPE"}
A_PIN = [h for rel, h, _ in L.FILE_PINS["A"] if rel == L.RECORD["A"]][0]


def jw(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def a_outputs(T: Path, args: list, code: int, sc: dict) -> None:
    script = args[2]
    opt = {args[i]: args[i + 1] for i in range(3, len(args) - 1) if args[i].startswith("--")}
    ok = code == 0
    base = T / "A_out"
    if script == "s0_data.py":
        d = base / "s0"
        bl = json.dumps({"names": ["b0", "b1"], "count_derived": [True, False], "n_position_bits": 1})
        (d).mkdir(parents=True, exist_ok=True)
        (d / "bit_list.json").write_text(bl)
        (d / "meta.json").write_text("{}")
        (d / "corpus.npz").write_bytes(b"corpus")
        jw(d / "done.json", {"bit_list_sha256": hashlib.sha256(bl.encode()).hexdigest(), "bits": 2,
                             "count_derived_bits": 1, "bits_A_nf": 1, "frozen_record_sha256": A_PIN,
                             "counts": {"replies": 10, "fit_replies": 8, "tune_cal_replies": 1, "tune_hold_replies": 1,
                                        "fit_tokens_with_end": 100, "vocab": 20}})
    elif script == "s1_floors.py":
        d = base / "s1"
        for f in ("summ_cal.npz", "summ_hold.npz"):
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_bytes(b"x")
        jw(d / "results.json", {"floor": "F1n4", "K1": {"rel_ppl": 2e-5 if ok else 3e-3, "pass": ok},
                                "tune_cal_bits": {"F1n4": 5.0}, "tune_hold_bits": {"F1n4": 5.1}})
        jw(d / "done.json", {"floor": "F1n4", "K1": ok, "frozen_record_sha256": A_PIN})
    elif script == "s2a_rows.py":
        d = base / "s2a"
        for f in ("twin_rows.npz", "train_rows.npz", "cal_rows.npz", "hold_rows.npz"):
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_bytes(f.encode())
        jw(d / "results.json", {"rows": {"train": 5}, "leak_test": {"pass": True}, "nofuture": {"pass": True}})
        jw(d / "done.json", {"rows": {"train": 5}, "frozen_record_sha256": A_PIN})
    elif script == "s2b_twin.py":
        d = base / "s2b_twin"
        (d).mkdir(parents=True, exist_ok=True)
        (d / "bits.npz").write_bytes(b"b")
        (d / "B.pkl").write_bytes(b"B")
        jw(d / "results.json", {"pick": "J2_w3", "B": {"tune_hold_bits": 4.9},
                                "K9": {"b": 0.0, "d_hold": 0.0 if ok else -0.01, "pass": ok}})
        jw(d / "done.json", {"pick": "J2_w3", "K9": ok, "frozen_record_sha256": A_PIN})
    elif script == "s2c_aa.py":
        d = base / "s2c_aa"
        (d).mkdir(parents=True, exist_ok=True)
        (d / "bits.npz").write_bytes(b"b")
        jw(d / "results.json", {})
        jw(d / "k8.json", {"K8": {"mean": 0.001 if ok else 0.009, "lo": -0.001, "hi": 0.003, "pass": ok}})
        jw(d / "done.json", {"K8": ok, "frozen_record_sha256": A_PIN})
    elif script == "s2b_arm.py":
        d = base / f"s2b_{opt['--arm']}_{opt['--family']}"
        (d).mkdir(parents=True, exist_ok=True)
        (d / "bits.npz").write_bytes(b"b")
        (d / "pick.pkl").write_bytes(b"p")
        jw(d / "results.json", {"pick": "c1", "configs": {"c1": {"tune_hold_minus_B": -0.05}},
                                "K9": {"b": 0.0, "d_hold": 0.0, "pass": ok}})
        jw(d / "done.json", {"pick": "c1", "K9": ok, "frozen_record_sha256": A_PIN})
    elif script == "s2d_gates.py":
        d = base / "s2d_gates"
        go = dict(sc.get("go_to_K3c", {"full": True, "nf": True})) if ok else {"full": False, "nf": False}
        k3 = {a: {"K3": {"best_trees_lr2": -0.05, "i_pass": go[a], "d_lr1": -0.03, "ii_pass": go[a]}} for a in go}
        jw(d / "gates.json", {"go_to_K3c": go, "K8": {"pass": True}, "K9_all_pass": True, "arms": k3})
        (d / "words.txt").write_text("gates\n")
        jw(d / "done.json", {"go_to_K3c": go, "frozen_record_sha256": A_PIN})
    elif script == "k3c_pilot.py":
        arm, s = opt["--arm"], int(float(opt["--s"]))
        sh = "--shuffled" in args
        d = base / (f"k3c_{arm}_s{s}" + ("_shuffled" if sh else ""))
        (d).mkdir(parents=True, exist_ok=True)
        (d / "state.npz").write_bytes(b"s")
        (d / "bits.npz").write_bytes(b"b")
        tc = sc.get("pilots", {}).get(arm, {}).get(str(s), 5.0)
        jw(d / "results.json", {"tag": d.name, "b": 0.0 if sh else 0.3, "tune_cal_bits": tc,
                                "tune_hold_minus_B": 0.0 if sh else -0.02})
        jw(d / "done.json", {"tune_cal_bits": tc, "frozen_record_sha256": A_PIN})
    elif script == "k3c_gate.py":
        arm = opt["--arm"]
        if code == 5:
            s = sc.get("gate_s", {}).get(arm) or L.better_s(arm, L.paths())
            print(f"K3c passes for A-{arm} (better s {s}); K9 needs the shuffled pilot first:\n"
                  f"  chain/part3_cpu_job.sh A dl k3c_pilot.py --arm {arm} --s {s} --shuffled")
            return
        d = base / f"k3c_gate_{arm}"
        best = L.better_s(arm, L.paths())
        jw(d / "k3c.json", {"better_s": best, "K3c": {"gain_pilot": 0.02, "gain_trees": 0.05, "pass": ok},
                            "K9_tm": {"pass": True} if ok else None, "go_to_GPU": ok})
        (d / "words.txt").write_text("k3c\n")
        jw(d / "done.json", {"go_to_GPU": ok, "frozen_record_sha256": A_PIN})


def c_outputs(T: Path, stage: str, code: int, sc: dict) -> None:
    data = T / "C_data"
    data.mkdir(parents=True, exist_ok=True)
    if stage == "gates":
        out = data / "step6_decision.json"
        dec = (sc.get("c_decision_stop", "STOP at K0b: LR and LightGBM both more than 0.0072 bpw behind the twin system")
               if code == 3 else "PASS: K0 and K0b passed; the GPU TM step may be scheduled")
        jw(out, {"decision": dec})
    else:
        out = data / f"{stage}.npz"
        out.write_bytes(f"{stage} {time.time()}".encode())
    n = len(list((T / "C_runs").glob("*"))) if (T / "C_runs").exists() else 0
    run = T / "C_runs" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{n:03d}_{stage}"
    jw(run / "run.json", {"state": "finished"})
    jw(run / "outputs.json", {str(out.relative_to(L.ROOT)): L._sha256_file(out)})


def main() -> int:
    T = Path(os.environ["ROUND1_TEST_DIR"]).resolve()
    if not T.is_relative_to((CH / "tmp").resolve()):
        print("fake wrapper: the test dir must be inside chain/tmp")
        return 2
    args = sys.argv[1:]
    desc = " ".join(args)
    sc = json.loads((T / "scenario.json").read_text()) if (T / "scenario.json").exists() else {}
    st_p = T / "fake_state.json"
    state = json.loads(st_p.read_text()) if st_p.exists() else {}
    i = state.get(desc, 0)
    state[desc] = i + 1
    st_p.write_text(json.dumps(state))
    codes = sc.get("codes", {}).get(desc, [0])
    code = codes[min(i, len(codes) - 1)]
    with open(T / "calls.txt", "a") as f:
        f.write(desc + "\n")
    if not (T / "env_seen.json").exists():
        env = dict(os.environ)
        jw(T / "env_seen.json", {
            "unexpected_variables": len([k for k in env if k not in ALLOWED_ENV]),
            "threads": {k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS")},
            "cuda_visible_devices_empty": env.get("CUDA_VISIBLE_DEVICES", None) == "",
            "nice": os.nice(0)})
    folder = a_folder(args) if args[0] == "A" else None
    if folder:  # like A's OutDir: a rerun removes a crashed run's .partial folder first
        part = T / "A_out" / (folder + ".partial")
        if part.exists():
            for p in sorted(part.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            part.rmdir()
    if desc not in sc.get("nowrite", []) and code in (0, 3, 4, 5):
        if args[0] == "A":
            a_outputs(T, args, code, sc)
        elif args[0] == "C":
            c_outputs(T, args[1], code, sc)
    if code == 1 and folder:  # a crashed A job leaves only its .partial folder
        part = T / "A_out" / (folder + ".partial")
        part.mkdir(parents=True, exist_ok=True)
        (part / "log.txt").write_text("crashed\n")
    print(f"part3_cpu_job: {desc} -> exit {code} (fake)")
    return code


def a_folder(args: list):
    script = args[2]
    opt = {args[i]: args[i + 1] for i in range(3, len(args) - 1) if args[i].startswith("--")}
    simple = {"s0_data.py": "s0", "s1_floors.py": "s1", "s2a_rows.py": "s2a", "s2b_twin.py": "s2b_twin",
              "s2c_aa.py": "s2c_aa", "s2d_gates.py": "s2d_gates"}
    if script in simple:
        return simple[script]
    if script == "s2b_arm.py":
        return f"s2b_{opt['--arm']}_{opt['--family']}"
    if script == "k3c_pilot.py":
        return f"k3c_{opt['--arm']}_s{int(float(opt['--s']))}" + ("_shuffled" if "--shuffled" in args else "")
    if script == "k3c_gate.py":
        return f"k3c_gate_{opt['--arm']}"
    return None


if __name__ == "__main__":
    sys.exit(main())
